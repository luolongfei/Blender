"""Read-only R11 cloth collision-input capture from current body and hands.

Run only in a fresh background Blender opened on the explicit saved source.
The result is finite evaluated world-space surfaces and dependency evidence,
not a cloth cache. No collider object, modifier, action or Blend is created.
Subdivision is read at its actual active level; it is never disabled or edited.
Default captures the saved pose only. Explicit --frames samples at most eight
poses of the already active named action, then restores the original state.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

RIG = 'Character_Rig'
BODY = 'R01 Head'
HANDS = ('Hand L · continuous anatomy', 'Hand R · continuous anatomy')
SOURCES = (BODY,) + HANDS
HAND_MARKER = 'R11_refined_hands_integrated'
DEPENDENCIES = {
    'garment_neck_observe_r11': 'b9b38b2c9078ee656a491c1480b884551764dd3ab0cf23ecf042384acd17a9a6',
    'production_uv_r08': 'c8f66773c868bf8ace72710c1d6e9beb7123c41d8069d0cbb197ae2ab4610c6c',
    'foot_contact_audit_r03': 'd098a57bfc44305d1df0c989c2be825e6da4b049b7baa33a7d869f909499a426',
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def path(relative, suffix):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Explicit project-relative independent reconstruction path required')
    result = (project_path() / relative).resolve()
    result.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if result.suffix != suffix:
        raise ValueError('Unexpected source/output suffix')
    return result


def api(name):
    file = HERE / (name + '.py')
    if sha(file) != DEPENDENCIES[name]:
        raise RuntimeError('Frozen observation dependency changed: ' + name)
    return runpy.run_path(str(file))


def values(data, attribute, width=1, dtype=np.int32):
    result = np.empty((len(data), width), dtype)
    data.foreach_get(attribute, result.ravel())
    return result if width > 1 else result[:, 0]


def topology(mesh):
    return {
        'edges': values(mesh.edges, 'vertices', 2),
        'loops': values(mesh.loops, 'vertex_index'),
        'polygon_starts': values(mesh.polygons, 'loop_start'),
        'polygon_sizes': values(mesh.polygons, 'loop_total'),
    }


def topology_record(mesh, arrays=None, prefix=None):
    data = topology(mesh)
    if arrays is not None:
        arrays.update({prefix + '_' + key: value for key, value in data.items()})
    return {'vertices': len(mesh.vertices), 'edges': len(mesh.edges),
            'polygons': len(mesh.polygons), 'loops': len(mesh.loops),
            'connectivity': {key: array_hash(value) for key, value in data.items()}}


def object_dependencies(obj, garment, uv):
    mesh = obj.data
    keys = mesh.shape_keys
    weights = [[(g.group, float(g.weight)) for g in v.groups] for v in mesh.vertices]
    return {
        'object': obj.name, 'mesh': mesh.name, 'objectType': obj.type,
        'objectLibrary': obj.library.filepath if obj.library else None,
        'meshLibrary': mesh.library.filepath if mesh.library else None,
        'meshUsers': mesh.users, 'rawTopology': topology_record(mesh),
        'rawCoordinatesSha256': array_hash(values(mesh.vertices, 'co', 3, np.float32)),
        'modifiers': garment['_modifiers'](obj),
        'groups': [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups],
        'weightsSha256': digest(weights),
        'parent': obj.parent.name if obj.parent else None,
        'parentType': obj.parent_type, 'parentBone': obj.parent_bone,
        'matrixParentInverse': np.asarray(obj.matrix_parent_inverse).tolist(),
        'matrixBasis': np.asarray(obj.matrix_basis).tolist(),
        'matrixWorld': np.asarray(obj.matrix_world).tolist(),
        'visibility': [obj.hide_viewport, obj.hide_render, obj.hide_get()],
        'objectAnimation': uv['_animation'](obj),
        'meshAnimation': uv['_animation'](mesh),
        'shapeAnimation': uv['_animation'](keys) if keys else None,
        'shapeKeys': {key.name: {
            'coordinatesSha256': array_hash(values(key.data, 'co', 3, np.float32)),
            'value': float(key.value), 'mute': key.mute,
            'relativeKey': key.relative_key.name, 'vertexGroup': key.vertex_group,
        } for key in keys.key_blocks} if keys else {},
    }


def subdivisions(obj):
    rows = []
    for m in obj.modifiers:
        if m.type in {'SUBSURF', 'MULTIRES'}:
            rows.append({'name': m.name, 'type': m.type, 'viewport': m.show_viewport,
                         'render': m.show_render, 'levels': m.levels,
                         'renderLevels': m.render_levels,
                         'adaptive': bool(getattr(m, 'use_adaptive_subdivision', False))})
    return rows


def surface(obj, graph, arrays, prefix, reference_topology=None):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        if not mesh or not mesh.vertices or not mesh.polygons:
            raise RuntimeError('Actual source evaluates to an empty collision surface: ' + obj.name)
        local = values(mesh.vertices, 'co', 3, np.float32)
        matrix = np.asarray(evaluated.matrix_world, np.float64)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-10:
            raise RuntimeError('Invalid actual object transform: ' + obj.name)
        points = local.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(points).all() or np.abs(points).max() > 5.:
            raise RuntimeError('Nonfinite or implausible actual body/hand surface: ' + obj.name)
        current_topology = topology_record(mesh, arrays, prefix)
        if reference_topology is not None and current_topology != reference_topology:
            raise RuntimeError('Evaluated polygon/edge connectivity changed between requested poses: ' + obj.name)
        mesh.calc_loop_triangles()
        triangles = values(mesh.loop_triangles, 'vertices', 3)
        polygon_ids = values(mesh.loop_triangles, 'polygon_index')
        cross = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                         points[triangles[:, 2]] - points[triangles[:, 0]])
        area2 = np.linalg.norm(cross, axis=1)
        triangle_normals = cross / np.maximum(area2[:, None], 1e-30)
        arrays[prefix + '_world'] = points.astype(np.float32)
        arrays[prefix + '_local'] = local
        arrays[prefix + '_matrix_world'] = matrix
        arrays[prefix + '_triangles'] = triangles
        arrays[prefix + '_triangle_polygons'] = polygon_ids
        arrays[prefix + '_triangle_normals_world'] = triangle_normals.astype(np.float32)
        # Face edge incidences diagnose open/nonmanifold surfaces. No capping,
        # winding repair, offset, merging or collision padding is introduced.
        loops = values(mesh.loops, 'vertex_index')
        starts = values(mesh.polygons, 'loop_start')
        sizes = values(mesh.polygons, 'loop_total')
        successor = np.arange(len(loops), dtype=np.int32) + 1
        successor[starts + sizes - 1] = starts
        face_edges = np.sort(np.column_stack((loops, loops[successor])), axis=1)
        _, incidence = np.unique(face_edges, axis=0, return_counts=True)
        return {
            'sourceObject': obj.name, 'sourceMesh': obj.data.name, 'arrayPrefix': prefix,
            'coordinateSpace': 'current evaluated object to world, float32 output; no skin offset',
            'depsgraphMode': str(graph.mode), 'topology': current_topology,
            'worldCoordinatesSha256': array_hash(arrays[prefix + '_world']),
            'trianglesSha256': array_hash(triangles),
            'worldBoundsMeters': [points.min(0).tolist(), points.max(0).tolist()],
            'actualSubdivision': subdivisions(obj),
            'triangleCount': len(triangles), 'degenerateTriangleCount': int(np.sum(area2 <= 1e-14)),
            'boundaryEdgeCount': int(np.sum(incidence == 1)),
            'nonmanifoldFaceEdgeCount': int(np.sum(incidence > 2)),
            'negativeWorldDeterminant': bool(np.linalg.det(matrix[:3, :3]) < 0.),
        }
    finally:
        evaluated.to_mesh_clear()


def old_cloth_inventory(bpy):
    rows = []
    for obj in bpy.context.scene.objects:
        if 'Cloth' not in obj.name and 'r02_collision_source' not in obj:
            continue
        rows.append({
            'object': obj.name, 'data': obj.data.name if obj.data else None,
            'sourceTag': obj.get('r02_collision_source'),
            'rawVertices': len(obj.data.vertices) if obj.type == 'MESH' else None,
            'rawCoordinatesSha256': array_hash(values(obj.data.vertices, 'co', 3, np.float32)) if obj.type == 'MESH' else None,
            'visibility': [obj.hide_viewport, obj.hide_render, obj.hide_get()],
            'modifiers': [(m.name, m.type, m.show_viewport, m.show_render) for m in obj.modifiers],
        })
    return sorted(rows, key=lambda row: row['object'])


def scene_ids(bpy):
    return {name: sorted((item.name, item.as_pointer()) for item in getattr(bpy.data, name))
            for name in ('objects', 'meshes', 'actions', 'node_groups', 'collections')}


def capture(source_relative, source_sha, output_relative, frames_text='current', action_name=None):
    import bpy
    source = path(source_relative, '.blend')
    output = path(output_relative, '.json')
    archive = output.with_suffix('.npz')
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve prior capture; choose unused JSON and NPZ paths')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open exact saved independent source in fresh background Blender')
    if sha(source) != source_sha:
        raise RuntimeError('Explicit saved source SHA does not match')
    garment = api('garment_neck_observe_r11')
    uv = api('production_uv_r08')
    audit = api('foot_contact_audit_r03')
    scene = bpy.context.scene
    rig = bpy.data.objects.get(RIG)
    if rig is None or rig.type != 'ARMATURE':
        raise RuntimeError('Current character rig missing')
    if HAND_MARKER not in scene:
        raise RuntimeError('Final refined-hand integration marker required; do not capture old hands as current')
    if abs(scene.unit_settings.scale_length - 1.) > 1e-6:
        raise RuntimeError('Current pipeline requires actual metre units')
    if abs(scene.render.fps / scene.render.fps_base - 30.) > 1e-6:
        raise RuntimeError('Explicit 30 fps timebase required')
    simplify = {'enabled': scene.render.use_simplify,
                'viewportSubdivisionLimit': scene.render.simplify_subdivision,
                'renderSubdivisionLimit': scene.render.simplify_subdivision_render}
    if simplify['enabled'] and min(simplify['viewportSubdivisionLimit'], simplify['renderSubdivisionLimit']) < 1:
        raise RuntimeError('Scene Simplify would reduce required L1 evaluation; do not change it automatically')
    objects = []
    for name in SOURCES:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'MESH':
            raise RuntimeError('Current source surface missing: ' + name)
        if obj.hide_viewport or obj.hide_get() or obj.hide_render:
            raise RuntimeError('Current collision input must be the visible production surface: ' + name)
        sub = subdivisions(obj)
        if len(sub) != 1 or sub[0]['levels'] != 1 or sub[0]['renderLevels'] != 1 or sub[0]['adaptive'] or not sub[0]['viewport'] or not sub[0]['render']:
            raise RuntimeError('Actual current L1 viewport/render subdivision required; do not change it automatically: ' + name)
        if any(m.show_viewport != m.show_render for m in obj.modifiers):
            raise RuntimeError('Viewport/render modifier flags differ; collision source needs explicit review: ' + name)
        arms = [m for m in obj.modifiers if m.type == 'ARMATURE' and m.show_viewport]
        if len(arms) != 1 or arms[0].object != rig:
            raise RuntimeError('Expected one actual character Armature without adding deformation: ' + name)
        objects.append(obj)
    active_solvers = [(o.name, m.name, m.type) for o in scene.objects for m in o.modifiers
                      if m.show_viewport and m.type in {'CLOTH', 'SOFT_BODY', 'FLUID', 'PARTICLE_SYSTEM'}]
    if active_solvers:
        raise RuntimeError('Read-only probe refuses active simulation modifiers: ' + repr(active_solvers))
    handlers = [getattr(h, '__name__', repr(h)) for h in bpy.app.handlers.frame_change_post
                if getattr(h, '_r02_secondary_hair', False)]
    if handlers:
        raise RuntimeError('Retire known legacy Python hair playback before frame sampling: ' + repr(handlers))
    bpy.context.view_layer.update()
    action = rig.animation_data.action if rig.animation_data else None
    action_sha = audit['_action_hash'](action) if action else None
    original_frame = (scene.frame_current, scene.frame_subframe)
    if frames_text == 'current':
        frames = [float(original_frame[0] + original_frame[1])]
    else:
        frames = [float(value.strip()) for value in frames_text.split(',')]
        if not 1 <= len(frames) <= 8 or len(set(frames)) != len(frames) or not all(math.isfinite(f) for f in frames):
            raise ValueError('Choose one to eight distinct finite action frames')
        if action is None or not action_name or action.name != action_name or rig.data.pose_position != 'POSE':
            raise RuntimeError('Explicit already-active named action in POSE required; no automatic selection')
        if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):
            raise RuntimeError('Active NLA composition needs separate explicit review')
        if min(frames) < action.frame_range[0] or max(frames) > action.frame_range[1]:
            raise ValueError('Capture frames must stay in the actual active action range')
    if action_name is not None and (action is None or action.name != action_name):
        raise RuntimeError('Explicit action does not match the current active action')
    before = {obj.name: object_dependencies(obj, garment, uv) for obj in objects}
    original_pose = uv['_pose_contract']()
    original_ids = scene_ids(bpy)
    old_inventory = old_cloth_inventory(bpy)
    original_position = rig.data.pose_position
    arrays = {}
    states = []
    reference = {}
    error = None
    try:
        for index, frame in enumerate(frames):
            if frames_text != 'current':
                scene.frame_set(math.floor(frame), subframe=frame % 1.)
            bpy.context.view_layer.update()
            graph = bpy.context.evaluated_depsgraph_get()
            state = {'index': index, 'frame': frame, 'posePosition': rig.data.pose_position,
                     'poseContract': uv['_pose_contract'](), 'surfaces': [],
                     'rigMatrixWorld': np.asarray(rig.matrix_world).tolist(),
                     'bonePoseWorld': {b.name: np.asarray(rig.matrix_world @ b.matrix).tolist() for b in rig.pose.bones},
                     'shapeValues': {obj.name: {k.name: float(k.value) for k in obj.data.shape_keys.key_blocks}
                                     if obj.data.shape_keys else {} for obj in objects}}
            for object_index, obj in enumerate(objects):
                prefix = f'pose_{index}_surface_{object_index}'
                row = surface(obj, graph, arrays, prefix, reference.get(obj.name))
                reference.setdefault(obj.name, row['topology'])
                state['surfaces'].append(row)
            states.append(state)
            print('R11_CLOTH_CURRENT_INPUT '+json.dumps({'frame': frame, 'surfaces': [
                {'object': r['sourceObject'], 'vertices': r['topology']['vertices']} for r in state['surfaces']]}), flush=True)
    except Exception as exc:
        error = {'error': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if frames_text != 'current':
            scene.frame_set(original_frame[0], subframe=original_frame[1])
        bpy.context.view_layer.update()
    preserved = (original_position == rig.data.pose_position and original_pose == uv['_pose_contract']()
                 and original_ids == scene_ids(bpy) and old_inventory == old_cloth_inventory(bpy)
                 and before == {obj.name: object_dependencies(obj, garment, uv) for obj in objects}
                 and sha(source) == source_sha
                 and (audit['_action_hash'](action) if action else None) == action_sha)
    report = {
        'schemaVersion': 1, 'source': source_relative, 'sourceSha256': source_sha,
        'helperSha256': sha(__file__), 'dependencyHelperSha256': DEPENDENCIES,
        'blender': bpy.app.version_string, 'fps': 30., 'metreScale': 1.,
        'sceneSimplify': simplify, 'gravityEnabled': scene.use_gravity,
        'gravityMetersPerSecondSquared': list(scene.gravity),
        'sourceObjects': list(SOURCES), 'sourceDependencies': before,
        'sourceDependenciesSha256': digest(before), 'originalPose': original_pose,
        'rigRestBones': {b.name: np.asarray(b.matrix_local).tolist() for b in rig.data.bones},
        'activeAction': {'name': action.name, 'sha256': action_sha} if action else None,
        'requestedFrames': frames, 'states': states, 'oldClothInventory': old_inventory,
        'refinedHandIntegrationRecord': str(scene[HAND_MARKER]),
        'headSkinBindingPresent': 'R11_head_neck_skin_bind' in scene,
        'sourceAndScenePreserved': preserved, 'createdBlenderObjects': [],
        'oldObjectsOrCacheEdited': False, 'sourceSubdivisionEdited': False,
        'savedBlend': False, 'simulationRun': False, 'clothCacheCreated': False,
        'appearanceAccepted': False, 'surfaceCollisionAccepted': False,
        'status': 'captured-current-surfaces' if preserved and error is None else 'failed-evidence-only',
        'limits': [
            'Finite current body plus hand surface samples only; no continuous collision or cloth simulation.',
            'R01 Head actual L1 modifier stack is evaluated intact; legacy 24001-point torso copies are never input.',
            'Captured depsgraph is VIEWPORT; matched subdivision and modifier flags alone do not prove all Geometry Nodes render branches identical.',
            'Each source stays a separate surface; open neck boundaries and hidden overlap are recorded, not capped or welded.',
            'These world-space points must not receive another Armature or parent transform in a later collider adapter.',
            'A later collision stage must rebuild at each real action frame, or use a separately validated matching surface cache.',
            'Any body, hand, weights, rig, action, modifier level or contact garment change invalidates dependent simulation.',
            'No collider thickness, contact gap, material calibration or current garment fit is approved by this report.',
            'No new or old Blender objects are mutated or created; old cache files remain untouched.',
        ],
    }
    if error:
        report.update(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report.update(arrays=str(archive.relative_to(project_path())), arraysSha256=sha(archive))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not preserved or error:
        raise RuntimeError('R11 collision-input capture failed; retained diagnostic: ' + str(output))
    return {'report': output_relative, 'reportSha256': sha(output), 'arraysSha256': sha(archive),
            'states': len(states), 'savedBlend': False, 'simulationRun': False}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', default='current')
    parser.add_argument('--action')
    args = parser.parse_args(argv)
    result = capture(args.source, args.source_sha, args.output, args.frames, args.action)
    print('R11_CLOTH_CURRENT_CAPTURE '+json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
