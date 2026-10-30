"""Read-only local wrist/hand/cuff geometry for a concrete seam correction.

Exports actual evaluated local triangle patches and raw rest hand/body patches.
No Blender save, pose change, modifier switch, material change or render.
Overlaps are evidence for diagnosis, not a complete collision certification.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def _path(relative):
    value = (project_path() / relative).resolve(); value.relative_to(project_path().resolve()); return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _co(mesh):
    values = np.empty((len(mesh.vertices), 3), np.float64); mesh.vertices.foreach_get('co', values.ravel()); return values


def _transform(values, matrix):
    matrix = np.asarray(matrix); return values @ matrix[:3, :3].T + matrix[:3, 3]


def _store_patch(mesh, positions, center, radius, key, payload):
    mesh.calc_loop_triangles()
    all_triangles = np.asarray([list(t.vertices) for t in mesh.loop_triangles], np.int32)
    distance = np.linalg.norm(positions - center, axis=1)
    chosen = np.flatnonzero(np.any(distance[all_triangles] < radius, axis=1))
    if not len(chosen): return None
    triangles = all_triangles[chosen]; ids = np.unique(triangles); mapping = np.full(len(positions), -1, np.int32); mapping[ids] = np.arange(len(ids))
    payload[key + '_positions'] = positions[ids]
    payload[key + '_triangles'] = mapping[triangles]
    payload[key + '_originalVertexIds'] = ids
    payload[key + '_originalTriangleIds'] = chosen
    payload[key + '_polygonIds'] = np.asarray([mesh.loop_triangles[int(i)].polygon_index for i in chosen], np.int32)
    payload[key + '_materialIndices'] = np.asarray([mesh.polygons[int(i)].material_index for i in payload[key + '_polygonIds']], np.int32)
    payload[key + '_vertexNormalsLocal'] = np.asarray([list(mesh.vertices[int(i)].normal) for i in ids], np.float64)
    return {'arrayPrefix': key, 'vertices': len(ids), 'triangles': len(chosen), 'selectionRadiusMeters': radius,
            'bounds': [positions[ids].min(0).tolist(), positions[ids].max(0).tolist()],
            'materialSlots': [m.name if m else None for m in mesh.materials]}


def _raw_patch(obj, rig, rest_center, key, payload):
    mesh = obj.data; raw = _co(mesh)
    mixed = raw.copy()
    if mesh.shape_keys:
        blocks = mesh.shape_keys.key_blocks; mixed = np.empty_like(raw); blocks[0].data.foreach_get('co', mixed.ravel())
        for block in blocks[1:]:
            if block.value:
                a = np.empty_like(raw); b = np.empty_like(raw)
                block.data.foreach_get('co', a.ravel()); block.relative_key.data.foreach_get('co', b.ravel())
                mixed += block.value * (a - b)
    matrix = rig.matrix_world.inverted() @ obj.matrix_world
    positions = _transform(mixed, matrix)
    row = _store_patch(mesh, positions, rest_center, .085, key, payload)
    if row is None: return None
    ids = payload[key + '_originalVertexIds']; payload[key + '_basePositionsLocal'] = raw[ids]
    weights = np.zeros((len(ids), len(obj.vertex_groups)), np.float32)
    for i, index in enumerate(ids):
        for group in mesh.vertices[int(index)].groups: weights[i, group.group] = group.weight
    payload[key + '_weights'] = weights
    row.update({'coordinateSpace': 'rig-local current raw shape mixture before Armature',
                'groupNamesInIndexOrder': [g.name for g in obj.vertex_groups],
                'objectToRigMatrix': np.asarray(matrix).tolist(),
                'shapeKeyValues': {b.name: float(b.value) for b in mesh.shape_keys.key_blocks} if mesh.shape_keys else {}})
    return row


def _skin_overlap(hand, body, payload):
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    hp = payload[hand['arrayPrefix'] + '_positions']; bp = payload[body['arrayPrefix'] + '_positions']; bt = payload[body['arrayPrefix'] + '_triangles']
    tree = BVHTree.FromPolygons(bp.tolist(), bt.tolist(), all_triangles=True)
    rows = []
    for index, point in enumerate(hp):
        hit, normal, triangle, distance = tree.find_nearest(Vector(point), .025)
        if hit is None: continue
        signed = float((Vector(point) - hit).dot(normal))
        rows.append((index, triangle, distance, signed))
    values = np.asarray(rows, np.float64).reshape((-1, 4)); key = hand['arrayPrefix'] + '_bodyNearest'
    payload[key] = values
    return {'array': key, 'columns': ['localHandVertex', 'localBodyTriangle', 'distanceMeters', 'normalSignedDistanceMeters'],
            'sampleCount': len(rows), 'negativeNormalDistanceSamples': int(np.sum(values[:, 3] < -.0001)) if len(rows) else 0,
            'limitation': 'Nearest surface normal sign on clipped patches is not a watertight inside/outside test; no collision pass claimed.'}


def observe(source_relative, output_relative):
    import bpy
    source = _path(source_relative); output = _path(output_relative); archive = output.with_suffix('.npz')
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Use exact source in disposable background Object Mode')
    if output.exists() or archive.exists(): raise FileExistsError('Preserve existing seam observations')
    source_hash = _sha(source); scene = bpy.context.scene; rig = bpy.data.objects['Character_Rig']
    if scene.frame_current != 1 or scene.frame_subframe != 0 or rig.data.pose_position != 'POSE':
        raise RuntimeError('Observe the actual current frame-one standing pose')
    graph = bpy.context.evaluated_depsgraph_get(); payload = {}; sides = {}
    for side, prefix in (('L', 'left'), ('R', 'right')):
        bone = rig.pose.bones[prefix + 'Hand']; wrist = np.asarray(rig.matrix_world @ bone.head)
        rest_center = np.asarray(rig.data.bones[prefix + 'Hand'].head_local)
        names = ['Hand ' + side + ' · continuous anatomy', 'R01 Head', 'Blouse_Cuff_' + side]
        patches = {}; raw = {}
        for index, name in enumerate(names):
            obj = bpy.data.objects[name]
            if obj.hide_render or obj.hide_viewport or obj.hide_get():
                raise RuntimeError('Expected visible wrist surface is hidden: ' + name)
            ev = obj.evaluated_get(graph); mesh = ev.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
            try:
                points = _transform(_co(mesh), ev.matrix_world)
                row = _store_patch(mesh, points, wrist, .075, side + '_eval_' + str(index), payload)
                if row:
                    row['coordinateSpace'] = 'evaluated world'; row['normalMatrix'] = np.asarray(ev.matrix_world.to_3x3().inverted().transposed()).tolist()
                    row['object'] = name; row['modifiers'] = [{'name': m.name, 'type': m.type, 'viewport': m.show_viewport, 'render': m.show_render} for m in obj.modifiers]
                    patches[name] = row
            finally: ev.to_mesh_clear()
            if name != 'Blouse_Cuff_' + side:
                row = _raw_patch(obj, rig, rest_center, side + '_raw_' + str(index), payload)
                if row: row['object'] = name; raw[name] = row
        hand_name = names[0]
        if hand_name not in patches or 'R01 Head' not in patches:
            raise RuntimeError('Actual hand or legacy body wrist patch missing; inspect object scope explicitly')
        sides[side] = {'wristWorld': wrist.tolist(), 'wristRestRigLocal': rest_center.tolist(),
                       'handPoseMatrix': np.asarray(bone.matrix).tolist(), 'handRestMatrix': np.asarray(bone.bone.matrix_local).tolist(),
                       'evaluated': patches, 'raw': raw,
                       'handBodyNearestSurface': _skin_overlap(patches[hand_name], patches['R01 Head'], payload)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream: np.savez_compressed(stream, **payload)
    result = {'source': source_relative, 'sourceSha256': source_hash, 'helperSha256': _sha(__file__),
              'arrays': str(archive.relative_to(project_path())), 'arraysSha256': _sha(archive), 'sides': sides,
              'blenderVersion': bpy.app.version_string, 'sourceUnchangedOnDisk': _sha(source) == source_hash,
              'savedBlend': False, 'poseChanged': False, 'appearanceAccepted': False,
              'scope': 'Current two wrist patches only; local overlap diagnosis and future correction inputs.'}
    if not result['sourceUnchangedOnDisk']: raise RuntimeError('Source changed during observation')
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print('R11_WRIST_OBSERVATION ' + json.dumps(observe(args.source, args.output), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
