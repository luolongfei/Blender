"""Read-only current-pose shoulder, blouse and collar diagnostic.

Integrator-only fresh background Blender. No pose/frame changes, rendering,
cloth solver, collider creation, material edits or Blend save. Retains native
modifier evaluation and exports full raw garment data plus finite actual
upper-body triangles for subsequent offline fitting/contact diagnosis.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CAPTURE_HELPER_SHA = 'c2a3bffb94a2231bac4a96337bc10094d5e4e0367ce48a35703361e3b2e453ad'
GARMENTS = (
    'Blouse_Tucked_Bodice', 'Blouse_Button_Placket',
    'Blouse_Puff_Sleeve_L', 'Blouse_Puff_Sleeve_R',
    'Blouse_PeterPan_Collar_L', 'Blouse_PeterPan_Collar_R',
    'Collar_Edge_Stitch_L', 'Collar_Edge_Stitch_R',
    'Sleeve_Shoulder_Seam_L', 'Sleeve_Shoulder_Seam_R',
)
SKIN = ('R01 Head', 'R11 Female head study', 'R11 Continuous neck bridge')
ROI_LOW = np.array((-.70, -.30, .95), dtype=np.float64)
ROI_HIGH = np.array((.70, .30, 1.42), dtype=np.float64)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(value, suffix):
    root = project_path()
    if Path(value).is_absolute() or '..' in str(value).replace('\\', '/').split('/'):
        raise ValueError('Use an independent project-relative reconstruction path')
    path = (root / value).resolve()
    path.relative_to((root / 'work/reconstruction-r02').resolve())
    if path.suffix != suffix:
        raise ValueError('Unexpected source or output extension')
    return path


def values(data, prop, width=1, dtype=np.int32):
    array = np.empty((len(data), width), dtype=dtype)
    data.foreach_get(prop, array.ravel())
    return array[:, 0] if width == 1 else array


def raw_geometry(obj, arrays, prefix):
    """Preserve native source coordinates, topology, UV and group provenance."""
    mesh = obj.data
    row = {'object': obj.name, 'type': obj.type, 'data': mesh.name,
           'prefix': prefix, 'matrixWorld': np.asarray(obj.matrix_world).tolist()}
    if obj.type == 'MESH':
        arrays[prefix + '_local'] = values(mesh.vertices, 'co', 3, np.float32)
        arrays[prefix + '_loops'] = values(mesh.loops, 'vertex_index')
        arrays[prefix + '_face_starts'] = values(mesh.polygons, 'loop_start')
        arrays[prefix + '_face_sizes'] = values(mesh.polygons, 'loop_total')
        arrays[prefix + '_face_materials'] = values(mesh.polygons, 'material_index')
        arrays[prefix + '_face_smooth'] = values(mesh.polygons, 'use_smooth', dtype=np.bool_)
        row['vertexGroups'] = [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups]
        counts = np.array([len(v.groups) for v in mesh.vertices], np.int32)
        arrays[prefix + '_weight_offsets'] = np.r_[0, np.cumsum(counts)].astype(np.int32)
        arrays[prefix + '_weight_groups'] = np.array([g.group for v in mesh.vertices for g in v.groups], np.int32)
        arrays[prefix + '_weights'] = np.array([g.weight for v in mesh.vertices for g in v.groups], np.float32)
        row['uvLayers'] = []
        for i, layer in enumerate(mesh.uv_layers):
            key = prefix + '_uv_' + str(i)
            arrays[key] = values(layer.data, 'uv', 2, np.float32)
            row['uvLayers'].append({'name': layer.name, 'array': key})
        row['shapeKeys'] = []
        if mesh.shape_keys:
            for i, key in enumerate(mesh.shape_keys.key_blocks):
                array_key = prefix + '_shape_' + str(i)
                arrays[array_key] = values(key.data, 'co', 3, np.float32)
                row['shapeKeys'].append({'name': key.name, 'array': array_key,
                                         'value': float(key.value), 'mute': key.mute,
                                         'relativeKey': key.relative_key.name})
    elif obj.type == 'CURVE':
        row['splines'] = []
        for i, spline in enumerate(mesh.splines):
            key = prefix + '_spline_' + str(i)
            data = spline.bezier_points if spline.type == 'BEZIER' else spline.points
            arrays[key + '_co'] = values(data, 'co', 3 if spline.type == 'BEZIER' else 4, np.float32)
            if spline.type == 'BEZIER':
                arrays[key + '_left'] = values(data, 'handle_left', 3, np.float32)
                arrays[key + '_right'] = values(data, 'handle_right', 3, np.float32)
            row['splines'].append({'type': spline.type, 'prefix': key,
                                   'cyclic': spline.use_cyclic_u})
    else:
        raise RuntimeError('Unexpected upper-blouse source object type: ' + obj.name)
    return row


def actual_geometry(obj, graph, arrays, prefix, cap):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        if not mesh or not len(mesh.polygons):
            raise RuntimeError('No actual evaluated surface for ' + obj.name)
        local = values(mesh.vertices, 'co', 3, np.float32).astype(np.float64)
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.isfinite(world).all():
            raise RuntimeError('Nonfinite actual surface: ' + obj.name)
        mesh.calc_loop_triangles()
        triangles = values(mesh.loop_triangles, 'vertices', 3)
        polygons = values(mesh.loop_triangles, 'polygon_index')
        corners = world[triangles]
        # AABB overlap keeps crossing triangles whose vertices lie outside ROI.
        keep = (corners.max(1) >= ROI_LOW).all(1) & (corners.min(1) <= ROI_HIGH).all(1)
        tri_ids = np.flatnonzero(keep).astype(np.int32)
        tri = triangles[keep]
        vertex_ids = np.unique(tri).astype(np.int32)
        if not len(vertex_ids):
            raise RuntimeError('Expected upper-body geometry misses finite ROI: ' + obj.name)
        remap = np.full(len(world), -1, np.int32)
        remap[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int32)
        points = world[vertex_ids].astype(np.float32)
        tri = remap[tri]
        arrays[prefix + '_world'] = points
        arrays[prefix + '_triangles'] = tri
        arrays[prefix + '_native_vertex_ids'] = vertex_ids
        arrays[prefix + '_native_triangle_ids'] = tri_ids
        arrays[prefix + '_native_polygon_ids'] = polygons[keep]
        area2 = np.linalg.norm(np.cross(points[tri[:, 1]] - points[tri[:, 0]],
                                       points[tri[:, 2]] - points[tri[:, 0]]), axis=1)
        return {'object': obj.name, 'prefix': prefix, 'depsgraphMode': str(graph.mode),
                'nativeVerticesTotal': len(world), 'nativeTrianglesTotal': len(triangles),
                'roiVertices': len(points), 'roiTriangles': len(tri),
                'boundsMeters': [points.min(0).tolist(), points.max(0).tolist()],
                'degenerateTriangleCount': int(np.count_nonzero(area2 <= 1e-14)),
                'worldSha256': cap['array_hash'](points),
                'trianglesSha256': cap['array_hash'](tri),
                'modifierViewportRenderFlagsMatch': all(m.show_viewport == m.show_render for m in obj.modifiers),
                'subdivision': cap['subdivisions'](obj)}
    finally:
        evaluated.to_mesh_clear()


def capture(source_relative, source_sha, output_relative):
    import bpy
    source = relative_path(source_relative, '.blend')
    output = relative_path(output_relative, '.json')
    archive = output.with_suffix('.npz')
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve prior evidence; choose unused output paths')
    if (not bpy.app.background or bpy.data.is_dirty
            or Path(bpy.data.filepath).resolve() != source or sha(source) != source_sha):
        raise RuntimeError('Open exact saved source and supply its current SHA in fresh background Blender')
    helper = HERE / 'cloth_current_inputs_capture_r11.py'
    if sha(helper) != CAPTURE_HELPER_SHA:
        raise RuntimeError('Frozen common capture dependency changed')
    cap = runpy.run_path(str(helper))
    garment, uv = cap['api']('garment_neck_observe_r11'), cap['api']('production_uv_r08')
    scene = bpy.context.scene
    objects = [bpy.data.objects[name] for name in GARMENTS + SKIN]
    if any(m.show_viewport and m.type in {'CLOTH', 'SOFT_BODY', 'FLUID', 'PARTICLE_SYSTEM'}
           for obj in scene.objects for m in obj.modifiers):
        raise RuntimeError('Active simulation forbidden; old caches must stay untouched')
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    before = garment['_contract'](objects)
    pose_before = uv['_pose_contract']()
    ids_before = cap['scene_ids'](bpy)
    old_inventory = cap['old_cloth_inventory'](bpy)
    arrays, raw, actual, error = {}, [], [], None
    try:
        for i, obj in enumerate(objects):
            # Raw skin and UV are omitted: the actual skin ROI is sufficient for
            # finite garment diagnosis and no skin editing is authorized here.
            if obj.name in GARMENTS:
                raw.append(raw_geometry(obj, arrays, 'raw_' + str(i)))
            actual.append(actual_geometry(obj, graph, arrays, 'actual_' + str(i), cap))
            print('R11_UPPER_BLOUSE_CAPTURE ' + obj.name, flush=True)
    except Exception as exc:
        error = {'error': str(exc), 'traceback': traceback.format_exc()}
    preserved = (before == garment['_contract'](objects)
                 and pose_before == uv['_pose_contract']()
                 and ids_before == cap['scene_ids'](bpy)
                 and old_inventory == cap['old_cloth_inventory'](bpy)
                 and sha(source) == source_sha)
    report = {'schemaVersion': 1, 'status': 'captured-current-static-upper-blouse' if preserved and error is None else 'failed-evidence-only',
              'source': source_relative, 'sourceSha256': source_sha,
              'helperSha256': sha(__file__), 'commonCaptureSha256': CAPTURE_HELPER_SHA,
              'dependencyHashes': cap['DEPENDENCIES'], 'blender': bpy.app.version_string,
              'frame': [scene.frame_current, scene.frame_subframe], 'poseContract': pose_before,
              'roiWorldMeters': [ROI_LOW.tolist(), ROI_HIGH.tolist()],
              'sceneSimplify': {'enabled': scene.render.use_simplify,
                               'subdivision': scene.render.simplify_subdivision,
                               'renderSubdivision': scene.render.simplify_subdivision_render},
              'sourceContract': before, 'rawGarments': raw, 'actualSurfaces': actual,
              'sourceAndScenePreserved': preserved, 'savedBlend': False,
              'simulationRun': False, 'appearanceAccepted': False, 'surfaceContactAccepted': False,
              'limitations': [
                  'Saved pose only; no new pose, action, frame or cloth simulation was evaluated.',
                  'Actual native VIEWPORT surfaces are captured without changing modifier flags or levels.',
                  'This capture does not claim render parity, contact clearance or collision freedom.',
                  'Raw garments retain seam/UV/weight evidence; current skin is only a finite triangle ROI.',
                  'No skirt, cuff, new feet, hand or dynamic cache acceptance is implied.',
              ]}
    if error:
        report.update(error)
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report.update(arrays=archive.relative_to(project_path()).as_posix(), arraysSha256=sha(archive))
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if not preserved or error:
        raise RuntimeError('Upper blouse capture failed; diagnostic retained at ' + str(output))
    return {'report': output_relative, 'sha256': sha(output), 'arraysSha256': sha(archive),
            'objects': len(actual), 'savedBlend': False}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(capture(args.source, args.source_sha, args.output), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
