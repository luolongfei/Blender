"""Seven actual eye-region states for a local intermediate blink correction.

Read-only Blender background observation. No render, binding audit, contour ray
grid, source save, shape-key creation or modifier change. Neutral/full endpoint
coordinates are reference data, not a claim that future corrections preserve them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
BLINK = {'L': 'R11 blink L', 'R': 'R11 blink R'}
WARM = 'R11 warm closed smile'
SMILES = ('R11 gentle smile', 'R11 natural soft smile', WARM)
TOPOLOGY = ('edges', 'loops', 'starts', 'sizes')
STATES = tuple(
    {'id': 'blink_'+str(round(v*100)).zfill(3), 'L': v, 'R': v, 'warm': 0.}
    for v in (0., .2, .5, .8, .9, 1.)
) + ({'id': 'warm_L090_R050', 'L': .9, 'R': .5, 'warm': 1.},)


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(json.dumps(list(value.shape)).encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def project_file(relative, suffix):
    result = (project_path()/relative).resolve()
    result.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if result.suffix != suffix:
        raise ValueError('Unexpected file extension: '+str(result))
    return result


def fixed_roi(neutral, eyes, extra_ids=()):
    """Current eye bounds plus generous physical halo; include complete faces."""
    world = neutral['world']
    keep = np.zeros(len(world), bool)
    boxes = {}
    padding = np.asarray((.012, .010, .013))
    for side, eye in eyes.items():
        lo = eye['world'].min(axis=0)-padding
        hi = eye['world'].max(axis=0)+padding
        keep |= np.all((world >= lo) & (world <= hi), axis=1)
        boxes[side] = {'minimumWorld': lo.tolist(), 'maximumWorld': hi.tolist()}
    extra_ids = np.asarray(extra_ids, dtype=np.int64)
    if len(extra_ids):
        if extra_ids.min() < 0 or extra_ids.max() >= len(world):
            raise RuntimeError('Installed canthal field indices exceed current evaluated head')
        keep[extra_ids] = True
    face_mask = np.logical_or.reduceat(keep[neutral['loops']], neutral['starts'])
    face_ids = np.flatnonzero(face_mask).astype(np.int32)
    corner_mask = np.repeat(face_mask, neutral['sizes'])
    vertex_ids = np.unique(neutral['loops'][corner_mask]).astype(np.int32)
    if len(vertex_ids) < 1000 or len(face_ids) < 1000:
        raise RuntimeError('Current eye-region selection is unexpectedly empty')
    mapping = np.full(len(world), -1, np.int32)
    mapping[vertex_ids] = np.arange(len(vertex_ids), dtype=np.int32)
    return vertex_ids, face_ids, mapping, boxes


def roi_triangles(snapshot, face_ids, mapping):
    # polygon IDs, unlike loop-triangle row order, remain invariant on deformation.
    selected = np.isin(snapshot['trianglePolygons'], face_ids)
    global_ids = np.flatnonzero(selected).astype(np.int32)
    result = mapping[snapshot['triangles'][selected]]
    if np.any(result < 0):
        raise RuntimeError('A selected polygon uses a point outside the fixed ROI')
    return result, global_ids, snapshot['trianglePolygons'][selected]


def capture_mesh(obj, binding):
    import bpy
    snapshot = binding['_head_mesh'](obj, triangulate=True)
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    polygon_ids = np.empty(len(mesh.loop_triangles), np.int32)
    mesh.loop_triangles.foreach_get('polygon_index', polygon_ids)
    snapshot['trianglePolygons'] = polygon_ids
    return snapshot


def exact_topology(a, b):
    return a['counts'] == b['counts'] and all(np.array_equal(a[k], b[k]) for k in TOPOLOGY)


def capture_lashes(binding):
    """Actual curve points only, for geometric attribution of the combo highlight."""
    import bpy
    result = {}
    for side in ('L', 'R'):
        for layer in ('upper', 'lower'):
            obj = bpy.data.objects['R11 Adult lashes '+layer+' '+side]
            ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            data = ev.data
            if not hasattr(data, 'position_data'):
                raise RuntimeError('Expected actual evaluated lash curves: '+obj.name)
            points = binding['_array'](data.position_data, 'vector')
            counts = np.asarray([c.points_length for c in data.curves], np.int32)
            if counts.sum() != len(points):
                raise RuntimeError('Actual lash curve indexing is inconsistent: '+obj.name)
            tag = side+'_'+layer
            result[tag+'_world'] = binding['_transform'](points, ev.matrix_world)
            result[tag+'_counts'] = counts
            if 'radius' in data.attributes:
                result[tag+'_radius'] = binding['_array'](data.attributes['radius'].data, 'value', 1).ravel()
    return result


def observe(source_relative, output_relative):
    import bpy
    source = project_file(source_relative, '.blend')
    output = project_file(output_relative, '.json')
    npz_path = output.with_suffix('.npz')
    if output.exists() or npz_path.exists():
        raise RuntimeError('Preserve earlier observations; fresh output paths required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh saved independent background source required')
    if 'R11_blink_corner_surface_refinement' not in bpy.context.scene:
        raise RuntimeError('Observe the installed current canthal candidate or its descendant')
    binding = api('face_fibre_binding_r11')
    pose_api = api('production_uv_r08')
    pose_before = pose_api['_pose_contract']()
    head = bpy.data.objects[HEAD]
    keys = head.data.shape_keys.key_blocks
    controlled = (*BLINK.values(), *SMILES)
    missing = [name for name in controlled if name not in keys]
    if missing:
        raise RuntimeError('Required current public controls absent: '+repr(missing))
    previous = {name: float(keys[name].value) for name in controlled}
    saved_values = {key.name: float(key.value) for key in keys}
    source_sha = sha(source)
    payload = {}
    marker = json.loads(bpy.context.scene['R11_blink_corner_surface_refinement'])
    cache = bpy.data.objects.get(marker.get('lookupObject', ''))
    attribute_name = 'R11_canthal_surface_angle_radius_side'
    extra_ids = np.empty(0, np.int32)
    if cache is None or attribute_name not in cache.data.attributes:
        raise RuntimeError('Installed canthal lookup evidence is unavailable')
    lookup = binding['_array'](cache.data.attributes[attribute_name].data, 'vector')
    extra_ids = np.flatnonzero(np.any(lookup[:, :2] != 0., axis=1)).astype(np.int32)
    payload['installedCanthalNonzeroGlobalVertexIds'] = extra_ids
    relief_control = 'R11 upper lid relief'
    report = {
        'source': source_relative, 'sourceSha256': source_sha,
        'helperSha256': sha(__file__), 'head': HEAD, 'states': [], 'eyes': {},
        'savedPublicControlValues': previous, 'savedAllKeyValues': saved_values,
        'sourceFrame': [bpy.context.scene.frame_current, bpy.context.scene.frame_subframe],
        'evaluation': 'Current saved scene frame, armature pose-position temporarily REST; no gaze/relief/static-key override.',
        'relief': {'present': relief_control in head,
                   'savedValue': float(head[relief_control]) if relief_control in head else None},
        'headModifierOrder': [m.name for m in head.modifiers],
        'canthalField': {'lookupObject': cache.name, 'attribute': attribute_name,
                         'nonzeroPoints': len(extra_ids), 'attributeSha256': array_sha(lookup)},
        'sourceSaved': False, 'appearanceAccepted': False, 'actualContactVerified': False,
        'scope': 'Actual geometry observations for a local intermediate blink correction; no old static observation is assumed compatible.',
        'limits': ['Polygon connectivity is stored once; per-state loop triangulation may legally change diagonals.',
                   'Endpoint arrays establish this source reference; they do not certify future endpoint preservation.',
                   'No contour rays, binding validation, rendering or general collision certification is performed.'],
    }

    def values(state):
        for name in SMILES:
            keys[name].value = state['warm'] if name == WARM else 0.
        for side in ('L', 'R'):
            keys[BLINK[side]].value = state[side]
        bpy.context.view_layer.update()
        actual = {name: float(keys[name].value) for name in controlled}
        wanted = {BLINK['L']: state['L'], BLINK['R']: state['R'],
                  SMILES[0]: 0., SMILES[1]: 0., WARM: state['warm']}
        if any(abs(actual[name]-wanted[name]) > 1e-6 for name in controlled):
            raise RuntimeError('Driver or animation overrides a requested observation control')
        return actual

    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            eyes = {side: capture_mesh(bpy.data.objects['R11 Adult sclera '+side], binding) for side in ('L', 'R')}
            for side, eye in eyes.items():
                for field in ('world', 'co', 'matrix', 'triangles', 'trianglePolygons', *TOPOLOGY):
                    payload['eye_'+side+'_'+field] = eye[field]
                payload['eye_'+side+'_originalVertexIds'] = np.arange(len(eye['world']), dtype=np.int32)
                report['eyes'][side] = {'object': 'R11 Adult sclera '+side, 'counts': eye['counts'],
                                       'topologySha256': eye['topologySha256'], 'worldSha256': array_sha(eye['world']),
                                       'centerWorld': ((eye['world'].min(0)+eye['world'].max(0))*.5).tolist(),
                                       'diametersWorld': np.ptp(eye['world'], axis=0).tolist()}
            neutral = None
            for index, state in enumerate(STATES):
                actual = values(state)
                mesh = capture_mesh(head, binding)
                if neutral is None:
                    neutral = mesh
                    ids, faces, mapping, boxes = fixed_roi(neutral, eyes, extra_ids)
                    payload['head_originalVertexIds'] = ids
                    payload['head_originalPolygonIds'] = faces
                    for field in TOPOLOGY:
                        payload['head_'+field] = neutral[field]
                    report.update(headCounts=neutral['counts'], headTopologySha256=neutral['topologySha256'],
                                  headMatrixWorld=neutral['matrix'].tolist(), roiBoxes=boxes,
                                  roiVertexCount=len(ids), roiPolygonCount=len(faces),
                                  roiSelection='Union of current actual globe AABBs + (12,10,13) mm halo and installed nonzero canthal points; all incident polygon vertices included.')
                if not exact_topology(neutral, mesh):
                    raise RuntimeError('Actual polygon/loop/edge/index topology changed in state '+state['id'])
                if not np.array_equal(neutral['matrix'], mesh['matrix']):
                    raise RuntimeError('Head frame changed during expression-only sampling')
                triangles, triangle_ids, triangle_polygons = roi_triangles(mesh, faces, mapping)
                tag = 'state_'+str(index)
                world = mesh['world'][ids]
                payload[tag+'_head_world'] = world
                payload[tag+'_head_triangles'] = triangles
                if index == 0:
                    payload['head_originalTriangleIds'] = triangle_ids
                    payload['head_trianglePolygonIds'] = triangle_polygons
                index_map_same = (np.array_equal(triangle_ids, payload['head_originalTriangleIds']) and
                                  np.array_equal(triangle_polygons, payload['head_trianglePolygonIds']))
                if not index_map_same:
                    # Preserve any genuine per-state ordering change; do not mistake
                    # legal triangulation ordering for changed polygon topology.
                    payload[tag+'_head_originalTriangleIds'] = triangle_ids
                    payload[tag+'_head_trianglePolygonIds'] = triangle_polygons
                eye_checks = {}
                for side in ('L', 'R'):
                    now = capture_mesh(bpy.data.objects['R11 Adult sclera '+side], binding)
                    same = exact_topology(eyes[side], now) and np.array_equal(eyes[side]['world'], now['world'])
                    eye_checks[side] = same
                    if not same:
                        raise RuntimeError('Blink/warm unexpectedly changed actual globe '+side)
                if state['warm']:
                    for name, array in capture_lashes(binding).items():
                        payload[tag+'_lash_'+name] = array
                delta = world-neutral['world'][ids]
                row = {'index': index, **state, 'publicControls': actual,
                       'headWorldArray': tag+'_head_world', 'headWorldSha256': array_sha(world),
                       'trianglesArray': tag+'_head_triangles', 'trianglesSha256': array_sha(triangles),
                       'triangleIndexMapShared': index_map_same,
                       'originalTriangleIdsArray': 'head_originalTriangleIds' if index_map_same else tag+'_head_originalTriangleIds',
                       'trianglePolygonIdsArray': 'head_trianglePolygonIds' if index_map_same else tag+'_head_trianglePolygonIds',
                       'polygonConnectivityExact': True, 'headTransformExact': True, 'globesExact': eye_checks,
                       'maximumDisplacementFromCurrentNeutralMeters': float(np.linalg.norm(delta, axis=1).max()),
                       'drivenWarmLowerLids': {side: float(keys['R11 warm smile lower lid '+side].value)
                                              for side in ('L', 'R') if 'R11 warm smile lower lid '+side in keys}}
                report['states'].append(row)
                print('R11_BLINK_MIDPATH_STATE '+json.dumps({'index': index, 'id': state['id'], 'roiPoints': len(ids), 'controls': actual}), flush=True)
            report['endpoints'] = {name: {'stateIndex': index, 'worldArray': 'state_'+str(index)+'_head_world',
                                         'worldSha256': report['states'][index]['headWorldSha256'],
                                         'trianglesArray': 'state_'+str(index)+'_head_triangles',
                                         'globalVertexIdsArray': 'head_originalVertexIds'}
                                   for name, index in (('neutral', 0), ('fullClosed', 5))}
    finally:
        for name, value in previous.items():
            keys[name].value = value
        bpy.context.view_layer.update()
    if any(float(keys[name].value) != value for name, value in previous.items()):
        raise RuntimeError('Public controls were not restored')
    if pose_api['_pose_contract']() != pose_before or sha(source) != source_sha:
        raise RuntimeError('Observation changed rig pose or source file')
    report.update(controlsRestored=True, poseRestored=True, sourceFileUnchanged=True,
                  arrays=str(npz_path.relative_to(project_path().resolve())), storedStateCount=len(STATES))
    np.savez_compressed(npz_path, **payload)
    report['arraysSha256'] = sha(npz_path)
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R11_BLINK_MIDPATH_OBSERVED '+json.dumps({'output': output_relative, 'arraysSha256': report['arraysSha256'], 'states': len(STATES)}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.output)
