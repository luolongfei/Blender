"""Read-only REST geometry for reference-led low twin-tail tie alignment.

Run in a fresh background Blender loaded from --source. No render and no blend
save. --output accepts either member of a new .json/.npz pair. All coordinates
are collected from the actual loaded source, not an older geometry report.

Reusable collectors expose world vertices and triangle indices directly for
BVH construction. Ear triangles are selected by evaluated .sculpt_face_set 4
(positive X) and 5 (negative X), with explicit original vertex/triangle maps.
Visibility is evidence only: hidden old geometry is never automatically
declared a required collision surface.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

P = project_path()
MAIN = 'R11 Donor / long hair main'
HEAD = 'R11 Female head study'
ROLE = 'R11 groom role'
TAILS = (('tail_positive', 'R02 Strands / R02 Twin tail L'),
         ('tail_negative', 'R02 Strands / R02 Twin tail R'))
TIES = tuple((f'tie_{"positive" if side > 0 else "negative"}_{"a" if ring < 0 else "b"}',
              f'R02 Tie {side} {ring}') for side in (-1, 1) for ring in (-1, 1))
COLLIDERS = (
    ('head', HEAD),
    ('neck_bridge', 'R11 Continuous neck bridge'),
    ('old_body', 'R01 Head'),
    ('bodice', 'Blouse_Tucked_Bodice'),
    ('sleeve_L', 'Blouse_Puff_Sleeve_L'),
    ('sleeve_R', 'Blouse_Puff_Sleeve_R'),
    ('collar_L', 'Blouse_PeterPan_Collar_L'),
    ('collar_R', 'Blouse_PeterPan_Collar_R'),
)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(values):
    a = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def add_array(arrays, key, values):
    if key in arrays:
        raise RuntimeError('Duplicate observation array: ' + key)
    value = np.ascontiguousarray(values)
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise RuntimeError('Nonfinite actual geometry: ' + key)
    arrays[key] = value
    return value


def coordinates(obj):
    if obj.type == 'MESH':
        values = np.empty(len(obj.data.vertices) * 3, np.float32)
        obj.data.vertices.foreach_get('co', values)
    elif obj.type == 'CURVES':
        values = np.empty(len(obj.data.position_data) * 3, np.float32)
        obj.data.position_data.foreach_get('vector', values)
    else:
        raise RuntimeError('Expected mesh or hair curves: ' + obj.name)
    return values.reshape(-1, 3)


def world_positions(local, matrix):
    matrix = np.asarray(matrix, np.float64)
    return np.asarray(local, np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def bounds(world):
    return [world.min(0).tolist(), world.max(0).tolist()] if len(world) else None


def attribute_metadata(data):
    return [{'name': a.name, 'dataType': a.data_type, 'domain': a.domain,
             'elements': len(a.data)} for a in data.attributes]


def scalar_attribute(data, name, prefix, arrays, *, required=False,
                     expected_type=None, expected_domain=None):
    """Return exact existence/type/domain state; never synthesize an attribute."""
    attr = data.attributes.get(name)
    if attr is None:
        if required:
            raise RuntimeError('Required actual attribute missing: ' + name)
        return {'exists': False, 'name': name}
    if expected_type and attr.data_type != expected_type:
        raise RuntimeError('Unexpected attribute type: ' + name + ' ' + attr.data_type)
    if expected_domain and attr.domain != expected_domain:
        raise RuntimeError('Unexpected attribute domain: ' + name + ' ' + attr.domain)
    if attr.data_type not in ('INT', 'FLOAT', 'BOOLEAN'):
        raise RuntimeError('Expected scalar actual attribute: ' + name)
    dtype = {'INT': np.int32, 'FLOAT': np.float32, 'BOOLEAN': np.bool_}[attr.data_type]
    values = np.empty(len(attr.data), dtype)
    attr.data.foreach_get('value', values)
    add_array(arrays, prefix, values)
    return {'exists': True, 'name': name, 'dataType': attr.data_type,
            'domain': attr.domain, 'elements': len(values), 'array': prefix,
            'valuesSha256': array_sha(values)}


def visibility(obj):
    scene_paths = []
    layer_paths = []

    def scene_visit(collection, path=(), blocked=False):
        path = path + (collection.name,)
        blocked = blocked or collection.hide_render
        if obj.name in collection.objects:
            scene_paths.append({'path': list(path), 'ancestorHideRender': bool(blocked)})
        for child in collection.children:
            scene_visit(child, path, blocked)

    def layer_visit(layer, path=(), excluded=False, hidden=False, render_blocked=False):
        path = path + (layer.name,)
        excluded = excluded or layer.exclude
        hidden = hidden or layer.hide_viewport or layer.collection.hide_viewport
        render_blocked = render_blocked or layer.collection.hide_render
        if obj.name in layer.collection.objects:
            layer_paths.append({'path': list(path), 'ancestorExcluded': bool(excluded),
                                'ancestorViewportHidden': bool(hidden),
                                'ancestorHideRender': bool(render_blocked)})
        for child in layer.children:
            layer_visit(child, path, excluded, hidden, render_blocked)

    scene_visit(bpy.context.scene.collection)
    layer_visit(bpy.context.view_layer.layer_collection)
    return {'hideRender': obj.hide_render, 'hideViewport': obj.hide_viewport,
            'hiddenInCurrentViewLayer': obj.hide_get(),
            'visibleInCurrentViewLayer': obj.visible_get(),
            'sceneCollectionPaths': scene_paths, 'viewLayerCollectionPaths': layer_paths,
            'renderEligibleByCollectedFlags': bool(not obj.hide_render
                and any(not p['ancestorExcluded'] and not p['ancestorHideRender']
                        for p in layer_paths)),
            'collisionUse': 'Observed candidate only. The installer must explicitly select current visible/physical surfaces; hidden legacy meshes are not mandatory obstacles.'}


def json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bpy.types.ID):
        return {'idType': value.bl_rna.identifier, 'name': value.name}
    try:
        return [json_value(v) for v in value]
    except TypeError:
        return str(value)


def modifier_state(obj):
    rows = []
    for modifier in obj.modifiers:
        row = {'name': modifier.name, 'type': modifier.type,
               'showViewport': modifier.show_viewport, 'showRender': modifier.show_render}
        row['rnaType'] = modifier.bl_rna.identifier
        if modifier.type == 'NODES':
            tree = modifier.node_group
            row['nodeGroup'] = tree.name if tree else None
            row['nodeGroupInterface'] = [
                {'name': item.name, 'identifier': item.identifier, 'direction': item.in_out,
                 'socketType': item.socket_type,
                 'groupDefaultValue': json_value(item.default_value) if hasattr(item, 'default_value') else None}
                for item in tree.interface.items_tree if item.item_type == 'SOCKET'
            ] if tree else []
            # Some actual NodesModifier RNA instances do not expose IDProperties.
            # Reading metadata must not invent empty inputs or create storage.
            try:
                keys = list(modifier.keys())
            except (TypeError, AttributeError) as exc:
                row['customInputsAccess'] = {'status': 'unsupported', 'mechanism': 'IDProperties',
                    'exceptionType': type(exc).__name__, 'reason': str(exc)}
            else:
                values = {}; unreadable = {}
                for key in keys:
                    try:
                        values[str(key)] = json_value(modifier[key])
                    except (TypeError, AttributeError, KeyError) as exc:
                        unreadable[str(key)] = {'exceptionType': type(exc).__name__, 'reason': str(exc)}
                row['customInputs'] = values
                row['customInputsAccess'] = {'status': 'partial' if unreadable else 'read',
                    'mechanism': 'IDProperties', 'keysReported': len(keys), 'unreadableKeys': unreadable}
        else:
            row['customInputsAccess'] = {'status': 'not_requested_for_non_geometry_nodes_modifier'}
        rows.append(row)
    playback = [row for row in rows if 'playback' in row['name'].lower()
                or row['name'] == 'R04 Native hair playback']
    return {'modifiers': rows,
            'activeGeometryNodesRender': [r['name'] for r in rows if r['type'] == 'NODES' and r['showRender']],
            'activeGeometryNodesViewport': [r['name'] for r in rows if r['type'] == 'NODES' and r['showViewport']],
            'playbackModifiers': playback,
            'legacyNativePlaybackPresent': any(r['name'] == 'R04 Native hair playback' for r in rows)}


def collect_mesh(obj, prefix, arrays, *, evaluated=True, graph=None,
                 require_face_sets=False, require_actual_evaluation=False):
    """Collect an actual mesh; *_world and *_triangles can build a triangle BVH."""
    value = obj.evaluated_get(graph or bpy.context.evaluated_depsgraph_get()) if evaluated else obj
    if value.type != 'MESH':
        raise RuntimeError('Actual mesh required: ' + obj.name)
    actual_evaluated = bool(getattr(value, 'is_evaluated', False))
    if evaluated and not actual_evaluated:
        if require_actual_evaluation:
            raise RuntimeError('Required evaluated mesh is unavailable in the current dependency graph: ' + obj.name)
        # An excluded object can return its original ID from evaluated_get.
        # Keep any fallback under a different prefix, so its raw vertices can
        # never be mistaken for the requested evaluated collision surface.
        fallback = collect_mesh(obj, prefix + '_unavailable_raw', arrays, evaluated=False)
        return {'object': obj.name, 'prefix': prefix, 'evaluationRequested': True,
                'evaluated': False, 'evaluationAvailable': False,
                'evaluationStatus': 'Original ID returned; evaluated surface unavailable',
                'visibility': visibility(obj), 'modifierState': modifier_state(obj),
                'rawFallback': fallback,
                'collisionUse': 'Unavailable evaluated surface. Raw fallback is diagnostic only and must not be silently substituted as a production collider.'}
    mesh = value.data
    local = add_array(arrays, prefix + '_local', coordinates(value))
    matrix = add_array(arrays, prefix + '_matrix', np.asarray(value.matrix_world, np.float64))
    world = add_array(arrays, prefix + '_world', world_positions(local, matrix))
    if not len(world) or not len(mesh.polygons):
        raise RuntimeError('Actual collision/attachment mesh is empty: ' + obj.name)
    mesh.calc_loop_triangles()
    triangles = add_array(arrays, prefix + '_triangles', np.asarray(
        [tuple(t.vertices) for t in mesh.loop_triangles], np.int32).reshape(-1, 3))
    triangle_polygon = add_array(arrays, prefix + '_triangle_polygon', np.asarray(
        [t.polygon_index for t in mesh.loop_triangles], np.int32))
    starts = add_array(arrays, prefix + '_polygon_start', np.asarray(
        [p.loop_start for p in mesh.polygons], np.int32))
    sizes = add_array(arrays, prefix + '_polygon_size', np.asarray(
        [p.loop_total for p in mesh.polygons], np.int32))
    loop_vertices = add_array(arrays, prefix + '_loop_vertex', np.asarray(
        [loop.vertex_index for loop in mesh.loops], np.int32))
    face_sets = scalar_attribute(mesh, '.sculpt_face_set', prefix + '_face_set', arrays,
                                 required=require_face_sets, expected_type='INT', expected_domain='FACE')
    if face_sets['exists']:
        labels = arrays[prefix + '_face_set']
        if len(labels) != len(mesh.polygons):
            raise RuntimeError('Actual face-set count does not match polygons')
        add_array(arrays, prefix + '_triangle_face_set', labels[triangle_polygon])
    return {'object': obj.name, 'prefix': prefix, 'evaluationRequested': evaluated,
            'evaluated': bool(evaluated and actual_evaluated),
            'evaluationAvailable': actual_evaluated if evaluated else None,
            'returnedIDIsEvaluated': getattr(value, 'is_evaluated', None),
            'vertices': len(world), 'polygons': len(mesh.polygons), 'triangles': len(triangles),
            'worldBounds': bounds(world), 'localPositionsSha256': array_sha(local),
            'worldPositionsSha256': array_sha(world), 'trianglesSha256': array_sha(triangles),
            'polygonStartsSha256': array_sha(starts), 'polygonSizesSha256': array_sha(sizes),
            'loopVerticesSha256': array_sha(loop_vertices), 'faceSets': face_sets,
            'visibility': visibility(obj), 'modifierState': modifier_state(obj)}


def collect_ear(arrays, face_set, prefix):
    """Extract ear-only triangles and vertex maps from the evaluated actual head."""
    labels = arrays['head_triangle_face_set']
    triangle_ids = np.flatnonzero(labels == face_set).astype(np.int32)
    if not len(triangle_ids):
        raise RuntimeError('Actual evaluated ear face set is empty: ' + str(face_set))
    full_triangles = arrays['head_triangles'][triangle_ids]
    ids, inverse = np.unique(full_triangles, return_inverse=True)
    ids = ids.astype(np.int32)
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    world = arrays['head_world'][ids]
    local = arrays['head_local'][ids]
    sign = 1 if face_set == 4 else -1
    if not np.all(world[:, 0] * sign > 0.):
        raise RuntimeError('Actual face-set ear side disagrees with verified 4/5 assignment')
    for suffix, value in (('world', world), ('local', local), ('triangles', triangles),
                          ('head_vertex_ids', ids), ('head_triangle_ids', triangle_ids)):
        add_array(arrays, prefix + '_' + suffix, value)
    low = int(np.argmin(world[:, 2]))
    outer = int(np.argmax(world[:, 0] * sign))
    posterior = int(np.argmax(world[:, 1]))
    anterior = int(np.argmin(world[:, 1]))
    return {'faceSet': face_set, 'side': 'positive_x' if sign > 0 else 'negative_x',
            'prefix': prefix, 'vertices': len(ids), 'triangles': len(triangles),
            'worldBounds': bounds(world), 'minimumWorldZ': float(world[low, 2]),
            'lowestPoint': world[low].tolist(), 'lowestHeadVertex': int(ids[low]),
            'outermostWorldX': float(world[outer, 0]), 'outermostPoint': world[outer].tolist(),
            'outermostHeadVertex': int(ids[outer]), 'posteriorMostWorldY': float(world[posterior, 1]),
            'posteriorMostPoint': world[posterior].tolist(), 'frontMostWorldY': float(world[anterior, 1]),
            'worldPositionsSha256': array_sha(world), 'trianglesSha256': array_sha(triangles),
            'axisConvention': 'Front camera looks along +Y; larger world Y is posterior. This is a world-frame observation at neutral REST, not an anatomical local-frame fit.'}


def collect_curves(obj, prefix, arrays, *, expected_curves=None,
                   expected_points_per_curve=None, require_roles=False):
    if obj.type != 'CURVES':
        raise RuntimeError('Actual native hair curves required: ' + obj.name)
    local = add_array(arrays, prefix + '_local', coordinates(obj))
    matrix = add_array(arrays, prefix + '_matrix', np.asarray(obj.matrix_world, np.float64))
    world = add_array(arrays, prefix + '_world', world_positions(local, matrix))
    starts = add_array(arrays, prefix + '_starts', np.asarray(
        [c.first_point_index for c in obj.data.curves], np.int32))
    lengths = add_array(arrays, prefix + '_lengths', np.asarray(
        [c.points_length for c in obj.data.curves], np.int32))
    if not len(starts) or np.any(lengths < 2) or int(lengths.sum()) != len(local):
        raise RuntimeError('Invalid raw curve topology: ' + obj.name)
    if not np.array_equal(starts, np.cumsum(np.r_[0, lengths[:-1]], dtype=np.int32)):
        raise RuntimeError('Noncontiguous raw curve starts: ' + obj.name)
    if expected_curves is not None and len(starts) != expected_curves:
        raise RuntimeError('Reviewed raw curve count changed: ' + obj.name)
    if expected_points_per_curve is not None and np.any(lengths != expected_points_per_curve):
        raise RuntimeError('Raw tail point counts changed: ' + obj.name)
    radius = scalar_attribute(obj.data, 'radius', prefix + '_radius', arrays, expected_type='FLOAT')
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    if radius['exists']:
        values = arrays[prefix + '_radius']
        if radius['domain'] not in ('POINT', 'CURVE') or len(values) != (len(local) if radius['domain'] == 'POINT' else len(starts)):
            raise RuntimeError('Unexpected actual radius domain/count: ' + obj.name)
        radius['rawLocalP0P50P100'] = np.percentile(values, [0, 50, 100]).tolist()
        radius['maximumWorldRadiusBound'] = float(values.max() * singular.max())
    roles = scalar_attribute(obj.data, ROLE, prefix + '_role', arrays, required=require_roles,
                             expected_type='INT', expected_domain='CURVE')
    if roles['exists'] and len(arrays[prefix + '_role']) != len(starts):
        raise RuntimeError('Actual curve-role count mismatch: ' + obj.name)
    if require_roles:
        unique, counts = np.unique(arrays[prefix + '_role'], return_counts=True)
        roles['counts'] = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        if not {40, 41}.issubset(set(map(int, unique))):
            raise RuntimeError('Both actual left/right gather roles must exist')
    if expected_points_per_curve is not None:
        pins = starts[:, None] + np.arange(3, dtype=np.int32)[None, :]
        add_array(arrays, prefix + '_first3_point_indices', pins)
        add_array(arrays, prefix + '_first3_world', world[pins])
    return {'object': obj.name, 'prefix': prefix, 'evaluated': False, 'curves': len(starts),
            'points': len(world), 'pointCountsP0P50P100': np.percentile(lengths, [0, 50, 100]).tolist(),
            'worldBounds': bounds(world), 'localPositionsSha256': array_sha(local),
            'worldPositionsSha256': array_sha(world), 'startsSha256': array_sha(starts),
            'lengthsSha256': array_sha(lengths), 'matrixSingularValues': singular.tolist(),
            'radius': radius, 'roles': roles, 'attributes': attribute_metadata(obj.data),
            'visibility': visibility(obj), 'modifierState': modifier_state(obj),
            'pinScope': 'The first three native points of every tail curve are exported explicitly. Their current pin interpretation must be confirmed against the active dynamics installer.' if expected_points_per_curve is not None else None}


def raw_contract(obj):
    """Same-pose raw geometry/keys/transform signature for read-only verification."""
    h = hashlib.sha256()
    h.update(coordinates(obj).tobytes())
    if obj.type == 'MESH':
        h.update(np.asarray([len(p.vertices) for p in obj.data.polygons], np.int32).tobytes())
        for polygon in obj.data.polygons:
            h.update(np.asarray(polygon.vertices, np.int32).tobytes())
    else:
        h.update(np.asarray([(c.first_point_index, c.points_length) for c in obj.data.curves], np.int32).tobytes())
    keys = getattr(obj.data, 'shape_keys', None)
    if keys:
        for key in keys.key_blocks:
            co = np.empty(len(key.data) * 3, np.float32)
            key.data.foreach_get('co', co)
            h.update(key.name.encode()); h.update(co.tobytes()); h.update(repr(float(key.value)).encode())
    return {'geometryKeysSha256': h.hexdigest(), 'data': obj.data.name,
            'matrixLocal': [list(row) for row in obj.matrix_local],
            'materials': [m.name if m else None for m in obj.data.materials],
            'parent': obj.parent.name if obj.parent else None, 'parentBone': obj.parent_bone,
            'hideRender': obj.hide_render, 'hideViewport': obj.hide_viewport,
            'hideCurrentViewLayer': obj.hide_get(), 'modifiers': modifier_state(obj)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source = (P / args.source).resolve()
    requested = (P / args.output).resolve()
    if requested.suffix.lower() not in ('.json', '.npz'):
        raise RuntimeError('--output must name a new .json or .npz observation pair')
    report_path = requested.with_suffix('.json')
    array_path = requested.with_suffix('.npz')
    for path in (source, report_path, array_path):
        path.relative_to((P / 'work/reconstruction-r02').resolve())
    if (not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source
            or report_path.exists() or array_path.exists()):
        raise RuntimeError('Fresh explicit source and unused observation outputs required')
    names = dict(COLLIDERS)
    names.update(dict(TAILS)); names.update(dict(TIES)); names['main'] = MAIN
    objects = {}
    for prefix, name in names.items():
        obj = bpy.data.objects.get(name)
        expected_type = 'CURVES' if prefix in ('main', 'tail_positive', 'tail_negative') else 'MESH'
        if obj is None or obj.type != expected_type:
            raise RuntimeError('Required actual object/type missing: ' + name + ' / ' + expected_type)
        objects[prefix] = obj
    donor = objects['main']
    if not donor.get('R11_reference_fringe_v8'):
        raise RuntimeError('An actual saved v8-or-later reference groom is required')
    rig = bpy.data.objects.get('Character_Rig')
    if rig is None or rig.type != 'ARMATURE':
        raise RuntimeError('Actual character rig required for REST observation')
    digest = file_sha(source)
    pose = rig.data.pose_position
    frame = (bpy.context.scene.frame_current, bpy.context.scene.frame_subframe)
    bpy.context.view_layer.update()
    before = {prefix: raw_contract(obj) for prefix, obj in objects.items()}
    arrays = {}
    report = {'schemaVersion': 1, 'source': str(source.relative_to(P)), 'sourceSha256': digest,
              'helperSha256': file_sha(__file__), 'blenderVersion': bpy.app.version_string,
              'sourcePose': pose, 'observationPose': 'REST', 'frame': frame,
              'savedBlend': False, 'rendered': False, 'status': 'collecting',
              'coordinateNotes': 'world arrays use actual matrix_world in neutral REST; float32 original local coordinates are retained separately. +Y is posterior for the verified character front camera.',
              'collisionSelectionPolicy': 'All requested meshes are observed, including hidden old meshes. Visibility flags and collection exclusions are preserved as evidence; no mesh is automatically selected as mandatory collision geometry.'}
    error = None
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        report['dependencyGraphMode'] = getattr(graph, 'mode', 'unknown')
        report['collisionSurfaces'] = []
        report['unavailableHiddenCollisionSurfaces'] = []
        for prefix, name in COLLIDERS:
            row = collect_mesh(objects[prefix], prefix, arrays, graph=graph,
                               require_face_sets=(prefix == 'head'),
                               require_actual_evaluation=(prefix == 'head'))
            report['collisionSurfaces'].append(row)
            if not row['evaluationAvailable']:
                if row['visibility']['renderEligibleByCollectedFlags']:
                    raise RuntimeError('Visible collision candidate has no actual evaluated surface: ' + name)
                report['unavailableHiddenCollisionSurfaces'].append(name)
        report['ears'] = [collect_ear(arrays, 4, 'ear_positive'),
                          collect_ear(arrays, 5, 'ear_negative')]
        report['tails'] = [collect_curves(objects[prefix], prefix, arrays,
                           expected_curves=2480, expected_points_per_curve=35)
                           for prefix, name in TAILS]
        report['ties'] = []
        for prefix, name in TIES:
            obj = objects[prefix]
            report['ties'].append({'object': name,
                'raw': collect_mesh(obj, prefix + '_raw', arrays, evaluated=False),
                'evaluated': collect_mesh(obj, prefix + '_eval', arrays, graph=graph,
                                           require_actual_evaluation=True)})
        report['mainGroom'] = collect_curves(donor, 'main', arrays, expected_curves=346,
                                            require_roles=True)
        if report['mainGroom']['points'] != 6770:
            raise RuntimeError('Reviewed main donor raw point count changed')
        report['allRequestedEvaluatedSurfacesAvailable'] = not report['unavailableHiddenCollisionSurfaces']
        report['status'] = ('collected' if report['allRequestedEvaluatedSurfacesAvailable']
                            else 'collected_with_unavailable_hidden_surfaces')
    except Exception as exc:
        error = exc
        report['status'] = 'failed'
        report['error'] = str(exc)
    finally:
        rig.data.pose_position = pose
        bpy.context.view_layer.update()
    after = {prefix: raw_contract(obj) for prefix, obj in objects.items()}
    differences = {prefix: {key: {'before': before[prefix].get(key), 'after': after[prefix].get(key)}
                           for key in before[prefix] if before[prefix].get(key) != after[prefix].get(key)}
                   for prefix in before if before[prefix] != after[prefix]}
    report['readOnlySourcePoseContractChanges'] = differences
    report['sourcePoseRestored'] = rig.data.pose_position == pose
    report['sourceFrameUnchanged'] = (bpy.context.scene.frame_current, bpy.context.scene.frame_subframe) == frame
    report['sourceFileUnchanged'] = file_sha(source) == digest
    if differences or not all(report[k] for k in ('sourcePoseRestored', 'sourceFrameUnchanged', 'sourceFileUnchanged')):
        report['status'] = 'failed'
        message = 'Read-only source/pose contract changed'
        report['error'] = (report.get('error', '') + '; ' + message).strip('; ')
        error = error or RuntimeError(message)
    if arrays:
        with array_path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        report['arrays'] = str(array_path.relative_to(P))
        report['arraysSha256'] = file_sha(array_path)
        report['arrayManifest'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                        'sha256': array_sha(value)} for key, value in arrays.items()}
    with report_path.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_REFERENCE_TIE_ALIGNMENT_OBSERVED ' + json.dumps(
        {'report': str(report_path.relative_to(P)), 'sourceSha256': digest,
         'status': report['status'], 'arrays': len(arrays), 'savedBlend': False}), flush=True)
    if error:
        raise error


if __name__ == '__main__':
    main()
