"""Embed one checked PBD guide cache in Blender shape keys + Geometry Nodes.

No handler, external cache, save, render, or source-point rewrite is needed for
subsequent playback. run() preserves the current body action and scene frame.
Use enable_for_action(select=True) to select/validate the recorded body clip;
disable() before selecting a different action. This is deliberately one clip,
not an automatic action-to-physics switcher. restore() removes only this pass's
modifiers/mapping attributes and retains its guide/Action/node data as history.

The parent integrator must call validate_evaluated() in actual Blender before
claiming parity with secondary_hair_r02.py or delivering the saved source.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
try:
    import bpy
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

VERSION = 1
RIG = 'Character_Rig'
GUIDE = 'R04 Native hair guide cache'
MODIFIER = 'R04 Native hair playback'
OWNER = 'r04_native_hair_owner'
RECORD = 'r04_native_hair_record'
ENABLE = 'R04_native_hair_enabled'
RIG_GUIDE = 'R04_native_hair_guide'
ATTRS = tuple(f'r04_hair_{kind}_{i}' for kind in ('index', 'weight') for i in range(3))
DEFAULT_CACHE = 'work/reconstruction-r02/hair-performance-r03/R03-Performance-review.guides.npz'


def _digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        q = np.ascontiguousarray(array)
        h.update(str(q.shape).encode()); h.update(q.dtype.str.encode()); h.update(q.tobytes())
    return h.hexdigest()


def _fcurves(action):
    curves = []
    if hasattr(action, 'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, 'channelbags'):
                    for bag in strip.channelbags:
                        curves.extend(bag.fcurves)
    if not curves and hasattr(action, 'fcurves'):
        curves = list(action.fcurves)
    return curves


def _action_fingerprint(action):
    curves = _fcurves(action)
    if not curves:
        raise RuntimeError('No inspectable action curves: ' + action.name)
    h = hashlib.sha256()
    for curve in sorted(curves, key=lambda c: (c.data_path, c.array_index)):
        h.update(f'{curve.data_path}:{curve.array_index}'.encode())
        points = np.array([(*k.co, *k.handle_left, *k.handle_right)
                           for k in curve.keyframe_points], dtype=np.float32)
        h.update(points.tobytes())
        h.update('|'.join(k.interpolation for k in curve.keyframe_points).encode())
    return h.hexdigest()


def _cache_path(relative):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Use an explicit project-relative cache path without ..')
    path = (project_path() / relative).resolve()
    path.relative_to((project_path() / 'work').resolve())
    return path


def load_cache(relative=DEFAULT_CACHE):
    """Read/validate numeric cache files only; this also works outside Blender."""
    path = _cache_path(relative)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays.pop('meta')))
    if meta.get('version') != 1:
        raise RuntimeError('Unsupported PBD cache version')
    base_path = (path.parent / meta['baseFile']).resolve()
    base_path.relative_to(path.parent.resolve())
    with np.load(base_path, allow_pickle=False) as data:
        base = {name: data[name].copy() for name in data.files}
    base_meta = json.loads(str(base.pop('meta')))
    if _digest(*[base[name] for name in sorted(base)]) != meta['baseHash']:
        raise RuntimeError('Static cache checksum mismatch')
    if base_meta.get('baseHash') != meta['baseHash']:
        raise RuntimeError('Static metadata checksum mismatch')
    frames = arrays['frames']; delta = arrays['guide_delta']
    if frames.ndim != 1 or not 2 <= len(frames) <= 600 or np.any(np.diff(frames) != 1):
        raise RuntimeError('Need a bounded consecutive integer-frame cache')
    if delta.ndim != 5 or delta.shape[0] != len(frames) or delta.shape[-1] != 3:
        raise RuntimeError('Unexpected guide delta dimensions')
    tails, guides, points = delta.shape[1:4]
    if tails != len(meta['objects']) or guides > 48 or points < 8:
        raise RuntimeError('Unexpected guide scope')
    if arrays['object_matrices'].shape != (len(frames), tails, 4, 4):
        raise RuntimeError('Missing sampled attachment matrices')
    if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(arrays['object_matrices'])):
        raise RuntimeError('Non-finite cache values')
    pin = int(meta['params']['pinned_points'])
    if not 1 <= pin <= 4:
        raise RuntimeError('Invalid root pin count')
    for i, name in enumerate(meta['objects']):
        rest, ids, weights = (base[f'{key}_{i}'] for key in ('rest', 'ids', 'weights'))
        if rest.ndim != 3 or rest.shape[1:] != (points, 3):
            raise RuntimeError('Invalid authored strand coordinates: ' + name)
        if ids.shape != (len(rest), 3) or weights.shape != ids.shape:
            raise RuntimeError('Invalid guide map: ' + name)
        if np.any(ids < 0) or np.any(ids >= guides):
            raise RuntimeError('Guide index outside cache')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.max(np.abs(weights.sum(axis=1) - 1)) > 2e-6:
            raise RuntimeError('Invalid guide weights')
        if _digest(rest) != base_meta['geometryHashes'][name]:
            raise RuntimeError('Static strand fingerprint mismatch: ' + name)
    return path, meta, arrays, base, base_meta


def _point_maps(base, meta, tail, guides, points):
    ids = base[f'ids_{tail}'].astype(np.int32)
    weights = base[f'weights_{tail}'].astype(np.float32)
    count = len(ids)
    sample = np.arange(points, dtype=np.int32)
    indices = tail * guides * points + ids[:, :, None] * points + sample[None, None, :]
    indices = indices.transpose(0, 2, 1).reshape(count * points, 3)
    mapped_weights = np.repeat(weights[:, None, :], points, axis=1)
    mapped_weights[:, :int(meta['params']['pinned_points']), :] = 0
    return indices, np.ascontiguousarray(mapped_weights.reshape(count * points, 3))


def expected_positions(relative=DEFAULT_CACHE, frame=1.0):
    """Independent original-cache oracle for the integrator's GN comparison."""
    _, meta, arrays, base, _ = load_cache(relative)
    first, last = int(arrays['frames'][0]), int(arrays['frames'][-1])
    if not first <= frame <= last:
        raise ValueError('Frame outside recorded range')
    pos = float(frame) - first; lo = int(math.floor(pos)); hi = min(lo + 1, len(arrays['frames']) - 1)
    t = pos - lo
    delta = arrays['guide_delta'][lo].astype(np.float32) * (1 - t) + arrays['guide_delta'][hi].astype(np.float32) * t
    result = {}
    for i, name in enumerate(meta['objects']):
        displacement = np.einsum('nkpq,nk->npq', delta[i][base[f'ids_{i}']], base[f'weights_{i}'])
        co = np.ascontiguousarray(base[f'rest_{i}'] + displacement, dtype=np.float32)
        pin = int(meta['params']['pinned_points']); co[:, :pin] = base[f'rest_{i}'][:, :pin]
        result[name] = co.reshape(-1, 3)
    return result


def numerical_probe(relative=DEFAULT_CACHE):
    """Compare embedded index/weight evaluation with the original cache formula."""
    _, meta, arrays, base, _ = load_cache(relative)
    tails, guides, points = arrays['guide_delta'].shape[1:4]
    frames = arrays['frames']
    samples = sorted(set([float(frames[0]), float(frames[-1]), float(frames[len(frames) // 2]),
                          float(frames[len(frames) // 3]) + .375]))
    worst = 0.; root = 0.
    for frame in samples:
        pos = frame - int(frames[0]); lo = int(math.floor(pos)); hi = min(lo + 1, len(frames) - 1); t = pos - lo
        delta = (arrays['guide_delta'][lo].astype(np.float32) * (1 - t) +
                 arrays['guide_delta'][hi].astype(np.float32) * t).reshape(-1, 3)
        expected = expected_positions(relative, frame)
        for tail, name in enumerate(meta['objects']):
            ids, weights = _point_maps(base, meta, tail, guides, points)
            actual = base[f'rest_{tail}'].reshape(-1, 3) + (delta[ids] * weights[:, :, None]).sum(axis=1)
            worst = max(worst, float(np.linalg.norm(actual - expected[name], axis=1).max()))
            root = max(root, float(np.abs(actual.reshape(-1, points, 3)[:, :int(meta['params']['pinned_points'])] - base[f'rest_{tail}'][:, :int(meta['params']['pinned_points'])]).max()))
    if worst > 1e-7 or root != 0:
        raise RuntimeError('Index embedding differs from PBD playback')
    return {'scope': 'NumPy index/weight parity only; Blender graph still requires evaluation',
            'sampleFrames': samples, 'maxErrorMeters': worst, 'rootErrorMeters': root,
            'guideVertices': tails * guides * points,
            'shapeCoordinateBytes': len(frames) * tails * guides * points * 3 * 4,
            'pointMappingBytes': sum(base[f'rest_{i}'].shape[0] * points * 6 * 4 for i in range(tails))}


def _guard(require_work_copy=False):
    if bpy is None:
        raise RuntimeError('This operation requires the integrator Blender session')
    file = Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath:
        raise RuntimeError('A saved project source is required')
    file.relative_to(project_path().resolve())
    if require_work_copy:
        if file == (project_path() / 'work/character.blend').resolve():
            raise RuntimeError('Build/restore only in a saved independent work copy')
        file.relative_to((project_path() / 'work').resolve())
    if bpy.context.mode != 'OBJECT':
        raise RuntimeError('Object Mode is required')
    rig = bpy.data.objects.get(RIG)
    if rig is None or rig.type != 'ARMATURE' or rig.library:
        raise RuntimeError('Local Character_Rig required')
    return rig


def _enum(owner, name, value):
    if value not in {item.identifier for item in owner.bl_rna.properties[name].enum_items}:
        raise RuntimeError(f'Unsupported {owner.bl_rna.identifier}.{name}: {value}')
    setattr(owner, name, value)


def _read_points(obj):
    data = obj.data
    arr = np.empty(len(data.position_data) * 3, dtype=np.float32)
    data.position_data.foreach_get('vector', arr)
    return arr.reshape(-1, 3)


def _shape_action(guide, frames, delta):
    guide.shape_key_add(name='Basis', from_mix=False)
    basis = guide.data.shape_keys.key_blocks[0]
    for fi, frame in enumerate(frames):
        key = guide.shape_key_add(name=f'Cache frame {int(frame):04d}', from_mix=False)
        key.relative_key = basis
        key.data.foreach_set('co', np.ascontiguousarray(delta[fi], dtype=np.float32).ravel())
        key.value = 0.; key.slider_min = 0.; key.slider_max = 1.
        for when, value in ((int(frame) - 1, 0.), (int(frame), 1.), (int(frame) + 1, 0.)):
            key.value = value; key.keyframe_insert(data_path='value', frame=when, group='Embedded guide cache')
        key.value = 0.
    action = guide.data.shape_keys.animation_data.action
    action.name = 'R04 Embedded hair guide offsets'
    action.use_fake_user = True
    for curve in _fcurves(action):
        curve.extrapolation = 'CONSTANT'
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
    return action


def _node_group(guide, group):
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = group.nodes, group.links
    source = nodes.new('NodeGroupInput'); source.location = (-800, 300)
    output = nodes.new('NodeGroupOutput'); output.location = (720, 300)
    object_info = nodes.new('GeometryNodeObjectInfo'); object_info.location = (-800, 80)
    _enum(object_info, 'transform_space', 'ORIGINAL')
    object_info.inputs['Object'].default_value = guide
    object_info.inputs['As Instance'].default_value = False
    position = nodes.new('GeometryNodeInputPosition'); position.location = (-800, -120)
    products = []
    for i in range(3):
        index = nodes.new('GeometryNodeInputNamedAttribute'); _enum(index, 'data_type', 'INT')
        index.inputs['Name'].default_value = f'r04_hair_index_{i}'; index.location = (-590, -200 - i * 220)
        weight = nodes.new('GeometryNodeInputNamedAttribute'); _enum(weight, 'data_type', 'FLOAT')
        weight.inputs['Name'].default_value = f'r04_hair_weight_{i}'; weight.location = (-160, -300 - i * 220)
        sample = nodes.new('GeometryNodeSampleIndex'); _enum(sample, 'data_type', 'FLOAT_VECTOR'); _enum(sample, 'domain', 'POINT')
        sample.clamp = False; sample.location = (-380, -80 - i * 220)
        links.new(object_info.outputs['Geometry'], sample.inputs['Geometry'])
        links.new(position.outputs['Position'], sample.inputs['Value'])
        links.new(index.outputs['Attribute'], sample.inputs['Index'])
        scale = nodes.new('ShaderNodeVectorMath'); _enum(scale, 'operation', 'SCALE'); scale.location = (60, -60 - i * 220)
        links.new(sample.outputs['Value'], scale.inputs[0]); links.new(weight.outputs['Attribute'], scale.inputs['Scale'])
        products.append(scale.outputs['Vector'])
    add01 = nodes.new('ShaderNodeVectorMath'); _enum(add01, 'operation', 'ADD'); add01.location = (270, 40)
    links.new(products[0], add01.inputs[0]); links.new(products[1], add01.inputs[1])
    add2 = nodes.new('ShaderNodeVectorMath'); _enum(add2, 'operation', 'ADD'); add2.location = (440, 40)
    links.new(add01.outputs['Vector'], add2.inputs[0]); links.new(products[2], add2.inputs[1])
    setpos = nodes.new('GeometryNodeSetPosition'); setpos.location = (510, 300)
    links.new(source.outputs['Geometry'], setpos.inputs['Geometry'])
    links.new(add2.outputs['Vector'], setpos.inputs['Offset'])
    links.new(setpos.outputs['Geometry'], output.inputs['Geometry'])
    return group


def _visibility_drivers(modifier, rig, start, end):
    for prop in ('show_viewport', 'show_render'):
        curve = modifier.driver_add(prop)
        curve.mute = False
        driver = curve.driver
        driver.type = 'SCRIPTED'; driver.use_self = False
        # driver_add may return an existing curve at this RNA path. Only this
        # newly owned modifier is in scope, so remove any stale variables.
        for existing in list(driver.variables):
            driver.variables.remove(existing)
        variable = driver.variables.new(); variable.name = 'enabled'; variable.type = 'SINGLE_PROP'
        target = variable.targets[0]
        target.id_type = 'OBJECT'; target.id = rig; target.data_path = f'["{ENABLE}"]'
        # Compile only after the variable target exists. Assigning the expression
        # first can leave a cached invalid driver until it is explicitly changed.
        # Numeric multiplication was verified in Blender 5.2; the boolean-chain
        # expression initially left both visibility channels invalid there.
        driver.expression = f'enabled * (frame >= {start}) * (frame <= {end})'


def _check_visibility_drivers(objects_and_modifiers, rig, start, end):
    """Check evaluated state; permit one bounded two-stage driver recompilation.

    No visibility flags, source points, action, or frame number are changed.
    Same-frame frame_set is needed to rebuild animation relations in a fresh
    background session; update_tag alone can retain a new driver's invalid flag.
    """
    scene = bpy.context.scene; view_layer = bpy.context.view_layer
    current_frame, subframe = scene.frame_current, scene.frame_subframe
    frame = current_frame + subframe
    expected = bool(float(rig[ENABLE]) * (start <= frame <= end))
    curves = []
    for obj, modifier in objects_and_modifiers:
        for prop in ('show_viewport', 'show_render'):
            path = modifier.path_from_id(prop)
            curve = obj.animation_data.drivers.find(path) if obj.animation_data else None
            curves.append((obj, modifier, prop, path, curve))

    def refresh(same_frame=False):
        rig.update_tag()
        for obj, _ in objects_and_modifiers:
            obj.update_tag()
        if same_frame:
            scene.frame_set(current_frame, subframe=subframe)
        view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        graph.update()
        return graph

    def snapshot(graph, phase):
        rows = []
        for obj, modifier, prop, path, curve in curves:
            visibility = {}
            try:
                visibility['objectHideSet'] = obj.hide_get(view_layer=view_layer)
                visibility['visibleGet'] = obj.visible_get(view_layer=view_layer)
            except RuntimeError as exc:
                visibility['visibilityQueryError'] = repr(exc)
            row = {'object': obj.name, 'path': path, 'property': prop,
                   'originalValue': bool(getattr(modifier, prop)),
                   'evaluatedValue': None, 'driverValid': None, 'evaluatedDriverValid': None,
                   'fcurveValid': None, 'evaluatedFcurveValid': None, 'expectedValue': expected,
                   'objectHideViewport': obj.hide_viewport, 'objectHideRender': obj.hide_render,
                   'inViewLayer': obj.name in view_layer.objects,
                   'collections': [{'name': col.name, 'hideViewport': col.hide_viewport,
                                    'hideRender': col.hide_render} for col in obj.users_collection],
                   'variables': []}
            row.update(visibility)
            if curve:
                driver = curve.driver
                row.update(driverValid=driver.is_valid, fcurveValid=curve.is_valid,
                           driverType=driver.type, muted=curve.mute,
                           expression=driver.expression,
                           simpleExpression=getattr(driver, 'is_simple_expression', None))
                for variable in driver.variables:
                    values = []
                    for target in variable.targets:
                        value = {'id': target.id.name if target.id else None,
                                 'idType': target.id_type, 'dataPath': target.data_path}
                        try:
                            resolved = target.id.path_resolve(target.data_path) if target.id else None
                            value['resolvedValue'] = resolved if isinstance(resolved, (str, int, float, bool, type(None))) else repr(resolved)
                        except (ValueError, TypeError, AttributeError) as exc:
                            value['resolutionError'] = repr(exc)
                        values.append(value)
                    row['variables'].append({'name': variable.name, 'type': variable.type, 'targets': values})
            try:
                evaluated = obj.evaluated_get(graph)
                evaluated_modifier = evaluated.modifiers.get(modifier.name)
                if evaluated_modifier:
                    row['evaluatedValue'] = bool(getattr(evaluated_modifier, prop))
                evaluated_curve = evaluated.animation_data.drivers.find(path) if evaluated.animation_data else None
                if evaluated_curve:
                    row['evaluatedDriverValid'] = evaluated_curve.driver.is_valid
                    row['evaluatedFcurveValid'] = evaluated_curve.is_valid
            except (RuntimeError, ReferenceError) as exc:
                row['evaluationError'] = repr(exc)
            rows.append(row)
        return {'phase': phase, 'frame': frame, 'rigToggle': float(rig[ENABLE]),
                'viewLayer': view_layer.name, 'background': bpy.app.background,
                'autoexecFail': getattr(bpy.app, 'autoexec_fail', None),
                'autoexecFailMessage': getattr(bpy.app, 'autoexec_fail_message', None),
                'rows': rows}

    def healthy(state):
        # Evaluated data is authoritative for playback. Original RNA is retained
        # in diagnostics because it can lag evaluation in background mode.
        return all((row['evaluatedFcurveValid'] if row['evaluatedFcurveValid'] is not None else row['fcurveValid']) and not row.get('muted', False)
                   and (row['evaluatedDriverValid'] if row['evaluatedDriverValid'] is not None else row['driverValid'])
                   and row['evaluatedValue'] == expected for row in state['rows'])

    initial = snapshot(refresh(), 'initial-evaluation')
    if healthy(initial):
        return {'boundedRecompileUsed': False, 'states': [initial]}
    # Match the successful R04 live repair, once only. Do not unhide objects or
    # suppress an unresolved invalid driver merely because its object is hidden.
    for _, _, _, _, curve in curves:
        if curve:
            curve.driver.expression = 'enabled'
    warmup = snapshot(refresh(same_frame=True), 'enabled-only-recompile')
    expression = f'enabled * (frame >= {start}) * (frame <= {end})'
    for _, _, _, _, curve in curves:
        if curve:
            curve.driver.expression = expression
    final = snapshot(refresh(same_frame=True), 'range-expression-recompile')
    result = {'boundedRecompileUsed': True, 'states': [initial, warmup, final]}
    print('R04_HAIR_DRIVER_DIAGNOSTICS ' + json.dumps(result, ensure_ascii=False))
    if not healthy(final):
        raise RuntimeError('Native hair visibility driver remained invalid/stale after one bounded refresh: ' + json.dumps(result, ensure_ascii=False))
    return result


def _recorded_state(guide, rig, check_geometry=True):
    record = json.loads(guide[RECORD]); meta = record['cacheMetadata']
    action = rig.animation_data.action if rig.animation_data else None
    if action is None or action.name != meta['action']:
        raise RuntimeError('Select the recorded body action: ' + meta['action'])
    if _action_fingerprint(action) != meta['actionSha256']:
        raise RuntimeError('The recorded body action changed; rebuild this cache')
    if rig.data.pose_position != 'POSE':
        raise RuntimeError('Use Pose Position for cached motion playback')
    if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA is outside this single-action cache scope')
    scene = bpy.context.scene
    if abs(scene.render.fps / scene.render.fps_base - meta['fps']) > 1e-6:
        raise RuntimeError('Scene fps differs from the recorded cache')
    frames = np.array(guide['r04_cache_frames'], dtype=np.int32)
    frame = scene.frame_current + scene.frame_subframe
    if not frames[0] <= frame <= frames[-1]:
        raise RuntimeError('Current frame is outside the embedded cache')
    pos = frame - int(frames[0]); lo = int(math.floor(pos)); hi = min(lo + 1, len(frames) - 1); t = pos - lo
    matrices = np.array(guide['r04_cache_matrices'], dtype=np.float32).reshape(len(frames), len(meta['objects']), 4, 4)
    tolerance = 2e-4
    values = (np.array(guide['r04_cache_ik'], dtype=np.float32).reshape(len(frames), 4)
              if 'r04_cache_ik' in guide else None)

    def current_matrices():
        return np.array([np.array(bpy.data.objects[name].matrix_world)
                         for name in meta['objects']], dtype=float)

    def compare(actual, expected):
        rows = {}
        for i, name in enumerate(meta['objects']):
            difference = actual[i] - expected[i]
            error = float(np.abs(difference).max())
            rows[name] = {'maxAbsMatrixElementError': error,
                          'translationDifferenceMeters': float(np.linalg.norm(difference[:3, 3]))}
            if error > tolerance:
                rows[name]['actualMatrix'] = actual[i].tolist()
                rows[name]['expectedMatrix'] = expected[i].tolist()
        return rows

    def check_anchor(index):
        row = {'frame': int(frames[index]), 'objects': compare(current_matrices(), matrices[index])}
        if values is not None:
            master = rig.pose.bones.get('CTRL_R02_Master')
            actual = np.array([master.get('ik_' + key, 0) if master else 0
                               for key in ('leg_L', 'leg_R', 'arm_L', 'arm_R')])
            row['ikSwitches'] = {'maxAbsError': float(np.abs(actual - values[index]).max()),
                                'actual': actual.tolist(), 'expected': values[index].tolist()}
        return row

    evidence = {'requestedFrame': frame, 'actionFingerprintMatches': True,
                'matrixElementTolerance': tolerance,
                'mode': 'exact-integer-cache-sample' if t == 0 else 'fingerprint-and-adjacent-integer-samples',
                'anchorChecks': []}
    for i, name in enumerate(meta['objects']):
        obj = bpy.data.objects[name]
        if check_geometry:
            original = _read_points(obj).reshape(record['strandCounts'][i], record['pointsPerStrand'], 3)
            if _digest(original) != record['geometryHashes'][name]:
                raise RuntimeError(f'The authored groom has changed at frame {frame}: {name}')
    if t == 0:
        evidence['anchorChecks'].append(check_anchor(lo))
    else:
        # An action can contain subframe keys (for example 120 Hz mocap in a
        # 30 fps scene). Its true fractional pose need not equal linear/SVD
        # interpolation of integer matrices. Keep that difference informational;
        # validate unchanged action data and both exact cached anchor poses.
        before = current_matrices()
        linear = matrices[lo].astype(float) * (1 - t) + matrices[hi].astype(float) * t
        u, _, vh = np.linalg.svd(linear[:, :3, :3]); linear[:, :3, :3] = u @ vh
        evidence['fractionalDifferenceFromLegacyLinearSVD'] = compare(before, linear)
        evidence['fractionalMatrixInterpolationIsValidationCriterion'] = False
        evidence['scope'] = ('True fractional body pose is retained. Attachment validity uses the full action '
                             'fingerprint and two integer cache samples; local hair-offset interpolation '
                             'is tested separately against the actual Geometry Nodes output.')
        old_frame, old_subframe = scene.frame_current, scene.frame_subframe
        try:
            for index in dict.fromkeys((lo, hi)):
                scene.frame_set(int(frames[index]), subframe=0.)
                bpy.context.view_layer.update()
                evidence['anchorChecks'].append(check_anchor(index))
        finally:
            scene.frame_set(old_frame, subframe=old_subframe)
            bpy.context.view_layer.update()
        evidence['fractionalRestoreError'] = compare(current_matrices(), before)
        evidence['restoredFrame'] = scene.frame_current + scene.frame_subframe

    invalid = any(check['objects'][name]['maxAbsMatrixElementError'] > tolerance
                  for check in evidence['anchorChecks'] for name in meta['objects'])
    invalid_ik = any(check.get('ikSwitches', {}).get('maxAbsError', 0.) > tolerance
                     for check in evidence['anchorChecks'])
    invalid_restore = any(row['maxAbsMatrixElementError'] > tolerance
                          for row in evidence.get('fractionalRestoreError', {}).values())
    if invalid or invalid_ik or invalid_restore:
        print('R04_HAIR_ATTACHMENT_DIAGNOSTICS ' + json.dumps(evidence, ensure_ascii=False))
        raise RuntimeError('Body/head attachment, IK or fractional restoration differs from recorded state: ' +
                           json.dumps(evidence, ensure_ascii=False))
    record['currentStateValidation'] = evidence
    return record


def _active_guide(rig):
    guide = bpy.data.objects.get(rig.get(RIG_GUIDE, ''))
    if guide is None or not guide.get('r04_native_hair_active'):
        raise RuntimeError('No active R04 embedded guide cache')
    return guide


def run(cache_relative=DEFAULT_CACHE):
    """Build in memory only; preserve source curves and current action/frame."""
    rig = _guard(require_work_copy=True)
    if RIG_GUIDE in rig or ENABLE in rig:
        raise RuntimeError('A native cache/toggle already exists; restore it explicitly first')
    if any(getattr(h, '_r02_secondary_hair', False) for h in bpy.app.handlers.frame_change_post):
        raise RuntimeError('Detach the old Python hair playback handler first')
    path, meta, arrays, base, base_meta = load_cache(cache_relative)
    action = bpy.data.actions.get(meta['action'])
    if action is None or _action_fingerprint(action) != meta['actionSha256']:
        raise RuntimeError('The recorded body action is absent or modified')
    objects = []
    tails, guides, points = arrays['guide_delta'].shape[1:4]
    for i, name in enumerate(meta['objects']):
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'CURVES' or obj.library or obj.data.library or obj.data.users != 1:
            raise RuntimeError('Expected local single-user native strands: ' + name)
        if obj.get('r02_secondary_hair_session') or obj.get(OWNER):
            raise RuntimeError('Detach/restore previous hair playback first')
        if len(obj.modifiers):
            raise RuntimeError('Existing modifiers require an explicit integration: ' + name)
        if any(obj.data.attributes.get(n) for n in ATTRS):
            raise RuntimeError('Reserved point attributes already exist')
        if any(curve.points_length != points or curve.first_point_index != j * points
               for j, curve in enumerate(obj.data.curves)):
            raise RuntimeError('Strand topology changed: ' + name)
        original = _read_points(obj).reshape(-1, points, 3)
        if original.shape != base[f'rest_{i}'].shape or _digest(original) != base_meta['geometryHashes'][name]:
            raise RuntimeError('Authored strand positions differ from the cache: ' + name)
        objects.append(obj)
    owner = hashlib.sha256(path.read_bytes()).hexdigest()
    start, end = int(arrays['frames'][0]), int(arrays['frames'][-1])
    scene = bpy.context.scene; old_frame = scene.frame_current; old_subframe = scene.frame_subframe
    created_mods = []; created_attrs = []; guide = mesh = group = key_action = text = None
    try:
        mesh = bpy.data.meshes.new('R04 native guide offset points')
        mesh.from_pydata([(0., 0., 0.)] * (tails * guides * points), [], []); mesh.update()
        guide = bpy.data.objects.new(GUIDE, mesh); scene.collection.objects.link(guide)
        guide.hide_render = True; guide.hide_select = True
        _enum(guide, 'display_type', 'WIRE')
        key_action = _shape_action(guide, arrays['frames'], arrays['guide_delta'])
        key_action.name = 'R04 Embedded hair / ' + meta['action']
        group = bpy.data.node_groups.new('R04 Native hair cached offsets', 'GeometryNodeTree')
        _node_group(guide, group)
        rig[ENABLE] = 0.
        rig.id_properties_ui(ENABLE).update(min=0., max=1., soft_min=0., soft_max=1.,
            description='仅用于已记录动作；切换动作前关闭。使用 enable_for_action() 检查并启用。')
        for i, obj in enumerate(objects):
            indices, weights = _point_maps(base, meta, i, guides, points)
            for j in range(3):
                attr = obj.data.attributes.new(f'r04_hair_index_{j}', 'INT', 'POINT')
                created_attrs.append((obj, attr.name)); attr.data.foreach_set('value', np.ascontiguousarray(indices[:, j], dtype=np.int32))
                attr = obj.data.attributes.new(f'r04_hair_weight_{j}', 'FLOAT', 'POINT')
                created_attrs.append((obj, attr.name)); attr.data.foreach_set('value', np.ascontiguousarray(weights[:, j], dtype=np.float32))
            modifier = obj.modifiers.new(MODIFIER, 'NODES'); created_mods.append((obj, modifier))
            modifier.node_group = group; _visibility_drivers(modifier, rig, start, end)
            obj[OWNER] = owner; obj.data.update_tag(); obj.update_tag()
        record = {'version': VERSION, 'owner': owner, 'cacheRelativePath': str(path.relative_to(project_path())),
                  'cacheSha256': owner, 'staticSha256': hashlib.sha256((path.parent / meta['baseFile']).read_bytes()).hexdigest(),
                  'constructionSource': str(Path(bpy.data.filepath).relative_to(project_path())),
                  'cacheMetadata': meta, 'geometryHashes': base_meta['geometryHashes'],
                  'guideVertices': tails * guides * points, 'pointsPerStrand': points,
                  'strandCounts': [len(base[f'rest_{i}']) for i in range(tails)],
                  'shapeFrames': len(arrays['frames']), 'shapeAction': key_action.name,
                  'nodeGroup': group.name, 'modifiers': {obj.name: modifier.name for obj, modifier in created_mods},
                  'attributes': list(ATTRS), 'playbackDependency': 'Embedded Blender datablocks only; no Python handler or external NPZ needed after saving',
                  'limitations': ['Only this recorded body action is supported; selecting other actions does not auto-switch the cache',
                                  'Disable the rig toggle before changing an action; enable_for_action validates the matching clip',
                                  'Groom, bind/parent transforms or body-action edits invalidate the recorded dynamics',
                                  'Only the two tails use this cache; scalp/fringe remain head-attached',
                                  'Source PBD sphere collision limitations remain; native playback adds no new physical accuracy',
                                  'Geometry Nodes and drivers require actual evaluated-coordinate and reopen verification'],
                  'evaluatedValidated': False, 'savedBlend': False}
        guide[RECORD] = json.dumps(record, ensure_ascii=False)
        guide['r04_cache_frames'] = arrays['frames'].astype(int).tolist()
        guide['r04_cache_matrices'] = arrays['object_matrices'].ravel().astype(float).tolist()
        if 'ik_switches' in arrays:
            guide['r04_cache_ik'] = arrays['ik_switches'].ravel().astype(float).tolist()
        guide['r04_native_hair_active'] = True; rig[RIG_GUIDE] = guide.name
        text = bpy.data.texts.new('R04 native hair playback - readme')
        text.write('原生毛发缓存\n\n适用动作：' + meta['action'] + f'\n帧范围：{start}–{end}，{meta["fps"]} fps。\n'
                   '发丝点位通过内嵌引导形态键和 Geometry Nodes 求值，保存后无需 Python handler 或外部 NPZ。\n'
                   'Character_Rig 自定义属性 R04_native_hair_enabled 控制两侧毛发。\n'
                   '切换动作前必须关闭；该开关不会自动识别其他动作。\n'
                   '项目脚本 native_hair_playback_r04.py 的 enable_for_action(select=True) 可选择并验证记录动作。\n'
                   'validate_evaluated() 用原缓存逐点对照；restore() 撤销本次绑定并保留引导数据作历史。\n'
                   '内嵌播放不等于碰撞、外观或客户验收通过。\n')
        scene.frame_set(old_frame, subframe=old_subframe); bpy.context.view_layer.update()
        try:
            _recorded_state(guide, rig); rig[ENABLE] = 1.; status = 'Enabled for the matching current action/frame'
        except RuntimeError as exc:
            rig[ENABLE] = 0.; status = 'Built but disabled: ' + str(exc)
        driver_check = _check_visibility_drivers(created_mods, rig, start, end)
        record['visibilityDriverCheck'] = driver_check
        guide[RECORD] = json.dumps(record, ensure_ascii=False)
        report = dict(record, guideObject=guide.name, initiallyEnabled=bool(rig[ENABLE]), initialStatus=status)
        print(json.dumps(report, ensure_ascii=False))
        return report
    except Exception:
        for obj, modifier in reversed(created_mods):
            for prop in ('show_viewport', 'show_render'):
                try: modifier.driver_remove(prop)
                except (TypeError, RuntimeError): pass
            obj.modifiers.remove(modifier)
            if OWNER in obj: del obj[OWNER]
        for obj, name in reversed(created_attrs):
            attr = obj.data.attributes.get(name)
            if attr: obj.data.attributes.remove(attr)
        if key_action is None and mesh and mesh.shape_keys and mesh.shape_keys.animation_data:
            key_action = mesh.shape_keys.animation_data.action
        if guide: bpy.data.objects.remove(guide, do_unlink=True)
        if mesh and mesh.users == 0: bpy.data.meshes.remove(mesh)
        if group and group.users == 0: bpy.data.node_groups.remove(group)
        if key_action and key_action.users <= int(key_action.use_fake_user): bpy.data.actions.remove(key_action)
        if text: bpy.data.texts.remove(text)
        for prop in (ENABLE, RIG_GUIDE):
            if prop in rig: del rig[prop]
        scene.frame_set(old_frame, subframe=old_subframe); bpy.context.view_layer.update()
        raise


def disable():
    rig = _guard(); _active_guide(rig)
    rig[ENABLE] = 0.; rig.update_tag(); bpy.context.view_layer.update()
    return {'enabled': False, 'savedBlend': False}


def enable_for_action(action_name=None, select=False, frame=None):
    """Validate then enable. Selecting the recorded action is explicit/optional."""
    rig = _guard(); guide = _active_guide(rig); record = json.loads(guide[RECORD])
    expected = record['cacheMetadata']['action']
    if action_name is not None and action_name != expected:
        raise ValueError('This native cache only supports ' + expected)
    rig[ENABLE] = 0.; rig.update_tag()
    if select:
        action = bpy.data.actions.get(expected)
        if action is None: raise RuntimeError('Recorded action missing')
        rig.animation_data_create(); rig.animation_data.action = action
        if hasattr(action, 'slots'):
            slots = [slot for slot in action.slots if slot.target_id_type == 'OBJECT']
            if len(slots) != 1: raise RuntimeError('Expected exactly one object action slot')
            rig.animation_data.action_slot = slots[0]
        rig.data.pose_position = 'POSE'
    if frame is not None:
        integer = math.floor(frame); bpy.context.scene.frame_set(integer, subframe=float(frame) - integer)
    bpy.context.view_layer.update(); _recorded_state(guide, rig)
    rig[ENABLE] = 1.; rig.update_tag(); bpy.context.view_layer.update()
    return {'enabled': True, 'action': expected, 'savedBlend': False}


def validate_evaluated(cache_relative=DEFAULT_CACHE, frames=None, tolerance=2e-6):
    """Actual native CURVES positions versus original PBD oracle; no render/save."""
    rig = _guard(); guide = _active_guide(rig); record = json.loads(guide[RECORD])
    path, meta, arrays, _, _ = load_cache(cache_relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['cacheSha256']:
        raise RuntimeError('Validation source differs from the embedded cache')
    if frames is None:
        a, b = int(arrays['frames'][0]), int(arrays['frames'][-1])
        frames = sorted(set([a, a + (b - a) // 3 + .375, a + (b - a) // 2, b]))
    scene = bpy.context.scene; old_frame = scene.frame_current; old_subframe = scene.frame_subframe; old_enable = float(rig[ENABLE])
    rows = []; maximum = 0.
    try:
        for frame in frames:
            scene.frame_set(math.floor(frame), subframe=float(frame) - math.floor(frame))
            bpy.context.view_layer.update(); checked_state = _recorded_state(guide, rig)
            rig[ENABLE] = 1.; rig.update_tag(); bpy.context.view_layer.update()
            expected = expected_positions(cache_relative, frame); graph = bpy.context.evaluated_depsgraph_get()
            row = {'frame': frame, 'objects': {},
                   'attachmentValidation': checked_state['currentStateValidation']}
            for name in meta['objects']:
                obj = bpy.data.objects[name]; evaluated = obj.evaluated_get(graph)
                if evaluated.type != 'CURVES' or not hasattr(evaluated.data, 'position_data'):
                    raise RuntimeError('Evaluated native curve data is unavailable: ' + name)
                actual = _read_points(evaluated)
                if actual.shape != expected[name].shape:
                    raise RuntimeError('GN changed the strand point count: ' + name)
                error = float(np.linalg.norm(actual - expected[name], axis=1).max())
                maximum = max(maximum, error); row['objects'][name] = {'maximumPositionErrorMeters': error, 'points': len(actual)}
            rows.append(row)
        report = {'maximumErrorMeters': maximum, 'toleranceMeters': tolerance, 'passed': maximum <= tolerance,
                  'frames': rows, 'scope': 'Evaluated native point parity only; no new visual/collision/portability certification', 'savedBlend': False}
        if not report['passed']:
            raise RuntimeError('Native guide playback mismatch: ' + json.dumps(report))
        record['evaluatedValidated'] = True; record['evaluatedValidation'] = report
        guide[RECORD] = json.dumps(record, ensure_ascii=False)
        print(json.dumps(report)); return report
    finally:
        rig[ENABLE] = old_enable; rig.update_tag(); scene.frame_set(old_frame, subframe=old_subframe); bpy.context.view_layer.update()


def restore():
    """Remove this pass's modifiers/maps; preserve guide/action/GN data as history."""
    rig = _guard(require_work_copy=True); guide = _active_guide(rig); record = json.loads(guide[RECORD])
    for name, modifier_name in record['modifiers'].items():
        obj = bpy.data.objects.get(name)
        if obj is None or obj.get(OWNER) != record['owner']:
            raise RuntimeError('Ownership changed; restore halted before removing this object')
        modifier = obj.modifiers.get(modifier_name)
        if modifier is None or modifier.type != 'NODES' or modifier.node_group is None or modifier.node_group.name != record['nodeGroup']:
            raise RuntimeError('Modifier changed; inspect before restore')
    rig[ENABLE] = 0.; rig.update_tag(); bpy.context.view_layer.update()
    for name, modifier_name in record['modifiers'].items():
        obj = bpy.data.objects[name]; modifier = obj.modifiers[modifier_name]
        for prop in ('show_viewport', 'show_render'):
            modifier.driver_remove(prop)
        obj.modifiers.remove(modifier)
        for attr_name in record['attributes']:
            attr = obj.data.attributes.get(attr_name)
            if attr: obj.data.attributes.remove(attr)
        del obj[OWNER]; obj.data.update_tag(); obj.update_tag()
    group = bpy.data.node_groups.get(record['nodeGroup'])
    if group: group.use_fake_user = True
    guide['r04_native_hair_active'] = False; guide.hide_set(True)
    guide.name = guide.name + ' preserved after restore'
    for prop in (ENABLE, RIG_GUIDE): del rig[prop]
    bpy.context.view_layer.update()
    return {'restoredSourceCurves': True, 'preservedGuide': guide.name, 'savedBlend': False}


if __name__ == '__main__':
    print('Loaded only; run(cache_relative), then validate_evaluated() in Blender. No scene change.')
