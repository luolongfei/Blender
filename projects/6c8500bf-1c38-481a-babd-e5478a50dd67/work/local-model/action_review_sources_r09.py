"""Explicit, independent R09 portrait and retained-walk review copies.

Integrator-only background Blender CLI (this module never launches Blender):
  --clip portrait --mode prepare --source <final-r09.blend> --output <new.blend>
  --clip walk     --mode prepare --source <final-r09.blend> --output <new.blend>
  --clip <clip> --mode sparse --source <prepared.blend> --label <new-label>
  --clip <clip> --mode continuous --source <prepared.blend> --label <same-label>

No animation authoring, simulation bake, geometry transfer, or main-source save.
Portrait is 181 frames with a STATIC skirt, not cloth-dynamics evidence. Walk is
the retained 80-frame captured cloth/body/hair combination, not newly simulated.
Both require a fresh-file sparse proof of their OWN saved SHA before continuous.
"""
import argparse
import hashlib
import json
import math
import re
import runpy
import sys
from pathlib import Path

import numpy as np
try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

FOLDER = Path(__file__).resolve().parent
WORK = 'work/reconstruction-r02'
HEAD = 'R01 Head'
RIG = 'Character_Rig'
WALK_CLOTH = 'R07 Native Walk Cloth playback'
WALK_CLOTH_ACTION = 'R07 Cloth baked payload - CMU walk 1-80'
STATIC_SKIRT = ('Skirt_Sixteen_Pleats', 'Skirt_Turned_Hem_Stitch', 'Skirt_Center_Back_Seam')
CONTROLS = ('R02_blink_L', 'R02_blink_R', 'R02_jaw_open', 'R02_mouth_wide',
            'R02_pucker', 'R02_lip_press', 'R02_smile', 'R02_frown',
            'R02_brow_raise', 'R02_brow_furrow', 'R03_squint_L', 'R03_squint_R')
ORAL = ('R02 Upper teeth', 'R02 Lower teeth', 'R02 Tongue', 'R03 Upper gingiva', 'R03 Lower gingiva')
CLIPS = {
    'portrait': {
        'body': 'R03 Performance review',
        'face': 'R03 Silent vowels and expression combinations',
        'cache': WORK+'/hair-batch-r07/R03-Performance-review.guides.npz',
        'end': 181, 'sparse': [1, 20, 42, 63, 82, 98, 110, 128, 146, 162, 181],
        'parity': [1, 19.375, 42, 81.5, 121, 145.375, 162, 181],
        'scope': 'Authored silent facial/head/gaze performance on current R09 geometry. Matched embedded twin-tail playback; original STATIC skirt. No speech audio or cloth-dynamics claim.'},
    'walk': {
        'body': 'R06 CMU 07_01 walk - cloth preroll',
        'face': 'R07 Walk authored natural blink',
        'cache': WORK+'/hair-walk-r07/R06-CMU-07_01-walk---cloth-preroll.guides.npz',
        'end': 80, 'sparse': [1, 15, 35, 39, 61, 80],
        'parity': [1, 7.5, 15.375, 35, 61, 79.75, 80],
        'scope': 'Retained CMU-derived body/head walk and authored blink, original embedded 80-frame cloth simulation and matched twin-tail playback, on current R09 appearance. Non-looping clip; no rebake or new collision guarantee.'}}
DEPENDENCIES = ('action_review_sources_r09.py', 'face_performance_r08.py',
                'face_keyframes_r09.py', 'native_hair_playback_r04.py',
                'walk_cloth_native_r07.py', 'walk_cloth_r06.py',
                'walk_cloth_cache_recovery_r07.py', 'secondary_hair_r02.py')


def _api(name):
    return runpy.run_path(str(FOLDER/(name+'.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _path(relative):
    path = (project_path()/relative).resolve()
    path.relative_to((project_path()/WORK).resolve())
    return path


def _relative(path):
    return Path(path).resolve().relative_to(project_path().resolve()).as_posix()


def _write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _dependencies():
    return {name: _sha(FOLDER/name) for name in DEPENDENCIES}


def _guard(relative):
    if bpy is None or not bpy.app.background:
        raise RuntimeError('Use a fresh disposable background Blender process')
    source = _path(relative)
    if (not source.is_file() or source.suffix != '.blend' or
            source != Path(bpy.data.filepath).resolve() or bpy.data.is_dirty or bpy.context.mode != 'OBJECT'):
        raise RuntimeError('Open the exact saved independent --source in a fresh process')
    for handlers in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        if any(getattr(h, '_r02_secondary_hair', False) for h in handlers):
            raise RuntimeError('Old Python hair handler is outside review scope')
    head = bpy.data.objects[HEAD]; rig = bpy.data.objects[RIG]
    if (not head.get('R09_lower_face_candidate') or not head.get('R09_perioral_delta_flow')
            or 'R08 ProductionUV' not in head.data.uv_layers):
        raise RuntimeError('Expected the actual R09 lower-face/perioral/production-UV source')
    key = head.data.shape_keys.key_blocks.get('R09_lower_jaw_curvature')
    if key is None or abs(key.value-1.) > 1e-6:
        raise RuntimeError('Preserve the current accepted static lower-jaw layer at 1')
    if 'R06_head_pivot_calibration' not in rig:
        raise RuntimeError('Calibrated head/neck source required')
    return source


def _array(collection, property_name, width=3, dtype=np.float32):
    result = np.empty(len(collection)*width, dtype=dtype)
    collection.foreach_get(property_name, result)
    return result.reshape(len(collection), width)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.shape).encode()+str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _drivers(owner):
    ad = getattr(owner, 'animation_data', None)
    rows = []
    for fc in ad.drivers if ad else ():
        # Only the cache module owns/replaces these tail-modifier switches.
        if fc.data_path.startswith('modifiers["R04 Native hair playback'):
            continue
        driver = fc.driver
        rows.append({'path': fc.data_path, 'index': fc.array_index, 'mute': fc.mute,
            'type': driver.type, 'expression': driver.expression, 'useSelf': driver.use_self,
            'variables': [{'name': v.name, 'type': v.type, 'targets': [
                {'id': t.id.name if t.id else None, 'idType': t.id_type,
                 'dataPath': t.data_path, 'bone': t.bone_target,
                 'transformType': t.transform_type, 'transformSpace': t.transform_space}
                for t in v.targets]} for v in driver.variables]})
    return rows


def _unanimated_object_channels(obj):
    """Protect oral/local placement without freezing legitimately driven axes."""
    ad = obj.animation_data
    occupied = {(fc.data_path, fc.array_index) for fc in ad.drivers} if ad else set()
    if ad and ad.action:
        native = _api('native_hair_playback_r04')
        occupied.update((fc.data_path, fc.array_index) for fc in native['_fcurves'](ad.action))
    if ad:
        native = _api('native_hair_playback_r04')
        for track in ad.nla_tracks:
            for strip in track.strips:
                if strip.action:
                    occupied.update((fc.data_path, fc.array_index) for fc in native['_fcurves'](strip.action))
    rows = {'rotationMode': obj.rotation_mode}
    for name in ('location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle',
                 'scale', 'delta_location', 'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale'):
        value = getattr(obj, name)
        rows[name] = {str(i): float(v) for i, v in enumerate(value) if (name, i) not in occupied}
    return rows


def _geometry_contract():
    """Raw coordinates/ALL key coordinates, radii, topology and bind data only.

    Does not confuse the raw mesh with the evaluated Basis+static-key mix.
    Values/actions are checked separately; no coordinate field is written here.
    """
    rig = bpy.data.objects[RIG]
    names = {HEAD, *ORAL, 'R02 Soft brow -1', 'R02 Soft brow 1', 'R06 attached scalp root substrate'}
    names.update(o.name for o in bpy.data.objects if o.type == 'CURVES')
    rows = {}
    for name in sorted(names):
        obj = bpy.data.objects[name]; data = obj.data
        row = {'type': obj.type, 'parent': obj.parent.name if obj.parent else None,
               'parentType': obj.parent_type, 'parentBone': obj.parent_bone,
               'parentInverse': [list(v) for v in obj.matrix_parent_inverse],
               'unanimatedObjectChannels': _unanimated_object_channels(obj),
               'drivers': _drivers(obj),
               'materials': [s.material.name if s.material else None for s in obj.material_slots]}
        if obj.type == 'MESH':
            row.update(vertices=_array_sha(_array(data.vertices, 'co')),
                       edges=_array_sha(_array(data.edges, 'vertices', 2, np.int32)),
                       loops=_array_sha(_array(data.loops, 'vertex_index', 1, np.int32)),
                       uv={uv.name: _array_sha(_array(uv.data, 'uv', 2)) for uv in data.uv_layers})
        elif obj.type == 'CURVES':
            radius = data.attributes.get('radius')
            row.update(points=_array_sha(_array(data.position_data, 'vector')),
                       radii=_array_sha(_array(radius.data, 'value', 1)) if radius else None,
                       starts=_array_sha(np.asarray([c.first_point_index for c in data.curves], np.int32)),
                       lengths=_array_sha(np.asarray([c.points_length for c in data.curves], np.int32)))
        elif obj.type == 'CURVE':
            row['splines'] = [{'type': s.type, 'cyclic': s.use_cyclic_u,
                'points': [{'co': list(p.co), 'radius': p.radius, 'tilt': p.tilt} for p in s.points],
                'bezier': [{'co': list(p.co), 'left': list(p.handle_left), 'right': list(p.handle_right),
                            'radius': p.radius, 'tilt': p.tilt} for p in s.bezier_points]} for s in data.splines]
        keys = getattr(data, 'shape_keys', None)
        if keys:
            row['keys'] = {k.name: {'co': _array_sha(_array(k.data, 'co')),
                'relativeTo': k.relative_key.name, 'mute': k.mute, 'group': k.vertex_group,
                'slider': [k.slider_min, k.slider_max]} for k in keys.key_blocks}
            row['keyDrivers'] = _drivers(keys)
        rows[name] = row
    rows['rigRest'] = {b.name: {'matrix': [list(v) for v in b.matrix_local],
        'parent': b.parent.name if b.parent else None, 'length': b.length,
        'deform': b.use_deform} for b in rig.data.bones}
    rows['rigDrivers'] = _drivers(rig)
    return {'sha256': _hash(rows), 'objects': {k: _hash(v) for k, v in rows.items()}}


def _static_values():
    keys = bpy.data.objects[HEAD].data.shape_keys
    driven = {d['path'] for d in _drivers(keys)}
    return {k.name: float(k.value) for k in keys.key_blocks
            if k.name not in CONTROLS and k.path_from_id('value') not in driven}


def _check_static(expected):
    actual = _static_values()
    if actual.keys() != expected.keys() or any(abs(actual[n]-v) > 1e-6 for n, v in expected.items()):
        raise RuntimeError('A static facial layer changed: '+json.dumps({'expected': expected, 'actual': actual}))


def _frame(value):
    bpy.context.scene.frame_set(math.floor(value), subframe=value % 1)
    bpy.context.view_layer.update()


def _pose_neutral():
    """Reset only pose/control values; retain every non-control static key/driver."""
    rig = bpy.data.objects[RIG]
    for owner in (rig, bpy.data.objects[HEAD].data.shape_keys):
        ad = owner.animation_data
        if ad and ad.action:
            ad.action.use_fake_user = True
            ad.action = None
        if ad:
            for track in ad.nla_tracks:
                track.mute = True
    for bone in rig.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    master = rig.pose.bones.get('CTRL_R02_Master')
    if master:
        for side in ('leg_L', 'leg_R', 'arm_L', 'arm_R'):
            if 'ik_'+side in master:
                master['ik_'+side] = 0.
    for prop in ('R03_look_yaw', 'R03_look_pitch'):
        if prop in rig:
            rig[prop] = 0.
    keys = bpy.data.objects[HEAD].data.shape_keys.key_blocks
    for name in CONTROLS:
        keys[name].value = .45 if name == 'R02_smile' else 0.
    rig.data.pose_position = 'POSE'


def _cloth_visibility(clip, change=False):
    rows = []
    if any(name not in bpy.data.objects for name in STATIC_SKIRT):
        raise RuntimeError('Original static skirt history is missing')
    if clip == 'walk' and WALK_CLOTH not in bpy.data.objects:
        raise RuntimeError('Final source lacks the retained 80-frame walk cloth; no automatic old-source import')
    for obj in bpy.context.scene.objects:
        dynamic = (any(m.type == 'CLOTH' for m in obj.modifiers)
                   or obj.name in ('R02 Cloth Skirt control', 'R06 Walk cloth')
                   or ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
        if not dynamic and obj.name not in STATIC_SKIRT:
            continue
        visible = obj.name in STATIC_SKIRT if clip == 'portrait' else obj.name == WALK_CLOTH
        if change:
            obj.hide_render = obj.hide_viewport = not visible
            obj.hide_set(not visible)
        if (obj.hide_render != (not visible) or obj.hide_viewport != (not visible)
                or obj.hide_get() != (not visible)):
            raise RuntimeError('Cloth visibility mismatch: '+obj.name)
        rows.append({'object': obj.name, 'visible': visible,
                     'role': 'STATIC original skirt' if obj.name in STATIC_SKIRT else 'recorded dynamics/history'})
    return rows


def _actions(clip, bind=False):
    config = CLIPS[clip]; helper = _api('face_performance_r08'); native = _api('native_hair_playback_r04')
    targets = [(bpy.data.objects[RIG], config['body'], 'OBJECT'),
               (bpy.data.objects[HEAD].data.shape_keys, config['face'], 'KEY')]
    if clip == 'walk':
        targets.append((bpy.data.objects[WALK_CLOTH].data.shape_keys, WALK_CLOTH_ACTION, 'KEY'))
    result = []
    for owner, name, id_type in targets:
        if bind:
            helper['_bind'](owner, name, id_type)
        ad = owner.animation_data
        if (ad is None or ad.action is None or ad.action.name != name or ad.action_slot is None
                or ad.action_slot.target_id_type != id_type or any(not t.mute for t in ad.nla_tracks)):
            raise RuntimeError('Saved action/slot/NLA mismatch: '+name)
        result.append({'owner': owner.name, 'action': name, 'slot': ad.action_slot.identifier,
                       'targetIdType': id_type, 'sha256': native['_action_fingerprint'](ad.action)})
    permitted = set(CONTROLS) if clip == 'portrait' else {'R02_blink_L', 'R02_blink_R'}
    head_keys = bpy.data.objects[HEAD].data.shape_keys.key_blocks
    paths = {head_keys[name].path_from_id('value') for name in permitted}
    for fc in native['_fcurves'](bpy.data.actions[config['face']]):
        if fc.data_path not in paths:
            raise RuntimeError('Review action drives a non-expression/static facial field: '+fc.data_path)
    return result


def _hair_preflight(clip):
    config = CLIPS[clip]; native = _api('native_hair_playback_r04')
    path, meta, arrays, base, base_meta = native['load_cache'](config['cache'])
    if (meta['action'] != config['body'] or meta['fps'] != 30 or arrays['frames'][0] != 1
            or arrays['frames'][-1] < config['end']):
        raise RuntimeError('Wrong clip guide cache')
    action = bpy.data.actions.get(config['body'])
    if action is None or native['_action_fingerprint'](action) != meta['actionSha256']:
        raise RuntimeError('Retained recorded body action was changed or removed')
    rows = {}
    for i, name in enumerate(meta['objects']):
        obj = bpy.data.objects[name]; rest = native['_read_points'](obj).reshape(base[f'rest_{i}'].shape)
        actual = native['_digest'](rest)
        if actual != base_meta['geometryHashes'][name]:
            raise RuntimeError('Current tail rest no longer matches the old cache: '+name)
        rows[name] = {'sha256': actual, 'shape': list(rest.shape),
                      'localBounds': [rest.min(axis=(0, 1)).tolist(), rest.max(axis=(0, 1)).tolist()],
                      'parent': obj.parent.name if obj.parent else None, 'parentBone': obj.parent_bone}
    return {'cache': config['cache'], 'cacheSha256': _sha(path),
            'baseSha256': _sha(path.parent/meta['baseFile']), 'cacheRange': meta['frameRange'],
            'actionSha256': meta['actionSha256'], 'actualTailRest': rows,
            'scope': 'Exact current raw tail rest and action compatibility; attachment matrices and evaluated points must also pass below. Not a new collider solve.'}


def _walk_cloth_check():
    """Reuse the decoded real payload; do not run old save/report entry points."""
    module = _api('walk_cloth_native_r07'); api = module['API']; recovery = module['RECOVERY']
    obj = bpy.data.objects[WALK_CLOTH]; source = bpy.data.objects['R06 Walk cloth']
    report = json.loads(obj['r07_native_cloth_report'])
    reference_path = recovery['REFERENCE']
    if _sha(reference_path) != report['payload_reference_sha256']:
        raise RuntimeError('Recorded cloth payload reference identity changed')
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if set(reference['positions_world']) != {str(f) for f in range(1, 81)}:
        raise RuntimeError('Expected exactly the original 80 cloth samples')
    if [m.type for m in obj.modifiers] != ['SUBSURF', 'SOLIDIFY']:
        raise RuntimeError('Walk cloth has unreviewed deformation modifiers')
    for actual, key in ((module['_topology'](obj.data), 'topology_sha256'),
                        (module['_uv_digest'](obj.data), 'uv_sha256'),
                        (api['_digest'](api['_records'](module['_shape_action'](obj), False)), 'shape_action_sha256'),
                        (api['_digest'](api['_records'](bpy.data.objects[RIG].animation_data.action, False)), 'body_action_sha256')):
        if actual != report[key]:
            raise RuntimeError('Retained cloth/body contract changed: '+key)
    errors = module['_parity'](obj, reference)
    surface_error = api['_coordinate_error'](report['surface_reference'], api['_cloth_coordinates'](obj, module['SURFACE_FRAMES']))
    if surface_error > module['TOLERANCE']:
        raise RuntimeError('Current visible cloth surface differs from original embedded payload')
    setup = json.loads(source['r06_walk_cloth_report'])
    current = api['_matrices'](bpy.data.objects[RIG], api['CHECK_FRAMES'])
    maximum = 0.; oral_changes = []
    # R09 intentionally maps the jaw/oral articulation. It must not influence
    # any garment vertex; every other original deform-bone matrix stays strict.
    jaw_group = source.vertex_groups.get('R02_Jaw')
    if jaw_group and any(g.group == jaw_group.index and g.weight > 0 for v in source.data.vertices for g in v.groups):
        raise RuntimeError('R09 jaw affects cloth vertices; old payload cannot be reused')
    for frame, bones in setup['source_matrices'].items():
        for bone, expected in bones.items():
            actual = current[frame][bone]
            error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
            if bone == 'R02_Jaw':
                oral_changes.append({'frame': frame, 'maxAbsMatrixElementDifference': error})
            else:
                maximum = max(maximum, error)
    if maximum > 1e-6:
        raise RuntimeError('Non-oral body pose differs from original cloth simulation: '+str(maximum))
    _frame(1)
    return {'payloadReference': _relative(reference_path), 'payloadReferenceSha256': _sha(reference_path),
            'integerFrames': 80, 'fractionalFrames': list(module['FRACTIONS']),
            'payloadMaxErrorMeters': max(errors.values()), 'surfaceMaxErrorMeters': surface_error,
            'nonOralBodyMatrixMaxError': maximum, 'intentionalJawMatrixDifferences': oral_changes,
            'jawHasClothWeights': False,
            'scope': 'Original simulated cloth coordinates, render subdivision/thickness surface and non-oral body matrices checked in the current source. Facial/skin changes are not a new cloth collision validation.'}


def _expression_checks(clip, static):
    helper = _api('face_performance_r08'); keyframes = _api('face_keyframes_r09')
    result = {'face': helper['_face_probe'](_api('native_hair_playback_r04'))} if clip == 'portrait' else {}
    rows = []
    for frame in CLIPS[clip]['sparse']:
        _frame(frame); _check_static(static)
        keys = bpy.data.objects[HEAD].data.shape_keys.key_blocks
        if clip == 'walk' and (abs(keys['R02_jaw_open'].value) > 1e-6 or abs(keys['R02_smile'].value-.45) > 1e-6):
            raise RuntimeError('Walk inherited an unrelated mouth expression')
        if clip == 'portrait' and frame == 42:
            for name, value in keyframes['AA42_EXPECTED'].items():
                if abs(keys[name].value-value) > 2e-5:
                    raise RuntimeError('Actual AA42 expression mismatch: '+name)
        rows.append({'frame': frame, 'controls': {name: float(keys[name].value) for name in CONTROLS},
                     'oral': keyframes['_oral_observation'](bpy.data.objects[HEAD])})
    _frame(1)
    result['currentR09OralAndStaticChecks'] = rows
    return result


def prepare(clip, source_relative, output_relative):
    config = CLIPS[clip]; source = _guard(source_relative); target = _path(output_relative)
    report_path = target.with_suffix('.prepare.json')
    if target.suffix != '.blend' or target == source or target.exists() or report_path.exists():
        raise RuntimeError('Select a NEW independent .blend output; previous sources/evidence are retained')
    source_sha = _sha(source); dependencies = _dependencies()
    geometry = _geometry_contract(); static = _static_values(); hair_preflight = _hair_preflight(clip)
    native = _api('native_hair_playback_r04'); rig = bpy.data.objects[RIG]
    restored = native['restore']() if native['RIG_GUIDE'] in rig else None
    # Hide every unrelated simulation before stepping a different body action.
    cloth = _cloth_visibility(clip, True)
    _pose_neutral(); actions = _actions(clip, True)
    scene = bpy.context.scene
    scene.render.fps = 30; scene.render.fps_base = 1.
    scene.frame_start = scene.frame_preview_start = 1
    scene.frame_end = scene.frame_preview_end = config['end']; scene.use_preview_range = True
    _frame(1); _check_static(static)
    native['run'](cache_relative=config['cache'])
    native['enable_for_action'](action_name=config['body'], select=False, frame=1)
    hair = native['validate_evaluated'](cache_relative=config['cache'], frames=config['parity'])
    cloth_checks = _walk_cloth_check() if clip == 'walk' else {'dynamicsClaim': False, 'skirt': 'STATIC'}
    expression = _expression_checks(clip, static)
    _actions(clip); _cloth_visibility(clip); _check_static(static); _frame(1)
    if _geometry_contract() != geometry:
        raise RuntimeError('Review preparation changed geometry, key coordinates, material references, bind data or non-cache drivers')
    if _sha(source) != source_sha or _dependencies() != dependencies:
        raise RuntimeError('Input source or helper code changed during preparation')
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    result = {'source': _relative(source), 'sourceSha256': source_sha, 'output': _relative(target),
        'sha256': _sha(target), 'clip': clip, 'blender': bpy.app.version_string,
        'dependencies': dependencies, 'actions': actions, 'frames': [1, config['end']], 'fps': 30,
        'frameCount': config['end'], 'durationSeconds': config['end']/30,
        'staticFacialValues': static, 'geometryContract': geometry, 'geometryAndDriversUnchanged': True,
        'previousNativeCacheRestored': restored, 'hairCompatibility': hair_preflight,
        'hairEvaluatedParity': hair, 'clothVisibility': cloth, 'clothChecks': cloth_checks,
        'expressionChecks': expression, 'scope': config['scope'], 'appearanceAccepted': False,
        'nextRequired': 'Fresh-file sparse check/render of THIS saved SHA, then continuous frames and visual review'}
    _write_new(report_path, result)
    print('R09_ACTION_SOURCE_SAVED '+json.dumps({'clip': clip, 'output': _relative(target), 'sha256': result['sha256']}), flush=True)
    return result


def render(clip, source_relative, mode, label):
    if mode not in ('sparse', 'continuous') or not re.fullmatch(r'[a-z0-9-]+', label or ''):
        raise ValueError('Explicit sparse/continuous and safe new --label required')
    config = CLIPS[clip]; source = _guard(source_relative); source_sha = _sha(source)
    prepared = json.loads(source.with_suffix('.prepare.json').read_text(encoding='utf-8'))
    if (prepared['sha256'] != source_sha or prepared['clip'] != clip
            or prepared['dependencies'] != _dependencies()):
        raise RuntimeError('Preparation proof does not identify this source, clip and current helper code')
    out = project_path()/'previews/reconstruction-r02'/f'{clip}-{mode}-{label}'
    proof = _path(f'{WORK}/{clip}-reopen-{label}.json')
    if out.exists() or (mode == 'sparse' and proof.exists()):
        raise RuntimeError('Preserve previous evidence; choose a new --label')
    if mode == 'continuous':
        earlier = json.loads(proof.read_text(encoding='utf-8'))
        if (earlier['sha256'] != source_sha or earlier['clip'] != clip or earlier['status'] != 'sparse-rendered'
                or earlier['dependencies'] != _dependencies()):
            raise RuntimeError('Fresh sparse proof must match this saved source/clip/helper SHA')
        for item in earlier['images']:
            if _sha(project_path()/item['path']) != item['sha256']:
                raise RuntimeError('Sparse image identity changed')
    scene = bpy.context.scene
    if (scene.frame_start, scene.frame_end, scene.render.fps, scene.render.fps_base) != (1, config['end'], 30, 1.):
        raise RuntimeError('Saved clip duration/fps mismatch')
    if _geometry_contract() != prepared['geometryContract']:
        raise RuntimeError('Fresh-file geometry/key/bind/driver contract differs from saved preparation')
    _check_static(prepared['staticFacialValues']); actions = _actions(clip); cloth = _cloth_visibility(clip)
    native = _api('native_hair_playback_r04')
    native['enable_for_action'](action_name=config['body'], select=False, frame=1)
    checks = {'hairCompatibility': _hair_preflight(clip)}
    if mode == 'sparse':
        checks['hairEvaluatedParity'] = native['validate_evaluated'](cache_relative=config['cache'], frames=config['parity'])
        checks['cloth'] = _walk_cloth_check() if clip == 'walk' else {'dynamicsClaim': False, 'skirt': 'STATIC'}
        checks['expression'] = _expression_checks(clip, prepared['staticFacialValues'])
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
    scene.cycles.samples = 12; scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 6; scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = (480, 540) if clip == 'portrait' else (480, 640)
    scene.render.resolution_percentage = 100; scene.render.use_sequencer = False
    scene.render.use_file_extension = True; scene.render.use_overwrite = False
    scene.render.image_settings.file_format = 'PNG'; scene.render.film_transparent = False
    camera = bpy.data.objects.new('R09 '+clip+' review camera', bpy.data.cameras.new('R09 '+clip+' review camera'))
    scene.collection.objects.link(camera); scene.camera = camera
    camera.data.type = 'ORTHO'; camera.data.dof.use_dof = False
    camera.data.ortho_scale = .36 if clip == 'portrait' else 1.83
    lights = [(o, o.location.copy()) for o in scene.objects if o.type == 'LIGHT']
    frames = list(range(1, config['end']+1)) if mode == 'continuous' else config['sparse']
    out.mkdir(parents=True); images = []; rows = []
    for frame in frames:
        _frame(frame); _check_static(prepared['staticFacialValues'])
        if clip == 'portrait':
            camera.location = (0., -3., 1.429); target = Vector((0., -.005, 1.429))
        else:
            rig = bpy.data.objects[RIG]; hips = rig.matrix_world@rig.pose.bones['hips'].head
            offset = Vector((hips.x, hips.y, 0.))
            camera.location = offset+Vector((1.5, -4.5, 1.04)); target = offset+Vector((0., 0., .78))
            for light, initial in lights:
                light.location = initial+offset
        camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
        path = out/f'frame-{frame:04d}.png'; scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError('Expected PNG missing after actual render')
        images.append({'path': _relative(path), 'sha256': _sha(path)})
        rows.append({'frame': frame, 'camera': list(camera.location), 'target': list(target)})
        print('R09_ACTION_FRAME '+clip+' '+str(frame), flush=True)
    if _sha(source) != source_sha or _dependencies() != prepared['dependencies']:
        raise RuntimeError('Saved source or helper changed while rendering')
    report = {'source': _relative(source), 'sha256': source_sha, 'clip': clip, 'mode': mode,
        'dependencies': _dependencies(), 'fps': 30, 'fpsBase': 1, 'framePattern': 'frame-%04d.png',
        'frames': rows, 'images': images, 'resolution': [scene.render.resolution_x, scene.render.resolution_y],
        'samples': 12, 'engine': 'Cycles CPU', 'actions': actions, 'checks': checks,
        'clothVisibility': cloth, 'staticFacialValues': prepared['staticFacialValues'],
        'scope': config['scope'], 'savedBlend': False, 'appearanceAccepted': False,
        'fullClipFrames': config['end'], 'fullClipDurationSeconds': config['end']/30}
    _write_new(out/'render-report.json', report)
    if mode == 'sparse':
        _write_new(proof, dict(report, status='sparse-rendered'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', choices=tuple(CLIPS), required=True)
    parser.add_argument('--mode', choices=('prepare', 'sparse', 'continuous'), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output')
    parser.add_argument('--label')
    arguments = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if arguments.mode == 'prepare':
        if not arguments.output:
            parser.error('prepare requires an explicit NEW --output')
        prepare(arguments.clip, arguments.source, arguments.output)
    else:
        if not arguments.label:
            parser.error('render requires --label; reuse sparse label only for its continuous run')
        render(arguments.clip, arguments.source, arguments.mode, arguments.label)
