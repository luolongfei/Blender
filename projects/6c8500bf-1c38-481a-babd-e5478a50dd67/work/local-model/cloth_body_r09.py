"""Fresh R09 cloth: capture actual evaluated Cloth points, then native playback.

Explicit background Blender only, executed by the integrator, serially:
  --python cloth_body_r09.py -- --source work/reconstruction-r02/body-action-r09.blend
Default mode 'all' saves settings, simulates -59..301 once, captures 1..301,
and saves a separate native playback Blend. No rendering or old-cache rebake.
Modes capture / embed / verify split the same workflow into separate processes.

Captured coordinates come from ONE evaluated mesh with downstream subdivision
and thickness disabled throughout simulation. No external bphys decoding.
The native playback has no Armature modifier; hip motion is already in points.
Integer samples are simulated; between-frame playback is linear interpolation.
"""
import argparse
import hashlib
import json
import math
import runpy
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

ACTION = 'R09 Body performance - authored squat bend reach'
PREFIX = 'R09 Body Cloth '
CONTROL = PREFIX + 'Skirt control'
COLLECTION = PREFIX + 'Simulation'
COLLIDERS = PREFIX + 'Body colliders'
NATIVE = 'R09 Native Body Cloth playback'
NATIVE_COLLECTION = 'R09 Native Body Cloth Playback'
PAYLOAD_ACTION = 'R09 Cloth simulated payload - body 1-301'
START, FIRST, END, COUNT = -59, 1, 301, 2176
CHECK_FRAMES = (1., 31.5, 55., 55.25, 151., 151.5, 247., 247.5, 300.5, 301.)
TOLERANCE = 2e-6
API = runpy.run_path(str(Path(__file__).with_name('walk_cloth_r06.py')))
BASE = runpy.run_path(str(Path(__file__).with_name('cloth_setup_r02.py')))
HELPER = API['HELPER']


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _path(value):
    path = (project_path() / value).resolve()
    path.relative_to((project_path() / 'work').resolve())
    if path == (project_path() / 'work/character.blend').resolve():
        raise RuntimeError('The formal source is excluded from this independent experiment')
    return path


def _relative(path):
    return Path(path).resolve().relative_to(project_path().resolve()).as_posix()


def _action_hash(action):
    return _digest(API['_records'](action, False))


def _topology(mesh):
    return _digest({'vertices': len(mesh.vertices),
                    'faces': [list(p.vertices) for p in mesh.polygons],
                    'materials': [p.material_index for p in mesh.polygons]})


def _uv_hash(mesh):
    return _digest({uv.name: [list(item.uv) for item in uv.data] for uv in mesh.uv_layers})


def _coords(mesh):
    values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', values)
    return values.reshape((-1, 3))


def _world(local, matrix):
    transform = np.asarray(matrix, dtype=np.float64)
    return (local.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def _evaluated(obj):
    """Mesh coordinates AND transform are read from the same evaluated object."""
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        return _world(_coords(mesh), evaluated.matrix_world), _topology(mesh)
    finally:
        evaluated.to_mesh_clear()


def _set_frame(frame):
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    bpy.context.view_layer.update()


def _open(path):
    if not bpy.app.background:
        raise RuntimeError('Only the integrator may run this in an isolated background Blender')
    if not path.is_file() or path.suffix.lower() != '.blend':
        raise RuntimeError('Explicit existing project work Blend required')
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    if Path(bpy.data.filepath).resolve() != path:
        raise RuntimeError('The requested exact source was not opened')


def _rig(readonly_master=False):
    if readonly_master:
        # Only finite verification/review callers opt into this path. All
        # simulation, capture and mutation callers retain the original guard.
        if (not bpy.app.background or bpy.context.mode!='OBJECT'
                or Path(bpy.data.filepath).resolve()!=(project_path()/'work/character.blend').resolve()):
            raise RuntimeError('Read-only master review requires the exact saved master in background Object Mode')
        rig=bpy.data.objects.get('Character_Rig')
        if rig is None or rig.type!='ARMATURE':raise RuntimeError('Character_Rig required')
        if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):
            raise RuntimeError('Active NLA blending needs explicit integration')
        if abs(bpy.context.scene.render.fps/bpy.context.scene.render.fps_base-30)>1e-6:
            raise RuntimeError('Preserve the validated 30 fps timebase')
    else:
        _, _, rig = API['_guard']()
    if any(o.get('r02_secondary_hair_session') for o in bpy.data.objects):
        raise RuntimeError('An old Python hair session needs explicit detachment before this new simulation')
    return rig


def _activate(rig):
    action = bpy.data.actions.get(ACTION)
    if action is None or 'r09_performance_report' not in action:
        raise RuntimeError('The saved validated R09 body action is required')
    curves = HELPER['_fcurves'](action)
    if not curves or any(c.extrapolation != 'CONSTANT' for c in curves):
        raise RuntimeError('R09 pre-roll requires the unchanged constant first-pose extrapolation')
    if tuple(round(x, 6) for x in action.frame_range) != (1., 301.):
        raise RuntimeError('Expected the unchanged R09 1..301 action')
    HELPER['_assign'](rig, action)
    rig.data.pose_position = 'POSE'
    return action


def _suspend_old_dynamics(rig):
    """Only visibility/enable flags in this derivative; no cache free or removal."""
    report = {'objects': [], 'modifiers': [], 'mutedDrivers': [], 'rigSwitch': {}}
    if 'R04_native_hair_enabled' in rig:
        report['rigSwitch']['R04_native_hair_enabled'] = rig['R04_native_hair_enabled']
        rig['R04_native_hair_enabled'] = 0
    for obj in bpy.context.scene.objects:
        dynamic = (any(m.type == 'CLOTH' for m in obj.modifiers)
                   or ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
        if dynamic or obj.name in BASE['ORIGINALS']:
            report['objects'].append({'name': obj.name, 'hideViewport': obj.hide_viewport,
                                      'hideRender': obj.hide_render, 'hideSet': obj.hide_get()})
            obj.hide_viewport = True
            obj.hide_render = True
            obj.hide_set(True)
        for modifier in obj.modifiers:
            hair = modifier.type == 'NODES' and modifier.name.startswith('R04 Native hair playback')
            if modifier.type == 'CLOTH' or hair:
                report['modifiers'].append({'object': obj.name, 'name': modifier.name,
                                            'viewport': modifier.show_viewport, 'render': modifier.show_render})
                for prop in ('show_viewport', 'show_render'):
                    curve = obj.animation_data.drivers.find(modifier.path_from_id(prop)) if obj.animation_data else None
                    if curve:
                        report['mutedDrivers'].append({'object': obj.name, 'path': curve.data_path, 'mute': curve.mute})
                        curve.mute = True
                modifier.show_viewport = False
                modifier.show_render = False
    return report


def _input_fingerprint(rig, control):
    """Retained rig rest and new collider deform inputs; no whole-scene scan."""
    objects = {}
    for obj in bpy.data.collections[COLLIDERS].objects:
        mesh = obj.data
        keys = mesh.shape_keys
        key_rows = []
        if keys:
            for key in keys.key_blocks:
                co = np.empty(len(key.data) * 3, dtype=np.float32)
                key.data.foreach_get('co', co)
                key_rows.append({'name': key.name, 'value': float(key.value),
                    'coordinatesSha256': hashlib.sha256(co.tobytes()).hexdigest(),
                    'relativeKey': key.relative_key.name, 'vertexGroup': key.vertex_group, 'mute': key.mute})
        objects[obj.name] = {'topology': _topology(mesh),
            'rawCoordinatesSha256': hashlib.sha256(_coords(mesh).tobytes()).hexdigest(),
            'keys': key_rows, 'groups': [g.name for g in obj.vertex_groups],
            'weightsSha256': _digest([[(g.group, g.weight) for g in v.groups] for v in mesh.vertices]),
            'matrixWorld': [list(row) for row in obj.matrix_world],
            'modifiers': [(m.name, m.type, m.show_viewport, m.show_render) for m in obj.modifiers]}
    data = {'colliders': objects, 'controlTopology': _topology(control.data),
        'controlCoordinatesSha256': hashlib.sha256(_coords(control.data).tobytes()).hexdigest(),
        'rigWorld': [list(row) for row in rig.matrix_world],
        'boneRest': {bone.name: [list(row) for row in bone.matrix_local] for bone in rig.data.bones}}
    return {'sha256': _digest(data), 'inputs': data}


def _setup(source, settings_path, out):
    _open(source)
    rig = _rig()
    action = _activate(rig)
    if bpy.data.objects.get(CONTROL) or bpy.data.collections.get(COLLECTION) or bpy.data.objects.get(NATIVE):
        raise RuntimeError('R09 cloth already exists; preserve it and use a fresh derivative')
    if settings_path.exists():
        raise RuntimeError('New settings output already exists; do not overwrite it')
    suspended = _suspend_old_dynamics(rig)
    # runpy's returned dict is not assumed identical to a function's globals.
    namespace = BASE['run'].__globals__
    for name, value in {'PREFIX': PREFIX, 'CONTROL': CONTROL, 'COLLECTION': COLLECTION,
                        'COLLIDERS': COLLIDERS, 'PIN': 'R09 waist pins'}.items():
        namespace[name] = value
    # The base helper copies this object's flags into its temporary shape-key
    # evaluation object. Keep it evaluable while deriving the fresh rest cloth.
    original_skirt = bpy.data.objects[BASE['ORIGINALS'][0]]
    original_skirt.hide_viewport = False
    original_skirt.hide_set(False)
    configured = BASE['run'](frame_start=START, frame_end=END, follow_rig=True)
    original_skirt.hide_viewport = True
    original_skirt.hide_set(True)
    control = bpy.data.objects[CONTROL]
    if len(control.data.vertices) != COUNT or [m.type for m in control.modifiers] != ['ARMATURE', 'CLOTH', 'SUBSURF', 'SOLIDIFY']:
        raise RuntimeError('Expected the reviewed 2176-point Armature/Cloth/Subsurf/Solidify control')
    if len(bpy.data.collections[COLLIDERS].objects) != 5:
        raise RuntimeError('Expected exactly five fresh R09 body collision copies')
    cloth = next(m for m in control.modifiers if m.type == 'CLOTH')
    cache = cloth.point_cache
    if (cache.frame_start, cache.frame_end) != (START, END):
        raise RuntimeError('This Blender did not retain the requested negative pre-roll cache range')
    cache.name = 'R09_Body_Skirt'
    cache.use_disk_cache = True
    cache.use_external = False
    post_flags = {}
    for modifier in control.modifiers:
        if modifier.type in {'SUBSURF', 'SOLIDIFY'}:
            post_flags[modifier.name] = {'viewport': modifier.show_viewport, 'render': modifier.show_render}
            modifier.show_viewport = False
            modifier.show_render = False
    control.hide_viewport = False
    control.hide_render = True
    control.hide_set(False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = START, END
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = FIRST, END
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = 6
    _set_frame(START)
    inputs = _input_fingerprint(rig, control)
    report = {'schemaVersion': 1, 'status': 'settings-created; capture-pending',
        'sourceBlend': _relative(source), 'sourceSha256': _sha(source),
        'settingsBlend': _relative(settings_path), 'blender': bpy.app.version_string,
        'action': ACTION, 'actionSha256': _action_hash(action), 'control': CONTROL,
        'controlTopologySha256': _topology(control.data), 'uvSha256': _uv_hash(control.data),
        'inputFingerprint': inputs, 'vertexCount': COUNT, 'simulationFrames': [START, END],
        'playbackFrames': [FIRST, END], 'prerollFrames': FIRST - START, 'fps': 30,
        'postModifierFlags': post_flags, 'suspendedOldDynamics': suspended,
        'configuration': configured, 'rigWorld': [list(row) for row in rig.matrix_world],
        'pointCache': API['_cache_snapshot'](cache),
        'coordinateSpace': 'world-space float32, from actual evaluated object.to_mesh after Cloth only',
        'scope': 'Fresh R09 native Cloth solver samples. Finite sanity and playback parity; no full collision or visual acceptance claim.',
        'limitations': ['Simplified 2176-point pleated control surface, not full garment tailoring',
            'Five body colliders and self-collision; no sleeve cloth simulation',
            'Old action caches are retained but disabled in this new derivative',
            'Hair dynamics remain disabled until a matching groom/cache is integrated']}
    control['r09_cloth_capture'] = json.dumps({k: v for k, v in report.items() if k != 'inputFingerprint'})
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(settings_path))
    report['settingsSha256'] = _sha(settings_path)
    _write(out / 'capture-report.json', report)
    return rig, control, report


def capture(source, settings_path, out):
    """One continuous pass. The settings copy is saved before the first step."""
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use a new empty output directory; preserve existing reports/payloads')
    rig, control, report = _setup(source, settings_path, out)
    points = np.empty((END, COUNT, 3), dtype=np.float32)
    rows = []
    started = time.monotonic()
    original_matrix = np.asarray(rig.matrix_world, dtype=np.float64)
    previous = None
    max_step = 0.
    try:
        for frame in range(START, END + 1):
            _set_frame(frame)
            world, topology = _evaluated(control)
            if world.shape != (COUNT, 3) or topology != report['controlTopologySha256']:
                raise RuntimeError(f'Unexpected evaluated control topology at frame {frame}')
            if not np.isfinite(world).all() or float(np.abs(world).max()) > 5.:
                raise RuntimeError(f'Non-finite or exploded cloth bounds at frame {frame}')
            # Rig-object motion would be double counted by the native parent.
            if float(np.max(np.abs(np.asarray(rig.matrix_world) - original_matrix))) > 1e-8:
                raise RuntimeError('Animated rig object transform needs an explicit playback adaptation')
            step = 0. if previous is None else float(np.linalg.norm(world - previous, axis=1).max())
            if step > .20:
                raise RuntimeError(f'Cloth moved over 20 cm in one 30 fps step at {frame}: {step}')
            max_step = max(max_step, step)
            previous = world
            if frame >= FIRST:
                points[frame - FIRST] = world
            if frame in {START, 0, 1, 55, 151, 247, END}:
                rows.append({'frame': frame, 'minimum': world.min(axis=0).tolist(),
                             'maximum': world.max(axis=0).tolist(), 'maxStepMeters': step})
            if (frame - START) % 30 == 0 or frame == END:
                progress = {'frame': frame, 'last': END, 'elapsedSeconds': round(time.monotonic() - started, 2)}
                _write(out / 'progress.json', progress)
                print('R09_CLOTH_CAPTURE ' + json.dumps(progress), flush=True)
        if _action_hash(rig.animation_data.action) != report['actionSha256']:
            raise RuntimeError('Body action changed during capture')
        if _sha(source) != report['sourceSha256']:
            raise RuntimeError('Input source file changed during capture')
        payload = out / 'cloth-world-1-301.npz'
        np.savez_compressed(payload, frames=np.arange(FIRST, END + 1, dtype=np.int32),
                            positions_world=points, rig_world=original_matrix.astype(np.float64))
        cache = next(m for m in control.modifiers if m.type == 'CLOTH').point_cache
        cache_dir = settings_path.parent / ('blendcache_' + settings_path.stem)
        disk = [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(cache_dir.glob('*')) if p.is_file()]
        report.update(status='captured-actual-evaluated-cloth; native-playback-pending',
            payload=_relative(payload), payloadSha256=_sha(payload), payloadBytes=payload.stat().st_size,
            positionBytes=points.nbytes, frameCount=END, captureSeconds=time.monotonic() - started,
            sampledBounds=rows, maxOneFrameVertexDisplacementMeters=max_step,
            diskCacheDirectory=_relative(cache_dir), observedDiskFiles=disk,
            pointCacheAfter={'isBaked': bool(cache.is_baked), 'isOutdated': bool(cache.is_outdated), 'info': cache.info},
            note='Sequential evaluated simulation captured directly. Disk cache is supplementary; isBaked is reported, not assumed.')
        _write(out / 'capture-report.json', report)
        print('R09_CLOTH_CAPTURE_COMPLETE ' + json.dumps({'payload': str(payload), 'seconds': report['captureSeconds']}), flush=True)
        return report
    except Exception as error:
        report.update(status='capture-failed', error=str(error), traceback=traceback.format_exc(), sampledBounds=rows)
        _write(out / 'capture-report.json', report)
        raise


def _load(out):
    report = json.loads((out / 'capture-report.json').read_text(encoding='utf-8'))
    if not report['status'].startswith('captured-'):
        raise RuntimeError('A successfully completed actual capture is required')
    path = _path(report['payload'])
    if _sha(path) != report['payloadSha256']:
        raise RuntimeError('Captured payload identity changed')
    with np.load(path, allow_pickle=False) as data:
        frames = data['frames'].copy()
        points = data['positions_world'].copy()
        matrix = data['rig_world'].copy()
    if not np.array_equal(frames, np.arange(FIRST, END + 1)) or points.shape != (END, COUNT, 3) or points.dtype != np.float32:
        raise RuntimeError('Expected exactly 301 world-space float32 samples with 2176 vertices')
    if not np.isfinite(points).all():
        raise RuntimeError('Payload contains non-finite positions')
    return report, points, matrix


def _parity(obj, points, frames=CHECK_FRAMES):
    flags = [(m, m.show_viewport) for m in obj.modifiers]
    errors = {}
    try:
        for modifier, _ in flags:
            modifier.show_viewport = False
        for frame in frames:
            _set_frame(frame)
            actual, _ = _evaluated(obj)
            low, high = math.floor(frame) - FIRST, math.ceil(frame) - FIRST
            t = frame % 1
            expected = (1. - t) * points[low] + t * points[high]
            if actual.shape != expected.shape:
                raise RuntimeError('Native playback point count changed')
            errors[str(frame)] = float(np.max(np.abs(actual - expected)))
        if max(errors.values()) > TOLERANCE:
            raise RuntimeError('Native point parity failed: ' + repr(errors))
        return errors
    finally:
        for modifier, flag in flags:
            modifier.show_viewport = flag


def _surface_samples(obj):
    result = {}
    for frame in (1, 55, 151, 247, 301):
        _set_frame(frame)
        points, topology = _evaluated(obj)
        # All actual downstream viewport points, stored separately in compressed NPZ.
        result[str(frame)] = points
    return result


def embed(out, target, already_open=False):
    report, points, matrix = _load(out)
    settings = _path(report['settingsBlend'])
    if _sha(settings) != report['settingsSha256']:
        raise RuntimeError('Settings source changed since capture')
    if target.exists() or target == settings or target == _path(report['sourceBlend']):
        raise RuntimeError('Use a fresh separate native-playback output Blend')
    if not already_open:
        _open(settings)
    elif Path(bpy.data.filepath).resolve() != settings:
        raise RuntimeError('Expected the exact current capture settings copy')
    rig = _rig()
    action = _activate(rig)
    source = bpy.data.objects[CONTROL]
    if _action_hash(action) != report['actionSha256'] or _topology(source.data) != report['controlTopologySha256']:
        raise RuntimeError('Action or simulation topology differs from captured input')
    if bpy.data.objects.get(NATIVE) or bpy.data.actions.get(PAYLOAD_ACTION):
        raise RuntimeError('Native R09 playback already exists; do not replace it')
    # Disable simulation before any seek; preserve settings/cache, never replay it
    # backwards to make the parity check depend on potentially missing bphys files.
    for modifier in source.modifiers:
        if modifier.type == 'CLOTH':
            modifier.show_viewport = False
            modifier.show_render = False
    source.hide_viewport = source.hide_render = True
    source.hide_set(True)
    _set_frame(FIRST)
    if float(np.max(np.abs(np.asarray(rig.matrix_world) - matrix))) > 1e-8:
        raise RuntimeError('Rig object transform differs from capture')
    inverse = np.linalg.inv(matrix)
    local = (points.astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.float32)
    mesh = source.data.copy()
    mesh.name = 'R09 simulated cloth payload mesh'
    mesh.vertices.foreach_set('co', local[0].ravel())
    mesh.update()
    obj = bpy.data.objects.new(NATIVE, mesh)
    collection = bpy.data.collections.new(NATIVE_COLLECTION)
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)
    obj.parent = rig
    obj.parent_type = 'OBJECT'
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    # Deliberately no copied vertex groups, Armature or Cloth deformation.
    obj.shape_key_add(name='Basis', from_mix=False)
    for frame in range(FIRST, END + 1):
        key = obj.shape_key_add(name=f'R09 simulated frame {frame:03d}', from_mix=False)
        key.data.foreach_set('co', local[frame - FIRST].ravel())
        for at, value in ((frame - 1, 0.), (frame, 1.), (frame + 1, 0.)):
            if FIRST <= at <= END:
                key.value = value
                key.keyframe_insert(data_path='value', frame=at, group='Recorded R09 Cloth simulation')
        key.value = 0.
    keys = obj.data.shape_keys
    keys.use_relative = True
    shape_action = keys.animation_data.action
    shape_action.name = PAYLOAD_ACTION
    for curve in HELPER['_fcurves'](shape_action):
        curve.extrapolation = 'CONSTANT'
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'
        curve.update()
    for original in source.modifiers:
        if original.type in {'SUBSURF', 'SOLIDIFY'}:
            modifier = obj.modifiers.new(original.name, original.type)
            API['_copy_settings'](original, modifier, {'object'})
            flags = report['postModifierFlags'][original.name]
            modifier.show_viewport, modifier.show_render = flags['viewport'], flags['render']
    obj.hide_viewport = obj.hide_render = False
    obj.hide_set(False)
    obj.display_type = 'TEXTURED'
    if _topology(mesh) != report['controlTopologySha256'] or _uv_hash(mesh) != report['uvSha256']:
        raise RuntimeError('Native topology/UV/material assignment differs from simulation')
    if [m.type for m in obj.modifiers] != ['SUBSURF', 'SOLIDIFY']:
        raise RuntimeError('Unexpected native modifier stack; no Armature is permitted')
    errors = _parity(obj, points, tuple(range(FIRST, END + 1)) + CHECK_FRAMES)
    surface = _surface_samples(obj)
    surface_path = out / 'native-viewport-surface.npz'
    np.savez_compressed(surface_path, **surface)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FIRST, END
    scene.frame_preview_start, scene.frame_preview_end = FIRST, END
    _set_frame(FIRST)
    if _action_hash(action) != report['actionSha256']:
        raise RuntimeError('Body action changed while embedding cloth')
    result = {'status': 'native-playback-saved; fresh-process-reopen-and-visual-review-pending',
        'settingsBlend': report['settingsBlend'], 'targetBlend': _relative(target),
        'payload': report['payload'], 'payloadSha256': report['payloadSha256'],
        'action': ACTION, 'actionSha256': report['actionSha256'],
        'object': NATIVE, 'sourceControl': CONTROL, 'sourceControlRetainedDisabled': True,
        'shapeKeyCount': len(keys.key_blocks), 'shapeActionSha256': _action_hash(shape_action),
        'topologySha256': _topology(mesh), 'uvSha256': _uv_hash(mesh),
        'pointParityErrorsMeters': errors, 'maxPointParityErrorMeters': max(errors.values()),
        'surfaceReference': _relative(surface_path), 'surfaceReferenceSha256': _sha(surface_path),
        'scope': report['scope'], 'integerFramesVerified': END,
        'fractionalPlayback': 'Linear interpolation of actual simulated world positions; not subframe physical simulation',
        'hairDynamics': 'Old caches disabled; no matching R09 hair is claimed'}
    obj['r09_native_cloth_report'] = json.dumps(result)
    text = bpy.data.texts.new('R09 Cloth direct-capture provenance.json')
    text.write(json.dumps({'capture': report, 'native': result}, ensure_ascii=False, indent=2))
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    result['targetSha256'] = _sha(target)
    _write(out / 'native-report.json', result)
    print('R09_CLOTH_NATIVE_SAVED ' + json.dumps({'target': str(target), 'maxPointError': max(errors.values())}), flush=True)
    return result


def verify(out, target):
    """Fresh process mode, no new simulation, no save, no render."""
    report, points, matrix = _load(out)
    native = json.loads((out / 'native-report.json').read_text(encoding='utf-8'))
    if target != _path(native['targetBlend']) or _sha(target) != native['targetSha256']:
        raise RuntimeError('Open the exact saved R09 native result')
    _open(target)
    rig = _rig()
    obj = bpy.data.objects[NATIVE]
    source = bpy.data.objects[CONTROL]
    if not rig.animation_data or not rig.animation_data.action or rig.animation_data.action.name != ACTION:
        raise RuntimeError('The saved matching R09 body action is not active')
    if _action_hash(rig.animation_data.action) != report['actionSha256']:
        raise RuntimeError('Body action differs from captured input')
    if [m.type for m in obj.modifiers] != ['SUBSURF', 'SOLIDIFY'] or obj.parent != rig or obj.parent_type != 'OBJECT':
        raise RuntimeError('Native playback parenting/modifiers changed')
    if any(m.type == 'CLOTH' and (m.show_viewport or m.show_render) for m in source.modifiers):
        raise RuntimeError('Preserved simulation unexpectedly enabled')
    if _topology(obj.data) != native['topologySha256'] or _uv_hash(obj.data) != native['uvSha256']:
        raise RuntimeError('Native topology/UV/material indices changed')
    keys = obj.data.shape_keys
    if len(keys.key_blocks) != END + 1 or _action_hash(keys.animation_data.action) != native['shapeActionSha256']:
        raise RuntimeError('Native shape-key animation changed')
    errors = _parity(obj, points)
    path = _path(native['surfaceReference'])
    if _sha(path) != native['surfaceReferenceSha256']:
        raise RuntimeError('Downstream surface reference identity changed')
    surface_errors = {}
    with np.load(path, allow_pickle=False) as reference:
        for frame, actual in _surface_samples(obj).items():
            expected = reference[frame]
            if actual.shape != expected.shape:
                raise RuntimeError('Downstream viewport surface point count changed')
            surface_errors[frame] = float(np.max(np.abs(actual - expected)))
    if max(surface_errors.values()) > TOLERANCE:
        raise RuntimeError('Reopened downstream viewport surface differs')
    result = {'status': 'finite-fresh-reopen-parity-passed; visual-cloth-review-still-required',
        'targetBlend': _relative(target), 'targetSha256': _sha(target),
        'pointErrorsMeters': errors, 'viewportSurfaceErrorsMeters': surface_errors,
        'scope': 'Finite integer/fractional native playback and downstream viewport surface parity. No new simulation, rendering, full collision or visual acceptance.'}
    _write(out / 'reopen-report.json', result)
    _set_frame(FIRST)
    print('R09_CLOTH_REOPEN ' + json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='work/reconstruction-r02/body-action-r09.blend')
    parser.add_argument('--settings', default='work/reconstruction-r02/body-cloth-settings-r09.blend')
    parser.add_argument('--output', default='work/reconstruction-r02/body-cloth-native-r09.blend')
    parser.add_argument('--data-dir', default='work/reconstruction-r02/cloth-body-r09')
    parser.add_argument('--mode', choices=('all', 'capture', 'embed', 'verify'), default='all')
    args = parser.parse_args(argv)
    if not bpy.app.background:
        raise RuntimeError('Explicit serial background Blender required')
    source, settings, target, out = map(_path, (args.source, args.settings, args.output, args.data_dir))
    if len({source, settings, target}) != 3:
        raise RuntimeError('Source, settings and native output must be distinct work copies')
    if args.mode in {'all', 'capture'}:
        if settings.exists() or target.exists():
            raise RuntimeError('Keep existing outputs; use new settings and native paths')
        capture(source, settings, out)
    if args.mode in {'all', 'embed'}:
        embed(out, target, already_open=args.mode == 'all')
    if args.mode == 'verify':
        verify(out, target)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
