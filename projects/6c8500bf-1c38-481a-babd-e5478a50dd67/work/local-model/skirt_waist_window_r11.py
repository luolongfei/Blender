"""Bounded native Cloth experiments for the retained R09 waist transition.

Offline analysis reads the real captured solver points. A capture creates a
fresh control/modifier/cache from the exact saved R09 settings and never edits
the production skirt or old cache. Every trial solves continuously from -59 to
193; only frame 1 and 121..193 are retained. It does NOT restart at frame 121
without velocity, nor inject a previous cache as a substitute simulation.

Profiles: baseline8, quality12, seam12. The seam profile changes only solver
quality and a tapered structural-stiffness group at the waist; pins, gravity,
mass, bending, global stiffness, collisions and body action remain unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

PROFILES = ('baseline8', 'quality12', 'seam12')
SOURCE = 'work/reconstruction-r02/body-cloth-settings-r09.blend'
CAPTURE = 'work/reconstruction-r02/cloth-body-r09/capture-report.json'
FIRST, WINDOW_START, LAST = -59, 121, 193
KEEP = np.array([1]+list(range(WINDOW_START, LAST+1)), np.int32)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _new_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _old_payload():
    p = project_path()
    record = _read(p/CAPTURE)
    payload = p/record['payload']
    if _sha(payload) != record['payloadSha256']:
        raise RuntimeError('Original solver payload identity changed')
    with np.load(payload, allow_pickle=False) as data:
        frames = data['frames'].copy()
        points = data['positions_world'].astype(np.float64)
    if points.shape != (301, 2176, 3) or not np.array_equal(frames, np.arange(1, 302)):
        raise RuntimeError('Expected the actual original 301-frame control payload')
    return record, points


def _metrics(points, frames, reference):
    q = np.asarray(points, float).reshape(len(frames), 17, 128, 3)
    reference = np.asarray(reference, float).reshape(17, 128, 3)
    circum = np.linalg.norm(q-np.roll(q, -1, axis=2), axis=3)
    vertical = np.linalg.norm(np.diff(q, axis=1), axis=3)
    ref_circum = np.linalg.norm(reference-np.roll(reference, -1, axis=1), axis=2)
    ref_vertical = np.linalg.norm(np.diff(reference, axis=0), axis=2)
    cr, vr = circum/ref_circum, vertical/ref_vertical
    worst = tuple(map(int, np.unravel_index(np.argmax(cr), cr.shape)))
    f, r, c = worst
    return {'circumferenceRatiosToOriginalFrameOne': np.quantile(cr, (0, .01, .5, .99, 1)).tolist(),
        'verticalRatiosToOriginalFrameOne': np.quantile(vr, (0, .01, .5, .99, 1)).tolist(),
        'worstCircumferenceEdge': {'frame': int(frames[f]), 'row': r, 'column': c,
            'referenceLengthMeters': float(ref_circum[r, c]), 'lengthMeters': float(circum[worst]),
            'absoluteIncreaseMeters': float(circum[worst]-ref_circum[r, c]), 'ratio': float(cr[worst])},
        'perFrame': [{'frame': int(frame), 'maximumCircumferenceRatio': float(cr[i].max()),
                      'maximumVerticalRatio': float(vr[i].max()),
                      'hemMinimumZ': float(q[i, -1, :, 2].min())} for i, frame in enumerate(frames)]}


def analyze(out):
    record, points = _old_payload()
    q = points.reshape(301, 17, 128, 3)
    base = q[0, 0]
    centered = base-base.mean(0)
    residual = []
    for pose in q:
        pin = pose[0]
        u, singular, vt = np.linalg.svd(centered.T@(pin-pin.mean(0)))
        rotation = u@vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u@vt
        target = (q[0]-base.mean(0))@rotation+pin.mean(0)
        residual.append(np.linalg.norm(pose-target, axis=2))
    residual = np.asarray(residual)
    result = {'sourceCapture': CAPTURE, 'sourceCaptureSha256': _sha(project_path()/CAPTURE),
        'payloadSha256': record['payloadSha256'], 'metrics': _metrics(points, np.arange(1, 302), points[0]),
        'pinnedRowRigidFitMaximumResidualMeters': float(residual[:, 0].max()),
        'softRowOneRigidTargetResidualQuantilesMeters': np.quantile(residual[:, 1], (.5, .95, .99, 1)).tolist(),
        'frame181SoftRowOneResidualQuantilesMeters': np.quantile(residual[180, 1], (0, .5, .95, 1)).tolist(),
        'interpretation': '19% refers to a 2.175mm fold edge lengthening by about 0.415mm, not a 19% global skirt stretch. Physical/visible significance remains unproven.',
        'trialDesign': {'solverInterval': [FIRST, LAST], 'retainedFrames': KEEP.tolist(),
            'baseline8': 'Unchanged solver/weights for reproducibility witness.',
            'quality12': 'Only quality 8 to 12; isolates solver convergence sensitivity.',
            'seam12': 'Quality12 plus 45 maximum tension/compression restricted to original top 3 rows with weights 1/.85/.25; no pin-weight change.',
            'allOtherProperties': 'Preserved. New independent modifier/cache for each trial.'},
        'officialPropertyReference': 'https://docs.blender.org/manual/en/latest/physics/cloth/settings/property_weights.html',
        'limits': ['Solver parameters are not laboratory fabric calibration.',
            'A smaller edge ratio alone does not establish better cloth motion or appearance.',
            'No simulation or current R11 source edit is performed by this offline analysis.']}
    _new_json(out, result)
    print('R11_SKIRT_WAIST_ANALYZED '+str(out), flush=True)
    return result


def _settings(owner):
    result = {}
    for prop in owner.bl_rna.properties:
        if prop.identifier == 'rna_type' or prop.is_readonly or prop.type == 'COLLECTION':
            continue
        value = getattr(owner, prop.identifier)
        if prop.type == 'POINTER':
            result[prop.identifier] = value.name if value and hasattr(value, 'name') else None
        elif isinstance(value, (int, float, str, bool)):
            result[prop.identifier] = value
        elif isinstance(value, set):
            result[prop.identifier] = sorted(value)
        else:
            try:
                result[prop.identifier] = list(value)
            except TypeError:
                result[prop.identifier] = str(value)
    return result


def _weights(obj):
    return {'groups': [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups],
            'points': [[(g.group, g.weight) for g in v.groups] for v in obj.data.vertices]}


def capture(profile, out, baseline_path=None):
    import bpy
    C = _api('cloth_body_r09')
    p = project_path()
    original, old_points = _old_payload()
    source = (p/SOURCE).resolve()
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the exact original saved R09 cloth settings in a fresh background process')
    if _sha(source) != original['settingsSha256']:
        raise RuntimeError('Original settings source is no longer the recorded source')
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use a new empty trial directory')
    baseline = None
    if profile != 'baseline8':
        if baseline_path is None:
            raise RuntimeError('Run and supply the actual baseline8 report before changed profiles')
        baseline = _read(baseline_path)
        if (baseline.get('profile') != 'baseline8' or not baseline.get('baselineReproduced')
                or baseline.get('settingsSourceSha256') != original['settingsSha256']
                or baseline.get('originalPayloadSha256') != original['payloadSha256']):
            raise RuntimeError('A matching successful unchanged baseline witness is required')
        for field in ('payload', 'settingsBlend'):
            if _sha(p/baseline[field]) != baseline[field+'Sha256']:
                raise RuntimeError('Baseline output changed: '+field)
    rig = bpy.data.objects['Character_Rig']
    old = bpy.data.objects[C['CONTROL']]
    old_cloth = next(m for m in old.modifiers if m.type == 'CLOTH')
    if old.data.shape_keys or old.constraints or old.animation_data:
        raise RuntimeError('The observed original control no longer matches its plain mesh contract')
    if C['_input_fingerprint'](rig, old) != original['inputFingerprint']:
        raise RuntimeError('Actual original rig/collision/control inputs differ from captured settings')
    if C['_action_hash'](rig.animation_data.action) != original['actionSha256']:
        raise RuntimeError('Keep the exact original body action; new arm timing is a separate future comparison')
    original_weights = _weights(old)
    original_settings = _settings(old_cloth.settings)
    original_collision = _settings(old_cloth.collision_settings)
    C['_suspend_old_dynamics'](rig)
    suffix = 'R11 Waist '+profile
    collection = bpy.data.collections.new(suffix+' trial')
    bpy.context.scene.collection.children.link(collection)
    obj = bpy.data.objects.new(suffix+' control', old.data.copy())
    collection.objects.link(obj)
    obj.parent, obj.parent_type, obj.parent_bone = old.parent, old.parent_type, old.parent_bone
    obj.matrix_parent_inverse = old.matrix_parent_inverse.copy()
    obj.matrix_world = old.matrix_world.copy()
    for group in old.vertex_groups:
        new = obj.vertex_groups.new(name=group.name)
        if new.index != group.index:
            raise RuntimeError('Fresh cloth group index differs')
        new.lock_weight = group.lock_weight
    if _weights(obj) != original_weights:
        raise RuntimeError('Mesh copy did not preserve exact original pin/hip weights')
    for old_modifier in old.modifiers:
        modifier = obj.modifiers.new(suffix+' '+old_modifier.type, old_modifier.type)
        if modifier.type == 'CLOTH':
            C['API']['_copy_settings'](old_modifier.settings, modifier.settings)
            C['API']['_copy_settings'](old_modifier.collision_settings, modifier.collision_settings, {'collection'})
            C['API']['_copy_settings'](old_modifier.settings.effector_weights, modifier.settings.effector_weights, {'collection'})
            modifier.show_viewport, modifier.show_render = True, True
            cloth = modifier
        else:
            C['API']['_copy_settings'](old_modifier, modifier, {'object'})
            if modifier.type in {'SUBSURF', 'SOLIDIFY'}:
                modifier.show_viewport = modifier.show_render = False
    if _settings(cloth.settings) != original_settings or _settings(cloth.collision_settings) != original_collision:
        raise RuntimeError('Fresh solver copy did not preserve its original parameters')
    if profile in {'quality12', 'seam12'}:
        cloth.settings.quality = 12
    seam = None
    if profile == 'seam12':
        if cloth.settings.vertex_group_structural_stiffness:
            raise RuntimeError('Existing structural group requires explicit integration')
        seam = obj.vertex_groups.new(name='R11 local waist structural reinforcement')
        for row, weight in ((0, 1.), (1, .85), (2, .25)):
            seam.add(list(range(row*128, (row+1)*128)), weight, 'REPLACE')
        cloth.settings.vertex_group_structural_stiffness = seam.name
        cloth.settings.tension_stiffness_max = 45.
        cloth.settings.compression_stiffness_max = 45.
    changed_settings = {key: {'old': value, 'new': _settings(cloth.settings)[key]}
                        for key, value in original_settings.items() if _settings(cloth.settings)[key] != value}
    allowed = set() if profile == 'baseline8' else {'quality'}
    if profile == 'seam12':
        allowed |= {'vertex_group_structural_stiffness', 'tension_stiffness_max', 'compression_stiffness_max'}
    if set(changed_settings) != allowed:
        raise RuntimeError('Unexpected solver-setting change: '+repr(changed_settings))
    # The two original groups retain exactly their original per-point values.
    now = _weights(obj)
    filtered = {'groups': now['groups'][:len(original_weights['groups'])],
                'points': [[pair for pair in row if pair[0] < len(original_weights['groups'])] for row in now['points']]}
    if filtered != original_weights or _weights(old) != original_weights:
        raise RuntimeError('Original pins or hip weights changed')
    cache = cloth.point_cache
    cache.frame_start, cache.frame_end, cache.frame_step = FIRST, LAST, 1
    cache.use_disk_cache, cache.use_external = True, False
    cache.name = 'R11_Waist_'+profile
    if cache.is_baked or cache.as_pointer() == old_cloth.point_cache.as_pointer():
        raise RuntimeError('Trial did not receive an independent empty cache')
    if (cache.frame_start, cache.frame_end) != (FIRST, LAST):
        raise RuntimeError('Solver cache did not retain requested continuous warmup range')
    obj.hide_viewport, obj.hide_render = False, True
    obj.hide_set(False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FIRST, LAST
    scene.frame_preview_start, scene.frame_preview_end = WINDOW_START, LAST
    scene.render.threads_mode, scene.render.threads = 'FIXED', 6
    C['_set_frame'](FIRST)
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out/'trial-settings.blend'
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(settings_path)):
        raise RuntimeError('Cannot save independent trial settings')
    report = {'profile': profile, 'settingsSource': SOURCE, 'settingsSourceSha256': _sha(source),
        'originalCapture': CAPTURE, 'originalCaptureSha256': _sha(p/CAPTURE),
        'originalPayloadSha256': original['payloadSha256'], 'helperSha256': _sha(Path(__file__)),
        'settingsBlend': settings_path.relative_to(p).as_posix(), 'settingsBlendSha256': _sha(settings_path),
        'object': obj.name, 'solverFrames': [FIRST, LAST], 'retainedFrames': KEEP.tolist(),
        'changedSettings': changed_settings, 'pinWeightsChanged': False,
        'originalInputsFingerprint': original['inputFingerprint']['sha256'],
        'actionSha256': original['actionSha256'], 'baselineReport': str(baseline_path) if baseline_path else None,
        'baselineReportSha256': _sha(baseline_path) if baseline_path else None,
        'originalSourceAndCacheRetained': True, 'status': 'running; no result or acceptance yet'}
    samples = []
    started = time.monotonic()
    try:
        for frame in range(FIRST, LAST+1):
            C['_set_frame'](frame)
            world, topology = C['_evaluated'](obj)
            if world.shape != (2176, 3) or topology != original['controlTopologySha256']:
                raise RuntimeError('Trial control topology changed')
            if not np.isfinite(world).all() or np.abs(world).max() > 5.:
                raise RuntimeError('Invalid or exploded trial geometry')
            if frame in KEEP:
                samples.append(world.copy())
            if (frame-FIRST) % 30 == 0 or frame == LAST:
                print('R11_WAIST_TRIAL_PROGRESS '+json.dumps({'profile': profile, 'frame': frame,
                      'last': LAST, 'seconds': round(time.monotonic()-started, 2)}), flush=True)
        points = np.asarray(samples, np.float32)
        if points.shape != (len(KEEP), 2176, 3):
            raise RuntimeError('Incomplete retained solver window')
        if _sha(source) != report['settingsSourceSha256'] or C['_action_hash'](rig.animation_data.action) != original['actionSha256']:
            raise RuntimeError('Original source/body action changed during trial')
        payload = out/'trial-window.npz'
        np.savez_compressed(payload, frames=KEEP, positions_world=points)
        delta = np.linalg.norm(points.astype(float)-old_points[KEEP-1], axis=2)
        baseline_error = float(delta.max())
        report.update(status='captured actual solver window; visual and collision review pending',
            payload=payload.relative_to(p).as_posix(), payloadSha256=_sha(payload), elapsedSeconds=time.monotonic()-started,
            metrics=_metrics(points, KEEP, old_points[0]),
            differenceToOriginalSolver={'maximumMeters': baseline_error,
                'perFrameRmsMeters': [{'frame': int(frame), 'rms': float(np.sqrt(np.mean(delta[i]**2)))}
                                     for i, frame in enumerate(KEEP)]},
            baselineReproduced=bool(profile == 'baseline8' and baseline_error <= 2e-5),
            baselineToleranceMeters=2e-5,
            limits=['Parameter study on original R09 action and collision inputs, not the new R11 combined character.',
                'Window warms up from the original start; no guessed initial velocity or old cached state injection.',
                'Smaller edge ratios are not automatic acceptance. Inspect waist folds, free skirt motion and collisions.',
                'No source promotion, runtime export, physical calibration or final appearance claim.'])
        _new_json(out/'trial-report.json', report)
        if profile == 'baseline8' and not report['baselineReproduced']:
            raise RuntimeError('Unchanged trial does not reproduce old solver within 20um; do not compare changed profiles')
        print('R11_WAIST_TRIAL_COMPLETE '+json.dumps({'profile': profile, 'output': str(out),
              'maximumDifferenceToOldMeters': baseline_error}), flush=True)
        return report
    except Exception as error:
        if not (out/'trial-report.json').exists():
            report.update(status='trial failed; no acceptance', error=str(error), traceback=traceback.format_exc())
            _new_json(out/'trial-report.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('analyze', 'capture'), required=True)
    parser.add_argument('--profile', choices=PROFILES)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-report')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    output = (project_path()/args.output).resolve()
    output.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if args.mode == 'analyze':
        analyze(output)
    else:
        if not args.profile:
            raise RuntimeError('Supply an explicit trial --profile')
        baseline = (project_path()/args.baseline_report).resolve() if args.baseline_report else None
        capture(args.profile, output, baseline)
