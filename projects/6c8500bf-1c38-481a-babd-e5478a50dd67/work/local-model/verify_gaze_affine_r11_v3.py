"""Offline affine gaze audit using actual saved optical matrices and eye export.

No Blender process, driver dependency graph, rendered appearance or eyelid
contact is evaluated here. Production v3 prepare and fresh verify remain due.
"""
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    folder = project_path() / 'work/reconstruction-r02'
    optical_path = folder / 'reference-adult-eye-depth-r11-v2.eye-depth.json'
    optical = json.loads(optical_path.read_text())
    if optical['outputSha256'] != '055a4e6a1e59a687f06770d3ded58d0fe4ac02b5c8ff14ff0189706c4d5eccfd':
        raise RuntimeError('Expected actual 055a saved optical source report')
    helper = Path(__file__).with_name('adult_gaze_control_r11_v3.py')
    frozen = Path(__file__).with_name('adult_gaze_control_r11_v2.py')
    if sha(frozen) != '7d064176ec7fffc9c0126f149b67a72e4a6396bc3ddc4eb74b80b9515c66f55d':
        raise RuntimeError('Frozen v2 changed')
    api = runpy.run_path(str(helper))
    meta_path = folder / 'adult-blink-current-shape-r11.json'
    array_path = folder / 'adult-blink-current-shape-r11.npz'
    meta = json.loads(meta_path.read_text())
    if sha(array_path) != meta['arraysSha256']: raise RuntimeError('Observed geometry checksum mismatch')
    arrays = np.load(array_path)
    actual_a = np.asarray(optical['eyes']['L']['plan']['opticalFrame']['worldMatrix'])[:3, :3]
    rz = api['_metric_rotation'](23, 13)
    cases = [('actual-055a', actual_a), ('synthetic-rigid-head-turn', rz @ actual_a),
        ('synthetic-anisotropic', rz @ actual_a @ np.diag([1.025, .985, 1.01])),
        ('synthetic-shear', rz @ actual_a @ np.array([[1.01, .008, 0], [.018, .99, .012], [0, 0, 1.015]]))]
    offsets = [(0, -2, 0), (.4, -2, 0), (-.4, -2, 0), (0, -.4, 0), (0, -2, .35), (0, -2, -.35),
        (3, -.1, 3), (-3, -.1, -3), (-3, -.1, 3), (3, -.1, -3), (0, .2, 0)]
    rows = []; longest = 0; max_error = 0.; current_geometry_error = 0.
    env = {n: getattr(math, n) for n in ('atan2', 'sqrt', 'sin', 'cos', 'pi')}
    env.update(min=min, max=max)
    for side in ('L', 'R'):
        eye = optical['eyes'][side]
        if eye['outerWorldCoordinatesSha256Before'] != eye['outerWorldCoordinatesSha256After']:
            raise RuntimeError('Actual optical report does not preserve outer globe')
        observed = arrays['eye_' + side + '_world']
        observed_center = (observed.min(0) + observed.max(0)) * .5
        original_scale = np.asarray(meta['headMatrixWorld'])[0, 0]
        local_vertices = (observed - observed_center) / original_scale
        center = np.array([.032 if side == 'L' else -.032, -.07, .15])
        for label, linear in cases:
            matrix = np.eye(4); matrix[:3, :3] = linear; matrix[:3, 3] = [.1, -.2, 1.3]
            frame = api['metric_frame'](matrix)
            metric = np.asarray(frame['localToMetricLinear']); inverse = np.asarray(frame['metricToLocalLinear'])
            q = np.asarray(frame['worldBasisColumns'])
            turned = matrix.copy(); turned[:3, :3] = rz @ linear
            metric_rigid_error = float(np.max(np.abs(np.asarray(api['metric_frame'](turned)['localToMetricLinear']) - metric)))
            if metric_rigid_error > 1e-12: raise RuntimeError('Rigid head motion changes metric')
            plan = api['driver_plan'](center, metric, inverse, side)
            longest = max(longest, *(len(e['expression']) for e in plan))
            max_direction = 0.; max_rigid = 0.; max_neutral = 0.; max_driver = 0.; ray_error = 0.
            for offset in offsets:
                target = inverse @ offset
                for influence in (0., .5, 1.):
                    result = api['solve_angles'](center, target, influence, metric)
                    values = {('target', 'location['+str(i)+']'): v for i, v in enumerate(target)}
                    values[('control', '["R11_gaze_influence"]')] = influence
                    euler = [0., 0., 0.]
                    for entry in plan:
                        variables = {n: values[(role, path)] for n, role, path in entry['variables']}
                        value = eval(entry['expression'], {'__builtins__': {}}, dict(env, **variables))
                        if entry['role'] == 'pivot': euler[entry['index']] = value
                        else: values[(entry['role'], entry['path'])] = value
                    max_driver = max(max_driver, max(abs(a-b) for a, b in zip(euler, result['localEulerXYZRadians'])))
                    local_rotation = api['_metric_rotation'](math.degrees(euler[2]), -math.degrees(euler[0]))
                    direction = linear @ local_rotation @ [0, -1, 0]; direction /= np.linalg.norm(direction)
                    desired = q @ result['forward']
                    max_direction = max(max_direction, float(np.linalg.norm(direction-desired)))
                    ideal = q @ api['_metric_rotation'](result['yawDegrees'], result['pitchDegrees']) @ q.T
                    actual = local_vertices @ (linear @ local_rotation).T
                    rigid = local_vertices @ (ideal @ linear).T
                    max_rigid = max(max_rigid, float(np.linalg.norm(actual-rigid, axis=1).max()))
                    if influence == 0: max_neutral = max(max_neutral, float(np.max(np.abs(local_rotation-np.eye(3)))))
                    if influence == 1 and not result['depthClamped'] and not result['angularClamped']:
                        miss = float(np.linalg.norm(np.cross(direction, linear @ (target-center))))
                        ray_error = max(ray_error, miss)
            if max_direction > 1e-12 or max_neutral > 1e-12 or max_driver > 1e-12 or ray_error > 1e-12:
                raise RuntimeError('Offline affine direction/neutral/driver failure')
            physical = max_rigid <= api['RIGID_TOLERANCE_METERS']
            if label in ('actual-055a', 'synthetic-rigid-head-turn'):
                if not physical: raise RuntimeError('Actual physical eye frame distorted')
                current_geometry_error = max(current_geometry_error, max_rigid)
            elif physical: raise RuntimeError('Severe synthetic scale/shear unexpectedly passes physical guard')
            max_error = max(max_error, max_direction)
            rows.append(dict(side=side, case=label, targetInfluenceCases=len(offsets)*3,
                maximumWorldDirectionError=max_direction, maximumDriverEulerErrorRadians=max_driver,
                maximumWorldRayMissMeters=ray_error, maximumNeutralMatrixError=max_neutral,
                maximumRigidVertexErrorMeters=max_rigid, physicalGlobeGuardWouldPass=physical,
                rigidHeadMetricInvarianceError=metric_rigid_error))
    report = dict(helper=str(helper.relative_to(project_path())), helperSha256=sha(helper),
        frozenV2Unchanged=True, actualOpticalReport=str(optical_path.relative_to(project_path())),
        actualOpticalReportSha256=sha(optical_path), actualOpticalSourceSha256=optical['outputSha256'],
        actualEyeExportSha256=sha(array_path), totalTargetInfluenceCases=sum(r['targetInfluenceCases'] for r in rows),
        driverMaximumExpressionCharacters=longest, maximumDirectionError=max_error,
        actualFrameMaximumRigidVertexErrorMeters=current_geometry_error, cases=rows,
        blenderExecuted=False, appearanceAccepted=False,
        limits=['Actual optical matrices and prior verified unchanged outer globe samples; no current driver graph evaluation.',
            'Synthetic shear/anisotropy proves ray calculation and physical rejection, not suitability for eye deformation.',
            'Real prepare, fresh-process verify, carrier poses and blink/lid follow combinations remain required.'])
    output = folder / 'adult-gaze-affine-v3-offline.json'
    with output.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('totalTargetInfluenceCases', 'driverMaximumExpressionCharacters',
        'maximumDirectionError', 'actualFrameMaximumRigidVertexErrorMeters', 'blenderExecuted')}))
    return report


if __name__ == '__main__': run()
