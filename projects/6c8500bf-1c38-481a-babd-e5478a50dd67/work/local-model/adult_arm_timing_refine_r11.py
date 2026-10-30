"""Copy the actual R09 action and refine only its two arm-raise transitions.

No motion generation, source-action editing, cache rebake or rendering. The new
action is left inactive. A later continuous review must use matching dynamics.
Existing keyframes 205/241/253/289 and the complete high-pose plateau are exact.
This changes shoulder/elbow timing, not the constant source wrist articulation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

try:
    import bpy
    from mathutils import Quaternion, Vector
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SOURCE_ACTION = 'R09 Body performance - authored squat bend reach'
SOURCE_SHA = '7a9ade79bbe6216d3accc73431baa48e691b55256ffaeb67788a27ab30b4b318'
ACTION = 'R11 R09 arm raise - shoulder elbow timing candidate'
RIG = 'Character_Rig'
MARKER = 'R11_arm_timing_candidate'
RAMPS = ((205., 241.), (253., 289.))
# Upper-arm displacement is at most one frame. The two sides differ by one
# frame and each forearm trails its corresponding upper arm by 0.75 frame.
DELAYS = {'leftShoulder': -.75, 'leftUpperArm': 0., 'leftLowerArm': .75,
          'rightShoulder': .25, 'rightUpperArm': 1., 'rightLowerArm': 1.75}
PROBE_FRAMES = (205., 213., 221., 221.5, 223., 229., 237., 241., 247.,
                253., 259., 267., 267.5, 269., 275., 283., 289.)


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def weight(frame):
    """Compact C2 window: exact zero at every source transition boundary."""
    for start, end in RAMPS:
        if start < frame < end:
            t = (frame - start) / (end - start)
            return 64. * t ** 3 * (1. - t) ** 3
    return 0.


def sample_time(frame, delay):
    return float(frame) - float(delay) * weight(float(frame))


def normalize(values):
    length = math.sqrt(sum(x * x for x in values))
    if not math.isfinite(length) or length < .5:
        raise RuntimeError('Invalid existing quaternion sample')
    return [x / length for x in values]


def _key(k):
    return {'co': list(k.co), 'left': list(k.handle_left), 'right': list(k.handle_right),
            'interpolation': k.interpolation, 'leftType': k.handle_left_type,
            'rightType': k.handle_right_type, 'easing': k.easing}


def curves(action):
    rows = api('full_body_review_r03')['_fcurves'](action)
    result = {(c.data_path, c.array_index): c for c in rows}
    if len(result) != len(rows):
        raise RuntimeError('Duplicate action channels need explicit handling')
    return result


def _snapshot(action):
    return {key: [_key(k) for k in curve.keyframe_points] for key, curve in curves(action).items()}


def create_action():
    if bpy is None:
        raise RuntimeError('Run explicitly in a fresh background Blender copy')
    rig = bpy.data.objects[RIG]
    if bpy.data.actions.get(ACTION):
        raise RuntimeError('Existing timing candidate must be preserved')
    source = bpy.data.actions.get(SOURCE_ACTION)
    audit = api('foot_contact_audit_r03')
    if source is None or audit['_action_hash'](source) != SOURCE_SHA:
        raise RuntimeError('Actual preserved R09 source action fingerprint differs')
    if any(rig.pose.bones[n].rotation_mode != 'QUATERNION' for n in DELAYS):
        raise RuntimeError('Expected existing quaternion arm channels')
    before = _snapshot(source)
    source_curves = curves(source)
    clone = None
    try:
        clone = source.copy()
        clone.name = ACTION
        clone.use_fake_user = True
        target_curves = curves(clone)
        changed = []
        for bone, delay in DELAYS.items():
            if delay == 0.:
                continue
            data_path = rig.pose.bones[bone].path_from_id('rotation_quaternion')
            channels = [source_curves[(data_path, i)] for i in range(4)]
            targets = [target_curves[(data_path, i)] for i in range(4)]
            frames = [float(k.co.x) for k in channels[0].keyframe_points]
            if frames != list(range(1, 302)) or any(
                [float(k.co.x) for k in c.keyframe_points] != frames or
                any(k.interpolation != 'LINEAR' for k in c.keyframe_points) or c.modifiers
                for c in channels):
                raise RuntimeError('Expected actual 301-sample LINEAR source channels')
            for index, frame in enumerate(frames):
                if weight(frame) == 0.:
                    continue
                t = sample_time(frame, delay)
                value = normalize([c.evaluate(t) for c in channels])
                original = [float(c.keyframe_points[index].co.y) for c in channels]
                if sum(a * b for a, b in zip(value, original)) < 0.:
                    value = [-v for v in value]
                for component, curve in enumerate(targets):
                    point = curve.keyframe_points[index]
                    delta = value[component] - float(point.co.y)
                    # Key times are untouched. LINEAR evaluation does not use
                    # handles; translate them without recalculating other keys.
                    left, right = point.handle_left.copy(), point.handle_right.copy()
                    point.co.y = value[component]
                    point.handle_left = (left.x, left.y + delta)
                    point.handle_right = (right.x, right.y + delta)
                changed.append((bone, frame, t))
        clone.update_tag()
        after = _snapshot(clone)
        expected_keys = {(rig.pose.bones[n].path_from_id('rotation_quaternion'), i)
                         for n, delay in DELAYS.items() if delay for i in range(4)}
        if set(before) != set(after):
            raise RuntimeError('Action channel inventory changed')
        actual_changed = []
        for channel in before:
            if len(before[channel]) != len(after[channel]):
                raise RuntimeError('Action sample count changed')
            for old, new in zip(before[channel], after[channel]):
                if old == new:
                    continue
                if channel not in expected_keys or weight(old['co'][0]) == 0.:
                    raise RuntimeError('A protected channel or boundary key changed: ' + repr(channel))
                if old['co'][0] != new['co'][0] or any(old[k] != new[k] for k in
                    ('interpolation', 'leftType', 'rightType', 'easing')):
                    raise RuntimeError('Source timing or interpolation metadata changed')
                actual_changed.append([channel[0], channel[1], old['co'][0]])
        if audit['_action_hash'](source) != SOURCE_SHA or _snapshot(source) != before:
            raise RuntimeError('Original R09 action changed')
        retained = {k: clone[k] for k in ('r09_performance_report', 'r09_performance_origin') if k in clone}
        for key in retained:
            del clone[key]
        clone['R11_retained_source_metadata'] = json.dumps(retained, ensure_ascii=False)
        record = {'sourceAction': SOURCE_ACTION, 'sourceActionSha256': SOURCE_SHA,
            'candidateAction': clone.name, 'candidateActionSha256': audit['_action_hash'](clone),
            'helperSha256': sha(__file__), 'ramps': RAMPS, 'delayFrames': DELAYS,
            'actualChangedScalarSamples': len(actual_changed), 'actualRetimedBoneSamples': len(changed),
            'allOtherKeyframesAndHandlesExact': True, 'maximumUpperArmTimeShiftFrames': 1.,
            'protectedBoundariesAndPlateau': [205, 241, 247, 253, 289],
            'originalActionUnchanged': True, 'leftInactive': rig.animation_data.action != clone,
            'actualPoseVerification': None, 'savedBlend': False, 'rendered': False,
            'appearanceAccepted': False,
            'limits': ['Only shoulder/elbow timing; constant local wrist articulation is unchanged.',
                'World-directed source arms include shoulder compensation; actual world pose checks remain required.',
                'Does not alter squat, bend, legs, pelvis, head, fingers or source key timing.',
                'Old cloth acceleration/waist expansion at 32/133/181 is outside this correction.',
                'Prior cloth/hair action fingerprints do not match this new action; do not activate against old caches.',
                'Candidate is an authored timing edit, not physical simulation or naturalness acceptance.']}
        clone[MARKER] = json.dumps(record, ensure_ascii=False)
        return record
    except Exception:
        if clone is not None:
            bpy.data.actions.remove(clone)
        raise


def verify_candidate(frames=PROBE_FRAMES):
    """Seventeen actual pose comparisons; no render or geometry modification."""
    rig = bpy.data.objects[RIG]
    scene = bpy.context.scene
    audit = api('foot_contact_audit_r03')
    source, candidate = (bpy.data.actions[n] for n in (SOURCE_ACTION, ACTION))
    record = json.loads(candidate[MARKER])
    if audit['_action_hash'](source) != SOURCE_SHA or audit['_action_hash'](candidate) != record['candidateActionSha256']:
        raise RuntimeError('Recorded source or candidate action changed')
    if any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA would contaminate the isolated action comparison')
    action, slot = rig.animation_data.action, rig.animation_data.action_slot
    frame = (scene.frame_current, scene.frame_subframe)
    basis = {b.name: b.matrix_basis.copy() for b in rig.pose.bones}
    master = rig.pose.bones['CTRL_R02_Master']
    switches = {k: master[k] for k in audit['IK_KEYS']}
    pose_position = rig.data.pose_position
    before = api('production_uv_r08')['_pose_contract']()
    selection = audit['select_sole_vertices'](objects=audit['DEFAULT_OBJECTS'], max_points=12)
    unchanged = ('hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
        'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
        'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes')
    if any(n not in rig.pose.bones for n in unchanged):
        raise RuntimeError('Expected unchanged skeleton channels missing')
    watched = unchanged + tuple(DELAYS) + ('leftHand', 'rightHand')
    sampled = {}
    try:
        rig.data.pose_position = 'POSE'
        for selected in (source, candidate):
            for name, value in basis.items():
                rig.pose.bones[name].matrix_basis = value
            for name in switches:
                master[name] = 0.
            audit['_assign_action'](rig, selected)
            rows = {}
            for value in frames:
                scene.frame_set(math.floor(value), subframe=value % 1.)
                bpy.context.view_layer.update()
                bones = {n: rig.matrix_world @ rig.pose.bones[n].matrix for n in watched}
                row = {'bones': bones, 'feet': audit['sample_sole_points'](selection),
                    'joints': {n: list(rig.matrix_world @ rig.pose.bones[n].tail) for n in watched}}
                row['lengths'] = {n: (rig.pose.bones[n].tail - rig.pose.bones[n].head).length for n in watched}
                row['elbowAngles'] = {side: math.degrees((rig.pose.bones[side+'UpperArm'].tail-rig.pose.bones[side+'UpperArm'].head).angle(
                    rig.pose.bones[side+'LowerArm'].tail-rig.pose.bones[side+'LowerArm'].head)) for side in ('left','right')}
                rows[str(value)] = row
            sampled[selected.name] = rows
    finally:
        audit['_assign_action'](rig, action)
        if slot is not None:
            rig.animation_data.action_slot = slot
        rig.data.pose_position = pose_position
        for name, value in basis.items():
            rig.pose.bones[name].matrix_basis = value
        for name, value in switches.items():
            master[name] = value
        scene.frame_set(frame[0], subframe=frame[1])
        bpy.context.view_layer.update()
    if before != api('production_uv_r08')['_pose_contract']():
        raise RuntimeError('Original action/pose/frame was not restored')
    results = []
    for value in frames:
        old, new = (sampled[n][str(value)] for n in (SOURCE_ACTION, ACTION))
        fixed_error = max(abs(old['bones'][n][i][j]-new['bones'][n][i][j])
                          for n in unchanged for i in range(4) for j in range(4))
        sole_error = max((Vector(a)-Vector(b)).length for name in selection
                         for a,b in zip(old['feet'][name],new['feet'][name]))
        length_error = max(abs(old['lengths'][n]-new['lengths'][n]) for n in watched)
        angles = {n: math.degrees(old['bones'][n].to_quaternion().rotation_difference(
            new['bones'][n].to_quaternion()).angle) for n in DELAYS}
        angles = {n: min(a, 360.-a) for n,a in angles.items()}
        elbow_change = {side: new['elbowAngles'][side]-old['elbowAngles'][side] for side in ('left','right')}
        protected = weight(value) == 0.
        maximum = max(angles.values())
        if fixed_error > 2e-6 or sole_error > 2e-6 or length_error > 2e-6:
            raise RuntimeError('Candidate changed planted feet, fixed body or bone lengths')
        if maximum > 5. or max(abs(v) for v in elbow_change.values()) > 3.:
            raise RuntimeError('Timing change exceeds conservative actual world-angle bounds')
        protected_error = max(abs(old['bones'][n][i][j]-new['bones'][n][i][j])
                              for n in DELAYS for i in range(4) for j in range(4)) if protected else None
        if protected and protected_error > 2e-6:
            raise RuntimeError('Protected boundary/high-pose arm orientation changed')
        results.append({'frame': value, 'unchangedBodyMatrixError': fixed_error,
            'protectedArmMatrixError': protected_error,
            'maximumSolePointDifferenceMeters': sole_error, 'maximumBoneLengthDifferenceMeters': length_error,
            'armWorldRotationDifferenceDegrees': angles, 'elbowFlexionDifferenceDegrees': elbow_change,
            'candidateElbowFlexionDegrees': new['elbowAngles'],
            'oldJointTailsWorld': old['joints'], 'newJointTailsWorld': new['joints']})
    result = {'sampleCount': len(results), 'samples': results, 'finitePoseBoundsPassed': True,
        'scope': 'Bone/finite shoe comparisons only; clothing, hair, skin contacts and visual acceptance not checked.'}
    record['actualPoseVerification'] = result
    candidate[MARKER] = json.dumps(record, ensure_ascii=False)
    return result


def prepare(source_relative, output_relative, verify=False):
    source = (project_path() / source_relative).resolve()
    output = (project_path() / output_relative).resolve()
    for p in (source, output):
        p.relative_to((project_path() / 'work/reconstruction-r02').resolve())
        if p.suffix != '.blend':
            raise ValueError('Independent Blend paths required')
    report_path = output.with_suffix('.arm-timing.json')
    if output.exists() or report_path.exists() or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh input and new independent output required')
    source_sha = sha(source)
    record = create_action()
    if verify:
        record['actualPoseVerification'] = verify_candidate()
    if sha(source) != source_sha:
        raise RuntimeError('Input file changed')
    record.update(source=str(source.relative_to(project_path())), sourceSha256=source_sha)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
        raise RuntimeError('Independent candidate save failed')
    record.update(savedBlend=True, output=str(output.relative_to(project_path())), outputSha256=sha(output))
    with report_path.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print('R11_ARM_TIMING_CANDIDATE '+json.dumps({'output': record['output'], 'verified': bool(record['actualPoseVerification'])}), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.output, args.verify)
