"""Persistent read-only posture observer with exact per-path restore diagnostics.

Uses the frozen v1 mesh sampler without editing it. No Blend is saved. Every
completed state and the final restore diagnosis are persisted before any
restore failure is raised; no tolerance or physical-equivalence bypass exists.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import traceback
import numpy as np

BASE_SHA = '885e5abb846c690b6efbbc78751eb7bb8666056ecf01e71723d760d576ea21e4'
BASE_PATH = Path(__file__).with_name('head_neck_posture_observe_r11.py')
base = runpy.run_path(str(BASE_PATH))
sha, matrix, evaluated, bone_evidence = (base[k] for k in ('sha', 'matrix', 'evaluated', 'bone_evidence'))
HEAD, RIG, BODY, GARMENTS = (base[k] for k in ('HEAD', 'RIG', 'BODY', 'GARMENTS'))
project_path = base['project_path']


def leaf_diff(before, after, path='$'):
    """Exact JSON-leaf differences, with no threshold and no truncated rows."""
    result = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = path+'['+json.dumps(str(key), ensure_ascii=False)+']'
            if key not in before:
                result.append({'path': child, 'kind': 'added', 'after': after[key]})
            elif key not in after:
                result.append({'path': child, 'kind': 'removed', 'before': before[key]})
            else:
                result.extend(leaf_diff(before[key], after[key], child))
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            result.append({'path': path+'.length', 'kind': 'length', 'before': len(before), 'after': len(after)})
        for i in range(min(len(before), len(after))):
            result.extend(leaf_diff(before[i], after[i], path+'['+str(i)+']'))
        for i in range(min(len(before), len(after)), max(len(before), len(after))):
            result.append({'path': path+'['+str(i)+']', 'kind': 'added' if i >= len(before) else 'removed',
                'before': before[i] if i < len(before) else None, 'after': after[i] if i < len(after) else None})
    elif before != after or type(before) is not type(after):
        row = {'path': path, 'kind': 'value', 'before': before, 'after': after}
        if isinstance(before, (float, int)) and isinstance(after, (float, int)):
            row['numericDelta'] = after-before
        result.append(row)
    return result


def snapshot(uv, head):
    import bpy
    rows, positions = [], {}
    for rig in bpy.context.scene.objects:
        if rig.type != 'ARMATURE':
            continue
        positions[rig.name] = rig.data.pose_position
        rows.append({'name': rig.name, 'world': uv['_plain'](rig.matrix_world), 'animation': uv['_animation'](rig),
            'bones': [{'name': bone.name, 'basis': uv['_plain'](bone.matrix_basis), 'matrix': uv['_plain'](bone.matrix),
                'custom': uv['_custom'](bone)} for bone in rig.pose.bones]})
    scene = bpy.context.scene
    contract = uv['_pose_contract']()
    return {'frame': scene.frame_current, 'subframe': scene.frame_subframe,
        'armaturePosePositions': positions, 'poseContract': contract, 'poseContractRows': rows,
        'rowsHashMatchesContract': uv['_hash'](rows) == contract['sha256'],
        'headKeyValues': {k.name: float(k.value) for k in head.data.shape_keys.key_blocks},
        'headMatrixWorld': matrix(head.matrix_world)}


def observe(source_relative, output_relative):
    import bpy
    if sha(BASE_PATH) != BASE_SHA:
        raise RuntimeError('Frozen v1 sampler changed')
    project = project_path()
    source, output = (project/source_relative).resolve(), (project/output_relative).resolve()
    for p in (source, output):
        p.relative_to((project/'work/reconstruction-r02').resolve())
    npz = output.with_suffix('.npz')
    if source.suffix != '.blend' or output.suffix != '.json' or output.exists() or npz.exists():
        raise RuntimeError('Explicit new diagnostic JSON/NPZ output required; preserve earlier observations')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh background process on explicit saved source required')
    head, rig, scene = bpy.data.objects[HEAD], bpy.data.objects[RIG], bpy.context.scene
    uv = runpy.run_path(str(Path(__file__).with_name('production_uv_r08.py')))
    source_sha = sha(source)
    arrays = {}
    report = {'source': source_relative, 'sourceSha256': source_sha, 'helperSha256': sha(__file__),
        'samplerSha256': BASE_SHA, 'savedBlend': False, 'geometryModified': False,
        'technicalAcceptance': False, 'sourcePoseAndExpressionRestored': False,
        'scope': 'Exact read-only frame-one POSE/REST geometry observation. Failed restoration diagnostics preserve samples and every changed contract leaf; no float tolerance or equivalence exemption.',
        'states': {}, 'steps': [], 'errors': [], 'restorationErrors': []}

    def persist(step):
        report['steps'].append(step)
        report['lastCompletedStep'] = step
        # This observer owns these newly created outputs. Replace only its own
        # checkpoint files so an exception never discards completed samples.
        payload_tmp = npz.with_suffix('.checkpoint.npz')
        np.savez_compressed(payload_tmp, **arrays)
        payload_tmp.replace(npz)
        report.update(arrays=npz.relative_to(project).as_posix(), arraysSha256=sha(npz),
            payloadKeys=sorted(arrays), payloadArrayCount=len(arrays))
        json_tmp = output.with_suffix('.checkpoint.json')
        json_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        json_tmp.replace(output)
        print('R11_HEAD_NECK_CHECKPOINT '+json.dumps({'step': step, 'arrays': len(arrays), 'output': output_relative}), flush=True)

    preupdate = snapshot(uv, head)
    report['preUpdateSnapshot'] = preupdate
    bpy.context.view_layer.update()
    baseline = snapshot(uv, head)
    report['baselineSnapshot'] = baseline
    report['preUpdateToBaselineExactDiff'] = leaf_diff(preupdate, baseline)
    report['sourceFrame'] = [baseline['frame'], baseline['subframe']]
    report['reviewFrame'] = 1
    old_pose = rig.data.pose_position
    persist('baseline_stabilized')
    try:
        report['reviewFrameSetPerformed'] = (scene.frame_current != 1 or scene.frame_subframe != 0.)
        if report['reviewFrameSetPerformed']:
            scene.frame_set(1)
        bpy.context.view_layer.update()
        report['frameOneKeyValues'] = {k.name: float(k.value) for k in head.data.shape_keys.key_blocks}
        report['frameOneSnapshot'] = snapshot(uv, head)
        action = getattr(rig.animation_data, 'action', None) if rig.animation_data else None
        report['rigAction'] = action.name if action else None
        for pose in ('POSE', 'REST'):
            rig.data.pose_position = pose
            rig.data.update_tag()
            bpy.context.view_layer.update()
            row = {'rigMatrixWorld': matrix(rig.matrix_world), 'posePosition': pose,
                'bones': bone_evidence(rig), 'objects': {}}
            report['states'][pose] = row
            for name, role in [(HEAD, 'head'), (BODY, 'body')]+[(n, 'garment') for n in GARMENTS]:
                report['activeSamplingObject'] = name
                prefix = pose.lower()+'_'+role+('_'+str(GARMENTS.index(name)) if role == 'garment' else '')
                row['objects'][name] = evaluated(bpy.data.objects[name], prefix, arrays, role)
                report['lastCompletedObject'] = name
            report.pop('activeSamplingObject', None)
            row['samplingCompleted'] = True
            persist(pose.lower()+'_sampling_completed')
    except BaseException as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        try:
            rig.data.pose_position = old_pose
            rig.data.update_tag()
            report['restoreFrameSetPerformed'] = (scene.frame_current != baseline['frame'] or scene.frame_subframe != baseline['subframe'])
            if report['restoreFrameSetPerformed']:
                scene.frame_set(baseline['frame'], subframe=baseline['subframe'])
            bpy.context.view_layer.update()
        except BaseException as error:
            report['restorationErrors'].append({'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        try:
            after = snapshot(uv, head)
            report['afterRestoreSnapshot'] = after
            report['baselineToAfterRestoreExactDiff'] = leaf_diff(baseline, after)
            report['preUpdateToAfterRestoreExactDiff'] = leaf_diff(preupdate, after)
            report['sourceSha256After'] = sha(source)
            report['sourceSha256Exact'] = report['sourceSha256After'] == source_sha
            report['sourcePoseAndExpressionRestored'] = (not report['baselineToAfterRestoreExactDiff'] and report['sourceSha256Exact'])
            report['technicalAcceptance'] = (report['sourcePoseAndExpressionRestored'] and not report['errors'] and not report['restorationErrors'] and all(report['states'].get(p, {}).get('samplingCompleted', False) for p in ('POSE', 'REST')))
        except BaseException as error:
            report['restorationErrors'].append({'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        persist('final_restore_diagnostics_persisted')
    if not report['technicalAcceptance']:
        raise RuntimeError('Exact read-only observation did not pass; completed samples and per-path differences retained in '+output_relative)
    print('R11_HEAD_NECK_POSTURE_OBSERVED '+json.dumps({'sourceSha256': source_sha, 'output': output_relative, 'arraysSha256': report['arraysSha256'], 'exactRestoration': True}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.output)
