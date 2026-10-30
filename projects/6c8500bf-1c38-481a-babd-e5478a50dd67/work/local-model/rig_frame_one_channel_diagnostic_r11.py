"""Light source-channel / active-action frame-one diagnostic, without geometry.

Reads original authored channels, performs an ordinary view update and one
explicit frame_set(1), then compares actual changed channels with FCurve(1).
No matrix writeback, pose edits, render, evaluated mesh or Blender save.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CHANNELS = ('location', 'rotation_mode', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle', 'scale')
SIDE_REPORT = 'previews/reconstruction-r02/chin-throat-contour-side-completion-r11/side-0.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [plain(v) for v in value]
    except TypeError:
        return {'type': type(value).__name__, 'name': getattr(value, 'name', None)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def leaf_diff(before, after, path='$'):
    rows = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            child = path+'['+json.dumps(str(key), ensure_ascii=False)+']'
            if key not in before or key not in after:
                rows.append({'path': child, 'kind': 'presence', 'before': before.get(key), 'after': after.get(key)})
            else:
                rows.extend(leaf_diff(before[key], after[key], child))
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            rows.append({'path': path+'.length', 'kind': 'length', 'before': len(before), 'after': len(after)})
        for i in range(min(len(before), len(after))):
            rows.extend(leaf_diff(before[i], after[i], path+'['+str(i)+']'))
    elif before != after or type(before) is not type(after):
        row = {'path': path, 'kind': 'value', 'before': before, 'after': after}
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            row['numericDelta'] = after-before
        rows.append(row)
    return rows


def object_channels(block):
    return {n: plain(getattr(block, n)) for n in CHANNELS}


def snapshot(bpy):
    rigs = {}
    for rig in bpy.context.scene.objects:
        if rig.type != 'ARMATURE':
            continue
        rigs[rig.name] = {'posePosition': rig.data.pose_position, 'channels': object_channels(rig),
            'bones': {bone.name: {'channels': object_channels(bone), 'basis': [list(r) for r in bone.matrix_basis],
                'matrix': [list(r) for r in bone.matrix]} for bone in rig.pose.bones},
            'world': [list(r) for r in rig.matrix_world]}
    return {'frame': bpy.context.scene.frame_current, 'subframe': bpy.context.scene.frame_subframe, 'rigs': rigs}


def handlers(bpy):
    return {name: [{'module': getattr(f, '__module__', None), 'name': getattr(f, '__qualname__', repr(f))}
        for f in getattr(bpy.app.handlers, name)] for name in
        ('frame_change_pre', 'frame_change_post', 'depsgraph_update_pre', 'depsgraph_update_post', 'render_pre', 'render_post')}


def action_rows(rig):
    animation = rig.animation_data
    if animation is None:
        return {'action': None, 'curves': [], 'nla': [], 'drivers': []}
    action = animation.action
    slot_handle = getattr(animation, 'action_slot_handle', None)
    out = {'action': action.name if action else None, 'actionPointer': action.as_pointer() if action else None,
        'activeSlotHandle': slot_handle, 'activeSlotIdentifier': getattr(getattr(animation, 'action_slot', None), 'identifier', None),
        'actionBlendType': getattr(animation, 'action_blend_type', None),
        'actionInfluence': getattr(animation, 'action_influence', None),
        'useNla': getattr(animation, 'use_nla', None), 'curves': [], 'layers': [],
        'nla': [{'name': t.name, 'mute': t.mute, 'solo': t.is_solo,
            'strips': [{'name': s.name, 'action': s.action.name if s.action else None,
                'mute': getattr(s, 'mute', None), 'influence': s.influence,
                'frameStart': s.frame_start, 'frameEnd': s.frame_end,
                'actionFrameStart': s.action_frame_start, 'actionFrameEnd': s.action_frame_end,
                'blendType': s.blend_type} for s in t.strips]} for t in animation.nla_tracks],
        'drivers': [{'path': c.data_path, 'index': c.array_index, 'mute': c.mute, 'expression': c.driver.expression,
            'type': c.driver.type} for c in animation.drivers]}
    if action is None:
        return out
    collected = []
    if hasattr(action, 'layers'):
        for layer in action.layers:
            out['layers'].append({n: plain(getattr(layer, n)) for n in ('name', 'influence', 'mix_mode') if hasattr(layer, n)})
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', []):
                    handle = getattr(bag, 'slot_handle', None)
                    for curve in bag.fcurves:
                        collected.append((curve, handle, layer.name, getattr(strip, 'type', None),
                            None if handle is None or slot_handle is None else handle == slot_handle))
    if not collected and hasattr(action, 'fcurves'):
        collected = [(c, None, None, None, True) for c in action.fcurves]
    for curve, handle, layer, strip_type, active in collected:
        path, index = curve.data_path, curve.array_index
        try:
            resolved = rig.path_resolve(path)
            current = float(resolved[index]) if hasattr(resolved, '__len__') else plain(resolved)
            resolution_error = None
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            current, resolution_error = None, str(exc)
        keys = [{'co': list(k.co), 'left': list(k.handle_left), 'right': list(k.handle_right),
            'interpolation': k.interpolation, 'easing': k.easing} for k in curve.keyframe_points]
        out['curves'].append({'path': path, 'index': index, 'slotHandle': handle, 'activeSlot': active,
            'layer': layer, 'stripType': strip_type, 'mute': curve.mute,
            'groupMute': getattr(curve.group, 'mute', None) if curve.group else None,
            'evaluateAtFrameOne': float(curve.evaluate(1.0)), 'currentOriginalProperty': current,
            'resolutionError': resolution_error, 'keysSha256': digest(keys),
            'modifierTypes': [m.type for m in curve.modifiers]})
    return out


def observe(source_relative, source_sha, output_relative):
    import bpy
    project = project_path().resolve()
    source, output = [(project/r).resolve() for r in (source_relative, output_relative)]
    for path in (source, output):
        path.relative_to(project/'work/reconstruction-r02')
    if source.suffix != '.blend' or output.suffix != '.json' or output.exists():
        raise RuntimeError('Independent saved source and unused report path required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source or sha(source) != source_sha.lower():
        raise RuntimeError('Fresh explicit background source and matching SHA required')
    if (bpy.context.scene.frame_current, bpy.context.scene.frame_subframe) != (1, 0.0):
        raise RuntimeError('Inspect the actual saved frame-one source')
    prior_path = project/SIDE_REPORT
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    report = {'source': source_relative, 'sourceSha256': source_sha.lower(), 'helperSha256': sha(__file__),
        'sourceSaved': False, 'rendersPerformed': 0, 'evaluatedGeometryCaptured': False,
        'manualChannelWritesPerformed': False, 'manualMatrixWritesPerformed': False,
        'sideReport': SIDE_REPORT, 'sideReportSha256': sha(prior_path),
        'sideReportSourceSha256': prior['sourceSha256'], 'sameSourceAsSideReport': prior['sourceSha256'] == source_sha.lower(),
        'scope': 'Original RNA channels and Action FCurve(1), one view update and one explicit frame_set(1). No geometry capture or render. FCurve equality is reported separately from NLA/driver/handler causality.',
        'states': {}, 'actions': {}, 'errors': [], 'handlersBefore': handlers(bpy)}
    def persist(step):
        report['lastCompletedStep'] = step
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print('R11_RIG_FRAME_ONE_DIAGNOSTIC '+step, flush=True)
    def capture(label):
        report['states'][label] = snapshot(bpy)
        report['actions'][label] = {r.name: action_rows(r) for r in bpy.context.scene.objects if r.type == 'ARMATURE'}
        persist(label)
    try:
        capture('fresh_load')
        bpy.context.view_layer.update()
        capture('after_view_update')
        bpy.context.scene.frame_set(1, subframe=0.0)
        capture('after_frame_set_one')
        report['freshToViewUpdateDiff'] = leaf_diff(report['states']['fresh_load'], report['states']['after_view_update'])
        report['viewUpdateToFrameSetDiff'] = leaf_diff(report['states']['after_view_update'], report['states']['after_frame_set_one'])
        report['freshToFrameSetDiff'] = leaf_diff(report['states']['fresh_load'], report['states']['after_frame_set_one'])
        expression = re.compile(r'^\$\["rigs"\]\["([^"\\]+)"\]\["bones"\]\["([^"\\]+)"\]\["channels"\]\["([^"\\]+)"\]\[(\d+)\]$')
        comparisons = []
        for row in prior['authorDiff']:
            match = expression.match(row['path'])
            if match is None:
                comparisons.append({'path': row['path'], 'supportedPath': False})
                continue
            rig_name, bone_name, channel, text_index = match.groups()
            index = int(text_index)
            rig = bpy.data.objects[rig_name]
            data_path = rig.pose.bones[bone_name].path_from_id(channel)
            values = {label: state['rigs'][rig_name]['bones'][bone_name]['channels'][channel][index]
                for label, state in report['states'].items()}
            curves = [c for c in report['actions']['after_frame_set_one'][rig_name]['curves']
                if c['path'] == data_path and c['index'] == index and c['activeSlot'] is not False]
            comparison = {'path': row['path'], 'supportedPath': True, 'rig': rig_name, 'bone': bone_name,
                'channel': channel, 'index': index, 'sideBefore': row['before'], 'sideAfter': row['after'],
                'sourceValues': values, 'freshEqualsSideBefore': values['fresh_load'] == row['before'],
                'frameSetEqualsSideAfter': values['after_frame_set_one'] == row['after'], 'matchingCurves': curves}
            if len(curves) == 1:
                expected = curves[0]['evaluateAtFrameOne']
                comparison['curveEquality'] = {label: value == expected for label, value in values.items()}
                comparison['curveEqualsSideBefore'] = expected == row['before']
                comparison['curveEqualsSideAfter'] = expected == row['after']
                comparison['curveMinusFresh'] = expected-values['fresh_load']
            comparisons.append(comparison)
        report['sideAuthorLeafComparisons'] = comparisons
        supported = [r for r in comparisons if r['supportedPath']]
        report['summary'] = {'sideAuthorLeaves': len(comparisons), 'supportedLeaves': len(supported),
            'freshEqualsSideBefore': sum(r['freshEqualsSideBefore'] for r in supported),
            'frameSetEqualsSideAfter': sum(r['frameSetEqualsSideAfter'] for r in supported),
            'singleMatchingCurve': sum(len(r['matchingCurves']) == 1 for r in supported),
            'curveEqualsSideAfter': sum(r.get('curveEqualsSideAfter', False) for r in supported),
            'curveEqualsFresh': sum(r.get('curveEquality', {}).get('fresh_load', False) for r in supported),
            'curveEqualsFrameSet': sum(r.get('curveEquality', {}).get('after_frame_set_one', False) for r in supported)}
        report['completed'] = True
    except BaseException as exc:
        report['completed'] = False
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        raise
    finally:
        report['handlersAfter'] = handlers(bpy)
        report['sourceFileExact'] = sha(source) == source_sha.lower()
        report['sourceFrameStillOne'] = (bpy.context.scene.frame_current, bpy.context.scene.frame_subframe) == (1, 0.0)
        report['inMemoryChannelsReevaluatedButNotSaved'] = True
        persist('final_report_persisted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.source_sha, args.output)
