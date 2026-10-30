"""Offline assembly of one continuous neutral eye-contour surface proposal."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
W = P/'work/reconstruction-r02'


def api(name):
    return runpy.run_path(str(HERE/(name+'.py')))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def head_opening(slices, x, center, depths):
    lines = slices.lines(x)
    lo, hi = center[2]-.004, center[2]+.004
    lines = lines[(lines[:, 1] >= lo) & (lines[:, 0] <= hi)]
    roots = np.divide(center[1]-lines[:, 3], lines[:, 2], out=np.full(len(lines), np.nan), where=abs(lines[:, 2]) > 1e-15)
    valid = (roots >= lines[:, 0]) & (roots <= lines[:, 1]) & (roots >= lo) & (roots <= hi)
    breaks = np.unique(np.r_[lo, hi, np.clip(lines[:, :2].ravel(), lo, hi), roots[valid]])
    mid = (breaks[:-1]+breaks[1:])*.5
    values = depths(lines, mid)
    selected = np.flatnonzero(np.isfinite(values) & (values >= center[1]))
    if not len(selected):
        return None
    groups = np.split(selected, np.flatnonzero(np.diff(selected) != 1)+1)
    ranges = [(breaks[g[0]], breaks[g[-1]+1]) for g in groups]
    return min(ranges, key=lambda pair: abs((pair[0]+pair[1])*.5-center[2]))


def tail(delta, derivative, t, length):
    return delta*(1.-10.*t**3+15.*t**4-6.*t**5)+derivative*length*(t-6.*t**3+8.*t**4-3.*t**5)


def run():
    output = W/'adult-outer-eye-contour-surface-preview-r11-v2'
    if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Preserve previous proposal evidence')
    source = P/'previews/reconstruction-r02/outer-eye-neutral-direction-preview-r11/neutral-geometry-preview.npz'
    source_report = json.loads(source.with_suffix('.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_report['arraysSha256']:
        raise RuntimeError('Actual neutral input changed')
    data = dict(np.load(source))
    target_report = json.loads((W/'adult-outer-eye-continuous-target-r11.json').read_text())
    raw = dict(np.load(W/'adult-outer-eye-visible-contour-r11.npz'))
    target = api('adult_outer_eye_continuous_target_r11')
    slices_api = api('adult_outer_eye_visible_contour_r11')
    transport = api('adult_outer_eye_contour_transport_preview_r11_v2')
    ids = data['originalVertexIds'];world = data['beforeFullWorld'][ids];triangles = data['triangles']
    slices = slices_api['Slices'](world, triangles)
    eyes = {s: {'world': data['eye_'+s+'_world'], 'triangles': data['eye_'+s+'_triangles']} for s in ('L', 'R')}
    profiles, anchors = {}, {}
    profile_report = {}
    for side in ('L', 'R'):
        details = source_report['plan']['sides'][side]
        center = np.asarray(details['centerWorld']);sign = details['outwardSign']
        report = target_report['sides'][side]
        start, end = report['startOutwardMeters'], report['endOutwardMeters']
        extension = report['extensionLimitMeters']-end
        extension_end = end+extension*.999
        tail_length = .00015
        outward = np.unique(np.r_[np.arange(start-.001, extension_end+tail_length, .000025), start, end, extension_end, extension_end+tail_length])
        x = center[0]+sign*outward
        rows = raw[side+'_before_contour']
        old_u = sign*(rows[:, 0]-center[0])
        order = np.argsort(old_u)
        lower = np.interp(outward, old_u[order], rows[order, 1])
        upper = np.interp(outward, old_u[order], rows[order, 2])
        globe_end = sign*(details['actualGlobeOuterX']-center[0])
        transition = transport['smooth5']((outward-(globe_end-.0004))/.0003)
        head_bounds = np.zeros((len(x), 2))
        valid = np.zeros(len(x), bool)
        for i in np.flatnonzero(transition > 0.):
            opening = head_opening(slices, float(x[i]), center, slices_api['depths'])
            if opening is None:
                opening = (report['endCenterWorldZ']-.000075, report['endCenterWorldZ']+.000075)
            else:
                valid[i] = True
            head_bounds[i] = opening
        lower = lower*(1.-transition)+head_bounds[:, 0]*transition
        upper = upper*(1.-transition)+head_bounds[:, 1]*transition
        source_gap = np.maximum(upper-lower, .00015)
        source_mid = (upper+lower)*.5
        lower, upper = source_mid-source_gap*.5, source_mid+source_gap*.5
        inside = outward <= extension_end
        goal_lower, goal_upper = lower.copy(), upper.copy()
        goal_lower[inside] = target['evaluate_target'](outward[inside], report, edge='lower')
        goal_upper[inside] = target['evaluate_target'](outward[inside], report, edge='upper')
        end_index = int(np.argmin(abs(outward-extension_end)))
        for edge, original, goal in (('lower', lower, goal_lower), ('upper', upper, goal_upper)):
            delta = goal[end_index]-original[end_index]
            derivative = report['edges'][edge]['endJet']['slope']
            t = np.clip((outward[~inside]-extension_end)/tail_length, 0., 1.)
            goal[~inside] = original[~inside]+tail(delta, derivative, t, tail_length)
        if np.any(goal_upper <= goal_lower):
            raise RuntimeError('Short non-visible bridge continuation crossed its target margins')
        xorder = np.argsort(x)
        profiles[side] = {'x': x[xorder], 'lower': lower[xorder], 'upper': upper[xorder],
            'targetLower': goal_lower[xorder], 'targetUpper': goal_upper[xorder],
            'startOutwardMeters': start, 'endOutwardMeters': extension_end+tail_length}
        points, rows = [], []
        for edge in ('lower', 'upper'):
            margin = data['edge_'+side+'_'+edge+'_headWorld']
            mask = data['edge_'+side+'_'+edge+'_coveredMask'] & (sign*(margin[:, 0]-center[0]) >= start)
            points.append(margin[mask]);rows.append(data['edge_'+side+'_'+edge+'_headTriangleIds'][mask])
        anchors[side] = {'world': np.concatenate(points), 'triangleRows': np.concatenate(rows)}
        profile_report[side] = {'bridgeContinuation': 'Actual head front-halfspace opening replaces the globe-clipped silhouette near its outer limit; after the real bridge, ordered virtual margins only taper the tiny deformation to zero.',
            'centralProtectedOutwardMeters': start, 'endOutwardMeters': extension_end+tail_length,
            'headOpeningSamples': int(valid.sum()), 'targetMinimumSeparationMeters': float((goal_upper-goal_lower).min())}
    delta, arrays, report = transport['transport'](world, triangles, ids, eyes, source_report['plan'], profiles, anchors)
    report.update(actualInput=str(source.relative_to(P)), actualInputSha256=source_report['arraysSha256'], profiles=profile_report,
                  targetHelperSha256=target_report['helperSha256'], surfaceValidation='Source tessellation only; actual post-modifier tessellation and image are pending.')
    payload = dict(originalVertexIds=ids, beforeWorld=world, deltaWorld=delta, triangles=triangles, **arrays)
    for side in profiles:
        for name, values in profiles[side].items():
            payload['profile_'+side+'_'+name] = np.asarray(values)

    def persist():
        np.savez_compressed(output.with_suffix('.npz'), **payload)
        report['arraysSha256'] = hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
        output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=serial), encoding='utf8')

    persist()
    try:
        contact = api('adult_blink_midpath_contact_r11')
        after = world+delta
        changed = np.linalg.norm(delta, axis=1) > 1e-10
        crosses = contact['changed_intersections'](after, triangles, changed)
        payload['crossingPairs'] = np.asarray(sorted(crosses), np.int32).reshape(-1, 2)
        report['sourceTessellationCrossings'] = len(crosses)
        report['proposedGlobeAndLayers'], report['pendingInnerGlobalVertexIds'] = api('adult_neutral_eye_outerends_radial_install_r11_v2')['actual_layers'](
            {'world': world, 'triangles': triangles, 'originalVertexIds': ids},
            {'world': after, 'triangles': triangles, 'originalVertexIds': ids}, eyes, contact)
        persist()
        print(json.dumps({'maximumMeters': float(np.linalg.norm(delta, axis=1).max()), 'crossings': len(crosses), 'output': str(output)}, ensure_ascii=False), flush=True)
    except Exception as error:
        report['failure'] = str(error)
        persist()
        raise


if __name__ == '__main__':
    run()
