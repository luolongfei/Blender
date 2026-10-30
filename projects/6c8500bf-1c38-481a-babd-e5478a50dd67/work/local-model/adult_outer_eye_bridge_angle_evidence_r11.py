"""One independent, strictly outside-globe bridge-angle continuation candidate.

This is an offline comparison for the failed continuous-contour v2 proposal.
It does not edit Blender data, does not approve an appearance, and does not
replace the parent agent's transport. The existing candidate inside the exact
globe X extent is retained bit-for-bit. One fitted periodic angular map replaces
the narrow gap-dependent map in the adjoining bridge core; original rho and X
are retained. Actual discrete triangles, rather than the analytical map alone,
decide whether this candidate is usable.
"""
from pathlib import Path
import hashlib
import json
import runpy

import numpy as np


HERE = Path(__file__).resolve().parent
P = HERE.parent.parent
W = P / 'work/reconstruction-r02'


def smooth5(t):
    t = np.clip(t, 0., 1.)
    return t ** 3 * (10. - 15. * t + 6. * t ** 2)


def repair(world, candidate_delta, metadata):
    """Return candidate delta, independent masks, and fitted constants.

    metadata is the current radial plan with actual globe center/extreme X.
    The explicit 75 micrometre join is wholly outside the globe, and completes
    before the measured failing faces begin at +117 micrometres. The 1.2 mm
    outgoing fade removes the former 150 micrometre profile termination.
    No amplitude search or automatic retries are performed.
    """
    world = np.asarray(world, float)
    candidate_delta = np.asarray(candidate_delta, float)
    if world.ndim != 2 or world.shape[1] != 3 or candidate_delta.shape != world.shape:
        raise ValueError('Matching N by 3 actual world and candidate delta required')
    if not np.isfinite(world).all() or not np.isfinite(candidate_delta).all():
        raise ValueError('Non-finite geometry')
    before = world + candidate_delta
    result = before.copy()
    payload = {}
    report = {'previewOnly': True, 'productionReady': False,
              'appearanceAccepted': False, 'allPhysicalContactPassed': False,
              'method': 'Strictly outside-globe common periodic angular continuation; original X and rho retained.',
              'joinWidthMeters': .000075, 'outgoingFadeStartMeters': .0002,
              'outgoingFadeLengthMeters': .0012,
              'radialCoreMeters': .0028, 'radialEndMeters': .005,
              'sides': {}}
    for side in ('L', 'R'):
        details = metadata['sides'][side]
        center = np.asarray(details['centerWorld'], float)
        sign = float(details['outwardSign'])
        outside = sign * (world[:, 0] - details['actualGlobeOuterX'])
        rho = np.linalg.norm(world[:, 1:] - center[1:], axis=1)
        theta = np.arctan2(world[:, 2] - center[2], center[1] - world[:, 1])
        old_theta = np.arctan2(before[:, 2] - center[2], center[1] - before[:, 1])
        delta_theta = np.arctan2(np.sin(old_theta-theta), np.cos(old_theta-theta))
        fit = ((outside >= -.0002) & (outside <= 0.) &
               (rho >= .001) & (rho <= .0028) & (abs(theta) < 1.3))
        if fit.sum() < 12:
            raise ValueError('Insufficient actual boundary samples')
        design = np.column_stack([np.sin(theta[fit]), np.cos(theta[fit])])
        coefficients = np.linalg.lstsq(design, delta_theta[fit], rcond=None)[0]
        amplitude = float(np.linalg.norm(coefficients))
        if amplitude >= .9:
            raise ValueError('Observed fitted periodic map has insufficient angular ordering margin')
        periodic = coefficients[0] * np.sin(theta) + coefficients[1] * np.cos(theta)
        join = smooth5(outside / report['joinWidthMeters'])
        radial = 1. - smooth5((rho-report['radialCoreMeters']) /
                             (report['radialEndMeters']-report['radialCoreMeters']))
        outgoing = 1. - smooth5((outside-report['outgoingFadeStartMeters']) /
                               report['outgoingFadeLengthMeters'])
        blend = join * radial
        active = (outside > 0.) & (outside < .0014) & (rho > 1e-7) & (blend > 0.)
        mapped = theta[active] + ((1.-blend[active])*delta_theta[active] +
                                  blend[active]*outgoing[active]*periodic[active])
        result[active, 1] = center[1] - rho[active] * np.cos(mapped)
        result[active, 2] = center[2] + rho[active] * np.sin(mapped)
        payload['repairMask_'+side] = active
        payload['joinWeight_'+side] = join
        payload['radialWeight_'+side] = radial
        payload['deltaThetaBefore_'+side] = delta_theta
        report['sides'][side] = {
            'fitCount': int(fit.sum()), 'sinCosCoefficients': coefficients.tolist(),
            'fitRmsRadians': float(np.sqrt(np.mean((design@coefficients-delta_theta[fit])**2))),
            'periodicAngularDerivativeLowerBound': 1.-amplitude,
            'modifiedPoints': int(active.sum()),
            'maxExtraDisplacementMeters': float(np.linalg.norm(result[active]-before[active], axis=1).max()),
            'sourceInsideGlobeXPreservedExactly': bool(np.array_equal(result[outside <= 0.], before[outside <= 0.])),
            'note': 'Analytical derivative applies only to the full common angular map. Joining and radial transitions require discrete checks.'}
    return result-world, payload, report


def pair_hits(points, triangles, pairs):
    """Actual interior edge-plane hits using the same strict crossing convention."""
    answer = []
    for rows in pairs:
        hits = []
        for reverse in (False, True):
            a = points[triangles[rows[int(reverse)]]]
            b = points[triangles[rows[int(not reverse)]]]
            e1, e2 = b[1]-b[0], b[2]-b[0]
            for edge in range(3):
                origin, direction = a[edge], a[(edge+1)%3]-a[edge]
                cross = np.cross(direction, e2)
                det = np.dot(e1, cross)
                if abs(det) <= 1e-15:
                    continue
                offset = origin-b[0]
                u = np.dot(offset, cross)/det
                q = np.cross(offset, e1)
                v, t = np.dot(direction, q)/det, np.dot(e2, q)/det
                if u>1e-7 and v>1e-7 and u+v<1.-1e-7 and t>1e-6 and t<1.-1e-6:
                    hits.append(origin+t*direction)
        hit = np.asarray(hits).reshape(-1, 3)
        answer.append({'triangleRows': list(map(int, rows)), 'intersectionHits': hit.tolist(),
                       'intersectionSpanMeters': float(np.linalg.norm(hit[:, None]-hit[None], axis=2).max()) if len(hit) else 0.})
    return answer


def run():
    output = W / 'adult-outer-eye-bridge-angle-evidence-r11'
    if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Preserve this independent single candidate')
    source = W / 'adult-outer-eye-contour-surface-preview-r11-v2.npz'
    source_report = json.loads(source.with_suffix('.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_report['arraysSha256']:
        raise RuntimeError('Parent v2 proposal hash mismatch')
    data = dict(np.load(source))
    current_path = P / source_report['actualInput']
    if not current_path.exists():
        current_path = P / source_report['actualInput'].replace('\\', '/')
    if hashlib.sha256(current_path.read_bytes()).hexdigest() != source_report['actualInputSha256']:
        raise RuntimeError('Actual neutral observation hash mismatch')
    actual_report = json.loads(current_path.with_suffix('.json').read_text())
    actual = dict(np.load(current_path))
    world, tri, ids = data['beforeWorld'], data['triangles'], data['originalVertexIds']
    delta, payload, report = repair(world, data['deltaWorld'], actual_report['plan'])
    report.update(sourceProposalSha256=source_report['arraysSha256'], actualInputSha256=source_report['actualInputSha256'],
                  helperSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    before_candidate = world+data['deltaWorld']
    after = world+delta
    pairs = data['crossingPairs']
    report['failedPairsBefore'] = pair_hits(before_candidate, tri, pairs)
    report['failedPairsAfter'] = pair_hits(after, tri, pairs)
    payload.update(originalVertexIds=ids, beforeWorld=world, parentDeltaWorld=data['deltaWorld'],
                   deltaWorld=delta, triangles=tri, oldCrossingPairs=pairs)

    def persist():
        np.savez_compressed(output.with_suffix('.npz'), **payload)
        report['arraysSha256'] = hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
        output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    persist()
    contact = runpy.run_path(str(HERE/'adult_blink_midpath_contact_r11.py'))
    changed = np.linalg.norm(delta, axis=1)>1e-10
    crosses = contact['changed_intersections'](after, tri, changed)
    payload['crossingPairs'] = np.asarray(sorted(crosses), np.int32).reshape(-1, 2)
    report['sourceTessellationCrossings'] = len(crosses)
    report['sourceTessellationOnly'] = True
    report['candidateRequiresActualPostTessellationAndAppearance'] = True
    report['candidateRejected'] = bool(crosses)
    q0, q1 = world[tri], after[tri]
    area0 = np.linalg.norm(np.cross(q0[:,1]-q0[:,0],q0[:,2]-q0[:,0]),axis=1)*.5
    area1 = np.linalg.norm(np.cross(q1[:,1]-q1[:,0],q1[:,2]-q1[:,0]),axis=1)*.5
    touched = changed[tri].any(1)
    report['minimumChangedAreaSquareMeters'] = float(area1[touched].min())
    report['minimumChangedAreaRatio'] = float((area1[touched]/area0[touched]).min())
    report['degenerateChangedTriangles'] = int((area1[touched]<=1e-12).sum())
    payload['changedAreaBefore'] = area0[touched]
    payload['changedAreaAfter'] = area1[touched]
    payload['changedTriangleRows'] = np.flatnonzero(touched)
    persist()
    print(json.dumps({'output': str(output), 'crossings': len(crosses), 'oldPairsAfter': report['failedPairsAfter'], 'sides': report['sides']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
