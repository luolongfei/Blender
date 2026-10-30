"""Local intermediate eyelid arc/contact fields from seven actual L3 states.

Neutral and fully closed fields are identically zero. The upper aperture uses
the actual neutral aperture profile, scaled by current central opening, to
avoid the old flat middle/late-closing corners. Inner tissue follows the real
triangle globe front surface; no ideal sphere or old cage delta is assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

KNOTS = (0., .2, .5, .8, .9, 1.)


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def front_depth(points, center, eye_world, eye_triangles, ray_radius):
    origins = np.asarray(points, float).copy()
    origins[:, 1] = center[1]-.04
    directions = np.zeros_like(origins)
    directions[:, 1] = 1.
    distance = ray_radius(origins, directions, eye_world, eye_triangles)
    return origins[:, 1]+distance


def contour_arrays(row):
    contour = row['contour']
    x = np.asarray([p['x'] for p in contour])
    low = np.asarray([p['lower']['worldZ'] for p in contour])
    high = np.asarray([p['upper']['worldZ'] for p in contour])
    return x, low, high


def make_side(state, side, arrays, observation, contours, ray_radius,
              arc_strength=1., clearance=.00006):
    index = state['index']
    world = np.asarray(arrays['state_'+str(index)+'_head_world'], float)
    neutral = np.asarray(arrays['state_0_head_world'], float)
    eye = np.asarray(arrays['eye_'+side+'_world'], float)
    triangles = arrays['eye_'+side+'_triangles']
    center = np.asarray(observation['eyes'][side]['centerWorld'], float)
    delta = np.zeros_like(world)
    x0, low0, up0 = contour_arrays(contours['blink_000'][side])
    xs, low, up = contour_arrays(contours[state['id']][side])
    central = abs(xs-center[0]) < .002
    neutral_gap = np.interp(xs, x0, up0-low0)
    fraction = float(np.median((up-low)[central]/neutral_gap[central]))
    desired_upper = low+neutral_gap*fraction
    blink = float(state[side])
    # A full proportional closure of the tiny canthal bridge folds that
    # joined upper/lower sheet. Keep that bridge and use a bounded central
    # arc correction, tapering before the existing late canthal solution.
    dx = xs-center[0]
    central_arc = np.maximum(1.-(dx/.011)**2, 0.)**2
    late_fade = 1.-smooth((blink-.6)/.35)
    upper_change = (np.clip(desired_upper-up, -.0005, .0002)+
                    .0007*np.sin(np.pi*blink)**2*central_arc)*late_fade
    offset = world-center
    # Spatial support comes from the current eye and original upper/lower
    # tissue identity, not a moving coarse coordinate box around the whole face.
    candidates = ((world[:, 0]>eye[:, 0].min()+.00002) &
                  (world[:, 0]<eye[:, 0].max()-.00002) &
                  (offset[:, 1]<0.) & (abs(offset[:, 2])<.013) &
                  (np.linalg.norm(offset[:, 1:], axis=1)<.020))
    ids = np.flatnonzero(candidates)
    old_depth = front_depth(world[ids], center, eye, triangles, ray_radius)
    valid = np.isfinite(old_depth)
    ids = ids[valid]
    old_depth = old_depth[valid]
    old_gap = old_depth-world[ids, 1]
    neutral_low = np.interp(neutral[ids, 0], x0, low0)
    neutral_high = np.interp(neutral[ids, 0], x0, up0)
    upper = smooth((neutral[ids, 2]-neutral_low)/np.maximum(neutral_high-neutral_low, .0001))
    dz_edge = world[ids, 2]-np.interp(world[ids, 0], xs, up)
    upper_support = 1.-smooth((np.maximum(dz_edge, 0.)-.0006)/.0045)
    upper_support *= upper
    upper_support *= 1.-smooth((old_gap-.004)/.002)
    lateral = smooth((world[ids, 0]-eye[:, 0].min()-.00005)/.0006)
    lateral *= smooth((eye[:, 0].max()-world[ids, 0]-.00005)/.0006)
    lateral *= 1.-smooth((abs(world[ids, 0]-center[0])-.010)/.0012)
    desired_dz = np.interp(world[ids, 0], xs, upper_change)*upper_support*lateral*arc_strength
    proposed = world[ids].copy()
    proposed[:, 2] += desired_dz
    new_depth = front_depth(proposed, center, eye, triangles, ray_radius)
    mapped = np.isfinite(new_depth)
    proposed[~mapped] = world[ids[~mapped]]
    new_depth[~mapped] = old_depth[~mapped]
    # Tangential travel carries the existing positive tissue thickness along
    # the actual globe. Only the inner sheet below the allowance is lifted.
    target_gap = np.maximum(old_gap, clearance)
    proposed[:, 1] = new_depth-target_gap
    delta[ids] = proposed-world[ids]
    inner_delta = np.zeros_like(world)
    inner_delta[ids, 1] = old_gap-target_gap
    changed = np.linalg.norm(delta[ids], axis=1)>1e-10
    contact_ids = ids[old_gap<clearance]
    return delta, {
        'centralApertureFraction': fraction,
        'changedVertices': int(changed.sum()),
        'maximumDisplacementMeters': float(np.linalg.norm(delta, axis=1).max()),
        'maximumUpperEdgeShiftMeters': float(abs(upper_change).max()*arc_strength),
        'priorInnerFrontNegativeGapCount': int((old_gap<0).sum()),
        'priorMinimumFrontDepthGapMeters': float(old_gap.min()),
        'newMinimumFrontDepthGapMeters': float((new_depth-proposed[:, 1]).min()),
        'unmappedNewHeightCount': int((~mapped).sum()),
        'contactPointIds': contact_ids,
        'allMappedPointIds': ids,
        'arcDelta': delta-inner_delta,
        'innerDelta': inner_delta,
    }


def evaluate_delta(value, fields):
    """Piecewise smoothstep interpolation of independently measured knot fields."""
    value = float(np.clip(value, 0., 1.))
    index = min(int(np.searchsorted(KNOTS, value, side='right')-1), len(KNOTS)-2)
    index = max(index, 0)
    blend = float(smooth((value-KNOTS[index])/(KNOTS[index+1]-KNOTS[index])))
    return fields[index]*(1.-blend)+fields[index+1]*blend


def run(observation_relative, contour_relative, output_relative, arc_strength=1.):
    project = project_path().resolve()
    observation_path = (project/observation_relative).resolve()
    contour_path = (project/contour_relative).resolve()
    output = (project/output_relative).resolve()
    for path in (observation_path, contour_path, output):
        path.relative_to((project/'work/reconstruction-r02').resolve())
    if output.exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Fresh proposal outputs required')
    observation = json.loads(observation_path.read_text(encoding='utf8'))
    if sha(observation_path.with_suffix('.npz')) != observation['arraysSha256']:
        raise RuntimeError('Actual observation checksum changed')
    with np.load(observation_path.with_suffix('.npz')) as data:
        arrays = {k: data[k].copy() for k in data.files}
    contours = json.loads(contour_path.read_text(encoding='utf8'))
    corner = api('adult_blink_corner_plan_r11')
    full = np.zeros((len(KNOTS), len(arrays['head_originalVertexIds']), 3), float)
    fields = {side: full.copy() for side in ('L', 'R')}
    arcs = {side: full.copy() for side in ('L', 'R')}
    inner = {side: full.copy() for side in ('L', 'R')}
    report = {'observation': observation_relative, 'observationSha256': sha(observation_path),
              'arraysSha256': observation['arraysSha256'], 'source': observation['source'],
              'sourceSha256': observation['sourceSha256'], 'helperSha256': sha(__file__),
              'knots': list(KNOTS), 'arcStrength': arc_strength, 'innerFrontAllowanceMeters': .00006,
              'states': [], 'combination': {}, 'actualInstallationPending': True,
              'appearanceAccepted': False, 'actualContactVerified': False,
              'preservedEndpoints': ['neutral', 'fullClosed'],
              'limits': ['Exact zero at endpoints preserves existing endpoint inner-lining intersections too.',
                         'Front-depth clearance is measured against the actual triangle globe, not an anatomical wet-line certification.',
                         'Knot fields still require real evaluation between knots and actual rendered appearance review.']}
    contact_arrays = {}
    for state in observation['states'][1:5]:
        index = state['index']
        row = {'index': index, 'blink': state['L'], 'sides': {}}
        for side in ('L', 'R'):
            delta, metrics = make_side(state, side, arrays, observation, contours, corner['ray_radius'], arc_strength)
            contact_arrays['state_'+str(index)+'_'+side+'_contactPointIds'] = metrics.pop('contactPointIds')
            contact_arrays['state_'+str(index)+'_'+side+'_mappedPointIds'] = metrics.pop('allMappedPointIds')
            arcs[side][index] = metrics.pop('arcDelta')
            inner[side][index] = metrics.pop('innerDelta')
            fields[side][index] = delta
            changed = arrays['state_'+str(index)+'_head_world']+delta
            visibility = corner['front_visibility'](changed, arrays['state_'+str(index)+'_head_triangles'],
                                                   arrays['eye_'+side+'_world'], arrays['eye_'+side+'_triangles'], refine=True)
            metrics['visibility'] = visibility
            row['sides'][side] = metrics
            print('R11_MIDPATH_PROPOSAL '+json.dumps({'blink': state['L'], 'side': side,
                  'maxMillimeters': metrics['maximumDisplacementMeters']*1000,
                  'visibleAreaSquareMillimeters': visibility['frontVisibleAreaSquareMeters']*1e6}), flush=True)
        report['states'].append(row)
    combo = observation['states'][6]
    combined_delta = sum(evaluate_delta(combo[side], fields[side]) for side in ('L', 'R'))
    combo_world = arrays['state_6_head_world']+combined_delta
    for side in ('L', 'R'):
        report['combination'][side] = corner['front_visibility'](combo_world, arrays['state_6_head_triangles'],
                                                               arrays['eye_'+side+'_world'], arrays['eye_'+side+'_triangles'], refine=True)
    np.savez_compressed(output.with_suffix('.npz'), originalVertexIds=arrays['head_originalVertexIds'],
                        fields_L=fields['L'], fields_R=fields['R'], arc_L=arcs['L'], arc_R=arcs['R'],
                        inner_L=inner['L'], inner_R=inner['R'], **contact_arrays)
    report.update(fields=str(output.with_suffix('.npz').relative_to(project)),
                  fieldsSha256=sha(output.with_suffix('.npz')))
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--contours', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arc-strength', type=float, default=1.)
    args = parser.parse_args()
    run(args.observation, args.contours, args.output, args.arc_strength)
