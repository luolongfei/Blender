"""Current-size spherical eyelid path candidate, with explicit residuals.

Only consumes an actual adult_blink_recompute_r11 observation. Public shape
keys R11 blink L/R are independent 0..1 controls. Fifteen driven corrective
keys per eye keep interpolation close to the spherical trajectory instead
of cutting through the globe. Existing keys, topology and materials stay.

This is an executable candidate authoring tool, not a declaration that a
blink is natural or fully closed. The actual L3 audit records residual globe
visibility at intermediate strengths and closure; images remain required.
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

HEAD = 'R11 Female head study'
MARKER = 'R11_spherical_blink_candidate'
CONTROLS = {s: 'R11 blink '+s for s in ('L', 'R')}
INTERVALS = 16
PARAMETERS = {'lowerLidShare': .10, 'cornerFadeMeters': .006,
              'remainingAngularGapFraction': .025,
              'radialFullInfluenceMeters': .015, 'radialOuterMeters': .025}


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _array(items):
    result = np.empty((len(items), 3), np.float32)
    items.foreach_get('co', result.ravel())
    return result


def _path(relative, suffix):
    path = (project_path()/relative).resolve()
    path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix != suffix:
        raise ValueError('Independent reconstruction artifact required')
    return path


def angular_displacement(points, center, angle, value):
    """Rotate each point in its YZ plane, preserving X and radial thickness."""
    if not 0. <= value <= 1.:
        raise ValueError('Bounded blink strength required')
    points = np.asarray(points, float)
    v = points-np.asarray(center, float)
    theta = np.asarray(angle, float)*value
    # phi=atan2(z,-y), so positive phi raises the visible point.
    delta = np.zeros_like(points)
    delta[:, 1] = v[:, 1]*np.cos(theta)+v[:, 2]*np.sin(theta)-v[:, 1]
    delta[:, 2] = -v[:, 1]*np.sin(theta)+v[:, 2]*np.cos(theta)-v[:, 2]
    return delta


def _triangles(donor):
    result = []
    for start, count in zip(donor['face_starts'], donor['face_lengths']):
        face = donor['corner_vertices'][start:start+count]
        result.extend((int(face[0]), int(face[j]), int(face[j+1])) for j in range(1, len(face)-1))
    return np.asarray(result, np.int32)


def _strict_triangle_intersections(points, triangles):
    """Nonadjacent interior crossings, distinct from changed normal direction.

    A normal may rotate beyond ninety degrees during a valid lid fold;
    dot(originalNormal, newNormal) alone is not a triangle inversion test.
    This local test records new actual crossings instead. Coplanar contact
    is explicitly outside its scope and requires evaluated-surface review.
    """
    p = points[triangles]; lo = p.min(axis=1); hi = p.max(axis=1)
    pairs = []
    for i in range(len(triangles)):
        ids = np.flatnonzero(np.all(hi[i]>lo[i+1:]+1e-9, axis=1) & np.all(hi[i+1:]>lo[i]+1e-9, axis=1))+i+1
        for j in ids:
            if not np.intersect1d(triangles[i], triangles[j]).size:
                pairs.append((i, int(j)))
    if not pairs:
        return set()
    pairs = np.asarray(pairs, np.int32); hits = np.zeros(len(pairs), bool)
    for reverse in (False, True):
        a = p[pairs[:, int(reverse)]]; b = p[pairs[:, int(not reverse)]]
        e1 = b[:, 1]-b[:, 0]; e2 = b[:, 2]-b[:, 0]
        for edge in range(3):
            origin = a[:, edge]; direction = a[:, (edge+1)%3]-origin
            cross = np.cross(direction, e2); det = np.sum(e1*cross, axis=1)
            valid = np.abs(det)>1e-15
            inv = np.zeros_like(det); inv[valid] = 1./det[valid]
            offset = origin-b[:, 0]; u = np.sum(offset*cross, axis=1)*inv
            q = np.cross(offset, e1); v = np.sum(direction*q, axis=1)*inv
            t = np.sum(e2*q, axis=1)*inv
            hits |= valid & (u>1e-7) & (v>1e-7) & (u+v<1.-1e-7) & (t>1e-6) & (t<1.-1e-6)
    return {tuple(map(int, pair)) for pair in pairs[hits]}


def _raw_metrics(points, deltas, donor):
    selected = np.any(np.linalg.norm(deltas, axis=2)>1e-10, axis=0)
    edges = donor['edges']; edges = edges[selected[edges].any(axis=1)]
    triangles = _triangles(donor); triangles = triangles[selected[triangles].any(axis=1)]
    p = points[triangles]
    n0 = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]); a0 = np.linalg.norm(n0, axis=1)
    valid = a0>1e-12; n0 = n0[valid]; a0 = a0[valid]; triangles = triangles[valid]
    e0 = np.linalg.norm(points[edges[:, 1]]-points[edges[:, 0]], axis=1)
    reports = []; baseline_crossings = _strict_triangle_intersections(points, triangles)
    for number, delta in enumerate(deltas):
        out = points+delta; p = out[triangles]
        n = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]); area = np.linalg.norm(n, axis=1)
        alignment = np.sum(n0*n, axis=1)/np.maximum(a0*area, 1e-20)
        ratio = np.linalg.norm(out[edges[:, 1]]-out[edges[:, 0]], axis=1)/np.maximum(e0, 1e-12)
        crossings = _strict_triangle_intersections(out, triangles)
        reports.append({'strength': number/INTERVALS, 'minimumNormalAlignment': float(alignment.min()),
                        'edgeRatioRange': [float(ratio.min()), float(ratio.max())],
                        'maximumWorldDisplacementMeters': float(np.linalg.norm(delta, axis=1).max()),
                        'normalDirectionChangesOver90Degrees': int(np.sum(alignment<=0.)),
                        'minimumTriangleAreaRatio': float((area/a0).min()),
                        'newNonadjacentTriangleCrossings': [list(v) for v in sorted(crossings-baseline_crossings)],
                        'originalNonadjacentTriangleCrossingCount': len(baseline_crossings)})
    return reports


def build_plan(observation, arrays):
    """Pure-array plan; actual observation, not the donor's old eye scale."""
    if not observation.get('proportionMarker') or observation.get('newBlinkInstalled'):
        raise RuntimeError('Actual new-size neutral eye observation required')
    skin = _api('natural_skin_adult_r11'); donor, landmarks, provenance = skin['_donor']()
    mixed = np.asarray(arrays['currentMixedCageLocal'], float)
    if mixed.shape != (10582, 3):
        raise RuntimeError('Preserved complete female cage required')
    matrix = np.asarray(observation['headMatrixWorld'], float)
    world = mixed@matrix[:3, :3].T+matrix[:3, 3]
    allowed = skin['_vertices_of_groups'](donor, {2, 3, 17})
    graph = skin['_graph'](world, donor['edges'], allowed)
    output = {}; report = {'parameters': PARAMETERS, 'provenance': provenance, 'sides': {}}
    for side in ('L', 'R'):
        eye = observation['eyes'][side]; center = np.asarray(eye['centerWorld'], float)
        diameter = np.mean(np.asarray(eye['diametersWorld'])[[0, 2]])
        if abs(diameter-.0246)>1e-5:
            raise RuntimeError('Observation is not the actual 24.6 mm eye: '+side)
        rows = [r for r in eye['contour'] if all(not r[e].get('unboundedAtGlobeRim', True)
                and r[e].get('globe') and r[e].get('headOutside120Micrometers') for e in ('upper', 'lower'))]
        if len(rows)<80:
            raise RuntimeError('Insufficient bounded actual lid-edge observation: '+side)
        rows.sort(key=lambda r: r['x']); xs = np.asarray([r['x'] for r in rows])
        contour = {e: np.asarray([r[e]['headOutside120Micrometers']['world'] for r in rows]) for e in ('upper', 'lower')}
        angles = {}
        for edge in ('upper', 'lower'):
            globe = np.asarray([r[edge]['globe']['world'] for r in rows])-center
            angles[edge] = np.arctan2(globe[:, 2], -globe[:, 1])
        gap = angles['upper']-angles['lower']
        if np.any(gap<=0.) or gap.max()>np.pi:
            raise RuntimeError('Invalid upper/lower actual angular aperture')
        # Do not call either topological ring a wet line. Actual visible L3
        # boundaries seed nearby editable cage tissue; both sheets receive
        # the same spatial rotation, retaining their radial separation.
        side_allowed = allowed & (world[:, 0]*center[0]>0.)
        ids = np.flatnonzero(side_allowed)
        seeds = {}
        for edge in ('upper', 'lower'):
            distance = np.sum((contour[edge][5:-5, None, :]-world[ids][None, :, :])**2, axis=2)
            seeds[edge] = sorted(set(ids[np.argsort(distance, axis=1)[:, :2]].ravel().tolist()))
        # A monotone angular map prevents upper/lower sheets overtaking.
        # The same map applies at every tissue radius; radial fade depends
        # on radius and X only, not on angular ordering within a section.
        central = np.abs(xs-center[0])<diameter*.36
        upper_phi = np.interp(world[:, 0], xs[central], angles['upper'][central])
        lower_phi = np.interp(world[:, 0], xs[central], angles['lower'][central])
        angular_gap = upper_phi-lower_phi
        rel = world-center; radius = np.linalg.norm(rel[:, 1:3], axis=1)
        phi = np.arctan2(rel[:, 2], -rel[:, 1])
        remaining_fraction = PARAMETERS['remainingAngularGapFraction']
        mid = lower_phi+PARAMETERS['lowerLidShare']*angular_gap
        target = mid+(phi-mid)*remaining_fraction
        upper_shift = -(1.-remaining_fraction)*(1.-PARAMETERS['lowerLidShare'])*angular_gap
        lower_shift = (1.-remaining_fraction)*PARAMETERS['lowerLidShare']*angular_gap
        above = phi>upper_phi; below = phi<lower_phi
        target[above] = phi[above]+upper_shift[above]*(1.-skin['_smooth']((phi[above]-upper_phi[above])/.85))
        target[below] = phi[below]+lower_shift[below]*(1.-skin['_smooth']((lower_phi[below]-phi[below])/.70))
        hinges = landmarks['eyes'][side]
        hinge_ids = [hinges['innerCanthusCandidate'], hinges['outerCanthusCandidate']]
        hx = sorted(world[hinge_ids, 0])
        corner_weight = skin['_smooth']((world[:, 0]-hx[0])/PARAMETERS['cornerFadeMeters'])*skin['_smooth']((hx[1]-world[:, 0])/PARAMETERS['cornerFadeMeters'])
        radial_weight = 1.-skin['_smooth']((radius-PARAMETERS['radialFullInfluenceMeters'])/
                      (PARAMETERS['radialOuterMeters']-PARAMETERS['radialFullInfluenceMeters']))
        angle = (target-phi)*corner_weight*radial_weight
        angle[~side_allowed] = 0.
        delta_world = np.stack([angular_displacement(world, center, angle, t)
                                for t in np.linspace(0., 1., INTERVALS+1)])
        metrics = _raw_metrics(world, delta_world, donor)
        selected = np.linalg.norm(delta_world[-1], axis=1)>1e-10
        output[side] = delta_world@np.linalg.inv(matrix[:3, :3]).T
        report['sides'][side] = {'centerWorld': center.tolist(), 'diameterMeters': float(diameter),
             'actualContourSamples': len(rows), 'seedIndices': seeds, 'affectedCageVertices': int(selected.sum()),
             'maximumAngleRadians': float(np.max(np.abs(angle))), 'hingeControlIndices': [hinges['innerCanthusCandidate'], hinges['outerCanthusCandidate']],
             'cageMetrics': metrics,
             'rawGeometrySafe': all(not r['newNonadjacentTriangleCrossings'] and r['minimumTriangleAreaRatio']>.05 and r['edgeRatioRange'][0]>.10
                                     and r['edgeRatioRange'][1]<3.5 for r in metrics)}
    report.update(intervals=INTERVALS, controls=CONTROLS, appearanceAccepted=False, actualEvaluationPending=True,
                  actualContactVerified=False, referenceObservationSha256=None)
    return output, report


def _load_observation(path):
    observation = json.loads(path.read_text(encoding='utf-8'))
    payload = path.with_suffix('.npz')
    if _sha(payload) != observation['arraysSha256']:
        raise RuntimeError('Current-size observation payload changed')
    with np.load(payload) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return observation, arrays


def _assert_current_observation(head, observation, arrays):
    import bpy
    eye_api = _api('adult_eye_proportion_r11')
    actual = eye_api['_mixed'](head.data)
    observed = arrays['currentMixedCageLocal']
    matrix = np.asarray(observation['headMatrixWorld'], float)
    world = observed@matrix[:3, :3].T+matrix[:3, 3]
    zone = (abs(world[:, 0])<.075) & (world[:, 1]<-.035) & (world[:, 2]>1.39) & (world[:, 2]<1.48)
    if np.max(np.abs(actual[zone]-observed[zone]))>2e-7:
        raise RuntimeError('Eye-region mix changed; renew actual current-size observation')
    if np.max(np.abs(np.asarray(head.matrix_world)-matrix))>2e-7:
        raise RuntimeError('Observed head transform changed')
    for side in ('L', 'R'):
        eye = _api('adult_blink_recompute_r11')['_snapshot'](bpy.data.objects['R11 Adult sclera '+side])
        known = arrays['eye_'+side+'_world']
        if eye['fullWorld'].shape != known.shape or np.max(np.abs(eye['fullWorld']-known))>2e-7:
            raise RuntimeError('Actual globe changed since observation: '+side)


def _add_corrective_driver(key, control, time):
    driver = key.driver_add('value').driver
    driver.type = 'SCRIPTED'
    variable = driver.variables.new(); variable.name = 'b'; variable.type = 'SINGLE_PROP'
    variable.targets[0].id_type = 'KEY'; variable.targets[0].id = key.id_data
    variable.targets[0].data_path = 'key_blocks['+json.dumps(control)+'].value'
    driver.expression = 'max(0.0,1.0-abs(b-'+repr(time)+')*'+str(INTERVALS)+')'


def audit_values(values=((0., 0.), (.25, .25), (.5, .5), (.75, .75), (1., 1.), (1., 0.), (0., 1.))):
    """Actual L3 visible-surface and bound-fibre observations, no source save."""
    import bpy
    observation_api = _api('adult_blink_recompute_r11')
    keys = bpy.data.objects[HEAD].data.shape_keys.key_blocks
    baseline = {s: float(keys[name].value) for s, name in CONTROLS.items()}
    reports = []
    try:
        for left, right in values:
            for side, value in (('L', left), ('R', right)):
                keys[CONTROLS[side]].value = value
            bpy.context.view_layer.update()
            full, _ = observation_api['_collect']()
            reports.append({'values': {'L': left, 'R': right}, 'eyes': {
                side: {k: v for k, v in full['eyes'][side].items() if k not in ('contour', 'disconnectedVisibleColumns')}
                for side in ('L', 'R')}, 'bindingValidation': full['bindingValidation']})
            print('R11_BLINK_ACTUAL '+json.dumps(reports[-1]), flush=True)
    finally:
        for side, value in baseline.items():
            keys[CONTROLS[side]].value = value
        bpy.context.view_layer.update()
    closure = [r for r in reports if r['values']=={'L': 1., 'R': 1.}]
    complete = bool(closure) and all(closure[0]['eyes'][s]['visibleSampleCount']==0 for s in ('L', 'R'))
    return {'samples': reports, 'fullClosureFrontVisibilityPassed': complete,
            'actualContactVerified': False,
            'limitations': ['Zero visible globe is necessary but does not prove lid thickness, absence of lid overlap, or natural appearance.',
                            'This audit does not yet validate gaze or smile combinations, eyelash self-contact, or a continuous video.']}


def prepare(source_relative, observation_relative, output_relative):
    import bpy
    source = _path(source_relative, '.blend'); output = _path(output_relative, '.blend')
    obs_path = _path(observation_relative, '.json'); report_path = output.with_suffix('.blink.json')
    if source == output or output.exists() or report_path.exists():
        raise RuntimeError('Preserve source/history and use a new independent output')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh independent background source required')
    if bpy.context.scene.get(MARKER):
        raise RuntimeError('A blink candidate is already installed; preserve/edit it explicitly')
    head = bpy.data.objects[HEAD]; keys = head.data.shape_keys
    if any(name in keys.key_blocks for name in CONTROLS.values()):
        raise RuntimeError('Existing blink controls must not be overwritten')
    if abs(keys.key_blocks['R11 adult eye proportion'].value-1.)>1e-7:
        raise RuntimeError('New globe and static eye-proportion key must be enabled together')
    observation, arrays = _load_observation(obs_path)
    old = _api('adult_jaw_contour_r11')['_keys'](head.data)
    pose = _api('production_uv_r08')['_pose_contract'](); source_sha = _sha(source)
    basis = _array(keys.reference_key.data); active = head.active_shape_key_index
    created = []; success = False
    try:
        with _api('face_contact_audit_r11')['rest_evaluation']():
            _assert_current_observation(head, observation, arrays)
            paths, report = build_plan(observation, arrays)
            if not all(r['rawGeometrySafe'] for r in report['sides'].values()):
                raise RuntimeError('Candidate folds/strains raw tissue; no keys installed: '+json.dumps(report['sides']))
            for side, path in paths.items():
                full = path[-1]
                key = head.shape_key_add(name=CONTROLS[side], from_mix=False); created.append(key.name)
                key.relative_key = keys.reference_key; key.value = 0.; key.slider_min = 0.; key.slider_max = 1.
                key.data.foreach_set('co', (basis+full).astype(np.float32).ravel())
                for step in range(1, INTERVALS):
                    time = step/INTERVALS
                    correction = path[step]-time*full
                    key = head.shape_key_add(name='R11 blink path '+side+' '+str(step).zfill(2), from_mix=False)
                    created.append(key.name); key.relative_key = keys.reference_key
                    key.value = 0.; key.slider_min = 0.; key.slider_max = 1.
                    key.data.foreach_set('co', (basis+correction).astype(np.float32).ravel())
                    _add_corrective_driver(key, CONTROLS[side], time)
            bpy.context.view_layer.update()
            report['actualEvaluation'] = audit_values()
        head.active_shape_key_index = active
        now = _api('adult_jaw_contour_r11')['_keys'](head.data)
        for name, original in old.items():
            current = now[name]
            if not np.array_equal(original['coords'], current['coords']) or any(original[k]!=current[k] for k in original if k!='coords'):
                raise RuntimeError('Existing shape key changed: '+name)
        if _api('production_uv_r08')['_pose_contract']()!=pose or _sha(source)!=source_sha:
            raise RuntimeError('Source or pose changed')
        report.update(source=source_relative, sourceSha256=source_sha, helperSha256=_sha(__file__),
                      observation=observation_relative, observationSha256=_sha(obs_path),
                      createdKeys=created, controlDefaults={'L': 0., 'R': 0.},
                      originalKeysPreserved=list(old), actualEvaluationPending=False,
                      status='independent-blink-path-candidate-requires-visible-review', appearanceAccepted=False)
        bpy.context.scene[MARKER] = json.dumps(report)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
            raise RuntimeError('Independent blink candidate save failed')
        report.update(output=output_relative, outputSha256=_sha(output), savedBlend=True)
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        success = True
        print('R11_BLINK_CANDIDATE_SAVED '+json.dumps({'output': output_relative,
              'sha256': report['outputSha256'], 'fullClosureFrontVisibilityPassed': report['actualEvaluation']['fullClosureFrontVisibilityPassed']}), flush=True)
        return report
    finally:
        if not success:
            for name in reversed(created):
                if head.data.shape_keys and name in head.data.shape_keys.key_blocks:
                    key = head.data.shape_keys.key_blocks[name]
                    try:
                        key.driver_remove('value')
                    except TypeError:
                        pass
                    head.shape_key_remove(key)
            head.active_shape_key_index = active
            if MARKER in bpy.context.scene:
                del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.observation, args.output)
