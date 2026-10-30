"""Pure neutral contour-to-tissue transport for an unaccepted shape preview.

Profiles describe actual lower/upper margins and a separate continuous target.
The central aperture and far cheek stay fixed. A C2 extension carries each
margin displacement into neighbouring tissue; actual triangle-globe radial
transport determines depth. This is not a blink path or inner contact repair.
"""
from pathlib import Path
import runpy
import json

import numpy as np


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def smooth5(value):
    value = np.clip(value, 0., 1.)
    return value**3*(10.-15.*value+6.*value**2)


def vertical_extension(z, lower, upper, lower_delta, upper_delta, padding=.004):
    """C2 extension of an affine opening map into the upper and lower tissue."""
    height = upper-lower
    if np.any(height <= .00001):
        raise ValueError('Finite positive source margin separation required')
    slope = (upper_delta-lower_delta)/height
    answer = lower_delta+slope*(z-lower)
    below = z < lower
    t = np.clip((z[below]-(lower[below]-padding))/padding, 0., 1.)
    answer[below] = ((10.*t**3-15.*t**4+6.*t**5)*lower_delta[below] +
                     (-4.*t**3+7.*t**4-3.*t**5)*padding*slope[below])
    above = z > upper
    t = np.clip((z[above]-upper[above])/padding, 0., 1.)
    answer[above] = ((1.-10.*t**3+15.*t**4-6.*t**5)*upper_delta[above] +
                     (t-6.*t**3+8.*t**4-3.*t**5)*padding*slope[above])
    return answer


def transport(world, triangles, original_ids, eyes, radial_metadata, profiles,
              anchors, support_radius=.005):
    """Return proposed world delta, point arrays, and neutral-only diagnostics.

    Per-side profile arrays: x (ascending world X), lower, upper, targetLower,
    targetUpper; startOutwardMeters/endOutwardMeters are measured from the actual
    eye center in the side's outward direction. Source/target profiles must
    include a short zero-displacement continuation outside the real head bridge.
    Anchors contain current margin world points and actual local triangle rows.
    """
    world = np.asarray(world, float)
    triangles = np.asarray(triangles, np.int32)
    ids = np.asarray(original_ids, np.int32)
    if world.shape != (len(ids), 3):
        raise ValueError('Current indexed world geometry required')
    graph_api = api('adult_outer_eye_surface_support_preview_r11')
    radial = api('adult_neutral_eye_outerends_radial_r11_v3')
    rays = api('adult_blink_corner_plan_r11')['ray_radius']
    graph = graph_api['adjacency'](world, triangles)
    total = np.zeros_like(world)
    payload = {}
    report = {'previewOnly': True, 'productionReady': False, 'appearanceAccepted': False,
              'middleBlinkProvided': False, 'innerContactRepairProvided': False,
              'method': 'Continuous upper/lower target, C2 tissue extension, mesh-distance falloff, true-globe radial depth transport.',
              'sides': {}}
    side_support = {}
    for side in ('L', 'R'):
        profile = profiles[side]
        x = np.asarray(profile['x'], float)
        channels = {name: np.asarray(profile[name], float) for name in ('lower', 'upper', 'targetLower', 'targetUpper')}
        if len(x) < 8 or np.any(np.diff(x) <= 0.) or any(v.shape != x.shape or not np.isfinite(v).all() for v in channels.values()):
            raise ValueError('Finite continuous current and target profile samples required')
        if np.any(channels['upper']-channels['lower'] <= .00001) or np.any(channels['targetUpper'] <= channels['targetLower']):
            raise ValueError('Ordered source/target margins required; no exact mathematical sheet collapse')
        details = radial_metadata['sides'][side]
        center = np.asarray(details['centerWorld'], float)
        sign = float(details['outwardSign'])
        outward = sign*(world[:, 0]-center[0])
        distance = graph_api['surface_distance'](world, triangles, graph,
            np.asarray(anchors[side]['world']), np.asarray(anchors[side]['triangleRows'], np.int32), support_radius)
        weight = 1.-smooth5((distance-.0004)/(support_radius-.0004))
        weight[~np.isfinite(distance)] = 0.
        weight *= smooth5((center[1]-world[:, 1])/.0008)
        weight[outward <= profile['startOutwardMeters']] = 0.
        weight[outward >= profile['endOutwardMeters']] = 0.
        side_support[side] = weight > 0.
        sampled = {name: np.interp(world[:, 0], x, values) for name, values in channels.items()}
        desired_delta_z = vertical_extension(world[:, 2], sampled['lower'], sampled['upper'],
            sampled['targetLower']-sampled['lower'], sampled['targetUpper']-sampled['upper'])*weight
        selected = np.flatnonzero(abs(desired_delta_z) > 1e-9)
        points = world[selected]
        rho = np.linalg.norm(points[:, 1:]-center[None, 1:], axis=1)
        if np.any(rho <= 1e-6):
            raise RuntimeError('Requested contour support reaches an angular singularity')
        theta_old = np.arctan2(points[:, 2]-center[2], center[1]-points[:, 1])
        query = radial['query_coordinates'](points, radial_metadata, side)
        origins = query.copy()
        origins[:, 1:] = center[1:]

        def radius(angle):
            directions = np.column_stack([np.zeros(len(angle)), -np.cos(angle), np.sin(angle)])
            answer = rays(origins, directions, eyes[side]['world'], eyes[side]['triangles'])
            if not np.isfinite(answer).all() or np.any(answer <= 0.):
                raise RuntimeError('A current real-globe radial query failed')
            return answer

        old_radius = radius(theta_old)
        radial_weight = radial['smooth'](sign*(details['actualGlobeOuterX']-points[:, 0])/radial_metadata['radialTransportFadeMeters'])
        desired_z = points[:, 2]+desired_delta_z[selected]-center[2]
        # The actual triangle-section radius varies with theta. Near its upper
        # or lower pole, repeated asin(target/radius) can diverge even when a
        # valid forward-surface root exists. Bracket on the requested side of
        # the original theta and solve the actual section, without clamping Z.
        increasing = desired_delta_z[selected] > 0.
        low = np.where(increasing, theta_old, -np.pi/2.+1e-7)
        high = np.where(increasing, np.pi/2.-1e-7, theta_old)

        def height(angle):
            r = rho+radial_weight*(radius(angle)-old_radius)
            if np.any(r <= 1e-6):
                raise RuntimeError('Contour transport collapses a real tissue section')
            return r*np.sin(angle), r

        low_z, _ = height(low)
        high_z, _ = height(high)
        bad = np.flatnonzero((low_z > desired_z+1e-9) | (high_z < desired_z-1e-9))
        if len(bad):
            raise RuntimeError('Actual radial section cannot bracket requested contour height: '+json.dumps({
                'side': side, 'count': len(bad), 'globalIds': ids[selected[bad[:12]]].tolist()}))
        for _ in range(21):
            theta = (low+high)*.5
            z_at_theta, new_rho = height(theta)
            low = np.where(z_at_theta < desired_z, theta, low)
            high = np.where(z_at_theta >= desired_z, theta, high)
        theta = (low+high)*.5
        _, new_rho = height(theta)
        target_error = abs(new_rho*np.sin(theta)-desired_z)
        if len(target_error) and target_error.max() > 2e-6:
            raise RuntimeError('True-globe contour transport did not converge to its height target')
        proposed = np.column_stack([points[:, 0], center[1]-new_rho*np.cos(theta), center[2]+new_rho*np.sin(theta)])
        delta = proposed-points
        total[selected] += delta
        payload['support_'+side] = weight
        payload['surfaceDistance_'+side] = distance
        payload['requestedDeltaZ_'+side] = desired_delta_z
        payload['changedLocalVertexIds_'+side] = selected
        report['sides'][side] = {'changedPoints': len(selected), 'supportRadiusMeters': support_radius,
            'centralApertureProtectedThroughOutwardMeters': profile['startOutwardMeters'],
            'maximumDisplacementMeters': float(np.linalg.norm(delta, axis=1).max()) if len(delta) else 0.,
            'targetHeightMaximumErrorMeters': float(target_error.max()) if len(target_error) else 0.,
            'radiusPreservedOutsideGlobePoints': int((radial_weight == 0.).sum()),
            'limits': 'Actual post-transport head intersections, globe signed clearance, layer order and visible silhouette must still be checked.'}
    if np.any(side_support['L'] & side_support['R']):
        raise RuntimeError('Independent outer-half tissue domains overlap')
    return total, payload, report
