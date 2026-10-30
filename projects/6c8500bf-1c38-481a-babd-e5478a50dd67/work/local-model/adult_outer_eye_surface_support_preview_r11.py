"""Neutral-only outer-eye direction study, not an accepted production field.

The rejected radial layer moved distant cheek skin and inherited a steep
canthal theta-mid derivative. This candidate spreads support by actual mesh
distance from the visible outer margins, and smooths the target angle profile.
It keeps the existing real-section transport formula for the local experiment.
"""
import hashlib
import heapq
from pathlib import Path
import runpy

import numpy as np

BASE = 'adult_neutral_eye_outerends_radial_r11_v3.py'
BASE_SHA = '9a508aff3d1dfd60814fe56658216adbc219c13bed3314e44ca22efe617d1394'


def base_api():
    file = Path(__file__).with_name(BASE)
    if hashlib.sha256(file.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError('Frozen radial reference changed')
    return runpy.run_path(str(file))


def smooth5(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10.-15.*t+6.*t*t)


def adjacency(world, triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                    triangles[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(world[edges[:, 1]]-world[edges[:, 0]], axis=1)
    origins = np.concatenate([edges[:, 0], edges[:, 1]])
    targets = np.concatenate([edges[:, 1], edges[:, 0]])
    lengths = np.tile(weights, 2)
    order = np.argsort(origins, kind='stable')
    offsets = np.concatenate([[0], np.cumsum(np.bincount(origins, minlength=len(world)))])
    return offsets, targets[order], lengths[order]


def surface_distance(world, triangles, graph, samples, triangle_ids, radius):
    """Multi-source edge-graph distance, seeded at actual visible-margin hits."""
    offsets, targets, lengths = graph
    distance = np.full(len(world), np.inf)
    queue = []
    for point, triangle_id in zip(samples, triangle_ids):
        if triangle_id < 0 or triangle_id >= len(triangles):
            raise RuntimeError('Current visible-edge hit does not index the current head')
        for vertex in triangles[triangle_id]:
            value = float(np.linalg.norm(world[vertex]-point))
            if value < distance[vertex]:
                distance[vertex] = value
                heapq.heappush(queue, (value, int(vertex)))
    while queue:
        value, vertex = heapq.heappop(queue)
        if value != distance[vertex] or value >= radius:
            continue
        for edge in range(offsets[vertex], offsets[vertex+1]):
            neighbor = int(targets[edge])
            proposed = value+lengths[edge]
            if proposed < min(distance[neighbor], radius):
                distance[neighbor] = proposed
                heapq.heappush(queue, (float(proposed), neighbor))
    return distance


def smooth_profile(x, values, sample_x, sigma=.0009):
    """Gaussian continuation with constant tails, avoiding a canthal slope kink."""
    step = .00005
    grid = np.arange(float(x.min())-4.*sigma, float(x.max())+4.*sigma+step/2., step)
    profile = np.interp(grid, x, values)
    half = int(np.ceil(4.*sigma/step))
    offset = np.arange(-half, half+1)*step
    kernel = np.exp(-.5*(offset/sigma)**2)
    kernel /= kernel.sum()
    smoothed = np.convolve(np.pad(profile, half, mode='edge'), kernel, mode='valid')
    return np.interp(sample_x, grid, smoothed)


def plan(neutral_world, triangles, original_ids, eyes, contours):
    reference = base_api()
    original_fields, arrays, metadata = reference['plan'](
        neutral_world, triangles, original_ids, eyes, contours)
    world = np.asarray(neutral_world, float)
    triangles = np.asarray(triangles, np.int32)
    graph = adjacency(world, triangles)
    fields = {}
    for side in ('L', 'R'):
        details = metadata['sides'][side]
        center = np.asarray(details['centerWorld'])
        sign = details['outwardSign']
        endpoint = details['endpointX']
        samples, hit_ids = [], []
        for edge in ('lower', 'upper'):
            points = arrays['edge_'+side+'_'+edge+'_headWorld']
            valid = arrays['edge_'+side+'_'+edge+'_coveredMask']
            inside = sign*(endpoint-points[:, 0])
            selected = valid & (inside >= 0.) & (inside <= .0022)
            samples.append(points[selected])
            hit_ids.append(arrays['edge_'+side+'_'+edge+'_headTriangleIds'][selected])
        samples = np.concatenate(samples)
        hit_ids = np.concatenate(hit_ids)
        if len(samples) < 12:
            raise RuntimeError('Too few actual outer-margin anchors for '+side)
        distance = surface_distance(world, triangles, graph, samples, hit_ids, .006)
        falloff = 1.-smooth5((distance-.00025)/.00575)
        inside = sign*(endpoint-world[:, 0])
        central_protection = smooth5((.0038-inside)/.0016)
        support = .65*falloff*central_protection
        support[~np.isfinite(distance)] = 0.
        lower = arrays['edge_'+side+'_lower_headWorld']
        upper = arrays['edge_'+side+'_upper_headWorld']
        valid = arrays['edge_'+side+'_lower_coveredMask'] & arrays['edge_'+side+'_upper_coveredMask']
        x = lower[valid, 0]
        low_angle = np.arctan2(lower[valid, 2]-center[2], center[1]-lower[valid, 1])
        high_angle = np.arctan2(upper[valid, 2]-center[2], center[1]-upper[valid, 1])
        mid = smooth_profile(x, (low_angle+high_angle)*.5, world[:, 0])
        half = smooth_profile(x, (high_angle-low_angle)*.5+.12, world[:, 0])
        fields[side] = np.column_stack([support, mid, half])
        arrays['field_'+side] = fields[side]
        arrays['originalField_'+side] = original_fields[side]
        arrays['surfaceDistance_'+side] = distance
        arrays['marginAnchorWorld_'+side] = samples
        arrays['marginAnchorTriangleIds_'+side] = hit_ids
        details['previewSupport'] = {
            'currentMarginAnchors': len(samples), 'edgeGraphRadiusMeters': .006,
            'innerApertureExactOutsideLastMeters': .0038,
            'angleProfileGaussianSigmaMeters': .0009,
            'supportPeakScale': .65, 'positiveSupportPoints': int((support > 0.).sum()),
            'interpretation': 'Actual mesh-distance falloff from the last 2.2 mm of visible margins; no whole-column cheek transport.'}
    if np.any((fields['L'][:, 0] > 0.) & (fields['R'][:, 0] > 0.)):
        raise RuntimeError('Preview surface support crossed between eyes')
    metadata.update(
        fieldChannels=['surfaceSupport', 'smoothedThetaMid', 'smoothedAngularCoreHalf'],
        previewOnly=True, productionReady=False, appearanceAccepted=False,
        previewDesign='Surface-distance support and smoothed canthal angle profile; actual neutral image must decide the direction.')
    return fields, arrays, metadata
