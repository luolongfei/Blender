"""Pure current-R11 partial nasal apparent-contour measurement.

No Blender, deformation or file writes. Recomputes apparent contours from the
supplied complete world surface. Original global vertex/triangle IDs are
preserved. These fragments are NOT a complete anatomical aperture boundary.
"""
from pathlib import Path
import runpy
import numpy as np

MIDPOINT_FRONT_TOLERANCE = 1e-6
CROSS_PROJECTED_OFFSET = .00005
MINIMUM_DEPTH_JUMP = .0005
UPPER_CAP_MINIMUM_DOWNWARD = .000035


def _components(edges):
    adjacency = {int(v): set() for v in np.unique(edges)}
    for a, b in edges:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    seen, sizes = set(), []
    for start in adjacency:
        if start in seen:
            continue
        seen.add(start)
        todo, count = [start], 0
        while todo:
            item = todo.pop()
            count += 1
            for other in adjacency[item]-seen:
                seen.add(other)
                todo.append(other)
        sizes.append(count)
    degrees = [len(v) for v in adjacency.values()]
    return {'componentVertexCounts': sizes,
            'degreeCounts': {str(d): degrees.count(d) for d in sorted(set(degrees))},
            'completeAnatomicalApertureClaim': False}


def measure(world, triangles):
    """Return (arrays, report) for the same frozen29e3 visibility criterion.

    Inputs are complete actual evaluated world coordinates and triangles.
    ``triangles`` row indices become the reported global triangle IDs. All
    bounds and thresholds are explicit current-R11 world-meter coordinates.
    NaN rays remain missing; they are never relabeled as original triangle0.
    """
    W = np.asarray(world, dtype=float)
    T = np.asarray(triangles)
    if W.ndim != 2 or W.shape[1] != 3 or not np.isfinite(W).all():
        raise ValueError('Finite complete actual world Nx3 surface required')
    if T.ndim != 2 or T.shape[1] != 3 or not len(T) or not np.issubdtype(T.dtype, np.integer):
        raise ValueError('Complete integer global triangle array required')
    if T.min() < 0 or T.max() >= len(W):
        raise ValueError('Global triangle indices exceed actual surface')
    P = W[T]
    normal = np.cross(P[:, 1]-P[:, 0], P[:, 2]-P[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    # Preserve the exact frozen diagnostic's source-edge candidate region.
    selected = ((np.abs(P[:, :, 0]).min(1) < .016) & (P[:, :, 0].max(1) > -.016)
        & (P[:, :, 2].max(1) > 1.389) & (P[:, :, 2].min(1) < 1.403)
        & (P[:, :, 1].min(1) < -.10))
    tids = np.flatnonzero(selected)
    local_tri = T[tids]
    E = np.sort(np.concatenate((local_tri[:, [0, 1]], local_tri[:, [1, 2]], local_tri[:, [2, 0]])), axis=1)
    owner = np.tile(tids, 3)
    order = np.lexsort((E[:, 1], E[:, 0]))
    E, owner = E[order], owner[order]
    shared = np.flatnonzero(np.all(E[1:] == E[:-1], axis=1))
    if len(shared) > 1 and np.any(np.diff(shared) == 1):
        raise RuntimeError('Nonmanifold edge in nasal source triangles; no unique two-sided contour')
    flip = shared[normal[owner[shared], 1]*normal[owner[shared+1], 1] < 0.]
    edges = E[flip]
    pairs = np.column_stack((owner[flip], owner[flip+1]))
    mid = W[edges].mean(1)
    if not len(edges):
        raise RuntimeError('No current-R11 nasal normal-flip candidates; inspect the supplied source/frame')
    # Every candidate ray is checked against the wider complete nasal region,
    # including the nearer nasal tip/skin, not just against candidate edges.
    rays = np.flatnonzero((P[:, :, 0].min(1) <= .024) & (P[:, :, 0].max(1) >= -.024)
        & (P[:, :, 2].min(1) < 1.414) & (P[:, :, 2].max(1) > 1.382))
    api = runpy.run_path(str(Path(__file__).with_name('adult_blink_inner_thickness_support_r11.py')))
    index = api['FrontTriangleIndex'](W, T[rays])

    def first(points):
        y, row, count = index.front(points)
        ids = np.full(len(row), -1, np.int64)
        valid = (row >= 0) & np.isfinite(y)
        ids[valid] = rays[row[valid]]
        return y, ids, count

    front_y, front_ids, _ = first(mid)
    occlusion = mid[:, 1]-front_y
    visible = np.isfinite(front_y) & (np.abs(occlusion) < MIDPOINT_FRONT_TOLERANCE)
    direction = W[edges[:, 1]][:, [0, 2]]-W[edges[:, 0]][:, [0, 2]]
    perp = np.column_stack((-direction[:, 1], direction[:, 0]))
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-20)
    qa, qb = mid.copy(), mid.copy()
    qa[:, [0, 2]] += perp*CROSS_PROJECTED_OFFSET
    qb[:, [0, 2]] -= perp*CROSS_PROJECTED_OFFSET
    ya, ia, _ = first(qa)
    yb, ib, _ = first(qb)
    A = np.column_stack((qa[:, 0], ya, qa[:, 2]))
    B = np.column_stack((qb[:, 0], yb, qb[:, 2]))
    jump = np.abs(ya-yb)
    valid_cross = np.isfinite(ya) & np.isfinite(yb)
    rim = (visible & valid_cross & (jump > MINIMUM_DEPTH_JUMP)
        & (np.abs(mid[:, 0]) > .003) & (np.abs(mid[:, 0]) < .015)
        & (mid[:, 2] > 1.390) & (mid[:, 2] < 1.402))
    inside = np.where((ya > yb)[:, None], A, B)
    interior_ids = np.where(ya > yb, ia, ib)
    upper = rim & (inside[:, 2] < mid[:, 2]-UPPER_CAP_MINIMUM_DOWNWARD)
    arrays = {'edges': edges, 'triangles': pairs, 'midpoints': mid,
        'normalY': normal[pairs, 1], 'frontY': front_y, 'depthOcclusion': occlusion,
        'frontTriangleIds': front_ids, 'visible': visible,
        'crossProjectedOffsetMeters': np.array(CROSS_PROJECTED_OFFSET),
        'crossRayAWorld': A, 'crossRayBWorld': B,
        'crossRayATriangleIds': ia, 'crossRayBTriangleIds': ib,
        'depthJumpMeters': jump, 'visibleRimWithDepthJump': rim,
        'insideRayWorld': inside, 'insideRayTriangleIds': interior_ids,
        'upperCapMask': upper, 'upperCapVertexIds': np.unique(edges[upper]),
        'visibleArcVertexIds': np.unique(edges[rim])}
    report = {'scope': 'Recomputed partial front-visible nasal apparent contour, not complete anatomical aperture boundary or area.',
        'inputVertices': len(W), 'inputGlobalTriangles': len(T), 'edgeCandidateSourceTriangles': len(tids),
        'firstHitOccluderTriangles': len(rays), 'normalFlipEdges': len(edges),
        'visibleDepthJumpEdges': int(rim.sum()), 'upperCapEdges': int(upper.sum()),
        'thresholdMeters': {'midpointFrontIdentification': MIDPOINT_FRONT_TOLERANCE,
            'projectedCrossOffsetEachSide': CROSS_PROJECTED_OFFSET,
            'minimumCrossDepthJump': MINIMUM_DEPTH_JUMP,
            'upperCapMinimumInsideDownward': UPPER_CAP_MINIMUM_DOWNWARD},
        'missingMidpointRays': int((~np.isfinite(front_y)).sum()),
        'missingCrossRayPairs': int((~valid_cross).sum()), 'sides': {},
        'geometryModified': False, 'completeAnatomicalApertureClaim': False}
    for side, sign in [('positiveX', 1), ('negativeX', -1)]:
        r = rim & (mid[:, 0]*sign > 0)
        u = upper & (mid[:, 0]*sign > 0)
        report['sides'][side] = {'visibleEdges': int(r.sum()), 'upperCapEdges': int(u.sum()),
            'visibleMidpointBoundsWorld': [mid[r].min(0).tolist(), mid[r].max(0).tolist()] if r.any() else None,
            'upperCapMidpointBoundsWorld': [mid[u].min(0).tolist(), mid[u].max(0).tolist()] if u.any() else None,
            'depthJumpMmPercentile0_50_100': (np.percentile(jump[r], [0, 50, 100])*1000).tolist() if r.any() else None,
            **_components(edges[r])}
    return arrays, report
