"""Pure-array reference for the separately bounded inner-lid contact layer."""
from pathlib import Path
import runpy
import numpy as np


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def normals(world, triangles):
    faces = world[triangles]
    weighted = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
    result = np.zeros_like(world)
    for corner in range(3):
        np.add.at(result, triangles[:, corner], weighted)
    return result/np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-15)


def changed_intersections(points, triangles, changed, cell_size=.0015):
    """Interior crossings with at least one changed face, against the whole ROI.

    A spatial hash limits candidate pairs; shared vertices and coplanar contact
    are excluded identically to the existing blink strict-triangle test.
    """
    tri = np.asarray(triangles, np.int32)
    p = points[tri]
    lo = p.min(1)
    hi = p.max(1)
    active = np.flatnonzero(changed[tri].any(1))
    if not len(active):
        return set()
    low = np.floor(lo/cell_size).astype(np.int32)
    high = np.floor(hi/cell_size).astype(np.int32)
    amin = lo[active].min(0)
    amax = hi[active].max(0)
    nearby = np.flatnonzero(np.all(hi>=amin, axis=1) & np.all(lo<=amax, axis=1))
    buckets = {}
    keys = {}
    for i in nearby:
        cells = [(x, y, z) for x in range(low[i, 0], high[i, 0]+1)
                 for y in range(low[i, 1], high[i, 1]+1)
                 for z in range(low[i, 2], high[i, 2]+1)]
        keys[i] = cells
        for key in cells:
            buckets.setdefault(key, []).append(int(i))
    pairs = set()
    for i in active:
        candidates = np.asarray(sorted(set(j for key in keys[i] for j in buckets[key])), np.int32)
        candidates = candidates[np.all(hi[i]>lo[candidates]+1e-9, axis=1) &
                                np.all(hi[candidates]>lo[i]+1e-9, axis=1)]
        if not len(candidates):
            continue
        shared = (tri[candidates, :, None] == tri[i][None, None, :]).any(axis=(1, 2))
        for j in candidates[~shared]:
            pairs.add((min(int(i), int(j)), max(int(i), int(j))))
    if not pairs:
        return set()
    pairs = np.asarray(sorted(pairs), np.int32)
    hits = np.zeros(len(pairs), bool)
    for reverse in (False, True):
        a = p[pairs[:, int(reverse)]]
        b = p[pairs[:, int(not reverse)]]
        e1 = b[:, 1]-b[:, 0]
        e2 = b[:, 2]-b[:, 0]
        for edge in range(3):
            origin = a[:, edge]
            direction = a[:, (edge+1)%3]-origin
            cross = np.cross(direction, e2)
            det = np.sum(e1*cross, axis=1)
            valid = abs(det)>1e-15
            inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
            offset = origin-b[:, 0]
            u = np.sum(offset*cross, axis=1)*inv
            q = np.cross(offset, e1)
            v = np.sum(direction*q, axis=1)*inv
            t = np.sum(e2*q, axis=1)*inv
            hits |= valid & (u>1e-7) & (v>1e-7) & (u+v<1.-1e-7) & (t>1e-6) & (t<1.-1e-6)
    return {tuple(map(int, pair)) for pair in pairs[hits]}


def build_masks(arrays, observation):
    index_class = api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
    masks = {side: np.zeros(len(arrays['head_originalVertexIds']), bool) for side in ('L', 'R')}
    eyes = {side: index_class(arrays['eye_'+side+'_world'], arrays['eye_'+side+'_triangles']) for side in ('L', 'R')}
    for state in observation['states']:
        world = arrays[state['headWorldArray']]
        normal = normals(world, arrays[state['trianglesArray']])
        for side in ('L', 'R'):
            center = np.asarray(observation['eyes'][side]['centerWorld'])
            candidate = (normal[:, 1]>.1) & (world[:, 1]<center[1]) & (abs(world[:, 0]-center[0])<.013)
            ids = np.flatnonzero(candidate)
            front, _, _ = eyes[side].front(world[ids])
            masks[side][ids[np.isfinite(front) & (front-world[ids, 1]<.00015)]] = True
    triangles = arrays['state_0_head_triangles']
    for side in ('L', 'R'):
        for _ in range(2):
            masks[side][np.unique(triangles[masks[side][triangles].any(1)])] = True
    if np.any(masks['L'] & masks['R']):
        raise RuntimeError('Independent inner-eye domains overlap')
    return masks


def correct(world, triangles, eye_worlds, eye_triangles, centers, masks,
            clearance=.00006, minimum_wall=.00008, point_normals=None):
    """Same front-axis formula as the native node layer; both sides read input."""
    index_class = api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
    head_index = index_class(world, triangles)
    normal = normals(world, triangles) if point_normals is None else point_normals
    total = np.zeros_like(world)
    report = {}
    for side in ('L', 'R'):
        ids = np.flatnonzero(masks[side] & (normal[:, 1]>.1) & (world[:, 1]<centers[side][1]))
        eye_index = index_class(eye_worlds[side], eye_triangles[side])
        globe_front, _, _ = eye_index.front(world[ids])
        head_front, _, _ = head_index.front(world[ids])
        mapped = np.isfinite(globe_front) & np.isfinite(head_front)
        ids = ids[mapped]
        globe_front = globe_front[mapped]
        head_front = head_front[mapped]
        requested = globe_front-clearance
        bounded = np.minimum(world[ids, 1], np.maximum(requested, head_front+minimum_wall))
        total[ids, 1] = bounded-world[ids, 1]
        moved = abs(total[ids, 1])>1e-10
        initial_negative = world[ids, 1]>globe_front+1e-8
        still_negative = bounded>globe_front+2e-6
        report[side] = {
            'mappedPoints': len(ids), 'changedPoints': int(moved.sum()),
            'initialNegativeFrontDepthCount': int(initial_negative.sum()),
            'remainingNegativeFrontDepthCount': int(still_negative.sum()),
            'minimumChangedFrontGapMeters': float((globe_front-bounded)[moved].min()) if moved.any() else None,
            'minimumChangedOuterThicknessMeters': float((bounded-head_front)[moved].min()) if moved.any() else None,
            'maximumDisplacementMeters': float(abs(total[ids, 1]).max()) if len(ids) else 0.,
            'blockedByOuterWallCount': int((moved & (requested<head_front+minimum_wall-1e-8)).sum()),
            'changedLocalVertexIds': ids[moved],
            'remainingNegativeLocalVertexIds': ids[still_negative],
        }
    return total, report
