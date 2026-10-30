"""Positive-weight harmonic continuation in the skin beyond the real globe.

Only a neutral preview extension: eye-side positions remain exact Dirichlet
data. The exterior bridge uses its actual mesh neighbours to relax the prior
narrow transition; the far mesh-distance and outer-X boundaries stay fixed.
"""
from pathlib import Path
import runpy

import numpy as np


def extend(world, triangles, delta, metadata, distances, bridge_outward):
    graph = runpy.run_path(str(Path(__file__).with_name('adult_outer_eye_surface_support_preview_r11.py')))
    offsets, targets, lengths = graph['adjacency'](world, triangles)
    origins = np.repeat(np.arange(len(world)), np.diff(offsets))
    weights = 1./np.maximum(lengths, 1e-8)
    degree = np.bincount(origins, weights=weights, minlength=len(world))
    unknown = np.zeros(len(world), bool)
    eye_inside = np.zeros(len(world), bool)
    for side in ('L', 'R'):
        details = metadata['sides'][side]
        center = np.asarray(details['centerWorld']);sign = details['outwardSign']
        outward = sign*(world[:, 0]-center[0])
        globe_end = sign*(details['actualGlobeOuterX']-center[0])
        unknown |= ((outward > globe_end+.00002) &
                    (outward < bridge_outward[side]+.0012) &
                    (distances[side] < .0048))
        eye_inside |= (outward <= globe_end+.00002) & (distances[side] < .005)
    selected = np.flatnonzero(unknown)
    if not len(selected) or np.any(unknown & eye_inside):
        raise RuntimeError('Strictly outside-globe continuation domain required')
    mapping = np.full(len(world), -1, np.int32)
    mapping[selected] = np.arange(len(selected))
    edge = unknown[origins]
    row = mapping[origins[edge]]
    neighbour = targets[edge]
    weight = weights[edge]
    interior = unknown[neighbour]
    ir, ic, iw = row[interior], mapping[neighbour[interior]], weight[interior]
    br, bc, bw = row[~interior], neighbour[~interior], weight[~interior]
    diagonal = degree[selected]

    def apply(value):
        return diagonal*value-np.bincount(ir, weights=iw*value[ic], minlength=len(selected))

    output = np.asarray(delta, float).copy()
    records = []
    for axis in (1, 2):
        right = np.bincount(br, weights=bw*delta[bc, axis], minlength=len(selected))
        value = delta[selected, axis].copy()
        residual = right-apply(value)
        direction = residual.copy()
        rr = float(residual@residual)
        tolerance = 1e-10*max(float(np.linalg.norm(right)), 1.)
        iterations = 0
        while rr**.5 > tolerance and iterations < 2000:
            product = apply(direction)
            denominator = float(direction@product)
            if denominator <= 0.:
                raise RuntimeError('Actual bridge Dirichlet system is not positive definite')
            alpha = rr/denominator
            value += alpha*direction
            residual -= alpha*product
            next_rr = float(residual@residual)
            direction = residual+(next_rr/rr)*direction
            rr = next_rr
            iterations += 1
        if rr**.5 > tolerance:
            raise RuntimeError('Actual bridge continuation did not converge')
        output[selected, axis] = value
        records.append({'axis': axis, 'iterations': iterations, 'residualNorm': rr**.5, 'tolerance': tolerance})
    output[np.linalg.norm(output, axis=1) < 1e-10] = 0.
    if not np.array_equal(output[~unknown], delta[~unknown]):
        # Thresholding must not alter any fixed boundary values.
        output[~unknown] = delta[~unknown]
    if np.any(output[:, 0] != 0.) or not np.array_equal(output[eye_inside], delta[eye_inside]):
        raise RuntimeError('Continuation changed the protected eye-side field or skin X')
    return output, selected, {'unknownPoints': len(selected), 'linearSolves': records,
        'minimumGlobeOuterXMarginMeters': .00002, 'outerContinuationBeyondBridgeMeters': .0012,
        'farGeodesicBoundaryMeters': .0048, 'eyeSideDeltaExact': True,
        'method': 'Positive 1/edge-length Dirichlet Laplacian on actual triangle connectivity; no compression or contour-target change.'}
