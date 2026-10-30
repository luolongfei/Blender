"""Bounded offline outer-skin thickness support for the proven inner-lid points.

No Blender import, source write, deformation or general collision audit. Uses
the recorded actual eye front intersection, then ray-tests the head triangles
at exactly the same X/Z with a projected-triangle spatial index.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrontTriangleIndex:
    def __init__(self, world, triangles, cell_size=.001):
        self.faces = np.asarray(world)[np.asarray(triangles)]
        self.e1 = self.faces[:, 1]-self.faces[:, 0]
        self.e2 = self.faces[:, 2]-self.faces[:, 0]
        self.determinant = self.e1[:, 0]*self.e2[:, 2]-self.e1[:, 2]*self.e2[:, 0]
        normal = np.cross(self.e1, self.e2)
        self.normal_y = normal[:, 1]/np.maximum(np.linalg.norm(normal, axis=1), 1e-20)
        xz = self.faces[:, :, [0, 2]]
        self.lo, self.hi = xz.min(axis=1), xz.max(axis=1)
        self.origin = self.lo.min(axis=0)
        self.cell_size = float(cell_size)
        low = np.floor((self.lo-self.origin-1e-10)/self.cell_size).astype(np.int32)
        high = np.floor((self.hi-self.origin+1e-10)/self.cell_size).astype(np.int32)
        self.buckets = {}
        for index in np.flatnonzero(abs(self.determinant)>1e-16):
            for x in range(low[index, 0], high[index, 0]+1):
                for z in range(low[index, 1], high[index, 1]+1):
                    self.buckets.setdefault((x, z), []).append(int(index))

    def front(self, points):
        result = np.full(len(points), np.nan)
        triangle_ids = np.full(len(points), -1, np.int32)
        tested = np.zeros(len(points), np.int32)
        for row, point in enumerate(points):
            cell = tuple(np.floor((point[[0, 2]]-self.origin)/self.cell_size).astype(np.int32))
            candidates = np.asarray(self.buckets.get(cell, []), np.int32)
            if not len(candidates):
                continue
            xz = point[[0, 2]]
            candidates = candidates[np.all((xz>=self.lo[candidates]-1e-10) & (xz<=self.hi[candidates]+1e-10), axis=1)]
            tested[row] = len(candidates)
            if not len(candidates):
                continue
            d = point-self.faces[candidates, 0]
            det = self.determinant[candidates]
            u = (d[:, 0]*self.e2[candidates, 2]-d[:, 2]*self.e2[candidates, 0])/det
            v = (self.e1[candidates, 0]*d[:, 2]-self.e1[candidates, 2]*d[:, 0])/det
            valid = (u>=-1e-8) & (v>=-1e-8) & (u+v<=1.+1e-8)
            if not valid.any():
                continue
            ids = candidates[valid]
            y = self.faces[ids, 0, 1]+u[valid]*self.e1[ids, 1]+v[valid]*self.e2[ids, 1]
            first = y.min()
            tied = ids[abs(y-first)<1e-10]
            chosen = tied[self.normal_y[tied].argmin()]
            result[row] = first
            triangle_ids[row] = chosen
        return result, triangle_ids, tested


def run(observation_relative, evidence_relative, contours_relative, output_relative):
    project = project_path().resolve()
    paths = [(project/relative).resolve() for relative in (observation_relative, evidence_relative, contours_relative, output_relative)]
    for path in paths:
        path.relative_to((project/'work/reconstruction-r02').resolve())
    observation_file, evidence_file, contour_file, output = paths
    if output.suffix!='.json' or output.exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Fresh independent thickness-support output required')
    observation = json.loads(observation_file.read_text(encoding='utf8'))
    evidence = json.loads(evidence_file.read_text(encoding='utf8'))
    contours = json.loads(contour_file.read_text(encoding='utf8'))
    if sha(observation_file.with_suffix('.npz'))!=observation['arraysSha256'] or sha(evidence_file.with_suffix('.npz'))!=evidence['arraysSha256']:
        raise RuntimeError('Actual source/evidence array checksum mismatch')
    if evidence['observationArraysSha256']!=observation['arraysSha256']:
        raise RuntimeError('Evidence belongs to another actual observation')
    with np.load(observation_file.with_suffix('.npz')) as data:
        observed = {key: data[key].copy() for key in data.files}
    with np.load(evidence_file.with_suffix('.npz')) as data:
        proven = {key: data[key].copy() for key in data.files}
    payload = {}
    record = {'source': observation['source'], 'sourceSha256': observation['sourceSha256'],
              'observation': observation_relative, 'observationArraysSha256': observation['arraysSha256'],
              'containmentEvidence': evidence_relative, 'containmentEvidenceSha256': sha(evidence_file),
              'helperSha256': sha(__file__), 'proposedGlobeFrontOffsetMeters': .00006,
              'minimumRemainingFrontThicknessMeters': .00008, 'groups': [],
              'method': 'Spatially indexed exact actual head triangle intersections from negative Y at each proven point X/Z. Outer front Y is the first captured head surface. Proposed inner Y equals recorded exact globe front Y minus 60 micrometres; X/Z stay unchanged.',
              'scope': 'Only the four previously proven neutral/full L/R inner-lining point sets. No new expression, whole-face audit or source deformation.',
              'limits': ['Thickness is front-axis separation, not a minimum Euclidean wall thickness or a future deformation acceptance.',
                         'Head geometry is the actual captured eye ROI. If no distinct front-facing outer cover is found, the point is flagged instead of assuming unseen coverage.',
                         'Neutral visible-boundary Z distance is a conservative margin proxy, not a named anatomical wet-line trace.',
                         'Vertex support does not alone exclude triangle folding or altered visible boundaries after smoothing.'],
              'sourceSaved': False, 'geometryModified': False}
    indices = {}
    for state_index in (0, 5):
        state = observation['states'][state_index]
        indices[state_index] = FrontTriangleIndex(observed[state['headWorldArray']], observed[state['trianglesArray']])
    for group in evidence['states']:
        state_index, side = group['stateIndex'], group['side']
        state = observation['states'][state_index]
        tag = group['arrayPrefix']
        points = proven[tag+'_world']
        global_ids = proven[tag+'_globalVertexIds']
        local_ids = proven[tag+'_localVertexIds']
        front, triangle_rows, tested = indices[state_index].front(points)
        finite = np.isfinite(front)
        first_normal_y = np.full(len(points), np.nan)
        first_normal_y[finite] = indices[state_index].normal_y[triangle_rows[finite]]
        globe_front_y = proven[tag+'_frontWorld'][:, 1]
        original_inner_y = points[:, 1]
        projected_inner_y = globe_front_y-.00006
        original_thickness = original_inner_y-front
        projected_thickness = projected_inner_y-front
        distinct_cover = finite & (original_thickness>1e-7) & (first_normal_y<-1e-5)
        no_cover = ~distinct_cover
        through_outer = finite & (projected_thickness < -1e-7)
        thin = finite & (projected_thickness<.00008)
        safe = distinct_cover & ~thin
        rows = contours['blink_000'][side]['contour']
        x = np.asarray([r['x'] for r in rows])
        low = np.asarray([r['lower']['worldZ'] for r in rows])
        upper = np.asarray([r['upper']['worldZ'] for r in rows])
        neutral = observed['state_0_head_world'][local_ids]
        upper_distance = neutral[:, 2]-np.interp(neutral[:, 0], x, upper)
        lower_distance = neutral[:, 2]-np.interp(neutral[:, 0], x, low)
        near_margin = np.minimum(abs(upper_distance), abs(lower_distance))<=.00030
        original_triangles = np.full(len(points), -1, np.int32)
        original_polygons = np.full(len(points), -1, np.int32)
        original_triangles[finite] = observed[state['originalTriangleIdsArray']][triangle_rows[finite]]
        original_polygons[finite] = observed[state['trianglePolygonIdsArray']][triangle_rows[finite]]
        payload.update({tag+'_originalVertexIds': global_ids,
                        tag+'_outerFrontY': front, tag+'_globeFrontY': globe_front_y,
                        tag+'_originalInnerY': original_inner_y, tag+'_projectedInnerY': projected_inner_y,
                        tag+'_originalFrontThicknessMeters': original_thickness,
                        tag+'_projectedFrontThicknessMeters': projected_thickness,
                        tag+'_outerNormalY': first_normal_y,
                        tag+'_outerOriginalTriangleIds': original_triangles,
                        tag+'_outerOriginalPolygonIds': original_polygons,
                        tag+'_headWorldXZ': points[:, [0, 2]],
                        tag+'_safeSupportMask': safe, tag+'_noDistinctOuterCoverMask': no_cover,
                        tag+'_projectsThroughOuterMask': through_outer, tag+'_below80MicrometresMask': thin,
                        tag+'_neutralMargin300MicrometresMask': near_margin,
                        tag+'_neutralUpperBoundaryDeltaZ': upper_distance,
                        tag+'_neutralLowerBoundaryDeltaZ': lower_distance,
                        tag+'_noOuterCoverVertexIds': global_ids[no_cover],
                        tag+'_throughOuterVertexIds': global_ids[through_outer],
                        tag+'_below80MicrometresVertexIds': global_ids[thin]})
        worst = np.argsort(np.where(finite, projected_thickness, -np.inf))[:8]
        row = {'stateIndex': state_index, 'blink': group['blink'], 'side': side,
               'inputPoints': len(points), 'exactRayHitPoints': int(finite.sum()),
               'safeSupportPoints': int(safe.sum()), 'noDistinctOuterCoverPoints': int(no_cover.sum()),
               'projectsThroughOuterPoints': int(through_outer.sum()), 'below80MicrometresPoints': int(thin.sum()),
               'neutralMargin300MicrometresPoints': int(near_margin.sum()),
               'safeButNearNeutralMarginPoints': int((safe&near_margin).sum()),
               'minimumOriginalFrontThicknessMeters': float(np.nanmin(original_thickness)),
               'minimumProjectedFrontThicknessMeters': float(np.nanmin(projected_thickness)),
               'minimumSafeProjectedFrontThicknessMeters': float(projected_thickness[safe].min()) if safe.any() else None,
               'projectedThicknessQuantilesMeters': np.nanquantile(projected_thickness, [0, .1, .5, .9, 1]).tolist(),
               'firstHeadNormalYRange': [float(np.nanmin(first_normal_y)), float(np.nanmax(first_normal_y))],
               'bboxCandidateTrianglesPerRay': [int(tested.min()), float(np.median(tested)), int(tested.max())],
               'noOuterCoverVertexIds': global_ids[no_cover].tolist(),
               'throughOuterVertexIds': global_ids[through_outer].tolist(),
               'below80MicrometresVertexIds': global_ids[thin].tolist(),
               'worstRemainingThicknessPoints': [{'globalVertexId': int(global_ids[i]),
                                                  'outerFrontY': float(front[i]) if finite[i] else None,
                                                  'globeFrontY': float(globe_front_y[i]),
                                                  'originalInnerY': float(original_inner_y[i]),
                                                  'projectedInnerY': float(projected_inner_y[i]),
                                                  'remainingFrontThicknessMeters': float(projected_thickness[i]) if finite[i] else None,
                                                  'firstHeadNormalY': float(first_normal_y[i]) if finite[i] else None,
                                                  'outerOriginalTriangleId': int(original_triangles[i]),
                                                  'outerOriginalPolygonId': int(original_polygons[i]),
                                                  'nearNeutralVisibleMargin': bool(near_margin[i])} for i in worst],
               'arrayPrefix': tag}
        record['groups'].append(row)
        print('R11_INNER_THICKNESS_SUPPORT '+json.dumps({k:row[k] for k in ('stateIndex','side','inputPoints','safeSupportPoints','noDistinctOuterCoverPoints','projectsThroughOuterPoints','below80MicrometresPoints','minimumProjectedFrontThicknessMeters')}), flush=True)
    np.savez_compressed(output.with_suffix('.npz'), **payload)
    record.update(arrays=str(output.with_suffix('.npz').relative_to(project)), arraysSha256=sha(output.with_suffix('.npz')))
    with output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--contours', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.observation, args.evidence, args.contours, args.output)
