"""Pure offline rigid twin-tail candidates from an actual REST observation.

No bpy, render, source modification, or claimed final collision acceptance.
The NumPy fallback deliberately bounds triangle/sample counts. It computes
exact distances only to a spatially sampled subset of triangles, so its
oriented distances are a coarse screening signal, not a mesh containment test.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rotation_x(degrees):
    a = math.radians(degrees)
    r = np.asarray([[1., 0., 0.], [0., math.cos(a), -math.sin(a)],
                    [0., math.sin(a), math.cos(a)]])
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-14) or abs(np.linalg.det(r) - 1.) > 1e-14:
        raise RuntimeError('Candidate must be a rigid proper rotation')
    return r


def transform(points, origin, rotation, target):
    return (points - origin) @ rotation.T + target


def closest_triangles(points, triangles):
    """Vectorized closest point on each corresponding candidate triangle."""
    p = points[:, None, :]
    a, b, c = triangles[:, :, 0], triangles[:, :, 1], triangles[:, :, 2]
    ab, ac = b-a, c-a
    normals = np.cross(ab, ac)
    nn = np.maximum(np.sum(normals*normals, axis=2), 1e-30)
    projected = p - normals * (np.sum((p-a)*normals, axis=2)/nn)[:, :, None]
    v = projected-a
    d00 = np.sum(ab*ab, axis=2); d01 = np.sum(ab*ac, axis=2); d11 = np.sum(ac*ac, axis=2)
    d20 = np.sum(v*ab, axis=2); d21 = np.sum(v*ac, axis=2)
    den = np.maximum(d00*d11-d01*d01, 1e-30)
    u = (d11*d20-d01*d21)/den; w = (d00*d21-d01*d20)/den
    valid_plane = (u >= 0) & (w >= 0) & (u+w <= 1)
    candidates = [projected]
    for start, end in ((a,b), (b,c), (c,a)):
        edge = end-start
        t = np.clip(np.sum((p-start)*edge, axis=2)/np.maximum(np.sum(edge*edge, axis=2), 1e-30), 0., 1.)
        candidates.append(start+t[:, :, None]*edge)
    q = np.stack(candidates, axis=2)
    dist = np.sum((q-p[:, :, None, :])**2, axis=3)
    dist[:, :, 0] = np.where(valid_plane, dist[:, :, 0], np.inf)
    which = np.argmin(dist, axis=2)
    hits = np.take_along_axis(q, which[:, :, None, None], axis=2)[:, :, 0, :]
    dsq = np.min(dist, axis=2)
    best = np.argmin(dsq, axis=1)
    row = np.arange(len(points))
    chosen = hits[row, best]
    normal = normals[row, best]/np.sqrt(nn[row, best])[:, None]
    signed = np.sum((points-chosen)*normal, axis=1)
    return np.sqrt(dsq[row, best]), signed


def spatial_subset(triangles, maximum):
    """Preserve spatial and normal-direction coverage while bounding cost."""
    if len(triangles) <= maximum:
        return np.arange(len(triangles), dtype=np.int32)
    centers = triangles.mean(1)
    span = np.ptp(centers, axis=0)
    surface_box_area = max(2*(span[0]*span[1]+span[0]*span[2]+span[1]*span[2]), 1e-8)
    cell = math.sqrt(surface_box_area/maximum)*.65
    xyz = np.floor((centers-centers.min(0))/cell).astype(np.int32)
    normal = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    axis = np.argmax(np.abs(normal), axis=1)
    direction = axis*2 + (normal[np.arange(len(normal)),axis] >= 0)
    keys = np.column_stack((xyz, direction))
    _, ids = np.unique(keys, axis=0, return_index=True)
    if len(ids) > maximum:
        ids = ids[np.linspace(0, len(ids)-1, maximum, dtype=np.int32)]
    return np.sort(ids).astype(np.int32)


class ApproxSurface:
    def __init__(self, prefix, arrays, maximum=6000, neighbors=6):
        vertices = arrays[prefix+'_world']
        faces = arrays[prefix+'_triangles']
        triangles = vertices[faces]
        self.prefix = prefix
        self.lo, self.hi = vertices.min(0), vertices.max(0)
        self.original_count = len(triangles)
        normals = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        nondegenerate = np.linalg.norm(normals, axis=1) > 1e-15
        triangles = triangles[nondegenerate]
        ids = spatial_subset(triangles, maximum)
        self.triangles = triangles[ids]
        self.origin = self.triangles.mean((0,1))
        self.centers = self.triangles.mean(1)-self.origin
        self.norm = np.sum(self.centers*self.centers, axis=1)
        self.neighbors = min(neighbors, len(self.triangles))

    def check(self, points, radii):
        # A point outside the expanded full-mesh AABB cannot be contained by
        # that mesh. Segment interiors remain a separately unverified scope.
        broad = np.all(points >= self.lo-.0015-radii[:,None], axis=1) & np.all(points <= self.hi+.0015+radii[:,None], axis=1)
        ids = np.flatnonzero(broad)
        gap = []
        for begin in range(0, len(ids), 128):
            sel = ids[begin:begin+128]
            q = points[sel]-self.origin
            distance_sq = np.maximum(np.sum(q*q, axis=1)[:,None]+self.norm[None,:]-2*q@self.centers.T, 0.)
            nearest = np.argpartition(distance_sq, self.neighbors-1, axis=1)[:, :self.neighbors]
            distance, signed = closest_triangles(points[sel], self.triangles[nearest])
            gap.extend((signed-radii[sel]).tolist())
        gap = np.asarray(gap)
        return {'prefix': self.prefix, 'samplePointsWithinExpandedFullAABB': len(ids),
                'minimumApproximateOrientedGapMeters': float(gap.min()) if len(gap) else None,
                'samplesBelowMinusHalfMillimeter': int(np.sum(gap < -.0005)),
                'samplesBelowOneMillimeter': int(np.sum(gap < .001)),
                'maximumApproximatePenetrationMeters': float(max(0., -gap.min())) if len(gap) else 0.,
                'sourceTriangles': self.original_count, 'sampledTriangles': len(self.triangles),
                'neighborsPerQuery': self.neighbors,
                'method': 'NumPy nearest sampled-triangle centroids, then closest point on those triangles; oriented normal distance, not closed-mesh containment.'}


def side_geometry(side, arrays, observation, sample_curves):
    tail = arrays['tail_'+side+'_world']
    starts = arrays['tail_'+side+'_starts']
    lengths = arrays['tail_'+side+'_lengths']
    if len(starts) != 2480 or not np.all(lengths == 35):
        raise RuntimeError('Actual tail topology differs from the observed production contract')
    grid = tail.reshape(2480, 35, 3)
    chosen_curves = np.linspace(0, 2479, sample_curves, dtype=np.int32)
    ids = list((chosen_curves[:,None]*35+np.arange(35)[None,:]).ravel())
    for station in range(35):
        for axis in range(3):
            ids.extend((int(np.argmin(grid[:,station,axis]))*35+station,
                        int(np.argmax(grid[:,station,axis]))*35+station))
    ids = np.unique(ids).astype(np.int32)
    tail_row = next(r for r in observation['tails'] if r['prefix'] == 'tail_'+side)
    radius_meta = tail_row['radius']
    if radius_meta['exists']:
        values = arrays['tail_'+side+'_radius']
        if radius_meta['domain'] == 'CURVE': values = np.repeat(values, 35)
        scale = max(tail_row['matrixSingularValues'])
        radii = values[ids]*scale
        radius_scope = 'Actual raw radii with maximum world matrix singular scale'
    else:
        radii = np.zeros(len(ids))
        radius_scope = 'Raw radius absent; sampled centerline only, no fabricated radius. Actual evaluated radius remains unverified.'
    raw_rings = np.concatenate([arrays[f'tie_{side}_{ring}_raw_world'] for ring in ('a','b')])
    eval_rings = np.concatenate([arrays[f'tie_{side}_{ring}_eval_world'] for ring in ('a','b')])
    origin = raw_rings.mean(0)
    points = np.concatenate((tail[ids], eval_rings))
    radii = np.r_[radii, np.zeros(len(eval_rings))]
    return {'tail': tail, 'rawRings': raw_rings, 'evaluatedRings': eval_rings,
            'origin': origin, 'samples': points, 'radii': radii,
            'sampleMetadata': {'tailSamples': len(ids), 'allTailPoints': len(tail),
                'tailUniformCurves': len(chosen_curves), 'eachStationAxisExtremaIncluded': True,
                'ringSamples': len(eval_rings), 'radiusScope': radius_scope}}


def make_side_candidates(side, sign, arrays, observation, surfaces, width, sample_curves):
    ear = arrays['ear_'+side+'_world']
    geom = side_geometry(side, arrays, observation, sample_curves)
    origin = geom['origin']
    outer = float(ear[:,0].max() if sign > 0 else ear[:,0].min())
    lobe = float(ear[:,2].min())
    posterior95 = float(np.percentile(ear[:,1],95))
    x = outer-sign*.02*width
    z = lobe+.04*width
    rows = []
    for pitch in (0., -5., -10., -15.):
        rotation = rotation_x(pitch)
        ring_offset = (geom['rawRings']-origin)@rotation.T
        y_base = posterior95-float(ring_offset[:,1].min())+.001
        for step in range(8):
            extra = step*.003
            target = np.asarray((x,y_base+extra,z))
            points = transform(geom['samples'], origin, rotation, target)
            checks = [surface.check(points, geom['radii']) for surface in surfaces]
            maximum_penetration = max(r['maximumApproximatePenetrationMeters'] for r in checks)
            negatives = sum(r['samplesBelowMinusHalfMillimeter'] for r in checks)
            near = sum(r['samplesBelowOneMillimeter'] for r in checks)
            transformed_tail = transform(geom['tail'], origin, rotation, target)
            lower = geom['tail'][:,2] < np.median(geom['tail'][:,2])
            lower_y_shift = float(np.median(transformed_tail[lower,1]-geom['tail'][lower,1]))
            row = {'originWorld': origin.tolist(), 'targetWorld': target.tolist(),
                   'rotationWorld': rotation.tolist(), 'pitchDegrees': pitch,
                   'rearExtraMeters': extra, 'posteriorSeedWorldY': y_base,
                   'centroidTranslationWorld': (target-origin).tolist(),
                   'approximateCollisionChecks': checks,
                   'approximateClearancePass': near == 0,
                   'maximumApproximatePenetrationMeters': maximum_penetration,
                   'negativeSampleCount': negatives, 'nearSampleCount': near,
                   'lowerTailMedianPosteriorShiftMeters': lower_y_shift,
                   'transformedTailWorldBounds': [transformed_tail.min(0).tolist(), transformed_tail.max(0).tolist()]}
            # Prefer a conservative clear candidate, then avoid a large rear
            # displacement of the lower hair or unnecessary pitch/rear travel.
            row['ranking'] = [int(near > 0), maximum_penetration, negatives, near,
                              abs(lower_y_shift)+extra*.4+abs(pitch)*.00025]
            rows.append(row)
        print('TIE_OFFLINE_SCREEN '+json.dumps({'side':side,'pitch':pitch,'candidates':len(rows)}), flush=True)
    rows.sort(key=lambda r: tuple(r['ranking']))
    best = rows[0]
    basis = {'earOutermostWorldX': outer, 'earLowestWorldZ': lobe,
             'earPosteriorY95': posterior95, 'originWorld': origin.tolist(),
             'targetWorldX': x, 'targetWorldZ': z,
             'originPolicy': 'Common centroid of all raw world vertices of both rings on this side',
             'horizontalPolicy': 'Ear outer X toward the center by 0.02 W',
             'verticalPolicy': 'Lowest actual ear Z plus 0.04 W',
             'posteriorPolicy': 'Actual ear Y95 minus minimum rotated ring relative Y plus 1 mm; then 0..21 mm in 3 mm steps',
             'sampleMetadata': geom['sampleMetadata'],
             'rawRingWorldBounds': [geom['rawRings'].min(0).tolist(), geom['rawRings'].max(0).tolist()],
             'ringHorizontalWidthOverEarSpan': float(np.ptp(geom['rawRings'][:,0])/width)}
    return best, rows, basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-triangles', type=int, default=6000)
    parser.add_argument('--tail-sample-curves', type=int, default=12)
    args = parser.parse_args()
    path = (P/args.observation).resolve(); out = (P/args.output).resolve()
    for item in (path,out): item.relative_to((P/'work/reconstruction-r02').resolve())
    if out.exists() or out.suffix != '.json': raise RuntimeError('New JSON output required')
    if not 500 <= args.max_triangles <= 20000 or not 4 <= args.tail_sample_curves <= 128:
        raise RuntimeError('Bounded approximate sampling required')
    observation = json.loads(path.read_text(encoding='utf8'))
    if observation['status'] not in ('collected','collected_with_unavailable_hidden_surfaces'):
        raise RuntimeError('Successful actual observation required')
    array_path = (P/observation['arrays']).resolve(); array_path.relative_to((P/'work/reconstruction-r02').resolve())
    if sha(array_path) != observation['arraysSha256']: raise RuntimeError('Observation array hash mismatch')
    arrays = np.load(array_path, allow_pickle=False)
    prefixes = []; excluded = []
    for row in observation['collisionSurfaces']:
        if row.get('evaluated') and row['visibility']['renderEligibleByCollectedFlags']:
            prefixes.append(row['prefix'])
        else:
            excluded.append({'prefix':row['prefix'],'object':row['object'],
                'evaluated':row.get('evaluated'),'visibility':row['visibility'],
                'reason':'Not both actual evaluated and currently render eligible; not promoted to mandatory by name.'})
    if 'head' not in prefixes: raise RuntimeError('Actual current head must be available as a mandatory surface')
    width = float(arrays['ear_positive_world'][:,0].max()-arrays['ear_negative_world'][:,0].min())
    if not .1 < width < .25: raise RuntimeError('Unexpected actual ear width')
    surfaces = [ApproxSurface(prefix, arrays, maximum=args.max_triangles) for prefix in prefixes]
    result = {'schemaVersion':1, 'observation':str(path.relative_to(P)),
        'observationSha256':sha(path), 'observationArraysSha256':observation['arraysSha256'],
        'source':observation['source'], 'sourceSha256':observation['sourceSha256'],
        'helperSha256':sha(__file__), 'status':'proposed', 'appearanceAccepted':False,
        'mandatoryCollisionPrefixes':prefixes, 'excludedCollisionSurfaces':excluded,
        'earOuterSpanWorldMeters':width, 'transforms':{}, 'candidateLists':{}, 'constraintBasis':{},
        'transformConvention':'T(p)=(p-originWorld)@rotationWorld.T+targetWorld; one rigid transform per side for the whole tail, all pins and both rings; no scaling.',
        'approximateOnly':True, 'productionCollisionAccepted':False,
        'unverified':['Full current Blender BVH against every tail point/segment and ring surface',
            'Mesh containment, edge/triangle intersections and hidden physical collider selection',
            'Actual post-install evaluated geometry and gather connection continuity',
            'Front and quarter appearance, ring size/material fidelity and the original C silhouette',
            'New pin mapping, dynamics/cache rebuild and motion collision tests']}
    for side,sign in (('positive',1),('negative',-1)):
        best,rows,basis = make_side_candidates(side,sign,arrays,observation,surfaces,width,args.tail_sample_curves)
        result['transforms'][side] = best
        result['candidateLists'][side] = rows
        result['constraintBasis'][side] = basis
    accepted = all(t['approximateClearancePass'] for t in result['transforms'].values())
    result['approximateScreeningAccepted'] = accepted
    if not accepted:
        result['status'] = 'blocked'
        result['blockingReason'] = 'No candidate met the conservative approximate 1 mm clearance target on both sides. Best diagnostic transforms are recorded but are not approved for production.'
    result['wholeSideRigidGeometryPreservedByConstruction'] = True
    with out.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print('TIE_OFFLINE_PLAN '+json.dumps({'output':str(out.relative_to(P)), 'status':result['status'],
        'selected':{s:{'pitch':t['pitchDegrees'],'rearExtra':t['rearExtraMeters'],'maxApproxPenetration':t['maximumApproximatePenetrationMeters']} for s,t in result['transforms'].items()}}),flush=True)


if __name__ == '__main__': main()
