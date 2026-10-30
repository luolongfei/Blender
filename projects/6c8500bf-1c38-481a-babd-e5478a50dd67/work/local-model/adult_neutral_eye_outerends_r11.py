"""Independent pure-array outer-aperture convergence proposal.

Only neutral Z fields are cached. Execution matches the native two-ray path:
the per-side early-blink fade scales Z, and actual globe front-depth change
carries Y. Neither an ideal sphere nor a new inner contact correction is used.
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

KEY = 'R11 refined outer eye corners'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


def fade(value):
    return 0. if float(value)>=.5 else 1.-float(smooth(max(float(value), 0.)/.5))


def _contour(row):
    rows = row['contour'] if isinstance(row, dict) else row
    x = np.asarray([p['x'] for p in rows])
    low = np.asarray([p['lower']['worldZ'] for p in rows])
    high = np.asarray([p['upper']['worldZ'] for p in rows])
    if len(x)<30 or np.any(np.diff(x)<=0) or np.any(high<low):
        raise ValueError('Actual ordered open-eye aperture contour required')
    return x, low, high


def _actual_projected_silhouette_distance(points, eye):
    """Distance to actual front/back triangle-edge projection, not a circle."""
    p = np.asarray(eye['world'], float)
    t = np.asarray(eye['triangles'], np.int32)
    q = p[t]
    ny = np.cross(q[:, 1]-q[:, 0], q[:, 2]-q[:, 0])[:, 1]
    edges = np.sort(np.concatenate([t[:, [0,1]], t[:, [1,2]], t[:, [2,0]]]), axis=1)
    signs = np.tile(ny<0., 3).astype(np.int32)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    front = np.bincount(inverse, weights=signs, minlength=len(unique))
    silhouette = unique[(front>0)&(front<counts)]
    if len(silhouette)<20:raise RuntimeError('Actual globe projected silhouette is incomplete')
    a, b = p[silhouette[:, 0]][:, [0,2]], p[silhouette[:, 1]][:, [0,2]]
    e = b-a
    query = np.asarray(points)[:, [0,2]]
    factor = np.clip(np.einsum('nfi,fi->nf', query[:, None]-a[None], e)/np.maximum(np.sum(e*e,axis=1), 1e-20), 0., 1.)
    distance = np.sqrt(np.sum((query[:, None]-(a[None]+factor[:,:,None]*e[None]))**2, axis=2).min(1))
    return distance, silhouette


def plan(neutral_world, triangles, original_ids, eyes, contours,
         width=.003, minimum_ratio=.25, compression=.75):
    """Return fields[side] (N,3), arrays, report. Only field Z is nonzero."""
    w = np.asarray(neutral_world, float)
    t = np.asarray(triangles, np.int32)
    ids = np.asarray(original_ids, np.int32)
    if w.shape!=(len(ids), 3) or width<=0 or not 0<minimum_ratio<1 or not 0<=compression<=1-minimum_ratio:
        raise ValueError('Bounded physical width/compression and aligned actual IDs required')
    index = api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
    fields, details = {}, {}
    arrays = {'originalVertexIds': ids, 'neutralWorld': w, 'triangles': t}
    for side in ('L', 'R'):
        eye = eyes[side]
        center = (eye['world'].min(0)+eye['world'].max(0))*.5
        sign = 1. if center[0]>0 else -1.
        x, lower, upper = _contour(contours[side])
        endpoint = x.max() if sign>0 else x.min()
        inward = sign*(endpoint-w[:, 0])
        mid = np.interp(w[:, 0], x, (lower+upper)*.5)
        half = np.interp(w[:, 0], x, (upper-lower)*.5)
        # Carry into the true lateral bridge; never truncate at the last
        # visibility sample, whose measured gap is still about 2 mm.
        lateral = (1.-smooth(np.maximum(inward, 0.)/width))*smooth((inward+.0015)/.0015)
        vertical = 1.-smooth((np.maximum(abs(w[:, 2]-mid)-half, 0.)-.0008)/.004)
        anterior = 1.-smooth((w[:, 1]-(center[1]-.004))/.006)
        support = lateral*vertical*anterior
        selected = np.flatnonzero(support>1e-10)
        front, hit_tri, _ = index(eye['world'], eye['triangles']).front(w[selected])
        # Actual projected horizon distance reaches zero at the ray boundary;
        # front-Y depth alone gives a steep square-root transition there.
        rim = np.zeros(len(selected))
        valid = np.isfinite(front)
        rim_distance, silhouette = _actual_projected_silhouette_distance(w[selected], eye)
        rim[valid] = smooth(rim_distance[valid]/.00075)
        support[selected] *= rim
        support[support<1e-9] = 0.
        field = np.zeros_like(w)
        field[:, 2] = -compression*support*(w[:, 2]-mid)
        fields[side] = field
        arrays['field_'+side] = field
        arrays['weight_'+side] = support
        arrays['neutralMidpointZ_'+side] = mid
        arrays['neutralInwardDistance_'+side] = inward
        arrays['neutralEyeFirstHitLocalIds_'+side] = selected[valid]
        arrays['neutralEyeFirstHitTriangleIds_'+side] = hit_tri[valid]
        arrays['actualGlobeSilhouetteEdges_'+side] = silhouette
        arrays['neutralSilhouetteDistanceCandidateLocalIds_'+side] = selected
        arrays['neutralSilhouetteDistanceMeters_'+side] = rim_distance
        details[side] = {'endpointX': float(endpoint), 'outwardSign': sign,
                        'nonzeroPoints': int((support>0).sum()),
                        'maximumAbsoluteDeltaZMeters': float(abs(field[:, 2]).max()),
                        'minimumCoreProportion': 1.-compression,
                        'actualProjectedGlobeBoundaryFadeMeters': .00075,
                        'actualLateralBridgeExtensionMeters': .0015}
    if np.any((fields['L'][:, 2]!=0)&(fields['R'][:, 2]!=0)):
        raise RuntimeError('Outer-end supports overlap')
    return fields, arrays, {'key': KEY, 'defaultStrength': 1., 'blinkFadeEnd': .5,
        'widthMeters': width, 'compression': compression, 'minimumRatio': minimum_ratio,
        'sides': details, 'nativeFieldContract': 'N by 3 world field; only Z nonzero; two actual globe front rays gate both Z and Y.',
        'appearanceAccepted': False, 'installed': False}


def apply(world, triangles, fields, eyes, blink=None, strength=1., verify=True):
    """Mirror fixed-Z native field and two-IsHit dynamic actual-eye transport."""
    w = np.asarray(world, float)
    t = np.asarray(triangles, np.int32)
    if not 0<=strength<=1:
        raise ValueError('Explicit bounded strength required')
    blink = blink or {'L': 0., 'R': 0.}
    index = api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
    normal = api('adult_blink_midpath_contact_r11')['normals'](w, t)
    total, reports = np.zeros_like(w), {}
    for side in ('L', 'R'):
        factor = fade(blink[side])*strength
        field = np.asarray(fields[side], float)
        if field.shape!=w.shape or np.any(field[:, :2]):
            raise ValueError('Native prototype accepts only finite Z fields')
        if factor==0:
            reports[side] = {'fade': 0., 'changedPoints': 0, 'maximumMeters': 0., 'twoHitRejectedCount': 0}
            continue
        candidate = np.flatnonzero(field[:, 2]*factor!=0.)
        proposed = w[candidate].copy()
        proposed[:, 2] += field[candidate, 2]*factor
        eye_index = index(eyes[side]['world'], eyes[side]['triangles'])
        before, first, _ = eye_index.front(w[candidate])
        after, second, _ = eye_index.front(proposed)
        valid = np.isfinite(before)&np.isfinite(after)
        ids = candidate[valid]
        total[ids, 2] += field[ids, 2]*factor
        total[ids, 1] += after[valid]-before[valid]
        gap_before = before[valid]-w[ids, 1]
        gap_after = after[valid]-(w[ids, 1]+total[ids, 1])
        outer = normal[ids, 1]<-.1
        newly_negative = outer&(gap_before>=-2e-8)&(gap_after<-2e-8)
        reports[side] = {'fade': factor, 'changedPoints': len(ids),
            'maximumMeters': float(np.linalg.norm(total[ids], axis=1).max()) if len(ids) else 0.,
            'twoHitRejectedCount': int((~valid).sum()), 'twoHitRejectedLocalIds': candidate[~valid],
            'changedLocalIds': ids, 'beforeEyeTriangleIds': first[valid], 'afterEyeTriangleIds': second[valid],
            'maximumFrontGapChangeMeters': float(abs(gap_after-gap_before).max()) if len(ids) else 0.,
            'originalNegativeInnerCount': int(((normal[ids, 1]>.1)&(gap_before<0)).sum()),
            'newOuterNegativeCount': int(newly_negative.sum()),
            'minimumPreviouslyPositiveOuterFrontGapMeters': float(gap_after[outer&(gap_before>=0)].min()) if (outer&(gap_before>=0)).any() else None}
        if verify and newly_negative.any():
            raise RuntimeError('Outer convergence newly put an outer point behind the actual globe')
    return total, reports


def aperture_slices(world, triangles, eye, endpoint, sign, distances=(.00015,.0005,.001,.0015,.002,.003,.004)):
    """Actual triangle first-hit visibility at a few useful X planes."""
    cls = api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
    head = cls(world, triangles)
    globe = cls(eye['world'], eye['triangles'])
    bounds = np.asarray([eye['world'][:, 2].min(), eye['world'][:, 2].max()])
    z = np.linspace(bounds[0]+.00001, bounds[1]-.00001, 801)
    result = []
    for inward in distances:
        x = endpoint-sign*inward
        points = np.column_stack([np.full(len(z), x), np.zeros(len(z)), z])
        gy, _, _ = globe.front(points)
        hy, _, _ = head.front(points)
        visible = np.isfinite(gy)&(~np.isfinite(hy)|(gy<hy-1e-8))
        ix = np.flatnonzero(visible)
        result.append({'inwardMeters': inward, 'x': x, 'lowerZ': float(z[ix[0]]) if len(ix) else None,
            'upperZ': float(z[ix[-1]]) if len(ix) else None,
            'gapMeters': float(z[ix[-1]]-z[ix[0]]) if len(ix) else 0., 'visibleSamples': len(ix)})
    return result


def _json(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    raise TypeError(type(value).__name__)


def run(observation_relative, contours_relative, output_relative, compression=.75):
    p = project_path()
    of, cf, out = [(p/q).resolve() for q in (observation_relative, contours_relative, output_relative)]
    for file in (of, cf, out):file.relative_to((p/'work/reconstruction-r02').resolve())
    if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('Fresh proposal paths required')
    obs = json.loads(of.read_text(encoding='utf8'))
    if sha(of.with_suffix('.npz'))!=obs['arraysSha256']:raise RuntimeError('Actual arrays checksum changed')
    with np.load(of.with_suffix('.npz')) as source:data={k:source[k].copy() for k in source.files}
    contours = json.loads(cf.read_text(encoding='utf8'))
    eyes = {s:{'world':data['eye_'+s+'_world'], 'triangles':data['eye_'+s+'_triangles']} for s in ('L','R')}
    fields, arrays, report = plan(data['state_0_head_world'],data['state_0_head_triangles'],data['head_originalVertexIds'],eyes,contours['blink_000'],compression=compression)
    roll = api('upper_lid_roll_surface_r11')
    weights, _, roll_plan = roll['plan'](data['state_0_head_world'],data['state_0_head_triangles'],data['head_originalVertexIds'],eyes,contours['blink_000'])
    contact = api('adult_blink_midpath_contact_r11')
    report.update(source=obs['source'],sourceSha256=obs['sourceSha256'],observation=observation_relative,
        observationArraysSha256=obs['arraysSha256'],helperSha256=sha(__file__),
        rollHelperSha256=sha(Path(__file__).with_name('upper_lid_roll_surface_r11.py')),states=[],
        limits=['Actual captured triangle topology is the offline reference; native evaluation can retessellate quads.',
                'Original inner-lining globe penetration is preserved and explicitly separated from new outer-surface penetration.',
                'Early fade can reopen the outer gap before the original blink closes it; new midpath must be derived from the new neutral aperture.',
                'This is an independent offline prototype, not a saved Blender or appearance acceptance.'])
    for n in (0,1):
        state=obs['states'][n];w=data[state['headWorldArray']];tri=data[state['trianglesArray']]
        rd,rm=roll['apply'](w,tri,weights,eyes,blink={s:state[s] for s in ('L','R')})
        delta,metrics=apply(w+rd,tri,fields,eyes,blink={s:state[s] for s in ('L','R')})
        result=w+rd+delta;changed=np.linalg.norm(result-w,axis=1)>1e-9
        print('OUTEREND_CHECK '+state['id']+' points '+str(changed.sum()),flush=True)
        crossings=contact['changed_intersections'](result,tri,changed)
        q0=w[tri[changed[tri].any(1)]];q1=result[tri[changed[tri].any(1)]]
        v0=np.cross(q0[:,1]-q0[:,0],q0[:,2]-q0[:,0]);v1=np.cross(q1[:,1]-q1[:,0],q1[:,2]-q1[:,0]);a0=np.linalg.norm(v0,axis=1);a1=np.linalg.norm(v1,axis=1);valid=a0>1e-14
        alignment=np.sum(v0[valid]*v1[valid],axis=1)/np.maximum(a0[valid]*a1[valid],1e-30)
        row={'stateIndex':n,'id':state['id'],'outerends':metrics,'roll':rm,'actualChangedInteriorCrossings':len(crossings),
             'minimumNormalAlignment':float(alignment.min()),'areaRatioRange':[float((a1[valid]/a0[valid]).min()),float((a1[valid]/a0[valid]).max())], 'aperture':{}}
        for s in ('L','R'):
            args=(eyes[s],report['sides'][s]['endpointX'],report['sides'][s]['outwardSign'])
            row['aperture'][s]={'before':aperture_slices(w,tri,*args),'after':aperture_slices(result,tri,*args)}
        arrays['state_'+str(n)+'_resultWorld']=result
        arrays['state_'+str(n)+'_outerDeltaWorld']=delta
        arrays['state_'+str(n)+'_rollDeltaWorld']=rd
        arrays['state_'+str(n)+'_crossingTrianglePairs']=np.asarray(sorted(crossings),np.int32).reshape((-1,2))
        report['states'].append(row)
        print('OUTEREND_RESULT '+json.dumps({'state':state['id'],'crossings':len(crossings),'minimumNormalAlignment':row['minimumNormalAlignment'],'gaps1mm':{s:row['aperture'][s]['after'][2]['gapMeters']*1000 for s in ('L','R')}}),flush=True)
    zeros={}
    for n in (2,3,4,5,6):
        state=obs['states'][n];delta,metrics=apply(data[state['headWorldArray']],data[state['trianglesArray']],fields,eyes,blink={s:state[s] for s in ('L','R')})
        if np.any(delta):raise RuntimeError('Late/full/new warm combination field is not identically zero')
        zeros[state['id']]=True
    report['lateStatesExactlyZero']=zeros
    report['offlineGeometryPassed']=all(r['actualChangedInteriorCrossings']==0 and r['minimumNormalAlignment']>0 for r in report['states'])
    np.savez_compressed(out.with_suffix('.npz'),**arrays)
    report['arraysSha256']=sha(out.with_suffix('.npz'))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=_json),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation',required=True);parser.add_argument('--contours',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--compression',type=float,default=.75)
    a=parser.parse_args();run(a.observation,a.contours,a.output,a.compression)
