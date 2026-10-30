"""Offline support-aware wrist-core surface fitting from actual failed faces.

This module never imports or launches Blender. A finished plan still requires
the unchanged actual continuous-surface and containment checks in all 3 poses.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

FAILURE='work/reconstruction-r02/reference-adult-lower-face-continuity-v2-r11-hand-surface-v5-failure.json'
FAILURE_SHA='41dad8177cd99075450827ac3e4f531b5f8ba2b1871c8040d732de05081e3890'
PROBE='work/reconstruction-r02/hand-integrated-wrist-probe-r11-v3.json'
PROBE_SHA='a940a192cb726623b6efb8601243b022d2a2d65ab8cda1f8626e9cc53ec64f3c'
PARENT='work/reconstruction-r02/hand-integrated-wrist-fit-plan-r11-v4.json'
PARENT_SHA='45d7d2348f7f3d7b5eddee376889de1eea33bc56628f6e252128171d68d836eb'
BASE='work/reconstruction-r02/hand-integrated-wrist-fit-plan-r11-v2.json'
BASE_SHA='88f306549c2f1f78327cfe363ac87772f3b04ed102228538304d5ad8fe7e8c56'
DEPS={'hand_wrist_subdivision_transfer_r11.py':'2f0ec272330c1131fb8c3260b0a32186742dc9416a9f7c4a14c4ed5903117aef',
      'hand_wrist_surface_contact_r11.py':'c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e',
      'hand_wrist_clearance_r11_v2.py':'715b366dc374f2755574b8ef640d6076d64711cc9e856e8d7a4a68475de26759'}


def path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(relative,digest):
    p=path(relative)
    if sha(p)!=digest:raise RuntimeError('Frozen input changed: '+relative)
    return json.loads(p.read_text(encoding='utf-8'))


class Stencils:
    def __init__(self,top):
        self.top=top;self.cache={};self.final={};self.neighbors=[set() for _ in range(top['nv']+top['ne']+len(top['faces']))]
        self.opposites=[[] for _ in self.neighbors]
        for q in top['quads']:
            for i,j in enumerate(q):
                self.neighbors[j].update((int(q[(i-1)%4]),int(q[(i+1)%4])))
                self.opposites[j].append(int(q[(i+2)%4]))

    @staticmethod
    def add(out,row,factor):
        for i,v in row.items():out[i]=out.get(i,0.)+v*factor

    def face(self,i):
        face=self.top['faces'][i];return {int(v):1/len(face) for v in face}

    def level(self,i):
        if i in self.cache:return self.cache[i]
        t=self.top;nv=t['nv'];ne=t['ne'];out={}
        if i>=nv+ne:out=self.face(i-nv-ne)
        elif i>=nv:
            ei=i-nv;a,b=t['edges'][ei];fs=t['edgeFaces'][ei]
            if len(fs)!=2:raise RuntimeError('Boundary stencil is not in the reviewed closed wrist')
            out={int(a):.25,int(b):.25}
            for fi in fs:self.add(out,self.face(fi),.25)
        else:
            es=t['vertexEdges'][i];fs=t['vertexFaces'][i];n=len(es)
            if not n or any(len(t['edgeFaces'][e])!=2 for e in es):raise RuntimeError('Nonmanifold raw support')
            out={i:(n-3)/n}
            for fi in fs:self.add(out,self.face(fi),1/(n*len(fs)))
            for ei in es:
                a,b=t['edges'][ei];self.add(out,{int(a):.5,int(b):.5},2/(n*len(es)))
        self.cache[i]=out;return out

    def limit(self,i):
        if i in self.final:return self.final[i]
        neighbors=self.neighbors[i];n=len(neighbors);den=n*(n+5);out={}
        self.add(out,self.level(i),n*n/den)
        for j in neighbors:self.add(out,self.level(j),4/den)
        for j in self.opposites[i]:self.add(out,self.level(j),1/den)
        if abs(sum(out.values())-1)>1e-12:raise RuntimeError('Non-unit subdivision stencil')
        self.final[i]=out;return out


def load():
    for name,digest in DEPS.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen math/geometry dependency changed: '+name)
    failure=read(FAILURE,FAILURE_SHA);probe=read(PROBE,PROBE_SHA)
    parent=read(PARENT,PARENT_SHA);base=read(BASE,BASE_SHA)
    if failure['helperSha256']!='b2317f453a3ecb59140aecb1e072084de86e62d5a3125438712230d7f0f8d190':raise RuntimeError('Not the actual frozen v5 failure')
    if sha(path(failure['source']))!=failure['sourceSha256']:raise RuntimeError('Actual failed source changed on disk')
    if [r['pose'] for r in failure['poses']]!=['relaxed','wrist-flex','wrist-extend']:raise RuntimeError('All 3 actual failure poses required')
    if any(not next(s for s in row['sides'] if s['side']=='L')['physicalCoreGatePassed'] for row in failure['poses']):
        raise RuntimeError('Accepted left wrist no longer passes actual physical gate')
    for report in (parent,base,probe):
        if sha(path(report['arrays']))!=report['arraysSha256']:raise RuntimeError('Frozen geometry arrays changed')
    with np.load(path(probe['arrays']),allow_pickle=False) as a:observed={k:a[k].copy() for k in a.files}
    with np.load(path(parent['arrays']),allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
    with np.load(path(base['arrays']),allow_pickle=False) as a:previous={k:a[k].copy() for k in a.files}
    api=runpy.run_path(str(HERE/'hand_wrist_subdivision_transfer_r11.py'))
    top=api['topology'](observed['raw_body_loops'],observed['raw_body_polygonStarts'],observed['raw_body_polygonSizes'],probe['bodyBeforeIntegration']['rawVertices'])
    values=np.zeros((top['nv'],6),np.float64);values[:,:3]=np.nan
    values[observed['raw_R_ids'],:3]=observed['raw_R_mix']
    values[arrays['R_ids'],3:]=arrays['R_delta'].astype(np.float64)-previous['R_delta'].astype(np.float64)
    _,limit=api['subdivide_limit'](values,top)
    finite=np.flatnonzero(np.isfinite(limit[:,:3]).all(1))
    rest=observed['relaxed_R_body_unposed'].astype(np.float64)
    mapping=np.empty(len(rest),np.int32);errors=np.empty(len(rest))
    for start in range(0,len(rest),128):
        squared=((rest[start:start+128,None]-limit[None,finite,:3])**2).sum(2)
        pick=squared.argmin(1);mapping[start:start+len(pick)]=finite[pick]
        errors[start:start+len(pick)]=np.sqrt(squared[np.arange(len(pick)),pick])
    ids=observed['relaxed_R_body_originalIds'];id_to_local={int(v):i for i,v in enumerate(ids)}
    current=rest+limit[mapping,3:]
    actual_rows={}
    for row in failure['poses']:
        right=next(s for s in row['sides'] if s['side']=='R')
        if right['continuousSurfaceContact'].get('contactDiagnosticsTruncated'):raise RuntimeError('Actual contact evidence truncated')
        for contact in right['continuousSurfaceContact']['contactPairsForDiagnosis']:
            for vi,point in zip(contact['bodyEvaluatedVertexIds'],contact['bodyOriginalTriangleUnposed']):actual_rows[int(vi)]=point
    match_rows=[]
    for vi,point in sorted(actual_rows.items()):
        local=id_to_local[vi];err=float(np.linalg.norm(current[local]-point))
        match_rows.append({'evaluatedId':vi,'localIndex':local,'l1Index':int(mapping[local]),
            'v3ReconstructionError':float(errors[local]),'v5TransferError':err})
        current[local]=point
    return {'failure':failure,'probe':probe,'parent':parent,'arrays':arrays,'previous':previous,'observed':observed,
        'top':top,'stencils':Stencils(top),'mapping':mapping,'mappingErrors':errors,'v3Limit':limit,
        'currentRest':current,'actualMatchRows':match_rows,'idToLocal':id_to_local}


def projected_overlap(body,hand):
    """Hand-plane overlap polygon with barycentric body-triangle coordinates."""
    normal=np.cross(hand[1]-hand[0],hand[2]-hand[0]);normal/=np.linalg.norm(normal)
    u=hand[1]-hand[0];u/=np.linalg.norm(u);v=np.cross(normal,u);basis=np.stack((u,v),axis=1)
    h=(hand-hand[0])@basis;b=(body-hand[0])@basis
    if abs(np.linalg.det(np.stack((b[1]-b[0],b[2]-b[0]))))<1e-14:return [],normal
    polygon=[(b[i].copy(),np.eye(3)[i]) for i in range(3)]
    for i in range(3):
        start,end=h[i],h[(i+1)%3];edge=end-start
        def side(p):return edge[0]*(p[1]-start[1])-edge[1]*(p[0]-start[0])
        clipped=[]
        for a,c in zip(polygon,polygon[1:]+polygon[:1]):
            da,dc=side(a[0]),side(c[0])
            if da>=0:clipped.append(a)
            if (da>=0)!=(dc>=0):
                t=da/(da-dc);clipped.append((a[0]+t*(c[0]-a[0]),a[1]+t*(c[1]-a[1])))
        polygon=clipped
        if not polygon:break
    return [{'barycentric':weights,'gap':float((weights@body-hand[0])@normal)} for _,weights in polygon],normal


def pose_data(data):
    observed=data['observed'];probe=data['probe'];rest=observed['relaxed_R_body_unposed'].astype(np.float64)
    package=json.loads(path(probe['package']).read_text(encoding='utf-8'));bones=package['bones']
    lower=np.asarray(bones['rightLowerArm']['pose'])@np.linalg.inv(bones['rightLowerArm']['rest'])
    hand=np.asarray(bones['rightHand']['pose'])@np.linalg.inv(bones['rightHand']['rest'])
    distal=np.abs(observed['relaxed_R_handRest_positions'][:,0])>.68
    reference=observed['relaxed_R_hand_positions'][distal].astype(np.float64)
    poses=[];numerator=np.zeros(len(rest));denominator=np.zeros(len(rest))
    for record in data['failure']['poses']:
        pose=record['pose'];prefix=pose.replace('-','_')+'_R';target=observed[prefix+'_hand_positions'][distal].astype(np.float64)
        xm,ym=reference.mean(0),target.mean(0);u,_,vt=np.linalg.svd((reference-xm).T@(target-ym));rotation=vt.T@u.T
        if np.linalg.det(rotation)<0:raise RuntimeError('Improper wrist pose fit')
        offset=ym-rotation@xm;transform=np.eye(4);transform[:3,:3]=rotation;transform[:3,3]=offset;current_hand=transform@hand
        low=rest@lower[:3,:3].T+lower[:3,3];high=rest@current_hand[:3,:3].T+current_hand[:3,3]
        actual=observed[prefix+'_body_actual'].astype(np.float64);difference=high-low
        numerator+=np.einsum('ij,ij->i',actual-low,difference);denominator+=np.einsum('ij,ij->i',difference,difference)
        poses.append({'pose':pose,'prefix':prefix,'lower':lower,'handTransform':current_hand,'actualV3':actual,
                      'hand':observed[prefix+'_hand_positions'].astype(np.float64),'handTriangles':observed[prefix+'_hand_triangles'],
                      'rigidFitError':float(np.linalg.norm(reference@rotation.T+offset-target,axis=1).max()),'failure':record})
    weights=numerator/np.maximum(denominator,1e-30)
    if np.any(weights<-.0001) or np.any(weights>1.0001):raise RuntimeError('Observed body is not explained by unchanged linear two-bone skinning')
    weights=np.clip(weights,0,1)
    for pose in poses:
        h=pose['handTransform'];l=pose['lower'];A=weights[:,None,None]*h[None,:3,:3]+(1-weights[:,None,None])*l[None,:3,:3]
        posed=pose['actualV3']+np.einsum('nij,nj->ni',A,data['currentRest']-rest)
        right=next(s for s in pose['failure']['sides'] if s['side']=='R');errors=[]
        for contact in right['continuousSurfaceContact']['contactPairsForDiagnosis']:
            for vi,point in zip(contact['bodyEvaluatedVertexIds'],contact['bodyOriginalTriangleWorld']):
                index=data['idToLocal'][vi];errors.append(np.linalg.norm(posed[index]-point));posed[index]=point
        pose.update({'body':posed,'deformation':A,'weightInference':weights,'actualContactTransferMaximumError':float(max(errors))})
    return poses


def support(data):
    contacts=data['actualMatchRows'];selected={}
    for row in contacts:
        selected[row['localIndex']]=data['stencils'].limit(row['l1Index'])
    all_raw=sorted(set(v for stencil in selected.values() for v in stencil))
    indices={int(v):i for i,v in enumerate(data['arrays']['R_ids'])}
    if any(v not in indices for v in all_raw):raise RuntimeError('Actual stencil leaves captured raw patch')
    baseline=data['arrays']['R_baselineMix'].astype(np.float64)
    def allowed(v):
        point=baseline[indices[v]]
        return -.631<point[0]<-.590 and abs(point[1])<.06 and abs(point[2]-1.235)<.06
    free=[v for v in all_raw if allowed(v)]
    if len(free)>96:raise RuntimeError('Actual support is unexpectedly broad')
    column={v:i for i,v in enumerate(free)};S=np.zeros((len(data['mapping']),len(free)))
    for i,l1 in enumerate(data['mapping']):
        if data['mappingErrors'][i]>1e-6:continue
        stencil=data['stencils'].limit(int(l1))
        for vi,w in stencil.items():
            if vi in column:S[i,column[vi]]=w
    affected=np.flatnonzero(S.sum(1)>0)
    if not len(affected):raise RuntimeError('Empty actual subdivision support')
    return {'ids':np.asarray(free,np.int32),'S':S,'affected':affected,'allContactStencilIds':all_raw,
            'excludedOutsideExistingRawDomain':[v for v in all_raw if v not in column],
            'localRawIndices':np.asarray([indices[v] for v in free],np.int32)}


def solve_qp(G,b,Q,max_cycles=5000):
    """Hildreth dual coordinate solve for min .5*x'Q*x, G*x <= b."""
    inverse=np.linalg.inv(Q);directions=inverse@G.T;diagonal=np.einsum('ij,ji->i',G,directions)
    if np.any(diagonal<1e-14):raise RuntimeError('Unsatisfiable constraint has no actual raw support')
    multipliers=np.zeros(len(b));x=np.zeros(G.shape[1]);max_violation=float('inf')
    for cycle in range(max_cycles):
        largest=0.
        for i in range(len(b)):
            new=max(0.,multipliers[i]+(G[i]@x-b[i])/diagonal[i]);change=new-multipliers[i]
            if change:
                x-=change*directions[:,i];multipliers[i]=new;largest=max(largest,abs(change))
        max_violation=float(np.max(G@x-b))
        if max_violation<1e-10 and largest<1e-9:break
    if max_violation>2e-9:raise RuntimeError('Local surface constraint solver did not converge: '+str(max_violation))
    return x,{'cycles':cycle+1,'maximumConstraintViolationMeters':max_violation,
              'activeConstraints':int((multipliers>1e-10).sum()),'objective':float(.5*x@Q@x)}


def contact_constraints(data,poses,sup,margin=20e-6):
    rows=[];bounds=[];evidence=[]
    for pose in poses:
        right=next(s for s in pose['failure']['sides'] if s['side']=='R')
        for pair in right['continuousSurfaceContact']['contactPairsForDiagnosis']:
            local=np.asarray([data['idToLocal'][vi] for vi in pair['bodyEvaluatedVertexIds']],np.int32)
            triangle=pose['body'][local];hand=np.asarray(pair['handTriangleWorld'])
            polygon,normal=projected_overlap(triangle,hand)
            for point in polygon:
                bary=point['barycentric'];G=np.zeros((len(sup['ids']),2))
                for i,weight in zip(local,bary):
                    direction=(pose['deformation'][i].T@normal)[1:]
                    G+=weight*sup['S'][i,:,None]*direction
                rows.append(G.ravel());bounds.append(-margin-point['gap'])
                evidence.append({'pose':pose['pose'],'bodyTriangle':pair['originalBodyTriangle'],
                    'handTriangle':pair['handTriangle'],'barycentric':bary.tolist(),'existingGap':point['gap']})
    return np.asarray(rows),np.asarray(bounds),evidence


def predicted_constraints(data,poses,sup,fit,checks,margin=20e-6):
    rows=[];bounds=[];evidence=[]
    for pose,check in zip(poses,checks):
        for pair in check['predictedCandidate'].get('_pairs',[]):
            local=np.asarray(pair['bodyLocalIds'],np.int32);hand=pose['hand'][pose['handTriangles'][pair['handTriangle']]]
            moved=pose['body'][local]+np.einsum('nij,nj->ni',pose['deformation'][local],fit['evaluatedDelta'][local])
            polygon,normal=projected_overlap(moved,hand)
            for point in polygon:
                bary=point['barycentric'];G=np.zeros((len(sup['ids']),2))
                for i,weight in zip(local,bary):G+=weight*sup['S'][i,:,None]*(pose['deformation'][i].T@normal)[None,1:]
                original_gap=float((bary@pose['body'][local]-hand[0])@normal)
                rows.append(G.ravel());bounds.append(-margin-original_gap)
                evidence.append({'pose':pose['pose'],'bodyLocalIds':local.tolist(),'handTriangle':pair['handTriangle'],
                    'barycentric':bary.tolist(),'existingGap':original_gap,'derivedFromPredictedDeformedOverlap':True})
    return np.asarray(rows),np.asarray(bounds),evidence


def candidate(data,poses):
    sup=support(data);G,b,evidence=contact_constraints(data,poses,sup)
    indices={int(v):i for i,v in enumerate(sup['ids'])};Q=np.eye(len(indices))
    # Penalize differential movement along actual raw cage edges, including
    # the zero-displacement boundary. This is not a new spatial brush.
    for a,c in data['top']['edges']:
        i,j=indices.get(int(a)),indices.get(int(c))
        if i is not None:Q[i,i]+=.25
        if j is not None:Q[j,j]+=.25
        if i is not None and j is not None:Q[i,j]-=.25;Q[j,i]-=.25
    rounds=[]
    for iteration in range(8):
        x,solver=solve_qp(G,b,np.kron(Q,np.eye(2)))
        delta=np.zeros((len(indices),3));delta[:,1:]=x.reshape(-1,2)
        moved=sup['S']@delta
        baseline=data['arrays']['R_baselineMix'].astype(np.float64)
        new_delta=data['arrays']['R_delta'].astype(np.float64).copy();new_delta[sup['localRawIndices']]+=delta
        wrist=runpy.run_path(str(HERE/'hand_wrist_clearance_r11_v2.py'))
        metrics=wrist['_triangle_metrics'](baseline,baseline+new_delta,data['arrays']['R_triangles'])
        fit={'support':sup,'delta':delta,'evaluatedDelta':moved,'newDelta':new_delta,
             'solver':solver,'triangleMetrics':metrics,'constraints':evidence}
        checks=offline_contacts(data,poses,fit)
        rounds.append({'iteration':iteration,'constraintCount':len(b),'maximumRawDisplacement':float(np.linalg.norm(delta,axis=1).max()),
            'poses':[{'pose':r['pose'],**{k:v for k,v in r['predictedCandidate'].items() if k not in ('contactPairs','_pairs')}} for r in checks]})
        if not any(r['predictedCandidate']['contactsAt1Micron'] for r in checks):
            fit['offlineChecks']=checks;fit['nonlinearConstraintRounds']=rounds
            linear=np.zeros((data['top']['nv'],3));linear[sup['ids']]=delta
            api=runpy.run_path(str(HERE/'hand_wrist_subdivision_transfer_r11.py'))
            _,whole=api['subdivide_limit'](linear,data['top'])
            influenced=np.flatnonzero(np.linalg.norm(whole,axis=1)>1e-12)
            mapped=data['mapping'][sup['affected']]
            if set(influenced)!=set(mapped):raise RuntimeError('Captured evaluated patch misses true affected subdivision supports')
            stencil_error=float(np.max(np.abs(whole[data['mapping']]-moved)))
            if stencil_error>1e-12:raise RuntimeError('Sparse and independent full subdivision operators differ')
            fit['linearSupportVerification']={'allAffectedL1PointsCaptured':True,'actualAffectedVertices':len(influenced),
                'sparseVersusFullOperatorMaximumError':stencil_error,
                'affectedCoordinateReconstructionMaximumError':float(data['mappingErrors'][sup['affected']].max())}
            return fit
        newG,newb,newevidence=predicted_constraints(data,poses,sup,fit,checks)
        if not len(newb):raise RuntimeError('Predicted contacts cannot produce local constraints')
        G=np.concatenate((G,newG));b=np.concatenate((b,newb));evidence.extend(newevidence)
    raise RuntimeError('Local support fit did not clear predicted surfaces: '+json.dumps(rounds))


def broad_pairs(core,hand_triangles,tolerance=1e-6):
    lo,hi=hand_triangles.min(1),hand_triangles.max(1)
    selected=np.flatnonzero(np.all(hi>=core.min((0,1))-tolerance,axis=1)&np.all(lo<=core.max((0,1))+tolerance,axis=1))
    lo,hi=lo[selected],hi[selected];pairs=[]
    for i,t in enumerate(core):
        hits=np.flatnonzero(np.all(hi>=t.min(0)-tolerance,axis=1)&np.all(lo<=t.max(0)+tolerance,axis=1))
        pairs.extend((i,int(selected[h])) for h in hits)
    if len(pairs)>100000:raise RuntimeError('Unexpected offline local broadphase size')
    return np.asarray(pairs,np.int32).reshape(-1,2)


def offline_contacts(data,poses,fit):
    api=runpy.run_path(str(HERE/'hand_wrist_surface_contact_r11.py'));result=[]
    rest=data['currentRest'];delta=fit['evaluatedDelta']
    for pose in poses:
        triangles=data['observed'][pose['prefix']+'_body_triangles']
        hand_triangles=pose['hand'][pose['handTriangles']]
        row={'pose':pose['pose']}
        for name,movement in (('reconstructedV5',np.zeros_like(delta)),('predictedCandidate',delta)):
            actual=pose['body']+np.einsum('nij,nj->ni',pose['deformation'],movement)
            core,original,domain=api['clipped_core']({'vertices':rest+movement},{'vertices':actual,'triangles':triangles},'R')
            pairs=broad_pairs(core,hand_triangles,20e-6)
            metrics=api['triangle_pair_metrics'](core[pairs[:,0]],hand_triangles[pairs[:,1]])
            hits=pairs[metrics['contactWithinTolerance']]
            row[name]={'coreTriangles':len(core),'broadphasePairs':len(pairs),
                'intersections':int(metrics['intersects'].sum()),'contactsAt1Micron':int(metrics['contactWithinTolerance'].sum()),
                'minimumBroadphaseDistance':float(metrics['distance'].min()),
                'actualGeometryEvaluation':False,
                '_pairs':[{'bodyLocalIds':triangles[original[c]].tolist(),'handTriangle':int(h)} for c,h in hits]}
        result.append(row)
    return result


def build(data,poses,fit,output_relative):
    output=path(output_relative);archive=output.with_suffix('.npz')
    if output.exists() or archive.exists():raise FileExistsError('Preserve previous wrist plan')
    arrays={k:v.copy() for k,v in data['arrays'].items()};old_delta=arrays['R_delta'].copy()
    arrays['R_delta']=fit['newDelta'].astype(np.float32)
    extra=arrays['R_delta'].astype(np.float64)-old_delta.astype(np.float64)
    affected=np.flatnonzero(np.linalg.norm(extra,axis=1)>1e-10)
    if not np.array_equal(arrays['R_ids'][affected],fit['support']['ids']):raise RuntimeError('Quantized correction leaves exact 19-point support')
    if np.any(extra[:,0]!=0) or np.linalg.norm(extra,axis=1).max()>.000825001:
        raise RuntimeError('Measured support correction exceeds precise 0.825 mm YZ guard')
    if np.linalg.norm(arrays['R_delta'],axis=1).max()>.0045:raise RuntimeError('Original total wrist bound exceeded')
    metrics=fit['triangleMetrics']
    if metrics['flippedTriangles'] or metrics['edgeLengthRatioRange'][0]<.4:raise RuntimeError('Original raw shape guard failed')
    if any(not np.array_equal(arrays[k],data['arrays'][k]) for k in arrays if k.startswith('L_')):raise RuntimeError('Left changed')
    ids=[];deltas=[]
    for side in ('L','R'):
        changed=np.flatnonzero(np.linalg.norm(arrays[side+'_delta'],axis=1)>1e-10)
        ids.extend(arrays[side+'_ids'][changed].tolist());deltas.extend(arrays[side+'_delta'][changed].tolist())
    arrays['changedIds']=np.asarray(ids,np.int32);arrays['delta']=np.asarray(deltas,np.float32)
    arrays['rightSurfaceFitIds']=arrays['R_ids'][affected]
    arrays['rightSurfaceFitExtra']=extra[affected].astype(np.float32)
    arrays['rightSurfaceFitEvaluatedIds']=data['observed']['relaxed_R_body_originalIds'][fit['support']['affected']]
    arrays['rightSurfaceFitStencils']=fit['support']['S'][fit['support']['affected']]
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as f:np.savez_compressed(f,**arrays)
    compact_checks=[{'pose':r['pose'],**{k:{a:b for a,b in v.items() if a!='_pairs'} for k,v in r.items() if k!='pose'}} for r in fit['offlineChecks']]
    result=dict(data['parent'])
    result.update({'helperSha256':sha(__file__),'dependencies':DEPS,'arrays':str(archive.relative_to(project_path())),
        'arraysSha256':sha(archive),'parentPlan':PARENT,'parentPlanSha256':PARENT_SHA,
        'parentArraysSha256':data['parent']['arraysSha256'],'totalChangedRawPoints':len(ids),
        'method':'Preserve v4; solve a smooth minimum-energy YZ correction only on the actual 19 raw cage supports of 9 intersecting wrist faces. Project actual hand triangles and use independently verified L1/skin operators; update local overlap constraints once without broadening support.',
        'rightSurfaceObservation':{'report':FAILURE,'reportSha256':FAILURE_SHA,'source':data['failure']['source'],
            'sourceSha256':data['failure']['sourceSha256'],'helperSha256':data['failure']['helperSha256'],
            'actualIntersections':[34,16,46],'completeDiagnosticPairs':True,
            'fullGeometryObservation':PROBE,'fullGeometryObservationSha256':PROBE_SHA,'fullGeometryArraysSha256':data['probe']['arraysSha256']},
        'rightSurfaceFit':{'supportIds':fit['support']['ids'].tolist(),'rawPoints':len(affected),
            'maximumAdditionalRawDisplacementMeters':float(np.linalg.norm(extra,axis=1).max()),
            'rawMaximumGuardMeters':.000825001,'leftAllArraysBitExact':True,'longitudinalCoordinatesUnchanged':True,
            'fitReserveMeters':20e-6,'fitReserveMeaning':'Numerical planning reserve only; does not replace the unchanged 1 micrometre actual continuous-surface contact gate.',
            'actualContactVertexMapping':data['actualMatchRows'],'linearSupportVerification':fit['linearSupportVerification'],
            'actualPoseTransferMaximumErrorMeters':max(p['actualContactTransferMaximumError'] for p in poses),
            'solver':fit['solver'],'constraints':fit['constraints'],'nonlinearConstraintRounds':fit['nonlinearConstraintRounds'],
            'offlinePrediction':compact_checks,
            'predictionLimit':'Offline deformation model, not Blender acceptance. Actual v5 intersection counts are reproduced; one flex pair is predicted within 1 micrometre while actual v5 reported 16 total contacts. Three real poses must run again unchanged.'},
        'actualPosesPassed':False,'appearanceAccepted':False,'savedBlend':False})
    result['sides']=dict(result['sides']);result['sides']['R']=dict(result['sides']['R'])
    result['sides']['R'].update({'triangleDiagnostics':metrics,
        'changedRawPoints':int((np.linalg.norm(arrays['R_delta'],axis=1)>1e-10).sum()),
        'maximumTotalWristKeyMeters':float(np.linalg.norm(arrays['R_delta'],axis=1).max()),
        'maximumAdditionalCorrectionMeters':float(np.linalg.norm(arrays['R_baselineMix']+arrays['R_delta']-arrays['R_integratedMixBefore'],axis=1).max())})
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return {'plan':str(output.relative_to(project_path())),'planSha256':sha(output),'arraysSha256':sha(archive),
        'rightRawSupportCount':len(affected),'maximumExtraMeters':result['rightSurfaceFit']['maximumAdditionalRawDisplacementMeters'],
        'offlinePredictedContacts':[r['predictedCandidate']['contactsAt1Micron'] for r in compact_checks]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analyze',action='store_true');parser.add_argument('--output');args=parser.parse_args()
    data=load();poses=pose_data(data);gaps=[]
    for pose in poses:
        contacts=next(s for s in pose['failure']['sides'] if s['side']=='R')['continuousSurfaceContact']['contactPairsForDiagnosis']
        values=[];normals=[]
        for c in contacts:
            poly,n=projected_overlap(np.asarray(c['bodyOriginalTriangleWorld']),np.asarray(c['handTriangleWorld']))
            values.extend(p['gap'] for p in poly);normals.append(n)
        gaps.append({'pose':pose['pose'],'minMaxProjectedOverlapGap':[float(min(values)),float(max(values))],
            'actualContactTransferError':pose['actualContactTransferMaximumError'],'normalBounds':[np.min(normals,0).tolist(),np.max(normals,0).tolist()]})
    fit=candidate(data,poses);contacts=fit['offlineChecks']
    if args.output:
        print(json.dumps(build(data,poses,fit,args.output),ensure_ascii=False));return
    print(json.dumps({'matchedActualContacts':data['actualMatchRows'],
        'localMapErrorsQuantiles':np.quantile(data['mappingErrors'],[0,.5,.9,.99,1]).tolist(),'projectedSurfaceGaps':gaps,
        'fit':{'supportCount':len(fit['support']['ids']),'supportIds':fit['support']['ids'].tolist(),
            'affectedEvaluatedVertices':len(fit['support']['affected']),
            'maximumAdditionalRawDisplacement':float(np.linalg.norm(fit['delta'],axis=1).max()),
            'maximumAdditionalEvaluatedDisplacement':float(np.linalg.norm(fit['evaluatedDelta'],axis=1).max()),
            'solver':fit['solver'],'rawTriangleGuard':fit['triangleMetrics']},'continuousSurfacePrediction':[
                {'pose':r['pose'],**{k:{a:b for a,b in v.items() if a!='_pairs'} for k,v in r.items() if k!='pose'}} for r in contacts],
            'nonlinearConstraintRounds':fit['nonlinearConstraintRounds']},indent=2))


if __name__=='__main__':main()
