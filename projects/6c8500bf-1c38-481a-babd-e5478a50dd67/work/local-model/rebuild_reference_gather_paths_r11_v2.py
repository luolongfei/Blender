"""Rebuild actual gather connectors offline; never runs or saves Blender.

The 46 guide-13 temple-curtain descendants are semantic exclusions, even
though their inherited group currently says 41. All untouched points remain
bit-exact. Collision values are nearest-centroid triangle-neighborhood
estimates, not a substitute for the installer full evaluated BVH checks.
"""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path()
HERE = Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def unit(v): return v / max(float(np.linalg.norm(v)), 1e-14)
def bezier(p0,p1,p2,p3,t):
    t=np.asarray(t)[:,None];u=1-t
    return u*u*u*p0+3*u*u*t*p1+3*u*t*t*p2+t*t*t*p3
def resample(q,count):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
    at=np.linspace(0,arc[-1],count)
    return np.stack([np.interp(at,arc,q[:,i]) for i in range(3)],axis=1)
def turns(q):
    v=np.diff(q,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-15)
    return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))
def nearest_indices(points,targets):
    result=[];dist=[]
    for start in range(0,len(points),128):
        d=np.sum((points[start:start+128,None]-targets[None])**2,axis=2)
        ix=d.argmin(1);result.extend(ix);dist.extend(np.sqrt(d[np.arange(len(ix)),ix]))
    return np.asarray(result,np.int32),np.asarray(dist)

def connector(path,tail,side,radial=None):
    """Preserve upper path and first three; connect with matched tangents.

    Tail segments are 11--15 mm, much longer than gather segments. Copying
    their lengths upstream would reverse the nape flow. The last two gather
    intervals therefore retain a bounded gather-appropriate length along
    the exact first-tail-segment direction, without transporting old tips.
    """
    n=len(path);end=tail[0];d1=unit(tail[1]-tail[0])
    spacing=float(np.clip(np.median(np.linalg.norm(np.diff(path,axis=0),axis=1))*.65,.0015,.003))
    last=np.stack([end-d1*(2*spacing),end-d1*spacing,end])
    eligible=np.flatnonzero(path[:,2]>=last[0,2]+.018)
    anchor=min(max(2,int(eligible[-1]) if len(eligible) else 2),n-9)
    # Avoid retaining a lower bend just because an unrelated later point rises.
    while anchor>2 and path[anchor,2]<last[0,2]+.012: anchor-=1
    p0=path[anchor];p3=last[0];d0=unit(path[anchor]-path[anchor-1])
    distance=float(np.linalg.norm(p3-p0));m=max(.003,distance*.34)
    t=np.linspace(0,1,128)
    dense=bezier(p0,p0+d0*m,p3-d1*m,p3,t)
    middle_path=resample(dense,n-anchor-2)
    new=path.copy();new[anchor:-2]=middle_path;new[-3:]=last
    new[:anchor+1]=path[:anchor+1]
    def radial_min(candidate,k):
        if radial is None:return 1.
        values=[]
        for point in candidate[k+1:]:
            hit,normal=radial(point-np.array([0.,-.005,1.430]))
            if hit is not None:values.append(float(np.dot(point-hit,normal)))
        return min(values) if values else 0.
    # Optimize the analytic endpoint handles and start the connector earlier
    # when a short retained lower bend would consume the needed turning room.
    # No interior support is inserted. All candidates are C1 cubic paths.
    score=float(turns(new)[anchor-1:].max());clearance=radial_min(new,anchor)
    if score<=45. and clearance>=.0005:return new,anchor,spacing
    candidates=[]
    for k in range(max(2,anchor-4),anchor+1):
        p0=path[k];d0=unit(p0-path[k-1]);dist=float(np.linalg.norm(p3-p0))
        for m0 in (.10,.20,.35,.50):
            for m1 in (.25,.50,.75,1.00,1.30):
                dense=bezier(p0,p0+d0*dist*m0,p3-d1*dist*m1,p3,t)
                candidate=path.copy();candidate[k:-2]=resample(dense,n-k-2)
                candidate[-3:]=last;candidate[:k+1]=path[:k+1]
                angle=float(turns(candidate)[k-1:].max())
                # Slightly prefer retaining more upper path at similar bend.
                candidates.append((angle+.20*(anchor-k),angle,k,candidate))
    candidates.sort(key=lambda row:row[0])
    best=(max(0.,-.0005-clearance),score,anchor,new)
    for _,angle,k,candidate in candidates[:16]:
        gap=radial_min(candidate,k)
        item=(max(0.,-.0005-gap),angle,k,candidate)
        if item[:2]<best[:2]:best=item
        if gap>=.0005 and angle<60.:
            return candidate,k,spacing
    return best[3],best[2],spacing

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--groom-arrays',required=True)
    ap.add_argument('--semantic-review',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    paths=[(P/v).resolve() for v in (args.plan,args.groom_arrays,args.semantic_review,args.output)]
    plan_path,groom_path,semantic_path,out=paths;npz=out.with_suffix('.npz')
    for p in paths+[npz]:p.relative_to((P/'work/reconstruction-r02').resolve())
    if out.exists() or npz.exists():raise RuntimeError('Independent unused candidate outputs required')
    plan=json.loads(plan_path.read_text());obs_path=(P/plan['observation']).resolve();obs=json.loads(obs_path.read_text());obs_arrays=(P/obs['arrays']).resolve()
    if sha(obs_path)!=plan['observationSha256'] or sha(obs_arrays)!=plan['observationArraysSha256']:raise RuntimeError('Observation hash changed')
    a=np.load(obs_arrays);g=np.load(groom_path);q=g['reference-front-detail_world'];starts=g['reference-front-detail_starts'];lengths=g['reference-front-detail_lengths'];group=g['reference-front-detail_group'];role=group%256
    semantic=json.loads(semantic_path.read_text());exclude=np.asarray(semantic['suspectDenseCurveIds'],np.int32)
    if semantic['denseArraysSha256']!=sha(groom_path) or semantic['actualObservationSha256']!=sha(obs_arrays) or semantic['sourceSha256']!=obs['sourceSha256']:raise RuntimeError('Semantic exclusion evidence does not match actual source arrays')
    if len(exclude)!=46 or not np.all(role[exclude]==41):raise RuntimeError('Expected actual guide-13 semantic exclusions')
    ids=np.setdiff1d(np.flatnonzero(np.isin(role,[40,41])),exclude).astype(np.int32)
    target=q.copy();mapping=np.full(len(ids),-1,np.int32);anchor_indices=np.zeros(len(ids),np.int32);spacing=np.zeros(len(ids));distances=np.zeros(len(ids));newtail={}
    for side,r in [('positive',40),('negative',41)]:
        tr=plan['transforms'][side];old=a['tail_'+side+'_world'].reshape(2480,35,3)
        newtail[r]=(old-np.asarray(tr['originWorld']))@np.asarray(tr['rotationWorld']).T+np.asarray(tr['targetWorld'])
        selected=np.flatnonzero(role[ids]==r);tip=q[starts[ids[selected]]+lengths[ids[selected]]-1]
        mapping[selected],distances[selected]=nearest_indices(tip,old[:,0])
    radial_class=runpy.run_path(str(HERE/'reference_fringe_layout_v6_r11.py'))['OfflineRadialSurface']
    radial=radial_class(a['head_world'],a['head_triangles'])
    for j,i in enumerate(ids):
        s=int(starts[i]);n=int(lengths[i]);new,anchor,sp=connector(q[s:s+n].astype(float),newtail[int(role[i])][mapping[j]],1 if role[i]==40 else -1,radial)
        target[s:s+n]=new;anchor_indices[j]=anchor;spacing[j]=sp
        if j and j%3000==0:print('REBUILD_C1_PROGRESS '+str(j),flush=True)
    lib=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'))
    near=runpy.run_path(str(HERE/'analyze_reference_gather_transport_r11.py'))['oriented_near']
    surface=lib['ApproxSurface']('head',a,maximum=16000,neighbors=16)
    def check(target):
        sample=[];owners=[];offsets=[]
        for j,i in enumerate(ids):
            s=int(starts[i]);n=int(lengths[i]);k=np.unique(np.round(np.linspace(anchor_indices[j],n-1,7)).astype(int))
            sample.extend(target[s+k]);owners.extend([j]*len(k));offsets.extend(k)
        sample=np.asarray(sample);signed,distance=near(sample,surface.triangles,surface.triangles.mean(1),16,lib['closest_triangles'])
        return sample,np.asarray(owners,np.int32),np.asarray(offsets,np.int32),signed
    sample,owners,offsets,signed=check(target)
    fallback=np.empty(0,np.int32)
    sample_original=q[starts[ids[owners]]+offsets]
    changed_samples=np.linalg.norm(sample-sample_original,axis=1)>1e-7
    changed_indices=np.flatnonzero(changed_samples)
    worst=changed_indices[np.argsort(signed[changed_indices])[:96]];full=a['head_world'][a['head_triangles']]
    refined,_=near(sample[worst],full,full.mean(1),32,lib['closest_triangles'],batch=12)
    original_refined,_=near(q[starts[ids[owners[worst]]]+offsets[worst]],full,full.mean(1),32,lib['closest_triangles'],batch=12)
    oldturn=[];newturn=[];maxmoves=[];endpoint_error=[];tangent_dot=[];first3=True;new_joint_max=[];joint65=[];joint95=[]
    changed_mask=np.zeros(len(q),bool)
    for j,i in enumerate(ids):
        s=int(starts[i]);n=int(lengths[i]);old=q[s:s+n];new=target[s:s+n];tail=newtail[int(role[i])][mapping[j]]
        ot=turns(old.astype(float));nt=turns(new.astype(float));oldturn.append(float(ot.max()));newturn.append(float(nt.max()));maxmoves.append(float(np.linalg.norm(new-old,axis=1).max()))
        changed_joint=np.linalg.norm(new[1:-1]-old[1:-1],axis=1)>1e-7
        changed_joint|=np.linalg.norm(new[:-2]-old[:-2],axis=1)>1e-7
        changed_joint|=np.linalg.norm(new[2:]-old[2:],axis=1)>1e-7
        new_joint_max.append(float(nt[changed_joint].max()) if changed_joint.any() else 0.)
        joint65.append(bool(np.any(changed_joint&(nt>65)&(nt>ot+20))))
        joint95.append(bool(np.any(changed_joint&(nt>95)&(nt>ot+20))))
        endpoint_error.append(float(np.linalg.norm(new[-1]-tail[0])));tangent_dot.append(float(unit(new[-1]-new[-2])@unit(tail[1]-tail[0])))
        first3 &= np.array_equal(old[:3],new[:3]);changed_mask[s:s+n]=True
    oldturn=np.asarray(oldturn);newturn=np.asarray(newturn);errors=[]
    if not first3:errors.append('First three gather points changed')
    if not np.array_equal(q[~changed_mask],target[~changed_mask]):errors.append('Non-gather or excluded face curtain changed')
    if float(refined.min())<-.001:errors.append('Approximate refined head penetration exceeds one millimetre')
    if float(signed.min())<-.003:errors.append('Coarse head penetration exceeds three millimetres')
    if any(joint95):errors.append('Connector has newly abrupt same-joint turns above 95 degrees')
    report={'schemaVersion':1,'status':'blocked' if errors else 'proposed','errors':errors,'appearanceAccepted':False,'productionCollisionAccepted':False,'approximateOnly':True,'actualGeometryNodesExecuted':False,
        'tiePlan':str(plan_path.relative_to(P)),'tiePlanSha256':sha(plan_path),'sourceGroomArrays':str(groom_path.relative_to(P)),'sourceGroomArraysSha256':sha(groom_path),'source':obs['source'],'sourceSha256':obs['sourceSha256'],'observation':str(obs_path.relative_to(P)),'observationSha256':sha(obs_path),'observationArraysSha256':sha(obs_arrays),'helperSha256':sha(__file__),
        'semanticReview':str(semantic_path.relative_to(P)),'semanticReviewSha256':sha(semantic_path),'preservedFaceCurtain':{'curves':46,'rawGuide':13,'originalKind':'temple-curtain','reason':'v6 threshold fallback mislabeled inherited role 41; complete actual paths protected'},
        'gatherCurves':len(ids),'fallbackCorridorCurves':len(fallback),'targetTailMapping':'Nearest actual old same-side tail root, rigidly transported; exact target endpoint, first tail segment direction, 1.5 to 3 mm gather intervals',
        'firstThreeGatherPointsExact':bool(first3),'allNonGatherAndExcludedPointsExact':bool(np.array_equal(q[~changed_mask],target[~changed_mask])),
        'headCoarseSignedMetersP0P1P5P50':np.percentile(signed,[0,1,5,50]).tolist(),'headChangedCoarseSignedMetersP0P1P5P50':np.percentile(signed[changed_samples],[0,1,5,50]).tolist(),'headUnchangedCoarseMinimumMeters':float(signed[~changed_samples].min()),'headRefinedWorstMinimumMeters':float(refined.min()),'originalAtRefinedWorstMinimumMeters':float(original_refined.min()),
        'headCoarseSamplesBelowMinus1mm':int(np.sum(signed<-.001)),'headRefinedWorstSamplesBelowMinus1mm':int(np.sum(refined<-.001)),
        'oldTurnDegreesP50P95P99P100':np.percentile(oldturn,[50,95,99,100]).tolist(),'newTurnDegreesP50P95P99P100':np.percentile(newturn,[50,95,99,100]).tolist(),'newChangedJointTurnDegreesP50P95P99P100':np.percentile(new_joint_max,[50,95,99,100]).tolist(),'newAbruptCurvesAbove65':ids[np.asarray(joint65)].tolist(),'newAbruptCurvesAbove95':ids[np.asarray(joint95)].tolist(),
        'preservedUpperControlPointCountP0P50P100':np.percentile(anchor_indices+1,[0,50,100]).tolist(),'maximumEndpointRoundoffMeters':max(endpoint_error),'minimumTangentDot':min(tangent_dot),'maximumPointMovementMeters':max(maxmoves),
        'worstSamples':[{'curve':int(ids[owners[w]]),'point':int(offsets[w]),'targetWorld':sample[w].tolist(),'refinedSignedMeters':float(refined[k]),'originalSignedMeters':float(original_refined[k])} for k,w in enumerate(worst)],
        'limitations':['Sampled controls only; parent installer must inspect full evaluated curves, segment interiors and all collision meshes','Nearest-centroid neighborhoods of actual triangles; oriented sign is not watertight containment proof','No contact displacement was used to rescue connector geometry','Appearance requires actual front and quarter rendering']}
    with npz.open('xb') as f:np.savez_compressed(f,source_world=q,source_starts=starts,source_lengths=lengths,source_group=group,source_radius=g['reference-front-detail_radius'],target_world=target,gather_curve_ids=ids,tail_curve_index=mapping,exclude_curve_ids=exclude,preserved_face_curtain_curve_ids=exclude,preserved_upper_last_point=anchor_indices,connector_spacing=spacing,old_tip_to_tail_root_distance=distances,corridor_curve_ids=ids[fallback],sample_world=sample,sample_gather_index=owners,sample_point_offset=offsets,sample_signed_head=signed,refined_sample_index=worst,refined_signed_head=refined,refined_original_signed_head=original_refined,old_max_turn_degrees=oldturn,target_max_turn_degrees=newturn)
    report.update(arrays=str(npz.relative_to(P)),arraysSha256=sha(npz))
    with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print('REBUILD_RESULT '+json.dumps({k:report[k] for k in ('status','errors','gatherCurves','fallbackCorridorCurves','headRefinedWorstMinimumMeters','newTurnDegreesP50P95P99P100','arraysSha256')}),flush=True)
if __name__=='__main__':main()
