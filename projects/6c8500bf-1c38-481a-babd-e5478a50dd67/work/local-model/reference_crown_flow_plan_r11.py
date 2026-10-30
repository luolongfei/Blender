"""Deterministic crown-seam flow proposal on measured post-tie coordinates.

Pure ``plan`` never imports Blender or changes its inputs. The optional CLI
builds an offline proposal from the frozen v10e observation and v3 tie target.
The proposal must be recalibrated against the actual saved post-tie stage.
"""
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np

PARAMETERS=dict(strength=.90, neighbour_radius=.012, primary_arc=.015,
                recovery_arc=.045, maximum_displacement=.009,
                clearance=.00040, maximum_changed_turn_degrees=68.)

def smooth(x):
    x=np.clip(x,0.,1.);return x*x*x*(10.+x*(-15.+6.*x))

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-15)

def arc(q):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]
def turns(q):
    t=unit(np.diff(q,axis=0))
    return np.degrees(np.arccos(np.clip(np.sum(t[:-1]*t[1:],axis=1),-1.,1.)))

def initial(q,distance=.005):
    s=arc(q)
    return unit(np.array([np.interp(distance,s,q[:,k]) for k in range(3)])-q[0])

def catmull_samples(q,segment_ids,count=6):
    """Conservative uniform Catmull-Rom estimate, not native Blender proof."""
    points=[];owners=[]
    for j in segment_ids:
        a=q[max(j-1,0)];b=q[j];c=q[j+1];d=q[min(j+2,len(q)-1)]
        for t in np.arange(1,count)/count:
            points.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t));owners.append(j)
    return np.asarray(points),np.asarray(owners,np.int32)

def plan(points,starts,lengths,groups,excluded_curve_ids,surface,*,parameters=None,connector_first_protected=None):
    """Return target, report and masks for one immutable measured point order.

    ``surface.find_nearest(point)`` returns world hit, unit outward normal,
    triangle id and distance. Only positive-crown front/gather roots near their
    actual nearest-neighbour role boundary participate. Both flow families
    use one locally averaged tangent field; no roles or interpolation change.
    Root p0, all last-three controls and all excluded curves are exact.
    """
    par=dict(PARAMETERS);par.update(parameters or {})
    source=np.asarray(points);q=source.astype(np.float64);s=np.asarray(starts,np.int32);n=np.asarray(lengths,np.int32);groups=np.asarray(groups,np.int32);g=groups%256
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or not np.array_equal(s,np.r_[0,np.cumsum(n[:-1])]) or int(n.sum())!=len(q):raise ValueError('Dense measured contiguous curves required')
    excluded=np.zeros(len(n),bool);excluded[np.asarray(excluded_curve_ids,np.int32)]=True
    root=q[s];eligible=(root[:,0]>0.)&(root[:,0]<.075)&(root[:,1]>-.025)&(root[:,1]<.050)&(root[:,2]>1.475)&(~excluded)&((g<40)|(g==40))
    front=np.flatnonzero(eligible&(g<40));rear=np.flatnonzero(eligible&(g==40))
    if not len(front) or not len(rear):raise ValueError('Measured crown families absent')
    pairdist=np.linalg.norm(root[front,None]-root[None,rear],axis=2)
    nearest=np.full(len(n),np.inf);nearest[front]=pairdist.min(1);nearest[rear]=pairdist.min(0)
    candidates=np.flatnonzero(eligible&(nearest<par['neighbour_radius']))
    direction=np.zeros_like(root)
    for i in np.r_[front,rear]:direction[i]=initial(q[s[i]:s[i]+n[i]])
    protected=np.asarray(connector_first_protected,np.int32) if connector_first_protected is not None else n-3
    if protected.shape!=n.shape or np.any(protected<1) or np.any(protected>n-3):raise ValueError('Per-curve connector protection required within native controls')
    target=q.copy();rows=[];moved_ids=[];ranges=[];failures=[]
    for i in candidates:
        a=int(s[i]);count=int(n[i]);original=q[a:a+count];ar=arc(original)
        recovery=min(par['recovery_arc'],float(ar[protected[i]])-.002)
        if recovery<.027:continue
        x=float(root[i,0]);z=float(root[i,2])
        edge=float(smooth(x/.005)*smooth((.075-x)/.005)*smooth((z-1.475)/.010))
        spatial=(1.-(nearest[i]/par['neighbour_radius'])**2)**2
        amount=par['strength']*edge*spatial
        if amount<.002:continue
        hit,normal,_,_=surface.find_nearest(root[i]);normal=np.asarray(normal)
        fields=[]
        for family in (front,rear):
            dd=np.linalg.norm(root[family]-root[i],axis=1);ids=np.argsort(dd)[:8];weight=np.exp(-.5*(dd[ids]/.006)**2)
            fields.append(np.sum(direction[family[ids]]*weight[:,None],axis=0)/max(float(weight.sum()),1e-15))
        common=sum(fields);common-=normal*np.dot(common,normal)
        if np.linalg.norm(common)<.20:
            common=np.array([1.,0.,0.]);common-=normal*np.dot(common,normal)
        common=unit(common)
        own=direction[i]-normal*np.dot(direction[i],normal);own=unit(own)
        change=common-own
        offsets=np.flatnonzero((ar>0)&(ar<recovery)&(np.arange(count)<protected[i]))
        if not len(offsets):continue
        # A compact C2 window: initial tangent changes promptly, then returns
        # smoothly to the same source path without moving the far/end region.
        envelope=ar*(1.-smooth(ar/recovery))
        baseline_turn=turns(original);old_lengths=np.linalg.norm(np.diff(original,axis=0),axis=1)
        old_height={}
        for j in offsets:
            h,no,_,_=surface.find_nearest(original[j]);old_height[int(j)]=max(par['clearance'],float(np.dot(original[j]-h,no)))
        accepted=None;reason=None
        for scale in (1.,.80,.64,.50,.40,.30,.20):
            candidate=original.copy();candidate[offsets]+=change*(amount*scale*envelope[offsets,None])
            project_max=0.
            for j in offsets:
                h,no,_,_=surface.find_nearest(candidate[j]);projected=np.asarray(h)+np.asarray(no)*old_height[int(j)]
                project_max=max(project_max,float(np.linalg.norm(projected-candidate[j])));candidate[j]=projected
            difference=np.linalg.norm(candidate-original,axis=1);changed=difference>1e-10
            segment=np.flatnonzero(changed[:-1]|changed[1:]);joint=np.flatnonzero(changed[:-2]|changed[1:-1]|changed[2:])
            turn=turns(candidate);new_lengths=np.linalg.norm(np.diff(candidate,axis=0),axis=1);ratio=new_lengths/np.maximum(old_lengths,1e-15)
            cap=np.maximum(par['maximum_changed_turn_degrees'],baseline_turn[joint]+10.)
            if np.any(turn[joint]>cap):reason='curvature';continue
            if np.max(difference)>par['maximum_displacement'] or project_max>.004:reason='displacement';continue
            if np.any(ratio[segment]<.45) or np.any(ratio[segment]>1.85):reason='segment-length';continue
            accepted=(candidate,scale,project_max,segment,joint,turn,difference);break
        if accepted is None:
            failures.append({'curve':int(i),'reason':reason,'keptExact':True});continue
        candidate,scale,project_max,segment,joint,turn,difference=accepted
        target[a:a+count]=candidate;moved_ids.append(int(i));ranges.append(recovery)
        rows.append({'curve':int(i),'role':int(g[i]),'nearestOtherFamilyRootMeters':float(nearest[i]),
                     'effectiveStrength':float(amount*scale),'recoveryArcMeters':recovery,'changedControlOffsets':offsets.tolist(),
                     'protectedFromPointOffset':int(protected[i]),'protectedFromArcMeters':float(ar[protected[i]]),
                     'maximumPointDisplacementMeters':float(difference.max()),'maximumSurfaceReprojectionMeters':project_max,
                     'beforeChangedMaximumTurnDegrees':float(baseline_turn[joint].max()),'afterChangedMaximumTurnDegrees':float(turn[joint].max()),
                     'sourceInitialDirection':direction[i].tolist(),'desiredSharedDirection':common.tolist()})
    target=target.astype(source.dtype)
    roots_exact=np.array_equal(target[s],source[s]);end=np.concatenate([np.arange(int(a+c-3),int(a+c)) for a,c in zip(s,n)])
    moved_point=np.any(target!=source,axis=1);curve_mask=np.zeros(len(n),bool);curve_mask[moved_ids]=True
    excluded_mask=np.repeat(excluded,n)
    if not roots_exact or not np.array_equal(target[end],source[end]) or not np.array_equal(target[excluded_mask],source[excluded_mask]):raise RuntimeError('Pure-plan preservation failed')
    protected_mask=np.concatenate([np.arange(count)>=protect for count,protect in zip(n,protected)])
    if not np.array_equal(target[protected_mask],source[protected_mask]):raise RuntimeError('Post-tie connector changed')
    extra={'selected_curve_ids':np.asarray(moved_ids,np.int32),'changed_point_mask':moved_point,'nearest_other_family_root':nearest,
           'recovery_arc_per_selected_curve':np.asarray(ranges),'source_world':source.copy(),'target_world':target,
           'source_starts':s.copy(),'source_lengths':n.copy(),'source_group':groups.copy(),'excluded_curve_ids':np.asarray(excluded_curve_ids,np.int32),
           'source_post_tie_world':source.copy(),'offset_world':(target.astype(float)-source.astype(float)),
           'target_point_mask':moved_point.copy(),'starts':s.copy(),'lengths':n.copy(),'group':groups.copy(),
           'protected_from_point_offset':protected.copy(),'connector_protected_point_mask':protected_mask}
    report={'parameters':par,'candidateCurves':len(candidates),'changedCurves':len(moved_ids),'changedPoints':int(moved_point.sum()),
            'rootP0Exact':roots_exact,'allFinalThreeControlsExact':True,'allExcludedCurvesExact':True,
            'allOtherCurvesExact':bool(np.array_equal(target[~np.repeat(curve_mask,n)],source[~np.repeat(curve_mask,n)])),
            'allPostTieConnectorsExact':True,
            'rows':rows,'unchangedFallbacks':failures,'topologyAndOrderedRolesUnchanged':True,
            'primaryArcPurpose':'Primary change of direction occurs in the first 5 to 15 mm; coarse original controls require recovery through up to 45 mm.',
            'restorePolicy':'Bypass this independent post-tie layer or restore source_world exactly. No upstream guide, tie carrier, radius or role edits.',
            'nativeCurvatureAccepted':False,'actualSourceRecalibrationRequired':True,'actualRenderAccepted':False}
    return target,report,extra

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reference-crown-flow-proposal-r11');args=parser.parse_args()
    root=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());p=root/'projects/6c8500bf-1c38-481a-babd-e5478a50dd67';w=p/'work/reconstruction-r02'
    out=w/args.output
    if any(out.with_suffix(ext).exists() for ext in ('.json','.npz')):raise RuntimeError('Independent new outputs required')
    name='reference-gather-rebuild-v3-r11.npz';source=np.load(w/name);head=np.load(w/'reference-tie-alignment-r11-v2.npz')
    Surface=runpy.run_path(str(Path(__file__).with_name('collar_continuous_offline_r11.py')))['Surface']
    hv=head['head_world'];tri=head['head_triangles']
    surface=Surface(hv,tri)
    protect=source['source_lengths'].copy()-3
    # Preserve the last original upper control and every subsequent connector
    # point, leaving a 2 mm recovery margin before that observed boundary.
    protect[source['gather_curve_ids']]=np.minimum(protect[source['gather_curve_ids']],source['preserved_upper_last_point'])
    print('CROWN_FLOW_OFFLINE_PLAN_START',flush=True)
    target,report,arrays=plan(source['target_world'],source['source_starts'],source['source_lengths'],source['source_group'],source['preserved_face_curtain_curve_ids'],surface,connector_first_protected=protect)
    print('CROWN_FLOW_PLAN_DONE '+json.dumps({'curves':report['changedCurves'],'points':report['changedPoints']}),flush=True)
    with out.with_suffix('.pending-plan.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    report.update(status='offline-proposal-awaiting-current-post-tie-calibration',sourcePlan=name,
                  sourcePlanSha256=hashlib.sha256((w/name).read_bytes()).hexdigest(),
                  headObservation='reference-tie-alignment-r11-v2.npz',headObservationSha256=hashlib.sha256((w/'reference-tie-alignment-r11-v2.npz').read_bytes()).hexdigest(),
                  helperSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),savedBlend=False,
                  insertionPoint='After R11 reference gather tie transport and before R11 full head clearance')
    signed=[];sample_meta=[];turn_before=[];turn_after=[]
    for index,i in enumerate(arrays['selected_curve_ids']):
        if index%200==0:print('CROWN_FLOW_PATH_CHECK '+str(index)+'/'+str(len(arrays['selected_curve_ids'])),flush=True)
        a=int(arrays['source_starts'][i]);n=int(arrays['source_lengths'][i]);before=arrays['source_world'][a:a+n].astype(float);after=target[a:a+n].astype(float)
        changed=np.any(before!=after,axis=1);seg=np.flatnonzero(changed[:-1]|changed[1:]);controls=np.flatnonzero(changed)
        mids=(after[seg]+after[seg+1])*.5;native,owner=catmull_samples(after,seg)
        for kind,pts,ids in (('control',after[controls],controls),('segment-mid',mids,seg),('catmull-estimate',native,owner)):
            for point,j in zip(pts,ids):
                hit,no,face,_=surface.find_nearest(point);signed.append(float(np.dot(point-hit,no)));sample_meta.append((int(i),int(j),kind))
        joint=np.flatnonzero(changed[:-2]|changed[1:-1]|changed[2:]);turn_before.extend(turns(before)[joint]);turn_after.extend(turns(after)[joint])
    signed=np.asarray(signed);arrays['sample_signed_head']=signed;arrays['sample_curve_id']=np.asarray([x[0] for x in sample_meta],np.int32);arrays['sample_control_or_segment_offset']=np.asarray([x[1] for x in sample_meta],np.int32)
    arrays['sample_kind']=np.asarray([{'control':0,'segment-mid':1,'catmull-estimate':2}[x[2]] for x in sample_meta],np.int8)
    report['headSamples']={'samples':len(signed),'signedMetersP0P1P50P100':np.percentile(signed,[0,1,50,100]).tolist(),'belowMinusHalfMillimeter':int((signed<-.0005).sum()),
                           'worst':[{'curve':sample_meta[k][0],'offset':sample_meta[k][1],'kind':sample_meta[k][2],'signedMeters':float(signed[k])} for k in np.argsort(signed)[:20]],
                           'scope':'Changed controls, all changed-segment midpoints and uniform Catmull-Rom 5 interior samples per changed segment, against observed current crown triangles. Actual native curve sampling after saved tie installation remains mandatory.'}
    report['changedJointTurns']={'beforeDegreesP50P95P100':np.percentile(turn_before,[50,95,100]).tolist(),'afterDegreesP50P95P100':np.percentile(turn_after,[50,95,100]).tolist()}
    report['errors']=[] if not np.any(signed<-.0005) else ['Offline changed path has samples under -0.5 mm; do not install without refinement']
    with out.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    report['arrays']=str(out.with_suffix('.npz').relative_to(p));report['arraysSha256']=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest()
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('changedCurves','changedPoints','headSamples','changedJointTurns','errors')}))

if __name__=='__main__':main()
