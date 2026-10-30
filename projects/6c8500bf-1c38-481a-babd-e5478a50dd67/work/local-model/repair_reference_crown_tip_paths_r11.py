"""Offline repair of fourteen observed, previously penetrating side-fringe tips.

Uses only the real failed Blender output and its current head surface. No
Blender import, new observation, or mutation of a frozen input is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
IDS=np.array([2071,2072,2183,2184,3474,3478,3509,3563,3590,3604,3607,21455,21845,21846],np.int32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(q,count=128):
    """Blender's measured clamped Catmull resolution-12, arc-length resample."""
    t=np.arange(12,dtype=float)/12.;t=t[:,None]
    pieces=[]
    for i in range(len(q)-1):
        a,b,c,d=q[max(0,i-1)],q[i],q[i+1],q[min(len(q)-1,i+2)]
        pieces.append(.5*((2*b)+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
    dense=np.concatenate(pieces+[q[-1:]])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    at=np.linspace(0,arc[-1],count)
    return np.column_stack([np.interp(at,arc,dense[:,j]) for j in range(3)])


def clearance(surface,q,r):
    result=[]
    for p,rad in zip(q,np.broadcast_to(r,len(q))):
        near,normal,_,_=surface.find_nearest(p)
        result.append(float((p-near)@normal-rad))
    return np.asarray(result)


def check(surface,q,r,tail_only=False):
    nr=native(q)
    points=np.concatenate((q,(q[:-1]+q[1:])*.5,nr,(nr[:-1]+nr[1:])*.5))
    if tail_only:
        # Everything below the p13 elevation contains every changed spline span.
        points=points[points[:,2]<q[13,2]+.002]
    gaps=clearance(surface,points,float(np.max(r)))
    return {'minimumSurfaceMeters':float(gaps.min()),'samples':len(gaps),
            'belowOriginalLimit':int((gaps<-.0005).sum()),'belowDesignMargin':int((gaps<.00010).sum())}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    w=P/'work/reconstruction-r02';out=P/args.output
    if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('Independent unused output required')
    actual_path=w/'reference-adult-crown-flow-r11.crown-flow.json'
    actual_record=json.loads(actual_path.read_text('utf8'));actual_npz=P/actual_record['arrays'].replace('\\','/')
    if sha(actual_npz)!=actual_record['arraysSha256']:raise RuntimeError('Actual failure arrays changed')
    with np.load(actual_npz) as z:a={k:z[k] for k in z.files}
    plan_path=w/'reference-crown-flow-actual-r11.json';plan_record=json.loads(plan_path.read_text('utf8'))
    with np.load(plan_path.with_suffix('.npz')) as z:plan={k:z[k] for k in z.files}
    head=a['crown-source-head_world'];tri=a['crown-source-head_triangles']
    surface=runpy.run_path(str(HERE/'collar_continuous_offline_r11.py'))['Surface'](head,tri)
    starts=a['after-final_starts'];lengths=a['after-final_lengths'];groups=a['after-final_group']
    final=a['after-final_world'].astype(float);source=a['before-final_world'].astype(float)
    final_radius=a['after-final_radius'];native_ids=a['native-final_original_curve_ids']
    exact_native={int(cid):a['native-final_world'][j*128:(j+1)*128] for j,cid in enumerate(native_ids)}
    rows=[];errors=[];patch=np.zeros(len(plan['source_post_tie_world']),bool)
    desired=plan['target_world'].copy().astype(float)
    predicted=final.copy()
    for cid in IDS:
        s=int(starts[cid]);n=int(lengths[cid]);ps=int(plan['starts'][cid])
        old=final[s:s+n].copy();rad=final_radius[s:s+n]
        if n!=20 or groups[cid]!=17:raise RuntimeError('Only fourteen measured twenty-point role-17 side fringes are authorized')
        if not np.array_equal(source[s+17:s+20],old[17:20]):raise RuntimeError('The known old tip penetration is not unchanged')
        native_error=float(np.linalg.norm(native(old)-exact_native[int(cid)],axis=1).max())
        if native_error>2e-6:raise RuntimeError('Native model differs from actual Blender by more than 2 micrometres')
        baseline=check(surface,old,rad,True)
        best=None;attempts=[]
        # Small lateral sweep clears the real ear fold while keeping the original
        # descending tip direction. A smooth ramp starts after p14, never a kink.
        weights=np.zeros(n);weights[15:]=[.2098765432,.7901234568,1.,1.,1.]
        for amplitude in np.arange(.0005,.01001,.0005):
            candidate=old.copy();candidate[:,0]+=weights*amplitude
            result=check(surface,candidate,rad,True)
            attempts.append({'lateralMeters':float(amplitude),**result})
            if result['minimumSurfaceMeters']>=.00010:
                best=candidate;break
        if best is None:
            errors.append('No bounded smooth ear-tip repair for curve '+str(cid));rows.append({'curve':int(cid),'baseline':baseline,'attempts':attempts});continue
        desired[ps+15:ps+n]=best[15:];patch[ps+15:ps+n]=True
        predicted[s+15:s+n]=best[15:]
        turns=lambda q:np.degrees(np.arccos(np.clip(np.sum(np.diff(q,axis=0)[:-1]*np.diff(q,axis=0)[1:],axis=1)/(np.linalg.norm(np.diff(q,axis=0)[:-1],axis=1)*np.linalg.norm(np.diff(q,axis=0)[1:],axis=1)),-1,1)))
        new_turn=turns(best);old_turn=turns(old)
        if np.any((new_turn>95)&(new_turn>old_turn+20)):errors.append('Introduced fold in curve '+str(cid))
        full=check(surface,best,rad)
        if full['belowOriginalLimit']:errors.append('Full repaired curve still penetrates: '+str(cid))
        rows.append({'curve':int(cid),'role':17,'modifiedWithinCurve':[15,16,17,18,19],
                     'originalLastThreeExactlyUnchangedBeforeRepair':True,'nativeReconstructionMaxErrorMeters':native_error,
                     'baselineTail':baseline,'candidateFullPath':full,'lateralAmplitudeMeters':float(amplitude),
                     'maximumActualFinalDisplacementMeters':float(np.linalg.norm(best-old,axis=1).max()),
                     'beforeWorld':old[15:].tolist(),'afterWorld':best[15:].tolist(),
                     'beforeArcMeters':float(np.linalg.norm(np.diff(old,axis=0),axis=1).sum()),
                     'afterArcMeters':float(np.linalg.norm(np.diff(best,axis=0),axis=1).sum()),
                     'beforeMaxControlTurnDegrees':float(old_turn.max()),'afterMaxControlTurnDegrees':float(new_turn.max()),
                     'attempts':attempts})
        print('TIP_REPAIR '+json.dumps({k:v for k,v in rows[-1].items() if k not in ('beforeWorld','afterWorld','attempts')}),flush=True)
    output={**plan,'target_world':desired,'offset_world':desired-plan['source_post_tie_world'],
            'target_point_mask':plan['target_point_mask']|patch,'tip_repair_point_mask':patch,
            'tip_repair_curve_ids':IDS,'predicted_final_world':predicted}
    # The former mask included terminal points of all roles; role-17 tips are
    # not a tie connector. Only the explicitly repaired points are excepted.
    output['connector_protected_point_mask']=plan['connector_protected_point_mask']&~patch
    record={**plan_record,'status':'actual-source-proposal' if not errors else 'rejected-offline-repair',
            'basePlan':str(plan_path.relative_to(P)),'basePlanSha256':sha(plan_path),
            'actualFailure':str(actual_path.relative_to(P)),'actualFailureSha256':sha(actual_path),
            'repairHelperSha256':sha(__file__),'repairScope':rows,'errors':errors,
            'scopeMeaning':'The same calibrated crown field plus only p15–19 of fourteen measured role-17 side-fringe tips. All other terminal controls, real gather connectors, side curtains, independent wisps and tails remain unchanged. Current Blender checks still mandatory.',
            'appearanceAccepted':False,'actualModifiedBlenderExecuted':False}
    with out.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**output)
    record.update(arrays=str(out.with_suffix('.npz').relative_to(P)),arraysSha256=sha(out.with_suffix('.npz')))
    with out.open('x',encoding='utf8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print('TIP_REPAIR_RESULT '+json.dumps({'errors':errors,'reportSha256':sha(out)}),flush=True)


if __name__=='__main__':main()
