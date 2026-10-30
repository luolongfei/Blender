"""Topology-aware adult mandibular motion field, independent of tooth fitting.

Pure NumPy planning only. Uses the current complete 10582 point female cage,
real upper/lower oral face sets, lip arcs, lower jaw and ear landmarks. The
estimated hinge is explicit and must be checked from actual side/open views.
No Blender startup, source mutation, or promise of anatomical acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rotation_displacement(world,weights,pivot,angle_radians,translation):
    """Weighted angle, not a straight interpolation between two lip positions."""
    q=np.asarray(world,float)-np.asarray(pivot,float)
    angles=np.asarray(weights,float)*float(angle_radians);c=np.cos(angles);s=np.sin(angles)
    rotated=q.copy();rotated[:,1]=q[:,1]*c-q[:,2]*s;rotated[:,2]=q[:,1]*s+q[:,2]*c
    return rotated-q+np.asarray(weights)[:,None]*np.asarray(translation,float)


def plan(current_world):
    skin=api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']();qa=api('face_contact_audit_r11')
    v=np.asarray(current_world,float)
    if v.shape!=(10582,3) or not np.isfinite(v).all():raise RuntimeError('Complete finite current adult cage required')
    edges=donor['edges'];mouth=landmarks['mouth'];sets=lambda ids:skin['_vertices_of_groups'](donor,set(ids))
    upper_oral=sets({7});lower_oral=sets({8});jaw=sets({22});ears=sets({4,5});eyes=sets({2,3})
    opening=list(map(int,mouth['openingRimCandidate']['sourceVertexIndices']))
    l=min(opening,key=lambda i:v[i,0]);r=max(opening,key=lambda i:v[i,0]);upper_arc,lower_arc=qa['_split'](opening,v,l,r)
    lip_pins={}
    def pin_arcs(upper,lower):
        for arc,is_lower in ((upper,False),(lower,True)):
            points=v[arc];dist=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            near=np.minimum(dist,dist[-1]-dist);t=np.clip(near/.012,0.,1.);t=t*t*(3.-2.*t)
            values=.5+.5*t if is_lower else .5-.5*t
            for i,w in zip(arc,values):lip_pins.setdefault(int(i),[]).append(float(w))
    pin_arcs(upper_arc,lower_arc)
    # Each true vermilion row follows its own real ring; never a rectangular lip mask.
    for label in ('vermilionMidLoop','outerVermilionLoopCandidate'):
        ring=list(map(int,mouth[label]['sourceVertexIndices']));l2=min(ring,key=lambda i:v[i,0]);r2=max(ring,key=lambda i:v[i,0])
        u,d=qa['_split'](ring,v,l2,r2);pin_arcs(u,d)
    lip_center=v[opening].mean(0);chin_candidates=jaw&(abs(v[:,0])<.015)&(v[:,1]<lip_center[1]+.04)
    if not chin_candidates.any():raise RuntimeError('Actual central chin support missing')
    chin=float(v[chin_candidates,2].min())
    ear_rows=[]
    for side,group in (('L',4),('R',5)):
        ids=np.flatnonzero(sets({group}));points=v[ids];lo=points.min(0);hi=points.max(0)
        # Visible pre-auricular estimate, not a hidden condyle measurement.
        estimate=np.array([float(np.median(points[:,0])),float(lo[1]-.004),float(lo[2]+.56*(hi[2]-lo[2]))])
        ear_rows.append({'side':side,'boundsWorld':[lo.tolist(),hi.tolist()],'estimatedCondyleWorld':estimate.tolist()})
    pivot=np.mean([r['estimatedCondyleWorld'] for r in ear_rows],axis=0);pivot[0]=float(lip_center[0])
    if not .020<pivot[2]-lip_center[2]<.065 or not .045<pivot[1]-lip_center[1]<.13:raise RuntimeError('Inspect current anatomy before estimating jaw hinge')
    active=(sets({1,7,8,17,22}))&(v[:,2]>chin-.055)&(v[:,2]<lip_center[2]+.042)&(v[:,1]<pivot[1]+.018)
    fixed_zero=~active|ears|eyes|(upper_oral&(v[:,2]>lip_center[2]+.011))
    # Exact zero at the neck and posterior skull, with a harmonic soft-tissue transition.
    fixed_zero|=(v[:,2]<=chin-.045)|(v[:,1]>=pivot[1]+.014)
    fixed_one=lower_oral&~upper_oral&(v[:,2]<lip_center[2]-.009)
    fixed_one|=jaw&(v[:,2]<lip_center[2]-.011)&(v[:,2]>chin+.003)&(v[:,1]<lip_center[1]+.044)&(abs(v[:,0])<.035)
    fixed_one&=~fixed_zero
    lip_ids=np.asarray(sorted(lip_pins),np.int32);lip_values=np.asarray([np.mean(lip_pins[i]) for i in lip_ids])
    if fixed_zero[lip_ids].any():raise RuntimeError('Lip pins unexpectedly outside anatomical jaw region')
    fixed_one[lip_ids]=False
    free=active&~fixed_zero&~fixed_one;free[lip_ids]=False
    weight=np.zeros(len(v),float);weight[fixed_one]=1.
    weight[lip_ids]=lip_values
    length=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1)
    conductance=1./np.maximum(length,.00025)
    degree=np.bincount(edges.ravel(),weights=np.repeat(conductance,2),minlength=len(v))
    # Edge endpoints are interleaved in ravel, so the symmetric Jacobi accumulator
    # must use the same ordering. Positive weights keep the maximum principle.
    error=None
    for iteration in range(2500):
        accum=np.bincount(edges[:,0],weights=conductance*weight[edges[:,1]],minlength=len(v))
        accum+=np.bincount(edges[:,1],weights=conductance*weight[edges[:,0]],minlength=len(v))
        target=accum/np.maximum(degree,1e-20);candidate=weight.copy();candidate[free]=.85*target[free]+.15*weight[free]
        error=float(np.max(abs(candidate-weight)));weight=candidate
        if error<2e-8:break
    if error is None or error>1e-6:raise RuntimeError('Jaw harmonic field has not converged')
    if np.any(weight[fixed_zero]) or np.max(abs(weight[fixed_one]-1.))>1e-12 or weight.min()<0 or weight.max()>1.:raise RuntimeError('Jaw field violates exact anatomical pins')
    triangles=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        f=donor['corner_vertices'][start:start+count]
        if (weight[f]>1e-8).any():triangles.extend((f[0],f[i],f[i+1]) for i in range(1,len(f)-1))
    tri=np.asarray(triangles,np.int32);a=v[tri];na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);la=np.linalg.norm(na,axis=1);valid=la>1e-11
    checks=[]
    for degrees in (4.5,9.,13.5,18.):
        t=degrees/18.;translation=np.array([0.,-.0025,-.0006])*t*t
        delta=rotation_displacement(v,weight,pivot,np.deg2rad(degrees),translation)
        b=(v+delta)[tri];nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);lb=np.linalg.norm(nb,axis=1)
        cosine=np.sum(na[valid]*nb[valid],axis=1)/(la[valid]*lb[valid])
        flipped=np.flatnonzero(cosine<=0)
        checks.append({'degrees':degrees,'translationWorld':translation.tolist(),'maximumDisplacementMeters':float(np.linalg.norm(delta,axis=1).max()),
                       'minimumNormalCosine':float(cosine.min()),'nonpositiveNormalCount':len(flipped),
                       'areaRatioP01P50P99':np.percentile(lb[valid]/la[valid],[1,50,99]).tolist(),
                       'openingLowerCenterWorld':(v+delta)[lower_arc[len(lower_arc)//2]].tolist()})
    report={'provenance':provenance,'method':'Positive inverse-edge-length harmonic mandibular field using actual oral and lip topology; weighted angle about an explicit preauricular estimate.',
            'earLandmarks':ear_rows,'estimatedPivotWorld':pivot.tolist(),'pivotIsExternalEstimate':True,
            'maximumAngleDegrees':18.,'maximumCondyleTranslationWorld':[0.,-.0025,-.0006],
            'rigidLowerSeeds':int(fixed_one.sum()),'fixedUpperAndNeckSeeds':int(fixed_zero.sum()),'freeTransitionPoints':int(free.sum()),
            'nonzeroPoints':int(np.count_nonzero(weight>1e-8)),'iterations':iteration+1,'maximumIterationDelta':error,
            'centralUpperLipAndPalatePinned':True,'lowerLipInteriorRigid':True,'mouthCornersMoveHalfJawAngle':True,
            'eyesEarsLowerNeckPinned':True,'submentalUpperThroatTransitionMeters':.045,
            'rawCageOpeningChecks':checks,'blenderApplied':False,'appearanceAccepted':False,
            'limits':['The external ear-derived hinge is a modelling estimate, not a measured skeletal condyle.',
                      'Actual Multires oral thickness, teeth clearance, jaw/neck silhouette and intermediate/composite expressions still require Blender evaluation and images.',
                      'No oral donor has yet been fitted; this field is not a functioning teeth or speech system.']}
    return {'weights':weight.astype(np.float32),'pivot_world':pivot,'rigid_ids':np.flatnonzero(fixed_one),'zero_ids':np.flatnonzero(fixed_zero),
            'transition_ids':np.flatnonzero(free),'reference_world':v,'check_triangles':tri},report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();out=(P/a.output).resolve();out.relative_to((P/'work/reconstruction-r02').resolve());npz=out.with_suffix('.npz')
    if out.exists() or npz.exists():raise RuntimeError('Retain existing plan')
    observed=P/'work/reconstruction-r02/adult-head-bind-observation-r11.npz';metadata=json.loads(observed.with_suffix('.json').read_text(encoding='utf8'))
    with np.load(observed) as n:arrays,report=plan(n['head_cage_current_world'])
    report.update(observation=observed.relative_to(P).as_posix(),observationSha256=sha(observed),observedSource=metadata['source'],observedSourceSha256=metadata['sourceSha256'],helperSha256=sha(__file__))
    np.savez_compressed(npz,**arrays);report['arrays']=npz.relative_to(P).as_posix();report['arraysSha256']=sha(npz)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'output':str(out.relative_to(P)),'checks':report['rawCageOpeningChecks'],'nonzeroPoints':report['nonzeroPoints']}))


if __name__=='__main__':main()
