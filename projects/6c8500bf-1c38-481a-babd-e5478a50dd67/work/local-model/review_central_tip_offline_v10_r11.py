"""Read-only actual-dense central fringe experiment; never imports Blender.

Outputs only independent review arrays/report. No production source is edited.
"""
from pathlib import Path
import argparse, hashlib, json, runpy, time
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'reconstruction-r02'
SOURCE=DATA/'reference-adult-front-groom-v9c-r11.groom-fringe-local.npz'
SURFACE=DATA/'front-groom-actual-v6-r11.npz'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def smooth(t):
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)

VARIANTS={
    'parent_proposal':{18:(-.017,.18,1.438,.55),19:(.002,.10,1.440,.35)},
    'feathered':{18:(-.016,.26,1.439,.65),19:(.0015,.16,1.440,.50)},
    'staggered':{18:(-.016,.26,1.438,.65),19:(.0015,.16,1.444,.50)},
}
MEDIANS={18:(-.01266,1.44493),19:(.00579,1.44004)}

def candidate(world,starts,lengths,roles,parameters):
    out=world.copy();rows=[]
    for i in np.flatnonzero(np.isin(roles,[18,19])):
        s=int(starts[i]);n=int(lengths[i]);p=world[s:s+n].astype(float);tip=p[-1].copy();r=int(roles[i])
        cx,wx,cz,wz=parameters[r];mx,mz=MEDIANS[r]
        target=tip.copy();target[0]=cx+wx*(tip[0]-mx);target[2]=cz+wz*(tip[2]-mz)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];factor=arc/arc[-1]
        weight=smooth((factor-.65)/.35)
        q=p+weight[:,None]*(target-tip)
        out[s:s+n]=q.astype(np.float32)
        oldvec=np.diff(p,axis=0);newvec=np.diff(q,axis=0)
        bend=np.degrees(np.arccos(np.clip(np.sum(oldvec*newvec,axis=1)/(np.maximum(np.linalg.norm(oldvec,axis=1)*np.linalg.norm(newvec,axis=1),1e-20)),-1,1)))
        rows.append((int(i),r,float(np.linalg.norm(target-tip)),float(bend.max())))
    return out,rows

def segment_eye(p,center,radius):
    a=p[:-1];ab=p[1:]-a
    t=np.clip(np.sum((center-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-20),0,1)
    return float(np.min(np.linalg.norm(a+t[:,None]*ab-center,axis=1))-radius)

def stats(world,starts,lengths,roles):
    tips=world[starts+lengths-1]
    return {str(r):{'curves':int(np.sum(roles==r)),'tipXYZPercentiles0_5_50_95_100':np.percentile(tips[roles==r],[0,5,50,95,100],axis=0).tolist()}
            for r in (14,15,18,19)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);a=parser.parse_args();output=DATA/a.output
    if output.exists() or output.with_suffix('.npz').exists():raise RuntimeError('Independent unused review outputs required')
    begun=time.time();source=np.load(SOURCE);surface=np.load(SURFACE)
    w=source['final-taper_world'];s=source['final-taper_starts'];n=source['final-taper_lengths'];roles=source['final-taper_group']%256
    selected=np.flatnonzero(np.isin(roles,[18,19]));pointrole=np.repeat(roles,n)
    Surface=runpy.run_path(str(HERE/'collar_continuous_offline_r11.py'))['Surface']
    # A conservative local patch contains the complete changed forehead region.
    # At least 25 mm margin to every changed point is verified below.
    v=surface['head_world'];tri=surface['head_triangles'];tp=v[tri]
    keep=(tp.max(1)[:,0]>-.080)&(tp.min(1)[:,0]<.060)&(tp.max(1)[:,2]>1.400)&(tp.min(1)[:,2]<1.530)&(tp.min(1)[:,1]<-.050)
    tree=Surface(v,tri[keep]);print('PATCH_READY '+str(int(keep.sum())),flush=True)
    sample=np.linspace(0,len(selected)-1,150).astype(int);baseline_diffs=[]
    for i in selected[sample]:
        ix=int(s[i]+n[i]-1);hit,normal,_,distance=tree.find_nearest(w[ix]);signed=float(np.dot(w[ix]-hit,normal))
        baseline_diffs.append(signed-float(source['final_signed_head_clearance'][ix]))
    report={'source':SOURCE.name,'sourceSha256':sha(SOURCE),'surface':SURFACE.name,'surfaceSha256':sha(SURFACE),'helperSha256':sha(Path(__file__)),
            'rendered':False,'savedBlend':False,'selectedRoles':[18,19],'selectedCurves':len(selected),'unchangedBoundaryRoles':[14,15],
            'surfaceBaselineTipChecks':len(sample),'surfaceBaselineSignedDifferenceAbsMaxMeters':float(np.max(np.abs(baseline_diffs))),
            'surfacePatchTriangles':int(keep.sum()),'original':stats(w,s,n,roles),'variants':{},
            'limitations':['Surface is the prior actual b8 head export; compared with saved actual v9c signed head distances before experiments. Final Blender current-source evaluation remains mandatory.',
                           'This is an endpoint-displacement field experiment on actual dense points, not a rendered appearance acceptance.']}
    if report['surfaceBaselineSignedDifferenceAbsMaxMeters']>.00005:raise RuntimeError('Prior skin export does not agree with actual saved local v9c clearance')
    arrays={'starts':s,'lengths':n,'roles':roles,'source_world':w}
    for name,parameters in VARIANTS.items():
        out,rows=candidate(w,s,n,roles,parameters);changed=np.flatnonzero(np.any(w!=out,axis=1))
        bad=[];clearance=[];eye_min={side:float('inf') for side in ('L','R')};max_segment=0.;max_joint=0.;bad_midpoints=[]
        for k,ix in enumerate(changed):
            hit,normal,_,distance=tree.find_nearest(out[ix]);signed=float(np.dot(out[ix]-hit,normal));clearance.append(signed)
            if signed<-.0005:bad.append(int(ix))
            if k and k%5000==0:print(name+' checked '+str(k),flush=True)
        for i in selected:
            p=out[s[i]:s[i]+n[i]].astype(float);old=w[s[i]:s[i]+n[i]]
            for side in ('L','R'):eye_min[side]=min(eye_min[side],segment_eye(p,surface['eye_'+side+'_center'],float(surface['eye_'+side+'_radius'])))
            vec=np.diff(p,axis=0);seg=np.linalg.norm(vec,axis=1);max_segment=max(max_segment,float(seg.max()))
            norms=vec/np.maximum(seg[:,None],1e-20);joint=np.degrees(np.arccos(np.clip(np.sum(norms[:-1]*norms[1:],axis=1),-1,1)));max_joint=max(max_joint,float(joint.max()))
            # Confirm changed segments at midpoint as well as stored points.
            active=np.any(old[:-1]!=p[:-1],axis=1)|np.any(old[1:]!=p[1:],axis=1)
            for j in np.flatnonzero(active):
                mid=(p[j]+p[j+1])*.5;hit,normal,_,distance=tree.find_nearest(mid);signed=float(np.dot(mid-hit,normal))
                if signed<-.0005:bad_midpoints.append([int(i),int(j),signed])
        rows=np.asarray(rows);arrays[name+'_world']=out
        item={'parameters':{str(k):list(v) for k,v in parameters.items()},'tipFormula':'x=cx+wx*(oldTipX-roleMedianX); z=cz+wz*(oldTipZ-roleMedianZ); y unchanged; delta*smoothstep((arcFraction-.65)/.35)',
              'changedPoints':len(changed),'rootPositionsPreservedExact':bool(np.array_equal(w[s],out[s])),
              'nonTargetPositionsPreservedExact':bool(np.array_equal(w[~np.isin(pointrole,[18,19])],out[~np.isin(pointrole,[18,19])])),
              'minimumChangedPointSkinClearanceMeters':float(min(clearance)),'skinPenetratingPointIndices':bad,'skinPenetratingSegmentMidpoints':bad_midpoints,
              'selectedSegmentEyeMinimumMeters':eye_min,'maxSelectedSegmentMeters':max_segment,'maxSelectedJointAngleDegrees':max_joint,
              'endDisplacementMetersP50P95Max':np.percentile(rows[:,2],[50,95,100]).tolist(),'segmentDirectionChangeDegreesP50P95Max':np.percentile(rows[:,3],[50,95,100]).tolist(),
              'roles':stats(out,s,n,roles)}
        report['variants'][name]=item
        print('RESULT '+name+' '+json.dumps({k:v for k,v in item.items() if k not in ['roles','skinPenetratingPointIndices','skinPenetratingSegmentMidpoints']}),flush=True)
    report['elapsedSeconds']=time.time()-begun
    with output.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    np.savez_compressed(output.with_suffix('.npz'),**arrays)
    print('REVIEW_DONE '+str(output),flush=True)

if __name__=='__main__':main()
