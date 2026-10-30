"""Offline warning probe for the proposed gather transport, never a GN proof."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def smooth5(t):
    t=np.clip(t,0.,1.);return t*t*t*(t*(t*6.-15.)+10.)
def turns(q):
    v=np.diff(q,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-15)
    return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1.,1.)))
def oriented_near(points,triangles,centers,neighbors,closest,batch=128):
    origin=centers.mean(0);c=centers-origin;norm=np.sum(c*c,axis=1);signed=[];distance=[]
    for start in range(0,len(points),batch):
        p=points[start:start+batch];q=p-origin
        d2=np.maximum(np.sum(q*q,axis=1)[:,None]+norm[None,:]-2*q@c.T,0.)
        k=min(neighbors,len(triangles));ids=np.argpartition(d2,k-1,axis=1)[:,:k]
        d,s=closest(p,triangles[ids]);signed.extend(s.tolist());distance.extend(d.tolist())
    return np.asarray(signed),np.asarray(distance)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--groom-arrays',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    plan_path=(P/args.plan).resolve();groom_path=(P/args.groom_arrays).resolve();out=(P/args.output).resolve();npz=out.with_suffix('.npz')
    for p in (plan_path,groom_path,out,npz):p.relative_to((P/'work/reconstruction-r02').resolve())
    if out.exists() or npz.exists():raise RuntimeError('Unused independent diagnostic outputs required')
    plan=json.loads(plan_path.read_text());observation_path=(P/plan['observation']).resolve();obs=json.loads(observation_path.read_text());actual_path=(P/obs['arrays']).resolve()
    if sha(observation_path)!=plan['observationSha256'] or sha(actual_path)!=plan['observationArraysSha256']:raise RuntimeError('Actual observation hashes changed')
    a=np.load(actual_path);g=np.load(groom_path);q=g['reference-front-detail_world'];starts=g['reference-front-detail_starts'];lengths=g['reference-front-detail_lengths'];groups=g['reference-front-detail_group'];roles=groups%256
    samples=[];old_samples=[];owners=[];offsets=[];sample_weights=[];rows=[]
    for i in np.flatnonzero(np.isin(roles,[40,41])):
        path=q[starts[i]:starts[i]+lengths[i]].astype(np.float64);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))];factor=arc/max(arc[-1],1e-15)
        weight=smooth5((factor-.45)/(.84-.45));weight[:3]=0.;weight[-3:]=1.
        tr=plan['transforms']['positive' if roles[i]==40 else 'negative'];origin=np.asarray(tr['originWorld']);target=np.asarray(tr['targetWorld']);rotation=np.asarray(tr['rotationWorld'])
        moved=(path-origin)@rotation.T+target;new=path+(moved-path)*weight[:,None]
        old_turn=turns(path);new_turn=turns(new);which=np.unique([int(np.argmin(abs(factor-v))) for v in (.45,.55,.65,.75,.84,.92,1.)]);which=which[which>=3]
        samples.extend(new[which]);old_samples.extend(path[which]);owners.extend([int(i)]*len(which));offsets.extend(which.tolist());sample_weights.extend(weight[which].tolist())
        rows.append({'curve':int(i),'role':int(roles[i]),'maximumOriginalTurnDegrees':float(old_turn.max()),'maximumSimulatedTurnDegrees':float(new_turn.max()),'maximumTurnIncreaseDegrees':float((new_turn-old_turn).max()),'maximumPointMovementMeters':float(np.linalg.norm(new-path,axis=1).max()),'rootsFirstThreeExact':bool(np.array_equal(path[:3],new[:3])),'lastThreeEqualRigidTransform':bool(np.array_equal(new[-3:],moved[-3:]))})
    samples=np.asarray(samples);old_samples=np.asarray(old_samples);lib=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'));surface=lib['ApproxSurface']('head',a,maximum=12000,neighbors=12)
    signed,distance=oriented_near(samples,surface.triangles,surface.triangles.mean(1),12,lib['closest_triangles'])
    old_signed,_=oriented_near(old_samples,surface.triangles,surface.triangles.mean(1),12,lib['closest_triangles'])
    # Refine the worst coarse samples against neighborhoods selected from ALL
    # actual head triangles. Still not a formal BVH nearest-surface proof.
    worst=np.argsort(signed)[:min(96,len(signed))];full=a['head_world'][a['head_triangles']]
    refined,refined_distance=oriented_near(samples[worst],full,full.mean(1),24,lib['closest_triangles'],batch=16)
    refined_old,_=oriented_near(old_samples[worst],full,full.mean(1),24,lib['closest_triangles'],batch=16)
    new_turns=np.asarray([r['maximumSimulatedTurnDegrees'] for r in rows]);increases=np.asarray([r['maximumTurnIncreaseDegrees'] for r in rows])
    abrupt=[r for r in rows if r['maximumSimulatedTurnDegrees']>60. and r['maximumTurnIncreaseDegrees']>15.]
    detailed=[{'sample':int(j),'curve':int(owners[j]),'pointOffset':int(offsets[j]),'weight':float(sample_weights[j]),'world':samples[j].tolist(),'coarseSignedMeters':float(signed[j]),'refinedSignedMeters':float(refined[k]),'refinedOriginalSignedMeters':float(refined_old[k])} for k,j in enumerate(worst)]
    report={'schemaVersion':1,'source':obs['source'],'sourceSha256':obs['sourceSha256'],'observation':str(observation_path.relative_to(P)),'observationSha256':sha(observation_path),'observationArraysSha256':sha(actual_path),'plan':str(plan_path.relative_to(P)),'planSha256':sha(plan_path),'groomArrays':str(groom_path.relative_to(P)),'groomArraysSha256':sha(groom_path),'helperSha256':sha(__file__),
        'status':'offline_warning_only','actualGeometryNodesExecuted':False,'appearanceAccepted':False,'gatherCurves':len(rows),'samplePoints':len(samples),
        'weightRule':'Quintic smootherstep over approximate control-polyline arc fraction .45 to .84; first three weights 0 and last three 1. Blender SplineParameter is NOT assumed numerically identical.',
        'coarseHeadSignedMetersP0P1P5P50':np.percentile(signed,[0,1,5,50]).tolist(),'coarseOriginalHeadSignedMetersP0P1P5P50':np.percentile(old_signed,[0,1,5,50]).tolist(),
        'coarseSamplesBelowMinus15mm':int(np.sum(signed<-.015)),'coarseSamplesBelowMinus5mm':int(np.sum(signed<-.005)),
        'refinedWorstSamples':detailed,'refinedMinimumMeters':float(refined.min()),'refinedSamplesBelowMinus15mm':int(np.sum(refined<-.015)),
        'maximumPerCurveTurnDegreesP50P95P99P100':np.percentile(new_turns,[50,95,99,100]).tolist(),'maximumPerCurveTurnIncreaseDegreesP50P95P99P100':np.percentile(increases,[50,95,99,100]).tolist(),
        'abruptCurvesCount':len(abrupt),'abruptCurves':sorted(abrupt,key=lambda r:r['maximumTurnIncreaseDegrees'],reverse=True)[:50],
        'allFirstThreeControlPointsExact':all(r['rootsFirstThreeExact'] for r in rows),'allLastThreeControlPointsRigid':all(r['lastThreeEqualRigidTransform'] for r in rows),
        'limitations':['Sampled control points; complete curve interpolation and segment interiors remain unchecked','Head distance uses nearest-centroid triangle neighborhoods, not exact all-triangle BVH','No post-contact correction applied, no clothing/gather evaluation, no dynamic acceptance']}
    with npz.open('xb') as f:np.savez_compressed(f,sample_world=samples,original_world=old_samples,curve_index=np.asarray(owners,np.int32),point_offset=np.asarray(offsets,np.int32),weight=np.asarray(sample_weights),coarse_signed=signed,coarse_original_signed=old_signed,refined_sample_index=worst,refined_signed=refined,refined_original_signed=refined_old)
    report.update(arrays=str(npz.relative_to(P)),arraysSha256=sha(npz))
    with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print('GATHER_OFFLINE_WARNING '+json.dumps({'curves':len(rows),'samples':len(samples),'coarseMin':float(signed.min()),'refinedMin':float(refined.min()),'abruptCurves':len(abrupt),'output':str(out.relative_to(P))}),flush=True)
if __name__=='__main__':main()
