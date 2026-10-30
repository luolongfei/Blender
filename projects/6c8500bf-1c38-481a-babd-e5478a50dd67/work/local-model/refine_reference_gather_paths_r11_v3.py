"""Local analytic-C1 ear connector refinement of the frozen v2 actual arrays.

This is an offline proposal. The existing contact GN must still be evaluated
and its displacement and complete final geometry checked in Blender.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--candidate',required=True);ap.add_argument('--output',required=True);ar=ap.parse_args()
 basepath=(P/ar.candidate).resolve();out=(P/ar.output).resolve();npz=out.with_suffix('.npz')
 for path in [basepath,out,npz]:path.relative_to((P/'work/reconstruction-r02').resolve())
 if out.exists() or npz.exists():raise RuntimeError('Independent unused output required')
 report=json.loads(basepath.read_text());basearray=P/report['arrays']
 if sha(basearray)!=report['arraysSha256']:raise RuntimeError('Frozen base arrays changed')
 arrays={k:v.copy() for k,v in np.load(basearray).items()};obs=json.loads((P/report['observation']).read_text());obsarray=P/obs['arrays'];ob=np.load(obsarray)
 if sha(P/report['observation'])!=report['observationSha256'] or sha(obsarray)!=report['observationArraysSha256']:raise RuntimeError('Actual head observation changed')
 plan=json.loads((P/report['tiePlan']).read_text())
 if sha(P/report['tiePlan'])!=report['tiePlanSha256']:raise RuntimeError('Rigid plan changed')
 lib=runpy.run_path(str(HERE/'rebuild_reference_gather_paths_r11_v2.py'));geom=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'));near=runpy.run_path(str(HERE/'analyze_reference_gather_transport_r11.py'))['oriented_near']
 unit=lib['unit'];turns=lib['turns'];bezier=lib['bezier'];resample=lib['resample'];closest=geom['closest_triangles']
 surface=geom['ApproxSurface']('head',ob,maximum=16000,neighbors=16);full=ob['head_world'][ob['head_triangles']];centers=full.mean(1)
 q=arrays['source_world'];target=arrays['target_world'];starts=arrays['source_starts'];lengths=arrays['source_lengths'];role=arrays['source_group']%256;ids=arrays['gather_curve_ids'];newtail={}
 for side,r in [('positive',40),('negative',41)]:
  tr=plan['transforms'][side];old=ob['tail_'+side+'_world'].reshape(2480,35,3);newtail[r]=(old-np.asarray(tr['originWorld']))@np.asarray(tr['rotationWorld']).T+np.asarray(tr['targetWorld'])
 fixes=[]
 for i in report['newAbruptCurvesAbove65']:
  j=int(np.flatnonzero(ids==i)[0]);s=int(starts[i]);n=int(lengths[i]);old=q[s:s+n].astype(float);current=target[s:s+n].astype(float);tail=newtail[int(role[i])][arrays['tail_curve_index'][j]];end=current[-3];d1=unit(tail[1]-tail[0]);anchor=int(arrays['preserved_upper_last_point'][j]);candidates=[]
  for k in range(max(2,anchor-4),anchor+1):
   p0=old[k];d0=unit(p0-old[k-1]);dist=float(np.linalg.norm(end-p0))
   for m0 in (.10,.20,.35,.50,.75,1.00,1.25):
    for m1 in (.25,.50,.75,1.00,1.30,1.60):
     dense=bezier(p0,p0+d0*dist*m0,end-d1*dist*m1,end,np.linspace(0,1,160));candidate=old.copy();candidate[k:-2]=resample(dense,n-k-2);candidate[-3:]=current[-3:];candidate[:k+1]=old[:k+1]
     angle=float(turns(candidate)[k-1:].max());sample=np.concatenate([candidate[k+1:],(candidate[k:-1]+candidate[k+1:])*.5]);candidates.append((angle,k,m0,m1,candidate,sample))
  candidates.sort(key=lambda row:row[0]);candidates=candidates[:40]
  samples=np.concatenate([c[5] for c in candidates]);coarse,_=near(samples,surface.triangles,surface.triangles.mean(1),16,closest);index=0;ranked=[]
  for c in candidates:
   count=len(c[5]);gap=float(coarse[index:index+count].min());index+=count;ranked.append((max(0.,-.0005-gap),c[0],gap,c))
  ranked.sort(key=lambda row:row[:2]);short=ranked[:8];samples=np.concatenate([x[3][5] for x in short]);refined,_=near(samples,full,centers,32,closest,batch=12);index=0;actual=[]
  for _,_,_,c in short:
   count=len(c[5]);gap=float(refined[index:index+count].min());index+=count;actual.append((max(0.,-.0005-gap),c[0],gap,c))
  actual.sort(key=lambda row:row[:2]);_,angle,gap,c=actual[0]
  target[s:s+n]=c[4];arrays['preserved_upper_last_point'][j]=c[1]
  fixes.append({'curve':int(i),'anchor':int(c[1]),'startHandleFraction':float(c[2]),'endHandleFraction':float(c[3]),'newConnectorMaximumTurnDegrees':angle,'actualTriangleNeighborhoodMinimumMeters':gap,'sampledControlsAndSegmentMidpoints':len(c[5])})
 print('LOCAL_C1_REFINEMENT '+json.dumps({'curves':len(fixes),'maxNewLocalTurn':max(x['newConnectorMaximumTurnDegrees'] for x in fixes),'minLocalGap':min(x['actualTriangleNeighborhoodMinimumMeters'] for x in fixes)}),flush=True)
 samples=[];owners=[];offsets=[];newjoint=[];oldmax=[];newmax=[];bad65=[];bad95=[];protected=True;mask=np.zeros(len(q),bool);enderror=[];tangent=[]
 for j,i in enumerate(ids):
  s=int(starts[i]);n=int(lengths[i]);old=q[s:s+n].astype(float);new=target[s:s+n].astype(float);anchor=int(arrays['preserved_upper_last_point'][j]);ot=turns(old);nt=turns(new)
  changed=np.linalg.norm(new-old,axis=1)>1e-7;changed_joint=changed[:-2]|changed[1:-1]|changed[2:];oldmax.append(float(ot.max()));newmax.append(float(nt.max()));newjoint.append(float(nt[changed_joint].max()) if changed_joint.any() else 0.)
  if np.any(changed_joint&(nt>65)&(nt>ot+20)):bad65.append(int(i))
  if np.any(changed_joint&(nt>95)&(nt>ot+20)):bad95.append(int(i))
  protected &= np.array_equal(q[s:s+anchor+1],target[s:s+anchor+1]);mask[s:s+n]=True
  tail=newtail[int(role[i])][arrays['tail_curve_index'][j]];enderror.append(float(np.linalg.norm(new[-1]-tail[0])));tangent.append(float(unit(new[-1]-new[-2])@unit(tail[1]-tail[0])))
  which=np.unique(np.round(np.linspace(anchor+1,n-1,7)).astype(int));samples.extend(new[which]);owners.extend([j]*len(which));offsets.extend(which)
 samples=np.asarray(samples);owners=np.asarray(owners,np.int32);offsets=np.asarray(offsets,np.int32);signed,_=near(samples,surface.triangles,surface.triangles.mean(1),16,closest)
 worst=np.argsort(signed)[:96];refined,_=near(samples[worst],full,centers,32,closest,batch=12);original=q[starts[ids[owners[worst]]]+offsets[worst]];original_refined,_=near(original,full,centers,32,closest,batch=12)
 untouched=np.array_equal(q[~mask],target[~mask]);errors=[]
 if not protected or not untouched:errors.append('Protected source positions changed')
 if bad95:errors.append('New same-joint reversal above 95 degrees and 20 degrees increase')
 if min(float(signed.min()),float(refined.min()))<-.015:errors.append('Offline pre-contact intrusion already exceeds installer 15 mm displacement limit')
 # status proposed describes a useful candidate requiring the native contact
 # stage, never collision acceptance. The strict Blender installer remains
 # responsible for every post-contact curve point, interior and radius.
 report.update(schemaVersion=1,status='blocked' if errors else 'proposed',errors=errors,appearanceAccepted=False,productionCollisionAccepted=False,approximateOnly=True,actualGeometryNodesExecuted=False,
  helperSha256=sha(__file__),baseCandidate=str(basepath.relative_to(P)),baseCandidateSha256=sha(basepath),baseCandidateArraysSha256=sha(basearray),localC1Refinements=fixes,
  firstThreeGatherPointsExact=bool(protected),allRetainedUpperControlPointsExact=bool(protected),allNonGatherAndExcludedPointsExact=bool(untouched),
  headChangedCoarseSignedMetersP0P1P5P50=np.percentile(signed,[0,1,5,50]).tolist(),headRefinedWorstMinimumMeters=float(refined.min()),originalAtRefinedWorstMinimumMeters=float(original_refined.min()),
  headChangedCoarseSamplesBelowMinus1mm=int(np.sum(signed<-.001)),headRefinedWorstSamplesBelowMinus1mm=int(np.sum(refined<-.001)),
  oldTurnDegreesP50P95P99P100=np.percentile(oldmax,[50,95,99,100]).tolist(),newTurnDegreesP50P95P99P100=np.percentile(newmax,[50,95,99,100]).tolist(),newChangedJointTurnDegreesP50P95P99P100=np.percentile(newjoint,[50,95,99,100]).tolist(),newAbruptCurvesAbove65=bad65,newAbruptCurvesAbove95=bad95,
  maximumEndpointRoundoffMeters=max(enderror),minimumTangentDot=min(tangent),
  preContactCorrectionNeeded=bool(np.any(refined<0)),preContactCandidateRule='A proposed path may need the existing native head-contact correction. This offline label does not pass collision. Installer must measure actual correction <=15 mm and all final/native64 free controls and interiors >=-0.5 mm.',
  worstSamples=[{'curve':int(ids[owners[w]]),'point':int(offsets[w]),'targetWorld':samples[w].tolist(),'refinedSignedMeters':float(refined[k]),'originalSignedMeters':float(original_refined[k])} for k,w in enumerate(worst)],
  limitations=['Nearest-centroid actual triangle neighborhoods are approximate; sample includes local refined candidate midpoints, but not every final native spline interior','Existing native head contact has NOT run; final free points, interior samples, radii and <=15 mm correction remain mandatory','No anatomy, raw roots, semantic face curtains, tail rigid plan or material changed','Actual full front and quarter render still required for appearance'])
 arrays.update(target_world=target,sample_world=samples,sample_gather_index=owners,sample_point_offset=offsets,sample_signed_head=signed,refined_sample_index=worst,refined_signed_head=refined,refined_original_signed_head=original_refined,old_max_turn_degrees=np.asarray(oldmax),target_max_turn_degrees=np.asarray(newmax),target_max_changed_joint_degrees=np.asarray(newjoint))
 with npz.open('xb') as f:np.savez_compressed(f,**arrays)
 report.update(arrays=str(npz.relative_to(P)),arraysSha256=sha(npz))
 with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
 print('REFINED_RESULT '+json.dumps({'status':report['status'],'errors':errors,'newAbove95':len(bad95),'newAbove65':len(bad65),'changedJointP50P95P99P100':report['newChangedJointTurnDegreesP50P95P99P100'],'headChangedMin':float(refined.min()),'arraysSha256':report['arraysSha256']}),flush=True)
if __name__=='__main__':main()
