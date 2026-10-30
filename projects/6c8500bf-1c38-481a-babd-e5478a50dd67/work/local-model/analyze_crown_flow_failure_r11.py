"""Diagnose actual crown-flow failures without modifying geometry or launching Blender."""
from pathlib import Path
import hashlib,json,runpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
import sys
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();W=P/'work/reconstruction-r02';H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_npz(p,keys=None):
 with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in (f.files if keys is None else keys)}
def native_basis(n,resolution=12):
 out=[]
 for i in range(n-1):
  for t in np.arange(resolution)/resolution:
   row=np.zeros(n);coef=[-.5*t+t*t-.5*t*t*t,1-2.5*t*t+1.5*t*t*t,.5*t+2*t*t-1.5*t*t*t,-.5*t*t+.5*t*t*t]
   for j,c in zip((max(0,i-1),i,i+1,min(n-1,i+2)),coef):row[j]+=c
   out.append(row)
 row=np.zeros(n);row[-1]=1;out.append(row);return np.asarray(out)
def native_model(q,count=128):
 basis=native_basis(len(q));points=basis@q;arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];at=np.linspace(0,arc[-1],count)
 B=np.stack([np.interp(at,arc,basis[:,i]) for i in range(len(q))],axis=1)
 return B@q,B,points

def main():
 out=W/'reference-crown-flow-failure-diagnostic-r11.json';payload=out.with_suffix('.npz')
 if out.exists() or payload.exists():raise RuntimeError('New diagnostic outputs required')
 failed=W/'reference-adult-crown-flow-r11.crown-flow.json';fr=json.loads(failed.read_text());fp=P/fr['arrays']
 assert sha(fp)==fr['arraysSha256']
 a=read_npz(fp);obs=W/'reference-post-tie-actual-r11.json';oj=json.loads(obs.read_text());op=P/oj['arrays'];assert sha(op)==oj['arraysSha256']
 o=read_npz(op,['head_world','head_triangles','native_final_world','native_final_starts','native_final_lengths','native_final_group','native_final_original_curve_ids'])
 plan_path=W/'reference-crown-flow-actual-r11.json';pj=json.loads(plan_path.read_text());pp=P/pj['arrays'];assert sha(pp)==pj['arraysSha256']
 p=read_npz(pp,['starts','lengths','target_point_mask','protected_from_point_offset','estimated_surface_curve_id','estimated_surface_control_or_segment_offset'])
 allids=set();witness=[];families={};native_ids=a['native-final_original_curve_ids']
 for family in ('control_midpoints','native_points','native_midpoints'):
  pref='tieproof_crown_selected_'+family;ix=a[pref+'_against_head_sample_index'];v=a[pref+'_against_head_signed_surface'];bad=np.flatnonzero((v<-.0005)&(a[pref+'_arc'][ix]>.001));rows=[]
  for k in bad:
   sample=int(ix[k]);ci=int(a[pref+'_curve'][sample]);i=int(native_ids[ci]) if family.startswith('native') else ci;allids.add(i)
   within=int(a[pref+'_within_curve'][sample]);flat=int(a[pref+'_source_point_or_vertex'][sample]);q=a['native-final_world'] if family.startswith('native') else a['after-final_world'];world=(q[flat]+q[flat+1])*.5 if 'midpoints' in family else q[flat]
   row={'family':family,'curve':i,'within':within,'signedSurfaceMeters':float(v[k]),'signedCenterMeters':float(a[pref+'_against_head_signed_center'][k]),'arcMeters':float(a[pref+'_arc'][sample]),'world':world.tolist()};rows.append(row);witness.append(row)
  families[family]={'count':len(rows),'curveIds':sorted(set(r['curve'] for r in rows)),'minimumSurfaceMeters':min(r['signedSurfaceMeters'] for r in rows)}
 Surface=runpy.run_path(str(H/'collar_continuous_offline_r11.py'))['Surface'];surface=Surface(a['crown-source-head_world'],a['crown-source-head_triangles'])
 def signed(points):
  out=[]
  for q in points:
   hit,no,_,_=surface.find_nearest(q);out.append(float(np.dot(q-hit,no)))
  return np.asarray(out)
 def stats(q,r):
  rad=np.broadcast_to(r,(len(q),));arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];combined=np.vstack([q,(q[:-1]+q[1:])*.5]);radii=np.r_[rad,(rad[:-1]+rad[1:])*.5];ar=np.r_[arc,(arc[:-1]+arc[1:])*.5];center=signed(combined);net=center-radii;free=ar>.001
  return {'minimumSurfaceMeters':float(net[free].min()),'freeSamplesBelowMinusHalfMillimeter':int(np.count_nonzero((net<-.0005)&free)),'minimumCenterMeters':float(center[free].min())},center,net,combined
 rows=[];arrays={};witness_oracle=[]
 for wi,row in enumerate(witness):
  v=float(signed(np.asarray([row['world']]))[0]);witness_oracle.append(v-row['signedCenterMeters'])
 for i in sorted(allids):
  print('CROWN_FAILURE_DIAGNOSTIC '+str(i),flush=True)
  s=int(a['after-final_starts'][i]);n=int(a['after-final_lengths'][i]);bs=int(a['before-final_starts'][i]);before=a['before-final_world'][bs:bs+n].astype(float);after=a['after-final_world'][s:s+n].astype(float);br=a['before-final_radius'][bs:bs+n];ar=a['after-final_radius'][s:s+n]
  j=int(np.flatnonzero(o['native_final_original_curve_ids']==i)[0]);os=int(o['native_final_starts'][j]);bn=o['native_final_world'][os:os+128].astype(float)
  k=int(np.flatnonzero(native_ids==i)[0]);ns=int(a['native-final_starts'][k]);an=a['native-final_world'][ns:ns+128].astype(float);nr=a['native-final_radius'][ns:ns+128]
  bm,BB,bpoly=native_model(before);am,AB,apoly=native_model(after)
  modelerr={'before128MaximumMeters':float(np.linalg.norm(bm-bn,axis=1).max()),'after128MaximumMeters':float(np.linalg.norm(am-an,axis=1).max())}
  before_stats,bc,bnet,bq=stats(bn,float(br.max()));after_stats,ac,anet,aq=stats(an,nr)
  affected=np.flatnonzero(np.any(before!=after,axis=1));support=np.any(np.abs(AB[:,affected])>1e-12,axis=1);pointfail=[r for r in witness if r['curve']==i and r['family']=='native_points'];midfail=[r for r in witness if r['curve']==i and r['family']=='native_midpoints']
  fail_support=any(bool(support[r['within']]) for r in pointfail) or any(bool(support[r['within']] or support[r['within']+1]) for r in midfail)
  controls=[]
  for r in witness:
   if r['curve']!=i or r['family']!='control_midpoints':continue
   seg=r['within'];bmid=(before[seg]+before[seg+1])*.5;amid=(after[seg]+after[seg+1])*.5;center=float(signed(bmid[None])[0]);net=center-float((br[seg]+br[seg+1])*.5)
   controls.append({'segment':seg,'world':amid.tolist(),'beforeAfterExact':bool(np.array_equal(bmid,amid)),'beforeSurfaceMeters':net,'afterActualSurfaceMeters':r['signedSurfaceMeters']})
  observed_offsets=np.unique(p['estimated_surface_control_or_segment_offset'][p['estimated_surface_curve_id']==i]).tolist()
  rows.append({'curve':i,'group':int(a['after-final_group'][i]),'controlCount':n,'changedFinalOffsets':affected.tolist(),'allControls14Through19Exact':bool(np.array_equal(before[14:],after[14:])),'protectedFrom':int(p['protected_from_point_offset'][i]),'failureControlMidpoints':controls,'beforeActualNative128':before_stats,'beforeRadiusPolicy':'Actual full curve maximum radius as conservative bound; before native radius was not exported. Center intrusion is independent of this radius bound.','afterActualNative128':after_stats,'nativeModelReproduction':modelerr,'failedNativeSamplesHaveAnyChangedControlSupport':fail_support,'offlineCalibratorCheckedOffsets':observed_offsets,'worldRoot':after[0].tolist(),'worldTip':after[-1].tolist()})
  for name,value in [('before_controls',before),('after_controls',after),('before_native_observed',bn),('after_native_observed',an),('before_native_model',bm),('after_native_model',am),('before_native_signed_center',bc),('before_native_signed_surface_conservative',bnet),('after_native_signed_center',ac),('after_native_signed_surface',anet)]:arrays[f'curve_{i}_{name}']=value
 report={'status':'diagnosed-read-only','failedReport':str(failed.relative_to(P)),'failedReportSha256':sha(failed),'failedArraysSha256':sha(fp),'baselineObservation':str(obs.relative_to(P)),'baselineObservationSha256':sha(obs),'baselineArraysSha256':sha(op),'crownPlan':str(plan_path.relative_to(P)),'crownPlanSha256':sha(plan_path),'helperSha256':sha(__file__),'curveIds':sorted(allids),'families':families,'witnesses':witness,'rows':rows,'currentHeadMatchesBaselineTopology':bool(np.array_equal(a['crown-source-head_triangles'],o['head_triangles'])),'currentHeadToBaselineMaximumMeters':float(np.linalg.norm(a['crown-source-head_world']-o['head_world'],axis=1).max()),'currentHeadExactOracleAgainstActualWitnessSignedCenterErrorMetersP0P50P100':np.percentile(witness_oracle,[0,50,100]).tolist(),'explanation':'All failing control midpoints are unchanged protected final segments 17/18 of role17. The offline crown calibrator intentionally estimated only changed segments; those protected end segments were outside its reported scope. Native paths are sampled at equal whole-curve arc distances, so before/after point indices can shift even on an unchanged geometric end; Catmull resolution12 support and actual before samples distinguish shape changes from sampling redistribution.','blenderExecuted':False,'geometryChanged':False}
 report['allFourteenAlreadyFailNativeBeforeCrown']=all(r['beforeActualNative128']['freeSamplesBelowMinusHalfMillimeter']>0 for r in rows)
 report['allFailureNativeSupportUnchanged']=all(not r['failedNativeSamplesHaveAnyChangedControlSupport'] for r in rows)
 report['nativeModelMaxErrorMeters']=max(max(r['nativeModelReproduction'].values()) for r in rows)
 with payload.open('xb') as f:np.savez_compressed(f,**arrays)
 report['arrays']=str(payload.relative_to(P));report['arraysSha256']=sha(payload)
 with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print('CROWN_FAILURE_DIAG_RESULT '+json.dumps({k:report[k] for k in ('curveIds','allFourteenAlreadyFailNativeBeforeCrown','allFailureNativeSupportUnchanged','nativeModelMaxErrorMeters','currentHeadToBaselineMaximumMeters')}));print('JSON_SHA256 '+sha(out));print('ARRAYS_SHA256 '+sha(payload))
if __name__=='__main__':main()
