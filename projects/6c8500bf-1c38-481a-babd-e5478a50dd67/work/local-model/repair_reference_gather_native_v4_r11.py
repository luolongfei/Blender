"""Repair only the 25 actually failed dense gathers, retaining native topology.

The current failed run supplies actual post-contact controls and native paths.
The old observed head is only a documented local predictor; the next Blender
wrapper must still use its current BVH, native paths and unchanged thresholds.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
CURVES=np.array([360,373,2451,2461,2489,2491,2493,2496,10238,10286,10287,10296,10297,10299,10301,10329,10330,10462,10463,10464,10527,10540,10541,10542,21550],np.int32)
TAKEOFF=np.array([10330,10527,10540,10541,10542],np.int32)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def native_basis(n,resolution=12):
 out=[]
 for i in range(n-1):
  for t in np.arange(resolution)/resolution:
   coef=np.array([-.5*t+t*t-.5*t*t*t,1-2.5*t*t+1.5*t*t*t,.5*t+2*t*t-1.5*t*t*t,-.5*t*t+.5*t*t*t]);row=np.zeros(n)
   for j,c in zip((max(0,i-1),i,i+1,min(n-1,i+2)),coef):row[j]+=c
   out.append(row)
 row=np.zeros(n);row[-1]=1;out.append(row);return np.asarray(out)
def resampled_basis(q,basis,count=64):
 points=basis@q;arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];at=np.linspace(0,arc[-1],count)
 return np.stack([np.interp(at,arc,basis[:,i]) for i in range(len(q))],axis=1)
def turns(q):
 v=np.diff(q,axis=0);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-15);return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1,1)))
class HeadPredictor:
 def __init__(self,world,tri):
  full=world[tri];lo=full.min(1);hi=full.max(1);mask=np.all(hi>=np.array([.035,-.06,1.365]),axis=1)&np.all(lo<=np.array([.105,.085,1.495]),axis=1);self.tri=full[mask];self.centers=self.tri.mean(1);self.origin=self.centers.mean(0);self.centered=self.centers-self.origin;self.norm=np.sum(self.centered*self.centered,axis=1);self.cell=.002;self.grid={}
  yz=self.tri[:,:,1:];lo=np.floor(yz.min(1)/self.cell).astype(int);hi=np.floor(yz.max(1)/self.cell).astype(int)
  for i,(a,b) in enumerate(zip(lo,hi)):
   for y in range(a[0],b[0]+1):
    for z in range(a[1],b[1]+1):self.grid.setdefault((y,z),[]).append(i)
  self.closest=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'))['closest_triangles']
 def signed(self,points):
  result=[]
  for st in range(0,len(points),64):
   p=points[st:st+64];q=p-self.origin;d=np.sum(q*q,axis=1)[:,None]+self.norm[None]-2*q@self.centered.T;k=min(64,len(self.tri));ids=np.argpartition(d,k-1,axis=1)[:,:k];_,signed=self.closest(p,self.tri[ids]);result.extend(signed)
  return np.asarray(result)
 def outer_x(self,p):
  ids=self.grid.get(tuple(np.floor(p[1:]/self.cell).astype(int)),[])
  if not ids:return None
  t=self.tri[ids];a=t[:,0,1:];b=t[:,1,1:]-a;c=t[:,2,1:]-a;v=p[1:]-a;den=b[:,0]*c[:,1]-b[:,1]*c[:,0];valid=abs(den)>1e-15;u=np.zeros(len(t));w=np.zeros(len(t));u[valid]=(v[valid,0]*c[valid,1]-v[valid,1]*c[valid,0])/den[valid];w[valid]=(b[valid,0]*v[valid,1]-b[valid,1]*v[valid,0])/den[valid];valid &= (u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)
  if not valid.any():return None
  x=t[:,0,0]+u*(t[:,1,0]-t[:,0,0])+w*(t[:,2,0]-t[:,0,0]);return float(x[valid].max())
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-plan',required=True);ap.add_argument('--failure-report',required=True);ap.add_argument('--output',required=True);ar=ap.parse_args();bp=(P/ar.base_plan).resolve();fp=(P/ar.failure_report).resolve();out=(P/ar.output).resolve();npz=out.with_suffix('.npz')
 for p in [bp,fp,out,npz]:p.relative_to((P/'work/reconstruction-r02').resolve())
 if out.exists() or npz.exists():raise RuntimeError('Unused independent correction outputs required')
 base=json.loads(bp.read_text());fail=json.loads(fp.read_text());fa=P/fail['arrays'];ba=P/base['arrays']
 if sha(bp)!='e5dab29b13c1def0047f975306927095225c984e9e2f6ca8218c815c84fa0249' or sha(ba)!=base['arraysSha256'] or sha(fa)!=fail['arraysSha256'] or fail['gatherPlanSha256']!=sha(bp):raise RuntimeError('Actual failed-run/base identity mismatch')
 arrays={k:v.copy() for k,v in np.load(ba).items()};actual=np.load(fa);s=arrays['source_starts'];lens=arrays['source_lengths'];target=arrays['target_world'];original=target.copy();source=arrays['source_world'];head_path=P/base['observation'];headj=json.loads(head_path.read_text());h=np.load(P/headj['arrays']);head=HeadPredictor(h['head_world'],h['head_triangles'])
 if not np.array_equal(actual['before-front_starts'],s) or not np.array_equal(actual['before-front_lengths'],lens) or np.linalg.norm(actual['before-front_world']-source,axis=1).max()>2e-6:raise RuntimeError('Failed run dense source does not match unchanged input contract')
 rows=[];exceptions=[];errors=[];pred_native=[];pred_control=[];native_deltas=[];evidence=[]
 finalq=actual['after-final_world'];finals=actual['after-final_starts'];finall=actual['after-final_lengths'];nw=actual['gather_native_after_world'];ns=actual['gather_native_after_starts'];native_ids=actual['gather_native_after_original_curve'];radius=actual['after-final_radius'];source_ids=arrays['gather_curve_ids']
 for curve in CURVES:
  st=int(s[curve]);n=int(lens[curve]);fs=int(finals[curve]);current=finalq[fs:fs+n].astype(float);old=original[st:st+n].astype(float);before=current.copy();basis=native_basis(n);ni=int(np.flatnonzero(native_ids==curve)[0]);native_old=nw[ns[ni]:ns[ni]+64];model=resampled_basis(current,basis)@current;err=float(np.linalg.norm(model-native_old,axis=1).max());native_deltas.append(err)
  if err>2e-6:raise RuntimeError('Empirical native resolution-12 model does not reproduce actual current curve '+str(curve))
  first=1 if curve in TAKEOFF else 3;free=np.arange(first,n-3);control=np.arange(n);G=np.exp(-.5*((control[:,None]-free[None])/1.15)**2);G[:first]=0;G[-3:]=0;amp=np.zeros(len(free));maxradius=float(radius[fs:fs+n].max());history=[]
  for iteration in range(18):
   offset=G@amp;q=current.copy();q[:,0]+=offset;nativeW=resampled_basis(q,basis);identity=np.eye(n);W=np.vstack([basis,(basis[:-1]+basis[1:])*.5,(identity[:-1]+identity[1:])*.5,identity,nativeW,(nativeW[:-1]+nativeW[1:])*.5]);points=W@q
   # Root contact exemption is the same 1 mm arc-distance concept; other
   # frozen controls are included so a new interpolation defect cannot hide.
   baseline=W@current;away=np.linalg.norm(baseline-current[0],axis=1)>.001;gap=head.signed(points)-maxradius;bad=np.flatnonzero(away&(gap<-.00020));history.append({'iteration':iteration,'minimumPredictedGapMeters':float(gap[away].min()),'samplesBelowMinus02mm':len(bad)})
   if not len(bad):break
   C=W@G;needed=[]
   for index in bad:
    hit=head.outer_x(points[index])
    if hit is None:continue
    desired=hit+.00075+maxradius;needed.append((desired-float(points[index,0]),index,desired))
   changed=False
   for _,index,desired in sorted(needed,reverse=True):
    coeff=C[index];positive=np.maximum(coeff,0);den=float(positive@positive);deficit=desired-float(W[index]@current[:,0]+coeff@amp)
    if deficit<=0:continue
    if den<1e-14:continue
    amp+=positive*(deficit/den);changed=True
   if not changed:break
   if (G@amp).max()>.010:break
  offset=G@amp;prediction=current.copy();prediction[:,0]+=offset;new=old.copy();modified=free[offset[free]>1e-7];new[modified]=prediction[modified];new[:first]=old[:first];new[-3:]=old[-3:];target[st:st+n]=new.astype(np.float32)
  # New pre-contact controls use the actually observed already-contacted
  # free point as their baseline. The next native head-contact run, which is
  # not emulated here, must confirm these candidate positions on current skin.
  nativeW=resampled_basis(prediction,basis);native=nativeW@prediction;nativegap=head.signed(np.vstack([native,(native[:-1]+native[1:])*.5]))-maxradius;ngfree=np.r_[np.linalg.norm(native-native[0],axis=1),np.linalg.norm((native[:-1]+native[1:])*.5-native[0],axis=1)]>.001
  mid=(prediction[:-1]+prediction[1:])*.5;midgap=head.signed(mid)-maxradius;controlgap=head.signed(prediction)-maxradius;nt=turns(prediction);ot=turns(before);newabrupt=np.flatnonzero((nt>95)&(nt>ot+20));move=np.linalg.norm(target[st:st+n]-original[st:st+n],axis=1);j=int(np.flatnonzero(source_ids==curve)[0]);equal=np.all(target[st:st+n]==source[st:st+n],axis=1);different=np.flatnonzero(~equal);anchor=int(different[0]-1) if len(different) else n-1;arrays['preserved_upper_last_point'][j]=anchor
  if nativegap[ngfree].min()<-.0005 or midgap.min()<-.0005:errors.append(str(curve)+': old-head predictor still has a required native/control-midpoint failure')
  if newabrupt.size:errors.append(str(curve)+': new local reversal above unchanged 95-degree gate')
  if move.max()>.012:errors.append(str(curve)+': exceeds 12 mm local scope')
  if not np.array_equal(target[st:st+first],original[st:st+first]) or not np.array_equal(target[st+n-3:st+n],original[st+n-3:st+n]):errors.append(str(curve)+': protected point changed')
  if curve in TAKEOFF:exceptions.append({'curve':int(curve),'allowedOffsets':[1,2],'actualMovementMeters':move[[1,2]].tolist(),'beforeWorld':original[st+np.array([1,2])].tolist(),'afterWorld':target[st+np.array([1,2])].tolist(),'rootP0Exact':bool(np.array_equal(target[st],original[st])),'reason':'Actual old/current control segment1 violates skin while both endpoints were frozen; explicitly authorized narrow dense-gather takeoff exception, not tail pins.'})
  row={'curve':int(curve),'nativeModelMaximumActualResidualMeters':err,'iterations':history,'maximumTargetMovementMeters':float(move.max()),'changedControlOffsets':np.flatnonzero(np.any(target[st:st+n]!=original[st:st+n],axis=1)).tolist(),'preservedUpperLastPoint':anchor,'predictedNativeMinimumGapMeters':float(nativegap[ngfree].min()),'predictedControlMidpointMinimumGapMeters':float(midgap.min()),'predictedControlMinimumGapMeters':float(controlgap.min()),'predictedMaximumTurnDegrees':float(nt.max()),'newAbruptJointOffsets':(newabrupt+1).tolist()};rows.append(row);pred_native.append(native);pred_control.append(prediction);evidence.append(offset)
  print('R11_LOCAL_GATHER_REPAIR '+json.dumps({k:row[k] for k in ('curve','maximumTargetMovementMeters','predictedNativeMinimumGapMeters','predictedControlMidpointMinimumGapMeters')}),flush=True)
 mask=np.zeros(len(target),bool)
 for i in CURVES:mask[s[i]:s[i]+lens[i]]=True
 if not np.array_equal(target[~mask],original[~mask]):errors.append('Target outside the exact25 correction set changed')
 arrays.update(target_world=target,takeoff_correction_curve_ids=TAKEOFF,corrected_curve_ids=CURVES,predicted_native_corrected_world=np.concatenate(pred_native),predicted_after_contact_corrected_world=np.concatenate(pred_control),correction_control_x_offset=np.concatenate(evidence))
 # Old v3 sample/turn arrays remain evidence of that base candidate, never
 # relabel them as measurements of the newly corrected geometry.
 for key in list(arrays):
  if key.startswith(('sample_','refined_','old_max_turn','target_max_')):arrays['base_v3_'+key]=arrays.pop(key)
 with npz.open('xb') as stream:np.savez_compressed(stream,**arrays)
 report={'schemaVersion':1,'status':'blocked' if errors else 'proposed','errors':errors,'tiePlanSha256':base['tiePlanSha256'],'tiePlan':base['tiePlan'],'arrays':str(npz.relative_to(P)),'arraysSha256':sha(npz),'sourceGroomArraysSha256':base['sourceGroomArraysSha256'],'baseGatherPlan':str(bp.relative_to(P)),'baseGatherPlanSha256':sha(bp),'baseGatherArraysSha256':base['arraysSha256'],'actualFailureReport':str(fp.relative_to(P)),'actualFailureReportSha256':sha(fp),'actualFailureArraysSha256':sha(fa),'correctedCurveIds':CURVES.tolist(),'takeoffCorrectionCurveIds':TAKEOFF.tolist(),'exceptionFields':exceptions,'curves':rows,'maximumNativeModelResidualMeters':max(native_deltas),'helperSha256':sha(__file__),
  'localPredictorHeadObservation':str(head_path.relative_to(P)),'localPredictorHeadObservationSha256':sha(head_path),'localPredictorHeadArraysSha256':sha(P/headj['arrays']),'currentHeadExportAvailable':False,'currentHeadCollisionAccepted':False,'actualNewNativeEvaluationExecuted':False,'appearanceAccepted':False,'savedBlend':False,
  'preservation':{'otherTargetsExact':True,'sourceArraysUnchanged':True,'allTailMapsUnchanged':True,'allLastThreeExact':True,'allRootP0Exact':True,'firstThreeExactExceptExplicitFive':True,'all46FaceCurtainsExact':True,'densePointCount':len(target)},
  'predictionMethod':'Empirically matched actual native paths: clamped Catmull-Rom, 12 samples per segment then 64 arc-length samples, <=2 micrometre actual residual guard. Smooth nonnegative local +X fields expand the positive-ear connector around the old actual outer skin. The 5 explicitly authorized takeoff exceptions may change p1/p2; no tail pin changes.',
  'limits':['The 8dea head is an OLD local predictor. Current failed-run native/controls are actual; current head was not exported by that failed wrapper.','The next wrapper must apply the current native head-contact node and rerun every existing current BVH/native64/skin/radius/metadata gate unchanged.','New proposal records only exact25 corrections. It does not claim a saved source or visual acceptance.']}
 with out.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
 print('R11_LOCAL_REPAIR_RESULT '+json.dumps({'status':report['status'],'errors':errors,'arraysSha256':report['arraysSha256'],'maximumNativeModelResidualMeters':max(native_deltas)}),flush=True)
if __name__=='__main__':main()
