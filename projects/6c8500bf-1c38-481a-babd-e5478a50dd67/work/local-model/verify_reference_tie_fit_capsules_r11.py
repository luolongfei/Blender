"""All selected measured control capsules against every proposed band triangle.

Exact finite segment/triangle minimum distance, conservatively subtracting
each segment's maximum endpoint radius. This still does not verify Blender's
native interpolated curves or a changed saved source.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def segment_triangles(a,b,triangles,closest,batch=48):
 """Segment endpoints, segment/triangle-plane interiors, and all 3 edges."""
 e0=triangles.reshape(-1,3,3);edge0=e0[:,[0,1,2]].reshape(-1,3);edge1=e0[:,[1,2,0]].reshape(-1,3);d2=edge1-edge0;ee=np.sum(d2*d2,axis=1)
 t0=triangles[:,0];v0=triangles[:,1]-t0;v1=triangles[:,2]-t0;n=np.cross(v0,v1);d00=np.sum(v0*v0,axis=1);d01=np.sum(v0*v1,axis=1);d11=np.sum(v1*v1,axis=1);baryden=d00*d11-d01*d01
 minima=[];signa=[];signb=[];crossing=[]
 for start in range(0,len(a),batch):
  aa0=a[start:start+batch];bb0=b[start:start+batch];d1=bb0-aa0;aa=np.sum(d1*d1,axis=1)
  tt=np.broadcast_to(triangles,(len(aa0),)+triangles.shape)
  da,sa=closest(aa0,tt);db,sb=closest(bb0,tt);minimum=np.minimum(da,db)
  r=aa0[:,None]-edge0[None];bb=np.einsum('ni,ki->nk',d1,d2);cc=np.sum(d1[:,None]*r,axis=2);ff=np.sum(d2[None]*r,axis=2);den=aa[:,None]*ee-bb*bb
  ss=np.zeros_like(den);np.divide(bb*ff-cc*ee,den,out=ss,where=abs(den)>1e-24);ss=np.clip(ss,0,1);zz=(bb*ss+ff)/ee
  below=zz<0;above=zz>1;ss=np.where(below,np.clip(-cc/np.maximum(aa[:,None],1e-30),0,1),ss);ss=np.where(above,np.clip((bb-cc)/np.maximum(aa[:,None],1e-30),0,1),ss);zz=np.clip(zz,0,1)
  delta=r+ss[:,:,None]*d1[:,None]-zz[:,:,None]*d2[None];minimum=np.minimum(minimum,np.linalg.norm(delta,axis=2).min(1))
  denom=d1@n.T;numer=np.einsum('nki,ki->nk',t0[None]-aa0[:,None],n);t=np.zeros_like(denom);np.divide(numer,denom,out=t,where=abs(denom)>1e-18)
  pp=aa0[:,None]+t[:,:,None]*d1[:,None]-t0[None];d20=np.einsum('nki,ki->nk',pp,v0);d21=np.einsum('nki,ki->nk',pp,v1)
  u=(d11*d20-d01*d21)/baryden;v=(d00*d21-d01*d20)/baryden;hit=(abs(denom)>1e-18)&(t>=0)&(t<=1)&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10);hit=hit.any(1)
  minimum[hit]=0.;minima.extend(minimum);signa.extend(sa);signb.extend(sb);crossing.extend(hit)
 return np.asarray(minima),np.asarray(signa),np.asarray(signb),np.asarray(crossing)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();planpath=(P/args.plan).resolve();out=(P/args.output).resolve();npz=out.with_suffix('.npz')
 for p in [planpath,out,npz]:p.relative_to((P/'work/reconstruction-r02').resolve())
 if out.exists() or npz.exists():raise RuntimeError('Unused verification output required')
 report=json.loads(planpath.read_text());a=np.load(P/report['arrays']);oj=json.loads((P/report['observation']).read_text());ob=np.load(P/oj['arrays']);gj=json.loads((P/report['gatherPlan']).read_text());g=np.load(P/gj['arrays']);tr=json.loads((P/report['tiePlan']).read_text())
 if sha(P/report['arrays'])!=report['arraysSha256'] or sha(P/gj['arrays'])!=report['gatherArraysSha256']:raise RuntimeError('Input frozen arrays changed')
 lib=runpy.run_path(str(HERE/'plan_reference_tie_fit_r11_v3.py'));geom=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'));saved={};rows=[];errors=[]
 # Stable, meaningful geometry self-checks: direct piercing and a disjoint
 # segment parallel to the test triangle at known separation.
 test=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]);distance,_,_,cross=segment_triangles(np.array([[.2,.2,-1.],[.2,.2,2.]]),np.array([[.2,.2,1.],[.4,.2,2.]]),test,geom['closest_triangles'])
 if not np.allclose(distance,[0,2],atol=1e-12) or cross.tolist()!=[True,False]:raise RuntimeError('Finite segment/triangle test failed')
 for side,role in [('positive',40),('negative',41)]:
  tailrow=next(x for x in oj['tails'] if x['prefix']=='tail_'+side);seg=lib['collect_segments'](ob,g,side,{'transform':tr['transforms'][side],'role':role,'tailRadiusScale':max(tailrow['matrixSingularValues'])})
  for ring in ['a','b']:
   pre='tie_'+side+'_'+ring;row=next(x for x in report['rings'] if x['prefix']==pre);world=a[pre+'_candidate_world'];tri=a[pre+'_triangles'];c=np.asarray(row['centerWorld']);n=np.asarray(row['axisWorld']);ids,_,_=lib['clipped_envelope'](seg,c,n,row['axialWidthMm']*.0005+.001)
   triangles=world[tri].copy()
   if lib['signed_volume'](world,tri)<0:triangles=triangles[:,[0,2,1]]
   dist,sa,sb,cross=segment_triangles(seg['a'][ids],seg['b'][ids],triangles,geom['closest_triangles']);radius=np.maximum(seg['ra'][ids],seg['rb'][ids]);lower=dist-radius
   endpoint_inside=(sa<-1e-7)|(sb<-1e-7);bad=(lower<0)|endpoint_inside;terminal=np.isin(ids,a[pre+'_envelope_segment_ids']);worst=np.argsort(lower)[:24]
   if np.any(bad):errors.append(pre+': measured control capsule intersects or lies inside band')
   value={'ring':side+'_'+ring,'segmentsAgainstAllTriangles':len(ids),'bandTriangles':len(tri),'conservativeCapsuleMinimumMeters':float(lower.min()),'centerlineSurfaceCrossings':int(cross.sum()),'endpointInsideByOrientedNearestFace':int(endpoint_inside.sum()),'capsulesWithNegativeConservativeClearance':int((lower<0).sum()),'terminalEnvelopeSegments':int(terminal.sum()),'otherSegmentsStillCollisionChecked':int((~terminal).sum()),
    'worst':[{'sourceSegment':int(ids[i]),'kind':'tail_pin' if seg['kind'][ids[i]]==0 else 'true_gather','curve':int(seg['owner'][ids[i]]),'fromFittedTerminalEnvelope':bool(terminal[i]),'conservativeClearanceMeters':float(lower[i])} for i in worst]}
   rows.append(value)
   for key,v in {'segment_ids':ids,'kind':seg['kind'][ids],'owner':seg['owner'][ids],'surface_distance':dist,'conservative_clearance':lower,'endpoint_a_signed':sa,'endpoint_b_signed':sb,'centerline_crossing':cross,'from_terminal_envelope':terminal}.items():saved[pre+'_'+key]=v
   print('BAND_ALL_TRIANGLES '+json.dumps({k:v for k,v in value.items() if k!='worst'}),flush=True)
 with npz.open('xb') as f:np.savez_compressed(f,**saved)
 result={'schemaVersion':1,'status':'measured_control_capsules_passed' if not errors else 'blocked','errors':errors,'productionCollisionAccepted':False,'appearanceAccepted':False,'nativeCurvesVerified':False,'bandPlan':str(planpath.relative_to(P)),'bandPlanSha256':sha(planpath),'bandArraysSha256':report['arraysSha256'],'helperSha256':sha(__file__),'method':__doc__,'rings':rows,'arrays':str(npz.relative_to(P)),'arraysSha256':sha(npz),'limits':['Every selected full control segment was tested against every band triangle; varying radius uses a conservative max-endpoint bound.','Endpoint containment uses the oriented nearest triangle face of this closed annular mesh; native interpolation and changed source remain unverified.','No strand, pin, gather, object transform, Blender file or band geometry changed.']}
 with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print('BAND_CAPSULE_PROOF '+json.dumps({'status':result['status'],'errors':errors,'output':str(out.relative_to(P))}),flush=True)
if __name__=='__main__':main()
