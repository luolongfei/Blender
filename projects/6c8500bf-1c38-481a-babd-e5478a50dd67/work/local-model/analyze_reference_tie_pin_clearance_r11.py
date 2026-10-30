"""Read-only pin-envelope clearance analysis for the actual v10e tie observation.

Run from the workspace root with NumPy. Prints JSON lines; optional --output
creates a new independent JSON/NPZ evidence pair. No bpy, render, or source
writes. Scope is ONLY the first three raw
tail points and their first two straight segments with actual linearly
interpolated radius. Upper gather hair and other tail points are excluded.

Seam statistics use actual edge-length weights. Inner wall statistics use
equal-area barycentric centroids over all 256 inward-facing mesh triangles.
A negative sampled clearance means the ring sample lies inside at least
one raw swept-radius segment, not an all-hair signed distance or physics
constraint. Contact/pressure and appearance acceptance are not inferred.
"""
from pathlib import Path
import numpy as np,json,hashlib,sys,argparse
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()/'work/reconstruction-r02/reference-tie-alignment-r11-v2'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output')
args=parser.parse_args()
output=(project_path()/args.output).resolve().with_suffix('') if args.output else None
if output:
 output.relative_to((project_path()/'work/reconstruction-r02').resolve())
 if output.with_suffix('.json').exists() or output.with_suffix('.npz').exists():raise RuntimeError('Unused independent output pair required')
D=np.load(P.with_suffix('.npz')); J=json.loads(P.with_suffix('.json').read_text())
assert J['sourceSha256']=='8dea7f21c32923ca82f3be208e5ed5e362e31f45d4e215c18f44a7137f89a706'
evidence={'schemaVersion':1,'status':'measured_tail_only','observation':str(P.relative_to(project_path())), 'sourceSha256':J['sourceSha256'], 'observationJsonSha256':hashlib.sha256(P.with_suffix('.json').read_bytes()).hexdigest(), 'observationNpzSha256':hashlib.sha256(P.with_suffix('.npz').read_bytes()).hexdigest(), 'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'method':__doc__, 'savedBlend':False, 'rendered':False,'rings':[]}
if evidence['observationNpzSha256']!=J['arraysSha256']:raise RuntimeError('Input NPZ identity mismatch')
saved_arrays={}
print(json.dumps({k:v for k,v in evidence.items() if k not in ('method','rings')}),flush=True)
def stats(v):
 return dict(zip(('min','p05','p25','median','p75','p95','max'),np.percentile(v,[0,5,25,50,75,95,100]).tolist()))
def capsule_distance(q,a,b,ra,rb):
 """Exact q to linearly interpolated radius ball sweep; positive outside."""
 v=b-a; L=np.linalg.norm(v,axis=1);u=v/L[:,None];k=(rb-ra)/L
 if np.any(L<=0) or np.any(np.abs(k)>=1):raise RuntimeError('Unsupported zero-length segment or radius slope')
 vals=[];nearest=[];hits=[]
 for beg in range(0,len(q),64):
  x=q[beg:beg+64,None,:]-a[None,:,:]
  p=np.sum(x*u,axis=2);h2=np.maximum(np.sum(x*x,axis=2)-p*p,0.)
  s=np.clip(p+k[None,:]*np.sqrt(h2/np.maximum(1.-k*k,1e-30)),0,L)
  # s has query x segment dimensions. Actual radius slopes are below one.
  f=np.sqrt((s-p)**2+h2)-(ra+k*s)
  ids=np.argmin(f,axis=1); row=np.arange(len(ids))
  vals.extend(f[row,ids]); nearest.extend(ids)
  hits.extend(a[ids]+u[ids]*s[row,ids,None])
 return np.asarray(vals),np.asarray(nearest),np.asarray(hits)
def weighted_stats(v,weights):
 ids=np.argsort(v);x=v[ids];w=weights[ids];cdf=(np.cumsum(w)-w*.5)/w.sum()
 return dict(zip(('min','p05','p25','median','p75','p95','max'),np.interp([0,.05,.25,.5,.75,.95,1],cdf,x).tolist()))
def area_points(tris,N=8):
 bary=[]
 for i in range(N):
  for j in range(N-i):
   bary.append(np.asarray([i+1/3,j+1/3])/N)
   if i+j<N-1:bary.append(np.asarray([i+2/3,j+2/3])/N)
 bary=np.asarray(bary);uv=np.c_[1-bary.sum(1),bary]
 pts=np.einsum('bi,tij->tbj',uv,tris).reshape(-1,3)
 areas=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)/2
 weights=np.repeat(areas/(N*N),N*N)
 cover=np.linalg.norm(tris-tris.mean(1)[:,None,:],axis=2).max()/N
 return pts,weights,cover
for side in ('positive','negative'):
 prefix='tail_'+side;pin=D[prefix+'_first3_world'];ids=D[prefix+'_first3_point_indices']
 r=D[prefix+'_radius'][ids]*max(next(x for x in J['tails'] if x['prefix']==prefix)['matrixSingularValues'])
 saved_arrays[prefix+'_first3_world']=pin
 saved_arrays[prefix+'_first3_radius_world_bound']=r
 saved_arrays[prefix+'_first3_point_indices']=ids
 a=pin[:,:2,:].reshape(-1,3);b=pin[:,1:,:].reshape(-1,3);ra=r[:,:2].reshape(-1);rb=r[:,1:].reshape(-1)
 for ring in ('a','b'):
  pfx='tie_'+side+'_'+ring+'_raw';w=D[pfx+'_world'];triids=D[pfx+'_triangles'];c=w.mean(0);blocks=w.reshape(32,8,3)
  _,v=np.linalg.eigh((w-c).T@(w-c));n=v[:,0]
  if n@(pin.mean(0)[2]-pin.mean(0)[0])<0:n=-n
  inner=blocks[:,4];f=np.arange(64)/64
  seam=(inner[:,None,:]*(1-f[None,:,None])+np.roll(inner,-1,axis=0)[:,None,:]*f[None,:,None]).reshape(-1,3)
  seamweight=np.repeat(np.linalg.norm(np.roll(inner,-1,axis=0)-inner,axis=1)/64,64)
  gap,near,hits=capsule_distance(seam,a,b,ra,rb)
  innerface=np.all((triids%8>=2)&(triids%8<=6),axis=1)
  tr=w[triids[innerface]];q,areaweight,cover=area_points(tr,8)
  agap,anear,ahits=capsule_distance(q,a,b,ra,rb)
  dist=(pin-c)@n
  no_before=(dist+r).max(1)<0; no_after=(dist-r).min(1)>0
  loose=int(gap.argmax())
  array_prefix='tie_'+side+'_'+ring
  for key,value in {'inner_edge_vertices_world':inner,'inner_edge_samples_world':seam,'inner_edge_arc_weights_m':seamweight,'inner_edge_clearance_m':gap,'inner_edge_nearest_segment':near,'inner_edge_nearest_centerline_world':hits,'inner_triangle_ids':np.flatnonzero(innerface),'inner_triangle_world':tr,'inner_wall_samples_world':q,'inner_wall_area_weights_m2':areaweight,'inner_wall_clearance_m':agap,'inner_wall_nearest_segment':anear,'inner_wall_nearest_centerline_world':ahits}.items():saved_arrays[array_prefix+'_'+key]=value
  bc=blocks.mean(1)
  geometry={'centerlineDiametersMajorMinorMm':(np.ptp((bc-c)@v[:,[2,1]],axis=0)*1000).tolist(),'innerDiametersMajorMinorMm':(np.ptp((inner-c)@v[:,[2,1]],axis=0)*1000).tolist(),'outerDiametersMajorMinorMm':(np.ptp((blocks[:,0]-c)@v[:,[2,1]],axis=0)*1000).tolist(),'tubeDiameterMm':float(2*np.linalg.norm(blocks-bc[:,None,:],axis=2).mean()*1000),'axialWidthMm':float(np.ptp((w-c)@n)*1000),'rawEvalMaxDeltaMm':float(np.linalg.norm(w-D['tie_'+side+'_'+ring+'_eval_world'],axis=1).max()*1000)}
  geometry['axialWidthDividedByInnerMajorMinorDiameter']=(geometry['axialWidthMm']/np.asarray(geometry['innerDiametersMajorMinorMm'])).tolist()
  out={'ring':side+'_'+ring,'centerMm':(c*1000).tolist(),'normalAlongTail':n.tolist(),
       'arrayPrefix':array_prefix,'nearestSegmentIndexMeaning':'curve_index * 2 + local_segment_index; local segment 0 joins pin0-pin1, segment1 joins pin1-pin2','ringGeometry':geometry,
       'seamLengthMm':float(seamweight.sum()*1000),'seamArcWeightedGapMm':weighted_stats(gap*1000,seamweight),
       'seamSamplePitchBoundMm':float(seamweight.max()*1000),
       'seamLargestGapPointMm':(seam[loose]*1000).tolist(),
       'seamLargestGapDirection':((seam[loose]-c)/np.linalg.norm(seam[loose]-c)).tolist(),
       'seamLargestGapAboveHighestRootMm':float((seam[loose,2]-pin[:,0,2].max())*1000),
       'seamArcFractions':{'over1mm':float(np.sum(seamweight[gap>.001])/seamweight.sum()),'over2mm':float(np.sum(seamweight[gap>.002])/seamweight.sum()),'under025mm':float(np.sum(seamweight[gap<.00025])/seamweight.sum()),'insideHairRadius':float(np.sum(seamweight[gap<0])/seamweight.sum())},
       'innerHemisphereTriangles':len(tr),'innerWallAreaMm2':float(areaweight.sum()*1e6),'innerWallSamples':len(q),'areaSamplingCoverRadiusBoundMm':float(cover*1000),
       'innerWallAreaWeightedGapMm':weighted_stats(agap*1000,areaweight),
       'innerWallAreaFractions':{'over1mm':float(np.sum(areaweight[agap>.001])/areaweight.sum()),'over2mm':float(np.sum(areaweight[agap>.002])/areaweight.sum()),'under025mm':float(np.sum(areaweight[agap<.00025])/areaweight.sum()),'insideHairRadius':float(np.sum(areaweight[agap<0])/areaweight.sum())},
       'innerWallAreaFractionGuaranteedOver1mmByLipschitzBound':float(np.sum(areaweight[agap>.001+cover])/areaweight.sum()),
       'innerWallLargestGapPointMm':(q[agap.argmax()]*1000).tolist(),
       'first3CapsuleCurvesWhollyBeforePlane':int(no_before.sum()),'first3CapsuleCurvesWhollyAfterPlane':int(no_after.sum()),
       'scope':'ONLY first3 raw tail points, first2 straight segments, linearly interpolated actual radius. Upper gather hair and remaining tail segments excluded; not all-hair support proof.'}
  print(json.dumps(out),flush=True)
  evidence['rings'].append(out)
if output:
 with output.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**saved_arrays)
 evidence['arrays']=str(output.with_suffix('.npz').relative_to(project_path()))
 evidence['arraysSha256']=hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest()
 evidence['arrayManifest']={k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k,v in saved_arrays.items()}
 with output.with_suffix('.json').open('x',encoding='utf8') as f:json.dump(evidence,f,ensure_ascii=False,indent=2)
 print(json.dumps({'evidenceJson':str(output.with_suffix('.json')),'evidenceJsonSha256':hashlib.sha256(output.with_suffix('.json').read_bytes()).hexdigest(),'evidenceNpzSha256':evidence['arraysSha256'],'scriptSha256':evidence['scriptSha256']}),flush=True)
