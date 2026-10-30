"""Fit-plan installation gate for an ACTUAL complete post-tie observation.

Pure functions load and check the frozen candidate against native samples.
install_checked imports bpy only when called in Blender, copies only four
ring mesh datablocks, never transports hair or saves a blend. No rigid T is
applied anywhere in this module. A wrapper must save only after all gates.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
PLAN_SHA='8a0820c84080f5048cd093fcd3362443dbbd3e7582c3dcef1bae4f01ca9ab815'
ARRAY_SHA='4ce71353ececc52441cca8d4f89a64d9540bcaa8cec6fe6dc7af4f2029182253'
PINS={
 'verify_reference_tie_fit_capsules_r11.py':'e4cc78174f29f8f18bd56c21105a17a18ee49793167763adeffbb257eccde998',
 'plan_reference_tie_transport_r11.py':'535d0a00e06cf3a5518c9bf4d81c314e32d3f0490ee728d107bed99e4c92f8ba',
 'observe_reference_tie_alignment_r11_v2.py':'c488fccec705d4a805f142dab3c2fa8fe902bf42d3a59fd6307d04f248cf7c5c',
}
TOL=2e-6

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()
def scoped(path):
 p=(P/str(path).replace('\\','/')).resolve();p.relative_to((P/'work/reconstruction-r02').resolve());return p
def pin_check():
 for name,digest in PINS.items():
  if sha(HERE/name)!=digest:raise RuntimeError('Frozen geometry helper changed: '+name)
def points_world(local,matrix):return np.asarray(local,float)@np.asarray(matrix,float)[:3,:3].T+np.asarray(matrix,float)[:3,3]
def topology(world,starts,lengths,radius,ids,label,expected_count=None):
 if world.ndim!=2 or world.shape[1]!=3 or starts.ndim!=1 or lengths.shape!=starts.shape or ids.shape!=starts.shape:raise RuntimeError(label+': malformed native topology')
 if expected_count is not None and len(ids)!=expected_count:raise RuntimeError(label+': wrong native curve count')
 if np.any(lengths<2) or not np.array_equal(starts,np.r_[0,np.cumsum(lengths[:-1])]) or int(lengths.sum())!=len(world):raise RuntimeError(label+': native curve offsets are not complete/contiguous')
 if radius.shape!=(len(world),) or not np.isfinite(world).all() or not np.isfinite(radius).all() or np.any(radius<0):raise RuntimeError(label+': missing/nonfinite actual world positions/radii')
 if len(np.unique(ids))!=len(ids):raise RuntimeError(label+': repeated original curve IDs')

def load_inputs(plan_path,observation_path):
 """Return a bound actual snapshot; deliberately fail before sampling exists."""
 pin_check();pp=scoped(plan_path);op=scoped(observation_path)
 if sha(pp)!=PLAN_SHA:raise RuntimeError('Expected the frozen v3 band fit manifest')
 plan=json.loads(pp.read_text());obs=json.loads(op.read_text());pa=scoped(plan['arrays']);oa=scoped(obs['arrays'])
 if sha(pa)!=ARRAY_SHA or plan['arraysSha256']!=ARRAY_SHA or sha(oa)!=obs['arraysSha256']:raise RuntimeError('Candidate or actual observation arrays changed')
 if obs.get('status')!='collected' or obs.get('observationPose')!='REST' or not all(obs.get(k) is True for k in ('sourceUnchanged','sourcePoseRestored','sourceFrameAndActionUnchanged')) or obs.get('readOnlyRestoredPoseChanges'):raise RuntimeError('A successfully restored read-only complete current REST observation is required')
 data=np.load(oa,allow_pickle=False);candidate=np.load(pa,allow_pickle=False)
 for key,meta in obs['arrayManifest'].items():
  if array_sha(data[key])!=meta['sha256']:raise RuntimeError('Actual observation manifest mismatch: '+key)
 for key in ('head_world','head_triangles','native_gather_final_world','native_gather_final_starts','native_gather_final_lengths','native_gather_final_group','native_gather_final_original_curve_ids','native_gather_final_radius_world'):
  if key not in data:raise RuntimeError('Complete post-tie capture missing '+key)
 checks=[]
 for ring in plan['rings']:
  p=ring['prefix'];current=data[p+'_raw_world'];expected=candidate[p+'_source_expected_after_transport_world'];matrix=data[p+'_raw_matrix'];tri=data[p+'_raw_triangles']
  if current.shape!=(256,3) or not np.array_equal(tri,candidate[p+'_triangles']) or not np.array_equal(candidate[p+'_vertex_correspondence'],np.arange(256)):raise RuntimeError(p+': actual ring topology/vertex identity changed')
  delta=float(np.linalg.norm(current-expected,axis=1).max());eval_delta=float(np.linalg.norm(data[p+'_eval_world']-current,axis=1).max())
  if delta>TOL or eval_delta>TOL:raise RuntimeError(p+': current raw/evaluated ring differs from the expected once-transported source')
  if not np.allclose(matrix,candidate[p+'_matrix'],atol=1e-10,rtol=0):raise RuntimeError(p+': object matrix changed; do not reinterpret local candidate')
  new=points_world(candidate[p+'_candidate_local'],matrix)
  if np.linalg.norm(new-candidate[p+'_candidate_world'],axis=1).max()>TOL:raise RuntimeError(p+': candidate local/world mismatch')
  checks.append({'ring':p,'sourceExpectedAfterTransportMaximumMeters':delta,'rawEvaluatedMaximumDifferenceMeters':eval_delta,'matrixUnchanged':True})
 return {'planPath':pp,'observationPath':op,'plan':plan,'observation':obs,'candidate':candidate,'actual':data,'identity':{'planSha256':sha(pp),'candidateArraysSha256':sha(pa),'observationSha256':sha(op),'observationArraysSha256':sha(oa),'sourceSha256':obs['sourceSha256']},'ringDependencyChecks':checks}

def native_segments(inputs):
 """All actual sampled paths, including protected face curtains, no transport."""
 a=inputs['actual'];c=inputs['candidate'];plan=inputs['plan'];old_gather=json.loads(scoped(plan['gatherPlan']).read_text());old=np.load(scoped(old_gather['arrays']),allow_pickle=False)
 expected_ids=np.sort(np.r_[old['gather_curve_ids'],old['preserved_face_curtain_curve_ids']]);parts=[];rows=[]
 for index,prefix in enumerate(('native_tail_positive','native_tail_negative','native_gather_final')):
  q=a[prefix+'_world'].astype(np.float64);starts=a[prefix+'_starts'];lengths=a[prefix+'_lengths'];ids=a[prefix+'_original_curve_ids'];r=a[prefix+('_radius_world' if index==2 else '_radius')].astype(np.float64)
  topology(q,starts,lengths,r,ids,prefix,14231 if index==2 else 2480)
  if index==2:
   if not np.array_equal(ids,expected_ids) or not np.all(lengths==64) or not np.array_equal(a[prefix+'_group'],old['source_group'][ids]):raise RuntimeError('Native gather/face-curtain ordered identities do not match the preserved source')
  elif not np.array_equal(ids,np.arange(2480)) or not np.all(lengths==128):raise RuntimeError(prefix+': expected actual complete 128-point native paths')
  mask=np.ones(len(q),bool);mask[starts+lengths-1]=False;beg=np.flatnonzero(mask)
  parts.append({'a':q[beg],'b':q[beg+1],'ra':r[beg],'rb':r[beg+1],'curve':np.repeat(ids,lengths-1),'kind':np.full(len(beg),index,np.int8),'point':np.concatenate([np.arange(n-1,dtype=np.int32) for n in lengths])})
  rows.append({'prefix':prefix,'curves':len(ids),'points':len(q),'segments':len(beg),'samplesPerCurve':int(lengths[0]),'worldRadiusMinMax':[float(r.min()),float(r.max())],'positionsWereTransformed':False})
 return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]},rows

def validate_native_snapshot(inputs):
 """Offline or Blender-safe full band/native sampled-capsule validation."""
 pin_check();segments,native_rows=native_segments(inputs);c=inputs['candidate'];rows=[];evidence={};errors=[]
 proof=runpy.run_path(str(HERE/'verify_reference_tie_fit_capsules_r11.py'));geometry=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'));distance=proof['segment_triangles'];closest=geometry['closest_triangles']
 maxr=np.maximum(segments['ra'],segments['rb']);margin=.003
 for ring in inputs['plan']['rings']:
  p=ring['prefix'];world=c[p+'_candidate_world'];tri=c[p+'_triangles'];lo=world.min(0);hi=world.max(0)
  # Every excluded capsule is separated from the ring AABB by >3 mm on at
  # least one axis. A path crossing the box is retained even if its endpoints
  # lie outside. Both sides and all protected native face-curtain paths count.
  ids=np.flatnonzero(np.all(np.maximum(segments['a'],segments['b'])+maxr[:,None]>=lo-margin,axis=1)&np.all(np.minimum(segments['a'],segments['b'])-maxr[:,None]<=hi+margin,axis=1))
  triangles=world[tri].copy();volume=np.sum((triangles[:,0]-world.mean(0))*np.cross(triangles[:,1]-world.mean(0),triangles[:,2]-world.mean(0)))/6
  if volume<0:triangles=triangles[:,[0,2,1]]
  d,sa,sb,cross=distance(segments['a'][ids],segments['b'][ids],triangles,closest);clear=d-maxr[ids];inside=(sa<-1e-7)|(sb<-1e-7);bad=(clear<0)|inside
  if bad.any():errors.append(p+': actual native sampled path capsule intersects or lies within the proposed band')
  worst=np.argsort(clear)[:32];row={'ring':p,'segmentsAgainstAll512Triangles':len(ids),'minimumConservativeClearanceMeters':float(clear.min()),'centerlineSurfaceCrossings':int(cross.sum()),'capsulesWithNegativeClearance':int((clear<0).sum()),'endpointsInsideByOrientedNearestFace':int(inside.sum()),'maximumSelectedNativeSegmentLengthMeters':float(np.linalg.norm(segments['b'][ids]-segments['a'][ids],axis=1).max()),'excludedSegmentSeparationLowerBoundMeters':margin,
   'worst':[{'segment':int(ids[j]),'kind':('tail_positive','tail_negative','gather_including_face_curtain')[segments['kind'][ids[j]]],'originalCurve':int(segments['curve'][ids[j]]),'nativeSegmentOffset':int(segments['point'][ids[j]]),'conservativeClearanceMeters':float(clear[j])} for j in worst]}
  rows.append(row)
  for key,value in {'segment_ids':ids,'clearance':clear,'endpoint_a_signed':sa,'endpoint_b_signed':sb,'centerline_crossing':cross}.items():evidence[p+'_'+key]=value
  print('R11_CURRENT_BAND_NATIVE '+json.dumps({k:v for k,v in row.items() if k!='worst'}),flush=True)
 result={'schemaVersion':1,'status':'passed_native_sample_capsules' if not errors else 'blocked','errors':errors,**inputs['identity'],'helperSha256':sha(__file__),'ringDependencyChecks':inputs['ringDependencyChecks'],'nativeInputs':native_rows,'rings':rows,'actualHeadBlenderBVHChecked':False,'savedBlend':False,'appearanceAccepted':False,'productionAcceptance':False,
  'limits':['Actual Blender native samples are checked as full finite segments with conservative maximum endpoint radii; unsampled native curvature between samples is not analytically bounded.','Current Blender head BVH, ring self/pair intersections, copied-data protection and actual post-install geometry must still pass before saving.']}
 return result,evidence

def live_head_and_ring_geometry(inputs,arrays=None):
 """Real current head BVH and band mesh intersections, in Blender REST."""
 import bpy
 from mathutils import Vector
 from mathutils.bvhtree import BVHTree
 observer=runpy.run_path(str(HERE/'observe_reference_tie_alignment_r11_v2.py'));a={};observer['collect_mesh'](bpy.data.objects['R11 Female head study'],'head',a,require_face_sets=True,require_actual_evaluation=True)
 actual=inputs['actual']
 if not np.array_equal(a['head_triangles'],actual['head_triangles']) or np.linalg.norm(a['head_world']-actual['head_world'],axis=1).max()>TOL:raise RuntimeError('Current evaluated head no longer matches the complete post-tie observation')
 head=BVHTree.FromPolygons([Vector(p) for p in a['head_world']],a['head_triangles'].tolist(),all_triangles=True);rings={};rows=[];errors=[]
 for ring in inputs['plan']['rings']:
  pre=ring['prefix'];obj=bpy.data.objects[ring['object']];out={};observer['collect_mesh'](obj,pre,out,require_actual_evaluation=True);world=out[pre+'_world'];tri=out[pre+'_triangles'];expected=inputs['candidate'][pre+'_candidate_world']
  if world.shape!=expected.shape or np.linalg.norm(world-expected,axis=1).max()>TOL or not np.array_equal(tri,inputs['candidate'][pre+'_triangles']):raise RuntimeError(pre+': evaluated installed ring differs from frozen candidate')
  tree=BVHTree.FromPolygons([Vector(p) for p in world],tri.tolist(),all_triangles=True);head_hits=tree.overlap(head);self_hits=[]
  for i,j in tree.overlap(tree):
   if i>=j or np.intersect1d(tri[i],tri[j]).size:continue
   self_hits.append([int(i),int(j)])
  triangles=world[tri];centers=triangles.mean(1);cover=np.linalg.norm(triangles-centers[:,None],axis=2).max(1);signed=[];dist=[]
  for p in centers:
   position,normal,index,d=head.find_nearest(Vector(p))
   if position is None:raise RuntimeError('Current head BVH nearest query failed')
   dist.append(float(d));signed.append(float(np.dot(p-np.asarray(position),np.asarray(normal))))
  dist=np.asarray(dist);signed=np.asarray(signed);lower=dist-cover
  if head_hits or self_hits or np.any(signed<0) or lower.min()<.0005:errors.append(pre+': current head/self-intersection or conservative whole-triangle 0.5 mm clearance gate failed')
  rows.append({'ring':pre,'headTriangleOverlaps':len(head_hits),'nonadjacentSelfTriangleOverlaps':len(self_hits),'minimumHeadTriangleCentroidDistanceMeters':float(dist.min()),'minimumHeadWholeTriangleDistanceLowerBoundMeters':float(lower.min()),'negativeHeadCentroidSigns':int((signed<0).sum()),'selfOverlapExamples':self_hits[:20]})
  rings[pre]=(tree,world)
  if arrays is not None:
   arrays[pre+'_installed_evaluated_world']=world;arrays[pre+'_installed_triangles']=tri;arrays[pre+'_head_centroid_distance']=dist;arrays[pre+'_head_centroid_signed']=signed;arrays[pre+'_head_triangle_cover_radius']=cover;arrays[pre+'_head_triangle_distance_lower_bound']=lower
 pairs=[]
 names=list(rings)
 for index,p in enumerate(names):
  for q in names[index+1:]:
   count=len(rings[p][0].overlap(rings[q][0]));pairs.append({'rings':[p,q],'triangleOverlaps':count})
   if count:errors.append(p+' / '+q+': ring meshes overlap')
 return {'errors':errors,'currentHeadMatchedObservation':True,'ringHeadAndSelfChecks':rows,'ringPairChecks':pairs,'method':'Exact current mesh BVH triangle overlap; centroid nearest-head distances minus each triangle cover radius give a conservative bound for every point of that triangle.'}

def install_checked(inputs,native_report,arrays=None):
 """Copy and edit only four mesh datablocks; roll them back on any failure.

 Caller must load the observed source freshly and put the rig in REST. The
 wrapper owns pose restoration, reporting, save and all client acceptance.
 """
 import bpy
 pin_check()
 if native_report.get('status')!='passed_native_sample_capsules' or native_report.get('errors'):raise RuntimeError('Actual current native sampled capsule validation has not passed')
 for k,v in inputs['identity'].items():
  if native_report.get(k)!=v:raise RuntimeError('Native validation is for another candidate/observation: '+k)
 if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Band installation requires the observed REST pose')
 if not bpy.data.filepath or sha(bpy.data.filepath)!=inputs['observation']['sourceSha256']:raise RuntimeError('Load the actually observed saved source; a derived source needs fresh native evidence')
 observer=runpy.run_path(str(HERE/'observe_reference_tie_alignment_r11_v2.py'));names={x['object'] for x in inputs['plan']['rings']};objects=[o for o in bpy.data.objects if o.type in ('MESH','CURVES')];before={o.name:observer['raw_contract'](o) for o in objects};olddata={};old_fake={};copies={};result=None
 for ring in inputs['plan']['rings']:
  p=ring['prefix'];obj=bpy.data.objects[ring['object']]
  if obj.type!='MESH' or obj.data.shape_keys or len(obj.data.vertices)!=256 or any(m.show_render or m.show_viewport for m in obj.modifiers):raise RuntimeError(p+': expected undeformed current ring mesh without shape keys')
  local=observer['coordinates'](obj);world=observer['world_positions'](local,obj.matrix_world)
  if np.linalg.norm(world-inputs['actual'][p+'_raw_world'],axis=1).max()>TOL or not np.allclose(np.asarray(obj.matrix_world),inputs['candidate'][p+'_matrix'],atol=1e-10,rtol=0):raise RuntimeError(p+': live ring moved since the actual observation')
  olddata[obj.name]=obj.data;old_fake[obj.name]=obj.data.use_fake_user
 try:
  for ring in inputs['plan']['rings']:
   p=ring['prefix'];obj=bpy.data.objects[ring['object']];old=olddata[obj.name];new=old.copy();new.name=old.name+' / R11 fitted band native verified';old.use_fake_user=True;obj.data=new;copies[obj.name]=new
   new.vertices.foreach_set('co',inputs['candidate'][p+'_candidate_local'].astype(np.float32).reshape(-1));new.update();obj.update_tag()
  bpy.context.view_layer.update();geometry=live_head_and_ring_geometry(inputs,arrays)
  after={o.name:observer['raw_contract'](o) for o in objects};protected={name for name in before if name not in names};changed=[name for name in protected if before[name]!=after[name]]
  if changed:raise RuntimeError('Protected non-ring data/positions/modifiers changed: '+', '.join(changed[:8]))
  # The copied ring data may only differ in positions and its data-block name.
  ring_changes={}
  for name in names:
   keys=[k for k in before[name] if before[name][k]!=after[name][k]];ring_changes[name]=keys
   allowed={'data','geometryKeysSha256'}
   unexpected=[k for k in keys if k not in allowed]
   if unexpected:raise RuntimeError('Unexpected ring contract change: '+name+' '+str(unexpected))
  if geometry['errors']:raise RuntimeError('; '.join(geometry['errors']))
  result={'installed':True,'savedBlend':False,'appearanceAccepted':False,'nativeObservationMatched':True,'onlyFourRingMeshCopiesChanged':True,'oldMeshDatablocksPreserved':[d.name for d in olddata.values()],'newMeshDatablocks':[d.name for d in copies.values()],'protectedNonRingObjects':len(protected),'ringContractChanges':ring_changes,'geometry':geometry,'rigidTransportApplied':False,'hairMoved':False}
  return result
 except Exception:
  for name,old in olddata.items():
   bpy.data.objects[name].data=old;old.use_fake_user=old_fake[name];bpy.data.objects[name].update_tag()
  bpy.context.view_layer.update()
  for data in copies.values():
   if data.users==0:bpy.data.meshes.remove(data)
  raise

def main():
 ap=argparse.ArgumentParser(description='Validate the frozen band against one actual post-tie native snapshot; never installs or saves Blender.');ap.add_argument('--plan',required=True);ap.add_argument('--observation',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();out=scoped(args.output).with_suffix('.json');npz=out.with_suffix('.npz')
 if out.exists() or npz.exists():raise RuntimeError('Independent unused validation outputs required')
 inputs=load_inputs(args.plan,args.observation);report,arrays=validate_native_snapshot(inputs)
 with npz.open('xb') as stream:np.savez_compressed(stream,**arrays)
 report.update(arrays=str(npz.relative_to(P)),arraysSha256=sha(npz),observation=str(inputs['observationPath'].relative_to(P)),plan=str(inputs['planPath'].relative_to(P)))
 with out.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
 print('R11_CURRENT_BAND_CHECK '+json.dumps({'status':report['status'],'errors':report['errors'],'report':str(out.relative_to(P))}),flush=True)
 if report['errors']:raise RuntimeError('Current native sample/band validation rejected candidate')
if __name__=='__main__':main()
