"""Install the independently measured relaxed lip-seal candidate; no render.

Current source must retain accepted lip/chin v2=1 and rejected v1=0. Existing
front inner-lip intersections are explicitly compared; this candidate is not
complete oral/dental acceptance. The current actual support, not whole Blend
SHA, determines compatibility with disjoint hair/eye/hand work.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';WARM='R11 warm closed smile';CONTROL='R11 relaxed lip seal';MARKER='R11_relaxed_lip_seal_candidate'
PLAN='work/reconstruction-r02/relaxed-lip-seal-plan-r11.json'
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def path(rel,suffix):
 p=(project_path()/rel).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
 if p.suffix!=suffix:raise ValueError('Explicit independent reconstruction artifact required')
 return p
def snapshot(head):
 import bpy
 dg=bpy.context.evaluated_depsgraph_get();ev=head.evaluated_get(dg);mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg);a=api('lower_face_surface_observe_r11')['array']
 try:
  sizes=a(mesh.polygons,'loop_total',1,np.int32).ravel()
  if not np.all(sizes==4):raise RuntimeError('Current fixed-L3 quad head required')
  faces=a(mesh.loops,'vertex_index',1,np.int32).ravel().reshape(-1,4);matrix=np.asarray(ev.matrix_world,float)
  world=a(mesh.vertices,'co',3)@matrix[:3,:3].T+matrix[:3,3];mesh.calc_loop_triangles()
  groups=mesh.attributes.get('.sculpt_face_set')
  if groups is None:raise RuntimeError('Actual anatomical face sets required')
  return {'world':world,'faces':faces,'groups':a(groups.data,'value',1,np.int32).ravel(),'matrix':matrix,
   'triangles':a(mesh.loop_triangles,'vertices',3,np.int32),'trianglePolygonIds':a(mesh.loop_triangles,'polygon_index',1,np.int32).ravel()}
 finally:ev.to_mesh_clear()
def prepare(source_relative,output_relative,plan_relative=PLAN):
 import bpy
 source=path(source_relative,'.blend');output=path(output_relative,'.blend');planpath=path(plan_relative,'.json')
 reportpath=output.with_suffix('.relaxed-lip-seal.json');attempt=output.with_suffix('.relaxed-lip-seal-attempt.json');payload=output.with_suffix('.relaxed-lip-seal.npz')
 if source==output or any(p.exists() for p in [output,reportpath,attempt,payload]) or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh saved background source and independent output required')
 head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
 if CONTROL in keys or MARKER in bpy.context.scene or WARM not in keys:raise RuntimeError('Existing warm smile and a fresh relaxed seal required')
 if head.get('R11 lower face continuity',None)!=0. or 'R11 lower face contour v2' not in keys or abs(keys['R11 lower face contour v2'].value-1.)>1e-7:raise RuntimeError('Retain old v1 zero and current lip/chin v2 one')
 if any(abs(k.value)>1e-7 for k in keys if 'blink' in k.name.lower() or 'smile' in k.name.lower() or k.name=='R11 jaw open'):raise RuntimeError('Neutral current blink/smile/jaw required')
 if 'R11 upper oral clearance' in keys and abs(keys['R11 upper oral clearance'].value)>1e-7:raise RuntimeError('This field was not measured with upper oral clearance active; new actual sampling required')
 plan=json.loads(planpath.read_text(encoding='utf8'));arraypath=path(plan['arrays'],'.npz')
 if sha(arraypath)!=plan['arraysSha256']:raise RuntimeError('Frozen relaxed-seal field changed')
 with np.load(arraypath) as z:data={k:z[k].copy() for k in z.files}
 refpath=path(plan['referenceActualArrays'],'.npz')
 if sha(refpath)!=plan['referenceActualArraysSha256']:raise RuntimeError('Actual current reference changed')
 with np.load(refpath) as z:reference={s:z['afterWarm'+str(s)+'World'].copy() for s in [0.,.5,1.]}
 uv=api('production_uv_r08');natural=api('natural_smile_adult_r11');groom=api('posterior_groom_r09');face=api('runtime_face_refresh_r08');shared=api('upper_lid_relief_install_r11');qa=api('face_contact_audit_r11')
 before_keys=natural['_key_records'](head.data,uv);before_data=groom['_data_hash'](head.data);before_materials=[face['_material_contract'](x.material) if x.material else None for x in head.material_slots]
 before_modifiers=shared['modifier_contract'](head,uv);before_external=shared['external_contract'](bpy,head);active_key=head.active_shape_key_index;source_sha=sha(source)
 modifier=group=cache=mesh=key=None;success=False;report={'source':source_relative,'sourceSha256':source_sha,'plan':plan_relative,'planSha256':sha(planpath),'field':plan,'actualChecks':[],'dependencies':[],
  'candidateSaved':False,'technicalAcceptance':False,'appearanceAccepted':False,'helperHashes':{n:sha(Path(__file__).with_name(n)) for n in ['relaxed_lip_seal_install_r11_v2.py','relaxed_lip_seal_nodes_r11_v2.py','relaxed_lip_seal_surface_r11.py','relaxed_lip_seal_contact_r11.py','relaxed_lip_seal_visibility_r11.py']}}
 diagnostic={k:v for k,v in data.items()};ids=data['originalVertexIds'];deps=data['dependencyOriginalVertexIds'];offset=np.zeros_like(reference[0.]);offset[ids]=data['worldOffsets'];active=np.any(offset!=0,1);upper_offset=np.zeros_like(offset);upper_offset[ids]=data['upperWorldOffsets']
 def persist():
  np.savez_compressed(payload,**diagnostic);report['arrays']=str(payload.relative_to(project_path()));report['arraysSha256']=sha(payload);attempt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 def values(warm,control=None):
  keys[WARM].value=float(warm)
  if control is not None:head.data.shape_keys.key_blocks[CONTROL].value=float(control)
  head.update_tag();bpy.context.view_layer.update()
 try:
  with qa['rest_evaluation']():
   baseline={}
   for warm in [0.,.5,1.]:
    values(warm);baseline[warm]=snapshot(head);now=baseline[warm];diagnostic['beforeWarm'+str(warm)+'World']=now['world'];diagnostic['beforeWarm'+str(warm)+'Triangles']=now['triangles'];diagnostic['beforeWarm'+str(warm)+'TrianglePolygonIds']=now['trianglePolygonIds'];persist()
    if not np.array_equal(now['faces'],data['fullFaces']):raise RuntimeError('Current actual topology differs from measured fixed L3 head')
    error=float(np.linalg.norm(now['world'][deps]-reference[warm][deps],axis=1).max())
    if error>2e-7:raise RuntimeError('Actual lip support/neighborhood differs from measured 31df: '+str(error))
    if not np.array_equal(now['groups'][data['roiOriginalFaceIds']],data['roiGroups']):raise RuntimeError('Observed mouth face regions changed')
    report['dependencies'].append({'warm':warm,'actualVertices':len(deps),'maximumMeters':error});persist()
   values(0.);key=head.shape_key_add(name=CONTROL,from_mix=False);key.relative_key=head.data.shape_keys.reference_key
   raw=np.empty((len(head.data.vertices),3),np.float32);head.data.shape_keys.reference_key.data.foreach_get('co',raw.ravel());key.data.foreach_set('co',raw.ravel());key.slider_min=0.;key.slider_max=1.;key.value=0.
   modifier,group,cache,mesh=api('relaxed_lip_seal_nodes_r11_v2')['build'](head,len(offset),ids,data['worldOffsets'],baseline[0.]['matrix'],data['upperWorldOffsets'])
   values(0.,0.);disabled=snapshot(head);diagnostic['disabledWorld']=disabled['world'];persist()
   if not np.array_equal(disabled['world'],baseline[0.]['world']):raise RuntimeError('Zero seal must return exact source surface')
   visibility=api('relaxed_lip_seal_visibility_r11');grid=visibility['union_grid']([visibility['grid_for'](s['world'],s['faces'],s['groups']) for s in baseline.values()])
   for warm,control in [(0.,.5),(0.,1.),(.5,1.),(1.,.5),(1.,1.)]:
    values(warm,control);after=snapshot(head);before=baseline[warm];tag=str(warm)+'_'+str(control);diagnostic['after_'+tag+'World']=after['world'];diagnostic['after_'+tag+'Triangles']=after['triangles'];persist()
    delta=after['world']-before['world'];actual=np.linalg.norm(delta,axis=1);error=float(np.linalg.norm(delta-(offset*control+upper_offset*control*(1-control)),axis=1).max())
    outside=bool(np.array_equal(after['world'][~active],before['world'][~active]));topology=bool(np.array_equal(after['faces'],before['faces']))
    if not np.array_equal(after['groups'],before['groups']) or not topology or error>3e-7 or not outside or abs(delta[:,0]).max()>2e-7:raise RuntimeError('Native seal differs from exact reviewed topology/support/width/vector')
    quality=api('lower_face_continuity_surface_r11')['quality'](before['world'],after['world'],before['faces'],active)
    row={'warm':warm,'seal':control,'maximumMeters':float(actual.max()),'expectedVectorErrorMeters':error,'outsideSupportExact':outside,'quality':quality}
    report['actualChecks'].append(row);persist()
    if quality['nonpositiveNormalCount'] or quality['minimumNormalAlignment']<.40 or quality['triangleAreaRatioRange'][0]<.50 or quality['triangleAreaRatioRange'][1]>2.30:raise RuntimeError('Actual lip unrolling folds/strains beyond measured range')
    # Depth/area and newly changed pair IDs are assessed independently; counts
    # never cancel a new contact region against removed baseline defects.
    row['contact']=api('relaxed_lip_seal_contact_r11')['contact'](before['world'],after['world'],before['faces'],before['groups'],after['triangles'],active,data['neutralCageWorld'],before_triangles=before['triangles']);persist()
    if not row['contact']['guardPassed']:raise RuntimeError('New contact region or worsened real section depth/area')
    for label,snap in [('before',before),('after',after)]:
     row['visibility'+label.title()]=visibility['measure'](snap['world'],snap['faces'],snap['groups'],grid,snap['triangles'],snap['trianglePolygonIds'])[0]
    persist();print('R11_RELAXED_LIP_SEAL_ACTUAL '+json.dumps({'warm':warm,'seal':control,'maximumMeters':float(actual.max())}),flush=True)
   values(0.,1.);head.active_shape_key_index=active_key
  after_keys=natural['_key_records'](head.data,uv);own=after_keys['blocks'].pop(CONTROL)
  if after_keys!=before_keys or groom['_data_hash'](head.data)!=before_data:raise RuntimeError('Existing raw head/keys/drivers changed')
  if own['coordinatesSha256']!=before_keys['blocks'][head.data.shape_keys.reference_key.name]['coordinatesSha256']:raise RuntimeError('Public control is not zero displacement')
  if [face['_material_contract'](x.material) if x.material else None for x in head.material_slots]!=before_materials or shared['modifier_contract'](head,uv,modifier)!=before_modifiers or shared['external_contract'](bpy,head,cache)!=before_external:raise RuntimeError('Existing materials/modifiers/external scene changed')
  if sha(source)!=source_sha:raise RuntimeError('Original source changed')
  report.update(control=CONTROL,controlType='Zero-displacement Key driving native fixed-L3 field',controlValue=1.,jawAttenuation=api('relaxed_lip_seal_nodes_r11_v2')['connect_jaw_control'](head,group),
   actualLocalGeometryChecksPassed=True,preexistingOralIntersectionsRemain=True,technicalAcceptance=False,appearanceAccepted=False,
   limitations=['Full-seal visible appearance needs the same-source A/B review.', 'Existing nonadjacent corner intersections remain and are not dental acceptance.', 'Front group7/8 intentionally move; old raw-cage or dental checks cannot certify this L3 inner-lip surface.', 'If jaw is absent, A new jaw successor must connect_jaw_control(head) inside its prepare transaction after Key creation, sample real upstream per jaw value with jaw GN disabled, validate downstream rotation, and save only after jaw 0/.03/.06/.12/1 and dental clearance. Old jaw prepare reports are incompatible.'])
  persist();bpy.context.scene[MARKER]=json.dumps(report)
  if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent seal save failed')
  report.update(candidateSaved=True,output=str(output.relative_to(project_path())),outputSha256=sha(output));reportpath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');success=True
  print('R11_RELAXED_LIP_SEAL_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True);return report
 except Exception as exc:report['guardError']=str(exc);persist();raise
 finally:
  if not success:
   keys[WARM].value=0.;head.active_shape_key_index=active_key
   if modifier is not None:head.modifiers.remove(modifier)
   if key is not None:head.shape_key_remove(key)
   if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
   if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
   if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
   if group is not None and group.users==0:bpy.data.node_groups.remove(group)
   bpy.context.view_layer.update()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',default=PLAN)
 a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.plan)
