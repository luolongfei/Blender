"""One-process, in-memory upper-lash neutral A/B on the current actual head.

Uses the frozen v4 root/path author and existing native binding, with a fresh
current head/globe capture. No external observation, late field, blink sweep,
model save or delivery claim. Temporary data and the exact marker are restored.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

PINS={
 'adult_upper_lash_regroom_r11_v4.py':'3b573fc79c69057e0123f8831373c850ac2b1700aae5837f9dfd28986b4e7b76',
 'face_fibre_binding_r11_v2.py':'47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49',
 'upper_lash_corrective_validation_r11.py':'36a8713e93a962504b6f1546fb466afd70e3a7af394c449b40176b32999e3c0a',
 'upper_lash_late_surface_r11.py':'2257f0c73f7b21d2bdbcb22cfb9320563c665189f0ba6b6c980777abc73efbeb',
}
REQUIRED_KEYS={'R11 blink L':0.,'R11 blink R':0.,'R11 gentle smile':0.,
 'R11 natural soft smile':0.,'R11 warm closed smile':0.,
 'R11 refined upper lid roll':1.,'R11 refined outer eye corners':0.,
 'R11 lower face contour v2':1.,'R11 relaxed lip seal':1.}


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def run(source_relative,source_sha,output_relative,samples=32):
 import bpy
 from mathutils import Vector
 project=project_path().resolve();source=(project/source_relative).resolve();out=(project/output_relative).resolve()
 source.relative_to(project/'work/reconstruction-r02');out.relative_to(project/'previews/reconstruction-r02')
 if source.suffix!='.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
  raise RuntimeError('Fresh explicit independent background source required')
 if sha(source)!=source_sha.lower():raise RuntimeError('Explicit current source hash mismatch')
 if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve previous preview outputs')
 if not 8<=samples<=32:raise ValueError('Bounded preview samples required')
 for filename,expected in PINS.items():
  if sha(Path(__file__).with_name(filename))!=expected:raise RuntimeError('Frozen preview dependency changed: '+filename)
 regroom=api('adult_upper_lash_regroom_r11_v4');binding=api('face_fibre_binding_r11_v2')
 qa=api('face_contact_audit_r11');detail=api('final_detail_review_r09');pose_api=api('production_uv_r08')
 scene=bpy.context.scene;head=bpy.data.objects[binding['HEAD']];keys=head.data.shape_keys.key_blocks
 if regroom['MARKER'] in scene or 'R11_upper_lash_late_surface' in scene:
  raise RuntimeError('Preview the retained old upper lashes, without a later regroom or late field')
 if binding['MARKER'] not in scene:raise RuntimeError('Existing recorded native fibre binding required')
 for name,value in REQUIRED_KEYS.items():
  if name not in keys or float(keys[name].value)!=value:raise RuntimeError('Saved neutral/static control differs: '+name)
 if 'R11 jaw open' in keys and float(keys['R11 jaw open'].value)!=0.:
  raise RuntimeError('Default relaxed jaw must remain zero')
 if head.get('R11 lower face continuity',None)!=0.:raise RuntimeError('Rejected old lower-face layer must remain zero')
 old_marker=str(scene[binding['MARKER']]);record=copy.deepcopy(json.loads(old_marker))
 old_data={name:bpy.data.objects[name].data for name in regroom['NAMES']}
 old_records={}
 for name in binding['OBJECTS']:
  obj=bpy.data.objects[name];data=obj.data
  old_records[name]={'data':data,'fakeUser':data.use_fake_user,'contract':binding['_curve_contract'](data),
                     'mapping':binding['_mapping_hash'](binding['_mapping'](data))}
 key_values={k.name:float(k.value) for k in keys};pose_before=pose_api['_pose_contract']()
 protected_before=binding['_protected'](head);external_before=regroom['external_contract'](bpy,binding)
 old_camera=scene.camera;old_render={name:getattr(scene.render,name) for name in ('engine','resolution_x','resolution_y','resolution_percentage','filepath')}
 old_cycles={name:getattr(scene.cycles,name) for name in ('device','samples','use_denoising')}
 old_format=scene.render.image_settings.file_format
 camera=camera_data=None;created=[];arrays={};success=False;failure=None
 report={'source':source_relative,'sourceSha256':source_sha.lower(),'helperSha256':sha(__file__),'pinnedHelpers':PINS,
  'evaluation':'Same source frame, temporary armature REST, saved neutral controls unchanged. Current evaluated L3 head and actual globes captured inside this process.',
  'savedKeyValues':key_values,'fixedHeadProperties':{'R11 lower face continuity':float(head['R11 lower face continuity'])},
  'oldBindingMarkerSha256':hashlib.sha256(old_marker.encode()).hexdigest(),'frames':[],'changes':[],
  'sourceSaved':False,'deliveryCandidate':False,'lateFieldInstalled':False,'dynamicStatesTested':[],
  'appearanceAccepted':False,'technicalAcceptance':False,'restoration':{},
  'limits':'Neutral diagnostic only. Passing local checks or a better image does not establish blink/other-expression compatibility. Formal v4 still needs final eye/midpath and fresh neutral observation.'}
 out.mkdir(parents=True,exist_ok=True)
 def controls_unchanged():
  if {k.name:float(k.value) for k in keys}!=key_values:raise RuntimeError('Preview changed a saved facial control')
  if head.get('R11 lower face continuity',None)!=0.:raise RuntimeError('Old lower-face scalar changed')
 def points(name):
  obj=bpy.data.objects[name];ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
  return binding['_transform'](binding['_array'](ev.data.position_data,'vector'),ev.matrix_world)
 def persist_attempt():
  (out/'attempt-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 def render_image(label):
  controls_unchanged();scene.render.filepath=str(out/(label+'.png'));bpy.ops.render.render(write_still=True)
  image=out/(label+'.png')
  if not image.is_file():raise RuntimeError('Preview image was not written')
  actual_matrix=[list(r) for r in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world]
  if actual_matrix!=report['camera']['matrixWorld']:raise RuntimeError('A/B camera changed')
  if detail['_lights'](scene)!=report['lights'] or detail['_world'](scene)!=report['world']:raise RuntimeError('A/B lighting changed')
  report['frames'].append({'label':label,'image':image.name,'sha256':sha(image),'actualKeyValues':{k.name:float(k.value) for k in keys}})
  persist_attempt();print('R11_NEUTRAL_LASH_PREVIEW_FRAME '+label,flush=True)
 try:
  with qa['rest_evaluation']():
   controls_unchanged();report['bindingBefore']=binding['validate'](neutral=False)
   hd=binding['_head_mesh'](head,triangulate=True)
   eyes={side:binding['_head_mesh'](bpy.data.objects['R11 Adult sclera '+side],triangulate=True) for side in ('L','R')}
   skin=regroom['bvh'](hd);globes={side:regroom['bvh'](eyes[side]) for side in ('L','R')}
   report['actualCurrentCapture']={'headTopologySha256':hd['topologySha256'],'headCounts':hd['counts'],
    'headWorldSha256':binding['_ah'](hd['world']),'eyes':{s:{'topologySha256':e['topologySha256'],'worldSha256':binding['_ah'](e['world'])} for s,e in eyes.items()}}
   unaffected={name:points(name) for name in binding['OBJECTS'] if name not in regroom['NAMES']}
   before_points={name:points(name) for name in regroom['NAMES']}
   centers=[(e['world'].min(0)+e['world'].max(0))*.5 for e in eyes.values()];target=np.mean(centers,axis=0);target[2]+=.003
   camera_data=bpy.data.cameras.new('TEMP R11 neutral upper lash review');camera=bpy.data.objects.new(camera_data.name,camera_data)
   scene.collection.objects.link(camera);camera.location=Vector(target+np.asarray((0.,-2.,0.)))
   camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
   camera_data.type='ORTHO';camera_data.ortho_scale=.115;camera_data.dof.use_dof=False;scene.camera=camera
   scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
   scene.render.resolution_x=1000;scene.render.resolution_y=570;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
   bpy.context.view_layer.update()
   report.update(camera={'matrixWorld':[list(r) for r in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world],
    'orthographicScale':.115,'depthOfField':False},resolution=[1000,570],samples=samples,lights=detail['_lights'](scene),world=detail['_world'](scene),
    colourManagement={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},groomVisibilityChanged=False)
   render_image('before')
   for side in ('L','R'):
    name='R11 Adult lashes upper '+side;obj=bpy.data.objects[name];prior=copy.deepcopy(record['objects'][name])
    reference=bpy.data.hair_curves[prior['originalData']]
    counts=np.asarray([c.points_length for c in obj.data.curves],np.int32)
    radius=binding['_array'](obj.data.attributes['radius'].data,'value',1).ravel();matrix=np.asarray(obj.matrix_world)
    scale=float(np.linalg.svd(matrix[:3,:3],compute_uv=False).max())
    world,design=regroom['new_paths'](side,before_points[name],counts,radius*scale,skin,globes[side],eyes[side]['world'])
    arrays[side+'_beforeWorld']=before_points[name];arrays[side+'_proposedWorld']=world;arrays[side+'_radius']=radius;arrays[side+'_counts']=counts
    new_reference=reference.copy();created.append(new_reference);new_reference.name='TEMP R11 neutral lash reference '+side;new_reference.use_fake_user=True
    new_reference.position_data.foreach_set('vector',np.asarray(binding['_transform'](world,np.linalg.inv(matrix)),np.float32).ravel())
    new_reference.materials.clear()
    for material in obj.data.materials:new_reference.materials.append(material)
    new_bound=new_reference.copy();created.append(new_bound);new_bound.name='TEMP R11 neutral lash bound '+side
    obj.data=new_reference;mapping,plan=binding['_plan'](obj,hd,skin)
    for i in range(3):
     for label,array in (('vertex',mapping['vertices']),('corner',mapping['corners'])):
      new_bound.attributes.new(binding['P']+label+'_'+str(i),'INT','CURVE').data.foreach_set('value',np.ascontiguousarray(array[:,i]))
    new_bound.attributes.new(binding['P']+'barycentric','FLOAT_VECTOR','CURVE').data.foreach_set('vector',mapping['weights'].ravel())
    new_bound.attributes.new(binding['P']+'frame_offset','FLOAT_VECTOR','POINT').data.foreach_set('vector',mapping['offsets'].ravel())
    obj.data=new_bound
    if not np.array_equal(binding['_array'](new_bound.attributes['radius'].data,'value',1).ravel(),radius):raise RuntimeError('Physical radius changed')
    if binding['_curve_contract'](new_reference)!=binding['_curve_contract'](new_bound):raise RuntimeError('Reference/bound curve contract differs')
    record['objects'][name]={**plan,'originalData':new_reference.name,'candidateData':new_bound.name,'originalFakeUser':True,
      'originalContractSha256':binding['_hash'](binding['_curve_contract'](new_reference)),
      'authoredPathRevision':{'helperSha256':sha(__file__),'sourceSha256':source_sha,'priorOriginalData':prior['originalData'],'priorBoundData':prior['candidateData'],'reason':'Temporary neutral preview only'}}
    arrays[side+'_boundTrianglesWorld']=hd['world'][mapping['vertices']]
    report['changes'].append({'object':name,'design':design,'bindingPlan':plan})
   scene[binding['MARKER']]=json.dumps(record,ensure_ascii=False);bpy.context.view_layer.update();controls_unchanged()
   report['bindingAfter']=binding['validate'](neutral=False)
   after_head=binding['_head_mesh'](head)
   if after_head['topologySha256']!=hd['topologySha256'] or not np.array_equal(after_head['world'],hd['world']):raise RuntimeError('Preview changed the evaluated head')
   for name,before in unaffected.items():
    if not np.array_equal(points(name),before):raise RuntimeError('Brow/lower-lash output changed: '+name)
   report['unaffectedFibresExact']=list(unaffected);report['neutralContact']={}
   for side in ('L','R'):
    obj=bpy.data.objects['R11 Adult lashes upper '+side]
    actual,radii,counts=regroom['sample_actual'](obj,binding)
    contacts=regroom['contact_report'](actual,radii,counts,skin,globes[side]);report['neutralContact'][side]=contacts
    arrays[side+'_actualNative33World']=actual;arrays[side+'_actualNative33Radii']=radii;arrays[side+'_actualNative33Counts']=counts
   persist_attempt()
   if any(info['nonRootPointsBeyondTolerance'] or info['rootPointsBeyondTolerance'] for side in report['neutralContact'].values() for info in side.values()):
    raise RuntimeError('Temporary neutral root/shaft exceeds unchanged physical contact limits')
   report['localNeutralChecksPassed']=True;render_image('after');success=True
 except Exception as exc:
  failure=exc;report['error']=str(exc)
 finally:
  restoration_errors=[]
  try:
   for name,data in old_data.items():bpy.data.objects[name].data=data
   scene[binding['MARKER']]=old_marker
   for data in reversed(created):
    data.use_fake_user=False
    if data.users==0:bpy.data.hair_curves.remove(data)
   scene.camera=old_camera
   if camera is not None:bpy.data.objects.remove(camera,do_unlink=True)
   if camera_data is not None and camera_data.users==0:bpy.data.cameras.remove(camera_data)
   for name,value in old_render.items():setattr(scene.render,name,value)
   for name,value in old_cycles.items():setattr(scene.cycles,name,value)
   scene.render.image_settings.file_format=old_format;bpy.context.view_layer.update()
   controls_unchanged()
   for name,row in old_records.items():
    data=bpy.data.objects[name].data
    if data!=row['data'] or data.use_fake_user!=row['fakeUser'] or binding['_curve_contract'](data)!=row['contract'] or binding['_mapping_hash'](binding['_mapping'](data))!=row['mapping']:
     restoration_errors.append('Original fibre data/attributes/fake-user differ: '+name)
   if str(scene[binding['MARKER']])!=old_marker:restoration_errors.append('Original binding marker text differs')
   if pose_api['_pose_contract']()!=pose_before:restoration_errors.append('Original pose differs')
   if binding['_protected'](head)!=protected_before:restoration_errors.append('Protected head differs')
   if regroom['external_contract'](bpy,binding)!=external_before:restoration_errors.append('Original external scene differs')
   if sha(source)!=source_sha.lower():restoration_errors.append('Saved source hash differs')
  except Exception as exc:restoration_errors.append(str(exc))
  report['restoration']={'exact':not restoration_errors,'errors':restoration_errors,'oldMarkerRestored':str(scene.get(binding['MARKER'],''))==old_marker}
  report['completed']=bool(success and not restoration_errors)
  if arrays:
   array_path=out/'neutral-arrays.npz';np.savez_compressed(array_path,**arrays)
   report['actualArrays']={'path':array_path.name,'sha256':sha(array_path)}
  persist_attempt()
  if report['completed']:(out/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
  if restoration_errors:raise RuntimeError('Preview restoration failed: '+json.dumps(restoration_errors)) from failure
 if failure is not None:raise failure
 print('R11_NEUTRAL_LASH_PREVIEW_DONE '+json.dumps({'output':output_relative,'images':len(report['frames']),'sourceSaved':False,'restored':True}),flush=True)
 return report


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
 parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=32)
 args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
 run(args.source,args.source_sha,args.output,args.samples)
