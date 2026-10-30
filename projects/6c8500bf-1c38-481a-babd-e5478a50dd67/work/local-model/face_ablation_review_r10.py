"""Read-only R10 face-design diagnosis: one temporary ablation and view.

These are causal observations, not adoptable adult-face candidates. Eye globes,
iris meshes, lid accessories and other attachments keep their current geometry;
reducing an old head-only design key can therefore expose mismatched contacts.
No Blender launch, source save, key coordinate edit or action clearing.
"""
import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
CASES=('current','expression-neutral','neutral-likeness-half')
VIEWS=('front','quarter','side')


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _key_coordinates(head):
    result={}
    for key in head.data.shape_keys.key_blocks:
        co=np.empty((len(key.data),3),np.float32);key.data.foreach_get('co',co.ravel())
        result[key.name]=hashlib.sha256(co.tobytes()).hexdigest()
    return result


def _mix(head):
    keys=head.data.shape_keys.key_blocks;arrays={}
    for key in keys:
        co=np.empty((len(key.data),3),np.float32);key.data.foreach_get('co',co.ravel());arrays[key.name]=co
    result=arrays[keys[0].name].astype(np.float64)
    for key in list(keys)[1:]:
        if not key.mute:result+=(arrays[key.name]-arrays[key.relative_key.name])*key.value
    return result


def _geometry_ids(bpy,head):
    return {o.name:{'object':o.as_pointer(),'data':o.data.as_pointer() if o.data else None,
        'parent':o.parent.as_pointer() if o.parent else None,'parentBone':o.parent_bone,
        'parentInverse':np.asarray(o.matrix_parent_inverse).tolist()}
        for o in bpy.data.objects if o!=head}


def run(source_relative,case,view,label,samples=24):
    import bpy
    from mathutils import Vector
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Use a disposable background Object Mode source')
    source=_path(source_relative)
    if Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Open the explicit saved source')
    if case not in CASES or view not in VIEWS:raise ValueError('One explicit diagnostic case/view required')
    if samples not in (24,32) or not re.fullmatch('[a-z0-9-]+',label):raise ValueError('Bounded samples and a new safe label required')
    digest=_sha(source);folder=_path('previews/reconstruction-r02/'+label);folder.mkdir(parents=True,exist_ok=True)
    image=folder/(case+'-'+view+'.png');report=image.with_suffix('.json')
    if image.exists() or report.exists():raise FileExistsError('Preserve previous actual diagnostic outputs')
    identity={'source':source_relative,'sourceSha256':digest,'mode':'R10 temporary head design ablation','headAttachmentsReconstructed':False}
    identity_file=folder/'face-ablation-source.json'
    if identity_file.exists():
        if json.loads(identity_file.read_text(encoding='utf-8'))!=identity:raise RuntimeError('Comparison label already identifies another source')
    else:
        with identity_file.open('x',encoding='utf-8') as f:json.dump(identity,f,ensure_ascii=False,indent=2)
    detail=_api('final_detail_review_r09');utility=_api('reference_final_stills_r08')
    rig,standing=detail['_matched_standing'](bpy);head=bpy.data.objects['R01 Head'];keys=head.data.shape_keys.key_blocks
    required=('R02_reference_likeness','R02_gentle_smile','R02_smile')
    if any(name not in keys for name in required):raise RuntimeError('Expected original design keys are absent')
    before_values={k.name:float(k.value) for k in keys};before_key_hashes=_key_coordinates(head)
    head_action=head.data.shape_keys.animation_data.action if head.data.shape_keys.animation_data else None
    rig_action=rig.animation_data.action;old_camera=bpy.context.scene.camera;camera=None
    before_other=_geometry_ids(bpy,head);before_rig=detail['_pose_state'](bpy,rig)
    before_mix=_mix(head);scene=bpy.context.scene;before_lights=detail['_lights'](scene)
    before_color={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
                  'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    eyes=detail['_eye_center'](bpy,utility)
    target=Vector(eyes);target.y+=.025;target.z-=.025
    desired={}
    if case!='current':desired.update(R02_gentle_smile=0.,R02_smile=0.)
    if case=='neutral-likeness-half':desired['R02_reference_likeness']=.5
    result={**identity,'case':case,'view':view,'sourceFrame':1,'matchedBodyHairCloth':standing,
        'originalKeyValues':before_values,'temporaryValueChanges':desired,
        'diagnosticOnly':True,'adoptableCandidate':False,'appearanceAccepted':False,'savedBlend':False,
        'limitations':['This is a head-key causal ablation, not a coordinated reconstruction.',
            'Current enlarged eye globes/irises, lid accessories, teeth and all attachment geometry remain unchanged.',
            'A reduced head-only likeness key may create eye/contact mismatch. Such artifacts cannot establish that a new adult face is acceptable.',
            'The old Basis also contains previous fitted proportions and is not a raw untouched Human Base Meshes asset.',
            'No material or lighting adjustment accompanies this comparison.']}
    try:
        for name,value in desired.items():keys[name].value=value
        bpy.context.view_layer.update()
        if any(abs(keys[name].value-value)>1e-6 for name,value in desired.items()):
            raise RuntimeError('A driver/action overrode the temporary key values; no action will be cleared')
        current_mix=_mix(head);delta=current_mix-before_mix
        basis=keys[0];raw=np.empty((len(basis.data),3),np.float32);basis.data.foreach_get('co',raw.ravel())
        roi=(raw[:,2]>1.30)&(raw[:,2]<1.62)&(np.abs(raw[:,0])<.12)&(raw[:,1]<.04)
        lengths=np.linalg.norm(delta[roi],axis=1)
        result['actualKeyValues']={k.name:float(k.value) for k in keys}
        result['headOnlyDisplacementMeters']={'max':float(lengths.max()),'p95':float(np.quantile(lengths,.95)),
            'mean':float(lengths.mean()),'over1mmCount':int((lengths>.001).sum()),'roiVertices':int(roi.sum())}
        camera=bpy.data.objects.new('R10 temporary face ablation camera',bpy.data.cameras.new('R10 temporary face ablation camera'))
        scene.collection.objects.link(camera);scene.camera=camera
        camera.data.type='ORTHO';camera.data.ortho_scale=.39;camera.data.dof.use_dof=False
        camera.data.clip_start=.01;camera.data.clip_end=100
        offset={'front':Vector((0,-3,0)),'quarter':Vector((2,-3,0)),'side':Vector((3,0,0))}[view]
        utility['_aim'](camera,target+offset,target)
        result['camera']={'target':list(target),'location':list(camera.location),'matrixWorld':np.asarray(camera.matrix_world).tolist(),
            'orthographicScale':.39,'baselineEyeCenter':eyes.tolist(),'headPoseUnchanged':True,
            'comparisonPolicy':'Same baseline eye anchors, view, framing, source lighting and exposure for every ablation case.'}
        scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_sequencer=False
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(image)
        result['lights']=before_lights;result['colorManagement']=before_color
        bpy.ops.render.render(write_still=True)
        if any(abs(keys[name].value-value)>1e-6 for name,value in desired.items()):raise RuntimeError('Render reevaluation changed temporary design values')
        result['image']=str(image.relative_to(project_path()));result['imageSha256']=_sha(image)
    finally:
        for name,value in before_values.items():keys[name].value=value
        bpy.context.view_layer.update();scene.camera=old_camera
        if camera:
            data=camera.data;bpy.data.objects.remove(camera,do_unlink=True);bpy.data.cameras.remove(data)
        result['keyValuesRestored']=all(abs(keys[name].value-value)<1e-6 for name,value in before_values.items())
        result['keyCoordinatesUnchanged']=_key_coordinates(head)==before_key_hashes
        result['otherObjectDataAndParentsUnchanged']=_geometry_ids(bpy,head)==before_other
        result['actionsUnchanged']=(rig.animation_data.action==rig_action and
            (head.data.shape_keys.animation_data.action if head.data.shape_keys.animation_data else None)==head_action)
        result['rigAndOriginalPoseRestored']=detail['_pose_state'](bpy,rig)==before_rig
        result['sourceUnchangedOnDisk']=_sha(source)==digest
        result['lightsUnchanged']=detail['_lights'](scene)==before_lights
        if not all(result[name] for name in ('keyValuesRestored','keyCoordinatesUnchanged','otherObjectDataAndParentsUnchanged',
                   'actionsUnchanged','rigAndOriginalPoseRestored','sourceUnchangedOnDisk','lightsUnchanged')):
            raise RuntimeError('Temporary ablation did not restore its read-only contract')
    with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return {'image':result['image'],'report':str(report.relative_to(project_path())),
            'diagnosticOnly':True,'adoptableCandidate':False}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--case',choices=CASES,required=True);p.add_argument('--view',choices=VIEWS,required=True)
    p.add_argument('--label',required=True);p.add_argument('--samples',type=int,default=24);a=p.parse_args(argv)
    print('R10_FACE_ABLATION '+json.dumps(run(a.source,a.case,a.view,a.label,a.samples),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
