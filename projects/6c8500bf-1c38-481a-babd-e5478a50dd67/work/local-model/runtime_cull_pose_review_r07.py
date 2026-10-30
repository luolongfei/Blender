"""Bounded collar/wrist QA using the existing authored diagnostic action.

No work on import; run(render=False) measures three actual posed camera targets.
run(render=True) additionally renders three sequential 480x540 CPU stills.
No action, keyframe, source Blend or simulation cache is created or saved.
"""
import hashlib,json,runpy,sys
from pathlib import Path
try:
    import bpy
    from mathutils import Vector
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
ACTION='R03 Full body review - authored diagnostic'
CASES=(('raised-collar',67,'neck',(0.,0.,.015),(2.,-4.,.8),.43),
       ('bent-collar-side',127,'neck',(0.,0.,.01),(4.,1.5,.7),.34),
       ('raised-left-wrist',67,'leftHand',(0.,0.,0.),(1.,-3.,1.),.19))
BONES=('spine','chest','neck','head','leftShoulder','rightShoulder','leftUpperArm','rightUpperArm',
       'leftLowerArm','rightLowerArm','leftHand','rightHand')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(render=False,output_relative='previews/reconstruction-r02/runtime-cull-poses-r07'):
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Saved background runtime Object Mode required')
    p=project_path();source=Path(bpy.data.filepath).resolve();source.relative_to((p/'work/runtime-r06').resolve())
    if not source.is_file():raise RuntimeError('Saved runtime source required')
    out=(p/output_relative).resolve();out.relative_to((p/'previews').resolve())
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new empty review directory')
    rig=bpy.data.objects['Character_Rig'];head=bpy.data.objects['RT06 Face / R01 Head'];scene=bpy.context.scene
    if not head.get('RT07_cull_candidate_unreviewed'):raise RuntimeError('This check expects the explicit culled runtime candidate')
    if abs(scene.render.fps/scene.render.fps_base-30)>1e-6:raise RuntimeError('Existing diagnostic requires 30 fps')
    action=bpy.data.actions.get(ACTION)
    if action is None or any(name not in rig.pose.bones for name in BONES):raise RuntimeError('Diagnostic action/bones missing')
    helper=runpy.run_path(str(Path(__file__).with_name('full_body_review_r03.py')))
    assembly=runpy.run_path(str(Path(__file__).with_name('runtime_assembly_r06.py')))
    inventory_rig,meshes=assembly['current_meshes']()
    if inventory_rig!=rig:raise RuntimeError('Runtime inventory rig mismatch')
    for obj in meshes:
        skins=[m for m in obj.modifiers if m.type=='ARMATURE' and (m.show_viewport or m.show_render)]
        if len(skins)!=1 or skins[0].object!=rig:raise RuntimeError('Expected one runtime skin: '+obj.name)
        if any(m.type in {'CLOTH','NODES','MESH_SEQUENCE_CACHE'} and (m.show_viewport or m.show_render) for m in obj.modifiers):
            raise RuntimeError('Runtime review cannot include playback/simulation modifiers: '+obj.name)
    manifest_path=p/'work/reconstruction-r02/full-body-review-r03.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['action']!=ACTION:raise RuntimeError('Diagnostic provenance mismatch')
    authored={int(row['frame']):row for row in manifest['keyPoses']}
    if any(case[1] not in authored for case in CASES):raise RuntimeError('Diagnostic pose provenance missing')
    source_sha=_sha(source);dirty=bool(bpy.data.is_dirty)
    rig.animation_data_create();ad=rig.animation_data
    old_action=ad.action;old_slot=ad.action_slot;old_frame=(scene.frame_current,scene.frame_subframe)
    old_pose=rig.data.pose_position;old_basis={b.name:b.matrix_basis.copy() for b in rig.pose.bones}
    nla=[(track,bool(track.mute)) for track in ad.nla_tracks]
    master=rig.pose.bones['CTRL_R02_Master'];switches={k:master[k] for k in ('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R') if k in master}
    native={k:rig[k] for k in ('R04_native_hair_enabled',) if k in rig}
    paused=[]
    settings=[(scene,'camera'),(scene.render,'engine'),(scene.render,'resolution_x'),(scene.render,'resolution_y'),
        (scene.render,'resolution_percentage'),(scene.render,'filepath'),(scene.render,'threads_mode'),(scene.render,'threads'),
        (scene.render,'use_persistent_data'),(scene.render.image_settings,'file_format'),
        (scene.cycles,'device'),(scene.cycles,'samples'),(scene.cycles,'use_denoising')]
    old_settings=[(item,name,getattr(item,name)) for item,name in settings]
    camera=None;camera_data=None;report=None
    try:
        # Hidden retained production objects must not trigger a cloth/GN solve
        # merely because the runtime review changes the current frame.
        for obj in bpy.data.objects:
            if obj in meshes or not obj.hide_render:continue
            for modifier in obj.modifiers:
                if modifier.type in {'CLOTH','NODES','MESH_SEQUENCE_CACHE'} and (modifier.show_viewport or modifier.show_render):
                    paused.append((obj,modifier,modifier.show_viewport,modifier.show_render))
                    modifier.show_viewport=False;modifier.show_render=False
        for track,_ in nla:track.mute=True
        for key in native:rig[key]=0.
        helper['_assign'](rig,action);rig.data.pose_position='POSE'
        camera_data=bpy.data.cameras.new('R07 temporary runtime seam camera');camera_data.type='ORTHO';camera_data.dof.use_dof=False
        camera=bpy.data.objects.new(camera_data.name,camera_data);scene.collection.objects.link(camera);scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
        scene.render.resolution_x=480;scene.render.resolution_y=540;scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG';out.mkdir(parents=True,exist_ok=True)
        rows=[]
        for label,frame,bone,offset,direction,scale in CASES:
            scene.frame_set(frame);rig.update_tag();bpy.context.view_layer.update()
            evaluated=rig.evaluated_get(bpy.context.evaluated_depsgraph_get())
            target=evaluated.matrix_world@evaluated.pose.bones[bone].head+Vector(offset)
            camera.location=target+Vector(direction).normalized()*1.4
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler();camera_data.ortho_scale=scale
            row={'name':label,'frame':frame,'action':action.name,'authoredPose':authored[frame],
                'cameraPosition':list(camera.location),'cameraTarget':list(target),'orthoScale':scale,
                'evaluatedBones':{name:{'head':list(evaluated.matrix_world@evaluated.pose.bones[name].head),
                    'tail':list(evaluated.matrix_world@evaluated.pose.bones[name].tail),
                    'matrixWorld':[list(r) for r in evaluated.matrix_world@evaluated.pose.bones[name].matrix]} for name in BONES},
                'actualIKSwitches':{k:float(master[k]) for k in switches},'rendered':False}
            if render:
                filename=out/(label+'.png');scene.render.filepath=str(filename)
                result=bpy.ops.render.render(write_still=True)
                if 'FINISHED' not in result or not filename.is_file():raise RuntimeError('Review render failed: '+label)
                row.update(rendered=True,image=str(filename),imageSha256=_sha(filename),resolution=[480,540],samples=12)
            rows.append(row);print('RT07_CULL_POSE',label,frame,'rendered' if render else 'measured',flush=True)
        report={'sourceBlend':str(source),'sourceFileSha256':source_sha,'sourceHadUnsavedChanges':dirty,
            'actionProvenance':str(manifest_path.relative_to(p)),'actionProvenanceSha256':_sha(manifest_path),
            'actionDescription':manifest['source'],'blenderVersion':bpy.app.version_string,
            'meshCount':len(meshes),'pausedHiddenModifiers':[{'object':o.name,'modifier':m.name,'type':m.type} for o,m,_,_ in paused],
            'cases':rows,'savedBlend':False,'accepted':False,
            'scope':'Three finite collar/neck and left-wrist views of existing authored arm-raise and bend poses. No simulation, automatic intersection test or bilateral/all-motion acceptance.'}
    finally:
        helper['_assign'](rig,old_action)
        if old_slot:ad.action_slot=old_slot
        for track,muted in nla:track.mute=muted
        rig.data.pose_position=old_pose
        scene.frame_set(old_frame[0],subframe=old_frame[1])
        for name,basis in old_basis.items():rig.pose.bones[name].matrix_basis=basis
        for key,value in switches.items():master[key]=value
        for key,value in native.items():rig[key]=value
        for obj,modifier,viewport,rendered in paused:modifier.show_viewport=viewport;modifier.show_render=rendered
        for item,name,value in old_settings:setattr(item,name,value)
        if camera is not None:bpy.data.objects.remove(camera,do_unlink=True)
        if camera_data is not None and camera_data.users==0:bpy.data.cameras.remove(camera_data)
        rig.update_tag();bpy.context.view_layer.update()
    if _sha(source)!=source_sha:raise RuntimeError('Saved source file changed during review')
    report['poseAndRenderSettingsRestored']=True
    (out/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':print('Loaded only. run(render=False) measures poses; explicit render=True writes three small stills without saving the Blend.')
