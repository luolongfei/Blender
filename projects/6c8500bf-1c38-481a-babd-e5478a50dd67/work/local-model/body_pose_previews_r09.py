"""Four actual R09 body-pose stills with all previous dynamics suspended.

Explicit render_key_poses() only; background Blender required. No save/bake.
The retained original skinned skirt is a static garment proxy, so skirt/leg
intersections in these images are not a test of the future matching cloth.
"""
import hashlib
import json
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
STATIC_SKIRT=('Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam')
FRAMES=(1,55,151,247)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


@contextmanager
def _static_layers(rig):
    visibility=[];modifiers=[];drivers=[];head=bpy.data.objects.get('R01 Head')
    keys=head.data.shape_keys if head else None
    ad=keys.animation_data if keys else None
    action=ad.action if ad else None;slot=ad.action_slot if ad else None
    nla=[(track,track.mute) for track in ad.nla_tracks] if ad else []
    values={k.name:float(k.value) for k in keys.key_blocks} if keys else {}
    records={'hiddenDynamicCloth':[],'visibleStaticSkirt':[],'disabledHairModifiers':[],
             'frozenHeadShapeAction':action.name if action else None}
    if bpy.data.objects.get(STATIC_SKIRT[0]) is None:raise RuntimeError('Retained original static skirt is required')
    try:
        # Face morphs are frozen at their current values for a body-only review;
        # driver connections and all sculpt keys remain intact.
        if ad:ad.action=None
        for track,_ in nla:track.mute=True
        for obj in bpy.context.scene.objects:
            dynamic=(any(m.type=='CLOTH' for m in obj.modifiers) or
                obj.name in ('R02 Cloth Skirt control','R06 Walk cloth') or
                ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
            if dynamic or obj.name in STATIC_SKIRT:
                if dynamic and obj.name in STATIC_SKIRT:raise RuntimeError('Original skirt unexpectedly contains simulation')
                visibility.append((obj,obj.hide_render,obj.hide_viewport,obj.hide_get()))
                obj.hide_render=dynamic;obj.hide_viewport=dynamic;obj.hide_set(dynamic)
                records['hiddenDynamicCloth' if dynamic else 'visibleStaticSkirt'].append(obj.name)
            for modifier in obj.modifiers:
                is_hair=(modifier.type=='NODES' and modifier.name.startswith('R04 Native hair playback'))
                if modifier.type=='CLOTH' or is_hair:
                    modifiers.append((modifier,modifier.show_viewport,modifier.show_render))
                    if is_hair:
                        for prop in ('show_viewport','show_render'):
                            curve=obj.animation_data.drivers.find(modifier.path_from_id(prop)) if obj.animation_data else None
                            if curve:drivers.append((curve,curve.mute));curve.mute=True
                        records['disabledHairModifiers'].append({'object':obj.name,'modifier':modifier.name})
                    modifier.show_viewport=False;modifier.show_render=False
        yield records
    finally:
        for modifier,viewport,render in modifiers:modifier.show_viewport=viewport;modifier.show_render=render
        for curve,mute in drivers:curve.mute=mute
        for obj,render,viewport,hidden in visibility:
            obj.hide_render=render;obj.hide_viewport=viewport;obj.hide_set(hidden)
        if ad:
            ad.action=action
            if slot:ad.action_slot=slot
        for track,mute in nla:track.mute=mute
        if keys:
            for name,value in values.items():keys.key_blocks[name].value=value


def render_key_poses(action_name='R09 Body performance - authored squat bend reach',
                     output_relative='previews/reconstruction-r02/body-poses-r09'):
    if bpy is None or not bpy.app.background:raise RuntimeError('Use the explicit serial background driver')
    body=runpy.run_path(str(Path(__file__).with_name('body_performance_r09.py')))
    helper,controls,_=body['_modules']();rig=helper['_guard']();scene=bpy.context.scene
    action=bpy.data.actions.get(action_name)
    if action is None or 'r09_performance_report' not in action:raise RuntimeError('Verified independent R09 action required')
    out=(project_path()/output_relative).resolve();out.relative_to((project_path()/'previews').resolve())
    if out.exists() and any(out.iterdir()):raise RuntimeError('Use an empty preview directory; old images are retained')
    settings=[(scene,'camera'),(scene.render,'engine'),(scene.render,'resolution_x'),(scene.render,'resolution_y'),
        (scene.render,'resolution_percentage'),(scene.render,'filepath'),(scene.render,'threads_mode'),
        (scene.render,'threads'),(scene.render,'use_persistent_data'),(scene.render.image_settings,'file_format'),
        (scene.cycles,'device'),(scene.cycles,'samples'),(scene.cycles,'use_denoising')]
    before=[(owner,name,getattr(owner,name)) for owner,name in settings]
    camera=None;camera_data=None;rows=[];report=None
    source=Path(bpy.data.filepath);source_sha=_sha(source)
    try:
        with body['_preserve'](rig,helper,controls),_static_layers(rig) as layers:
            helper['_assign'](rig,action)
            camera_data=bpy.data.cameras.new('R09 temporary body-pose camera');camera_data.type='ORTHO'
            camera_data.ortho_scale=1.90;camera_data.dof.use_dof=False
            camera=bpy.data.objects.new('R09 temporary body-pose camera',camera_data);scene.collection.objects.link(camera)
            camera.location=(3.,-4.,1.12);target=Vector((0.,0.,.83))
            camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera
            scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
            scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
            scene.render.resolution_x=480;scene.render.resolution_y=640;scene.render.resolution_percentage=100
            scene.render.image_settings.file_format='PNG';out.mkdir(parents=True,exist_ok=True)
            for frame in FRAMES:
                scene.frame_set(frame);rig.update_tag();bpy.context.view_layer.update()
                filename=f'frame-{frame:04d}.png';scene.render.filepath=str(out/filename)
                bpy.ops.render.render(write_still=True)
                rows.append({'frame':frame,'label':body['KEY_POSES'][frame],'path':str(out/filename)})
                print('R09_BODY_POSE_RENDER '+str(frame),flush=True)
            report={'sourceBlend':str(source),'sourceFileSha256':source_sha,'action':action_name,'frames':rows,
                'layers':layers,'resolution':[480,640],'samples':12,'cameraPosition':list(camera.location),
                'cameraTarget':list(target),'orthoScale':1.90,'lightsAndMaterialsUnchanged':True,
                'scope':'Body pose preflight only: authored-rest hair and original skinned skirt; all old native hair/cloth and simulations disabled. Static skirt intersections do not certify or reject a future matched cloth solve.',
                'savedBlend':False,'accepted':False}
    finally:
        for owner,name,value in before:setattr(owner,name,value)
        if camera is not None:bpy.data.objects.remove(camera,do_unlink=True)
        if camera_data is not None and camera_data.users==0:bpy.data.cameras.remove(camera_data)
        rig.update_tag();bpy.context.view_layer.update()
    if report:
        report['originalPoseActionsVisibilityAndRenderSettingsRestored']=True
        (out/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print('Loaded only. Call render_key_poses() explicitly in an isolated background Blender.')
