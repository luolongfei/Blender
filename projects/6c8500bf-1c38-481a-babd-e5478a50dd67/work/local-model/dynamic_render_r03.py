"""Actual serial Cycles frames; body/face actions and cached native hair together."""
import bpy,sys,json,hashlib,runpy,time
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();args=sys.argv[sys.argv.index('--')+1:];label=args[0];mode=args[1] if len(args)>1 else 'sparse'
folder=p/'previews/reconstruction-r02'/label;folder.mkdir(parents=True,exist_ok=True)
scene=bpy.context.scene;rig=bpy.data.objects['Character_Rig'];rig.data.pose_position='POSE'
action=bpy.data.actions['R03 Performance review'];rig.animation_data.action=action
if hasattr(action,'slots'):rig.animation_data.action_slot=next(s for s in action.slots if s.target_id_type=='OBJECT')
scene.frame_set(1);bpy.context.view_layer.update()
hair=runpy.run_path(str(Path(__file__).with_name('secondary_hair_r02.py')))
cache='work/reconstruction-r02/hair-performance-r03/R03-Performance-review.guides.npz'
hair['attach'](cache)
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=400;scene.render.resolution_y=480;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
camera=bpy.data.objects.new('R03 Performance camera',bpy.data.cameras.new('R03 Performance camera'));scene.collection.objects.link(camera)
camera.location=(.10,-3,1.405);camera.rotation_euler=(Vector((0,-.005,1.405))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type='ORTHO';camera.data.ortho_scale=.345;camera.data.dof.use_dof=False;scene.camera=camera
frames=list(range(1,182,2)) if mode=='full' else [1,20,42,63,82,98,110,121,128,146,162,181]
if len(args)>2:frames=[int(f) for f in args[2].split(',')]
prefix='frame' if mode=='full' else 'key';start=time.time()
try:
    for frame in frames:
        target=folder/f'{prefix}-{frame:04d}.png'
        if target.exists():continue
        scene.frame_set(frame);bpy.context.view_layer.update();hair['apply_frame']()
        scene.render.filepath=str(target);bpy.ops.render.render(write_still=True)
        print('R03_DYNAMIC_FRAME',frame,round(time.time()-start,1),flush=True)
finally:hair['detach']()
(folder/(mode+'-render-report.json')).write_text(json.dumps({'source':bpy.data.filepath,'sourceSha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),'action':action.name,'hairCache':cache,'frames':frames,'sourceFps':30,'outputFps':15 if mode=='full' else None,'resolution':[400,480],'samples':8,'note':'Actual diagnostic Cycles frames; not final photoreal or client acceptance'},indent=2),encoding='utf-8')
print('R03_DYNAMIC_COMPLETE',time.time()-start,flush=True)
