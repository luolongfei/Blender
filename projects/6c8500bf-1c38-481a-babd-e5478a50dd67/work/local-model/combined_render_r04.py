"""Reopen verification and serial CPU review of matching body/hair/cloth."""
import bpy,sys,json,hashlib,runpy,time
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();args=sys.argv[sys.argv.index('--')+1:]
label=args[0];frames=list(range(1,302,2)) if args[1]=='continuous' else [int(x) for x in args[1].split(',')]
scene=bpy.context.scene;out=p/'previews/reconstruction-r02'/label;out.mkdir(parents=True,exist_ok=True)
ns=runpy.run_path(str(p/'work/local-model/native_hair_playback_r04.py'))
cache='work/reconstruction-r02/hair-body-r04/R03-Full-body-review---authored-diagnostic.guides.npz'
ns['enable_for_action'](select=True,frame=1)
parity=ns['validate_evaluated'](cache,frames=[1,101.375,229,301])
cloth=bpy.data.objects['R02 Cloth Skirt control'];cm=next(m for m in cloth.modifiers if m.type=='CLOTH')
if not cm.point_cache.use_external or not Path(bpy.path.abspath(cm.point_cache.filepath)).is_dir():
    raise RuntimeError('Matching external cloth cache missing after reopen')
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=480;scene.render.resolution_y=640;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R04 Combined review camera',bpy.data.cameras.new('R04 Combined review camera'));scene.collection.objects.link(cam)
cam.location=(1.25,-4.5,1.05);cam.rotation_euler=(Vector((0,0,.79))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=1.79;cam.data.dof.use_dof=False;scene.camera=cam
start=time.monotonic()
for frame in frames:
    scene.frame_set(frame);bpy.context.view_layer.update();scene.render.filepath=str(out/f'frame-{frame:04d}.png')
    bpy.ops.render.render(write_still=True);print('R04_COMBINED_FRAME',frame,flush=True)
report={'source':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    'action':bpy.data.objects['Character_Rig'].animation_data.action.name,'frames':frames,'samples':8,'resolution':[480,640],
    'elapsedSeconds':time.monotonic()-start,'nativeHairReopenParity':parity,
    'clothCache':cm.point_cache.filepath,'clothExternal':True,
    'scope':'Authored body diagnostic with matching native tail dynamics and cloth cache. No motion-capture or final visual acceptance claim.'}
(out/'render-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
