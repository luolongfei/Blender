"""Read the matching baked cloth with its authored body action; diagnostic views."""
import bpy,sys,json,time,hashlib
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();args=sys.argv[sys.argv.index('--')+1:];label=args[0];frames=[int(x) for x in args[1].split(',')]
scene=bpy.context.scene;out=p/'previews/reconstruction-r02'/label;out.mkdir(parents=True,exist_ok=True)
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=400;scene.render.resolution_y=640;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R04 Body review camera',bpy.data.cameras.new('R04 Body review camera'));scene.collection.objects.link(cam)
cam.location=(1.6,-4.5,1.09);cam.rotation_euler=(Vector((0,0,.78))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=1.67;cam.data.dof.use_dof=False;scene.camera=cam
for frame in frames:
    scene.frame_set(frame);bpy.context.view_layer.update();scene.render.filepath=str(out/f'body-{frame:04d}.png')
    bpy.ops.render.render(write_still=True);print('R04_BODY_FRAME',frame,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),'action':bpy.data.objects['Character_Rig'].animation_data.action.name,'frames':frames,'samples':8,'resolution':[400,640],'scope':'Cloth/body key poses; hair remains head-attached in this first check'},indent=2),encoding='utf-8')
