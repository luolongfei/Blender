"""Human walk visual review. Hide cloth cached for a different diagnostic clip."""
import bpy,sys,json,hashlib,time
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();args=sys.argv[sys.argv.index('--')+1:];label,action_name=args[:2]
frames=list(range(1,80,2)) if args[2]=='continuous' else [int(x) for x in args[2].split(',')]
scene=bpy.context.scene;rig=bpy.data.objects['Character_Rig'];action=bpy.data.actions[action_name]
rig.animation_data_create();rig.animation_data.action=action;rig.animation_data.action_slot=action.slots[0]
rig.data.pose_position='POSE';scene.render.fps=30
if 'R04_native_hair_enabled' in rig:rig['R04_native_hair_enabled']=0.
bpy.data.objects['R02 Cloth Skirt control'].hide_render=True
bpy.data.objects['Skirt_Sixteen_Pleats'].hide_render=False
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=480;scene.render.resolution_y=640;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
out=p/'previews/reconstruction-r02'/label;out.mkdir(parents=True,exist_ok=True)
cam=bpy.data.objects.new('R05 Mocap review camera',bpy.data.cameras.new('R05 Mocap review camera'));scene.collection.objects.link(cam)
cam.data.type='ORTHO';cam.data.ortho_scale=1.83;cam.data.dof.use_dof=False;scene.camera=cam
lights=[(o,o.location.copy()) for o in scene.objects if o.type=='LIGHT']
rows=[]
for frame in frames:
    scene.frame_set(frame);bpy.context.view_layer.update()
    root=rig.matrix_world@rig.pose.bones['hips'].head;offset=Vector((root.x,root.y,0))
    cam.location=offset+Vector((1.5,-4.5,1.04));cam.rotation_euler=(offset+Vector((0,0,.78))-cam.location).to_track_quat('-Z','Y').to_euler()
    for light,start in lights:light.location=start+offset
    scene.render.filepath=str(out/f'frame-{frame:04d}.png');bpy.ops.render.render(write_still=True)
    rows.append({'frame':frame,'trackedRootXY':list(offset[:2])});print('R05_MOCAP_FRAME',frame,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    'action':action_name,'frames':rows,'scope':'CMU retargeted body-motion review. Original skinned skirt shown; mismatched cloth and hair caches disabled. Secondary physics not validated by these images.'},indent=2),encoding='utf-8')
