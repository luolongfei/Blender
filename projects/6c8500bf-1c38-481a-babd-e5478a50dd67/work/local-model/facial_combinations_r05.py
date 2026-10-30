"""Small fixed-camera expression/gaze review, with actual current shape drivers."""
import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();args=sys.argv[sys.argv.index('--')+1:];label=args[0]
out=p/'previews/reconstruction-r02'/label;out.mkdir(parents=True,exist_ok=True)
scene=bpy.context.scene;rig=bpy.data.objects['Character_Rig'];h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
rig.animation_data.action=None;rig.data.pose_position='POSE'
for b in rig.pose.bones:b.matrix_basis.identity()
if h.data.shape_keys.animation_data:h.data.shape_keys.animation_data.action=None
if 'R04_native_hair_enabled' in rig:rig['R04_native_hair_enabled']=0
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=480;scene.render.resolution_y=540;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R05 Facial review camera',bpy.data.cameras.new('R05 Facial review camera'));scene.collection.objects.link(cam)
cam.data.type='ORTHO';cam.data.ortho_scale=.34;cam.data.dof.use_dof=False;scene.camera=cam
cases=[('closed',{'R02_blink_L':1,'R02_blink_R':1},0,0,False),
       ('half-blink',{'R02_blink_L':.5,'R02_blink_R':.5},0,0,False),
       ('smile-wink',{'R02_smile':.8,'R02_blink_L':1,'R03_squint_R':.6},0,0,True),
       ('open-mouth',{'R02_jaw_open':.65,'R02_mouth_wide':.15},0,0,True),
       ('gaze-up',{},8,-14,True),('gaze-down',{},-8,14,True)]
if len(args)>1:cases=[c for c in cases if c[0] in args[1].split(',')]
controls=['R02_blink_L','R02_blink_R','R03_squint_L','R03_squint_R','R02_jaw_open','R02_mouth_wide','R02_smile','R02_frown','R02_pucker','R02_lip_press','R02_brow_raise','R02_brow_furrow']
for name,values,yaw,pitch,quarter in cases:
    for n in controls:ks[n].value=.45 if n=='R02_smile' else 0
    for n,value in values.items():ks[n].value=value
    rig['R03_look_yaw']=yaw;rig['R03_look_pitch']=pitch;rig.update_tag();h.update_tag();bpy.context.view_layer.update()
    cam.location=(1.65 if quarter else 0,-4,1.405);cam.rotation_euler=(Vector((0,-.005,1.405))-cam.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True);print('R05_FACE_CASE',name,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),'cases':cases,'samples':12,'resolution':[480,540],'scope':'Actual facial combination images, not automatic visual acceptance'},indent=2),encoding='utf-8')
