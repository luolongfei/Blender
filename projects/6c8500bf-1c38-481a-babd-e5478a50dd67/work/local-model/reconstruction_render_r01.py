"""Bounded, identical-camera comparison. Run only in background Blender."""
import bpy, sys, json, hashlib
from pathlib import Path
from mathutils import Vector
root = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0, str(root))
from pipeline.context import project_path
P = project_path()
source_hash=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()
label = sys.argv[sys.argv.index('--')+1]
out = P/'previews'/'reconstruction-r01'/label
out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.frame_set(1)
rig = bpy.data.objects.get('Character_Rig')
if rig: rig.data.pose_position='REST'
for obj in bpy.data.objects:
    if obj.type=='MESH' and obj.data.shape_keys:
        for k in obj.data.shape_keys.key_blocks:
            if not k.name.startswith('R01'): k.value=0
scene.render.engine='CYCLES'
scene.cycles.device='CPU'
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED'
scene.render.threads=6
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R01 Comparison Camera',bpy.data.cameras.new('R01 Comparison Camera'))
scene.collection.objects.link(cam)
cam.data.type='ORTHO'; cam.data.dof.use_dof=False
scene.camera=cam
views=[('face-front',(0,-3,1.429),(0,-.005,1.429),.36,640,720),
 ('face-quarter',(3,-4,1.429),(0,-.005,1.429),.36,640,720),
 ('face-side',(3,0,1.429),(0,-.005,1.429),.36,640,720),
 ('full',(2.4,-4.5,1.3),(0,0,.81),1.8,640,800),
 ('back',(0,4,1.3),(0,0,.97),1.35,640,800)]
if len(sys.argv)>sys.argv.index('--')+2:
    names=sys.argv[sys.argv.index('--')+2].split(','); views=[v for v in views if v[0] in names]
records=[]
for name,pos,target,scale,w,h in views:
    cam.location=pos; cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.ortho_scale=scale
    scene.render.resolution_x=w; scene.render.resolution_y=h
    scene.render.filepath=str(out/(name+'.png'))
    bpy.ops.render.render(write_still=True)
    records.append({'view':name,'position':pos,'target':target,'scale':scale,'size':[w,h]})
    print('R01_RENDER_DONE',label,name,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':bpy.data.filepath,'sha256':source_hash,'blender':bpy.app.version_string,'samples':24,'views':records,'lights':[{ 'name':o.name,'matrix':[list(r) for r in o.matrix_world],'energy':o.data.energy} for o in scene.objects if o.type=='LIGHT']},indent=2),encoding='utf-8')
