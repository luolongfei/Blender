"""Actual matched walk/cloth/hair playback, with bounded fresh-file checks."""
import bpy,sys,json,hashlib,runpy,re
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
args=sys.argv[sys.argv.index('--')+1:];mode=args[0]
if mode not in {'sparse','continuous'}:raise ValueError('Use sparse or continuous')
label=args[1] if len(args)>1 else 'r07'
if not re.fullmatch(r'[a-z0-9-]+',label):raise ValueError('Safe review stage label required')
source=Path(bpy.data.filepath);digest=hashlib.sha256(source.read_bytes()).hexdigest()
proof=p/'work/reconstruction-r02'/('combined-walk-reopen-'+label+'.json')
native=runpy.run_path(str(folder/'native_hair_playback_r04.py'))
native['enable_for_action'](action_name='R06 CMU 07_01 walk - cloth preroll',select=True,frame=1)
cache='work/reconstruction-r02/hair-walk-r07/R06-CMU-07_01-walk---cloth-preroll.guides.npz'
if mode=='sparse':
    cloth=runpy.run_path(str(folder/'walk_cloth_native_r07.py'))
    cloth_result=cloth['verify_reopened'](expected_relative=source.resolve().relative_to(p.resolve()).as_posix())
    hair_result=native['validate_evaluated'](cache_relative=cache,frames=[1,15.375,61,80])
    proof.write_text(json.dumps({'source':str(source),'sha256':digest,'cloth':cloth_result,'hair':hair_result},indent=2),encoding='utf8')
else:
    if not proof.exists() or json.loads(proof.read_text(encoding='utf8'))['sha256']!=digest:raise RuntimeError('Fresh-file sparse review must precede continuous render')
scene=bpy.context.scene;rig=bpy.data.objects['Character_Rig'];scene.render.fps=30
for name in ['R02 Cloth Skirt control','R06 Walk cloth','Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam']:
    obj=bpy.data.objects.get(name)
    if obj:obj.hide_render=True;obj.hide_set(True)
bpy.data.objects['R07 Native Walk Cloth playback'].hide_render=False
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=640 if mode=='continuous' else 480
scene.render.resolution_y=848 if mode=='continuous' else 640
scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
out=p/'previews/reconstruction-r02'/('combined-walk-'+mode+'-'+label);out.mkdir(parents=True,exist_ok=True)
cam=bpy.data.objects.new('R07 Combined walk camera',bpy.data.cameras.new('R07 Combined walk camera'));scene.collection.objects.link(cam)
cam.data.type='ORTHO';cam.data.ortho_scale=1.83;cam.data.dof.use_dof=False;scene.camera=cam
lights=[(o,o.location.copy()) for o in scene.objects if o.type=='LIGHT'];rows=[]
frames=range(1,81) if mode=='continuous' else [1,15,35,61]
for frame in frames:
    scene.frame_set(frame);bpy.context.view_layer.update()
    root=rig.matrix_world@rig.pose.bones['hips'].head;offset=Vector((root.x,root.y,0))
    cam.location=offset+Vector((1.5,-4.5,1.04));cam.rotation_euler=(offset+Vector((0,0,.78))-cam.location).to_track_quat('-Z','Y').to_euler()
    for light,start in lights:light.location=start+offset
    scene.render.filepath=str(out/f'frame-{frame:04d}.png');bpy.ops.render.render(write_still=True)
    rows.append({'frame':frame,'trackedRootXY':list(offset[:2])});print('R07_COMBINED_FRAME',frame,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':str(source),'sha256':digest,'fps':30,'mode':mode,'frames':rows,
    'scope':'CMU-derived body/head walk, authored blink, embedded native cloth simulation and matched embedded twin-tail guide dynamics; non-looping clip. Images still require visual contact review.'},indent=2),encoding='utf8')
