"""Isolated official long-hair GN preview. Never saves the donor blend."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import bpy
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();source=P/'assets/blender-hair-styles-bystedt-2023/hair_nodes-female_hair_styles.blend'
SOURCE_SHA='1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48'
args=argparse.ArgumentParser(description=__doc__);args.add_argument('--view',choices=['front','quarter','side'],default='quarter')
a=args.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out=P/'previews/reconstruction-r02/hair-donor-native-r11'/('donor-'+a.view+'.png');out.parent.mkdir(parents=True,exist_ok=True)
if out.exists() or out.with_suffix('.json').exists():raise RuntimeError('Preserve prior preview')
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source.resolve() or bpy.data.is_dirty:
    raise RuntimeError('Fresh official donor background source required')
if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Official donor checksum changed')
t0=time.monotonic();scene=bpy.data.scenes.new('R11 isolated official donor inspection')
for name in ('Long hair head','long hair growth mesh','long hair main','long hair strands'):
    scene.collection.objects.link(bpy.data.objects[name])
bpy.context.window.scene=scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=4
scene.render.resolution_x=720;scene.render.resolution_y=840;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out);scene.render.film_transparent=False
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0.;scene.view_settings.gamma=1.
world=bpy.data.worlds.new('R11 donor inspection world');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value=(.24,.24,.24,1.)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35;scene.world=world
target=Vector((5.,-.55,14.93))
cam_data=bpy.data.cameras.new('R11 donor inspection camera');cam=bpy.data.objects.new(cam_data.name,cam_data);scene.collection.objects.link(cam)
offset={'front':(0,-8,.12),'quarter':(5,-7,.35),'side':(8,0,.15)}[a.view]
cam.location=target+Vector(offset);cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
cam_data.type='ORTHO';cam_data.ortho_scale=4.0;cam_data.lens=70;scene.camera=cam
for name,location,power,size in [('key',(1,-5,18),1200,4.2),('fill',(9,-2.5,16),650,4.5),('rim',(6,3.2,18),1500,3.0)]:
    data=bpy.data.lights.new('R11 donor '+name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(data.name,data);scene.collection.objects.link(obj);obj.location=location
    obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
main=bpy.data.objects['long hair main'];ev=main.evaluated_get(deps)
metrics={'evaluatedType':ev.type,'storedGuideCurves':len(main.data.curves),'storedGuidePoints':len(main.data.position_data)}
if ev.type=='CURVES':
    metrics.update(evaluatedCurves=len(ev.data.curves),evaluatedPoints=len(ev.data.position_data))
    if metrics['evaluatedCurves']>250000 or metrics['evaluatedPoints']>18000000:
        raise RuntimeError('Native author GN exceeds this preview memory budget: '+json.dumps(metrics))
report={'source':str(source.relative_to(P)),'sourceSha256':SOURCE_SHA,'author':'Daniel Bystedt',
        'license':'CC-BY-SA (version not specified on official page or embedded credit)',
        'sourceUrl':'https://www.blender.org/download/demo-files/#hair','view':a.view,
        'kind':'Actual isolated official donor Blender render; not target character adaptation',
        'originalGuideAndGNParametersUnchanged':True,'metrics':metrics,'secondsBeforeRender':time.monotonic()-t0,
        'savedBlend':False,'targetAdapted':False,'appearanceAccepted':False}
print('R11_DONOR_NATIVE_PREVIEW '+json.dumps(report),flush=True)
bpy.ops.render.render(write_still=True)
if not out.is_file():raise RuntimeError('Preview missing')
if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Donor file changed unexpectedly')
report.update(output=str(out.relative_to(P)),outputSha256=hashlib.sha256(out.read_bytes()).hexdigest(),secondsTotal=time.monotonic()-t0)
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('R11_DONOR_NATIVE_PREVIEW_DONE '+json.dumps(report),flush=True)
