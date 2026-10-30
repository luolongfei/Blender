"""Actual donor head form review; temporary clay materials, no source save."""
import argparse,bpy,hashlib,json,re,sys
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();source=P/'assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
if not bpy.app.background or '--disable-autoexec' not in sys.argv or Path(bpy.data.filepath).resolve()!=source.resolve():
    raise RuntimeError('Open only the official donor in a separate background process')
if source_sha!='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137':raise RuntimeError('Donor mismatch')
parser=argparse.ArgumentParser();parser.add_argument('--label',default='female-donor-form-r11');parser.add_argument('--light-scale',type=float,default=1.)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
if not re.fullmatch(r'[a-z0-9-]+',args.label) or not .1<=args.light_scale<=1.:raise ValueError('Bounded diagnostic label/light scale required')
out=P/'previews/reconstruction-r02'/args.label
if out.exists():raise RuntimeError('Preserve earlier form studies')
out.mkdir(parents=True)
body=bpy.data.objects['GEO-body_female_realistic'];eyes=[bpy.data.objects['GEO-body_female_realistic.eye.'+s] for s in ('L','R')]
def unhide(layer):
    layer.exclude=False;layer.hide_viewport=False
    for child in layer.children:unhide(child)
unhide(bpy.context.view_layer.layer_collection)
for collection in bpy.data.collections:collection.hide_render=False
for obj in bpy.data.objects:obj.hide_render=True
for obj in [body]+eyes:obj.hide_render=False;obj.hide_viewport=False;obj.hide_set(False)
for obj in [body]+eyes:
    material=bpy.data.materials.new('R11 temporary donor clay '+obj.name);material.use_nodes=True
    shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value=(.42,.33,.28,1) if obj==body else (.35,.35,.35,1)
    shader.inputs['Roughness'].default_value=.62
    obj.data.materials.clear();obj.data.materials.append(material)
    for face in obj.data.polygons:face.material_index=0;face.use_smooth=True
bpy.context.view_layer.update()
centers=[]
for obj in eyes:
    points=[obj.matrix_world@v.co for v in obj.data.vertices]
    centers.append(Vector([(min(p[i] for p in points)+max(p[i] for p in points))*.5 for i in range(3)]))
center=(centers[0]+centers[1])*.5;target=center+Vector((0,0,-.035))
scene=bpy.context.scene
scene.world=bpy.data.worlds.new('R11 temporary form world');scene.world.use_nodes=True
bg=scene.world.node_tree.nodes.get('Background');bg.inputs['Color'].default_value=(.35,.35,.35,1);bg.inputs['Strength'].default_value=.35
def aim(obj,point):obj.rotation_euler=(point-obj.location).to_track_quat('-Z','Y').to_euler()
for name,offset,energy,size in [('key',(-.5,-.7,.6),80,.65),('fill',(.5,-.4,.1),35,.8),('rim',(.35,.5,.5),70,.6)]:
    data=bpy.data.lights.new('R11 donor '+name,'AREA');data.energy=energy*args.light_scale;data.shape='DISK';data.size=size
    obj=bpy.data.objects.new(data.name,data);scene.collection.objects.link(obj);obj.location=target+Vector(offset);aim(obj,target)
camera_data=bpy.data.cameras.new('R11 donor form camera');camera_data.type='ORTHO';camera_data.ortho_scale=.36;camera_data.dof.use_dof=False
camera=bpy.data.objects.new(camera_data.name,camera_data);scene.collection.objects.link(camera);scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.resolution_x=640;scene.render.resolution_y=720
scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
scene.render.use_sequencer=False;scene.render.use_overwrite=False
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0.;scene.view_settings.gamma=1.
images=[]
for label,direction in [('front',(0,-1,0)),('quarter',(.65,-1,.05)),('side',(1,0,0))]:
    camera.location=target+Vector(direction).normalized()*1.5;aim(camera,target)
    image=out/(label+'.png');scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
    images.append({'view':label,'path':image.relative_to(P).as_posix(),'cameraMatrix':[list(r) for r in camera.matrix_world]})
if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise RuntimeError('Donor file changed')
(out/'report.json').write_text(json.dumps({'source':str(source.relative_to(P)),'sourceSha256':source_sha,
    'bodyObject':body.name,'eyeObjects':[o.name for o in eyes],'eyeCenters':[list(c) for c in centers],
    'actualMultiresLevels':[(m.levels,m.render_levels,m.total_levels) for m in body.modifiers if m.type=='MULTIRES'],
    'views':images,'scope':'Actual donor geometry including authored Multires 3, temporary neutral clay and blank eye spheres. Asset form diagnostic only; not a character candidate, not likeness or skin acceptance.',
    'lightEnergyScale':args.light_scale,'geometryEdited':False,'sourceSaved':False,'appearanceAccepted':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('FEMALE_DONOR_FORM_RENDERED')
