"""Read-only same-camera lash reflection/denoising comparison, physical BSDF."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--output',required=True)
a=pa.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve();source.relative_to((P/'work/reconstruction-r02').resolve());out.relative_to((P/'previews/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists():raise RuntimeError('Fresh saved source and new output directory required')
out.mkdir(parents=True);digest=hashlib.sha256(source.read_bytes()).hexdigest();scene=bpy.context.scene;scene.frame_set(1)
objects=[bpy.data.objects['R11 Adult lashes '+p+' '+s] for p in ('upper','lower') for s in ('L','R')]
old=objects[0].data.materials[0]
if old.name!='R11 light opaque black mascara' or any(o.data.materials[0]!=old for o in objects):raise RuntimeError('Explicit same current mascara required')
material=old.copy();material.name='TEMP same-lash physical roughness study';shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
for obj in objects:obj.data.materials[0]=material
bpy.context.view_layer.update();ev=bpy.data.objects['R11 Adult sclera L'].evaluated_get(bpy.context.evaluated_depsgraph_get());co=np.asarray([ev.matrix_world@v.co for v in ev.data.vertices]);target=(co.min(0)+co.max(0))*.5;target[2]+=.003
camera=bpy.data.objects.new('TEMP lash optics camera',bpy.data.cameras.new('TEMP lash optics camera'));scene.collection.objects.link(camera);camera.location=Vector(target+np.array([0.,-2.,0.]));camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.045;camera.data.dof.use_dof=False;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=128;scene.cycles.seed=29
scene.render.resolution_x=640;scene.render.resolution_y=440;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
report={'source':a.source,'sourceSha256':digest,'helperSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'frames':[],
    'sourceSaved':False,'diagnosticOnly':True,'cameraScale':.045,'resolution':[640,440],'samples':128,'seed':29,
    'scope':'Only lash roughness or image denoising changes between cases. Same geometry, thickness, opaque base colour, IOR 1.45, specular 0.5, lights/exposure. No final appearance acceptance.'}
for name,roughness,denoise in [('rough050-raw',.5,False),('rough050-denoised',.5,True),('rough085-raw',.85,False)]:
    shader.inputs['Roughness'].default_value=roughness;scene.cycles.use_denoising=denoise;bpy.context.view_layer.update();png=out/(name+'.png');scene.render.filepath=str(png)
    bpy.ops.render.render(write_still=True)
    report['frames'].append({'name':name,'roughness':roughness,'denoising':denoise,'sha256':hashlib.sha256(png.read_bytes()).hexdigest()})
    print('R11_LASH_REFLECTION_CASE '+name,flush=True)
if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed')
with (out/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
