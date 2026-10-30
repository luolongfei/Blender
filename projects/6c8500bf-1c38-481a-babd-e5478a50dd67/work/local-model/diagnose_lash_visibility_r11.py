"""Unsaved red-emission diagnostic of the four current lash objects only."""
import argparse,hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--source',required=True);pa.add_argument('--output',required=True)
a=pa.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve();source.relative_to((P/'work/reconstruction-r02').resolve());out.relative_to((P/'previews/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh source required')
out.mkdir(parents=True,exist_ok=True);png=out/'red-lash-diagnostic.png';report=out/'red-lash-diagnostic.json'
if png.exists() or report.exists():raise RuntimeError('Preserve diagnostic')
digest=hashlib.sha256(source.read_bytes()).hexdigest();scene=bpy.context.scene;scene.frame_set(1)
material=bpy.data.materials.new('TEMP diagnostic red emission - never delivery');material.use_nodes=True;nt=material.node_tree;nt.nodes.clear();e=nt.nodes.new('ShaderNodeEmission');e.inputs['Color'].default_value=(.8,.005,.005,1.);e.inputs['Strength'].default_value=1.;o=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(e.outputs[0],o.inputs['Surface'])
changes=[]
for part in ('upper','lower'):
    for side in ('L','R'):
        obj=bpy.data.objects['R11 Adult lashes '+part+' '+side]
        changes.append({'object':obj.name,'previous':obj.data.materials[0].name});obj.data.materials[0]=material
bpy.context.view_layer.update();ev=bpy.data.objects['R11 Adult sclera L'].evaluated_get(bpy.context.evaluated_depsgraph_get());co=np.asarray([ev.matrix_world@v.co for v in ev.data.vertices]);target=(co.min(0)+co.max(0))*.5;target[2]+=.003
camera=bpy.data.objects.new('TEMP lash visibility camera',bpy.data.cameras.new('TEMP lash visibility camera'));scene.collection.objects.link(camera);camera.location=Vector(target+np.array([0.,-2.,0.]));camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.045;camera.data.dof.use_dof=False;scene.camera=camera
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True;scene.render.resolution_x=640;scene.render.resolution_y=440;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.filepath=str(png)
bpy.ops.render.render(write_still=True)
if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed')
with report.open('x',encoding='utf8') as f:json.dump({'source':a.source,'sourceSha256':digest,'imageSha256':hashlib.sha256(png.read_bytes()).hexdigest(),'changes':changes,'sourceSaved':False,'diagnosticOnly':True,'scope':'Red emission deliberately removes fibre optics; tests which current lash geometry appears in the actual image. Not a cosmetic candidate or physical render.'},f,indent=2)
print('R11_RED_LASH_DIAGNOSTIC '+str(png),flush=True)
