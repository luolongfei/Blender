"""Actual R11 eye or mouth macro with unchanged source lighting and no DOF."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--case',choices=('eyes','mouth'),required=True)
parser.add_argument('--samples',type=int,default=128)
a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
source.relative_to((P/'work/reconstruction-r02').resolve());out.relative_to((P/'previews/reconstruction-r02').resolve())
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent saved source required')
out.mkdir(parents=True,exist_ok=True);png=out/(a.case+'.png');report=out/(a.case+'.json')
if png.exists() or report.exists():raise RuntimeError('Preserve earlier review')
source_sha=sha(source);scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
head=bpy.data.objects['R11 Female head study']
if a.case=='eyes':
    centers=[]
    for side in ('L','R'):
        ev=bpy.data.objects['R11 Adult sclera '+side].evaluated_get(dg)
        points=np.asarray([ev.matrix_world@v.co for v in ev.data.vertices])
        centers.append((points.min(0)+points.max(0))*.5)
    target=np.mean(centers,axis=0);target[2]+=.003;scale=.115
else:
    # Authored upper-lip point used only as framing center, not a contact test.
    target=np.array(head.matrix_world@head.data.vertices[8005].co,copy=True);target[2]-=.002;scale=.095
camera=bpy.data.objects.new('R11 actual '+a.case+' macro',bpy.data.cameras.new('R11 '+a.case+' macro'))
scene.collection.objects.link(camera);camera.location=Vector(target+np.array([0.,-2.,0.]))
camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type='ORTHO';camera.data.ortho_scale=scale;camera.data.dof.use_dof=False;scene.camera=camera
if not 16<=a.samples<=256:raise ValueError('Use bounded actual detail sample count')
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
scene.render.resolution_x=1200;scene.render.resolution_y=680;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(png);bpy.context.view_layer.update()
detail=runpy.run_path(str(Path(__file__).with_name('final_detail_review_r09.py')))
record={'source':a.source,'sourceSha256':source_sha,'case':a.case,'frame':1,'target':target.tolist(),'orthographicScale':scale,
        'resolution':[1200,680],'depthOfField':False,'lights':detail['_lights'](scene),'world':detail['_world'](scene),
        'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
        'samples':a.samples,'denoising':True,'appearanceAccepted':False,'scope':'Actual saved neutral R11 detail; not blink, smile or speech validation.'}
bpy.ops.render.render(write_still=True)
if not png.exists() or sha(source)!=source_sha:raise RuntimeError('Render or source proof failed')
record['imageSha256']=sha(png)
with report.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
print('R11_MACRO_RENDERED '+json.dumps({'case':a.case,'sourceSha256':source_sha,'image':str(png)}),flush=True)
