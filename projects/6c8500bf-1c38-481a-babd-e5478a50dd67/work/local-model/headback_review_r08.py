"""Fixed close back camera for genuine production groom inspection."""
import hashlib,json,runpy
from pathlib import Path
import bpy
from mathutils import Vector
api=runpy.run_path(str(Path(__file__).with_name('reference_final_stills_r08.py')))
p=api['project_path']();source=Path(bpy.data.filepath)
out=p/'previews/reconstruction-r02/source-headback-r08'
if out.exists():raise RuntimeError('Retain previous review')
digest=hashlib.sha256(source.read_bytes()).hexdigest()
neutral=api['_neutral'](bpy.data.objects['Character_Rig'])
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6
scene.render.resolution_x=640;scene.render.resolution_y=720;scene.render.resolution_percentage=100
scene.render.use_sequencer=False;scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R08 Back hair review',bpy.data.cameras.new('R08 Back hair review'));scene.collection.objects.link(cam)
cam.data.type='ORTHO';cam.data.ortho_scale=.44;cam.data.dof.use_dof=False
cam.location=(0,3,1.425);cam.rotation_euler=(Vector((0,0,1.425))-cam.location).to_track_quat('-Z','Y').to_euler();scene.camera=cam
out.mkdir(parents=True);scene.render.filepath=str(out/'back.png');bpy.ops.render.render(write_still=True)
(out/'report.json').write_text(json.dumps({'source':str(source),'sha256':digest,'neutral':neutral,'camera':{'target':[0,0,1.425],'orthoScale':.44},'sourceSaved':False},indent=2),encoding='utf-8')
