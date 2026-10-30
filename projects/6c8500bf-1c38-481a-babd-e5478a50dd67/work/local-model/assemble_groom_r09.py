"""Save posterior supplement independently, then render real coverage views."""
import bpy,hashlib,json,runpy,sys
from pathlib import Path
from mathutils import Vector
folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'posterior_groom_r09.py'))
p=api['project_path']();source='work/reconstruction-r02/reference-lower-face-candidate-r09.blend'
target=p/'work/reconstruction-r02/reference-posterior-groom-r09.blend'
report_path=p/'work/reconstruction-r02/posterior-groom-r09.json'
out=p/'previews/reconstruction-r02/posterior-groom-r09'
if target.exists() or report_path.exists() or out.exists():raise RuntimeError('Preserve previous candidate/output')
result=api['run'](source_relative=source)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result.update(output=target.relative_to(p).as_posix(),outputSha256=hashlib.sha256(target.read_bytes()).hexdigest(),savedBlend=True)
report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_POSTERIOR_SAVED',flush=True)
stills=runpy.run_path(str(folder/'reference_final_stills_r08.py'))
stills['_neutral'](bpy.data.objects['Character_Rig'])
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
scene.render.resolution_x=640;scene.render.resolution_y=720;scene.render.resolution_percentage=100
scene.render.use_sequencer=False;scene.render.image_settings.file_format='PNG'
cam=bpy.data.objects.new('R09 Posterior review',bpy.data.cameras.new('R09 Posterior review'));scene.collection.objects.link(cam)
cam.data.type='ORTHO';cam.data.dof.use_dof=False;scene.camera=cam
out.mkdir(parents=True);rows=[]
for name,position,look,scale in [('back',(0,3,1.425),(0,0,1.425),.44),('rear-left',(-3,4,1.425),(0,0,1.425),.44),('rear-right',(3,4,1.425),(0,0,1.425),.44),('crown',(0,0,4),(0,0,1.48),.33)]:
    cam.location=position;cam.rotation_euler=(Vector(look)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale
    scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    rows.append({'view':name,'camera':position,'target':look,'scale':scale});print('R09_POSTERIOR_VIEW '+name,flush=True)
(out/'render-report.json').write_text(json.dumps({'source':target.relative_to(p).as_posix(),'sha256':result['outputSha256'],'samples':12,'views':rows,'sourceSavedByRender':False},indent=2),encoding='utf-8')
