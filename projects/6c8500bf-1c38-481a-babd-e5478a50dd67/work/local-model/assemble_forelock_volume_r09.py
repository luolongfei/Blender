"""One larger forelock candidate, with unchanged-camera visible proof."""
import bpy, hashlib, json, runpy, sys
from pathlib import Path
from mathutils import Vector

folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'forelock_volume_candidate_r09.py'))
p=api['project_path']()
source='work/reconstruction-r02/reference-root-flow-r09.blend'
target=p/'work/reconstruction-r02/reference-forelock-volume-r09.blend'
report=p/'work/reconstruction-r02/forelock-volume-r09.json'
out=p/'previews/reconstruction-r02/forelock-volume-r09'
if target.exists() or report.exists() or out.exists():raise RuntimeError('Preserve previous outputs')
result=api['run'](source_relative=source)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result.update(savedBlend=True,output=target.relative_to(p).as_posix(),outputSha256=hashlib.sha256(target.read_bytes()).hexdigest())
report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_FORELOCK_VOLUME_SAVED',flush=True)
stills=runpy.run_path(str(folder/'reference_final_stills_r08.py'))
stills['_neutral'](bpy.data.objects['Character_Rig'])
sys.argv=[str(folder/'reference_render_r02.py'),'--','forelock-volume-r09','face-front,face-quarter']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
scene=bpy.context.scene;scene.cycles.samples=12;scene.render.use_persistent_data=True
scene.render.resolution_x=640;scene.render.resolution_y=720;scene.render.resolution_percentage=100
cam=scene.camera;rows=[]
for name,position in [('back',(0,3,1.425)),('rear-quarter',(3,4,1.425))]:
    target_point=Vector((0,0,1.425));cam.location=position
    cam.rotation_euler=(target_point-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=.44
    scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    rows.append({'view':name,'position':position,'target':list(target_point),'scale':.44})
    print('R09_ROOT_VISIBLE_VIEW '+name,flush=True)
(out/'root-render-report.json').write_text(json.dumps({'source':target.relative_to(p).as_posix(),'sha256':result['outputSha256'],'samples':12,'views':rows,'sourceSavedByRender':False},indent=2),encoding='utf-8')
