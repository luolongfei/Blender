"""Refine only forelock flow in the saved posterior candidate, then review."""
import bpy,hashlib,json,runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'forelock_flow_r09.py'))
p=api['project_path']();source='work/reconstruction-r02/reference-posterior-groom-r09.blend'
target=p/'work/reconstruction-r02/reference-forelock-flow-r09.blend';report=p/'work/reconstruction-r02/forelock-flow-r09.json'
out=p/'previews/reconstruction-r02/forelock-flow-r09'
if target.exists() or report.exists() or out.exists():raise RuntimeError('Preserve previous outputs')
result=api['run'](source_relative=source)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result.update(savedBlend=True,output=target.relative_to(p).as_posix(),outputSha256=hashlib.sha256(target.read_bytes()).hexdigest())
report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_FORELOCK_SAVED',flush=True)
stills=runpy.run_path(str(folder/'reference_final_stills_r08.py'))
stills['_neutral'](bpy.data.objects['Character_Rig'])
sys.argv=[str(folder/'reference_render_r02.py'),'--','forelock-flow-r09','face-front,face-quarter']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
