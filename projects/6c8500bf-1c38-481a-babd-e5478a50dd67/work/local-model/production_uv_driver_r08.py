"""Add verified optional UV to an independent R08 production copy."""
import hashlib,json,runpy
from pathlib import Path
import bpy
folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'production_uv_r08.py'))
p=api['project_path']()
target=p/'work/reconstruction-r02/reference-production-uv-r08.blend'
report=p/'work/reconstruction-r02/production-uv-r08.json'
if target.exists() or report.exists():raise RuntimeError('Retain previous output')
result=api['run'](source_relative='work/reconstruction-r02/reference-face-flow-r08.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result.update(output=target.relative_to(p).as_posix(),outputSha256=hashlib.sha256(target.read_bytes()).hexdigest(),savedBlend=True)
report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R08_UV_SAVED '+str(target),flush=True)
