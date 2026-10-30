"""Inventory actual runtime materials before spending time on baking."""
import bpy,runpy,sys,json
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
assembly=runpy.run_path(str(folder/'runtime_assembly_r06.py'))
rig,meshes=assembly['current_meshes']()
api=runpy.run_path(str(folder/'runtime_material_bake_r06.py'))
report=api['probe'](meshes)
(p/'work/runtime-r06/material-probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
bad=[{'object':o['object'],'material':m['name'],'reason':m.get('reason')} for o in report['objects'] for m in o['materials'] if m.get('bakeableSurface') is False]
print(json.dumps({'meshes':len(meshes),'unsupported':bad}),flush=True)
sys.argv=['reference_render_r02.py','--','runtime-prototype-r06','face-front,full']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
