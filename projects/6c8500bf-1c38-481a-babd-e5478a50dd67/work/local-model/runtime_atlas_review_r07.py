"""Upgrade the retained hair-card prototype and render actual coverage."""
import bpy,runpy,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
api=runpy.run_path(str(folder/'runtime_haircards_r06.py'));api['upgrade_atlases']()
target=p/'work/runtime-r06/prototype-atlas-r07.blend'
if target.exists():raise RuntimeError('Retain existing prototype')
bpy.ops.wm.save_as_mainfile(filepath=str(target))
sys.argv=['reference_render_r02.py','--','runtime-atlas-r07','face-front,face-quarter']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
