"""Build only the disposable runtime geometry stage; no material bake/export."""
import bpy,runpy,sys,json
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
face=runpy.run_path(str(folder/'runtime_face_snapshot_r06.py'));face['prepare'](max_subdivision=1)
bpy.ops.wm.save_as_mainfile(filepath=str(p/'work/runtime-r06/prototype-face.blend'))
hair=runpy.run_path(str(folder/'runtime_haircards_r06.py'));hair['build']()
bpy.ops.wm.save_as_mainfile(filepath=str(p/'work/runtime-r06/prototype-hair.blend'))
assembly=runpy.run_path(str(folder/'runtime_assembly_r06.py'))
assembly['skirt_from_control']();assembly['bind_bone_parent_meshes']();assembly['configure_springs']();assembly['inventory']()
bpy.ops.wm.save_as_mainfile(filepath=str(p/'work/runtime-r06/prototype-assembled.blend'))
print('RT06_PREFLIGHT_DONE',flush=True)
