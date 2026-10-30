"""Reopen-check the integrated current source, then render bounded head poses."""
import bpy,runpy,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
cloth=runpy.run_path(str(folder/'walk_cloth_native_r07.py'))
cloth['verify_reopened'](expected_relative=Path(bpy.data.filepath).resolve().relative_to(p.resolve()).as_posix())
neck=runpy.run_path(str(folder/'head_neck_pose_review_r07.py'))
neck['run'](render_cases=[('look_down','face-side'),('look_up','face-side')],output_relative='previews/reconstruction-r02/neck-poses-fixed-r07')
sys.argv=['reference_render_r02.py','--','iris-r07','face-front']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
