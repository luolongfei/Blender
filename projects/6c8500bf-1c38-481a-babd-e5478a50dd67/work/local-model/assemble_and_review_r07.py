"""Serial combined-source build and bounded latest neck/eye visual checks."""
import runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
runpy.run_path(str(folder/'assemble_walk_r07.py'),run_name='__main__')
neck=runpy.run_path(str(folder/'head_neck_pose_review_r07.py'))
neck['run'](render_cases=[('look_down','face-side'),('look_up','face-side')],output_relative='previews/reconstruction-r02/neck-contour-poses-r07')
hair=runpy.run_path(str(folder/'native_hair_playback_r04.py'));hair['disable']()
sys.argv=['reference_render_r02.py','--','iris-optics-r07','face-front']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
