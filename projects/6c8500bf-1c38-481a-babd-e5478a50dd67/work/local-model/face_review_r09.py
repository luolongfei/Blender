"""Identical R08 cameras for the independent R09 proportion candidate."""
import bpy,runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'reference_final_stills_r08.py'))
api['_neutral'](bpy.data.objects['Character_Rig'])
label='face-proportion-r09'
out=api['project_path']()/'previews/reconstruction-r02'/label
if out.exists():raise RuntimeError('Retain previous images')
sys.argv=[str(folder/'reference_render_r02.py'),'--',label,'face-front,face-quarter,face-side']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
