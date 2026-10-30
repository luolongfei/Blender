"""Same neutral R09 cameras and lights for an explicitly saved candidate."""
import argparse, bpy, runpy, sys
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True);parser.add_argument('--label',required=True)
parser.add_argument('--views',default='face-front,face-quarter')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
folder=Path(__file__).resolve().parent
stills=runpy.run_path(str(folder/'reference_final_stills_r08.py'))
p=stills['project_path']();source=(p/args.source).resolve();out=p/'previews/reconstruction-r02'/args.label
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or out.exists():raise RuntimeError('Exact source and new review folder required')
if Path(args.label).name!=args.label:raise ValueError('Invalid review label')
stills['_neutral'](bpy.data.objects['Character_Rig'])
sys.argv=[str(folder/'reference_render_r02.py'),'--',args.label,args.views]
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
