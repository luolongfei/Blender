"""Serial background guide-dynamics trial, then the combined review cache."""
import bpy,runpy,json,time,sys
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('secondary_hair_r02.py')))
arg=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'trial'
start=time.time()
if arg=='trial':
    result=api['bake_actions'](actions=['R02 Idle - authored IK source'],frame_ranges={'R02 Idle - authored IK source':(1,61)},output='work/reconstruction-r02/hair-trial-r03',loop_preroll_cycles=0)
else:
    result=api['bake_actions'](actions=['R03 Performance review'],frame_ranges={'R03 Performance review':(1,181)},output='work/reconstruction-r02/hair-performance-r03',loop_preroll_cycles=0)
print('R03_HAIR_VALIDATION_COMPLETE',time.time()-start,flush=True)
