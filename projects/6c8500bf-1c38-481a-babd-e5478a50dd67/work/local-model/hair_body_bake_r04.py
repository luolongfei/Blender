"""Explicit PBD guide bake for the validated authored body review with keyed leg IK."""
import runpy,time
from pathlib import Path
api=runpy.run_path(str(Path(__file__).with_name('secondary_hair_r02.py')))
start=time.time()
api['bake_actions'](actions=['R03 Full body review - authored diagnostic'],
    frame_ranges={'R03 Full body review - authored diagnostic':(1,301)},
    output='work/reconstruction-r02/hair-body-r04',loop_preroll_cycles=0,allow_authored_ik=True)
print('R04_BODY_HAIR_BAKED',time.time()-start,flush=True)
