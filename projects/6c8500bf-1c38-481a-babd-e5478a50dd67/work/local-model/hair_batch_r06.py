"""Bounded sequential guide bakes for the current R06 groom, no Blend mutation."""
import bpy,runpy,sys,json
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();api=runpy.run_path(str(Path(__file__).with_name('secondary_hair_r02.py')))
actions=['R03 Full body review - authored diagnostic','R03 Performance review','R06 CMU 07_01 walk - forward gaze']
colliders=api['_collider_definitions'](bpy.data.objects['Character_Rig'])
for c in colliders:
    if c['name']=='head':c['radius']=max(c['radius'],.085)
report=api['bake_actions'](actions=actions,output='work/reconstruction-r02/hair-batch-r06',allow_authored_ik=True,colliders=colliders)
print('R06_HAIR_BATCH_COMPLETE',json.dumps({'actions':[r['action'] for r in report['actions']],'caches':[r['cache'] for r in report['actions']]}),flush=True)
