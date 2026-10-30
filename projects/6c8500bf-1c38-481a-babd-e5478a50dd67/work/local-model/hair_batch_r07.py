"""Fresh sequential hair dynamics after the corrected skull-base pivot."""
import bpy,runpy,sys,json
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();rig=bpy.data.objects['Character_Rig']
if not rig.get('R06_head_pivot_calibration'):raise RuntimeError('Calibrated head pivot required')
api=runpy.run_path(str(Path(__file__).with_name('secondary_hair_r02.py')))
colliders=api['_collider_definitions'](rig)
center=rig.data.bones['head'].matrix_local.inverted() @ Vector((0,.00202,1.441285))
for c in colliders:
    if c['name']=='head':c.update(localCenter=list(center),radius=.084)
actions=['R03 Full body review - authored diagnostic','R03 Performance review','R06 CMU 07_01 walk - forward gaze']
report=api['bake_actions'](actions=actions,output='work/reconstruction-r02/hair-batch-r07',allow_authored_ik=True,colliders=colliders)
print('R07_HAIR_BATCH_COMPLETE',json.dumps({'actions':[r['action'] for r in report['actions']],'caches':[r['cache'] for r in report['actions']]}),flush=True)
