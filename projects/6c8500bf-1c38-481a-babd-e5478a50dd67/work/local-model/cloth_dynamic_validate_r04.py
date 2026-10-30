"""One serial 301-frame cloth/body review bake on an explicit derivative copy."""
import bpy,runpy,sys,time,json
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
action=bpy.data.actions['R03 Full body review - authored diagnostic'];rig.animation_data.action=action
if hasattr(action,'slots'):rig.animation_data.action_slot=next(s for s in action.slots if s.target_id_type=='OBJECT')
rig.data.pose_position='POSE';scene.frame_start=1;scene.frame_end=301;scene.frame_set(1)
head=bpy.data.objects['R01 Head'];head.data.shape_keys.animation_data.action=None
for key in head.data.shape_keys.key_blocks:
    if key.name in ['R02_blink_L','R02_blink_R','R03_squint_L','R03_squint_R','R02_jaw_open','R02_pucker','R02_mouth_wide','R02_lip_press','R02_brow_raise','R02_brow_furrow','R02_frown']:key.value=0
api=runpy.run_path(str(Path(__file__).with_name('cloth_setup_r02.py')))
api['run'](frame_start=1,frame_end=301,follow_rig=True);api['preview'](True)
target=p/'work/reconstruction-r02/reference-cloth-body-r04.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(target));start=time.time()
report=api['bake']();report['elapsed_seconds']=time.time()-start
(target.parent/'cloth-dynamic-r04-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(target))
print('R04_CLOTH_BODY_COMPLETE',time.time()-start,flush=True)
