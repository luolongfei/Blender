"""Serial background static cloth experiment. Its saved output is a test copy."""
import bpy,runpy,sys,json,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();r=bpy.data.objects['Character_Rig'];r.data.pose_position='REST'
api=runpy.run_path(str(Path(__file__).with_name('cloth_setup_r02.py')))
api['run']();api['preview'](True)
target=p/'work/reconstruction-r02/cloth-static-test.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(target))
start=time.time();report=api['bake']();report['elapsed_seconds']=time.time()-start
(target.parent/'cloth-static-timing.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
bpy.context.scene.frame_set(61)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
print('CLOTH_STATIC_COMPLETE',time.time()-start,flush=True)
