import bpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,render_views
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/A-neutral-reference.blend'))
render_views('checks/before-matched',['front','back','right','face-front','face-side','hand-top','hand-palm','hand-oblique'])
print('MATCHED_BEFORE_RENDERED')
