import bpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,render_views
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D2b-face-continuity.blend'))
for ob in bpy.data.collections['Rebuilt_03_Hair_Editable'].objects:ob.hide_render=True
render_views('checks/face-shadow-diagnostic',['face-front'])
