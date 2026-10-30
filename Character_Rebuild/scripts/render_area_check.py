import bpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,render_views
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E4-final-refined.blend'))
render_views('checks/area-light-diagnostic',['face-front'])
