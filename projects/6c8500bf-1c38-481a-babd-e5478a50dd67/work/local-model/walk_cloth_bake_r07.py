"""Serial walk-cloth bake, then read-only neutral face renders."""
import bpy,runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent
api=runpy.run_path(str(folder/'walk_cloth_r06.py'))
api['bake']()
sys.argv=['reference_render_r02.py','--','face-continuity-r07','face-front,face-quarter']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
