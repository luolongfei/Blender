"""Read-only visual test of accumulated sculpt layers, never saves a source."""
import bpy,runpy,sys,json
from pathlib import Path
values={'R02_soft_tissue':.35,'R03_facial_balance':.65,'R04_reference_softness':.5,
        'R05_cheek_contour':0.,'R07_cheek_surface_continuity':0.,
        'R02_smile':.2,'R02_gentle_smile':.5}
sys.argv=['reference_render_r02.py','--','face-volume-study-r07','face-front,face-quarter','0',json.dumps(values)]
runpy.run_path(str(Path(__file__).with_name('reference_render_r02.py')),run_name='__main__')
