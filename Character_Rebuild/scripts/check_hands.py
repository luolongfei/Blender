import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import geometry_checks
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E-textured-refinement.blend'))
objects=[o for o in bpy.context.scene.objects if o.name.startswith('Rebuilt_Hand_')]
report=geometry_checks.inspect_scene(objects,evaluated=True,self_intersection_objects=[o.name for o in objects])
(OUT/'reports/hand-v2-actual-geometry.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
