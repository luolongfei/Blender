import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
from face_e7_mouth_annulus_delta import apply_e7_mouth_annulus_delta,_positions
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E7-uv-normals-corrected.blend'))
report=apply_e7_mouth_annulus_delta()
patched=_positions(bpy.data.objects['Head_Skin_Quad_Retopo'])
bpy.ops.wm.open_mainfile(filepath=str(OUT/'work/e7-head-contact-diagnostic/E7-mouth-local-test.blend'))
assert patched==_positions(bpy.data.objects['Head_Skin_Quad_Retopo'])
report['matchesVerifiedCandidateExactly']=True
(OUT/'work/e7-head-contact-diagnostic/module-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
