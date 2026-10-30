import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views,neutral_review
import face_v2,materials
expected='aa3f6d5b55b104f58ad58f4c350a5a2fe56e2ecb4e560ff3b6c5a3bc61821f31'
assert hashlib.sha256(Path(face_v2.__file__).read_bytes()).hexdigest()==expected
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D2-refined-face-hair-hands.blend'))
headcoll=bpy.data.collections['Rebuilt_01_Head_Editable']
for o in list(headcoll.objects):
    if not o.name.startswith('Ear_'):bpy.data.objects.remove(o,do_unlink=True)
spec=json.loads((OUT/'reports/measured-face-spec-v2.json').read_text())
mats=materials.create_palette(prefix='RebuiltV2b_',iris_image_path=OUT/'textures/iris-anime-1024.png')
materials.set_basecolor_image(mats['skin'],OUT/'textures/face-skin-2048.png','Face_Reference_UV')
materials.match_flat_skin_to_baked_corner(mats['skin_back'],OUT/'textures/face-skin-2048.png')
objects,report=face_v2.build_face(spec,headcoll,mats)
report['moduleSha256']=expected
(OUT/'reports/stage-d2b.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
neutral_review();save('D2b-face-continuity.blend')
render_views('checks/geometry-v2b',['face-front','face-side','face-quarter'])
