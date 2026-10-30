import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views
import face_v2,hands,hair,materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D-integrated-geometry.blend'))
headcoll=bpy.data.collections['Rebuilt_01_Head_Editable']
for o in list(headcoll.objects):
    if not o.name.startswith('Ear_'):bpy.data.objects.remove(o,do_unlink=True)
spec=json.loads((OUT/'reports/measured-face-spec-v2.json').read_text())
mats=materials.create_palette(prefix='RebuiltV2_',iris_image_path=OUT/'textures/iris-anime-1024.png')
materials.set_basecolor_image(mats['skin'],OUT/'textures/face-skin-2048.png','Face_Reference_UV')
objects,face_report=face_v2.build_face(spec,headcoll,mats)
for o in headcoll.objects:
    if o.name.startswith('Ear_Inner'):
        o.data.materials.clear();o.data.materials.append(mats['skin_back'])
haircoll=bpy.data.collections['Rebuilt_03_Hair_Editable']
for o in list(haircoll.objects):bpy.data.objects.remove(o,do_unlink=True)
hair_spec=json.loads((OUT/'reports/hair-spec-v2.json').read_text())
hair_objs=hair.build_from_spec(hair_spec,haircoll,mats['hair'],mats['hair_shadow'])
handcoll=bpy.data.collections['Rebuilt_02_Hands_Editable']
for o in list(handcoll.objects):bpy.data.objects.remove(o,do_unlink=True)
hs=json.loads((OUT/'reports/hand-spec-v2.json').read_text())
hand_reports=[]
for h in hs['hands']:
    ob,rep=hands.build_hand(h['name'],h['wrist_origin'],h['frame'],hs['dimensions'],handcoll,mats['skin_back'],subdivision=1);hand_reports.append(rep)
(OUT/'reports/stage-d2.json').write_text(json.dumps({'face':face_report,'hands':hand_reports,'status':'Second geometry candidate, pending appearance review; back-yoke not integrated'},indent=2),encoding='utf-8')
save('D2-refined-face-hair-hands.blend')
render_views('checks/geometry-v2',['face-front','face-side','face-quarter','back','hand-top','hand-palm'])
