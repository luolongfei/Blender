import bpy,sys,json,math
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,collection,save,render_views,neutral_review
import face,hands,hair,materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/A-neutral-reference.blend'))
original=bpy.data.objects['Original_Character'];original.hide_render=True;original.hide_set(True)
spec=json.loads((OUT/'reports/measured-face-spec.json').read_text())
skinpath=OUT/'textures/face-skin-2048.png'
if not skinpath.exists():materials.bake_face_skin_image(spec,skinpath)
mats=materials.create_palette(iris_image_path=OUT/'textures/iris-anime-1024.png')
materials.set_basecolor_image(mats['skin'],skinpath,'Face_Reference_UV')
headcoll=collection('Rebuilt_01_Head_Editable')
head,face_report=face.build_face(spec,headcoll,mats)
# Clean stylised ears are inferred behind the preserved front face outline.
for s,label in [(-1,'R'),(1,'L')]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=20,location=(s*.111,.023,.767))
    o=bpy.context.object;o.name='Ear_'+label;o.scale=(.017,.018,.032)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for c in list(o.users_collection):c.objects.unlink(o)
    headcoll.objects.link(o);o.data.materials.append(mats['skin_back'])
    for p in o.data.polygons:p.use_smooth=True
    o['inferred_detail']='Conservative stylised ear volume under side locks'
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,location=(s*.122,.010,.768))
    o=bpy.context.object;o.name='Ear_Inner_'+label;o.scale=(.005,.009,.020)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for c in list(o.users_collection):c.objects.unlink(o)
    headcoll.objects.link(o);o.data.materials.append(mats['ear_inner'])
    for p in o.data.polygons:p.use_smooth=True
save('B-rebuilt-face.blend')
hs=json.loads((OUT/'reports/hand-spec.json').read_text())
hc=collection('Rebuilt_02_Hands_Editable');hand_reports=[]
for h in hs['hands']:
    obj,rep=hands.build_hand(h['name'],h['wrist_origin'],h['frame'],hs['dimensions'],hc,mats['skin_back'],subdivision=1);hand_reports.append(rep)
save('C-rebuilt-hands.blend')
import clothing
clothing_result=clothing.cleanup_original(original)
original.hide_render=False;original.hide_set(False)
hair_spec=json.loads((OUT/'reports/hair-spec.json').read_text())
haircoll=collection('Rebuilt_03_Hair_Editable')
hair_objs=hair.build_from_spec(hair_spec,haircoll,mats['hair'],mats['hair_shadow'])
report={'face':face_report,'hands':hand_reports,'clothing':clothing_result,'hairObjects':[o.name for o in hair_objs],'status':'first integrated geometry; appearance review pending'}
(OUT/'reports/stage-bcd.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
save('D-integrated-geometry.blend')
render_views('checks/geometry-v1',['front','face-front','face-side','face-quarter','hand-top','hand-palm','back'])
