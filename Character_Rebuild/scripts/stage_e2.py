import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views,neutral_review
import face_v3,hands,hair,hair_finish_v4,front_collar,materials
locked={'face_v3.py':'e04464ee152bb445b6c3bc8a06921a49c3cc209c6fd5ca65bc58822bbafe81de','hands.py':'5d40c0625a1a76ab26b7b6488383fa2b9c6f6ba5e50568af39bdee82854406c5','hair.py':'cf1626835dbdbabe74ee52140c27ad3603dd31c646aa0a80c521b2ff6daf43c3'}
for file,sha in locked.items():assert hashlib.sha256((OUT/'scripts'/file).read_bytes()).hexdigest()==sha,file
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D3-repaired-yoke-and-hair.blend'))
body=bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
headcoll=bpy.data.collections['Rebuilt_01_Head_Editable']
for o in list(headcoll.objects):
    if not o.name.startswith('Ear_'):bpy.data.objects.remove(o,do_unlink=True)
spec=json.loads((OUT/'reports/measured-face-spec-v3.json').read_text())
mats=materials.create_palette(prefix='Rebuilt_Final_',iris_image_path=OUT/'textures/iris-anime-1024.png')
materials.set_basecolor_image(mats['skin'],OUT/'textures/face-skin-2048.png','Face_Reference_UV')
materials.match_flat_skin_to_baked_corner(mats['skin_back'],OUT/'textures/face-skin-2048.png')
objects,face_report=face_v3.build_face(spec,headcoll,mats)
hc=bpy.data.collections['Rebuilt_03_Hair_Editable']
for ob in list(hc.objects):bpy.data.objects.remove(ob,do_unlink=True)
hair_objects=hair.build_from_spec(json.loads((OUT/'reports/hair-spec-v4.json').read_text()),hc,mats['hair'],mats['hair_shadow'])
hair_report=hair_finish_v4.apply_crown_flow(bpy.data.objects['Hair_Crown_SourceEnvelope'],OUT/'textures')
handcoll=bpy.data.collections['Rebuilt_02_Hands_Editable']
for ob in list(handcoll.objects):bpy.data.objects.remove(ob,do_unlink=True)
hs=json.loads((OUT/'reports/hand-spec-v3.json').read_text());hand_reports=[]
for h in hs['hands']:
    ob,rep=hands.build_hand(h['name'],h['wrist_origin'],h['frame'],hs['dimensions'],handcoll,mats['skin_back'],subdivision=1);hand_reports.append(rep)
collar=front_collar.add_clean_front_collar(body,mats['blouse'])
neutral_review();save('E2-geometry-final-candidate.blend')
regions=[
 {'name':'Blouse baked-light reduction','center':[0,0,.492],'half_extents':[.77,.29,.204],'feather':.012,'strength':.70,'color_srgb':[.94,.89,.805]},
 {'name':'Back duplicate button placket removed','center':[0,.086,.445],'half_extents':[.035,.12,.140],'feather':.014,'normal_direction':[0,1,0],'normal_min':.20,'color_srgb':[.94,.89,.805]},
 {'name':'Front left collar hair colour contamination','center':[-.119,-.075,.576],'half_extents':[.074,.095,.060],'feather':.015,'dark_threshold':.40,'color_srgb':[.94,.89,.805]},
 {'name':'Front right collar hair colour contamination','center':[.119,-.075,.576],'half_extents':[.074,.095,.060],'feather':.015,'dark_threshold':.40,'color_srgb':[.94,.89,.805]},
 {'name':'Old front collar painted edge removed beneath new collar','center':[0,-.08,.602],'half_extents':[.162,.112,.048],'feather':.010,'color_srgb':[.94,.89,.805]},
 {'name':'Skirt white background specks','center':[0,0,.022],'half_extents':[.36,.29,.268],'feather':.014,'bright_threshold':.38,'color_srgb':[.31,.35,.276]},
 {'name':'Exposed legs colour shadow cleanup','center':[0,0,-.446],'half_extents':[.23,.19,.225],'feather':.018,'strength':.72,'color_srgb':[.96,.805,.690]},
]
material=materials.bake_repaired_body_atlas(body,regions,OUT/'textures/body-repaired-v2-4096.png',4096,repack_uv=True)
body.data.materials.clear();body.data.materials.append(material)
for poly in body.data.polygons:poly.material_index=0
report={'lockedModules':locked,'face':face_report,'hair':hair_report,'hands':hand_reports,'frontCollar':collar,'regions':regions,'newUV':body.data.uv_layers.active.name,'status':'Final candidate pending visual review and clean GLB reimport'}
(OUT/'reports/stage-e2.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
save('E2-textured-final-candidate.blend')
render_views('checks/textured-v2',['front','back','face-front','face-side','face-quarter','hand-top','hand-palm','hand-oblique'])
