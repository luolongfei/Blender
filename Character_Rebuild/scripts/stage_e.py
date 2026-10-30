import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views,neutral_review
import clothing_v2,hair,materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/D2b-face-continuity.blend'))
body=bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
yoke=clothing_v2.apply_integrated_back_yoke(body)
hs=json.loads((OUT/'reports/hair-spec-v3.json').read_text())
hc=bpy.data.collections['Rebuilt_03_Hair_Editable']
hair_mat=bpy.data.objects['Hair_Crown_SourceEnvelope'].data.materials[0]
tie_mat=bpy.data.objects['Ponytail_Left_FabricTie'].data.materials[0]
for ob in list(hc.objects):bpy.data.objects.remove(ob,do_unlink=True)
hair_objects=hair.build_from_spec(hs,hc,hair_mat,tie_mat)
neutral_review();save('D3-repaired-yoke-and-hair.blend')
regions=[
 {'name':'Back duplicate button placket removed','center':[0,.086,.423],'half_extents':[.033,.12,.16],'feather':.014,'normal_direction':[0,1,0],'normal_min':.20,'color_srgb':[.94,.89,.805]},
 {'name':'Front left collar hair colour contamination','center':[-.119,-.075,.576],'half_extents':[.074,.095,.060],'feather':.015,'dark_threshold':.40,'color_srgb':[.94,.89,.805]},
 {'name':'Front right collar hair colour contamination','center':[.119,-.075,.576],'half_extents':[.074,.095,.060],'feather':.015,'dark_threshold':.40,'color_srgb':[.94,.89,.805]},
 {'name':'Skirt white background specks','center':[0,0,.022],'half_extents':[.36,.29,.268],'feather':.014,'bright_threshold':.38,'color_srgb':[.31,.35,.276]},
 {'name':'Exposed legs colour shadow cleanup','center':[0,0,-.446],'half_extents':[.23,.19,.225],'feather':.018,'strength':.72,'color_srgb':[.96,.805,.690]},
]
material=materials.bake_repaired_body_atlas(body,regions,OUT/'textures/body-repaired-4096.png',4096,repack_uv=True)
body.data.materials.clear();body.data.materials.append(material)
for poly in body.data.polygons:poly.material_index=0
body['rebuild_note']='Preserved source garment volumes with repaired back yoke; original UV kept for editing, new UV and actual 4K colour rebake used for export'
report={'backYoke':yoke,'regions':regions,'newUV':body.data.uv_layers.active.name,'bodyMaterial':material.name,'hairSpecSha256':hashlib.sha256((OUT/'reports/hair-spec-v3.json').read_bytes()).hexdigest(),'status':'Textured candidate; pending neutral review and clean GLB reimport'}
(OUT/'reports/stage-e.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
save('E-textured-refinement.blend')
render_views('checks/textured-v1',['front','back','face-front','face-side','face-quarter','hand-oblique'])
