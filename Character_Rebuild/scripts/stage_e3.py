import bpy,sys,json,math
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views,neutral_review,collection
import materials,bmesh
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E2-textured-final-candidate.blend'))
head=bpy.data.objects['Head_Skin_Quad_Retopo'];cap=bpy.data.objects['Hair_Crown_SourceEnvelope']
bvh=BVHTree.FromObject(cap,bpy.context.evaluated_depsgraph_get())
centre=Vector((0,.02,.855));moves=[]
for v in head.data.vertices:
    if v.co.z<=.845:continue
    delta=v.co-centre;direction=delta.normalized()
    pos,normal,index,distance=bvh.ray_cast(centre,direction,1)
    if pos is not None and delta.length>distance-.003:
        target=centre+direction*max(.001,distance-.003)
        amount=(target-v.co).length;moves.append(amount);v.co=target
head.data.update()
bm=bmesh.new();bm.from_mesh(head.data);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(head.data);bm.free()
body=bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
bvh_body=BVHTree.FromObject(body,bpy.context.evaluated_depsgraph_get())
buttons=collection('Rebuilt_04_Blouse_Buttons')
mat=materials.principled_material('Rebuilt_Ivory_Buttons',(.90,.86,.78),.65)
button_report=[]
for i,z in enumerate([.53,.435,.34]):
    pos,no,idx,dist=bvh_body.ray_cast(Vector((0,-1,z)),Vector((0,1,0)),2)
    assert pos is not None
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,location=(0,pos.y-.0015,z))
    ob=bpy.context.object;ob.name='Blouse_Button_'+str(i+1);ob.scale=(.006,.0025,.006)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for c in list(ob.users_collection):c.objects.unlink(ob)
    buttons.objects.link(ob);ob.data.materials.append(mat)
    for p in ob.data.polygons:p.use_smooth=True
    button_report.append({'name':ob.name,'centre':list(ob.location)})
report={'hiddenScalpFit':{'heightLowerBound':.845,'method':'Actual crown inner-shell ray intersection from head interior; 3mm gap','movedVertices':len(moves),'maxMove':max(moves,default=0),'faceEyeMouthUnchanged':True},'buttons':button_report,'neutralLighting':'Broad front fill reduced to 40W after rejecting overexposed 300W diagnostic'}
(OUT/'reports/stage-e3.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
neutral_review();save('E3-review-ready.blend')
render_views('checks/textured-v3',['front','back','face-front','face-side','face-quarter','hand-top'])
