import bpy,sys,json,math,bmesh
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,render_views,neutral_review,collection
import materials
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E3-review-ready.blend'))
body=bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
before_custom=body.data.has_custom_normals
for o in bpy.context.scene.objects:o.select_set(False)
body.select_set(True);bpy.context.view_layer.objects.active=body
try:bpy.ops.mesh.customdata_custom_splitnormals_clear()
except Exception:body.data.normals_split_custom_set([(0,0,0)]*len(body.data.loops))
for p in body.data.polygons:p.use_smooth=True
body.data.update()
mat=materials.bake_repaired_body_atlas(body,[{'name':'Uniform ivory cloth albedo; geometry retains fold shape','center':[0,0,.492],'half_extents':[.78,.30,.204],'feather':.008,'strength':1,'color_srgb':[.94,.89,.805]}],OUT/'textures/body-final-4096.png',4096,repack_uv=False)
body.data.materials.clear();body.data.materials.append(mat)
for p in body.data.polygons:p.material_index=0
c=collection('Rebuilt_05_Cuff_Inner_Construction')
cloth=materials.principled_material('Rebuilt_Cuff_Inner',(.90,.85,.77),.9)
skin=bpy.data.materials['Rebuilt_Final_skin_back']
created=[]
for side,label in [(-1,'R'),(1,'L')]:
    # Continue the garment inward at its actual cut boundary; preserve opening.
    bm=bmesh.new();bm.from_mesh(body.data)
    boundary={v.index:v.co.copy() for e in bm.edges if e.is_boundary and all(abs(v.co.x-side*.7525)<1e-5 for v in e.verts) for v in e.verts}
    bm.free()
    center_y=.0365;center_z=.5475
    ring=sorted(boundary.values(),key=lambda v:math.atan2(v.z-center_z,v.y-center_y))
    n=len(ring);assert n>20
    vs=[];fs=[]
    for layer in range(3):
        for v in ring:
            depth=[0,.016,.038][layer];factor=[1,.94,.84][layer]
            vs.append((side*(.7525-depth),center_y+(v.y-center_y)*factor,center_z+(v.z-center_z)*factor))
    for j in range(2):
        for i in range(n):fs.append((j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i))
    me=bpy.data.meshes.new('Cuff_Inner_'+label);me.from_pydata(vs,[],fs);me.update()
    ob=bpy.data.objects.new('Cuff_Inner_'+label,me);c.objects.link(ob);me.materials.append(cloth)
    uv=me.uv_layers.new(name='CuffUV')
    for p in me.polygons:
        p.use_smooth=True
        for li in p.loop_indices:
            idx=me.loops[li].vertex_index;uv.data[li].uv=((idx%n)/n,(idx//n)/2)
    cloth.use_backface_culling=False
    created.append(ob.name)
    # Hidden continuation of the wrist, avoiding a disconnected capped hand.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,location=(side*.722,.0365,.5475))
    ob=bpy.context.object;ob.name='Forearm_UnderSleeve_'+label;ob.scale=(.055,.041,.028)
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for old in list(ob.users_collection):old.objects.unlink(ob)
    c.objects.link(ob);ob.data.materials.append(skin)
    for p in ob.data.polygons:p.use_smooth=True
    created.append(ob.name)
report={'retainedCustomNormalsBefore':before_custom,'retainedCustomNormalsAfter':body.data.has_custom_normals,'bodySmoothFaces':len(body.data.polygons),'cuffAndHiddenWristObjects':created,'bodyAlbedo':'Original garment silhouette and folds retained; blouse directional painted shading replaced with ivory intrinsic colour. Skirt, legs, socks and shoes remain previous repaired atlas.'}
(OUT/'reports/stage-e4.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
neutral_review();save('E4-final-refined.blend')
render_views('checks/textured-v4',['front','back','face-front','face-side','hand-oblique'])
