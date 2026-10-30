import bpy,bmesh,sys,json,math
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,neutral_review
bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E6-export-source.blend'))

def bad_uv(mesh):
    mesh.calc_loop_triangles();uv=mesh.uv_layers.active
    bad=[]
    for t in mesh.loop_triangles:
        a,b,c=[uv.data[i].uv for i in t.loops]
        if abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))*.5<=1e-14:bad.append(t)
    return bad

def unwrap(ob,faces=None,region=None):
    for other in bpy.context.scene.objects:other.select_set(False)
    ob.hide_set(False);ob.select_set(True);bpy.context.view_layer.objects.active=ob
    if faces is None:faces=set(range(len(ob.data.polygons)))
    for p in ob.data.polygons:p.select=p.index in faces
    bpy.context.tool_settings.mesh_select_mode=(False,False,True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(54),island_margin=.025,area_weight=0,correct_aspect=False,scale_to_bounds=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    if region:
        u0,v0,scale=region
        for p in ob.data.polygons:
            if p.index in faces:
                for li in p.loop_indices:
                    uv=ob.data.uv_layers.active.data[li].uv
                    uv.x=u0+uv.x*scale;uv.y=v0+uv.y*scale
    return len(faces)

neck=bpy.data.objects['Neck_Clean_Quad_Sleeve']
bm=bmesh.new();bm.from_mesh(neck.data)
bmesh.ops.reverse_faces(bm,faces=list(bm.faces));bm.to_mesh(neck.data);bm.free();neck.data.update()
radial=[v.normal.dot(Vector((v.co.x,v.co.y-.034,0)).normalized()) for v in neck.data.vertices]
assert min(radial)>.3,'Neck normals do not consistently point outward'
unwrap(neck)
report={'neck':{'normalsReversed':True,'minOutwardRadialDot':min(radial),'uvReunwrapped':True},'uvRepairs':[]}
head=bpy.data.objects['Head_Skin_Quad_Retopo']
bad=bad_uv(head.data);faces={t.polygon_index for t in bad}
original_polygons=[tuple(p.vertices) for p in head.data.polygons]
original_coords=[tuple(v.co) for v in head.data.vertices]
count=unwrap(head,faces,region=(.01,.90,.08))
after=bad_uv(head.data)
assert not after,'Head UV still contains collapsed triangles'
assert original_polygons==[tuple(p.vertices) for p in head.data.polygons]
assert original_coords==[tuple(v.co) for v in head.data.vertices]
report['uvRepairs'].append({'object':head.name,'collapsedTrianglesBefore':len(bad),'reunwrappedPolygons':count,'collapsedTrianglesAfter':len(after),'placement':'Separate non-collapsed UV islands in unpainted uniform-skin corner of original 2K face atlas','geometryUnchanged':True})
for name in ('Rebuilt_BackCollar_Band','Rebuilt_FrontCollar_L','Rebuilt_FrontCollar_R'):
    ob=bpy.data.objects[name];before=len(bad_uv(ob.data));unwrap(ob)
    after=len(bad_uv(ob.data));assert after==0,name
    report['uvRepairs'].append({'object':name,'collapsedTrianglesBefore':before,'collapsedTrianglesAfter':after,'material':'Pure ivory PBR; UV change does not alter albedo'})
neutral_review();save('E7-uv-normals-corrected.blend')
(OUT/'reports/stage-e7.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('E7_NORMALS_AND_UV_FIXED')
