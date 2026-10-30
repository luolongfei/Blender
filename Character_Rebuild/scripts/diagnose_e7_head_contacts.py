import bpy,sys,json,hashlib,math
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
from geometry_checks import triangles_contact,inspect_mesh
dest=OUT/'work/e7-head-contact-diagnostic';dest.mkdir(parents=True,exist_ok=True)
src=OUT/'stages/E7-uv-normals-corrected.blend'
glb=OUT/'character-rebuilt-textured.glb'
report=json.loads((OUT/'reports/clean-reimport.json').read_text(encoding='utf-8'))
record=next(x for x in report['objects'] if x.get('object')=='Head_Skin_Quad_Retopo' or x.get('name')=='Head_Skin_Quad_Retopo')
pairs=[x['triangles'] for x in record['selfContacts']['contactPairSamples']]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(glb))
ob=bpy.data.objects['Head_Skin_Quad_Retopo'];me=ob.data;me.calc_loop_triangles()
out={'sourceSha256':hashlib.sha256(src.read_bytes()).hexdigest(),'glbSha256':hashlib.sha256(glb.read_bytes()).hexdigest(),'pairs':[]}
for pair in pairs:
    tris=[me.loop_triangles[i] for i in pair]
    pts=[[ob.matrix_world@me.vertices[v].co for v in t.vertices] for t in tris]
    norms=[(p[1]-p[0]).cross(p[2]-p[0]).normalized() for p in pts]
    rec={'triangles':pair,'vertices':[list(t.vertices) for t in tris],'positions':[[list(v) for v in p] for p in pts],
        'sat':{str(tol):triangles_contact(*pts,tolerance=tol) for tol in (1e-8,1e-10,1e-12,0)},
        'normals':[list(n) for n in norms],
        'planeDistances':[[float((v-pts[side][0]).dot(norms[side])) for v in pts[1-side]] for side in (0,1)]}
    out['pairs'].append(rec)
bpy.ops.wm.open_mainfile(filepath=str(src))
ob=bpy.data.objects['Head_Skin_Quad_Retopo'];me=ob.data
tree=KDTree(len(me.vertices))
for v in me.vertices:tree.insert(ob.matrix_world@v.co,v.index)
tree.balance()
for rec in out['pairs']:
    rec['sourceNearest']=[[[int(i),float(d)] for p in pts for _,i,d in [tree.find(Vector(p))]] for pts in rec['positions']]
out['sourceHeadCheck']=inspect_mesh(me,ob.matrix_world,self_intersections=True,max_self_triangles=40000,max_self_candidates=3000000)
(dest/'diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('DIAGNOSIS',json.dumps({'pairs':out['pairs'],'sourceContacts':out['sourceHeadCheck']['selfContacts']},indent=2))
