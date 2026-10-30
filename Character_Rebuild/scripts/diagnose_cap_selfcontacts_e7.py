import bpy,sys,json,hashlib
from pathlib import Path
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.context import workspace_path
import geometry_checks
base=workspace_path('Character_Rebuild')
source=base/'stages/E7-uv-normals-corrected.blend'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
j=json.loads((base/'reports/clean-reimport.json').read_text())
entry=next(o for o in j['objects'] if o.get('object')=='Hair_Crown_SourceEnvelope')
bpy.ops.wm.open_mainfile(filepath=str(base/'stages/F-clean-glb-reimport.blend'))
ob=bpy.data.objects['Hair_Crown_SourceEnvelope'];me=ob.data;me.calc_loop_triangles()
import_pairs=[]
for pair in entry['selfContacts']['contactPairSamples']:
    tris=[]
    for index in pair['triangles']:
        t=me.loop_triangles[index];tris.append([tuple(ob.matrix_world@me.vertices[k].co) for k in t.vertices])
    import_pairs.append({'indices':pair,'points':tris,'contactAtTolerance':{str(t):geometry_checks.triangles_contact(*tris,tolerance=t) for t in [1e-8,1e-10,1e-12,0]}})
bpy.ops.wm.open_mainfile(filepath=str(source))
ob=bpy.data.objects['Hair_Crown_SourceEnvelope'];me=ob.data;me.calc_loop_triangles()
kd=KDTree(len(me.vertices))
for v in me.vertices:kd.insert(ob.matrix_world@v.co,v.index)
kd.balance()
for p in import_pairs:
    p['sourceVertexMatches']=[]
    for tri in p['points']:
        ar=[]
        for co in tri:
            location,index,distance=kd.find(co)
            ar.append({'index':index,'distance':distance,'position':list(location),
                       'surface':'outer' if index<1344 else 'inner' if index<2688 else 'pole'})
        p['sourceVertexMatches'].append(ar)
actual=geometry_checks.inspect_mesh(me,ob.matrix_world,self_intersections=True,max_self_triangles=20000,max_self_candidates=300000,tolerance=1e-10)
report={'source':str(source),'sourceSha256':source_hash,'sourceUnchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,
        'mappedReimportPairs':import_pairs,'sourceMeshSelfContacts':actual['selfContacts'],
        'vertices':len(me.vertices),'polygons':len(me.polygons)}
out=base/'reports/cap-selfcontact-diagnostic-e7.json';out.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
