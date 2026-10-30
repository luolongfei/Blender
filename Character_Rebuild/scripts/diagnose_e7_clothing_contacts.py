import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector,kdtree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
import geometry_checks as gc
src=OUT/'stages/E7-uv-normals-corrected.blend'
glb=OUT/'character-rebuilt-textured.glb'
input_report=json.loads((OUT/'reports/clean-reimport.json').read_text())
assert hashlib.sha256(glb.read_bytes()).hexdigest()==input_report['glbSha256']
bpy.ops.wm.open_mainfile(filepath=str(src))
names=['Rebuild_Body_Clothing_SourceRetained','Rebuilt_FrontCollar_R']
sources={n:bpy.data.objects[n] for n in names}
source_data={}
for n,o in sources.items():
    o.data.calc_loop_triangles()
    points=[tuple(o.matrix_world@v.co) for v in o.data.vertices]
    tris=[tuple(t.vertices) for t in o.data.loop_triangles]
    polygons=[t.polygon_index for t in o.data.loop_triangles]
    kd=kdtree.KDTree(len(points))
    for i,p in enumerate(points):kd.insert(p,i)
    kd.balance()
    source_data[n]=(points,tris,polygons,kd)
existing=set(bpy.data.objects)
temp=bpy.data.scenes.new('TEMP_E7_ClothingContactDiagnostic')
bpy.context.window.scene=temp
bpy.ops.import_scene.gltf(filepath=str(glb))
imported=[o for o in bpy.data.objects if o not in existing]
report={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'glbSha256':input_report['glbSha256'],'objects':{}}
for n in names:
    so=sources[n];p,t,pi,kd=source_data[n]
    io=next(o for o in imported if o.type=='MESH' and (o.name==n or o.name.startswith(n+'.')))
    io.data.calc_loop_triangles()
    ip=[tuple(io.matrix_world@v.co) for v in io.data.vertices]
    it=[tuple(x.vertices) for x in io.data.loop_triangles]
    entry=next(o for o in input_report['objects'] if o['object']==n)
    contacts=[]
    for item in entry['selfContacts']['contactPairSamples']:
        samples=[]
        for tri_index in item['triangles']:
            ids=it[tri_index];coords=[ip[i] for i in ids]
            matches=[]
            for co in coords:
                found=kd.find_range(co,2e-6)
                matches.append([{'sourceVertex':i,'distance':d,'position':p[i]} for _,i,d in found])
            samples.append({'glbTriangle':tri_index,'glbVertexIndices':ids,'coordinates':coords,'sourceMatches':matches})
        contacts.append({'triangles':samples,'sat':gc.triangles_contact(samples[0]['coordinates'],samples[1]['coordinates'],1e-8)})
    source_contact=gc._self_contacts(p,t,pi,enabled=True,max_triangles=200000,max_candidates=3000000,max_samples=30)
    source_samples=[]
    for pair in source_contact.get('contactPairSamples',[]):
        source_samples.append({'triangles':pair['triangles'],'polygons':pair['polygons'],
                               'vertexIndices':[t[i] for i in pair['triangles']],
                               'coordinates':[[p[v] for v in t[i]] for i in pair['triangles']]})
    report['objects'][n]={'glbContacts':contacts,'sourceSAT':source_contact,'sourceContactPositions':source_samples,'modifiers':[m.name for m in so.modifiers]}
dest=OUT/'work/clothing-contact-e7';dest.mkdir(parents=True,exist_ok=True)
(dest/'diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
