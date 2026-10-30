import bpy,sys,json,hashlib,struct,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
from geometry_checks import inspect_mesh
from face_e4_refine_delta_v3 import _uv_digest,_positions,_topology
dest=OUT/'work/e7-head-contact-diagnostic'
src=OUT/'stages/E7-uv-normals-corrected.blend'
candidate=dest/'E7-mouth-local-test.blend'
bpy.ops.wm.open_mainfile(filepath=str(src))
head=bpy.data.objects['Head_Skin_Quad_Retopo']
before=[tuple(v.co) for v in head.data.vertices]
uv=_uv_digest(head);topo=_topology(head)
others={o.name:_positions(o) for o in bpy.context.scene.objects if o.type=='MESH' and o!=head}
bpy.ops.wm.open_mainfile(filepath=str(candidate))
head=bpy.data.objects['Head_Skin_Quad_Retopo']
changes=[{'vertex':v.index,'before':list(before[v.index]),'after':list(v.co)} for v in head.data.vertices if tuple(v.co)!=before[v.index]]
assert len(changes)==217
assert _uv_digest(head)==uv and _topology(head)==topo
assert all(_positions(bpy.data.objects[n])==h for n,h in others.items())
assert all(tuple(head.data.vertices[i].co)==before[i] for i in range(13754,13878)), 'Mouth rim or socket changed'
report={'sourceSha256':hashlib.sha256(src.read_bytes()).hexdigest(),'candidateSha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'operation':'Move only 217 upper mouth surround vertices on the existing smooth skin surface to make the annular rows nested. Mouth opening and inner wall unchanged.',
        'sourceObject':'Head_Skin_Quad_Retopo','changes':changes,'uvDigest':uv,'topologyDigest':topo,
        'mouthRimAndWallExact':True,'allOtherObjectsExact':True,'uvExact':True,'topologyExact':True,
        'maximumMove':max(math.dist(c['before'],c['after']) for c in changes),
        'affectedBounds':[[min(c['before'][k] for c in changes),max(c['before'][k] for c in changes)] for k in range(3)]}
report['sourceHeadTests']={str(t):inspect_mesh(head.data,head.matrix_world,self_intersections=True,max_self_triangles=40000,max_self_candidates=3000000,tolerance=t)['selfContacts'] for t in (1e-8,1e-10,0)}
assert all(r['contacts']==0 for r in report['sourceHeadTests'].values())
for ob in bpy.context.scene.objects:ob.select_set(False)
head.hide_set(False);head.select_set(True);bpy.context.view_layer.objects.active=head
glb=dest/'head-only-validation.glb'
bpy.ops.export_scene.gltf(filepath=str(glb),export_format='GLB',use_selection=True,export_apply=True,export_yup=True,export_normals=True,export_texcoords=True,export_materials='EXPORT')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(glb))
head=bpy.data.objects['Head_Skin_Quad_Retopo']
report['reimportTests']={str(t):inspect_mesh(head.data,head.matrix_world,self_intersections=True,max_self_triangles=40000,max_self_candidates=3000000,tolerance=t) for t in (1e-8,1e-10,0)}
assert all(r['selfContacts']['contacts']==0 for r in report['reimportTests'].values())
(dest/'verified-local-delta.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
delta={k:report[k] for k in ['sourceSha256','operation','sourceObject','changes','uvDigest','topologyDigest','maximumMove']}
(OUT/'reports/e7-mouth-annulus-delta.json').write_text(json.dumps(delta,indent=2),encoding='utf-8')
print('VERIFIED',len(changes),'vertices maxmove',report['maximumMove'],'source/reimport SAT zero for 1e-8,1e-10,0')
