"""Read actual author surface-deformed guide inputs and signed skin distances.

Official source only. Disables the main style GN in memory, so the Surface
Deform result can be observed separately. No source save or render.
"""
import hashlib,json,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();source=P/'assets/blender-hair-styles-bystedt-2023/hair_nodes-female_hair_styles.blend'
stem=P/'work/reconstruction-r02/hair-donor-surface-deform-r11'
expected='1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48'
if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source.resolve():raise RuntimeError('Fresh official source required')
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise RuntimeError('Official source changed')
if stem.with_suffix('.json').exists() or stem.with_suffix('.npz').exists():raise RuntimeError('Preserve existing probe')
main=bpy.data.objects['long hair main']
for mod in main.modifiers:
    if mod.type=='NODES' and mod.node_group and not mod.node_group.name.startswith('Surface Deform'):
        mod.show_viewport=False;mod.show_render=False
scene=bpy.data.scenes.new('R11 donor guide deformation observation')
for name in ('Long hair head','long hair growth mesh','long hair main','long hair strands'):
    scene.collection.objects.link(bpy.data.objects[name])
bpy.context.window.scene=scene;bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()

def coordinates(obj):
    items,key=(obj.data.vertices,'co') if obj.type=='MESH' else (obj.data.position_data,'vector')
    a=np.empty(len(items)*3,dtype=np.float32);items.foreach_get(key,a);return a.reshape(-1,3)
def world(a,m):
    m=np.asarray(m,dtype=float);return a@m[:3,:3].T+m[:3,3]
head=bpy.data.objects['head'].evaluated_get(deps);hv=world(coordinates(head),head.matrix_world)
tree=BVHTree.FromPolygons([Vector(p) for p in hv],[tuple(p.vertices) for p in head.data.polygons],all_triangles=False)
layout_inverse=bpy.data.objects['Long hair head'].matrix_world.inverted();arrays={};rows=[]
for group,name in enumerate(('long hair main','long hair strands')):
    obj=bpy.data.objects[name];raw=coordinates(obj);ev=obj.evaluated_get(deps)
    if ev.type!='CURVES':raise RuntimeError('Unexpected surface-deformed type')
    evaluated=world(world(coordinates(ev),ev.matrix_world),layout_inverse)
    starts=np.array([c.first_point_index for c in ev.data.curves],dtype=np.int32)
    lengths=np.array([c.points_length for c in ev.data.curves],dtype=np.int32)
    signed=[];nearest=[];normal=[]
    for p in evaluated:
        hit,n,idx,d=tree.find_nearest(Vector(p));signed.append(float((Vector(p)-hit).dot(n)));nearest.append(tuple(hit));normal.append(tuple(n))
    signed=np.asarray(signed);prefix='guide_'+str(group)
    arrays[prefix+'_raw']=raw;arrays[prefix+'_surface_deformed']=evaluated;arrays[prefix+'_starts']=starts;arrays[prefix+'_lengths']=lengths
    arrays[prefix+'_signed_skin_distance']=signed;arrays[prefix+'_nearest_skin']=np.asarray(nearest);arrays[prefix+'_skin_normal']=np.asarray(normal)
    rows.append({'name':name,'storedPoints':len(raw),'evaluatedPoints':len(evaluated),'curves':len(starts),
                 'maximumStoredToSurfaceDeformedDistance':float(np.linalg.norm(evaluated-raw,axis=1).max()) if raw.shape==evaluated.shape else None,
                 'signedSkinDistanceDonorUnits':{'min':float(signed.min()),'median':float(np.median(signed)),'max':float(signed.max())},
                 'insidePointCount':int(np.count_nonzero(signed<-.002)),
                 'point142':{'worldInDonorLayout':evaluated[142].tolist(),'signedSkinDistanceDonorUnits':float(signed[142]),'nearest':nearest[142],'normal':normal[142]}})
growth=bpy.data.objects['long hair growth mesh'];ge=growth.evaluated_get(deps)
arrays['growth_raw']=coordinates(growth);arrays['growth_evaluated']=world(world(coordinates(ge),ge.matrix_world),layout_inverse)
report={'source':str(source.relative_to(P)),'sourceSha256':expected,'headEvaluatedVertices':len(hv),'headBounds':[hv.min(0).tolist(),hv.max(0).tolist()],
        'guides':rows,'growth':{'storedVertices':len(growth.data.vertices),'evaluatedVertices':len(ge.data.vertices),
             'modifiers':[{'name':m.name,'type':m.type,'showViewport':m.show_viewport} for m in growth.modifiers],
             'attributes':[{'name':a.name,'type':a.data_type,'domain':a.domain} for a in ge.data.attributes]},
        'savedBlend':False,'rendered':False,'sourceEditedOnDisk':False}
np.savez_compressed(stem.with_suffix('.npz'),**arrays)
report['arraysSha256']=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest()
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise RuntimeError('Official input changed')
stem.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('R11_DONOR_SURFACE_DEFORM_PROBE '+json.dumps(report),flush=True)
