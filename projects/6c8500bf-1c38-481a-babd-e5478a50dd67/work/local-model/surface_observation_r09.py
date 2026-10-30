"""Observe actual active skin/eye shaders and the existing forelock paths."""
import bpy, hashlib, json, runpy
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent
api = runpy.run_path(str(folder/'reference_final_stills_r08.py'))
p = api['project_path']()
out = p/'work/reconstruction-r02/surface-observation-r09.json'
arrays = out.with_suffix('.npz')
if out.exists() or arrays.exists():
    raise RuntimeError('Preserve existing observations')
source = Path(bpy.data.filepath)
api['_neutral'](bpy.data.objects['Character_Rig'])
bpy.data.objects['Character_Rig'].data.pose_position='REST'
bpy.context.view_layer.update()
names = ['R01 Head', 'R01 Sclera L', 'R01 Sclera R',
         'R01 Textured iris L', 'R01 Textured iris R', 'R03 Forelocks from crown']
result = {'source':str(source.relative_to(p)), 'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'objects':{}, 'savedBlend':False}
for name in names:
    obj = bpy.data.objects[name]
    row = {'materials':[], 'modifiers':[{'name':m.name,'type':m.type,'render':m.show_render} for m in obj.modifiers]}
    for mat in obj.data.materials:
        nodes=[]
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                values={}
                for socket in node.inputs:
                    if not hasattr(socket,'default_value'):continue
                    value=socket.default_value
                    if hasattr(value,'to_list'):value=value.to_list()
                    elif not isinstance(value,(int,float,str,bool,type(None))):
                        try:value=list(value)
                        except TypeError:value=str(value)
                    values[socket.name]={'value':value,'links':[(l.from_node.name,l.from_socket.name) for l in socket.links]}
                nodes.append({'name':node.name,'type':node.type,'attribute':getattr(node,'attribute_name',None),'inputs':values})
        row['materials'].append({'name':mat.name if mat else None,'nodes':nodes})
    result['objects'][name]=row
groom=bpy.data.objects['R03 Forelocks from crown']
points=np.empty((len(groom.data.position_data),3),np.float32)
groom.data.position_data.foreach_get('vector',points.ravel())
radii=np.empty(len(points),np.float32);groom.data.attributes['radius'].data.foreach_get('value',radii)
head=bpy.data.objects['R01 Head'];ev=head.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
keys=head.data.shape_keys.key_blocks
key_arrays={}
for index,key in enumerate(keys):
    co=np.empty((len(key.data),3),np.float32);key.data.foreach_get('co',co.ravel())
    key_arrays['key_'+str(index)]=co
result['headKeys']=[{'index':i,'name':k.name,'relative':k.relative_key.name,'value':float(k.value)} for i,k in enumerate(keys)]
result['headMatrix']=np.array(head.matrix_world).tolist()
try:
    headpoints=np.array([ev.matrix_world@v.co for v in mesh.vertices],np.float32)
    mesh.calc_loop_triangles();tris=np.array([t.vertices[:] for t in mesh.loop_triangles],np.int32)
    normals=np.array([ev.matrix_world.to_3x3()@v.normal for v in mesh.vertices],np.float32)
finally:ev.to_mesh_clear()
with arrays.open('xb') as stream:
    np.savez_compressed(stream,forelock_positions=points,forelock_radii=radii,
        forelock_starts=np.array([c.first_point_index for c in groom.data.curves]),
        forelock_lengths=np.array([c.points_length for c in groom.data.curves]),
        forelock_matrix=np.array(groom.matrix_world),head_world=headpoints,head_triangles=tris,head_normals=normals,
        head_edges=np.array([e.vertices[:] for e in head.data.edges],np.int32),**key_arrays)
result['arrays']=str(arrays.relative_to(p))
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_SURFACE_OBSERVED', flush=True)
