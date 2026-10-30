"""Read-only actual Blender inventory/export of official CC0 head candidates.

Run only in a separate background process on the official bundle, with
--disable-autoexec. No object or material edits, no scene save, no render.
"""
import bpy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

PROJECT = project_path()
SOURCE = PROJECT/'assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
OUT = PROJECT/'work/reconstruction-r02/female-basemesh-actual-r11'
EXPECTED = '3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
assert bpy.app.background, 'Separate background process required'
assert '--disable-autoexec' in sys.argv, 'Disable automatic script execution'
assert Path(bpy.data.filepath).resolve() == SOURCE.resolve(), 'Open only the official bundle'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
assert not OUT.with_suffix('.json').exists() and not OUT.with_suffix('.npz').exists(), 'Preserve prior outputs'

names = ['GEO-body_female_realistic', 'GEO-body_female_realistic.eye.L', 'GEO-body_female_realistic.eye.R']
for root in ['GEO-head_animation_realistic', 'GEO-head_sculpting_realistic']:
    names += [root, root+'.iris.L', root+'.iris.R', root+'.sclera.L', root+'.sclera.R']
arrays = {}
report = {'schema':1, 'source': str(SOURCE.relative_to(PROJECT)), 'source_sha256': EXPECTED,
          'blender': bpy.app.version_string, 'read_only':True,
          'coordinate_note':'Raw source mesh coordinates are local; separate world coordinates use object matrix_world. No modifiers evaluated or changes applied. Bundle display positions are not target character coordinates.',
          'objects':[], 'texts':{t.name:t.as_string() for t in bpy.data.texts},
          'all_scene_objects':[{'name':o.name,'type':o.type,'parent':o.parent.name if o.parent else None,'collections':[c.name for c in o.users_collection]} for o in bpy.data.objects],
          'missing':[], 'scene_units': {'system':bpy.context.scene.unit_settings.system, 'scale_length':bpy.context.scene.unit_settings.scale_length}}
for n, name in enumerate(names):
    obj = bpy.data.objects.get(name)
    if obj is None:
        report['missing'].append(name)
        continue
    prefix = 'object_%02d' % n
    item = {'name':name, 'prefix':prefix, 'type':obj.type, 'collections':[c.name for c in obj.users_collection],
            'parent':obj.parent.name if obj.parent else None, 'matrix_world':[list(r) for r in obj.matrix_world],
            'location':list(obj.location), 'rotation_euler':list(obj.rotation_euler), 'scale':list(obj.scale),
            'dimensions':list(obj.dimensions), 'hide_render':obj.hide_render,
            'modifiers':[], 'materials':[]}
    for mod in obj.modifiers:
        m={'name':mod.name,'type':mod.type,'show_viewport':mod.show_viewport,'show_render':mod.show_render}
        for k in ['levels','render_levels','sculpt_levels','total_levels','use_axis','use_clip','use_mirror_merge','merge_threshold']:
            if hasattr(mod,k):
                val=getattr(mod,k)
                m[k]=list(val) if hasattr(val,'__len__') and not isinstance(val,str) else val
        item['modifiers'].append(m)
    for slot in obj.material_slots:
        mat=slot.material
        if mat is None:
            item['materials'].append(None)
            continue
        item['materials'].append({'name':mat.name,'use_nodes':mat.use_nodes,'diffuse_color':list(mat.diffuse_color),
                                  'nodes':[{'name':node.name,'type':node.type,'image':node.image.name if getattr(node,'image',None) else None} for node in mat.node_tree.nodes] if mat.node_tree else []})
    if obj.type=='MESH':
        me=obj.data
        co=np.empty(len(me.vertices)*3,dtype=np.float32);me.vertices.foreach_get('co',co);co=co.reshape(-1,3)
        mat=np.asarray(obj.matrix_world,dtype=np.float64)
        world=co@mat[:3,:3].T+mat[:3,3]
        arrays[prefix+'_local']=co
        arrays[prefix+'_world']=world.astype(np.float32)
        loops=np.empty(len(me.loops),dtype=np.int32);me.loops.foreach_get('vertex_index',loops)
        starts=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('loop_start',starts)
        totals=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('loop_total',totals)
        arrays[prefix+'_corner_vertices']=loops;arrays[prefix+'_face_starts']=starts;arrays[prefix+'_face_lengths']=totals
        edges=np.empty(len(me.edges)*2,dtype=np.int32);me.edges.foreach_get('vertices',edges);arrays[prefix+'_edges']=edges.reshape(-1,2)
        material_indices=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('material_index',material_indices);arrays[prefix+'_material_indices']=material_indices
        item.update({'mesh':me.name,'vertices':len(me.vertices),'edges':len(me.edges),'faces':len(me.polygons),'corners':len(me.loops),
                     'local_bounds':[co.min(axis=0).tolist(),co.max(axis=0).tolist()], 'world_bounds':[world.min(axis=0).tolist(),world.max(axis=0).tolist()],
                     'polygon_sizes':{str(k):int(np.count_nonzero(totals==k)) for k in np.unique(totals)},
                     'uv_layers':[],'shape_keys':[], 'attributes':[{'name':a.name,'domain':a.domain,'data_type':a.data_type,'count':len(a.data)} for a in me.attributes]})
        edge_count={}
        for st,ln in zip(starts,totals):
            f=loops[st:st+ln]
            for a,b in zip(f,np.roll(f,-1)):
                e=tuple(sorted((int(a),int(b))));edge_count[e]=edge_count.get(e,0)+1
        boundary=[list(k) for k,v in edge_count.items() if v==1]
        arrays[prefix+'_boundary_edges']=np.asarray(boundary,dtype=np.int32).reshape(-1,2)
        item['boundary_edge_count']=len(boundary)
        item['nonmanifold_gt2_edge_count']=sum(v>2 for v in edge_count.values())
        for j,uv in enumerate(me.uv_layers):
            data=np.empty(len(me.loops)*2,dtype=np.float32);uv.data.foreach_get('uv',data)
            key=prefix+'_uv_%02d'%j;arrays[key]=data.reshape(-1,2)
            item['uv_layers'].append({'name':uv.name,'array':key,'min':arrays[key].min(axis=0).tolist(),'max':arrays[key].max(axis=0).tolist()})
        if me.shape_keys:
            for j,k in enumerate(me.shape_keys.key_blocks):
                data=np.empty(len(k.data)*3,dtype=np.float32);k.data.foreach_get('co',data)
                key=prefix+'_key_%02d'%j;arrays[key]=data.reshape(-1,3)
                item['shape_keys'].append({'name':k.name,'value':k.value,'relative_key':k.relative_key.name,'array':key})
        for attr_name in ['.sculpt_face_set']:
            attr=me.attributes.get(attr_name)
            if attr and attr.data_type=='INT':
                data=np.empty(len(attr.data),dtype=np.int32);attr.data.foreach_get('value',data)
                arrays[prefix+'_face_sets']=data
                item['face_set_ids']=[int(v) for v in np.unique(data)]
    report['objects'].append(item)
np.savez_compressed(OUT.with_suffix('.npz'), **arrays)
report['npz_sha256']=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
OUT.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('FEMALE_BASE_ACTUAL_R11 '+json.dumps({'json':str(OUT.with_suffix('.json')),'npz':str(OUT.with_suffix('.npz')),'objects':[{'name':o['name'],'verts':o.get('vertices'),'faces':o.get('faces'),'uv':o.get('uv_layers'),'keys':o.get('shape_keys'),'bounds':o.get('world_bounds')} for o in report['objects']],'missing':report['missing']},ensure_ascii=False))
