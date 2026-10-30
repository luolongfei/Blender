"""Export the author's actual evaluated Multires L3 female mesh, read-only.

Independent background process only. Does not change modifier levels, sculpt,
topology, visibility, materials, or save the source. The whole evaluated body is
kept so extraction decisions remain reversible. No rendering is performed.
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

P = project_path()
SOURCE=P/'assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
OUT=P/'work/reconstruction-r02/female-multires-actual-r11'
SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
assert bpy.app.background and '--disable-autoexec' in sys.argv
assert Path(bpy.data.filepath).resolve()==SOURCE.resolve()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
assert not OUT.with_suffix('.npz').exists() and not OUT.with_suffix('.json').exists()

body=bpy.data.objects['GEO-body_female_realistic']
multires=next(m for m in body.modifiers if m.type=='MULTIRES')
assert multires.show_viewport and multires.levels==3 and multires.total_levels==3
names=['GEO-body_female_realistic','GEO-body_female_realistic.eye.L','GEO-body_female_realistic.eye.R']
dg=bpy.context.evaluated_depsgraph_get()
arrays={}
report={'source':str(SOURCE.relative_to(P)),'source_sha256':SHA,'blender':bpy.app.version_string,
        'method':'evaluated_get at unchanged author Multires L3; no topology deletion, no modifier changes, no source save',
        'objects':[],'body_layout_offset':list(body.matrix_world.translation),
        'coordinates':'Evaluated local coordinates use original object frame; world includes asset-library display offset; body_space uses inverse female body matrix_world for all objects.'}
body_inv=np.linalg.inv(np.asarray(body.matrix_world,dtype=np.float64))
for n,name in enumerate(names):
    obj=bpy.data.objects[name]
    evaluated=obj.evaluated_get(dg)
    me=evaluated.data
    prefix='eval_%02d'%n
    co=np.empty(len(me.vertices)*3,dtype=np.float32);me.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    mx=np.asarray(evaluated.matrix_world,dtype=np.float64)
    world=co@mx[:3,:3].T+mx[:3,3]
    bm=body_inv@mx
    bodyco=co@bm[:3,:3].T+bm[:3,3]
    loops=np.empty(len(me.loops),dtype=np.int32);me.loops.foreach_get('vertex_index',loops)
    starts=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('loop_start',starts)
    lengths=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('loop_total',lengths)
    arrays[prefix+'_local']=co
    arrays[prefix+'_body_space']=bodyco.astype(np.float32)
    arrays[prefix+'_world']=world.astype(np.float32)
    arrays[prefix+'_corner_vertices']=loops
    arrays[prefix+'_face_starts']=starts
    arrays[prefix+'_face_lengths']=lengths
    material=np.empty(len(me.polygons),dtype=np.int32);me.polygons.foreach_get('material_index',material)
    arrays[prefix+'_material_indices']=material
    item={'name':name,'prefix':prefix,'vertices':len(me.vertices),'faces':len(me.polygons),'corners':len(me.loops),
          'matrix_world':mx.tolist(),'matrix_to_body_space':bm.tolist(),
          'local_bounds':[co.min(axis=0).tolist(),co.max(axis=0).tolist()],
          'body_space_bounds':[bodyco.min(axis=0).tolist(),bodyco.max(axis=0).tolist()],
          'uv_layers':[],'face_sets':None}
    for j,uv in enumerate(me.uv_layers):
        xy=np.empty(len(me.loops)*2,dtype=np.float32);uv.data.foreach_get('uv',xy)
        key=prefix+'_uv_%02d'%j;arrays[key]=xy.reshape(-1,2)
        item['uv_layers'].append({'name':uv.name,'array':key})
    fs=me.attributes.get('.sculpt_face_set')
    if fs and fs.domain=='FACE' and fs.data_type=='INT':
        a=np.empty(len(fs.data),dtype=np.int32);fs.data.foreach_get('value',a)
        arrays[prefix+'_face_sets']=a;item['face_sets']=prefix+'_face_sets'
    # Optional quick selection indices only: complete evaluated geometry stays intact.
    if n==0:
        minimum_z=np.minimum.reduceat(bodyco[loops,2], starts)
        for zval,label in [(1.30,'upper_shoulder'),(1.34,'head_neck'),(1.40,'head')]:
            key=prefix+'_faces_minz_'+label
            arrays[key]=np.flatnonzero(minimum_z>=zval).astype(np.int32)
            item[label+'_selection']={'array':key,'min_body_z':zval,'faces':len(arrays[key]),'note':'Whole faces only; heuristic selection, not final anatomical seam.'}
    report['objects'].append(item)
    print('EVALUATED_FEMALE_OBJECT '+json.dumps({'name':name,'vertices':len(me.vertices),'faces':len(me.polygons)}),flush=True)
np.savez_compressed(OUT.with_suffix('.npz'),**arrays)
report['npz_sha256']=hashlib.sha256(OUT.with_suffix('.npz').read_bytes()).hexdigest()
OUT.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('EVALUATED_FEMALE_R11_COMPLETE '+json.dumps({'json':str(OUT.with_suffix('.json')),'npz':str(OUT.with_suffix('.npz')),'sha256':report['npz_sha256']}),flush=True)
