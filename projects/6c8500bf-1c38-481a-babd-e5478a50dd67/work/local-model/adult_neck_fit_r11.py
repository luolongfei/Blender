"""Current R11 head/old-body/collar observation for a separate neck-fit study.

No import-time Blender activity, rendering or geometry edit. The first version
exports actual surfaces and restores every temporary display/pose change.
Neck fitting is deliberately not inferred from a screenshot or hidden by crop.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study'
OLD='R01 Head'
HIDE='R11 hide rejected head surface'
OUTPUT='work/reconstruction-r02/adult-neck-observation-r11.json'
GARMENTS=('Blouse_Tucked_Bodice','Blouse_PeterPan_Collar_L','Blouse_PeterPan_Collar_R',
          'Blouse_Puff_Sleeve_L','Blouse_Puff_Sleeve_R')


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _ahash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _array(data,field,width=3,dtype=np.float32):
    a=np.empty(len(data)*width,dtype);data.foreach_get(field,a);return a.reshape(-1,width)


def _world(points,matrix):
    m=np.asarray(matrix,float);return np.asarray(points)@m[:3,:3].T+m[:3,3]


def _path(relative,suffix):
    p=project_path().resolve();path=(p/relative).resolve();path.relative_to(p/'work/reconstruction-r02')
    if path.suffix!=suffix:raise ValueError('Unexpected independent artifact type')
    return path


def _mesh_geometry(mesh,prefix,arrays,matrix):
    arrays[prefix+'_local']=_array(mesh.vertices,'co')
    arrays[prefix+'_world']=_world(arrays[prefix+'_local'],matrix)
    for label,field,width,collection in (
        ('edges','vertices',2,mesh.edges),('loops','vertex_index',1,mesh.loops),
        ('face_starts','loop_start',1,mesh.polygons),('face_sizes','loop_total',1,mesh.polygons),
        ('face_materials','material_index',1,mesh.polygons)):
        a=_array(collection,field,width,np.int32);arrays[prefix+'_'+label]=a if width>1 else a.ravel()
    mesh.calc_loop_triangles();arrays[prefix+'_triangles']=_array(mesh.loop_triangles,'vertices',3,np.int32)
    arrays[prefix+'_triangle_polygons']=_array(mesh.loop_triangles,'polygon_index',1,np.int32).ravel()
    sets=mesh.attributes.get('.sculpt_face_set')
    if sets and sets.domain=='FACE':arrays[prefix+'_face_sets']=_array(sets.data,'value',1,np.int32).ravel()
    return {'prefix':prefix,'vertices':len(mesh.vertices),'faces':len(mesh.polygons),
        'worldBounds':[arrays[prefix+'_world'].min(0).tolist(),arrays[prefix+'_world'].max(0).tolist()],
        'matrixWorld':[list(r) for r in matrix]}


def _evaluated(obj,prefix,arrays,bpy):
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
    try:return _mesh_geometry(mesh,prefix,arrays,evaluated.matrix_world)
    finally:evaluated.to_mesh_clear()


def _object_contract(obj,uv):
    mesh=obj.data;keys=mesh.shape_keys
    return _hash({'raw':_ahash(_array(mesh.vertices,'co')),
        'loops':_ahash(_array(mesh.loops,'vertex_index',1,np.int32)),
        'polygons':_ahash(_array(mesh.polygons,'loop_total',1,np.int32)),
        'uvs':{layer.name:_ahash(_array(layer.data,'uv',2)) for layer in mesh.uv_layers},
        'weights':[[(g.group,float(g.weight)) for g in v.groups] for v in mesh.vertices],
        'groups':[(g.name,g.index) for g in obj.vertex_groups],
        'keys':[{'name':k.name,'co':_ahash(_array(k.data,'co')),'settings':uv['_scalar_rna'](k),
            'relative':k.relative_key.name} for k in keys.key_blocks] if keys else [],
        'keyAnimation':uv['_animation'](keys) if keys else None,'animation':uv['_animation'](obj),
        'matrixWorld':uv['_plain'](obj.matrix_world),'matrixBasis':uv['_plain'](obj.matrix_basis),
        'parentInverse':uv['_plain'](obj.matrix_parent_inverse),'parent':uv['_plain'](obj.parent),
        'parentType':obj.parent_type,'parentBone':obj.parent_bone,
        'modifiers':[uv['_scalar_rna'](m) for m in obj.modifiers],
        'materials':[s.material.name if s.material else None for s in obj.material_slots],
        'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hideSet':obj.hide_get()})


def observe(source_relative,output_relative=OUTPUT):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.json');payload=output.with_suffix('.npz')
    if not bpy.app.background or bpy.data.is_dirty or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh explicit saved R11 candidate required')
    if output.exists() or payload.exists():raise RuntimeError('Retain observations; use new output names')
    if not bpy.data.objects.get(HEAD) or not bpy.context.scene.get('R11_adult_female_head_candidate'):
        raise RuntimeError('Expected the new adult female head candidate')
    objects={name:bpy.data.objects[name] for name in (HEAD,OLD,*GARMENTS)}
    for name,obj in objects.items():
        if obj.type!='MESH':raise RuntimeError('Expected actual mesh surface: '+name)
    head=objects[HEAD];old=objects[OLD];hide=old.modifiers.get(HIDE)
    if not hide:raise RuntimeError('Old rejected face display modifier absent')
    uv=_api('production_uv_r08');bpy.context.view_layer.update()
    source_sha=_sha(source);pose_before=uv['_pose_contract']()
    contracts={name:_object_contract(obj,uv) for name,obj in objects.items()}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    scene=bpy.context.scene;frame=scene.frame_current;subframe=scene.frame_subframe
    hide_enabled=hide.show_viewport
    subdivision=[(m,m.levels) for obj in objects.values() for m in obj.modifiers if m.type=='SUBSURF']
    arrays={};rows={};report={}
    try:
        scene.frame_set(1);bpy.context.view_layer.update()
        rows['new_head_picture_pose']=_evaluated(head,'picture_new_head',arrays,bpy)
        rig.data.pose_position='REST'
        for modifier,_ in subdivision:modifier.levels=modifier.render_levels
        bpy.context.view_layer.update()
        for i,(name,obj) in enumerate(objects.items()):
            if name==OLD:continue
            prefix='rest_'+str(i).zfill(2)
            rows[name]=_evaluated(obj,prefix,arrays,bpy)
            rows[name]['materials']=[s.material.name if s.material else None for s in obj.material_slots]
            rows[name]['raw']=_mesh_geometry(obj.data,prefix+'_raw',arrays,obj.matrix_world)
            rows[name]['modifiers']=[uv['_scalar_rna'](m) for m in obj.modifiers]
            rows[name]['parent']=uv['_plain'](obj.parent);rows[name]['parentBone']=obj.parent_bone
        # The rejected old neck remains useful seam/body evidence. Only its new
        # display deletion is disabled, never the existing deformation stack.
        hide.show_viewport=False;bpy.context.view_layer.update()
        rows[OLD]=_evaluated(old,'rest_old_complete',arrays,bpy)
        mixed,mix_record=uv['current_mix'](old)
        rows[OLD]['raw']=_mesh_geometry(old.data,'rest_old_raw',arrays,old.matrix_world)
        arrays['rest_old_current_world']=_world(mixed,old.matrix_world)
        rows[OLD]['currentMix']=mix_record;rows[OLD]['materials']=[s.material.name if s.material else None for s in old.material_slots]
        groups=[];weights=[];starts=[0]
        for vertex in old.data.vertices:
            for group in vertex.groups:groups.append(group.group);weights.append(group.weight)
            starts.append(len(groups))
        arrays['old_weight_starts']=np.asarray(starts,np.int32);arrays['old_weight_groups']=np.asarray(groups,np.int32)
        arrays['old_weight_values']=np.asarray(weights,np.float32)
        rows[OLD]['vertexGroups']=[{'name':g.name,'index':g.index} for g in old.vertex_groups]
        actual_group=head.vertex_groups['R11 head and lower-neck context'];weights=np.zeros(len(head.data.vertices),np.float32)
        for vertex in head.data.vertices:
            for group in vertex.groups:
                if group.group==actual_group.index:weights[vertex.index]=group.weight
        arrays['new_head_actual_display_weights']=weights
        selection=json.loads((project_path()/'work/reconstruction-r02/female-head-selection-r11.json').read_text(encoding='utf-8'))
        exact=np.zeros(len(head.data.vertices),bool);exact[np.asarray(selection['original_vertex_indices'],np.int32)]=True
        arrays['new_head_original_anatomical_mask']=exact
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(Path(__file__)),
            'objects':rows,'femaleToTargetRestWorld':json.loads(head['R11_female_to_target']),
            'originalAnatomicalDisplayVertices':int(exact.sum()),'actualDisplayVertices':int(np.count_nonzero(weights)),
            'extraDisplayVertices':int(np.count_nonzero((weights>0)&~exact)),
            'sourceAnatomicalNeckRing':selection['neck_ring_old_indices'],
            'sourceAnatomicalNeckRingWorld':_world(_array(head.data.vertices,'co')[selection['neck_ring_old_indices']],head.matrix_world).tolist(),
            'bones':[{'name':b.name,'headLocal':list(b.head_local),'tailLocal':list(b.tail_local),
                'restMatrix':[list(r) for r in b.matrix_local]} for b in rig.data.bones if b.name in ('upperChest','neck','head')],
            'savedBlend':False,'geometryEdited':False,'rendered':False,
            'purpose':'Measure and join a real new neck to retained torso and collar; compare the 54-point anatomical neck boundary against the expanded display mask. Do not delete visible failures or crop the picture as a substitute for continuous anatomy.',
            'limits':['All fitting candidates remain pending. The observed R11 donor shoulder stub is visibly outside the blouse in side view.',
                'REST surfaces provide geometric fit evidence. Existing R11 head is still head-parented and has not been given final neck deformation weights.']}
    finally:
        hide.show_viewport=hide_enabled
        for modifier,levels in subdivision:modifier.levels=levels
        rig.data.pose_position=pose;scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before:raise RuntimeError('Observation did not restore source pose')
    for name,obj in objects.items():
        if _object_contract(obj,uv)!=contracts[name]:raise RuntimeError('Observation changed original object: '+name)
    if _sha(source)!=source_sha:raise RuntimeError('Saved source changed')
    with payload.open('xb') as handle:np.savez_compressed(handle,**arrays)
    report['arrays']={'path':str(payload.relative_to(project_path())),'sha256':_sha(payload)}
    with output.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R11_NECK_OBSERVED '+json.dumps({'report':str(output),'arraysSha256':report['arrays']['sha256'],'sourceSha256':source_sha}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('observe',),required=True)
    parser.add_argument('--source',required=True);parser.add_argument('--output',default=OUTPUT)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source,args.output)
