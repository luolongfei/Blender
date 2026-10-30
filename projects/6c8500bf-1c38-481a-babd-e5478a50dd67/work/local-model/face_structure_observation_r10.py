"""Read-only current adult-face redesign observation. No render or saved edit.

Exports actual Basis/all relative keys, topology, skin domain, weights and eye,
lid, brow and oral attachments in temporary frame-one REST. Historical prejoin
head coordinates are evidence, not an approved replacement or a new candidate.
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

HEAD='R01 Head'
ORAL=('R02 Upper teeth','R02 Lower teeth','R02 Tongue','R03 Upper gingiva','R03 Lower gingiva')
REQUIRED=(HEAD,'R01 Sclera L','R01 Sclera R','R01 Textured iris L','R01 Textured iris R',*ORAL)
HISTORICAL=('R03 prejoin Head preserved',)
OUTPUT='work/reconstruction-r02/face-structure-observation-r10.json'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _array(data,field,width=3,dtype=np.float32):
    out=np.empty(len(data)*width,dtype);data.foreach_get(field,out)
    return out.reshape(-1,width)


def _ahash(array):
    a=np.ascontiguousarray(array);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _path(relative,suffix,source=False):
    p=project_path().resolve();path=(p/relative).resolve()
    if not(source and path==p/'work/character.blend'):path.relative_to(p/'work/reconstruction-r02')
    if path.suffix!=suffix:raise ValueError('Unexpected observation artifact type')
    return path


def _world(points,matrix):
    m=np.asarray(matrix,float);return np.asarray(points)@m[:3,:3].T+m[:3,3]


def _custom_summary(obj,uv):
    result={}
    for name in obj.keys():
        value=uv['_plain'](obj[name]);encoded=json.dumps(value,sort_keys=True)
        result[name]=value if len(encoded)<1800 else {'length':len(encoded),'sha256':hashlib.sha256(encoded.encode()).hexdigest()}
    return result


def _positions(data,kind):
    if kind=='MESH':return _array(data.vertices,'co')
    if kind=='CURVES':return _array(data.points,'position')
    if kind=='CURVE':
        rows=[]
        for spline in data.splines:
            if spline.type=='BEZIER':rows.extend(tuple(p.co) for p in spline.bezier_points)
            else:rows.extend(tuple(p.co[:3]) for p in spline.points)
        return np.asarray(rows,np.float32).reshape(-1,3)
    raise RuntimeError('Unsupported observed attachment type '+kind)


def _curve_fields(data):
    """Retain every legacy-curve control point and both Bezier handles."""
    arrays={};rows=[]
    for i,spline in enumerate(data.splines):
        prefix='spline_'+str(i).zfill(4);bezier=spline.type=='BEZIER'
        points=spline.bezier_points if bezier else spline.points
        fields={'co':_array(points,'co',3 if bezier else 4)}
        if bezier:
            fields.update(handle_left=_array(points,'handle_left'),handle_right=_array(points,'handle_right'))
        for name in ('radius','tilt','weight_softbody'):
            if len(points) and hasattr(points[0],name):fields[name]=_array(points,name,1).ravel()
        for name,value in fields.items():arrays[prefix+'_'+name]=value
        row={'prefix':prefix,'type':spline.type,'pointCount':len(points),'cyclic':spline.use_cyclic_u,
             'order':spline.order_u,'endpoint':spline.use_endpoint_u,'bezierInterpolation':spline.use_bezier_u,
             'resolution':spline.resolution_u,'fields':list(fields)}
        if bezier:row['handleTypes']=[(p.handle_left_type,p.handle_right_type) for p in points]
        rows.append(row)
    return rows,arrays


def _key_fields(key):
    fields={'co':_array(key.data,'co')}
    if len(key.data):
        for name,width in (('handle_left',3),('handle_right',3),('radius',1),('tilt',1)):
            if hasattr(key.data[0],name):
                value=_array(key.data,name,width);fields[name]=value if width>1 else value.ravel()
    return fields


def _fingerprint(obj,uv):
    keys=getattr(obj.data,'shape_keys',None)
    return _hash({'positions':_ahash(_positions(obj.data,obj.type)),
        'curveFields':{n:_ahash(a) for n,a in _curve_fields(obj.data)[1].items()} if obj.type=='CURVE' else None,
        'keys':[{'name':k.name,'fields':{n:_ahash(a) for n,a in _key_fields(k).items()},'settings':uv['_scalar_rna'](k),
                 'relative':k.relative_key.name} for k in keys.key_blocks] if keys else [],
        'animation':uv['_animation'](obj),'dataAnimation':uv['_animation'](obj.data),
        'keyAnimation':uv['_animation'](keys) if keys else None,
        'world':uv['_plain'](obj.matrix_world),'basis':uv['_plain'](obj.matrix_basis),
        'parent':uv['_plain'](obj.parent),'parentType':obj.parent_type,'parentBone':obj.parent_bone,
        'parentInverse':uv['_plain'](obj.matrix_parent_inverse),
        'modifiers':[uv['_scalar_rna'](m) for m in obj.modifiers]})


def _export_object(obj,prefix,arrays,uv):
    data=obj.data;raw=_positions(data,obj.type);keys=getattr(data,'shape_keys',None)
    arrays[prefix+'_raw']=raw
    row={'prefix':prefix,'type':obj.type,'data':data.name,'verticesOrPoints':len(raw),
        'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'visibleNow':obj.visible_get(),
        'matrixWorld':uv['_plain'](obj.matrix_world),'matrixBasis':uv['_plain'](obj.matrix_basis),
        'parent':uv['_plain'](obj.parent),'parentType':obj.parent_type,'parentBone':obj.parent_bone,
        'parentInverse':uv['_plain'](obj.matrix_parent_inverse),
        'animation':uv['_animation'](obj),'dataAnimation':uv['_animation'](data),
        'modifiers':[{'settings':uv['_scalar_rna'](m),'target':uv['_plain'](getattr(m,'object',None))} for m in obj.modifiers],
        'constraints':[uv['_scalar_rna'](c) for c in obj.constraints],
        'materialSlots':[s.material.name if s.material else None for s in obj.material_slots],
        'customSummary':_custom_summary(obj,uv),'dataCustomSummary':_custom_summary(data,uv)}
    if keys:
        row['keys']=[];row['keyAnimation']=uv['_animation'](keys)
        row['relativeKeys']=keys.use_relative;row['referenceKey']=keys.reference_key.name
        for i,key in enumerate(keys.key_blocks):
            name=prefix+'_key_'+str(i).zfill(3);fields=_key_fields(key)
            arrays[name]=fields['co']
            for field,value in fields.items():
                if field!='co':arrays[name+'_'+field]=value
            row['keys'].append({'name':key.name,'array':name,'relative':key.relative_key.name,
                'value':float(key.value),'mute':key.mute,'vertexGroup':key.vertex_group,
                'sliderMin':key.slider_min,'sliderMax':key.slider_max,'interpolation':key.interpolation,
                'elementCount':len(key.data),'elementType':type(key.data[0]).__name__ if len(key.data) else None,
                'fields':{field:name if field=='co' else name+'_'+field for field in fields}})
        if obj.type=='MESH':
            mixed,mix_record=uv['current_mix'](obj);row['currentMix']=mix_record
        elif obj.type=='CURVE':
            # Legacy Curve key layout is not a Mesh vertex array. Blender owns
            # the evaluated centre/handle mix, including Bezier interpolation.
            import bpy
            evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            if evaluated.type!='CURVE':raise RuntimeError('Legacy curve evaluation changed type: '+obj.name)
            mixed=_positions(evaluated.data,'CURVE')
            spline_rows,spline_arrays=_curve_fields(evaluated.data)
            for field,value in spline_arrays.items():arrays[prefix+'_evaluated_'+field]=value
            row['evaluatedSplines']=spline_rows
            row['currentMix']={'method':'Actual depsgraph evaluated legacy Curve controls and handles in temporary REST; raw key element layout is exported separately.',
                               'rawControlPoints':len(raw),'evaluatedControlPoints':len(mixed)}
        else:raise RuntimeError('Unexpected non-Mesh/non-Curve shape-key owner: '+obj.name)
    else:mixed=raw
    arrays[prefix+'_current_local']=mixed;world=_world(mixed,obj.matrix_world)
    arrays[prefix+'_current_world']=world
    row['currentWorldBounds']=[world.min(0).tolist(),world.max(0).tolist()]
    if obj.type=='MESH':
        for field,width,dtype,collection in (
            ('edges',2,np.int32,data.edges),('loop_vertex',1,np.int32,data.loops),
            ('face_start',1,np.int32,data.polygons),('face_size',1,np.int32,data.polygons),
            ('face_material',1,np.int32,data.polygons)):
            property_name={'edges':'vertices','loop_vertex':'vertex_index','face_start':'loop_start',
                'face_size':'loop_total','face_material':'material_index'}[field]
            values=_array(collection,property_name,width,dtype)
            arrays[prefix+'_'+field]=values if width>1 else values.ravel()
        row['uvLayers']=[x.name for x in data.uv_layers]
        row['attributes']=[{'name':a.name,'type':a.data_type,'domain':a.domain} for a in data.attributes]
        groups=[];weights=[];starts=[0]
        for v in data.vertices:
            for g in v.groups:groups.append(g.group);weights.append(g.weight)
            starts.append(len(groups))
        arrays[prefix+'_weight_starts']=np.asarray(starts,np.int32)
        arrays[prefix+'_weight_groups']=np.asarray(groups,np.int32)
        arrays[prefix+'_weight_values']=np.asarray(weights,np.float32)
        row['vertexGroups']=[{'name':g.name,'index':g.index,'locked':g.lock_weight} for g in obj.vertex_groups]
        for attr in data.attributes:
            if 'face_set' in attr.name and attr.data_type=='INT' and attr.domain=='FACE':
                arrays[prefix+'_face_sets']=_array(attr.data,'value',1,np.int32).ravel();row['faceSetAttribute']=attr.name
        if obj.name==HEAD:
            for name,field,width in (('R01_region_color','color',4),('R04 skin roughness','value',1)):
                attr=data.attributes.get(name)
                if attr is not None:
                    key=prefix+('_colour' if width==4 else '_roughness')
                    a=_array(attr.data,field,width);arrays[key]=a if width>1 else a.ravel()
            physical=[i for i,s in enumerate(obj.material_slots) if s.material and s.material.name=='R04 physical skin']
            if len(physical)!=1:raise RuntimeError('Current physical skin domain differs')
            mask=np.zeros(len(raw),bool)
            for polygon in data.polygons:
                if polygon.material_index==physical[0]:mask[list(polygon.vertices)]=True
            arrays[prefix+'_skin_mask']=mask;row['skinMaterialSlot']=physical[0]
    elif obj.type=='CURVES':
        arrays[prefix+'_curve_lengths']=np.asarray([c.points_length for c in data.curves],np.int32)
        arrays[prefix+'_radii']=_array(data.points,'radius',1).ravel()
    elif obj.type=='CURVE':
        spline_rows,spline_arrays=_curve_fields(data);row['splines']=spline_rows
        for field,value in spline_arrays.items():arrays[prefix+'_'+field]=value
    return row


def observe(source_relative,output_relative=OUTPUT):
    import bpy
    source=_path(source_relative,'.blend',source=True);output=_path(output_relative,'.json');payload=output.with_suffix('.npz')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Read the exact saved source in a fresh background Object Mode process')
    if output.exists() or payload.exists():raise RuntimeError('Retain historical observations; use new output names')
    missing=[name for name in REQUIRED if bpy.data.objects.get(name) is None]
    if missing:raise RuntimeError('Missing current face attachments: '+repr(missing))
    uv=_api('production_uv_r08');groom=_api('posterior_groom_r09');head=bpy.data.objects[HEAD]
    selected={name:bpy.data.objects[name] for name in REQUIRED}
    for obj in bpy.context.scene.objects:
        if obj.hide_render or obj.type not in {'MESH','CURVES','CURVE'}:continue
        if any(word in obj.name.lower() for word in ('cornea','iris','sclera','lashes','lid rim','soft brow')):
            selected[obj.name]=obj
    for name in HISTORICAL:
        if bpy.data.objects.get(name):selected[name]=bpy.data.objects[name]
    bpy.context.view_layer.update();source_sha=_sha(source);pose_before=uv['_pose_contract']()
    head_before=uv['_original_contract'](head,tuple(x.name for x in head.data.uv_layers))['sha256']
    object_before={name:_fingerprint(obj,uv) for name,obj in selected.items()}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    scene=bpy.context.scene;frame=scene.frame_current;subframe=scene.frame_subframe
    subdivisions=[(m,m.levels) for m in head.modifiers if m.type=='SUBSURF']
    arrays={};records={};report={}
    try:
        scene.frame_set(1);rig.data.pose_position='REST';bpy.context.view_layer.update()
        for i,(name,obj) in enumerate(selected.items()):records[name]=_export_object(obj,'object_'+str(i).zfill(2),arrays,uv)
        bones=[]
        for bone in rig.data.bones:
            posebone=rig.pose.bones[bone.name]
            bones.append({'name':bone.name,'parent':bone.parent.name if bone.parent else None,'deform':bone.use_deform,
                'headLocal':list(bone.head_local),'tailLocal':list(bone.tail_local),'restMatrix':uv['_plain'](bone.matrix_local),
                'poseMatrix':uv['_plain'](posebone.matrix),'poseBasis':uv['_plain'](posebone.matrix_basis),
                'customSummary':_custom_summary(posebone,uv),'constraints':[uv['_scalar_rna'](c) for c in posebone.constraints]})
        # Evaluate the same smooth surface used for final shading, not viewport0.
        for mod,_ in subdivisions:mod.levels=mod.render_levels
        bpy.context.view_layer.update();evaluated=head.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            arrays['head_render_world']=_world(_array(mesh.vertices,'co'),evaluated.matrix_world)
            arrays['head_render_triangles']=_array(mesh.loop_triangles,'vertices',3,np.int32)
            arrays['head_render_face_material']=_array(mesh.polygons,'material_index',1,np.int32).ravel()
            arrays['head_render_triangle_polygon']=_array(mesh.loop_triangles,'polygon_index',1,np.int32).ravel()
        finally:evaluated.to_mesh_clear()
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'helperSha256':_sha(Path(__file__)),'objects':records,
            'rig':{'name':rig.name,'worldMatrix':uv['_plain'](rig.matrix_world),'bones':bones,'animation':uv['_animation'](rig)},
            'historicalHeadInventory':[{'name':o.name,'vertices':len(o.data.vertices),'data':o.data.name,
                'hideRender':o.hide_render,'matrixWorld':uv['_plain'](o.matrix_world)} for o in bpy.data.objects
                if o.type=='MESH' and 'head' in o.name.lower()],
            'renderSurfaceLevels':[{'modifier':m.name,'viewportOriginal':old,'renderUsed':m.render_levels} for m,old in subdivisions],
            'frameObserved':1,'temporaryRigPose':'REST','currentHeadContractSha256':head_before,
            'savedBlend':False,'rendered':False,'geometryEdited':False,
            'purpose':'Adult feminine facial reconstruction study. Current exaggerated or flattened features and old prejoin geometry are evidence, not approved shape targets.',
            'limits':['REST current mix still contains original always-on smile and static keys; individual relative keys are exported for explicit offline separation.',
                'An R02 key ablation alone does not rescale its already modified eyes/iris or restore all mouth/rig bindings.',
                'No fixed ethnicity-derived proportions are encoded. Candidate requires actual neutral front, quarter, side and facial-action review.']}
    finally:
        for mod,old in subdivisions:mod.levels=old
        rig.data.pose_position=pose;scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before:raise RuntimeError('Observation did not restore the source rig pose')
    if uv['_original_contract'](head,tuple(x.name for x in head.data.uv_layers))['sha256']!=head_before:
        raise RuntimeError('Original head state changed during observation')
    if any(_fingerprint(obj,uv)!=object_before[name] for name,obj in selected.items()):
        raise RuntimeError('An eye/lid/brow/oral/historical attachment changed')
    if _sha(source)!=source_sha:raise RuntimeError('Saved source changed during observation')
    with payload.open('xb') as handle:np.savez_compressed(handle,**arrays)
    report['arrays']={'path':str(payload.relative_to(project_path())),'sha256':_sha(payload)}
    with output.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R10_FACE_STRUCTURE_OBSERVED '+json.dumps({'report':str(output),'sourceSha256':source_sha,'arraysSha256':report['arrays']['sha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',default=OUTPUT)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source,args.output)
