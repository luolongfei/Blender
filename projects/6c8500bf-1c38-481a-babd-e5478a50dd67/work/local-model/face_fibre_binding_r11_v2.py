"""Native triangle-frame surface binding for the six actual R11 brow/lash Curves.

Run only after the final head Mask. Each root binds to one actual evaluated L3
triangle, never to old/raw head indices. A copied Curves datablock stores three
vertex/corner indices and barycentric weights per strand, plus a tangent-frame
offset per point. GN rebuilds position from the deformed triangle without a
Python handler. Original editable positions, physical radii and materials stay
exact in both the retained source and the copied datablock.

apply()/validate()/probe_deformation()/restore() do not save. prepare() saves a
new independent candidate only after actual evaluated parity succeeds. Runtime
topology checks hide invalid strands; validate/export still FAIL on mismatch.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
OBJECTS=tuple(['R11 Adult natural brow '+s for s in ('L','R')]+
              ['R11 Adult lashes '+part+' '+s for s in ('L','R') for part in ('upper','lower')])
MARKER='R11_face_fibre_binding'
MODIFIER='R11 native face fibre surface binding'
GROUP='R11 face fibre triangle frames'
P='r11_fibre_bind_'
ATTRS=tuple(P+'vertex_'+str(i) for i in range(3))+tuple(P+'corner_'+str(i) for i in range(3))+(P+'barycentric',P+'frame_offset')
TOLERANCE=2e-6


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _array(items,field,width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype=dtype);items.foreach_get(field,a.ravel());return a
def _ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()
def _hash(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _transform(points,matrix):
    m=np.asarray(matrix,float);return np.asarray(points)@m[:3,:3].T+m[:3,3]
def _path(relative):
    base=project_path().resolve();path=(base/relative).resolve();path.relative_to(base/'work/reconstruction-r02')
    if path.suffix!='.blend':raise ValueError('Independent reconstruction Blend required')
    return path


def triangle_frames(triangles):
    """Orthonormal [tangent, bitangent, normal], no triangle stretch of fibres."""
    tri=np.asarray(triangles,float);edge=tri[:,1]-tri[:,0]
    lengths=np.linalg.norm(edge,axis=1);normal=np.cross(edge,tri[:,2]-tri[:,0]);area2=np.linalg.norm(normal,axis=1)
    if np.any(lengths<1e-8) or np.any(area2<1e-12):raise RuntimeError('Degenerate binding triangle')
    tangent=edge/lengths[:,None];normal/=area2[:,None]
    return np.stack((tangent,np.cross(normal,tangent),normal),axis=2)


def barycentric(point,triangle):
    a,b,c=np.asarray(triangle,float);u=b-a;v=c-a;q=np.asarray(point,float)-a
    uu=u@u;uv=u@v;vv=v@v;den=uu*vv-uv*uv
    if den<1e-24:raise RuntimeError('Degenerate nearest triangle')
    y=(vv*(q@u)-uv*(q@v))/den;z=(uu*(q@v)-uv*(q@u))/den
    weights=np.array((1.-y-z,y,z),float)
    if weights.min()<-.001 or weights.max()>1.001:raise RuntimeError('Nearest surface point falls outside its triangle')
    weights=np.clip(weights,0.,1.);return weights/weights.sum()


def rebuild(points,triangles,weights,offsets,counts):
    tri=np.asarray(points,float)[np.asarray(triangles,np.int32)]
    frames=triangle_frames(tri);centers=np.einsum('si,sij->sj',weights,tri)
    return np.repeat(centers,counts,axis=0)+np.einsum('nij,nj->ni',np.repeat(frames,counts,axis=0),offsets)


def rebuild_gn_float32(points,mapping):
    """Reference the real node input and each float32 vector operation.

    Relative Object Info transforms its geometry to float32 before the
    triangle edge subtraction. Keeping this transform in float64 until the
    last step gives a different frame on sub-millimeter L3 triangles. The
    original double reference remains reported separately by validate().
    """
    tri=np.asarray(points,np.float32)[mapping['vertices']]
    edge=(tri[:,1]-tri[:,0]).astype(np.float32)
    other=(tri[:,2]-tri[:,0]).astype(np.float32)
    length=np.sqrt(np.sum(edge*edge,axis=1,dtype=np.float32))
    cross=np.cross(edge,other).astype(np.float32)
    area2=np.sqrt(np.sum(cross*cross,axis=1,dtype=np.float32))
    if np.any(length<1e-8) or np.any(area2<1e-12):
        raise RuntimeError('Degenerate actual GN binding triangle')
    tangent=edge/length[:,None];normal=cross/area2[:,None]
    bitangent=np.cross(normal,tangent).astype(np.float32)
    weights=np.asarray(mapping['weights'],np.float32)
    result=(tri[:,0]*weights[:,0,None]).astype(np.float32)
    result=(result+tri[:,1]*weights[:,1,None]).astype(np.float32)
    result=(result+tri[:,2]*weights[:,2,None]).astype(np.float32)
    result=np.repeat(result,mapping['counts'],axis=0)
    offsets=np.asarray(mapping['offsets'],np.float32)
    for index,frame in enumerate((tangent,bitangent,normal)):
        term=(np.repeat(frame,mapping['counts'],axis=0)*offsets[:,index,None]).astype(np.float32)
        result=(result+term).astype(np.float32)
    return result


@contextlib.contextmanager
def _relative_gn_inputs(head):
    """Capture actual Relative Object Info coordinates, once per matrix.

    The temporary object/group are always removed before validation returns;
    no original object, raw fibre, mapping, shader or node group is edited.
    Cache lifetime is one validate() call, so expressions cannot reuse stale
    head geometry. Temporary datablocks may mark the in-memory session dirty;
    this function never saves it and is intended for production validation.
    """
    import bpy
    mesh=None;probe=None;group=None;cache={}
    stats={'lookups':0,'uniqueMatrices':0,'temporaryCaptureRemoved':False}
    try:
        mesh=bpy.data.meshes.new('R11 temporary validated GN input')
        probe=bpy.data.objects.new('R11 temporary validated GN input',mesh)
        bpy.context.scene.collection.objects.link(probe);probe.hide_render=True
        group=bpy.data.node_groups.new('R11 temporary validated relative input','GeometryNodeTree')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        info=group.nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE'
        info.inputs['Object'].default_value=head;info.inputs['As Instance'].default_value=False
        output=group.nodes.new('NodeGroupOutput');group.links.new(info.outputs['Geometry'],output.inputs['Geometry'])
        modifier=probe.modifiers.new('Temporary numerical input capture','NODES');modifier.node_group=group
        def capture(obj,expected_count):
            stats['lookups']+=1
            key=np.asarray(obj.matrix_world,np.float32).tobytes()
            if key not in cache:
                probe.matrix_world=obj.matrix_world.copy();bpy.context.view_layer.update()
                evaluated=probe.evaluated_get(bpy.context.evaluated_depsgraph_get())
                if evaluated.type!='MESH' or len(evaluated.data.vertices)!=expected_count:
                    raise RuntimeError('Actual GN input capture changed head point count')
                cache[key]=_array(evaluated.data.vertices,'co');stats['uniqueMatrices']+=1
            return cache[key]
        yield capture,stats
    finally:
        if probe is not None:bpy.data.objects.remove(probe,do_unlink=True)
        if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
        if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        bpy.context.view_layer.update();stats['temporaryCaptureRemoved']=True


def _head_mesh(head,triangulate=False):
    import bpy
    ev=head.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.data
    data={'co':_array(me.vertices,'co'),'loops':_array(me.loops,'vertex_index',1,np.int32).ravel(),
        'edges':_array(me.edges,'vertices',2,np.int32),'starts':_array(me.polygons,'loop_start',1,np.int32).ravel(),
        'sizes':_array(me.polygons,'loop_total',1,np.int32).ravel(),'matrix':np.asarray(ev.matrix_world,float)}
    data['world']=_transform(data['co'],data['matrix'])
    data['counts']={'points':len(me.vertices),'edges':len(me.edges),'faces':len(me.polygons),'corners':len(me.loops)}
    data['topologySha256']=_hash({'counts':data['counts'],**{k:_ah(data[k]) for k in ('edges','loops','starts','sizes')}})
    if triangulate:
        # These evaluated loop triangles are a neutral binding triangulation.
        # Its original corner IDs remain fixed; no later re-triangulation is used.
        me.calc_loop_triangles()
        data['triangles']=_array(me.loop_triangles,'vertices',3,np.int32)
        data['triangleCorners']=_array(me.loop_triangles,'loops',3,np.int32)
    return data


def _curve_contract(data):
    groom=_api('posterior_groom_r09');attrs=groom['_attributes'](data)
    return {'counts':[c.points_length for c in data.curves],
        'attributes':{name:{'domain':row['domain'],'type':row['type'],'sha256':_ah(row['values'])} for name,row in attrs.items() if name not in ATTRS},
        'materials':[m.name if m else None for m in data.materials]}


def _mapping(data):
    counts=np.asarray([c.points_length for c in data.curves],np.int32)
    vertices=np.column_stack([_array(data.attributes[P+'vertex_'+str(i)].data,'value',1,np.int32).ravel() for i in range(3)])
    corners=np.column_stack([_array(data.attributes[P+'corner_'+str(i)].data,'value',1,np.int32).ravel() for i in range(3)])
    weights=_array(data.attributes[P+'barycentric'].data,'vector')
    offsets=_array(data.attributes[P+'frame_offset'].data,'vector')
    return {'counts':counts,'vertices':vertices,'corners':corners,'weights':weights,'offsets':offsets}


def _mapping_hash(mapping):return _hash({k:_ah(v) for k,v in mapping.items()})


def _plan(obj,head_data,bvh):
    from mathutils import Vector
    points=_array(obj.data.position_data,'vector');counts=np.asarray([c.points_length for c in obj.data.curves],np.int32)
    if len(counts)==0 or np.any(counts<2) or counts.sum()!=len(points):raise RuntimeError('Invalid actual fibre offsets '+obj.name)
    starts=np.r_[0,np.cumsum(counts)[:-1]];world=_transform(points,obj.matrix_world)
    root_vertices=[];root_corners=[];weights=[];distances=[];signed=[]
    for root in world[starts]:
        nearest,normal,index,distance=bvh.find_nearest(Vector(root))
        if nearest is None or distance>.0015:raise RuntimeError('Fibre root is not close to actual face: '+obj.name+' distance='+str(distance))
        ids=head_data['triangles'][index];tri=head_data['world'][ids]
        root_vertices.append(ids);root_corners.append(head_data['triangleCorners'][index]);weights.append(barycentric(nearest,tri))
        distances.append(float(distance));signed.append(float((Vector(root)-nearest).dot(normal)))
    ids=np.asarray(root_vertices,np.int32);corners=np.asarray(root_corners,np.int32);weights=np.asarray(weights,np.float32)
    head_local=_transform(head_data['world'],np.linalg.inv(np.asarray(obj.matrix_world,float)))
    triangles=head_local[ids];frames=triangle_frames(triangles);centers=np.einsum('si,sij->sj',weights,triangles)
    offsets=np.einsum('nji,nj->ni',np.repeat(frames,counts,axis=0),points-np.repeat(centers,counts,axis=0)).astype(np.float32)
    mapping={'counts':counts,'vertices':ids,'corners':corners,'weights':weights,'offsets':offsets}
    reconstructed=rebuild(head_local,ids,weights,offsets,counts)
    error=np.linalg.norm(_transform(reconstructed,obj.matrix_world)-world,axis=1)
    if error.max()>TOLERANCE:raise RuntimeError('Offline neutral triangle reconstruction exceeded tolerance')
    return mapping,{'strands':len(counts),'points':len(points),'rootDistanceMeters':[min(distances),float(np.median(distances)),max(distances)],
        'signedRootNormalDistanceMeters':[min(signed),float(np.median(signed)),max(signed)],
        'offlineReconstructionMaxMeters':float(error.max()),'mappingSha256':_mapping_hash(mapping)}


def _node_group(bpy,head,counts):
    group=bpy.data.node_groups.new(GROUP,'GeometryNodeTree');nodes=group.nodes;links=group.links
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    source=nodes.new('NodeGroupInput');output=nodes.new('NodeGroupOutput')
    obj=nodes.new('GeometryNodeObjectInfo');obj.transform_space='RELATIVE';obj.inputs['Object'].default_value=head;obj.inputs['As Instance'].default_value=False
    position=nodes.new('GeometryNodeInputPosition');geometry=obj.outputs['Geometry']
    def named(name,kind):
        n=nodes.new('GeometryNodeInputNamedAttribute');n.data_type=kind;n.inputs['Name'].default_value=name;return n.outputs['Attribute']
    def sample(kind,domain,value,index):
        n=nodes.new('GeometryNodeSampleIndex');n.data_type=kind;n.domain=domain;n.clamp=False
        links.new(geometry,n.inputs['Geometry']);links.new(value,n.inputs['Value']);links.new(index,n.inputs['Index']);return n.outputs['Value']
    def vec(operation,a,b=None,scale=None):
        n=nodes.new('ShaderNodeVectorMath');n.operation=operation;links.new(a,n.inputs[0])
        if b is not None:links.new(b,n.inputs[1])
        if scale is not None:links.new(scale,n.inputs['Scale'])
        return n.outputs['Vector']
    def component(vector):
        n=nodes.new('ShaderNodeSeparateXYZ');links.new(vector,n.inputs[0]);return [n.outputs[x] for x in ('X','Y','Z')]
    def equal(a,b):
        # Counts and all current indices are below float32's exact integer limit.
        n=nodes.new('ShaderNodeMath');n.operation='COMPARE';n.inputs[2].default_value=.1;links.new(a,n.inputs[0])
        if isinstance(b,(int,float)):n.inputs[1].default_value=b
        else:links.new(b,n.inputs[1])
        return n.outputs[0]
    def both(a,b):
        n=nodes.new('FunctionNodeBooleanMath');n.operation='AND';links.new(a,n.inputs[0]);links.new(b,n.inputs[1]);return n.outputs[0]
    size=nodes.new('GeometryNodeAttributeDomainSize');size.component='MESH';links.new(geometry,size.inputs['Geometry'])
    valid=equal(size.outputs['Point Count'],counts['points'])
    for socket,key in (('Edge Count','edges'),('Face Count','faces'),('Face Corner Count','corners')):valid=both(valid,equal(size.outputs[socket],counts[key]))
    corner_vertex=nodes.new('GeometryNodeVertexOfCorner')
    # Unconnected Corner Index is the evaluated face-corner domain Index.
    vertices=[]
    for i in range(3):
        index=named(P+'vertex_'+str(i),'INT');corner=named(P+'corner_'+str(i),'INT')
        vertices.append(sample('FLOAT_VECTOR','POINT',position.outputs['Position'],index))
        actual=sample('INT','CORNER',corner_vertex.outputs['Vertex Index'],corner)
        valid=both(valid,equal(actual,index))
    tangent=vec('NORMALIZE',vec('SUBTRACT',vertices[1],vertices[0]))
    cross=vec('CROSS_PRODUCT',vec('SUBTRACT',vertices[1],vertices[0]),vec('SUBTRACT',vertices[2],vertices[0]))
    normal=vec('NORMALIZE',cross);bitangent=vec('CROSS_PRODUCT',normal,tangent)
    # A collapsed triangle is another explicit runtime failure, never an origin snap.
    length=nodes.new('ShaderNodeVectorMath');length.operation='LENGTH';links.new(cross,length.inputs[0])
    usable=nodes.new('ShaderNodeMath');usable.operation='GREATER_THAN';usable.inputs[1].default_value=1e-12;links.new(length.outputs['Value'],usable.inputs[0]);valid=both(valid,usable.outputs[0])
    bary=component(named(P+'barycentric','FLOAT_VECTOR'));offset=component(named(P+'frame_offset','FLOAT_VECTOR'))
    terms=[vec('SCALE',v,scale=w) for v,w in zip(vertices,bary)]+[vec('SCALE',v,scale=w) for v,w in zip((tangent,bitangent,normal),offset)]
    target=terms[0]
    for term in terms[1:]:target=vec('ADD',target,term)
    setpos=nodes.new('GeometryNodeSetPosition');links.new(source.outputs['Geometry'],setpos.inputs['Geometry']);links.new(target,setpos.inputs['Position'])
    invalid=nodes.new('FunctionNodeBooleanMath');invalid.operation='NOT';links.new(valid,invalid.inputs[0])
    delete=nodes.new('GeometryNodeDeleteGeometry');delete.domain='CURVE';delete.mode='ALL'
    links.new(setpos.outputs['Geometry'],delete.inputs['Geometry']);links.new(invalid.outputs[0],delete.inputs['Selection']);links.new(delete.outputs['Geometry'],output.inputs['Geometry'])
    group['R11_runtime_guard']='Count + bound face-corner vertex identities + nondegenerate triangle. Full SHA validation is additionally mandatory before save/export.'
    return group


def _protected(head):
    eye=_api('eye_margin_refine_r11');face=_api('runtime_face_refresh_r08');state=eye['_state'](head,0)
    state['allHeadMaterials']=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    return _hash(state)


def _external(bpy):
    rows=[]
    for obj in bpy.context.scene.objects:
        row={'name':obj.name,'object':obj.as_pointer(),'data':None if obj.name in OBJECTS else getattr(getattr(obj,'data',None),'as_pointer',lambda:None)(),
            'parent':obj.parent.as_pointer() if obj.parent else None,'world':[list(r) for r in obj.matrix_world],
            'materials':[s.material.as_pointer() if s.material else None for s in obj.material_slots],
            'modifiers':[(m.name,m.type) for m in obj.modifiers if m.name!=MODIFIER]}
        if obj.type=='ARMATURE':row['pose']=[(b.name,[list(r) for r in b.matrix_basis]) for b in obj.pose.bones]
        rows.append(row)
    return _hash({'objects':rows,'actions':[(a.name,a.as_pointer()) for a in bpy.data.actions],
        'frame':[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe]})


def _validate_with_gn_inputs(neutral,capture):
    """Actual current-pose evaluation. Raises on topology/hidden fibres/parity/radius failure."""
    import bpy
    if MARKER not in bpy.context.scene:raise RuntimeError('No recorded R11 fibre binding')
    record=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[record['head']];current=_head_mesh(head)
    mismatch={};mappings={}
    for name,item in record['objects'].items():
        obj=bpy.data.objects[name];mapping=_mapping(obj.data);mappings[name]=mapping
        corners=mapping['corners'];ids=mapping['vertices'];in_range=np.all((corners>=0)&(corners<len(current['loops'])),axis=1)
        matches=np.zeros(len(ids),bool);matches[in_range]=np.all(current['loops'][corners[in_range]]==ids[in_range],axis=1)
        if current['counts']!=record['headCounts']:matches[:]=False
        mismatch[name]=int((~matches).sum())
    if current['topologySha256']!=record['headTopologySha256'] or any(mismatch.values()):
        raise RuntimeError('R11 fibre topology mismatch; save/export forbidden: '+json.dumps({'expected':record['headTopologySha256'],'actual':current['topologySha256'],
            'countsExpected':record['headCounts'],'countsActual':current['counts'],'runtimeInvalidStrands':mismatch,'hiddenIsNotPass':True}))
    results={};maximum=0.;maximum_double=0.
    for name,item in record['objects'].items():
        obj=bpy.data.objects[name];mapping=mappings[name];original=bpy.data.hair_curves.get(item['originalData'])
        if original is None or _hash(_curve_contract(original))!=item['originalContractSha256']:raise RuntimeError('Retained original fibre data changed '+name)
        original_contract=_curve_contract(original);current_contract=_curve_contract(obj.data)
        geometric_contract=dict(current_contract);geometric_contract['materials']=original_contract['materials']
        expected_current=item.get('currentMaterialContractSha256',item['originalContractSha256'])
        if _hash(current_contract)!=expected_current or _hash(geometric_contract)!=item['originalContractSha256'] or _mapping_hash(mapping)!=item['mappingSha256']:
            raise RuntimeError('Raw positions/radii/materials or binding map changed '+name)
        if obj.data.name!=item['candidateData'] or not obj.modifiers.get(MODIFIER):raise RuntimeError('Recorded binding stack missing '+name)
        if obj.modifiers[MODIFIER].node_group.name!=record['nodeGroup'] or not obj.modifiers[MODIFIER].show_viewport or not obj.modifiers[MODIFIER].show_render:
            raise RuntimeError('Native fibre binding is not active '+name)
        head_local=_transform(current['world'],np.linalg.inv(np.asarray(obj.matrix_world,float)))
        expected_double=rebuild(head_local,mapping['vertices'],mapping['weights'],mapping['offsets'],mapping['counts'])
        actual_gn_input=capture(obj,current['counts']['points'])
        expected=rebuild_gn_float32(actual_gn_input,mapping)
        expected=_api('upper_lash_corrective_validation_r11')['expected_final'](obj,expected,record)
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.data
        actual_counts=np.asarray([c.points_length for c in data.curves],np.int32)
        if not np.array_equal(actual_counts,mapping['counts']):raise RuntimeError('Runtime guard removed fibres; save/export forbidden: '+json.dumps({'object':name,'expectedStrands':len(mapping['counts']),'actualStrands':len(actual_counts),'missingStrands':len(mapping['counts'])-len(actual_counts)}))
        actual=_array(data.position_data,'vector');matrix=np.asarray(evaluated.matrix_world,float)
        world=_transform(actual,matrix);expected_world=_transform(expected,obj.matrix_world)
        error=float(np.linalg.norm(world-expected_world,axis=1).max());maximum=max(maximum,error)
        expected_double=_api('upper_lash_corrective_validation_r11')['expected_final'](obj,expected_double,record)
        double_error=float(np.linalg.norm(world-_transform(expected_double,obj.matrix_world),axis=1).max())
        maximum_double=max(maximum_double,double_error)
        if error>TOLERANCE:raise RuntimeError('Actual GN triangle-frame parity failed '+name+' '+str(error))
        radii=_array(data.attributes['radius'].data,'value',1)
        if not np.array_equal(radii,_array(original.attributes['radius'].data,'value',1)):raise RuntimeError('Actual physical fibre radii changed '+name)
        row={'expectedStrands':len(mapping['counts']),'actualStrands':len(actual_counts),'hiddenStrands':0,'actualGNParityMaxMeters':error,
             'actualGNFloat64ReferenceDifferenceMaxMeters':double_error,
             'float64ReferenceMeaning':'Float64 base triangle-frame binding; an applicable recorded late upper-lash correction casts that base to float32 and applies the same four sequential float32 fields. Without an applicable correction this remains the original pure float64 reference.',
             'parityReference':'Actual Relative Object Info float32 input and stepwise float32 node arithmetic',
             'rawOriginalContractExact':expected_current==item['originalContractSha256'],
             'rawGeometryAndAttributesExact':True,'documentedMaterialChange':item.get('materialChange'),
             'rawAndEvaluatedRadiiExact':True}
        if neutral:
            original_world=_transform(_array(original.position_data,'vector'),obj.matrix_world)
            neutral_error=float(np.linalg.norm(world-original_world,axis=1).max());row['actualNeutralParityMaxMeters']=neutral_error
            if neutral_error>TOLERANCE:raise RuntimeError('Actual neutral fibre shape changed '+name+' '+str(neutral_error))
        results[name]=row
    return {'headTopologySha256':current['topologySha256'],'objects':results,'actualGNParityMaxMeters':maximum,
        'actualGNFloat64ReferenceDifferenceMaxMeters':maximum_double,'parityToleranceMeters':TOLERANCE,
        'frame':[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe],'neutralRequired':neutral,
        'appearanceAccepted':False,'blinkGazeCompatibilityEstablished':False,'sourceSaved':False}


def validate(neutral=False):
    """Validate exact data and native float32 parity; neutral guard unchanged."""
    import bpy
    if MARKER not in bpy.context.scene:raise RuntimeError('No recorded R11 fibre binding')
    record=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[record['head']]
    with _relative_gn_inputs(head) as (capture,stats):
        result=_validate_with_gn_inputs(neutral,capture)
    result['gnInputCapture']=stats
    result['precisionEvidence']='work/reconstruction-r02/adult-blink-binding-precision-r11.json'
    return result


def apply():
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    if MARKER in bpy.context.scene:raise RuntimeError('Retain/restore the existing binding before rebinding')
    head=bpy.data.objects[HEAD]
    if head.library or head.data.library or len(head.data.vertices)!=10582:raise RuntimeError('Explicit complete current adult head required')
    if head.data.shape_keys and any(abs(k.value)>1e-8 for k in head.data.shape_keys.key_blocks[1:]):raise RuntimeError('Neutral face keys required for binding')
    multires=[m for m in head.modifiers if m.type=='MULTIRES']
    if len(multires)!=1 or multires[0].levels!=3 or multires[0].render_levels!=3:raise RuntimeError('Current viewport/render L3 must agree')
    for name in OBJECTS:
        obj=bpy.data.objects[name]
        if obj.type!='CURVES' or obj.library or obj.data.library or len(obj.modifiers):raise RuntimeError('Inspect existing fibre stack before binding '+name)
        if not obj.data.attributes.get('radius') or any(obj.data.attributes.get(n) for n in ATTRS):raise RuntimeError('Actual radii required and binding attributes must be fresh '+name)
    bpy.context.view_layer.update()
    source_sha=_sha(source);before=_protected(head);external=_external(bpy);qa=_api('face_contact_audit_r11')
    created=[];group=None;success=False
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),'head':HEAD,'objects':{},
        'method':'Actual evaluated L3 triangle barycentre + orthonormal tangent/bitangent/normal offset for every fibre point; native GN, no frame handler.',
        'neutralToleranceMeters':TOLERANCE,'appearanceAccepted':False,'sourceSaved':False,'facialPerformanceApproved':False,
        'limits':['Bind only after final Mask; full current topology SHA is mandatory in validate/save/export. Runtime hiding on mismatch is a failure, never a pass.',
            'Each short strand follows one root triangle rigidly; it does not bend along a changing face surface. Extreme brow compression may still require grooming correctives.',
            'This supplies surface following, not blink/squint/brow muscles or lash collision correction. Actual neutral, intermediate/full blink, brow and head-turn images remain required.']}
    try:
        with qa['rest_evaluation']():
            hd=_head_mesh(head,True)
            bvh=BVHTree.FromPolygons([Vector(p) for p in hd['world']],[tuple(t) for t in hd['triangles']],all_triangles=True)
            prepared={name:_plan(bpy.data.objects[name],hd,bvh) for name in OBJECTS}
            group=_node_group(bpy,head,hd['counts']);group['R11_head_topology_sha256']=hd['topologySha256']
            report.update(headTopologySha256=hd['topologySha256'],headCounts=hd['counts'],nodeGroup=group.name)
            for name,(mapping,plan) in prepared.items():
                obj=bpy.data.objects[name];original=obj.data;candidate=original.copy();candidate.name=original.name+' - R11 surface bound'
                original_fake=original.use_fake_user;created.append((obj,original,candidate,original_fake));contract=_curve_contract(original)
                for i in range(3):
                    for label,array in (('vertex',mapping['vertices']),('corner',mapping['corners'])):
                        candidate.attributes.new(P+label+'_'+str(i),'INT','CURVE').data.foreach_set('value',np.ascontiguousarray(array[:,i]))
                candidate.attributes.new(P+'barycentric','FLOAT_VECTOR','CURVE').data.foreach_set('vector',mapping['weights'].ravel())
                candidate.attributes.new(P+'frame_offset','FLOAT_VECTOR','POINT').data.foreach_set('vector',mapping['offsets'].ravel())
                if _curve_contract(candidate)!=contract:raise RuntimeError('Copied raw fibre geometry/radius/material changed '+name)
                original.use_fake_user=True;obj.data=candidate;modifier=obj.modifiers.new(MODIFIER,'NODES');modifier.node_group=group
                report['objects'][name]={**plan,'originalData':original.name,'candidateData':candidate.name,
                    'originalFakeUser':original_fake,'originalContractSha256':_hash(contract)}
            bpy.context.view_layer.update();bpy.context.scene[MARKER]=json.dumps(report)
            report['actualNeutralValidation']=validate(neutral=True)
        if _protected(head)!=before or _external(bpy)!=external or _sha(source)!=source_sha:raise RuntimeError('Head, external object/action/current pose, or source file changed')
        report['validationScope']='Original editable head/keys/UV/attributes/materials, original fibre complete attributes/point order/radii/material assignments, actual GN output and external object/data/action identities. Full immutable hair/cloth caches are not re-read.'
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            for obj,original,candidate,fake in reversed(created):
                modifier=obj.modifiers.get(MODIFIER)
                if modifier:obj.modifiers.remove(modifier)
                obj.data=original;original.use_fake_user=fake
                if candidate.users==0:bpy.data.hair_curves.remove(candidate)
            if group and group.users==0:bpy.data.node_groups.remove(group)
            bpy.context.view_layer.update()


def probe_deformation():
    """Temporary local cage lift/shear to exercise real rotation; fully removed.

    It is a mechanism test, not a blink shape or facial performance approval.
    No save/render. Head and original fibre contracts must be exact afterwards.
    """
    import bpy
    if MARKER not in bpy.context.scene:raise RuntimeError('Apply native binding first')
    head=bpy.data.objects[HEAD];qa=_api('face_contact_audit_r11');before=_protected(head)
    if head.data.shape_keys and any(abs(k.value)>1e-8 for k in head.data.shape_keys.key_blocks[1:]):raise RuntimeError('Neutral new-face keys required for the temporary probe')
    had_keys=head.data.shape_keys is not None;active=head.active_shape_key_index;temporary=None;result=None
    def snapshots():
        return {name:_transform(_array(bpy.data.objects[name].evaluated_get(bpy.context.evaluated_depsgraph_get()).data.position_data,'vector'),bpy.data.objects[name].matrix_world) for name in OBJECTS}
    try:
        with qa['rest_evaluation']():
            baseline=validate(neutral=True);start=snapshots();hd0=_head_mesh(head)
            neutral=_array(head.data.vertices,'co');world=_transform(neutral,head.matrix_world);delta=np.zeros_like(world)
            for side in ('L','R'):
                eye=bpy.data.objects['R11 Adult sclera '+side];ev=eye.evaluated_get(bpy.context.evaluated_depsgraph_get())
                eye_points=_transform(_array(ev.data.vertices,'co'),ev.matrix_world);center=(eye_points.min(0)+eye_points.max(0))*.5
                d=world-center;weight=np.exp(-((d[:,0]/.025)**2+(d[:,2]/.026)**2+(d[:,1]/.040)**2))
                delta[:,2]+=.0006*weight;delta[:,1]+=.0003*weight*(d[:,0]/.025)
            local_delta=delta@np.linalg.inv(np.asarray(head.matrix_world,float)[:3,:3]).T
            if not had_keys:head.shape_key_add(name='Basis',from_mix=False)
            temporary=head.shape_key_add(name='R11 temporary fibre following probe',from_mix=False)
            temporary.relative_key=head.data.shape_keys.reference_key;temporary.data.foreach_set('co',(neutral+local_delta).astype(np.float32).ravel());temporary.value=1.
            bpy.context.view_layer.update();changed=validate(neutral=False);finish=snapshots();hd1=_head_mesh(head)
            motion={};angles={}
            for name in OBJECTS:
                mapping=_mapping(bpy.data.objects[name].data);tri=mapping['vertices']
                f0=triangle_frames(hd0['world'][tri]);f1=triangle_frames(hd1['world'][tri])
                angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',f0,f1)-1.)*.5,-1.,1.)))
                motion[name]=float(np.linalg.norm(finish[name]-start[name],axis=1).max());angles[name]=float(angle.max())
            if min(motion.values())<.00003 or max(angles.values())<.05:raise RuntimeError('Temporary local deformation did not exercise actual translation/rotation')
            result={'baseline':baseline,'deformed':changed,'actualFibreMotionMaxMeters':motion,'actualTriangleRotationMaxDegrees':angles,
                'temporaryCageDisplacementMaxMeters':float(np.linalg.norm(delta,axis=1).max()),'probeIsBlink':False,'appearanceAccepted':False,'sourceSaved':False}
    finally:
        if temporary is not None and head.data.shape_keys:head.shape_key_remove(temporary)
        if not had_keys and head.data.shape_keys and len(head.data.shape_keys.key_blocks)==1:head.shape_key_clear()
        head.active_shape_key_index=active;bpy.context.view_layer.update()
    if _protected(head)!=before:raise RuntimeError('Temporary probe did not exactly restore head/keys/material state')
    result['restored']=validate(neutral=True);result['temporaryKeyRemoved']=True;return result


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    record=json.loads(bpy.context.scene[MARKER])
    for name,item in record['objects'].items():
        obj=bpy.data.objects[name];old=bpy.data.hair_curves[item['originalData']]
        if obj.data.name!=item['candidateData'] or _mapping_hash(_mapping(obj.data))!=item['mappingSha256'] or _hash(_curve_contract(obj.data))!=item['originalContractSha256'] or _hash(_curve_contract(old))!=item['originalContractSha256']:
            raise RuntimeError('Fibre data edited after binding; retain edits before rollback '+name)
    for name,item in record['objects'].items():
        obj=bpy.data.objects[name];candidate=obj.data;candidate.use_fake_user=True
        modifier=obj.modifiers.get(MODIFIER)
        if modifier:obj.modifiers.remove(modifier)
        obj.data=bpy.data.hair_curves[item['originalData']];obj.data.use_fake_user=item['originalFakeUser']
    bpy.data.node_groups[record['nodeGroup']].use_fake_user=True
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'originalFibreDataRestored':True,'candidateDataAndNodesRetained':True,'sourceSaved':False}


def prepare(source_relative,output_relative,probe=False):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.fibre-binding.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New output required; preserve historical candidates')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit independent source required')
    report=apply()
    if probe:report['actualTemporaryDeformationProbe']=probe_deformation()
    report['preSaveValidation']=validate(neutral=True)
    bpy.context.scene[MARKER]=json.dumps(report)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),candidateSaved=True)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_FACE_FIBRES_BOUND '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--probe',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.probe)
