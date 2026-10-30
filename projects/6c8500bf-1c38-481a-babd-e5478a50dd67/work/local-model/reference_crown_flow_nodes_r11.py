"""Downstream crown start-flow field on the retained measured tie carrier.

Only new offset/selection attributes are added to a copied carrier data block.
Its existing positions and attributes stay exact. The new field follows the
gather reconnection and precedes actual head contact, so the tie carrier cannot
overwrite it. Strength zero bypasses the complete new group.
"""
import hashlib
import json
import numpy as np

MAIN='R11 Donor / long hair main'
CARRIER='R11 reference tie gather target positions'
PREDECESSOR='R11 reference gather tie transport'
STAGE='R11 reference crown root flow'
OFFSET='R11 crown start local offset'
POINT_MASK='R11 crown start point selection'
CURVE_LOOKUP='R11 crown start curve lookup'
CURVE_TAG='R11 crown flow affected curve'
GROUP='R11 groom interpolation group'


def geometry(sockets):
    found=[s for s in sockets if s.type=='GEOMETRY']
    if len(found)!=1:raise RuntimeError('Expected one geometry socket')
    return found[0]


def math(tree,op,a,b):
    node=tree.nodes.new('ShaderNodeMath');node.operation=op
    for i,value in enumerate((a,b)):
        if isinstance(value,(int,float,np.integer,np.floating)):node.inputs[i].default_value=float(value)
        else:tree.links.new(value,node.inputs[i])
    return node.outputs[0]


def named_sample(tree,info,index,name,kind):
    named=tree.nodes.new('GeometryNodeInputNamedAttribute');named.data_type=kind;named.inputs['Name'].default_value=name
    sample=tree.nodes.new('GeometryNodeSampleIndex');sample.data_type=kind;sample.domain='POINT';sample.clamp=False
    tree.links.new(info.outputs['Geometry'],sample.inputs['Geometry']);tree.links.new(named.outputs['Attribute'],sample.inputs['Value']);tree.links.new(index.outputs['Index'],sample.inputs['Index'])
    return sample.outputs['Value']


def install(plan,actual,*,plan_sha256,strength=1.):
    import bpy
    if not np.isfinite(strength) or not 0.<=strength<=1.:raise RuntimeError('Crown flow strength must be within [0,1]')
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Crown field installation requires REST')
    donor=bpy.data.objects[MAIN];carrier=bpy.data.objects.get(CARRIER)
    if not donor.get('R11_reference_gather_reconnection_v2') or donor.get('R11_reference_crown_flow'):
        raise RuntimeError('An actual saved tie-reconnected source without crown flow is required')
    if carrier is None or carrier.type!='MESH' or carrier.parent!=donor.parent or carrier.parent_type!=donor.parent_type or carrier.parent_bone!=donor.parent_bone:
        raise RuntimeError('Existing same-parent measured tie carrier is required')
    if np.max(np.abs(np.asarray(carrier.matrix_world)-np.asarray(donor.matrix_world)))>2e-6:
        raise RuntimeError('Actual carrier and donor frames differ')
    source=np.asarray(plan['source_post_tie_world'],np.float64)
    offsets=np.asarray(plan['offset_world'],np.float64)
    selected_points=np.asarray(plan['target_point_mask'],bool)
    ids=np.asarray(plan['selected_curve_ids'],np.int32)
    starts=np.asarray(plan['starts'],np.int32);lengths=np.asarray(plan['lengths'],np.int32);groups=np.asarray(plan['group'],np.int32)
    q=np.asarray(actual['before-gather_world'],np.float64)
    if source.shape!=q.shape or offsets.shape!=q.shape or selected_points.shape!=(len(q),) or not np.isfinite(offsets).all():raise RuntimeError('Complete measured crown field shape/finite mismatch')
    for key,values in (('starts',starts),('lengths',lengths),('group',groups)):
        if not np.array_equal(values,actual['before-gather_'+key]):raise RuntimeError('Current post-tie point correspondence changed: '+key)
    residual=float(np.linalg.norm(q-source,axis=1).max())
    if residual>2e-6:raise RuntimeError('Actual post-tie positions differ from crown plan; recapture/rebase is required')
    if len(carrier.data.vertices)!=len(q):raise RuntimeError('Carrier point count no longer matches measured stage')
    if ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(starts)):
        raise RuntimeError('Explicit crown curve identities invalid')
    selected_curves=np.zeros(len(starts),bool);selected_curves[ids]=True
    point_owners=np.repeat(selected_curves,lengths)
    if np.any(selected_points&~point_owners) or np.any(offsets[~selected_points]!=0.):raise RuntimeError('Offsets escaped the explicit point/curve selection')
    if np.any(selected_points[starts]) or np.any(offsets[starts]!=0.):raise RuntimeError('Root positions must be exactly protected')
    terminal=np.concatenate([np.arange(s+n-3,s+n) for s,n in zip(starts,lengths)])
    if np.any(selected_points[terminal]) or np.any(offsets[terminal]!=0.):raise RuntimeError('Every terminal three-point shape must remain protected')
    curtains=np.asarray(json.loads(donor['R11_verified_face_curtain_curves']),np.int32)
    if np.any(selected_curves[curtains]):raise RuntimeError('Verified temple curtains may not be restyled by this field')
    if np.any(selected_points[np.asarray(plan['connector_protected_point_mask'],bool)]):raise RuntimeError('Crown field overlaps the protected tie connector')
    if float(np.linalg.norm(offsets,axis=1).max())>.020:raise RuntimeError('Crown start field exceeds the scoped 20 mm displacement envelope')
    inverse=np.linalg.inv(np.asarray(donor.matrix_world,np.float64)[:3,:3])
    local=(offsets@inverse.T).astype(np.float32)
    old_data=carrier.data;old_data.use_fake_user=True
    if any(old_data.attributes.get(name) is not None for name in (OFFSET,POINT_MASK,CURVE_LOOKUP)):
        raise RuntimeError('Crown field attributes already exist')
    new_data=old_data.copy();new_data.name=old_data.name+' crown flow attributes';carrier.data=new_data
    vector=new_data.attributes.new(OFFSET,'FLOAT_VECTOR','POINT');vector.data.foreach_set('vector',local.ravel())
    mask=new_data.attributes.new(POINT_MASK,'BOOLEAN','POINT');mask.data.foreach_set('value',selected_points)
    # Curve context Index intentionally addresses the first curve_count entries.
    # This is a typed lookup, independent of the carrier's geometric coordinates.
    lookup_values=np.zeros(len(q),bool);lookup_values[:len(starts)]=selected_curves
    lookup=new_data.attributes.new(CURVE_LOOKUP,'BOOLEAN','POINT');lookup.data.foreach_set('value',lookup_values)
    new_data.update();carrier.update_tag()
    modifier=next(m for m in donor.modifiers if m.type=='NODES' and m.show_render)
    old_tree=modifier.node_group;old_tree.use_fake_user=True
    if STAGE in old_tree.nodes:raise RuntimeError('Crown flow stage already exists')
    tree=old_tree.copy();tree.name='R11 reference crown-to-gather start flow';modifier.node_group=tree
    destination=geometry(tree.nodes['R11 full head clearance'].inputs)
    if len(destination.links)!=1 or destination.links[0].from_node.name!=PREDECESSOR:
        raise RuntimeError('New crown field must follow the actual measured gather reconnection')
    incoming=destination.links[0].from_socket;tree.links.remove(destination.links[0])
    group=bpy.data.node_groups.new('R11 smooth crown start field','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    control=group.interface.new_socket(name='Strength',in_out='INPUT',socket_type='NodeSocketFloat')
    control.default_value=float(strength);control.min_value=0.;control.max_value=1.
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    inputs=group.nodes.new('NodeGroupInput');output=group.nodes.new('NodeGroupOutput')
    info=group.nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE';info.inputs['Object'].default_value=carrier;info.inputs['As Instance'].default_value=False
    index=group.nodes.new('GeometryNodeInputIndex')
    displacement=named_sample(group,info,index,OFFSET,'FLOAT_VECTOR')
    point_selection=named_sample(group,info,index,POINT_MASK,'BOOLEAN')
    curve_selection=named_sample(group,info,index,CURVE_LOOKUP,'BOOLEAN')
    clamped=math(group,'MINIMUM',math(group,'MAXIMUM',inputs.outputs['Strength'],0.),1.)
    scale=group.nodes.new('ShaderNodeVectorMath');scale.operation='SCALE';group.links.new(displacement,scale.inputs[0]);group.links.new(clamped,scale.inputs['Scale'])
    change=group.nodes.new('GeometryNodeSetPosition');change.name='R11 local crown start offsets'
    group.links.new(inputs.outputs['Geometry'],change.inputs['Geometry']);group.links.new(point_selection,change.inputs['Selection']);group.links.new(scale.outputs['Vector'],change.inputs['Offset'])
    tag=group.nodes.new('GeometryNodeStoreNamedAttribute');tag.data_type='BOOLEAN';tag.domain='CURVE';tag.inputs['Name'].default_value=CURVE_TAG
    group.links.new(change.outputs['Geometry'],tag.inputs['Geometry']);group.links.new(curve_selection,tag.inputs['Value'])
    switch=group.nodes.new('GeometryNodeSwitch');switch.input_type='GEOMETRY'
    group.links.new(math(group,'GREATER_THAN',clamped,0.),switch.inputs['Switch']);group.links.new(inputs.outputs['Geometry'],switch.inputs['False']);group.links.new(tag.outputs['Geometry'],switch.inputs['True'])
    group.links.new(switch.outputs['Output'],output.inputs['Geometry'])
    node=tree.nodes.new('GeometryNodeGroup');node.node_tree=group;node.name=STAGE;node.inputs['Strength'].default_value=float(strength)
    tree.links.new(incoming,node.inputs['Geometry']);tree.links.new(node.outputs['Geometry'],destination)
    group.update_tag();tree.update_tag();donor.update_tag();bpy.context.view_layer.update()
    donor['R11_reference_crown_flow']=True;carrier['R11_crown_flow_plan_sha256']=plan_sha256
    return {'stage':STAGE,'positionAfter':PREDECESSOR,'positionBefore':'R11 full head clearance',
            'strength':float(strength),'range':[0.,1.],'zeroStrengthUsesExactGeometryBypass':True,
            'retainedCarrierData':old_data.name,'newCarrierData':new_data.name,'carrierObjectUnchanged':True,
            'newAttributes':[OFFSET,POINT_MASK,CURVE_LOOKUP],'curveSemanticTag':CURVE_TAG,
            'selectedCurves':len(ids),'selectedPoints':int(selected_points.sum()),'sourcePositionResidualMaxMeters':residual,
            'maximumWorldOffsetMeters':float(np.linalg.norm(offsets,axis=1).max()),'oldTreeRetained':old_tree.name,
            'rawGuidesUnchanged':True,'appearanceAccepted':False,'planSha256':plan_sha256}
