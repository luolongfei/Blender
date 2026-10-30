"""Explicit measured gather reconnection; preserves the author/front branch.

The target is an editable hidden point carrier with the same head parent and
matrix as the donor. A POINT-domain native Sample Index replaces positions
only for real gather roles 40/41, before current-head contact. The original
guides, node tree and all non-gather points remain. Source point order is an
explicit measured dependency, never inferred from a role label alone.
"""
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np

MAIN = 'R11 Donor / long hair main'
GROUP = 'R11 groom interpolation group'
CARRIER = 'R11 reference tie gather target positions'
STAGE = 'R11 reference gather tie transport'
TAKEOFF_CORRECTIONS = np.array([10330,10527,10540,10541,10542],np.int32)


def geometry(sockets):
    found=[s for s in sockets if s.type=='GEOMETRY']
    if len(found)!=1:raise RuntimeError('Expected one native geometry socket')
    return found[0]


def install(targets, measured, *, plan_sha256):
    import bpy
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Measured gather installation requires REST')
    obj=bpy.data.objects[MAIN]
    if not obj.get('R11_reference_detail_v10') or obj.get('R11_reference_tie_transport'):
        raise RuntimeError('Fresh saved v10-derived donor required')
    if bpy.data.objects.get(CARRIER):raise RuntimeError('Measured target carrier already exists')
    q=np.asarray(targets['target_world'],np.float64)
    source=np.asarray(targets['source_world'],np.float64)
    starts=np.asarray(targets['source_starts'],np.int32)
    lengths=np.asarray(targets['source_lengths'],np.int32)
    groups=np.asarray(targets['source_group'],np.int32)
    actual=measured['before-front_world']
    if q.shape!=actual.shape or source.shape!=actual.shape or not np.isfinite(q).all():raise RuntimeError('Measured target shape or finite check failed')
    for key,values in (('starts',starts),('lengths',lengths),('group',groups)):
        if not np.array_equal(values,measured['before-front_'+key]):raise RuntimeError('Actual before-contact gather topology changed: '+key)
    source_error=float(np.linalg.norm(actual-source,axis=1).max())
    if source_error>2e-6:raise RuntimeError('Actual before-contact positions changed since measured reconnection plan')
    ids=np.asarray(targets['gather_curve_ids'],np.int32)
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(groups)):
        raise RuntimeError('Explicit gathered-curve selection is invalid')
    if not np.all(np.isin(groups[ids]%256,(40,41))):raise RuntimeError('Plan selected a non-gather semantic region')
    selected=np.zeros(len(groups),bool);selected[ids]=True;point_selection=np.repeat(selected,lengths)
    if not np.array_equal(q[~point_selection],source[~point_selection]):raise RuntimeError('Offline plan moved a non-gather point')
    if not np.array_equal(np.asarray(targets['takeoff_correction_curve_ids'],np.int32),TAKEOFF_CORRECTIONS):raise RuntimeError('Only the five actually demonstrated gather takeoff corrections are allowed')
    if not np.all(selected[TAKEOFF_CORRECTIONS]):raise RuntimeError('Takeoff correction escaped the explicit true gather set')
    exception=set(TAKEOFF_CORRECTIONS.tolist())
    first=np.concatenate([np.arange(starts[i],starts[i]+(1 if int(i) in exception else 3)) for i in np.flatnonzero(selected)])
    if not np.array_equal(q[first],source[first]):raise RuntimeError('Offline plan changed a protected gather root/takeoff')
    # Keep all source-roundoff differences out of the unaffected branch. The
    # selected roots also use the current measured points, rather than a stale
    # equivalent point from another source serialization.
    q=q.copy();q[~point_selection]=actual[~point_selection];q[first]=actual[first]
    # Native selection skips p0 on all curves and first3 on every curve except
    # five actual old mid-segment penetrations. Tail pins are not edited here.
    point_selection[first]=False
    matrix=np.asarray(obj.matrix_world,np.float64);inverse=np.linalg.inv(matrix)
    local=(q@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32)
    mesh=bpy.data.meshes.new('R11 measured gather target points')
    mesh.vertices.add(len(local));mesh.vertices.foreach_set('co',local.ravel());mesh.update()
    mask_name='R11 measured reconnection selection'
    mask=mesh.attributes.new(mask_name,'BOOLEAN','POINT');mask.data.foreach_set('value',point_selection)
    curtains=np.asarray(targets['preserved_face_curtain_curve_ids'],np.int32)
    if np.any(curtains<0) or np.any(curtains>=len(groups)) or np.any(selected[curtains]):
        raise RuntimeError('Verified face curtains must be explicitly preserved')
    curtain_curve_mask=np.zeros(len(groups),bool);curtain_curve_mask[curtains]=True
    curtain_name='R11 verified temple curtain'
    curtain_attribute=mesh.attributes.new(curtain_name,'BOOLEAN','POINT')
    curtain_attribute.data.foreach_set('value',np.repeat(curtain_curve_mask,lengths))
    carrier=bpy.data.objects.new(CARRIER,mesh)
    if not obj.users_collection:raise RuntimeError('Main donor has no owning collection')
    obj.users_collection[0].objects.link(carrier)
    carrier.parent=obj.parent;carrier.parent_type=obj.parent_type;carrier.parent_bone=obj.parent_bone
    carrier.matrix_parent_inverse=obj.matrix_parent_inverse.copy();carrier.matrix_world=obj.matrix_world.copy()
    carrier.hide_render=True;carrier.display_type='WIRE';carrier.hide_set(True)
    carrier['R11_gather_target_source']=MAIN
    carrier['R11_gather_target_plan_sha256']=plan_sha256
    carrier['R11_gather_target_point_count']=len(q)
    carrier['R11_gather_target_contract']='Editable measured positions, POINT index correspondence to saved author/front output. Changing source guides, counts, interpolation order or scalp density requires rebuilding this plan; no animation cache is reused.'
    modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    old=modifier.node_group;old.use_fake_user=True
    if STAGE in old.nodes:raise RuntimeError('Gather reconnection stage already exists')
    # POINT indices are meaningful only at the measured density. Use the same
    # complete branch in viewport and render, retaining the original adaptive
    # tree for explicit later performance work rather than silently indexing a
    # render carrier with a lower-density viewport geometry.
    review_path=Path(__file__).with_name('adult_full_groom_review_r11.py')
    if hashlib.sha256(review_path.read_bytes()).hexdigest()!='699d0a96cd332682e9bc9ddf8e6d3d731053e19889196f7f92a65c569fa7f6ae':raise RuntimeError('Measured render-branch copier changed')
    copies={};replacements=[]
    tree=runpy.run_path(str(review_path))['copy_render_branch'](old,copies,replacements)
    for original,copied in copies.items():copied.name='R11 full-density dependency / '+original.name
    tree.name='R11 original-reference measured gather reconnection';modifier.node_group=tree
    destination=geometry(tree.nodes['R11 full head clearance'].inputs)
    if len(destination.links)!=1 or destination.links[0].from_node.name!='R11 reference local front detail':
        raise RuntimeError('Actual v10 front output must feed existing head contact')
    upstream=destination.links[0].from_socket;tree.links.remove(destination.links[0])
    info=tree.nodes.new('GeometryNodeObjectInfo');info.name='R11 measured gather position carrier'
    info.transform_space='RELATIVE';info.inputs['Object'].default_value=carrier;info.inputs['As Instance'].default_value=False
    position=tree.nodes.new('GeometryNodeInputPosition');index=tree.nodes.new('GeometryNodeInputIndex')
    sample=tree.nodes.new('GeometryNodeSampleIndex');sample.data_type='FLOAT_VECTOR';sample.domain='POINT';sample.clamp=False
    tree.links.new(info.outputs['Geometry'],sample.inputs['Geometry'])
    tree.links.new(position.outputs['Position'],sample.inputs['Value']);tree.links.new(index.outputs['Index'],sample.inputs['Index'])
    named_mask=tree.nodes.new('GeometryNodeInputNamedAttribute');named_mask.data_type='BOOLEAN';named_mask.inputs['Name'].default_value=mask_name
    sample_mask=tree.nodes.new('GeometryNodeSampleIndex');sample_mask.data_type='BOOLEAN';sample_mask.domain='POINT';sample_mask.clamp=False
    tree.links.new(info.outputs['Geometry'],sample_mask.inputs['Geometry']);tree.links.new(named_mask.outputs['Attribute'],sample_mask.inputs['Value']);tree.links.new(index.outputs['Index'],sample_mask.inputs['Index'])
    named=tree.nodes.new('GeometryNodeInputNamedAttribute');named.data_type='INT';named.inputs['Name'].default_value=GROUP
    role=tree.nodes.new('ShaderNodeMath');role.operation='MODULO';role.inputs[1].default_value=256.;tree.links.new(named.outputs['Attribute'],role.inputs[0])
    selected_field=tree.nodes.new('FunctionNodeBooleanMath');selected_field.operation='OR'
    for i,value in enumerate((40,41)):
        compare=tree.nodes.new('ShaderNodeMath');compare.operation='COMPARE';compare.inputs[1].default_value=value;compare.inputs[2].default_value=.1
        tree.links.new(role.outputs[0],compare.inputs[0]);tree.links.new(compare.outputs[0],selected_field.inputs[i])
    selected_mask=tree.nodes.new('FunctionNodeBooleanMath');selected_mask.operation='AND'
    tree.links.new(selected_field.outputs[0],selected_mask.inputs[0]);tree.links.new(sample_mask.outputs['Value'],selected_mask.inputs[1])
    change=tree.nodes.new('GeometryNodeSetPosition');change.name='R11 measured gather positions'
    tree.links.new(upstream,change.inputs['Geometry']);tree.links.new(selected_mask.outputs[0],change.inputs['Selection'])
    tree.links.new(sample.outputs['Value'],change.inputs['Position'])
    curtain_field=tree.nodes.new('GeometryNodeInputNamedAttribute');curtain_field.data_type='BOOLEAN';curtain_field.inputs['Name'].default_value=curtain_name
    curtain_sample=tree.nodes.new('GeometryNodeSampleIndex');curtain_sample.data_type='BOOLEAN';curtain_sample.domain='POINT';curtain_sample.clamp=False
    tree.links.new(info.outputs['Geometry'],curtain_sample.inputs['Geometry']);tree.links.new(curtain_field.outputs['Attribute'],curtain_sample.inputs['Value']);tree.links.new(index.outputs['Index'],curtain_sample.inputs['Index'])
    label=tree.nodes.new('GeometryNodeStoreNamedAttribute');label.name=STAGE;label.data_type='BOOLEAN';label.domain='POINT';label.inputs['Name'].default_value=curtain_name
    tree.links.new(change.outputs['Geometry'],label.inputs['Geometry']);tree.links.new(curtain_sample.outputs['Value'],label.inputs['Value']);tree.links.new(label.outputs['Geometry'],destination)
    tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
    obj['R11_reference_tie_transport']=True;obj['R11_reference_gather_reconnection_v2']=True;obj['R11_reference_gather_reconnection_v3']=True
    obj['R11_gather_takeoff_correction_curves']=json.dumps(TAKEOFF_CORRECTIONS.tolist())
    obj['R11_verified_face_curtain_curves']=json.dumps(curtains.tolist())
    target_sha=hashlib.sha256(q.astype(np.float64).tobytes()).hexdigest()
    carrier['R11_gather_target_world_sha256']=target_sha
    measured['gather_target_effective_world']=q
    return {'carrier':CARRIER,'carrierPoints':len(q),'gatherCurves':int(selected.sum()),'protectedOtherCurves':int((~selected).sum()),
            'sourceMeasuredDifferenceMaxMeters':source_error,'sourceParent':obj.parent.name if obj.parent else None,
            'explicitTargetCurveIdsSha256':hashlib.sha256(ids.tobytes()).hexdigest(),
            'allUnselectedCurvesProtectedIncludingMislabelledFaceWisps':True,
            'verifiedTempleCurtainCurves':curtains.tolist(),'semanticAttribute':curtain_name,'semanticAttributeDomain':'POINT; all points of each verified curve have the same value',
            'sourceParentBone':obj.parent_bone,'oldTreeRetained':old.name,'newTree':tree.name,'stage':STAGE,
            'beforeExistingHeadContact':True,'ownMeasuredPointIndexOnly':True,'crossRoleClumping':False,
            'rawDonorGuidesUnchanged':True,'allP0RootsPreserved':True,'firstThreeRootPointsPreservedExceptExplicitFive':True,'gatherP1P2CorrectionCurves':TAKEOFF_CORRECTIONS.tolist(),'effectiveTargetWorldSha256':target_sha,
            'viewportAndRenderUseSameCompleteDensity':True,'formerViewportSwitches':replacements,
            'planSha256':plan_sha256,'appearanceAccepted':False,
            'editingRequirement':'The original guide/scalp/interpolation order is a dependency. Edit the measured carrier directly or rebuild its plan after changes to that upstream groom.'}
