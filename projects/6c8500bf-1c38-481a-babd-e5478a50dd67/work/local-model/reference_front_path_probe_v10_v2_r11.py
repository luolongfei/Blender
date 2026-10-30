"""Current-head clearance of central free-segment midpoints and native curves.

The first 1 mm of arc is reported as the attachment zone. Everything beyond
that is a free shaft and retains the existing -0.5 mm maximum intrusion guard.
Native Blender resampling additionally observes the actual curved path rather
than assuming its control-point polygon is the final rendered hair shape.
"""
import runpy
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
HERE=Path(__file__).resolve().parent;GROUP='R11 groom interpolation group'
ROOT_ZONE=.001;LIMIT=-.0005;COUNT=128

def probe(arrays):
    errors=[];fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));_,skin,head=fit['_head_trees'](bpy.data.objects['R11 Female head study'])
    q=arrays['final-taper_world'];starts=arrays['final-taper_starts'];lengths=arrays['final-taper_lengths'];groups=arrays['final-taper_group'];selected=np.flatnonzero(np.isin(groups%256,(18,19)))
    mids=[];owners=[];segments=[];arcs=[]
    for i in selected:
        path=q[starts[i]:starts[i]+lengths[i]];distance=np.linalg.norm(np.diff(path,axis=0),axis=1);arc=np.r_[0.,np.cumsum(distance)]
        mids.extend((path[:-1]+path[1:])*.5);owners.extend([int(i)]*(len(path)-1));segments.extend(range(len(path)-1));arcs.extend((arc[:-1]+distance*.5).tolist())
    mids=np.asarray(mids,np.float32);owners=np.asarray(owners,np.int32);segments=np.asarray(segments,np.int32);arcs=np.asarray(arcs,np.float32)
    def clearance(points):
        values=np.empty(len(points),np.float32)
        for i,p in enumerate(points):
            hit,n,_,distance=skin.find_nearest(Vector(p))
            if hit is None:raise RuntimeError('Current actual head lookup failed during curve path probe')
            values[i]=float((Vector(p)-hit).dot(n.normalized()))
        return values
    signed=clearance(mids);free=arcs>ROOT_ZONE;bad=np.flatnonzero(free&(signed<LIMIT));worst=int(np.argmin(np.where(free,signed,np.inf)))
    arrays.update(central_midpoint_world=mids,central_midpoint_curve=owners,central_midpoint_segment=segments,central_midpoint_arc=arcs,central_midpoint_signed_clearance=signed)
    midpoint_report={'samples':len(mids),'freeSamples':int(free.sum()),'rootZoneSamples':int((~free).sum()),'freeMinimumMeters':float(signed[free].min()),
        'rootZoneMinimumMeters':float(signed[~free].min()) if np.any(~free) else None,'freeSamplesBelowLimit':len(bad),
        'worstFree':{'curve':int(owners[worst]),'segment':int(segments[worst]),'arcMeters':float(arcs[worst]),'world':mids[worst].tolist(),'signedMeters':float(signed[worst])}}
    located=np.flatnonzero((owners==14267)&(segments==11))
    midpoint_report['previouslyFlagged14267Segment11']=[{'signedMeters':float(signed[i]),'arcMeters':float(arcs[i]),'freeSegment':bool(free[i]),'world':mids[i].tolist()} for i in located]
    if len(bad):errors.append(str(len(bad))+' actual central free-segment midpoints exceed 0.5 mm skin intrusion')
    obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);original=modifier.node_group
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'));replacements=[]
    try:
        tree=review['copy_render_branch'](original,{},replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        source=next(s for s in tree.nodes['R11 strand tip taper'].outputs if s.type=='GEOMETRY')
        attr=tree.nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='INT';attr.inputs['Name'].default_value=GROUP
        role=tree.nodes.new('ShaderNodeMath');role.operation='MODULO';role.inputs[1].default_value=256.;tree.links.new(attr.outputs['Attribute'],role.inputs[0])
        comparisons=[]
        for value in (18,19):
            node=tree.nodes.new('ShaderNodeMath');node.operation='COMPARE';node.inputs[1].default_value=value;node.inputs[2].default_value=.1;tree.links.new(role.outputs[0],node.inputs[0]);comparisons.append(node.outputs[0])
        selected_field=tree.nodes.new('FunctionNodeBooleanMath');selected_field.operation='OR'
        for i,socket in enumerate(comparisons):tree.links.new(socket,selected_field.inputs[i])
        resample=tree.nodes.new('GeometryNodeResampleCurve')
        # Blender 5.2.3 exposes the measured menu input, not the old .mode RNA.
        mode=resample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Measured Blender 5.2 Resample Mode menu is missing')
        mode.default_value='Count'
        if mode.default_value!='Count':raise RuntimeError('Resample Count menu selection did not persist')
        resample.inputs['Count'].default_value=COUNT
        if resample.inputs['Count'].default_value!=COUNT:raise RuntimeError('Requested native path sample count did not persist')
        tree.links.new(source,resample.inputs['Curve']);tree.links.new(selected_field.outputs[0],resample.inputs['Selection'])
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(resample.outputs['Curve'],output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.data
        values=np.empty(len(data.position_data)*3,np.float32);data.position_data.foreach_get('vector',values);local=values.reshape(-1,3)
        matrix=np.asarray(evaluated.matrix_world);world=local@matrix[:3,:3].T+matrix[:3,3]
        native_starts=np.array([c.first_point_index for c in data.curves],np.int32);native_lengths=np.array([c.points_length for c in data.curves],np.int32)
        native_groups=np.empty(len(data.attributes[GROUP].data),np.int32);data.attributes[GROUP].data.foreach_get('value',native_groups)
        if not np.array_equal(native_groups,groups) or len(native_starts)!=len(starts):raise RuntimeError('Native read-only resampling changed ordered curve identities')
        paths=[];native_owners=[];native_arc=[]
        for i in selected:
            if native_lengths[i]!=COUNT:raise RuntimeError('Native selected curve did not receive requested sample count')
            path=world[native_starts[i]:native_starts[i]+native_lengths[i]]
            if float(np.linalg.norm(path[0]-q[starts[i]]))>2e-6:raise RuntimeError('Native sampled path root differs from actual dense measurement')
            paths.extend(path);native_owners.extend([int(i)]*len(path));native_arc.extend(np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))].tolist())
        paths=np.asarray(paths,np.float32);native_owners=np.asarray(native_owners,np.int32);native_arc=np.asarray(native_arc,np.float32)
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    native_signed=clearance(paths);native_free=native_arc>ROOT_ZONE;native_bad=np.flatnonzero(native_free&(native_signed<LIMIT));native_worst=int(np.argmin(np.where(native_free,native_signed,np.inf)))
    arrays.update(central_native_path_world=paths,central_native_path_curve=native_owners,central_native_path_arc=native_arc,central_native_path_signed_clearance=native_signed)
    if len(native_bad):errors.append(str(len(native_bad))+' native central curve free-path samples exceed 0.5 mm skin intrusion')
    native_report={'method':'Actual GeometryNodeResampleCurve on a temporary render-branch copy; no saved source edits','samplesPerSelectedCurve':COUNT,'curves':len(selected),'samples':len(paths),
        'freeSamples':int(native_free.sum()),'freeMinimumMeters':float(native_signed[native_free].min()),'freeSamplesBelowLimit':len(native_bad),
        'rootZoneSamples':int((~native_free).sum()),'rootZoneMinimumMeters':float(native_signed[~native_free].min()),
        'worstFree':{'curve':int(native_owners[native_worst]),'sampleWithinCurve':int(native_worst%COUNT),'arcMeters':float(native_arc[native_worst]),'world':paths[native_worst].tolist(),'signedMeters':float(native_signed[native_worst])}}
    return {'errors':errors,'targetRoles':[18,19],'rootAttachmentZoneArcMeters':ROOT_ZONE,'freeShaftMinimumAllowedSignedMeters':LIMIT,
        'currentHead':head,'actualFreeSegmentMidpoints':midpoint_report,'actualNativeCurvedPaths':native_report,
        'rootZoneReportedSeparatelyNotUsedToHideFreeShaftErrors':True,'scope':'Current-source neutral head BVH, every central control-segment midpoint and 128 native Blender samples per central curve.'}
