"""Install a calibrated crown-start field on an actually saved tie source.

Independent candidate only. The complete measured tie carrier remains editable;
the crown field is downstream of it. Current native paths and head/eye surfaces
are checked before saving. Original sources and all frozen predecessors remain.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
MAIN='R11 Donor / long hair main';CARRIER='R11 reference tie gather target positions'
PREDECESSOR='R11 reference gather tie transport';STAGE='R11 reference crown root flow'
GROUP='R11 groom interpolation group';TAG='R11 crown flow affected curve'
PINNED={
    'reference_crown_flow_nodes_v2_r11.py':'639b68ca9a80a74c636622183b28c391a4d9b673ec723182cdb6e5c2897b61b8',
    'reference_crown_flow_proof_v2_r11.py':'a6a5073a21b8468c9ef21b23ed3dc0c9605054de21b05dc943d9fed2642e5e78',
    'observe_reference_tie_alignment_r11_v2.py':'c488fccec705d4a805f142dab3c2fa8fe902bf42d3a59fd6307d04f248cf7c5c',
    'prepare_reference_tie_reconnection_v2_r11.py':'ef079aaa840043037181d6a55561394591bc14934271d9e97c40564162b9925d',
    'adult_full_groom_review_r11.py':'699d0a96cd332682e9bc9ddf8e6d3d731053e19889196f7f92a65c569fa7f6ae',
    'observe_donor_gn_stages_r11.py':'3c5da0c03dbbc9beb7c44884d271115c793a75fb5ca366ae2d7d8d1457f7e8cb',
    'prepare_adult_tail_shape_r11.py':'bf0be79636d01488f975c626edddca49f5f64c4d0909b0027f3e7a18feda095d',
    'reference_tie_rigid_objects_v2_r11.py':'6b798962e88d9bbf9d0917029b24349c7ae7eb7b447c76ef2f7e65dc4a145231',
    'reference_tie_surface_proof_r11.py':'aefcdf7328ccde22a59d70023f53852824c5cc3336676d6b4765d6ca455bebd3',
}
# Released after actual-source plan correspondence and frozen dependencies
# passed the offline preflight. Blender still validates actual native output.
RELEASED=True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def scoped(relative):
    p=(P/str(relative).replace('\\','/')).resolve();p.relative_to((P/'work/reconstruction-r02').resolve());return p


def curve_tag(obj):
    data=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data;attr=data.attributes.get(TAG)
    if attr is None or attr.data_type!='BOOLEAN' or attr.domain!='CURVE':raise RuntimeError('Actual crown tag must be CURVE Boolean')
    result=np.empty(len(attr.data),np.bool_);attr.data.foreach_get('value',result);return result


def native_selected(arrays,ids,capture_api):
    """Actual native interpolation of only the explicitly selected curves."""
    obj=bpy.data.objects[MAIN];mod=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    old=mod.node_group;replacements=[]
    try:
        tree=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))['copy_render_branch'](old,{},replacements);mod.node_group=tree
        source=next(s for s in tree.nodes['R11 strand tip taper'].outputs if s.type=='GEOMETRY')
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        tag=tree.nodes.new('GeometryNodeInputNamedAttribute');tag.data_type='BOOLEAN';tag.inputs['Name'].default_value=TAG
        sample=tree.nodes.new('GeometryNodeResampleCurve');mode=sample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Measured Resample Mode menu missing')
        mode.default_value='Count';sample.inputs['Count'].default_value=128
        if mode.default_value!='Count' or sample.inputs['Count'].default_value!=128:raise RuntimeError('Native sample settings did not persist')
        tree.links.new(source,sample.inputs['Curve']);tree.links.new(tag.outputs['Attribute'],sample.inputs['Selection'])
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(sample.outputs['Curve'],output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        temporary={};row=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture'](obj,'actual',temporary)
        groups=capture_api['integer_attribute'](obj,GROUP);lengths=temporary['actual_lengths'];starts=temporary['actual_starts']
        found=np.flatnonzero(lengths==128)
        if not np.array_equal(found,ids) or not np.array_equal(groups,arrays['after-final_group']):raise RuntimeError('Native curve selection/order differs from the measured crown identities')
        flat=np.concatenate([np.arange(starts[i],starts[i]+lengths[i]) for i in ids])
        for suffix in ('world','radius'):arrays['native-final_'+suffix]=temporary['actual_'+suffix][flat]
        arrays['native-final_starts']=np.arange(len(ids),dtype=np.int32)*128
        arrays['native-final_lengths']=np.full(len(ids),128,np.int32)
        arrays['native-final_group']=groups[ids];arrays['native-final_original_curve_ids']=np.asarray(ids,np.int32)
        error=float(np.linalg.norm(arrays['native-final_world'][::128]-arrays['after-final_world'][arrays['after-final_starts'][ids]],axis=1).max())
        if error>2e-6:raise RuntimeError('Native crown roots differ from actual final roots')
        return {'curves':len(ids),'samplesPerCurve':128,'actualPoints':len(flat),'maximumRootDifferenceMeters':error,'temporaryOnly':True,
                'radiusPolicy':'Actual native interpolated point radii in world metres','sourceStage':'R11 strand tip taper'}
    finally:
        mod.node_group=old;obj.update_tag();bpy.context.view_layer.update()


def contract_changes(before,after,*,carrier=False):
    rows={}
    if set(before)!=set(after):raise RuntimeError('Protected geometry object membership changed')
    for name,old in before.items():
        change={key:{'before':old[key],'after':after[name][key]} for key in old if old[key]!=after[name][key] and not (carrier and name==CARRIER and key=='data')}
        if change:rows[name]=change
    return rows


def calibration_head_dependency(observer,targets,arrays):
    """Protect the actual calibration surface while allowing distant edits."""
    row=observer['collect_mesh'](bpy.data.objects['R11 Female head study'],'crown-source-head',arrays,require_actual_evaluation=True)
    source=np.asarray(targets['source_head_world'],float);current=arrays['crown-source-head_world']
    source_tri=np.asarray(targets['source_head_triangles'],np.int32);current_tri=arrays['crown-source-head_triangles']
    if source.shape!=current.shape or source_tri.shape!=current_tri.shape:raise RuntimeError('Actual head correspondence differs from the calibrated source')
    ids=np.asarray(targets['selected_curve_ids'],np.int32);starts=np.asarray(targets['starts'],np.int32)
    selected=np.asarray(targets['target_point_mask'],bool)
    support=np.concatenate((targets['source_post_tie_world'][selected],targets['target_world'][selected],targets['source_post_tie_world'][starts[ids]]))
    margin=.04;lo=support.min(0)-margin;hi=support.max(0)+margin
    def overlaps(points,tri):
        corners=points[tri]
        return np.all(corners.max(1)>=lo,axis=1)&np.all(corners.min(1)<=hi,axis=1)
    region=overlaps(source,source_tri)|overlaps(current,current_tri)
    if not np.any(region):raise RuntimeError('Calibrated crown surface dependency region is empty')
    if not np.array_equal(source_tri[region],current_tri[region]):raise RuntimeError('Actual crown-support head triangle topology changed')
    vertices=np.unique(source_tri[region]);difference=np.linalg.norm(current-source,axis=1)
    maximum=float(difference[vertices].max())
    if maximum>2e-6:raise RuntimeError('Current crown-support head surface differs from calibration by more than 2 micrometres')
    arrays['crown-source-head_guarded_vertex_ids']=vertices.astype(np.int32)
    arrays['crown-source-head_guarded_triangle_ids']=np.flatnonzero(region).astype(np.int32)
    return {'actualHead':row,'guardedVertices':len(vertices),'guardedTriangles':int(region.sum()),
            'calibrationSupportBoundsWorld':[lo.tolist(),hi.tolist()],'calibrationSupportMarginMeters':margin,
            'maximumGuardedWorldResidualMeters':maximum,'maximumWholeHeadWorldResidualMeters':float(difference.max()),
            'guardedTrianglesExact':True,'sourceHeadPositionsSha256':observer['array_sha'](np.asarray(targets['source_head_world'])),
            'policy':'Same vertex/triangle correspondence and exact actual geometry within a conservative 40 mm support region around every changed control and selected root. Distant facial edits are recorded without locking an obsolete whole-head shape. Current complete head collisions are still re-evaluated after installation.'}


def main():
    if not RELEASED:raise RuntimeError('Unreleased crown wrapper draft: use only the final frozen released entrypoint')
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','plan','output'):parser.add_argument('--'+flag,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=scoped(args.source);plan_path=scoped(args.plan);out=scoped(args.output)
    report_path=out.with_suffix('.crown-flow.json');array_path=out.with_suffix('.crown-flow.npz')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,report_path,array_path)):
        raise RuntimeError('Fresh explicit source and independent unused outputs required')
    for name,digest in PINNED.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen crown dependency changed: '+name)
    plan=json.loads(plan_path.read_text(encoding='utf8'));observation_path=scoped(plan['observation']);observation=json.loads(observation_path.read_text(encoding='utf8'))
    if plan.get('status')!='actual-source-proposal' or plan.get('errors'):raise RuntimeError('An error-free calibrated actual-source proposal is required')
    if observation.get('status')!='collected' or sha(observation_path)!=plan['observationSha256'] or sha(scoped(observation['arrays']))!=plan['observationArraysSha256']:
        raise RuntimeError('Actual post-tie observation chain differs from calibrated proposal')
    target_path=scoped(plan['arrays'])
    if sha(target_path)!=plan['arraysSha256']:raise RuntimeError('Calibrated crown target arrays changed')
    targets=np.load(target_path,allow_pickle=False);observed=np.load(scoped(observation['arrays']),allow_pickle=False)
    observer=runpy.run_path(str(HERE/'observe_reference_tie_alignment_r11_v2.py'))
    capture_api=runpy.run_path(str(HERE/'prepare_reference_tie_reconnection_v2_r11.py'))
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    attrs=runpy.run_path(str(HERE/'reference_tie_rigid_objects_v2_r11.py'))['raw_attribute_contract']
    objects={o.name:o for o in bpy.data.objects if o.type in {'MESH','CURVES'}}
    donor=objects[MAIN];carrier=objects[CARRIER];rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    bpy.context.view_layer.update();pose=rig.data.pose_position;frame=(scene.frame_current,scene.frame_subframe);action=rig.animation_data.action if rig.animation_data else None
    before_source={name:contract(o) for name,o in objects.items()}
    before_attributes={name:attrs(o.data) for name,o in objects.items()}
    before_modifiers={o.name:observer['modifier_state'](o) for o in bpy.data.objects}
    before_visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
    arrays={};report={'source':args.source,'sourceSha256':sha(source),'plan':args.plan,'planSha256':sha(plan_path),
                     'wrapperSha256':sha(__file__),'dependencies':PINNED,'sourcePose':pose,'savedBlend':False,'technicalAcceptance':False,'appearanceAccepted':False,'dynamicsValid':False}
    error=None
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();before_rest={name:contract(o) for name,o in objects.items()}
        report['beforeStages']=capture_api['capture_stages'](arrays,(('before-gather',PREDECESSOR),('before-final','R11 strand tip taper')))
        for stage,observed_stage in (('before-gather','post-tie'),('before-final','final')):
            for key in ('starts','lengths','group','radius'):
                if not np.array_equal(arrays[stage+'_'+key],observed[observed_stage+'_'+key]):raise RuntimeError('Current actual crown dependency differs from observation: '+stage+' '+key)
            if float(np.linalg.norm(arrays[stage+'_world']-observed[observed_stage+'_world'],axis=1).max())>2e-6:raise RuntimeError('Current actual crown coordinates differ from observation')
        report['calibratedHeadDependency']=calibration_head_dependency(observer,targets,arrays)
        report['installation']=runpy.run_path(str(HERE/'reference_crown_flow_nodes_v2_r11.py'))['install'](targets,arrays,plan_sha256=sha(plan_path))
        mod=next(m for m in donor.modifiers if m.type=='NODES' and m.show_render);node=mod.node_group.nodes[STAGE]
        node.inputs['Strength'].default_value=0.;mod.node_group.update_tag();donor.update_tag();bpy.context.view_layer.update()
        capture_api['capture_stages'](arrays,(('zero-final','R11 strand tip taper'),))
        zero={key:bool(np.array_equal(arrays['zero-final_'+key],arrays['before-final_'+key])) for key in ('world','starts','lengths','group','radius')}
        report['zeroStrengthExactBypass']=zero
        if not all(zero.values()):raise RuntimeError('Crown strength zero changed original final geometry')
        node.inputs['Strength'].default_value=1.;mod.node_group.update_tag();donor.update_tag();bpy.context.view_layer.update()
        report['afterStages']=capture_api['capture_stages'](arrays,(('after-gather',PREDECESSOR),('after-flow',STAGE),('after-final','R11 strand tip taper')))
        arrays['after-flow_selected_curve_tag']=curve_tag(donor)
        proof=runpy.run_path(str(HERE/'reference_crown_flow_proof_v2_r11.py'))
        report['actualDensePreservation']=proof['compare'](arrays,targets)
        report['allActualFinalRadiiExact']=bool(np.array_equal(arrays['after-final_radius'],arrays['before-final_radius']))
        if not report['allActualFinalRadiiExact']:report['actualDensePreservation']['errors'].append('Crown-only flow change altered original measured final radii')
        ids=np.asarray(targets['selected_curve_ids'],np.int32);report['actualNativePaths']=native_selected(arrays,ids,capture_api)
        report['currentSurfaces']=proof['surfaces'](arrays,ids,observer,runpy.run_path(str(HERE/'reference_tie_surface_proof_r11.py')))
        errors=list(report['actualDensePreservation']['errors'])+list(report['currentSurfaces']['errors'])
        runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture'](donor,'viewport-final',arrays)
        arrays['viewport-final_group']=capture_api['integer_attribute'](donor,GROUP)
        parity={key:bool(np.array_equal(arrays['viewport-final_'+key],arrays['after-final_'+key])) for key in ('world','starts','lengths','group','radius')}
        report['viewportRenderParity']=parity
        if not all(parity.values()):errors.append('Crown viewport and full-render outputs differ')
        report['protectedSameRestChanges']=contract_changes(before_rest,{name:contract(o) for name,o in objects.items()},carrier=True)
        if report['protectedSameRestChanges']:errors.append('Protected raw geometry/keys/material/parent changed in REST')
        added=set(report['installation']['newAttributes']);attribute_changes={}
        for name,o in objects.items():
            current=attrs(o.data)
            if name==CARRIER:current=[row for row in current if row['name'] not in added]
            if current!=before_attributes[name]:attribute_changes[name]={'before':before_attributes[name],'afterWithoutDeclaredNewAttributes':current}
        report['protectedAttributeChanges']=attribute_changes
        if attribute_changes:errors.append('Existing raw non-position attributes changed')
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        report['protectedRestoredPoseChanges']=contract_changes(before_source,{name:contract(o) for name,o in objects.items()},carrier=True)
        if report['protectedRestoredPoseChanges']:errors.append('Protected raw geometry contract changed after source pose restoration')
        current_modifiers={o.name:observer['modifier_state'](o) for o in bpy.data.objects}
        expected=json.loads(json.dumps(before_modifiers));active=[r for r in expected[MAIN]['modifiers'] if r['type']=='NODES' and r['showRender']]
        if len(active)!=1:raise RuntimeError('Expected one main styling modifier')
        active[0]['nodeGroup']=mod.node_group.name
        report['modifierContractExactExceptDeclaredMainTree']=current_modifiers==expected
        if current_modifiers!=expected:errors.append('Unexpected modifier metadata changed')
        visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
        if visibility!=before_visibility:errors.append('Object membership/visibility changed')
        if sha(source)!=report['sourceSha256'] or (scene.frame_current,scene.frame_subframe)!=frame or (rig.animation_data.action if rig.animation_data else None)!=action:
            errors.append('Source file, frame or action changed')
        report['errors']=errors;report['technicalAcceptance']=not errors
        if errors:raise RuntimeError('; '.join(errors))
    except Exception as exc:
        error=exc;report['error']=str(exc)
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update();targets.close();observed.close()
    if error is None:
        scene['R11_reference_crown_start_flow']=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Independent crown candidate save failed')
        report.update(savedBlend=True,output=args.output,outputSha256=sha(out))
    if arrays:
        with array_path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        report.update(arrays=str(array_path.relative_to(P)),arraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_CROWN_FLOW_RESULT '+json.dumps({'savedBlend':report['savedBlend'],'errors':report.get('errors',[]),'error':report.get('error')}),flush=True)
    if error:raise error


if __name__=='__main__':main()
