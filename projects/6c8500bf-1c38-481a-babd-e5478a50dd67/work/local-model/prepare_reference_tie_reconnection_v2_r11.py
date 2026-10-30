"""Apply measured tail transport and true-gather reconnection on a current source.

The input blend may contain newer eye/lip/hand work. Only actual ear, main raw
groom, tail and ring dependencies are compared to the observation. Every
collision surface is recaptured from the loaded source before an independent
candidate can be saved. Old sources, helpers, caches and node trees remain.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path(); HERE = Path(__file__).resolve().parent
MAIN = 'R11 Donor / long hair main'; GROUP = 'R11 groom interpolation group'
STAGE = 'R11 reference gather tie transport'
CARRIER = 'R11 reference tie gather target positions'
SEMANTIC_REVIEW = 'work/reconstruction-r02/reference-role41-facelock-semantic-review-r11.json'
SEMANTIC_SHA = '2efc236d84d5e32b67c9e20082805535aae96bae31cfe69fca10bb1c75f30683'
PINNED = {
    'observe_reference_tie_alignment_r11_v2.py': 'c488fccec705d4a805f142dab3c2fa8fe902bf42d3a59fd6307d04f248cf7c5c',
    'reference_tie_rigid_objects_v2_r11.py': '6b798962e88d9bbf9d0917029b24349c7ae7eb7b447c76ef2f7e65dc4a145231',
    'reference_gather_reconnection_nodes_v2_r11.py': '1b91efbf60c26b9ff27b5f0496bec85960993ed04adb891e244d34657f17686c',
    'reference_gather_reconnection_proof_v2_r11.py': '90ecd33ee7d1e053d78b6321a946cb0ef41a1a26081e76ef3b1c7f650708d716',
    'reference_tie_surface_proof_r11.py': 'aefcdf7328ccde22a59d70023f53852824c5cc3336676d6b4765d6ca455bebd3',
    'adult_full_groom_review_r11.py': '699d0a96cd332682e9bc9ddf8e6d3d731053e19889196f7f92a65c569fa7f6ae',
    'observe_donor_gn_stages_r11.py': '3c5da0c03dbbc9beb7c44884d271115c793a75fb5ca366ae2d7d8d1457f7e8cb',
    'prepare_adult_tail_shape_r11.py': 'bf0be79636d01488f975c626edddca49f5f64c4d0909b0027f3e7a18feda095d',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def path(relative):
    p=(P/str(relative).replace('\\','/')).resolve()
    p.relative_to((P/'work/reconstruction-r02').resolve())
    return p


def integer_attribute(obj, name):
    data=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    attr=data.attributes.get(name)
    if not attr or attr.data_type!='INT' or attr.domain!='CURVE':
        raise RuntimeError('Actual ordered curve identity attribute missing')
    values=np.empty(len(attr.data),np.int32);attr.data.foreach_get('value',values)
    return values


def capture_stages(arrays, stages):
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    capture=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture']
    obj=bpy.data.objects[MAIN];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    original=modifier.node_group;replacements=[];rows=[]
    try:
        tree=review['copy_render_branch'](original,{},replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        for prefix,name in stages:
            source=next(s for s in tree.nodes[name].outputs if s.type=='GEOMETRY')
            for link in list(output.links):tree.links.remove(link)
            tree.links.new(source,output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
            rows.append(capture(obj,prefix,arrays));arrays[prefix+'_group']=integer_attribute(obj,GROUP)
            data=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
            tag=data.attributes.get('R11 verified temple curtain')
            if tag is not None:
                if tag.domain!='POINT' or tag.data_type!='BOOLEAN':raise RuntimeError('Verified temple-curtain tag must be actual POINT Boolean')
                values=np.empty(len(tag.data),np.bool_);tag.data.foreach_get('value',values);arrays[prefix+'_verified_temple_curtain']=values
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    return {'stages':rows,'renderBranchReplacements':replacements}


def capture_native_gather(arrays):
    """Actual native path at 64 samples per gather, with no saved resample."""
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    obj=bpy.data.objects[MAIN];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    original=modifier.node_group;replacements=[];count=64
    try:
        tree=review['copy_render_branch'](original,{},replacements);modifier.node_group=tree
        source=next(s for s in tree.nodes['R11 strand tip taper'].outputs if s.type=='GEOMETRY')
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        attr=tree.nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='INT';attr.inputs['Name'].default_value=GROUP
        role=tree.nodes.new('ShaderNodeMath');role.operation='MODULO';role.inputs[1].default_value=256.;tree.links.new(attr.outputs['Attribute'],role.inputs[0])
        select=tree.nodes.new('FunctionNodeBooleanMath');select.operation='OR'
        for index,value in enumerate((40,41)):
            compare=tree.nodes.new('ShaderNodeMath');compare.operation='COMPARE';compare.inputs[1].default_value=value;compare.inputs[2].default_value=.1
            tree.links.new(role.outputs[0],compare.inputs[0]);tree.links.new(compare.outputs[0],select.inputs[index])
        sample=tree.nodes.new('GeometryNodeResampleCurve');mode=sample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Measured Blender 5.2 Resample menu missing')
        mode.default_value='Count';sample.inputs['Count'].default_value=count
        if mode.default_value!='Count' or sample.inputs['Count'].default_value!=count:raise RuntimeError('Native resample settings not retained')
        tree.links.new(source,sample.inputs['Curve']);tree.links.new(select.outputs[0],sample.inputs['Selection'])
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(sample.outputs['Curve'],output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.data
        local=np.empty(len(data.position_data)*3,np.float32);data.position_data.foreach_get('vector',local);local=local.reshape(-1,3)
        matrix=np.asarray(evaluated.matrix_world,np.float64);world=local@matrix[:3,:3].T+matrix[:3,3]
        starts=np.asarray([c.first_point_index for c in data.curves],np.int32)
        lengths=np.asarray([c.points_length for c in data.curves],np.int32)
        groups=integer_attribute(obj,GROUP);before=arrays['after-final_group']
        if not np.array_equal(groups,before):raise RuntimeError('Native sampling changed curve identities/order')
        ids=np.flatnonzero(np.isin(groups%256,(40,41)))
        if np.any(lengths[ids]!=count):raise RuntimeError('Native gather sampling count mismatch')
        q=np.concatenate([world[starts[i]:starts[i]+count] for i in ids])
        reference=arrays['after-final_world'][arrays['after-final_starts'][ids]]
        if np.linalg.norm(q[::count]-reference,axis=1).max()>2e-6:raise RuntimeError('Native gather roots differ from final measured roots')
        arrays['gather_native_after_world']=q.astype(np.float32)
        arrays['gather_native_after_starts']=np.arange(len(ids),dtype=np.int32)*count
        arrays['gather_native_after_lengths']=np.full(len(ids),count,np.int32)
        arrays['gather_native_after_group']=groups[ids]
        arrays['gather_native_after_original_curve']=ids.astype(np.int32)
        radius=arrays['after-final_radius'];base_start=arrays['after-final_starts'];base_len=arrays['after-final_lengths']
        maximum=np.asarray([radius[base_start[i]:base_start[i]+base_len[i]].max() for i in ids],np.float32)
        arrays['gather_native_after_radius_world']=np.repeat(maximum,count)
        return {'curves':len(ids),'samplesPerCurve':count,'samples':len(q),'nativeNodeMode':'Count',
                'temporaryOnly':True,'radiusPolicy':'Actual per-curve maximum final radius as conservative path bound'}
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()


def current_dependencies(observer,old,arrays):
    current={};head=bpy.data.objects['R11 Female head study']
    observer['collect_mesh'](head,'head',current,require_face_sets=True,require_actual_evaluation=True)
    observer['collect_ear'](current,4,'ear_positive');observer['collect_ear'](current,5,'ear_negative')
    rows=[]
    for prefix in ('ear_positive','ear_negative'):
        for suffix in ('head_vertex_ids','triangles'):
            if not np.array_equal(current[prefix+'_'+suffix],old[prefix+'_'+suffix]):raise RuntimeError('Observed ear topology dependency changed')
        delta=float(np.linalg.norm(current[prefix+'_world']-old[prefix+'_world'],axis=1).max())
        if delta>2e-6:raise RuntimeError('Observed actual ear dependency moved: '+prefix)
        rows.append({'dependency':prefix,'maximumWorldDifferenceMeters':delta})
    for prefix,name in observer['TAILS']:
        observer['collect_curves'](bpy.data.objects[name],prefix,arrays,expected_curves=2480,expected_points_per_curve=35)
        for suffix in ('local','starts','lengths','radius'):
            if not np.array_equal(arrays[prefix+'_'+suffix],old[prefix+'_'+suffix]):raise RuntimeError('Observed tail dependency changed: '+prefix+' '+suffix)
        delta=float(np.linalg.norm(arrays[prefix+'_world']-old[prefix+'_world'],axis=1).max())
        if delta>2e-6:raise RuntimeError('Observed tail rest-world dependency changed')
        rows.append({'dependency':prefix,'maximumWorldDifferenceMeters':delta,'rawPositionsTopologyRadiiExact':True})
    for prefix,name in observer['TIES']:
        observer['collect_mesh'](bpy.data.objects[name],prefix+'_raw',arrays,evaluated=False)
        for suffix in ('local','triangles'):
            if not np.array_equal(arrays[prefix+'_raw_'+suffix],old[prefix+'_raw_'+suffix]):raise RuntimeError('Observed raw ring dependency changed')
        delta=float(np.linalg.norm(arrays[prefix+'_raw_world']-old[prefix+'_raw_world'],axis=1).max())
        if delta>2e-6:raise RuntimeError('Observed ring rest-world dependency changed')
        rows.append({'dependency':prefix,'maximumWorldDifferenceMeters':delta})
    observer['collect_curves'](bpy.data.objects[MAIN],'main',arrays,expected_curves=346,require_roles=True)
    for suffix in ('local','starts','lengths','role','matrix'):
        if not np.array_equal(arrays['main_'+suffix],old['main_'+suffix]):raise RuntimeError('Observed v10 raw main dependency changed: '+suffix)
    # Complete old head is deliberately not locked; newer facial work stays.
    return {'earTailRingRawMainDependencies':rows,'rawMainExact':True,
            'wholeOldHeadPositionsLocked':False,'currentHeadWorldSha256':observer['array_sha'](current['head_world'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','plan','gather-plan','output'):parser.add_argument('--'+flag,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    source=path(args.source);plan_path=path(args.plan);gather_plan_path=path(args.gather_plan);out=path(args.output)
    report_path=out.with_suffix('.tie-transport.json');array_path=out.with_suffix('.tie-transport.npz')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(q.exists() for q in (out,report_path,array_path)):
        raise RuntimeError('Fresh independent loaded source and unused outputs required')
    for name,digest in PINNED.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen transport dependency changed: '+name)
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    if plan.get('schemaVersion')!=1 or plan.get('status')!='proposed' or set(plan.get('transforms',{}))!={'positive','negative'}:
        raise RuntimeError('An explicit proposed two-sided rigid plan is required')
    observation_path=path(plan['observation']);observation=json.loads(observation_path.read_text(encoding='utf8'))
    observed_array_path=path(observation['arrays'])
    if sha(observation_path)!=plan['observationSha256'] or sha(observed_array_path)!=plan['observationArraysSha256'] or observation['arraysSha256']!=plan['observationArraysSha256']:
        raise RuntimeError('Measured plan/observation hashes do not match')
    observed=np.load(observed_array_path,allow_pickle=False)
    observer=runpy.run_path(str(HERE/'observe_reference_tie_alignment_r11_v2.py'))
    for key,metadata in observation['arrayManifest'].items():
        if observer['array_sha'](observed[key])!=metadata['sha256']:raise RuntimeError('Observation array manifest mismatch: '+key)
    gather_plan=json.loads(gather_plan_path.read_text(encoding='utf8'))
    if gather_plan.get('status')!='proposed' or gather_plan.get('tiePlanSha256')!=sha(plan_path):raise RuntimeError('Measured gather plan is not a proposal for this exact rigid tie plan')
    gather_array_path=path(gather_plan['arrays'])
    if sha(gather_array_path)!=gather_plan['arraysSha256']:raise RuntimeError('Measured gather target arrays hash changed')
    targets=np.load(gather_array_path,allow_pickle=False)
    semantic_path=path(SEMANTIC_REVIEW)
    if sha(semantic_path)!=SEMANTIC_SHA:raise RuntimeError('Verified temple-curtain classification evidence changed')
    semantic=json.loads(semantic_path.read_text(encoding='utf8'))
    if not np.array_equal(np.asarray(targets['preserved_face_curtain_curve_ids'],np.int32),np.asarray(semantic['suspectDenseCurveIds'],np.int32)):
        raise RuntimeError('All 46 actual guide-13 face-curtain curves must be protected by identity')
    core=runpy.run_path(str(HERE/'reference_tie_rigid_objects_v2_r11.py'))
    for value in plan['transforms'].values():core['validate_transform'](value)
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    changed={name for side in core['TAILS'] for name in [core['TAILS'][side]]+core['TIES'][side]}
    protected=[o for o in bpy.data.objects if o.type in {'MESH','CURVES'} and o.name not in changed]
    bpy.context.view_layer.update();source_contract={o.name:contract(o) for o in protected}
    changed_contract={name:observer['raw_contract'](bpy.data.objects[name]) for name in changed}
    modifiers={o.name:observer['modifier_state'](o) for o in bpy.data.objects}
    visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;scene=bpy.context.scene
    frame=(scene.frame_current,scene.frame_subframe);action=rig.animation_data.action if rig.animation_data else None
    arrays={};report={'source':args.source,'sourceSha256':sha(source),'plan':args.plan,'planSha256':sha(plan_path),
                    'observation':plan['observation'],'observationSourceSha256':observation['sourceSha256'],
                    'savedBlend':False,'technicalAcceptance':False,'appearanceAccepted':False,'dynamicsValid':False,
                    'wrapperSha256':sha(__file__),'dependencies':PINNED,'sourcePose':pose,
                    'gatherPlan':args.gather_plan,'gatherPlanSha256':sha(gather_plan_path),'gatherArraysSha256':gather_plan['arraysSha256']}
    report['verifiedFaceCurtainEvidence']={'path':SEMANTIC_REVIEW,'sha256':SEMANTIC_SHA,'curveIds':semantic['suspectDenseCurveIds']}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        before_rest={o.name:contract(o) for o in protected}
        print('R11_TIE_VALIDATE_CURRENT_DEPENDENCIES',flush=True)
        report['currentDependencies']=current_dependencies(observer,observed,arrays)
        report['beforeDense']=capture_stages(arrays,(('before-front','R11 reference local front detail'),('before-final','R11 strand tip taper')))
        print('R11_TIE_APPLY_RIGID_AND_GATHER',flush=True)
        report['installation']=core['apply'](plan['transforms'],observer,arrays)
        report['measuredGatherInstallation']=runpy.run_path(str(HERE/'reference_gather_reconnection_nodes_v2_r11.py'))['install'](targets,arrays,plan_sha256=sha(gather_plan_path))
        for prefix,name in observer['TAILS']:
            observer['collect_curves'](bpy.data.objects[name],prefix+'_after',arrays,expected_curves=2480,expected_points_per_curve=35)
        for prefix,name in observer['TIES']:
            observer['collect_mesh'](bpy.data.objects[name],prefix+'_after_eval',arrays,require_actual_evaluation=True)
        report['afterDense']=capture_stages(arrays,(('after-front','R11 reference local front detail'),('after-transport',STAGE),
                        ('after-contact','R11 full head clearance'),('after-final','R11 strand tip taper')))
        direct_capture=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture']
        report['savedTreeDirectViewportOutput']=direct_capture(bpy.data.objects[MAIN],'actual-viewport-final',arrays)
        arrays['actual-viewport-final_group']=integer_attribute(bpy.data.objects[MAIN],GROUP)
        parity={key:bool(np.array_equal(arrays['actual-viewport-final_'+key],arrays['after-final_'+key])) for key in ('world','starts','lengths','group','radius')}
        report['savedTreeViewportRenderParity']=parity
        if not all(parity.values()):raise RuntimeError('Saved tree direct viewport output differs from its measured complete render branch')
        report['actualDensePreservation']=runpy.run_path(str(HERE/'reference_gather_reconnection_proof_v2_r11.py'))['compare'](arrays,targets)
        print('R11_TIE_CAPTURE_NATIVE_GATHER_PATHS',flush=True)
        report['nativeGather']=capture_native_gather(arrays)
        print('R11_TIE_CURRENT_SURFACE_CHECKS',flush=True)
        report['currentSurfaces']=runpy.run_path(str(HERE/'reference_tie_surface_proof_r11.py'))['prove'](arrays,observer)
        errors=list(report['actualDensePreservation']['errors'])+list(report['currentSurfaces']['errors'])
        after_rest={o.name:contract(o) for o in protected}
        rest_changes={k:{'before':before_rest[k],'after':after_rest[k]} for k in before_rest if before_rest[k]!=after_rest[k]}
        report['protectedSameRestChanges']=rest_changes
        if rest_changes:errors.append('Protected raw geometry/keys/parent/material contract changed in REST')
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        after_source={o.name:contract(o) for o in protected}
        source_changes={k:{'before':source_contract[k],'after':after_source[k]} for k in source_contract if source_contract[k]!=after_source[k]}
        report['protectedRestoredPoseChanges']=source_changes
        if source_changes:errors.append('Protected raw contract changed after source pose restoration')
        unchanged_fields=('matrixLocal','materials','parent','parentBone','hideRender','hideViewport','hideCurrentViewLayer','modifiers')
        moved_meta={}
        for name in sorted(changed):
            actual=observer['raw_contract'](bpy.data.objects[name]);old=changed_contract[name]
            difference={key:{'before':old[key],'after':actual[key]} for key in unchanged_fields if old[key]!=actual[key]}
            if difference:moved_meta[name]=difference
        report['movedObjectsProtectedMetadataChanges']=moved_meta
        if moved_meta:errors.append('Moved tail/ring parent/transform/material/modifier contract changed')
        after_modifiers={o.name:observer['modifier_state'](o) for o in bpy.data.objects}
        if {k:v for k,v in modifiers.items() if k!=MAIN}!={k:v for k,v in after_modifiers.items() if k not in (MAIN,CARRIER)}:errors.append('Protected object modifier changed')
        after_visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
        if set(after_visibility)-set(visibility)!={CARRIER}:errors.append('Unexpected new object inventory')
        if visibility!={k:v for k,v in after_visibility.items() if k!=CARRIER}:errors.append('Existing object inventory or visibility changed')
        carrier=bpy.data.objects.get(CARRIER);donor=bpy.data.objects[MAIN]
        if carrier is None or carrier.type!='MESH' or not carrier.hide_render or carrier.parent!=donor.parent or carrier.parent_type!=donor.parent_type or carrier.parent_bone!=donor.parent_bone:
            errors.append('Measured target carrier parent/visibility contract invalid')
        elif np.max(np.abs(np.asarray(carrier.matrix_world)-np.asarray(donor.matrix_world)))>2e-6:
            errors.append('Measured target carrier does not follow the restored source pose with the donor')
        report['addedEditableCarrier']=CARRIER
        if (scene.frame_current,scene.frame_subframe)!=frame or (rig.animation_data.action if rig.animation_data else None)!=action:errors.append('Scene frame or body action changed')
        if sha(source)!=report['sourceSha256']:errors.append('Saved source was modified')
        report['errors']=errors
        print('R11_TIE_CHECKS_COMPLETE '+json.dumps({'errors':errors,'appearanceAccepted':False,'dynamicsValid':False}),flush=True)
        if errors:raise RuntimeError('; '.join(errors))
        report['technicalAcceptance']=True
        report['nextRequired']=['Same-camera front and quarter appearance review including posterior loops and tie-root continuity',
                                'Measure and independently correct any pre-existing loose ring fit',
                                'Rebuild maps/pins/dynamics and check actual continuous motion after static appearance adoption']
    except Exception as exc:
        report['error']=str(exc)
        if arrays:
            with array_path.open('xb') as stream:np.savez_compressed(stream,**arrays)
            report.update(arrays=str(array_path.relative_to(P)),arraysSha256=sha(array_path))
        with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update();observed.close();targets.close()
    scene['R11_reference_tie_reconnection_v2']=json.dumps(report,ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Independent tie candidate save failed')
    with array_path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report.update(savedBlend=True,output=args.output,outputSha256=sha(out),arrays=str(array_path.relative_to(P)),arraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_REFERENCE_TIE_RECONNECTION_V2_SAVED '+json.dumps({'output':args.output,'sha256':report['outputSha256'],'appearanceAccepted':False,'dynamicsValid':False}),flush=True)


if __name__=='__main__':main()
