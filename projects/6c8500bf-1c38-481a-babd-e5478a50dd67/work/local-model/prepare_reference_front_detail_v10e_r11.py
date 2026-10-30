"""Measure all before/after dense hair, then save the measured local front detail v10 with actual guide-influence and native path clearance."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent;MAIN='R11 Donor / long hair main'
PINS={
    'reference_front_preservation_v10_v2_r11.py':'b7ec130dc49929c627454755a3f5e0919bbab99cd1dfe0584233cb13aab493e9',
    'reference_front_path_probe_v10_v2_r11.py':'f5ad22fa311909ab16097294c3e01374f4d1ffed61ee18abb85cb18d241c2e6a',
    'reference_front_detail_v10_v2_r11.py':'7f8bbe1814914981c81608bd7a58fe6505242e78e1a66327b82bcfb484c27ce8',
    'reference_left_gather_v10_r11.py':'5340c545a0d1103db82ee7e8c559172a0d80ebfc18349a8a2d4da909579292ed',
    'reference_central_fan_v10_r11.py':'710bcda5d46ace79cb54302de571374381d5de4fd8e48f25cb0e11ebefb530df',
    'reference_front_preservation_v10_r11.py':'4004a40377243d7603e32b882fa8113e1d69021e472bd506308ca8bf54fef590',
    'reference_front_dense_proof_v10_r11.py':'c0259b6bd88cddfeb770562925d61ad10d2e4fc304e72bd1a7dc294bc952ff29',
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def raw_radius_contract(data):
    # Evaluated hair always receives a radius, but author guides may correctly
    # have no raw radius attribute. Preserve that absence without fabricating it.
    attr=data.attributes.get('radius')
    if attr is None:return {'exists':False}
    if attr.data_type!='FLOAT':raise RuntimeError('Unexpected existing raw radius attribute type')
    values=np.empty(len(attr.data),np.float32);attr.data.foreach_get('value',values)
    return {'exists':True,'dataType':attr.data_type,'domain':attr.domain,'elements':len(values),
            'valuesSha256':hashlib.sha256(values.tobytes()).hexdigest()}

def contract_difference(a,b):
    return {name:{field:{'before':a[name].get(field),'after':b[name].get(field)}
                  for field in set(a[name])|set(b[name]) if a[name].get(field)!=b[name].get(field)}
            for name in a if a[name]!=b[name]}

def modifier_contract(obj):
    return [(m.name,m.type,m.show_viewport,m.show_render,getattr(getattr(m,'node_group',None),'name',None)) for m in obj.modifiers]

def capture_before(arrays,proof_helpers):
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'));capture=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture']
    obj=bpy.data.objects[MAIN];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);original=modifier.node_group;replacements=[]
    try:
        tree=review['copy_render_branch'](original,{},replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(next(s for s in tree.nodes['R11 strand tip taper'].outputs if s.type=='GEOMETRY'),output)
        tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        result=capture(obj,'before-final',arrays);arrays['before-final_group']=proof_helpers['integer_attribute'](obj,'R11 groom interpolation group')
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(next(s for s in tree.nodes['Group.005'].outputs if s.type=='GEOMETRY'),output)
        tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        result['actualInterpolationBefore']=capture(obj,'before-interpolate',arrays)
        arrays['before-interpolate_group']=proof_helpers['integer_attribute'](obj,'R11 groom interpolation group')
        return result
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    report_path=out.with_suffix('.groom-front-detail.json');array_path=out.with_suffix('.groom-front-detail.npz')
    for path in (source,out,report_path,array_path):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,report_path,array_path)):raise RuntimeError('Fresh independent source and unused outputs required')
    for name,digest in PINS.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen helper changed: '+name)
    main_obj=bpy.data.objects[MAIN]
    if not main_obj.get('R11_reference_local_v9'):raise RuntimeError('Only a saved v9-derived groom may receive the local detail edit')
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    carriers=[o.name for o in bpy.data.objects if o.get('R11_groom_role_root_source')==MAIN]
    protected=[o for o in bpy.data.objects if o.type in {'MESH','CURVES'} and o.name!=MAIN]
    loaded_contracts={o.name:contract(o) for o in protected}
    bpy.context.view_layer.update()
    before={o.name:contract(o) for o in protected}
    modifiers={o.name:modifier_contract(o) for o in bpy.data.objects}
    visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));old_raw=fit['_coords'](main_obj)
    raw_radii=raw_radius_contract(main_obj.data)
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;scene=bpy.context.scene
    frame=(scene.frame_current,scene.frame_subframe);action=rig.animation_data.action if rig.animation_data else None
    arrays={};report={'source':a.source,'sourceSha256':sha(source),'savedBlend':False,'appearanceAccepted':False,'wrapperSha256':sha(__file__)}
    report['protectedContractsLoaded']=loaded_contracts
    report['protectedContractsSourcePoseBefore']=before
    report['sourcePose']=pose
    report['loadedToEvaluatedSourceContractChanges']=contract_difference(loaded_contracts,before)
    helpers=runpy.run_path(str(HERE/'reference_front_dense_proof_v10_r11.py'))
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        before_rest={o.name:contract(o) for o in protected}
        report['protectedContractsRestBefore']=before_rest
        report['sourceToRestContractChanges']=contract_difference(before,before_rest)
        print('R11_REFERENCE_V10_MEASURING_ACTUAL_SOURCE',flush=True)
        report['actualDenseBefore']=capture_before(arrays,helpers)
        print('R11_REFERENCE_V10_LOCAL_SHAPE',flush=True)
        raw_before=fit['_world'](fit['_coords'](main_obj),main_obj.matrix_world)
        raw_starts=np.array([c.first_point_index for c in main_obj.data.curves],np.int32)
        raw_lengths=np.array([c.points_length for c in main_obj.data.curves],np.int32)
        if any(raw_lengths[i]!=20 for i in (88,94)):raise RuntimeError('Measured two-guide basis topology changed')
        report.update(runpy.run_path(str(HERE/'reference_front_detail_v10_v2_r11.py'))['apply'](expected_source_relative=a.source))
        raw_after=fit['_world'](fit['_coords'](main_obj),main_obj.matrix_world)
        arrays['actual_raw_gather_displacement_basis']=np.stack([raw_after[raw_starts[i]:raw_starts[i]+20]-raw_before[raw_starts[i]:raw_starts[i]+20] for i in (88,94)])
        arrays['actual_raw_gather_before']=np.stack([raw_before[raw_starts[i]:raw_starts[i]+20] for i in (88,94)])
        arrays['actual_raw_gather_after']=np.stack([raw_after[raw_starts[i]:raw_starts[i]+20] for i in (88,94)])
        print('R11_REFERENCE_V10_MEASURING_ALL_ACTUAL_DENSE_OUTPUT',flush=True)
        proof=helpers['actual_dense_proof'](arrays);report['actualDenseOutput']=proof
        preservation=runpy.run_path(str(HERE/'reference_front_preservation_v10_v2_r11.py'))['compare'](arrays);report['allActualDenseBeforeAfter']=preservation
        print('R11_REFERENCE_V10_ACTUAL_FREE_PATH_CLEARANCE',flush=True)
        path_proof=runpy.run_path(str(HERE/'reference_front_path_probe_v10_v2_r11.py'))['probe'](arrays);report['actualCentralFreePathClearance']=path_proof
        report['fineWispsActualProof']=helpers['actual_wisp_proof'](arrays)
        errors=list(proof['errors'])+list(preservation['errors'])+list(path_proof['errors'])
        after_rest={o.name:contract(o) for o in protected}
        report['protectedContractsRestAfter']=after_rest
        rest_changes=contract_difference(before_rest,after_rest)
        report['protectedContractRestChanges']=rest_changes
        if rest_changes:errors.append('Protected geometry/keys/materials/parent contract changed in the same REST pose: '+json.dumps({name:list(fields) for name,fields in rest_changes.items()}))
        # Compare the original pose as well. A pose transition may legitimately
        # change a bone-parented matrix_local, but an edit must preserve both
        # sides of that transition, not hide it with a relaxed numeric tolerance.
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        after_source={o.name:contract(o) for o in protected}
        report['protectedContractsSourcePoseAfter']=after_source
        source_changes=contract_difference(before,after_source)
        report['protectedContractSourcePoseChanges']=source_changes
        if source_changes:errors.append('Protected geometry/keys/materials/parent contract changed after source pose restoration: '+json.dumps({name:list(fields) for name,fields in source_changes.items()}))
        after_modifiers={o.name:modifier_contract(o) for o in bpy.data.objects}
        if {k:v for k,v in modifiers.items() if k!=MAIN}!={k:v for k,v in after_modifiers.items() if k!=MAIN}:errors.append('A protected object modifier changed')
        if [v[:4] for v in modifiers[MAIN]]!=[v[:4] for v in after_modifiers[MAIN]]:errors.append('Main modifier count/type/visibility changed')
        report['mainModifierBefore']=modifiers[MAIN];report['mainModifierAfter']=after_modifiers[MAIN]
        if visibility!={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}:errors.append('Object visibility changed')
        new_radii=raw_radius_contract(main_obj.data)
        report['rawGuideRadiusAttributeBefore']=raw_radii;report['rawGuideRadiusAttributeAfter']=new_radii
        if raw_radii!=new_radii:errors.append('Raw guide radius attribute presence/type/domain/values changed')
        if (scene.frame_current,scene.frame_subframe)!=frame or (rig.animation_data.action if rig.animation_data else None)!=action:errors.append('Body action or scene frame changed')
        if sha(source)!=report['sourceSha256']:errors.append('Saved input changed')
        report['errors']=errors
        print('R11_REFERENCE_V10_ACTUAL_DENSE_CHECKED '+json.dumps({'curves':proof['stages'][-1]['curves'],'errors':errors}),flush=True)
        if errors:raise RuntimeError('; '.join(errors))
        report['protectedGeometryContracts']=before;report['protectedModifierContracts']=modifiers
        report['sourceVisibilityPreserved']=True;report['bodyActionAndFramePreserved']=True;report['rawGuideRadiiPreservedExact']=True
        report['faceFibreBinding']=runpy.run_path(str(HERE/'face_fibre_binding_r11.py'))['validate'](neutral=False)
    except Exception as exc:
        report['error']=str(exc)
        if arrays:
            with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
            report.update(actualDenseArrays=str(array_path.relative_to(P)),actualDenseArraysSha256=sha(array_path))
        with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    scene['R11_reference_front_groom_v10']=json.dumps(report,ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Candidate save failed')
    with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out),actualDenseArrays=str(array_path.relative_to(P)),actualDenseArraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_REFERENCE_FRONT_GROOM_V10_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

if __name__=='__main__':main()
