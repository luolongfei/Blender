"""Measure all before/after dense hair, then save the local original-reference v9 (optional raw radius fix)."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent;MAIN='R11 Donor / long hair main'
PINS={
    'reference_fringe_local_v9_r11.py':'c7400ffdaee88aa0807e97e95cd34bb3df7c0f99c03bf1dc62f85ac51fa03306',
    'reference_fringe_shape_v9_r11.py':'c04dd8a94f2a714fc4f3e32ce635c1f1e0b553c749beb553baedfa2ae933e3e5',
    'reference_fringe_preservation_v9_r11.py':'1318e622ac8f3f4c5007a9c9a15bc66a805744792a9513154638f534caeba963',
    'prepare_reference_fringe_selection_v8_r11.py':'497b618b9108ba3d705eb9251375f2aa8668005e0b97c8f24b8c691280a3c9b0',
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
        return result
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    report_path=out.with_suffix('.groom-fringe-local.json');array_path=out.with_suffix('.groom-fringe-local.npz')
    for path in (source,out,report_path,array_path):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,report_path,array_path)):raise RuntimeError('Fresh independent source and unused outputs required')
    for name,digest in PINS.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen helper changed: '+name)
    main_obj=bpy.data.objects[MAIN]
    if not main_obj.get('R11_reference_fringe_v8'):raise RuntimeError('Only a saved v8-derived groom may receive the local edit')
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    carriers=[o.name for o in bpy.data.objects if o.get('R11_groom_role_root_source')==MAIN]
    protected=[o for o in bpy.data.objects if o.type in {'MESH','CURVES'} and o.name not in [MAIN]+carriers]
    before={o.name:contract(o) for o in protected}
    modifiers={o.name:modifier_contract(o) for o in bpy.data.objects}
    visibility={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));old_raw=fit['_coords'](main_obj)
    raw_radii=raw_radius_contract(main_obj.data)
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;scene=bpy.context.scene
    frame=(scene.frame_current,scene.frame_subframe);action=rig.animation_data.action if rig.animation_data else None
    arrays={};report={'source':a.source,'sourceSha256':sha(source),'savedBlend':False,'appearanceAccepted':False,'wrapperSha256':sha(__file__)}
    helpers=runpy.run_path(str(HERE/'prepare_reference_fringe_selection_v8_r11.py'))
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        print('R11_REFERENCE_V9_MEASURING_ACTUAL_SOURCE',flush=True)
        report['actualDenseBefore']=capture_before(arrays,helpers)
        print('R11_REFERENCE_V9_LOCAL_SHAPE',flush=True)
        report.update(runpy.run_path(str(HERE/'reference_fringe_local_v9_r11.py'))['apply'](expected_source_relative=a.source))
        print('R11_REFERENCE_V9_MEASURING_ALL_ACTUAL_DENSE_OUTPUT',flush=True)
        proof=helpers['actual_dense_proof'](arrays);report['actualDenseOutput']=proof
        preservation=runpy.run_path(str(HERE/'reference_fringe_preservation_v9_r11.py'))['compare'](arrays);report['allActualDenseBeforeAfter']=preservation
        report['fineWispsActualProof']=helpers['actual_wisp_proof'](arrays)
        errors=list(proof['errors'])+list(preservation['errors'])
        if before!={o.name:contract(o) for o in protected}:errors.append('Protected geometry/keys/materials/parent contract changed')
        if modifiers!={o.name:modifier_contract(o) for o in bpy.data.objects}:errors.append('An existing modifier or source node group changed')
        if visibility!={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}:errors.append('Object visibility changed')
        new_radii=raw_radius_contract(main_obj.data)
        report['rawGuideRadiusAttributeBefore']=raw_radii;report['rawGuideRadiusAttributeAfter']=new_radii
        if raw_radii!=new_radii:errors.append('Raw guide radius attribute presence/type/domain/values changed')
        if (scene.frame_current,scene.frame_subframe)!=frame or (rig.animation_data.action if rig.animation_data else None)!=action:errors.append('Body action or scene frame changed')
        if sha(source)!=report['sourceSha256']:errors.append('Saved input changed')
        report['errors']=errors
        print('R11_REFERENCE_V9_ACTUAL_DENSE_CHECKED '+json.dumps({'curves':proof['stages'][-1]['curves'],'errors':errors}),flush=True)
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
    scene['R11_reference_front_groom_v9']=json.dumps(report,ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Candidate save failed')
    with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out),actualDenseArrays=str(array_path.relative_to(P)),actualDenseArraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_REFERENCE_FRONT_GROOM_V9_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

if __name__=='__main__':main()
