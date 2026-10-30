"""Explicit, reversible upper-lip contour candidate; no import-time Blender work.

Amplitude is a required argument, chosen after reviewing the adopted lower-face
images. The separately retained offline study is not overwritten. Only a new
relative key is added; actual evaluated lip contact and protected surfaces are
compared on this precise source before a new candidate file can be saved.
"""
import argparse
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
KEY='R11 soft upper vermilion contour'
MARKER='R11_upper_vermilion_contour_candidate'
STUDY_HELPER_SHA='bd4ad5c22354f7df0758dd4102a324b5024a7ead1a1bc3669cfaf3220fd60325'
STUDY_REPORT='work/reconstruction-r02/upper-lip-topology-study-r11.json'
STUDY_REPORT_SHA='507ffbe6e79039e1cb5ebe5617fad76b1cf110d4fa0c091f8d178a2326ebd7ab'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(data):
    a=np.empty((len(data),3),np.float32);data.foreach_get('co',a.ravel());return a


def _path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Explicit reconstruction artifact required')
    return p


def _strength(value):
    value=float(value)
    if not np.isfinite(value) or not 0.<value<=1.:raise ValueError('Choose an explicit candidate strength above0 and at most1 after reviewing the new lower face')
    return value


def _study():
    helper=Path(__file__).with_name('upper_lip_topology_study_r11.py');report=project_path()/STUDY_REPORT
    if _sha(helper)!=STUDY_HELPER_SHA or _sha(report)!=STUDY_REPORT_SHA:
        raise RuntimeError('Retained draft provenance changed; inspect and explicitly revise this candidate before use')
    return runpy.run_path(str(helper))


def _keys(mesh,uv,skin):
    return {'animation':uv['_animation'](mesh.shape_keys),'custom':uv['_custom'](mesh.shape_keys),
        'blocks':{k.name:{'coordinatesSha256':skin['_ah'](_array(k.data)),
            'settings':uv['_scalar_rna'](k),'relative':k.relative_key.name} for k in mesh.shape_keys.key_blocks}}


def _nonkey(head,face):
    state=_api('eye_margin_refine_r11')['_state'](head,0);state['keys']=None
    state['allMaterials']=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    return state


def _actual(head):
    qa=_api('face_contact_audit_r11');uv=_api('production_uv_r08')
    data=qa['_evaluated_mesh'](head);T=np.asarray(data['matrix'],float)
    cage,metadata=uv['current_mix'](head);cage=np.asarray(cage,float)
    body=qa['_transform'](data['world'],np.linalg.inv(T));faces=np.asarray(data['faces'],np.int32)
    sections=qa['lip_section_contacts'](body,faces,data['groups'],cage,T)
    return {'data':data,'cage':cage,'cageWorld':qa['_transform'](cage,T),'mixMetadata':metadata,'lipSections':sections}


def _audit(before,after,strength,changed_ids):
    """Actual evaluated evidence, not the retained donor-study closure numbers."""
    a=before['data'];b=after['data'];v0=np.asarray(a['world']);v1=np.asarray(b['world'])
    if a['faces']!=b['faces'] or not np.array_equal(a['groups'],b['groups']):raise RuntimeError('Actual topology or anatomical region identities changed')
    delta=v1-v0;distance=np.linalg.norm(delta,axis=1);faces=np.asarray(a['faces'],np.int32)
    if distance.max()>strength*.0006+2e-6:raise RuntimeError('Actual upper-lip deformation exceeded its bounded amplitude: '+str(distance.max()))
    old_sections=before['lipSections']['sections'];new_sections=after['lipSections']['sections']
    old_g=np.asarray([r['minimumSurfaceGapMeters'] for r in old_sections]);new_g=np.asarray([r['minimumSurfaceGapMeters'] for r in new_sections])
    gap_change=np.abs(new_g-old_g)
    if gap_change.max()>1e-5:raise RuntimeError('Actual lip-body contact changed by over10micrometers: '+str(gap_change.tolist()))
    for old,new in zip(old_sections,new_sections):
        if old['sectionCrossings']!=new['sectionCrossings']:raise RuntimeError('Actual lip-section crossing count changed; inspect the contact geometry')
    contact_points=np.asarray([r[name] for r in old_sections for name in ('upperNearestWorld','lowerNearestWorld')])
    contact_band=np.zeros(len(v0),bool)
    for point in contact_points:contact_band|=np.linalg.norm(v0-point,axis=1)<=.001
    if not np.any(contact_band):raise RuntimeError('Actual L3 contact neighborhood could not be identified')
    contact_error=float(distance[contact_band].max())
    if contact_error>1e-5:raise RuntimeError('Actual contact surface points moved despite pinned cage rows: '+str(contact_error))
    _,land,_=_api('natural_skin_adult_r11')['_donor']();opening=land['mouth']['openingRimCandidate']
    corner_seeds=[opening['leftCornerCandidate'],opening['rightCornerCandidate']]
    corners=np.zeros(len(v0),bool)
    for index in corner_seeds:corners|=np.linalg.norm(v0-before['cageWorld'][index],axis=1)<=.004
    corner_error=float(distance[corners].max()) if np.any(corners) else float('inf')
    if corner_error>2e-6:raise RuntimeError('Actual mouth-corner tissue moved: '+str(corner_error))
    internal=np.unique(faces[np.isin(a['groups'],[2,3,4,5,7,8,22])])
    fixed=np.zeros(len(v0),bool);fixed[internal]=True
    # An 8 mm neighborhood covers the actual Multires interpolation support.
    # Everything farther away, including nose body, eyes, jaw and low neck,
    # must stay unchanged. No smoothing or generalized face warp is allowed.
    near=np.zeros(len(v0),bool)
    for index in changed_ids:near|=np.linalg.norm(v0-before['cageWorld'][index],axis=1)<=.008
    fixed|=~near
    protected_error=float(distance[fixed].max())
    if protected_error>2e-6:raise RuntimeError('Actual lower lip/oral/eye/ear or outside-patch surface moved: '+str(protected_error))
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[count==1])
    boundary_error=float(distance[boundary].max()) if len(boundary) else 0.
    if boundary_error>2e-6:raise RuntimeError('Actual head/neck boundary changed')
    changed=distance>1e-7;region=faces[changed[faces].any(1)]
    if not len(region):raise RuntimeError('Nonzero upper-lip candidate has no actual evaluated effect')
    triangles=np.concatenate([region[:,[0,1,2]],region[:,[0,2,3]]])
    p=v0[triangles];q=v1[triangles]
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    ar0=np.linalg.norm(n0,axis=1);ar1=np.linalg.norm(n1,axis=1);valid=ar0>1e-14
    align=np.sum(n0[valid]*n1[valid],axis=1)/(ar0[valid]*ar1[valid]);ratio=ar1[valid]/ar0[valid]
    if not np.isfinite(align).all() or align.min()<=0 or ratio.min()<.60 or ratio.max()>1.60:
        raise RuntimeError('Actual L3 patch folded or strained excessively')
    return {'temporaryStrength':strength,'actualChangedVerticesAbove0_1Micrometer':int(changed.sum()),
        'maximumActualWorldDeltaMeters':float(distance.max()),'actualContactBandVertices':int(contact_band.sum()),
        'actualContactBandMaximumMeters':contact_error,'actualCornerPatchVertices':int(corners.sum()),
        'actualCornerPatchMaximumMeters':corner_error,'actualProtectedSurfaceMaximumMeters':protected_error,
        'actualBoundaryVertices':len(boundary),'actualBoundaryMaximumMeters':boundary_error,
        'maximumActualContactGapChangeMeters':float(gap_change.max()),
        'actualLipSectionsBefore':before['lipSections'],'actualLipSectionsAfter':after['lipSections'],
        'actualPatchNormalAlignmentMinimum':float(align.min()),'actualPatchAreaRatioRange':[float(ratio.min()),float(ratio.max())],
        'appearanceAccepted':False,'expressionCombinationsVerified':False,
        'limitations':'Nine true lip-body sections, fixed 1mm contact neighborhoods, mouth-corner patches and actual local triangle strain are tested. This is not a complete global self-collision or oral-volume proof; images and later expression combinations are still required.'}


def apply(strength):
    import bpy
    strength=_strength(strength);study=_study();source=Path(bpy.data.filepath).resolve()
    source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582 or not keys:raise RuntimeError('Actual complete local female head and retained keys required')
    if MARKER in bpy.context.scene or KEY in keys.key_blocks:raise RuntimeError('Retain/restore an earlier upper-lip candidate before preparing another')
    if any(abs(k.value)>1e-8 for k in keys.key_blocks if any(word in k.name.lower() for word in ('smile','blink'))):
        raise RuntimeError('Choose the adopted neutral expression; preserve but leave smile/blink controls at zero')
    skin=_api('natural_skin_adult_r11');uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    qa=_api('face_contact_audit_r11');binding=_api('face_fibre_binding_r11');gentle=_api('gentle_smile_adult_r11')
    if binding['MARKER'] not in bpy.context.scene:raise RuntimeError('Existing validated face fibre surface binding must be retained')
    donor,_,_=skin['_donor']()
    if not np.array_equal(skin['_array'](mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):raise RuntimeError('Actual female topology differs')
    bpy.context.view_layer.update();prior=_keys(mesh,uv,skin);contract=_nonkey(head,face)
    external=gentle['_external_identity'](bpy,head);source_sha=_sha(source);active=head.active_shape_key_index;key=None;success=False
    try:
        with qa['rest_evaluation']():
            before=_actual(head);T=np.asarray(head.matrix_world,float);scale=np.linalg.norm(T[:3,0])
            if not np.allclose(T[:3,:3],np.eye(3)*scale,atol=8e-7):raise RuntimeError('Documented upright female REST frame required')
            delta_world,plan=study['study'](before['cageWorld'])
            delta_local=(np.asarray(delta_world,float)@np.linalg.inv(T[:3,:3]).T).astype(np.float32)
            basis=_array(keys.reference_key.data)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',(basis+delta_local).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            inactive=binding['_head_mesh'](head)
            if inactive['world'].shape!=np.asarray(before['data']['world']).shape:raise RuntimeError('Inactive candidate changed evaluated count')
            inactive_error=float(np.max(np.linalg.norm(inactive['world']-np.asarray(before['data']['world']),axis=1)))
            if inactive_error>2e-8:raise RuntimeError('Inactive candidate changed actual geometry: '+str(inactive_error))
            key.value=strength;bpy.context.view_layer.update();after=_actual(head)
            validation=_audit(before,after,strength,plan['changedIndices']);fibre=binding['validate'](neutral=False)
            key.value=0.;bpy.context.view_layer.update()
        current=_keys(mesh,uv,skin);current['blocks'].pop(KEY)
        if current!=prior:raise RuntimeError('Original shape coordinates/settings/relative bases, drivers or actions changed')
        now=_nonkey(head,face)
        if gentle['_hash'](now)!=gentle['_hash'](contract):raise RuntimeError('Original head materials/UV/maps/weights/modifiers changed: '+json.dumps(skin['_differences'](contract,now)))
        if gentle['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('Other object/rig/action/pose or source disk changed')
        key.value=strength;bpy.context.view_layer.update()
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
            'head':HEAD,'key':KEY,'keyValue':strength,'originalActiveShapeKeyIndex':active,
            'retainedDraftReport':STUDY_REPORT,'retainedDraftReportSha256':STUDY_REPORT_SHA,'studyHelperSha256':STUDY_HELPER_SHA,
            'fullStrengthPlanRecomputedFromCurrentMix':plan,'currentMixMetadata':before['mixMetadata'],
            'newKeyCoordinatesSha256':skin['_ah'](_array(key.data)),'localDeltaSha256':skin['_ah'](delta_local),
            'existingKeysPreserved':list(prior['blocks']),'actualInactiveGeometryErrorMeters':inactive_error,
            'actualValidation':validation,'actualFibreBindingValidation':fibre,'appearanceAccepted':False,
            'actualImagesPending':True,'savedBlend':False,
            'scope':'Only a new optional shape key. Current material and pigment/roughness layers, original keys, eye shape/attachments, jaw and low-neck connection data are retained. Explicit strength is a candidate choice, never automatic acceptance.'}
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def validate(strength):
    """Read-only temporary neutral/candidate comparison on a prepared file."""
    import bpy
    strength=_strength(strength)
    if not bpy.app.background or MARKER not in bpy.context.scene:raise RuntimeError('Explicit prepared background candidate required')
    record=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[record['head']];key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=record['newKeyCoordinatesSha256']:raise RuntimeError('Candidate coordinates have changed')
    old=key.value;qa=_api('face_contact_audit_r11');source=Path(bpy.data.filepath);source_sha=_sha(source)
    try:
        with qa['rest_evaluation']():
            key.value=0.;bpy.context.view_layer.update();before=_actual(head)
            key.value=strength;bpy.context.view_layer.update();after=_actual(head)
            result=_audit(before,after,strength,record['fullStrengthPlanRecomputedFromCurrentMix']['changedIndices'])
            result.update(source=str(source.resolve().relative_to(project_path())),sourceSha256=source_sha,
                actualFibreBindingValidation=_api('face_fibre_binding_r11')['validate'](neutral=False),savedBlend=False)
    finally:key.value=old;bpy.context.view_layer.update()
    if _sha(source)!=source_sha:raise RuntimeError('Source disk changed during validation')
    return result


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    record=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[record['head']];keys=head.data.shape_keys;key=keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=record['newKeyCoordinatesSha256']:raise RuntimeError('Retain edited candidate before restoration')
    animation=keys.animation_data
    for curve in getattr(animation,'drivers',[]) if animation else []:
        paths=[curve.data_path]+[target.data_path for var in curve.driver.variables for target in var.targets]
        if any(KEY in path for path in paths):raise RuntimeError('A later driver depends on this lip candidate; preserve it explicitly')
    for action in bpy.data.actions:
        for curve in _api('native_hair_playback_r04')['_fcurves'](action):
            if KEY in curve.data_path:raise RuntimeError('An animation action references this lip key; retain that work before restoration')
    head.shape_key_remove(key);head.active_shape_key_index=record['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'removedKey':KEY,'savedBlend':False}


def prepare(source_relative,output_relative,strength):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.upper-lip.json')
    if source==output or output.exists() or report_path.exists() or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh explicit source and new output/report required; no historical candidate overwrite')
    report=apply(strength)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_UPPER_LIP_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256'],'strength':strength,'report':str(report_path.relative_to(project_path()))}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--strength',required=True,type=float,help='Explicit 0<value<=1, selected after actual new lower-face image review; no default amplitude')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output,args.strength)
