"""A single optional gentle smile for the new female head, neutral value zero.

No import-time Blender work. apply() changes only one new head shape key and
creates Basis only if absent. It never activates, renders or saves the smile.
prepare() is an explicit independent-file wrapper. The scalar field moves both
lip bodies and the nearby connected tissue together; it does not flatten the
lip rim, smooth the source coordinates, or reuse old R01/R02 expression keys.
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
KEY='R11 gentle smile'
MARKER='R11_gentle_smile_candidate'
PARAMETERS={'cornerTargetMeters':[.00065,.00035,.00148],
            'centralTargetMeters':[0.,.00010,.00012],
            'geodesicCoreBodyMeters':.007,'geodesicOuterBodyMeters':.042,
            'nasalPinTransitionBodyMeters':.014,'maximumTargetDisplacementMeters':.0018}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _array(items):
    a=np.empty((len(items),3),np.float32);items.foreach_get('co',a.ravel());return a


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Explicit independent reconstruction artifact required')
    return p


def plan(current_cage,female_to_target):
    """Pure data proposal; the actual Multires expression remains unvalidated."""
    skin=_api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']()
    neutral=np.asarray(current_cage,np.float32);points=np.asarray(donor['local'],float)
    T=np.asarray(female_to_target,float);linear=T[:3,:3];scale=float(np.linalg.norm(linear[:,0]))
    if neutral.shape!=(10582,3) or not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):
        raise RuntimeError('Complete female cage and uniform source-to-target scale required')
    mouth=landmarks['mouth'];ring=mouth['openingRimCandidate']['sourceVertexIndices']
    center=points[ring].mean(0);half=float(np.ptp(points[ring,0])*.5)
    allowed=skin['_vertices_of_groups'](donor,{1,7,8,17,22})
    graph=skin['_graph'](points,donor['edges'],allowed)
    distance=skin['_distance'](graph,mouth['vermilionStripVertexIndices'],PARAMETERS['geodesicOuterBodyMeters'])
    core=PARAMETERS['geodesicCoreBodyMeters'];outer=PARAMETERS['geodesicOuterBodyMeters']
    weight=1.-skin['_smooth']((distance-core)/(outer-core))
    eye_center=np.mean([landmarks['eyes'][s]['boundsEllipsoidCenter'] for s in ('L','R')],axis=0)
    eye_gap=eye_center[2]-center[2]
    facial=skin['_vertices_of_groups'](donor,{17})
    possible_nose=facial&(abs(points[:,0]-center[0])<half*.9)&(points[:,2]>center[2]+.30*eye_gap)&(points[:,2]<eye_center[2]-.12*eye_gap)
    nose_ids=np.flatnonzero(possible_nose)
    tip=int(nose_ids[np.argmin(points[nose_ids,1])])
    nasal_pin=facial&(abs(points[:,0]-center[0])<half*.95)&(points[:,2]>center[2]+.28*eye_gap)&(points[:,1]<points[tip,1]+.024)
    pin_distance=skin['_distance'](graph,np.flatnonzero(nasal_pin),PARAMETERS['nasalPinTransitionBodyMeters'])
    weight*=skin['_smooth'](pin_distance/PARAMETERS['nasalPinTransitionBodyMeters'])
    # Both the upper and lower lip bodies share this three-dimensional field.
    # In the 7 mm lip core, there is no Z/Y scaling or compression of lip bulk;
    # the existing cupids bow, convexity and wet-line depth remain the baseline.
    across=skin['_smooth'](np.clip(abs(points[:,0]-center[0])/half,0.,1.))
    local=np.column_stack([
        np.sign(points[:,0]-center[0])*.00065*across,
        .00010+.00025*across,
        .00012+.00136*across,
    ])*weight[:,None]/scale
    local[nasal_pin]=0.
    eyes=skin['_vertices_of_groups'](donor,{2,3,4,5})
    local[eyes]=0.
    # Neck, eyelids, ears and lower-jaw silhouette remain outside this pass.
    lower_pin=points[:,2]<center[2]-.045
    local[lower_pin]=0.
    target=local@linear.T;maximum=float(np.linalg.norm(target,axis=1).max())
    if maximum>PARAMETERS['maximumTargetDisplacementMeters']+1e-9:raise RuntimeError('Smile exceeded 1.8 mm target limit')
    selected=np.linalg.norm(local,axis=1)>1e-9
    if not 300<int(selected.sum())<950:raise RuntimeError('Unexpected connected mouth/cheek scope')
    edges=donor['edges'];old_edge=neutral[edges[:,1]].astype(float)-neutral[edges[:,0]]
    new_edge=old_edge+local[edges[:,1]]-local[edges[:,0]]
    edge_ratio=np.linalg.norm(new_edge,axis=1)/np.maximum(np.linalg.norm(old_edge,axis=1),1e-12)
    affected_edges=selected[edges].any(axis=1)
    triangles=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face=donor['corner_vertices'][start:start+count]
        if selected[face].any():
            triangles.extend((int(face[0]),int(face[j]),int(face[j+1])) for j in range(1,len(face)-1))
    triangles=np.asarray(triangles,np.int32);before=neutral[triangles].astype(float);after=(neutral+local)[triangles]
    normal0=np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0]);normal1=np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
    area0=np.linalg.norm(normal0,axis=1);area1=np.linalg.norm(normal1,axis=1);valid=area0>1e-12
    alignment=np.sum(normal0[valid]*normal1[valid],axis=1)/(area0[valid]*area1[valid])
    if not np.isfinite(local).all() or alignment.min()<=0 or edge_ratio[affected_edges].min()<.75 or edge_ratio[affected_edges].max()>1.25:
        raise RuntimeError('Raw-cage smile has a fold or excessive edge deformation')
    left=min(ring,key=lambda i:points[i,0]);right=max(ring,key=lambda i:points[i,0])
    report={'provenance':provenance,'parameters':PARAMETERS,'maximumTargetDisplacementMeters':maximum,
        'affectedCageVertices':int(selected.sum()),'mouthCenterSource':center.tolist(),
        'cornerSourceIndices':[left,right],'cornerDisplacementsTargetMeters':target[[left,right]].tolist(),
        'centralMotion':'Common upward 0.12 mm / backward 0.10 mm field, smoothly joined to both lip sheets and the cheeks; actual value depends on nasal transition.',
        'noseTipSourceIndex':tip,'nasalPinnedVertices':int(nasal_pin.sum()),
        'rawAffectedEdgeLengthRatioQuantiles':np.quantile(edge_ratio[affected_edges],[0,.05,.5,.95,1]).tolist(),
        'rawTriangleNormalAlignmentMinimum':float(alignment.min()),
        'rawTriangleAreaRatioRange':[float((area1[valid]/area0[valid]).min()),float((area1[valid]/area0[valid]).max())],
        'actualMultiresValidationPending':True,'appearanceAccepted':False,
        'scope':'One new optional mouth/near-cheek smile. No jaw rotation, eye squint, brow action, teeth, tongue, material or topology alteration.',
        'requiredReview':['Same-light neutral, 0.5 and 1.0 with actual current hair and brows, front/quarter/side.',
            'Check upper/lower lip bulk, cupids bow, central seam and corner depth; do not accept only the corner lift.',
            'Run actual evaluated lip-body sections; the 320-point control loop is not a closure target.',
            'Inspect nasolabial compression at the pinned nasal boundary, where the raw edge ratio is strongest.',
            'No closed-eye combination or speech compatibility is established by this single smile.']}
    return local.astype(np.float32),report


def _nonkey_head_state(head,uv,face,skin):
    state=skin['_head_state'](head,0,uv,face);state['keys']=None
    state['allActualMaterials']=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    return state


def _external_identity(bpy,head):
    # This local key insertion never edits other datablocks. Check their actual
    # identities, attachment/material assignments and current rig pose without
    # re-reading every frame of immutable cloth and hair carriers per variant.
    rows=[]
    for obj in bpy.context.scene.objects:
        if obj==head:continue
        row={'object':obj.as_pointer(),'data':getattr(getattr(obj,'data',None),'as_pointer',lambda:None)(),
             'parent':obj.parent.as_pointer() if obj.parent else None,'world':[list(r) for r in obj.matrix_world],
             'modifiers':[(m.name,m.type) for m in obj.modifiers],
             'materials':[s.material.as_pointer() if s.material else None for s in obj.material_slots]}
        if obj.type=='ARMATURE':row['pose']=[(b.name,[list(r) for r in b.matrix_basis]) for b in obj.pose.bones]
        rows.append(row)
    return _hash({'objects':rows,'actions':[(a.name,a.as_pointer()) for a in bpy.data.actions],
                  'frame':bpy.context.scene.frame_current,'subframe':bpy.context.scene.frame_subframe})


def apply(head_name=HEAD):
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    head=bpy.data.objects[head_name];mesh=head.data
    if mesh.library or head.library or mesh.users!=1 or len(mesh.vertices)!=10582:raise RuntimeError('Complete independent new female head required')
    if bpy.context.scene.get(MARKER):raise RuntimeError('Smile candidate already exists')
    previous=mesh.shape_keys
    if previous and (len(previous.key_blocks)!=1 or previous.reference_key.name!='Basis'):
        raise RuntimeError('Inspect existing new-face keys explicitly before adding this first smile')
    neutral=_array(mesh.vertices);T=np.asarray(json.loads(head['R11_female_to_target']),float)
    skin=_api('natural_skin_adult_r11');donor,_,_ =skin['_donor']()
    actual_loops=skin['_array'](mesh.loops,'vertex_index',1,np.int32).ravel()
    if not np.array_equal(actual_loops,donor['corner_vertices']):raise RuntimeError('New female cage topology changed')
    delta,report=plan(neutral,T)
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    bpy.context.view_layer.update();before=_nonkey_head_state(head,uv,face,skin);external=_external_identity(bpy,head)
    active=head.active_shape_key_index;source_sha=_sha(source);created_basis=previous is None;key=None;success=False
    try:
        if created_basis:head.shape_key_add(name='Basis',from_mix=False)
        basis=mesh.shape_keys.reference_key
        if not np.array_equal(_array(basis.data),neutral):raise RuntimeError('Existing Basis is not the current neutral cage')
        key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=basis;key.slider_min=0.;key.slider_max=1.;key.value=0.
        key.data.foreach_set('co',(neutral+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
        if not np.array_equal(_array(basis.data),neutral) or not np.array_equal(_array(mesh.vertices),neutral):
            raise RuntimeError('Neutral source coordinates changed while adding an inactive key')
        after=_nonkey_head_state(head,uv,face,skin)
        if _hash(after)!=_hash(before):raise RuntimeError('Non-key head state changed: '+json.dumps(skin['_differences'](before,after),ensure_ascii=False))
        if _external_identity(bpy,head)!=external or _sha(source)!=source_sha:
            raise RuntimeError('Body/attachment/material/action identity, pose, or source file changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,head=head_name,
            key=KEY,keyValue=0.,createdBasis=created_basis,originalActiveShapeKeyIndex=active,
            neutralCageSha256=skin['_ah'](neutral),deltaSha256=skin['_ah'](delta),
            newKeyCoordinatesSha256=skin['_ah'](_array(key.data)),helperSha256=_sha(Path(__file__)),
            sourceMayContainEarlierInMemoryModules=bool(bpy.data.is_dirty),writtenInMemory=True,savedBlend=False,
            validationScope='Head original coordinates/maps/UV/materials/weights/attachments are hashed. Other objects/datablocks/actions keep identity, attachment and current pose; full cache-frame arrays are not revalidated by this local operation.')
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            if created_basis and mesh.shape_keys and len(mesh.shape_keys.key_blocks)==1:head.shape_key_clear()
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];keys=head.data.shape_keys
    if not keys or KEY not in keys.key_blocks:raise RuntimeError('Recorded smile key missing')
    skin=_api('natural_skin_adult_r11')
    if skin['_ah'](_array(keys.key_blocks[KEY].data))!=report['newKeyCoordinatesSha256']:
        raise RuntimeError('Smile was edited; retain that work before removing it')
    head.shape_key_remove(keys.key_blocks[KEY])
    if report['createdBasis'] and head.data.shape_keys and len(head.data.shape_keys.key_blocks)==1:head.shape_key_clear()
    head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'removedKey':KEY,'sourceSaved':False}


def measure_strength(strength):
    """Actual evaluated lip-body evidence at one temporary strength; no render/save."""
    import bpy
    if not bpy.app.background or not 0.<=strength<=1.:raise ValueError('Bounded explicit background smile strength required')
    if MARKER not in bpy.context.scene:raise RuntimeError('Prepare the explicit new smile candidate first')
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']]
    key=head.data.shape_keys.key_blocks[KEY];previous=key.value;qa=_api('face_contact_audit_r11')
    source=Path(bpy.data.filepath);source_sha=_sha(source)
    try:
        with qa['rest_evaluation']():
            key.value=float(strength);bpy.context.view_layer.update()
            data=qa['_evaluated_mesh'](head);T=qa['_matrix'](head);inverse=np.linalg.inv(T)
            body=qa['_transform'](data['world'],inverse);faces=np.asarray(data['faces'],np.int32)
            basis=_array(key.relative_key.data).astype(float);cage=basis+strength*(_array(key.data)-basis)
            cage=qa['_transform'](qa['_transform'](cage,data['matrix']),inverse)
            landmarks=qa['evaluated_landmarks'](body,faces,data['groups'],cage)
            result={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
                'key':KEY,'temporaryStrength':strength,'actualLipControlLoopVertices':len(landmarks['mouth']['ring']),
                'actualLipSurfaceSections':qa['lip_section_contacts'](body,faces,data['groups'],cage,T),
                'appearanceAccepted':False,'savedBlend':False,
                'scope':'Actual evaluated Multires lip-body sections at one temporary smile value. Does not replace same-light images or establish all facial-expression compatibility.'}
    finally:
        key.value=previous;bpy.context.view_layer.update()
    if _sha(source)!=source_sha:raise RuntimeError('Source disk file changed during expression measurement')
    return result


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New independent output and report required')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit source required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_GENTLE_SMILE_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256'],'keyValue':0.}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
