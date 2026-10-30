"""Independent coordinated soft-smile candidate on the actual R11 female face.

Keeps the prior gentle-smile key, Basis and all other keys. The new optional key
uses its measured mouth component plus compact connected cheek and lower-lid
fields. No global Gaussian/full-face warp, globe move, nose lift, brow change,
new wrinkle texture or jaw action. A lower-lid rotation follows the actual eye
centre; it is not a blink. New key is saved at zero. Real images and blink
combinations remain required; neither AU naming nor numeric tests prove a smile.
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
KEY='R11 natural soft smile'
MOUTH_KEY='R11 gentle smile'
MARKER='R11_natural_soft_smile_candidate'
PARAMETERS={'mouthComponentScale':1.,'cheekLiftWorldMeters':.00095,
    'cheekMedialWorldMeters':.00022,'cheekForwardWorldMeters':.00018,
    'cheekGeodesicCoreWorldMeters':.005,'cheekGeodesicOuterWorldMeters':.032,
    'lowerLidMaximumRotationRadians':-.032,'lowerLidGeodesicCoreWorldMeters':.0025,
    'lowerLidGeodesicOuterWorldMeters':.0105,'maximumTotalWorldMeters':.0022}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _hash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _array(items):
    a=np.empty((len(items),3),np.float32);items.foreach_get('co',a.ravel());return a
def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!='.blend':raise ValueError('Independent reconstruction Blend required')
    return p
def _sparse(delta):
    ids=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-10)
    return {'sourceVertexIndices':ids.tolist(),'deltaLocal':np.asarray(delta)[ids].tolist()}


def plan(current_mix,mouth_delta,female_to_target):
    """Pure data; actual old mouth delta + current adopted neutral anatomy."""
    skin=_api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']()
    current=np.asarray(current_mix,np.float32);mouth_delta=np.asarray(mouth_delta,np.float32)
    T=np.asarray(female_to_target,float);linear=T[:3,:3];inverse=np.linalg.inv(linear)
    if current.shape!=(10582,3) or mouth_delta.shape!=current.shape:raise RuntimeError('Complete actual R11 cage and preserved mouth key required')
    world=current@linear.T+T[:3,3];mouth_world=mouth_delta@linear.T
    scale=float(np.linalg.norm(linear[:,0]))
    if not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):raise RuntimeError('Uniform actual female mapping required')
    audit_path=project_path()/'work/reconstruction-r02/adult-eyelid-visible-audit-r11.json'
    audit=json.loads(audit_path.read_text(encoding='utf8'))
    observation=project_path()/'work/reconstruction-r02/adult-neck-observation-r11.npz'
    if _sha(observation)!=audit['headArraySha256']:raise RuntimeError('Actual visible-eye observation payload changed')
    with np.load(observation) as payload:observed_world=payload['rest_00_raw_world'].copy()
    face=skin['_vertices_of_groups'](donor,{17});orbital=skin['_vertices_of_groups'](donor,{2,3,17})
    graph=skin['_graph'](world,donor['edges'],orbital)
    lip=np.asarray(land['mouth']['vermilionStripVertexIndices'],np.int32)
    opening=np.asarray(land['mouth']['openingRimCandidate']['sourceVertexIndices'],np.int32)
    mouth_center=world[opening].mean(0);eye_centers=np.array([audit['eyes'][s]['centerWorld'] for s in ('L','R')])
    eye_center=eye_centers.mean(0);ipd=float(abs(eye_centers[0,0]-eye_centers[1,0]));height=float(eye_center[2]-mouth_center[2])
    if not .05<ipd<.075 or not .045<height<.08:raise RuntimeError('Current face/eye mapping differs from measured adult anatomy')
    upper_pin=np.zeros(len(world),bool)
    for row in land['eyes'].values():upper_pin[np.asarray(row['orderedUpperInnerToOuter'],np.int32)]=True
    nasal=face&(abs(world[:,0]-mouth_center[0])<.018)&(world[:,2]>mouth_center[2]+.018)&(world[:,2]<eye_center[2]+.006)&(world[:,1]<eye_center[1]+.01)
    nose_distance=skin['_distance'](graph,np.flatnonzero(nasal),.012)
    lip_distance=skin['_distance'](graph,lip,.020)
    cheek=np.zeros_like(world);lid=np.zeros_like(world);sides={};eye_patch=np.zeros(len(world),bool)
    for side,row in audit['eyes'].items():
        sign=1 if side=='L' else -1;center=np.asarray(row['centerWorld']);contour=row['contour']
        xs=np.array([q['x'] for q in contour])
        # Diagnostic neighbours seed the tissue influence only, never a new
        # wet-line, eye-closure boundary, or fake extrapolated sphere contact.
        seeds=sorted({int(i) for q in contour[15:-15] for i in q['lower']['nearestRawDiagnosticVertices'][:2]})
        distance=skin['_distance'](graph,seeds,.020)
        t=(world[:,0]-xs[0])/(xs[-1]-xs[0])
        across=np.sin(np.pi*np.clip(t,0.,1.))**.65
        corners=skin['_smooth']((t-.08)/.12)*(1.-skin['_smooth']((t-.80)/.12))
        below=1.-skin['_smooth']((world[:,2]-center[2]+.001)/.0018)
        anterior=skin['_smooth']((center[1]-world[:,1]-.0005)/.004)
        lid_weight=1.-skin['_smooth']((distance-PARAMETERS['lowerLidGeodesicCoreWorldMeters'])/
            (PARAMETERS['lowerLidGeodesicOuterWorldMeters']-PARAMETERS['lowerLidGeodesicCoreWorldMeters']))
        lid_weight*=across*corners*below*anterior;lid_weight[upper_pin|(~orbital)]=0.
        active_lid=lid_weight>1e-9
        observed_error=float(np.max(np.abs(world[active_lid]-observed_world[active_lid])))
        if observed_error>4e-7:raise RuntimeError('Lower-lid rest anatomy changed; renew actual visible-contour observation '+side+' '+str(observed_error))
        theta=PARAMETERS['lowerLidMaximumRotationRadians']*lid_weight;v=world-center;d=np.zeros_like(world)
        d[:,1]=v[:,1]*np.cos(theta)-v[:,2]*np.sin(theta)-v[:,1]
        d[:,2]=v[:,1]*np.sin(theta)+v[:,2]*np.cos(theta)-v[:,2]
        lid+=d
        # Cheek apex is selected from real face-set 17 near the measured adult
        # eye/mouth relation; its support travels connected surface edges.
        eligible=face&(world[:,0]*sign>.01)&(world[:,1]<center[1]+.025)&(world[:,2]<center[2]-.008)&(world[:,2]>mouth_center[2]+.015)
        candidates=np.flatnonzero(eligible);target=np.array([sign*.57*ipd,eye_center[2]-.42*height])
        if not len(candidates):raise RuntimeError('No actual anatomical cheek domain '+side)
        seed=int(candidates[np.argmin(np.sum((world[candidates][:,[0,2]]-target)**2,axis=1))])
        cheek_distance=skin['_distance'](graph,[seed],PARAMETERS['cheekGeodesicOuterWorldMeters'])
        weight=1.-skin['_smooth']((cheek_distance-PARAMETERS['cheekGeodesicCoreWorldMeters'])/
            (PARAMETERS['cheekGeodesicOuterWorldMeters']-PARAMETERS['cheekGeodesicCoreWorldMeters']))
        weight*=skin['_smooth'](nose_distance/.012)*skin['_smooth']((lip_distance-.010)/.010)
        # Contact-band movement is supplied by the spherical lid rotation,
        # not by a cheek translation that would push skin through the globe.
        weight*=skin['_smooth']((distance-.004)/.006)
        weight*=1.-skin['_smooth']((world[:,2]-(center[2]-.002))/.003)
        weight[~face|upper_pin|nasal|(world[:,0]*sign<=.006)|(world[:,2]<=mouth_center[2]+.008)]=0.
        direction=np.array([-sign*PARAMETERS['cheekMedialWorldMeters'],-PARAMETERS['cheekForwardWorldMeters'],PARAMETERS['cheekLiftWorldMeters']])
        cheek_side=weight[:,None]*direction;cheek+=cheek_side
        patch=(distance<.014)&(world[:,0]*sign>.006)&(world[:,2]<center[2]+.001)
        eye_patch|=patch
        sides[side]={'cheekSeed':seed,'cheekSeedCurrentWorld':world[seed].tolist(),'lowerContourDiagnosticSeeds':seeds,
            'cheekAffectedCageVertices':int(np.count_nonzero(weight>1e-9)),
            'lowerLidAffectedCageVertices':int(np.count_nonzero(lid_weight>1e-9)),
            'cheekMaximumWorldMeters':float(np.linalg.norm(cheek_side,axis=1).max()),
            'lowerLidMaximumWorldMeters':float(np.linalg.norm(d,axis=1).max()),
            'maximumLowerLidRotationRadians':float(abs(theta).max())}
    oral=skin['_vertices_of_groups'](donor,{7,8});protected=oral.copy();protected[lip]=True
    if np.any(cheek[protected]!=0.) or np.any(lid[protected]!=0.):raise RuntimeError('New eye/cheek fields entered lip/oral tissue')
    if np.any(cheek[upper_pin]!=0.) or np.any(lid[upper_pin]!=0.):raise RuntimeError('Upper-lid/canthus controls moved')
    if np.any(cheek[nasal]!=0.) or np.any(lid[nasal]!=0.):raise RuntimeError('Nose/philtrum protected tissue moved')
    total_world=mouth_world*PARAMETERS['mouthComponentScale']+cheek+lid
    delta=(total_world@inverse.T).astype(np.float32)
    if not np.array_equal(delta[protected],mouth_delta[protected]):
        # Avoid a matrix round-trip changing the protected mouth component.
        delta[protected]=mouth_delta[protected]
    selected=np.linalg.norm(delta,axis=1)>1e-9
    if np.linalg.norm(total_world,axis=1).max()>PARAMETERS['maximumTotalWorldMeters'] or not 500<int(selected.sum())<1600:raise RuntimeError('Unexpected coordinated-smile scope or amplitude')
    edges=donor['edges'];a=current[edges[:,1]].astype(float)-current[edges[:,0]];b=a+delta[edges[:,1]]-delta[edges[:,0]]
    active=selected[edges].any(1);ratio=np.linalg.norm(b,axis=1)/np.maximum(np.linalg.norm(a,axis=1),1e-12)
    triangles=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face_ids=donor['corner_vertices'][start:start+count]
        if selected[face_ids].any():triangles.extend((int(face_ids[0]),int(face_ids[j]),int(face_ids[j+1])) for j in range(1,len(face_ids)-1))
    triangles=np.asarray(triangles,np.int32);old=current[triangles].astype(float);new=(current+delta)[triangles]
    n0=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);n1=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0])
    a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1);valid=a0>1e-12
    alignment=np.sum(n0[valid]*n1[valid],axis=1)/(a0[valid]*a1[valid])
    if alignment.min()<=0 or ratio[active].min()<.75 or ratio[active].max()>1.25:raise RuntimeError('Raw-cage fold/excess strain; do not call this a visual success')
    report={'provenance':provenance,'visibleContourAuditSha256':_sha(audit_path),'parameters':PARAMETERS,'sides':sides,
        'affectedCageVertices':int(selected.sum()),'maximumWorldDisplacementMeters':float(np.linalg.norm(total_world,axis=1).max()),
        'mouthComponentKeptExactOnLipAndOralVertices':True,'nosePhiltrumUpperLidCanthusControlsPinnedForNewFields':True,
        'rawEdgeRatioQuantiles':np.quantile(ratio[active],[0,.05,.5,.95,1]).tolist(),
        'rawTriangleAreaRatioRange':[float((a1[valid]/a0[valid]).min()),float((a1[valid]/a0[valid]).max())],
        'rawTriangleNormalAlignmentMinimum':float(alignment.min()),
        'lowerLidComponent':_sparse((lid@inverse.T).astype(np.float32)),
        'eyeContactPatchFullSmileDelta':_sparse(np.where(eye_patch[:,None],delta,0.)),
        'eyeContactPatchMeaning':'Diagnostic same-topology lower-orbital support for the separate blink author. Not a certified contact band or automatic blink corrective; use actual closed-eye anatomy and combination views.',
        'appearanceAccepted':False,'evaluatedReviewPending':True,
        'sources':['https://studio.blender.org/training/realistic-human-research/the-shapes-approach/',
                   'https://pmc.ncbi.nlm.nih.gov/articles/PMC8317937/'],
        'scope':'One coordinated optional smile. Existing mouth tissue delta plus compact geodesic cheek raise and lower-lid sphere-following rotation. No new wrinkle detail, nose/brow action, globe movement, jaw rotation or whole-face warp.',
        'requiredReview':['Same-light neutral, 0.5 and 1.0, front/quarter/side with actual current hair and lip colour.',
            'Check cheek volume and nasolabial continuity; no hard diagonal crease, inflated cheek or central lip flattening.',
            'Check real lower-lid/globe contact, eyelash surface following and mouth sections. Preserve subtle lower-lid thickness.',
            'The separate blink keys must be combined and visually checked; adding cheek raising is not by itself proof of a natural smile.']}
    return delta,report


def _key_records(mesh,uv):
    if not mesh.shape_keys:raise RuntimeError('Preserved existing R11 keys required')
    return {'animation':uv['_animation'](mesh.shape_keys),'custom':uv['_custom'](mesh.shape_keys),
        'blocks':{key.name:{'coordinatesSha256':_api('natural_skin_adult_r11')['_ah'](_array(key.data)),
            'settings':uv['_scalar_rna'](key),'relative':key.relative_key.name} for key in mesh.shape_keys.key_blocks}}


def _nonkey_state(head,uv,face):
    state=_api('eye_margin_refine_r11')['_state'](head,0);state['keys']=None
    state['allMaterials']=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    return state


def apply():
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    if MARKER in bpy.context.scene:raise RuntimeError('Retain or restore the earlier natural-smile candidate')
    binding=_api('face_fibre_binding_r11')
    if binding['MARKER'] not in bpy.context.scene:raise RuntimeError('Actual brow/lash surface binding required before adding smile eye movement')
    head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582 or not keys or MOUTH_KEY not in keys.key_blocks:
        raise RuntimeError('Complete current R11 female cage and preserved gentle-smile key required')
    if KEY in keys.key_blocks or abs(keys.key_blocks[MOUTH_KEY].value)>1e-8:raise RuntimeError('Keep original gentle smile at zero while authoring/using this alternative')
    if any(abs(k.value)>1e-7 for k in keys.key_blocks if k.name.startswith('R11 blink')):raise RuntimeError('Open-eye adopted rest required; blink combination is a later explicit check')
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08');skin=_api('natural_skin_adult_r11');qa=_api('face_contact_audit_r11')
    donor,_,_=skin['_donor']()
    if not np.array_equal(skin['_array'](mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):raise RuntimeError('Actual female topology identity changed')
    template=keys.key_blocks[MOUTH_KEY]
    if template.relative_key!=keys.reference_key or template.vertex_group or template.mute:raise RuntimeError('Inspect altered original mouth-key semantics before combining')
    mouth_delta=(_array(template.data)-_array(template.relative_key.data)).astype(np.float32)
    prior_smile=json.loads(bpy.context.scene['R11_gentle_smile_candidate'])
    if skin['_ah'](_array(template.data))!=prior_smile['newKeyCoordinatesSha256']:raise RuntimeError('Existing mouth key was edited; inspect before reusing its tissue component')
    bpy.context.view_layer.update();before=_nonkey_state(head,uv,face);prior_keys=_key_records(mesh,uv)
    external=_api('gentle_smile_adult_r11')['_external_identity'](bpy,head);source_sha=_sha(source)
    active=head.active_shape_key_index;key=None;success=False;was_dirty=bool(bpy.data.is_dirty)
    try:
        with qa['rest_evaluation']():
            current,current_mix_metadata=uv['current_mix'](head)
            current=np.asarray(current,np.float32);T=np.asarray(head.matrix_world,float)
            delta,report=plan(current,mouth_delta,T);baseline=_array(keys.reference_key.data)
            actual_before=binding['_head_mesh'](head)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',(baseline+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            actual_after=binding['_head_mesh'](head)
            if actual_before['topologySha256']!=actual_after['topologySha256'] or not np.array_equal(actual_before['co'],actual_after['co']):raise RuntimeError('Inactive new smile changed actual neutral geometry')
            neutral_binding=binding['validate'](neutral=False)
        now_keys=_key_records(mesh,uv);now_keys['blocks'].pop(KEY)
        if now_keys!=prior_keys:raise RuntimeError('Original keys, drivers, relative bases or values changed')
        after=_nonkey_state(head,uv,face)
        if _hash(before)!=_hash(after):raise RuntimeError('Original non-key source changed: '+json.dumps(skin['_differences'](before,after)))
        if _api('gentle_smile_adult_r11')['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('External objects/actions/pose or source file changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,sourceMayContainEarlierInMemoryModules=was_dirty,
            helperSha256=_sha(__file__),head=HEAD,key=KEY,keyValue=0.,mouthTemplateKey=MOUTH_KEY,
            mouthTemplateCoordinatesSha256=prior_smile['newKeyCoordinatesSha256'],originalActiveShapeKeyIndex=active,
            baselineActualMixSha256=skin['_ah'](current),currentMixMetadata=current_mix_metadata,newKeyCoordinatesSha256=skin['_ah'](_array(key.data)),
            localDeltaSha256=skin['_ah'](delta),actualInactiveGeometryDifferenceMeters=0.,actualInactiveFibreValidation=neutral_binding,
            existingKeysPreserved=list(prior_keys['blocks']),savedBlend=False,
            doNotDoubleMouth='Alternative to R11 gentle smile, whose value must be zero. No driver silently changes other controls.',
            blinkCompatibility='Separate blink author may read lowerLidComponent / eyeContactPatchFullSmileDelta. No blink corrective is installed or approved by this module.')
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def measure_strength(strength):
    """Actual temporary evaluated eye/lip and fibre evidence; no render/save."""
    import bpy
    if not bpy.app.background or not 0.<=strength<=1.:raise ValueError('Bounded explicit background strength required')
    if MARKER not in bpy.context.scene:raise RuntimeError('Prepare the independent natural-smile key first')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if abs(keys[MOUTH_KEY].value)>1e-8:raise RuntimeError('Do not double the shared mouth component')
    if any(abs(k.value)>1e-7 for k in keys if k.name.startswith('R11 blink')):raise RuntimeError('This open-eye measurement is not the separate combined-blink audit')
    key=keys[KEY];previous=key.value;qa=_api('face_contact_audit_r11');uv=_api('production_uv_r08');binding=_api('face_fibre_binding_r11')
    def lip_summary():
        data=qa['_evaluated_mesh'](head);T=qa['_matrix'](head);inverse=np.linalg.inv(T)
        mix_coordinates,mix_metadata=uv['current_mix'](head)
        body=qa['_transform'](data['world'],inverse);cage=qa['_transform'](qa['_transform'](mix_coordinates,data['matrix']),inverse)
        return qa['lip_section_contacts'](body,np.asarray(data['faces'],np.int32),data['groups'],cage,T)
    result=None
    try:
        with qa['rest_evaluation']():
            key.value=0.;bpy.context.view_layer.update()
            before=_api('adult_upper_lid_r11')['actual_visible_summary']();lip_before=lip_summary()
            key.value=float(strength);bpy.context.view_layer.update()
            after=_api('adult_upper_lid_r11')['actual_visible_summary']();lip_after=lip_summary();movements={}
            for side in ('L','R'):
                lower=after[side][2]['lower']-before[side][2]['lower']
                upper=max(abs(a['upper']-b['upper']) for a,b in zip(after[side],before[side]))
                if strength>0 and (not .000015*strength<lower<.0007*strength or upper>.00015):
                    raise RuntimeError('Actual coordinated-eye movement differs from bounded lower-lid intent: '+json.dumps({'side':side,'lowerRise':lower,'upperChange':upper,'strength':strength}))
                movements[side]={'centralLowerLidRiseMeters':lower,'maximumUpperVisibleChangeMeters':upper}
            baseline_gap=max(r['minimumSurfaceGapMeters'] for r in lip_before['sections'])
            new_gap=max(r['minimumSurfaceGapMeters'] for r in lip_after['sections'])
            if new_gap>baseline_gap+.00025:raise RuntimeError('Light-smile lip-body section gap increased excessively: '+json.dumps({'baselineMax':baseline_gap,'candidateMax':new_gap}))
            result={'source':str(Path(bpy.data.filepath).resolve().relative_to(project_path())),'sourceSha256':_sha(Path(bpy.data.filepath)),
                'key':KEY,'temporaryStrength':strength,'actualEyeMovements':movements,'actualVisibleBefore':before,'actualVisibleAfter':after,
                'actualLipSectionsBefore':lip_before,'actualLipSectionsAfter':lip_after,
                'actualFibreBindingValidation':binding['validate'](neutral=False),
                'appearanceAccepted':False,'blinkCombinationValidated':False,'savedBlend':False,
                'limits':'Front-visible aperture and sagittal lip-body sections do not prove complete eye/oral collision freedom. Actual front/quarter/side and combined blink are still required.'}
    finally:
        key.value=previous;bpy.context.view_layer.update()
    return result


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['newKeyCoordinatesSha256']:raise RuntimeError('Retain edited natural-smile key before removing it')
    if any(c.data_path.find(KEY)>=0 for c in getattr(head.data.shape_keys.animation_data,'drivers',[]) if head.data.shape_keys.animation_data):raise RuntimeError('A later driver uses this key; preserve/deactivate that later expression system explicitly')
    head.shape_key_remove(key);head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'removedKey':KEY,'preservedOldGentleSmile':True,'sourceSaved':False}


def prepare(source_relative,output_relative,measure=False):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.natural-smile.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Retain history; choose a new output')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit independent source required')
    report=apply()
    if measure:report['actualStrengthMeasurements']=[measure_strength(v) for v in (.5,1.)]
    bpy.context.scene[MARKER]=json.dumps(report)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),candidateSaved=True)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NATURAL_SMILE_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256'],'keyValue':0.}),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--measure',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.measure)
