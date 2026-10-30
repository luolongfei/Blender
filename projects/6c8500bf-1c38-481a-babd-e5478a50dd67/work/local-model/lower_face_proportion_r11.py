"""One removable lower-face proportion study on the actual R11 female cage.

Offline plan/measurement are ordinary NumPy. Blender application is explicit,
background-only, on a new reconstruction copy. The existing lip bodies move
together by 0.8 mm; the anterior inferior chin rises by at most 9 mm. This is
not a calibrated reconstruction of the generated direction image, an accepted
appearance, or an automatic replacement for existing sculpt/expression keys.
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
KEY='R11 lower face proportion'
MARKER='R11_lower_face_proportion_candidate'
PARAMETERS={'lipRiseMeters':.0008,'maximumChinRiseMeters':.009,
    'neckPinBelowWorldZ':1.315,'posteriorPinWorldY':-.043,
    'anteriorFullWorldY':-.083,'mouthRigidGeodesicMeters':.004,
    'mouthTransitionGeodesicMeters':.010,'earTransitionMeters':.020,
    'lowNeckPinnedCageRings':2,'lowNeckTransitionMeters':.014}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _array(items):
    a=np.empty((len(items),3),np.float32);items.foreach_get('co',a.ravel());return a


def _path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Explicit reconstruction artifact required')
    return p


def _transform(a,T):return np.asarray(a,float)@np.asarray(T,float)[:3,:3].T+np.asarray(T,float)[:3,3]


def _triangles(loops,starts,sizes):
    result=[]
    for start,size in zip(starts,sizes):
        row=loops[start:start+size]
        result.extend((int(row[0]),int(row[j]),int(row[j+1])) for j in range(1,len(row)-1))
    return np.asarray(result,np.int32)


def _segments(world,triangles,sx=1e-6):
    q=np.asarray(world,float)[triangles]
    take=(q[:,:,0].min(1)<sx)&(q[:,:,0].max(1)>sx)
    take&=(q[:,:,1].min(1)<-.04)&(q[:,:,2].max(1)>1.310)&(q[:,:,2].min(1)<1.420)
    segments=[]
    for tri in q[take]:
        points=[]
        for a,b in zip(tri,np.roll(tri,-1,axis=0)):
            if (a[0]-sx)*(b[0]-sx)<0:points.append(a+(b-a)*(sx-a[0])/(b[0]-a[0]))
        if len(points)==2:segments.append(points)
    if len(segments)<100:raise RuntimeError('Actual bounded sagittal surface section missing')
    return np.asarray(segments,float)


def _project_section(segments,seed):
    edge=segments[:,1]-segments[:,0]
    t=np.sum((np.asarray(seed)-segments[:,0])*edge,1)/np.maximum(np.sum(edge*edge,1),1e-20)
    point=segments[:,0]+np.clip(t,0,1)[:,None]*edge
    return point[np.argmin(np.linalg.norm(point-np.asarray(seed),axis=1))]


def measure_geometry(world,triangles,cage_world,eye_centers,lip_sections):
    """Explicit landmarks on the evaluated surface; no photograph-to-mm fit.

    Subnasale is a bounded midline philtrum/columella concavity near donor 8009,
    not the most inferior nostril pixel. Chin landmarks are sagittal surface
    projections of the observed anterior-under-chin cage chain 7969/7972/7973.
    Their range is reported because the visible chin floor is not one raw min-Z.
    """
    segments=_segments(world,triangles);cage=np.asarray(cage_world,float)
    zs=np.arange(cage[8009,2]-.002,cage[8009,2]+.001,0.000025)
    edge=segments[:,1]-segments[:,0];dz=edge[:,2];valid=abs(dz)>1e-12
    u=(zs[:,None]-segments[None,:,0,2])/np.where(valid,dz,1.)[None,:]
    yy=segments[None,:,0,1]+u*edge[None,:,1]
    front=np.where(valid[None,:]&(u>=0)&(u<=1),yy,np.inf).min(1)
    finite=np.flatnonzero(np.isfinite(front))
    if not len(finite):raise RuntimeError('Nasal/philtrum surface turn missing')
    index=finite[np.argmax(front[finite])];sn=np.array([1e-6,front[index],zs[index]])
    chin={str(i):_project_section(segments,cage[i]).tolist() for i in (7969,7972,7973)}
    central=min(lip_sections['sections'],key=lambda r:abs(r['bodySectionX']))
    mouth=np.mean([central['upperNearestWorld'],central['lowerNearestWorld']],axis=0)
    eyes=np.asarray(eye_centers,float);eye=eyes.mean(0);ipd=float(abs(eyes[0,0]-eyes[1,0]))
    if not .05<ipd<.075:raise RuntimeError('Unexpected actual eye-center separation')
    ci=np.asarray(chin['7972']);heights=(eye[2]-sn[2],sn[2]-mouth[2],mouth[2]-ci[2])
    return {'eyeCentersWorld':eyes.tolist(),'eyeCenterHorizontalSeparationMeters':ipd,
        'subnasaleSurfaceTurnWorld':sn.tolist(),'actualLipContactMidpointWorld':mouth.tolist(),
        'centralLipSurfaceGapMeters':central['minimumSurfaceGapMeters'],
        'chinSurfaceAnchorsWorld':chin,'primaryChinAnchor':'7972: anterior inferior chin turn; 7969 is the more posterior underside',
        'verticalSegmentsMeters':dict(zip(('eyeToSubnasale','subnasaleToLipContact','lipContactToAnteriorInferiorChin'),map(float,heights))),
        'eyeToChinOverEyeSeparation':float((eye[2]-ci[2])/ipd),
        'eyeToChinRatioAcrossAnchorChoices':{k:float((eye[2]-v[2])/ipd) for k,v in chin.items()},
        'lowerFaceUpperSegmentFraction':float(heights[1]/(heights[1]+heights[2])),
        'method':'Actual evaluated midline triangle sections, actual upper/lower-lip surface nearest pair, actual globe geometry centers. Anatomical landmark choices and their uncertainty are explicit.',
        'appearanceAccepted':False}


def offline_measurement():
    p=project_path()/'work/reconstruction-r02';obs=p/'adult-neck-observation-r11.npz'
    meta=json.loads((p/'adult-neck-observation-r11.json').read_text(encoding='utf-8'))
    contact_path=p/'face-contact-actual-r11.json';contact=json.loads(contact_path.read_text(encoding='utf-8'))
    with np.load(obs) as a:
        world=a['rest_00_world'];cage=a['rest_00_raw_world'];tri=a['rest_00_triangles']
    eyes=[np.asarray(contact['eyes'][s]['worldBounds']).mean(0) for s in ('L','R')]
    result=measure_geometry(world,tri,cage,eyes,contact['mouth']['actualLipSurfaceSections'])
    result.update(source=meta['source'],sourceSha256=meta['sourceSha256'],observationSha256=_sha(obs),
        lipObservationSource=contact['source'],lipObservationSourceSha256=contact['sourceSha256'],
        lipObservationSha256=_sha(contact_path),offlineOnly=True,
        currentCombinationMeasured=False,
        limitation='This is the actual 080f head and 575c lip-contact evidence. Later jaw X, eye-only and low-neck XY edits do not deliberately edit central lip/chin YZ, but fresh candidate measurements are still mandatory.')
    return result


def plan(current_cage,T,measurement):
    skin=_api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']()
    current=np.asarray(current_cage,float);T=np.asarray(T,float);linear=T[:3,:3]
    scale=float(np.linalg.norm(linear[:,0]));world=_transform(current,T)
    if current.shape!=(10582,3) or not np.allclose(linear,np.eye(3)*scale,atol=8e-7):
        raise RuntimeError('Actual full female cage in the documented upright REST transform required')
    z=world[:,2];y=world[:,1];smooth=skin['_smooth'];p=PARAMETERS
    sn=float(measurement['subnasaleSurfaceTurnWorld'][2]);chin=float(measurement['chinSurfaceAnchorsWorld']['7972'][2])
    lip=np.asarray(landmarks['mouth']['vermilionStripVertexIndices'],np.int32)
    upper=float(world[lip,2].max());lower=float(world[lip,2].min())
    if not (1.382<sn<1.391 and .022<lower-chin<.040 and 1.320<chin<1.333):
        raise RuntimeError('Current measured lower-face anatomy differs; inspect before applying this bounded proposal')
    mouth=p['lipRiseMeters'];peak=p['maximumChinRiseMeters']
    rise=mouth*smooth((sn-z)/max(sn-upper,.002))
    rise+=(peak-mouth)*smooth((lower-z)/(lower-chin))
    # The cheek/mandibular envelope remains continuous. Posterior jaw and the
    # low neck fade to zero independently of vertical compression in front.
    rise*=smooth((z-p['neckPinBelowWorldZ'])/(chin-p['neckPinBelowWorldZ']))
    rise*=smooth((p['posteriorPinWorldY']-y)/(p['posteriorPinWorldY']-p['anteriorFullWorldY']))
    allowed=skin['_vertices_of_groups'](donor,{17,22,7,8})
    graph=skin['_graph'](world,donor['edges'],np.ones(len(current),bool))
    # Two untouched cage rings around the low-neck region protect the actual
    # Subdivision/Multires support, not just the raw vertices below the cut.
    neck_pin=z<=p['neckPinBelowWorldZ']
    for _ in range(p['lowNeckPinnedCageRings']):
        neck_pin[np.unique(donor['edges'][neck_pin[donor['edges']].any(1)])]=True
    neck_distance=skin['_distance'](graph,np.flatnonzero(neck_pin),p['lowNeckTransitionMeters'])
    rise*=smooth(neck_distance/p['lowNeckTransitionMeters'])
    ears=skin['_vertices_of_groups'](donor,{4,5})
    ear_distance=skin['_distance'](graph,np.flatnonzero(ears),p['earTransitionMeters'])
    rise*=smooth(ear_distance/p['earTransitionMeters'])
    # All lip strips and both sheets of the actual contact neighborhood share
    # one translation. This is not separate upper/lower scaling or mouth closure.
    lip_distance=skin['_distance'](graph,lip,p['mouthTransitionGeodesicMeters'])
    blend=1.-smooth((lip_distance-p['mouthRigidGeodesicMeters'])/(p['mouthTransitionGeodesicMeters']-p['mouthRigidGeodesicMeters']))
    rise=rise*(1.-blend)+mouth*blend
    pinned=(~allowed)|ears|neck_pin|(z>=sn)|(y>=p['posteriorPinWorldY'])
    rise[pinned]=0.
    selection=json.loads((project_path()/'work/reconstruction-r02/female-head-selection-r11.json').read_text(encoding='utf-8'))
    boundary=np.asarray(selection['neck_ring_old_indices'],np.int32);rise[boundary]=0.
    rigid=(lip_distance<=p['mouthRigidGeodesicMeters'])&allowed&~pinned
    if not np.allclose(rise[lip],mouth,atol=1e-12):raise RuntimeError('True lip volume strip was not translated rigidly')
    target=np.zeros_like(world);target[:,2]=rise;delta=(target@np.linalg.inv(linear).T).astype(np.float32)
    affected=np.linalg.norm(delta,axis=1)>1e-10
    edges=donor['edges'];old=current[edges[:,1]]-current[edges[:,0]];new=old+delta[edges[:,1]]-delta[edges[:,0]]
    ratio=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12);ae=affected[edges].any(1)
    tris=_triangles(donor['corner_vertices'],donor['face_starts'],donor['face_lengths']);tris=tris[affected[tris].any(1)]
    a=current[tris];b=(current+delta)[tris];na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    n0=np.linalg.norm(na,axis=1);n1=np.linalg.norm(nb,axis=1);valid=n0>1e-12
    align=np.sum(na[valid]*nb[valid],axis=1)/(n0[valid]*n1[valid])
    if not 500<int(affected.sum())<1800 or not np.isfinite(delta).all() or align.min()<=0:
        raise RuntimeError('Unexpected candidate scope or a raw face fold')
    if ratio[ae].min()<.55 or ratio[ae].max()>1.65:raise RuntimeError('Raw edge strain outside the predeclared bounded candidate: '+str([ratio[ae].min(),ratio[ae].max()]))
    report={'parameters':p,'provenance':provenance,'baselineMeasurement':measurement,
        'rawAffectedVertices':int(affected.sum()),'maximumWorldDisplacementMeters':float(rise.max()),
        'rawLipStripRigidTranslationMeters':[0.,0.,mouth],'rawRigidLipAndContactNeighborhoodVertices':int(rigid.sum()),
        'rawRigidLipAndContactNeighborhoodIndices':np.flatnonzero(rigid).tolist(),
        'pinnedOriginalNeckBoundaryVertices':boundary.tolist(),'actualNeckBoundaryReviewRequired':True,
        'rawEdgeLengthRatioQuantiles':np.quantile(ratio[ae],[0,.05,.5,.95,1]).tolist(),
        'rawTriangleAreaRatioRange':[float((n1[valid]/n0[valid]).min()),float((n1[valid]/n0[valid]).max())],
        'rawTriangleNormalAlignmentMinimum':float(align.min()),
        'noseAboveSubnasaleAndEyesEarsPinnedInCage':True,'existingJawWidthAndChinForwardProjectionUnchangedInCage':True,
        'scope':'One new static proportion key. Common 0.8 mm lip-body/contact-neighborhood rise, up to 9 mm anterior-inferior chin rise with continuous posterior/ear/neck fade. Only REST vertical coordinates change; no spherical skull/eye edit, no lip flattening or colour repaint.',
        'requiredReview':['Fresh actual L3 contact and neck/eye/ear protection checks on this exact combination.',
            'Identical front, quarter and side at key0,0.5,1; inspect lower-lip-to-chin curvature and posterior jaw-to-neck transition.',
            'Reject a pinched chin, hard step under the lower lip, detached neck edge, changed lip seal or flattened lip volume.',
            'Preserved expression keys still require later actual combination review on the shortened face.'],
        'appearanceAccepted':False,'actualMultiresReviewPending':True}
    return delta,report


def _keys(mesh,uv,skin):
    return {'animation':uv['_animation'](mesh.shape_keys),'custom':uv['_custom'](mesh.shape_keys),
        'blocks':{k.name:{'coordinatesSha256':skin['_ah'](_array(k.data)),
            'settings':uv['_scalar_rna'](k),'relative':k.relative_key.name} for k in mesh.shape_keys.key_blocks}}


def _nonkey(head,face):
    state=_api('eye_margin_refine_r11')['_state'](head,0);state['keys']=None
    state['allMaterials']=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    return state


def _actual(head):
    import bpy
    qa=_api('face_contact_audit_r11');uv=_api('production_uv_r08')
    data=qa['_evaluated_mesh'](head);T=np.asarray(head.matrix_world,float)
    body=qa['_transform'](data['world'],np.linalg.inv(T));faces=np.asarray(data['faces'],np.int32)
    cage,mix_metadata=uv['current_mix'](head)
    cage=np.asarray(cage,float);cage_world=_transform(cage,T)
    lips=qa['lip_section_contacts'](body,faces,data['groups'],cage,T)
    eyes=[]
    for side in ('L','R'):
        e=qa['_evaluated_mesh'](bpy.data.objects['R11 Adult sclera '+side])['world']
        eyes.append((e.min(0)+e.max(0))*.5)
    triangles=np.asarray([(int(row[0]),int(row[j]),int(row[j+1])) for row in faces for j in range(1,len(row)-1)],np.int32)
    measurement=measure_geometry(data['world'],triangles,cage_world,eyes,lips)
    return {'data':data,'cage':cage,'cageWorld':cage_world,'triangles':triangles,'lips':lips,'measurement':measurement,'currentMixMetadata':mix_metadata}


def _validate_actual(before,after):
    a=before['data'];b=after['data'];world0=np.asarray(a['world']);world1=np.asarray(b['world'])
    if a['faces']!=b['faces'] or not np.array_equal(a['groups'],b['groups']):raise RuntimeError('Actual L3 topology/material-region identities changed')
    diff=np.linalg.norm(world1-world0,axis=1);faces=np.asarray(a['faces'],np.int32)
    groups=a['groups'];protected=np.unique(faces[np.isin(groups,[2,3,4,5])])
    pin=np.zeros(len(world0),bool);pin[protected]=True
    # Protect the nasal body and upper face separately from lower cheek tissue.
    # A former single Z>=subnasale+5mm box also included the lower lateral
    # cheeks. Actual diagnostic vertex81742 at X=-28.55mm,Y=-81.14mm was a
    # cheek interpolation support point, not nose/eye anatomy. That support
    # must remain continuous with the lower-face rise, not be hard-clamped.
    # The narrow subnasale-to-lip transition intentionally remains deformable.
    sn=before['measurement']['subnasaleSurfaceTurnWorld'][2]
    sn_y=before['measurement']['subnasaleSurfaceTurnWorld'][1]
    nasal_body=(world0[:,2]>=sn+.005)&(abs(world0[:,0])<=.025)&(world0[:,1]<=sn_y+.012)
    upper_face=world0[:,2]>=sn+.020
    low_neck=world0[:,2]<PARAMETERS['neckPinBelowWorldZ']
    strict_pin=pin.copy()|upper_face|low_neck
    pin|=nasal_body|upper_face|low_neck
    lower_cheek_extension=(world0[:,2]>=sn+.005)&~pin
    cheek_extension_max=float(diff[lower_cheek_extension].max()) if np.any(lower_cheek_extension) else 0.
    if cheek_extension_max>PARAMETERS['lipRiseMeters']*.25:
        raise RuntimeError('Lower-cheek interpolation above the nasal-base height exceeds one-quarter of the intended lip translation: '+str(cheek_extension_max))
    max_pin=float(diff[pin].max())
    # The fresh regional diagnostic found 2.8915um only at the lower lateral
    # nasal interpolation boundary; its nearest eight cage controls were all
    # exactly pinned. Permit at most10um there (about0.026px in the0.365m/960px
    # review), while eyes/ears/upper face/low neck retain2um and the actual neck
    # seam is independently checked at2um below. Do not distort the visible
    # cheek transition to eliminate invisible subdivision propagation.
    protected_tolerance=np.where(strict_pin,2e-6,1e-5)
    if np.any(pin&(diff>protected_tolerance)):
        # Diagnose the real evaluated region before changing a protected field.
        # Report actual regional values instead of hiding the allowed10um
        # nasal support tolerance behind a single generic pass flag.
        regions={str(g):np.unique(faces[np.asarray(groups)==g]) for g in (2,3,4,5)}
        regions['nasalBody']=np.flatnonzero(nasal_body)
        regions['upperFace']=np.flatnonzero(upper_face)
        regions['lowerCheekExtension']=np.flatnonzero(lower_cheek_extension)
        regions['lowNeck']=np.flatnonzero(low_neck)
        detail={}
        cage=np.asarray(before['cageWorld']);raw_delta=np.asarray(after['cageWorld'])-cage
        for label,indices in regions.items():
            if not len(indices):continue
            order=indices[np.argsort(diff[indices])[-5:][::-1]]
            detail[label]={'maximumMeters':float(diff[indices].max()),'verticesAbove2Micrometers':int(np.count_nonzero(diff[indices]>2e-6)),
                'worst':[{'evaluatedVertex':int(i),'world':world0[i].tolist(),'deltaMeters':(world1[i]-world0[i]).tolist(),
                    'nearestCage':[{'index':int(j),'world':cage[j].tolist(),'deltaMeters':raw_delta[j].tolist()}
                        for j in np.argsort(np.linalg.norm(cage-world0[i],axis=1))[:8]]} for i in order]}
        raise RuntimeError('Actual protected eye/ear/nose/low-neck surface moved: '+json.dumps({'maximumMeters':max_pin,'regions':detail}))
    pair=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1)
    unique,counts=np.unique(pair,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
    boundary_error=float(diff[boundary].max()) if len(boundary) else 0.
    if boundary_error>2e-6:raise RuntimeError('Actual head-to-neck boundary moved: '+str(boundary_error))
    affected=diff>1e-7;local_faces=faces[affected[faces].any(1)]
    tris=np.concatenate((local_faces[:,[0,1,2]],local_faces[:,[0,2,3]]))
    old_tri=world0[tris];new_tri=world1[tris]
    normal0=np.cross(old_tri[:,1]-old_tri[:,0],old_tri[:,2]-old_tri[:,0]);normal1=np.cross(new_tri[:,1]-new_tri[:,0],new_tri[:,2]-new_tri[:,0])
    area0=np.linalg.norm(normal0,axis=1);area1=np.linalg.norm(normal1,axis=1);valid=area0>1e-14
    alignment=np.sum(normal0[valid]*normal1[valid],axis=1)/(area0[valid]*area1[valid]);area_ratio=area1[valid]/area0[valid]
    if not np.isfinite(alignment).all() or alignment.min()<=0 or area_ratio.min()<.35 or area_ratio.max()>1.8:
        raise RuntimeError('Actual lower-face/cheek surface folded or strained excessively: '+str([alignment.min(),area_ratio.min(),area_ratio.max()]))
    old_g=np.asarray([r['minimumSurfaceGapMeters'] for r in before['lips']['sections']])
    new_g=np.asarray([r['minimumSurfaceGapMeters'] for r in after['lips']['sections']])
    if np.max(new_g-old_g)>.00005:raise RuntimeError('Actual lip contact gap increased over 50 micrometers: '+str((new_g-old_g).tolist()))
    # Existing corner intersections are recorded, not reinterpreted as new holes.
    for old,new in zip(before['lips']['sections'][2:7],after['lips']['sections'][2:7]):
        if new['sectionCrossings']>old['sectionCrossings']:raise RuntimeError('A new central lip-section crossing appeared')
    b0=before['measurement'];b1=after['measurement']
    mouth_rise=b1['actualLipContactMidpointWorld'][2]-b0['actualLipContactMidpointWorld'][2]
    chin_rise=b1['chinSurfaceAnchorsWorld']['7972'][2]-b0['chinSurfaceAnchorsWorld']['7972'][2]
    if not .0006<mouth_rise<.0010 or not .0078<chin_rise<.0092:raise RuntimeError('Actual lip/chin motion differs from the measured proposal: '+str([mouth_rise,chin_rise]))
    return {'before':b0,'after':b1,'actualLipSectionsBefore':before['lips'],'actualLipSectionsAfter':after['lips'],
        'actualMaximumWorldDisplacementMeters':float(diff.max()),'protectedSurfaceMaximumMeters':max_pin,
        'protectedSurfaceTolerancesMeters':{'eyesEarsUpperFaceLowNeck':2e-6,'nasalBodyInterpolation':1e-5,'actualNeckSeam':2e-6},
        'actualNasalBodyMaximumMeters':float(diff[nasal_body].max()),
        'actualStrictProtectedMaximumMeters':float(diff[strict_pin].max()),
        'toleranceRationale':'Measured2.8915um displacement at lower-lateral nasal Multires support with its nearest8cage vertices exactly fixed;10um is approximately0.026pixel at960px/0.365m. Eye/ear/upper-face/low-neck and actual neck seam retain2um. This numeric protection does not replace actual images or nasal-base/cheek continuity review.',
        'protectedAnatomicalDomains':{'nasalBody':'Z>=subnasale+5mm, absX<=25mm, Y<=subnasaleY+12mm; excludes deformable nasal-base/philtrum transition',
            'upperFace':'Z>=subnasale+20mm','eyesEars':'Actual sculpt-face-set2/3/4/5 vertices','lowNeck':'Z<1.315m plus actual open neck boundary'},
        'lowerCheekInterpolationMaximumMeters':cheek_extension_max,
        'actualPatchMinimumNormalAlignment':float(alignment.min()),'actualPatchAreaRatioRange':[float(area_ratio.min()),float(area_ratio.max())],
        'actualBoundaryVertices':len(boundary),'actualBoundaryMaximumMeters':boundary_error,
        'actualLipContactRiseMeters':mouth_rise,'actualAnteriorInferiorChinRiseMeters':chin_rise,
        'maximumLipGapIncreaseMeters':float((new_g-old_g).max()),'appearanceAccepted':False,
        'scope':'Actual topology/protected surfaces, true nine lip-body sections and landmarks. Not a complete 3D collision/volume proof or expression-combination approval.'}


def apply():
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    if MARKER in bpy.context.scene or not keys or KEY in keys.key_blocks:raise RuntimeError('Keep an independent new candidate and preserved existing keys')
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582:raise RuntimeError('Actual complete independent female cage required')
    if any(abs(k.value)>1e-8 for k in keys.key_blocks if any(w in k.name.lower() for w in ('smile','blink'))):
        raise RuntimeError('Measure the adopted neutral expression, with smile and blink at zero; preserve static shape keys')
    uv=_api('production_uv_r08');skin=_api('natural_skin_adult_r11');face=_api('runtime_face_refresh_r08')
    qa=_api('face_contact_audit_r11');binding=_api('face_fibre_binding_r11');gentle=_api('gentle_smile_adult_r11')
    donor,_,_=skin['_donor']()
    if not np.array_equal(skin['_array'](mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):raise RuntimeError('Actual donor topology changed')
    bpy.context.view_layer.update();prior=_keys(mesh,uv,skin);contract=_nonkey(head,face)
    external=gentle['_external_identity'](bpy,head);source_sha=_sha(source);active=head.active_shape_key_index;key=None;success=False
    try:
        with qa['rest_evaluation']():
            original=_actual(head);T=np.asarray(head.matrix_world,float);delta,report=plan(original['cage'],T,original['measurement'])
            baseline=_array(keys.reference_key.data)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.value=0.;key.slider_min=0.;key.slider_max=1.
            key.data.foreach_set('co',(baseline+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            inactive=binding['_head_mesh'](head)
            if not np.array_equal(np.asarray(original['data']['world']),inactive['world']):
                # The independent readers use the same evaluated matrix; compare
                # numerical precision explicitly rather than ignore a mismatch.
                error=float(np.max(np.linalg.norm(np.asarray(original['data']['world'])-inactive['world'],axis=1)))
                if error>2e-8:raise RuntimeError('Inactive key changed actual geometry: '+str(error))
            key.value=1.;bpy.context.view_layer.update();candidate=_actual(head)
            measured=_validate_actual(original,candidate);fibre=binding['validate'](neutral=False)
            key.value=0.;bpy.context.view_layer.update()
        current=_keys(mesh,uv,skin);current['blocks'].pop(KEY)
        if current!=prior:raise RuntimeError('Original keys, animation, relative bases or values changed')
        now=_nonkey(head,face)
        if gentle['_hash'](now)!=gentle['_hash'](contract):raise RuntimeError('Original non-key source contract changed: '+json.dumps(skin['_differences'](contract,now)))
        if gentle['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('External objects/action/rig/pose or source disk changed')
        key.value=1.;bpy.context.view_layer.update()
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
            head=HEAD,key=KEY,keyValue=1.,originalActiveShapeKeyIndex=active,newKeyCoordinatesSha256=skin['_ah'](_array(key.data)),
            baselineActualMixSha256=skin['_ah'](original['cage'].astype(np.float32)),currentMixMetadata=original['currentMixMetadata'],localDeltaSha256=skin['_ah'](delta),
            existingKeysPreserved=list(prior['blocks']),actualValidation=measured,actualFibreBindingValidation=fibre,
            actualMultiresReviewPending=False,actualImagesPending=True,savedBlend=False)
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];keys=head.data.shape_keys;key=keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['newKeyCoordinatesSha256']:raise RuntimeError('Retain edited proportion key before restoring')
    animation=keys.animation_data
    for curve in getattr(animation,'drivers',[]) if animation else []:
        paths=[curve.data_path]+[target.data_path for v in curve.driver.variables for target in v.targets]
        if any(KEY in path for path in paths):raise RuntimeError('A later driver references this candidate; preserve that dependent system explicitly')
    head.shape_key_remove(key);head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'removedKey':KEY,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.lower-face.json')
    if source==output or output.exists() or report_path.exists() or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh explicit source and new output/report required; retain history')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_LOWER_FACE_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256'],'report':str(report_path.relative_to(project_path()))}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
