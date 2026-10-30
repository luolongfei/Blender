"""Bounded hand candidate from the Human Base Meshes 1.4.1 evaluated mesh.

plan() is offline NumPy only. run() is an explicitly invoked disposable Blender
candidate operation, never a launcher or master installer. The actual level-one
Multires evaluated asset is fitted; no procedural hand is reconstructed.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
OBS='work/reconstruction-r02/hand-observation-r10.json'
CHAINS={'Index':(19,16,3),'Middle':(18,15,4),'Ring':(17,14,5),
        'Little':(13,12,11),'Thumb':(23,22,21)}
PLAN='work/reconstruction-r02/hand-asset-fit-plan-r10-v3.json'


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(obs,npz,row,key):return np.asarray(npz[row['arrays'][key]['array']])


def _faces(row,npz):
    loops=npz[row['arrays']['loopVertex']['array']]
    return [loops[a:a+b].copy() for a,b in zip(npz[row['arrays']['polygonStart']['array']],
                                             npz[row['arrays']['polygonSize']['array']])]


def _groups(row,npz):
    return np.asarray(npz[next(a for a in row['attributes'] if a['name']=='.sculpt_face_set')['data']['array']])


def _unit(v):return v/max(np.linalg.norm(v),1e-12)


def _cross_section(v,edges,z):
    a=v[edges[:,0]];b=v[edges[:,1]]
    mask=(a[:,2]-z)*(b[:,2]-z)<0
    a=a[mask];b=b[mask]
    return a+(b-a)*((z-a[:,2])/(b[:,2]-a[:,2]))[:,None]


def _rbf(source,target):
    center=source.mean(0);unit=.1;q=(source-center)/unit
    distance=np.linalg.norm(q[:,None]-q[None,:],axis=2)
    poly=np.column_stack([np.ones(len(q)),q])
    system=np.block([[distance**3,poly],[poly.T,np.zeros((4,4))]])
    coefficients=np.linalg.solve(system,np.vstack([target,np.zeros((4,3))]))
    def apply(points):
        x=(np.asarray(points)-center)/unit
        return np.linalg.norm(x[:,None]-q[None,:],axis=2)**3@coefficients[:-4]+np.column_stack([np.ones(len(x)),x])@coefficients[-4:]
    return apply,float(np.linalg.cond(system))


def _fit(obs,npz):
    asset=obs['asset'][0]
    if asset['name']!='Hand  - Realistic' or asset['type']!='MESH':raise RuntimeError('Exact observed asset identity changed')
    raw=asset['raw'];ev=asset['evaluated'];v=np.asarray(npz[raw['arrays']['positions']['array']],np.float64)
    dense=np.asarray(npz[ev['arrays']['positions']['array']],np.float64)
    faces=_faces(raw,npz);groups=_groups(raw,npz)
    ids={int(g):np.unique(np.concatenate([f for f,k in zip(faces,groups) if k==g])) for g in np.unique(groups)}
    rig=next(o for o in obs['current'] if o['name']=='Character_Rig')
    bones={b['name']:b for b in rig['bones']}
    anchors=[];targets=[];labels=[];source_joints={};target_joints={}
    def add(p,t,label):anchors.append(np.asarray(p));targets.append(np.asarray(t));labels.append(label)
    for finger,chain in CHAINS.items():
        joints=[]
        for before,after in zip((24,)+chain,chain):
            shared=np.intersect1d(ids[before],ids[after])
            if len(shared)<6:raise RuntimeError('Observed face-group joint ring incomplete')
            joints.append(v[shared].mean(0))
        axis=_unit(joints[2]-joints[1]);distal=v[ids[chain[-1]]]
        tip=joints[2]+axis*((distal-joints[2])@axis).max()
        joints.append(tip-axis*.0008)
        suffix=('Metacarpal','Proximal','Distal') if finger=='Thumb' else ('Proximal','Intermediate','Distal')
        b=[bones['left'+finger+s] for s in suffix]
        target=[np.array(x['headLocal']) for x in b]+[np.array(b[-1]['tailLocal'])]
        source_joints[finger]=np.array(joints);target_joints[finger]=np.array(target)
        for i,(p,t) in enumerate(zip(joints,target)):add(p,t,finger+' joint '+str(i))
    edges=npz[raw['arrays']['edges']['array']]
    section=_cross_section(v,edges,0.)
    if len(section)<12:raise RuntimeError('Observed wrist section is incomplete')
    wrist=section.mean(0);wrist[2]=0
    target_wrist=np.array(bones['leftHand']['headLocal'])
    add(wrist,target_wrist,'wrist center')
    # Exact observed wrist contour, not a generated hand or sphere. The upper
    # arm stub is compressed into the existing concealed cuff extent.
    for source_z,target_x in ((0.,float(target_wrist[0])),(.025,.602),(.070,.5876)):
        section=_cross_section(v,edges,source_z)
        middle=section.mean(0)
        for j in range(0,len(section),max(1,len(section)//12)):
            p=section[j]
            target=np.array([target_x,-.64*(p[0]-middle[0]),1.235-.64*(p[1]-middle[1])])
            add(p,target,'wrist contour '+str(source_z)+' '+str(j))
    transform,condition=_rbf(np.array(anchors),np.array(targets))
    fitted=transform(dense)
    anchor_error=float(np.max(np.linalg.norm(transform(np.array(anchors))-np.array(targets),axis=1)))
    # A center-only field must preserve surface neighborhoods; assess its real
    # local derivative everywhere before allowing this single candidate.
    epsilon=1e-5
    jac=np.stack([(transform(dense+np.eye(3)[i]*epsilon)-transform(dense-np.eye(3)[i]*epsilon))/(2*epsilon)
                  for i in range(3)],axis=2)
    determinant=np.linalg.det(jac);singular=np.linalg.svd(jac,compute_uv=False)
    stats={'rbfCondition':condition,'anchorCount':len(anchors),'anchorMaxErrorMeters':anchor_error,
           'jacobianDeterminantRange':[float(determinant.min()),float(determinant.max())],
           'localSingularValueRange':[float(singular.min()),float(singular.max())],
           'localFoldCount':int(np.count_nonzero(determinant>=0)),
           'fittedLeftBounds':[fitted.min(0).tolist(),fitted.max(0).tolist()],
           'sourceJointCenters':{k:x.tolist() for k,x in source_joints.items()},
           'targetJointCenters':{k:x.tolist() for k,x in target_joints.items()},
           'anchors':[{'label':l,'source':p.tolist(),'target':t.tolist()} for l,p,t in zip(labels,anchors,targets)]}
    return fitted,stats


def _weights(obs,npz,points,face_groups,faces):
    """Interpolate the actual existing weights; explicit finger regions cannot
    borrow another finger's bone merely because their surfaces are nearby."""
    current=obs['current'][0]
    old=npz[current['raw']['arrays']['positions']['array']]
    names=[g['name'] for g in current['vertexGroups']]
    source=np.zeros((len(old),len(names)),np.float64)
    source[npz[current['weights']['vertex']['array']],npz[current['weights']['group']['array']]]=npz[current['weights']['value']['array']]
    finger_names=list(CHAINS);membership=np.zeros((len(points),len(finger_names)),np.int32)
    for face,group in zip(faces,face_groups):
        for i,finger in enumerate(finger_names):
            if group in CHAINS[finger]:membership[face,i]+=1
    # Nail interiors have no finger-labelled adjacent face: use their actual
    # spatial nearest distal segment, then enforce that same finger's bones.
    bones={b['name']:b for b in obs['current'][-1]['bones']}
    tips=np.array([bones['left'+finger+'Distal']['tailLocal'] for finger in finger_names])
    nail_vertices=np.unique(np.concatenate([f for f,g in zip(faces,face_groups) if g==10]))
    for index in nail_vertices:
        if not membership[index].any():membership[index,np.argmin(np.linalg.norm(tips-points[index],axis=1))]=1
    result=np.zeros((len(points),len(names)),np.float64);nearest=[]
    for start in range(0,len(points),128):
        p=points[start:start+128]
        distance=np.sum((p[:,None,:]-old[None,:,:])**2,axis=2)
        indices=np.argpartition(distance,7,axis=1)[:,:8]
        d=np.take_along_axis(distance,indices,axis=1)
        influence=1/np.maximum(d,1e-10);influence/=influence.sum(1)[:,None]
        result[start:start+len(p)]=(source[indices]*influence[:,:,None]).sum(1)
        nearest.extend(np.sqrt(d.min(1)).tolist())
    for i,votes in enumerate(membership):
        if votes.any():
            finger=finger_names[int(votes.argmax())]
            allowed=np.array([name in ('leftHand','leftLowerArm') or name.startswith('left'+finger) for name in names])
            result[i,~allowed]=0
    if np.min(result.sum(1))<.01:raise RuntimeError('Weight transfer lost a finger region')
    result/=result.sum(1)[:,None]
    return result.astype(np.float32),names,{'method':'8 nearest actual source vertices, inverse squared-distance blend, explicit digit bone restriction, normalize',
        'nearestSurfaceVertexDistanceMaxMeters':max(nearest),
        'nearestSurfaceVertexDistance95Meters':float(np.quantile(nearest,.95)),
        'weightSumMaxError':float(np.max(np.abs(result.sum(1)-1))),
        'sourceGroups':names,'note':'This is a candidate weight transfer, not evidence that flexed-finger deformation has passed.'}


def plan(report_relative=PLAN):
    output=_path(report_relative);payload=output.with_suffix('.npz')
    if output.exists() or payload.exists():raise FileExistsError('Choose a new plan path; keep historical evidence')
    obs_path=_path(OBS);obs=json.loads(obs_path.read_text(encoding='utf-8'))
    npz_path=_path(obs['arrays'])
    if _sha(npz_path)!=obs['arraysSha256']:raise RuntimeError('Observed arrays no longer match their recorded identity')
    with np.load(npz_path) as arrays:
        fitted,stats=_fit(obs,arrays)
        asset=obs['asset'][0];mesh=asset['evaluated'];groups=_groups(mesh,arrays)
        loops=arrays[mesh['arrays']['loopVertex']['array']]
        starts=arrays[mesh['arrays']['polygonStart']['array']];sizes=arrays[mesh['arrays']['polygonSize']['array']]
        faces=[loops[a:a+b] for a,b in zip(starts,sizes)]
        weights,names,weight_report=_weights(obs,arrays,fitted,groups,faces)
        uv=arrays[mesh['uvLayers'][0]['data']['array']]
        result={'schemaVersion':1,'observation':OBS,'observationSha256':_sha(obs_path),
            'observedArraysSha256':obs['arraysSha256'],'observedSourceSha256':obs['sourceSha256'],
            'assetName':asset['name'],'assetMetadata':{k:v for k,v in asset['assetMetadata'].items() if k!='description'},
            'assetDescription':'Human Base Meshes 1.4.1 actual evaluated hand mesh; no acquisition-method claim.',
            'originalAssetMetadataRecordedIn':OBS,'bundleSha256':obs['bundleSha256'],
            'assetEvaluatedVertices':len(fitted),'assetEvaluatedFaces':len(groups),
            'fitting':'One smooth 3D polyharmonic field fitted to exact sculpt-face-group joint interfaces, current rig rest joints and observed wrist contours. Existing rig and actions unchanged.',
            'diagnostics':stats,'appearanceAccepted':False,'candidateReady':False}
        result['weightTransfer']=weight_report
        result['sourceHandCoordinatesSha256']={o['name']:o['raw']['arrays']['positions']['sha256'] for o in obs['current'][:2]}
        result['wristScope']='A closed 25 mm proximal stump is retained under the sleeve; no forearm/wrist welding is claimed. The new stump extends about 11.5 mm proximally beyond the prior hand cap and needs cuff checks in actual poses.'
        # Record even rejected numerical fits for diagnosis, never silently save
        # a Blender candidate with foldovers or extreme local collapse/stretch.
        result['candidateReady']=(stats['localFoldCount']==0 and stats['anchorMaxErrorMeters']<1e-7
            and stats['localSingularValueRange'][0]>.09 and stats['localSingularValueRange'][1]<1.65)
        output.parent.mkdir(parents=True,exist_ok=True)
        with payload.open('xb') as f:np.savez_compressed(f,left=fitted.astype(np.float32),
            right=(fitted*np.array([-1,1,1])).astype(np.float32),loopVertex=loops,
            polygonStart=starts,polygonSize=sizes,faceGroup=groups,uv=uv,weights=weights)
    result['payload']=str(payload.relative_to(project_path()));result['payloadSha256']=_sha(payload)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return result


def _coordinates(mesh):
    co=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',co.ravel());return co


def _rig_contract(rig):
    return {'matrix':np.array(rig.matrix_world).tolist(),
        'bones':[{'name':b.name,'parent':b.parent.name if b.parent else None,
                  'rest':np.array(b.matrix_local).tolist(),'pose':np.array(rig.pose.bones[b.name].matrix_basis).tolist()}
                 for b in rig.data.bones],
        'action':rig.animation_data.action.as_pointer() if rig.animation_data and rig.animation_data.action else None,
        'slot':getattr(getattr(rig.animation_data,'action_slot',None),'identifier',None),
        'posePosition':rig.data.pose_position}


def _group_columns(actual,observed,plan_names,side):
    """Keep each hand's observed index order; map payload columns by bone name.

    The real right-hand group order differs from the left's. Symmetry changes
    bone names, not Blender's existing per-object vertex-group indices.
    """
    expected=[{'name':g['name'],'index':g['index'],'lockWeight':g['lockWeight']} for g in observed]
    if actual!=expected:
        raise RuntimeError('Current hand groups differ from its actual observation: '+json.dumps(
            {'side':side,'expected':expected,'actual':actual},ensure_ascii=False))
    source_names=[name.replace('left',side,1) for name in plan_names]
    actual_names=[g['name'] for g in actual]
    if len(source_names)!=len(set(source_names)) or set(actual_names)!=set(source_names):
        raise RuntimeError('Plan and observed hand bone identities differ: '+json.dumps(
            {'side':side,'planBoneNames':source_names,'actualBoneNames':actual_names},ensure_ascii=False))
    return {g['index']:source_names.index(g['name']) for g in actual}


def _source_weights_unchanged(obj,observed,arrays):
    count=len(obj.data.vertices);width=len(obj.vertex_groups)
    expected=np.zeros((count,width),np.float32);actual=np.zeros_like(expected)
    row=observed['weights']
    expected[arrays[row['vertex']['array']],arrays[row['group']['array']]]=arrays[row['value']['array']]
    for vertex in obj.data.vertices:
        for group in vertex.groups:actual[vertex.index,group.group]=group.weight
    if not np.array_equal(actual,expected):
        index=np.unravel_index(np.abs(actual-expected).argmax(),actual.shape)
        raise RuntimeError('Original source hand weights changed: '+json.dumps({'object':obj.name,
            'vertex':int(index[0]),'groupIndex':int(index[1]),'groupName':obj.vertex_groups[index[1]].name,
            'expected':float(expected[index]),'actual':float(actual[index])}))


def _installed_weights(obj,expected):
    actual=np.zeros_like(expected)
    for vertex in obj.data.vertices:
        for group in vertex.groups:actual[vertex.index,group.group]=group.weight
    error=float(np.max(np.abs(actual-expected)))
    sums=actual.sum(1);unweighted=int(np.count_nonzero(sums<.99999))
    if error>1e-7 or unweighted or np.max(sums)>1.00001:
        raise RuntimeError('Actual new mesh weights were not installed: '+json.dumps(
            {'object':obj.name,'groups':len(obj.vertex_groups),'maxWeightError':error,
             'unweightedVertices':unweighted,'weightSumRange':[float(sums.min()),float(sums.max())]}))
    return {'readBackMaxWeightError':error,'unweightedVertices':unweighted,
            'weightSumRange':[float(sums.min()),float(sums.max())]}


def _actual_armature_motion(bpy,obj,rig):
    """Compare identical point topology with the real modifier off and on.
    Restore all modifier visibility afterward; body/action/rig pose never move.
    """
    armatures=[m for m in obj.modifiers if m.type=='ARMATURE']
    if len(armatures)!=1 or armatures[0].object!=rig or not armatures[0].use_vertex_groups:
        raise RuntimeError('Expected one actual rig modifier using vertex groups: '+obj.name)
    armature=armatures[0]
    if not armature.show_viewport or not armature.show_render:raise RuntimeError('Actual hand rig modifier is disabled')
    states=[(m,m.show_viewport) for m in obj.modifiers]
    def points():
        bpy.context.view_layer.update();ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
        try:return _coordinates(mesh).astype(np.float64)
        finally:ev.to_mesh_clear()
    try:
        for modifier,_ in states:modifier.show_viewport=False
        undeformed=points()
        armature.show_viewport=True
        posed=points()
    finally:
        for modifier,enabled in states:modifier.show_viewport=enabled
        bpy.context.view_layer.update()
    raw=_coordinates(obj.data)
    if undeformed.shape!=raw.shape or posed.shape!=raw.shape:raise RuntimeError('Isolated actual Armature changed hand point topology')
    rest_error=float(np.max(np.linalg.norm(undeformed-raw,axis=1)))
    displacement=np.linalg.norm(posed-undeformed,axis=1)
    if rest_error>1e-6 or float(np.median(displacement))<.05:
        raise RuntimeError('Standing source did not actually move the newly weighted hand: '+json.dumps(
            {'object':obj.name,'rawRestError':rest_error,'medianArmatureDisplacementMeters':float(np.median(displacement)),
             'displacementRange':[float(displacement.min()),float(displacement.max())]}))
    return {'modifier':armature.name,'rig':rig.name,'vertexCount':len(raw),
        'realModifierDisabledVsEnabled':True,'rawRestErrorMeters':rest_error,
        'displacementRangeMeters':[float(displacement.min()),float(displacement.max())],
        'medianDisplacementMeters':float(np.median(displacement)),
        'scope':'Actual matched standing frame with the hand Armature toggled only in disposable memory; all modifier visibility restored.'}


def run(source_relative,source_sha256,plan_relative=PLAN,
        output_relative='work/reconstruction-r02/reference-asset-hands-r10.blend'):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report=output.with_suffix('.hands.json')
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Use a disposable background Object Mode source')
    if Path(bpy.data.filepath).resolve()!=source or _sha(source)!=source_sha256:raise RuntimeError('Explicit source identity mismatch')
    if output==source or output==_path('work/character.blend') or not output.is_relative_to(_path('work/reconstruction-r02')):
        raise RuntimeError('Only save a new independent reconstruction candidate')
    if output.exists() or report.exists():raise FileExistsError('Preserve existing candidate')
    plan_path=_path(plan_relative);recipe=json.loads(plan_path.read_text(encoding='utf-8'))
    if not recipe['candidateReady']:raise RuntimeError('Offline fit did not satisfy its bounded geometry checks')
    if _sha(_path(recipe['payload']))!=recipe['payloadSha256']:raise RuntimeError('Fitted points were modified')
    obs_path=_path(recipe['observation'])
    if _sha(obs_path)!=recipe['observationSha256']:raise RuntimeError('Observed source identity changed')
    obs=json.loads(obs_path.read_text(encoding='utf-8'))
    observe=runpy.run_path(str(Path(__file__).with_name('hand_asset_observe_r10.py')))
    rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    if scene.frame_current!=1 or scene.frame_subframe!=0:raise RuntimeError('Keep the matched body/cloth/hair frame-one source; no pose reset performed')
    original_rig=_rig_contract(rig)
    actual_bones={b.name:b for b in rig.data.bones}
    for bone in obs['current'][-1]['bones']:
        actual=actual_bones.get(bone['name'])
        if actual is None or not np.allclose(np.array(actual.matrix_local),bone['restMatrixLocal'],atol=2e-7,rtol=0):
            raise RuntimeError('Current rig rest differs from observed fitting rig: '+bone['name'])
    hands=[bpy.data.objects[name] for name in observe['HANDS']]
    nails=[bpy.data.objects[name] for name in observe['NAILS']]
    for obj in hands:
        if obj.data.shape_keys:raise RuntimeError('Candidate would discard existing hand shape keys')
        if hashlib.sha256(_coordinates(obj.data).tobytes()).hexdigest()!=recipe['sourceHandCoordinatesSha256'][obj.name]:
            raise RuntimeError('Current hand geometry differs from the observed baseline')
        if not np.allclose(np.array(obj.matrix_world),np.eye(4),atol=1e-7):raise RuntimeError('Observed hand space changed')
    groups_left=recipe['weightTransfer']['sourceGroups'];weight_columns={}
    observed_hands={o['name']:o for o in obs['current'][:2]}
    observed_arrays_path=_path(obs['arrays'])
    if _sha(observed_arrays_path)!=recipe['observedArraysSha256']:raise RuntimeError('Observed weight arrays changed')
    with np.load(observed_arrays_path) as observed_arrays:
        for obj,side in zip(hands,('left','right')):
            observed_hand=observed_hands[obj.name]
            actual=[{'name':g.name,'index':g.index,'lockWeight':g.lock_weight} for g in obj.vertex_groups]
            weight_columns[obj.name]=_group_columns(actual,observed_hand['vertexGroups'],groups_left,side)
            _source_weights_unchanged(obj,observed_hand,observed_arrays)
    protected={o.as_pointer():(o.name,o.data.as_pointer() if o.data else None,
               o.parent.as_pointer() if o.parent else None,np.array(o.matrix_parent_inverse).tolist())
               for o in bpy.data.objects if o not in hands}
    old_data=[obj.data for obj in hands]
    nail_visibility=[(obj,obj.hide_render,obj.hide_viewport,obj.hide_get()) for obj in nails]
    all_before=observe['_ids'](bpy);backups=[];success=False
    result={'source':source_relative,'sourceSha256':source_sha256,'plan':plan_relative,
        'planSha256':_sha(plan_path),'payloadSha256':recipe['payloadSha256'],
        'asset':{'name':recipe['assetName'],'bundleSha256':recipe['bundleSha256'],'metadata':recipe['assetMetadata']},
        'changes':[],'appearanceAccepted':False,'masterUpdated':False,
        'limitations':['Actual closeup and relaxed/flexed-finger poses remain required.',recipe['wristScope'],
                       'Existing body action, rig rest, cloth and native hair are preserved; they do not prove this new hand candidate passes visual review.']}
    try:
        backup_col=bpy.data.collections.new('R10 retained original hands - reversible')
        scene.collection.children.link(backup_col);backup_col.hide_render=True;backup_col.hide_viewport=True
        with np.load(_path(recipe['payload'])) as payload:
            loops=payload['loopVertex'];starts=payload['polygonStart'];sizes=payload['polygonSize'];face_groups=payload['faceGroup']
            for index,(obj,side) in enumerate(zip(hands,('L','R'))):
                backup=obj.copy();backup.name='R10 retained '+obj.name;backup_col.objects.link(backup)
                backup.hide_render=True;backup.hide_viewport=True;backups.append(backup)
                mesh=bpy.data.meshes.new('R10 HBM fitted hand '+side)
                mesh.use_fake_user=True
                faces=[loops[a:a+b].tolist() for a,b in zip(starts,sizes)]
                # The source-to-left coordinate map reverses handedness.
                reverse=side=='L';faces=[list(reversed(f)) if reverse else f for f in faces]
                mesh.from_pydata(payload['left' if side=='L' else 'right'].tolist(),[],faces);mesh.update()
                uv=mesh.uv_layers.new(name='R10 HBM Hand UV')
                source_uv=payload['uv'];uv_values=np.empty((len(mesh.loops),2),np.float32)
                for p,start,size in zip(mesh.polygons,starts,sizes):
                    values=source_uv[start:start+size]
                    uv_values[p.loop_start:p.loop_start+p.loop_total]=values[::-1] if reverse else values
                uv.data.foreach_set('uv',uv_values.ravel())
                sculpt=mesh.attributes.new('R10_hbm_face_group','INT','FACE');sculpt.data.foreach_set('value',face_groups)
                mesh.materials.append(old_data[index].materials[0])
                mesh.materials.append(bpy.data.objects['Nail '+side+' Index · polished'].data.materials[0])
                for p,g in zip(mesh.polygons,face_groups):p.material_index=1 if g==10 else 0;p.use_smooth=True
                obj.data=mesh
                # Blender 5.2 switches the group-name storage with the mesh.
                # Recreate the exact observed definitions on this new mesh,
                # rather than iterating the now-empty collection silently.
                expected_groups=observed_hands[obj.name]['vertexGroups']
                if len(obj.vertex_groups)==0:
                    for row in expected_groups:
                        group=obj.vertex_groups.new(name=row['name']);group.lock_weight=row['lockWeight']
                        if group.index!=row['index']:raise RuntimeError('New mesh group index differs from observed original')
                actual_groups=[{'name':g.name,'index':g.index,'lockWeight':g.lock_weight} for g in obj.vertex_groups]
                _group_columns(actual_groups,expected_groups,groups_left,'left' if side=='L' else 'right')
                weights=payload['weights']
                expected_weights=np.zeros((len(mesh.vertices),len(obj.vertex_groups)),np.float32)
                for g in obj.vertex_groups:
                    column=weight_columns[obj.name][g.index]
                    expected_weights[:,g.index]=weights[:,column]
                    for vertex in np.flatnonzero(weights[:,column]>0):
                        g.add([int(vertex)],float(weights[vertex,column]),'REPLACE')
                weight_readback=_installed_weights(obj,expected_weights)
                # Retain the existing Armature + one Subdivision modifier, which
                # now refine the actual evaluated HBM surface and its joint loops.
                obj['R10_hbm_hand_original_object']=backup.name
                obj['R10_hbm_hand_provenance']=json.dumps({'asset':recipe['assetName'],
                    'assetMetadata':recipe['assetMetadata'],'bundleSha256':recipe['bundleSha256'],
                    'observationSha256':recipe['observationSha256'],'fitPayloadSha256':recipe['payloadSha256'],
                    'faceGroupAttribute':'R10_hbm_face_group','nailFaceGroup':10,'side':side},ensure_ascii=False)
                result['changes'].append({'hand':obj.name,'backupObject':backup.name,
                    'oldData':old_data[index].name,'newData':mesh.name,'vertices':len(mesh.vertices),
                    'polygons':len(mesh.polygons),'integratedNailFaces':int(np.count_nonzero(face_groups==10)),
                    'uvLoops':len(uv.data),'groups':len(obj.vertex_groups),
                    'originalGroupIndicesAndNamesPreserved':True,
                    'originalObservedWeightsExactMatch':True,
                    'installedNewMeshWeights':weight_readback,
                    'weightColumnMapping':[{'groupName':g.name,'actualGroupIndex':g.index,
                        'leftPayloadColumn':weight_columns[obj.name][g.index]} for g in obj.vertex_groups]})
        for obj,*_ in nail_visibility:obj.hide_render=True;obj.hide_viewport=True;obj.hide_set(True)
        bpy.context.view_layer.update()
        result['actualHandArmatureMotion']=[_actual_armature_motion(bpy,obj,rig) for obj in hands]
        result['actualNailBoneAttachment']=[]
        for side in ('L','R'):
            diagnosis=hand_camera(None,rig,side,None,diagnostics_only=True)
            if max(n['centerTipDistance'] for n in diagnosis['nails'])>.020:
                raise RuntimeError('Evaluated integrated nails are not near their actual posed distal bones: '+json.dumps(diagnosis))
            if abs(diagnosis['normalLongitudinalDot'])>.6:
                raise RuntimeError('Evaluated nail normals are inconsistent with the actual posed hand: '+json.dumps(diagnosis))
            result['actualNailBoneAttachment'].append(diagnosis)
        if _rig_contract(rig)!=original_rig:raise RuntimeError('Hand adaptation changed rig/action/rest')
        for pointer,row in protected.items():
            obj=next((o for o in bpy.data.objects if o.as_pointer()==pointer),None)
            if obj is None or (obj.name,obj.data.as_pointer() if obj.data else None,
                 obj.parent.as_pointer() if obj.parent else None,np.array(obj.matrix_parent_inverse).tolist())!=row:
                raise RuntimeError('A protected original object/data/parent changed')
        result['rigActionRestUnchanged']=True;result['otherOriginalObjectDataAndParentsUnchanged']=True
        result['separateOriginalNailsHidden']=[obj.name for obj in nails]
        result['originalNailVisibility']={obj.name:{'render':r,'viewport':v,'hideGet':h} for obj,r,v,h in nail_visibility}
        result['sourceUnchangedOnDisk']=_sha(source)==source_sha256
        if not result['sourceUnchangedOnDisk']:raise RuntimeError('Source changed on disk during preparation')
        # No metadata says this is accepted. Root renders the bounded candidate.
        scene['R10_hand_asset_candidate']=json.dumps(result,ensure_ascii=False)
        bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False,relative_remap=True)
        success=True
        result['candidate']=output_relative;result['candidateSha256']=_sha(output)
        with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    finally:
        if not success:
            for obj,data in zip(hands,old_data):
                obj.data=data
                for key in ('R10_hbm_hand_original_object','R10_hbm_hand_provenance'):
                    if key in obj:del obj[key]
            for obj,r,v,h in nail_visibility:obj.hide_render=r;obj.hide_viewport=v;obj.hide_set(h)
            if 'R10_hand_asset_candidate' in scene:del scene['R10_hand_asset_candidate']
            new=[item for ptr,item in observe['_ids'](bpy).items() if ptr not in all_before]
            if new:bpy.data.batch_remove(ids=new)
    return result


def restore():
    """Only current in-memory candidate; caller chooses a new save path."""
    import bpy
    scene=bpy.context.scene
    record=json.loads(scene['R10_hand_asset_candidate'])
    for row in record['changes']:
        obj=bpy.data.objects[row['hand']];backup=bpy.data.objects[row['backupObject']]
        if obj.data.name!=row['newData']:raise RuntimeError('Current hand was edited after candidate creation; preserve it first')
        obj.data=backup.data
        for key in ('R10_hbm_hand_original_object','R10_hbm_hand_provenance'):
            if key in obj:del obj[key]
    for name,old in record['originalNailVisibility'].items():
        obj=bpy.data.objects[name];obj.hide_render=old['render'];obj.hide_viewport=old['viewport'];obj.hide_set(old['hideGet'])
    del scene['R10_hand_asset_candidate'];bpy.context.view_layer.update()
    return {'restored':True,'savedBlend':False,'retainedNewMeshes':True}


def hand_camera(camera,rig,side,utility,diagnostics_only=False):
    """Camera for the candidate's actual integrated nail faces, not hidden old plates."""
    import bpy
    import itertools
    from mathutils import Vector
    graph=bpy.context.evaluated_depsgraph_get();prefix='left' if side=='L' else 'right'
    hand=bpy.data.objects['Hand '+side+' · continuous anatomy']
    if 'R10_hbm_hand_provenance' not in hand:raise RuntimeError('This camera needs the Human Base Meshes hand candidate')
    ev=hand.evaluated_get(graph);mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        nail_faces=[p for p in mesh.polygons if p.material_index==1]
        if len(nail_faces)<20:raise RuntimeError('Actual integrated nail surface is missing')
        # Connected nail face islands remain separate although the complete
        # mesh is continuous through their skin borders.
        by_vertex={}
        for p in nail_faces:
            for v in p.vertices:by_vertex.setdefault(v,[]).append(p.index)
        pending={p.index for p in nail_faces};islands=[]
        while pending:
            seed=pending.pop();island={seed};stack=[seed]
            while stack:
                for vertex in mesh.polygons[stack.pop()].vertices:
                    for neighbor in by_vertex[vertex]:
                        if neighbor in pending:pending.remove(neighbor);island.add(neighbor);stack.append(neighbor)
            islands.append([mesh.polygons[i] for i in island])
        if len(islands)!=5:raise RuntimeError('Expected five actual integrated nail islands, got '+str(len(islands)))
        normal_matrix=ev.matrix_world.to_3x3().inverted().transposed()
        centers=[];normals=[];island_areas=[];island_sizes=[]
        for island in islands:
            area=sum(p.area for p in island)
            island_areas.append(area);island_sizes.append(len(island))
            centers.append(ev.matrix_world@(sum((p.center*p.area for p in island),Vector())/area))
            normals.append((normal_matrix@sum((p.normal*p.area for p in island),Vector())).normalized())
    finally:ev.to_mesh_clear()
    fingers=('Thumb','Index','Middle','Ring','Little')
    tips=[rig.matrix_world@rig.pose.bones[prefix+f+'Distal'].tail for f in fingers]
    order=min(itertools.permutations(range(5)),key=lambda seq:sum((centers[j]-t).length_squared for j,t in zip(seq,tips)))
    centers=[centers[i] for i in order];normals=[normals[i] for i in order]
    wrist=rig.matrix_world@rig.pose.bones[prefix+'Hand'].head
    tip=rig.matrix_world@rig.pose.bones[prefix+'MiddleDistal'].tail
    index=rig.matrix_world@rig.pose.bones[prefix+'IndexProximal'].head
    little=rig.matrix_world@rig.pose.bones[prefix+'LittleProximal'].head
    axis=(tip-wrist).normalized();across=(little-index).normalized()
    normal=sum(normals[1:],Vector()).normalized()
    diagnosis={'side':side,'handObject':hand.name,'handMatrixWorld':np.asarray(hand.matrix_world).tolist(),
        'wrist':list(wrist),'middleDistalTip':list(tip),'longitudinalAxis':list(axis),
        'fourFingerMeanNormal':list(normal),'normalLongitudinalDot':float(normal.dot(axis)),
        'matchingOrder':list(order),'nails':[{'finger':finger,'originalIsland':int(i),
            'faces':island_sizes[i],'area':island_areas[i],'center':list(center),
            'normal':list(norm),'boneTip':list(btip),'centerTipDistance':(center-btip).length,
            'normalLongitudinalDot':float(norm.dot(axis))}
            for finger,i,center,norm,btip in zip(fingers,order,centers,normals,tips)],
        'normalMethod':'Area-weighted polygon normals from actual evaluated integrated nail islands; thumb excluded from mean'}
    if diagnostics_only:return diagnosis
    if abs(normal.dot(axis))>.6:raise RuntimeError('Actual nail normals do not describe a dorsal view: '+json.dumps(diagnosis))
    target=(wrist+tip)*.5;position=target+normal*.65-axis*.12+across*.04
    utility['_aim'](camera,position,target,axis)
    camera.data.ortho_scale=utility['_fit_scale'](camera,utility['_bounds']([hand]),1.12)
    visibility=[]
    for finger,point in zip(fingers,centers):
        direction=point-camera.location
        hit,location,_,_,first,_=bpy.context.scene.ray_cast(graph,camera.location,direction.normalized(),distance=direction.length+.003)
        first_name=first.original.name if hit else None
        error=(location-point).length if hit else None
        visibility.append({'finger':finger,'nailCenterWorld':list(point),'firstObject':first_name,
                           'distanceToNailCenterMeters':error,'centerVisible':first_name==hand.name and error<.003})
    if sum(v['centerVisible'] for v in visibility[1:])<3:raise RuntimeError('Candidate hand view does not expose at least three actual nail islands: '+json.dumps(visibility))
    return {'target':list(target),'dorsalNormal':list(normal),'integratedNailVisibility':visibility,
            'actualNailAxisDiagnosis':diagnosis,
            'hiddenOriginalNailsUsedForCamera':False,'geometrySource':'Human Base Meshes 1.4.1 actual evaluated hand with integrated nail material islands'}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    offline=sub.add_parser('plan');offline.add_argument('--report',default=PLAN)
    build=sub.add_parser('prepare');build.add_argument('--source',required=True);build.add_argument('--source-sha',required=True)
    build.add_argument('--plan',default=PLAN);build.add_argument('--output',default='work/reconstruction-r02/reference-asset-hands-r10.blend')
    args=parser.parse_args(argv)
    if args.mode=='plan':
        r=plan(args.report);print(json.dumps({'report':args.report,'candidateReady':r['candidateReady'],
            'folds':r['diagnostics']['localFoldCount'],'localScaleRange':r['diagnostics']['localSingularValueRange']},ensure_ascii=False))
    else:
        r=run(args.source,args.source_sha,args.plan,args.output)
        print(json.dumps({'candidate':r['candidate'],'sha256':r['candidateSha256'],'appearanceAccepted':False},ensure_ascii=False))


if __name__=='__main__':main()
