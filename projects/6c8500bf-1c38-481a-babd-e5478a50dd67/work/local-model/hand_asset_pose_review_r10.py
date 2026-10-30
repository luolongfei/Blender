"""One temporary HBM hand pose/detail per explicit Blender invocation.

No process launch, blend save, action editing, material change or body pose
reset. Actual hand geometry and rest matrices define axes. Numeric checks are
finite evidence only; every generated view still needs visual review.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
POSES=('relaxed','half-grip','fist','opposition','wrist-flex','wrist-extend')
VIEWS=('dorsal','palm','quarter','cuff')
DIGITS=('Thumb','Index','Middle','Ring','Little')
GROUPS={'Index':(19,16,3),'Middle':(18,15,4),'Ring':(17,14,5),'Little':(13,12,11),'Thumb':(23,22,21)}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _unit(value):
    v=np.asarray(value,dtype=float);n=np.linalg.norm(v)
    if n<1e-9:raise RuntimeError('Degenerate observed hand axis')
    return v/n


def _rotation(axis,degrees):
    x,y,z=_unit(axis);a=math.radians(degrees);c=math.cos(a);s=math.sin(a)
    k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    result=np.eye(4);result[:3,:3]=np.eye(3)*c+(1-c)*np.outer((x,y,z),(x,y,z))+s*k
    return result


def _rest_frame(hand,rig,side):
    """Native fitted mesh data are rig-local rest, not evaluated standing data."""
    prefix='left' if side=='L' else 'right';mesh=hand.data
    if not np.allclose(np.asarray(hand.matrix_world),np.eye(4),atol=1e-7):raise RuntimeError('Expected observed rig-local hand coordinates')
    if 'R10_hbm_hand_provenance' not in hand:raise RuntimeError('Actual HBM candidate required')
    if len(hand.vertex_groups)!=17 or any(sum(g.weight for g in v.groups)<.9999 for v in mesh.vertices):
        raise RuntimeError('Actual candidate hand is not completely weighted; preserve and reject the earlier unbound candidate')
    attr=mesh.attributes.get('R10_hbm_face_group')
    if not attr:raise RuntimeError('Actual HBM face-group attribute is missing')
    groups=np.empty(len(attr.data),np.int32);attr.data.foreach_get('value',groups)
    co=np.empty((len(mesh.vertices),3),np.float64);mesh.vertices.foreach_get('co',co.ravel())
    nails=[p for p,g in zip(mesh.polygons,groups) if g==10]
    normals=[]
    # Distinguish thumb by its actual nearest distal rest endpoint; do not infer
    # dorsal direction from the thumb's differently oriented nail.
    endpoints={f:np.asarray(rig.data.bones[prefix+f+'Distal'].tail_local) for f in DIGITS}
    for face in nails:
        points=co[np.asarray(face.vertices)];center=points.mean(0)
        finger=min(DIGITS,key=lambda f:np.linalg.norm(center-endpoints[f]))
        if finger!='Thumb':
            for i in range(1,len(points)-1):normals.append(np.cross(points[i]-points[0],points[i+1]-points[0]))
    dorsal=_unit(np.sum(normals,axis=0))
    wrist=np.asarray(rig.data.bones[prefix+'Hand'].head_local)
    longitudinal=_unit(endpoints['Middle']-wrist)
    raw_dot=float(dorsal@longitudinal)
    if abs(raw_dot)>.55:raise RuntimeError('Raw actual HBM nail direction is not dorsal: '+str(raw_dot))
    dorsal=_unit(dorsal-longitudinal*raw_dot)
    across=_unit(np.asarray(rig.data.bones[prefix+'LittleProximal'].head_local)-
                 np.asarray(rig.data.bones[prefix+'IndexProximal'].head_local))
    axes={};rest={}
    for finger in DIGITS:
        suffixes=('Metacarpal','Proximal','Distal') if finger=='Thumb' else ('Proximal','Intermediate','Distal')
        for suffix in suffixes:
            name=prefix+finger+suffix;bone=rig.data.bones[name]
            matrix=np.asarray(bone.matrix_local,dtype=float);direction=_unit(np.asarray(bone.tail_local)-bone.head_local)
            flex_rig=_unit(np.cross(direction,-dorsal))
            axes[name]=_unit(matrix[:3,:3].T@flex_rig)
            rest[name]=matrix
    handbone=rig.data.bones[prefix+'Hand'];handrest=np.asarray(handbone.matrix_local)
    handaxis=_unit(np.asarray(handbone.tail_local)-handbone.head_local)
    axes[prefix+'Hand']=_unit(handrest[:3,:3].T@_unit(np.cross(handaxis,-dorsal)))
    rest[prefix+'Hand']=handrest
    thumb=rig.data.bones[prefix+'ThumbMetacarpal'];thumbdir=_unit(np.asarray(thumb.tail_local)-thumb.head_local)
    toward=np.asarray(rig.data.bones[prefix+'MiddleProximal'].head_local)-thumb.head_local
    toward=_unit(toward-dorsal*(toward@dorsal))
    adduction=_unit(np.cross(thumbdir,toward))
    return {'prefix':prefix,'dorsal':dorsal,'longitudinal':longitudinal,'across':across,'wrist':wrist,
        'axes':axes,'rest':rest,'thumbAdductionLocal':_unit(rest[prefix+'ThumbMetacarpal'][:3,:3].T@adduction),
        'thumbLongitudinalLocal':_unit(rest[prefix+'ThumbMetacarpal'][:3,:3].T@thumbdir),
        'axisEvidence':{'restDorsalFromFourActualNailRegions':dorsal.tolist(),
            'restLongitudinal':longitudinal.tolist(),'rawNailLongitudinalDot':raw_dot,
            'flexionAxesBoneLocal':{k:v.tolist() for k,v in axes.items()},
            'method':'For each real bone: rest direction cross palmar direction, converted by that bone actual rest inverse rotation. No assumed global/local X axis.'}}


def _pose_spec(pose):
    if pose in ('relaxed','wrist-flex','wrist-extend'):return {f:(0,0,0) for f in DIGITS}
    if pose=='half-grip':return {'Thumb':(10,15,10),'Index':(30,40,20),'Middle':(35,45,25),'Ring':(40,50,28),'Little':(45,55,30)}
    if pose=='fist':return {'Thumb':(15,25,15),'Index':(70,90,50),'Middle':(75,95,55),'Ring':(80,95,60),'Little':(80,90,60)}
    return {'Thumb':(15,35,25),'Index':(65,80,30),'Middle':(10,15,5),'Ring':(15,20,5),'Little':(20,25,10)}


def _basis_for(frame,bases,spec,thumb_extra=(0,0)):
    result={name:value.copy() for name,value in bases.items()};prefix=frame['prefix']
    for finger,angles in spec.items():
        suffixes=('Metacarpal','Proximal','Distal') if finger=='Thumb' else ('Proximal','Intermediate','Distal')
        for suffix,angle in zip(suffixes,angles):
            name=prefix+finger+suffix
            result[name]=bases[name]@_rotation(frame['axes'][name],angle)
            if finger=='Thumb' and suffix=='Metacarpal':
                result[name]=result[name]@_rotation(frame['thumbAdductionLocal'],thumb_extra[0])@_rotation(frame['thumbLongitudinalLocal'],thumb_extra[1])
    return result


def _finger_fk(rig,frame,bases,finger):
    prefix=frame['prefix'];parent=rig.data.bones[prefix+'Hand'];world=np.asarray(rig.pose.bones[parent.name].matrix)
    suffixes=('Metacarpal','Proximal','Distal') if finger=='Thumb' else ('Proximal','Intermediate','Distal')
    for suffix in suffixes:
        bone=rig.data.bones[prefix+finger+suffix]
        if bone.parent!=parent or bone.inherit_scale!='FULL':raise RuntimeError('Unexpected actual finger hierarchy/scale inheritance')
        world=world@np.linalg.inv(np.asarray(parent.matrix_local))@np.asarray(bone.matrix_local)@bases[bone.name]
        parent=bone
    length=(parent.tail_local-parent.head_local).length
    return world[:3,3]+world[:3,1]*length,world


def _apply_pose(bpy,rig,frame,pose,snapshot):
    from mathutils import Matrix
    prefix=frame['prefix'];bases={name:np.asarray(row['basis']).copy() for name,row in snapshot.items()}
    spec=_pose_spec(pose);extra=(25,15) if pose=='fist' else (12,5) if pose=='half-grip' else (0,0)
    solver=None
    if pose=='opposition':
        # Pure matrix FK search; no 100x depsgraph/cloth/hair reevaluation. It
        # searches only bounded thumb degrees, with index already flexed.
        trial=_basis_for(frame,bases,spec)
        index_tip,_=_finger_fk(rig,frame,trial,'Index')
        thumb_root=np.asarray(rig.pose.bones[prefix+'ThumbMetacarpal'].head)
        target=index_tip+_unit(thumb_root-index_tip)*.003
        parameters=np.array([15.,35.,25.,30.,15.]);limits=np.array([[0,45],[0,75],[0,70],[0,65],[-20,45]],float)
        def score(values):
            trial_spec=dict(spec);trial_spec['Thumb']=tuple(values[:3])
            b=_basis_for(frame,bases,trial_spec,tuple(values[3:]))
            tip,_=_finger_fk(rig,frame,b,'Thumb')
            return float(np.linalg.norm(tip-target)),tip
        best,_=score(parameters)
        for step in (15.,7.5,3.75,1.875):
            for index in range(5):
                for sign in (-1,1):
                    proposed=parameters.copy();proposed[index]=np.clip(proposed[index]+sign*step,*limits[index])
                    error,_=score(proposed)
                    if error<best:parameters=proposed;best=error
        spec['Thumb']=tuple(parameters[:3]);extra=tuple(parameters[3:]);_,tip=score(parameters)
        solver={'method':'Bounded thumb-only matrix FK coordinate search, 40 trials; no source action edits',
            'angleLimitsDegrees':limits.tolist(),'targetIndexTipWith3mmOffsetRigLocal':target.tolist(),
            'predictedThumbTipRigLocal':tip.tolist(),'endpointGapMeters':best,
            'contactClaimed':False,'note':'Bone endpoint proximity is not skin-pad contact; evaluated surface checks and actual image are required.'}
    desired=_basis_for(frame,bases,spec,extra)
    if pose in ('wrist-flex','wrist-extend'):
        angle=25 if pose=='wrist-flex' else -20
        desired[prefix+'Hand']=bases[prefix+'Hand']@_rotation(frame['axes'][prefix+'Hand'],angle)
    for name,matrix in desired.items():
        pb=rig.pose.bones[name]
        if any(not c.mute and c.influence>1e-6 for c in pb.constraints):raise RuntimeError('Active constraint prevents isolated hand pose: '+name)
        pb.rotation_mode='QUATERNION';pb.matrix_basis=Matrix(matrix.tolist())
    bpy.context.view_layer.update()
    actual={name:np.asarray(rig.pose.bones[name].matrix_basis) for name in desired}
    errors={name:float(np.max(np.abs(actual[name]-value))) for name,value in desired.items()}
    if max(errors.values())>2e-5:raise RuntimeError('Animation or driver overrode temporary local hand pose: '+json.dumps(errors))
    return {'requestedPose':pose,'additionalFlexionDegrees':{k:list(v) for k,v in spec.items()},
        'thumbAdditionalAdductionAndLongitudinalDegrees':list(extra),
        'wristAdditionalFlexionDegrees':25 if pose=='wrist-flex' else -20 if pose=='wrist-extend' else 0,
        'poseMatrixErrorMax':max(errors.values()),'oppositionSolver':solver}


def _surface(bpy,obj):
    graph=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(graph)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        mesh.calc_loop_triangles();vertices=np.empty((len(mesh.vertices),3),np.float64);mesh.vertices.foreach_get('co',vertices.ravel())
        matrix=np.asarray(ev.matrix_world);vertices=vertices@matrix[:3,:3].T+matrix[:3,3]
        triangles=np.asarray([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
        polygons=np.asarray([t.polygon_index for t in mesh.loop_triangles],np.int32)
        attr=mesh.attributes.get('R10_hbm_face_group');groups=None
        if attr:
            values=np.empty(len(attr.data),np.int32);attr.data.foreach_get('value',values);groups=values[polygons]
        return {'vertices':vertices,'triangles':triangles,'faceGroups':groups,
            'edges':np.asarray([tuple(e.vertices) for e in mesh.edges],np.int32),'object':obj.name}
    finally:ev.to_mesh_clear()


def _volume(surface):
    v=surface['vertices'];t=surface['triangles'];center=v.mean(0);a,b,c=(v[t[:,i]]-center for i in range(3))
    return float(abs(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6))


def _crossings(a,b,same_mesh=False,limit=100000):
    """Actual edge/triangle narrow phase after BVH candidates, not just AABB hits."""
    from mathutils.bvhtree import BVHTree
    av=a['vertices'];at=a['triangles'];bv=b['vertices'];bt=b['triangles']
    if not len(at) or not len(bt):return {'complete':True,'trianglePairCrossings':0,'candidatePairs':0}
    tree_a=BVHTree.FromPolygons(av.tolist(),at.tolist(),all_triangles=True)
    tree_b=BVHTree.FromPolygons(bv.tolist(),bt.tolist(),all_triangles=True)
    pairs=tree_a.overlap(tree_b)
    if len(pairs)>limit:return {'complete':False,'candidatePairs':len(pairs),'limit':limit,'reason':'Bounded local check exceeded its pair budget'}
    keep=[]
    for i,j in pairs:
        if same_mesh and (i>=j and a is b or len(set(at[i])&set(bt[j]))):continue
        keep.append((i,j))
    if not keep:return {'complete':True,'trianglePairCrossings':0,'candidatePairs':len(pairs),'testedPairs':0}
    indices=np.asarray(keep);ta=av[at[indices[:,0]]];tb=bv[bt[indices[:,1]]]
    hit=np.zeros(len(keep),bool);parallel=np.zeros(len(keep),bool)
    for source,target in ((ta,tb),(tb,ta)):
        e1=target[:,1]-target[:,0];e2=target[:,2]-target[:,0]
        for edge in range(3):
            origin=source[:,edge];direction=source[:,(edge+1)%3]-origin
            p=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,p)
            stable=np.abs(det)>1e-13;inverse=np.zeros_like(det);inverse[stable]=1/det[stable]
            delta=origin-target[:,0];u=np.einsum('ij,ij->i',delta,p)*inverse
            q=np.cross(delta,e1);v=np.einsum('ij,ij->i',direction,q)*inverse
            t=np.einsum('ij,ij->i',e2,q)*inverse
            hit|=stable&(u>1e-5)&(v>1e-5)&(u+v<1-1e-5)&(t>1e-5)&(t<1-1e-5)
            parallel|=~stable
    points=(ta[hit].mean(1)+tb[hit].mean(1))*.5
    return {'complete':True,'candidatePairs':len(pairs),'testedPairs':len(keep),
        'trianglePairCrossings':int(hit.sum()),'crossingRegionBoundsWorld':
        [points.min(0).tolist(),points.max(0).tolist()] if len(points) else None,
        'parallelOrCoplanarCandidatesNotCertified':int((parallel&~hit).sum()),
        'scope':'Strict surface crossings only; tangent/copanar contacts and closed containment are not certified.'}


def _metrics(bpy,hand,cuff,before,after):
    v=after['vertices'];t=after['triangles'];groups=after['faceGroups']
    if groups is None or len(groups)!=len(t):raise RuntimeError('Actual evaluated hand face groups are missing')
    if not np.array_equal(t,before['triangles']):raise RuntimeError('Pose changed evaluated hand topology')
    edges=after['edges'];old=np.linalg.norm(before['vertices'][edges[:,1]]-before['vertices'][edges[:,0]],axis=1)
    new=np.linalg.norm(v[edges[:,1]]-v[edges[:,0]],axis=1);ratio=new/np.maximum(old,1e-12)
    domains={'Palm':(24,),**GROUPS};regions={}
    provenance=json.loads(hand['R10_hbm_hand_provenance']);prefix='left' if provenance['side']=='L' else 'right'
    rig=bpy.data.objects['Character_Rig']
    tips=np.asarray([list(rig.matrix_world@rig.pose.bones[prefix+finger+'Distal'].tail) for finger in DIGITS])
    nail_digit=np.argmin(np.linalg.norm(v[t].mean(1)[:,None]-tips[None],axis=2),axis=1)
    for name,values in domains.items():
        mask=np.isin(groups,values)
        if name in DIGITS:mask|=(groups==10)&(nail_digit==DIGITS.index(name))
        selected=t[mask];regions[name]={'vertices':v,'triangles':selected}
    collisions={}
    for name in DIGITS:
        collisions[name+' / Palm']=_crossings(regions[name],regions['Palm'],True)
    for a,b in itertools.combinations(DIGITS,2):collisions[a+' / '+b]=_crossings(regions[a],regions[b],True)
    for name in DIGITS:collisions[name+' self']=_crossings(regions[name],regions[name],True)
    cuff_surface=_surface(bpy,cuff);collisions['Hand / visible cuff']=_crossings(after,cuff_surface)
    old_volume=_volume(before);new_volume=_volume(after)
    return {'evaluatedVertices':len(v),'evaluatedTriangles':len(t),'handWorldBounds':[v.min(0).tolist(),v.max(0).tolist()],
        'closedHandSignedVolumeMagnitudeM3':new_volume,'standingVolumeM3':old_volume,
        'volumeRatioToStanding':new_volume/max(old_volume,1e-12),
        'surfaceEdgeLengthRatioRange':[float(ratio.min()),float(ratio.max())],
        'surfaceEdgeLengthRatioPercentiles':np.quantile(ratio,[.01,.05,.5,.95,.99]).tolist(),
        'localSurfaceCrossingChecks':collisions,
        'numericPassOrAppearanceAcceptanceClaimed':False,
        'limitations':['Finite pose and local hand/cuff surface checks only; no whole-body collision certification.',
                       'Tangent contacts, coplanar cases, completely enclosed volumes, friction and forces are not resolved.',
                       'Bone endpoint proximity and volume preservation cannot prove natural motion or visible hand quality.']}


def _camera(bpy,rig,hand,cuff,frame,view,utility,after):
    from mathutils import Vector
    matrix=np.asarray(rig.matrix_world)@np.asarray(rig.pose.bones[frame['prefix']+'Hand'].matrix)@np.linalg.inv(frame['rest'][frame['prefix']+'Hand'])
    dorsal=_unit(matrix[:3,:3]@frame['dorsal']);along=_unit(matrix[:3,:3]@frame['longitudinal']);across=_unit(matrix[:3,:3]@frame['across'])
    wrist=np.asarray(rig.matrix_world@rig.pose.bones[frame['prefix']+'Hand'].head)
    v=after['vertices'];target=(v.min(0)+v.max(0))*.5
    direction=dorsal if view=='dorsal' else -dorsal if view=='palm' else _unit(dorsal+across*.75)
    camera=bpy.data.objects.new('R10 temporary hand pose camera',bpy.data.cameras.new('R10 temporary hand pose camera'))
    bpy.context.scene.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.dof.use_dof=False
    camera.data.clip_start=.01;camera.data.clip_end=100
    if view=='cuff':
        target=wrist;direction=_unit(dorsal+across*.85);scale=.105
    else:scale=None
    utility['_aim'](camera,Vector(target+direction*.65-along*.025),Vector(target),Vector(along))
    camera.data.ortho_scale=scale or utility['_fit_scale'](camera,utility['_bounds']([hand]),1.15)
    if not .07<camera.data.ortho_scale<.32:raise RuntimeError('Hand-only framing outside compact bounds')
    bpy.context.scene.camera=camera
    return camera,{'view':view,'target':target.tolist(),'orthographicScale':camera.data.ortho_scale,
        'dorsalDirectionFromCurrentHandBoneAndRealRestNails':dorsal.tolist(),
        'actualCuffObject':cuff.name,'wholeBodyBoundsUsed':False,'noDepthOfField':True}


def run(source_relative,side,pose,view,label,samples=24):
    import bpy
    from mathutils import Matrix
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Disposable background Object Mode required')
    source=_path(source_relative)
    if Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Open the exact requested source')
    if side not in ('L','R') or pose not in POSES or view not in VIEWS:raise ValueError('Choose one explicit side, pose and view')
    if not re.fullmatch('[a-z0-9-]+',label):raise ValueError('Use a unique safe label')
    if samples not in (24,32):raise ValueError('Use 24 or 32 bounded review samples')
    digest=_sha(source);folder=_path('previews/reconstruction-r02/'+label);folder.mkdir(parents=True,exist_ok=True)
    source_record=folder/'hand-pose-review-source.json'
    identity={'source':source_relative,'sourceSha256':digest,'review':'R10 temporary hand poses, unchanged matched body frame 1'}
    if source_record.exists():
        if json.loads(source_record.read_text(encoding='utf-8'))!=identity:raise RuntimeError('Review label already belongs to another source')
    else:
        with source_record.open('x',encoding='utf-8') as f:json.dump(identity,f,ensure_ascii=False,indent=2)
    stem=side.lower()+'-'+pose+'-'+view;image=folder/(stem+'.png');report=folder/(stem+'.json')
    if image.exists() or report.exists():raise FileExistsError('Keep existing actual images and pose evidence')
    detail=_api('final_detail_review_r09');utility=_api('reference_final_stills_r08')
    rig,evidence=detail['_matched_standing'](bpy)
    hand=bpy.data.objects['Hand '+side+' · continuous anatomy'];cuff=bpy.data.objects['Blouse_Cuff_'+side]
    if any(o.hide_render or o.hide_viewport or o.hide_get() for o in (hand,cuff)):raise RuntimeError('Actual hand and cuff must be visible')
    attachment=_api('hand_asset_fit_r10')['hand_camera'](None,rig,side,None,diagnostics_only=True)
    if max(n['centerTipDistance'] for n in attachment['nails'])>.020:raise RuntimeError('Actual evaluated hand is not attached to its standing finger bones')
    frame=_rest_frame(hand,rig,side);names=tuple(frame['axes'])
    snapshot={name:{'basis':np.asarray(rig.pose.bones[name].matrix_basis).tolist(),
                    'rotationMode':rig.pose.bones[name].rotation_mode} for name in names}
    protected={b.name:np.asarray(b.matrix).copy() for b in rig.pose.bones if b.name not in names}
    rest={b.name:np.asarray(b.matrix_local).copy() for b in rig.data.bones}
    action=rig.animation_data.action;old_camera=bpy.context.scene.camera;camera=None;render_hidden=[]
    before=_surface(bpy,hand);lights_before=detail['_lights'](bpy.context.scene)
    result={'source':source_relative,'sourceSha256':digest,'side':side,'pose':pose,'view':view,
        'matchedStandingEvidence':evidence,'actualStandingHandAttachment':attachment,'axisEvidence':frame['axisEvidence'],
        'savedBlend':False,'appearanceAccepted':False,'physicsAccepted':False,
        'materialsEdited':False,'scope':'Temporary local hand/finger offsets on the unchanged matched body frame 1; actual image plus finite surface diagnostics.'}
    try:
        result['poseParameters']=_apply_pose(bpy,rig,frame,pose,snapshot)
        if rig.animation_data.action!=action:raise RuntimeError('Body action changed')
        for name,matrix in rest.items():
            if not np.array_equal(np.asarray(rig.data.bones[name].matrix_local),matrix):raise RuntimeError('Rig rest changed')
        for name,matrix in protected.items():
            if not np.allclose(np.asarray(rig.pose.bones[name].matrix),matrix,atol=2e-6,rtol=0):raise RuntimeError('Non-hand pose bone changed: '+name)
        after=_surface(bpy,hand);result['surfaceDiagnostics']=_metrics(bpy,hand,cuff,before,after)
        scene=bpy.context.scene;scene.render.resolution_x=scene.render.resolution_y=768;scene.render.resolution_percentage=100
        camera,framing=_camera(bpy,rig,hand,cuff,frame,view,utility,after);result['framing']=framing
        # With arms at the real body side the palm faces the skirt. Isolate
        # only this inspection view, keeping evaluated poses/cache/geometry
        # untouched; the contextual cuff views remain fully visible.
        if view=='palm':
            for obj in scene.objects:
                if obj not in (hand,cuff,camera) and obj.type in ('MESH','CURVES','CURVE','SURFACE','FONT','META','VOLUME') and not obj.hide_render:
                    render_hidden.append(obj);obj.hide_render=True
        result['inspectionIsolation']={'enabled':view=='palm','temporarilyRenderHidden':[o.name for o in render_hidden],
            'reason':'Standing palm faces the garment; isolated palm inspection avoids treating garment occlusion as palm geometry. Contextual wrist/cuff views retain all surroundings.',
            'bodyPoseOrCacheChanged':False}
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_sequencer=False
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(image)
        result['lighting']=lights_before;result['colorManagement']={'view':scene.view_settings.view_transform,
            'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
        bpy.ops.render.render(write_still=True)
        after_render=_surface(bpy,hand)
        if after_render['vertices'].shape!=after['vertices'].shape or np.max(np.abs(after_render['vertices']-after['vertices']))>2e-6:
            raise RuntimeError('Render reevaluation changed the temporary hand pose; image is not valid pose evidence')
        result['temporaryPoseSurvivedRenderReevaluation']=True
        result['image']=str(image.relative_to(project_path()));result['imageSha256']=_sha(image)
        result['lightingUnchanged']=detail['_lights'](scene)==lights_before
    finally:
        for obj in render_hidden:obj.hide_render=False
        for name,row in snapshot.items():
            pb=rig.pose.bones[name];pb.rotation_mode=row['rotationMode'];pb.matrix_basis=Matrix(row['basis'])
        bpy.context.view_layer.update();bpy.context.scene.camera=old_camera
        if camera:
            data=camera.data;bpy.data.objects.remove(camera,do_unlink=True);bpy.data.cameras.remove(data)
        result['localPoseRestored']=all(np.allclose(np.asarray(rig.pose.bones[name].matrix_basis),row['basis'],atol=2e-6,rtol=0) for name,row in snapshot.items())
        result['sourceUnchangedOnDisk']=_sha(source)==digest
        if not result['localPoseRestored'] or not result['sourceUnchangedOnDisk']:raise RuntimeError('Temporary review failed restoration/source identity check')
    with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    return {'image':result['image'],'report':str(report.relative_to(project_path())),
            'sourceSha256':digest,'appearanceAccepted':False,'physicsAccepted':False}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--side',choices=('L','R'),required=True);parser.add_argument('--pose',choices=POSES,required=True)
    parser.add_argument('--view',choices=VIEWS,required=True);parser.add_argument('--label',required=True)
    parser.add_argument('--samples',type=int,default=24);a=parser.parse_args(argv)
    print('R10_HAND_POSE_REVIEW '+json.dumps(run(a.source,a.side,a.pose,a.view,a.label,a.samples),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
