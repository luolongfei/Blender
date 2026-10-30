"""Independent first adult female head integration. No import-time Blender work.

Preserves the author's Multires L3 on a new head, supplies new adult eyes, and
hides the rejected face by an evaluated-only material-domain deletion. This is
a static form candidate, NOT migrated expressions, collision or runtime proof.
The official donor, saved current source and all old raw meshes/keys stay intact.
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
HEAD='R11 Female head study'
COLLECTION='R11 Adult female face candidate'
MARKER='R11_adult_female_head_candidate'
HIDE_MODIFIER='R11 hide rejected head surface'
DONOR='assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
DONOR_SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
OBS='work/reconstruction-r02/face-structure-observation-r10.json'
SELECTION='work/reconstruction-r02/female-head-selection-r11.json'
LANDMARKS='work/reconstruction-r02/female-face-landmarks-r11.json'
SCALE=1.53119993/1.6414071321487427
TRANSLATION=(0.,.030,0.)
OLD_ATTACHMENTS=('R01 Sclera L','R01 Sclera R','R01 Textured iris L','R01 Textured iris R',
    'R02 Upper teeth','R02 Lower teeth','R02 Tongue','R03 Upper gingiva','R03 Lower gingiva',
    'R02 Soft brow -1','R02 Soft brow 1','R02 Upper lashes L','R02 Upper lashes R',
    'R02 Upper lid rim L','R02 Upper lid rim R')


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _path(relative,suffix):
    p=project_path().resolve();path=(p/relative).resolve();path.relative_to(p/'work/reconstruction-r02')
    if path.suffix!=suffix:raise ValueError('Unexpected independent artifact type')
    return path


def _array(data,field,width=3,dtype=np.float32):
    result=np.empty(len(data)*width,dtype);data.foreach_get(field,result);return result.reshape(-1,width)


def _smooth(t):
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)


def _read(relative):return json.loads((project_path()/relative).read_text(encoding='utf-8'))


def _world(points,matrix):
    m=np.asarray(matrix,float);return np.asarray(points)@m[:3,:3].T+m[:3,3]


def female_matrix():
    matrix=np.diag([SCALE,SCALE,SCALE,1.]);matrix[:3,3]=TRANSLATION;return matrix


def _mouth_record(landmarks):
    if 'mouth' not in landmarks:raise RuntimeError('Explicit female mouth topology required')
    return landmarks['mouth']


def soften_chin(raw,face_sets,loops,starts,sizes,landmarks):
    """Bounded lower-chin roundness; no eye, nose, lip or global-face scaling."""
    points=np.asarray(raw,float);x,y,z=points.T
    weight=np.exp(-((abs(x)-.044)/.029)**2-((z-1.435)/.022)**4)
    weight*=_smooth((-y-.040)/.055)
    weight*=_smooth((z-1.402)/.010)*(1.-_smooth((z-1.459)/.010))
    oral=set()
    for st,n,label in zip(starts,sizes,face_sets):
        if int(label) in (7,8):oral.update(map(int,loops[int(st):int(st+n)]))
    mouth=_mouth_record(landmarks)
    oral.update(map(int,mouth['vermilionStripVertexIndices']))
    weight[list(oral)]=0.
    delta=np.zeros_like(points);delta[:,0]=-np.sign(x)*.0016*weight
    rounding=np.exp(-((abs(x)-.026)/.024)**2-((z-1.419)/.011)**4)
    delta[:,2]=.00075*rounding*_smooth((-y-.040)/.055)*_smooth((z-1.402)/.010)
    delta[list(oral)]=0.
    # The correction remains far below the nose/eyes and fades to exact zero.
    delta[z>=1.469]=0.;delta[z<=1.402]=0.
    if np.linalg.norm(delta,axis=1).max()>.00181:raise RuntimeError('Unexpected chin correction amplitude')
    return delta.astype(np.float32),{'maximumDonorSpaceDisplacementMeters':float(np.linalg.norm(delta,axis=1).max()),
        'changedVertices':int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-8)),
        'oralVerticesProtected':len(oral),'eyesNoseLipsUnchanged':True,
        'scope':'Up to 1.6 mm lateral lower-jaw softening and 0.75 mm rounding of chin corners in donor space; all nose/lip/eye geometry retained.'}


def _load_data():
    project=project_path();raw_report=_read('work/reconstruction-r02/female-basemesh-actual-r11.json')
    raw_path=project/'work/reconstruction-r02/female-basemesh-actual-r11.npz'
    if _sha(raw_path)!=raw_report['npz_sha256']:raise RuntimeError('Actual female raw observation changed')
    with np.load(raw_path) as f:raw={k:f[k] for k in f.files if k.startswith('object_00')}
    selection=_read(SELECTION);landmarks=_read(LANDMARKS)
    return raw_report,raw,selection,landmarks


def _source_match(head,observation,uv):
    record=observation['objects']['R01 Head'];path=project_path()/observation['arrays']['path']
    if _sha(path)!=observation['arrays']['sha256']:raise RuntimeError('Current-face observation changed')
    with np.load(path) as arrays:
        prefix=record['prefix']
        if not np.array_equal(_array(head.data.vertices,'co'),arrays[prefix+'_raw']):
            raise RuntimeError('Current raw head does not match the observed rejected face')
        if list(head.data.shape_keys.key_blocks.keys())!=[k['name'] for k in record['keys']]:
            raise RuntimeError('Current shape-key inventory changed')
        for k in record['keys']:
            if not np.array_equal(_array(head.data.shape_keys.key_blocks[k['name']].data,'co'),arrays[k['array']]):
                raise RuntimeError('Current head key changed since observation: '+k['name'])
    return {'actualRawHeadAndAllKeysMatch':True,'observationSourceSha256':observation['sourceSha256'],
            'observationArraysSha256':observation['arrays']['sha256']}


def _delete_old_face(bpy,head,material):
    group=bpy.data.node_groups.new(HIDE_MODIFIER,'GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=group.nodes;links=group.links
    inp=nodes.new('NodeGroupInput');out=nodes.new('NodeGroupOutput')
    selected=nodes.new('GeometryNodeMaterialSelection');selected.inputs['Material'].default_value=material
    delete=nodes.new('GeometryNodeDeleteGeometry');delete.domain='FACE';delete.mode='ALL'
    links.new(inp.outputs['Geometry'],delete.inputs['Geometry'])
    links.new(selected.outputs['Selection'],delete.inputs['Selection'])
    links.new(delete.outputs['Geometry'],out.inputs['Geometry'])
    mod=head.modifiers.new(HIDE_MODIFIER,'NODES');mod.node_group=group
    return mod


def _skin_material(bpy,head,source,raw,landmarks):
    old=bpy.data.objects['R01 Head'];original=old.data.color_attributes['R01_region_color']
    colours=_array(original.data,'color',4);support=set()
    slots={i for i,s in enumerate(old.material_slots) if s.material==source}
    for polygon in old.data.polygons:
        if polygon.material_index in slots:support.update(map(int,polygon.vertices))
    if not support:raise RuntimeError('No actual current physical-skin colour domain')
    base=np.median(colours[sorted(support),:3],axis=0)
    colour=np.tile(np.r_[base,1.],(len(head.data.vertices),1)).astype(np.float32)
    mouth=_mouth_record(landmarks);strip=np.asarray(mouth['vermilionStripVertexIndices'],np.int32)
    # Existing source-derived warm lip pigment, restricted to real lip topology.
    lip=np.asarray((.4468,.2488,.1890))
    colour[strip,:3]=base*.40+lip*.60
    outer=mouth['outerVermilionLoopCandidate']
    if isinstance(outer,dict):outer=outer.get('sourceVertexIndices')
    if outer is None:raise RuntimeError('Female outer vermilion ring layout needs explicit inspection')
    colour[np.asarray(outer,np.int32),:3]=base*.80+lip*.20
    ca=head.data.color_attributes.new(name='R01_region_color',type='FLOAT_COLOR',domain='POINT')
    ca.data.foreach_set('color',colour.ravel())
    rough=np.full(len(colour),.50,np.float32);rough[strip]=.43
    ra=head.data.attributes.new(name='R04 skin roughness',type='FLOAT',domain='POINT');ra.data.foreach_set('value',rough)
    rest=head.data.attributes.new(name='R04 skin rest position',type='FLOAT_VECTOR',domain='POINT')
    rest.data.foreach_set('vector',_world(_array(head.data.vertices,'co'),female_matrix()).astype(np.float32).ravel())
    skin=source.copy();skin.name='R11 Female head neutral skin study'
    skin['R11_source_material']=source.name;skin['R11_scope']='Same current optical constants, new donor point maps; no makeup treatment.'
    oral=bpy.data.materials.new('R11 Oral cavity interior study');oral.use_nodes=True
    bs=next(n for n in oral.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Base Color'].default_value=(.038,.012,.010,1.);bs.inputs['Roughness'].default_value=.44
    bs.inputs['IOR'].default_value=1.4
    head.data.materials.clear();head.data.materials.append(skin);head.data.materials.append(oral)
    labels=raw['object_00_face_sets']
    for polygon,label in zip(head.data.polygons,labels):
        polygon.material_index=1 if int(label) in (7,8) else 0;polygon.use_smooth=True
    return {'skinMaterial':skin.name,'oralMaterial':oral.name,'lipVertexCount':len(strip),
        'sourceOpticsPreserved':True,'newPigmentMapping':'Uniform source warm baseline plus restrained true vermilion strip. Existing donor UDIM UV retained unchanged.',
        'skinMicrodetailNotAccepted':True}


def _old_head_state(head,uv):
    import bpy
    display=head.modifiers.get(HIDE_MODIFIER)
    enabled=display.show_viewport if display else None
    try:
        # Compare the same original evaluated stack, including strict dimensions.
        # The explicit new face-display deletion is the only disabled modifier.
        if display:display.show_viewport=False;bpy.context.view_layer.update()
        state=uv['_original_contract'](head,tuple(v.name for v in head.data.uv_layers));state.pop('sha256')
        state['modifiers']=[m for m in state['modifiers'] if m['settings'].get('name')!=HIDE_MODIFIER]
        return state
    finally:
        if display:display.show_viewport=enabled;bpy.context.view_layer.update()


def _eye_rest_bounds_guard(bpy,eyes):
    graph=bpy.context.evaluated_depsgraph_get();records=[]
    for side,row in eyes['eyes'].items():
        for part in ('sclera','iris'):
            obj=bpy.data.objects[row['objects'][part]];evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh()
            try:points=_world(_array(mesh.vertices,'co'),evaluated.matrix_world)
            finally:evaluated.to_mesh_clear()
            actual=np.asarray([points.min(0),points.max(0)])
            expected=np.asarray(row['target_'+part+'_world_bounds'])
            error=float(np.max(np.abs(actual-expected)))
            record={'side':side,'part':part,'object':obj.name,'actualWorldBounds':actual.tolist(),
                'expectedRestWorldBounds':expected.tolist(),'maximumErrorMeters':error}
            records.append(record)
            if error>2e-7:
                raise RuntimeError('Integrated eye parent failed exact REST-world bounds: '+json.dumps(record))
    return {'temporaryRigPose':'REST','toleranceMeters':2e-7,'parts':records,
        'scope':'Actual evaluated meshes with parent_object=new head; does not claim gaze or blink validation.'}


def _state_diff(before,after,path=()):
    differences=[]
    if isinstance(before,dict) and isinstance(after,dict):
        for key in sorted(set(before)|set(after)):
            if key not in before or key not in after:
                differences.append({'path':list(path+(key,)),'before':before.get(key,'MISSING'),'after':after.get(key,'MISSING')})
            else:differences.extend(_state_diff(before[key],after[key],path+(key,)))
    elif isinstance(before,(list,tuple)) and isinstance(after,(list,tuple)) and len(before)==len(after):
        for index,(a,b) in enumerate(zip(before,after)):differences.extend(_state_diff(a,b,path+(index,)))
    elif before!=after:differences.append({'path':list(path),'before':before,'after':after})
    return differences


def _modifier_active_ui_only(row):
    path=row['path']
    return (len(path)==4 and path[0]=='modifiers' and isinstance(path[1],int)
        and path[2:] == ['settings','is_active']
        and isinstance(row['before'],bool) and isinstance(row['after'],bool))


def prepare(source_relative,output_relative):
    import bpy
    from mathutils import Matrix
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Use new independent candidate/report paths')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh explicit independent source in background Object Mode required')
    if bpy.context.scene.get(MARKER) or bpy.data.objects.get(HEAD):raise RuntimeError('A previous adult head candidate is already present')
    donor=project_path()/DONOR
    if _sha(donor)!=DONOR_SHA:raise RuntimeError('Official donor changed')
    uv=_api('production_uv_r08');groom=_api('posterior_groom_r09');old=bpy.data.objects['R01 Head']
    bpy.context.view_layer.update();source_sha=_sha(source);pose_before=uv['_pose_contract']()
    match=_source_match(old,_read(OBS),uv);old_state=_old_head_state(old,uv)
    raw_report,raw,selection,landmarks=_load_data()
    donor_co=raw['object_00_local']
    delta,softening=soften_chin(donor_co,raw['object_00_face_sets'],raw['object_00_corner_vertices'],
        raw['object_00_face_starts'],raw['object_00_face_lengths'],landmarks)
    rig=bpy.data.objects['Character_Rig'];rig_pose=rig.data.pose_position
    scene=bpy.context.scene;frame=scene.frame_current;subframe=scene.frame_subframe
    visibility={};collection=None;head=None;new_modifier=None
    old_material=bpy.data.materials['R04 physical skin']
    try:
        scene.frame_set(1);rig.data.pose_position='REST';bpy.context.view_layer.update()
        with bpy.data.libraries.load(str(donor),link=False) as (available,requested):
            if 'GEO-body_female_realistic' not in available.objects:raise RuntimeError('Female donor absent')
            requested.objects=['GEO-body_female_realistic']
        head=requested.objects[0]
        if len(head.data.vertices)!=len(donor_co) or not np.array_equal(_array(head.data.vertices,'co'),donor_co):
            raise RuntimeError('Appended female cage differs from actual exported donor')
        collection=bpy.data.collections.new(COLLECTION);scene.collection.children.link(collection)
        collection.objects.link(head);head.name=HEAD
        for other in list(head.users_collection):
            if other!=collection:other.objects.unlink(head)
        donor_data=head.data;donor_data.name='R11 Original female cage and Multires retained';donor_data.use_fake_user=True
        head.data=donor_data.copy();head.data.name='R11 Female head softened cage'
        head.data.vertices.foreach_set('co',(donor_co+delta).ravel());head.data.update()
        multires=[m for m in head.modifiers if m.type=='MULTIRES']
        if len(multires)!=1 or multires[0].total_levels!=3:raise RuntimeError('Actual author Multires L3 unavailable')
        multires[0].levels=3;multires[0].render_levels=3;multires[0].show_viewport=True;multires[0].show_render=True
        head.parent=rig;head.parent_type='BONE';head.parent_bone='head'
        # A newly assigned bone parent must enter the depsgraph before deriving
        # its transform. Measure the actual effective parent matrix (including
        # Blender's bone-tail convention) with identity local transforms, then
        # retain its inverse explicitly. Do not guess an edit-bone transform.
        head.matrix_parent_inverse=Matrix.Identity(4);head.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update();effective_parent=head.matrix_world.copy()
        head.matrix_parent_inverse=effective_parent.inverted()
        head.matrix_basis=Matrix(female_matrix().tolist());bpy.context.view_layer.update()
        parenting={'effectiveParentRestWorld':[list(r) for r in effective_parent],
            'parentInverse':[list(r) for r in head.matrix_parent_inverse],
            'targetRestWorld':female_matrix().tolist(),'actualRestWorld':[list(r) for r in head.matrix_world],
            'maximumMatrixError':float(np.max(np.abs(np.asarray(head.matrix_world)-female_matrix())))}
        if parenting['maximumMatrixError']>2e-7:
            raise RuntimeError('Head parenting did not preserve the target REST-world matrix: '+json.dumps(parenting))
        keep=np.zeros(len(donor_co),bool);keep[np.asarray(selection['original_vertex_indices'],np.int32)]=True
        # Retain a short continuous author upper-chest/neck context below the
        # anatomical neck ring. This lies beneath the blouse; no facial hole or
        # truncated posterior neck is hidden by deleting Multires cage faces.
        keep|=donor_co[:,2]>=1.305
        vg=head.vertex_groups.new(name='R11 head and lower-neck context');vg.add(np.flatnonzero(keep).tolist(),1.,'REPLACE')
        mask=head.modifiers.new('R11 reversible head neck display','MASK');mask.vertex_group=vg.name;mask.threshold=.5
        skin=_skin_material(bpy,head,old_material,raw,landmarks)
        head['R11_female_to_target']=json.dumps(female_matrix().tolist())
        head['R11_parenting_record']=json.dumps(parenting)
        head['R11_original_female_data']=donor_data.name
        head['R11_original_donor_sha256']=DONOR_SHA
        head['R11_cage_softening']=json.dumps(softening)
        head['R11_static_candidate_only']=True
        new_modifier=_delete_old_face(bpy,old,old_material)
        for name in OLD_ATTACHMENTS:
            obj=bpy.data.objects.get(name)
            if obj:
                visibility[name]={'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hideSet':obj.hide_get()}
                obj.hide_render=True;obj.hide_viewport=True;obj.hide_set(True)
        eye_api=_api('eye_adult_fit_r11')
        eyes=eye_api['build_adult_eyes'](collection,Matrix(female_matrix().tolist()),parent_object=head)
        bpy.context.view_layer.update()
        eyes['integratedRESTWorldBoundsAudit']=_eye_rest_bounds_guard(bpy,eyes)
    except Exception:
        # No saved candidate is produced after a failed preparation. Roll back
        # existing scene surfaces/visibility before letting the fresh process exit.
        if new_modifier is not None:old.modifiers.remove(new_modifier)
        for name,state in visibility.items():
            obj=bpy.data.objects[name];obj.hide_render=state['hideRender'];obj.hide_viewport=state['hideViewport'];obj.hide_set(state['hideSet'])
        if collection is not None:
            for obj in list(collection.objects):bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.collections.remove(collection)
        raise
    finally:
        rig.data.pose_position=rig_pose;scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before:raise RuntimeError('Original body/rig pose changed')
    old_differences=_state_diff(old_state,_old_head_state(old,uv))
    if old_differences:
        print('R11_OLD_HEAD_FIELD_DIFF '+json.dumps(old_differences,ensure_ascii=False),flush=True)
        non_ui=[row for row in old_differences if not _modifier_active_ui_only(row)]
        if non_ui:
            raise RuntimeError('Original head differs beyond modifier active-selection UI: '+json.dumps(non_ui,ensure_ascii=False))
    if _sha(source)!=source_sha or _sha(donor)!=DONOR_SHA:raise RuntimeError('Saved source or official donor changed')
    delta_path=output.with_suffix('.cage-delta.npz')
    if delta_path.exists():raise RuntimeError('Retain prior cage-delta evidence')
    with delta_path.open('xb') as handle:np.savez_compressed(handle,donor_local=donor_co,delta_local=delta,
        candidate_local=donor_co+delta,head_display_mask=keep,female_to_target=female_matrix())
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'sourceMatch':match,
        'helperSha256':_sha(Path(__file__)),'eyeHelperSha256':_sha(Path(__file__).with_name('eye_adult_fit_r11.py')),
        'donor':DONOR,'donorSha256':DONOR_SHA,'femaleToTargetRestWorld':female_matrix().tolist(),
        'headObject':head.name,'newCollection':collection.name,'donorDataBackup':donor_data.name,'parenting':parenting,
        'authorMultiresPreserved':{'levels':3,'renderLevels':3,'topologyDeletedBeforeMultires':False},
        'softening':softening,'cageDelta':{'path':str(delta_path.relative_to(project_path())),'sha256':_sha(delta_path)},
        'anatomicalSelection':SELECTION,'headDisplayVertices':int(keep.sum()),
        'lowerNeckContextDonorZMinimum':1.305,'oldAttachmentsHidden':visibility,
        'oldFaceDisplayModifier':HIDE_MODIFIER,'oldRawHeadKeysUVRigAndMaterialsUnchanged':True,
        'oldModifierActiveUIChanges':old_differences,
        'skin':skin,'eyes':eyes,'appearanceAccepted':False,'savedBlend':False,
        'scope':'First neutral adult female head, eyes and true nose/lip volume in existing costume/groom context. No old exaggerated eye/nose/smile keys applied to new head.',
        'limits':['New head currently follows the head parent rigidly; neck weights and facial action migration are not completed.',
            'Old brows, lashes, lid rims, teeth/gingiva/tongue are explicitly hidden; new donor oral pocket remains. No expression or speech approval.',
            'A short author upper-chest region lies under the blouse and may overlap the retained old body. This is a visible-form candidate, not final seamless-body topology.',
            'Existing scalp/tails retain their old shape and cache. Groom-context views expose fit mismatches; no new collision or cache equivalence is claimed.',
            'Skin uses simple source-derived point colour and current optical constants. Skin detail and a finalized character-specific face are not accepted.']}
    scene[MARKER]=json.dumps(report)
    if output.exists():raise RuntimeError('Output appeared during preparation')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R11_ADULT_HEAD_SAVED '+json.dumps({'output':str(output),'outputSha256':report['outputSha256']}),flush=True)
    return report


def review(source_relative,case,output_relative,smile_strength=None):
    import bpy
    from mathutils import Vector
    source=_path(source_relative,'.blend')
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Review the saved candidate in a fresh background process')
    if MARKER not in bpy.context.scene:raise RuntimeError('Not an explicit R11 adult-head candidate')
    mode,view=case.split('-',1)
    if mode not in ('structure','groom') or view not in ('front','quarter','side'):raise ValueError('Explicit review case required')
    out=(project_path()/output_relative).resolve();out.relative_to((project_path()/'previews/reconstruction-r02').resolve())
    out.mkdir(parents=True,exist_ok=True);image_path=out/(case+'.png');report_path=out/(case+'.json')
    if image_path.exists() or report_path.exists():raise RuntimeError('Retain actual prior renders; use a new output directory')
    source_sha=_sha(source);scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update()
    expression=None
    if smile_strength is not None:
        if not 0.<=smile_strength<=1.:raise ValueError('Smile review strength must be within zero and one')
        keys=bpy.data.objects[HEAD].data.shape_keys
        if keys is None or 'R11 gentle smile' not in keys.key_blocks:raise RuntimeError('Explicit new adult smile key required')
        key=keys.key_blocks['R11 gentle smile'];previous=float(key.value);key.value=float(smile_strength)
        bpy.context.view_layer.update()
        expression={'key':key.name,'temporaryValue':float(key.value),'savedValue':previous,'sourceSaved':False}
    hidden=[]
    if mode=='structure':
        for obj in scene.objects:
            if obj.hide_render:continue
            if obj.name.startswith(('R11 Recombed ','R11 Donor / ')):
                obj.hide_render=True;hidden.append(obj.name);continue
            if obj.name.startswith('R11 '):continue
            name=obj.name.lower()
            if any(token in name for token in ('hair','forelock','scalp','gather','cheek wisps','tail','tie ring')) or name.startswith('r02 tie '):
                if obj.type in ('MESH','CURVES','CURVE'):obj.hide_render=True;hidden.append(obj.name)
    head=bpy.data.objects[HEAD];ev=head.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    try:points=_world(_array(mesh.vertices,'co'),ev.matrix_world)
    finally:ev.to_mesh_clear()
    # Only actual displayed new head/neck bounds, never the merged old body.
    lo=points.min(0);hi=points.max(0);target=(lo+hi)/2
    target[2]=float(hi[2]-.145)
    direction=np.asarray({'front':(0.,-1.,0.),'quarter':(.6,-.8,0.),'side':(1.,0.,0.)}[view])
    camera=bpy.data.objects.new('R11 actual head review camera',bpy.data.cameras.new('R11 actual head review camera'))
    scene.collection.objects.link(camera);camera.location=Vector(target+direction*2.)
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=.365;camera.data.lens=85.;camera.data.dof.use_dof=False
    scene.camera=camera;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24
    scene.cycles.use_denoising=True;scene.render.resolution_x=960;scene.render.resolution_y=1080;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.filepath=str(image_path)
    # Existing source lights and exposure are retained for all six cases.
    bpy.context.view_layer.update()
    detail=_api('final_detail_review_r09')
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'case':case,
        'newHeadActualBounds':[lo.tolist(),hi.tolist()],'headTarget':target.tolist(),
        'camera':{'matrixWorld':[list(r) for r in camera.matrix_world],'orthographicScale':camera.data.ortho_scale,
                  'resolution':[960,1080],'depthOfField':False},
        'lights':detail['_lights'](scene),'world':detail['_world'](scene),
        'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
            'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
        'hiddenForStructure':hidden,'expression':expression,'frame':1,'engine':'Cycles CPU','samples':24,
        'scope':'Actual static adult head candidate. Structure and groom-context use identical face, eyes, pose, camera and source lights. No facial-motion or collision acceptance.'}
    bpy.context.view_layer.update();bpy.ops.render.render(write_still=True)
    if not image_path.exists() or _sha(source)!=source_sha:raise RuntimeError('Render/source proof failed')
    report['imageSha256']=_sha(image_path)
    with report_path.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R11_ADULT_HEAD_REVIEW '+json.dumps({'image':str(image_path),'sourceSha256':source_sha}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('prepare','review'),required=True)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--case',choices=tuple(a+'-'+b for a in ('structure','groom') for b in ('front','quarter','side')))
    parser.add_argument('--smile',type=float,help='Temporary saved-key review strength; never saves the source')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if args.mode=='prepare':prepare(args.source,args.output)
    else:
        if not args.case:parser.error('--case required for review')
        review(args.source,args.case,args.output,args.smile)
