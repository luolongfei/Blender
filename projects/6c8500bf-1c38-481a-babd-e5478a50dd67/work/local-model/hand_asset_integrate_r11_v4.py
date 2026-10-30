"""Merge actual refined hands with a measured dense-target wrist correction.

capture: read the finished wrist+nail candidate into a SHA-bound small package.
prepare: append only its two hand MESH datablocks and dependencies; keep target
objects/modifiers/rig and all target face/groom data. Add only its sparse wrist
key to target R01 body. No old head, rig, action, hair or clothing is appended.
This independent v4 retains failed v2/v3, preserves the left v3 wrist exactly,
and adds at most 0.3 mm to 38 measured right-wrist cage points. All original
dense-pose gates remain. The check command rolls back without saving a Blend.
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
MARKER='R11_refined_hands_integrated'
WRIST_KEY='R11 wrist overlap clearance'
NAIL_KEY='R11 natural nail plate'
HANDS=('Hand L · continuous anatomy','Hand R · continuous anatomy')
HAND_PLAN='work/reconstruction-r02/hand-asset-fit-plan-r10-v3.json'
PAYLOAD_SHA='c86aaacc0fde9a15d6d9009531727b8a3e7d8ecff931fea81f38572b70ad6bbc'
TARGET_WRIST_PLAN='work/reconstruction-r02/hand-integrated-wrist-fit-plan-r11-v4.json'
TARGET_WRIST_PLAN_SHA='45d7d2348f7f3d7b5eddee376889de1eea33bc56628f6e252128171d68d836eb'
TARGET_WRIST_ARRAYS_SHA='ac42ed638a46a1f464fb06ef077056d3968462f49d85c4f18c0d7d6a02825a35'
TARGET_OBS_ARRAYS_SHA='078943738b320e77de53e94f9a6f2e25f52b8af787dd14eef2c9781b2c877852'
PARENT_WRIST_PLAN='work/reconstruction-r02/hand-integrated-wrist-fit-plan-r11-v2.json'
PARENT_WRIST_PLAN_SHA='88f306549c2f1f78327cfe363ac87772f3b04ed102228538304d5ad8fe7e8c56'
PARENT_WRIST_ARRAYS_SHA='e920cc2039c80eed7ddee94a3141fb6ad0f8e1b43c4943e359782d14e6d3ea41'
DEPS={'hand_asset_integrate_r11.py':'12c522866f3736d7f1aa6438539499f730b9497c734743e0926ef6ce363252de',
      'hand_wrist_clearance_r11_v2.py':'715b366dc374f2755574b8ef640d6076d64711cc9e856e8d7a4a68475de26759',
      'hand_asset_observe_r10.py':'b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'}


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _apis():
    result={}
    for name,digest in DEPS.items():
        path=Path(__file__).with_name(name)
        if _sha(path)!=digest:raise RuntimeError('Frozen integration dependency changed: '+name)
        result[name]=runpy.run_path(str(path))
    return result['hand_asset_integrate_r11.py'],result['hand_wrist_clearance_r11_v2.py'],result['hand_asset_observe_r10.py']


def _co(data):
    a=np.empty((len(data),3),np.float64);data.foreach_get('co',a.ravel());return a


def _groups(obj):return [{'name':g.name,'index':g.index,'lockWeight':g.lock_weight} for g in obj.vertex_groups]


def _weights(obj):
    a=np.zeros((len(obj.data.vertices),len(obj.vertex_groups)),np.float32)
    for vertex in obj.data.vertices:
        for group in vertex.groups:a[vertex.index,group.group]=group.weight
    return a


def _mesh_record(mesh,prefix,observe,arrays):
    row=observe['_mesh'](mesh,prefix,arrays)
    # These derived values can be recomputed after append. Actual positions,
    # polygon connectivity, smooth flags, UVs and every stored attribute remain
    # exact requirements; do not confuse a chosen quad diagonal with topology.
    derived=('normals','faceNormals','triangles','trianglePolygon','cornerNormals')
    for key in derived:row['arrays'].pop(key,None)
    row.pop('triangleCount',None)
    return row


def _keys(mesh,prefix,arrays):
    if not mesh.shape_keys:raise RuntimeError('Refined hand shape keys missing')
    animation=mesh.shape_keys.animation_data
    if animation and (animation.action or len(animation.drivers) or len(animation.nla_tracks)):
        raise RuntimeError('Animated hand keys need a separate explicit action merge; refuse hidden dependencies')
    result=[]
    for i,key in enumerate(mesh.shape_keys.key_blocks):
        name=prefix+'_key_'+str(i);values=_co(key.data).astype(np.float32);arrays[name]=values
        result.append({'name':key.name,'relative':key.relative_key.name,'value':float(key.value),'mute':key.mute,
                       'vertexGroup':key.vertex_group,'min':key.slider_min,'max':key.slider_max,
                       'array':name,'sha256':hashlib.sha256(values.tobytes()).hexdigest()})
    return result


def _mods(obj):
    result=[]
    for modifier in obj.modifiers:
        row={'name':modifier.name,'type':modifier.type,'viewport':modifier.show_viewport,'render':modifier.show_render}
        if modifier.type=='ARMATURE':
            row.update({'rig':modifier.object.name if modifier.object else None,
                        'vertexGroups':modifier.use_vertex_groups,'envelopes':modifier.use_bone_envelopes,
                        'preserveVolume':modifier.use_deform_preserve_volume,'vertexGroup':modifier.vertex_group})
        elif modifier.type=='SUBSURF':
            row.update({k:getattr(modifier,k) for k in ('subdivision_type','levels','render_levels','quality','uv_smooth','boundary_smooth','use_creases')})
        else:raise RuntimeError('Unreviewed hand modifier dependency: '+modifier.type)
        result.append(row)
    return result


def _material(mat):
    """Stable rendering inputs; omit UI positions, IDs and suffix-renamed names."""
    if not mat or not mat.use_nodes:raise RuntimeError('Expected node-based hand material')
    nodes=[]
    for node in mat.node_tree.nodes:
        row={'name':node.name,'type':node.bl_idname,'inputs':[]}
        for socket in node.inputs:
            value=getattr(socket,'default_value',None)
            if hasattr(value,'to_list'):value=value.to_list()
            elif value is not None and not isinstance(value,(str,int,float,bool)):
                try:value=list(value)
                except TypeError:raise RuntimeError('Unsupported node input in hand material')
            row['inputs'].append([socket.identifier,socket.is_linked,value])
        for name in ('attribute_name','uv_map','operation','blend_type','distribution','subsurface_method','data_type'):
            if hasattr(node,name):row[name]=getattr(node,name)
        image=getattr(node,'image',None)
        if image:
            row['image']={'size':list(image.size),'colorspace':image.colorspace_settings.name,
                          'path':image.filepath,'packedSha256':hashlib.sha256(bytes(image.packed_file.data)).hexdigest() if image.packed_file else None}
        if getattr(node,'node_tree',None):raise RuntimeError('Nested material node group needs explicit dependency fingerprint')
        nodes.append(row)
    return {'nodes':nodes,'links':[[l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier] for l in mat.node_tree.links]}


def _bone_names(rig):
    names=set()
    for bone in rig.data.bones:
        if any(bone.name.startswith(p) for p in ('leftHand','rightHand','leftThumb','rightThumb','leftIndex','rightIndex','leftMiddle','rightMiddle','leftRing','rightRing','leftLittle','rightLittle')):
            current=bone
            while current:names.add(current.name);current=current.parent
    return sorted(names)


def _bones(rig,names):
    return {name:{'parent':rig.data.bones[name].parent.name if rig.data.bones[name].parent else None,
                  'rest':np.asarray(rig.data.bones[name].matrix_local).tolist(),
                  'pose':np.asarray(rig.pose.bones[name].matrix).tolist()} for name in names}


def _source(bpy,relative,digest):
    path=_path(relative)
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=path or _sha(path)!=digest:
        raise RuntimeError('Exact source SHA in disposable background Object Mode required')
    if bpy.context.scene.frame_current!=1 or bpy.context.scene.frame_subframe!=0:raise RuntimeError('Keep current frame 1')
    return path


def capture(source_relative,source_sha,package_relative):
    import bpy
    source=_source(bpy,source_relative,source_sha);output=_path(package_relative);archive=output.with_suffix('.npz')
    if output.exists() or archive.exists():raise FileExistsError('Preserve previous hand packages')
    integrate,wrist,observe=_apis();recipe,baseline=integrate['_recipe']()
    body=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    if 'R11_wrist_overlap_clearance' not in scene or 'R11_natural_nail_finish' not in scene:
        raise RuntimeError('Source must contain actual wrist correction and nail finish')
    if rig.data.pose_position!='POSE':raise RuntimeError('Actual standing source pose required')
    key=body.data.shape_keys.key_blocks.get(WRIST_KEY)
    if key is None or key.relative_key!=body.data.shape_keys.key_blocks[0] or abs(key.value-1)>1e-7:
        raise RuntimeError('Actual active wrist key required')
    arrays={};base=_co(key.relative_key.data);delta=_co(key.data)-base;ids=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-8)
    if not len(ids) or not np.all((np.abs(base[ids,0])>.590)&(np.abs(base[ids,0])<.631)&(np.abs(base[ids,1])<.06)&(np.abs(base[ids,2]-1.235)<.06)):
        raise RuntimeError('Wrist key edits leave the independently observed wrist region')
    if np.linalg.norm(delta[ids],axis=1).max()>.0045:raise RuntimeError('Wrist delta exceeds reviewed bound')
    obs=json.loads(_path(wrist['OBS']).read_text(encoding='utf-8'))
    if _sha(_path(obs['arrays']))!=wrist['ARRAYS_SHA']:raise RuntimeError('Actual wrist observation changed')
    with np.load(_path(obs['arrays'])) as old:
        domain=np.unique(np.concatenate([old[s+'_raw_1_originalVertexIds'] for s in ('L','R')]))
    arrays['wrist_ids']=ids.astype(np.int32);arrays['wrist_delta']=delta[ids].astype(np.float32)
    arrays['body_domain']=domain;arrays['body_base']=base[domain].astype(np.float32)
    arrays['body_mix_without_wrist']=(wrist['_mix'](body)-delta)[domain].astype(np.float32)
    arrays['body_weights']=_weights(body)[domain]
    rows=[]
    with np.load(_path(recipe['payload'])) as original:
        for side,name in zip(('L','R'),HANDS):
            obj=bpy.data.objects[name];mesh=obj.data
            if np.max(np.abs(_co(mesh.vertices)-original['left' if side=='L' else 'right']))>2e-7:
                raise RuntimeError('Refined hand Basis no longer matches actual fitted asset: '+name)
            if not mesh.shape_keys or NAIL_KEY not in mesh.shape_keys.key_blocks or abs(mesh.shape_keys.key_blocks[NAIL_KEY].value-1)>1e-7:
                raise RuntimeError('Actual active nail finish key missing')
            if any(slot.link!='DATA' for slot in obj.material_slots):raise RuntimeError('Object material overrides need explicit handling')
            record={'object':name,'mesh':mesh.name,'side':side,'raw':_mesh_record(mesh,'hand_'+side,observe,arrays),
                    'keys':_keys(mesh,'hand_'+side,arrays),'groups':_groups(obj),'modifiers':_mods(obj),
                    'materials':[_material(m) for m in mesh.materials],
                    'parent':obj.parent.name if obj.parent else None,'parentType':obj.parent_type,'parentBone':obj.parent_bone,
                    'matrixWorld':np.asarray(obj.matrix_world).tolist(),'provenance':obj['R10_hbm_hand_provenance']}
            arrays['hand_'+side+'_weights']=_weights(obj);rows.append(record)
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as stream:np.savez_compressed(stream,**arrays)
    result={'schemaVersion':2,'source':source_relative,'sourceSha256':source_sha,'helperSha256':_sha(__file__),
            'dependencies':DEPS,'fittedHandPayloadSha256':PAYLOAD_SHA,'hands':rows,
            'wrist':{'key':WRIST_KEY,'value':1.,'body':'R01 Head','vertexCount':len(base),'groups':_groups(body),'changedCount':len(ids)},
            'bones':_bones(rig,_bone_names(rig)),'rigMatrix':np.asarray(rig.matrix_world).tolist(),
            'arrays':str(archive.relative_to(project_path())),'arraysSha256':_sha(archive),
            'sourceWristRecord':json.loads(scene['R11_wrist_overlap_clearance']),
            'sourceNailRecord':json.loads(scene['R11_natural_nail_finish']),
            'savedBlend':False,'appearanceAccepted':False,'scope':'Exact two refined hand meshes plus sparse wrist contribution; not old head/rig/groom.'}
    if _sha(source)!=source_sha:raise RuntimeError('Source changed during capture')
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return {'package':package_relative,'packageSha256':_sha(output),'arraysSha256':result['arraysSha256'],'sourceSha256':source_sha}


def _target_hand_compatible(obj,side,recipe,baseline,integrate):
    fit=integrate['_fit']();old=next(o for o in baseline['current'] if o['name']==obj.name)
    fit['_group_columns'](_groups(obj),old['vertexGroups'],recipe['weightTransfer']['sourceGroups'],'left' if side=='L' else 'right')
    if obj.data.shape_keys:raise RuntimeError('Target has existing hand shapes; preserve them rather than overwrite')
    digest=hashlib.sha256(_co(obj.data.vertices).astype(np.float32).tobytes()).hexdigest()
    if digest==recipe['sourceHandCoordinatesSha256'][obj.name]:
        with np.load(_path(baseline['arrays'])) as arrays:fit['_source_weights_unchanged'](obj,old,arrays)
        return 'original procedural target, exact observed geometry and weights'
    with np.load(_path(recipe['payload'])) as arrays:
        expected=arrays['left' if side=='L' else 'right']
        if _co(obj.data.vertices).shape!=expected.shape or not np.allclose(_co(obj.data.vertices),expected,atol=2e-7,rtol=0):
            raise RuntimeError('Target hand has unmerged geometry changes')
        names=recipe['weightTransfer']['sourceGroups'];expected_weights=np.column_stack([arrays['weights'][:,names.index(g.name.replace('right','left',1))] for g in obj.vertex_groups])
        fit['_installed_weights'](obj,expected_weights)
    if 'R10_hbm_hand_provenance' not in obj:raise RuntimeError('Expected explicit HBM target provenance')
    return 'unrefined HBM target, exact fitted geometry and weights'


def _check_rest_pose(rig,package):
    if not np.allclose(np.asarray(rig.matrix_world),package['rigMatrix'],atol=1e-7,rtol=0):raise RuntimeError('Rig world matrix differs')
    for name,row in package['bones'].items():
        bone=rig.data.bones.get(name)
        if bone is None or (bone.parent.name if bone.parent else None)!=row['parent']:
            raise RuntimeError('Target hand hierarchy differs: '+name)
        if not np.allclose(np.asarray(bone.matrix_local),row['rest'],atol=2e-7,rtol=0):raise RuntimeError('Target hand/arm rest differs: '+name)
        if not np.allclose(np.asarray(rig.pose.bones[name].matrix),row['pose'],atol=3e-6,rtol=0):raise RuntimeError('Target standing hand/arm pose differs: '+name)


def _pose_checks(bpy,body,rig,wrist):
    from mathutils import Matrix
    adapter=runpy.run_path(str(Path(__file__).with_name('hand_asset_pose_review_r11.py')))['_api']()
    frames={};snapshots={};rows=[]
    for side in ('L','R'):
        frame=adapter['_rest_frame'](bpy.data.objects['Hand '+side+' · continuous anatomy'],rig,side);frames[side]=frame
        snapshots[side]={name:{'basis':np.asarray(rig.pose.bones[name].matrix_basis).tolist(),'rotationMode':rig.pose.bones[name].rotation_mode} for name in frame['axes']}
    try:
        for pose in ('relaxed','wrist-flex','wrist-extend'):
            for side in ('L','R'):adapter['_apply_pose'](bpy,rig,frames[side],pose,snapshots[side])
            observed=[wrist['_actual_overlap'](bpy,body,side,adapter) for side in ('L','R')]
            rows.append({'pose':pose,'sides':observed})
        failed=[row for row in rows if any(r['outsideByThreeRayMajority'] or r['minimumSurfaceDistanceMeters']<.00015 for r in row['sides'])]
        if failed:
            raise RuntimeError('Integrated wrist/hand failed unchanged actual pose clearance: '+json.dumps(failed))
    finally:
        for snapshot in snapshots.values():
            for name,row in snapshot.items():
                bone=rig.pose.bones[name];bone.rotation_mode=row['rotationMode'];bone.matrix_basis=Matrix(row['basis'])
        bpy.context.view_layer.update()
        if any(not np.allclose(np.asarray(rig.pose.bones[name].matrix_basis),row['basis'],atol=2e-6,rtol=0) for snapshot in snapshots.values() for name,row in snapshot.items()):
            raise RuntimeError('Integrated wrist pose failed restoration')
    return rows


def _target_wrist_revision(bpy,body,key,package_relative,wrist,observe,plan_relative):
    """Replace only the added wrist contribution, using copied real-target data."""
    path=_path(plan_relative)
    if _sha(path)!=TARGET_WRIST_PLAN_SHA:raise RuntimeError('Frozen measured dense-target wrist plan changed')
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan['observedArraysSha256']!=TARGET_OBS_ARRAYS_SHA or plan['arraysSha256']!=TARGET_WRIST_ARRAYS_SHA:
        raise RuntimeError('Unreviewed dense-target wrist plan identity')
    if _sha(_path(plan['arrays']))!=plan['arraysSha256'] or _sha(_path(package_relative))!=plan['packageSha256']:
        raise RuntimeError('Target plan or original refined-hand package changed')
    parent_path=_path(PARENT_WRIST_PLAN)
    if _sha(parent_path)!=PARENT_WRIST_PLAN_SHA:raise RuntimeError('Accepted left wrist parent plan changed')
    parent=json.loads(parent_path.read_text(encoding='utf-8'))
    if _sha(_path(parent['arrays']))!=PARENT_WRIST_ARRAYS_SHA:raise RuntimeError('Accepted left wrist parent arrays changed')
    if plan['requiredActualGate']!={'minimumDistanceMeters':.00015,'outsideByThreeRayMajority':0,'poses':['relaxed','wrist-flex','wrist-extend']}:
        raise RuntimeError('Dense wrist plan changes the required actual gate')
    modifiers=[observe['_rna'](m) for m in body.modifiers]
    if modifiers!=plan['targetBodyModifiers'] or observe['_animation'](body)!=plan['targetBodyObjectAnimation']:
        raise RuntimeError('Current dense body modifiers/driver state differs from actual observation')
    if len(body.data.vertices)!=plan['targetBodyRawVertices'] or [{'name':g.name,'index':g.index} for g in body.vertex_groups]!=plan['targetBodyGroups']:
        raise RuntimeError('Current body cage/group indexing changed')
    basis=_co(key.relative_key.data);before_mix=wrist['_mix'](body);old_delta=_co(key.data)-basis
    weights=_weights(body);domain=[]
    with np.load(_path(plan['arrays'])) as arrays:
        with np.load(_path(parent['arrays'])) as previous:
            if any(not np.array_equal(arrays[name],previous[name]) for name in previous.files if name.startswith('L_')):
                raise RuntimeError('Right micro-correction changed an accepted left wrist array')
            extra=arrays['R_delta'].astype(np.float64)-previous['R_delta'].astype(np.float64)
            if np.max(np.linalg.norm(extra,axis=1))>.000300001 or np.any(extra[:,0]!=0):
                raise RuntimeError('Right-only refinement exceeds its 0.3 mm transverse limit')
            affected=np.flatnonzero(np.linalg.norm(extra,axis=1)>1e-10)
            if len(affected)!=38 or not np.array_equal(arrays['R_ids'][affected],arrays['rightLocalAffectedIds']):
                raise RuntimeError('Right correction support differs from the measured 38-point patch')
        for side in ('L','R'):
            ids=arrays[side+'_ids'];domain.extend(ids.tolist())
            if np.max(np.abs(basis[ids]-arrays[side+'_base']))>2e-7 or np.max(np.abs(before_mix[ids]-arrays[side+'_integratedMixBefore']))>2e-7:
                raise RuntimeError('Actual current wrist Basis/mix differs from measured target')
            if not np.allclose(weights[ids],arrays[side+'_weights'],rtol=0,atol=1e-7):raise RuntimeError('Actual target wrist weights changed')
        changed=arrays['changedIds'];delta=arrays['delta'].astype(np.float64)
    if len(changed)!=plan['totalChangedRawPoints'] or len(np.unique(changed))!=len(changed):raise RuntimeError('Invalid sparse wrist indices')
    p=basis[changed]
    if not np.all((np.abs(p[:,0])>.590)&(np.abs(p[:,0])<.631)&(np.abs(p[:,1])<.06)&(np.abs(p[:,2]-1.235)<.06)):
        raise RuntimeError('Dense wrist revision leaves the original local wrist region')
    if np.linalg.norm(delta,axis=1).max()>.0045 or np.any(delta[:,0]!=0):raise RuntimeError('Dense revision exceeds original bounded no-X-displacement rule')
    domain=np.unique(domain);outside=np.ones(len(basis),bool);outside[domain]=False
    if np.any(np.linalg.norm(old_delta[outside],axis=1)>1e-8):raise RuntimeError('Original wrist contribution extends outside observed domain')
    coordinates=basis.copy();coordinates[changed]+=delta;key.data.foreach_set('co',coordinates.ravel());key.value=1.
    bpy.context.view_layer.update();after_mix=wrist['_mix'](body)
    if np.max(np.abs(after_mix[outside]-before_mix[outside]))>1e-10:raise RuntimeError('Dense wrist revision moved an unrelated body/head coordinate')
    if [observe['_rna'](m) for m in body.modifiers]!=modifiers or not np.array_equal(_weights(body),weights):
        raise RuntimeError('Dense revision changed a body modifier or weight')
    return {'plan':plan_relative,'planSha256':_sha(path),'arraysSha256':plan['arraysSha256'],
            'observedTargetSource':plan['observedTargetSource'],'observedTargetSourceSha256':plan['observedTargetSourceSha256'],
            'actualTargetProbeArraysSha256':TARGET_OBS_ARRAYS_SHA,'changedRawPoints':len(changed),
            'maximumAdditionalCorrectionMeters':float(np.linalg.norm(after_mix-before_mix,axis=1).max()),
            'maximumTotalWristKeyMeters':float(np.linalg.norm(delta,axis=1).max()),
            'newKeySha256':hashlib.sha256(_co(key.data).astype(np.float32).tobytes()).hexdigest(),
            'bodySubdivisionModifiersAndWeightsUnchanged':True,'unrelatedBodyHeadCoordinatesExact':True,
            'leftWristParentArraysBitExact':True,'rightLocalRefinement':plan['rightLocalRefinement'],
            'inheritedGeometryBaselineObservation':plan['inheritedGeometryBaselineObservation'],
            'rightLocalObservation':plan['rightLocalObservation'],
            'measuredHandMarginMeters':plan['marginMeters'],'planGeometryDiagnostics':plan['sides'],
            'actualPoseGateUnchanged':plan['requiredActualGate']}


def prepare(source_relative,source_sha,package_relative,output_relative,wrist_plan=TARGET_WRIST_PLAN,check_only=False):
    import bpy
    source=_source(bpy,source_relative,source_sha);output=_path(output_relative);report=output.with_suffix('.refined-hands.json')
    if output==source or not output.is_relative_to(_path('work/reconstruction-r02')) or output.exists() or report.exists():
        raise RuntimeError('Choose a new independent reconstruction output')
    package_path=_path(package_relative);package=json.loads(package_path.read_text(encoding='utf-8'))
    donor=_path(package['source'])
    if package['schemaVersion']!=2 or _sha(donor)!=package['sourceSha256'] or _sha(_path(package['arrays']))!=package['arraysSha256']:
        raise RuntimeError('Refined-hand package/source identity changed')
    integrate,wrist,observe=_apis();recipe,baseline=integrate['_recipe']();fit=integrate['_fit']()
    scene=bpy.context.scene;body=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig']
    if MARKER in scene or WRIST_KEY in body.data.shape_keys.key_blocks:raise RuntimeError('Target already has refined wrist work; reconcile explicitly')
    if len([o for o in bpy.data.objects if o.type=='MESH' and o.data==body.data])!=1:
        raise RuntimeError('Target body mesh is shared by another object; avoid changing retained data')
    if not np.allclose(np.asarray(body.matrix_world),np.eye(4),atol=1e-7,rtol=0):raise RuntimeError('Target body coordinate space differs')
    _check_rest_pose(rig,package)
    hands=[bpy.data.objects[row['object']] for row in package['hands']];compatibility=[]
    for obj,row in zip(hands,package['hands']):
        compatibility.append(_target_hand_compatible(obj,row['side'],recipe,baseline,integrate))
        if _mods(obj)!=row['modifiers']:raise RuntimeError('Target hand modifier stack differs')
        if (obj.parent.name if obj.parent else None,obj.parent_type,obj.parent_bone)!=(row['parent'],row['parentType'],row['parentBone']):
            raise RuntimeError('Target hand parent space differs')
        if not np.allclose(np.asarray(obj.matrix_world),row['matrixWorld'],atol=1e-7,rtol=0):raise RuntimeError('Target hand world space differs')
    with np.load(_path(package['arrays'])) as arrays:
        domain=arrays['body_domain'];ids=arrays['wrist_ids'];delta=arrays['wrist_delta'];mix=wrist['_mix'](body)
        if len(body.data.vertices)!=package['wrist']['vertexCount'] or np.max(np.abs(mix[domain]-arrays['body_mix_without_wrist']))>2e-7:
            raise RuntimeError('Target wrist geometry differs from source baseline')
        if np.max(np.abs(_co(body.data.shape_keys.key_blocks[0].data)[domain]-arrays['body_base']))>2e-7:
            raise RuntimeError('Target wrist Basis differs')
        source_names=[r['name'] for r in package['wrist']['groups']];current_names=[g.name for g in body.vertex_groups]
        if set(source_names)!=set(current_names):raise RuntimeError('Target body bone groups differ')
        target_weights=_weights(body)[domain][:,[current_names.index(name) for name in source_names]]
        if not np.allclose(target_weights,arrays['body_weights'],atol=1e-7,rtol=0):raise RuntimeError('Target wrist weights changed')
    protected=integrate['_protected_objects'](bpy,hands);keys_before=wrist['_key_contract'](body);rig_before=fit['_rig_contract'](rig)
    original_data=[o.data for o in hands];original_props=[dict(o.items()) for o in hands]
    original_fake_users=[mesh.use_fake_user for mesh in original_data]
    all_ids=observe['_ids'](bpy);backups=[];key=None;success=False
    nails=[bpy.data.objects['Nail '+side+' '+finger+' · polished'] for side in ('L','R') for finger in ('Thumb','Index','Middle','Ring','Little')]
    nail_state={o.name:(o.hide_render,o.hide_viewport,o.hide_get()) for o in nails}
    try:
        with bpy.data.libraries.load(str(donor),link=False) as (available,requested):
            wanted=[row['mesh'] for row in package['hands']]
            if any(name not in available.meshes for name in wanted):raise RuntimeError('Package mesh names missing from exact donor')
            requested.meshes=wanted
        meshes=list(requested.meshes)
        if len(meshes)!=2 or any(m is None for m in meshes):raise RuntimeError('Failed exact two-mesh append')
        new_ids=[item for pointer,item in observe['_ids'](bpy).items() if pointer not in all_ids]
        if any(isinstance(item,(bpy.types.Object,bpy.types.Armature,bpy.types.Action)) for item in new_ids):
            raise RuntimeError('Mesh-only append unexpectedly pulled an object, rig or action')
        collection=bpy.data.collections.new('R11 retained pre-refinement hands');scene.collection.children.link(collection)
        collection.hide_render=True;collection.hide_viewport=True
        imported=[]
        with np.load(_path(package['arrays'])) as arrays:
            for obj,mesh,row,old in zip(hands,meshes,package['hands'],original_data):
                backup=obj.copy();backup.name='R11 retained before refined '+obj.name;collection.objects.link(backup)
                backup.hide_render=True;backup.hide_viewport=True;backups.append(backup);old.use_fake_user=True
                obj.data=mesh
                if not len(obj.vertex_groups):
                    for group in row['groups']:
                        created=obj.vertex_groups.new(name=group['name']);created.lock_weight=group['lockWeight']
                if _groups(obj)!=row['groups']:raise RuntimeError('Appended mesh group definitions differ from source')
                if not np.allclose(_weights(obj),arrays['hand_'+row['side']+'_weights'],atol=1e-7,rtol=0):
                    raise RuntimeError('Appended mesh deform weights differ from source')
                scratch={}
                if _mesh_record(mesh,'hand_'+row['side'],observe,scratch)!=row['raw']:raise RuntimeError('Appended mesh geometry/UV/attributes differ from exact source')
                if _keys(mesh,'hand_'+row['side'],scratch)!=row['keys']:raise RuntimeError('Appended shape keys differ')
                if [_material(m) for m in mesh.materials]!=row['materials']:raise RuntimeError('Appended material rendering inputs differ')
                obj['R10_hbm_hand_original_object']=backup.name;obj['R10_hbm_hand_provenance']=row['provenance']
                imported.append({'object':obj.name,'sourceMesh':row['mesh'],'installedMesh':mesh.name,'backupObject':backup.name,
                                 'rawGeometryUVAttributesKeysWeightsMaterialsExact':True})
        key=body.shape_key_add(name=WRIST_KEY,from_mix=False);key.relative_key=body.data.shape_keys.key_blocks[0]
        co=_co(key.relative_key.data);co[ids]+=delta;key.data.foreach_set('co',co.ravel());key.value=1.
        bpy.context.view_layer.update()
        target_revision=_target_wrist_revision(bpy,body,key,package_relative,wrist,observe,wrist_plan)
        for obj in nails:obj.hide_render=True;obj.hide_viewport=True;obj.hide_set(True)
        bpy.context.view_layer.update();pose_checks=_pose_checks(bpy,body,rig,wrist)
        if wrist['_key_contract'](body,(WRIST_KEY,))!=keys_before:raise RuntimeError('Existing body/face keys changed')
        after=fit['_rig_contract'](rig);pose_error=max(float(np.max(np.abs(np.asarray(a['pose'])-np.asarray(b['pose'])))) for a,b in zip(after['bones'],rig_before['bones']))
        for a,b in zip(after['bones'],rig_before['bones']):a['pose']=b['pose']
        if after!=rig_before or pose_error>2e-6:raise RuntimeError('Target rig/rest/action changed or pose failed restoration')
        current=integrate['_protected_objects'](bpy,hands)
        if any(current.get(pointer)!=value for pointer,value in protected.items()):raise RuntimeError('A protected target object/data/parent/transform changed')
        result={'source':source_relative,'sourceSha256':source_sha,'package':package_relative,'packageSha256':_sha(package_path),
                'refinedHandSource':package['source'],'refinedHandSourceSha256':package['sourceSha256'],'helperSha256':_sha(__file__),
                'dependencies':DEPS,'targetHandCompatibility':compatibility,'imported':imported,
                'wristChangedPoints':target_revision['changedRawPoints'],'targetDenseWristRevision':target_revision,
                'actualFiniteWristPoses':pose_checks,'restoredPoseMatrixError':pose_error,
                'originalNailVisibility':nail_state,'protectedTargetFaceGroomObjectsUnchanged':True,
                'targetRigActionsAndOriginalBodyKeysUnchanged':True,'masterChanged':False,'appearanceAccepted':False,
                'sourceWristRecord':package['sourceWristRecord'],'sourceNailRecord':package['sourceNailRecord'],
                'limits':['Actual integrated hand/wrist/nail images remain required.',
                          'The internal closed wrist overlap is retained; no welded wrist or all-pose certification.',
                          'Existing runtime hand bakes/exports require refresh after adoption.']}
        if _sha(source)!=source_sha or _sha(donor)!=package['sourceSha256']:raise RuntimeError('An input source changed on disk')
        if check_only:
            result.update({'savedBlend':False,'checkOnly':True,'actualDensePoseGatePassed':True})
            return result
        scene[MARKER]=json.dumps(result,ensure_ascii=False)
        bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False,relative_remap=True);success=True
        result.update({'candidate':output_relative,'candidateSha256':_sha(output)})
        with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        return result
    finally:
        if not success:
            if key and key.name in body.data.shape_keys.key_blocks:body.shape_key_remove(key)
            for obj,mesh,properties in zip(hands,original_data,original_props):
                obj.data=mesh
                for name in tuple(obj.keys()):
                    if name not in properties:del obj[name]
                for name,value in properties.items():obj[name]=value
            for mesh,fake_user in zip(original_data,original_fake_users):mesh.use_fake_user=fake_user
            for obj in nails:
                r,v,h=nail_state[obj.name];obj.hide_render=r;obj.hide_viewport=v;obj.hide_set(h)
            if MARKER in scene:del scene[MARKER]
            added=[item for pointer,item in observe['_ids'](bpy).items() if pointer not in all_ids]
            if added:bpy.data.batch_remove(ids=added)
            bpy.context.view_layer.update()


def check(source_relative,source_sha,package_relative,report_relative):
    """Actual install/poses followed by verified rollback; no Blender save."""
    import bpy
    source=_source(bpy,source_relative,source_sha);report=_path(report_relative)
    if report.exists():raise FileExistsError('Preserve previous check reports')
    integrate,wrist,observe=_apis();body=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig']
    before_ids=set(observe['_ids'](bpy));before_objects=integrate['_protected_objects'](bpy,[])
    before_visibility={o.as_pointer():(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}
    before_fake_users={m.as_pointer():m.use_fake_user for m in bpy.data.meshes}
    before_keys=wrist['_key_contract'](body);before_rig=integrate['_fit']()['_rig_contract'](rig)
    dummy=report.with_suffix('.never-save.blend')
    result=prepare(source_relative,source_sha,package_relative,str(dummy.relative_to(project_path())),check_only=True)
    if dummy.exists() or set(observe['_ids'](bpy))!=before_ids:raise RuntimeError('Check did not restore original ID set or unexpectedly saved')
    if integrate['_protected_objects'](bpy,[])!=before_objects or wrist['_key_contract'](body)!=before_keys:
        raise RuntimeError('Check did not restore original objects/body keys')
    if {o.as_pointer():(o.hide_render,o.hide_viewport,o.hide_get()) for o in bpy.data.objects}!=before_visibility:
        raise RuntimeError('Check did not restore original visibility')
    if {m.as_pointer():m.use_fake_user for m in bpy.data.meshes}!=before_fake_users:
        raise RuntimeError('Check did not restore original mesh retention flags')
    after_rig=integrate['_fit']()['_rig_contract'](rig)
    pose_error=max(float(np.max(np.abs(np.asarray(a['pose'])-np.asarray(b['pose'])))) for a,b in zip(after_rig['bones'],before_rig['bones']))
    for a,b in zip(after_rig['bones'],before_rig['bones']):a['pose']=b['pose']
    if after_rig!=before_rig or pose_error>2e-6:raise RuntimeError('Check did not restore target rig/pose')
    if _sha(source)!=source_sha:raise RuntimeError('Read-only check source changed on disk')
    result.update({'rollbackVerified':True,'rollbackPoseMaximumError':pose_error})
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return {'report':report_relative,'reportSha256':_sha(report),'savedBlend':False,'rollbackVerified':True,
            'poseMinima':[{row['pose']:{r['side']:{k:r[k] for k in ('samples','outsideByThreeRayMajority','minimumSurfaceDistanceMeters')} for r in row['sides']}} for row in result['actualFiniteWristPoses']]}


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if '--' in argv:argv=argv[argv.index('--')+1:]
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    read=sub.add_parser('capture');read.add_argument('--source',required=True);read.add_argument('--source-sha',required=True);read.add_argument('--package',required=True)
    build=sub.add_parser('prepare');build.add_argument('--source',required=True);build.add_argument('--source-sha',required=True);build.add_argument('--package',required=True);build.add_argument('--output',required=True)
    build.add_argument('--wrist-plan',default=TARGET_WRIST_PLAN)
    verify=sub.add_parser('check');verify.add_argument('--source',required=True);verify.add_argument('--source-sha',required=True);verify.add_argument('--package',required=True);verify.add_argument('--report',required=True)
    a=parser.parse_args(argv)
    if a.mode=='capture':result=capture(a.source,a.source_sha,a.package)
    elif a.mode=='check':result=check(a.source,a.source_sha,a.package,a.report)
    else:result=prepare(a.source,a.source_sha,a.package,a.output,a.wrist_plan)
    print('R11_REFINED_HAND_INTEGRATION '+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
