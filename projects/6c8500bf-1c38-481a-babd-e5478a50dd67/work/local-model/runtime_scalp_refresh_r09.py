"""Explicit native numerical payload -> one reversible runtime scalp swap.

No Blender launch, render, Blend save or dependency-library import occurs here.
prepare_payload() reads a saved production candidate. apply_payload() changes
only existing RT06 scalp card mesh/UV/material data in a saved runtime copy.
The other six hair meshes, tail bones/springs, face and body stay untouched.
"""
import hashlib,importlib.util,json,re,sys,types
from pathlib import Path
import numpy as np
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SOURCE='R06 continuous scalp flow';TARGET='RT06 Cards / '+SOURCE
MARKER='RT09_scalp_refresh_manifest';BACKUP='RT09_scalp_mesh_backup'
REGION='scalp';POLICY='append';PRIOR_DATA_KEY='R09_root_projection_original_data';PROPERTY_PREFIX='RT09_scalp'


def _api(name):
    spec=importlib.util.spec_from_file_location('scalp09_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def _path(relative,area):
    p=project_path();value=(p/relative).resolve();value.relative_to((p/area).resolve());return value


def _guard(area):
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Saved background Object Mode required')
    path=Path(bpy.data.filepath).resolve();path.relative_to((project_path()/area).resolve())
    if not path.is_file():raise RuntimeError('Saved source file is required')
    return path,bpy.data.objects['Character_Rig']


def _json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def prepare_payload(source_relative,stage='r09-scalp',_source_lineage=None):
    """Read appended native cohorts in their exact rest-rig coordinates.

    The original 6720-strand prefix is hashed separately and later checked
    against the existing runtime scalp provenance. No .blend is written.
    """
    source,rig=_guard('work/reconstruction-r02')
    expected=_path(source_relative,'work/reconstruction-r02')
    if source!=expected:raise RuntimeError('Open the exact explicit production candidate before preparing')
    if not re.fullmatch(r'[a-z0-9-]+',stage):raise ValueError('Simple explicit stage name required')
    folder=_path('work/runtime-r06/'+stage,'work/runtime-r06');folder.mkdir(exist_ok=False)
    cards=_api('runtime_haircards_r06');refresh=_api('runtime_face_refresh_r08')
    obj=bpy.data.objects[SOURCE];spec=cards.resolved_spec(obj)
    if POLICY=='append' and len(spec['groups'])<=len(cards.SPECS[SOURCE]['groups']):raise RuntimeError('Expected explicit appended R09 scalp cohorts')
    if POLICY not in ('append','replace_same_cohorts'):raise RuntimeError('Unknown regional transfer policy')
    source_sha=_sha(source);pose=rig.data.pose_position
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        paths,radii=cards._read(obj,rig)
        coordinates=np.concatenate(paths).astype(np.float64);radius=np.concatenate(radii).astype(np.float32)
        lengths=np.array([len(path) for path in paths],dtype=np.int32)
        old_count=sum(cards.SPECS[SOURCE]['groups']);old_points=int(lengths[:old_count].sum())
        prior_data={};previous_name=None;previous_sha=None;prior_motion=None
        if POLICY=='replace_same_cohorts' or (PRIOR_DATA_KEY and obj.get(PRIOR_DATA_KEY)):
            previous_name=obj.get(PRIOR_DATA_KEY,'');previous=bpy.data.hair_curves.get(previous_name)
            if previous is None:raise RuntimeError('Retained pre-edit native data required: '+str(PRIOR_DATA_KEY))
            proxy=types.SimpleNamespace(name=obj.name,type=obj.type,hide_render=False,modifiers=obj.modifiers,matrix_world=obj.matrix_world,data=previous)
            prior_paths,prior_radii=cards._read(proxy,rig)
            prior_lengths=np.array([len(path) for path in prior_paths],dtype=np.int32)
            if not np.array_equal(prior_lengths,lengths) or len(paths)!=sum(spec['groups']):raise RuntimeError('Regional replacement changed curve topology/cohorts')
            prior_coordinates=np.concatenate(prior_paths).astype(np.float64);prior_radius=np.concatenate(prior_radii).astype(np.float32)
            if not np.array_equal(prior_radius,radius):raise RuntimeError('Regional replacement unexpectedly changed source radii')
            prior_motion=float(np.max(np.linalg.norm(coordinates-prior_coordinates,axis=1)))
            if PRIOR_DATA_KEY=='R09_root_projection_original_data' and prior_motion>.00205:
                raise RuntimeError('Root-transported fibres exceed the documented 2 mm scope: '+str(prior_motion))
            prior_data={'prior_coordinates':prior_coordinates,'prior_radius':prior_radius,'prior_lengths':prior_lengths}
            # Appended scalp cohorts may be moved by the later documented root
            # clearance correction. Verify the OLD R06 prefix against retained
            # pre-correction data, not the newly lifted first 6720 fibres.
            previous_sha=hashlib.sha256(prior_coordinates[:old_points].astype(np.float32).tobytes()).hexdigest()
        payload=folder/('native-'+REGION+'.npz')
        with payload.open('xb') as stream:np.savez_compressed(stream,coordinates=coordinates,radius=radius,lengths=lengths,**prior_data)
        report={'schema':1,'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'sourceObject':SOURCE,'targetObject':TARGET,'sourceRig':refresh._rig_contract(rig),'region':REGION,'transferPolicy':POLICY,
            'spec':spec,'nativeCurves':len(paths),'nativePoints':len(coordinates),
            'originalNativeCurves':old_count,'originalNativePoints':old_points,
            'originalPrefixPointSha256':previous_sha or hashlib.sha256(coordinates[:old_points].astype(np.float32).tobytes()).hexdigest(),
            'retainedPreEditNativeData':previous_name,
            'sourceLayerLineage':_source_lineage,
            'maximumRetainedNativeMotionMeters':prior_motion,
            'allPointSha256':hashlib.sha256(coordinates.astype(np.float32).tobytes()).hexdigest(),
            'payload':str(payload.relative_to(project_path())),'payloadSha256':_sha(payload),
            'savedBlend':False,'sourceGeometryChanged':False,
            'rootSubstrateTransferred':False,'scope':'Only explicitly selected '+REGION+' native paths/radii; no root-shell, rig, face, cloth or spring transfer.'}
        if _sha(source)!=source_sha:raise RuntimeError('Production candidate changed on disk during payload preparation')
        _json(folder/'manifest.json',report);return report
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()


def _read_payload(manifest_relative):
    path=_path(manifest_relative,'work/runtime-r06');spec=json.loads(path.read_text(encoding='utf-8'))
    if spec.get('schema')!=1 or spec['sourceObject']!=SOURCE or spec['targetObject']!=TARGET or spec.get('region','scalp')!=REGION or spec.get('transferPolicy','append')!=POLICY:raise RuntimeError('Unrecognized regional payload')
    payload=_path(spec['payload'],'work/runtime-r06')
    if _sha(payload)!=spec['payloadSha256']:raise RuntimeError('Native scalp payload checksum mismatch')
    with np.load(payload,allow_pickle=False) as data:
        co=data['coordinates'];radius=data['radius'];lengths=data['lengths'];prior=None
        if spec.get('retainedPreEditNativeData'):
            pc=data['prior_coordinates'];pr=data['prior_radius'];pl=data['prior_lengths']
            if pc.shape!=co.shape or not np.array_equal(pl,lengths) or not np.array_equal(pr,radius) or not np.isfinite(pc).all():raise RuntimeError('Invalid pre-edit native payload')
            if hashlib.sha256(pc[:spec['originalNativePoints']].astype(np.float32).tobytes()).hexdigest()!=spec['originalPrefixPointSha256']:raise RuntimeError('Prior native fingerprint mismatch')
            offsets=np.r_[0,np.cumsum(pl)];prior=([pc[a:b] for a,b in zip(offsets,offsets[1:])],[pr[a:b] for a,b in zip(offsets,offsets[1:])])
    if co.ndim!=2 or co.shape[1]!=3 or radius.shape!=(len(co),) or lengths.ndim!=1 or lengths.dtype.kind not in 'iu':raise RuntimeError('Invalid native payload dimensions')
    if len(lengths)!=spec['nativeCurves'] or len(co)!=spec['nativePoints'] or int(lengths.sum())!=len(co):raise RuntimeError('Native payload count mismatch')
    if not np.isfinite(co).all() or not np.isfinite(radius).all() or np.any(radius<0) or np.any(lengths<4):raise RuntimeError('Invalid native payload values')
    if hashlib.sha256(co.astype(np.float32).tobytes()).hexdigest()!=spec['allPointSha256']:raise RuntimeError('Native point fingerprint mismatch')
    offsets=np.r_[0,np.cumsum(lengths)]
    return path,spec,[co[a:b] for a,b in zip(offsets,offsets[1:])],[radius[a:b] for a,b in zip(offsets,offsets[1:])],prior


def _untouched_state(target):
    # Direct datablock identity and slot identity prevent accidental replacement
    # without materializing all 21 facial morph coordinate arrays again.
    return {obj.name:(obj.data.as_pointer(),tuple(m.as_pointer() if m else 0 for m in obj.data.materials),
        tuple(float(v) for row in obj.matrix_world for v in row),obj.hide_render,obj.hide_get())
        for obj in bpy.context.scene.objects if obj.type=='MESH' and obj!=target}


def apply_payload(manifest_relative,_shared=None):
    """Replace one scalp card mesh; does not rebuild tail springs or all atlases."""
    source,rig=_guard('work/runtime-r06');manifest,spec,paths,radii,prior=_read_payload(manifest_relative)
    cards=_api('runtime_haircards_r06');refresh=_api('runtime_face_refresh_r08')
    obj=bpy.data.objects[TARGET]
    if obj.get(BACKUP) or obj.get(MARKER):raise RuntimeError('Scalp candidate already applied; restore first')
    if obj.type!='MESH' or obj.data.shape_keys or obj.data.users-int(obj.data.use_fake_user)!=1:raise RuntimeError('Single-user morph-free scalp mesh required')
    if [g.name for g in obj.vertex_groups]!=['head']:raise RuntimeError('Existing fixed scalp must have exactly one head group')
    armatures=[m for m in obj.modifiers if m.type=='ARMATURE']
    if len(obj.modifiers)!=1 or len(armatures)!=1 or armatures[0].object!=rig:raise RuntimeError('Unexpected scalp modifiers')
    if len(obj.data.materials)!=1 or not obj.data.materials[0].get('RT06_source_projected_atlas'):raise RuntimeError('Current source-projected scalp atlas is required')
    if obj.get('RT06_source_points_sha256')!=spec['originalPrefixPointSha256']:raise RuntimeError('Original R06 native scalp prefix differs from current runtime provenance')
    names=[bone['name'] for bone in spec['sourceRig']['data']['bones']]
    current=refresh._rig_contract(rig,names)
    if current['sha256']!=spec['sourceRig']['sha256']:
        diagnostic=refresh._record_rig_mismatch('r09-scalp',spec['sourceRig'],current)
        raise RuntimeError('Source/runtime rest rig differs: '+str(diagnostic))
    refresh._runtime_extra_bones(rig,names)
    # Start each attempt in a new report folder; previous failures are retained.
    folder=manifest.parent/'apply-001';index=1
    while folder.exists():index+=1;folder=manifest.parent/f'apply-{index:03d}'
    folder.mkdir();pose=rig.data.pose_position;original=obj.data;prepared=None
    rig_before=refresh._rig_contract(rig)
    old_hash=obj.get('RT06_source_points_sha256','');old_material=original.materials[0]
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();before=_untouched_state(obj)
        packet=cards.card_packet(paths,radii,spec['spec'])
        if _shared is not None:
            shared_packet=_shared['packets'][SOURCE]
            if not np.array_equal(packet['vertices'],shared_packet['vertices']) or packet['faces']!=shared_packet['faces']:
                raise RuntimeError('Shared regional atlas packet differs from checked payload geometry')
            packet=shared_packet
        co=np.empty(len(original.vertices)*3,np.float32);original.vertices.foreach_get('co',co);co=co.reshape(-1,3)
        matrix=np.asarray(rig.matrix_world.inverted()@obj.matrix_world)
        co=(co@matrix[:3,:3].T+matrix[:3,3]).astype(np.float32)
        expected_prior=cards.card_packet(prior[0],prior[1],spec['spec']) if prior else packet
        prefix=expected_prior['vertices'][:len(co)]
        if len(prefix)!=len(co) or not np.array_equal(prefix,co):
            error=float(np.max(np.linalg.norm(prefix-co,axis=1))) if len(prefix)==len(co) else None
            _json(folder/'prefix-mismatch.json',{'maximumPointDeltaMeters':error,'existingVertices':len(co),'newVertices':len(packet['vertices'])})
            raise RuntimeError('Original scalp card geometry prefix changed; see preserved prefix-mismatch.json')
        if _shared is None:materials,atlas=cards._projected_materials(folder, {SOURCE:packet})
        else:materials,atlas=_shared['materials'],_shared['atlas']
        material=materials[REGION];old_bs=next(n for n in old_material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        new_bs=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        constants={}
        for name in ('Roughness','Metallic','Specular IOR Level','Coat Weight','Coat Roughness'):
            if old_bs.inputs[name].is_linked:raise RuntimeError('Unexpected linked scalp scalar: '+name)
            value=float(old_bs.inputs[name].default_value);new_bs.inputs[name].default_value=value;constants[name]=value
        prepared=bpy.data.meshes.new(original.name+' / R09 '+REGION+' cohorts')
        # packet coordinates are rest rig-local. Preserve the existing object's
        # exact transform by converting them back to its local coordinates.
        inverse=np.linalg.inv(matrix);local=packet['vertices']@inverse[:3,:3].T+inverse[:3,3]
        prepared.from_pydata(local.tolist(),[],packet['faces']);prepared.update()
        uv=prepared.uv_layers.new(name='RuntimeHairUV')
        for face in prepared.polygons:
            face.use_smooth=True
            for li in face.loop_indices:uv.data[li].uv=packet['uv'][prepared.loops[li].vertex_index]
        prepared.materials.append(material);original.use_fake_user=True;old_material.use_fake_user=True
        obj.data=prepared
        # Blender 5 can carry group definitions on the mesh datablock. This
        # new from_pydata mesh must own exactly one definition regardless of
        # whether assigning obj.data retained the old object's definition.
        obj.vertex_groups.clear();head=obj.vertex_groups.new(name='head')
        head.add(list(range(len(prepared.vertices))),1,'REPLACE')
        if any(len(v.groups)!=1 or v.groups[0].group!=obj.vertex_groups['head'].index or abs(v.groups[0].weight-1)>1e-7 for v in prepared.vertices):raise RuntimeError('Fixed scalp weights were not assigned exactly')
        if _untouched_state(obj)!=before or refresh._rig_contract(rig)!=rig_before:raise RuntimeError('Non-scalp object or rig state changed')
        obj[BACKUP]=original.name;obj[MARKER]=str(manifest.relative_to(project_path()))
        obj[PROPERTY_PREFIX+'_prior_source_hash']=old_hash;obj['RT06_source_points_sha256']=spec['allPointSha256']
        obj['RT09_native_payload_override']=spec['payload']
        obj[PROPERTY_PREFIX+'_apply_report']=str((folder/'apply-report.json').relative_to(project_path()))
        result={'sourceRuntime':str(source.relative_to(project_path())),'nativeManifest':str(manifest.relative_to(project_path())),
            'object':obj.name,'retainedMesh':original.name,'newMesh':prepared.name,
            'oldVertices':len(original.vertices),'newVertices':len(prepared.vertices),'triangles':packet['triangles'],
            'groups':len(spec['spec']['groups']),'nativeCurves':len(paths),'priorCardGeometryExact':True,'region':REGION,'transferPolicy':POLICY,
            'sourceLayerLineage':spec.get('sourceLayerLineage'),
            'maximumChangedOldVertexMeters':float(np.max(np.linalg.norm(packet['vertices'][:len(co)]-co,axis=1))),
            'texture':atlas,'preservedPBRConstants':constants,'otherMeshDatablocksAndRigUnchanged':True,
            'rootSubstrateTransferred':False,'sourceNativeObjectInRuntime':'Historical native data retained hidden; provenance of new region is this checked numerical payload.',
            'savedBlend':False,'appearanceAccepted':False,
            'reviewRequired':'Actual front, side and posterior views with new opacity atlas; no posterior completeness guarantee from numerical transfer alone.'}
        _json(folder/'apply-report.json',result);prepared=None;return result
    except Exception:
        if obj.data!=original:obj.data=original
        obj['RT06_source_points_sha256']=old_hash
        for key in (BACKUP,MARKER,PROPERTY_PREFIX+'_prior_source_hash','RT09_native_payload_override',PROPERTY_PREFIX+'_apply_report'):
            if key in obj:del obj[key]
        raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        if prepared is not None and prepared.users==0:prepared.use_fake_user=True


def restore():
    _,rig=_guard('work/runtime-r06');obj=bpy.data.objects[TARGET]
    if not obj.get(BACKUP):return False
    old=bpy.data.meshes.get(obj[BACKUP])
    if old is None:raise RuntimeError('Retained scalp mesh missing')
    obj.data.use_fake_user=True;obj.data=old
    obj['RT06_source_points_sha256']=obj[PROPERTY_PREFIX+'_prior_source_hash']
    for key in (BACKUP,MARKER,PROPERTY_PREFIX+'_prior_source_hash','RT09_native_payload_override',PROPERTY_PREFIX+'_apply_report'):del obj[key]
    bpy.context.view_layer.update();return True


if __name__=='__main__':print('Loaded only. prepare_payload(explicit_source), then apply_payload(explicit_manifest) in separate source/runtime processes. Neither saves or renders.')
