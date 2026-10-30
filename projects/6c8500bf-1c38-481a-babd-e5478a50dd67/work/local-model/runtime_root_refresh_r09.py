"""Transfer the documented PN root shell; deliberately invalidate old maps.

Source and runtime remain separate. The library contains one parentless mesh,
its original material graph and all source UV/attributes, never an armature.
Only runtime root data/materials are swapped. Its object, binding and rest rig
stay fixed. A NEW actual root bake is mandatory before export.
"""
import importlib.util,json,re,sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
NAME='R06 attached scalp root substrate';KEY='R09_root_projection_original_data'
PAYLOAD='RT09 isolated root payload';MARKER='RT09_root_refresh_manifest';BACKUP='RT09_root_mesh_backup'


def _api(name):
    s=importlib.util.spec_from_file_location('root09_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def _base():return _api('runtime_scalp_refresh_r09')


def _co(mesh):
    values=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',values);return values.reshape(-1,3)


def _topology(mesh):return [(int(p.material_index),tuple(p.vertices)) for p in mesh.polygons]


def _attrs(mesh):
    api=_api('runtime_torso_cull_r07')
    return {a.name:(a.domain,a.data_type,api._attribute(a).copy()) for a in mesh.attributes
        if a.name not in {'position','.edge_verts','.corner_vert','.corner_edge'} and not a.name.startswith(('.select','.vs.','.es.'))}


def _same_attrs(left,right):
    return set(left)==set(right) and all(left[n][:2]==right[n][:2] and np.array_equal(left[n][2],right[n][2]) for n in left)


def prepare_payload(source_relative,stage='r09-root'):
    api=_base();source,rig=api._guard('work/reconstruction-r02')
    if source!=api._path(source_relative,'work/reconstruction-r02'):raise RuntimeError('Explicit source not open')
    if not re.fullmatch(r'[a-z0-9-]+',stage):raise ValueError('Simple stage name required')
    obj=bpy.data.objects[NAME];prior=bpy.data.meshes.get(obj.get(KEY,''))
    if obj.type!='MESH' or obj.data.shape_keys or prior is None:raise RuntimeError('Documented retained pre-PN root mesh is required')
    if obj.constraints or any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Root requires original-to-evaluated mapping before transfer')
    if len(prior.vertices)!=len(obj.data.vertices) or _topology(prior)!=_topology(obj.data):raise RuntimeError('PN root changed topology/material domains')
    if not _same_attrs(_attrs(prior),_attrs(obj.data)):raise RuntimeError('PN root changed source UV/alpha/custom attributes')
    if any(m is None or m.get('r06_baked_material') for m in obj.data.materials):raise RuntimeError('Prepare the physical source root shader, not previous runtime maps')
    # This scene may contain motion. A rest-matrix snapshot is required, not an
    # animated object matrix from the current frame.
    folder=api._path('work/runtime-r06/'+stage,'work/runtime-r06');folder.mkdir(exist_ok=False)
    old_pose=rig.data.pose_position;source_sha=api._sha(source);temp=None
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        frame=rig.matrix_world.inverted()@obj.matrix_world
        pre=_co(prior).astype(np.float64);now=_co(obj.data).astype(np.float64)
        matrix=np.asarray(frame);pre=pre@matrix[:3,:3].T+matrix[:3,3];now=now@matrix[:3,:3].T+matrix[:3,3]
        maximum=float(np.max(np.linalg.norm(now-pre,axis=1)))
        if maximum>.00205:raise RuntimeError('Root motion exceeds documented 2 mm PN scope: '+str(maximum))
        with (folder/'prior-root.npz').open('xb') as stream:np.savez_compressed(stream,priorRigLocal=pre)
        temp=obj.copy();temp.data=obj.data.copy();temp.name=PAYLOAD
        world=obj.matrix_world.copy();temp.parent=None;temp.matrix_world=world;temp.animation_data_clear()
        for m in list(temp.modifiers):temp.modifiers.remove(m)
        temp.vertex_groups.clear();temp.hide_render=False;temp.hide_viewport=False
        refresh=_api('runtime_face_refresh_r08');contract=refresh._mesh_contract(temp)
        materials=[refresh._material_contract(m) for m in temp.data.materials]
        payload=folder/'root-payload.blend';bpy.data.libraries.write(str(payload),{temp},path_remap='RELATIVE',fake_user=True,compress=True)
        with bpy.data.libraries.load(str(payload),link=False) as (available,requested):
            if set(available.objects)!={temp.name}:raise RuntimeError('Root payload contains object dependencies')
        report={'schema':1,'source':source_relative,'sourceSha256':source_sha,'sourceRig':refresh._rig_contract(rig),
            'sourceObject':NAME,'payloadObject':temp.name,'mesh':contract,'materialHashes':materials,
            'payload':str(payload.relative_to(project_path())),'payloadSha256':api._sha(payload),
            'priorPayload':str((folder/'prior-root.npz').relative_to(project_path())),'priorPayloadSha256':api._sha(folder/'prior-root.npz'),
            'maximumSourceMotionMeters':maximum,'sourceUVAlphaAttributesExact':True,'retainedPrePNMesh':prior.name,
            'rootBakeRequired':True,'oldNormalMapReusable':False,'savedBlend':False}
        if api._sha(source)!=source_sha:raise RuntimeError('Production file changed during root preparation')
        api._json(folder/'manifest.json',report);return report
    finally:
        if temp:
            mesh=temp.data;bpy.data.objects.remove(temp,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        rig.data.pose_position=old_pose;bpy.context.view_layer.update()


def apply_payload(manifest_relative):
    api=_base();source,rig=api._guard('work/runtime-r06')
    manifest=api._path(manifest_relative,'work/runtime-r06');spec=json.loads(manifest.read_text(encoding='utf-8'))
    if spec.get('schema')!=1 or spec['sourceObject']!=NAME:raise RuntimeError('Wrong root payload')
    payload=api._path(spec['payload'],'work/runtime-r06');prior_path=api._path(spec['priorPayload'],'work/runtime-r06')
    if api._sha(payload)!=spec['payloadSha256'] or api._sha(prior_path)!=spec['priorPayloadSha256']:raise RuntimeError('Root payload checksum mismatch')
    refresh=_api('runtime_face_refresh_r08');names=[r['name'] for r in spec['sourceRig']['data']['bones']]
    current=refresh._rig_contract(rig,names)
    if current['sha256']!=spec['sourceRig']['sha256']:
        path=refresh._record_rig_mismatch('r09-root',spec['sourceRig'],current);raise RuntimeError('Root rest rig mismatch: '+str(path))
    refresh._runtime_extra_bones(rig,names);obj=bpy.data.objects[NAME]
    if obj.get(MARKER) or obj.get(BACKUP):raise RuntimeError('Root already refreshed')
    if obj.data.users-int(obj.data.use_fake_user)!=1 or obj.data.shape_keys or [g.name for g in obj.vertex_groups]!=['head']:raise RuntimeError('Expected one fixed head-weighted runtime root')
    if len(obj.modifiers)!=1 or obj.modifiers[0].type!='ARMATURE' or obj.modifiers[0].object!=rig:raise RuntimeError('Unexpected runtime root modifiers')
    pose=rig.data.pose_position;old=obj.data;imported=None;complete=False
    ids={o.as_pointer() for o in bpy.data.objects if o.type=='ARMATURE'}
    folder=manifest.parent/'apply-001';index=1
    while folder.exists():index+=1;folder=manifest.parent/f'apply-{index:03d}'
    folder.mkdir()
    oldprops={k:obj[k] for k in ('r06_material_bake_source_sha256','r06_material_originals','r06_material_bake_domains') if k in obj}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();before=api._untouched_state(obj);rig_before=refresh._rig_contract(rig)
        matrix=np.asarray(rig.matrix_world.inverted()@obj.matrix_world);co=_co(old).astype(np.float64);co=co@matrix[:3,:3].T+matrix[:3,3]
        with np.load(prior_path,allow_pickle=False) as data:prior=data['priorRigLocal']
        if co.shape!=prior.shape:raise RuntimeError('Runtime root no longer matches retained pre-PN vertex count')
        old_error=float(np.max(np.linalg.norm(co-prior,axis=1)))
        # Only float32 mesh-transform roundoff is admitted (1 micrometre).
        # The actual delta is reported; this is not a rig fingerprint waiver.
        if old_error>1e-6:raise RuntimeError('Existing root differs from retained pre-PN geometry: '+str(old_error))
        expected=(spec['payloadObject'],)
        with bpy.data.libraries.load(str(payload),link=False) as (available,requested):requested.objects=list(expected)
        imported=requested.objects[0]
        if imported is None or {o.as_pointer() for o in bpy.data.objects if o.type=='ARMATURE'}!=ids:raise RuntimeError('Root import failed or appended rig')
        actual=refresh._mesh_contract(imported);actual['materials']=spec['mesh']['materials']
        if actual!=spec['mesh'] or [refresh._material_contract(m) for m in imported.data.materials]!=spec['materialHashes']:raise RuntimeError('Imported root attributes/geometry/material graph changed')
        if imported.parent or imported.modifiers or imported.constraints:raise RuntimeError('Root payload retained external dependency')
        if _topology(imported.data)!=_topology(old):raise RuntimeError('Runtime root topology or material domains changed')
        mesh=imported.data;attributes_before=_attrs(mesh)
        transform=obj.matrix_world.inverted()@Matrix(spec['mesh']['matrixWorld'])
        mesh.transform(transform);mesh.update()
        if not _same_attrs(attributes_before,_attrs(mesh)):raise RuntimeError('Root transform changed preserved UV/alpha/custom attributes')
        old.use_fake_user=True
        for m in old.materials:m.use_fake_user=True
        obj.data=mesh;obj.vertex_groups.clear();group=obj.vertex_groups.new(name='head');group.add(list(range(len(mesh.vertices))),1,'REPLACE')
        if any(len(v.groups)!=1 or abs(v.groups[0].weight-1)>1e-7 for v in mesh.vertices):raise RuntimeError('Root head weights failed')
        for key in oldprops:del obj[key]
        obj[BACKUP]=old.name;obj[MARKER]=manifest_relative;obj['RT09_root_old_bake_properties']=json.dumps(oldprops)
        obj['RT09_root_bake_required']=True;obj['RT09_root_geometry_source_sha256']=spec['sourceSha256']
        if api._untouched_state(obj)!=before or refresh._rig_contract(rig)!=rig_before:raise RuntimeError('Non-root mesh or rig changed')
        result={'sourceRuntime':str(source.relative_to(project_path())),'manifest':manifest_relative,'productionSource':spec['source'],
            'productionSourceSha256':spec['sourceSha256'],'oldRootMaximumDeltaMeters':old_error,
            'newSourceMaximumMotionMeters':spec['maximumSourceMotionMeters'],'retainedMesh':old.name,
            'vertices':len(mesh.vertices),'sourceUVAlphaAttributesPreserved':True,'bindingPreserved':'fixed head, weight 1',
            'oldBakedRootMapsDiscardedFromCurrentRoot':True,'rootBakeRequired':True,'oldNormalMapReusable':False,
            'savedBlend':False,'appearanceAccepted':False}
        api._json(folder/'apply-report.json',result);complete=True;return result
    finally:
        if not complete:
            if obj.data!=old:obj.data=old
            for key,value in oldprops.items():obj[key]=value
            for key in (BACKUP,MARKER,'RT09_root_old_bake_properties','RT09_root_bake_required','RT09_root_geometry_source_sha256','RT09_root_baked_geometry_source_sha256'):
                if key in obj:del obj[key]
        if imported:bpy.data.objects.remove(imported,do_unlink=True)
        rig.data.pose_position=pose;bpy.context.view_layer.update()


def restore():
    api=_base();api._guard('work/runtime-r06');obj=bpy.data.objects[NAME]
    if not obj.get(BACKUP):return False
    old=bpy.data.meshes.get(obj[BACKUP])
    if old is None:raise RuntimeError('Retained runtime root mesh missing')
    obj.data.use_fake_user=True;obj.data=old
    for key in ('r06_material_bake_source_sha256','r06_material_originals','r06_material_bake_domains'):
        if key in obj:del obj[key]
    for key,value in json.loads(obj['RT09_root_old_bake_properties']).items():obj[key]=value
    for key in (BACKUP,MARKER,'RT09_root_old_bake_properties','RT09_root_bake_required','RT09_root_geometry_source_sha256','RT09_root_baked_geometry_source_sha256'):
        if key in obj:del obj[key]
    bpy.context.view_layer.update();return True


if __name__=='__main__':print('Loaded only. prepare_payload/apply_payload/restore. Re-bake root before exporting.')
