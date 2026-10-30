"""Separate 301-frame authored full-body diagnostic from the saved R07 walk.

First run fullbody_payload_r07.extract() using Python 3.14+, without Blender.
Then, in an isolated background process opened on the selected current source
(default reference-face-flow-r08.blend, derived from the preserved R07 walk):
    audit_inputs()    # optional library-read comparison; never saves
    run()             # embeds old actual cloth + matching R07 hair; saves NEW copy
Reopen the new copy and call verify_reopened(). No operation runs on import.

The old cloth is replayed exactly. A static R04/current input audit is recorded;
changed collider weights/bone rest poses are not declared collision-equivalent.
No new solve, collision certification, live session or original-walk overwrite.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

P=project_path(); BASE=P/'work/reconstruction-r02'; FOLDER=Path(__file__).parent
SOURCE=BASE/'reference-face-flow-r08.blend'
OLD_SOURCE=BASE/'reference-cloth-body-r04.blend'
TARGET=BASE/'reference-combined-fullbody-r08.blend'
REPORT=BASE/'combined-fullbody-r08.json'
ACTION='R03 Full body review - authored diagnostic'
CONTROL='R02 Cloth Skirt control'
OBJECT='R07 Native Fullbody Cloth playback'
HAIR='work/reconstruction-r02/hair-batch-r07/R03-Full-body-review---authored-diagnostic.guides.npz'
FRAMES=tuple(range(1,302)); FRACTIONS=(1.375,49.5,150.25,228.75,300.5)
CHECK_FRAMES=(1,49,109,151,229,301)
PAYLOAD_API=runpy.run_path(str(FOLDER/'fullbody_payload_r07.py'))


def _sha(path): return PAYLOAD_API['sha'](path)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _work_file(relative, default, suffix):
    value=(P/relative).resolve() if relative is not None else default.resolve()
    value.relative_to((P/'work').resolve())
    if value.suffix!=suffix or value==(P/'work/character.blend').resolve():
        raise ValueError('Use a separate project work file of the expected type')
    return value


def _runtime(expected=SOURCE):
    import bpy
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=expected.resolve():
        raise RuntimeError('Open the exact independent source in a fresh background Blender process')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None: raise RuntimeError('Character_Rig missing')
    native=runpy.run_path(str(FOLDER/'native_hair_playback_r04.py'))
    api=runpy.run_path(str(FOLDER/'walk_cloth_r06.py'))
    return bpy,rig,native,api


def _mesh_snapshot(obj):
    mesh=obj.data
    points=np.empty(len(mesh.vertices)*3,dtype=np.float32); mesh.vertices.foreach_get('co',points)
    weights=[[(obj.vertex_groups[g.group].name,float(g.weight)) for g in v.groups if g.weight>0]
             for v in mesh.vertices]
    used=sorted({name for row in weights for name,weight in row})
    topology={'vertices':len(mesh.vertices),'faces':[list(p.vertices) for p in mesh.polygons],
              'materialIndices':[p.material_index for p in mesh.polygons]}
    return {'vertices':len(mesh.vertices),'topologySha256':_digest(topology),
            'coordinatesSha256':hashlib.sha256(points.tobytes()).hexdigest(),
            'weightsSha256':_digest(weights),'weightedGroups':used,
            'objectWorldMatrix':[list(row) for row in obj.matrix_world],
            'parentType':obj.parent_type,'parentBone':obj.parent_bone,
            'parentInverse':[list(row) for row in obj.matrix_parent_inverse],
            'uvSha256':_digest({uv.name:[list(v.uv) for v in uv.data] for uv in mesh.uv_layers}),
            'modifiers':[{'type':m.type,'armatureTarget':getattr(m,'object',None).type if getattr(m,'object',None) else None}
                         for m in obj.modifiers],
            'sourceObject':obj.get('r02_collision_source')}


def _rig_rest(rig):
    return {b.name:{'matrix':[list(row) for row in b.matrix_local],
                    'parent':b.parent.name if b.parent else None,'deform':b.use_deform}
            for b in rig.data.bones}


def _all_ids(bpy):
    buckets=('objects','meshes','armatures','materials','actions','collections','images',
             'node_groups','curves','hair_curves','shape_keys','texts','worlds','cameras','lights')
    return {block for bucket in buckets for block in getattr(bpy.data,bucket,[])}


def _load_requested_objects(bpy,library_file,requested_names):
    # Blender replaces the assigned destination list's strings in place with
    # loaded IDs on context exit. Keep immutable source-name keys separately.
    requested_names=tuple(requested_names)
    if len(set(requested_names))!=len(requested_names):
        raise RuntimeError('Duplicate requested library object names')
    with bpy.data.libraries.load(str(library_file),link=False) as (available,loaded):
        missing=[name for name in requested_names if name not in available.objects]
        if missing:raise RuntimeError('R04 source lacks requested objects: '+repr(missing))
        loaded.objects=list(requested_names)
    objects=tuple(loaded.objects)
    if len(objects)!=len(requested_names) or any(obj is None for obj in objects):
        raise RuntimeError('R04 library object load incomplete: '+repr(requested_names))
    return dict(zip(requested_names,objects))


def audit_inputs(source_relative=None):
    """Append old input IDs temporarily; inspect static data; remove only added IDs."""
    source_file=_work_file(source_relative,SOURCE,'.blend')
    bpy,rig,native,api=_runtime(source_file)
    old_report=json.loads((BASE/'cloth-dynamic-r04-report.json').read_text(encoding='utf-8'))
    names=[CONTROL]+[row['object'] for row in old_report['colliders']]
    if any(bpy.data.objects.get(name) is None for name in names):
        raise RuntimeError('A preserved R04 simulation control/collider is missing from this walk source')
    current={name:_mesh_snapshot(bpy.data.objects[name]) for name in names}
    current_rest=_rig_rest(rig)
    current_action=bpy.data.actions.get(ACTION)
    if current_action is None: raise RuntimeError('Preserved authored fullbody action missing')
    current_action_hash=native['_action_fingerprint'](current_action)
    before=_all_ids(bpy)
    try:
        requested_names=tuple(names)+('Character_Rig',)
        old_objects=_load_requested_objects(bpy,OLD_SOURCE,requested_names)
        old={name:_mesh_snapshot(old_objects[name]) for name in names}
        old_rig=old_objects['Character_Rig']; old_rest=_rig_rest(old_rig)
        old_action=old_rig.animation_data.action if old_rig.animation_data else None
        if old_action is None: raise RuntimeError('R04 source has no active body action')
        old_action_hash=native['_action_fingerprint'](old_action)
        changed_bones=sorted(name for name in set(old_rest)|set(current_rest)
                             if old_rest.get(name)!=current_rest.get(name))
        comparisons={}
        for name in names:
            affected=set(current[name]['weightedGroups'])|set(old[name]['weightedGroups'])
            for rest in (old_rest,current_rest):
                for bone_name in list(affected):
                    parent=rest.get(bone_name,{}).get('parent')
                    while parent:
                        affected.add(parent);parent=rest.get(parent,{}).get('parent')
            comparisons[name]={'old':old[name],'current':current[name],
                'geometryWeightsTopologyMatch':all(old[name][k]==current[name][k] for k in
                    ('coordinatesSha256','weightsSha256','topologySha256')),
                'objectTransformMatch':all(old[name][k]==current[name][k] for k in
                    ('objectWorldMatrix','parentType','parentBone','parentInverse')),
                'weightedBoneOrAncestorRestChanges':sorted(affected&set(changed_bones))}
        result={'oldSource':str(OLD_SOURCE.relative_to(P)),'oldSourceSha256':_sha(OLD_SOURCE),
                'currentSource':str(source_file.relative_to(P)),'currentSourceSha256':_sha(source_file),
                'bodyAction':ACTION,'oldActionName':old_action.name,
                'oldActionSha256':old_action_hash,'currentActionSha256':current_action_hash,
                'actionMatches':old_action_hash==current_action_hash,
                'rigObjectTransformMatches':bool(np.array_equal(np.array(old_rig.matrix_world),np.array(rig.matrix_world))),
                'changedRigRestBones':changed_bones,'objects':comparisons,
                'directFaceOrHeadColliderPresent':any('head' in name.lower() or 'face' in name.lower() for name in names[1:]),
                'scope':'Static geometry/weights/topology/weighted-bone-rest audit only. Does not prove evaluated collision equivalence, unchanged constraints or visual clearance.'}
        result['staticInputsMatch']=result['actionMatches'] and bool(result['rigObjectTransformMatches']) and all(row['geometryWeightsTopologyMatch'] and row['objectTransformMatch'] and
            not row['weightedBoneOrAncestorRestChanges'] for row in comparisons.values())
        result['collisionReviewRequired']=True
        return result
    finally:
        added=_all_ids(bpy)-before
        if added: bpy.data.batch_remove(ids=list(added))


def _payload():
    reference=json.loads(PAYLOAD_API['REFERENCE'].read_text(encoding='utf-8'))
    if reference['manifest']!=PAYLOAD_API['inventory'](): raise RuntimeError('R04 source payload changed')
    if _sha(PAYLOAD_API['PAYLOAD'])!=reference['payloadSha256']: raise RuntimeError('Decoded payload changed')
    values=np.fromfile(PAYLOAD_API['PAYLOAD'],dtype='<f4').reshape(301,2176,3)
    if not np.isfinite(values).all(): raise RuntimeError('Non-finite cloth payload')
    return reference,values


def _select(bpy,rig,action):
    rig.animation_data_create();rig.animation_data.action=action
    slots=[slot for slot in action.slots if slot.target_id_type=='OBJECT']
    if len(slots)!=1: raise RuntimeError('Expected one object slot')
    rig.animation_data.action_slot=slots[0];rig.data.pose_position='POSE'


def _parity(bpy,api,obj,values,frames=FRAMES+FRACTIONS):
    flags=[(m,m.show_viewport) for m in obj.modifiers];errors={}
    old=(bpy.context.scene.frame_current,bpy.context.scene.frame_subframe)
    try:
        for m,_ in flags:m.show_viewport=False
        for frame in frames:
            actual=np.array(api['_cloth_coordinates'](obj,(frame,))[str(frame)])
            lo=math.floor(frame)-1;hi=math.ceil(frame)-1;t=frame-math.floor(frame)
            expected=values[lo].astype(float)*(1-t)+values[hi].astype(float)*t
            if actual.shape!=expected.shape:raise RuntimeError('Cloth point order/count changed')
            errors[str(frame)]=float(np.abs(actual-expected).max())
    finally:
        for m,flag in flags:m.show_viewport=flag
        bpy.context.scene.frame_set(old[0],subframe=old[1]);bpy.context.view_layer.update()
    if max(errors.values())>2e-6:raise RuntimeError('Cloth payload parity failed: '+json.dumps(errors))
    return errors


def run(source_relative=None,target_relative=None,report_relative=None):
    source_file=_work_file(source_relative,SOURCE,'.blend')
    target_file=_work_file(target_relative,TARGET,'.blend')
    report_file=_work_file(report_relative,REPORT,'.json')
    bpy,rig,native,api=_runtime(source_file)
    from mathutils import Matrix
    if target_file.exists() or report_file.exists() or bpy.data.objects.get(OBJECT):
        raise RuntimeError('Existing fullbody output preserved')
    reference,world=_payload();audit=audit_inputs(source_file.relative_to(P).as_posix())
    if not audit['actionMatches']:raise RuntimeError('Body action differs from the actual R04 cloth action')
    waist=audit['objects'][CONTROL]
    if not audit['rigObjectTransformMatches'] or not waist['geometryWeightsTopologyMatch'] or not waist['objectTransformMatch'] or waist['weightedBoneOrAncestorRestChanges']:
        raise RuntimeError('Historical cloth waist/control inputs changed; do not silently replay an unmatched waistband: '+json.dumps(waist))
    control=bpy.data.objects[CONTROL]
    if control.data.shape_keys or len(control.data.vertices)!=2176:
        raise RuntimeError('Preserved R04 cloth control is not the expected raw mesh')
    if audit['objects'][CONTROL]['old']['topologySha256']!=audit['objects'][CONTROL]['current']['topologySha256']:
        raise RuntimeError('Payload point order cannot be mapped onto changed control topology')
    if [m.type for m in control.modifiers]!=['ARMATURE','CLOTH','SUBSURF','SOLIDIFY']:
        raise RuntimeError('Unexpected preserved simulation modifier stack')
    _,hair_meta,_,_,_=native['load_cache'](HAIR)
    if hair_meta['actionSha256']!=audit['currentActionSha256']:
        raise RuntimeError('R07 hair cache does not match this authored action')
    # From this point changes exist only in this isolated process; save only a
    # new target after parity. Failure must exit without saving the source.
    if native['RIG_GUIDE'] in rig:native['restore']()
    preserved_visibility={}
    for obj in bpy.context.scene.objects:
        if any(m.type=='CLOTH' for m in obj.modifiers) or obj.name in {
            'R07 Native Walk Cloth playback','Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam'}:
            preserved_visibility[obj.name]={'viewport':obj.hide_viewport,'render':obj.hide_render,'hideSet':obj.hide_get()}
            obj.hide_viewport=True;obj.hide_render=True;obj.hide_set(True)
    _select(bpy,rig,bpy.data.actions[ACTION])
    keys=bpy.data.objects['R01 Head'].data.shape_keys
    face_action=keys.animation_data.action if keys.animation_data else None
    if face_action:face_action.use_fake_user=True;keys.animation_data.action=None
    for name in ('R02_blink_L','R02_blink_R'):
        if name in keys.key_blocks:keys.key_blocks[name].value=0.
    bpy.context.scene.render.fps=30;bpy.context.scene.render.fps_base=1.
    rig_matrices=[]
    for frame in FRAMES:
        api['_set_frame'](frame);rig_matrices.append(np.array(rig.matrix_world))
    world_change=float(np.max(np.abs(np.array(rig_matrices)-rig_matrices[0])))
    if world_change>1e-8:raise RuntimeError('Rig object matrix changes; world-space cloth embedding requires adaptation')
    inverse=np.linalg.inv(rig_matrices[0]); local=world@inverse[:3,:3].T+inverse[:3,3]
    mesh=control.data.copy();mesh.name='R07 fullbody simulated cloth payload'
    obj=bpy.data.objects.new(OBJECT,mesh);bpy.context.scene.collection.objects.link(obj)
    obj.parent=rig;obj.parent_type='OBJECT';obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    for group in control.vertex_groups:obj.vertex_groups.new(name=group.name)
    for i,slot in enumerate(control.material_slots):obj.material_slots[i].link=slot.link;obj.material_slots[i].material=slot.material
    mesh.vertices.foreach_set('co',np.ascontiguousarray(local[0],dtype=np.float32).ravel());mesh.update()
    basis=obj.shape_key_add(name='Basis',from_mix=False)
    for i,frame in enumerate(FRAMES):
        key=obj.shape_key_add(name=f'R07 fullbody simulated frame {frame:03d}',from_mix=False)
        key.relative_key=basis;key.data.foreach_set('co',np.ascontiguousarray(local[i],dtype=np.float32).ravel())
        for when,value in ((frame-1,0.),(frame,1.),(frame+1,0.)):
            if 1<=when<=301:key.value=value;key.keyframe_insert(data_path='value',frame=when)
        key.value=0.
    shape_action=mesh.shape_keys.animation_data.action;shape_action.name='R07 Cloth actual payload - authored fullbody 1-301';shape_action.use_fake_user=True
    for curve in native['_fcurves'](shape_action):
        curve.extrapolation='CONSTANT'
        for point in curve.keyframe_points:point.interpolation='LINEAR'
        curve.update()
    for source in control.modifiers:
        if source.type in {'SUBSURF','SOLIDIFY'}:
            target=obj.modifiers.new(source.name,source.type);api['_copy_settings'](source,target,{'object'})
    obj.hide_viewport=False;obj.hide_render=False;obj.hide_set(False)
    errors=_parity(bpy,api,obj,world)
    api['_set_frame'](1)
    native['run'](HAIR);native['enable_for_action'](select=True,frame=1)
    hair_parity=native['validate_evaluated'](HAIR,frames=[1,49.5,109,151,229,301])
    surface=api['_cloth_coordinates'](obj,CHECK_FRAMES);api['_set_frame'](1)
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=301
    scene.use_preview_range=True;scene.frame_preview_start=1;scene.frame_preview_end=301
    result={'source':str(source_file.relative_to(P)),'sourceSha256':_sha(source_file),'target':str(target_file.relative_to(P)),
            'bodyAction':ACTION,'bodyActionSha256':audit['currentActionSha256'],'bodyProvenance':'Authored diagnostic, not motion capture',
            'clothObject':OBJECT,'payloadReference':str(PAYLOAD_API['REFERENCE'].relative_to(P)),
            'payloadReferenceSha256':_sha(PAYLOAD_API['REFERENCE']),'clothPayloadSha256':reference['payloadSha256'],
            'payloadManifest':reference['manifest'],'clothErrors':errors,'maximumClothErrorMeters':max(errors.values()),
            'clothShapeActionSha256':native['_action_fingerprint'](shape_action),'clothShapeKeys':302,
            'rigObjectMatrixChange':world_change,'staticSimulationInputAudit':audit,
            'surfaceReferences':surface,'hairCache':HAIR,'hairParity':hair_parity,
            'preservedVisibilityBefore':preserved_visibility,'preservedDetachedFaceAction':face_action.name if face_action else None,
            'playbackDependency':'Embedded native Blender shape keys and hair Geometry Nodes; no external cloth or hair payload needed for playback. Reference files are used only by QA helpers. No VRM equivalence claimed.',
            'scope':'301-frame historical native cloth payload + matching R07 hair, with current face/neck/groom retained. Linear cloth interpolation between samples. Original walk source remains separate.',
            'pending':['Fresh-file parity/reopen','Current body/cloth and hair contact review','Continuous rendered review','No collision-equivalence or client acceptance claimed']}
    obj['r07_fullbody_playback_report']=json.dumps(result)
    text=bpy.data.texts.new('R07 Fullbody cloth provenance.json')
    text.write(json.dumps({k:v for k,v in result.items() if k not in {'surfaceReferences','clothErrors'}},indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(target_file))
    result['savedSha256']=_sha(target_file)
    report_file.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'target':str(target_file),'maximumClothErrorMeters':max(errors.values()),
                      'staticInputsMatch':audit['staticInputsMatch'],'reopenRequired':True}))
    return result


def verify_reopened(expected_relative=None,report_relative=None):
    expected=_work_file(expected_relative,TARGET,'.blend')
    bpy,rig,native,api=_runtime(expected)
    _,world=_payload();obj=bpy.data.objects[OBJECT];report=json.loads(obj['r07_fullbody_playback_report'])
    if rig.animation_data.action.name!=ACTION or native['_action_fingerprint'](rig.animation_data.action)!=report['bodyActionSha256']:
        raise RuntimeError('Fullbody action changed')
    if _sha(PAYLOAD_API['REFERENCE'])!=report['payloadReferenceSha256']:
        raise RuntimeError('Validation payload reference changed')
    if native['_action_fingerprint'](obj.data.shape_keys.animation_data.action)!=report['clothShapeActionSha256']:
        raise RuntimeError('Cloth shape animation changed')
    if obj.hide_viewport or obj.hide_render:raise RuntimeError('Native fullbody cloth became hidden')
    for name in report['preservedVisibilityBefore']:
        preserved=bpy.data.objects.get(name)
        if preserved is None or not preserved.hide_viewport or not preserved.hide_render:
            raise RuntimeError('A preserved simulation or duplicate skirt became visible: '+name)
    if [m.type for m in obj.modifiers]!=['SUBSURF','SOLIDIFY']:raise RuntimeError('Unexpected playback modifier stack')
    errors=_parity(bpy,api,obj,world)
    surface_error=api['_coordinate_error'](report['surfaceReferences'],api['_cloth_coordinates'](obj,CHECK_FRAMES))
    if surface_error>2e-6:raise RuntimeError('Reopened visible cloth surface differs')
    hair=native['validate_evaluated'](HAIR,frames=[1,49.5,151,229,301]);api['_set_frame'](1)
    result={'source':str(expected.relative_to(P)),'sha256':_sha(expected),'rawClothMaximumErrorMeters':max(errors.values()),
            'surfaceMaximumErrorMeters':surface_error,'hair':hair,'staticInputsMatch':report['staticSimulationInputAudit']['staticInputsMatch'],
            'scope':'Reopen numerical parity only. Historical collider/input differences and fresh visual/contact review remain separate.'}
    output=_work_file(report_relative,BASE/'combined-fullbody-reopen-r08.json','.json')
    if output.exists():raise RuntimeError('Existing reopen report preserved')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    print('Loaded only. Use audit_inputs(), run(), and fresh-file verify_reopened() explicitly.')
