"""Evaluate preserved R04 and current fullbody collider vertices side by side.

Use only in a fresh background Blender opened on the independent fullbody copy:
    report = run()
    report = run(report_relative='work/reconstruction-r02/fullbody-colliders-r08.json')

No execution on import, source save, solve, bake, render, or pass certificate.
Library IDs are temporary; old/current constraints and drivers remain active.
The lower-body mask is frozen in OLD rig-local REST coordinates, never animated
world Z. Six sampled frames measure input differences, not skirt clearance.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

P=project_path();FOLDER=Path(__file__).parent;BASE=P/'work/reconstruction-r02'
SOURCE=BASE/'reference-combined-fullbody-r08.blend'
OLD_SOURCE=BASE/'reference-cloth-body-r04.blend'
FRAMES=(1,49,109,151,229,301)
ACTION='R03 Full body review - authored diagnostic'
EXPECTED_ACTION_SHA='df1f4bc9ceaadb069cfced3585a3564dde22b1878a1b3d6cd01fe69dd62b619c'
IK=('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R')
REST_Z_MAX=1.05


def _api(name):return runpy.run_path(str(FOLDER/(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _path(relative,default,suffix):
    path=(P/relative).resolve() if relative is not None else default.resolve()
    path.relative_to(BASE.resolve())
    if path.suffix!=suffix:raise ValueError('Unexpected audit path suffix')
    return path


def _coordinates(mesh):
    a=np.empty(len(mesh.vertices)*3,dtype=np.float32);mesh.vertices.foreach_get('co',a)
    return a.reshape(-1,3).astype(np.float64)


def _topology(mesh):
    loops=np.empty(len(mesh.loops),dtype=np.int32);mesh.loops.foreach_get('vertex_index',loops)
    starts=np.empty(len(mesh.polygons),dtype=np.int32);mesh.polygons.foreach_get('loop_start',starts)
    counts=np.empty(len(mesh.polygons),dtype=np.int32);mesh.polygons.foreach_get('loop_total',counts)
    return {'vertices':len(mesh.vertices),'loops':_array_sha(loops),'starts':_array_sha(starts),'counts':_array_sha(counts)}


def _world(co,matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    return co@matrix[:3,:3].T+matrix[:3,3]


def _drivers(block):
    data=getattr(block,'animation_data',None)
    return tuple(data.drivers) if data else ()


def _constraints(obj):
    rows=[('object',constraint) for constraint in obj.constraints]
    if obj.type=='ARMATURE':
        rows.extend((bone.name,constraint) for bone in obj.pose.bones for constraint in bone.constraints)
    return rows


def _object_dependencies(bpy,obj):
    """Only actual evaluation targets, not custom-shape display objects."""
    found=set()
    if obj.parent:found.add(obj.parent)
    for block in [m for m in obj.modifiers]+[c for _,c in _constraints(obj)]:
        for prop in block.bl_rna.properties:
            if prop.identifier!='rna_type' and prop.type=='POINTER':
                target=getattr(block,prop.identifier)
                if isinstance(target,bpy.types.Object):found.add(target)
        for entry in getattr(block,'targets',()):
            target=getattr(entry,'target',None)
            if isinstance(target,bpy.types.Object):found.add(target)
    for owner in (obj,obj.data,getattr(obj.data,'shape_keys',None)):
        if owner is None:continue
        for curve in _drivers(owner):
            for variable in curve.driver.variables:
                for target in variable.targets:
                    if isinstance(target.id,bpy.types.Object):found.add(target.id)
    return found


def _closure(bpy,roots,allowed):
    result=set();pending=list(roots)
    while pending:
        obj=pending.pop()
        if obj in result:continue
        if obj not in allowed:raise RuntimeError('Historical dependency refers to an existing/current object: '+obj.name)
        if any(mod.type=='CLOTH' for mod in obj.modifiers):
            raise RuntimeError('Do not evaluate a simulation dependency during this bounded comparison: '+obj.name)
        result.add(obj);pending.extend(_object_dependencies(bpy,obj)-result)
    return result


def _validate_rig(rig,native):
    animation=rig.animation_data
    if not animation or not animation.action or rig.data.pose_position!='POSE':
        raise RuntimeError('Expected an active fullbody action and POSE evaluation: '+rig.name)
    if animation.use_nla and any(not track.mute and len(track.strips) for track in animation.nla_tracks):
        raise RuntimeError('Active NLA layers require explicit alignment: '+rig.name)
    action_sha=native['_action_fingerprint'](animation.action)
    if action_sha!=EXPECTED_ACTION_SHA:
        raise RuntimeError('Unexpected fullbody action fingerprint: '+json.dumps({'rig':rig.name,'action':animation.action.name,'sha256':action_sha}))
    slot=getattr(animation,'action_slot',None)
    slots=[s for s in animation.action.slots if s.target_id_type=='OBJECT']
    if len(slots)!=1 or slot!=slots[0]:raise RuntimeError('Expected the unique fullbody OBJECT action slot: '+rig.name)
    if rig.pose.bones.get('CTRL_R02_Master') is None:raise RuntimeError('IK master missing: '+rig.name)
    return {'action':animation.action.name,'sha256':action_sha,'slot':slot.identifier,
            'nlaEnabled':animation.use_nla,'posePosition':rig.data.pose_position}


def _ik(rig):
    master=rig.pose.bones['CTRL_R02_Master']
    if any(key not in master for key in IK):raise RuntimeError('Missing explicit IK switch: '+rig.name)
    return {key:float(master[key]) for key in IK}


def _driver_status(objects):
    rows=[]
    for obj in sorted(objects,key=lambda o:o.name):
        for owner in (obj,obj.data,getattr(obj.data,'shape_keys',None)):
            if owner is None:continue
            for curve in _drivers(owner):
                rows.append({'owner':owner.name,'path':curve.data_path,'index':curve.array_index,
                             'valid':bool(curve.driver.is_valid),'muted':bool(curve.mute),
                             'expression':curve.driver.expression})
    return rows


def _constraint_state(objects,uv):
    # Include writable settings and target pointers in preservation checks. The
    # only mutations allowed in this helper are temporary links/visibility/frame.
    return {obj.name:{'constraints':[{'owner':owner,'settings':uv['_scalar_rna'](c),
                      'pointers':{p.identifier:uv['_plain'](getattr(c,p.identifier))
                                  for p in c.bl_rna.properties if p.type=='POINTER' and p.identifier!='rna_type'}}
                     for owner,c in _constraints(obj)],
                    'animation':uv['_animation'](obj),
                    'dataAnimation':uv['_animation'](obj.data),
                    'modifiers':[uv['_scalar_rna'](m) for m in obj.modifiers]}
            for obj in sorted(objects,key=lambda o:o.name)}


def _mesh_guard(obj,rig):
    if obj.type!='MESH' or obj.data.shape_keys:raise RuntimeError('Expected original collider mesh without shape keys: '+obj.name)
    if obj.animation_data or obj.constraints or obj.parent_type!='OBJECT':
        raise RuntimeError('Collider object attachment needs an explicit rest-space strategy: '+obj.name)
    if obj.parent not in (None,rig):raise RuntimeError('Unexpected collider parent: '+obj.name)
    active=[m for m in obj.modifiers if m.show_viewport]
    if any(m.type not in {'ARMATURE','COLLISION'} for m in active):
        raise RuntimeError('Unexpected active collider modifier: '+repr([(m.name,m.type) for m in active]))
    arms=[m for m in active if m.type=='ARMATURE']
    if len(arms)!=1 or arms[0].object!=rig:raise RuntimeError('Collider must evaluate through its own active armature: '+obj.name)
    return {'topology':_topology(obj.data),'rawCoordinatesSha256':_array_sha(_coordinates(obj.data)),
            'activeModifiers':[{'name':m.name,'type':m.type} for m in active]}


def _evaluated_world(obj,graph,topology):
    evaluated=obj.evaluated_get(graph)
    if evaluated==obj:raise RuntimeError('No evaluated collider instance: '+obj.name)
    mesh=evaluated.to_mesh(preserve_all_data_layers=False,depsgraph=graph)
    try:
        if _topology(mesh)!=topology:raise RuntimeError('Evaluated topology/vertex order changed: '+obj.name)
        result=_world(_coordinates(mesh),evaluated.matrix_world)
        if not np.isfinite(result).all():raise RuntimeError('Non-finite evaluated collider: '+obj.name)
        return result
    finally:evaluated.to_mesh_clear()


def _difference(old,current,ids=None):
    if ids is None:ids=np.arange(len(old),dtype=np.int32)
    if not len(ids):return {'vertices':0,'maximumDistanceMeters':None}
    delta=current[ids]-old[ids];distance=np.linalg.norm(delta,axis=1);worst=int(np.argmax(distance))
    return {'vertices':len(ids),'maximumDistanceMeters':float(distance[worst]),
            'maximumAbsoluteComponentMeters':float(np.abs(delta).max()),
            'meanDistanceMeters':float(distance.mean()),'rmsDistanceMeters':float(np.sqrt(np.mean(distance**2))),
            'vertexAtMaximum':int(ids[worst]),'oldWorldAtMaximum':old[ids[worst]].tolist(),
            'currentWorldAtMaximum':current[ids[worst]].tolist()}


def run(source_relative=None,report_relative=None,frames=FRAMES):
    import bpy
    source=_path(source_relative,SOURCE,'.blend')
    output=_path(report_relative,BASE/'fullbody-colliders-r08.json','.json') if report_relative is not None else None
    if source==OLD_SOURCE.resolve() or not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the exact independent fullbody source in background Object Mode')
    if output and output.exists():raise RuntimeError('Preserve existing diagnostic report: '+str(output))
    if tuple(frames)!=FRAMES:raise ValueError('Use the six reviewed bounded sample frames')
    assembly=_api('assemble_fullbody_r07');native=_api('native_hair_playback_r04');uv=_api('production_uv_r08')
    source_sha=_sha(source);old_sha=_sha(OLD_SOURCE)
    original_report=json.loads((BASE/'cloth-dynamic-r04-report.json').read_text(encoding='utf-8'))
    names=tuple(row['object'] for row in original_report['colliders'])
    if not names or len(set(names))!=len(names):raise RuntimeError('Invalid historical collider inventory')
    current={name:bpy.data.objects.get(name) for name in names};rig=bpy.data.objects.get('Character_Rig')
    if rig is None or any(o is None for o in current.values()):raise RuntimeError('A current rig/collider is missing')
    scene=bpy.context.scene;view=bpy.context.view_layer;view.update()
    frame_before=(scene.frame_current,scene.frame_subframe)
    current_action=_validate_rig(rig,native)
    current_objects=set(current.values())|{rig}
    current_contract=_constraint_state(current_objects,uv)
    current_mesh={name:_mesh_guard(obj,rig) for name,obj in current.items()}
    original_ids=assembly['_all_ids'](bpy)
    visibility={};temp=None;result=None;cleanup_ok=False
    try:
        requested=names+('Character_Rig',)
        old=assembly['_load_requested_objects'](bpy,OLD_SOURCE,requested)
        old_rig=old['Character_Rig'];old_action=_validate_rig(old_rig,native)
        old_ids=assembly['_all_ids'](bpy)-original_ids
        old_objects=_closure(bpy,old.values(),{obj for obj in bpy.data.objects if obj in old_ids})
        old_mesh={name:_mesh_guard(old[name],old_rig) for name in names}
        for name in names:
            if old_mesh[name]['topology']!=current_mesh[name]['topology']:
                raise RuntimeError('Old/current vertex correspondence is not established: '+name)
        temp=bpy.data.collections.new('R08 temporary collider evaluation')
        scene.collection.children.link(temp)
        for obj in current_objects|old_objects:
            if obj in current_objects:
                visibility[obj]={'hideViewport':obj.hide_viewport,'hideRender':obj.hide_render,
                                 'hideSet':obj.hide_get() if obj.name in view.objects else None,
                                 'originalCollections':[c.name for c in obj.users_collection]}
            temp.objects.link(obj)
            obj.hide_viewport=False;obj.hide_set(False)
        view.update()
        for obj in current_objects|old_objects:
            if obj.name not in view.objects or not obj.visible_get(view_layer=view):
                raise RuntimeError('Collider/rig dependency is not enabled in the view layer: '+obj.name)
        scene.frame_set(frame_before[0],subframe=frame_before[1]);view.update()
        old_contract=_constraint_state(old_objects,uv)
        scene.frame_set(FRAMES[0]);view.update();graph=bpy.context.evaluated_depsgraph_get()
        old_rig_world=np.array(old_rig.evaluated_get(graph).matrix_world,dtype=float)
        masks={};mask_reports={}
        for name in names:
            # Raw original geometry, transformed only by its static object
            # attachment into the old rig space. Pose deformation is excluded.
            transform=np.linalg.inv(old_rig_world)@np.array(old[name].evaluated_get(graph).matrix_world)
            rest=_world(_coordinates(old[name].data),transform)
            ids=np.flatnonzero(rest[:,2]<=REST_Z_MAX).astype(np.int32);masks[name]=ids
            mask_reports[name]={'definition':'old rig-local rest vertex Z <= 1.05 m',
                'fixedVertexIndicesSha256':_array_sha(ids),'vertices':len(ids),'totalVertices':len(rest),
                'restBounds':[rest.min(axis=0).tolist(),rest.max(axis=0).tolist()],
                'restAttachmentMatrix':transform.tolist(),'restCoordinatesSha256':_array_sha(rest),
                'scope':'Conservative lower-trunk/leg region. Fixed original vertex IDs; never animated world Z.'}
        rows=[]
        for frame in FRAMES:
            scene.frame_set(frame);view.update();graph=bpy.context.evaluated_depsgraph_get()
            old_evaluated_rig=old_rig.evaluated_get(graph);current_evaluated_rig=rig.evaluated_get(graph)
            old_ik=_ik(old_evaluated_rig);current_ik=_ik(current_evaluated_rig)
            ik_error=max(abs(old_ik[k]-current_ik[k]) for k in IK)
            if ik_error>1e-7:
                raise RuntimeError('Actual IK switches differ: '+json.dumps({'frame':frame,'old':old_ik,'current':current_ik,'error':ik_error}))
            statuses=_driver_status(current_objects|old_objects)
            invalid=[row for row in statuses if not row['valid'] and not row['muted']]
            if invalid:raise RuntimeError('Invalid active evaluation driver: '+json.dumps({'frame':frame,'drivers':invalid}))
            objects={}
            for name in names:
                previous=_evaluated_world(old[name],graph,old_mesh[name]['topology'])
                actual=_evaluated_world(current[name],graph,current_mesh[name]['topology'])
                objects[name]={'wholeMesh':_difference(previous,actual),
                               'fixedLowerBodyRegion':_difference(previous,actual,masks[name])}
            basis_differences={bone.name:float(np.abs(np.array(bone.matrix_basis)-np.array(
                current_evaluated_rig.pose.bones[bone.name].matrix_basis)).max())
                for bone in old_evaluated_rig.pose.bones if bone.name in current_evaluated_rig.pose.bones}
            rows.append({'frame':frame,'ik':{'old':old_ik,'current':current_ik,'maximumDifference':ik_error},
                         'poseBasisMaximumElementDifferenceByBone':{name:error for name,error in basis_differences.items() if error>0},
                         'activeDriversValid':True,'driverCount':len(statuses),'objects':objects})
            print('R08_COLLIDER_COMPARISON_FRAME '+json.dumps(rows[-1]),flush=True)
        # Read-only settings must stay unchanged, including action/constraint
        # definitions. Frame-driven values are checked after restoring the frame.
        scene.frame_set(frame_before[0],subframe=frame_before[1]);view.update()
        # old source was appended at the caller frame, so restore it likewise.
        if _constraint_state(old_objects,uv)!=old_contract:
            raise RuntimeError('Historical constraint/driver/modifier definitions changed during comparison')
        result={'currentSource':str(source.relative_to(P)),'currentSourceSha256':source_sha,
            'historicalSource':str(OLD_SOURCE.relative_to(P)),'historicalSourceSha256':old_sha,
            'historicalCacheReportSha256':_sha(BASE/'cloth-dynamic-r04-report.json'),
            'blenderVersion':bpy.app.version_string,'frames':list(FRAMES),
            'currentAction':current_action,'historicalAction':old_action,
            'actionFingerprintMatched':True,'ikAlignment':'Actual evaluated switches checked at every sampled frame; no override applied.',
            'originalModifiersAndConstraintsPreserved':True,
            'oldDependenciesLinkedAndVisible':[o.name for o in sorted(old_objects,key=lambda o:o.name)],
            'currentColliderData':current_mesh,'historicalColliderData':old_mesh,
            'fixedRegionRestZUpperBoundMeters':REST_Z_MAX,'fixedRegions':mask_reports,'samples':rows,
            'limits':['Six actual evaluated frames, not all 301 frames or continuous collision detection.',
                      'Historical and current world coordinates use the same action fingerprint and explicit IK switches with original constraints/drivers active.',
                      'Regional masks use old rig-local raw rest coordinates and fixed vertex IDs; entire-mesh errors are also retained.',
                      'These are collider input differences only. They do not establish cloth clearance, unchanged contact, cloth resimulation, or visual acceptance.']}
    finally:
        for obj,state in visibility.items():
            obj.hide_viewport=state['hideViewport'];obj.hide_render=state['hideRender']
            if state['hideSet'] is not None:obj.hide_set(state['hideSet'])
        added=assembly['_all_ids'](bpy)-original_ids
        if added:bpy.data.batch_remove(ids=list(added))
        scene.frame_set(frame_before[0],subframe=frame_before[1]);view.update()
        cleanup_ok=assembly['_all_ids'](bpy)==original_ids
        if not cleanup_ok:raise RuntimeError('Temporary library IDs were not fully removed')
    if _constraint_state(current_objects,uv)!=current_contract:
        raise RuntimeError('Current constraint/driver/modifier definitions did not restore exactly')
    if {name:_mesh_guard(obj,rig) for name,obj in current.items()}!=current_mesh:
        raise RuntimeError('Current raw collider geometry or modifier configuration changed')
    for obj,state in visibility.items():
        actual={'hideViewport':obj.hide_viewport,'hideRender':obj.hide_render,
                'hideSet':obj.hide_get() if obj.name in view.objects else None,
                'originalCollections':[c.name for c in obj.users_collection]}
        if actual!=state:raise RuntimeError('Current object visibility/collection membership did not restore: '+obj.name)
    if _sha(source)!=source_sha or _sha(OLD_SOURCE)!=old_sha:raise RuntimeError('A source file changed during this diagnostic')
    result.update(temporaryIDsRemoved=cleanup_ok,currentSettingsRestored=True,sourceFilesUnchanged=True,
                  restoredFrame={'frame':scene.frame_current,'subframe':scene.frame_subframe},savedBlend=False)
    result['sampledMaximaByCollider']={name:{domain:max(
        (row['objects'][name][domain]['maximumDistanceMeters'] for row in result['samples']
         if row['objects'][name][domain]['maximumDistanceMeters'] is not None),default=None)
        for domain in ('wholeMesh','fixedLowerBodyRegion')} for name in names}
    if output:
        # Exclusive creation preserves previous reports even if another process
        # created the path after the initial check.
        with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


if __name__=='__main__':print('Loaded only. Call run() in the independent fullbody background copy; no auto execution.')
