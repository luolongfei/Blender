"""New R09 authored body action + matched native tail hair, in a new copy only.

Integrator entry (already opened exact independent final face/groom source):
    run(source_relative=..., action_library_relative=...,
        output_relative='work/reconstruction-r02/reference-body-hair-r09.blend')

Explicit serial CLI, never launched by this module:
    blender --background --python body_hair_driver_r09.py -- --source <relative>

Only the Action is appended, never another rig/object. Old native hair is
restored into retained history. Old cloth playback is retained but hidden in
this new copy; the original skinned skirt is a labeled STATIC display proxy.
This helper does not generate matching R09 cloth or claim garment contact.
"""
import argparse
import hashlib
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

ACTION='R09 Body performance - authored squat bend reach'
HEAD_CENTER=(0.,.00202,1.441285)
HEAD_RADIUS=.084
FRAMES=(1,55,109.375,151,247,301)
STATIC_SKIRT=('Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam')
CACHE_DIR='work/reconstruction-r02/hair-body-r09'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _stable_ids(value):
    # Runtime RNA pointers change after reopening, so they must not enter a
    # persistent cache input fingerprint. ID names/data paths remain explicit.
    if isinstance(value,dict):return {key:_stable_ids(item) for key,item in value.items() if key!='pointer'}
    if isinstance(value,list):return [_stable_ids(item) for item in value]
    return value


def _path(relative,suffix=None):
    p=project_path();value=(p/relative).resolve();value.relative_to((p/'work/reconstruction-r02').resolve())
    if suffix and value.suffix!=suffix:raise ValueError('Unexpected path type: '+str(value))
    return value


def _relative(path):return Path(path).relative_to(project_path()).as_posix()


def _json_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def _array(collection,field,width,dtype=np.float32):
    a=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(field,a)
    return a.reshape(-1,width)


def _geometry(obj):
    """Raw editable geometry, including all morphs, radii, weights and topology."""
    data=obj.data;h=hashlib.sha256()
    if obj.type=='CURVES':
        h.update(_array(data.position_data,'vector',3).tobytes())
        h.update(np.asarray([c.points_length for c in data.curves],np.int32).tobytes())
        if data.attributes.get('radius'):h.update(_array(data.attributes['radius'].data,'value',1).tobytes())
    elif obj.type=='MESH':
        h.update(_array(data.vertices,'co',3).tobytes())
        h.update(_array(data.loops,'vertex_index',1,np.int32).tobytes())
        h.update(_array(data.polygons,'loop_start',1,np.int32).tobytes())
        h.update(_array(data.polygons,'loop_total',1,np.int32).tobytes())
        h.update(_array(data.polygons,'material_index',1,np.int32).tobytes())
        if data.shape_keys:
            for key in data.shape_keys.key_blocks:h.update(key.name.encode());h.update(_array(key.data,'co',3).tobytes())
        for layer in data.uv_layers:h.update(layer.name.encode());h.update(_array(layer.data,'uv',2).tobytes())
        h.update(_digest([[(obj.vertex_groups[g.group].name,float(g.weight)) for g in v.groups] for v in data.vertices]).encode())
    else:raise ValueError('Geometry fingerprint only supports Mesh/Curves')
    h.update(_digest([m.name if m else None for m in data.materials]).encode())
    return h.hexdigest()


def _rig_rest(rig):
    return {'world':[list(row) for row in rig.matrix_world],
        'bones':{bone.name:{'parent':bone.parent.name if bone.parent else None,
                    'matrix':[list(row) for row in bone.matrix_local],'deform':bone.use_deform}
                 for bone in rig.data.bones}}


def _append_action(bpy,library,name,copy_name,native):
    if bpy.data.actions.get(copy_name):raise RuntimeError('Preserve existing named action; choose a distinct copy name: '+copy_name)
    before_objects={o.as_pointer() for o in bpy.data.objects}
    requested_names=(name,)
    with bpy.data.libraries.load(str(library),link=False) as (available,loaded):
        if name not in available.actions:raise RuntimeError('Action library lacks '+name)
        # The destination list is mutated to IDs by Blender on context exit.
        # The immutable requested tuple must never be the assigned destination.
        loaded.actions=list(requested_names)
    loaded_ids=tuple(loaded.actions)
    if len(loaded_ids)!=1 or loaded_ids[0] is None:raise RuntimeError('Incomplete action append')
    result=dict(zip(requested_names,loaded_ids));action=result[name]
    if {o.as_pointer() for o in bpy.data.objects}!=before_objects:
        raise RuntimeError('Action append unexpectedly imported object dependencies; refuse mixed rigs')
    if action.library or 'r09_performance_report' not in action:raise RuntimeError('Independent authored R09 action metadata required')
    metadata=json.loads(action['r09_performance_report'])
    if metadata.get('sampleCount')!=301 or metadata.get('keySampleFps')!=30 or list(action.frame_range)!=[1.,301.]:
        raise RuntimeError('Expected the verified 301-frame, 30 fps authored body action')
    action.name=copy_name;action.use_fake_user=True
    return action,{'libraryAction':name,'independentAction':action.name,
                   'actionSha256':native['_action_fingerprint'](action),'authoringMetadata':metadata,
                   'objectCountUnchanged':True}


def _select(bpy,rig,action,native):
    for curve in native['_fcurves'](action):
        try:rig.path_resolve(curve.data_path)
        except Exception as error:raise RuntimeError('Action channel does not resolve on current rig: '+curve.data_path) from error
    rig.animation_data_create();rig.animation_data.action=action
    slots=[s for s in action.slots if s.target_id_type=='OBJECT']
    if len(slots)!=1:raise RuntimeError('Expected one OBJECT action slot')
    rig.animation_data.action_slot=slots[0];rig.data.pose_position='POSE'
    bpy.context.scene.frame_set(1);bpy.context.view_layer.update()


def _static_cloth_display(bpy):
    states={};disabled=[]
    if any(bpy.data.objects.get(name) is None for name in STATIC_SKIRT):raise RuntimeError('Retained original skinned skirt is missing')
    for obj in bpy.context.scene.objects:
        dynamic=(any(m.type=='CLOTH' for m in obj.modifiers) or obj.name in ('R02 Cloth Skirt control','R06 Walk cloth')
                 or ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
        if dynamic or obj.name in STATIC_SKIRT:
            if dynamic and obj.name in STATIC_SKIRT:raise RuntimeError('Static skirt unexpectedly contains a simulation')
            states[obj.name]={'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hideSet':obj.hide_get(),
                             'newHidden':bool(dynamic)}
            obj.hide_render=dynamic;obj.hide_viewport=dynamic;obj.hide_set(dynamic)
        for modifier in obj.modifiers:
            if modifier.type=='CLOTH':
                disabled.append({'object':obj.name,'modifier':modifier.name,
                                 'viewport':modifier.show_viewport,'render':modifier.show_render})
                modifier.show_viewport=False;modifier.show_render=False
    return {'objectStatesBefore':states,'clothModifierStatesBefore':disabled,
            'display':'Original skinned skirt is a static preview proxy; previous action-specific cloth is retained and hidden.',
            'matchingR09ClothPresent':False,'clothCollisionAccepted':False}


def _colliders(bpy,rig,secondary):
    from mathutils import Vector
    if not rig.get('R06_head_pivot_calibration'):raise RuntimeError('Current calibrated head pivot required')
    world=np.array(rig.matrix_world,dtype=float)
    if np.max(np.abs(world[:3,:3].T@world[:3,:3]-np.eye(3)))>1e-5:
        raise RuntimeError('World-space collider radii require the existing rigid unit rig transform')
    specs=secondary['_collider_definitions'](rig)
    rest_matrix=rig.matrix_world@rig.data.bones['head'].matrix_local
    center=rest_matrix.inverted()@Vector(HEAD_CENTER)
    for spec in specs:
        if spec['name']=='head':spec.update(localCenter=list(center),radius=HEAD_RADIUS)
    head_check=np.array(rest_matrix@center)-np.array(HEAD_CENTER)
    if np.max(np.abs(head_check))>2e-7:raise RuntimeError('Measured head sphere rest conversion failed')
    fit=[];pose=rig.data.pose_position
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        for side,suffix in (('left','L'),('right','R')):
            bone=rig.data.bones[side+'UpperArm'];sleeve=bpy.data.objects.get('Blouse_Puff_Sleeve_'+suffix)
            if sleeve is None or sleeve.type!='MESH':raise RuntimeError('Actual puff sleeve missing for conservative upper-arm envelope')
            evaluated=sleeve.evaluated_get(graph);mesh=evaluated.to_mesh()
            try:
                coordinates=_array(mesh.vertices,'co',3).astype(float)
                transform=np.array((rig.matrix_world@bone.matrix_local).inverted()@evaluated.matrix_world)
                local=coordinates@transform[:3,:3].T+transform[:3,3]
            finally:evaluated.to_mesh_clear()
            if len(local)<24:raise RuntimeError('Insufficient actual sleeve samples')
            for index,fraction in enumerate((.22,.52)):
                y=bone.length*fraction;distance=np.abs(local[:,1]-y)
                selected=distance<=max(.020,bone.length*.10)
                if np.count_nonzero(selected)<24:
                    selected=np.zeros(len(local),bool);selected[np.argsort(distance)[:24]]=True
                radial=np.linalg.norm(local[selected][:,(0,2)],axis=1)
                measured=float(np.quantile(radial,.85))+.0015
                # Sphere envelope is intentionally finite; raw fit and the cap
                # are reported rather than pretending it is a garment surface.
                radius=float(np.clip(measured,.028,.065))
                spec={'name':side+' upper arm / sleeve '+str(index+1),'bone':bone.name,
                      'localCenter':[0.,float(y),0.],'radius':radius}
                specs.append(spec)
                fit.append({'collider':spec,'sourceSleeve':sleeve.name,'boneLength':bone.length,
                            'axialFraction':fraction,'selectedVertices':int(selected.sum()),
                            'maximumSelectedAxialDistanceMeters':float(distance[selected].max()),
                            'measured85PercentRadialEnvelopePlusMarginMeters':measured,
                            'radiusClamped':abs(radius-measured)>1e-9,
                            'scope':'Bone-following sphere approximation fitted to actual REST sleeve cross section; not cloth collision.'})
    finally:rig.data.pose_position=pose;bpy.context.view_layer.update()
    return specs,{'headWorldCenter':list(HEAD_CENTER),'headRadiusMeters':HEAD_RADIUS,
        'headRestWorldMatrix':[list(row) for row in rest_matrix],'headLocalCenter':list(center),
        'headCenterRoundtripMaximumErrorMeters':float(np.abs(head_check).max()),
        'headMeasurementSource':'R07 measured posterior/side/top cranial envelope, independent of head bone length.',
        'preservedBaseSpheres':[c['name'] for c in specs[:5]],'addedUpperArmSleeveSpheres':fit,
        'limitations':['Head, neck, shoulder, chest and sleeve spheres are approximate body envelopes.',
                      'No garment mesh collision, strand self-collision, dynamic sleeve deformation or full cloth contact is certified.']}


def _cache(secondary,native,action,colliders,manifest,output_relative=CACHE_DIR):
    folder=_path(output_relative);folder.mkdir(parents=True,exist_ok=True)
    manifest_file=folder/'inputs-r09.json'
    cache=folder/(re.sub('[^a-zA-Z0-9_-]+','-',action.name).strip('-')+'.guides.npz')
    if manifest_file.exists():
        existing=json.loads(manifest_file.read_text(encoding='utf-8'))
        if existing!=manifest:raise RuntimeError('R09 cache inputs differ; retain this cache and choose an explicitly new cache stage')
    else:
        if any(folder.iterdir()):raise RuntimeError('Unrecorded existing R09 cache files must be retained for inspection')
        _json_new(manifest_file,manifest)
    if cache.exists():
        _,meta,arrays,_,_=native['load_cache'](_relative(cache))
        if (meta['actionSha256']!=manifest['actionSha256'] or meta['colliders']!=colliders or
            meta['frameRange']!=[1,301] or meta['params']!=manifest['simulationParameters']):
            raise RuntimeError('Existing R09 cache metadata does not match these explicit inputs')
        if arrays['guide_delta'].shape[0]!=301:raise RuntimeError('Existing cache is incomplete')
        print('R09_BODY_HAIR_REUSE '+str(cache),flush=True)
        return cache,{'reusedValidatedInputs':True,'metadata':meta}
    print('R09_BODY_HAIR_BAKE 301 frames, tail guides only',flush=True)
    report=secondary['bake_actions'](actions=[action.name],frame_ranges={action.name:(1,301)},
        output=_relative(folder),allow_authored_ik=True,colliders=colliders,loop_preroll_cycles=0)
    actual=_path(report['actions'][0]['cache'],'.npz')
    if actual!=cache:raise RuntimeError('Unexpected output cache path')
    return cache,{'reusedValidatedInputs':False,'report':report}


def run(source_relative,action_library_relative='work/reconstruction-r02/body-action-r09.blend',
        output_relative='work/reconstruction-r02/reference-body-hair-r09.blend',action_copy_name=ACTION,
        cache_directory_relative=CACHE_DIR):
    import bpy
    source=_path(source_relative,'.blend');library=_path(action_library_relative,'.blend');target=_path(output_relative,'.blend')
    report_path=target.with_suffix('.json')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the exact final face/groom source in background Object Mode')
    if source==target or library==target or target.exists() or report_path.exists():raise RuntimeError('A new independent output and report are required')
    rigs=[obj for obj in bpy.context.scene.objects if obj.type=='ARMATURE']
    if len(rigs)!=1 or rigs[0].name!='Character_Rig':raise RuntimeError('Expected one production Character_Rig')
    rig=rigs[0];scene=bpy.context.scene
    if abs(scene.render.fps/scene.render.fps_base-30)>1e-6:raise RuntimeError('Preserve the authored 30 fps timebase')
    if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):raise RuntimeError('Active NLA needs explicit action alignment')
    native=_api('native_hair_playback_r04');secondary=_api('secondary_hair_r02');uv=_api('production_uv_r08')
    if any(getattr(handler,'_r02_secondary_hair',False) for handler in bpy.app.handlers.frame_change_post):
        raise RuntimeError('Detach the historical Python hair handler before this explicit native-cache workflow')
    if any(bpy.data.objects[name].get(secondary['MARKER']) for name in secondary['OBJECTS']):
        raise RuntimeError('Restore session-driven tail points before baking the new source')
    source_sha=_sha(source);library_sha=_sha(library);bpy.context.view_layer.update()
    originals=[obj for obj in bpy.context.scene.objects if obj.type in {'MESH','CURVES'}]
    geometry={obj.as_pointer():_geometry(obj) for obj in originals};rest=_rig_rest(rig)
    old_actions={action.as_pointer():native['_action_fingerprint'](action) for action in bpy.data.actions if native['_fcurves'](action)}
    previous_action=rig.animation_data.action if rig.animation_data else None
    report={'source':_relative(source),'sourceSha256':source_sha,'actionLibrary':_relative(library),
            'actionLibrarySha256':library_sha,'target':_relative(target),'savedBlend':False}
    try:
        action,import_report=_append_action(bpy,library,ACTION,action_copy_name,native)
        report['actionImport']=import_report
        if previous_action:previous_action.use_fake_user=True
        if native['RIG_GUIDE'] in rig:report['previousNativeHairRetained']=native['restore']()
        _select(bpy,rig,action,native);report['clothDisplay']=_static_cloth_display(bpy)
        specs,collider_report=_colliders(bpy,rig,secondary);report['colliders']=collider_report
        parameters=dict(secondary['PARAMS']);parameters['loop_preroll_cycles']=0
        manifest={'version':1,'sourceSha256':source_sha,'actionLibrarySha256':library_sha,
            'actionName':action.name,'actionSha256':native['_action_fingerprint'](action),
            'rigRestSha256':_digest(rest),'rigDriversSha256':_digest(_stable_ids(uv['_animation'](rig)['drivers'])),
            'tailGeometrySha256':{name:_geometry(bpy.data.objects[name]) for name in secondary['OBJECTS']},
            'colliders':specs,'simulationParameters':parameters,'frameRange':[1,301]}
        cache,cache_report=_cache(secondary,native,action,specs,manifest,cache_directory_relative)
        scene.frame_start=1;scene.frame_end=301;scene.frame_set(1);bpy.context.view_layer.update()
        report['cache']=_relative(cache);report['cacheSha256']=_sha(cache);report['cacheRun']=cache_report
        print('R09_BODY_HAIR_NATIVE_BUILD',flush=True)
        report['nativeBuild']=native['run'](cache_relative=_relative(cache))
        native['enable_for_action'](action_name=action.name,select=True,frame=1)
        report['nativePointParity']=native['validate_evaluated'](cache_relative=_relative(cache),frames=list(FRAMES))
        scene.frame_set(1);bpy.context.view_layer.update()
        for obj in originals:
            if _geometry(obj)!=geometry[obj.as_pointer()]:raise RuntimeError('Original editable geometry changed: '+obj.name)
        if _rig_rest(rig)!=rest:raise RuntimeError('Current rig rest data changed')
        for old in bpy.data.actions:
            if old.as_pointer() in old_actions and native['_action_fingerprint'](old)!=old_actions[old.as_pointer()]:
                raise RuntimeError('An existing action was modified: '+old.name)
        if [obj for obj in bpy.context.scene.objects if obj.type=='ARMATURE']!=rigs:raise RuntimeError('Unexpected second rig')
        if _sha(source)!=source_sha or _sha(library)!=library_sha:raise RuntimeError('An input file changed externally')
        if target.exists():raise RuntimeError('Output appeared during processing; preserve it')
        target.parent.mkdir(parents=True,exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(target))
        report['savedBlend']=True
        report.update(targetSha256=_sha(target),activeAction=action.name,frameRange=[1,301],fps=30,
            originalGeometryAndActionsPreserved=True,singleRigPreserved=True,
            requiresFreshReopen=True,appearanceAccepted=False,
            limitations=['Only the two tails have new PBD dynamics. Scalp/fringe/gathers remain attached to the head.',
                        'This action is authored, not motion capture. Sphere contact is an approximation.',
                        'Previous R07 cloth remains in the file but is hidden. Static skinned skirt is a temporary display proxy; R09 matching cloth is outstanding.',
                        'Actual fresh-file reopening and continuous combined motion/clearance review are still required.',
                        'Native Curves/Geometry Nodes playback is Blender production functionality, not a VRM capability.'])
        _json_new(report_path,report)
        print('R09_BODY_HAIR_SAVED '+str(target),flush=True)
        return report
    except Exception as error:
        report.update(error=str(error),status='failed-after-save' if report['savedBlend'] else 'failed-no-save')
        failure=report_path.with_name(report_path.stem+'.failed.json');index=1
        while failure.exists():failure=report_path.with_name(report_path.stem+f'.failed-{index}.json');index+=1
        _json_new(failure,report)
        raise


def verify_reopened(source_relative='work/reconstruction-r02/reference-body-hair-r09.blend'):
    import bpy
    source=_path(source_relative,'.blend');report=json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or _sha(source)!=report['targetSha256']:
        raise RuntimeError('Reopen the exact saved R09 body/hair output in a fresh background process')
    native=_api('native_hair_playback_r04')
    parity=native['validate_evaluated'](cache_relative=report['cache'],frames=list(FRAMES))
    for name,row in report['clothDisplay']['objectStatesBefore'].items():
        obj=bpy.data.objects[name]
        if row['newHidden'] and (not obj.hide_render or not obj.hide_viewport):
            raise RuntimeError('Unmatched historical cloth became visible: '+name)
    return {'source':_relative(source),'sourceSha256':_sha(source),'nativePointParity':parity,
            'historicalClothStillHidden':True,'matchingR09ClothPresent':False,'savedBlend':False,'appearanceAccepted':False}


def main(argv=None):
    import bpy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--action-library',default='work/reconstruction-r02/body-action-r09.blend')
    parser.add_argument('--output',default='work/reconstruction-r02/reference-body-hair-r09.blend')
    parser.add_argument('--cache-directory',default=CACHE_DIR)
    args=parser.parse_args(argv)
    if not bpy.app.background:raise RuntimeError('Use an explicitly launched isolated background Blender')
    source=_path(args.source,'.blend')
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    return run(args.source,args.action_library,args.output,cache_directory_relative=args.cache_directory)


if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
