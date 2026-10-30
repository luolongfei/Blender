"""Copy the audited CMU walk and author a head-only forward-gaze adjustment.

Read the complete source first. Never alter the source action or bone rest pose.
Only the four copied head quaternion curves are edited; every other FCurve,
including head location/scale, is compared point-for-point with the source.
This is an authored adaptation of mocap, not untouched capture or acceptance.
No Blender launch, rendering, cache rebake or Blend save occurs here.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SOURCE='R03 CMU 07_01 walk - proportional IK preview v4'
ACTION='R06 CMU 07_01 walk - forward gaze'
HEAD_ROTATION='pose.bones["head"].rotation_quaternion'
HEAD_PREFIX='pose.bones["head"]'
KEYS=('leg_L','leg_R','arm_L','arm_R')


def head_pitch(matrix,rest):
    """Use the confirmed model-facing -Y direction transformed from rest."""
    forward=np.asarray(matrix)[:3,:3]@np.linalg.inv(np.asarray(rest)[:3,:3])@np.array([0.,-1.,0.])
    forward/=np.linalg.norm(forward)
    return math.atan2(float(forward[2]),float(np.linalg.norm(forward[:2]))),forward


def adjusted_head(matrix,rest,mean_pitch,target_degrees=-4.0,source_variation=.35):
    """Rotate about world UP x horizontal forward while keeping the head pivot."""
    matrix=np.asarray(matrix,dtype=float);pitch,forward=head_pitch(matrix,rest)
    target=math.radians(target_degrees)+source_variation*(pitch-mean_pitch)
    horizontal=forward.copy();horizontal[2]=0
    if np.linalg.norm(horizontal)<1e-6: raise RuntimeError('Near-vertical facing direction has no stable gaze axis')
    horizontal/=np.linalg.norm(horizontal)
    axis=np.cross(np.array([0.,0.,1.]),horizontal);axis/=np.linalg.norm(axis)
    # With axis = UP x forward, a positive right-handed rotation lowers pitch.
    angle=pitch-target;x,y,z=axis
    k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    rotation=np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
    result=matrix.copy();result[:3,:3]=rotation@matrix[:3,:3]
    achieved,_=head_pitch(result,rest)
    if abs(achieved-target)>1e-8: raise RuntimeError('Gaze pitch construction failed')
    return result,{'sourcePitchDegrees':math.degrees(pitch),'targetPitchDegrees':math.degrees(target),
                   'rotationDegrees':math.degrees(angle),'worldRightAxis':axis.tolist()}


def _value(value):
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    try: return [_value(x) for x in value]
    except TypeError: return repr(value)


def _properties(item):
    """Scalar/array RNA settings; exclude identity pointers and UI containers."""
    result={}
    for prop in item.bl_rna.properties:
        if prop.identifier=='rna_type' or prop.type in {'POINTER','COLLECTION'}: continue
        result[prop.identifier]=_value(getattr(item,prop.identifier))
    return result


def _curve_record(curve):
    return {'dataPath':curve.data_path,'arrayIndex':curve.array_index,
            'group':curve.group.name if curve.group else None,
            'settings':_properties(curve),
            'keyframes':[_properties(p) for p in curve.keyframe_points],
            'sampledPoints':[_properties(p) for p in curve.sampled_points],
            'modifiers':[_properties(m) for m in curve.modifiers]}


def _records(action,helper,exclude_rotation=False,exclude_head=False):
    curves=helper['_fcurves'](action)
    if exclude_rotation: curves=[c for c in curves if c.data_path!=HEAD_ROTATION]
    if exclude_head: curves=[c for c in curves if not c.data_path.startswith(HEAD_PREFIX)]
    return [_curve_record(c) for c in sorted(curves,key=lambda c:(c.data_path,c.array_index))]


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _set_frame(scene,frame):
    scene.frame_set(math.floor(frame),subframe=float(frame%1))
    bpy.context.view_layer.update()


def create_action(action_name=ACTION,source_action=SOURCE,target_degrees=-4.0,source_variation=.35,
                  verification_frames=(1,7,15,23,35,45,61,79,79.75),
                  report_relative='work/reconstruction-r02/forward-gaze-r06.json'):
    """Explicitly create a copied, independently named action; restore scene state.

    Runtime evaluation happens only when the integrator invokes this in the
    existing saved candidate. All 316 source head/parent matrices are sampled
    before making the copy. The source FCurves must remain exactly unchanged.
    """
    if bpy is None: raise RuntimeError('Run explicitly in the existing Blender candidate')
    helper=runpy.run_path(str(Path(__file__).with_name('full_body_review_r03.py')))
    rig=helper['_guard']();scene=bpy.context.scene
    if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6:
        raise RuntimeError('This confirmed world-direction convention requires the identity rig object transform')
    if not math.isfinite(target_degrees) or not -20<=target_degrees<=10 or not 0<=source_variation<=1:
        raise ValueError('Use a bounded forward-gaze target and retained variation in [0,1]')
    if bpy.data.actions.get(action_name): raise RuntimeError('Action exists; choose a new name instead of overwriting')
    source=bpy.data.actions.get(source_action)
    if source is None: raise RuntimeError('The audited v4 source action is required')
    head=rig.pose.bones['head'];parent=head.parent
    if parent is None or head.rotation_mode!='QUATERNION':
        raise RuntimeError('This operation requires the current quaternion head with its unchanged parent')
    if head.constraints or any(c.data_path.startswith(HEAD_PREFIX) for c in rig.animation_data.drivers):
        raise RuntimeError('Head constraints or direct channel drivers require a separately scoped edit')
    curves={c.array_index:c for c in helper['_fcurves'](source) if c.data_path==HEAD_ROTATION}
    if set(curves)!={0,1,2,3} or any(c.modifiers or c.sampled_points for c in curves.values()):
        raise RuntimeError('Expected four ordinary keyed head quaternion curves')
    frames=[float(k.co.x) for k in curves[0].keyframe_points]
    if len(frames)!=316 or not np.allclose(frames,1+np.arange(316)*.25,rtol=0,atol=1e-6):
        raise RuntimeError('Expected the preserved 316 source-rate keys at frames 1..79.75')
    if any([float(k.co.x) for k in c.keyframe_points]!=frames for c in curves.values()):
        raise RuntimeError('Head quaternion key times differ between components')
    checks=sorted(set(float(f) for f in verification_frames))
    if not checks or any(f<frames[0] or f>frames[-1] for f in checks): raise ValueError('Verification frames must lie inside the source clip')
    source_all=_records(source,helper);preserved=_records(source,helper,exclude_rotation=True)
    nonhead=_records(source,helper,exclude_head=True)
    rest=np.asarray(head.bone.matrix_local).copy()
    old_action=rig.animation_data.action
    old_slot=rig.animation_data.action_slot if hasattr(rig.animation_data,'action_slot') else None
    old_frame,old_subframe=scene.frame_current,scene.frame_subframe
    old_basis={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
    master=rig.pose.bones['CTRL_R02_Master'];old_switch={k:float(master['ik_'+k]) for k in KEYS}
    created=None;complete=False
    try:
        helper['_assign'](rig,source)
        samples=[];check_source={}
        for frame in frames:
            _set_frame(scene,frame)
            if any(abs(float(master['ik_'+k]))>1e-8 for k in KEYS):
                raise RuntimeError('Audited FK source must retain all four IK switches at zero')
            sample={'frame':frame,'head':np.asarray(head.matrix).copy(),
                    'parent':parent.matrix.copy(),'basis':head.matrix_basis.copy()}
            samples.append(sample)
            if frame in checks:
                check_source[frame]={'head':head.matrix.copy(),
                    'feet':{name:rig.pose.bones[name].matrix.copy() for name in ('leftFoot','rightFoot')},
                    'neck':rig.pose.bones['neck'].matrix.copy()}
        # Verification normally uses source key frames. Arbitrary check frames
        # are also sampled now, before the copied action can exist.
        for frame in checks:
            if frame not in check_source:
                _set_frame(scene,frame)
                check_source[frame]={'head':head.matrix.copy(),
                    'feet':{name:rig.pose.bones[name].matrix.copy() for name in ('leftFoot','rightFoot')},
                    'neck':rig.pose.bones['neck'].matrix.copy()}
        pitches=[head_pitch(s['head'],rest)[0] for s in samples]
        mean_pitch=float(np.mean(pitches));quaternions=[];metadata=[];previous=None
        max_basis_location_error=0.;max_basis_scale_error=0.
        for sample in samples:
            desired,details=adjusted_head(sample['head'],rest,mean_pitch,target_degrees,source_variation)
            basis=head.bone.convert_local_to_pose(Matrix(desired.tolist()),head.bone.matrix_local,
                parent_matrix=sample['parent'],parent_matrix_local=parent.bone.matrix_local,invert=True)
            max_basis_location_error=max(max_basis_location_error,(basis.translation-sample['basis'].translation).length)
            max_basis_scale_error=max(max_basis_scale_error,(basis.to_scale()-sample['basis'].to_scale()).length)
            quaternion=basis.to_quaternion()
            if previous is not None and previous.dot(quaternion)<0: quaternion.negate()
            previous=quaternion.copy();quaternions.append(list(quaternion))
            metadata.append(dict(frame=sample['frame'],**details))
        if max_basis_location_error>2e-6 or max_basis_scale_error>2e-6:
            raise RuntimeError('Head correction would change its local pivot or scale')
        # All source reads are complete. Copy now; edit only copied head values.
        created=source.copy();created.name=action_name;created.use_fake_user=True
        copied={c.array_index:c for c in helper['_fcurves'](created) if c.data_path==HEAD_ROTATION}
        for component,curve in copied.items():
            for point,value in zip(curve.keyframe_points,quaternions):
                change=float(value[component])-point.co.y
                point.co.y=float(value[component])
                point.handle_left.y+=change;point.handle_right.y+=change
                point.interpolation='LINEAR'
            curve.update()
        if _records(source,helper)!=source_all: raise RuntimeError('Source action changed unexpectedly')
        if _records(created,helper,exclude_rotation=True)!=preserved:
            raise RuntimeError('A copied channel outside head rotation changed')
        if _records(created,helper,exclude_head=True)!=nonhead:
            raise RuntimeError('Non-head animation channels are not exactly equivalent')
        helper['_assign'](rig,created);verification=[]
        for frame in checks:
            _set_frame(scene,frame);before=check_source[frame]
            source_pitch,_=head_pitch(np.asarray(before['head']),rest)
            actual_pitch,_=head_pitch(np.asarray(head.matrix),rest)
            expected=math.radians(target_degrees)+source_variation*(source_pitch-mean_pitch)
            pivot_error=(head.matrix.translation-before['head'].translation).length
            foot_error=max(max(abs(rig.pose.bones[n].matrix[i][j]-before['feet'][n][i][j]) for i in range(4) for j in range(4)) for n in before['feet'])
            neck_error=max(abs(rig.pose.bones['neck'].matrix[i][j]-before['neck'][i][j]) for i in range(4) for j in range(4))
            if pivot_error>2e-6 or foot_error>2e-6 or neck_error>2e-6 or abs(actual_pitch-expected)>math.radians(.02):
                raise RuntimeError(f'Unexpected gaze/pivot/body evaluation mismatch at frame {frame}')
            verification.append({'frame':frame,'sourcePitchDegrees':math.degrees(source_pitch),
                'actualPitchDegrees':math.degrees(actual_pitch),'expectedPitchDegrees':math.degrees(expected),
                'headPivotErrorMeters':pivot_error,'footMatrixMaxError':foot_error,'neckMatrixMaxError':neck_error})
        if _records(source,helper)!=source_all or _records(created,helper,exclude_rotation=True)!=preserved:
            raise RuntimeError('Action channel equivalence changed during evaluation')
        audit_path=project_path()/'work/reconstruction-r02/cmu-v4-foot-audit-r06-eighth.json'
        upstream=None
        if audit_path.exists():
            audit=json.loads(audit_path.read_text(encoding='utf-8-sig'))
            foot_helper=runpy.run_path(str(Path(__file__).with_name('foot_contact_audit_r03.py')))
            matching=[row for row in audit.get('actions',[]) if row.get('action')==source.name]
            upstream={'path':str(audit_path.relative_to(project_path())),
                      'sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                      'sourceActionHashMatches':len(matching)==1 and matching[0].get('actionSha256')==foot_helper['_action_hash'](source)}
            if not upstream['sourceActionHashMatches']: raise RuntimeError('The claimed upstream foot audit does not match the actual source action')
        report={'action':created.name,'sourceAction':source.name,'sourceBlend':str(Path(bpy.data.filepath).relative_to(project_path())),
            'authoring':'Manual forward-gaze adaptation of the identified CMU retarget; not untouched mocap',
            'sourceSampleCount':len(frames),'sourceFrameRange':[frames[0],frames[-1]],'sceneFps':30,'sourceSampleFps':120,
            'forwardDefinition':'headPoseRotation @ inverse(headRestRotation) @ (0,-1,0)',
            'rotationAxis':'world UP cross normalized horizontal forward',
            'targetPitchFormulaDegrees':f'{target_degrees} + {source_variation} * (sourcePitch - arithmeticMeanOf316SourcePitches)',
            'sourceMeanPitchDegrees':math.degrees(mean_pitch),'onlyChangedChannels':HEAD_ROTATION,
            'allNonHeadFCurvesExactlyEqual':True,'headLocationAndScaleFCurvesExactlyEqual':True,
            'sourceActionExactlyUnchanged':True,'sourceAllFCurvesSha256':_digest(source_all),
            'sourceNonHeadFCurvesSha256':_digest(nonhead),'resultNonHeadFCurvesSha256':_digest(_records(created,helper,exclude_head=True)),
            'allPreservedFCurvesSha256':_digest(preserved),'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'maxComputedLocalPivotErrorMeters':max_basis_location_error,'maxComputedLocalScaleError':max_basis_scale_error,
            'upstreamFootAudit':upstream,'verification':verification,'sourceAndTargetPitches':metadata,
            'evidenceScope':'Unchanged non-head action channels preserve the independent leg-chain animation; prior foot evidence does not certify new hair, cloth or whole-scene collisions.',
            'cacheScope':'Rebake/rebind caches whose head motion or action fingerprint changed; no cache is modified here.',
            'upstreamMotionEvidence':'Copied r03_motion_report remains the original v4 provenance; this separate r06 report records the added head edit.',
            'savedBlend':False,'rendered':False,'accepted':False}
        created['r06_motion_origin']=report['authoring'];created['r06_forward_gaze_report']=json.dumps(report,ensure_ascii=False)
        out=(project_path()/report_relative).resolve();out.relative_to((project_path()/'work').resolve())
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        complete=True
        print(json.dumps({'action':created.name,'sourceKeys':len(frames),'sourceMeanPitchDegrees':report['sourceMeanPitchDegrees'],
            'nonHeadFCurvesExactlyEqual':True,'report':str(out),'savedBlend':False},ensure_ascii=False))
        return report
    finally:
        helper['_assign'](rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        scene.frame_set(old_frame,subframe=old_subframe)
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        for key,value in old_switch.items(): master['ik_'+key]=value
        rig.update_tag();bpy.context.view_layer.update()
        if created is not None and not complete: bpy.data.actions.remove(created)


if __name__=='__main__':
    print('Loaded only. create_action() explicitly creates a head-only authored derivative in the existing candidate.')
