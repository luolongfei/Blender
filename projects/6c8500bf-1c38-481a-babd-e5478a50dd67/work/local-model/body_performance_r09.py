"""Independent 10 s authored body-performance study, using existing R02 IK.

13.5 cm hip drop, 30 degree hip/torso bend, and arms rising to 80 degrees from
downward. Not mocap, a certified balance solution, or a cloth/hair acceptance.
Existing actions/caches are never edited. Import has no Blender side effects.
create_action() creates a new inactive FK action; validate_static_poses() checks
four major poses and three between-key poses. Neither renders, bakes or saves.
"""
import hashlib
import json
import math
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix,Vector
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
ACTION='R09 Body performance - authored squat bend reach'
MASTER='CTRL_R02_Master';IK=('leg_L','leg_R','arm_L','arm_R')
KEY_POSES={1:'Standing',55:'13.5 cm squat',151:'30 degree bend',247:'80 degree arm raise'}
FRAME_RANGE=(1,301)


def _ease(t):
    t=max(0.,min(1.,float(t)))
    return t*t*t*(10.-15.*t+6.*t*t)


def _envelope(frame,start,peak,hold,end):
    if frame<=start or frame>=end:return 0.
    if frame<peak:return _ease((frame-start)/(peak-start))
    if frame<=hold:return 1.
    return 1.-_ease((frame-hold)/(end-hold))


def pose_inputs(frame):
    """C2 onset/settle envelopes; no oscillatory substitute for performance."""
    squat=_envelope(frame,13,49,61,97)
    bend=_envelope(frame,109,145,157,193)
    raise_arms=_envelope(frame,205,241,253,289)
    # Arms anticipate the squat by four frames and settle four frames later.
    reach=_envelope(frame,9,45,65,101)
    return {'frame':float(frame),'squat':squat,'bend':bend,'raise':raise_arms,
        'hipDropMeters':.135*squat+.025*bend,'hipBackMeters':.055*squat+.040*bend,
        'hipHingeDegrees':5.*squat+20.*bend,'spineBendDegrees':3.*squat+4.*bend,
        'chestBendDegrees':4.*squat+6.*bend,
        'armForwardDegrees':38.*reach+12.*raise_arms,
        'armRaiseTargetFromDownDegrees':80.,'armRaiseWeight':raise_arms,
        'elbowAdditionalFlexDegrees':15.*reach+5.*bend+8.*raise_arms,
        'clavicleAdditionalRaiseDegrees':7.*raise_arms}


def timeline():
    return {'frameRange':list(FRAME_RANGE),'fps':30,'durationSeconds':10.,
        'segments':[{'name':'Squat and rise','frames':[1,109],'descent':[13,49],'hold':[49,61],'rise':[61,97]},
                    {'name':'Hip-led bend and return','frames':[109,205],'bend':[109,145],'hold':[145,157],'return':[157,193]},
                    {'name':'Arm raise and lower','frames':[205,301],'raise':[205,241],'hold':[241,253],'lower':[253,289]}],
        'keyPoses':[dict(label=label,**pose_inputs(frame)) for frame,label in KEY_POSES.items()],
        'authoring':'Authored coordinated pose sequence; not mocap or physical balance simulation.',
        'interpolation':'Inputs use fifth-order ease-in/out. The result is sampled at every 30 fps frame and stored as sign-continuous LINEAR FK keys.'}


def offline_reach_review(target_relative='work/reconstruction-r02/cmu-target-r04-actual.json'):
    """Read exported rest lengths only; this is not a live IK result."""
    path=project_path()/target_relative;target=json.loads(path.read_text(encoding='utf-8'))
    result={}
    for side in ('left','right'):
        upper=target['bones'][side+'UpperLeg'];lower=target['bones'][side+'LowerLeg'];foot=target['bones'][side+'Foot']
        hip=np.asarray(upper['head']);ankle=np.asarray(foot['head']);a=float(upper['length']);b=float(lower['length'])
        rows=[]
        for frame,label in KEY_POSES.items():
            inputs=pose_inputs(frame);moved=hip+np.array([0.,inputs['hipBackMeters'],-inputs['hipDropMeters']])
            distance=float(np.linalg.norm(moved-ankle));cosine=np.clip((distance*distance-a*a-b*b)/(2*a*b),-1,1)
            rows.append({'frame':frame,'label':label,'hipAnkleDistanceMeters':distance,
                         'extensionReserveMeters':a+b-distance,'estimatedKneeFlexionDegrees':math.degrees(math.acos(cosine)),
                         'withinTwoBoneReach':abs(a-b)<distance<a+b+1e-5})
        result[side]={'upperLengthMeters':a,'lowerLengthMeters':b,'poses':rows}
    return {'targetFile':str(path),'targetFileSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Existing exported leg lengths; rest-hip translation estimate. Does not include actual baseline FK, small hip-pivot rotation offsets, skin or clothing.',
        'legs':result,'timeline':timeline()}


def _modules():
    folder=Path(__file__).parent
    return tuple(runpy.run_path(str(folder/name)) for name in
                 ('full_body_review_r03.py','production_controls_r02.py','foot_contact_audit_r03.py'))


@contextmanager
def _preserve(rig,helper,controls):
    """Suspend simulation evaluation while solving, then restore all touched state."""
    scene=bpy.context.scene;master=rig.pose.bones[MASTER]
    action=rig.animation_data.action if rig.animation_data else None
    slot=rig.animation_data.action_slot if rig.animation_data else None
    frame=(scene.frame_current,scene.frame_subframe);pose_position=rig.data.pose_position
    basis={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
    switches={k:float(master['ik_'+k]) for k in IK}
    constraints={k:rig.pose.bones[controls['LIMBS'][k][1]].constraints[controls['PREFIX']+k+' IK'] for k in IK}
    angles={k:c.pole_angle for k,c in constraints.items()}
    cache={k:rig[k] for k in ('R04_native_hair_enabled',) if k in rig}
    cloth=[(m,m.show_viewport) for obj in bpy.data.objects for m in obj.modifiers if m.type=='CLOTH']
    try:
        for key in cache:rig[key]=0.
        for modifier,_ in cloth:modifier.show_viewport=False
        rig.data.pose_position='POSE'
        yield constraints
    finally:
        helper['_assign'](rig,action)
        if slot:rig.animation_data.action_slot=slot
        scene.frame_set(frame[0],subframe=frame[1])
        for name,value in basis.items():rig.pose.bones[name].matrix_basis=value
        for key,value in switches.items():master['ik_'+key]=value
        for key,value in angles.items():constraints[key].pole_angle=value
        rig.data.pose_position=pose_position
        for key,value in cache.items():rig[key]=value
        for modifier,value in cloth:modifier.show_viewport=value
        rig.update_tag();bpy.context.view_layer.update()


def _rotation(matrix,rotation):
    m=rotation.to_4x4();m.translation=matrix.translation
    return m


def _flatten(samples):
    return {f'{name}:{i}':Vector(point) for name,points in samples.items() for i,point in enumerate(points)}


def _probe(rig,selection,audit,ground,reference=None):
    sampled=audit['sample_sole_points'](selection);points=_flatten(sampled)
    heights={side:min(p[2]-ground for key,p in points.items() if key.split(':')[0].endswith('_'+side)) for side in ('L','R')}
    knees={};arms={}
    for side in ('left','right'):
        u=rig.pose.bones[side+'UpperLeg'];l=rig.pose.bones[side+'LowerLeg']
        knees[side]=math.degrees((u.tail-u.head).angle(l.tail-l.head))
        arm=rig.pose.bones[side+'UpperArm'];arms[side]=math.degrees((arm.tail-arm.head).angle(Vector((0,0,-1))))
    # A geometric torso/hip marker is useful for visual balance review, but it
    # is deliberately not labelled a measured human center of mass.
    guide=rig.pose.bones['hips'].head*.60+rig.pose.bones['chest'].head*.40
    return {'shoeLowestHeightMeters':heights,'maximumSelectedSolePointDisplacementMeters':
            max((p-reference[k]).length for k,p in points.items()) if reference else 0.,
            'kneeFlexionDegrees':knees,'upperArmAngleFromWorldDownDegrees':arms,
            'pelvisWorld':list(rig.matrix_world@rig.pose.bones['hips'].head),
            'torsoPelvisProjectionGuideXY':list(guide[:2]),
            'projectionGuideScope':'60% pelvis + 40% chest geometric marker, not physical COM or force/balance proof.'},points


def _pose(rig,inputs,base_basis,base_pose,feet,angles,constraints,axes):
    master=rig.pose.bones[MASTER];up,lateral,forward=axes
    for name,basis in base_basis.items():rig.pose.bones[name].matrix_basis=basis
    for key in IK:master['ik_'+key]=1. if key.startswith('leg_') else 0.
    for key,value in angles.items():constraints[key].pole_angle=value
    hip=base_pose['hips'].copy();hip.translation+=-up*inputs['hipDropMeters']-forward*inputs['hipBackMeters']
    hip=_rotation(hip,Matrix.Rotation(math.radians(inputs['hipHingeDegrees']),3,lateral)@base_pose['hips'].to_3x3())
    rig.pose.bones['hips'].matrix=hip
    for name,key in (('spine','spineBendDegrees'),('chest','chestBendDegrees')):
        axis=base_pose[name].to_3x3().inverted()@lateral
        rig.pose.bones[name].matrix_basis=base_basis[name]@Matrix.Rotation(math.radians(inputs[key]),4,axis)
    for side,sign in (('left',1.),('right',-1.)):
        name=side+'Shoulder';axis=base_pose[name].to_3x3().inverted()@forward
        rig.pose.bones[name].matrix_basis=base_basis[name]@Matrix.Rotation(sign*math.radians(inputs['clavicleAdditionalRaiseDegrees']),4,axis)
    for side,matrix in feet.items():rig.pose.bones['CTRL_R02_Foot_'+side].matrix=matrix
    rig.update_tag();bpy.context.view_layer.update()
    rotations={}
    for side,sign in (('left',1.),('right',-1.)):
        u=base_pose[side+'UpperArm'].to_3x3();l=base_pose[side+'LowerArm'].to_3x3();h=base_pose[side+'Hand'].to_3x3()
        udir=u.col[1].normalized();ldir=l.col[1].normalized()
        rest_elevation=math.degrees(math.atan2(sign*udir.dot(lateral),-udir.dot(up)))
        raise_delta=inputs['armRaiseWeight']*(inputs['armRaiseTargetFromDownDegrees']-rest_elevation)
        delta=Matrix.Rotation(-math.radians(inputs['armForwardDegrees']),3,lateral)@Matrix.Rotation(sign*math.radians(raise_delta),3,forward)
        plane=ldir-udir*ldir.dot(udir)
        if plane.length<1e-5:plane=forward-udir*forward.dot(udir)
        plane.normalize();bend=udir.angle(ldir)+math.radians(inputs['elbowAdditionalFlexDegrees'])
        desired_lower=delta@(udir*math.cos(bend)+plane*math.sin(bend))
        upper=delta@u;pre_lower=delta@l
        lower=pre_lower.col[1].normalized().rotation_difference(desired_lower.normalized()).to_matrix()@pre_lower
        rotations[side]={'UpperArm':upper,'LowerArm':lower,'Hand':lower@l.inverted()@h}
    # World-directed arm gestures keep the bend's relaxed arms hanging instead
    # of pulling both hands rigidly backwards with the tilted chest.
    for suffix in ('UpperArm','LowerArm','Hand'):
        for side in ('left','right'):
            pb=rig.pose.bones[side+suffix];pb.matrix=_rotation(pb.matrix,rotations[side][suffix])
        rig.update_tag();bpy.context.view_layer.update()


def create_action(action_name=ACTION,base_action='R02 Idle - authored IK source',base_frame=1,
                  ground_z=None,report_relative='work/reconstruction-r02/body-performance-r09.json'):
    """Create 301 FK samples through the existing IK controls; leave inactive."""
    if bpy is None:raise RuntimeError('Explicit invocation in existing Blender required')
    helper,controls,audit=_modules();rig=helper['_guard']();scene=bpy.context.scene
    if bpy.data.actions.get(action_name):raise RuntimeError('Use a new action name; existing actions are retained')
    source=bpy.data.actions.get(base_action)
    if source is None:raise RuntimeError('Existing relaxed standing baseline action is required')
    path=(project_path()/report_relative).resolve();path.relative_to((project_path()/'work').resolve())
    if path.exists():raise RuntimeError('Use a new report path; prior evidence is retained')
    ground,ground_source=helper['_ground_height'](ground_z)
    created=None;complete=False;probes=[];max_snap=0.;max_drift=0.;max_ground=0.
    try:
        with _preserve(rig,helper,controls) as constraints:
            helper['_assign'](rig,source);scene.frame_set(int(base_frame))
            master=rig.pose.bones[MASTER]
            for key in IK:master['ik_'+key]=0.
            rig.update_tag();bpy.context.view_layer.update()
            initial_match={key:controls['snap_ik_to_fk'](key,enable=True,keyframe=False) for key in ('leg_L','leg_R')}
            for key in IK:master['ik_'+key]=0.
            rig.update_tag();bpy.context.view_layer.update()
            base_basis={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
            base_pose={p.name:p.matrix.copy() for p in rig.pose.bones}
            angles={k:c.pole_angle for k,c in constraints.items()}
            up=Vector((0,0,1));lateral=base_pose['leftUpperArm'].translation-base_pose['rightUpperArm'].translation
            lateral-=up*lateral.dot(up);lateral.normalize();forward=lateral.cross(up).normalized()
            toe_forward=sum((rig.pose.bones[s+'Foot'].tail-rig.pose.bones[s+'Foot'].head for s in ('left','right')),Vector())
            if toe_forward.dot(forward)<=0:raise RuntimeError('Body lateral/forward inference disagrees with feet')
            axes=(up,lateral,forward)
            selection=audit['select_sole_vertices'](objects=helper['SHOES'],max_points=12)
            first,_=_probe(rig,selection,audit,ground)
            offsets={side:-first['shoeLowestHeightMeters'][side] for side in ('L','R')}
            if max(abs(v) for v in offsets.values())>.020:raise RuntimeError('Baseline sole is over 20 mm from ground')
            feet={side:rig.pose.bones['CTRL_R02_Foot_'+side].matrix.copy() for side in ('L','R')}
            for side,matrix in feet.items():matrix.translation+=up*offsets[side]
            drivers={f.data_path for f in rig.animation_data.drivers};keyprops={}
            for pb in rig.pose.bones:
                rotation='rotation_quaternion' if pb.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if pb.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
                keyprops[pb.name]=[prop for prop in ('location',rotation,'scale') if pb.path_from_id(prop) not in drivers]
            created=bpy.data.actions.new(action_name);created.use_fake_user=True;helper['_assign'](rig,created)
            previous={};reference=None
            for frame in range(1,302):
                scene.frame_set(frame);inputs=pose_inputs(frame)
                _pose(rig,inputs,base_basis,base_pose,feet,angles,constraints,axes)
                # Existing, previously tested controller API performs the FK
                # bake. No animated IK or new constraint survives in output.
                for key in ('leg_L','leg_R'):
                    result=controls['snap_fk_to_ik'](key,disable=True,keyframe=False,max_matrix_error=2e-5)
                    max_snap=max(max_snap,result['maxPoseMatrixError'])
                row,points=_probe(rig,selection,audit,ground,reference)
                if reference is None:reference=points
                drift=row['maximumSelectedSolePointDisplacementMeters'];height=max(abs(v) for v in row['shoeLowestHeightMeters'].values())
                max_drift=max(max_drift,drift);max_ground=max(max_ground,height)
                if drift>.001 or height>.002:raise RuntimeError(f'Planted shoe check failed at frame {frame}: {row}')
                if max(row['kneeFlexionDegrees'].values())>100.:raise RuntimeError('Actual knee bend exceeds the moderate-squat scope')
                if frame in KEY_POSES:probes.append(dict(label=KEY_POSES[frame],inputs=inputs,**row))
                for pb in rig.pose.bones:
                    if pb.rotation_mode=='QUATERNION':
                        if pb.name in previous and pb.rotation_quaternion.dot(previous[pb.name])<0:pb.rotation_quaternion.negate()
                        previous[pb.name]=pb.rotation_quaternion.copy()
                    for prop in keyprops[pb.name]:pb.keyframe_insert(prop,frame=frame,group=pb.name)
                for key in IK:
                    master['ik_'+key]=0.;master.keyframe_insert(data_path='["ik_'+key+'"]',frame=frame,group=MASTER)
            for curve in helper['_fcurves'](created):
                for key in curve.keyframe_points:key.interpolation='LINEAR'
            report={'action':action_name,'sourceBlend':bpy.data.filepath,'baseAction':base_action,'baseFrame':base_frame,
                'timeline':timeline(),'initialLegMatch':initial_match,'footVerticalCorrectionMeters':offsets,
                'groundZ':ground,'groundSource':ground_source,'soleVertexSelection':selection,
                'allProductionIKSwitches':0,'sampleCount':301,'keySampleFps':30,
                'maximumIKtoFKMatrixError':max_snap,'maximumIntegerSolePointDriftMeters':max_drift,
                'maximumIntegerLowestSoleHeightAbsoluteMeters':max_ground,'keyPoseChecks':probes,
                'createdActionIsActiveOnReturn':False,'savedBlend':False,'rendered':False,'accepted':False,
                'limitations':['Explicitly authored performance study, not captured natural motion.',
                    '13.5 cm lowering on approximately 70 cm legs produces about 72 degrees knee flexion; no fixed quarter-squat angle is claimed.',
                    'Pelvis/trunk and arms are coordinated smoothly; dynamic balance and contact forces are not simulated.',
                    'Only finite original shoe cages are checked; bevel, body, sleeve, skirt and hair collisions require the new combination review.',
                    'New cloth/hair caches must be produced for this action; old R03/R06 cached motion is not reused.',
                    'Integer FK sampling does not prove arbitrary subframe contact; validate_static_poses includes three bounded between-key probes.']}
            created['r09_performance_origin']=report['timeline']['authoring'];created['r09_performance_report']=json.dumps(report,ensure_ascii=False)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            complete=True
        return report
    finally:
        if created is not None and not complete:bpy.data.actions.remove(created)


def validate_static_poses(action_name=ACTION,ground_z=None,
                          report_relative='work/reconstruction-r02/body-performance-r09-static.json'):
    """Four major poses and three mid-transition half frames; no render/save."""
    if bpy is None:raise RuntimeError('Explicit invocation in existing Blender required')
    helper,controls,audit=_modules();rig=helper['_guard']();action=bpy.data.actions.get(action_name)
    if action is None or 'r09_performance_report' not in action:raise RuntimeError('Create the independent R09 action first')
    path=(project_path()/report_relative).resolve();path.relative_to((project_path()/'work').resolve())
    if path.exists():raise RuntimeError('Use a new static report path')
    ground,_=helper['_ground_height'](ground_z);selection=audit['select_sole_vertices'](objects=helper['SHOES'],max_points=12)
    rows=[];reference=None
    with _preserve(rig,helper,controls):
        helper['_assign'](rig,action)
        for frame in (1.,31.5,55.,127.5,151.,223.5,247.):
            bpy.context.scene.frame_set(math.floor(frame),subframe=frame%1)
            rig.update_tag();bpy.context.view_layer.update()
            row,points=_probe(rig,selection,audit,ground,reference)
            if reference is None:reference=points
            rows.append(dict(frame=frame,label=KEY_POSES.get(frame,'Between-key transition'),**row))
    report={'action':action_name,'probes':rows,'passedFiniteShoeCriteria':all(
        r['maximumSelectedSolePointDisplacementMeters']<.001 and
        max(abs(v) for v in r['shoeLowestHeightMeters'].values())<.002 for r in rows),
        'scope':'Four major poses plus three half frames. Numerical shoe-cage check only; render poses 1/55/151/247 and review combined new cloth/hair.',
        'rendered':False,'savedBlend':False,'accepted':False}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print('Loaded only. create_action() authors a new inactive FK action. validate_static_poses() performs seven bounded probes. Neither renders, bakes or saves.')
