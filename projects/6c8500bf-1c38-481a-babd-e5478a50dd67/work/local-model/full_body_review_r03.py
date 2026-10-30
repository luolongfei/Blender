"""Create a bounded authored deformation review, not mocap or a gait clip.

Uses the existing R02 IK controls and preserves the active action/pose on exit.
Only run() creates an independent action; no Blend is saved or rendered.
The optional contact helper samples a few named shoe-cage vertices, not a full
collision pass. Continuous motion and garment/hair combinations remain QA work.
"""
import json
import math
import runpy
import sys
from pathlib import Path
import bpy
from mathutils import Matrix,Vector

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

RIG='Character_Rig';MASTER='CTRL_R02_Master';PREFIX='R02 Production '
KEYS=('leg_L','leg_R','arm_L','arm_R')
SHOES=('Shoe_Outsole_L','Shoe_Stacked_Heel_L','Shoe_Outsole_R','Shoe_Stacked_Heel_R')
DEFAULTS={'arm_raise_degrees':38.0,'elbow_flex_degrees':10.0,'bend_degrees':14.0,
          'turn_degrees':12.0,'squat_meters':0.04,'hip_back_meters':0.012}
LIMITS={'arm_raise_degrees':55.0,'elbow_flex_degrees':25.0,'bend_degrees':22.0,
        'turn_degrees':20.0,'squat_meters':0.06,'hip_back_meters':0.025}


def timeline(amplitudes=None):
    """Pure-data specification; all durations are explicit at 30 fps."""
    p=dict(DEFAULTS)
    if amplitudes:
        if set(amplitudes)-set(p): raise ValueError('Unknown amplitude keys')
        p.update(amplitudes)
    if any(not math.isfinite(v) or v<0 or v>LIMITS[k] for k,v in p.items()):
        raise ValueError('Use finite, nonnegative amplitudes within the bounded review limits')
    def pose(frame,label,ik=1.0,arms=0.0,bend=0.0,turn=0.0,squat=0.0):
        return {'frame':frame,'seconds':(frame-1)/30,'label':label,'leg_ik':ik,
                'arm_raise':arms*math.radians(p['arm_raise_degrees']),
                'elbow_flex':arms*math.radians(p['elbow_flex_degrees']),
                'bend':bend*math.radians(p['bend_degrees']),
                'turn':turn*math.radians(p['turn_degrees']),
                'hip_drop':squat*p['squat_meters'],
                'hip_back':max(bend*.65,squat)*p['hip_back_meters']}
    return p,[pose(1,'Baseline standing / all IK off',ik=0),
              pose(10,'Feet gently plant / legs IK on'),pose(31,'Standing hold'),
              pose(49,'Moderate bilateral arm raise',arms=1),pose(67,'Arm raise hold',arms=1),
              pose(91,'Arms lower to baseline'),pose(109,'Small forward torso bend',bend=1),
              pose(127,'Forward bend hold',bend=1),pose(151,'Torso returns upright'),
              pose(169,'Small torso turn left',turn=1),pose(187,'Small torso turn right',turn=-1),
              pose(211,'Torso returns to center'),pose(229,'Small squat',squat=1,bend=.5),
              pose(247,'Small squat hold',squat=1,bend=.5),pose(271,'Rise to baseline'),
              pose(286,'Final standing hold'),pose(301,'Baseline standing / all IK off',ik=0)]


def _guard():
    path=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or path==(project_path()/'work/character.blend').resolve():
        raise RuntimeError('Use a saved independent candidate, never the formal main source')
    path.relative_to((project_path()/'work').resolve())
    rig=bpy.data.objects.get(RIG)
    if rig is None or rig.type!='ARMATURE' or rig.library or rig.data.library:
        raise RuntimeError('Local Character_Rig required')
    if rig.get('r02_controls_version')!=1 or MASTER not in rig.pose.bones:
        raise RuntimeError('The previously validated R02 production controls are required')
    if bpy.context.mode not in {'OBJECT','POSE'}: raise RuntimeError('Leave edit/sculpt mode first')
    if abs(bpy.context.scene.render.fps/bpy.context.scene.render.fps_base-30)>1e-6:
        raise RuntimeError('Use the existing 30 fps action timebase')
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA needs explicit integration')
    if any(o.get('r02_secondary_hair_session') for o in bpy.data.objects):
        raise RuntimeError('Detach the hair playback cache before creating another action')
    return rig


def _assign(rig,action):
    rig.animation_data_create();rig.animation_data.action=action
    if action and hasattr(action,'slots'):
        slots=[s for s in action.slots if s.target_id_type=='OBJECT']
        if len(slots)>1: raise RuntimeError('Expected one object slot for this rig action')
        if slots: rig.animation_data.action_slot=slots[0]


def _fcurves(action):
    result=[]
    if hasattr(action,'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip,'channelbags'):
                    for bag in strip.channelbags: result.extend(bag.fcurves)
    if not result and hasattr(action,'fcurves'): result=list(action.fcurves)
    return result


def _ground_height(value):
    if value is not None:
        if not math.isfinite(value): raise ValueError('Non-finite ground height')
        return float(value),'explicit horizontal ground plane'
    obj=bpy.data.objects.get('Cyclorama floor')
    if obj is None or obj.type!='MESH' or len(obj.data.vertices)>16:
        raise RuntimeError('Pass ground_z explicitly when the small existing floor plane is unavailable')
    heights=[(obj.matrix_world@v.co).z for v in obj.data.vertices]
    if not heights or max(heights)-min(heights)>1e-5:
        raise RuntimeError('Only a horizontal ground plane is supported')
    return sum(heights)/len(heights),'current Cyclorama floor world vertices'


def _contacts(sample,ground):
    report={}
    for side in ('L','R'):
        points=[q for name,values in sample.items() if name.endswith('_'+side) for q in values]
        if not points: raise RuntimeError('No sole samples for '+side)
        heights=[q[2]-ground for q in points]
        report[side]={'lowestHeightMeters':min(heights),'highestSelectedHeightMeters':max(heights),
                      'sampleCount':len(points)}
    return report


def run(action_name='R03 Full body review - authored diagnostic',
        base_action='R02 Idle - authored IK source',base_frame=1,ground_z=None,
        amplitudes=None,report_relative='work/reconstruction-r02/full-body-review-r03.json'):
    """Create a repeatable 10-second diagnostic action, leaving it inactive.

    Frames 1/301 have all four switches zero. Legs blend on by frame 10 and
    remain planted through frame 286; arms use the original FK bones throughout.
    No action is overwritten. Foot planting is calibrated against a few shoe
    cage points; a large starting height mismatch aborts instead of stretching.
    """
    rig=_guard();parameters,poses=timeline(amplitudes)
    if bpy.data.actions.get(action_name): raise RuntimeError('Action already exists; choose a distinct name')
    source=bpy.data.actions.get(base_action)
    if source is None: raise RuntimeError('Existing base body action missing: '+base_action)
    ground,ground_source=_ground_height(ground_z)
    out=(project_path()/report_relative).resolve();out.relative_to((project_path()/'work').resolve())
    controls=runpy.run_path(str(Path(__file__).with_name('production_controls_r02.py')))
    audit=runpy.run_path(str(Path(__file__).with_name('foot_contact_audit_r03.py')))
    scene=bpy.context.scene;master=rig.pose.bones[MASTER]
    old_action=rig.animation_data.action if rig.animation_data else None
    old_slot=rig.animation_data.action_slot if rig.animation_data and hasattr(rig.animation_data,'action_slot') else None
    old_frame=scene.frame_current;old_subframe=scene.frame_subframe
    old_basis={b.name:b.matrix_basis.copy() for b in rig.pose.bones}
    old_switch={k:float(master['ik_'+k]) for k in KEYS}
    constraints={k:rig.pose.bones[controls['LIMBS'][k][1]].constraints[PREFIX+k+' IK'] for k in KEYS}
    old_angles={k:c.pole_angle for k,c in constraints.items()}
    created=None;completed=False
    try:
        _assign(rig,source);scene.frame_set(int(base_frame))
        for k in KEYS: master['ik_'+k]=0.0
        rig.update_tag();bpy.context.view_layer.update()
        snapshots={b.name:b.matrix.copy() for b in rig.pose.bones}
        match={}
        for k in ('leg_L','leg_R'): match[k]=controls['snap_ik_to_fk'](k,enable=True,keyframe=False)
        for k in KEYS: master['ik_'+k]=0.0
        rig.update_tag();bpy.context.view_layer.update()
        baseline={b.name:b.matrix_basis.copy() for b in rig.pose.bones}
        angles={k:c.pole_angle for k,c in constraints.items()}
        selection=audit['select_sole_vertices'](objects=SHOES,max_points=12)
        initial_contacts=_contacts(audit['sample_sole_points'](selection),ground)
        offsets={side:-initial_contacts[side]['lowestHeightMeters'] for side in ('L','R')}
        if any(abs(v)>0.020 for v in offsets.values()):
            raise RuntimeError('Baseline shoe cage is over 20 mm from the floor; choose a valid standing baseline: '+str(offsets))
        feet={side:rig.pose.bones['CTRL_R02_Foot_'+side].matrix.copy() for side in ('L','R')}
        planted={side:m.copy() for side,m in feet.items()}
        # matrix_world is confirmed unit at controller creation, but use its
        # inverse vector transform so the grounding offset is explicitly world Z.
        world_to_rig=rig.matrix_world.to_3x3().inverted()
        for side in planted: planted[side].translation+=world_to_rig@Vector((0,0,offsets[side]))
        up=(snapshots['head'].translation-snapshots['hips'].translation).normalized()
        lateral=(snapshots['leftUpperArm'].translation-snapshots['rightUpperArm'].translation).normalized()
        forward=up.cross(lateral).normalized()
        foot_forward=sum((rig.pose.bones[s+'Foot'].tail-rig.pose.bones[s+'Foot'].head for s in ('left','right')),Vector())
        if forward.dot(foot_forward)<0: forward=-forward
        axes={name:{key:(matrix.to_3x3().inverted()@axis).normalized()
                    for key,axis in [('up',up),('lateral',lateral),('forward',forward)]}
              for name,matrix in snapshots.items()}
        # Directly driven transform channels (e.g. jaw opening) remain driven.
        driver_paths={f.data_path for f in (rig.animation_data.drivers if rig.animation_data else [])}
        keyprops={}
        for b in rig.pose.bones:
            rotation='rotation_quaternion' if b.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if b.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
            props=('location',rotation,'scale')
            keyprops[b.name]=[p for p in props if b.path_from_id(p) not in driver_paths]
        created=bpy.data.actions.new(action_name);created.use_fake_user=True;_assign(rig,created)
        previous_quaternions={};probes=[]
        for pose in poses:
            scene.frame_set(pose['frame'])
            for name,basis in baseline.items(): rig.pose.bones[name].matrix_basis=basis
            for k in KEYS: master['ik_'+k]=pose['leg_ik'] if k.startswith('leg_') else 0.0
            for k,c in constraints.items(): c.pole_angle=angles[k]
            hips=rig.pose.bones['hips'];m=snapshots['hips'].copy()
            m.translation+=-up*pose['hip_drop']-forward*pose['hip_back'];hips.matrix=m
            def rotate(name,axis,angle):
                if abs(angle)>1e-12:
                    b=rig.pose.bones[name];b.matrix_basis=b.matrix_basis@Matrix.Rotation(angle,4,axes[name][axis])
            rotate('spine','lateral',pose['bend']*.45);rotate('chest','lateral',pose['bend']*.55)
            rotate('spine','up',pose['turn']*.40);rotate('chest','up',pose['turn']*.60)
            for side,sign in [('left',1),('right',-1)]:
                rotate(side+'UpperArm','forward',pose['arm_raise']*sign)
                rotate(side+'LowerArm','lateral',-pose['elbow_flex'])
            for side in ('L','R'):
                rig.pose.bones['CTRL_R02_Foot_'+side].matrix=planted[side] if pose['leg_ik'] else feet[side]
            rig.update_tag();bpy.context.view_layer.update()
            for b in rig.pose.bones:
                if b.rotation_mode=='QUATERNION':
                    q=b.rotation_quaternion
                    if b.name in previous_quaternions and q.dot(previous_quaternions[b.name])<0: q.negate()
                    previous_quaternions[b.name]=q.copy()
                for prop in keyprops[b.name]: b.keyframe_insert(prop,frame=pose['frame'],group=b.name)
            for k in KEYS:
                master.keyframe_insert(data_path='["ik_'+k+'"]',frame=pose['frame'],group=MASTER)
                constraints[k].keyframe_insert('pole_angle',frame=pose['frame'],group=controls['LIMBS'][k][1])
            contact=_contacts(audit['sample_sole_points'](selection),ground)
            errors={side:(rig.pose.bones[side+'Foot'].head-rig.pose.bones['CTRL_R02_Foot_'+label].head).length
                    for side,label in [('left','L'),('right','R')]}
            knees={}
            for side in ('left','right'):
                a=rig.pose.bones[side+'UpperLeg'].head;b=rig.pose.bones[side+'LowerLeg'].head;c=rig.pose.bones[side+'LowerLeg'].tail
                knees[side]=math.degrees(math.pi-(a-b).angle(c-b))
            probes.append({'frame':pose['frame'],'label':pose['label'],'shoeCageContact':contact,
                           'ankleTargetDistanceMeters':errors,'kneeBendDegrees':knees})
            if pose['leg_ik'] and (max(errors.values())>0.002 or any(abs(c['lowestHeightMeters'])>0.003 for c in contact.values())):
                raise RuntimeError('Planted key pose failed the limited ankle/sole check: '+json.dumps(probes[-1]))
        for curve in _fcurves(created):
            is_switch='["ik_' in curve.data_path
            for key in curve.keyframe_points:
                key.interpolation='LINEAR' if is_switch else 'BEZIER'
                key.handle_left_type='AUTO_CLAMPED';key.handle_right_type='AUTO_CLAMPED'
        report={'action':created.name,'source':'Explicitly authored bounded deformation review; not mocap',
                'baseAction':base_action,'baseFrame':base_frame,'frameRange':[1,301],'fps':30,'durationSeconds':10.0,
                'amplitudes':parameters,'keyPoses':poses,'initialIkSwitches':dict.fromkeys(KEYS,0.0),
                'finalIkSwitches':dict.fromkeys(KEYS,0.0),'footPlantFrames':[10,286],
                'groundZ':ground,'groundSource':ground_source,'soleVertexSelection':selection,
                'initialShoeCageContact':initial_contacts,'footVerticalCorrectionsMeters':offsets,
                'initialLegMatch':match,'keyPoseChecks':probes,'createdActionIsActiveOnReturn':False,
                'savedBlend':False,'limitations':['Key-pose checks only; audit interpolated frames and render the continuous sequence.',
                    'Shoe samples may use rigid source-cage reconstruction where bevel changes topology; not the final bevel surface.',
                    'This does not validate garment penetration, shoulder deformation, balance, or foot pressure.',
                    'No foot roll, toe pivot, full locomotion, or deep squat is claimed.',
                    'Independent face/other-ID actions remain as assigned; review combined performance separately.',
                    'Hair-cache playback must support the action\'s animated leg IK switches before combining.']}
        created['r03_diagnostic_source']=report['source'];created['r03_diagnostic_report']=json.dumps(report,ensure_ascii=False)
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        completed=True;print(json.dumps(report,ensure_ascii=False));return report
    finally:
        _assign(rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        scene.frame_set(old_frame,subframe=old_subframe)
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        for k,v in old_switch.items(): master['ik_'+k]=v
        for k,v in old_angles.items(): constraints[k].pole_angle=v
        rig.update_tag();bpy.context.view_layer.update()
        if created is not None and not completed: bpy.data.actions.remove(created)


if __name__=='__main__':
    print('Loaded only. Call run() explicitly to create the independent diagnostic action.')
