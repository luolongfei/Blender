"""Proportional, contact-aware CMU 07_01 walk adaptation for this character.

The original ASF/AMC stays unchanged. Source timing/rotations are evaluated by
cmu_asf_amc_r03.py. A two-bone solve uses the actual target limb lengths; a small
set of shoe-cage support points constrains ground contact. Output is a new FK
action with existing IK switches off, suitable for subsequent finite audits.
No Blender launch, rendering, Blend save, motion-loop claim, or acceptance.
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

RIG='Character_Rig';MASTER='CTRL_R02_Master';IK_KEYS=('leg_L','leg_R','arm_L','arm_R')
CORE={'hips':'root','spine':'lowerback','chest':'upperback','upperChest':'thorax',
      'neck':'upperneck','head':'head','leftShoulder':'lclavicle','rightShoulder':'rclavicle'}
LIMITATIONS=[
    'Retargeted preview from identified CMU mocap, not an untouched source or validated final animation.',
    'The captured clip begins/ends in motion; it is not an idle transition or seamless loop.',
    'Target limb proportions, shoe contact and pelvis reach corrections change the source trajectory.',
    'Contact is inferred from source ball/toe height and speed; no measured contact forces are available.',
    'Shoe constraints use finite rigid original-cage points and omit bevel-generated surface and floor compliance.',
    'Source finger/thumb channels are not transferred; existing relaxed fingers are retained.',
    'Source toes are noisy and are not copied independently; shoes remain governed by the foot bone.',
    'No garment/skin/hair collision acceptance follows from numerical bone or contact checks.',
    'Source files remain local dependencies under the recorded CMU terms; do not sell converted motion data by itself.']


def _unit(v,fallback=None):
    n=np.linalg.norm(v)
    if n>1e-10: return v/n
    if fallback is None: raise RuntimeError('Degenerate motion direction')
    return _unit(np.asarray(fallback,dtype=float))


def _align(a,b):
    a=_unit(a);b=_unit(b);c=float(np.clip(a@b,-1,1));v=np.cross(a,b)
    if c>1-1e-10: return np.eye(3)
    if c<-1+1e-8:
        axis=_unit(np.cross(a,np.eye(3)[int(np.argmin(np.abs(a)))]))
        return 2*np.outer(axis,axis)-np.eye(3)
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k/(1+c)


def _blend_rotation(a,b,t):
    u,_,vh=np.linalg.svd(a*(1-t)+b*t)
    if np.linalg.det(u@vh)<0: u[:,-1]*=-1
    return u@vh


def _two_bone(a,goal,pole,l1,l2,fallback):
    direction=_unit(goal-a,fallback);distance=np.linalg.norm(goal-a)
    reach=np.clip(distance,abs(l1-l2)+1e-5,(l1+l2)*(1-1e-5))
    end=a+direction*reach
    plane=pole-a-direction*((pole-a)@direction)
    if np.linalg.norm(plane)<1e-5:
        plane=fallback-direction*(fallback@direction)
    if np.linalg.norm(plane)<1e-5:
        candidate=np.eye(3)[int(np.argmin(np.abs(direction)))]
        plane=candidate-direction*(candidate@direction)
    plane=_unit(plane)
    along=(l1*l1-l2*l2+reach*reach)/(2*reach)
    perpendicular=math.sqrt(max(0,l1*l1-along*along))
    elbow=a+direction*along+plane*perpendicular
    return elbow,end,float(np.linalg.norm(end-goal))


def _smoothstep(a,b,x):
    q=np.clip((x-a)/(b-a),0,1)
    return q*q*(3-2*q)


def _smooth(values):
    return np.convolve(np.pad(values,(1,1),mode='edge'),[.25,.5,.25],mode='valid')


def _runs(mask):
    result=[];start=None
    for i,value in enumerate(list(mask)+[False]):
        if value and start is None: start=i
        if not value and start is not None:
            result.append((start,i-1));start=None
    return result


def source_contacts(clip):
    """Finite source-only contact candidates; no assertion of measured support."""
    index={n:i for i,n in enumerate(clip['names'])};p=clip['positions'];times=clip['times']
    raw={}
    for side in ('l','r'):
        ball=p[:,index[side+'foot']];toe=p[:,index[side+'toes']]
        height=_smooth(np.minimum(ball[:,2],toe[:,2]))
        speed=np.minimum(np.linalg.norm(np.gradient(ball[:,:2],times,axis=0),axis=1),
                         np.linalg.norm(np.gradient(toe[:,:2],times,axis=0),axis=1))
        raw[side]=(height,_smooth(speed))
    floor=float(np.percentile(np.concatenate([x[0] for x in raw.values()]),5))
    low=np.minimum(raw['l'][0],raw['r'][0])
    # Early CMU data can drift vertically. For this identified walking clip,
    # estimate a slowly varying source support plane from the low-foot envelope;
    # report and remove this drift, not the pelvis bob relative to that plane.
    padded=np.pad(low,(3,3),mode='edge')
    floor_track=_smooth(np.array([np.percentile(padded[i:i+7],20) for i in range(len(low))]))
    result={}
    for side,(height,speed) in raw.items():
        confidence=(1-_smoothstep(.015,.065,height-floor_track))*(1-_smoothstep(.10,.55,speed))
        mask=confidence>.40
        for f in range(1,len(mask)-1):
            if not mask[f] and mask[f-1] and mask[f+1]: mask[f]=True
        for a,b in _runs(mask):
            if b-a+1<3: mask[a:b+1]=False
        weight=np.zeros(len(mask));flat_weight=np.zeros(len(mask));runs=_runs(mask)
        for a,b in runs:
            for f in range(a,b+1):
                weight[f]=min(1,(f-a+1)/2,(b-f+1)/2)
                phase=(f-a)/max(1,b-a)
                flat_weight[f]=weight[f]*_smoothstep(0,.20,phase)*(1-_smoothstep(.55,.90,phase))
        result[side]={'height':height,'speed':speed,'confidence':confidence,'weight':weight,
                      'flatWeight':flat_weight,'runs':runs,'floorTrack':floor_track}
    if not any(x['runs'] for x in result.values()): raise RuntimeError('No stable source contact candidates detected')
    return floor,result


def _bone(target,name):
    return target['bones'][name]


def _rest_matrix(target,name):
    return np.asarray(_bone(target,name)['matrix'],dtype=float)


def _rest_point(target,name,key='head'):
    return np.asarray(_bone(target,name)[key],dtype=float)


def _length(target,name):
    return float(_bone(target,name)['length'])


def build_plan(clip,target,root_motion=True):
    """NumPy-only source-to-target plan. target is from read_target()."""
    times=clip['times'];F=len(times);index={n:i for i,n in enumerate(clip['names'])}
    srcp=clip['positions'];srcr=clip['rotations'];ground=float(target['groundZ'])
    required=set(CORE)|{side+suffix for side in ('left','right') for suffix in ('UpperArm','LowerArm','Hand','UpperLeg','LowerLeg','Foot')}
    missing=required-set(target['bones'])
    if missing: raise RuntimeError('Required target bones missing: '+str(sorted(missing)))
    leg_ratios={side:(_length(target,long+'UpperLeg')+_length(target,long+'LowerLeg'))/
                (clip['lengths'][side+'femur']+clip['lengths'][side+'tibia']) for side,long in [('l','left'),('r','right')]}
    arm_ratios={side:(_length(target,long+'UpperArm')+_length(target,long+'LowerArm'))/
                (clip['lengths'][side+'humerus']+clip['lengths'][side+'radius']+clip['lengths'][side+'wrist']) for side,long in [('l','left'),('r','right')]}
    root_scale=sum(leg_ratios.values())/2
    if not .4<root_scale<1.5: raise RuntimeError('Unexpected source/target scale; inspect units before adapting')
    root=srcp[:,index['root']].copy();root-=root[0]
    if not root_motion: root[:,:2]=0
    root=root*root_scale+_rest_point(target,'hips')
    rotations={name:srcr[:,index[source]]@_rest_matrix(target,name)[:3,:3] for name,source in CORE.items()}
    def limb_rotation(name,source):
        rest=_rest_matrix(target,name)[:3,:3]
        correction=_align(rest[:,1],clip['rest_directions'][index[source]])
        return srcr[:,index[source]]@correction@rest
    for side,long in [('l','left'),('r','right')]:
        for suffix,source in [('UpperLeg','femur'),('LowerLeg','tibia'),('Foot','foot'),
                              ('UpperArm','humerus'),('LowerArm','wrist'),('Hand','hand')]:
            rotations[long+suffix]=limb_rotation(long+suffix,side+source)
    def core_heads(f,root_position):
        heads={'hips':root_position.copy()}
        for name in CORE:
            if name=='hips': continue
            parent=_bone(target,name)['parent']
            offset=_rest_point(target,name)-_rest_point(target,parent)
            heads[name]=heads[parent]+rotations[parent][f]@(_rest_matrix(target,parent)[:3,:3].T@offset)
        return heads
    def attached_head(f,name,heads):
        parent=_bone(target,name)['parent']
        offset=_rest_point(target,name)-_rest_point(target,parent)
        return heads[parent]+rotations[parent][f]@(_rest_matrix(target,parent)[:3,:3].T@offset)
    source_floor,contacts=source_contacts(clip)
    source_floor_track=contacts['l']['floorTrack']
    root[:,2]-=root_scale*(source_floor_track-source_floor_track[0])
    ankles={};poles={};foot_r={};sole={};raw_height={};contact_records={}
    for side,long in [('l','left'),('r','right')]:
        raw=np.empty((F,3));pole=np.empty((F,3));r=rotations[long+'Foot'].copy()
        foot_rest=_rest_matrix(target,long+'Foot')[:3,:3]
        forward=foot_rest[:,1].copy();forward[2]=0;forward=_unit(forward)
        for f in range(F):
            heads=core_heads(f,root[f]);hip=attached_head(f,long+'UpperLeg',heads)
            source_hip=srcp[f,index[side+'hipjoint']]
            raw[f]=hip+leg_ratios[side]*(srcp[f,index[side+'tibia']]-source_hip)
            pole[f]=hip+leg_ratios[side]*(srcp[f,index[side+'femur']]-source_hip)
            direction=srcp[f,index[side+'foot']]-srcp[f,index[side+'tibia']];direction[2]=0
            yaw=_align(forward,_unit(direction,forward))
            flat=yaw@foot_rest
            r[f]=_blend_rotation(r[f],flat,contacts[side]['flatWeight'][f])
        points=np.asarray(target['footLocalPoints'][long],dtype=float)
        offsets=np.einsum('fij,pj->fpi',r,points)
        sole[side]=offsets;foot_r[side]=r;ankles[side]=raw;poles[side]=pole
        raw_height[side]=np.min(offsets[:,:,2]+raw[:,None,2],axis=1)-ground
    support=np.concatenate([raw_height[s][contacts[s]['weight']>=.5] for s in ('l','r')])
    global_height_shift=-float(np.median(support))
    if abs(global_height_shift)>.15: raise RuntimeError('Large pelvis-ground calibration indicates incompatible source or target')
    root[:,2]+=global_height_shift
    for side in ('l','r'):
        ankles[side][:,2]+=global_height_shift;poles[side][:,2]+=global_height_shift
    support_weight=contacts['l']['weight']+contacts['r']['weight']
    valid=support_weight>.05
    support_error=sum(contacts[s]['weight']*(raw_height[s]+global_height_shift) for s in ('l','r'))
    correction=np.zeros(F)
    correction[valid]=-support_error[valid]/support_weight[valid]
    correction=np.interp(np.arange(F),np.where(valid)[0],correction[valid])
    correction=_smooth(_smooth(correction))
    if np.max(np.abs(correction))>.10: raise RuntimeError('Per-frame shoe support fitting exceeds 100 mm')
    root[:,2]+=correction
    for side in ('l','r'):
        ankles[side][:,2]+=correction;poles[side][:,2]+=correction
    max_ground_correction=0.0;anchor_local={};anchor_xy_samples={}
    for side,long in [('l','left'),('r','right')]:
        points=np.asarray(target['footLocalPoints'][long]);w=contacts[side]['weight'];records=[]
        rest_r=_rest_matrix(target,long+'Foot')[:3,:3]
        rest_forward=rest_r[:,1].copy();rest_forward[2]=0;rest_forward=_unit(rest_forward)
        projected=(points@rest_r.T)@rest_forward
        heel_local=np.mean(points[projected<=projected.min()+.020],axis=0)
        toe_local=np.mean(points[projected>=projected.max()-.020],axis=0)
        anchor_local[side]=np.zeros((F,3));anchor_xy_samples[side]=np.zeros((F,2))
        for a,b in contacts[side]['runs']:
            middle=[f for f in range(a,b+1) if contacts[side]['flatWeight'][f]>.8]
            if not middle: middle=list(range(a,b+1))
            anchor_frame=min(middle,key=lambda f:contacts[side]['speed'][f])
            heel_anchor=(ankles[side][anchor_frame]+foot_r[side][anchor_frame]@heel_local)[:2]
            toe_anchor=(ankles[side][anchor_frame]+foot_r[side][anchor_frame]@toe_local)[:2]
            for f in range(a,b+1):
                # Preserve captured heel/toe roll: the support location travels
                # across a stationary shoe footprint rather than locking a flat
                # ankle through terminal stance and forcing the pelvis down.
                phase=(f-a)/max(1,b-a);pivot=float(_smoothstep(.35,.90,phase))
                support_local=heel_local*(1-pivot)+toe_local*pivot
                anchor_xy=heel_anchor*(1-pivot)+toe_anchor*pivot
                anchor_local[side][f]=support_local;anchor_xy_samples[side][f]=anchor_xy
                fixed_ankle_xy=anchor_xy-(foot_r[side][f]@support_local)[:2]
                ankles[side][f,:2]=(1-w[f])*ankles[side][f,:2]+w[f]*fixed_ankle_xy
            records.append({'sampleRange':[a,b],'timeRange':[float(times[a]),float(times[b])],
                            'anchorSample':anchor_frame,'heelAnchorXY':heel_anchor.tolist(),'toeAnchorXY':toe_anchor.tolist(),
                            'heelSupportLocal':heel_local.tolist(),'toeSupportLocal':toe_local.tolist(),
                            'rollPolicy':'Source rotation near heel strike/toe off; flat blend only in middle stance'})
        contact_records[long]=records
        for f in range(F):
            minimum=ground-float(np.min(sole[side][f,:,2]));old=ankles[side][f,2]
            desired=(1-w[f])*old+w[f]*minimum
            ankles[side][f,2]=max(minimum,desired)
            max_ground_correction=max(max_ground_correction,abs(ankles[side][f,2]-old))
    pelvis_reach=np.zeros(F)
    # Lower the pelvis only when a planted ankle would otherwise overextend a
    # target leg. Report every correction; do not enable rig bone stretching.
    for f in range(F):
        heads=core_heads(f,root[f]);lower=0.0
        for side,long in [('l','left'),('r','right')]:
            hip=attached_head(f,long+'UpperLeg',heads);goal=ankles[side][f]
            reach=(_length(target,long+'UpperLeg')+_length(target,long+'LowerLeg'))*(1-1e-4)
            horizontal=float(np.linalg.norm(goal[:2]-hip[:2]))
            if horizontal>=reach: raise RuntimeError('Stride cannot fit target legs without changing horizontal motion')
            allowed=goal[2]+math.sqrt(max(0,reach*reach-horizontal*horizontal))
            lower=max(lower,hip[2]-allowed)
        pelvis_reach[f]=max(0,lower);root[f,2]-=max(0,lower)
    if np.max(pelvis_reach)>.10: raise RuntimeError('Target walk would require over 100 mm of extra pelvis reach correction')
    targets={name:np.tile(np.eye(4),(F,1,1)) for name in required};max_clamp=0.0;max_length_error=0.0
    floor_height={side:[] for side in ('left','right')};anchor_error={'left':0.0,'right':0.0}
    for f in range(F):
        heads=core_heads(f,root[f])
        for name in CORE:
            targets[name][f,:3,:3]=rotations[name][f];targets[name][f,:3,3]=heads[name]
        for side,long in [('l','left'),('r','right')]:
            hip=attached_head(f,long+'UpperLeg',heads)
            # Preserve the source knee plane relative to the new hip height.
            source_hip=srcp[f,index[side+'hipjoint']]
            knee_hint=hip+leg_ratios[side]*(srcp[f,index[side+'femur']]-source_hip)
            fallback=srcr[f,index['root']]@np.array([0.,-1.,0.])
            knee,ankle,clamp=_two_bone(hip,ankles[side][f],knee_hint,
                _length(target,long+'UpperLeg'),_length(target,long+'LowerLeg'),fallback)
            max_clamp=max(max_clamp,clamp)
            for name,a,b in [(long+'UpperLeg',hip,knee),(long+'LowerLeg',knee,ankle)]:
                pre=rotations[name][f];r=_align(pre[:,1],b-a)@pre
                targets[name][f,:3,:3]=r;targets[name][f,:3,3]=a
                max_length_error=max(max_length_error,abs(np.linalg.norm(b-a)-_length(target,name)))
            targets[long+'Foot'][f,:3,:3]=foot_r[side][f];targets[long+'Foot'][f,:3,3]=ankle
            floor_height[long].append(float(np.min(sole[side][f,:,2]+ankle[2])-ground))
            if contacts[side]['weight'][f]>.999:
                support_xy=(ankle+foot_r[side][f]@anchor_local[side][f])[:2]
                anchor_error[long]=max(anchor_error[long],float(np.linalg.norm(support_xy-anchor_xy_samples[side][f])))
            shoulder=attached_head(f,long+'UpperArm',heads)
            source_shoulder=srcp[f,index[side+'clavicle']]
            wrist_goal=shoulder+arm_ratios[side]*(srcp[f,index[side+'wrist']]-source_shoulder)
            elbow_hint=shoulder+arm_ratios[side]*(srcp[f,index[side+'humerus']]-source_shoulder)
            elbow,wrist,clamp=_two_bone(shoulder,wrist_goal,elbow_hint,
                _length(target,long+'UpperArm'),_length(target,long+'LowerArm'),-fallback)
            max_clamp=max(max_clamp,clamp)
            for name,a,b in [(long+'UpperArm',shoulder,elbow),(long+'LowerArm',elbow,wrist)]:
                pre=rotations[name][f];r=_align(pre[:,1],b-a)@pre
                targets[name][f,:3,:3]=r;targets[name][f,:3,3]=a
                max_length_error=max(max_length_error,abs(np.linalg.norm(b-a)-_length(target,name)))
            targets[long+'Hand'][f,:3,:3]=rotations[long+'Hand'][f];targets[long+'Hand'][f,:3,3]=wrist
    if max_clamp>.01: raise RuntimeError(f'End-effector reach correction exceeds 10 mm: {max_clamp}')
    ordered=sorted(targets,key=lambda n:target['boneOrder'].index(n))
    matrices=np.array([targets[n] for n in ordered]).transpose(1,0,2,3)
    report={'source':clip['source'],'sourceChecks':clip['checks'],'targetScale':{'root':root_scale,'legs':leg_ratios,'arms':arm_ratios},
            'sourceContactReferenceFloorMeters':source_floor,'contactSegments':contact_records,
            'sourceSupportPlaneHeightRangeMeters':[float(source_floor_track.min()),float(source_floor_track.max())],
            'sourceSupportPlaneDriftRemovedMeters':float(source_floor_track[-1]-source_floor_track[0]),
            'globalPelvisHeightCalibrationMeters':global_height_shift,'maxExtraPelvisLoweringMeters':float(np.max(pelvis_reach)),
            'maxSupportFittingPelvisCorrectionMeters':float(np.max(np.abs(correction))),
            'maxExtraPelvisLoweringAtTargetFrame':float(clip['target_frames'][int(np.argmax(pelvis_reach))]),
            'bothUnplantedIntervalsSeconds':[[float(times[a]),float(times[b])] for a,b in _runs(~valid)],
            'maxAnkleGroundCorrectionMeters':max_ground_correction,'maxReachClampMeters':max_clamp,
            'maxTargetLimbLengthErrorMeters':max_length_error,
            'minimumShoeCageHeightMeters':{side:min(v) for side,v in floor_height.items()},
            'maxFullWeightBlendSupportXYErrorMeters':anchor_error,
            'supportErrorScope':'Interpolated heel/toe support point, not proof that every rendered sole vertex is stationary',
            'rootTravelMeters':(root[-1]-root[0]).tolist(),'rootMotion':root_motion,
            'targetFrames':clip['target_frames'].tolist(),'durationSeconds':float(times[-1]-times[0]),
            'sourceFingerChannelsTransferred':False,'sourceToeChannelsTransferred':False,'limitations':LIMITATIONS}
    return {'names':ordered,'matrices':matrices,'target_frames':clip['target_frames'].copy(),
            'times':times.copy(),'report':report,'contact_weights':np.array([contacts[s]['weight'] for s in ('l','r')]).T}


def _guard():
    if bpy is None: raise RuntimeError('This operation needs the existing Blender candidate')
    helper=runpy.run_path(str(Path(__file__).with_name('full_body_review_r03.py')))
    rig=helper['_guard']()
    if np.max(np.abs(np.array(rig.matrix_world)-np.eye(4)))>1e-6:
        raise RuntimeError('Retargeting currently requires the confirmed identity armature object transform')
    return rig,helper


def read_target(ground_z=None):
    """Read actual rest lengths and finite shoe support points; no pose mutation."""
    rig,_=_guard();audit=runpy.run_path(str(Path(__file__).with_name('foot_contact_audit_r03.py')))
    selection=audit['select_sole_vertices'](objects=audit['DEFAULT_OBJECTS'],max_points=12)
    descriptions=audit['probe_description'](selection,mode='rigid_cage')
    bones={b.name:{'head':list(b.head_local),'tail':list(b.tail_local),'matrix':[list(r) for r in b.matrix_local],
                   'length':b.length,'parent':b.parent.name if b.parent else None} for b in rig.data.bones}
    def depth(name):
        parent=bones[name]['parent'];return 0 if parent is None else 1+depth(parent)
    points={'left':[],'right':[]}
    for name,ids in selection.items():
        obj=bpy.data.objects[name];side='left' if name.endswith('_L') else 'right'
        transform=rig.data.bones[side+'Foot'].matrix_local.inverted()@rig.matrix_world.inverted()@obj.matrix_world
        points[side].extend([list(transform@obj.data.vertices[i].co) for i in ids])
    return {'bones':bones,'boneOrder':sorted(bones,key=depth),'footLocalPoints':points,
            'soleVertexSelection':selection,'soleProbeDescription':descriptions,
            'groundZ':audit['resolve_ground_z'](ground_z),'sourceBlend':str(Path(bpy.data.filepath).relative_to(project_path()))}


def create_action(action_name='R03 CMU 07_01 walk - proportional IK preview',
                  baseline_action='R02 Idle - authored IK source',baseline_frame=1,
                  root_motion=True,ground_z=None,report_relative='work/reconstruction-r02/cmu-retarget-r03.json'):
    """Create an independent action from source data; restore original scene state.

    30 Hz sampling preserves the 2.625 s source endpoint at frame 79.75.
    All existing production IK switches are keyed to zero because the target
    proportion solve is baked to the existing FK bones. No bone/weight edits.
    """
    rig,helper=_guard()
    if bpy.data.actions.get(action_name): raise RuntimeError('Choose a new action name; nothing will be overwritten')
    source_action=bpy.data.actions.get(baseline_action)
    if source_action is None: raise RuntimeError('Existing relaxed baseline action required')
    parser=runpy.run_path(str(Path(__file__).with_name('cmu_asf_amc_r03.py')))
    clip=parser['sample_clip'](parser['load_clip'](),30,include_endpoint=True)
    target=read_target(ground_z);plan=build_plan(clip,target,root_motion=root_motion)
    scene=bpy.context.scene;master=rig.pose.bones[MASTER]
    old_action=rig.animation_data.action if rig.animation_data else None
    old_slot=rig.animation_data.action_slot if rig.animation_data and hasattr(rig.animation_data,'action_slot') else None
    old_frame=scene.frame_current;old_subframe=scene.frame_subframe
    old_bases={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
    old_switches={k:float(master['ik_'+k]) for k in IK_KEYS}
    created=None;complete=False
    try:
        helper['_assign'](rig,source_action);scene.frame_set(int(baseline_frame))
        for k in IK_KEYS: master['ik_'+k]=0.0
        rig.update_tag();bpy.context.view_layer.update()
        baseline={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
        baseline_pose={p.name:p.matrix.copy() for p in rig.pose.bones}
        direct_drivers={f.data_path for f in rig.animation_data.drivers}
        created=bpy.data.actions.new(action_name);created.use_fake_user=True;helper['_assign'](rig,created)
        previous={};max_error=0.0
        for frame,matrices in zip(plan['target_frames'],plan['matrices']):
            scene.frame_set(int(math.floor(frame)),subframe=float(frame%1))
            for name,basis in baseline.items(): rig.pose.bones[name].matrix_basis=basis
            for k in IK_KEYS: master['ik_'+k]=0.0
            desired={name:Matrix(m.tolist()) for name,m in zip(plan['names'],matrices)}
            for name in plan['names']:
                pb=rig.pose.bones[name];args={}
                if pb.parent:
                    args={'parent_matrix':desired.get(pb.parent.name,baseline_pose[pb.parent.name]),
                          'parent_matrix_local':pb.parent.bone.matrix_local}
                pb.matrix_basis=pb.bone.convert_local_to_pose(desired[name],pb.bone.matrix_local,invert=True,**args)
            rig.update_tag();bpy.context.view_layer.update()
            error=max(max(abs(rig.pose.bones[n].matrix[i][j]-desired[n][i][j]) for i in range(4) for j in range(4)) for n in desired)
            max_error=max(max_error,error)
            if error>2e-4: raise RuntimeError(f'Blender evaluated target matrix mismatch at {frame}: {error}')
            for pb in rig.pose.bones:
                if pb.rotation_mode=='QUATERNION':
                    if pb.name in previous and pb.rotation_quaternion.dot(previous[pb.name])<0: pb.rotation_quaternion.negate()
                    previous[pb.name]=pb.rotation_quaternion.copy()
                rotation='rotation_quaternion' if pb.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if pb.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
                for prop in ('location',rotation,'scale'):
                    if pb.path_from_id(prop) not in direct_drivers: pb.keyframe_insert(prop,frame=float(frame),group=pb.name)
            for k in IK_KEYS: master.keyframe_insert(data_path='["ik_'+k+'"]',frame=float(frame),group=MASTER)
        for c in helper['_fcurves'](created):
            for k in c.keyframe_points: k.interpolation='LINEAR'
        report=dict(plan['report'],action=created.name,targetSourceBlend=target['sourceBlend'],
                    targetRestAndShoeSha256=hashlib.sha256(json.dumps(target,sort_keys=True).encode()).hexdigest(),
                    retargetScriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    targetSoleProbeDescription=target['soleProbeDescription'],targetGroundZ=target['groundZ'],
                    blenderEvaluatedMatrixMaxError=max_error,allProductionIKSwitches=0,
                    frameRange=[float(plan['target_frames'][0]),float(plan['target_frames'][-1])],fps=30,
                    savedBlend=False,sourceUnmodified=True,rendered=False,accepted=False)
        created['r03_motion_origin']='CMU subject 07 trial 01, retargeted to actual character proportions; preview'
        created['r03_motion_report']=json.dumps(report,ensure_ascii=False)
        path=(project_path()/report_relative).resolve();path.relative_to((project_path()/'work').resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        complete=True;print(json.dumps({'action':created.name,'frames':report['frameRange'],'matrixError':max_error,'report':str(path),'savedBlend':False}));return report
    finally:
        helper['_assign'](rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        scene.frame_set(old_frame,subframe=old_subframe)
        for name,basis in old_bases.items(): rig.pose.bones[name].matrix_basis=basis
        for k,v in old_switches.items(): master['ik_'+k]=v
        rig.update_tag();bpy.context.view_layer.update()
        if created is not None and not complete: bpy.data.actions.remove(created)


if __name__=='__main__':
    print('Loaded only. read_target()/build_plan() inspect; create_action() explicitly creates an independent preview.')
