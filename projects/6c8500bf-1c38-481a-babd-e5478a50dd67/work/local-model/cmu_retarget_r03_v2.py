"""Version 2 of proportional, contact-aware CMU 07_01 walk adaptation.

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
    'Pure NumPy shoe checks cover sampled keyframes; Blender evaluation, interpolation and continuous visual review remain required.',
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


def _axis_rotation(axis,angle):
    x,y,z=_unit(axis);k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)


def _swing_intervals(runs,count):
    """Intervals include stance boundary samples to impose continuous joins."""
    out=[]
    if not runs: return [(0,count-1,False,False)]
    if runs[0][0]>0: out.append((0,runs[0][0],False,True))
    out.extend((b,c,True,True) for (a,b),(c,d) in zip(runs,runs[1:]) if c>b)
    if runs[-1][1]<count-1: out.append((runs[-1][1],count-1,True,False))
    return out


def _foot_orientations(raw,rest,contact,times):
    """Hold stance yaw/bank and preserve bounded, smooth source pitch roll."""
    count=len(raw);result=raw.copy();forward=rest[:,1].copy();forward[2]=0;forward=_unit(forward)
    segments=[]
    for a,b in contact['runs']:
        middle=list(range(a+max(1,(b-a)//4),max(a+2,b-(b-a)//3)))
        middle=[f for f in middle if a<=f<=b] or list(range(a,b+1))
        anchor=min(middle,key=lambda f:contact['speed'][f])
        direction=raw[anchor,:,1].copy();direction[2]=0
        flat=_align(forward,_unit(direction,forward))@rest;axis=flat[:,0]
        def pitch(f):
            src=raw[f,:,1];src=src-axis*(axis@src);src=_unit(src)
            return math.atan2(axis@np.cross(flat[:,1],src),flat[:,1]@src)
        heel=float(np.clip(pitch(a),math.radians(-25),0));toe=float(np.clip(pitch(b),0,math.radians(32)))
        for f in range(a,b+1):
            phase=(times[f]-times[a])/max(times[b]-times[a],1e-8)
            # Retain the measured pitch rate during toe-off. Flatten only
            # the middle of stance; forcing zero rate at release delayed the
            # ankle while the pelvis was already moving forwards.
            raw_pitch=pitch(f)
            angle=float(np.clip(raw_pitch,math.radians(-25),0))*(1-_smoothstep(0,.25,phase))
            angle+=float(np.clip(raw_pitch,0,math.radians(32)))*_smoothstep(.55,.80,phase)
            result[f]=_axis_rotation(axis,angle)@flat
        segments.append({'start':a,'end':b,'anchor':anchor,'flat':flat,
                         'heelPitchRadians':heel,'toePitchRadians':toe})
    for a,b,left,right in _swing_intervals(contact['runs'],count):
        left_offset=result[a]@raw[a].T;right_offset=result[b]@raw[b].T
        for f in range(a,b+1):
            phase=(times[f]-times[a])/max(times[b]-times[a],1e-8);r=raw[f]
            if left: r=_blend_rotation(left_offset,np.eye(3),float(_smoothstep(0,.35,phase)))@r
            if right: r=_blend_rotation(np.eye(3),right_offset,float(_smoothstep(.65,1,phase)))@r
            result[f]=r
    return result,segments


def _foot_trajectory(raw,rotations,rest,points,segments,contact,times,ground):
    """Actual rigid sole pivots in stance; Hermite correction and clearance in swing."""
    count=len(raw);out=raw.copy();anchor_local=np.zeros((count,3));anchor_xy=np.zeros((count,2));records=[]
    rest_forward=rest[:,1].copy();rest_forward[2]=0;rest_forward=_unit(rest_forward)
    rest_points=points@rest.T;projected=rest_points@rest_forward
    def lowest_support(mask):
        ids=np.where(mask)[0]
        ids=ids[rest_points[ids,2]<=np.min(rest_points[ids,2])+.0001]
        return np.mean(points[ids],axis=0)
    # The rear outsole cage is about 5 mm above the stacked heel. Averaging
    # those together produces a pivot in the air and visible micro-sliding.
    heel=lowest_support(projected<=projected.min()+.001)
    toe=lowest_support(projected>=projected.max()-.001)
    offsets=np.einsum('fij,pj->fpi',rotations,points)
    for segment in segments:
        a,b,anchor=segment['start'],segment['end'],segment['anchor'];flat=segment['flat']
        placement=raw[anchor,:2].copy()
        for f in range(a,b+1):
            phase=(times[f]-times[a])/max(times[b]-times[a],1e-8)
            # Switch while pitch is exactly zero: both pivots imply the same
            # ankle placement, so there is no spatial jump at this change.
            pivot=heel if phase<=.50 else toe
            fixed=placement+(flat@pivot)[:2]
            out[f,:2]=fixed-(rotations[f]@pivot)[:2]
            out[f,2]=ground-float(np.min(offsets[f,:,2]))
            anchor_local[f]=pivot;anchor_xy[f]=fixed
        records.append({'sampleRange':[a,b],'timeRange':[float(times[a]),float(times[b])],
                        'anchorSample':anchor,'footprintOriginXY':placement.tolist(),
                        'heelSupportLocal':heel.tolist(),'toeSupportLocal':toe.tolist(),
                        'heelPitchDegrees':math.degrees(segment['heelPitchRadians']),
                        'toePitchDegrees':math.degrees(segment['toePitchRadians']),
                        'rollPolicy':'Fixed footprint yaw/bank; bounded source pitch rate through release; actual lowest heel/forefoot pivots'})
    raw_velocity=np.gradient(raw,times,axis=0)
    for a,b,left,right in _swing_intervals(contact['runs'],count):
        duration=times[b]-times[a]
        if duration<=0: continue
        start=out[a].copy() if left else raw[a].copy();end=out[b].copy() if right else raw[b].copy()
        def continued_support(f,reference):
            value=np.empty(3)
            value[:2]=anchor_xy[reference]-(rotations[f]@anchor_local[reference])[:2]
            value[2]=ground-float(np.min(offsets[f,:,2]))
            return value
        va=(continued_support(a+1,a)-out[a])/(times[a+1]-times[a]) if left and a+1<count else raw_velocity[a]
        vb=(out[b]-continued_support(b-1,b))/(times[b]-times[b-1]) if right and b>0 else raw_velocity[b]
        c0=start-raw[a];c1=end-raw[b]
        dv0=va-raw_velocity[a];dv1=vb-raw_velocity[b]
        for f in range(a,b+1):
            if (f==a and left) or (f==b and right): continue
            t=(times[f]-times[a])/duration
            h00=2*t**3-3*t*t+1;h10=t**3-2*t*t+t;h01=-2*t**3+3*t*t;h11=t**3-t*t
            out[f]=raw[f]+h00*c0+h10*duration*dv0+h01*c1+h11*duration*dv1
            # A fast swing must clear the actual shoe heel, not merely avoid
            # penetration. This directly addresses v1 frames 44->45 grazing.
            rise=float(_smoothstep(0,.12,t)) if left else 1.0
            fall=float(_smoothstep(0,.12,1-t)) if right else 1.0
            clearance=.012*rise*fall
            out[f,2]=max(out[f,2],ground+clearance-float(np.min(offsets[f,:,2])))
    return out,anchor_local,anchor_xy,records


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
                weight[f]=1.0
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
    ankles={};poles={};foot_r={};sole={};raw_height={};contact_records={};foot_segments={}
    for side,long in [('l','left'),('r','right')]:
        raw=np.empty((F,3));pole=np.empty((F,3))
        foot_rest=_rest_matrix(target,long+'Foot')[:3,:3]
        r,segments=_foot_orientations(rotations[long+'Foot'],foot_rest,contacts[side],times)
        foot_segments[side]=segments
        for f in range(F):
            heads=core_heads(f,root[f]);hip=attached_head(f,long+'UpperLeg',heads)
            source_hip=srcp[f,index[side+'hipjoint']]
            raw[f]=hip+leg_ratios[side]*(srcp[f,index[side+'tibia']]-source_hip)
            pole[f]=hip+leg_ratios[side]*(srcp[f,index[side+'femur']]-source_hip)
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
        points=np.asarray(target['footLocalPoints'][long])
        rest_r=_rest_matrix(target,long+'Foot')[:3,:3]
        old=ankles[side].copy()
        ankles[side],anchor_local[side],anchor_xy_samples[side],contact_records[long]=_foot_trajectory(
            old,foot_r[side],rest_r,points,foot_segments[side],contacts[side],times,ground)
        max_ground_correction=max(max_ground_correction,float(np.max(np.abs(ankles[side][:,2]-old[:,2]))))
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
            'supportErrorScope':'Fixed rigid heel/forefoot pivots; the separate finite shoe audit tracks each selected cage point',
            'rootTravelMeters':(root[-1]-root[0]).tolist(),'rootMotion':root_motion,
            'targetFrames':clip['target_frames'].tolist(),'durationSeconds':float(times[-1]-times[0]),
            'sourceFingerChannelsTransferred':False,'sourceToeChannelsTransferred':False,'limitations':LIMITATIONS}
    return {'names':ordered,'matrices':matrices,'target_frames':clip['target_frames'].copy(),
            'times':times.copy(),'report':report,'contact_weights':np.array([contacts[s]['weight'] for s in ('l','r')]).T}


def audit_plan(plan,target,contact_height=.004,vertical_speed=.03,sliding_speed=.05):
    """Track the same actual shoe-cage vertex at consecutive sampled poses.

    Mirrors the finite audit's near-ground/low-vertical-speed criterion. This
    does not substitute for the Blender audit: interpolation between the keyed
    poses, deformation outside the rigid shoe and rendered bevels are omitted.
    """
    times=np.asarray(plan['times']);frames=np.asarray(plan['target_frames']);dt=np.diff(times)
    if np.any(dt<=0): raise RuntimeError('Audit times must be strictly increasing')
    ground=float(target['groundZ']);sides={}
    for side_index,side in enumerate(('left','right')):
        matrices=plan['matrices'][:,plan['names'].index(side+'Foot')]
        points=np.asarray(target['footLocalPoints'][side],dtype=float)
        world=np.einsum('fij,pj->fpi',matrices[:,:3,:3],points)+matrices[:,:3,3][:,None,:]
        height=world[:,:,2]-ground;velocity=np.diff(world,axis=0)/dt[:,None,None]
        horizontal=np.linalg.norm(velocity[:,:,:2],axis=2)
        near=(height>=-.002)&(height<=contact_height)
        contact=near[:-1]&near[1:]&(np.abs(velocity[:,:,2])<=vertical_speed)
        masked=np.where(contact,horizontal,0);worst=np.unravel_index(np.argmax(masked),masked.shape)
        stance=plan['contact_weights'][:,side_index]>.999
        swing=~stance
        worst_intervals=[]
        for interval,point in zip(*np.where(masked>sliding_speed)):
            worst_intervals.append({'frames':[float(frames[interval]),float(frames[interval+1])],
                                    'footLocalPointIndex':int(point),'horizontalSpeedMetersPerSecond':float(masked[interval,point]),
                                    'classifiedSupport':[bool(stance[interval]),bool(stance[interval+1])]})
        worst_intervals.sort(key=lambda x:x['horizontalSpeedMetersPerSecond'],reverse=True)
        sides[side]={'pointCount':len(points),'minimumCageHeightMeters':float(height.min()),
                     'minimumSwingCageHeightMeters':float(height[swing].min()) if np.any(swing) else None,
                     'maximumNearGroundHorizontalSpeedMetersPerSecond':float(masked.max()),
                     'worstNearGroundFrames':[float(frames[worst[0]]),float(frames[worst[0]+1])],
                     'worstNearGroundFootLocalPointIndex':int(worst[1]),
                     'suspectedSlidingPointIntervals':int(np.sum(masked>sliding_speed)),
                     'worstSuspectedSlidingIntervals':worst_intervals[:12],
                     'maximumAllPhaseHorizontalSpeedMetersPerSecond':float(horizontal.max()),
                     'minimumHeightByTargetFrame':{str(float(f)):float(h.min()) for f,h in zip(frames,height)}}
    return {'scope':'NumPy planned keyframe matrices with same actual rigid shoe-cage points; not Blender evaluation or subframe audit',
            'criteria':{'contactHeightMeters':contact_height,'penetrationAllowanceMeters':.002,
                        'maximumContactVerticalSpeedMetersPerSecond':vertical_speed,'suspectedSlidingSpeedMetersPerSecond':sliding_speed},
            'targetFrames':[float(frames[0]),float(frames[-1])],'durationSeconds':float(times[-1]-times[0]),
            'sides':sides,'passedFiniteKeyframeCriteria':all(v['minimumCageHeightMeters']>=-.002 and v['suspectedSlidingPointIntervals']==0 for v in sides.values())}


def audit_exported_target(target_relative='work/reconstruction-r02/cmu-target-r04-actual.json',
                          report_relative='work/reconstruction-r02/cmu-retarget-v2-numerical-audit.json'):
    """Plain Python/NumPy comparison on an actual exported target; no Blender."""
    path=(project_path()/target_relative).resolve();path.relative_to(project_path().resolve())
    target=json.loads(path.read_text(encoding='utf-8-sig'))
    parser=runpy.run_path(str(Path(__file__).with_name('cmu_asf_amc_r03.py')))
    clip=parser['sample_clip'](parser['load_clip'](),30,include_endpoint=True)
    old=runpy.run_path(str(Path(__file__).with_name('cmu_retarget_r03.py')))
    before=old['build_plan'](clip,target);after=build_plan(clip,target)
    report={'targetFile':target_relative,'targetFileSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'targetSourceBlend':target.get('sourceBlend'),'source':clip['source'],
            'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'before':audit_plan(before,target),'after':audit_plan(after,target),
            'beforeMaxExtraPelvisLoweringMeters':before['report']['maxExtraPelvisLoweringMeters'],
            'afterRetargetReport':after['report'],'blenderExecuted':False,'rendered':False,'accepted':False}
    destination=(project_path()/report_relative).resolve();destination.relative_to((project_path()/'work').resolve())
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


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


def create_action(action_name='R03 CMU 07_01 walk - proportional IK preview v2',
                  baseline_action='R02 Idle - authored IK source',baseline_frame=1,
                  root_motion=True,ground_z=None,report_relative='work/reconstruction-r02/cmu-retarget-r03-v2.json'):
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
    finite_audit=audit_plan(plan,target)
    if not finite_audit['passedFiniteKeyframeCriteria']:
        raise RuntimeError('The current target fails finite shoe contact criteria; inspect build_plan()/audit_plan() before creating an action')
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
                    finitePlannedShoeAudit=finite_audit,retargetRevision='v2 fixed stance pivots and continuous source-derived foot roll',
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
