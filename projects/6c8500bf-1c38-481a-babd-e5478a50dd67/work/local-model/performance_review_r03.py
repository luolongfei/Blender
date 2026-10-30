"""Authored six-second silent facial/turn review, with original idle body motion."""
import bpy,math
from mathutils import Quaternion,Vector

def _smooth(points,f):
    for i in range(len(points)-1):
        a,x=points[i];b,y=points[i+1]
        if f<=b:
            t=max(0,min(1,(f-a)/(b-a)));t=t*t*(3-2*t);return x+(y-x)*t
    return points[-1][1]

def run():
    rig=bpy.data.objects['Character_Rig'];head=bpy.data.objects['R01 Head'];scene=bpy.context.scene
    if bpy.data.actions.get('R03 Performance review'):raise RuntimeError('Review exists')
    original=bpy.data.actions['R02 Idle - authored IK source']
    rig.animation_data.action=original
    if hasattr(original,'slots'):rig.animation_data.action_slot=next(s for s in original.slots if s.target_id_type=='OBJECT')
    rig.data.pose_position='POSE';scene.render.fps=30
    bone=rig.pose.bones['head'];bone.rotation_mode='QUATERNION'
    poses=[]
    for frame in range(1,182):
        scene.frame_set(frame);poses.append(bone.rotation_quaternion.copy())
    action=original.copy();action.name='R03 Performance review';action.use_fake_user=True;rig.animation_data.action=action
    if hasattr(action,'slots'):rig.animation_data.action_slot=next(s for s in action.slots if s.target_id_type=='OBJECT')
    keys=head.data.shape_keys.key_blocks
    channels=['R02_blink_L','R02_blink_R','R02_jaw_open','R02_mouth_wide','R02_pucker','R02_lip_press','R02_smile','R02_frown','R02_brow_raise','R02_brow_furrow','R03_squint_L','R03_squint_R']
    for frame in range(1,182):
        scene.frame_set(frame)
        yaw=math.radians(_smooth([(1,0),(30,0),(62,12),(102,-12),(140,0),(181,0)],frame))
        pitch=math.radians(_smooth([(1,0),(62,2),(102,-2),(140,0),(181,0)],frame))
        world=Quaternion(Vector((0,0,1)),yaw)@Quaternion(Vector((1,0,0)),pitch)
        rest=rig.data.bones['head'].matrix_local.to_quaternion()
        bone.rotation_quaternion=poses[frame-1]@rest.inverted()@world@rest
        bone.keyframe_insert(data_path='rotation_quaternion',frame=frame,group='R03 Head turn')
        rig['R03_look_yaw']=_smooth([(1,0),(18,0),(29,8),(57,2),(75,-10),(100,-2),(130,0),(181,0)],frame)
        rig['R03_look_pitch']=_smooth([(1,0),(62,3),(102,-3),(140,0),(181,0)],frame)
        for prop in ['R03_look_yaw','R03_look_pitch']:rig.keyframe_insert(data_path=f'["{prop}"]',frame=frame,group='R03 Gaze')
        values={n:0. for n in channels};values['R02_smile']=.45
        for center,width in [(20,4),(121,4),(171,4)]:
            blink=max(0,1-abs(frame-center)/width)
            for side in ['L','R']:values['R02_blink_'+side]=max(values['R02_blink_'+side],blink)
        pulses=[(42,12,{'R02_jaw_open':.65,'R02_mouth_wide':.15}),
                (63,10,{'R02_jaw_open':.2,'R02_mouth_wide':.75}),
                (82,9,{'R02_jaw_open':.45,'R02_pucker':.6}),
                (98,8,{'R02_jaw_open':.2,'R02_pucker':.85}),
                (110,7,{'R02_lip_press':.85}),
                (128,13,{'R02_smile':.55,'R02_brow_raise':.2}),
                (146,11,{'R02_brow_furrow':.6,'R02_lip_press':.4}),
                (162,9,{'R02_brow_raise':.65,'R02_jaw_open':.35})]
        for center,width,data in pulses:
            strength=max(0,1-abs(frame-center)/width);strength=strength*strength*(3-2*strength)
            for name,value in data.items():values[name]+=value*strength
        for name in channels:
            keys[name].value=min(1,values[name]);keys[name].keyframe_insert(data_path='value',frame=frame,group='R03 Facial performance')
    head.data.shape_keys.animation_data.action.name='R03 Silent vowels and expression combinations'
    head.data.shape_keys.animation_data.action.use_fake_user=True
    scene.frame_start=1;scene.frame_end=181;scene.frame_set(1)
    rig['R03_review_note']='Authored six-second review, silent visemes and eye/head coordination over authored idle. Not motion capture or speech audio.'
    print('R03_PERFORMANCE_REVIEW_CREATED')

if __name__=='__main__':run()
