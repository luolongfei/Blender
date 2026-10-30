"""Eyebrow follow, gaze controls, squint, and reusable viseme presets."""
import bpy, math, json
from mathutils import Vector

VISEMES={
 'rest':{},'AA':{'R02_jaw_open':.65,'R02_mouth_wide':.15},
 'EE':{'R02_jaw_open':.2,'R02_mouth_wide':.75},
 'IH':{'R02_jaw_open':.12,'R02_mouth_wide':.35},
 'OH':{'R02_jaw_open':.45,'R02_pucker':.60},
 'OU':{'R02_jaw_open':.2,'R02_pucker':.85},
 'MBP':{'R02_lip_press':.85}}

def run():
    h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];ks=h.data.shape_keys.key_blocks
    for o in bpy.data.objects:
        if not o.name.startswith('R02 Soft brow'):continue
        if o.data.shape_keys:continue
        o.shape_key_add(name='Basis',from_mix=False)
        for keyname in ['R02_brow_raise','R02_brow_furrow']:
            k=o.shape_key_add(name=keyname,from_mix=False)
            for v in k.data:
                q=v.co;x,y,z=q
                if keyname=='R02_brow_raise':
                    w=math.exp(-((z-1.471)/.018)**2-((abs(x)-.032)/.024)**4)*max(0,min(1,(-y-.04)/.03))
                    v.co.z+=.005*w
                else:
                    w=math.exp(-((z-1.467)/.015)**2-((abs(x)-.022)/.021)**4)*max(0,min(1,(-y-.04)/.03))
                    v.co+=Vector((-math.copysign(.002,x),-.001,-.003))*w
            d=k.driver_add('value').driver;d.expression='value';v=d.variables.new();v.name='value';v.targets[0].id_type='KEY';v.targets[0].id=h.data.shape_keys;v.targets[0].data_path=f'key_blocks["{keyname}"].value'
    # The imported VRMA normalizes bone axes; derive the gaze axes from actual rest bones.
    rig['R03_look_yaw']=0.;rig['R03_look_pitch']=0.
    for name in ['R03_look_yaw','R03_look_pitch']:rig.id_properties_ui(name).update(min=-18,max=18,soft_min=-12,soft_max=12)
    for side,bname in [('L','leftEye'),('R','rightEye')]:
        pb=rig.pose.bones[bname];pb.rotation_mode='XYZ'
        basis=rig.data.bones[bname].matrix_local.to_3x3().inverted()
        localpitch=basis@Vector((1,0,0));localyaw=basis@Vector((0,0,1))
        for i in range(3):
            d=pb.driver_add('rotation_euler',i).driver
            d.expression=f'({localpitch[i]}*pitch+{localyaw[i]}*yaw)*0.01745329252'
            for n,prop in [('pitch','R03_look_pitch'),('yaw','R03_look_yaw')]:
                v=d.variables.new();v.name=n;v.targets[0].id=rig;v.targets[0].data_path=f'["{prop}"]'
        if not ks.get('R03_squint_'+side):
            k=h.shape_key_add(name='R03_squint_'+side,from_mix=False)
            blink=ks['R02_blink_'+side]
            for i,v in enumerate(k.data):v.co+=(blink.data[i].co-blink.relative_key.data[i].co)*.36
            k.value=0.
    rig['R03_viseme_presets']=json.dumps(VISEMES)
    rig['R03_facial_controls']='R01 Head shape keys: blink_L/R, squint_L/R, brow_raise/furrow, smile/frown, jaw_open, mouth_wide, pucker, lip_press. Rig custom properties: look_yaw/pitch in degrees.'
    print('R03_FACIAL_CONTROLS_READY')

def viseme(name):
    ks=bpy.data.objects['R01 Head'].data.shape_keys.key_blocks
    for n in ['R02_jaw_open','R02_mouth_wide','R02_pucker','R02_lip_press']:ks[n].value=VISEMES[name].get(n,0.)

if __name__=='__main__':run()
