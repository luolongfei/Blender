"""Optional minimal IK/FK authoring controls for the existing humanoid rig.

Run run() only in a saved project working copy. The original deform bones,
weights and actions are kept. IK switches start at zero (FK). No Blend is saved.
This is a small authoring rig, not a full production/retargeting rig package.
"""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Matrix, Vector

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

VERSION=1
RIG='Character_Rig'
MASTER='CTRL_R02_Master'
COLLECTION='R02 Production Controls'
PREFIX='R02 Production '
LIMBS={
    'leg_L':('leftUpperLeg','leftLowerLeg','leftFoot','CTRL_R02_Foot_L','CTRL_R02_Knee_L'),
    'leg_R':('rightUpperLeg','rightLowerLeg','rightFoot','CTRL_R02_Foot_R','CTRL_R02_Knee_R'),
    'arm_L':('leftUpperArm','leftLowerArm','leftHand','CTRL_R02_Hand_L','CTRL_R02_Elbow_L'),
    'arm_R':('rightUpperArm','rightLowerArm','rightHand','CTRL_R02_Hand_R','CTRL_R02_Elbow_R'),
}

def _matrix_error(a,b):
    return max(abs(a[i][j]-b[i][j]) for i in range(4) for j in range(4))

def _enum(owner,prop,value):
    """Validate runtime enum identifiers instead of assuming a Blender build."""
    valid={item.identifier for item in owner.bl_rna.properties[prop].enum_items}
    if value not in valid: raise RuntimeError(f'{prop}: {value} unavailable; choices {sorted(valid)}')
    setattr(owner,prop,value)

def _guard():
    p=project_path(); file=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or file==(p/'work/character.blend').resolve():
        raise RuntimeError('Production controls may only be built in a saved independent working copy')
    try: file.relative_to((p/'work').resolve())
    except ValueError: raise RuntimeError('Working copy must be inside this project work directory')
    rig=bpy.data.objects.get(RIG)
    if rig is None or rig.type!='ARMATURE': raise RuntimeError('Character_Rig armature required')
    if rig.library or rig.data.library: raise RuntimeError('Linked rigs must not be modified here')
    if rig.data.users!=1: raise RuntimeError('Use a working-copy rig with single-user armature data')
    if _matrix_error(rig.matrix_world,Matrix.Identity(4))>1e-7:
        raise RuntimeError('This project control setup expects the confirmed identity rig transform')
    return rig

def _forward(bones):
    """Infer body up and forward from the actual spine and foot rest positions."""
    up=bones['head'].tail_local-bones['hips'].head_local
    if up.length<1e-6: raise RuntimeError('Degenerate torso direction')
    up.normalize()
    forward=sum((bones[side+'Foot'].tail_local-bones[side+'Foot'].head_local
                 for side in ('left','right')),Vector())
    forward-=up*forward.dot(up)
    if forward.length<1e-6: raise RuntimeError('Cannot infer a forward direction from the feet')
    return forward.normalized(),up

def _pole(a,b,c,preferred,alternative):
    """Stable two-segment pole, including an almost straight rest chain."""
    line=c-a; total=(b-a).length+(c-b).length
    if line.length<1e-6 or min((b-a).length,(c-b).length)<1e-6:
        raise RuntimeError('Degenerate limb chain')
    unit=line.normalized()
    offset=b-(a+unit*(b-a).dot(unit))
    fallback=offset.length<total*.02
    if fallback:
        offset=preferred-unit*preferred.dot(unit)
        if offset.length<1e-6: offset=alternative-unit*alternative.dot(unit)
        if offset.length<1e-6: raise RuntimeError('Cannot derive a stable pole direction')
    direction=offset.normalized()
    return b+direction*total*.65, fallback, direction

def _pole_angle_from_matrix(matrix,head,tail,endpoint,pole):
    """Angle preserving this upper-bone basis with the chosen pole position.

    Blender's standard IK solver forms its pole-up vector from the current
    chain-root X/Z axes, not the rest axes. A snap must therefore pass the FK
    pose matrix; a rest-only angle fails after arm lowering or axial twist.
    Solver convention reference (independent analytical implementation):
    https://github.com/blender/blender/blob/main/intern/iksolver/intern/IK_QJacobianSolver.cpp
    """
    direction=endpoint-head
    if direction.length<1e-7: raise RuntimeError('Pole angle undefined for a fully folded chain')
    direction.normalize()
    pole_up=pole-head;pole_up-=direction*pole_up.dot(direction)
    if pole_up.length<1e-7: raise RuntimeError('Pole target lies on the limb axis')
    pole_up.normalize()
    axes=matrix.to_3x3()
    x=axes.col[0].normalized();y=axes.col[1].normalized();z=axes.col[2].normalized()
    candidate=direction.cross(pole_up).cross(y)
    if candidate.length<1e-7:
        raise RuntimeError('Pole roll is singular in this pose; slightly bend/reposition the limb')
    if candidate.dot(pole_up)<0: candidate=-candidate
    candidate.normalize()
    return math.atan2(candidate.dot(z),candidate.dot(x))

def _pole_angle(upper,lower,pole):
    return _pole_angle_from_matrix(upper.matrix_local,upper.head_local,upper.tail_local,
                                   lower.tail_local,pole)

def _widget(name,kind):
    if bpy.data.objects.get(name): raise RuntimeError('Existing controller widget name: '+name)
    if kind=='ring':
        verts=[(math.cos(i*math.tau/32),math.sin(i*math.tau/32),0) for i in range(32)]
        edges=[(i,(i+1)%32) for i in range(32)]
    elif kind=='box':
        verts=[(x,y,z) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)]
        edges=[(i,j) for i,v in enumerate(verts) for j,w in enumerate(verts)
               if j>i and sum(a!=b for a,b in zip(v,w))==1]
    else:
        verts=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
        edges=[(i,j) for i,v in enumerate(verts) for j,w in enumerate(verts)
               if j>i and sum(a*b for a,b in zip(v,w))==0]
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,edges,[]);mesh.update()
    obj=bpy.data.objects.new(name,mesh);obj.hide_render=True
    # The object is only a pose custom-shape datablock, not scene geometry.
    return obj

def _shape(pb,obj,size):
    pb.custom_shape=obj
    if hasattr(pb,'use_custom_shape_bone_size'): pb.use_custom_shape_bone_size=False
    pb.custom_shape_scale_xyz=(size,size,size)

def _driver(constraint,rig,property_name):
    constraint.influence=0.0
    curve=constraint.driver_add('influence');driver=curve.driver
    _enum(driver,'type','SCRIPTED')
    variable=driver.variables.new();variable.name='switch'
    _enum(variable,'type','SINGLE_PROP')
    variable.targets[0].id=rig
    variable.targets[0].data_path=f'pose.bones["{MASTER}"]["{property_name}"]'
    driver.expression='min(1.0, max(0.0, switch))'

def run():
    rig=_guard()
    expected=[MASTER]+[n for row in LIMBS.values() for n in row[3:]]
    if rig.get('r02_controls_version')==VERSION:
        if any(n not in rig.pose.bones for n in expected):
            raise RuntimeError('Control marker exists but bones are incomplete; inspect instead of rebuilding')
        return {'alreadyBuilt':True,'version':VERSION,'savedBlend':False}
    if bpy.context.mode not in {'OBJECT','POSE'}:
        raise RuntimeError('Finish the active edit operation before adding controls')
    if any(n in rig.data.bones for n in expected): raise RuntimeError('Conflicting control bones already exist')
    if any(bpy.data.objects.get('R02_WGT_'+kind) for kind in ('ring','box','diamond')):
        raise RuntimeError('Conflicting control-shape objects already exist')
    required={'hips','head'}|{n for row in LIMBS.values() for n in row[:3]}
    missing=required-set(rig.data.bones.keys())
    if missing: raise RuntimeError('Missing required bones: '+str(sorted(missing)))
    for key,(upper,lower,end,*_) in LIMBS.items():
        if rig.data.bones[lower].parent!=rig.data.bones[upper]:
            raise RuntimeError('Expected two-bone limb hierarchy for '+key)
        if rig.data.bones[end].parent!=rig.data.bones[lower]:
            raise RuntimeError('Expected wrist/ankle child for '+key)
        if rig.pose.bones[lower].constraints or rig.pose.bones[end].constraints:
            raise RuntimeError('Existing endpoint constraints need explicit integration: '+key)
    valid_constraints={item.identifier for item in bpy.types.Constraint.bl_rna.properties['type'].enum_items}
    if not {'IK','COPY_ROTATION'}.issubset(valid_constraints):
        raise RuntimeError('Required constraint types unavailable')
    bpy.context.view_layer.update()
    original_names=list(rig.data.bones.keys())
    original_rest={n:rig.data.bones[n].matrix_local.copy() for n in original_names}
    original_pose={n:rig.pose.bones[n].matrix.copy() for n in original_names}
    original_basis={n:rig.pose.bones[n].matrix_basis.copy() for n in original_names}
    original_parents={n:(rig.data.bones[n].parent.name if rig.data.bones[n].parent else None,
                         rig.data.bones[n].use_connect) for n in original_names}
    original_active=bpy.context.view_layer.objects.active
    original_selection=list(bpy.context.selected_objects)
    original_mode=original_active.mode if original_active else 'OBJECT'
    original_front=rig.show_in_front
    forward,up=_forward(rig.data.bones)
    height=max(b.tail_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
    plans={}
    for key,(upper,lower,end,target,pole) in LIMBS.items():
        ub,lb,eb=(rig.data.bones[n] for n in (upper,lower,end))
        preferred=forward if key.startswith('leg_') else -forward
        location,fallback,direction=_pole(ub.head_local,lb.head_local,lb.tail_local,preferred,up)
        plans[key]={'pole':location,'fallback':fallback,'direction':direction,
                    'angle':_pole_angle(ub,lb,location),'target_head':eb.head_local.copy(),
                    'target_tail':eb.tail_local.copy(),'end':end,'target':target,'pole_name':pole}
    created_constraints=[];created_widgets=[];new_collection=None
    try:
        if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
        bpy.context.view_layer.objects.active=rig
        bpy.ops.object.mode_set(mode='EDIT')
        eb=rig.data.edit_bones
        master=eb.new(MASTER);master.head=(0,0,0);master.tail=(0,max(.10,height*.08),0)
        master.roll=0;master.use_deform=False
        # Identity rest orientation means an untouched master preserves the old
        # root channels and all bone rest matrices while providing global motion.
        for name,(parent,connected) in original_parents.items():
            if parent is None: eb[name].parent=master;eb[name].use_connect=False
        for key,plan in plans.items():
            end=eb[plan['end']]
            target=eb.new(plan['target']);target.head=plan['target_head'];target.tail=plan['target_tail']
            target.roll=end.roll;target.parent=master;target.use_deform=False
            pole=eb.new(plan['pole_name']);pole.head=plan['pole']
            pole.tail=plan['pole']+up*max(.025,height*.025)
            pole.roll=0;pole.parent=master;pole.use_deform=False
        bpy.ops.object.mode_set(mode='OBJECT')
        new_collection=rig.data.collections.new(COLLECTION)
        for name in expected: new_collection.assign(rig.data.bones[name])
        shapes={}
        for kind in ('ring','box','diamond'):
            shapes[kind]=_widget('R02_WGT_'+kind,kind)
            created_widgets.append(shapes[kind])
        mp=rig.pose.bones[MASTER];_enum(mp,'rotation_mode','QUATERNION')
        _shape(mp,shapes['ring'],height*.27)
        for key,(upper,lower,end,target,pole) in LIMBS.items():
            prop='ik_'+key;mp[prop]=0.0
            mp.id_properties_ui(prop).update(min=0.0,max=1.0,soft_min=0.0,soft_max=1.0,
                description='0=FK，1=IK；先用 snap_ik_to_fk / snap_fk_to_ik 匹配，避免跳变')
            tp,pp=rig.pose.bones[target],rig.pose.bones[pole]
            for pb in (tp,pp):
                _enum(pb,'rotation_mode','QUATERNION');pb.lock_scale=(True,True,True)
            pp.lock_rotation=(True,True,True);pp.lock_rotation_w=True
            _shape(tp,shapes['box'],height*.035);_shape(pp,shapes['diamond'],height*.021)
            ik=rig.pose.bones[lower].constraints.new('IK');created_constraints.append((lower,ik))
            ik.name=PREFIX+key+' IK';ik.target=rig;ik.subtarget=target
            ik.pole_target=rig;ik.pole_subtarget=pole;ik.pole_angle=plans[key]['angle']
            ik.chain_count=2;ik.use_tail=True;ik.use_stretch=False;ik.use_rotation=False
            _driver(ik,rig,prop)
            rotation=rig.pose.bones[end].constraints.new('COPY_ROTATION')
            created_constraints.append((end,rotation));rotation.name=PREFIX+key+' orientation'
            rotation.target=rig;rotation.subtarget=target
            _enum(rotation,'owner_space','POSE');_enum(rotation,'target_space','POSE')
            _enum(rotation,'mix_mode','REPLACE');_driver(rotation,rig,prop)
        rig.show_in_front=True
        bpy.context.view_layer.update()
        rest_error=max(_matrix_error(rig.data.bones[n].matrix_local,original_rest[n]) for n in original_names)
        pose_error=max(_matrix_error(rig.pose.bones[n].matrix,original_pose[n]) for n in original_names)
        if max(rest_error,pose_error)>2e-6:
            raise RuntimeError(f'Default FK state changed: rest={rest_error}, pose={pose_error}')
        if any(abs(c.influence)>1e-8 for _,c in created_constraints):
            raise RuntimeError('An IK driver did not initialize at zero')
        report={'version':VERSION,'addedControlBones':expected,'addedConstraints':len(created_constraints),
                'defaultIKInfluence':0,'originalBoneNamesPreserved':True,'weightsModified':False,
                'originalRestMatrixMaxError':rest_error,'defaultPoseMatrixMaxError':pose_error,
                'reparentedOriginalRoots':[n for n,(parent,_) in original_parents.items() if parent is None],
                'switches':{key:'pose.bones["'+MASTER+'"]["ik_'+key+'"]' for key in LIMBS},
                'poleSetup':{key:{'nearStraightFallback':p['fallback'],'position':list(p['pole']),
                                  'fallbackPreference':'forward' if key.startswith('leg_') else 'backward',
                                  'poleAngleRadians':p['angle']} for key,p in plans.items()},
                'savedBlend':False,'scope':'Minimal optional authoring controls; FK is the existing deform bones',
                'unverified':['IK target/pole motion and singularity behavior in live poses',
                              'Visual elbow/knee roll and hand/foot orientation at full IK influence',
                              'IK/FK matching across animation and limb extremes',
                              'Shoulder volume, foot roll, toe pivot and twist-bone corrections',
                              'Export must bake/evaluate required deformation; controls are not a VRM deliverable']}
        rig['r02_controls_version']=VERSION
        rig['r02_controls_forward']=list(forward)
        rig['r02_controls_report']=json.dumps(report,ensure_ascii=False)
        print(json.dumps(report,ensure_ascii=False))
        return report
    except Exception:
        if rig.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        for owner,constraint in reversed(created_constraints):
            try: constraint.driver_remove('influence')
            except (RuntimeError,TypeError): pass
            if rig.pose.bones.get(owner): rig.pose.bones[owner].constraints.remove(constraint)
        bpy.context.view_layer.objects.active=rig;rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT');eb=rig.data.edit_bones
        for name,(parent,connected) in original_parents.items():
            eb[name].parent=eb.get(parent) if parent else None;eb[name].use_connect=connected
        for name in reversed(expected):
            if eb.get(name): eb.remove(eb[name])
        bpy.ops.object.mode_set(mode='OBJECT')
        for name,basis in original_basis.items(): rig.pose.bones[name].matrix_basis=basis
        if new_collection: rig.data.collections.remove(new_collection)
        for obj in created_widgets:
            mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        rig.show_in_front=original_front
        for key in ('r02_controls_version','r02_controls_forward','r02_controls_report'):
            if key in rig: del rig[key]
        bpy.context.view_layer.update()
        raise
    finally:
        if bpy.context.object and bpy.context.object.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in original_selection:
            if obj.name in bpy.context.view_layer.objects: obj.select_set(True)
        bpy.context.view_layer.objects.active=original_active
        if original_active and original_mode=='POSE': bpy.ops.object.mode_set(mode='POSE')

def snap_ik_to_fk(limb_key,*,enable=True,keyframe=False,max_matrix_error=2e-3):
    """Match controls to the currently evaluated FK pose before enabling IK."""
    rig=_guard()
    if rig.get('r02_controls_version')!=VERSION: raise RuntimeError('Build controls first')
    upper,lower,end,target,pole=LIMBS[limb_key]
    bpy.context.view_layer.update()
    ub,lb,ep=(rig.pose.bones[n] for n in (upper,lower,end))
    wanted={n:rig.pose.bones[n].matrix.copy() for n in (upper,lower,end)}
    desired=ep.matrix.copy()
    master=rig.pose.bones[MASTER]
    old_switch=master['ik_'+limb_key]
    old_basis={n:rig.pose.bones[n].matrix_basis.copy() for n in (target,pole)}
    ik=rig.pose.bones[lower].constraints.get(PREFIX+limb_key+' IK')
    if ik is None or ik.type!='IK': raise RuntimeError('Expected R02 IK constraint missing')
    old_angle=ik.pole_angle
    forward=master.matrix.to_3x3()@Vector(rig['r02_controls_forward'])
    if limb_key.startswith('arm_'): forward=-forward
    alternative=master.matrix.to_3x3()@Vector((0,0,1))
    location,_,_=_pole(ub.head.copy(),lb.head.copy(),lb.tail.copy(),forward,alternative)
    angle=_pole_angle_from_matrix(ub.matrix.copy(),ub.head.copy(),ub.tail.copy(),lb.tail.copy(),location)
    rig.pose.bones[target].matrix=desired
    matrix=rig.pose.bones[pole].matrix.copy();matrix.translation=location
    rig.pose.bones[pole].matrix=matrix
    ik.pole_angle=angle
    if enable: master['ik_'+limb_key]=1.0
    rig.update_tag();bpy.context.view_layer.update()
    error=max(_matrix_error(rig.pose.bones[n].matrix,wanted[n]) for n in wanted) if enable else None
    if error is not None and error>max_matrix_error:
        master['ik_'+limb_key]=old_switch
        ik.pole_angle=old_angle
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        rig.update_tag();bpy.context.view_layer.update()
        raise RuntimeError(f'IK matching exceeded tolerance ({error}); restored controls and switch. Check pole roll and limb pose.')
    if keyframe:
        for name in (target,pole):
            for prop in ('location','rotation_quaternion'):
                rig.pose.bones[name].keyframe_insert(prop,group=name)
        master.keyframe_insert(data_path='["ik_'+limb_key+'"]',group=MASTER)
        ik.keyframe_insert(data_path='pole_angle',group=lower)
    return {'limb':limb_key,'IK':master['ik_'+limb_key],'savedBlend':False,
            'keyframed':keyframe,'maxPoseMatrixError':error,
            'poleAngleRadians':ik.pole_angle,'previousPoleAngleRadians':old_angle}

def update_pole_angles(limb_keys=('arm_L','arm_R')):
    """Refresh existing controls against the current FK pose; keep IK off.

    Call in the saved candidate after loading this revised module. No edit mode,
    new constraints, keyframes, action changes, or Blend save is performed.
    This is a current-frame calibration, not proof of a continuous action.
    Future snaps also compute their pole angle from the current FK basis.
    """
    rig=_guard()
    if rig.get('r02_controls_version')!=VERSION: raise RuntimeError('Build controls first')
    keys=tuple(limb_keys)
    if not keys or len(set(keys))!=len(keys): raise ValueError('Choose one or more unique limb keys')
    mp=rig.pose.bones[MASTER]
    for key in keys:
        if key not in LIMBS: raise ValueError('Unknown limb key: '+key)
        if abs(mp['ik_'+key])>1e-8: raise RuntimeError('Set '+key+' to FK (0) before calibration')
    old={}
    for key in keys:
        upper,lower,end,target,pole=LIMBS[key]
        con=rig.pose.bones[lower].constraints.get(PREFIX+key+' IK')
        if con is None or con.type!='IK': raise RuntimeError('Missing IK constraint: '+key)
        old[key]=(con.pole_angle,{n:rig.pose.bones[n].matrix_basis.copy() for n in (target,pole)})
    bpy.context.view_layer.update()
    before={b.name:b.matrix.copy() for b in rig.pose.bones if b.bone.use_deform}
    results={}
    try:
        for key in keys: results[key]=snap_ik_to_fk(key,enable=False,keyframe=False)
        error=max(_matrix_error(rig.pose.bones[n].matrix,m) for n,m in before.items())
        if error>2e-6: raise RuntimeError(f'FK changed while calibrating disabled IK: {error}')
    except Exception:
        for key,(angle,bases) in old.items():
            lower=LIMBS[key][1]
            rig.pose.bones[lower].constraints[PREFIX+key+' IK'].pole_angle=angle
            for name,basis in bases.items(): rig.pose.bones[name].matrix_basis=basis
        rig.update_tag();bpy.context.view_layer.update()
        raise
    report={'revision':2,'frame':bpy.context.scene.frame_current,'limbs':results,
            'defaultPoseMatrixMaxError':error,'savedBlend':False,
            'scope':'Current FK pose calibration; perform separate enabled-IK and continuous-motion validation'}
    rig['r02_controls_pole_revision']=2
    rig['r02_controls_pole_update']=json.dumps(report,ensure_ascii=False)
    print(json.dumps(report,ensure_ascii=False))
    return report

def snap_fk_to_ik(limb_key,*,disable=True,keyframe=False,max_matrix_error=2e-3):
    """Bake the evaluated IK limb pose into existing FK bones before disabling."""
    rig=_guard()
    if rig.get('r02_controls_version')!=VERSION: raise RuntimeError('Build controls first')
    upper,lower,end,*_=LIMBS[limb_key]
    bpy.context.view_layer.update()
    targets={n:rig.pose.bones[n].matrix.copy() for n in (upper,lower,end)}
    old_basis={n:rig.pose.bones[n].matrix_basis.copy() for n in targets}
    master=rig.pose.bones[MASTER]
    old_switch=master['ik_'+limb_key]
    if disable: master['ik_'+limb_key]=0.0
    rig.update_tag();bpy.context.view_layer.update()
    for name in (upper,lower,end):
        pb=rig.pose.bones[name];parents={}
        if pb.parent:
            parents={'parent_matrix':targets.get(pb.parent.name,pb.parent.matrix),
                     'parent_matrix_local':pb.parent.bone.matrix_local}
        pb.matrix_basis=pb.bone.convert_local_to_pose(targets[name],pb.bone.matrix_local,invert=True,**parents)
    bpy.context.view_layer.update()
    error=max(_matrix_error(rig.pose.bones[n].matrix,targets[n]) for n in targets) if disable else None
    if error is not None and error>max_matrix_error:
        master['ik_'+limb_key]=old_switch
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        rig.update_tag();bpy.context.view_layer.update()
        raise RuntimeError(f'FK matching exceeded tolerance ({error}); restored FK pose and switch.')
    for name in (upper,lower,end):
        pb=rig.pose.bones[name]
        if keyframe:
            pb.keyframe_insert('location',group=name)
            if pb.rotation_mode=='QUATERNION': pb.keyframe_insert('rotation_quaternion',group=name)
            elif pb.rotation_mode=='AXIS_ANGLE': pb.keyframe_insert('rotation_axis_angle',group=name)
            else: pb.keyframe_insert('rotation_euler',group=name)
            pb.keyframe_insert('scale',group=name)
    if keyframe: master.keyframe_insert(data_path='["ik_'+limb_key+'"]',group=MASTER)
    bpy.context.view_layer.update()
    return {'limb':limb_key,'IK':master['ik_'+limb_key],'savedBlend':False,
            'keyframed':keyframe,'maxPoseMatrixError':error}

if __name__=='__main__': run()
