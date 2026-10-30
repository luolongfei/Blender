"""Bounded shoe support-point and limb-endpoint audit; no rendering or saves.

Public read-only probes:
  select_sole_vertices(objects=(...), max_points=12) -> {object: [vertex IDs]}
  sample_sole_points(selection, mode='auto') -> {object: [[world x,y,z], ...]}
  resolve_ground_z(None) -> measured horizontal Cyclorama floor height

auto uses evaluated vertex indices only for a proven topology-preserving stack.
For the authored beveled, foot-rigid shoes it may instead evaluate selected
original cage points analytically, after checking all selected skin weights.
Cage probes are not the post-bevel surface and cannot certify visual contact.
"""
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

RIG='Character_Rig'
MASTER='CTRL_R02_Master'
DEFAULT_OBJECTS=('Shoe_Outsole_L','Shoe_Stacked_Heel_L','Shoe_Outsole_R','Shoe_Stacked_Heel_R')
IK_KEYS=('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R')
ENDPOINTS=('leftHand','rightHand','leftFoot','rightFoot')
DEFORM_ONLY={'ARMATURE','LATTICE','CURVE','SIMPLE_DEFORM','CAST','SMOOTH',
             'CORRECTIVE_SMOOTH','LAPLACIANSMOOTH','LAPLACIANDEFORM','DISPLACE',
             'SHRINKWRAP','WARP','WAVE','SURFACE_DEFORM','MESH_DEFORM','HOOK',
             'WEIGHTED_NORMAL','NORMAL_EDIT','DATA_TRANSFER'}
LIMITS=['Finite, explicitly indexed shoe support probes; no full surface or scene collision test.',
        'Rigid cage sampling excludes bevel-generated surface vertices and may differ by the bevel width.',
        'A geometrically low foot is only a contact candidate, not measured support force.',
        'Contact-point speed can flag sliding but cannot distinguish every intentional pivot or toe roll.',
        'Hand/foot endpoint speed uses the named bone tail, not fingertips or the complete shoe surface.',
        'Temporal samples can miss between-frame events; results do not establish naturalness or all-pose acceptance.',
        'Changing frames still evaluates active scene systems; this script does not disable unrelated simulations.',
        'Only the armature action is switched; other animated IDs keep their existing actions.']


def _object(name):
    obj=bpy.data.objects.get(name)
    if obj is None or obj.type!='MESH': raise RuntimeError('Mesh missing: '+name)
    return obj


def _validate_selection(selection):
    if not isinstance(selection,dict) or not selection: raise ValueError('Supply a nonempty object-to-vertex-ID mapping')
    for name,ids in selection.items():
        obj=_object(name)
        if not (1<=len(ids)<=12) or len(set(ids))!=len(ids):
            raise ValueError('Use 1..12 unique vertex IDs per object: '+name)
        if any(not isinstance(i,int) or isinstance(i,bool) or i<0 or i>=len(obj.data.vertices) for i in ids):
            raise ValueError('Invalid source vertex ID: '+name)


def select_sole_vertices(objects=('Shoe_Outsole_L','Shoe_Outsole_R'),max_points=12):
    """Pick deterministic bottom-ring extrema from each small authored shoe cage.

    Uses original mesh coordinates transformed into armature rest space. The
    source shoes were authored with +Z up and -Y forward. Bone-parented meshes
    require explicit IDs because their object transform already includes pose.
    The selection scans only the named small source meshes, once.
    """
    if not isinstance(max_points,int) or not (4<=max_points<=12): raise ValueError('max_points must be 4..12')
    rig=bpy.data.objects.get(RIG)
    if rig is None: raise RuntimeError('Character_Rig required')
    selection={}
    for name in objects:
        obj=_object(name)
        if obj.parent_type=='BONE': raise RuntimeError('Use explicit source vertex IDs for bone-parented shoe objects')
        if not (4<=len(obj.data.vertices)<=10000): raise RuntimeError('Automatic selection is limited to small shoe cages')
        transform=rig.matrix_world.inverted()@obj.matrix_world
        coords=[transform@v.co for v in obj.data.vertices]
        low=min(p.z for p in coords);high=max(p.z for p in coords)
        tolerance=max(0.0005,(high-low)*0.04)
        candidates=[i for i,p in enumerate(coords) if p.z<=low+tolerance]
        if len(candidates)<3: raise RuntimeError('Cannot find a stable lower support ring: '+name)
        center=sum((coords[i] for i in candidates),Vector())/len(candidates)
        picked=[]
        for j in range(max_points):
            angle=2*math.pi*j/max_points
            direction=Vector((math.cos(angle),math.sin(angle),0))
            idx=max(candidates,key=lambda i:((coords[i]-center).dot(direction),-i))
            if idx not in picked: picked.append(idx)
        selection[name]=picked
    _validate_selection(selection)
    return selection


def _active_modifiers(obj):
    # Evaluation is viewport depsgraph geometry. Report/reject a render mismatch
    # rather than silently presenting it as the rendered surface.
    if any(bool(m.show_viewport)!=bool(m.show_render) for m in obj.modifiers):
        raise RuntimeError('Viewport/render modifier visibility differs: '+obj.name)
    return [m for m in obj.modifiers if m.show_viewport]


def _rigid_cage_bone(obj,ids,mods):
    if any(m.type not in {'BEVEL','ARMATURE','WEIGHTED_NORMAL','NORMAL_EDIT'} for m in mods):
        raise RuntimeError('Cannot prove a rigid original-cage mapping: '+obj.name)
    armatures=[m for m in mods if m.type=='ARMATURE']
    if len(armatures)!=1: raise RuntimeError('Rigid cage sampling needs exactly one Armature modifier')
    mod=armatures[0];rig=mod.object
    if rig is None or rig.name!=RIG or not mod.use_vertex_groups or mod.use_bone_envelopes:
        raise RuntimeError('Unsupported armature weighting configuration: '+obj.name)
    if mod.vertex_group or mod.use_multi_modifier:
        raise RuntimeError('Unsupported masked or multi-modifier cage evaluation')
    if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks)>1:
        raise RuntimeError('Shape-keyed shoe cages require evaluated point-preserving geometry')
    expected='leftFoot' if obj.name.endswith('_L') else 'rightFoot' if obj.name.endswith('_R') else None
    if expected is None or expected not in rig.pose.bones: raise RuntimeError('Cannot identify the authored foot bone')
    if not rig.data.bones[expected].use_deform: raise RuntimeError('Foot bone is not deforming')
    for idx in ids:
        weights={obj.vertex_groups[g.group].name:g.weight for g in obj.data.vertices[idx].groups
                 if obj.vertex_groups[g.group].name in rig.data.bones
                 and rig.data.bones[obj.vertex_groups[g.group].name].use_deform and g.weight>1e-7}
        if set(weights)!={expected} or abs(weights[expected]-1)>1e-5:
            raise RuntimeError(f'Vertex {obj.name}[{idx}] is not exclusively weighted 1.0 to {expected}')
    if mod.use_deform_preserve_volume:
        # One unit-weight bone with a proper rigid transform gives the same
        # point under dual-quaternion and linear skinning. Do not extend this
        # shortcut to scaled bones, mixed weights, or general DQ deformation.
        rotation=(rig.pose.bones[expected].matrix@rig.data.bones[expected].matrix_local.inverted()).to_3x3()
        product=rotation.transposed()@rotation
        error=max(abs(product[i][j]-(1.0 if i==j else 0.0)) for i in range(3) for j in range(3))
        if error>2e-5 or abs(rotation.determinant()-1.0)>2e-5:
            raise RuntimeError('Dual-quaternion cage shortcut requires a rigid unscaled foot transform')
    return rig,expected


def probe_description(selection,mode='auto'):
    """Return the actual coordinate source and restrictions without sampling."""
    _validate_selection(selection)
    if mode not in {'auto','evaluated','rigid_cage'}: raise ValueError('mode must be auto, evaluated, or rigid_cage')
    result={}
    for name,ids in selection.items():
        obj=_object(name);mods=_active_modifiers(obj)
        preserve=all(m.type in DEFORM_ONLY for m in mods)
        selected='evaluated' if preserve and mode!='rigid_cage' else 'rigid_cage'
        if mode=='evaluated' and not preserve:
            raise RuntimeError('Topology-changing modifier prevents source-index evaluation: '+name)
        bone=None
        if selected=='rigid_cage': _,bone=_rigid_cage_bone(obj,ids,mods)
        result[name]={'mode':selected,'vertexIds':list(ids),'sourceVertexCount':len(obj.data.vertices),
                      'sourceCoordinates':[list(obj.data.vertices[i].co) for i in ids],
                      'footBone':bone,'modifiers':[{'name':m.name,'type':m.type} for m in mods],
                      'bevelWidthsMeters':[float(m.width) for m in mods if m.type=='BEVEL'],
                      'surfaceScope':'evaluated source-index vertices' if selected=='evaluated' else
                                     'original weighted cage points; excludes bevel-generated surface'}
    return result


def sample_sole_points(selection,mode='auto'):
    """Read only the requested points; do not alter modifiers or scene state."""
    descriptions=probe_description(selection,mode)
    dg=bpy.context.evaluated_depsgraph_get();result={}
    for name,ids in selection.items():
        obj=_object(name)
        if descriptions[name]['mode']=='evaluated':
            evaluated=obj.evaluated_get(dg)
            if evaluated.type!='MESH' or len(evaluated.data.vertices)!=len(obj.data.vertices):
                raise RuntimeError('Evaluated topology no longer preserves source vertex indices: '+name)
            result[name]=[list(evaluated.matrix_world@evaluated.data.vertices[i].co) for i in ids]
        else:
            rig,bone=_rigid_cage_bone(obj,ids,_active_modifiers(obj))
            deform=rig.pose.bones[bone].matrix@rig.data.bones[bone].matrix_local.inverted()
            transform=rig.matrix_world@deform@rig.matrix_world.inverted()@obj.matrix_world
            result[name]=[list(transform@obj.data.vertices[i].co) for i in ids]
    return result


def resolve_ground_z(value=None,floor_name='Cyclorama floor'):
    """Read the current small, horizontal floor plane, or use an explicit height."""
    if value is not None:
        if not math.isfinite(float(value)): raise ValueError('Finite ground height required')
        return float(value)
    floor=_object(floor_name)
    if floor.modifiers: raise RuntimeError('Explicit ground_z required for a modified floor')
    if not (3<=len(floor.data.vertices)<=32): raise RuntimeError('Explicit ground_z required for a complex floor')
    zs=[(floor.matrix_world@v.co).z for v in floor.data.vertices]
    if max(zs)-min(zs)>1e-5: raise RuntimeError('This audit requires a horizontal floor plane')
    return sum(zs)/len(zs)


def _guard():
    p=project_path();source=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or source==(p/'work/character.blend').resolve():
        raise RuntimeError('Run action audits in a saved independent candidate, never the formal main source')
    source.relative_to((p/'work').resolve())
    rig=bpy.data.objects.get(RIG)
    if rig is None or rig.type!='ARMATURE' or rig.library: raise RuntimeError('Local Character_Rig required')
    if bpy.context.mode not in {'OBJECT','POSE'}: raise RuntimeError('Finish the active editing operation first')
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA requires a separately scoped evaluated audit')
    if any(getattr(h,'_r02_secondary_hair',False) for h in bpy.app.handlers.frame_change_post):
        raise RuntimeError('Detach the hair playback handler before changing body actions')
    return rig


def _entries(actions):
    inventory=json.loads((project_path()/'work/reconstruction-r02/imported-motions.json').read_text(encoding='utf-8'))['actions']
    known={e['action']:e for e in inventory}
    if isinstance(actions,str): raise ValueError('Pass a list of action names')
    names=list(known) if actions is None else list(actions)
    if not names or len(names)!=len(set(names)): raise ValueError('Choose one or more unique actions')
    result=[]
    for name in names:
        if bpy.data.actions.get(name) is None: raise RuntimeError('Existing action missing: '+name)
        result.append({'action':name,'sourceMotion':known[name]['file'] if name in known else None,
                       'sourceFromImportInventory':name in known,
                       'source':'Existing imported authored VRMA; not mocap' if name in known else
                                'Explicit existing Blender action; no import provenance claimed'})
    return result


def _assign_action(rig,action):
    rig.animation_data_create();rig.animation_data.action=action
    if action and hasattr(action,'slots'):
        slots=[s for s in action.slots if s.target_id_type=='OBJECT']
        if len(slots)!=1: raise RuntimeError('Expected one object slot in the body action')
        rig.animation_data.action_slot=slots[0]


def _action_hash(action):
    curves=[]
    if hasattr(action,'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip,'channelbags'):
                    for bag in strip.channelbags: curves.extend(bag.fcurves)
    if not curves and hasattr(action,'fcurves'): curves=list(action.fcurves)
    if not curves: raise RuntimeError('Cannot fingerprint body action curves: '+action.name)
    rows=[]
    for c in sorted(curves,key=lambda c:(c.data_path,c.array_index)):
        rows.append([c.data_path,c.array_index,[[list(k.co),list(k.handle_left),list(k.handle_right),k.interpolation]
                                               for k in c.keyframe_points]])
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def _distance(a,b,horizontal=False):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(2 if horizontal else 3)))


def _zones(selection):
    zones={}
    for name,ids in selection.items():
        obj=_object(name);ys=[obj.data.vertices[i].co.y for i in ids]
        low,high=min(ys),max(ys);span=max(high-low,1e-8)
        zones[name]={'toe':[j for j,y in enumerate(ys) if y<=low+span*.25],
                     'heel':[j for j,y in enumerate(ys) if y>=high-span*.25]}
    return zones


def _sample_frames(start,end,frame_step,max_count):
    """Pure bounded sample construction, preserving both fractional endpoints."""
    if isinstance(frame_step,bool): raise ValueError('Positive finite frame step required')
    start,end,step=float(start),float(end),float(frame_step)
    if not all(math.isfinite(v) for v in (start,end,step)) or step<=0:
        raise ValueError('Finite endpoints and positive finite frame step required')
    if end<=start: raise ValueError('At least two distinct sampled frames are needed')
    if not isinstance(max_count,int) or isinstance(max_count,bool) or max_count<2:
        raise ValueError('Bounded audit sample allowance exhausted')
    ratio=(end-start)/step
    # Check the count before range/list allocation, including underflowing steps.
    if not math.isfinite(ratio) or ratio>max_count:
        raise ValueError(f'Bounded audit sample allowance exceeded: {max_count}')
    steps=math.floor(ratio);last=start+steps*step
    tolerance=min((end-start)*1e-6,8*max(math.ulp(start),math.ulp(end),math.ulp(step)))
    at_end=steps>0 and abs(last-end)<=tolerance
    count=steps+1+(not at_end)
    if count>max_count: raise ValueError(f'Bounded audit sample allowance exceeded: {count}>{max_count}')
    frames=[start+i*step for i in range(steps+1)]
    if at_end: frames[-1]=end
    else: frames.append(end)
    if any(b<=a for a,b in zip(frames,frames[1:])):
        raise ValueError('Frame step is below representable precision for this range')
    return frames


def audit_actions(actions=None,frame_ranges=None,sole_vertex_ids=None,objects=DEFAULT_OBJECTS,
                  mode='auto',ground_z=None,frame_step=1,max_total_frames=1600,
                  contact_height=0.004,penetration_tolerance=0.002,liftoff_height=0.012,
                  sliding_speed=0.05,contact_vertical_speed=0.03,
                  output='work/reconstruction-r02/foot-contact-audit-r03.json'):
    """Audit a bounded sequence and restore action/slot/frame/pose/IK state.

    frame_ranges is {action_name:(start,end)}; fractional endpoints are included.
    frame_step accepts a positive finite float (e.g. .25 for quarter frames).
    Defaults
    to the four imported R02 actions. An explicit R03 action keeps its own keyed
    IK switches; imported actions start from disabled IK before evaluation.
    Sampling does not write keyframes, change source geometry, or save a Blend.
    """
    rig=_guard();scene=bpy.context.scene
    fps=scene.render.fps/scene.render.fps_base
    if isinstance(frame_step,bool): raise ValueError('Positive finite frame step required')
    frame_step=float(frame_step)
    if not (math.isfinite(fps) and fps>0 and math.isfinite(frame_step) and frame_step>0):
        raise ValueError('Positive finite fps and frame step required')
    if not isinstance(max_total_frames,int) or isinstance(max_total_frames,bool) or max_total_frames<2:
        raise ValueError('max_total_frames must be an integer of at least two')
    if not (0<=penetration_tolerance<contact_height<liftoff_height) or sliding_speed<=0 or contact_vertical_speed<0:
        raise ValueError('Invalid contact/speed thresholds')
    entries=_entries(actions)
    selection=sole_vertex_ids if sole_vertex_ids is not None else select_sole_vertices(objects,max_points=12)
    descriptions=probe_description(selection,mode);zones=_zones(selection)
    floor=resolve_ground_z(ground_z)
    grouped={}
    for name in selection:
        side='L' if name.endswith('_L') else 'R' if name.endswith('_R') else name
        grouped.setdefault(side,[]).append(name)
    planned=[];total=0
    for entry in entries:
        action=bpy.data.actions[entry['action']]
        start,end=(frame_ranges or {}).get(action.name,action.frame_range)
        frames=_sample_frames(start,end,frame_step,max_total_frames-total)
        total+=len(frames);planned.append((entry,action,frames))
    missing=set(ENDPOINTS)-set(rig.pose.bones.keys())
    if missing: raise RuntimeError('Missing endpoint bones: '+str(sorted(missing)))
    old_action=rig.animation_data.action if rig.animation_data else None
    old_slot=rig.animation_data.action_slot if rig.animation_data and hasattr(rig.animation_data,'action_slot') else None
    old_frame,old_subframe=scene.frame_current,scene.frame_subframe
    old_basis={b.name:b.matrix_basis.copy() for b in rig.pose.bones}
    master=rig.pose.bones.get(MASTER)
    old_switches={k:float(master[k]) for k in IK_KEYS if master and k in master}
    old_angles=[(c,c.pole_angle) for b in rig.pose.bones for c in b.constraints if c.type=='IK']
    report={'version':1,'sourceBlend':str(Path(bpy.data.filepath).relative_to(project_path())),
            'sourceHasUnsavedChanges':bool(bpy.data.is_dirty),
            'footBoneRestMatrices':{n:[list(row) for row in rig.data.bones[n].matrix_local] for n in ('leftFoot','rightFoot')},
            'fps':fps,'groundZ':floor,'groundSource':'explicit' if ground_z is not None else 'current Cyclorama floor plane',
            'sampledFrames':total,'probes':descriptions,'toeHeelLabelBasis':'authored rest coordinates: -Y forward',
            'thresholds':{'contactHeightMeters':contact_height,'penetrationToleranceMeters':penetration_tolerance,
                          'liftoffHeightMeters':liftoff_height,'slidingSpeedMetersPerSecond':sliding_speed,
                          'contactVerticalSpeedMetersPerSecond':contact_vertical_speed},
            'endpointDefinition':{n:'world-space pose bone tail' for n in ENDPOINTS},
            'limitations':LIMITS,'actions':[],'savedBlend':False}
    try:
        for entry,action,frames in planned:
            # Do not let unkeyed values left by another action contaminate this one.
            for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
            if master:
                for key in IK_KEYS:
                    if key in master: master[key]=0.0
            for constraint,angle in old_angles: constraint.pole_angle=angle
            _assign_action(rig,action)
            rows=[];previous=None;endpoint_max={n:0.0 for n in ENDPOINTS}
            summaries={side:{'minimumClearanceMeters':math.inf,'maximumMinimumClearanceMeters':-math.inf,
                       'maximumProbeClearanceMeters':-math.inf,
                       'penetrationFrames':[],'airborneFrames':[],'contactCandidateFrames':[],
                       'suspectedSlidingIntervals':[],'maxContactPointHorizontalSpeed':0.0}
                       for side in grouped}
            for index,frame in enumerate(frames):
                scene.frame_set(math.floor(frame),subframe=frame%1.0);bpy.context.view_layer.update()
                points=sample_sole_points(selection,mode)
                ends={n:list(rig.matrix_world@rig.pose.bones[n].tail) for n in ENDPOINTS}
                dt=(frame-previous['frame'])/fps if previous else None
                row={'frame':frame,'feet':{},'endpointSpeedMetersPerSecond':{}}
                if previous:
                    for n in ENDPOINTS:
                        speed=_distance(ends[n],previous['ends'][n])/dt
                        row['endpointSpeedMetersPerSecond'][n]=speed;endpoint_max[n]=max(endpoint_max[n],speed)
                for side,names in grouped.items():
                    all_points=[p for name in names for p in points[name]]
                    clearance=[p[2]-floor for p in all_points]
                    low,high=min(clearance),max(clearance)
                    toe=[points[name][j][2]-floor for name in names if 'Stacked_Heel' not in name for j in zones[name]['toe']]
                    heel=[points[name][j][2]-floor for name in names for j in zones[name]['heel']]
                    point_speed=[];suspects=[]
                    if previous:
                        for name in names:
                            for j,p in enumerate(points[name]):
                                old=previous['points'][name][j]
                                near=(-penetration_tolerance<=p[2]-floor<=contact_height and
                                      -penetration_tolerance<=old[2]-floor<=contact_height)
                                vertical=abs(p[2]-old[2])/dt
                                if near and vertical<=contact_vertical_speed:
                                    speed=_distance(p,old,True)/dt;point_speed.append(speed)
                                    if speed>sliding_speed:
                                        suspects.append({'object':name,'vertexId':selection[name][j],'speed':speed})
                    contact=(-penetration_tolerance<=low<=contact_height)
                    row['feet'][side]={'minimumClearanceMeters':low,'maximumClearanceMeters':high,
                                      'toeClearanceMeters':min(toe) if toe else None,
                                      'heelClearanceMeters':min(heel) if heel else None,
                                      'contactCandidate':contact,'maxContactPointHorizontalSpeed':max(point_speed,default=0.0)}
                    summary=summaries[side]
                    summary['minimumClearanceMeters']=min(summary['minimumClearanceMeters'],low)
                    summary['maximumMinimumClearanceMeters']=max(summary['maximumMinimumClearanceMeters'],low)
                    summary['maximumProbeClearanceMeters']=max(summary['maximumProbeClearanceMeters'],high)
                    summary['maxContactPointHorizontalSpeed']=max(summary['maxContactPointHorizontalSpeed'],max(point_speed,default=0.0))
                    if low < -penetration_tolerance: summary['penetrationFrames'].append(frame)
                    if low > liftoff_height: summary['airborneFrames'].append(frame)
                    if contact: summary['contactCandidateFrames'].append(frame)
                    if suspects: summary['suspectedSlidingIntervals'].append({'fromFrame':previous['frame'],'toFrame':frame,'points':suspects})
                rows.append(row);previous={'frame':frame,'points':points,'ends':ends}
                if index%60==0: print(f'R03_FOOT_AUDIT {action.name}: {index+1}/{len(frames)}')
            report['actions'].append(dict(entry,actionSha256=_action_hash(action),frameRange=[frames[0],frames[-1]],frameStep=frame_step,
                                          feet=summaries,maxEndpointSpeedMetersPerSecond=endpoint_max,samples=rows))
    finally:
        _assign_action(rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        scene.frame_set(old_frame,subframe=old_subframe)
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        if master:
            for key,value in old_switches.items(): master[key]=value
        for constraint,angle in old_angles: constraint.pole_angle=angle
        rig.update_tag();bpy.context.view_layer.update()
    if output is not None:
        relative=Path(output)
        if relative.is_absolute() or '..' in str(output).replace('\\','/').split('/'):
            raise ValueError('Use a project-relative output path')
        path=(project_path()/relative).resolve();path.relative_to((project_path()/'work').resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        report['reportPath']=str(path.relative_to(project_path()))
    print(json.dumps({'actions':[e['action'] for e in entries],'sampledFrames':total,
                      'reportPath':report.get('reportPath'),'savedBlend':False,'scope':'finite shoe support and bone-tail audit'},ensure_ascii=False))
    return report


if __name__=='__main__':
    print('Loaded only. Call select_sole_vertices(), sample_sole_points(), or audit_actions() explicitly.')
