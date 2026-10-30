"""Candidate-only correction of the collapsed neck/head rest pivot.

Only neck.tail and head.head change. No mesh/key/curve coordinate, weight,
action, parent relation or deformation modifier is rewritten. Bone-parented
attachments retain their REST world transforms through parent-inverse offsets.
run() applies and reports; apply_and_save() additionally saves a new candidate.
Both require explicit invocation in the existing Blender; neither launches it.
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
    from mathutils import Matrix,Vector
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
RIG='Character_Rig'
MARKER='R06_head_pivot_calibration'
PIVOT=(0.,.020,1.370)
REPORT='work/reconstruction-r02/head-pivot-calibration-r06.json'
SOURCE_INPUT='work/reconstruction-r02/neck-calibration-input-r06.json'
FOOT_ACTIONS=('R03 CMU 07_01 walk - proportional IK preview v4','R06 CMU 07_01 walk - forward gaze')


def _array(data,prop):
    out=np.empty(len(data)*3,dtype=np.float32);data.foreach_get(prop,out)
    return out.reshape(-1,3)


def _blocks(obj):
    """Raw coordinates only: no evaluated cloth, subdivision or collision job."""
    if obj.type=='MESH':
        yield 'mesh',_array(obj.data.vertices,'co')
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks: yield 'key:'+key.name,_array(key.data,'co')
    elif obj.type=='CURVES': yield 'native-curves',_array(obj.data.position_data,'vector')
    elif obj.type=='CURVE':
        for i,spline in enumerate(obj.data.splines):
            for prop in ('co','handle_left','handle_right'):
                if spline.bezier_points:
                    yield f'spline:{i}:bezier:{prop}',np.asarray([list(getattr(p,prop)) for p in spline.bezier_points],dtype=np.float32)
            if spline.points: yield f'spline:{i}:points',np.asarray([list(p.co)[:3] for p in spline.points],dtype=np.float32)


def _bone_records(rig):
    return {b.name:{'head':list(b.head_local),'tail':list(b.tail_local),'matrix':[list(r) for r in b.matrix_local],
                   'parent':b.parent.name if b.parent else None,'use_connect':bool(b.use_connect),
                   'use_deform':bool(b.use_deform),'length':float(b.length)} for b in rig.data.bones}


def _rest_difference(before,after):
    """Separate metric positions from orientation and float stretch residuals."""
    a=np.asarray(before['matrix']);b=np.asarray(after['matrix'])
    ua,sa,va=np.linalg.svd(a[:3,:3]);ub,sb,vb=np.linalg.svd(b[:3,:3])
    ra=ua@va;rb=ub@vb;relative=rb@ra.T
    sine=.5*np.linalg.norm([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])
    angle=math.degrees(math.atan2(float(sine),float((np.trace(relative)-1)*.5)))
    stretch_a=va.T@np.diag(sa)@va;stretch_b=vb.T@np.diag(sb)@vb
    return {'endpointPositionMeters':max(float(np.linalg.norm(np.asarray(before[k])-after[k])) for k in ('head','tail')),
            'matrixOriginPositionMeters':float(np.linalg.norm(a[:3,3]-b[:3,3])),
            'normalizedOrientationDegrees':angle,
            'stretchMatrixComponentDelta':float(np.max(np.abs(stretch_b-stretch_a))),
            'rawMatrixComponentDelta':float(np.max(np.abs(a-b)))}


def _affected_bones(rig):
    result=set()
    for bone in rig.data.bones:
        ancestor=bone
        while ancestor:
            if ancestor.name=='neck': result.add(bone.name);break
            ancestor=ancestor.parent
    return result


def _affected_objects(rig,affected):
    attachments=[o for o in bpy.data.objects if o.parent==rig and o.parent_type=='BONE' and o.parent_bone in affected]
    selected=set(attachments)
    for obj in bpy.data.objects:
        ancestor=obj.parent
        while ancestor and ancestor!=rig:
            if ancestor in selected: selected.add(obj);break
            ancestor=ancestor.parent
        if obj.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers):
            if affected.intersection(obj.vertex_groups.keys()): selected.add(obj)
    return attachments,sorted(selected,key=lambda o:o.name)


def _binding_signature(obj):
    modifiers=[{'name':m.name,'type':m.type,'rig':m.object.name if m.object else None,
        'preserveVolume':bool(m.use_deform_preserve_volume),'vertexGroups':bool(m.use_vertex_groups),
        'envelopes':bool(m.use_bone_envelopes),'vertexGroup':m.vertex_group,'multi':bool(m.use_multi_modifier),
        'viewport':bool(m.show_viewport),'render':bool(m.show_render)} for m in obj.modifiers if m.type=='ARMATURE']
    weights=hashlib.sha256()
    if obj.type=='MESH':
        weights.update('|'.join(v.name for v in obj.vertex_groups).encode())
        for v in obj.data.vertices:
            for g in v.groups: weights.update(np.asarray([v.index,g.group,g.weight],dtype=np.float64).tobytes())
    return {'armatureModifiers':modifiers,'weightsSha256':weights.hexdigest()}


def _geometry_snapshot(objects):
    result={}
    for obj in objects:
        blocks=[];digest=hashlib.sha256()
        for label,array in _blocks(obj):
            digest.update(label.encode());digest.update(array.tobytes());blocks.append((label,array))
        result[obj.name]={'world':np.asarray(obj.matrix_world).copy(),'blocks':blocks,
                          'rawSha256':digest.hexdigest(),'binding':_binding_signature(obj)}
    return result


def _geometry_compare(before,objects):
    rows={};maximum=0.;total=0
    for obj in objects:
        old=before[obj.name];new=list(_blocks(obj));digest=hashlib.sha256()
        if [n for n,_ in new]!=[n for n,_ in old['blocks']]: raise RuntimeError('Coordinate block identity changed: '+obj.name)
        mw=np.asarray(obj.matrix_world);delta=0.;count=0
        for (label,array),(_,original) in zip(new,old['blocks']):
            digest.update(label.encode());digest.update(array.tobytes())
            if array.shape!=original.shape or not np.array_equal(array,original):
                raise RuntimeError('Raw geometry changed: '+obj.name+'/'+label)
            if len(array):
                old_world=original@old['world'][:3,:3].T+old['world'][:3,3]
                new_world=array@mw[:3,:3].T+mw[:3,3]
                delta=max(delta,float(np.max(np.linalg.norm(new_world-old_world,axis=1))))
                count+=len(array)
        if _binding_signature(obj)!=old['binding']: raise RuntimeError('Skin weights or Armature configuration changed: '+obj.name)
        if delta>2e-6: raise RuntimeError(f'Neutral original-cage geometry moved: {obj.name}, {delta} m')
        rows[obj.name]={'rawSha256Before':old['rawSha256'],'rawSha256After':digest.hexdigest(),
                        'rawCoordinatesExactlyEqual':True,'weightAndArmatureSettingsEqual':True,
                        'pointCountAcrossRawBlocks':count,'maxRestWorldPointDeltaMeters':delta}
        maximum=max(maximum,delta);total+=count
    return {'maxRestWorldPointDeltaMeters':maximum,'rawPointCountAcrossBlocks':total,'objects':rows,
            'scope':'Every affected original mesh cage, shape-key block and curve control point in rig REST; not evaluated cloth/subdivision or a scene collision check'}


def _foot_samples(rig,helper,audit,actions,frames):
    result={};rig.data.pose_position='POSE'
    selection=audit['select_sole_vertices'](objects=audit['DEFAULT_OBJECTS'],max_points=12)
    for name in actions:
        helper['_assign'](rig,bpy.data.actions[name]);rows={}
        for frame in frames:
            bpy.context.scene.frame_set(math.floor(frame),subframe=frame%1);bpy.context.view_layer.update()
            rows[str(frame)]=audit['sample_sole_points'](selection,mode='rigid_cage')
        result[name]=rows
    return result


def collider_recommendation(rig=None):
    """Fit only the exported cranial rear/side/top; return an explicit center."""
    path=project_path()/SOURCE_INPUT;r=json.loads(path.read_text(encoding='utf-8-sig'))
    points=np.asarray(r['headSkinRestRegionVertices'],dtype=float)
    sample=points[(points[:,2]>1.39)&((points[:,1]>.02)|(points[:,2]>1.48)|(np.abs(points[:,0])>.065))]
    fit=np.linalg.lstsq(np.column_stack([2*sample,np.ones(len(sample))]),np.sum(sample*sample,axis=1),rcond=None)[0]
    center=np.array([0.,.00202,1.441285]);radius=.084
    distances=np.linalg.norm(sample-center,axis=1)
    local=None
    if rig is not None: local=list(rig.data.bones['head'].matrix_local.inverted()@Vector(center))
    return {'name':'head','bone':'head','restWorldCenter':center.tolist(),'localCenter':local,'radius':radius,
        'fitSampleCount':len(sample),'fitCenterMeters':fit[:3].tolist(),
        'unreducedFitRadiusMeters':float(math.sqrt(fit[3]+np.dot(fit[:3],fit[:3]))),
        'sampleDistanceRangeMeters':[float(distances.min()),float(distances.max())],
        'scope':'Conservative inner sphere from cranial rear/side/top fit; not an exact head surface. Recheck guide collision and tune after the new pose/hair bake.',
        'inputSha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def _edit_mode(rig):
    if bpy.context.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    rig.hide_set(False);bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
    bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')


def run(pivot=PIVOT,report_relative=REPORT,foot_frames=(1.,7.,15.,23.,35.,45.,61.,79.75)):
    """Apply only in a saved independent candidate; retain original actions."""
    if bpy is None: raise RuntimeError('Explicit Blender candidate invocation required')
    helper=runpy.run_path(str(Path(__file__).with_name('full_body_review_r03.py')))
    audit=runpy.run_path(str(Path(__file__).with_name('foot_contact_audit_r03.py')))
    rig=helper['_guard']();audit['_guard']();scene=bpy.context.scene
    if MARKER in rig: raise RuntimeError('Already calibrated; inspect the existing report instead of applying twice')
    if 'R04_native_hair_guide' in rig: raise RuntimeError('Restore/unembed the old native hair cache before changing its rest binding')
    if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6: raise RuntimeError('Confirmed identity rig object transform required')
    pivot=np.asarray(pivot,dtype=float)
    if pivot.shape!=(3,) or not np.all(np.isfinite(pivot)) or not (abs(pivot[0])<.005 and -.005<pivot[1]<.045 and 1.35<pivot[2]<1.39):
        raise ValueError('Pivot must remain inside the scoped lower-cranium candidate region')
    if rig.data.bones['head'].parent.name!='neck': raise RuntimeError('Unexpected head hierarchy')
    if any(b.name!='head' and b.use_connect for b in rig.data.bones['neck'].children):
        raise RuntimeError('Another connected neck child requires a separately scoped rest edit')
    affected=_affected_bones(rig);attachments,objects=_affected_objects(rig,affected)
    if any(o.library or o.constraints for o in attachments): raise RuntimeError('Linked/constrained bone attachments require an explicit evaluated offset strategy')
    actions=[name for name in FOOT_ACTIONS if bpy.data.actions.get(name)]
    if not actions: raise RuntimeError('Existing audited source walk required for the finite foot-chain preservation check')
    action_hashes={name:audit['_action_hash'](bpy.data.actions[name]) for name in actions}
    original_bones=_bone_records(rig);source_path=bpy.data.filepath
    old_action=rig.animation_data.action;old_slot=rig.animation_data.action_slot
    old_frame,old_subframe=scene.frame_current,scene.frame_subframe;old_pose=rig.data.pose_position
    old_basis={p.name:p.matrix_basis.copy() for p in rig.pose.bones};master=rig.pose.bones['CTRL_R02_Master']
    switches={k:float(master['ik_'+k]) for k in ('leg_L','leg_R','arm_L','arm_R')}
    old_active=bpy.context.view_layer.objects.active;old_selected=list(bpy.context.selected_objects)
    old_hidden=rig.hide_get();old_mode=bpy.context.mode
    attachments_before={};edit_before=None;changed=False;complete=False
    try:
        feet_before=_foot_samples(rig,helper,audit,actions,foot_frames)
        helper['_assign'](rig,None)
        for pb in rig.pose.bones: pb.matrix_basis=Matrix.Identity(4)
        for key in switches: master['ik_'+key]=0.
        rig.data.pose_position='REST';rig.update_tag();bpy.context.view_layer.update()
        attachments_before={o.name:{'world':o.matrix_world.copy(),'inverse':o.matrix_parent_inverse.copy(),
                                    'basis':o.matrix_basis.copy()} for o in attachments}
        geometry_before=_geometry_snapshot(objects)
        _edit_mode(rig)
        edit_before={b.name:{'head':b.head.copy(),'tail':b.tail.copy(),'roll':b.roll,'connect':b.use_connect} for b in rig.data.edit_bones}
        head=rig.data.edit_bones['head'];neck=rig.data.edit_bones['neck'];connect=head.use_connect
        changed=True
        head.use_connect=False;neck.tail=Vector(pivot);head.head=Vector(pivot);head.use_connect=connect
        bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
        # Derive Blender's evaluated bone-parent transform rather than assuming
        # its tail-origin convention. Keep object matrix_basis exactly intact.
        for obj in attachments:
            before=attachments_before[obj.name]
            parent_world=obj.matrix_world@before['basis'].inverted()@before['inverse'].inverted()
            obj.matrix_parent_inverse=parent_world.inverted()@before['world']@before['basis'].inverted()
        bpy.context.view_layer.update()
        attachment_report={}
        for obj in attachments:
            before=attachments_before[obj.name]
            world_error=float(np.max(np.abs(np.asarray(obj.matrix_world)-np.asarray(before['world']))))
            basis_error=float(np.max(np.abs(np.asarray(obj.matrix_basis)-np.asarray(before['basis']))))
            if world_error>2e-6 or basis_error>1e-8: raise RuntimeError('REST attachment offset mismatch: '+obj.name)
            attachment_report[obj.name]={'bone':obj.parent_bone,'restWorldMatrixBefore':[list(r) for r in before['world']],
                'restWorldMatrixAfter':[list(r) for r in obj.matrix_world],
                'parentInverseBefore':[list(r) for r in before['inverse']],
                'parentInverseAfter':[list(r) for r in obj.matrix_parent_inverse],
                'maxRestWorldMatrixDelta':world_error,'matrixBasisMaxDelta':basis_error}
        geometry_report=_geometry_compare(geometry_before,objects)
        after_bones=_bone_records(rig);other_error=0.;leg_error=0.;rest_differences={}
        for name,before in original_bones.items():
            after=after_bones[name]
            if any(before[k]!=after[k] for k in ('parent','use_connect','use_deform')):
                raise RuntimeError('Bone hierarchy/connect/deform flags changed: '+name)
            if name not in {'neck','head'}:
                delta=max(float(np.max(np.abs(np.asarray(before[k])-np.asarray(after[k])))) for k in ('head','tail','matrix'))
                other_error=max(other_error,delta)
                measures=_rest_difference(before,after);rest_differences[name]=measures
                # Actual failed-state diagnosis found ZERO endpoint motion,
                # <=15 nm matrix-origin drift, <=0.000162 degree rotation and
                # ~4.4e-6 stretch residual in old long hair chains. Comparing a
                # dimensionless rotation entry to a 2 micrometre distance was
                # the wrong test. Keep the metric position limit unchanged;
                # .001 degree means <1.75 micrometres at a 100 mm head radius.
                if (max(measures['endpointPositionMeters'],measures['matrixOriginPositionMeters'])>2e-6
                    or measures['normalizedOrientationDegrees']>.001
                    or measures['stretchMatrixComponentDelta']>1e-5):
                    raise RuntimeError(f'Unscoped bone rest transform changed: {name}; {measures}')
                if any(t in name for t in ('Leg','Foot','Toes','Knee')):
                    leg_error=max(leg_error,delta)
                    if delta>2e-6: raise RuntimeError('Leg-chain raw rest matrix changed: '+name)
        if np.max(np.abs(np.asarray(after_bones['head']['tail'])-original_bones['head']['tail']))>1e-7:
            raise RuntimeError('Head tail changed')
        if np.max(np.abs(np.asarray(after_bones['neck']['head'])-original_bones['neck']['head']))>1e-7:
            raise RuntimeError('Neck base changed')
        collider=collider_recommendation(rig)
        feet_after=_foot_samples(rig,helper,audit,actions,foot_frames);foot_error=0.
        for name,rows in feet_before.items():
            for frame,points in rows.items():
                for obj_name,values in points.items():
                    delta=np.linalg.norm(np.asarray(feet_after[name][frame][obj_name])-np.asarray(values),axis=1)
                    foot_error=max(foot_error,float(np.max(delta)))
        if foot_error>2e-6: raise RuntimeError('Finite foot samples changed after the neck/head rest edit')
        if action_hashes!={name:audit['_action_hash'](bpy.data.actions[name]) for name in actions}:
            raise RuntimeError('Existing source action curves changed')
        report={'sourceBlend':source_path,'pivotRestWorldMeters':pivot.tolist(),
            'choice':'Lower cranium, 7.2 mm below the current jaw hinge and about 1.6 mm behind it; aligned with the measured lower-neck sagittal center.',
            'changedRestEndpoints':['neck.tail','head.head'],'bonesBefore':original_bones,'bonesAfter':after_bones,
            'maximumOtherBoneRestDelta':other_error,'maximumLegBoneRestDelta':leg_error,
            'nonHeadNeckRestDifferences':rest_differences,
            'restPreservationLimits':{'positionMeters':2e-6,'normalizedOrientationDegrees':.001,
                'stretchMatrixComponentDelta':1e-5,'legRawMatrixComponentDelta':2e-6,
                'rationale':'Measured edit-mode float reconstruction of long hair chains; head/tail positions were exactly unchanged. Geometry/world-point and attachment limits remain 2 micrometres.'},
            'headTailUnchanged':True,'neckBaseUnchanged':True,'parentConnectDeformFlagsUnchanged':True,
            'geometry':geometry_report,'attachments':attachment_report,
            'maximumAttachmentRestWorldMatrixDelta':max((v['maxRestWorldMatrixDelta'] for v in attachment_report.values()),default=0.),
            'originalActionHashesUnchanged':action_hashes,'finiteFootFrames':list(foot_frames),
            'finiteFootMaximumWorldPointDeltaMeters':foot_error,'suggestedHeadCollider':collider,
            'scope':'Neutral original cages and shape-key/control points are preserved; existing head/neck poses change because their rest pivot changes. Old head/hair animation evidence is historical; leg checks are finite.',
            'needsNewChecks':['Head and neck motion/skin deformation','Forward-gaze head action on the new rest rig','Rebaked head-relative hair cache with explicit collider','Cloth/whole-scene intersections'],
            'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'savedBlend':False,'accepted':False}
        out=(project_path()/report_relative).resolve();out.relative_to((project_path()/'work').resolve())
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        rig[MARKER]=json.dumps({'pivot':pivot.tolist(),'report':report_relative,'source':source_path})
        complete=True;return report
    except Exception as error:
        # Preserve the offending state BEFORE finally rolls the candidate back.
        # This file is diagnostic evidence, never a success/acceptance report.
        if bpy.context.mode=='EDIT_ARMATURE': bpy.ops.object.mode_set(mode='OBJECT')
        failed_bones=_bone_records(rig);deltas={}
        for name,before in original_bones.items():
            if name not in failed_bones: continue
            after=failed_bones[name]
            components={key:float(np.max(np.abs(np.asarray(before[key])-np.asarray(after[key]))))
                        for key in ('head','tail','matrix')}
            if max(components.values())>0:
                deltas[name]={'maxComponentDelta':components,'before':before,'after':after}
        failed={'success':False,'error':str(error),'sourceBlend':source_path,
                'requestedPivot':pivot.tolist(),'state':'Captured before automatic rollback',
                'changedBoneRestRecords':deltas,'allowedChangedBones':['neck','head'],
                'positionThresholdUnchangedMeters':2e-6,'scriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        failure_path=project_path()/'work/reconstruction-r02/head-pivot-calibration-r06-failed.json'
        failure_path.write_text(json.dumps(failed,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'calibrationFailed':True,'error':str(error),'preRollbackDiagnostic':str(failure_path)},ensure_ascii=False))
        raise
    finally:
        if not complete and changed and edit_before:
            _edit_mode(rig)
            for bone in rig.data.edit_bones: bone.use_connect=False
            for bone in rig.data.edit_bones:
                old=edit_before[bone.name];bone.head=old['head'];bone.tail=old['tail'];bone.roll=old['roll']
            for bone in rig.data.edit_bones: bone.use_connect=edit_before[bone.name]['connect']
            bpy.ops.object.mode_set(mode='OBJECT')
            for obj in attachments:
                if obj.name in attachments_before: obj.matrix_parent_inverse=attachments_before[obj.name]['inverse']
        if bpy.context.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
        helper['_assign'](rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        rig.data.pose_position=old_pose;scene.frame_set(old_frame,subframe=old_subframe)
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        for key,value in switches.items(): master['ik_'+key]=value
        rig.update_tag();bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT')
        for obj in old_selected:
            if obj.name in bpy.context.view_layer.objects: obj.select_set(True)
        bpy.context.view_layer.objects.active=old_active;rig.hide_set(old_hidden)
        if old_mode=='POSE' and old_active==rig: bpy.ops.object.mode_set(mode='POSE')


def apply_and_save(output_relative='work/reconstruction-r02/reference-neck-calibrated-r06.blend',**kwargs):
    """Apply, restore scene state, then save a new nonexisting candidate path."""
    if bpy is None: raise RuntimeError('Explicit Blender candidate invocation required')
    output=(project_path()/output_relative).resolve();output.relative_to((project_path()/'work').resolve())
    if output==(project_path()/'work/character.blend').resolve() or output.exists():
        raise RuntimeError('Choose a new candidate path; no file will be overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    report=run(**kwargs)
    result=bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if 'FINISHED' not in result or not output.exists(): raise RuntimeError('Calibration applied but candidate save was not confirmed')
    report.update(savedBlend=True,outputBlend=str(output))
    report_path=project_path()/kwargs.get('report_relative',REPORT)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'neckLength':report['bonesAfter']['neck']['length'],
        'geometryMaxDelta':report['geometry']['maxRestWorldPointDeltaMeters'],
        'attachmentMaxDelta':report['maximumAttachmentRestWorldMatrixDelta'],
        'footMaxDelta':report['finiteFootMaximumWorldPointDeltaMeters']},ensure_ascii=False))
    return report


if __name__=='__main__':
    print('Loaded only. run() applies/reports; apply_and_save() additionally saves a new independent candidate.')
