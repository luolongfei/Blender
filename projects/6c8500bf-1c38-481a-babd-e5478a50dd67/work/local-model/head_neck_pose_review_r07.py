"""Small real-render neck/pivot review; explicit run(), no launch and no save.

Five available poses: neutral, yaw +/-30, pitch +/-15. At most four stills,
480x540, Cycles 12 samples. Head-only articulation isolates the corrected pivot
and the neck/head skin blend; neck/body remain neutral. Existing eye/jaw pose
and facial drivers remain enabled. Every changed scene setting is restored.
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
POSES={'neutral':(0.,0.),'turn_left':(30.,0.),'turn_right':(-30.,0.),
       'look_down':(0.,-15.),'look_up':(0.,15.)}
VIEWS={'face-quarter':((3.,-4.,1.429),(0.,-.005,1.429),.36),
       'face-side':((3.,0.,1.429),(0.,-.005,1.429),.36)}
DEFAULT_RENDERS=(('turn_left','face-quarter'),('turn_right','face-quarter'),
                 ('look_down','face-side'),('look_up','face-side'))
DEFORM_ONLY={'ARMATURE','LATTICE','CURVE','SIMPLE_DEFORM','CAST','SMOOTH',
             'CORRECTIVE_SMOOTH','LAPLACIANSMOOTH','LAPLACIANDEFORM','DISPLACE',
             'SHRINKWRAP','WARP','WAVE','SURFACE_DEFORM','MESH_DEFORM','HOOK',
             'WEIGHTED_NORMAL','NORMAL_EDIT','DATA_TRANSFER'}


def _sha_file(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def _boundary_vertices(head):
    required={n:head.vertex_groups.get(n) for n in ('neck','head')}
    if any(v is None for v in required.values()): raise RuntimeError('Existing neck/head skin groups are required')
    indices={n:g.index for n,g in required.items()};chosen=[]
    for vertex in head.data.vertices:
        weights={g.group:g.weight for g in vertex.groups}
        if all(weights.get(i,0.)>=.05 for i in indices.values()): chosen.append(vertex.index)
    if not chosen or len(chosen)>8000: raise RuntimeError('Unexpected or oversized neck/head blend region')
    return chosen


def _driver_states(rig,head):
    ids=[rig,head]
    if head.data.shape_keys: ids.append(head.data.shape_keys)
    return [(item.name,f.data_path,f.array_index,bool(f.mute)) for item in ids
            if item.animation_data for f in item.animation_data.drivers]


def _deform_probe(head,scene):
    # Linked mesh/key data and copied vertex groups preserve the actual drivers
    # and dual-quaternion Armature deformation. Remove topology-changing
    # modifiers only from this temporary, render-hidden probe object.
    probe=head.copy();probe.name='R07 temporary neck cage probe';scene.collection.objects.link(probe)
    probe.hide_render=True;probe.hide_viewport=False;probe.hide_set(False)
    removed=[]
    for modifier in list(probe.modifiers):
        if modifier.type not in DEFORM_ONLY:
            removed.append({'name':modifier.name,'type':modifier.type});probe.modifiers.remove(modifier)
    return probe,removed


def _sample_probe(probe,indices,count):
    evaluated=probe.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if len(evaluated.data.vertices)!=count: raise RuntimeError('Probe modifier stack does not preserve source indices')
    return np.asarray([list(evaluated.matrix_world@evaluated.data.vertices[i].co) for i in indices])


def _pose(rig,name):
    yaw,pitch=POSES[name];head=rig.pose.bones['head'];rest=head.bone.matrix_local.copy()
    yaw_rotation=Matrix.Rotation(math.radians(yaw),3,'Z')
    horizontal=yaw_rotation@Vector((0.,-1.,0.));right=Vector((0.,0.,1.)).cross(horizontal).normalized()
    rotation=Matrix.Rotation(-math.radians(pitch),3,right)@yaw_rotation
    desired=rest.copy();r=rotation@rest.to_3x3()
    for i in range(3):
        for j in range(3): desired[i][j]=r[i][j]
    head.matrix_basis=head.bone.convert_local_to_pose(desired,head.bone.matrix_local,
        parent_matrix=head.parent.matrix,parent_matrix_local=head.parent.bone.matrix_local,invert=True)
    rig.update_tag();bpy.context.view_layer.update()
    return {'headWorldYawDegrees':yaw,'headWorldPitchDegrees':pitch,'neckBasisRotationDegrees':[0.,0.,0.],
            'headPivotWorld':list(rig.matrix_world@head.head)}


def run(render_cases=DEFAULT_RENDERS,output_relative='previews/reconstruction-r02/neck-poses-r07'):
    """Render selected pose/view pairs, measure all five poses, restore source.

    Example: render_cases=[('neutral','face-quarter'),('look_down','face-side')].
    Existing eye/jaw bases and all face drivers remain active; no shape key is
    zeroed and no action/keyframe or blend file is written. Displacement reports
    movement of the blend region, not a numerical score for skin collapse.
    """
    if bpy is None: raise RuntimeError('Explicit invocation in the existing Blender candidate required')
    cases=[tuple(x) for x in render_cases]
    if not 1<=len(cases)<=4 or len(set(cases))!=len(cases):
        raise ValueError('Choose one to four unique pose/view pairs')
    if any(len(c)!=2 or c[0] not in POSES or c[1] not in VIEWS for c in cases): raise ValueError('Unknown pose or camera view')
    helper=runpy.run_path(str(Path(__file__).with_name('full_body_review_r03.py')))
    rig=helper['_guard']();scene=bpy.context.scene
    if 'R06_head_pivot_calibration' not in rig: raise RuntimeError('Run on the calibrated candidate')
    if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6: raise RuntimeError('Confirmed identity rig object transform required')
    if rig.pose.bones['head'].constraints: raise RuntimeError('Unexpected head constraints require a separate review setup')
    head=bpy.data.objects.get('R01 Head')
    if head is None or head.type!='MESH': raise RuntimeError('Existing skinned R01 Head required')
    if not any(m.type=='ARMATURE' and m.object==rig for m in head.modifiers): raise RuntimeError('Head skin is not bound to the existing rig')
    if any(m.show_viewport!=m.show_render for m in head.modifiers if m.type in DEFORM_ONLY):
        raise RuntimeError('Head deform modifier visibility differs between viewport and render')
    out=(project_path()/output_relative).resolve();out.relative_to((project_path()/'previews').resolve())
    if out.exists() and any(out.iterdir()): raise RuntimeError('Use a new empty review directory; existing images are retained')
    source=Path(bpy.data.filepath);source_sha=_sha_file(source);dirty_before=bool(bpy.data.is_dirty)
    indices=_boundary_vertices(head);drivers_before=_driver_states(rig,head)
    old_action=rig.animation_data.action;old_slot=rig.animation_data.action_slot
    old_pose=rig.data.pose_position;old_basis={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
    master=rig.pose.bones['CTRL_R02_Master'];old_switches={k:float(master['ik_'+k]) for k in ('leg_L','leg_R','arm_L','arm_R')}
    old_cache={key:rig[key] for key in ('R04_native_hair_enabled',) if key in rig}
    settings=[(scene,'camera'),(scene.render,'engine'),(scene.render,'resolution_x'),(scene.render,'resolution_y'),
              (scene.render,'resolution_percentage'),(scene.render,'filepath'),(scene.render,'threads_mode'),
              (scene.render,'threads'),(scene.render,'use_persistent_data'),(scene.render.image_settings,'file_format'),
              (scene.cycles,'device'),(scene.cycles,'samples'),(scene.cycles,'use_denoising')]
    old_settings=[(item,name,getattr(item,name)) for item,name in settings]
    probe=None;camera=None;camera_data=None;report=None
    try:
        helper['_assign'](rig,None);rig.data.pose_position='POSE'
        for pb in rig.pose.bones:
            # Preserve the independent gaze/jaw basis; attached facial objects
            # and shape-key drivers are never disabled or reassigned.
            if 'eye' not in pb.name.lower() and 'jaw' not in pb.name.lower(): pb.matrix_basis=Matrix.Identity(4)
        for key in old_switches: master['ik_'+key]=0.
        for key in old_cache: rig[key]=0.
        rig.update_tag();bpy.context.view_layer.update()
        probe,omitted=_deform_probe(head,scene);bpy.context.view_layer.update()
        _pose(rig,'neutral');neutral=_sample_probe(probe,indices,len(head.data.vertices))
        camera_data=bpy.data.cameras.new('R07 neck pose review camera')
        camera=bpy.data.objects.new('R07 neck pose review camera',camera_data);scene.collection.objects.link(camera)
        camera_data.type='ORTHO';camera_data.dof.use_dof=False;scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
        scene.render.resolution_x=480;scene.render.resolution_y=540;scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG';out.mkdir(parents=True,exist_ok=True)
        measurements=[];renders=[]
        for pose_name in POSES:
            inputs=_pose(rig,pose_name);points=_sample_probe(probe,indices,len(head.data.vertices))
            displacement=np.linalg.norm(points-neutral,axis=1);worst=int(np.argmax(displacement))
            measurements.append({'pose':pose_name,'inputs':inputs,
                'maximumBoundaryDisplacementMeters':float(displacement[worst]),'maximumBoundaryVertexId':indices[worst],
                'meanBoundaryDisplacementMeters':float(displacement.mean())})
            for _,view in [case for case in cases if case[0]==pose_name]:
                position,target,scale=VIEWS[view];camera.location=position
                camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
                camera_data.ortho_scale=scale;filename=f'{pose_name}--{view}.png'
                scene.render.filepath=str(out/filename);bpy.ops.render.render(write_still=True)
                renders.append({'pose':pose_name,'view':view,'path':str(out/filename),
                    'cameraPosition':position,'cameraTarget':target,'orthoScale':scale,'resolution':[480,540],'samples':12})
                print('R07_NECK_POSE_RENDER',pose_name,view,flush=True)
        if _driver_states(rig,head)!=drivers_before: raise RuntimeError('Facial driver mute/identity state changed')
        report={'sourceBlend':str(source),'sourceFileSha256':source_sha,'sourceHadUnsavedChanges':dirty_before,
            'sourceIdentityScope':'SHA identifies the saved file; any unsaved in-memory edits at invocation are separately disclosed.',
            'blenderVersion':bpy.app.version_string,'currentFrame':scene.frame_current+scene.frame_subframe,
            'headRestMatrix':[list(row) for row in rig.data.bones['head'].matrix_local],
            'neckRestMatrix':[list(row) for row in rig.data.bones['neck'].matrix_local],
            'skinObject':head.name,'boundaryCriterion':'Existing head and neck vertex weights both >=0.05',
            'boundaryVertexCount':len(indices),'boundaryVertexIds':indices,
            'boundaryMeasurementScope':'Actual evaluated original cage through retained deform-only modifiers, with current shape keys and drivers; subdivision/bevel omitted only on a temporary probe.',
            'probeOmittedModifiers':omitted,'measurements':measurements,'renders':renders,
            'facialDriverIdentityAndMuteStatesUnchanged':True,'preservedFacialPoseBones':[n for n in old_basis if 'eye' in n.lower() or 'jaw' in n.lower()],
            'lightsUnchanged':True,'savedBlend':False,'accepted':False,
            'reviewScope':'Finite head-only pose stress review of the corrected pivot and neck/head skin blend. Displacement is movement magnitude, not proof of collapse or freedom from intersections.'}
    finally:
        helper['_assign'](rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        rig.data.pose_position=old_pose
        for name,basis in old_basis.items(): rig.pose.bones[name].matrix_basis=basis
        for key,value in old_switches.items(): master['ik_'+key]=value
        for key,value in old_cache.items(): rig[key]=value
        for item,name,value in old_settings: setattr(item,name,value)
        if probe is not None: bpy.data.objects.remove(probe,do_unlink=True)
        if camera is not None: bpy.data.objects.remove(camera,do_unlink=True)
        if camera_data is not None and camera_data.users==0: bpy.data.cameras.remove(camera_data)
        rig.update_tag();bpy.context.view_layer.update()
    if report is not None:
        report['originalPoseActionAndRenderSettingsRestored']=True
        (out/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print('Loaded only. run(render_cases=[(pose, view), ...]) explicitly renders at most four small stills, then restores the scene.')
