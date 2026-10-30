"""Bake verified raw-bind deformation matrices to an editable Blender Action.

This imports project-authored procedural choreography, NOT mocap. It never saves
the Blend. Run only in a working copy after binding the new head/eyes/groom.
"""
import bpy
import json
import math
import hashlib
import sys
from pathlib import Path
from mathutils import Matrix

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def matrix_column_major(values):
    if len(values)!=16 or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected 16 finite matrix values')
    return Matrix([values[i:i+4] for i in range(0,16,4)]).transposed()

def difference(a,b):
    return max(abs(a[i][j]-b[i][j]) for i in range(4) for j in range(4))

def bake(relative_json, action_name=None):
    P=project_path()
    current=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or current==(P/'work/character.blend').resolve():
        raise RuntimeError('Use a saved independent working copy, never the master')
    try: current.relative_to((P/'work').resolve())
    except ValueError: raise RuntimeError('Working copy must be inside this project work directory')
    rel=Path(relative_json)
    if rel.is_absolute() or '..' in rel.parts: raise ValueError('Use a project-relative JSON path')
    source=(P/rel).resolve()
    data=json.loads(source.read_text(encoding='utf-8'))
    if data.get('version')!=3 or not data.get('sourceUnchanged'):
        raise RuntimeError('Expected version 3 reconstruction_motion_sample output')
    if data.get('physics') is not False: raise RuntimeError('Body-only source required')
    vrm=P/data['input']
    if hashlib.sha256(vrm.read_bytes()).hexdigest()!=data['vrmSha256']:
        raise RuntimeError('Motion source VRM fingerprint changed')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.type!='ARMATURE': raise RuntimeError('Character_Rig required')
    if difference(rig.matrix_world,Matrix.Identity(4))>1e-7:
        raise RuntimeError('Rig transform is not identity; explicit conversion required')
    if any(pb.constraints for pb in rig.pose.bones):
        raise RuntimeError('Active constraints require an explicit control-rig bake route')
    if rig.animation_data and (rig.animation_data.drivers or rig.animation_data.nla_tracks):
        raise RuntimeError('Drivers/NLA present; preserve them and use an explicit integration route')
    names={row['name'] for row in data['poses'][0]['bones']}
    if not names.issubset(rig.pose.bones.keys()):
        raise RuntimeError('Missing target bones: '+str(sorted(names-set(rig.pose.bones.keys()))))
    if any({r['name'] for r in pose['bones']}!=names for pose in data['poses']):
        raise RuntimeError('Inconsistent bone rows')
    ordered=sorted((rig.pose.bones[n] for n in names),key=lambda pb:len(pb.parent_recursive))
    previous_action=rig.animation_data.action if rig.animation_data else None
    previous_basis={pb.name:pb.matrix_basis.copy() for pb in rig.pose.bones}
    previous_modes={pb.name:pb.rotation_mode for pb in rig.pose.bones}
    scene=bpy.context.scene
    previous_frame=scene.frame_current;previous_pose=rig.data.pose_position
    previous_range=(scene.frame_start,scene.frame_end,scene.render.fps,scene.render.fps_base)
    preferences=bpy.context.preferences.edit
    interpolation=preferences.keyframe_new_interpolation_type
    identifiers={e.identifier for e in preferences.bl_rna.properties['keyframe_new_interpolation_type'].enum_items}
    if 'LINEAR' not in identifiers: raise RuntimeError('Linear keyframe interpolation unavailable')
    action=bpy.data.actions.new(action_name or 'R02 Authored '+data['mode'])
    action.use_fake_user=True
    rig.animation_data_create();rig.animation_data.action=action
    scene.render.fps=data['fps'];scene.render.fps_base=1
    rig.data.pose_position='POSE'
    max_matrix_error=0.0;last_quaternions={}
    try:
        preferences.keyframe_new_interpolation_type='LINEAR'
        for pose in data['poses']:
            frame=1+pose['frame'];scene.frame_set(frame)
            rows={r['name']:matrix_column_major(r['restToPosedBlender']) for r in pose['bones']}
            targets={n:rows[n]@rig.data.bones[n].matrix_local for n in names}
            for pb in ordered:
                pb.rotation_mode='QUATERNION'
                parent_args={}
                if pb.parent:
                    parent_args={'parent_matrix':targets.get(pb.parent.name,pb.parent.matrix),
                                 'parent_matrix_local':pb.parent.bone.matrix_local}
                # Compute the local basis from desired parent matrices directly;
                # this does not depend on per-bone dependency-graph evaluation order.
                pb.matrix_basis=pb.bone.convert_local_to_pose(
                    targets[pb.name],pb.bone.matrix_local,invert=True,**parent_args)
                # Make quaternion signs continuous for interpolation.
                q=pb.rotation_quaternion
                if pb.name in last_quaternions and q.dot(last_quaternions[pb.name])<0:q.negate()
                last_quaternions[pb.name]=q.copy()
                for prop in ('location','rotation_quaternion','scale'):
                    pb.keyframe_insert(prop,frame=frame,group=pb.name)
            bpy.context.view_layer.update()
            max_matrix_error=max(max_matrix_error,max(difference(rig.pose.bones[n].matrix,targets[n]) for n in names))
        scene.frame_start=1;scene.frame_end=1+data['poses'][-1]['frame']
        scene.frame_set(scene.frame_start)
        if max_matrix_error>2e-4:
            raise RuntimeError(f'Pose replay mismatch: {max_matrix_error}')
        action['provenance']=data['provenance'];action['source_json']=str(rel.as_posix())
        action['source_json_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        action['verified_max_matrix_error']=max_matrix_error
        report={'action':action.name,'frames':len(data['poses']),'fps':data['fps'],
                'maxMatrixElementError':max_matrix_error,'previousAction':previous_action.name if previous_action else None,
                'savedBlend':False,'mocap':False,
                'limitations':['No face expression keys imported','No new groom or cloth simulation',
                               'Matrix replay checks do not establish visual/contact quality']}
        print(json.dumps(report,ensure_ascii=False))
        return report
    except Exception:
        rig.animation_data.action=previous_action
        bpy.data.actions.remove(action)
        rig.data.pose_position=previous_pose
        scene.frame_start,scene.frame_end,scene.render.fps,scene.render.fps_base=previous_range
        scene.frame_set(previous_frame)
        for pb in rig.pose.bones:
            pb.rotation_mode=previous_modes[pb.name];pb.matrix_basis=previous_basis[pb.name]
        bpy.context.view_layer.update()
        raise
    finally:
        preferences.keyframe_new_interpolation_type=interpolation

if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    if not args: raise RuntimeError('Pass project-relative motion JSON after --')
    bake(args[0],args[1] if len(args)>1 else None)
