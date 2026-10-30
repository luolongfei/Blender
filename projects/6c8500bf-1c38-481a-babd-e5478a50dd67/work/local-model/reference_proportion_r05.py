"""Coherent head enlargement and neck shortening in a saved work copy.

Transforms every face key, attached eye/oral/hair component and head rest bone.
The central neck mask never touches the arms. Existing tail dynamics must first
be restored/disabled and regenerated for the changed rest attachment/groom.
"""
import bpy,json,sys
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run(scale=1.10,drop=.015):
    p=project_path();source=Path(bpy.data.filepath).resolve()
    source.relative_to((p/'work').resolve())
    if source==(p/'work/character.blend').resolve():raise RuntimeError('Independent saved copy required')
    h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    if h.get('R05_reference_proportion'):raise RuntimeError('Already applied')
    if 'R04_native_hair_guide' in rig:raise RuntimeError('Restore old native tail cache first; this pass changes geometry and attachment')
    pose=rig.data.pose_position;frame=scene.frame_current;rig.data.pose_position='REST';bpy.context.view_layer.update()
    objects=[h]+[o for o in scene.objects if not o.hide_render and o.parent==rig and o.parent_type=='BONE'
        and o.type in {'MESH','CURVE','CURVES'} and o.parent_bone in {'head','leftEye','rightEye','R02_Jaw'}]
    matrices={o.name:o.matrix_world.copy() for o in objects};center=np.array([0,.003,1.423])
    def mask(world):
        t=np.clip((world[:,2]-1.245)/.065,0,1);w=t*t*(3-2*t)
        x=np.clip((.14-np.abs(world[:,0]))/.04,0,1)
        return w*x*x*(3-2*x)
    def change(world,w):
        delta=(world-center)*(scale-1);delta[:,2]-=drop
        return world+delta*w[:,None]
    def read(data,prop):
        a=np.empty(len(data)*3,np.float32);data.foreach_get(prop,a);return a.reshape(-1,3)
    changes={}
    for o in objects:
        mw=np.array(matrices[o.name]);inv=np.linalg.inv(mw)
        if o.type=='CURVES':blocks=[(o.data.position_data,'vector')]
        elif o.data.shape_keys:blocks=[(k.data,'co') for k in o.data.shape_keys.key_blocks]
        elif o.type=='MESH':blocks=[(o.data.vertices,'co')]
        else:
            blocks=[]
            for spline in o.data.splines:
                for pt in spline.bezier_points:
                    for prop in ('co','handle_left','handle_right'):
                        q=np.array(matrices[o.name]@getattr(pt,prop));q=change(q[None,:],np.ones(1))[0]
                        local=(q-mw[:3,3])@inv[:3,:3].T;setattr(pt,prop,local)
                for pt in spline.points:
                    q=np.array(matrices[o.name]@Vector(pt.co[:3]));q=change(q[None,:],np.ones(1))[0]
                    local=(q-mw[:3,3])@inv[:3,:3].T;pt.co=(*local,pt.co.w)
        weight=None;maxdelta=0.
        for data,prop in blocks:
            a=read(data,prop);world=a@mw[:3,:3].T+mw[:3,3]
            if weight is None:weight=mask(world) if o==h else np.ones(len(a))
            altered=change(world,weight);maxdelta=max(maxdelta,float(np.linalg.norm(altered-world,axis=1).max()))
            local=(altered-mw[:3,3])@inv[:3,:3].T
            data.foreach_set(prop,np.ascontiguousarray(local,dtype=np.float32).ravel())
        o.data.update_tag();changes[o.name]=maxdelta
    rig_hidden=rig.hide_get();rig.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in rig.data.edit_bones:
        ancestor=bone;head_chain=False
        while ancestor:
            if ancestor.name=='head':head_chain=True;break
            ancestor=ancestor.parent
        if bone.name=='neck' or head_chain:
            for attr in ('head','tail'):
                q=np.array(getattr(bone,attr));weight=mask(q[None,:]) if bone.name in {'neck','head'} else np.ones(1)
                setattr(bone,attr,change(q[None,:],weight)[0])
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
    for o in objects:o.matrix_world=matrices[o.name]
    rig.hide_set(rig_hidden);rig.data.pose_position=pose;scene.frame_set(frame);bpy.context.view_layer.update()
    h['R05_reference_proportion']=json.dumps({'scale':scale,'drop':drop,'center':center.tolist()})
    report={'source':str(source),'scale':scale,'dropMeters':drop,'changedObjects':changes,
        'centralNeckMask':{'z':[1.245,1.31],'xFull':.10,'xZero':.14},
        'validation':'Requires new visual face/side/blink checks and regenerated tail dynamics; previous reports are historical'}
    (p/'work/reconstruction-r02/reference-proportion-r05.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));return report
