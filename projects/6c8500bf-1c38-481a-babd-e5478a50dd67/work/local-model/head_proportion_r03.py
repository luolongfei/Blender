"""Shorten the exposed neck coherently, retaining every facial shape and attachment."""
import bpy, numpy as np
from mathutils import Vector

def run(drop=.018):
    head=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig']
    if head.get('R03_head_lowering'):raise RuntimeError('Already proportioned')
    scene=bpy.context.scene;frame=scene.frame_current;pose=rig.data.pose_position
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    objects=[head]+[o for o in scene.objects if not o.hide_render and o.parent==rig and o.parent_type=='BONE' and o.type in {'MESH','CURVE','CURVES'} and (o.parent_bone=='head' or o.parent_bone in {'leftEye','rightEye','R02_Jaw'})]
    matrices={o.name:o.matrix_world.copy() for o in objects}
    def transform(data,o,prop='co'):
        arr=np.zeros(len(data)*3,dtype=np.float32);data.foreach_get(prop,arr);arr=arr.reshape(-1,3)
        mw=np.array(matrices[o.name]);inv=np.linalg.inv(mw)
        world=arr@mw[:3,:3].T+mw[:3,3]
        if o==head:
            t=np.clip((world[:,2]-1.26)/.09,0,1);world[:,2]-=drop*t*t*(3-2*t)
        else:world[:,2]-=drop
        local=(world-mw[:3,3])@inv[:3,:3].T
        data.foreach_set(prop,np.asarray(local,dtype=np.float32).reshape(-1))
    for o in objects:
        if o.type=='CURVES':transform(o.data.position_data,o,'vector')
        elif o.data.shape_keys:
            for key in o.data.shape_keys.key_blocks:transform(key.data,o)
        elif o.type=='MESH':transform(o.data.vertices,o)
        else:
            # Legacy polynomial curves retain their homogeneous weight.
            for spline in o.data.splines:
                for pt in spline.points:
                    q=matrices[o.name]@Vector(pt.co[:3]);q.z-=drop;q=matrices[o.name].inverted()@q;pt.co=(*q,pt.co.w)
        o.data.update_tag()
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for bone in rig.data.edit_bones:
        ancestor=bone;ishead=False
        while ancestor:
            if ancestor.name=='head':ishead=True;break
            ancestor=ancestor.parent
        if bone.name=='neck':
            for attr in ['head','tail']:
                q=getattr(bone,attr).copy();t=max(0,min(1,(q.z-1.26)/.09));q.z-=drop*t*t*(3-2*t);setattr(bone,attr,q)
        elif ishead:
            for attr in ['head','tail']:
                q=getattr(bone,attr).copy();t=max(0,min(1,(q.z-1.26)/.09)) if bone.name=='head' else 1.;q.z-=drop*t*t*(3-2*t);setattr(bone,attr,q)
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
    for o in objects:o.matrix_world=matrices[o.name]
    rig.data.pose_position=pose;scene.frame_set(frame);bpy.context.view_layer.update()
    head['R03_head_lowering']=drop
    print('R03_NECK_PROPORTION',len(objects),drop)

if __name__=='__main__':run()
