"""Recess globes and the matching eyelid envelope together, preserving blink deltas."""
import bpy,math
from mathutils import Vector

def run(depth=.003):
    h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];ks=h.data.shape_keys.key_blocks
    if ks.get('R04_orbital_depth'):raise RuntimeError('Already recessed')
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    centers={};worlds={}
    for side in ['L','R']:
        obj=bpy.data.objects['R01 Sclera '+side];co=[obj.matrix_world@v.co for v in obj.data.vertices]
        centers[side]=Vector([(min(q[i] for q in co)+max(q[i] for q in co))/2 for i in range(3)])
        for prefix in ['R01 Sclera ','R01 Textured iris ']:
            o=bpy.data.objects[prefix+side];worlds[o.name]=o.matrix_world.copy()
            delta=o.matrix_world.inverted().to_3x3()@Vector((0,depth,0))
            for v in o.data.vertices:v.co+=delta
    neutral=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        neutral.append(q)
    key=h.shape_key_add(name='R04_orbital_depth',from_mix=False)
    for i,q in enumerate(neutral):
        weight=0.
        for c in centers.values():
            dx=max(0,abs(q.x-c.x)-.022);dz=max(0,abs(q.z-c.z)-.017)
            weight=max(weight,math.exp(-(dx/.009)**2-(dz/.008)**2))
        weight*=max(0,min(1,(-q.y-.025)/.025))
        key.data[i].co.y+=depth*weight
    key.value=1.
    for o in bpy.data.objects:
        if o.hide_render or not o.name.startswith(('R02 Upper lashes','R02 Upper lid rim','R02 Soft brow')):continue
        delta=o.matrix_world.inverted().to_3x3()@Vector((0,depth if 'brow' not in o.name else depth*.55,0))
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks:
                for v in k.data:v.co+=delta
        else:
            for spline in o.data.splines:
                for v in spline.points:v.co.x+=delta.x;v.co.y+=delta.y;v.co.z+=delta.z
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for side,name in [('L','leftEye'),('R','rightEye')]:
        b=rig.data.edit_bones[name];axis=b.tail-b.head;b.head=rig.matrix_world.inverted()@(centers[side]+Vector((0,depth,0)));b.tail=b.head+axis
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
    for name,matrix in worlds.items():bpy.data.objects[name].matrix_world=matrix
    rig.data.pose_position=pose;bpy.context.view_layer.update();print('R04_ORBITS_RECESSED',depth)

if __name__=='__main__':run()
