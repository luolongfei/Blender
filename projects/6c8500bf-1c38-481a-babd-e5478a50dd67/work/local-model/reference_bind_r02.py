"""Bind new facial construction to existing humanoid and expose eye controls."""
import bpy,math
from mathutils import Vector
def run():
    if 'reconstruction-r02' not in bpy.data.filepath:raise RuntimeError('R02 working copy only')
    rig=bpy.data.objects['Character_Rig'];head=bpy.data.objects['R01 Head']
    if any(m.type=='ARMATURE' for m in head.modifiers):raise RuntimeError('Already bound')
    # Add true gaze pivots at each new eyeball center.
    centers={}
    for side in ['L','R']:
        vs=[v.co for v in bpy.data.objects['R01 Sclera '+side].data.vertices]
        centers[side]=Vector([(min(v[i] for v in vs)+max(v[i] for v in vs))/2 for i in range(3)])
    bpy.ops.object.select_all(action='DESELECT');rig.hide_set(False);rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    for side,bn in [('L','leftEye'),('R','rightEye')]:
        b=rig.data.edit_bones.new(bn);b.head=centers[side];b.tail=b.head+Vector((0,-.018,0));b.parent=rig.data.edit_bones['head'];b.use_deform=True
    bpy.ops.object.mode_set(mode='OBJECT')
    # Neck uses a graded blend rather than head-rigid attachment at the collar.
    for name in ['neck','head','chest']:
        if not head.vertex_groups.get(name):head.vertex_groups.new(name=name)
    for v in head.data.vertices:
        z=v.co.z;head_w=max(0,min(1,(z-1.29)/.055));head_w=head_w*head_w*(3-2*head_w)
        chest_w=max(0,min(.65,(1.27-z)/.045));neck_w=(1-head_w)*(1-chest_w);chest_w*=1-head_w
        for name,w in [('head',head_w),('neck',neck_w),('chest',chest_w)]:
            if w>0:head.vertex_groups[name].add([v.index],w,'REPLACE')
    mod=head.modifiers.new('R02 humanoid deformation','ARMATURE');mod.object=rig;mod.use_deform_preserve_volume=True
    head.parent=rig
    for side,bn in [('L','leftEye'),('R','rightEye')]:
        for prefix in ['R01 Sclera ','R01 Textured iris ']:
            o=bpy.data.objects[prefix+side];world=o.matrix_world.copy();o.parent=rig;o.parent_type='BONE';o.parent_bone=bn;o.matrix_world=world
    for o in bpy.context.scene.objects:
        if o.hide_render:continue
        if o.name.startswith(('R01 Groom','R02 Surface','R02 Curved','R02 Twin','R02 Soft brow','R02 Upper','R02 Tie')):
            world=o.matrix_world.copy();o.parent=rig;o.parent_type='BONE';o.parent_bone='head';o.matrix_world=world
            if o.name.startswith('R02 Twin'):o['dynamic_status']='head parent established; secondary motion pending'
    rig['R02_gaze_note']='leftEye/rightEye rest axes point forward; rotate independently, eyelid follow pending'
    rig.data.pose_position='POSE';bpy.context.view_layer.update()
    print('R02_HEAD_BOUND',len(rig.data.bones))
if __name__=='__main__':run()
