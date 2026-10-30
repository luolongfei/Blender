"""Editable jaw and basic facial action units. Requires visual combination QA."""
import bpy,math,json,sys
from pathlib import Path
from mathutils import Vector,Matrix
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run():
    p=project_path();h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];ks=h.data.shape_keys.key_blocks
    if ks.get('R02_jaw_open'):raise RuntimeError('Already added')
    co=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        co.append(q)
    data=json.loads((p/'work/reconstruction-r02/lip-landmarks.json').read_text())
    up=data['upper_R_to_L'];low=data['lower_R_to_L'];inner=set(data['oral_interior_face_ids'])
    mucosa=bpy.data.materials.new('R02 oral mucosa');mucosa.use_nodes=True
    b=next(n for n in mucosa.node_tree.nodes if n.type=='BSDF_PRINCIPLED');b.inputs['Base Color'].default_value=(.065,.012,.015,1);b.inputs['Roughness'].default_value=.5;b.inputs['Subsurface Weight'].default_value=.15;b.inputs['Subsurface Scale'].default_value=.0015
    mi=len(h.data.materials);h.data.materials.append(mucosa)
    for f in h.data.polygons:
        if f.index in inner:f.material_index=mi
    pivot=Vector((0,.017,1.413));rot=Matrix.Rotation(.22,3,'X')
    jaw=h.shape_key_add(name='R02_jaw_open',from_mix=False)
    for i,q in enumerate(co):
        x,y,z=q
        lower=max(0,min(1,(1.392-z)/.018));lower=lower*lower*(3-2*lower)
        neck=max(0,min(1,(z-1.315)/.04));front=max(0,min(1,(-y+.01)/.045))
        weight=lower*neck*front
        if i in low[1:-1]:weight=1
        if i in up[1:-1]:weight=0
        jaw.data[i].co+=(pivot+rot@(q-pivot)-q)*weight
    jaw.value=0
    def action(name,fun):
        k=h.shape_key_add(name=name,from_mix=False)
        for i,q in enumerate(co):k.data[i].co+=fun(q)
        k.value=0;return k
    def mouthmask(q):
        x,y,z=q
        return math.exp(-((z-1.391)/.019)**4-(abs(x)/.039)**6)*max(0,min(1,(-y-.05)/.025))
    action('R02_mouth_wide',lambda q:Vector((q.x*.20,0,0))*mouthmask(q))
    action('R02_pucker',lambda q:Vector((-q.x*.30,-.006,0))*mouthmask(q))
    action('R02_lip_press',lambda q:Vector((0,.001,(1.3892-q.z)*.3))*mouthmask(q))
    def smile_delta(q):
        x,y,z=q
        a=math.exp(-((abs(x)-.019)/.011)**2-((z-1.39)/.016)**2)*max(0,min(1,(-y-.05)/.025))
        return Vector((math.copysign(.0015,x),-.0008,.004))*a
    smile=action('R02_smile',smile_delta);smile.value=.45
    action('R02_frown',lambda q:Vector((0,0,-.004))*math.exp(-((abs(q.x)-.019)/.012)**2-((q.z-1.39)/.018)**2)*max(0,min(1,(-q.y-.05)/.025)))
    action('R02_brow_raise',lambda q:Vector((0,0,.005))*math.exp(-((q.z-1.471)/.018)**2-((abs(q.x)-.032)/.024)**4)*max(0,min(1,(-q.y-.04)/.03)))
    action('R02_brow_furrow',lambda q:Vector((-math.copysign(.002,q.x),-.001,-.003))*math.exp(-((q.z-1.467)/.015)**2-((abs(q.x)-.022)/.021)**4)*max(0,min(1,(-q.y-.04)/.03)))
    # A real jaw pivot also moves the separate lower dental arch and tongue.
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    bone=rig.data.edit_bones.new('R02_Jaw');bone.head=pivot;bone.tail=(0,-.075,1.39);bone.parent=rig.data.edit_bones['head'];bone.use_deform=True
    bpy.ops.object.mode_set(mode='OBJECT');pb=rig.pose.bones['R02_Jaw'];pb.rotation_mode='XYZ'
    sign=1 if rig.data.bones['R02_Jaw'].x_axis.x>0 else -1
    drv=pb.driver_add('rotation_euler',0).driver;drv.expression=f'{sign*.22} * opening';var=drv.variables.new();var.name='opening';var.targets[0].id_type='KEY';var.targets[0].id=h.data.shape_keys;var.targets[0].data_path='key_blocks["R02_jaw_open"].value'
    for source,name,parent,zshift in [('Upper dental arch','R02 Upper teeth','head',.022),('Lower dental arch','R02 Lower teeth','R02_Jaw',.025),('Tongue surface','R02 Tongue','R02_Jaw',.017)]:
        old=bpy.data.objects[source];o=old.copy();o.data=old.data.copy();o.name=name;bpy.context.scene.collection.objects.link(o)
        if o.data.shape_keys:o.shape_key_clear()
        o.modifiers.clear();o.animation_data_clear();o.parent=None;o.matrix_world=Matrix.Identity(4);o.hide_render=False;o.hide_set(False)
        for v in o.data.vertices:v.co.x*=.83;v.co.y+=.014;v.co.z+=zshift
        world=o.matrix_world.copy();o.parent=rig;o.parent_type='BONE';o.parent_bone=parent;o.matrix_world=world
    # Region-scaled physical skin response, with restrained micro relief.
    mat=h.data.materials[0];nt=mat.node_tree;bs=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Subsurface Scale'].default_value=.0022;bs.inputs['Subsurface Weight'].default_value=.16;bs.inputs['Roughness'].default_value=.49
    tex=nt.nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=1100;tex.inputs['Detail'].default_value=2
    bump=nt.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00008
    nt.links.new(tex.outputs['Fac'],bump.inputs['Height']);nt.links.new(bump.outputs['Normal'],bs.inputs['Normal'])
    rig['R02_face_status']='Basic editable face controls; combinations/visemes not accepted until render checks'
    bpy.context.view_layer.update();print('R02_PERFORMANCE_CREATED')
if __name__=='__main__':run()
