"""Separate, reversible interactive rig for the delivered E8 static character."""
import bpy, sys, json, math, hashlib
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
OUT=workspace_path('Character_Rebuild/interactive')
WORK=OUT/'work';WORK.mkdir(parents=True,exist_ok=True)
SOURCE=workspace_path('Character_Rebuild/character-rebuilt.blend')
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
model=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.name.startswith('Reference_') and not o.hide_render]
assert len(model)==47, 'Expected current E8 geometry'
assert not any(o.type=='ARMATURE' for o in bpy.context.scene.objects)

def point_signature(o):
    import struct
    h=hashlib.sha256()
    for v in o.data.vertices:h.update(struct.pack('<3f',*v.co))
    for uv in o.data.uv_layers:
        h.update(uv.name.encode())
        for d in uv.data:h.update(struct.pack('<2f',*d.uv))
    return h.hexdigest()
before={o.name:point_signature(o) for o in model}
armdata=bpy.data.armatures.new('PreviewHumanoid_Skeleton')
arm=bpy.data.objects.new('PreviewHumanoid',armdata)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active=arm
arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bones={}
def bone(name,head,tail,parent=None,deform=True):
    b=armdata.edit_bones.new(name);b.head=head;b.tail=tail;b.roll=0;b.use_deform=deform
    if parent:b.parent=bones[parent]
    bones[name]=b
bone('Root',(0,0,-.999),(0,0,-.90),deform=False)
bone('Hips',(0,.02,.16),(0,.02,.28),'Root')
bone('Spine',(0,.02,.28),(0,.02,.42),'Hips')
bone('Chest',(0,.02,.42),(0,.02,.56),'Spine')
bone('Neck',(0,.02,.56),(0,.02,.69),'Chest')
bone('Head',(0,.02,.69),(0,.02,.91),'Neck')
for side,s in [('Left',1),('Right',-1)]:
    bone(side+'Shoulder',(s*.05,.025,.55),(s*.19,.025,.55),'Chest')
    bone(side+'UpperArm',(s*.19,.025,.55),(s*.47,.040,.539),side+'Shoulder')
    bone(side+'LowerArm',(s*.47,.040,.539),(s*.746,.0365,.5475),side+'UpperArm')
    bone(side+'Hand',(s*.746,.0365,.5475),(s*.91,.0365,.5475),side+'LowerArm')
    bone(side+'UpperLeg',(s*.10,.02,.16),(s*.09,.027,-.43),'Hips')
    bone(side+'LowerLeg',(s*.09,.027,-.43),(s*.085,.033,-.85),side+'UpperLeg')
    bone(side+'Foot',(s*.085,.033,-.85),(s*.10,-.14,-.95),side+'LowerLeg')
    bone(side+'Toes',(s*.10,-.14,-.95),(s*.10,-.20,-.95),side+'Foot')
bpy.ops.object.mode_set(mode='OBJECT')
arm.show_in_front=True;armdata.display_type='STICK'
arm['purpose']='Basic bounded motion preview; separate from static delivery; no cloth simulation or facial animation'
for p in arm.pose.bones:p.rotation_mode='QUATERNION'

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)));return t*t*(3-2*t)
def mix(a,b,t):
    d={k:v*(1-t) for k,v in a.items()}
    for k,v in b.items():d[k]=d.get(k,0)+v*t
    return d
def torso(z):
    if z<.27:return {'Hips':1.}
    if z<.37:return mix({'Hips':1},{'Spine':1},smooth(.27,.37,z))
    return mix({'Spine':1},{'Chest':1},smooth(.37,.49,z))
def sleeve(x,z):
    side='Left' if x>=0 else 'Right';a=abs(x)
    if a<.23:
        return mix({'Chest':1},{side+'UpperArm':1},smooth(.135,.265,a))
    return mix({side+'UpperArm':1},{side+'LowerArm':1},smooth(.40,.54,a))
def body_weights(p):
    x,y,z=p;side='Left' if x>=0 else 'Right';a=abs(x)
    if z>.30:
        t=smooth(.135,.235,a)*smooth(.34,.44,z)
        return mix(torso(z),sleeve(x,z),t)
    # The skirt remains with the pelvis. Below the hem, legs blend into it.
    # No left/right skirt split and no heat transfer across adjacent surfaces.
    if z>-.20:return {'Hips':1.}
    leg=mix({side+'UpperLeg':1},{side+'LowerLeg':1},1-smooth(-.50,-.36,z))
    # Whole shoes follow foot; no toe roll in these conservative preview clips.
    leg=mix(leg,{side+'Foot':1},1-smooth(-.895,-.80,z))
    return mix({'Hips':1},leg,1-smooth(-.265,-.20,z))
def weights(o,p):
    n=o.name
    if n=='Rebuild_Body_Clothing_SourceRetained':return body_weights(p)
    if n.startswith('Rebuilt_Hand_'):
        return {('Left' if n.endswith('_L') else 'Right')+'Hand':1.}
    if n.startswith('Cuff_Inner_'):
        return {('Left' if n.endswith('_L') else 'Right')+'LowerArm':1.}
    if n.startswith('Forearm_UnderSleeve_'):
        side='Left' if p.x>=0 else 'Right'
        return mix({side+'LowerArm':1},{side+'Hand':1},smooth(.72,.75,abs(p.x)))
    if n=='Neck_Clean_Quad_Sleeve':
        z=p.z
        if z<.645:return mix({'Chest':1},{'Neck':1},smooth(.599,.645,z))
        return mix({'Neck':1},{'Head':1},smooth(.645,.695,z))
    if 'Collar' in n or n.startswith('Blouse_Button_'):return torso(p.z)
    return {'Head':1.}

stats=[]
for o in model:
    for vg in list(o.vertex_groups):o.vertex_groups.remove(vg)
    groups={name:o.vertex_groups.new(name=name) for name in bones if name!='Root'}
    hist={};max_count=0;max_err=0
    for v in o.data.vertices:
        p=o.matrix_world@v.co
        d={k:w for k,w in weights(o,p).items() if w>1e-7}
        total=sum(d.values());assert total>0 and all(math.isfinite(w) for w in d.values())
        d={k:w/total for k,w in d.items()}
        assert len(d)<=4
        for name,w in d.items():groups[name].add([v.index],w,'REPLACE');hist[name]=hist.get(name,0)+1
        max_count=max(max_count,len(d));max_err=max(max_err,abs(sum(d.values())-1))
    for vg in list(o.vertex_groups):
        if vg.name not in hist:o.vertex_groups.remove(vg)
    matrix=o.matrix_world.copy();o.parent=arm;o.matrix_world=matrix
    mod=o.modifiers.new('Preview_Skin_Deformation','ARMATURE');mod.object=arm;mod.use_deform_preserve_volume=False
    stats.append({'object':o.name,'vertices':len(o.data.vertices),'maxInfluences':max_count,'maxWeightSumError':max_err,'weightedBones':hist})
assert all(point_signature(o)==before[o.name] for o in model), 'Rig operation changed geometry or UV'
bpy.context.view_layer.update()
# Measure bind/rest invariance against original vertex positions after existing modifiers.
rest_test={}
for o in model:
    skin=o.modifiers.get('Preview_Skin_Deformation')
    deps=bpy.context.evaluated_depsgraph_get()
    ev=o.evaluated_get(deps);mesh=ev.to_mesh();with_skin=[(ev.matrix_world@v.co).copy() for v in mesh.vertices];ev.to_mesh_clear()
    skin.show_viewport=False;bpy.context.view_layer.update()
    ev=o.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();without_skin=[(ev.matrix_world@v.co).copy() for v in mesh.vertices];ev.to_mesh_clear()
    skin.show_viewport=True;bpy.context.view_layer.update()
    assert len(with_skin)==len(without_skin)
    delta=max(((a-b).length for a,b in zip(with_skin,without_skin)),default=0)
    assert delta<2e-6,(o.name,delta)
    rest_test[o.name]=delta
contract={'armature':'PreviewHumanoid','coordinateSystem':'Blender world: X lateral, -Y front, Z up; identity armature','groundZ':-.9990763664,'bones':[{'name':b.name,'parent':b.parent.name if b.parent else None,'head':list(b.head_local),'tail':list(b.tail_local),'matrixLocal':[list(r) for r in b.matrix_local],'deform':b.use_deform} for b in armdata.bones]}
(WORK/'rig-contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
report={'source':str(SOURCE),'sourceSha256':source_sha,'meshCount':len(model),'boneCount':len(armdata.bones),'geometryUvUnchanged':True,'restMaximumDisplacement':max(rest_test.values()),'restDisplacements':rest_test,'weights':stats,'limitations':['Analytic weights for a limited motion preview, not a production rig.','Skirt follows pelvis; no cloth collision/simulation.','Head, eyes and hair follow the head rigidly; no facial or hair dynamics.','Hands preserve their five-finger geometry and follow hand bones; fingers have no independent controls.','Numerical model scale retained; physical human height is not calibrated.']}
(WORK/'rig-build-report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
arm['source_static_sha256']=source_sha
bpy.context.scene['interactive_preview_scope']='E8 geometry plus basic humanoid skin, limited motions; static delivery preserved'
for o in bpy.context.scene.objects:o.select_set(False)
arm.select_set(True);bpy.context.view_layer.objects.active=arm
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(WORK/'rigged-rest.blend'))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_sha
print('RIGGED_REST_READY',json.dumps({'meshCount':len(model),'bones':len(armdata.bones),'maxRestDisplacement':max(rest_test.values())}))
