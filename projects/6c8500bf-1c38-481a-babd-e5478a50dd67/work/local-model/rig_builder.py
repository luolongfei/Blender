"""Explicit original humanoid rig and reference-specific skinning.

No image reconstruction, donor meshes, or bone-name guessing is performed here.
"""
import bpy
import math
import json
import importlib.util
from pathlib import Path
from mathutils import Vector, Quaternion

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COL_NAME="CHARACTER · Original geometry"
RIG_NAME="Character_Rig"


def helper(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def smooth(a,b,v):
    t=max(0.,min(1.,(v-a)/(b-a)))
    return t*t*(3-2*t)


def blend(a,b,f):
    return {a:1-f,b:f}


def make_rig():
    col=bpy.data.collections[COL_NAME]
    if bpy.data.objects.get(RIG_NAME):
        raise ValueError("Rig already exists; do not overwrite a posed/edited rig.")
    arm=bpy.data.armatures.new("Original humanoid skeleton")
    rig=bpy.data.objects.new(RIG_NAME,arm);col.objects.link(rig)
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True);bpy.context.view_layer.objects.active=rig
    rig.show_in_front=True
    bpy.ops.object.mode_set(mode="EDIT")
    specs=[]
    def bone(name,a,b,parent=None,connected=False,deform=True):
        ob=arm.edit_bones.new(name);ob.head=a;ob.tail=b;ob.use_deform=deform
        if parent:
            ob.parent=arm.edit_bones[parent];ob.use_connect=connected
        specs.append(name)
        return ob
    bone("hips",(0,.006,.83),(0,.006,.965))
    bone("spine",(0,.006,.965),(0,.006,1.09),"hips",True)
    bone("chest",(0,.006,1.09),(0,.006,1.20),"spine",True)
    bone("upperChest",(0,.006,1.20),(0,.006,1.27),"chest",True)
    bone("neck",(0,.006,1.27),(0,.006,1.335),"upperChest",True)
    bone("head",(0,.006,1.335),(0,.006,1.565),"neck",True)
    for sign,side,label in ((1,"left","L"),(-1,"right","R")):
        bone(side+"Shoulder",(sign*.025,0,1.235),(sign*.185,0,1.235),"upperChest")
        bone(side+"UpperArm",(sign*.185,0,1.235),(sign*.385,0,1.235),side+"Shoulder",True)
        bone(side+"LowerArm",(sign*.385,0,1.235),(sign*.610,0,1.235),side+"UpperArm",True)
        bone(side+"Hand",(sign*.610,0,1.235),(sign*.675,0,1.235),side+"LowerArm",True)
        for idx,(name,y,length) in enumerate([("Index",-.020,.058),("Middle",-.005,.064),("Ring",.010,.058),("Little",.024,.045)]):
            start=.672
            fcoords=[(sign*(start+length*t),y+(idx-1.2)*.003*t,1.235-.003*t*t) for t in (0,.42,.73,1)]
            parent=side+"Hand"
            for j,seg in enumerate(("Proximal","Intermediate","Distal")):
                namebone=side+name+seg
                bone(namebone,fcoords[j],fcoords[j+1],parent,j>0);parent=namebone
        points=[(sign*.633,-.022,1.231),(sign*.647,-.037,1.230),(sign*.661,-.047,1.2285),(sign*.677,-.054,1.228)]
        parent=side+"Hand"
        for j,seg in enumerate(("Metacarpal","Proximal","Distal")):
            namebone=side+"Thumb"+seg
            bone(namebone,points[j],points[j+1],parent,j>0);parent=namebone
        bone(side+"UpperLeg",(sign*.068,.006,.83),(sign*.068,-.004,.46),"hips")
        bone(side+"LowerLeg",(sign*.068,-.004,.46),(sign*.068,.002,.13),side+"UpperLeg",True)
        bone(side+"Foot",(sign*.068,.002,.13),(sign*.068,-.095,.055),side+"LowerLeg",True)
        bone(side+"Toes",(sign*.068,-.095,.055),(sign*.068,-.15,.042),side+"Foot",True)
        hpoints=[(sign*.105,.046,1.383),(sign*.140,.025,1.315),(sign*.159,-.068,1.250),(sign*.135,-.104,1.195)]
        parent="head"
        for j in range(3):
            name="hair_"+label+"_"+str(j+1)
            bone(name,hpoints[j],hpoints[j+1],parent,j>0);parent=name
    for i in range(12):
        t=math.tau*i/12
        points=[(.118*math.cos(t),.080*math.sin(t),.975),
                (.165*math.cos(t),.105*math.sin(t),.850),
                (.210*math.cos(t),.128*math.sin(t),.720),
                (.242*math.cos(t),.145*math.sin(t),.610)]
        parent="hips"
        for j in range(3):
            name=f"skirt_{i:02d}_{j+1}"
            bone(name,points[j],points[j+1],parent,j>0);parent=name
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type="OCTAHEDRAL"
    rig["production_route"]="Original local humanoid, explicit named skin weights"
    rig["neutral_expression"]="fixed neutral face; no expression/LookAt rig claimed"
    return rig


def convert_curves():
    col=bpy.data.collections[COL_NAME]
    converted=[]
    for obj in list(col.objects):
        if obj.type=="CURVE":
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True);bpy.context.view_layer.objects.active=obj
            bpy.ops.object.convert(target="MESH")
            converted.append(obj.name)
    return converted


def spatial_weights(obj,co):
    x,y,z=co
    side="left" if (obj.get("side")=="L" or (not obj.get("side") and x>=0)) else "right"
    label="L" if side=="left" else "R"
    part=obj.get("part","")
    role=obj.get("deform_role","")
    ax=abs(x)
    if part=="head":
        return {"head":1.}
    if part=="neck":
        return blend("upperChest","neck",smooth(1.265,1.30,z))
    if part=="hair":
        if obj.get("hair_region")!="ponytail":
            return {"head":1.}
        if z>1.355:
            return blend("head","hair_"+label+"_1",1-smooth(1.355,1.382,z))
        if z>1.29:
            return blend("hair_"+label+"_2","hair_"+label+"_1",smooth(1.29,1.34,z))
        return blend("hair_"+label+"_3","hair_"+label+"_2",smooth(1.23,1.29,z))
    if part in ("finger","nail"):
        name=obj["finger"]
        if name=="Thumb":
            if part=="nail":
                return {side+"ThumbDistal":1}
            if ax<.650:
                return blend(side+"ThumbMetacarpal",side+"ThumbProximal",smooth(.642,.652,ax))
            return blend(side+"ThumbProximal",side+"ThumbDistal",smooth(.654,.666,ax))
        lengths={"Index":.058,"Middle":.064,"Ring":.058,"Little":.045}
        t=(ax-.672)/lengths[name]
        if part=="nail":
            return {side+name+"Distal":1}
        if t<.55:
            return blend(side+name+"Proximal",side+name+"Intermediate",smooth(.30,.52,t))
        return blend(side+name+"Intermediate",side+name+"Distal",smooth(.62,.84,t))
    if part=="hand":
        return blend(side+"LowerArm",side+"Hand",smooth(.605,.625,ax))
    if part=="leg":
        return blend(side+"LowerLeg",side+"UpperLeg",smooth(.425,.505,z))
    if role in ("shoe",):
        return {side+"Foot":1.}
    if role=="sock":
        return blend(side+"Foot",side+"LowerLeg",smooth(.095,.150,z))
    if role in ("sleeve","cuff"):
        if ax<.205:
            return blend("upperChest",side+"UpperArm",smooth(.12,.205,ax))
        return blend(side+"UpperArm",side+"LowerArm",smooth(.345,.425,ax))
    if role=="skirt":
        theta=math.atan2(y/.14,x/.24)%math.tau
        index=theta/math.tau*12
        i0=int(index)%12;i1=(i0+1)%12;f=index-int(index)
        root=smooth(.90,.97,z)
        result={"hips":root}
        for i,angular in ((i0,1-f),(i1,f)):
            if z>.79:
                w1=smooth(.80,.88,z);ws={1:w1,2:1-w1}
            else:
                w2=smooth(.68,.79,z);ws={2:w2,3:1-w2}
            for segment,w in ws.items():
                result[f"skirt_{i:02d}_{segment}"]=angular*(1-root)*w
        return result
    if role=="waistband":
        return {"hips":1.}
    if role=="blouse" or part=="upper_body":
        if ax>.12 and z>1.18:
            arm=smooth(.12,.20,ax)
            if arm>.01:
                return blend("upperChest",side+"UpperArm",arm)
        if z<1.075:
            return blend("hips","spine",smooth(.985,1.07,z))
        if z<1.18:
            return blend("spine","chest",smooth(1.07,1.18,z))
        return blend("chest","upperChest",smooth(1.18,1.245,z))
    raise ValueError(f"Unclassified mesh {obj.name}: {part}/{role}")


def skin(rig):
    col=bpy.data.collections[COL_NAME]
    for obj in col.objects:
        if obj.type!="MESH":
            continue
        if obj.vertex_groups:
            raise ValueError(f"{obj.name} already has weights; preserve edits.")
        obj.parent=rig
        groups={}
        for vertex in obj.data.vertices:
            weights=spatial_weights(obj,vertex.co)
            weights={n:max(0.,v) for n,v in weights.items() if v>1e-8}
            total=sum(weights.values())
            if total<=0:raise ValueError(obj.name+" missing weights")
            for name,weight in weights.items():
                if name not in rig.data.bones:raise ValueError(name)
                if name not in groups:
                    groups[name]=obj.vertex_groups.new(name=name)
                groups[name].add([vertex.index],weight/total,"REPLACE")
        mod=obj.modifiers.new("Humanoid deformation","ARMATURE")
        mod.object=rig;mod.use_deform_preserve_volume=True
        # Skin prior to subdivision and solidify so fine geometry follows base.
        bpy.context.view_layer.objects.active=obj
        while list(obj.modifiers).index(mod)>0:
            bpy.ops.object.modifier_move_up(modifier=mod.name)
    return [o for o in col.objects if o.type=="MESH"]


def pose(rig,style="relaxed"):
    for b in rig.pose.bones:
        b.rotation_mode="QUATERNION";b.rotation_quaternion=Quaternion()
        b.location=(0,0,0);b.scale=(1,1,1)
    if style=="rest":
        bpy.context.view_layer.update()
        return
    for sign,side in ((1,"left"),(-1,"right")):
        bone=rig.pose.bones[side+"UpperArm"]
        worldrot=Quaternion(Vector((0,1,0)),sign*math.radians(73))
        rest=bone.bone.matrix_local.to_quaternion()
        bone.rotation_quaternion=rest.inverted()@worldrot@rest
        # Slightly soften elbow, with palm remaining close to the thigh.
        lower=rig.pose.bones[side+"LowerArm"]
        q=lower.bone.matrix_local.to_quaternion()
        lower.rotation_quaternion=q.inverted()@Quaternion(Vector((0,0,1)),sign*math.radians(-5))@q
        for finger in ("Index","Middle","Ring","Little"):
            for seg,angle in (("Proximal",7),("Intermediate",9),("Distal",5)):
                b=rig.pose.bones[side+finger+seg]
                rest=b.bone.matrix_local.to_quaternion()
                b.rotation_quaternion=rest.inverted()@Quaternion(Vector((0,1,0)),sign*math.radians(angle))@rest
    bpy.context.view_layer.update()


def setup_springs(rig,meshes):
    api=helper("vrm_helpers")
    collider_report=api.add_colliders(rig,[
        {"name":"Collider · head","bone":"head","center":(0,.015,1.443),"radius":.091},
        {"name":"Collider · chest","bone":"chest","center":(0,.004,1.15),"radius":.115},
        # The 55 mm capsules were smaller than the thigh surface and missed
        # the front-centre skirt particles. 80 mm + 25 mm cloth hit radius
        # covers the interpolated skirt panels in the measured standard walk.
        {"name":"Collider · left thigh","bone":"leftUpperLeg","center":(.068,.006,.82),"tail":(.068,0,.51),"radius":.080},
        {"name":"Collider · right thigh","bone":"rightUpperLeg","center":(-.068,.006,.82),"tail":(-.068,0,.51),"radius":.080}
    ],"Character body")
    chains=[]
    for label in ("L","R"):
        chains.append({"name":"ponytail_"+label,"bones":["hair_"+label+"_"+str(i) for i in (1,2,3)],"stiffness":.8,"gravityPower":.08,"dragForce":.55,"hitRadius":.01,"colliderGroups":["Character body"]})
    for i in range(12):
        chains.append({"name":f"skirt_panel_{i:02d}","bones":[f"skirt_{i:02d}_{j}" for j in (1,2,3)],"stiffness":2.0,"gravityPower":.05,"dragForce":.65,"hitRadius":.025,"colliderGroups":["Character body"]})
    targets=[]
    for obj in meshes:
        if obj.get("part")=="hair":
            if obj.get("hair_region")=="ponytail":
                targets.append({"mesh":obj.name,"source":"hair","mode":"spring","chains":["ponytail_"+obj["side"]]})
            else:
                targets.append({"mesh":obj.name,"source":"hair","mode":"fixed","reason":"Short scalp, bangs, temple strands and hair ties remain attached to the head to preserve the approved silhouette."})
        elif obj.get("deform_role")=="skirt":
            group_names={g.index:g.name for g in obj.vertex_groups}
            used={group_names[g.group] for v in obj.data.vertices for g in v.groups if g.weight>1e-6}
            active=[f"skirt_panel_{i:02d}" for i in range(12) if any(f"skirt_{i:02d}_{j}" in used for j in (1,2,3))]
            if active:
                targets.append({"mesh":obj.name,"source":"skirt","mode":"spring","chains":active})
            else:
                targets.append({"mesh":obj.name,"source":"skirt","mode":"fixed","reason":"This seam is wholly inside the fixed waistband and follows the hips."})
        elif obj.get("deform_role") in ("sleeve","cuff"):
            targets.append({"mesh":obj.name,"source":"sleeves","mode":"fixed","reason":"Tailored puff sleeves deform with upper and lower arm bones; no independent dangling cuff is present."})
    return {"colliders":collider_report,"plan":api.register_springs(rig,chains,targets)}


def build():
    converted=convert_curves()
    rig=make_rig()
    meshes=skin(rig)
    api=helper("vrm_helpers")
    mapping=api.configure_vrm(rig,meshes)
    if mapping["humanoidErrors"]:
        raise RuntimeError(mapping["humanoidErrors"])
    binding=api.audit_bindings(rig,meshes)
    # Curve seams may have no UV. Add simple UVs only for untextured details.
    for obj in meshes:
        if not obj.data.uv_layers:
            uv=obj.data.uv_layers.new(name="UVMap")
            for loop in obj.data.loops:
                v=obj.data.vertices[loop.vertex_index].co
                uv.data[loop.index].uv=(v.x+0.5,v.z)
    springs=setup_springs(rig,meshes)
    report={"convertedCurves":len(converted),"humanoid":mapping,"bindings":binding,"springs":springs}
    (ROOT/"work/local-model/rig-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    pose(rig,"relaxed")
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/"work/character.blend"))
    return {"bones":len(rig.data.bones),"meshes":len(meshes),"humanoidErrors":mapping["humanoidErrors"],
            "unweighted":sum(row["unweightedVertices"] for row in binding),
            "nonNormalized":sum(row["nonNormalizedVertices"] for row in binding)}


def finish_existing():
    rig=bpy.data.objects[RIG_NAME]
    meshes=[o for o in bpy.data.collections[COL_NAME].objects if o.type=="MESH"]
    api=helper("vrm_helpers")
    mapping=api.configure_vrm(rig,meshes)
    for obj in meshes:
        if not obj.data.uv_layers:
            uv=obj.data.uv_layers.new(name="UVMap")
            for loop in obj.data.loops:
                v=obj.data.vertices[loop.vertex_index].co
                uv.data[loop.index].uv=(v.x+0.5,v.z)
    binding=api.audit_bindings(rig,meshes)
    springs=setup_springs(rig,meshes)
    report={"humanoid":mapping,"bindings":binding,"springs":springs}
    (ROOT/"work/local-model/rig-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    pose(rig,"relaxed")
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/"work/character.blend"))
    return {"bones":len(rig.data.bones),"meshes":len(meshes),"humanoidErrors":mapping["humanoidErrors"],
            "unweighted":sum(row["unweightedVertices"] for row in binding),
            "nonNormalized":sum(row["nonNormalizedVertices"] for row in binding)}
