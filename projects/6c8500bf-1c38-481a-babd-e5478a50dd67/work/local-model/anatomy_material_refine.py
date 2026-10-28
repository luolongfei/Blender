"""Authored proportions and a neutral, portable skin/material palette.

Apply to the retained comprehensive-refinement-start source, after the current
hair patch. Every facial shape receives the same continuous anatomical field.
This module neither exports nor overwrites the source Blend.
"""
from pathlib import Path
import math
import json
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

P = Path(__file__).resolve().parents[2]
VERSION = "anatomy-material-03"
FACE = "Face · sculpted planes & original illustration"
COL = "CHARACTER · Original geometry"
EYE_Z = 1.4395
BACK_CURVE = []


def smooth(a, b, x):
    t = min(1., max(0., (x-a)/(b-a)))
    return t*t*(3-2*t)


def gauss(x, z, cx, cz, rx, rz):
    return math.exp(-((x-cx)/rx)**2-((z-cz)/rz)**2)


def skull(co):
    x, y, z = co
    x *= 1-.14*smooth(1.365, 1.425, z)
    # Restore the high occipital curve and remove the previous low posterior
    # bulge, exposed when the hair was hidden for the anatomy review.
    if BACK_CURVE and 1.34 < z < 1.552:
        old=float(np.interp(z,[a[0] for a in BACK_CURVE],[a[1] for a in BACK_CURVE]))
        oval=.091*math.sqrt(max(0.,1-((z-1.454)/.102)**2))
        lower=float(np.interp(z,[1.34,1.36,1.38,1.40,1.42],
                              [.016,.033,.052,.070,.085]))
        blend=smooth(1.39,1.425,z)
        target=lower*(1-blend)+oval*blend
        y+=(target-old)*smooth(.002,.050,y)*smooth(1.34,1.355,z)
    return Vector((x, y, z))


def face_field(co, role):
    x, y, z = co
    front = 1-smooth(-.035, .01, y)
    if not front:
        return skull(co)
    eye = (1-smooth(.72, 1.38, math.sqrt(
        ((abs(x)-.0365)/.023)**2+((z-EYE_Z)/.023)**2)))*front
    # Less exposed white below the iris and a softer, almond-shaped lid.
    # The iris retains its circular outline, behind the actual eyelid rim.
    if role != "eye_iris":
        z += ((EYE_Z+(z-EYE_Z)*.74)-z)*eye
    else:
        z = EYE_Z+.0005+(z-EYE_Z)*.86
    # Small orbital recess, with continuous brow and cheek support.
    y += .0016*eye
    for side in (-1, 1):
        y -= .0015*gauss(x,z,side*.037,1.459,.023,.007)*front
        y -= .0014*gauss(x,z,side*.051,1.407,.022,.014)*front
        y += .0008*gauss(x,z,side*.031,1.425,.022,.005)*front
    mouth = gauss(x,z,0,1.367,.032,.016)*front
    x += .11*x*mouth
    # A rounded nasal tip, paired alar volumes, and recessed nostril floors.
    y -= .0048*gauss(x,z,0,1.396,.010,.008)*front
    y -= .0018*gauss(x,z,0,1.416,.011,.018)*front
    for side in (-1, 1):
        y -= .0026*gauss(x,z,side*.0115,1.388,.005,.0045)*front
        y += .0024*gauss(x,z,side*.0086,1.3845,.0028,.002)*front
        x += side*.0016*gauss(x,z,side*.010,1.388,.006,.006)*front
    y += .0011*gauss(x,z,0,1.380,.002,.005)*front
    # Bring chin/labial transition into a gradual profile; keep jaw rounded.
    y -= .0021*gauss(x,z,0,1.343,.028,.011)*front
    y += .0011*gauss(x,z,0,1.354,.025,.004)*front
    z += .0018*(1-smooth(1.326,1.345,z))*front
    return skull((x,y,z))


def apply_geometry():
    global BACK_CURVE
    face = bpy.data.objects[FACE]
    if face.get("anatomy_material_version") == VERSION:
        return {"alreadyApplied": True, "version": VERSION}
    BACK_CURVE=sorted((float(v.co.z),float(v.co.y)) for v in face.data.vertices
                      if abs(v.co.x)<.0001 and v.co.y>.001)
    col = bpy.data.collections[COL]
    objects = [o for o in list(col.all_objects) if o.type == "MESH" and (
        o.get("facial_role") or o.get("part") in {"head","hair"}
        or o.name.startswith(("Ear_", "EarHelix_", "EarConcha_", "Hair_")))]
    rows = []
    for obj in objects:
        role = obj.get("facial_role")
        transform = (lambda co: face_field(co, role)) if role else skull
        before = len(obj.data.vertices)
        blocks = obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
        groups = blocks if blocks else [obj.data.vertices]
        displacement = 0.
        for group in groups:
            points = group.data if blocks else group
            output = []
            for point in points:
                value = transform(point.co)
                displacement = max(displacement, (value-point.co).length)
                output.append(value)
            points.foreach_set("co", np.asarray(output, dtype=np.float32).ravel())
        if blocks:
            obj.data.vertices.foreach_set("co", np.asarray(
                [v.co for v in blocks[0].data], dtype=np.float32).ravel())
        obj.data.update()
        obj["anatomy_material_version"] = VERSION
        rows.append({"object":obj.name,"vertices":before,
                     "maxDisplacementMeters":displacement})
    # Project the iris onto the revised ocular surface, preserving retraction
    # at blink. Outside the aperture its edge remains beneath the skin.
    for side in ("L","R"):
        iris = bpy.data.objects["Iris brown "+side]
        white = bpy.data.objects["Eye sclera "+side]
        polys = [list(p.vertices) for p in white.data.polygons]
        for block in iris.data.shape_keys.key_blocks:
            other = white.data.shape_keys.key_blocks.get(block.name)
            other = other or white.data.shape_keys.key_blocks[0]
            bvh = BVHTree.FromPolygons([v.co for v in other.data],polys)
            for point in block.data:
                hit = bvh.ray_cast(Vector((point.co.x,-.5,point.co.z)),
                                   Vector((0,1,0)))[0]
                if hit:
                    point.co.y = hit.y-.00045
        for point, basis in zip(iris.data.vertices,
                                iris.data.shape_keys.key_blocks[0].data):
            point.co = basis.co
        iris.data.update()
    # Match hair rest-joint positions to the same upper-head field. No human
    # limb bone is changed; the field is zero below the high ponytail root.
    rig = bpy.data.objects["Character_Rig"]
    active = bpy.context.view_layer.objects.active
    old_selected = list(bpy.context.selected_objects)
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bones = []
    for bone in rig.data.edit_bones:
        if bone.name.startswith("hair_"):
            bone.head = skull(bone.head)
            bone.tail = skull(bone.tail)
            bones.append(bone.name)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.select_set(False)
    for obj in old_selected:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active
    return {"version":VERSION,"objects":rows,"hairRestBones":bones,
            "humanRestBonesChanged":False,"expressionTopologyPreserved":True}


def linear_hex(value):
    result = []
    for i in (0,2,4):
        v = int(value[i:i+2],16)/255
        result.append(v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4)
    return (*result,1)


def set_principled(material, color, roughness, specular=.3):
    material.use_nodes = True
    shader = next((n for n in material.node_tree.nodes
                   if n.type=="BSDF_PRINCIPLED"),None)
    if shader is None:
        return None
    material.diffuse_color = color
    for link in list(shader.inputs["Base Color"].links):
        material.node_tree.links.remove(link)
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Specular IOR Level"].default_value = specular
    return shader


def skin_material():
    material = bpy.data.materials.new("Character / neutral anatomical skin source")
    material.use_nodes = True
    n, l = material.node_tree.nodes, material.node_tree.links
    shader = n.get("Principled BSDF")
    color = linear_hex("DAB39D")
    material.diffuse_color = color
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = .48
    shader.inputs["Specular IOR Level"].default_value = .32
    shader.inputs["Subsurface Weight"].default_value = .09
    shader.inputs["Subsurface Radius"].default_value = (1,.48,.28)
    shader.inputs["Subsurface Scale"].default_value = .004
    coordinates = n.new("ShaderNodeTexCoord")
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1600
    noise.inputs["Detail"].default_value = 2.1
    noise.inputs["Roughness"].default_value = .65
    l.new(coordinates.outputs["Object"],noise.inputs["Vector"])
    tint = n.new("ShaderNodeValToRGB")
    tint.color_ramp.elements[0].color = tuple(c*.975 for c in color[:3])+(1,)
    tint.color_ramp.elements[1].color = tuple(c*1.025 for c in color[:3])+(1,)
    l.new(noise.outputs["Fac"],tint.inputs["Fac"])
    lip = n.new("ShaderNodeAttribute");lip.attribute_name = "LipWeight"
    vermilion = n.new("ShaderNodeAttribute");vermilion.attribute_name="NaturalLipColor"
    maximum=n.new("ShaderNodeMath");maximum.operation="MAXIMUM"
    l.new(lip.outputs["Fac"],maximum.inputs[0])
    l.new(vermilion.outputs["Fac"],maximum.inputs[1])
    mix = n.new("ShaderNodeMixRGB")
    mix.inputs[2].default_value = linear_hex("AD7568")
    l.new(maximum.outputs[0],mix.inputs[0])
    l.new(tint.outputs["Color"],mix.inputs[1])
    nostril=n.new("ShaderNodeAttribute");nostril.attribute_name="NostrilTissue"
    nose_mix=n.new("ShaderNodeMixRGB")
    nose_mix.inputs[2].default_value=linear_hex("6B3F32")
    l.new(nostril.outputs["Fac"],nose_mix.inputs[0])
    l.new(mix.outputs[0],nose_mix.inputs[1])
    l.new(nose_mix.outputs[0],shader.inputs["Base Color"])
    bump = n.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .14
    bump.inputs["Distance"].default_value = .00007
    l.new(noise.outputs["Fac"],bump.inputs["Height"])
    l.new(bump.outputs["Normal"],shader.inputs["Normal"])
    return material


def bake_skin(face, material, resolution=2048):
    """Bake actual 3D surface shading; never alter supplied reference images."""
    scene = bpy.context.scene
    old_engine = scene.render.engine
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = 64
    scene.render.bake.use_clear = False
    bpy.ops.object.select_all(action="DESELECT")
    face.select_set(True)
    bpy.context.view_layer.objects.active = face
    face.data.uv_layers.active = face.data.uv_layers["ExpressiveSkinUV"]
    face.data.uv_layers.active.active_render = True
    images = {}
    for kind, colorspace, basename in (
        ("DIFFUSE","sRGB","skin-neutral-anatomical-basecolor.png"),
        ("NORMAL","Non-Color","skin-neutral-anatomical-normal.png"),
    ):
        image = bpy.data.images.new(basename,width=resolution,height=resolution,
                                    alpha=False)
        image.generated_color = linear_hex("DAB39D") if kind=="DIFFUSE" else (.5,.5,1,1)
        image.colorspace_settings.name = colorspace
        tex = material.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        material.node_tree.nodes.active = tex
        bpy.ops.object.bake(type=kind,normal_space="TANGENT")
        image.filepath_raw = str(P/"textures"/basename)
        image.file_format = "PNG"
        image.save()
        image.pack()
        images[kind] = image
    portable = bpy.data.materials.new("Character / natural unified facial skin")
    portable.use_nodes = True
    n,l = portable.node_tree.nodes,portable.node_tree.links
    s = n.get("Principled BSDF")
    s.inputs["Roughness"].default_value=.48
    s.inputs["Specular IOR Level"].default_value=.32
    s.inputs["Subsurface Weight"].default_value=.09
    s.inputs["Subsurface Radius"].default_value=(1,.48,.28)
    s.inputs["Subsurface Scale"].default_value=.004
    uv=n.new("ShaderNodeUVMap");uv.uv_map="ExpressiveSkinUV"
    t=n.new("ShaderNodeTexImage");t.image=images["DIFFUSE"];t.extension="EXTEND"
    l.new(uv.outputs["UV"],t.inputs["Vector"]);l.new(t.outputs["Color"],s.inputs["Base Color"])
    tn=n.new("ShaderNodeTexImage");tn.image=images["NORMAL"];tn.extension="EXTEND"
    nm=n.new("ShaderNodeNormalMap");nm.uv_map="ExpressiveSkinUV"
    nm.inputs["Strength"].default_value=.45
    l.new(uv.outputs["UV"],tn.inputs["Vector"]);l.new(tn.outputs["Color"],nm.inputs["Color"])
    l.new(nm.outputs["Normal"],s.inputs["Normal"])
    face.data.materials[0]=portable
    scene.render.engine=old_engine
    return {kind:image.filepath_raw for kind,image in images.items()}


def apply_materials(bake=True):
    import bmesh
    neck=bpy.data.objects.get("Neck")
    if neck and not neck.data.shape_keys:
        bm=bmesh.new();bm.from_mesh(neck.data)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
        bm.to_mesh(neck.data);bm.free()
        neck.data.update()
    face=bpy.data.objects[FACE]
    material=skin_material()
    face.data.materials.clear()
    face.data.materials.append(material)
    for polygon in face.data.polygons:
        polygon.material_index=0
    skin_color=linear_hex("DAB39D")
    rows=[]
    palette={
        "Character / blouse":("E7DFD2",.78),
        "Character / trim":("DDD3C2",.73),
        "Character / button":("CCBEA8",.42),
        "Character / skirt":("4D5945",.8),
        "Character / skirt_shadow":("424E3B",.83),
        "Character / sock":("DAD4C5",.84),
        "Character / leather":("49392E",.37),
        "Character / sole":("2E2925",.68),
        "Character / nail":("C89986",.38),
        "Character / warm sclera":("D4D0C3",.27),
        "Character / subtle lip tissue":("AD7568",.43),
    }
    # Preserve existing garment normal maps and every UV/material assignment.
    for name,(color,roughness) in palette.items():
        mat=bpy.data.materials.get(name)
        if mat:
            shader=set_principled(mat,linear_hex(color),roughness)
            if shader and any(word in name for word in ("blouse","trim","skirt","sock")):
                shader.inputs["Sheen Weight"].default_value=.18
                shader.inputs["Sheen Roughness"].default_value=.68
            if shader and "leather" in name:
                shader.inputs["Coat Weight"].default_value=.08
                shader.inputs["Coat Roughness"].default_value=.36
            rows.append({"material":name,"baseColorSRGB":color,"roughness":roughness})
    col=bpy.data.collections[COL]
    skin_materials={}
    for obj in list(col.all_objects):
        if obj.type!="MESH" or obj==face:
            continue
        for mat in obj.data.materials:
            if mat and any(t in mat.name.lower() for t in (
                "/ skin","neutral skin","expressive neutral skin",
                "body skin","anatomical skin")):
                skin_materials[mat.name]=mat
    for name,mat in skin_materials.items():
        shader=set_principled(mat,skin_color,.48)
        if shader:
            shader.inputs["Subsurface Weight"].default_value=.09
            shader.inputs["Subsurface Radius"].default_value=(1,.48,.28)
            shader.inputs["Subsurface Scale"].default_value=.004
    face["default_blush"]=False
    textures=bake_skin(face,material) if bake else {}
    return {"palette":rows,"bodySkinMaterials":list(skin_materials),
            "textures":textures,"defaultBlush":False,
            "referenceImagesUnchanged":True}


def apply(bake=True):
    geometry=apply_geometry()
    import runpy
    detail=runpy.run_path(str(Path(__file__).with_name("facial_microstructure.py")))["apply"]()
    dental=runpy.run_path(str(Path(__file__).with_name("dental_refine.py")))["apply"]()
    materials=apply_materials(bake)
    return {"geometry":geometry,"microstructure":detail,"dental":dental,"materials":materials}


if __name__=="__main__":
    print(json.dumps(apply(),ensure_ascii=False,indent=2))
