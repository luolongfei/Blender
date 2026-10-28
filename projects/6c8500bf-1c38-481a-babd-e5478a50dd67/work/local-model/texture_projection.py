"""Reference-guided painted detail, baked later to portable model UV atlases."""
import bpy
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
COL="CHARACTER · Original geometry"


def source_image(part,view):
    img=bpy.data.images.load(str(ROOT/"images"/part/(view+".png")),check_existing=True)
    img.pack()
    return img


def shader(name,base,rough):
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    ns=mat.node_tree.nodes;ns.clear();links=mat.node_tree.links
    out=ns.new("ShaderNodeOutputMaterial")
    bsdf=ns.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value=base
    bsdf.inputs["Roughness"].default_value=rough
    bsdf.inputs["Specular IOR Level"].default_value=.15
    links.new(bsdf.outputs[0],out.inputs[0])
    uv=ns.new("ShaderNodeUVMap");uv.uv_map="ReferenceUV"
    return mat,ns,links,bsdf,uv


def set_uv(obj,fn):
    layer=obj.data.uv_layers.get("ReferenceUV") or obj.data.uv_layers.new(name="ReferenceUV")
    for loop in obj.data.loops:
        layer.data[loop.index].uv=fn(obj.data.vertices[loop.vertex_index].co)
    attr=obj.data.attributes.get("FrontReference") or obj.data.attributes.new("FrontReference","FLOAT","POINT")
    strength=obj.data.attributes.get("ReferenceStrength") or obj.data.attributes.new("ReferenceStrength","FLOAT","POINT")
    for v in obj.data.vertices:
        y=v.co.y
        t=max(0,min(1,(.025-y)/.050))
        attr.data[v.index].value=t*t*(3-2*t)
        strength.data[v.index].value=1
    return strength


def hair_detail():
    objs=[o for o in bpy.data.collections[COL].objects if o.type=="MESH" and o.get("part")=="hair" and o.get("hair_region")!="tie"]
    base=bpy.data.materials["Character / hair"].diffuse_color[:]
    mat,n,l,bsdf,uv=shader("Reference paint / hair strands",base,.68)
    tex=n.new("ShaderNodeTexImage");tex.image=source_image("head","front");tex.extension="EXTEND"
    l.new(uv.outputs[0],tex.inputs["Vector"])
    bw=n.new("ShaderNodeRGBToBW");l.new(tex.outputs["Color"],bw.inputs[0])
    dark=n.new("ShaderNodeMath");dark.operation="LESS_THAN";dark.inputs[1].default_value=.22;l.new(bw.outputs[0],dark.inputs[0])
    alpha=n.new("ShaderNodeMath");alpha.operation="MULTIPLY";l.new(dark.outputs[0],alpha.inputs[0]);l.new(tex.outputs["Alpha"],alpha.inputs[1])
    mix=n.new("ShaderNodeMixRGB");mix.inputs[1].default_value=base
    l.new(alpha.outputs[0],mix.inputs[0]);l.new(tex.outputs["Color"],mix.inputs[2]);l.new(mix.outputs[0],bsdf.inputs["Base Color"])
    for obj in objs:
        set_uv(obj,lambda v:((627+v.x*2590)/1254,1-(858-(v.z-1.32)*2590)/1254))
        obj.data.materials.clear();obj.data.materials.append(mat)
    return objs


def hair_strands_material():
    objs=[o for o in bpy.data.collections[COL].objects if o.type=="MESH" and o.get("part")=="hair" and o.get("hair_region")!="tie"]
    mat=bpy.data.materials.new("Character / longitudinal hair fibres")
    mat.use_nodes=True
    p=mat.node_tree.nodes.get("Principled BSDF")
    p.inputs["Roughness"].default_value=.67
    p.inputs["Specular IOR Level"].default_value=.16
    image=bpy.data.images.load(str(ROOT/"textures/hair-strands.png"),check_existing=True)
    image.pack()
    tex=mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image=image;tex.extension="REPEAT"
    mat.node_tree.links.new(tex.outputs["Color"],p.inputs["Base Color"])
    for obj in objs:
        obj.data.uv_layers.active_index=obj.data.uv_layers.find("UVMap")
        obj.data.uv_layers["UVMap"].active_render=True
        obj.data.materials.clear();obj.data.materials.append(mat)
    return objs


def cloth_shader(name,key,strength):
    base=bpy.data.materials["Character / "+key].diffuse_color[:]
    mat,n,l,bsdf,uv=shader(name,base,.80)
    front=n.new("ShaderNodeTexImage");front.image=source_image("body","front");front.extension="EXTEND"
    back=n.new("ShaderNodeTexImage");back.image=source_image("body","back");back.extension="EXTEND"
    for node in (front,back):l.new(uv.outputs[0],node.inputs["Vector"])
    frontmix=n.new("ShaderNodeAttribute");frontmix.attribute_name="FrontReference"
    image=n.new("ShaderNodeMixRGB")
    l.new(frontmix.outputs["Fac"],image.inputs[0]);l.new(back.outputs["Color"],image.inputs[1]);l.new(front.outputs["Color"],image.inputs[2])
    a=n.new("ShaderNodeMixRGB")
    l.new(frontmix.outputs["Fac"],a.inputs[0]);l.new(back.outputs["Alpha"],a.inputs[1]);l.new(front.outputs["Alpha"],a.inputs[2])
    mask=n.new("ShaderNodeAttribute");mask.attribute_name="ReferenceStrength"
    factor=n.new("ShaderNodeMath");factor.operation="MULTIPLY"
    l.new(a.outputs[0],factor.inputs[0]);l.new(mask.outputs["Fac"],factor.inputs[1])
    gain=n.new("ShaderNodeMath");gain.operation="MULTIPLY";gain.inputs[1].default_value=strength;l.new(factor.outputs[0],gain.inputs[0])
    mix=n.new("ShaderNodeMixRGB");mix.inputs[1].default_value=base
    l.new(gain.outputs[0],mix.inputs[0]);l.new(image.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],bsdf.inputs["Base Color"])
    return mat


def clothes_detail():
    col=bpy.data.collections[COL]
    blouse=cloth_shader("Reference paint / blouse folds","blouse",.68)
    skirt=cloth_shader("Reference paint / woven skirt","skirt",.38)
    objs=[]
    for obj in col.objects:
        if obj.name=="Blouse_Tucked_Bodice":
            attr=set_uv(obj,lambda v:((627+v.x*650)/1254,1-(380-(v.z-.99)*745)/1254))
            for v in obj.data.vertices:
                # Center placket/buttons are actual geometry, not duplicate paint.
                middle=max(0,min(1,(abs(v.co.x)-.009)/.01))
                neck=max(0,min(1,(1.215-v.co.z)/.026))
                attr.data[v.index].value=middle*neck*neck*(3-2*neck)
            obj.data.materials.clear();obj.data.materials.append(blouse);objs.append(obj)
        elif obj.name.startswith("Blouse_Puff_Sleeve"):
            set_uv(obj,lambda v:((627+v.x*755)/1254,1-(233-(v.z-1.235)*850)/1254))
            obj.data.materials.clear();obj.data.materials.append(blouse);objs.append(obj)
        elif obj.name=="Skirt_Sixteen_Pleats":
            set_uv(obj,lambda v:((627+v.x*812.5)/1254,1-(395-(v.z-.99)*885)/1254))
            obj.data.materials.clear();obj.data.materials.append(skirt);objs.append(obj)
    return objs


def refine_collar():
    names=[]
    for obj in bpy.data.collections[COL].objects:
        if obj.type=="MESH" and (obj.name.startswith("Blouse_PeterPan_Collar") or obj.name.startswith("Collar_Edge_Stitch")):
            if obj.get("collar_refined"):
                continue
            for v in obj.data.vertices:
                front=max(0,min(1,-v.co.y/.073))
                v.co.x*=1+.18*front
                v.co.y*=1+.17*front
                v.co.z-=.014*front
            obj["collar_refined"]=True
            obj.data.update();names.append(obj.name)
    return names
