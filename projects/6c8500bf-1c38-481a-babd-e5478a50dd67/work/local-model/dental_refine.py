"""Rounded individual tooth crowns with a stable upper dental arch."""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix

VERSION="individual-dental-crowns-01"


def crowns(lower=False):
    bm=bmesh.new()
    # Central and lateral incisors, canines and two premolars on each side.
    specs=[(.0030,.0058,.0070),(.0084,.0048,.0063),(.0130,.0044,.0068),
           (.0174,.0044,.0057),(.0217,.0042,.0054)]
    for sign in (-1,1):
        for i,(x,width,height) in enumerate(specs):
            x*=sign
            y=-.096+.012*(abs(x)/.025)**1.65
            if lower:
                width*=.93;height*=.68
                bottom=1.362-height
                z=bottom+height/2
                y+=.0076
            else:
                bottom=(1.3638+[0,.0006,-.0002,.0011,.0016][i]
                        +(.00010 if sign>0 and i==1 else 0))
                z=bottom+height/2
            added=bmesh.ops.create_cube(bm,size=1)["verts"]
            angle=-sign*(abs(x)/.025)*.63
            rot=Matrix.Rotation(angle,3,"Z")
            for vertex in added:
                local=Vector((vertex.co.x*width,vertex.co.y*.0034,vertex.co.z*height))
                vertex.co=rot@local+Vector((x,y,z))
    bmesh.ops.bevel(bm,geom=list(bm.edges),offset=.00032,segments=3,
                    affect="EDGES",profile=.5)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    mesh=bpy.data.meshes.new("Rounded lower crowns" if lower else "Rounded upper crowns")
    bm.to_mesh(mesh);bm.free()
    for polygon in mesh.polygons:polygon.use_smooth=True
    uv=mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co=mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv=((co.x+.025)/.05,(co.z-1.350)/.026)
    return mesh


def assign_rig(obj,rig):
    obj.vertex_groups.clear()
    group=obj.vertex_groups.new(name="head")
    group.add(list(range(len(obj.data.vertices))),1,"REPLACE")
    for mod in list(obj.modifiers):
        if mod.type=="ARMATURE":obj.modifiers.remove(mod)
    mod=obj.modifiers.new("Humanoid deformation","ARMATURE");mod.object=rig
    obj["part"]="head";obj["facial_role"]="mouth_teeth"
    obj["dental_refine_version"]=VERSION


def apply():
    upper=bpy.data.objects["Upper dental arch"]
    if upper.get("dental_refine_version")==VERSION:
        return {"alreadyApplied":True}
    rig=bpy.data.objects["Character_Rig"]
    col=bpy.data.collections["CHARACTER · Original geometry"]
    material=bpy.data.materials["Character / soft ivory teeth"]
    bsdf=material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=(.66,.59,.45,1)
    bsdf.inputs["Roughness"].default_value=.32
    bsdf.inputs["Specular IOR Level"].default_value=.34
    bsdf.inputs["Subsurface Weight"].default_value=.025
    old=upper.data
    old.use_fake_user=True
    keys=[k.name for k in old.shape_keys.key_blocks if k.name!="Basis"]
    upper.data=crowns()
    upper.data.materials.append(material)
    assign_rig(upper,rig)
    # Maxillary teeth remain rigid under a smile/pucker instead of stretching
    # with the lip border. Lower teeth follow the modeled jaw opening.
    lower=bpy.data.objects.new("Lower dental arch",crowns(True))
    col.objects.link(lower)
    lower.data.materials.append(material)
    assign_rig(lower,rig)
    lower.shape_key_add(name="Basis")
    opening={"aa":.0040,"oh":.0030,"ou":.0010,"ee":.0018,"ih":.0008,
             "surprised_mouth":.0040,"happy_mouth":.0012}
    for name in keys:
        key=lower.shape_key_add(name=name)
        value=opening.get(name,0)
        for vertex in key.data:
            vertex.co.z-=value
            vertex.co.y+=value*.28
    return {"version":VERSION,"upperCrowns":10,"lowerCrowns":10,
            "upperTeethStretchWithExpression":False,
            "lowerJawKeys":keys,"oldUpperRibbonMeshRetained":old.name,
            "anatomicalScope":"visible crowns; no tooth roots or full oral anatomy"}


if __name__=="__main__":
    print(apply())
