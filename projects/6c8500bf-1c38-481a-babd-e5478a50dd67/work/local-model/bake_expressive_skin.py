"""Portable skin-only albedo on the newly authored head topology."""
from pathlib import Path
import math
import bpy

P=Path(__file__).resolve().parents[2]

def smooth(a,b,x):
    t=min(1,max(0,(x-a)/(b-a)))
    return t*t*(3-2*t)

def apply(force=False):
    face=bpy.data.objects["Face · sculpted planes & original illustration"]
    if face.get("expressive_skin_bake") and not force:
        return {"alreadyBaked":True}
    mesh=face.data
    ref=mesh.uv_layers.get("SkinReference") or mesh.uv_layers.new(name="SkinReference")
    uv=mesh.uv_layers.get("ExpressiveSkinUV") or mesh.uv_layers.new(name="ExpressiveSkinUV")
    for poly in mesh.polygons:
        us=[]
        for li in poly.loop_indices:
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            ref.data[li].uv=((627+co.x*2590)/1254,1-(858-(co.z-1.32)*2590)/1254)
            us.append(math.atan2(co.x,.006-co.y)/math.tau+.5)
        if max(us)-min(us)>.5:
            us=[u+1 if u<.5 else u for u in us]
        for li,u in zip(poly.loop_indices,us):
            co=mesh.vertices[mesh.loops[li].vertex_index].co
            uv.data[li].uv=(u,(co.z-1.310)/.275)
    mesh.uv_layers.active_index=mesh.uv_layers.find("ExpressiveSkinUV")
    uv.active_render=True
    mask=mesh.attributes.get("SkinProjection") or mesh.attributes.new("SkinProjection","FLOAT","POINT")
    widths=[(1.318,.018),(1.340,.036),(1.360,.052),(1.380,.055),(1.400,.058),(1.410,.065),(1.422,.077),(1.450,.081),(1.475,.080),(1.572,.080)]
    for v in mesh.vertices:
        x,y,z=v.co
        width=.08
        for a,b in zip(widths,widths[1:]):
            if z<=b[0]:
                t=smooth(a[0],b[0],z)
                width=a[1]*(1-t)+b[1]*t
                break
        value=(1-smooth(width-.008,width+.003,abs(x)))*smooth(1.333,1.350,z)*(1-smooth(-.055,-.015,y))
        # Source bangs are preserved in the texture file, but must never be
        # projected onto moving lids or visible forehead below real 3D hair.
        value*=1-smooth(1.426,1.437,z)
        mask.data[v.index].value=value
    source=bpy.data.materials.new("Character / skin-only projection baking source")
    source.use_nodes=True
    n,l=source.node_tree.nodes,source.node_tree.links
    n.clear()
    out=n.new("ShaderNodeOutputMaterial")
    bsdf=n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value=.68
    bsdf.inputs["Specular IOR Level"].default_value=.14
    l.new(bsdf.outputs[0],out.inputs[0])
    mapping=n.new("ShaderNodeUVMap")
    mapping.uv_map="SkinReference"
    tex=n.new("ShaderNodeTexImage")
    tex.image=bpy.data.images.load(str(P/"textures/face-skin-only-projection.png"),check_existing=True)
    tex.image.pack()
    l.new(mapping.outputs[0],tex.inputs["Vector"])
    attr=n.new("ShaderNodeAttribute")
    attr.attribute_name="SkinProjection"
    mix=n.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value=(.89,.665,.495,1)
    l.new(attr.outputs["Fac"],mix.inputs[0])
    l.new(tex.outputs["Color"],mix.inputs[2])
    lips=n.new("ShaderNodeAttribute")
    lips.attribute_name="LipWeight"
    lipmix=n.new("ShaderNodeMixRGB")
    lipmix.inputs[2].default_value=(.80,.53,.40,1)
    l.new(lips.outputs["Fac"],lipmix.inputs[0])
    l.new(mix.outputs[0],lipmix.inputs[1])
    l.new(lipmix.outputs[0],bsdf.inputs["Base Color"])
    mesh.materials.clear()
    mesh.materials.append(source)
    for poly in mesh.polygons:
        poly.material_index=0
    image=bpy.data.images.new("Character · expressive skin albedo",width=2048,height=2048,alpha=False)
    image.colorspace_settings.name="sRGB"
    target=n.new("ShaderNodeTexImage")
    target.image=image
    n.active=target
    for node in n:
        node.select=node==target
    for obj in bpy.context.scene.objects:
        if obj.select_get():
            obj.select_set(False)
    face.select_set(True)
    bpy.context.view_layer.objects.active=face
    scene=bpy.context.scene
    engine=scene.render.engine
    scene.render.engine="CYCLES"
    scene.cycles.samples=1
    scene.render.bake.use_pass_direct=False
    scene.render.bake.use_pass_indirect=False
    scene.render.bake.use_pass_color=True
    scene.render.bake.margin=16
    bpy.ops.object.bake(type="DIFFUSE")
    path=P/"textures/face-expressive-skin-basecolor.png"
    image.filepath_raw=str(path)
    image.file_format="PNG"
    image.save()
    image.pack()
    material=bpy.data.materials.new("Character / portable expressive skin")
    material.use_nodes=True
    nt=material.node_tree
    shader=nt.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value=.68
    shader.inputs["Specular IOR Level"].default_value=.14
    shader.inputs["Subsurface Weight"].default_value=.025
    texture=nt.nodes.new("ShaderNodeTexImage")
    texture.image=image
    mapping=nt.nodes.new("ShaderNodeUVMap")
    mapping.uv_map="ExpressiveSkinUV"
    nt.links.new(mapping.outputs[0],texture.inputs["Vector"])
    nt.links.new(texture.outputs["Color"],shader.inputs["Base Color"])
    mesh.materials[0]=material
    scene.render.engine=engine
    face["expressive_skin_bake"]=True
    face["skin_basecolor"]=str(path)
    face["default_blush"]=False
    return {"texture":str(path),"size":2048,"no_painted_eyes_mouth_brows":True}
