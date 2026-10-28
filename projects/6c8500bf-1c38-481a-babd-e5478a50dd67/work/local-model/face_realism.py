"""Natural-proportion refinement of this character's existing animated face.

Works from the retained realism-direction-start checkpoint. Every existing
expression receives the same smooth spatial deformation as the neutral mesh.
Eye/mouth apertures, UV topology, skin weights and all existing keys survive.
"""
from pathlib import Path
import json
import math
import runpy
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

P = Path(__file__).resolve().parents[2]
VERSION = "natural-facial-proportions-03"
EX, EZ, MZ = .0435, 1.4308, 1.3688


def smooth(a, b, x):
    t = min(1., max(0., (x-a)/(b-a)))
    return t*t*(3-2*t)


def gaussian(x, z, cx, cz, rx, rz):
    return math.exp(-((x-cx)/rx)**2-((z-cz)/rz)**2)


def transform(co, role=""):
    x, y, z = co
    front = 1-smooth(-.035, .015, y)
    if not front:
        return Vector(co)
    side = 1 if x >= 0 else -1
    q = math.sqrt(((abs(x)-EX)/.045)**2+((z-(EZ+.006))/.038)**2)
    eye = (1-smooth(.68, 1.32, q))*front
    # Less exaggerated aperture, closer spacing, and a shorter eye-to-brow gap.
    nx = x + ((side*.0365+(x-side*EX)*.73)-x)*eye
    nz = z + ((EZ+.008+(z-EZ)*.71)-z)*eye
    ny = y - .0009*eye
    # Broaden the small illustrated mouth and bring the lip mound forward.
    mouth = gaussian(x,z,0,MZ,.035,.018)*front
    nx += .18*x*mouth
    nz -= .0016*mouth
    ny -= .0021*mouth
    # The bridge, alae and philtrum are three-dimensional, restrained forms.
    ny -= .0030*gaussian(x,z,0,1.418,.010,.022)*front
    ny += .0028*gaussian(x,z,0,1.397,.011,.007)*front
    ny -= .0016*gaussian(x,z,0,1.387,.008,.005)*front
    ny -= .0015*gaussian(x,z,0,MZ+.0028,.016,.0020)*front
    ny -= .0023*gaussian(x,z,0,MZ-.0034,.017,.0024)*front
    ny -= .0025*gaussian(x,z,0,1.343,.033,.009)*front
    for side_nose in (-1,1):
        nx += side_nose*.0012*gaussian(x,z,side_nose*.012,1.387,.006,.006)*front
        ny -= .0025*gaussian(x,z,side_nose*.012,1.387,.0055,.006)*front
        ny += .00125*gaussian(x,z,side_nose*.009,1.383,.0028,.0028)*front
        ny -= .00055*gaussian(x,z,side_nose*.0034,1.379,.0025,.007)*front
        ny -= .0018*gaussian(x,z,side_nose*.047,1.405,.025,.015)*front
        ny += .00055*gaussian(x,z,side_nose*.025,1.376,.005,.013)*front
    return Vector((nx, ny, nz))


def lip_color_weight(co):
    x,y,z=co
    u=x/.022
    if abs(u)>=1 or y>-.050:
        return 0.
    arc=max(0,1-u*u)**.7
    middle=MZ+.0012*abs(u)**1.8
    upper=middle+.0030*arc+.0010*math.exp(-((abs(u)-.33)/.22)**2)*arc
    lower=middle-.0037*arc
    return smooth(lower-.0010,lower+.0007,z)*(1-smooth(upper-.0007,upper+.0010,z))*.67


def make_iris_material():
    material=bpy.data.materials.new("Character / natural brown iris")
    material.use_nodes=True
    shader=material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value=.32
    shader.inputs["Specular IOR Level"].default_value=.20
    shader.inputs["Coat Weight"].default_value=.0
    shader.inputs["Coat Roughness"].default_value=.10
    tex=material.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image=bpy.data.images.load(str(P/"textures/iris-brown-realism.png"),check_existing=True)
    tex.image.pack()
    material.node_tree.links.new(tex.outputs["Color"],shader.inputs["Base Color"])
    return material


def bake_skin(face, old_material):
    mesh=face.data
    old_shader=next(n for n in old_material.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
    link=old_shader.inputs["Base Color"].links[0]
    if link.from_node.type!="TEX_IMAGE":
        raise ValueError("The preceding skin material must be a portable image.")
    image=link.from_node.image
    material=bpy.data.materials.new("Character / natural skin baking source")
    material.use_nodes=True
    nodes,links=material.node_tree.nodes,material.node_tree.links
    shader=nodes.get("Principled BSDF")
    source=nodes.new("ShaderNodeTexImage")
    source.image=image
    uv=nodes.new("ShaderNodeUVMap");uv.uv_map="ExpressiveSkinUV"
    links.new(uv.outputs["UV"],source.inputs["Vector"])
    attr=nodes.new("ShaderNodeAttribute");attr.attribute_name="NaturalLipColor"
    mix=nodes.new("ShaderNodeMixRGB")
    mix.inputs[2].default_value=(.57,.285,.235,1)
    links.new(attr.outputs["Fac"],mix.inputs[0])
    links.new(source.outputs["Color"],mix.inputs[1])
    noise=nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value=1850
    noise.inputs["Detail"].default_value=2
    coords=nodes.new("ShaderNodeTexCoord")
    links.new(coords.outputs["Object"],noise.inputs["Vector"])
    ramp=nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color=(.965,.965,.965,1)
    ramp.color_ramp.elements[1].color=(1.025,1.025,1.025,1)
    links.new(noise.outputs["Fac"],ramp.inputs["Fac"])
    multiply=nodes.new("ShaderNodeMixRGB");multiply.blend_type="MULTIPLY"
    multiply.inputs[0].default_value=1
    links.new(mix.outputs[0],multiply.inputs[1]);links.new(ramp.outputs[0],multiply.inputs[2])
    links.new(multiply.outputs[0],shader.inputs["Base Color"])
    bump=nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value=.18
    bump.inputs["Distance"].default_value=.00012
    links.new(noise.outputs["Fac"],bump.inputs["Height"])
    links.new(bump.outputs["Normal"],shader.inputs["Normal"])
    mesh.materials[0]=material
    scene=bpy.context.scene
    scene.render.engine="CYCLES";scene.cycles.device="CPU";scene.cycles.samples=1
    scene.render.bake.use_pass_direct=False;scene.render.bake.use_pass_indirect=False
    scene.render.bake.use_pass_color=True;scene.render.bake.margin=16
    for obj in scene.objects:
        obj.select_set(False)
    face.select_set(True);bpy.context.view_layer.objects.active=face
    baked={}
    for kind,colorspace,name in (
        ("DIFFUSE","sRGB","face-natural-basecolor.png"),
        ("NORMAL","Non-Color","face-natural-normal.png"),
    ):
        target_image=bpy.data.images.new(name,width=2048,height=2048,alpha=False)
        target_image.colorspace_settings.name=colorspace
        target=nodes.new("ShaderNodeTexImage");target.image=target_image
        nodes.active=target
        bpy.ops.object.bake(type=kind,normal_space="TANGENT")
        target_image.filepath_raw=str(P/"textures"/name);target_image.file_format="PNG"
        target_image.save();target_image.pack();baked[kind]=target_image
    portable=bpy.data.materials.new("Character / natural neutral skin")
    portable.use_nodes=True
    n,l=portable.node_tree.nodes,portable.node_tree.links
    s=n.get("Principled BSDF")
    s.inputs["Roughness"].default_value=.50;s.inputs["Specular IOR Level"].default_value=.28
    s.inputs["Subsurface Weight"].default_value=.085
    s.inputs["Subsurface Radius"].default_value=(1,.45,.25)
    s.inputs["Subsurface Scale"].default_value=.006
    u=n.new("ShaderNodeUVMap");u.uv_map="ExpressiveSkinUV"
    t=n.new("ShaderNodeTexImage");t.image=baked["DIFFUSE"]
    l.new(u.outputs["UV"],t.inputs["Vector"]);l.new(t.outputs["Color"],s.inputs["Base Color"])
    nm=n.new("ShaderNodeNormalMap");nm.uv_map="ExpressiveSkinUV"
    nm.inputs["Strength"].default_value=.50
    tn=n.new("ShaderNodeTexImage");tn.image=baked["NORMAL"]
    l.new(u.outputs["UV"],tn.inputs["Vector"]);l.new(tn.outputs["Color"],nm.inputs["Color"])
    l.new(nm.outputs["Normal"],s.inputs["Normal"])
    mesh.materials[0]=portable
    return {"basecolor":baked["DIFFUSE"].filepath_raw,"normal":baked["NORMAL"].filepath_raw}


def apply():
    face=bpy.data.objects["Face · sculpted planes & original illustration"]
    if face.get("face_realism_version"):
        raise RuntimeError("Use the retained pre-realism source; this spatial transform runs once.")
    collection=bpy.data.collections["CHARACTER · Original geometry"]
    objects=[o for o in collection.all_objects if o.type=="MESH" and o.get("facial_role")]
    old_material=face.data.materials[0]
    old_material.use_fake_user=True
    before={o.name:len(o.data.vertices) for o in objects}
    face_basis=[v.co.copy() for v in face.data.vertices]
    color=face.data.attributes.get("NaturalLipColor") or face.data.attributes.new(
        "NaturalLipColor","FLOAT","POINT")
    for value,co in zip(color.data,face_basis):
        value.value=lip_color_weight(co)
    iris_material=make_iris_material()
    for obj in objects:
        role=obj["facial_role"]
        original=[v.co.copy() for v in obj.data.vertices]
        blocks=obj.data.shape_keys.key_blocks if obj.data.shape_keys else []
        for block in blocks:
            transformed=[]
            for vertex in block.data:
                co=transform(vertex.co,role)
                if obj==face:
                    # Lip relief follows each expression rather than floating
                    # as a separate fixed mouth strip.
                    co.y-=.0012*color.data[len(transformed)].value
                if role=="eye_iris":
                    side=1 if vertex.co.x>=0 else -1
                    center=transform((side*EX,vertex.co.y,EZ+.0007),role)
                    co.x=center.x+(co.x-center.x)*.79
                    co.z=center.z+(co.z-center.z)*.79
                transformed.append(co)
            block.data.foreach_set("co",np.asarray(transformed,dtype=np.float32).ravel())
        if blocks:
            obj.data.vertices.foreach_set("co",np.asarray(
                [v.co for v in blocks[0].data],dtype=np.float32).ravel())
        else:
            for vertex,co in zip(obj.data.vertices,original):
                vertex.co=transform(co,role)
        if role=="eye_iris":
            uv=obj.data.uv_layers.active
            uv.name="NaturalIrisUV";uv.active_render=True
            side=1 if original[0].x>=0 else -1
            for loop in obj.data.loops:
                co=original[loop.vertex_index]
                uv.data[loop.index].uv=(.5+(co.x-side*EX)/.012*.477,
                                       .5+(co.z-(EZ+.0007))/.0126*.477)
            obj.data.materials.clear();obj.data.materials.append(iris_material)
        obj.data.update()
        obj["face_realism_version"]=VERSION
    # Resizing a curved iris in X/Z also requires placing it on the resized
    # sclera. Keeping the former depth makes the white intersect its centre.
    for side_label in ("L","R"):
        iris=bpy.data.objects["Iris brown "+side_label]
        white=bpy.data.objects["Eye sclera "+side_label]
        polygons=[list(p.vertices) for p in white.data.polygons]
        for block in iris.data.shape_keys.key_blocks:
            white_block=white.data.shape_keys.key_blocks.get(block.name)
            white_block=white_block or white.data.shape_keys.key_blocks[0]
            bvh=BVHTree.FromPolygons([v.co for v in white_block.data],polygons)
            for vertex in block.data:
                hit=bvh.ray_cast(Vector((vertex.co.x,-.5,vertex.co.z)),Vector((0,1,0)))[0]
                if hit:
                    vertex.co.y=hit.y-.0007
        for vertex,basis in zip(iris.data.vertices,iris.data.shape_keys.key_blocks[0].data):
            vertex.co=basis.co
        iris.data.update()
    # Real highlights come from the studio lights in PBR, rather than painted
    # white dots. The retained iris geometry now has natural radial texture.
    sclera=bpy.data.materials.get("Character / warm sclera")
    if sclera:
        s=next(n for n in sclera.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
        s.inputs["Base Color"].default_value=(.72,.70,.65,1)
        s.inputs["Roughness"].default_value=.28
        s.inputs["Specular IOR Level"].default_value=.25
    textures=bake_skin(face,old_material)
    face["default_blush"]=False
    face["human_proportion_notes"]="smaller apertures and irises; closer eyes; dimensional nasal bridge, alae, philtrum and lips"
    after={o.name:len(o.data.vertices) for o in objects}
    assert before==after
    return {"version":VERSION,"meshes":len(objects),"topologyPreserved":True,
            "existingExpressionKeysPreserved":True,"defaultBlush":False,
            "textures":textures}


if __name__=="__main__":
    report=apply()
    report["headProfile"]=runpy.run_path(
        str(Path(__file__).with_name("head_proportion_refine.py")))["apply"]()
    out=P/"work/realism-authoring";out.mkdir(exist_ok=True,parents=True)
    (out/"face-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"face-review.blend"))
    render=runpy.run_path(str(Path(__file__).with_name("render_expressions.py")))
    render["OUT"]=out
    # Function globals belong to runpy's returned namespace; explicit paths
    # below avoid changing the previous expression review renders.
    render["render"].__globals__["OUT"]=out
    for name,pose,angle in [
        ("face-neutral",{},0),("face-three-quarter",{},40),("face-side",{},90),
        ("face-blink",{"blink":1},0),("face-happy",render["module"]["presets"]()["happy"],0),
    ]:
        render["render"](name,pose,angle,size=800,samples=20)
