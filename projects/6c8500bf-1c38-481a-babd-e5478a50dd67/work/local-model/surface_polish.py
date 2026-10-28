"""Reference-specific neutral facial planes and portable textile finish.

No eye separation, expressions, donor meshes or facial rig is introduced.
"""
from pathlib import Path
import importlib.util
import math
import bpy

ROOT = Path(__file__).resolve().parents[2]


def helper(name):
    spec = importlib.util.spec_from_file_location(
        "surface_polish_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smooth_profile(table, z, column):
    for index, (a, b) in enumerate(zip(table, table[1:])):
        if a[0] <= z <= b[0]:
            before = table[max(0, index - 1)]
            after = table[min(len(table) - 1, index + 2)]
            h = b[0] - a[0]
            t = (z - a[0]) / h
            ma = (b[column] - before[column]) / (b[0] - before[0])
            mb = (after[column] - a[column]) / (after[0] - a[0])
            return ((2*t**3 - 3*t*t + 1)*a[column] +
                    (t**3 - 2*t*t + t)*h*ma +
                    (-2*t**3 + 3*t*t)*b[column] +
                    (t**3 - t*t)*h*mb)
    return table[0 if z < table[0][0] else -1][column]


def face_and_skin():
    col = bpy.data.collections["CHARACTER · Original geometry"]
    face = next(o for o in col.objects if o.name.startswith("Face ·"))
    if face.get("client_surface_polish") == 1:
        return {"alreadyApplied": True}
    cb = helper("character_builder")
    # This colour comes from clear cheek/chin samples in the approved PNG.
    skin = cb.linear_hex("FCE3CE")
    for key, color in (("skin", skin),
                       ("skin_shadow", cb.linear_hex("DBB398")),
                       ("nail", cb.linear_hex("E8CBBB"))):
        mat = bpy.data.materials["Character / " + key]
        mat.diffuse_color = color
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = .61 if key != "nail" else .42
        bsdf.inputs["Specular IOR Level"].default_value = .18

    table = [
        (1.318, -.050, .004, .004), (1.326, -.037, .040, .025),
        (1.340, -.015, .069, .041), (1.360, .000, .083, .061),
        (1.380, .005, .0875, .0775), (1.400, .005, .0875, .089),
    ]
    for vert in face.data.vertices:
        x, y, z = vert.co
        angle = math.tau * (vert.index % 112) / 112
        cosine = math.cos(angle)
        if z < 1.38:
            cy, front, back = (smooth_profile(table, z, i) for i in (1, 2, 3))
            y = cy - front*max(cosine, 0)**.57 if cosine >= 0 else cy + back*(-cosine)
            if cosine > 0:
                nose = .014*math.exp(-(x/.012)**2 - ((z-1.390)/.015)**2)
                bridge = .007*math.exp(-(x/.009)**2 - ((z-1.415)/.028)**2)
                lips = .003*math.exp(-(x/.024)**4 - ((z-1.365)/.008)**2)
                cheek = .003*(math.exp(-((x-.055)/.025)**2) +
                              math.exp(-((x+.055)/.025)**2))*math.exp(-((z-1.387)/.020)**2)
                y -= (nose + bridge + lips + cheek)*cosine**4
        if cosine > 0:
            # Sub-millimetre eyelid and lower-lip relief on the existing surface.
            # It supports the approved fixed painting without adding eyeballs.
            local_eye_x = (abs(x) - .0445) / .023
            if abs(local_eye_x) < 1.25:
                eye_mask = math.exp(-(local_eye_x / 1.04)**8)
                lid_z = 1.430 + .0125*max(0, 1-local_eye_x**2)
                y -= .00075*eye_mask*math.exp(-((z-lid_z)/.0028)**2)*cosine**4
            y -= .00065*math.exp(-(x/.021)**4 - ((z-1.361)/.0038)**2)*cosine**6
        vert.co.y = y
    face.data.update()
    bake = helper("bake_face")
    bake.restore_face_projection(face)
    source = face.data.materials[0]
    for node in source.node_tree.nodes:
        if node.type == "MIX_RGB" and not node.inputs[1].is_linked:
            node.inputs[1].default_value = skin
    source.diffuse_color = skin
    result = bake.bake_face(face, str(ROOT/"textures/face-basecolor.png"), 2048)
    face["client_surface_polish"] = 1
    return {"face": face.name, "skinHex": "FCE3CE", "bake": result}


def neck_join():
    obj = bpy.data.objects["Neck"]
    if obj.get("neck_join_version") == 1:
        return {"alreadyApplied": True}
    top = max(vertex.co.z for vertex in obj.data.vertices)
    changed = []
    for vertex in obj.data.vertices:
        if abs(vertex.co.z - top) < 1e-5:
            # Put the open top of the neck completely inside the head, rather
            # than ending its visible cylinder at the back of the jaw.
            vertex.co.x *= .035/.045
            vertex.co.y = .006 + (vertex.co.y-.006)*.031/.040
            vertex.co.z = 1.382
            changed.append(vertex.index)
    head = obj.vertex_groups.get("head") or obj.vertex_groups.new(name="head")
    neck = obj.vertex_groups.get("neck") or obj.vertex_groups.new(name="neck")
    for vertex in obj.data.vertices:
        if vertex.co.z < 1.32:
            continue
        t = max(0, min(1, (vertex.co.z-1.32)/.045))
        t = t*t*(3-2*t)
        for group in obj.vertex_groups:
            group.remove([vertex.index])
        if t < 1:
            neck.add([vertex.index], 1-t, "REPLACE")
        if t > 0:
            head.add([vertex.index], t, "REPLACE")
    obj.data.update()
    obj["neck_join_version"] = 1
    return {"mesh": obj.name, "embeddedTopVertices": len(changed)}


def cloth_normal_bake(resolution=4096):
    """Bake a very shallow woven normal into the existing shared delivery UV."""
    names = ["Blouse_Tucked_Bodice", "Blouse_Puff_Sleeve_L",
             "Blouse_Puff_Sleeve_R", "Skirt_Sixteen_Pleats"]
    objs = [bpy.data.objects[name] for name in names]
    mats = {mat for obj in objs for mat in obj.data.materials}
    if all(m.get("portable_weave_version") == 1 for m in mats):
        return {"alreadyApplied": True}
    if len(mats) != 1:
        raise ValueError("Expected the four garments to share their checked color atlas.")
    mat = next(iter(mats))
    scene = bpy.context.scene
    rig = bpy.data.objects["Character_Rig"]
    original_pose = rig.data.pose_position
    engine = scene.render.engine
    samples = scene.cycles.samples
    bake_settings = {key: getattr(scene.render.bake, key) for key in
                     ("use_selected_to_active", "use_clear", "margin", "normal_space")}
    previous_active = bpy.context.view_layer.objects.active
    previous_selection = list(bpy.context.selected_objects)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    if shader.inputs["Normal"].is_linked:
        raise ValueError("Preserve the existing normal connection before changing this material.")
    created = []

    def node(kind):
        value = nodes.new(kind)
        created.append(value)
        return value

    texcoord = node("ShaderNodeTexCoord")
    noise = node("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 950
    noise.inputs["Detail"].default_value = 1
    links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    waves = []
    for axis in ("X", "Z"):
        wave = node("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = axis
        wave.inputs["Scale"].default_value = 420
        wave.inputs["Distortion"].default_value = .25
        links.new(texcoord.outputs["Object"], wave.inputs["Vector"])
        waves.append(wave)
    multiply = node("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    links.new(waves[0].outputs["Color"], multiply.inputs[0])
    links.new(waves[1].outputs["Color"], multiply.inputs[1])
    mix = node("ShaderNodeMixRGB")
    mix.inputs[0].default_value = .3
    links.new(multiply.outputs[0], mix.inputs[1])
    links.new(noise.outputs["Fac"], mix.inputs[2])
    bump = node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .16
    bump.inputs["Distance"].default_value = .00012
    links.new(mix.outputs[0], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    image = bpy.data.images.new("Garment · subtle woven tangent normal",
                                width=resolution, height=resolution, alpha=False)
    image.colorspace_settings.name = "Non-Color"
    image.generated_color = (.5, .5, 1, 1)
    target = nodes.new("ShaderNodeTexImage")
    target.name = "Portable textile normal"
    target.image = image
    for n in nodes:
        n.select = False
    target.select = True
    nodes.active = target
    output = ROOT/"textures/clothing-normal.png"
    try:
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 8
        scene.render.bake.use_selected_to_active = False
        scene.render.bake.use_clear = True
        scene.render.bake.margin = 16
        scene.render.bake.normal_space = "TANGENT"
        bpy.ops.object.bake(type="NORMAL")
        image.filepath_raw = str(output)
        image.file_format = "PNG"
        image.save()
        image.pack()
        for n in created:
            nodes.remove(n)
        uv = nodes.new("ShaderNodeUVMap")
        uv.uv_map = objs[0].data.uv_layers.active.name
        normal = nodes.new("ShaderNodeNormalMap")
        normal.inputs["Strength"].default_value = .50
        links.new(uv.outputs["UV"], target.inputs["Vector"])
        links.new(target.outputs["Color"], normal.inputs["Color"])
        links.new(normal.outputs["Normal"], shader.inputs["Normal"])
        shader.inputs["Roughness"].default_value = .78
        shader.inputs["Specular IOR Level"].default_value = .18
        mat["portable_weave_version"] = 1
        return {"image": str(output), "resolution": resolution, "meshes": names}
    finally:
        rig.data.pose_position = original_pose
        scene.render.engine = engine
        scene.cycles.samples = samples
        for key, value in bake_settings.items():
            setattr(scene.render.bake, key, value)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in previous_selection:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = previous_active
        bpy.context.view_layer.update()
