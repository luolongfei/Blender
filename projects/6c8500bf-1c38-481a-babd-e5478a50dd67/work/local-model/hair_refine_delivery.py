"""Reversible refinement of existing bound hair, without changing its topology.

Scalp/crown relief, fringe layering and softer tail sections are computed from
a checkpoint. All object names, UVs, weights, rig bindings and spring membership
remain intact. Fine surface fibres are baked to a portable tangent normal map.
"""
from pathlib import Path
import hashlib
import json
import math
import bpy
from mathutils import Vector

VERSION = "coherent-layered-hair-02"
BACKUP = "HairDeliveryOriginal"
MATERIAL = "Character / coherent layered hair"
P = Path(__file__).resolve().parents[2]
CENTER = Vector((0, .004, 1.450))


def smooth(a, b, value):
    value = max(0, min(1, (value-a)/(b-a)))
    return value*value*(3-2*value)


def _objects(collection):
    return [o for o in collection.all_objects if o.type == "MESH"
            and o.name.startswith("Hair_") and not o.name.startswith("Hair_Tie_")]


def _parameters(mesh):
    layer = mesh.uv_layers.get("UVMap")
    if layer is None:
        raise RuntimeError("Hair UVMap missing: "+mesh.name)
    result = {}
    for loop in mesh.loops:
        result.setdefault(loop.vertex_index, tuple(layer.data[loop.index].uv))
    return result


def _rings(parameters):
    groups = {}
    for index, uv in parameters.items():
        groups.setdefault(round(uv[1], 6), []).append(index)
    return [(value, indices) for value, indices in sorted(groups.items())]


def _backup(obj):
    attr = obj.data.attributes.get(BACKUP)
    if attr is None:
        attr = obj.data.attributes.new(BACKUP, "FLOAT_VECTOR", "POINT")
        for vertex in obj.data.vertices:
            attr.data[vertex.index].vector = vertex.co
        obj["hair_delivery_original_materials"] = json.dumps(
            [m.name if m else None for m in obj.data.materials])
        obj["hair_delivery_checkpoint_vertices"] = len(obj.data.vertices)
    if len(attr.data) != len(obj.data.vertices):
        raise RuntimeError("Checkpoint topology changed: "+obj.name)
    for name in json.loads(obj["hair_delivery_original_materials"]):
        if name:
            material = bpy.data.materials.get(name)
            if material is None:
                raise RuntimeError("Checkpoint material is missing: "+name)
            # Hair slot replacement leaves the old material with zero users.
            # Keep it across save/reload so restoration remains reversible.
            material.use_fake_user = True
    return [value.vector.copy() for value in attr.data]


def _scalp(co):
    delta = co-CENTER
    ry = .116 if delta.y >= 0 else .108
    radius = math.sqrt((delta.x/.121)**2+(delta.y/ry)**2+(delta.z/.145)**2)
    p = CENTER+delta/max(radius, 1e-5)
    n = Vector((p.x/.121**2, (p.y-.004)/ry**2, (p.z-1.450)/.145**2)).normalized()
    phi = math.atan2(p.x, -(p.y-.004))
    theta = math.acos(max(-1, min(1, (p.z-1.450)/.145)))
    envelope = smooth(.09, .40, theta)*(1-smooth(1.90, 2.20, theta))
    # Low-relief grouped flow, not separate cylindrical strands. Its amplitude
    # is smaller than one millimetre and smoothly vanishes at the central part.
    flow = 12*phi+1.25*math.sin(theta*1.7)+.3*math.sin(phi*3)
    relief = envelope*(.00030*math.cos(flow)+.00012*math.cos(21*phi-.7*theta))
    return p+n*relief, n


def _longwise_smooth(coords, parameters, strength=.18, count=3):
    rings = _rings(parameters)
    sizes = {len(indices) for _, indices in rings}
    if len(rings) < 6 or len(sizes) != 1:
        return coords
    rings = [sorted(indices, key=lambda i: parameters[i][0]) for _, indices in rings]
    for _ in range(count):
        changed = [co.copy() for co in coords]
        for row in range(1, len(rings)-1):
            amount = strength*math.sin(math.pi*row/(len(rings)-1))
            for column, index in enumerate(rings[row]):
                neighbour = (coords[rings[row-1][column]]+coords[rings[row+1][column]])*.5
                changed[index] = coords[index].lerp(neighbour, amount)
        coords = changed
    return coords


def _geometry(obj, original, parameters):
    name = obj.name
    coords = [co.copy() for co in original]
    if name == "Hair_ScalpFoundation":
        return [_scalp(co)[0] for co in coords]
    if name.startswith("Hair_Crown_") and "_Flow" not in name:
        # Previously buried crown clumps return as very shallow joined layers.
        # Their roots and tips sink into the backing; neither has an exposed cap.
        for index, co in enumerate(coords):
            u, t = parameters[index]
            surface, normal = _scalp(co)
            emergence = smooth(.16, .55, t)*(1-smooth(.76, 1.0, t))
            depth = -.00080+emergence*(.00075+.00030*math.sin(math.tau*u))
            coords[index] = surface+normal*depth
        return coords
    if "_Flow" in name:
        for t, indices in _rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            _, normal = _scalp(center)
            for index in indices:
                coords[index] = center+(coords[index]-center)*.32-normal*.00012
        return coords
    if name.startswith("Hair_BackGather_"):
        for index, co in enumerate(coords):
            u, t = parameters[index]
            surface, normal = _scalp(co)
            emergence = smooth(.08, .28, t)*(1-smooth(.74, .94, t))
            depth = -.00035+emergence*(.00075+.00045*math.sin(math.tau*u))
            blend = 1-smooth(.72, .95, t)
            coords[index] = co.lerp(surface+normal*depth, blend)
        return _longwise_smooth(coords, parameters, .12, 2)
    if name.startswith("Hair_Fringe_"):
        number = int(name.split("_")[2])
        for t, indices in _rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            width_factor = 1-.18*smooth(.42, .92, t)
            for index in indices:
                co = coords[index]
                co.x = center.x+(co.x-center.x)*width_factor
                # A short curved terminal wisp softens the thick triangular
                # silhouette while maintaining the approved clear eye line.
                tip = smooth(.73, 1, t)
                co.x += (.0018*math.sin(number*1.7+.5))*tip
                co.z -= (.0015+.0015*(number % 3)/2)*tip
                surface, normal = _scalp(co)
                root_blend = 1-smooth(.10, .50, t)
                co = co.lerp(surface-normal*.0015, root_blend)
                coords[index] = co
        return _longwise_smooth(coords, parameters, .16, 3)
    if name.startswith(("Hair_Ponytail_", "Hair_TailEdge_", "Hair_Temple_")):
        for t, indices in _rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            factor = 1-.10*smooth(.40, .95, t)
            for index in indices:
                coords[index] = center+(coords[index]-center)*factor
        return _longwise_smooth(coords, parameters, .20, 4)
    if name.startswith("Hair_CrownFlyaway_"):
        for index, co in enumerate(coords):
            u, t = parameters[index]
            surface, normal = _scalp(co)
            coords[index] = surface+normal*(.00055+.00035*math.sin(math.pi*t))
        return coords
    return _longwise_smooth(coords, parameters, .12, 2)


def _normal_texture(base, path):
    """Bake a restrained fibre bump into an exportable tangent normal image."""
    material = bpy.data.materials.new("Hair delivery · temporary normal bake")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = base
    gray = nodes.new("ShaderNodeRGBToBW")
    remap = nodes.new("ShaderNodeMapRange")
    remap.inputs["From Min"].default_value = .015
    remap.inputs["From Max"].default_value = .040
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .35
    bump.inputs["Distance"].default_value = .00006
    links.new(tex.outputs["Color"], gray.inputs[0])
    links.new(gray.outputs[0], remap.inputs["Value"])
    links.new(remap.outputs[0], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    image = bpy.data.images.new("Hair · fine directional normal", width=1254, height=1254)
    image.colorspace_settings.name = "Non-Color"
    target = nodes.new("ShaderNodeTexImage")
    target.image = image
    nodes.active = target
    scene = bpy.data.scenes.new("Hair delivery · isolated normal bake")
    mesh = bpy.data.meshes.new("Hair delivery · UV plane")
    mesh.from_pydata([(-.025, -.075, 0), (.025, -.075, 0),
                      (.025, .075, 0), (-.025, .075, 0)], [], [(0, 1, 2, 3)])
    layer = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        layer.data[loop.index].uv = ((0, 0), (1, 0), (1, 1), (0, 1))[loop.vertex_index]
    obj = bpy.data.objects.new("Hair delivery · UV plane", mesh)
    scene.collection.objects.link(obj)
    mesh.materials.append(material)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    view_layer = scene.view_layers[0]
    view_layer.objects.active = obj
    obj.select_set(True, view_layer=view_layer)
    try:
        with bpy.context.temp_override(scene=scene, view_layer=view_layer,
                                       active_object=obj, object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", margin=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
        image.pack()
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(scene)
        bpy.data.materials.remove(material)
    return image


def _material():
    material = bpy.data.materials.get(MATERIAL)
    if material and material.get("hair_delivery_version") == VERSION:
        return material
    if material:
        material.name = MATERIAL+" previous"
    source = bpy.data.materials.get("Character / polished charcoal hair")
    if source is None:
        raise RuntimeError("Expected portable charcoal hair material is missing.")
    material = source.copy()
    material.name = MATERIAL
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    links = shader.inputs["Base Color"].links
    if not links or links[0].from_node.type != "TEX_IMAGE":
        raise RuntimeError("Hair base colour must already be a portable direct image.")
    base = links[0].from_node.image
    image = _normal_texture(base, P/"textures/hair-fine-normal.png")
    nodes, links = material.node_tree.nodes, material.node_tree.links
    tex = nodes.new("ShaderNodeTexImage")
    tex.name = "Fine directional fibre normal"
    tex.image = image
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    normal = nodes.new("ShaderNodeNormalMap")
    normal.uv_map = "UVMap"
    normal.inputs["Strength"].default_value = .40
    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = .64
    shader.inputs["Specular IOR Level"].default_value = .14
    socket = shader.inputs.get("Anisotropic IOR Level") or shader.inputs.get("Anisotropic")
    if socket:
        socket.default_value = .18
    material["hair_delivery_version"] = VERSION
    return material


def apply(collection=None, rig=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    rig = rig or bpy.data.objects["Character_Rig"]
    objects = _objects(collection)
    if not any(o.name == "Hair_ScalpFoundation" for o in objects):
        raise RuntimeError("Expected original hair scalp is missing.")
    if any(o.data.shape_keys for o in objects):
        raise RuntimeError("Hair unexpectedly has shape keys; checkpoint needs review.")
    for obj in objects:
        if len(obj.data.vertices) and not obj.vertex_groups:
            raise RuntimeError("Unbound hair mesh: "+obj.name)
        armatures = [m for m in obj.modifiers if m.type == "ARMATURE"]
        if len(armatures) != 1 or armatures[0].object != rig:
            raise RuntimeError("Hair rig binding changed: "+obj.name)
    material = _material()
    for obj in objects:
        original = _backup(obj)
        parameters = _parameters(obj.data)
        for vertex, co in zip(obj.data.vertices, _geometry(obj, original, parameters)):
            vertex.co = co
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj.data.update()
        obj["hair_delivery_version"] = VERSION
    return {"version": VERSION, "objects": [o.name for o in objects],
            "material": material.name, "new_meshes": 0,
            "preserved": ["object names", "UVs", "weights", "rig", "spring membership",
                          "colliders", "facial shape keys", "topology"]}


def restore(collection=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    for obj in _objects(collection):
        if obj.data.attributes.get(BACKUP) is not None:
            for name in json.loads(obj["hair_delivery_original_materials"]):
                if name and name not in bpy.data.materials:
                    raise RuntimeError("Cannot restore missing checkpoint material: "+name)
    for obj in _objects(collection):
        attr = obj.data.attributes.get(BACKUP)
        if attr is None:
            continue
        if len(attr.data) != len(obj.data.vertices):
            raise RuntimeError("Cannot restore changed topology: "+obj.name)
        for vertex in obj.data.vertices:
            vertex.co = attr.data[vertex.index].vector
        obj.data.materials.clear()
        for name in json.loads(obj["hair_delivery_original_materials"]):
            if name:
                obj.data.materials.append(bpy.data.materials[name])
        obj.data.update()
        if "hair_delivery_version" in obj:
            del obj["hair_delivery_version"]


def qa(after_only=False):
    out = P/"work/qa-hair-delivery"
    out.mkdir(parents=True, exist_ok=True)
    collection = bpy.data.collections["CHARACTER · Original geometry"]
    objects = _objects(collection)

    def invariant():
        return hashlib.sha256(repr([
            (o.name, len(o.data.vertices), [tuple(p.vertices) for p in o.data.polygons],
             [[tuple(uv.uv) for uv in layer.data] for layer in o.data.uv_layers],
             [[(g.group, g.weight) for g in v.groups] for v in o.data.vertices])
            for o in objects]).encode()).hexdigest()

    original = {o.name: [tuple(v.co) for v in o.data.vertices] for o in objects}
    baseline = invariant()
    report = apply(collection)
    first = {o.name: [tuple(v.co) for v in o.data.vertices] for o in objects}
    apply(collection)
    assert first == {o.name: [tuple(v.co) for v in o.data.vertices] for o in objects}
    assert invariant() == baseline
    restore(collection)
    assert original == {o.name: [tuple(v.co) for v in o.data.vertices] for o in objects}
    apply(collection)
    report["validation"] = {"idempotent": True, "restore_exact": True,
                            "uv_weights_topology_names_preserved": True}
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    # CPU avoids the expensive first-time Metal-kernel build on this host.
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 744
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    camera = bpy.data.objects.get("Hair delivery review camera")
    if camera is None:
        camera = bpy.data.objects.new("Hair delivery review camera",
                                      bpy.data.cameras.new("Hair delivery review camera"))
        scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = .41
    scene.camera = camera
    target = Vector((0, .002, 1.405))
    positions = {"front": Vector((0, -3, 1.405)),
                 "front45": Vector((2.5, -2.5, 1.405)),
                 "side": Vector((3, .002, 1.405))}
    camera.location = positions["front"]
    camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
    clay = bpy.data.materials.new("Hair delivery review clay")
    clay.use_nodes = True
    bsdf = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (.24, .21, .18, 1)
    bsdf.inputs["Roughness"].default_value = .7
    bsdf.inputs["Specular IOR Level"].default_value = .15
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"hair-delivery-review.blend"))
    for state in (("after",) if after_only else ("after", "before")):
        apply(collection) if state == "after" else restore(collection)
        for style in ("textured", "clay"):
            scene.view_layers[0].material_override = clay if style == "clay" else None
            for view, position in positions.items():
                camera.location = position
                camera.rotation_euler = (target-position).to_track_quat("-Z", "Y").to_euler()
                scene.render.filepath = str(out/f"{state}-{view}-{style}.png")
                bpy.ops.render.render(write_still=True)
                print("HAIR_DELIVERY_RENDER", state, view, style, flush=True)
    (out/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("HAIR_DELIVERY_COMPLETE", json.dumps(report["validation"]), flush=True)


if __name__ == "__main__":
    import sys
    if "--hair-delivery-qa" in sys.argv:
        qa(after_only="--after-only" in sys.argv)
