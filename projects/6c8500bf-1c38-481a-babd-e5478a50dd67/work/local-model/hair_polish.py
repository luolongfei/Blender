"""Reversible polish for the bound local character's original hair.

Only edits tagged hair mesh coordinates and material assignments. Vertex counts,
UVs, vertex-group membership, armature modifiers and the rig remain unchanged.
Call apply(collection, rig) repeatedly safely; restore(collection) restores the
original coordinates and material slots recorded on first application.
"""
import json
import math
from pathlib import Path
import bpy
from mathutils import Vector

VERSION = "charcoal-continuous-roots-05"
ORIGINAL = "HairPolishOriginal"
MATERIAL = "Character / polished charcoal hair"
CENTER = Vector((0.0, 0.004, 1.450))
RADII = Vector((0.121, 0.108, 0.145))


def _smoothstep(low, high, value):
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    return t * t * (3 - 2 * t)


def _radial(co):
    delta = co - CENTER
    # The sculpted back of the head is slightly fuller than the early hair
    # placeholder. The real envelope must clear it, not reveal skin patches.
    radii = Vector((RADII.x, 0.116 if delta.y >= 0 else RADII.y, RADII.z))
    radius = math.sqrt(sum((delta[i] / radii[i]) ** 2 for i in range(3)))
    if radius < 0.01:
        return co.copy(), Vector((0, 0, 1)), 1.0
    surface = CENTER + delta / radius
    normal = Vector(((surface.x - CENTER.x) / radii.x ** 2,
                     (surface.y - CENTER.y) / radii.y ** 2,
                     (surface.z - CENTER.z) / radii.z ** 2)).normalized()
    return surface, normal, radius


def _material():
    mat = bpy.data.materials.get(MATERIAL)
    if mat and mat.get("hair_polish_version") == VERSION:
        return mat
    source = bpy.data.materials.get("Character / longitudinal hair fibres")
    if not source:
        raise RuntimeError("Expected longitudinal hair material is missing.")
    if mat:
        mat.name = MATERIAL + " previous"
    mat = source.copy()
    mat.name = MATERIAL
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    texture = next((node for node in nodes if node.type == "TEX_IMAGE" and node.image), None)
    if not shader or not texture:
        raise RuntimeError("The original hair material has no Principled/image pair.")
    for link in list(links):
        if link.to_node == shader and link.to_socket.name == "Base Color":
            links.remove(link)
    luminance = nodes.new("ShaderNodeRGBToBW")
    luminance.name = "Polish · neutral fibre luminance"
    luminance.location = (-250, 100)
    tone = nodes.new("ShaderNodeValToRGB")
    tone.name = "Polish · quiet charcoal tonal range"
    tone.location = (-40, 100)
    ramp = tone.color_ramp
    ramp.interpolation = "EASE"
    ramp.elements[0].position = 0.010
    ramp.elements[0].color = (0.017, 0.0165, 0.0155, 1)
    ramp.elements[1].position = 0.105
    ramp.elements[1].color = (0.034, 0.032, 0.029, 1)
    mid = ramp.elements.new(0.045)
    mid.color = (0.0250, 0.0240, 0.0225, 1)
    links.new(texture.outputs["Color"], luminance.inputs[0])
    links.new(luminance.outputs[0], tone.inputs[0])
    links.new(tone.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = 0.69
    shader.inputs["Specular IOR Level"].default_value = 0.12
    anisotropic = shader.inputs.get("Anisotropic IOR Level") or shader.inputs.get("Anisotropic")
    if anisotropic:
        anisotropic.default_value = 0.12
    mat.diffuse_color = (0.025, 0.024, 0.0225, 1)
    mat["hair_polish_version"] = VERSION
    mat["hair_polish_note"] = "Neutral charcoal; subdued retained source strand variation"
    return mat


def _backup(obj):
    data = obj.data
    attribute = data.attributes.get(ORIGINAL)
    if not attribute:
        attribute = data.attributes.new(ORIGINAL, "FLOAT_VECTOR", "POINT")
        for vertex in data.vertices:
            attribute.data[vertex.index].vector = vertex.co
        obj["hair_polish_original_materials"] = json.dumps(
            [mat.name if mat else None for mat in data.materials], ensure_ascii=False)
    if len(attribute.data) != len(data.vertices):
        raise RuntimeError("Hair topology changed after polish backup: " + obj.name)
    return [datum.vector.copy() for datum in attribute.data]


def _uv_parameters(mesh):
    uv = mesh.uv_layers.get("UVMap")
    if not uv:
        raise RuntimeError("Hair UVMap missing: " + mesh.name)
    values = {}
    for loop in mesh.loops:
        values.setdefault(loop.vertex_index, tuple(uv.data[loop.index].uv))
    return values


def _ring_smooth(original, uv, strength=0.12, iterations=2):
    """Smooth longwise curvature while preserving independent ring cross-sections."""
    groups = {}
    for index, pair in uv.items():
        groups.setdefault(round(pair[1], 6), []).append(index)
    rings = [groups[key] for key in sorted(groups)]
    if len(rings) < 6 or len(set(len(ring) for ring in rings)) != 1:
        return [co.copy() for co in original]
    count = len(rings[0])
    if count not in (8, 10, 12, 16):
        return [co.copy() for co in original]
    rings = [sorted(ring, key=lambda index: uv[index][0]) for ring in rings]
    coords = [co.copy() for co in original]
    for _ in range(iterations):
        updated = [co.copy() for co in coords]
        for i in range(1, len(rings) - 1):
            t = i / (len(rings) - 1)
            fade = math.sin(math.pi * t)
            for j, index in enumerate(rings[i]):
                average = (coords[rings[i - 1][j]] + coords[rings[i + 1][j]]) * 0.5
                updated[index] = coords[index].lerp(average, strength * fade)
        coords = updated
    return coords


def _separate_rear_part(coords, parameters):
    """Keep opposite rear gather leaves on their own side of the centre part."""
    groups = {}
    for index, pair in parameters.items():
        groups.setdefault(round(pair[1], 6), []).append(index)
    for indices in groups.values():
        center_x = sum(coords[index].x for index in indices) / len(indices)
        width = max(abs(coords[index].x - center_x) for index in indices)
        if width <= 1e-7:
            continue
        factor = min(1.0, max(0.04, (abs(center_x) - 0.0005) / width))
        for index in indices:
            coords[index].x = center_x + (coords[index].x - center_x) * factor
    return coords


def _polished_position(name, co, uv):
    u, t = uv
    surface, normal, _ = _radial(co)
    # Original upper clumps were swept flattened tubes. Their transverse surface
    # can cut through neighbouring locks, exposing cap-like corners. In the
    # rooted area, conform both sides to the same scalp envelope at low relief.
    section_top = math.sin(math.tau * u)
    shallow = 0.00020 + section_top * 0.00030
    if name == "Hair_ScalpFoundation":
        phi = math.tau * u
        theta = math.acos(max(-1, min(1, (co.z - CENTER.z) / RADII.z)))
        # Raise only the hidden front edge; the old edge sliced through the
        # free-hanging fringe at mid-length and produced the visible zigzag.
        theta -= 0.20 * max(0, math.cos(phi)) ** 4 * t ** 3
        ry = 0.116 if math.cos(phi) < 0 else RADII.y
        return Vector((RADII.x * math.sin(theta) * math.sin(phi),
                       CENTER.y - ry * math.sin(theta) * math.cos(phi),
                       CENTER.z + RADII.z * math.cos(theta)))
    if name.startswith("Hair_CrownFlyaway_"):
        return surface + normal * (0.0009 + 0.00020 * section_top)
    if name.startswith("Hair_Crown_"):
        # These old coarse crown strips now remain below the continuous cap.
        # Keep them and their bindings for a reversible topology-preserving edit.
        lift = -0.0012 + 0.00010 * section_top
        return surface + normal * lift
    if name.startswith("Hair_Fringe_"):
        # All adjacent fringe locks share one smooth forehead depth field.
        # Their original varying tube frames crossed each other mid-fringe,
        # creating the conspicuous staircase even with a smooth centreline.
        q = max(0.0, 1 - (co.x / RADII.x) ** 2 - ((co.z - CENTER.z) / RADII.z) ** 2)
        y = CENTER.y - RADII.y * math.sqrt(q)
        root = _smoothstep(0.02, 0.42, t)
        lift = -0.0010 + 0.0048 * root
        index = int(name.split("_")[2])
        # Fixed tiny layer separation avoids coincident overlapping surfaces.
        lift += root * 0.00035 * index
        return Vector((co.x, y - lift - 0.00012 * section_top, co.z))
    if name.startswith("Hair_Temple_"):
        blend = 1 - _smoothstep(0.28, 0.59, t)
        return co.lerp(surface + normal * shallow, blend)
    if name.startswith("Hair_BackGather_"):
        blend = 1 - _smoothstep(0.75, 0.93, t)
        q = max(0.0, 1 - (co.x / RADII.x) ** 2 - ((co.z - CENTER.z) / RADII.z) ** 2)
        index = int(name.rsplit("_", 1)[1])
        lift = -0.001 + (0.0025 + index * 0.0011) * _smoothstep(0.01, 0.18, t)
        target = Vector((co.x, CENTER.y + 0.116 * math.sqrt(q)
                         + lift + 0.00015 * section_top, co.z))
        return co.lerp(target, blend)
    if name.startswith("Hair_NapeWisp_"):
        blend = 1 - _smoothstep(0.18, 0.48, t)
        result = co.lerp(surface + normal * 0.00035, blend)
        result.z += 0.010 * t * t
        return result
    return co


def apply(collection=None, rig=None):
    if collection is None:
        collection = bpy.data.collections["CHARACTER · Original geometry"]
    hair = [obj for obj in collection.all_objects
            if obj.type == "MESH" and obj.get("part") == "hair"
            and obj.get("hair_region") != "tie"]
    if not hair:
        raise RuntimeError("No original tagged hair meshes in collection.")
    material = _material()
    changed = []
    for obj in hair:
        original = _backup(obj)
        parameters = _uv_parameters(obj.data)
        coords = _ring_smooth(original, parameters)
        if obj.name.startswith("Hair_BackGather_"):
            coords = _separate_rear_part(coords, parameters)
        for vertex in obj.data.vertices:
            uv = parameters.get(vertex.index, (0.0, 0.0))
            vertex.co = _polished_position(obj.name, coords[vertex.index], uv)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj["hair_polish_version"] = VERSION
        changed.append(obj.name)
    return {"version": VERSION, "objects": changed, "material": material.name,
            "preserved": ["vertex counts", "UV maps", "weights", "armature", "source artwork"]}


def restore(collection=None):
    if collection is None:
        collection = bpy.data.collections["CHARACTER · Original geometry"]
    restored = []
    for obj in collection.all_objects:
        if obj.type != "MESH" or obj.get("part") != "hair":
            continue
        attr = obj.data.attributes.get(ORIGINAL)
        if not attr:
            continue
        if len(attr.data) != len(obj.data.vertices):
            raise RuntimeError("Cannot restore changed topology: " + obj.name)
        for vertex in obj.data.vertices:
            vertex.co = attr.data[vertex.index].vector
        names = json.loads(obj.get("hair_polish_original_materials", "[]"))
        obj.data.materials.clear()
        for name in names:
            if name and bpy.data.materials.get(name):
                obj.data.materials.append(bpy.data.materials[name])
        obj.data.update()
        if "hair_polish_version" in obj:
            del obj["hair_polish_version"]
        restored.append(obj.name)
    return restored


def bake_portable_material(filepath=None):
    """Bake the exact charcoal colour into a packed, exportable base-colour map.

    Uses an isolated UV plane and scene. Neither the rigged meshes nor their UVs
    are modified. The resulting material has a direct image-to-Principled link
    so glTF and MToon conversion can retain its colour.
    """
    material = bpy.data.materials.get(MATERIAL)
    if material is None or material.get("hair_polish_version") != VERSION:
        raise RuntimeError("Apply the current hair polish before baking.")
    if material.get("hair_polish_portable_version") == VERSION:
        image = bpy.data.images.get(material.get("hair_polish_portable_image", ""))
        if image:
            return {"image": image.name, "filepath": image.filepath,
                    "packed": bool(image.packed_file), "reused": True}
    if filepath is None:
        filepath = Path(__file__).resolve().parents[2] / "textures/hair-charcoal-basecolor.png"
    filepath = Path(filepath).resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    source = next(node for node in nodes if node.type == "TEX_IMAGE" and node.image)
    width, height = source.image.size[:]
    image = bpy.data.images.new("Hair · portable charcoal base colour",
                                width=width, height=height, alpha=False)
    image.colorspace_settings.name = "sRGB"
    target = nodes.new("ShaderNodeTexImage")
    target.name = "Polish · portable charcoal image"
    target.image = image
    target.location = (-300, -200)
    for node in nodes:
        node.select = False
    target.select = True
    nodes.active = target
    scene = bpy.data.scenes.new("Hair polish · isolated colour bake")
    mesh = bpy.data.meshes.new("Hair polish · temporary UV square")
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
                     [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new(name="UVMap")
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for loop in mesh.loops:
        uv.data[loop.index].uv = corners[loop.vertex_index]
    obj = bpy.data.objects.new("Hair polish · temporary UV square", mesh)
    scene.collection.objects.link(obj)
    mesh.materials.append(material)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.margin = 2
    layer = scene.view_layers[0]
    layer.objects.active = obj
    obj.select_set(True, view_layer=layer)
    try:
        with bpy.context.temp_override(scene=scene, view_layer=layer, object=obj,
                                       active_object=obj, selected_objects=[obj],
                                       selected_editable_objects=[obj]):
            bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"},
                               use_clear=True, margin=2)
        image.filepath_raw = str(filepath)
        image.file_format = "PNG"
        image.save()
        image.pack()
        for link in list(links):
            if link.to_node == shader and link.to_socket.name == "Base Color":
                links.remove(link)
        uv_node = nodes.new("ShaderNodeUVMap")
        uv_node.name = "Polish · retained UVMap"
        uv_node.uv_map = "UVMap"
        uv_node.location = (-520, -200)
        links.new(uv_node.outputs["UV"], target.inputs["Vector"])
        links.new(target.outputs["Color"], shader.inputs["Base Color"])
        material["hair_polish_portable_version"] = VERSION
        material["hair_polish_portable_image"] = image.name
        return {"image": image.name, "filepath": str(filepath),
                "packed": bool(image.packed_file), "size": [width, height],
                "direct_base_color_image": True, "reused": False}
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(scene)
