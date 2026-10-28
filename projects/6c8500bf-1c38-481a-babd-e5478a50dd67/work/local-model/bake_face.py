"""Bake the approved-image facial shader into a portable base-color texture.

Intended for character_builder.build_head's 112-column, 101-ring face mesh.
Run bake_face(face_object, "/absolute/project/path/face-basecolor.png", 2048)
inside Blender. This module does not execute work on import.
"""
from pathlib import Path
import math
import uuid

import bpy
import numpy as np


COLUMNS = 112
RINGS = 101


def restore_face_projection(obj):
    """Restore the retained single source material before an intentional rebake."""
    names = list(obj.get("face_bake_original_materials", []))
    if len(names) != 1 or names[0] not in bpy.data.materials:
        raise ValueError("The retained original single face material is unavailable.")
    uv_name = obj.get("face_bake_source_uv", "")
    uv = obj.data.uv_layers.get(uv_name)
    if uv is None:
        raise ValueError("The retained projection UV is unavailable.")
    obj.data.materials.clear()
    obj.data.materials.append(bpy.data.materials[names[0]])
    for poly in obj.data.polygons:
        poly.material_index = 0
    obj.data.uv_layers.active_index = obj.data.uv_layers.find(uv_name)
    uv.active_render = True
    if "face_bake_completed" in obj:
        del obj["face_bake_completed"]


def _delivery_uv(mesh, layer):
    """Latitudinal face island with explicit loop UV seam and two cap islands."""
    for poly in mesh.polygons:
        vertex_indices = [mesh.loops[i].vertex_index for i in poly.loop_indices]
        rings = {i // COLUMNS for i in vertex_indices}
        if len(rings) == 1 and len(vertex_indices) == COLUMNS:
            row = next(iter(rings))
            if row not in (0, RINGS - 1):
                raise ValueError("Unexpected intermediate cap in original face grid.")
            # Reserved bottom margin: disjoint from the main 0.045..0.975 island.
            center_u = 0.25 if row == 0 else 0.75
            for loop_index, vertex_index in zip(poly.loop_indices, vertex_indices):
                angle = math.tau * (vertex_index % COLUMNS) / COLUMNS
                layer.data[loop_index].uv = (
                    center_u + 0.013 * math.sin(angle),
                    0.020 + 0.013 * math.cos(angle),
                )
            continue
        if len(vertex_indices) != 4 or len(rings) != 2:
            raise ValueError("Expected intact original head ring quads and end caps.")
        columns = [i % COLUMNS for i in vertex_indices]
        seam = max(columns) - min(columns) > COLUMNS // 2
        for loop_index, vertex_index in zip(poly.loop_indices, vertex_indices):
            column, row = vertex_index % COLUMNS, vertex_index // COLUMNS
            u = column / COLUMNS
            if seam and column == 0:
                u = 1.0
            layer.data[loop_index].uv = (
                0.02 + 0.96 * u,
                0.045 + 0.93 * row / (RINGS - 1),
            )


def _freeze_source_uv(material, uv_name):
    """Prevent the approved source PNG from following the new active UV layer."""
    if not material or not material.use_nodes:
        raise ValueError("The face source must have its original node material.")
    nodes, links = material.node_tree.nodes, material.node_tree.links
    uv = nodes.new("ShaderNodeUVMap")
    uv.name = "Original approved image projection — fixed UV"
    uv.label = uv_name
    uv.uv_map = uv_name
    frozen = 0
    for node in list(nodes):
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        vector = node.inputs["Vector"]
        if not vector.is_linked:
            links.new(uv.outputs["UV"], vector)
            frozen += 1
        elif all(link.from_node.type == "TEX_COORD" and
                 link.from_socket.name == "UV" for link in vector.links):
            for link in list(vector.links):
                links.remove(link)
            links.new(uv.outputs["UV"], vector)
            frozen += 1
        elif all(link.from_node.type == "UVMAP" and
                 link.from_node.uv_map == uv_name for link in vector.links):
            frozen += 1
        else:
            raise ValueError("Unexpected source image mapping: verify projection UV.")
    return frozen


def bake_face(obj, output_path, resolution=2048):
    """Bake only diffuse COLOR, save/reload/pack PNG, then assign portable material.

The original shader and original projection UV are preserved as datablocks.
On bake failure, original material slots/UV activation are restored. Scene
engine, samples, device, bake settings, visibility, selection and active object
are restored on both success and failure. No lights, camera or geometry change.
"""
    if obj.type != "MESH":
        raise TypeError("bake_face requires the original face mesh object.")
    if len(obj.data.vertices) != COLUMNS * RINGS:
        raise ValueError(
            f"Expected {COLUMNS * RINGS} original grid vertices, got "
            f"{len(obj.data.vertices)}. Do not apply topology modifiers first."
        )
    if obj.get("face_bake_completed"):
        raise ValueError("Already baked; restore the original face shader before rebaking.")
    if not isinstance(resolution, int) or not 256 <= resolution <= 8192:
        raise ValueError("Resolution must be an integer from 256 through 8192.")
    if not obj.data.materials or any(m is None for m in obj.data.materials):
        raise ValueError("Every face material slot must have a source material.")
    bpy.context.view_layer.update()
    if obj.name not in bpy.context.view_layer.objects:
        raise ValueError("The face must be in the active view layer for baking.")
    mesh = obj.data
    projection = (mesh.uv_layers.get("Projection") or
                  mesh.uv_layers.get("UVMap") or mesh.uv_layers.active)
    if projection is None:
        raise ValueError("The original face projection UV layer is missing.")
    if projection.name == "DeliveryUV":
        raise ValueError("An original projection UV is required, not a baked UV.")
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".png":
        raise ValueError("Use a PNG path for the portable face base-color texture.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        destination.stem + ".bake-tmp-" + uuid.uuid4().hex[:10] + ".png"
    )
    original_materials = list(mesh.materials)
    original_polygon_slots = [p.material_index for p in mesh.polygons]
    original_active_uv = mesh.uv_layers.active_index
    original_render_uv = [uv.name for uv in mesh.uv_layers if uv.active_render]
    previous_delivery = mesh.uv_layers.get("DeliveryUV")
    previous_delivery_values = None
    if previous_delivery is not None:
        previous_delivery_values = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        previous_delivery.data.foreach_get("uv", previous_delivery_values)
    layer = previous_delivery or mesh.uv_layers.new(name="DeliveryUV")
    scene = bpy.context.scene
    render = scene.render
    bake = render.bake
    original_settings = {
        "engine": render.engine,
        "samples": scene.cycles.samples,
        "device": scene.cycles.device,
        "bake_type": scene.cycles.bake_type,
    }
    bake_fields = [
        "use_selected_to_active", "use_clear", "use_pass_color",
        "use_pass_direct", "use_pass_indirect", "margin", "margin_type", "target",
    ]
    original_bake = {key: getattr(bake, key) for key in bake_fields}
    original_selection = list(bpy.context.selected_objects)
    original_active = bpy.context.view_layer.objects.active
    original_mode = bpy.context.object.mode if bpy.context.object else "OBJECT"
    original_hidden = obj.hide_get()
    original_hide_render = obj.hide_render
    bake_materials = []
    image = None
    finished = False
    result = None
    try:
        if original_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        _delivery_uv(mesh, layer)
        # Duplicate materials before freezing texture-coordinate inputs.
        for index, source in enumerate(original_materials):
            duplicate = source.copy()
            duplicate.name = source.name + " · bake source"
            bake_materials.append(duplicate)
            frozen = _freeze_source_uv(duplicate, projection.name)
            if frozen == 0:
                raise ValueError("Could not verify the approved image projection UV.")
            mesh.materials[index] = duplicate
        image = bpy.data.images.new(
            "Character · facial base color · portable",
            width=resolution, height=resolution, alpha=False, float_buffer=False,
        )
        image.colorspace_settings.name = "sRGB"
        image.generated_color = (0.0, 0.0, 0.0, 1.0)
        for material in bake_materials:
            nodes = material.node_tree.nodes
            for node in nodes:
                node.select = False
            target = nodes.new("ShaderNodeTexImage")
            target.name = "BAKE TARGET · diffuse color only"
            target.image = image
            target.select = True
            nodes.active = target
        mesh.uv_layers.active_index = mesh.uv_layers.find(layer.name)
        layer.active_render = True
        bpy.ops.object.select_all(action="DESELECT")
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.cycles.bake_type = "DIFFUSE"
        bake.use_selected_to_active = False
        bake.use_clear = True
        bake.use_pass_color = True
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.margin = 16
        bake.margin_type = "EXTEND"
        bake.target = "IMAGE_TEXTURES"
        operation = bpy.ops.object.bake(
            type="DIFFUSE", pass_filter={"COLOR"}, uv_layer=layer.name,
            use_clear=True, margin=16, margin_type="EXTEND",
        )
        if "FINISHED" not in operation:
            raise RuntimeError(f"Face color baking did not finish: {operation}")
        pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, 4)[:, :3]
        if not np.isfinite(rgb).all() or float(rgb.max()) < 1e-5:
            raise RuntimeError("The baked facial image is invalid or completely black.")
        rgb_range = (float(rgb.min()), float(rgb.max()))
        image.file_format = "PNG"
        image.filepath_raw = str(temporary_path)
        image.save()
        if not temporary_path.is_file() or temporary_path.stat().st_size < 128:
            raise RuntimeError("Baked image was not saved successfully.")
        temporary_path.replace(destination)
        image.filepath_raw = str(destination)
        image.source = "FILE"
        image.reload()
        image.pack()
        # A direct image -> Principled graph is understood by glTF and MToon.
        material = bpy.data.materials.new("Character / portable baked facial color")
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        nodes.clear()
        uv = nodes.new("ShaderNodeUVMap")
        uv.uv_map = layer.name
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "EXTEND"
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.inputs["Roughness"].default_value = .72
        principled.inputs["Specular IOR Level"].default_value = .22
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(uv.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], principled.inputs["Base Color"])
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])
        material.diffuse_color = original_materials[0].diffuse_color
        mesh.materials.clear()
        mesh.materials.append(material)
        for poly in mesh.polygons:
            poly.material_index = 0
        obj["face_bake_completed"] = True
        obj["face_bake_original_materials"] = [m.name for m in original_materials]
        obj["face_bake_source_uv"] = projection.name
        obj["face_bake_texture"] = str(destination)
        result = {
            "object": obj.name,
            "texture": str(destination),
            "resolution": [resolution, resolution],
            "sourceUV": projection.name,
            "deliveryUV": layer.name,
            "bakePass": "DIFFUSE_COLOR_ONLY",
            "packed": bool(image.packed_file),
            "fileBytes": destination.stat().st_size,
            "pixelRangeLinear": rgb_range,
        }
        finished = True
    finally:
        render.engine = original_settings["engine"]
        scene.cycles.samples = original_settings["samples"]
        scene.cycles.device = original_settings["device"]
        scene.cycles.bake_type = original_settings["bake_type"]
        for key, value in original_bake.items():
            setattr(bake, key, value)
        if not finished:
            mesh.materials.clear()
            for material in original_materials:
                mesh.materials.append(material)
            for poly, index in zip(mesh.polygons, original_polygon_slots):
                poly.material_index = index
            if previous_delivery is None:
                mesh.uv_layers.remove(layer)
            elif previous_delivery_values is not None:
                layer.data.foreach_set("uv", previous_delivery_values)
            mesh.uv_layers.active_index = original_active_uv
            for uv in mesh.uv_layers:
                uv.active_render = uv.name in original_render_uv
        bpy.ops.object.select_all(action="DESELECT")
        for selected in original_selection:
            if selected.name in bpy.context.view_layer.objects:
                selected.select_set(True)
        bpy.context.view_layer.objects.active = original_active
        obj.hide_set(original_hidden)
        obj.hide_render = original_hide_render
        if original_mode != "OBJECT" and original_active is not None:
            bpy.ops.object.mode_set(mode=original_mode)
        for material in bake_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        if temporary_path.exists():
            temporary_path.unlink()
        if not finished and image is not None and image.users == 0:
            bpy.data.images.remove(image)
    return result
