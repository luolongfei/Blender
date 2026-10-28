"""Bake separate selected meshes into one portable diffuse-color atlas.

No work executes on import.  Call bake_selected_atlas(objects, png_path, 2048)
inside Blender.  Original objects, metadata, modifiers, source materials, and
source UVs remain available; only a new packed UV and output material are used.
"""
import math
from pathlib import Path
import uuid

import bpy
import numpy as np


def _freeze_uv(tree, uv_name, owned_groups):
    """Pin implicit source UVs, including through Mapping nodes and node groups."""
    uv_node = None

    def source_uv():
        nonlocal uv_node
        if not uv_name:
            raise ValueError("A source shader uses UV coordinates but has no source UV.")
        if uv_node is None:
            uv_node = tree.nodes.new("ShaderNodeUVMap")
            uv_node.name = "Original source UV · pinned during atlas bake"
            uv_node.uv_map = uv_name
        return uv_node.outputs["UV"]

    for node in list(tree.nodes):
        if node.type == "GROUP" and node.node_tree:
            node.node_tree = node.node_tree.copy()
            owned_groups.append(node.node_tree)
            _freeze_uv(node.node_tree, uv_name, owned_groups)
        elif node.type == "TEX_COORD":
            for link in list(node.outputs["UV"].links):
                destination = link.to_socket
                tree.links.remove(link)
                tree.links.new(source_uv(), destination)
        elif node.type == "UVMAP" and not node.uv_map:
            if not uv_name:
                raise ValueError("An unnamed source UV node has no original UV layer.")
            node.uv_map = uv_name
        elif node.type == "TEX_IMAGE" and node.image and not node.inputs["Vector"].is_linked:
            tree.links.new(source_uv(), node.inputs["Vector"])


def restore_atlas_sources(objects):
    """Restore retained material slots and active source UV before a deliberate rebake."""
    for obj in objects:
        names = list(obj.get("atlas_source_materials", []))
        indices = list(obj.get("atlas_source_polygon_materials", []))
        if not names or any(name not in bpy.data.materials for name in names):
            raise ValueError(f"{obj.name}: source materials are unavailable.")
        if len(indices) != len(obj.data.polygons):
            raise ValueError(f"{obj.name}: mesh topology changed since the atlas bake.")
        obj.data.materials.clear()
        for name in names:
            obj.data.materials.append(bpy.data.materials[name])
        for poly, index in zip(obj.data.polygons, indices):
            poly.material_index = index
        source_uv = obj.get("atlas_source_uv", "")
        if source_uv:
            if source_uv not in obj.data.uv_layers:
                raise ValueError(f"{obj.name}: source UV is unavailable.")
            obj.data.uv_layers.active_index = obj.data.uv_layers.find(source_uv)
            obj.data.uv_layers[source_uv].active_render = True
        if "atlas_bake_completed" in obj:
            del obj["atlas_bake_completed"]


def bake_selected_atlas(objs, output_path, resolution=2048):
    """Globally pack mesh UVs and bake all selected objects in one operation.

The shared target image is cleared only once by the single object.bake call.
Diffuse direct/indirect lighting is excluded. Roughness/metallic become one
surface-area-weighted scalar each; procedural normal/bump is not color data and
is not included by this function.
"""
    objects = list(dict.fromkeys(objs))
    if not objects or any(o.type != "MESH" or not o.data.polygons for o in objects):
        raise ValueError("Supply a nonempty set of polygon mesh objects.")
    if not isinstance(resolution, int) or not 256 <= resolution <= 8192:
        raise ValueError("Resolution must be an integer from 256 through 8192.")
    destination = Path(output_path).resolve()
    if destination.suffix.lower() != ".png":
        raise ValueError("The atlas destination must be a PNG file.")
    bpy.context.view_layer.update()
    for obj in objects:
        if obj.name not in bpy.context.view_layer.objects:
            raise ValueError(f"{obj.name}: not present in the active view layer.")
        if obj.get("atlas_bake_completed"):
            raise ValueError(f"{obj.name}: restore its atlas sources before rebaking.")
        if not obj.data.materials or any(m is None for m in obj.data.materials):
            raise ValueError(f"{obj.name}: every mesh material slot needs a material.")
        if any(slot.link != "DATA" for slot in obj.material_slots):
            raise ValueError(f"{obj.name}: object-linked material slots need explicit handling.")
    scene, view_layer = bpy.context.scene, bpy.context.view_layer
    original_active = view_layer.objects.active
    original_selection = list(bpy.context.selected_objects)
    original_mode = bpy.context.object.mode if bpy.context.object else "OBJECT"
    original_engine = scene.render.engine
    original_cycles = {key: getattr(scene.cycles, key) for key in
                       ("device", "samples", "bake_type")}
    bake = scene.render.bake
    bake_fields = ("use_selected_to_active", "use_clear", "use_pass_color",
                   "use_pass_direct", "use_pass_indirect", "margin",
                   "margin_type", "target")
    original_bake = {key: getattr(bake, key) for key in bake_fields}
    tool_settings = scene.tool_settings
    original_sync = tool_settings.use_uv_select_sync
    original_mesh_mode = tuple(tool_settings.mesh_select_mode)
    records, copies, groups = [], {}, []
    image = None
    output_material = None
    finished = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.stem + ".bake-tmp-" + uuid.uuid4().hex[:10] + ".png"
    )
    report = None
    try:
        if original_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        # Keep any shared original mesh datablocks untouched.
        for obj in objects:
            old_mesh = obj.data
            if old_mesh.users > 1:
                obj.data = old_mesh.copy()
            mesh = obj.data
            source_uv = next((u for u in mesh.uv_layers if u.active_render),
                             mesh.uv_layers.active)
            source_uv_name = source_uv.name if source_uv else None
            existing = mesh.uv_layers.get("DeliveryUV")
            previous_uv = None
            if existing:
                previous_uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
                existing.data.foreach_get("uv", previous_uv)
            selections = {}
            for attr in ("vertices", "edges", "polygons"):
                values = np.empty(len(getattr(mesh, attr)), dtype=np.bool_)
                getattr(mesh, attr).foreach_get("select", values)
                selections[attr] = values
            rec = {
                "object": obj, "oldMesh": old_mesh, "mesh": mesh,
                "materials": list(mesh.materials),
                "indices": [p.material_index for p in mesh.polygons],
                "activeUV": mesh.uv_layers.active_index,
                "renderUV": [u.name for u in mesh.uv_layers if u.active_render],
                "sourceUV": source_uv_name, "previousUV": previous_uv,
                "existingDelivery": existing is not None,
                "hidden": obj.hide_get(), "hideRender": obj.hide_render,
                "selections": selections,
            }
            records.append(rec)
            # Material copies are keyed by UV name because objects may differ.
            for index, source in enumerate(rec["materials"]):
                key = (source.as_pointer(), source_uv_name)
                if key not in copies:
                    material = source.copy()
                    copies[key] = material
                    material.name = source.name + " · atlas bake source"
                    if not material.use_nodes:
                        material.use_nodes = True
                        principled = next(n for n in material.node_tree.nodes
                                          if n.type == "BSDF_PRINCIPLED")
                        principled.inputs["Base Color"].default_value = source.diffuse_color
                    _freeze_uv(material.node_tree, source_uv_name, groups)
                mesh.materials[index] = copies[key]
            layer = existing or mesh.uv_layers.new(name="DeliveryUV")
            rec["delivery"] = layer
            mesh.uv_layers.active_index = mesh.uv_layers.find(layer.name)
            layer.active_render = True
            obj.hide_set(False)
            obj.hide_render = False
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        view_layer.objects.active = objects[0]
        tool_settings.use_uv_select_sync = False
        tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66),
            island_margin=0.0,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False,
        )
        bpy.ops.uv.select_all(action="SELECT")
        packed = bpy.ops.uv.pack_islands(
            udim_source="CLOSEST_UDIM",
            rotate=True, scale=True, merge_overlap=False,
            margin_method="FRACTION", margin=10.0 / resolution,
            shape_method="CONCAVE",
        )
        if "FINISHED" not in packed:
            raise RuntimeError(f"Global atlas packing did not finish: {packed}")
        bpy.ops.object.mode_set(mode="OBJECT")
        # Edit-mode BMesh flush can invalidate previously held UV layer RNA.
        for rec in records:
            rec["delivery"] = rec["mesh"].uv_layers["DeliveryUV"]
        image = bpy.data.images.new(
            destination.stem + " · portable color atlas",
            width=resolution, height=resolution, alpha=True, float_buffer=False,
        )
        image.generated_color = (0.0, 0.0, 0.0, 0.0)
        image.colorspace_settings.name = "sRGB"
        for material in copies.values():
            for node in material.node_tree.nodes:
                node.select = False
            node = material.node_tree.nodes.new("ShaderNodeTexImage")
            node.name = "ATLAS BAKE TARGET · diffuse color only"
            node.image = image
            node.select = True
            material.node_tree.nodes.active = node
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.cycles.bake_type = "DIFFUSE"
        bake.use_selected_to_active = False
        bake.use_clear = True
        bake.use_pass_color = True
        bake.use_pass_direct = bake.use_pass_indirect = False
        bake.margin = 8
        bake.margin_type = "EXTEND"
        bake.target = "IMAGE_TEXTURES"
        # A single call is essential: clearing for each object would erase peers.
        result = bpy.ops.object.bake(
            type="DIFFUSE", pass_filter={"COLOR"}, uv_layer="DeliveryUV",
            use_selected_to_active=False, use_clear=True,
            margin=8, margin_type="EXTEND",
        )
        if "FINISHED" not in result:
            raise RuntimeError(f"Atlas color baking did not finish: {result}")
        pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgba = pixels.reshape(resolution, resolution, 4)
        if not np.isfinite(rgba).all():
            raise RuntimeError("The baked atlas contains non-finite pixel values.")
        samples = []
        weight, roughness, metallic = 0.0, 0.0, 0.0
        for rec in records:
            mesh = rec["mesh"]
            colors = []
            # Show sampled object coverage; black is allowed as authored color.
            step = max(1, len(mesh.polygons) // 24)
            for poly in list(mesh.polygons)[::step][:24]:
                uv = np.mean([tuple(rec["delivery"].data[i].uv)
                              for i in poly.loop_indices], axis=0)
                x = min(resolution - 1, max(0, int(uv[0] * resolution)))
                y = min(resolution - 1, max(0, int(uv[1] * resolution)))
                colors.append(rgba[y, x, :].tolist())
            samples.append({"object": rec["object"].name,
                            "sampleCount": len(colors),
                            "meanUVSampleRGBA": np.mean(colors, axis=0).tolist()})
            for poly, material_index in zip(mesh.polygons, rec["indices"]):
                material = rec["materials"][material_index]
                node = next((n for n in material.node_tree.nodes
                             if n.type == "BSDF_PRINCIPLED"), None) if material.use_nodes else None
                area = max(1e-12, poly.area)
                weight += area
                roughness += area * (node.inputs["Roughness"].default_value if node else .6)
                metallic += area * (node.inputs["Metallic"].default_value if node else 0)
        image.file_format = "PNG"
        image.filepath_raw = str(temporary)
        image.save()
        if not temporary.is_file() or temporary.stat().st_size < 128:
            raise RuntimeError("The atlas PNG was not saved successfully.")
        temporary.replace(destination)
        image.filepath_raw = str(destination)
        image.source = "FILE"
        image.reload()
        image.pack()
        output_material = bpy.data.materials.new(destination.stem + " · portable material")
        output_material.use_nodes = True
        nodes, links = output_material.node_tree.nodes, output_material.node_tree.links
        nodes.clear()
        uv = nodes.new("ShaderNodeUVMap")
        uv.uv_map = "DeliveryUV"
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.extension = "EXTEND"
        texture.interpolation = "Linear"
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = roughness / weight
        shader.inputs["Metallic"].default_value = metallic / weight
        shader.inputs["Specular IOR Level"].default_value = .22
        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(uv.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], shader.inputs["Base Color"])
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])
        for rec in records:
            obj, mesh = rec["object"], rec["mesh"]
            obj["atlas_source_materials"] = [m.name for m in rec["materials"]]
            obj["atlas_source_polygon_materials"] = rec["indices"]
            obj["atlas_source_uv"] = rec["sourceUV"] or ""
            obj["atlas_bake_texture"] = str(destination)
            obj["atlas_bake_completed"] = True
            mesh.materials.clear()
            mesh.materials.append(output_material)
            for poly in mesh.polygons:
                poly.material_index = 0
        report = {
            "texture": str(destination),
            "resolution": [resolution, resolution],
            "objects": len(objects), "materials": 1,
            "bakeOperations": 1, "clearOperations": 1,
            "bakePass": "DIFFUSE_COLOR_ONLY",
            "packed": bool(image.packed_file),
            "fileBytes": destination.stat().st_size,
            "roughness": roughness / weight,
            "metallic": metallic / weight,
            "objectUVSamples": samples,
        }
        finished = True
    finally:
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine = original_engine
        for key, value in original_cycles.items():
            setattr(scene.cycles, key, value)
        for key, value in original_bake.items():
            setattr(bake, key, value)
        tool_settings.use_uv_select_sync = original_sync
        tool_settings.mesh_select_mode = original_mesh_mode
        for rec in records:
            obj, mesh = rec["object"], rec["mesh"]
            if not finished:
                if mesh != rec["oldMesh"]:
                    obj.data = rec["oldMesh"]
                    if mesh.users == 0:
                        bpy.data.meshes.remove(mesh)
                else:
                    mesh.materials.clear()
                    for material in rec["materials"]:
                        mesh.materials.append(material)
                    for poly, index in zip(mesh.polygons, rec["indices"]):
                        poly.material_index = index
                    layer = mesh.uv_layers.get("DeliveryUV")
                    if layer and not rec["existingDelivery"]:
                        mesh.uv_layers.remove(layer)
                    elif layer and rec["previousUV"] is not None:
                        layer.data.foreach_set("uv", rec["previousUV"])
                    mesh.uv_layers.active_index = rec["activeUV"]
                    for uv in mesh.uv_layers:
                        uv.active_render = uv.name in rec["renderUV"]
            if finished or mesh == rec["oldMesh"]:
                for attr, values in rec["selections"].items():
                    getattr(obj.data, attr).foreach_set("select", values)
            obj.hide_set(rec["hidden"])
            obj.hide_render = rec["hideRender"]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in original_selection:
            if obj.name in view_layer.objects:
                obj.select_set(True)
        view_layer.objects.active = original_active
        if original_mode != "OBJECT" and original_active:
            bpy.ops.object.mode_set(mode=original_mode)
        for material in copies.values():
            if material.users == 0:
                bpy.data.materials.remove(material)
        for group in reversed(groups):
            if group.users == 0:
                bpy.data.node_groups.remove(group)
        if not finished and output_material and output_material.users == 0:
            bpy.data.materials.remove(output_material)
        if not finished and image and image.users == 0:
            bpy.data.images.remove(image)
        if temporary.exists():
            temporary.unlink()
    return report
