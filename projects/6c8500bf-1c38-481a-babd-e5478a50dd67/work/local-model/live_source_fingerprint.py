"""Read-only model fingerprint for a saved Blend or the current Blender session.

API: fingerprint(collection=None) returns modelSha256 and detailed components.
write(path, collection=None) writes that result without saving/modifying a Blend.
CLI: blender -b input.blend --python this.py -- --output snapshot.json
"""
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import bpy

COLLECTION = "CHARACTER · Original geometry"
VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def scalar(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, set):
        return sorted(scalar(item) for item in value)
    if isinstance(value, bpy.types.ID):
        return {"idType": value.bl_rna.identifier, "name": value.name_full,
                "library": value.library.filepath if value.library else None}
    if hasattr(value, "to_list"):
        return scalar(value.to_list())
    if hasattr(value, "items"):
        return {str(k): scalar(v) for k, v in value.items()}
    try:
        return [scalar(v) for v in value]
    except TypeError:
        return str(value)


def custom(owner):
    return {key: scalar(owner[key]) for key in sorted(owner.keys()) if key != "_RNA_UI"}


RNA_SKIP = {"rna_type", "id_data", "original", "is_evaluated", "is_runtime_data"}


def rna(owner, depth=0, skip=()):
    """Serialize settings, including nested PropertyGroups; ID links by name."""
    if owner is None:
        return None
    if depth > 16:
        raise ValueError("RNA nesting exceeds inspected scope: " + owner.bl_rna.identifier)
    result = {"rnaType": owner.bl_rna.identifier}
    active_collider = owner.bl_rna.identifier == "SpringBone1ColliderPropertyGroup"
    for prop in owner.bl_rna.properties:
        name = prop.identifier
        if name in RNA_SKIP or name in skip or (active_collider and name in {"shape", "extensions"}):
            continue
        try:
            value = getattr(owner, name)
        except (AttributeError, RuntimeError):
            continue
        if prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            result[name] = scalar(value)
        elif prop.type == "POINTER":
            if value is None or isinstance(value, bpy.types.ID):
                result[name] = scalar(value)
            elif isinstance(value, bpy.types.PropertyGroup):
                result[name] = rna(value, depth + 1)
        elif prop.type == "COLLECTION":
            result[name] = [rna(v, depth + 1) for v in value]
    if active_collider:
        shape_name = "sphere" if owner.shape_type == "Sphere" else "capsule"
        result["shape"] = {shape_name: rna(getattr(owner.shape, shape_name), depth + 1)}
        extended = owner.extensions.vrmc_spring_bone_extended_collider
        result["extendedColliderEnabled"] = extended.enabled
        if extended.enabled:
            result["extensions"] = rna(owner.extensions, depth + 1)
    return result


def buffer_hash(collection, attr, width, dtype="f"):
    values = array.array(dtype, [0]) * (len(collection) * width)
    if values:
        collection.foreach_get(attr, values)
    if sys.byteorder != "little":
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def attributes(mesh):
    result = {}
    for attr in mesh.attributes:
        fields = {}
        if len(attr.data):
            for prop in attr.data[0].bl_rna.properties:
                if prop.identifier == "rna_type" or prop.type not in {"FLOAT", "INT", "BOOLEAN"}:
                    continue
                dtype = "f" if prop.type == "FLOAT" else "i"
                width = prop.array_length if prop.is_array else 1
                fields[prop.identifier] = buffer_hash(attr.data, prop.identifier, width, dtype)
        result[attr.name] = {"domain": attr.domain, "type": attr.data_type,
                            "length": len(attr.data), "fields": fields}
    return result


def mesh_state(obj):
    mesh = obj.data
    weights = hashlib.sha256()
    for vertex in mesh.vertices:
        groups = sorted((g.group, g.weight) for g in vertex.groups)
        weights.update(struct.pack("<I", len(groups)))
        for group, weight in groups:
            weights.update(struct.pack("<If", group, weight))
    shape = mesh.shape_keys
    keys = None if shape is None else {
        "use_relative": shape.use_relative,
        "eval_time": shape.eval_time,
        "custom": custom(shape),
        "keys": [{
            "name": key.name, "value": key.value, "mute": key.mute,
            "relative": key.relative_key.name if key.relative_key else None,
            "slider_min": key.slider_min, "slider_max": key.slider_max,
            "vertex_group": key.vertex_group, "interpolation": key.interpolation,
            "coordinates": buffer_hash(key.data, "co", 3),
        } for key in shape.key_blocks],
        "animation": animation_state(shape),
    }
    return {
        "dataName": mesh.name,
        "counts": [len(mesh.vertices), len(mesh.edges), len(mesh.loops), len(mesh.polygons)],
        "coordinates": buffer_hash(mesh.vertices, "co", 3),
        "edges": buffer_hash(mesh.edges, "vertices", 2, "i"),
        "loopVertices": buffer_hash(mesh.loops, "vertex_index", 1, "i"),
        "loopEdges": buffer_hash(mesh.loops, "edge_index", 1, "i"),
        "polygonLoopStart": buffer_hash(mesh.polygons, "loop_start", 1, "i"),
        "polygonLoopTotal": buffer_hash(mesh.polygons, "loop_total", 1, "i"),
        "materialIndices": buffer_hash(mesh.polygons, "material_index", 1, "i"),
        "smooth": buffer_hash(mesh.polygons, "use_smooth", 1, "i"),
        "attributes": attributes(mesh),
        "uvLayers": [{"name": u.name, "active_render": u.active_render,
                      "active_clone": u.active_clone} for u in mesh.uv_layers],
        "vertexGroups": [{"name": g.name, "index": g.index, "lock_weight": g.lock_weight}
                         for g in obj.vertex_groups],
        "weights": weights.hexdigest(),
        "shapeKeys": keys,
        "custom": custom(mesh),
    }


def animation_state(owner):
    data = owner.animation_data
    if not data:
        return None
    return rna(data, skip={"action_slot_handle"})


def armature_state(obj):
    data = obj.data
    result = {
        "pose_position": data.pose_position,
        "custom": custom(data),
        "rest": {},
        "pose": {},
    }
    bone_skip = {"children", "children_recursive", "collections", "color",
                 "select", "select_head", "select_tail", "hide", "hide_select",
                 "show_wire", "use_deform"}  # use_deform is captured explicitly below.
    for bone in data.bones:
        result["rest"][bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "matrix_local": scalar(bone.matrix_local),
            "head_local": scalar(bone.head_local), "tail_local": scalar(bone.tail_local),
            "use_connect": bone.use_connect, "use_deform": bone.use_deform,
            "inherit_scale": bone.inherit_scale, "custom": custom(bone),
            "settings": rna(bone, skip=bone_skip | {"parent"}),
        }
    for bone in obj.pose.bones:
        result["pose"][bone.name] = {
            "location": scalar(bone.location), "rotation_mode": bone.rotation_mode,
            "rotation_euler": scalar(bone.rotation_euler),
            "rotation_quaternion": scalar(bone.rotation_quaternion),
            "rotation_axis_angle": scalar(bone.rotation_axis_angle),
            "scale": scalar(bone.scale), "matrix_basis": scalar(bone.matrix_basis),
            "constraints": [rna(c) for c in bone.constraints],
            "custom": custom(bone),
        }
    if hasattr(data, "vrm_addon_extension"):
        result["vrm"] = rna(data.vrm_addon_extension)
    return result


def socket_state(socket):
    result = {"name": socket.name, "identifier": socket.identifier,
              "type": socket.bl_idname}
    if hasattr(socket, "default_value"):
        result["default_value"] = scalar(socket.default_value)
    return result


def tree_state(tree, images, trees, key):
    # Embedded material node trees all use "Shader Nodetree"; owner is identity.
    if key in trees:
        return
    trees[key] = None  # Guard reused/cyclic node groups.
    node_skip = {"location", "location_absolute", "width", "height", "dimensions", "select",
                 "show_options", "show_preview", "show_texture", "hide",
                 "color", "use_custom_color", "label", "parent",
                 "inputs", "outputs", "internal_links", "rna_type"}
    nodes = {}
    for node in tree.nodes:
        entry = rna(node, skip=node_skip)
        entry["custom"] = custom(node)
        entry["inputs"] = [socket_state(s) for s in node.inputs]
        entry["outputs"] = [socket_state(s) for s in node.outputs]
        if getattr(node, "image", None):
            images[node.image.name] = node.image
        if getattr(node, "node_tree", None):
            tree_state(node.node_tree, images, trees, "group:" + node.node_tree.name)
        if hasattr(node, "color_ramp"):
            entry["color_ramp"] = rna(node.color_ramp)
        nodes[node.name] = entry
    trees[key] = {
        "nodes": nodes,
        "links": sorted([link.from_node.name, link.from_socket.identifier,
                         link.to_node.name, link.to_socket.identifier] for link in tree.links),
        "custom": custom(tree),
        "animation": animation_state(tree),
    }


def image_state(image):
    # Packed-byte and decoded-pixel hashes detect both persisted and live paint edits.
    pixels = array.array("f", [0]) * len(image.pixels)
    if pixels:
        image.pixels.foreach_get(pixels)
    if sys.byteorder != "little":
        pixels.byteswap()
    packed = [hashlib.sha256(bytes(item.packed_file.data)).hexdigest()
              for item in image.packed_files]
    if not packed and image.packed_file:
        packed = [hashlib.sha256(bytes(image.packed_file.data)).hexdigest()]
    return {
        "name": image.name, "source": image.source, "type": image.type,
        "size": list(image.size), "channels": image.channels,
        "alpha_mode": image.alpha_mode,
        "colorSpace": rna(image.colorspace_settings),
        "pixelFloats": len(pixels), "pixelSha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "packedSha256": packed,
        # A packed image's original filesystem spelling does not alter its model.
        "externalFilepath": image.filepath if not packed else None,
        "custom": custom(image),
    }


def fingerprint(collection=None):
    started = time.monotonic()
    collection = collection or bpy.data.collections.get(COLLECTION)
    if collection is None:
        raise ValueError("Character collection not found")
    objects = sorted(collection.all_objects, key=lambda obj: obj.name)
    if any(obj.mode == "EDIT" for obj in objects):
        raise ValueError("Cannot fingerprint unsynchronized live edit-mode data")
    material_ids, image_ids, trees = {}, {}, {}
    components = {"collection": collection.name, "objects": {}, "materials": {},
                  "nodeTrees": trees, "images": {}}
    for obj in objects:
        entry = {
            "type": obj.type, "parent": obj.parent.name if obj.parent else None,
            "parent_type": obj.parent_type, "parent_bone": obj.parent_bone,
            "matrix_parent_inverse": scalar(obj.matrix_parent_inverse),
            "location": scalar(obj.location), "rotation_mode": obj.rotation_mode,
            "rotation_euler": scalar(obj.rotation_euler),
            "rotation_quaternion": scalar(obj.rotation_quaternion),
            "rotation_axis_angle": scalar(obj.rotation_axis_angle),
            "scale": scalar(obj.scale), "delta_location": scalar(obj.delta_location),
            "delta_rotation_euler": scalar(obj.delta_rotation_euler),
            "delta_rotation_quaternion": scalar(obj.delta_rotation_quaternion),
            "delta_scale": scalar(obj.delta_scale),
            "hide_render": obj.hide_render, "custom": custom(obj),
            "modifiers": [rna(m) for m in obj.modifiers],
            "constraints": [rna(c) for c in obj.constraints],
            "animation": animation_state(obj),
            "materials": [{"name": s.material.name if s.material else None, "link": s.link}
                          for s in obj.material_slots],
        }
        for slot in obj.material_slots:
            if slot.material:
                material_ids[slot.material.name] = slot.material
        if obj.type == "MESH":
            entry["mesh"] = mesh_state(obj)
        elif obj.type == "ARMATURE":
            entry["armature"] = armature_state(obj)
        components["objects"][obj.name] = entry
    for name, material in sorted(material_ids.items()):
        settings = rna(material, skip={"preview", "node_tree", "animation_data",
                                      "texture_paint_images", "texture_paint_slots",
                                      "paint_active_slot", "paint_clone_slot", "use_fake_user",
                                      "users", "is_updated", "is_updated_data", "is_updated_transform",
                                      "is_embedded_data", "is_library_indirect", "session_uid"})
        settings["custom"] = custom(material)
        settings["animation"] = animation_state(material)
        components["materials"][name] = settings
        if material.node_tree:
            tree_state(material.node_tree, image_ids, trees, "material:" + name)
    for name, image in sorted(image_ids.items()):
        components["images"][name] = image_state(image)
    # Kept outside modelSha256: selection, editor layout, inspection visibility.
    presentation = {
        "activeScene": bpy.context.scene.name,
        "scenes": {s.name: {"frame_current": s.frame_current, "frame_subframe": s.frame_subframe,
                            "camera": s.camera.name if s.camera else None} for s in bpy.data.scenes},
        "activeObject": bpy.context.view_layer.objects.active.name
                        if bpy.context.view_layer.objects.active else None,
        "objects": {obj.name: {"selected": obj.select_get(), "hide_viewport": obj.hide_viewport,
                               "hidden": obj.hide_get(), "show_in_front": obj.show_in_front,
                               "activeShapeKeyIndex": obj.active_shape_key_index}
                    for obj in objects},
        "imageDirtyFlags": {name: image.is_dirty for name, image in image_ids.items()},
        "screens": {screen.name: [{"type": area.type, "x": area.x, "y": area.y,
                                   "width": area.width, "height": area.height}
                                  for area in screen.areas] for screen in bpy.data.screens},
    }
    return {
        "version": VERSION,
        "sourceFilepath": bpy.data.filepath,
        "modelSha256": digest(components),
        "componentSha256": {key: digest(value) for key, value in components.items()},
        "components": components,
        "presentationSha256": digest(presentation),
        "presentation": presentation,
        "meshCount": sum(obj.type == "MESH" for obj in objects),
        "armatureCount": sum(obj.type == "ARMATURE" for obj in objects),
        "elapsedSeconds": time.monotonic() - started,
        "scope": "Character raw geometry/topology/attributes/UV/skin/shape keys, transforms, modifiers/constraints/animation, armature rest/pose, custom data and actual VRM properties, material/node graphs, used image packed bytes and decoded pixels. UI/selection/inspection visibility recorded separately. No Blend save or object mutation.",
    }


def write(path, collection=None):
    result = fingerprint(collection)
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in result.items()
            if key not in {"components", "presentation"}}


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    output = args[args.index("--output") + 1]
    print(json.dumps(write(output), ensure_ascii=False))
