"""Expression preservation and VRM 1.0 bindings for disposable export copies.

The shape keys are authored by the modeling workflow. This module does not
create expressions, infer facial anatomy, or modify the saved source Blend.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct


PRESETS = {
    "neutral": (),
    **{mood: (mood + "_eye", mood + "_mouth", "brow_" + mood)
       for mood in ("happy", "angry", "sad", "relaxed", "surprised")},
    "blink": ("blink",),
    "blinkLeft": ("blink_L",),
    "blinkRight": ("blink_R",),
    **{vowel: (vowel,) for vowel in ("aa", "ih", "ou", "ee", "oh")},
}


def inventory(meshes):
    result = {}
    for obj in meshes:
        keys = obj.data.shape_keys
        if not keys:
            continue
        if not keys.use_relative:
            raise ValueError(obj.name + ": absolute/time-based shape keys are unsupported")
        basis = keys.reference_key
        shapes = {}
        for key in keys.key_blocks:
            if key == basis:
                continue
            if key.relative_key != basis:
                raise ValueError(
                    obj.name + "/" + key.name + ": export requires a Basis-relative key"
                )
            if len(key.data) != len(basis.data):
                raise ValueError(obj.name + ": shape-key topology differs from Basis")
            maximum = 0.0
            changed = 0
            for a, b in zip(key.data, basis.data):
                delta = (a.co - b.co).length
                if not math.isfinite(delta):
                    raise ValueError(obj.name + "/" + key.name + ": nonfinite shape")
                maximum = max(maximum, delta)
                changed += delta > 1e-8
            shapes[key.name] = {
                "maxDeltaMeters": maximum,
                "movedVertices": changed,
                "value": key.value,
                "muted": key.mute,
                "coordinatesSha256": hashlib.sha256(b"".join(
                    struct.pack("<fff", *point.co) for point in key.data
                )).hexdigest(),
            }
        if shapes:
            result[obj.name] = {"vertices": len(obj.data.vertices), "keys": shapes}
    return result


def configure(rig, meshes):
    """Bind every nonzero authored component, keeping unrequested presets intact."""
    import bpy

    if not bpy.app.background:
        raise RuntimeError("Configure expressions only in a disposable export process")
    source = inventory(meshes)
    by_key = {}
    for name, item in source.items():
        for key, info in item["keys"].items():
            if info["movedVertices"]:
                by_key.setdefault(key, []).append(name)
    required = {key for keys in PRESETS.values() for key in keys}
    missing = sorted(required - by_key.keys())
    if missing:
        raise ValueError("Missing nonzero authored expression components: " + ", ".join(missing))
    expressions = rig.data.vrm_addon_extension.vrm1.expressions
    expressions.initial_automatic_expression_assignment = False
    expressions.fill_missing_expression_names()
    # Add-on previews can otherwise restore stale expression values on updates.
    for expression in expressions.all_name_to_expression_dict().values():
        expression.preview = 0
    mapping = expressions.preset.name_to_expression_dict()
    report = {}
    for preset, components in PRESETS.items():
        expression = mapping[preset]
        expression.morph_target_binds.clear()
        expression.material_color_binds.clear()
        expression.texture_transform_binds.clear()
        expression.is_binary = False
        # A mood's eye and mouth poses reduce competing automatic blink/lip-sync
        # proportionally. All individual blink/vowel channels remain continuous.
        mood = preset in {"happy", "angry", "sad", "relaxed", "surprised"}
        expression.override_blink = "blend" if mood else "none"
        expression.override_mouth = "blend" if mood else "none"
        expression.override_look_at = "none"
        binds = []
        for component in components:
            for object_name in sorted(by_key[component]):
                bind = expression.morph_target_binds.add()
                bind.node.mesh_object_name = object_name
                bind.index = component
                bind.weight = 1
                binds.append({"mesh": object_name, "key": component, "weight": 1})
        report[preset] = {
            "components": list(components), "binds": binds,
            "overrideBlink": expression.override_blink,
            "overrideMouth": expression.override_mouth,
        }
    for obj in meshes:
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                key.value = 0
                key.mute = False
    bpy.context.view_layer.update()
    return {"source": inventory(meshes), "presets": report,
            "defaultExpression": "neutral", "sourceBlendModified": False}


@contextmanager
def evaluated_pbr_meshes(meshes):
    """Temporarily evaluate modifiers without discarding shape-key targets.

    Blender's export_apply=True evaluates a mesh without retaining its keys.
    Here each key is evaluated with skin modifiers disabled, then reconstructed
    on the identical evaluated topology. Restore data/modifier flags afterward.
    """
    import bpy

    backups = []
    try:
        for obj in meshes:
            flags = [(mod, mod.show_viewport, mod.show_render) for mod in obj.modifiers]
            original = obj.data
            key_values = [(key, key.value, key.mute)
                          for key in original.shape_keys.key_blocks] if original.shape_keys else []
            backup = {"object": obj, "data": original, "flags": flags,
                      "keyValues": key_values, "temporary": None}
            backups.append(backup)
            active_non_skin = any(mod.type != "ARMATURE" and mod.show_viewport
                                  for mod in obj.modifiers)
            if not active_non_skin:
                continue
            for mod in obj.modifiers:
                if mod.type == "ARMATURE":
                    mod.show_viewport = mod.show_render = False
            for key, _, _ in key_values:
                key.value = 0
                key.mute = False
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            evaluated = obj.evaluated_get(depsgraph)
            basis_mesh = bpy.data.meshes.new_from_object(
                evaluated, preserve_all_data_layers=True, depsgraph=depsgraph
            )
            backup["temporary"] = basis_mesh
            topology = (len(basis_mesh.vertices),
                        tuple(tuple(p.vertices) for p in basis_mesh.polygons))
            shape_coordinates = {}
            if original.shape_keys:
                basis = original.shape_keys.reference_key
                for key in original.shape_keys.key_blocks:
                    if key == basis:
                        continue
                    if key.relative_key != basis:
                        raise ValueError(obj.name + ": non-Basis-relative key")
                    key.value = 1
                    bpy.context.view_layer.update()
                    evaluated_key = obj.evaluated_get(depsgraph)
                    temporary = evaluated_key.to_mesh(
                        preserve_all_data_layers=True, depsgraph=depsgraph
                    )
                    try:
                        actual = (len(temporary.vertices),
                                  tuple(tuple(p.vertices) for p in temporary.polygons))
                        if actual != topology:
                            raise ValueError(
                                obj.name + "/" + key.name +
                                ": modifiers change topology between expression states"
                            )
                        shape_coordinates[key.name] = [tuple(v.co) for v in temporary.vertices]
                    finally:
                        evaluated_key.to_mesh_clear()
                        key.value = 0
            obj.data = basis_mesh
            if shape_coordinates:
                obj.shape_key_add(name=original.shape_keys.reference_key.name, from_mix=False)
                for name, coords in shape_coordinates.items():
                    key = obj.shape_key_add(name=name, from_mix=False)
                    for point, coordinate in zip(key.data, coords):
                        point.co = coordinate
                    source_key = original.shape_keys.key_blocks[name]
                    key.slider_min = source_key.slider_min
                    key.slider_max = source_key.slider_max
                    # A source key's vertex-group mask has already been
                    # evaluated into these coordinates; do not apply it twice.
                    key.interpolation = source_key.interpolation
                    key.value = source_key.value
            for mod, viewport, render in flags:
                mod.show_viewport = viewport if mod.type == "ARMATURE" else False
                mod.show_render = render if mod.type == "ARMATURE" else False
        bpy.context.view_layer.update()
        yield
    finally:
        for backup in reversed(backups):
            obj = backup["object"]
            obj.data = backup["data"]
            for mod, viewport, render in backup["flags"]:
                mod.show_viewport, mod.show_render = viewport, render
            for key, value, mute in backup["keyValues"]:
                key.value, key.mute = value, mute
            temporary = backup["temporary"]
            if temporary and temporary.users == 0:
                bpy.data.meshes.remove(temporary)
        bpy.context.view_layer.update()


def _read_glb(path):
    raw = Path(path).read_bytes()
    if struct.unpack_from("<III", raw) != (0x46546C67, 2, len(raw)):
        raise ValueError("Invalid binary glTF")
    offset, document, binary = 12, None, None
    while offset < len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if document is None or binary is None:
        raise ValueError("Missing glTF JSON/binary chunk")
    return raw, document, binary


def _accessor(document, binary, index):
    a = document["accessors"][index]
    components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    fmt, size = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4),
                 5126: ("f", 4)}[a["componentType"]]
    values = [[0] * components for _ in range(a["count"])]
    if "bufferView" in a:
        v = document["bufferViews"][a["bufferView"]]
        offset = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride", components * size)
        values = [struct.unpack_from("<" + fmt * components, binary, offset + i * stride)
                  for i in range(a["count"])]
    if "sparse" in a:
        sparse = a["sparse"]
        ind, val = sparse["indices"], sparse["values"]
        iv, vv = (document["bufferViews"][x["bufferView"]] for x in (ind, val))
        ifmt, isize = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4)}[ind["componentType"]]
        io = iv.get("byteOffset", 0) + ind.get("byteOffset", 0)
        vo = vv.get("byteOffset", 0) + val.get("byteOffset", 0)
        for i in range(sparse["count"]):
            target = struct.unpack_from("<" + ifmt, binary, io + i * isize)[0]
            if not 0 <= target < len(values):
                raise ValueError("Morph sparse accessor index is out of bounds")
            values[target] = struct.unpack_from(
                "<" + fmt * components, binary, vo + i * components * size
            )
    return values


def audit(path, manifest, *, require_vrm=False):
    """Check real exported deltas/binds; this is not an appearance judgement."""
    raw, document, binary = _read_glb(path)
    nodes = {node["name"]: (i, node) for i, node in enumerate(document["nodes"])
             if "mesh" in node}
    checked = {}
    for name, source in manifest["source"].items():
        if name not in nodes:
            raise ValueError("Expression mesh omitted: " + name)
        _, node = nodes[name]
        mesh = document["meshes"][node["mesh"]]
        names = mesh.get("extras", {}).get("targetNames", [])
        if set(names) != set(source["keys"]) or len(names) != len(set(names)):
            raise ValueError("Exported shape keys differ: " + name)
        if any(abs(w) > 1e-7 for w in node.get("weights", mesh.get("weights", []))):
            raise ValueError("Exported model does not start with neutral morph weights")
        maxima = {key: 0.0 for key in names}
        for primitive in mesh["primitives"]:
            targets = primitive.get("targets", [])
            if len(targets) != len(names):
                raise ValueError("Primitive lost expression targets: " + name)
            vertex_count = document["accessors"][primitive["attributes"]["POSITION"]]["count"]
            for key, target in zip(names, targets):
                if "POSITION" not in target:
                    raise ValueError("Expression target lost position deltas: " + name + "/" + key)
                for semantic, accessor in target.items():
                    values = _accessor(document, binary, accessor)
                    if len(values) != vertex_count or any(
                            not all(math.isfinite(v) for v in row) for row in values):
                        raise ValueError("Invalid expression accessor: " + name + "/" + key)
                    if semantic == "POSITION":
                        maxima[key] = max(maxima[key], max(
                            (math.sqrt(sum(v * v for v in row)) for row in values), default=0
                        ))
        for key, source_key in source["keys"].items():
            if source_key["movedVertices"] and maxima[key] < 1e-8:
                raise ValueError("Nonzero authored key became empty: " + name + "/" + key)
        checked[name] = {"shapeKeys": names, "maxDeltasMeters": maxima}
    presets = {}
    if require_vrm:
        exported = document["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        for preset, expected in manifest["presets"].items():
            expression = exported.get(preset)
            if expression is None:
                raise ValueError("VRM preset missing: " + preset)
            actual_binds = []
            for bind in expression.get("morphTargetBinds", []):
                node = document["nodes"][bind["node"]]
                mesh = document["meshes"][node["mesh"]]
                names = mesh.get("extras", {}).get("targetNames", [])
                if not 0 <= bind["index"] < len(names):
                    raise ValueError("VRM morph binding points outside the exported mesh")
                actual_binds.append((node["name"], names[bind["index"]], bind["weight"]))
            expected_binds = [(b["mesh"], b["key"], b["weight"]) for b in expected["binds"]]
            if Counter(actual_binds) != Counter(expected_binds):
                raise ValueError("VRM preset binding mismatch: " + preset)
            if expression.get("isBinary", False):
                raise ValueError("Continuous expression became binary: " + preset)
            for property_name in ("overrideBlink", "overrideMouth"):
                if expression.get(property_name, "none") != expected[property_name]:
                    raise ValueError("Expression override mismatch: " + preset)
            presets[preset] = {"binds": len(actual_binds), "continuous": True}
    return {
        "file": str(Path(path).resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shapeKeyMeshes": len(checked),
        "shapeKeyTargets": sum(len(x["shapeKeys"]) for x in checked.values()),
        "meshes": checked, "presets": presets,
        "neutralMorphWeights": True, "finiteMorphDeltas": True,
        "scope": "Serialized targets, nonzero deltas and exact VRM bindings; appearance, "
                 "surface collisions and expression combinations require visual review.",
    }
