"""Synthetic background test; never opens or changes an authored character."""
import importlib.util
import json
from pathlib import Path

import bpy


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "expression-export-qa"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    if not bpy.app.background or bpy.data.filepath:
        raise RuntimeError("Run against Blender's factory startup, with no input file.")
    OUT.mkdir(parents=True, exist_ok=True)
    api, expressions = load("vrm_helpers"), load("expression_export")
    api._enabled()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    arm = bpy.data.armatures.new("QA skeleton")
    rig = bpy.data.objects.new("QA rig", arm)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    positions = {
        "hips": ((0, 0, .9), (0, 0, 1.0), None),
        "spine": ((0, 0, 1.0), (0, 0, 1.2), "hips"),
        "chest": ((0, 0, 1.2), (0, 0, 1.4), "spine"),
        "neck": ((0, 0, 1.4), (0, 0, 1.5), "chest"),
        "head": ((0, 0, 1.5), (0, 0, 1.7), "neck"),
    }
    for side, sign in (("left", 1), ("right", -1)):
        positions.update({
            side + "UpperArm": ((sign*.1, 0, 1.4), (sign*.3, 0, 1.4), "chest"),
            side + "LowerArm": ((sign*.3, 0, 1.4), (sign*.5, 0, 1.4), side+"UpperArm"),
            side + "Hand": ((sign*.5, 0, 1.4), (sign*.6, 0, 1.4), side+"LowerArm"),
            side + "UpperLeg": ((sign*.08, 0, .9), (sign*.08, 0, .5), "hips"),
            side + "LowerLeg": ((sign*.08, 0, .5), (sign*.08, 0, .1), side+"UpperLeg"),
            side + "Foot": ((sign*.08, 0, .1), (sign*.08, -.12, .05), side+"LowerLeg"),
        })
    for name, (head, tail, parent) in positions.items():
        bone = arm.edit_bones.new(name)
        bone.head, bone.tail = head, tail
        if parent:
            bone.parent = arm.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    objects = []
    for i, name in enumerate(("QA face", "QA brow")):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16, ring_count=8, radius=.08 if not i else .02,
            location=(i*.15, 0, 1.6),
        )
        obj = bpy.context.object
        obj.name = name
        obj["part"] = "head"
        obj["facial_role"] = "qa"
        material = bpy.data.materials.new(name + " material")
        material.use_nodes = True
        material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (.8, .6, .5, 1)
        obj.data.materials.append(material)
        if i:
            # Pure-color new facial parts need no artificial UV atlas.
            for uv in list(obj.data.uv_layers):
                obj.data.uv_layers.remove(uv)
            optimizer = load("runtime_optimize")
            assert not optimizer.requires_uv(obj)
            optimizer.active_uv_only(obj)
            assert not obj.data.uv_layers
        obj.shape_key_add(name="Basis")
        keys = sorted({key for values in expressions.PRESETS.values() for key in values})
        if i:
            keys = [key for key in keys if key.startswith("brow_")]
        for index, name in enumerate(keys):
            key = obj.shape_key_add(name=name, from_mix=False)
            for point in key.data:
                point.co.x += (index + 1) * .0002
                point.co.z += .001 * (point.co.z + .1)
        subdivision = obj.modifiers.new("QA surface", "SUBSURF")
        subdivision.levels = subdivision.render_levels = 1
        group = obj.vertex_groups.new(name="head")
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
        modifier = obj.modifiers.new("QA skin", "ARMATURE")
        modifier.object = rig
        objects.append(obj)
    mapping = api.configure_vrm(rig, objects)
    if mapping["humanoidErrors"]:
        raise ValueError(mapping["humanoidErrors"])
    manifest = expressions.configure(rig, objects)
    assert max(item["maxDeltaMeters"] for mesh in manifest["source"].values()
               for item in mesh["keys"].values()) < .01
    original = [(obj.data, len(obj.data.vertices), len(obj.data.shape_keys.key_blocks))
                for obj in objects]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [rig, *objects]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    glb = OUT / "synthetic-expressions.glb"
    with expressions.evaluated_pbr_meshes(objects):
        assert len(objects[0].data.vertices) > original[0][1]
        bpy.ops.export_scene.gltf(
            filepath=str(glb), export_format="GLB", use_selection=True,
            export_apply=False, export_morph=True, export_morph_normal=True,
            export_skins=True, export_animations=False, export_current_frame=True,
        )
    for obj, (data, vertices, keys) in zip(objects, original):
        assert obj.data == data and len(obj.data.vertices) == vertices
        assert len(obj.data.shape_keys.key_blocks) == keys
        assert obj.modifiers[0].show_viewport and obj.modifiers[1].show_viewport
    assert expressions.inventory(objects) == manifest["source"], "Source keys changed during PBR evaluation"
    pbr = expressions.audit(glb, manifest)
    _, document, _ = expressions._read_glb(glb)
    brow_node = next(node for node in document["nodes"] if node.get("name") == "QA brow")
    assert all("TEXCOORD_0" not in primitive["attributes"]
               for primitive in document["meshes"][brow_node["mesh"]]["primitives"])
    vrm = OUT / "synthetic-expressions.vrm"
    api.export_review(rig, objects, vrm)
    runtime = expressions.audit(vrm, manifest, require_vrm=True)
    report = {"syntheticOnly": True, "sourceCharacterOpened": False,
              "untexturedFacialMeshWithoutUVSupported": True,
              "temporaryGeometryRestored": True, "pbr": pbr, "vrm": runtime}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("EXPRESSION_QA_REPORT", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    run()
