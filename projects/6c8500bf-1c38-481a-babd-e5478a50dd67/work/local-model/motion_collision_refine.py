"""Apply measured local collision refinements without changing skin or bones.

Call apply() after any rig_builder.finish_existing(), which reconstructs the
base spring setup. This helper itself does not save a file or set a pose.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
GROUP = "Shoulders · ponytails"
PLAN_KEY = "local_motion_collision_plan_v1"
SHOULDER_RADIUS = 0.070
PONYTAIL_HIT_RADIUS = 0.025
FRONT_SKIRT_HIT_RADIUS = 0.038
BACK_GROUP = "Back garment · ponytails"
BACK_SPECS = [
    {"name": "Collider · left back garment", "bone": "chest",
     "center": (0.090, 0.025, 1.090), "tail": (0.090, 0.025, 1.230),
     "radius": 0.060},
    {"name": "Collider · right back garment", "bone": "chest",
     "center": (-0.090, 0.025, 1.090), "tail": (-0.090, 0.025, 1.230),
     "radius": 0.060},
]


def _helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rig_signature(rig):
    """Check actual rest bones, transforms and all existing mesh/shape weights."""
    digest = hashlib.sha256()
    for bone in rig.data.bones:
        digest.update(repr((bone.name, bone.parent.name if bone.parent else None,
                            tuple(tuple(row) for row in bone.matrix_local),
                            bone.length, bone.use_deform)).encode())
    for obj in sorted((o for o in bpy.data.objects if o.type == "MESH"
                       and any(m.type == "ARMATURE" and m.object == rig
                               for m in o.modifiers)), key=lambda o: o.name):
        digest.update(obj.name.encode())
        digest.update(repr(tuple(tuple(row) for row in obj.matrix_world)).encode())
        digest.update(repr([(g.name, g.index) for g in obj.vertex_groups]).encode())
        for vertex in obj.data.vertices:
            digest.update(repr((tuple(vertex.co),
                                [(g.group, g.weight) for g in vertex.groups])).encode())
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                digest.update(key.name.encode())
                for point in key.data:
                    digest.update(repr(tuple(point.co)).encode())
    return digest.hexdigest()


def apply_back_coverage(rig=None):
    """Add two local back capsules after dense setup; preserve every joint setting."""
    rig = rig or bpy.data.objects.get("Character_Rig")
    if not rig or rig.type != "ARMATURE":
        raise ValueError("Character_Rig armature is required.")
    api = _helper("vrm_helpers")
    api._enabled()
    sb = rig.data.vrm_addon_extension.spring_bone1
    springs = {s.vrm_name: s for s in sb.springs}
    if any(name not in springs for name in ("ponytail_L", "ponytail_R")):
        raise ValueError("Both ponytail chains are required.")
    before = _rig_signature(rig)
    created = api.add_colliders(rig, BACK_SPECS, BACK_GROUP)
    group = next(g for g in sb.collider_groups if g.vrm_name == BACK_GROUP)
    for name in ("ponytail_L", "ponytail_R"):
        spring = springs[name]
        if not any(r.collider_group_uuid == group.uuid for r in spring.collider_groups):
            spring.collider_groups.add().collider_group_uuid = group.uuid
    bpy.context.view_layer.update()
    after = _rig_signature(rig)
    if before != after:
        raise RuntimeError("Back coverage unexpectedly modified geometry, skin or bones.")
    plan = {
        "version": 1, "colliderGroup": BACK_GROUP, "specs": BACK_SPECS,
        "scope": "Only two chest-attached posterior garment capsules, referenced by both ponytails.",
        "evidence": "work/qa-hair-physics/v07-standard-contact-binding.json",
        "measuredChestRestContact": {
            "x": [0.098, 0.100], "y": [0.068, 0.071], "z": [1.194, 1.200],
        },
        "jointParametersPreserved": True,
    }
    rig["local_back_collision_plan_v1"] = json.dumps(plan, ensure_ascii=False)
    actual = _helper("hair_chain_refine").sync_actual_spring_plan(rig)
    return {
        "colliders": created, "plan": plan, "actualSpringSettings": actual,
        "meshShapeSkinRestUnchanged": before == after,
        "meshShapeSkinRestSignatureBefore": before,
        "meshShapeSkinRestSignatureAfter": after,
    }


def apply(rig=None):
    rig = rig or bpy.data.objects.get("Character_Rig")
    if not rig or rig.type != "ARMATURE":
        raise ValueError("Character_Rig armature is required.")
    api = _helper("vrm_helpers")
    api._enabled()
    sb = rig.data.vrm_addon_extension.spring_bone1
    springs = {spring.vrm_name: spring for spring in sb.springs}
    expected = ["ponytail_L", "ponytail_R"] + [
        f"skirt_panel_{index:02d}" for index in (8, 9, 10)
    ]
    if any(name not in springs for name in expected):
        raise ValueError("The measured ponytail and skirt chains must already exist.")
    before = _rig_signature(rig)
    specs = [
        {"name": "Collider · left shoulder", "bone": "upperChest",
         "center": (0.12, 0.004, 1.23), "radius": SHOULDER_RADIUS},
        {"name": "Collider · right shoulder", "bone": "upperChest",
         "center": (-0.12, 0.004, 1.23), "radius": SHOULDER_RADIUS},
    ]
    colliders = api.add_colliders(rig, specs, GROUP)
    group = next(group for group in sb.collider_groups if group.vrm_name == GROUP)
    changes = []
    for name in ("ponytail_L", "ponytail_R"):
        spring = springs[name]
        refs = spring.collider_groups
        if not any(ref.collider_group_uuid == group.uuid for ref in refs):
            refs.add().collider_group_uuid = group.uuid
        previous = [joint.hit_radius for joint in spring.joints]
        for joint in spring.joints:
            joint.hit_radius = PONYTAIL_HIT_RADIUS
        changes.append({"chain": name, "beforeHitRadius": previous,
                        "hitRadius": PONYTAIL_HIT_RADIUS,
                        "additionalColliderGroup": GROUP})
    for index in (8, 9, 10):
        spring = springs[f"skirt_panel_{index:02d}"]
        previous = [joint.hit_radius for joint in spring.joints]
        for joint in spring.joints:
            joint.hit_radius = FRONT_SKIRT_HIT_RADIUS
        changes.append({"chain": spring.vrm_name, "beforeHitRadius": previous,
                        "hitRadius": FRONT_SKIRT_HIT_RADIUS})
    bpy.context.view_layer.update()
    after = _rig_signature(rig)
    if before != after:
        raise RuntimeError("Unexpected mesh, shape-key, skin or rest-bone modification.")
    # The kit's target/chain plan remains authoritative and is audited below.
    # This separate record keeps the measured parameter rationale with the Blend.
    plan = {
        "version": 1, "armature": rig.name, "shoulderColliders": specs,
        "colliderGroup": GROUP, "ponytailHitRadius": PONYTAIL_HIT_RADIUS,
        "frontSkirtChains": [f"skirt_panel_{i:02d}" for i in (8, 9, 10)],
        "frontSkirtHitRadius": FRONT_SKIRT_HIT_RADIUS,
        "evidence": {
            "hair": "work/qa-hair-physics/shoulder-r070-h025-walk-5-90.png",
            "skirt": "work/qa-skirt-motion/hem-front038-full.json",
        },
        "scope": "Only shoulder colliders and five named chains' collision radii. "
                 "Existing body groups, gravity, stiffness, drag, centers and bones preserved.",
    }
    rig[PLAN_KEY] = json.dumps(plan, ensure_ascii=False)
    spring_api = api._module("spring-setup.py", "local_motion_collision_audit")
    audit = spring_api.audit(rig)
    return {
        "colliders": colliders, "changes": changes,
        "meshShapeSkinRestSignatureBefore": before,
        "meshShapeSkinRestSignatureAfter": after,
        "meshShapeSkinRestUnchanged": before == after,
        "springChains": audit["chains"], "plan": plan,
    }


if __name__ == "__main__":
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--output" not in args:
        raise SystemExit("Explicit --output disposable copy is required.")
    output = Path(args[args.index("--output") + 1]).resolve()
    if output == Path(bpy.data.filepath).resolve():
        raise SystemExit("Refusing to overwrite the opened source.")
    source = Path(bpy.data.filepath).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    report = apply()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), copy=True)
    report.update(source=str(source), sourceSha256=source_sha,
                  sourceUnchanged=hashlib.sha256(source.read_bytes()).hexdigest() == source_sha,
                  output=str(output))
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
