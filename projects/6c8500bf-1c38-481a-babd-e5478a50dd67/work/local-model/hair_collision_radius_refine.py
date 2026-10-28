"""Update only ponytail mid/tip hit radii to cover measured actual strand volume.

The v07 exported surface has middle/lower centerline radii up to 39.35 mm.
The former 25 mm particle radius did not represent that volume. This patch
keeps roots and all stiffness, gravity, drag, collider geometry and skins.
"""
import importlib.util
import json
import re
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
RADIUS = 0.040


def _helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply(rig=None):
    rig = rig or bpy.data.objects.get("Character_Rig")
    if not rig or rig.type != "ARMATURE":
        raise ValueError("Character_Rig armature is required.")
    sb = rig.data.vrm_addon_extension.spring_bone1
    signature = _helper("motion_collision_refine")._rig_signature
    before = signature(rig)
    changes = []
    chains = [s for s in sb.springs if s.vrm_name in ("ponytail_L", "ponytail_R")]
    if len(chains) != 2:
        raise ValueError("Expected the two dense ponytail chains.")
    for spring in chains:
        for joint in spring.joints:
            name = joint.node.bone_name
            if re.match(r"^hair_[LR]_[23](?:_|$)", name):
                changes.append({"chain": spring.vrm_name, "bone": name,
                                "before": float(joint.hit_radius), "after": RADIUS})
                joint.hit_radius = RADIUS
    if not changes:
        raise ValueError("No middle/tip ponytail joints matched.")
    bpy.context.view_layer.update()
    after = signature(rig)
    if before != after:
        raise RuntimeError("Unexpected mesh/shape/skin/rest-bone change.")
    report = {
        "version": 1, "radius": RADIUS, "changes": changes,
        "evidence": {
            "surfaceRadiusMaximumMeters": 0.03934992570414143,
            "middleLower99PercentileRangeMeters": [0.03039, 0.03825],
            "surfaceInputSha256": "65acb1a66f5164e28603d1db1e4fff9b5307ce72b961ea5b66d493f526c39f09",
            "contactReport": "work/qa-hair-physics/v07-standard-contact-binding.json",
        },
        "meshShapeSkinRestUnchanged": before == after,
        "scope": "Only middle/tip hair joint hit radius. Root joints remain 25mm.",
    }
    rig["local_measured_hair_radius_v1"] = json.dumps(report, ensure_ascii=False)
    report["actualSpringSettings"] = _helper("hair_chain_refine").sync_actual_spring_plan(rig)
    return report
