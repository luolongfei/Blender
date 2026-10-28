"""Dense ponytail collision chains and arc-length skinning in a work copy.

The old three main bone names and their head positions are retained. Each
span is divided to <=32 mm; the original non-deforming terminal is retained.
Call after geometry refinements and after the last generic spring setup.
"""
import bisect
import importlib.util
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
KEY = "local_dense_hair_chain_v1"
BIND_ATTRIBUTE = "HairRefinementBindPosition"
MAX_SEGMENT = 0.032


def _helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rig(rig):
    result = rig or bpy.data.objects.get("Character_Rig")
    if not result or result.type != "ARMATURE":
        raise ValueError("Character_Rig armature is required.")
    return result


def _plan(rig):
    plan = json.loads(rig.get(KEY, "null"))
    if not plan or plan.get("version") != 1:
        raise ValueError("Create the dense chain with apply() before binding.")
    return plan


def sync_actual_spring_plan(rig=None):
    """Keep the kit target plan plus an exact snapshot of final addon settings."""
    rig = _rig(rig)
    plan = json.loads(rig["charakit_spring_plan_v1"])
    sb = rig.data.vrm_addon_extension.spring_bone1
    groups = {g.uuid: g.vrm_name for g in sb.collider_groups}
    colliders = []
    for collider in sb.colliders:
        capsule = collider.shape_type == "Capsule"
        shape = collider.shape.capsule if capsule else collider.shape.sphere
        row = {
            "uuid": collider.uuid,
            "name": collider.bpy_object.name if collider.bpy_object else "",
            "bone": collider.node.bone_name,
            "shape": "capsule" if capsule else "sphere",
            "offset": list(shape.offset), "radius": float(shape.radius),
        }
        if capsule:
            row["tail"] = list(shape.tail)
        colliders.append(row)
    chains = []
    for spring in sb.springs:
        chains.append({
            "name": spring.vrm_name, "center": spring.center.bone_name or None,
            "colliderGroups": [groups[r.collider_group_uuid] for r in spring.collider_groups],
            "joints": [{
                "bone": j.node.bone_name, "stiffness": float(j.stiffness),
                "gravityPower": float(j.gravity_power),
                "gravityDir": list(j.gravity_dir), "dragForce": float(j.drag_force),
                "hitRadius": float(j.hit_radius),
            } for j in spring.joints],
        })
    actual = {
        "format": "Actual VRM1 addon values; offsets in assigned bone local space",
        "chains": chains, "colliders": colliders,
        "colliderGroups": [{
            "name": g.vrm_name,
            "colliders": [r.collider_uuid for r in g.colliders],
        } for g in sb.collider_groups],
    }
    plan["actualSpringSettings"] = actual
    rig["charakit_spring_plan_v1"] = json.dumps(plan, ensure_ascii=False)
    return actual


def weights_for_point(co, side, rig=None):
    """co is the original bind point in rig coordinates; side is L or R."""
    rig = _rig(rig)
    data = _plan(rig)["sides"][side]
    return _weights_from_info(co, data)


def _weights_from_info(co, data):
    p = Vector(co)
    points = [Vector(point) for point in data["originalControlPoints"]]
    nearest = None
    distance_before = 0.0
    for a, b in zip(points, points[1:]):
        ab = b - a
        length = ab.length
        t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
        closest = a + t * ab
        candidate = ((p - closest).length_squared, distance_before + t * length)
        if nearest is None or candidate[0] < nearest[0]:
            nearest = candidate
        distance_before += length
    arc = nearest[1]
    names, positions = data["deformBones"], data["arcHeadPositions"]
    index = min(len(names) - 1, max(0, bisect.bisect_right(positions, arc) - 1))
    if index == len(names) - 1:
        result = {names[index]: 1.0}
    else:
        t = (arc - positions[index]) / (positions[index + 1] - positions[index])
        result = {names[index]: 1 - t, names[index + 1]: t}
    # A 12 mm attachment transition, then purely spring-driven hair.
    t = max(0.0, min(1.0, arc / 0.012))
    root_blend = t * t * (3 - 2 * t)
    result = {name: value * root_blend for name, value in result.items()}
    result["head"] = 1.0 - root_blend
    result = {name: value for name, value in result.items() if value > 1e-8}
    total = sum(result.values())
    return {name: value / total for name, value in result.items()}


def bind_mesh(obj, side=None, rig=None):
    """Rebind a ponytail using preserved pre-refinement point coordinates."""
    rig = _rig(rig)
    side = side or obj.get("side")
    if side not in ("L", "R") or obj.type != "MESH":
        raise ValueError("A mesh with explicit L/R side is required.")
    attr = obj.data.attributes.get(BIND_ATTRIBUTE)
    if attr and (attr.domain != "POINT" or attr.data_type != "FLOAT_VECTOR"):
        raise ValueError(obj.name + ": bind attribute must be POINT FLOAT_VECTOR.")
    transform = rig.matrix_world.inverted() @ obj.matrix_world
    info = _plan(rig)["sides"][side]
    expected = {"head", *info["deformBones"]}
    old_hair = {g.index for g in obj.vertex_groups
                if g.name == "head" or g.name.startswith("hair_")}
    for vertex in obj.data.vertices:
        if any(g.weight > 1e-8 and g.group not in old_hair for g in vertex.groups):
            raise ValueError(obj.name + ": unexpected non-hair weight; inspect first.")
    indices = list(range(len(obj.data.vertices)))
    for group in obj.vertex_groups:
        if group.index in old_hair:
            group.remove(indices)
    groups = {name: obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
              for name in expected}
    maximum_sum_error = 0.0
    for vertex in obj.data.vertices:
        co = attr.data[vertex.index].vector if attr else vertex.co
        weights = _weights_from_info(transform @ co, info)
        maximum_sum_error = max(maximum_sum_error, abs(sum(weights.values()) - 1.0))
        for name, value in weights.items():
            groups[name].add([vertex.index], value, "REPLACE")
    obj["dense_hair_binding_version"] = 1
    return {"mesh": obj.name, "vertices": len(indices), "side": side,
            "bindPositions": BIND_ATTRIBUTE if attr else "current original coordinates",
            "maxWeightSumError": maximum_sum_error}


def align_centerline(rig, transform):
    """Apply the hair geometry's measured rest field to nonhumanoid bones.

    The bind arc remains the original centerline, because the saved POINT
    attribute is expressed before the geometry refinement.  This function
    moves only the rest path used by physics, not the arc used for weights.
    Run before any subsequent whole-head geometry/bone warp.
    """
    rig = _rig(rig)
    data = _plan(rig)
    if not callable(transform):
        raise TypeError("centerline transform must be callable(co, side).")
    if bpy.context.mode != "OBJECT":
        raise ValueError("Run from Object mode.")
    moved = {}
    bpy.ops.object.select_all(action="DESELECT")
    rig.hide_set(False)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for side, info in data["sides"].items():
            original = info.setdefault("originalDensePoints", info["points"])
            source = [Vector(point) for point in original]
            points = [Vector(transform(point.copy(), side)) for point in source]
            if any(not all(math.isfinite(value) for value in point) for point in points):
                raise ValueError("Nonfinite dense-chain centerline.")
            terminal = rig.data.edit_bones[info["terminal"]]
            delta = Vector(info.setdefault(
                "originalTerminalDelta", tuple(terminal.tail - terminal.head)))
            for index, name in enumerate(info["deformBones"]):
                bone = rig.data.edit_bones[name]
                bone.head, bone.tail = points[index], points[index + 1]
            terminal.head = points[-1]
            terminal.tail = Vector(transform(source[-1] + delta, side))
            info["points"] = [tuple(point) for point in points]
            info["centerlineTransform"] = "measured ponytail geometry field"
            moved[side] = {
                "maximumRestShiftMeters": max((a-b).length for a, b in zip(points, source)),
                "maximumSegmentMeters": max((b-a).length for a, b in zip(points, points[1:])),
                "points": info["points"],
            }
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    rig[KEY] = json.dumps(data, ensure_ascii=False)
    bpy.context.view_layer.update()
    return moved


def apply(rig=None, rebind=True, centerline_transform=None, dynamics="uniform"):
    if dynamics not in ("uniform", "graduated"):
        raise ValueError("Unknown dense-hair dynamics profile.")
    rig = _rig(rig)
    api = _helper("vrm_helpers")
    api._enabled()
    if bpy.context.mode != "OBJECT":
        raise ValueError("Run from Object mode.")
    human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    mapping_before = {name.value: entry.node.bone_name
                      for name, entry in human.human_bone_name_to_human_bone().items()}
    stored = json.loads(rig.get(KEY, "null"))
    data = stored or {"version": 1, "maxSegmentMeters": MAX_SEGMENT, "sides": {}}
    original_count = len(rig.data.bones)
    if not stored:
        for side in ("L", "R"):
            bones = [rig.data.bones[f"hair_{side}_{i}"] for i in (1, 2, 3)]
            points = [bones[0].head_local.copy()] + [bone.tail_local.copy() for bone in bones]
            for a, b in zip(bones, bones[1:]):
                if (a.tail_local - b.head_local).length > 1e-6:
                    raise ValueError("The original hair chain is not contiguous.")
            names, heads, dense_points, spans, arc = [], [], [], [], 0.0
            for segment, (a, b) in enumerate(zip(points, points[1:]), 1):
                subdivisions = max(1, math.ceil((b - a).length / MAX_SEGMENT))
                # The geometry field can lengthen or curve a rest span.
                # Choose density in the final physics space while retaining
                # the original parameterization for the bind-position map.
                if centerline_transform is not None:
                    while True:
                        mapped = [
                            Vector(centerline_transform(a.lerp(b, i/subdivisions), side))
                            for i in range(subdivisions + 1)
                        ]
                        if max((q-p).length for p, q in zip(mapped, mapped[1:])) <= MAX_SEGMENT:
                            break
                        subdivisions += 1
                        if subdivisions > 32:
                            raise ValueError("Hair rest field requires an unexpectedly dense span.")
                span_names = []
                for index in range(subdivisions):
                    name = (f"hair_{side}_{segment}" if index == 0 else
                            f"hair_{side}_{segment}_mid_{index:02d}")
                    names.append(name)
                    span_names.append(name)
                    heads.append(arc + (b - a).length * index / subdivisions)
                    dense_points.append(tuple(a.lerp(b, index / subdivisions)))
                spans.append({"originalBone": f"hair_{side}_{segment}",
                              "subdivisions": subdivisions, "bones": span_names})
                arc += (b - a).length
            dense_points.append(tuple(points[-1]))
            data["sides"][side] = {
                "originalControlPoints": [tuple(point) for point in points],
                "deformBones": names, "arcHeadPositions": heads, "totalArcLength": arc,
                "points": dense_points, "spans": spans,
                "terminal": f"hair_{side}_3_CharakitEnd",
            }
        bpy.ops.object.select_all(action="DESELECT")
        rig.hide_set(False)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for side, info in data["sides"].items():
            parent = rig.data.edit_bones["head"]
            for index, name in enumerate(info["deformBones"]):
                bone = rig.data.edit_bones.get(name) or rig.data.edit_bones.new(name)
                bone.head = info["points"][index]
                bone.tail = info["points"][index + 1]
                bone.parent = parent
                bone.use_connect = index > 0
                bone.use_deform = True
                parent = bone
            terminal = rig.data.edit_bones[info["terminal"]]
            terminal.parent = parent
            terminal.use_connect = True
            terminal.use_deform = False
        bpy.ops.object.mode_set(mode="OBJECT")
        rig[KEY] = json.dumps(data, ensure_ascii=False)
    else:
        for info in data["sides"].values():
            if any(name not in rig.data.bones for name in info["deformBones"]):
                raise ValueError("Stored dense chain has missing bones.")
    alignment = None
    if centerline_transform is not None:
        alignment = align_centerline(rig, centerline_transform)
        data = _plan(rig)
    bindings = []
    if rebind:
        for obj in bpy.data.objects:
            if (obj.type == "MESH" and obj.get("part") == "hair"
                    and obj.get("hair_region") == "ponytail"
                    and any(m.type == "ARMATURE" and m.object == rig for m in obj.modifiers)):
                bindings.append(bind_mesh(obj, rig=rig))
    sb = rig.data.vrm_addon_extension.spring_bone1
    group_names = {group.uuid: group.vrm_name for group in sb.collider_groups}
    chains = []
    for side, info in data["sides"].items():
        name = "ponytail_" + side
        previous = next(spring for spring in sb.springs if spring.vrm_name == name)
        groups = [group_names[ref.collider_group_uuid] for ref in previous.collider_groups]
        chains.append({
            "name": name, "bones": info["deformBones"] + [info["terminal"]],
            "includesTerminal": True, "center": previous.center.bone_name or None,
            # Shorter spans distribute curvature; avoid multiplying the old
            # gravity bend across ten joints. Motion is verified after export.
            "stiffness": 3.0, "gravityPower": 0.025, "dragForce": 0.48,
            "hitRadius": 0.025, "colliderGroups": groups,
        })
    plan = json.loads(rig["charakit_spring_plan_v1"])
    targets = plan["targets"]
    known = {target["mesh"] for target in targets}
    for binding in bindings:
        if binding["mesh"] not in known:
            targets.append({"mesh": binding["mesh"], "source": "hair", "mode": "spring",
                            "chains": ["ponytail_" + binding["side"]]})
    audit = api.register_springs(rig, chains, targets)
    joint_dynamics = {}
    for side, info in data["sides"].items():
        spring = next(s for s in sb.springs if s.vrm_name == "ponytail_" + side)
        rows = []
        for index, joint in enumerate(spring.joints):
            if dynamics == "graduated":
                fraction = (info["arcHeadPositions"][index] / info["totalArcLength"]
                            if index < len(info["arcHeadPositions"]) else 1.0)
                joint.stiffness = 3.0 if fraction < .30 else 2.0 if fraction < .73 else 1.1
                joint.gravity_power = .065
                joint.drag_force = .50
            rows.append({
                "bone": joint.node.bone_name, "stiffness": joint.stiffness,
                "gravityPower": joint.gravity_power, "dragForce": joint.drag_force,
                "hitRadius": joint.hit_radius,
            })
        joint_dynamics[spring.vrm_name] = rows
    rig["local_dense_hair_dynamics_v1"] = json.dumps(
        {"profile": dynamics, "chains": joint_dynamics}, ensure_ascii=False)
    actual_settings = sync_actual_spring_plan(rig)
    mapping_after = {name.value: entry.node.bone_name
                     for name, entry in human.human_bone_name_to_human_bone().items()}
    if mapping_before != mapping_after:
        raise RuntimeError("Humanoid mapping unexpectedly changed.")
    bpy.context.view_layer.update()
    return {"bonesBefore": original_count, "bonesAfter": len(rig.data.bones),
            "newNonHumanoidBones": len(rig.data.bones) - original_count,
            "humanoidMappingUnchanged": True, "plan": data,
            "springChains": audit["chains"], "settings": chains, "bindings": bindings,
            "centerlineAlignment": alignment, "dynamicsProfile": dynamics,
            "jointDynamics": joint_dynamics,
            "actualSpringSettings": actual_settings}


if __name__ == "__main__":
    import hashlib
    import sys
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--output" not in args:
        raise SystemExit("Pass --output for a disposable QA Blend.")
    output = Path(args[args.index("--output") + 1]).resolve()
    source = Path(bpy.data.filepath).resolve()
    if output == source:
        raise SystemExit("Refusing to overwrite source.")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    transform = (_helper("hair_refinement").motion_center
                 if "--align-refinement" in args else None)
    profile = args[args.index("--dynamics") + 1] if "--dynamics" in args else "uniform"
    report = apply(centerline_transform=transform, dynamics=profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), copy=True)
    report.update(source=str(source), sourceSha256=source_sha,
                  sourceUnchanged=hashlib.sha256(source.read_bytes()).hexdigest() == source_sha,
                  output=str(output))
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))
