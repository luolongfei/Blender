"""Conservative, measured runtime optimization in a disposable Blender process.

The editable master and posed PBR export are never passed through this module.
Face/hand/finger/nail meshes and every shape-key mesh keep their topology.
Decimation is accepted only after bidirectional surface sampling and weight
checks. This measures geometric error, not user approval of appearance.
"""
from collections import defaultdict
import hashlib
import json
import math

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


PROTECTED_PARTS = {
    "head", "face", "eye", "iris", "pupil", "eyelid", "eyelash", "eyebrow",
    "brow", "lip", "mouth", "teeth", "tongue", "hand", "finger", "nail",
}


def protected_geometry(obj):
    return bool(obj.data.shape_keys or obj.get("facial_role")
                or obj.get("part", "") in PROTECTED_PARTS)


def requires_uv(obj):
    """Solid-color facial geometry is valid without an unused UV layer."""
    for material in obj.data.materials:
        if material is None:
            continue
        extension = getattr(material, "vrm_addon_extension", None)
        mtoon = getattr(extension, "mtoon1", None)
        if mtoon and mtoon.enabled:
            if any(info.index.source for info in mtoon.all_texture_info()):
                return True
        elif material.use_nodes and material.node_tree:
            if any(node.type == "TEX_IMAGE" and node.image
                   and any(socket.is_linked for socket in node.outputs)
                   for node in material.node_tree.nodes):
                return True
    return False


def triangles(mesh):
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def select_one(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def snapshot_surface(mesh):
    mesh.calc_loop_triangles()
    vertices = [v.co.copy() for v in mesh.vertices]
    faces = [tuple(t.vertices) for t in mesh.loop_triangles]
    samples = list(vertices)
    # Every triangle centre is checked, not just a sparse random subset.
    samples.extend((vertices[a] + vertices[b] + vertices[c]) / 3
                   for a, b, c in faces)
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True), samples


def surface_distance(source, candidate):
    other_tree, other_samples = snapshot_surface(candidate)
    tree, samples = source
    first = max((other_tree.find_nearest(p)[3] for p in samples), default=0)
    second = max((tree.find_nearest(p)[3] for p in other_samples), default=0)
    return max(first, second)


def weight_error(obj, rig):
    valid = {g.index for g in obj.vertex_groups
             if g.name in rig.data.bones and rig.data.bones[g.name].use_deform}
    max_dropped = 0.0
    max_sum_error = 0.0
    for vertex in obj.data.vertices:
        weights = sorted((g.weight for g in vertex.groups if g.group in valid),
                         reverse=True)
        max_sum_error = max(max_sum_error, abs(sum(weights) - 1))
        max_dropped = max(max_dropped, sum(weights[4:]))
    return max_sum_error, max_dropped


def normalize_weights(obj, rig):
    groups = {g.index: g for g in obj.vertex_groups
              if g.name in rig.data.bones and rig.data.bones[g.name].use_deform}
    for vertex in obj.data.vertices:
        weights = sorted(((g.group, g.weight) for g in vertex.groups
                          if g.group in groups and g.weight > 0),
                         key=lambda item: item[1], reverse=True)
        total = sum(w for _, w in weights[:4])
        if total <= 0:
            raise ValueError(obj.name + ": optimization lost skin weights")
        for index, weight in weights[:4]:
            groups[index].add([vertex.index], weight / total, "REPLACE")
        for index, _ in weights[4:]:
            groups[index].remove([vertex.index])


def policy(obj, count):
    part = obj.get("part", "")
    if protected_geometry(obj):
        return None
    # Only authored, known categories are eligible. New unclassified details
    # remain intact rather than guessing what the geometry represents.
    name = obj.name
    if name.startswith("Sock_Ribbed_"):
        return .18, .00016, "sock longitudinal oversampling"
    if name == "Skirt_Sixteen_Pleats":
        return .20, .00030, "planar pleat panels"
    if name.startswith(("Blouse_Tucked_", "Blouse_Puff_")):
        return .32, .00025, "dense cloth surface"
    if name.startswith("Shoe_Leather_Upper_"):
        return .35, .00020, "dense shoe upper"
    if part == "leg":
        return .40, .00020, "dense leg longitudinal sampling"
    if part == "hair" and count >= 400:
        return .50, .00016, "hair lock sampling"
    if part in {"upper_body", "lower_body", "accessory"} and count >= 280:
        return .25, .00010, "small garment or shoe detail"
    return None


def freeze_non_skin_modifiers(obj):
    """Apply evaluated non-skin geometry while preserving the original skin."""
    select_one(obj)
    if obj.data.shape_keys:
        # Shape keys are protected; the exporter evaluates their modifiers.
        return
    for modifier in list(obj.modifiers):
        if modifier.type == "ARMATURE":
            continue
        if not modifier.show_viewport or (
                modifier.type == "SUBSURF" and modifier.levels == 0):
            obj.modifiers.remove(modifier)
            continue
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        if "FINISHED" not in result:
            raise ValueError(obj.name + ": cannot freeze " + modifier.name)


def reduce_mesh(obj, rig):
    original = obj.data
    count = triangles(original)
    choice = policy(obj, count)
    row = {"mesh": obj.name, "beforeTriangles": count,
           "afterTriangles": count, "accepted": False}
    if choice is None:
        row["reason"] = "protected or already modest topology"
        return row
    ratio, tolerance, reason = choice
    row.update(reason=reason, toleranceMeters=tolerance, attempts=[])
    source = snapshot_surface(original)
    # If the aggressive ratio exceeds the small absolute surface tolerance,
    # retry at a denser level; otherwise retain the entire original mesh.
    attempts = [("COLLAPSE", ratio), ("COLLAPSE", min(.82, math.sqrt(ratio))),
                ("DISSOLVE", .075)]
    for method, requested in attempts:
        candidate = original.copy()
        obj.data = candidate
        select_one(obj)
        modifier = obj.modifiers.new("Measured runtime reduction", "DECIMATE")
        modifier.decimate_type = method
        if method == "COLLAPSE":
            modifier.ratio = requested
            modifier.use_collapse_triangulate = True
        else:
            modifier.angle_limit = requested
            modifier.delimit = {"NORMAL", "MATERIAL", "SEAM", "UV"}
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        if "FINISHED" not in result:
            raise ValueError(obj.name + ": reduction failed")
        candidate = obj.data
        deviation = surface_distance(source, candidate)
        sum_error, dropped = weight_error(obj, rig)
        after = triangles(candidate)
        attempt = {"method": method, "ratioOrAngle": requested, "triangles": after,
                   "sampledSurfaceErrorMeters": deviation,
                   "maxSkinWeightSumError": sum_error,
                   "maxDiscardedSkinWeight": dropped}
        row["attempts"].append(attempt)
        if (after < count and deviation <= tolerance
                and dropped <= .015 and sum_error <= .0002):
            normalize_weights(obj, rig)
            row.update(accepted=True, afterTriangles=after,
                       sampledSurfaceErrorMeters=deviation)
            if original.users == 0:
                bpy.data.meshes.remove(original)
            return row
        obj.data = original
        if candidate.users == 0:
            bpy.data.meshes.remove(candidate)
    return row


def active_uv_only(obj):
    active = next((layer for layer in obj.data.uv_layers if layer.active_render), None)
    if active is None:
        if requires_uv(obj):
            raise ValueError(obj.name + ": image material has no active render UV")
        return
    keep = active.name
    for layer in list(obj.data.uv_layers):
        if layer.name != keep:
            obj.data.uv_layers.remove(layer)
    obj.data.uv_layers[0].name = "RuntimeUV"
    obj.data.uv_layers.active_index = 0
    obj.data.uv_layers[0].active_render = True


def join_buckets(rig, meshes, plan):
    """Join only equal material + physics buckets, retaining full provenance."""
    targets = {target["mesh"]: target for target in plan["targets"]}
    buckets = defaultdict(list)
    for obj in meshes:
        # A protected facial/hand mesh stays individually editable in runtime.
        # Shape-key meshes cannot safely share mesh topology.
        if protected_geometry(obj):
            key = ("individual", obj.name)
        else:
            target = targets.get(obj.name)
            motion = ((target["mode"], tuple(sorted(target.get("chains", []))))
                      if target else ("humanoid", ()))
            material_names = tuple(material.name for material in obj.data.materials)
            key = (material_names, motion, obj.get("part", "unclassified"))
        buckets[key].append(obj)
    outputs, provenance, new_targets = [], {}, []
    for index, members in enumerate(buckets.values()):
        sources = sorted(obj.name for obj in members)
        existing_targets = [targets[name] for name in sources if name in targets]
        active = members[0]
        if len(members) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for obj in members:
                obj.hide_set(False)
                obj.select_set(True)
            bpy.context.view_layer.objects.active = active
            result = bpy.ops.object.join()
            if "FINISHED" not in result:
                raise ValueError("Could not join runtime material bucket")
            suffix = hashlib.sha256("\0".join(sources).encode()).hexdigest()[:8]
            active.name = "Runtime_" + active.get("part", "detail") + "_" + suffix
        active["runtime_source_meshes"] = json.dumps(sources, ensure_ascii=False)
        provenance[active.name] = sources
        outputs.append(active)
        if existing_targets:
            first = existing_targets[0]
            target = {"mesh": active.name, "mode": first["mode"],
                      "source": first["source"], "sourceMeshes": sources}
            if first["mode"] == "spring":
                target["chains"] = sorted(set(first["chains"]))
            else:
                target["reason"] = "; ".join(dict.fromkeys(
                    target["reason"] for target in existing_targets))
            new_targets.append(target)
    updated = {"version": 1, "targets": new_targets}
    rig["charakit_spring_plan_v1"] = json.dumps(updated, ensure_ascii=False)
    return outputs, updated, provenance


def optimize(rig, meshes, plan, *, join=True):
    if not bpy.app.background:
        raise RuntimeError("Runtime optimization requires a disposable background process")
    source_names = {obj.name for obj in meshes}
    rows = []
    for obj in meshes:
        freeze_non_skin_modifiers(obj)
        rows.append(reduce_mesh(obj, rig))
        active_uv_only(obj)
    if join:
        meshes, plan, provenance = join_buckets(rig, meshes, plan)
    else:
        provenance = {obj.name: [obj.name] for obj in meshes}
    flattened = [name for names in provenance.values() for name in names]
    if len(flattened) != len(set(flattened)) or set(flattened) != source_names:
        raise ValueError("Runtime provenance is not an exact partition of source meshes")
    return meshes, plan, {
        "sourceMeshCount": len(source_names), "runtimeMeshCount": len(meshes),
        "beforeTriangles": sum(row["beforeTriangles"] for row in rows),
        "afterTriangles": sum(row["afterTriangles"] for row in rows),
        "protectedParts": sorted(PROTECTED_PARTS) + [
            "all shape-key meshes", "explicit facial_role meshes",
        ],
        "surfaceCheck": "bidirectional nearest-surface distances at all vertices and triangle centroids",
        "visualReviewRequired": True, "meshes": rows, "provenance": provenance,
    }
