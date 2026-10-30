"""Read-only local neck/body topology, cross-section and skin-weight audit.

No mesh, keys, object visibility, frame, rig pose or modifiers are changed.
No full-scene collision query is performed. run() prints and returns a report;
run(write_report=True) also writes a small JSON beside the current work copy.
Raw Basis/mixed-key sections are unposed production coordinates. Optional
evaluated sections include existing modifiers and the current pose, clearly
labelled separately; they are not a collision certification.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

TARGETS = ("R01 Head", "Body_Complete_Trunk_Arms_Pelvis")
LEVELS = (1.245, 1.255, 1.265, 1.275, 1.285, 1.300, 1.320, 1.340)
LOW, HIGH = 1.235, 1.345


def _bounds(points):
    if not points:
        return None
    return [[min(v[i] for v in points), max(v[i] for v in points)] for i in range(3)]


def _components(adjacency, vertices):
    left = set(vertices)
    result = []
    while left:
        seed = left.pop()
        stack, group = [seed], [seed]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in left:
                    left.remove(neighbor)
                    stack.append(neighbor)
                    group.append(neighbor)
        result.append(group)
    return result


def _mixed(obj):
    keys = obj.data.shape_keys
    if keys is None:
        return [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not keys.use_relative:
        raise RuntimeError("Absolute shape keys need separate inspection: " + obj.name)
    points = [v.co.copy() for v in keys.reference_key.data]
    for key in keys.key_blocks:
        if key == keys.reference_key or key.mute or abs(key.value) < 1e-10:
            continue
        group = obj.vertex_groups.get(key.vertex_group) if key.vertex_group else None
        for i, point in enumerate(points):
            factor = key.value
            if group:
                try:
                    factor *= group.weight(i)
                except RuntimeError:
                    factor = 0.0
            point += (key.data[i].co - key.relative_key.data[i].co) * factor
    return [obj.matrix_world @ point for point in points]


def _sections(points, edges):
    records = []
    for z in LEVELS:
        intersections = []
        for a, b in edges:
            p, q = points[a], points[b]
            if (p.z - z) * (q.z - z) > 0 or abs(p.z - q.z) < 1e-10:
                continue
            t = (z - p.z) / (q.z - p.z)
            if 0.0 <= t <= 1.0:
                hit = p.lerp(q, t)
                # Exclude arms/shoulders from the axial neck section only.
                if abs(hit.x) < .085 and abs(hit.y) < .095:
                    intersections.append(hit)
        records.append({"z_m": z, "edge_intersections": len(intersections),
                        "axial_section_bounds_m": _bounds(intersections)})
    return records


def _raw_report(obj):
    mesh = obj.data
    keys = mesh.shape_keys
    basis = keys.reference_key.data if keys else mesh.vertices
    coordinates = [obj.matrix_world @ v.co for v in basis]
    mixed = _mixed(obj)
    adjacency = defaultdict(set)
    incidence = defaultdict(int)
    edges = [tuple(e.vertices) for e in mesh.edges]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    for face in mesh.polygons:
        for key in face.edge_keys:
            incidence[tuple(sorted(key))] += 1
    boundary = [(a, b) for a, b in edges if incidence[tuple(sorted((a, b)))] == 1]
    boundary_adjacency = defaultdict(set)
    for a, b in boundary:
        boundary_adjacency[a].add(b)
        boundary_adjacency[b].add(a)
    groups = _components(boundary_adjacency, boundary_adjacency.keys())
    loops = []
    for group in groups:
        pts = [coordinates[i] for i in group]
        bb = _bounds(pts)
        if bb[2][0] > HIGH or bb[2][1] < LOW:
            continue
        loops.append({"vertices": sorted(group), "vertex_count": len(group),
                      "basis_bounds_m": bb,
                      "mixed_bounds_m": _bounds([mixed[i] for i in group]),
                      "closed_simple_loop": all(len(boundary_adjacency[i]) == 2 for i in group)})
    components = _components(adjacency, range(len(coordinates)))
    local_ids = [i for i, v in enumerate(coordinates)
                 if LOW < v.z < HIGH and abs(v.x) < .085 and abs(v.y) < .095]
    weights = defaultdict(list)
    unweighted = []
    rig = bpy.data.objects.get("Character_Rig")
    deform_names = {b.name for b in rig.data.bones if b.use_deform} if rig else set()
    for i in local_ids:
        total = 0.0
        for weight in mesh.vertices[i].groups:
            name = obj.vertex_groups[weight.group].name
            if name in deform_names:
                weights[name].append(weight.weight)
                total += weight.weight
        if total < 1e-6:
            unweighted.append(i)
    key_rows = []
    if keys:
        for key in keys.key_blocks:
            delta = [(key.data[i].co - key.relative_key.data[i].co).length for i in local_ids]
            key_rows.append({"name": key.name, "value": key.value, "mute": key.mute,
                             "point_count": len(key.data), "relative_to": key.relative_key.name,
                             "local_max_relative_delta_m": max(delta, default=0.0)})
    cap_candidates = []
    for face in mesh.polygons:
        pts = [coordinates[i] for i in face.vertices]
        center = sum(pts, Vector()) / len(pts)
        if not (1.265 < center.z < 1.295 and abs(center.x) < .055 and abs(center.y) < .055):
            continue
        normal = (pts[1] - pts[0]).cross(pts[2] - pts[0])
        if normal.length > 1e-12 and abs(normal.normalized().z) > .8:
            cap_candidates.append(face.index)
    return {
        "vertices": len(mesh.vertices), "faces": len(mesh.polygons),
        "connected_components": [{"vertices": len(group),
                                  "bounds_m": _bounds([coordinates[i] for i in group])}
                                 for group in sorted(components, key=len, reverse=True)],
        "total_boundary_edges": len(boundary), "local_boundary_loops": loops,
        "local_band_vertex_count": len(local_ids), "local_unweighted_vertices": unweighted,
        "local_weight_groups": {name: {"count": len(values), "min": min(values), "max": max(values)}
                                for name, values in weights.items()},
        "shape_keys": key_rows,
        "modifiers": [{"name": m.name, "type": m.type,
                       "viewport": m.show_viewport, "render": m.show_render} for m in obj.modifiers],
        "upward_or_downward_cap_candidate_faces": cap_candidates,
        "basis_sections": _sections(coordinates, edges),
        "mixed_key_sections": _sections(mixed, edges),
        "visible": {"viewport": not obj.hide_get(), "render": not obj.hide_render},
    }


def run(write_report=False, evaluated=True):
    report = {"scope": "Read-only local neck boundary/topology/section/weight audit",
              "source": bpy.data.filepath, "frame": bpy.context.scene.frame_current,
              "objects": {}, "limitations": [
                  "Section bounds are geometric evidence, not proof of visible overlap",
                  "Cap face candidates require visual selection review before removal",
                  "No topology, weights, keys, modifiers, frame or pose changed",
                  "No whole-body collision or motion testing performed"]}
    graph = bpy.context.evaluated_depsgraph_get() if evaluated else None
    for name in TARGETS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            report["objects"][name] = {"missing": True}
            continue
        row = _raw_report(obj)
        if evaluated:
            evaluated_obj = obj.evaluated_get(graph)
            mesh = evaluated_obj.to_mesh()
            try:
                pts = [evaluated_obj.matrix_world @ v.co for v in mesh.vertices]
                local = {i for i, p in enumerate(pts)
                         if LOW - .02 < p.z < HIGH + .02 and abs(p.x) < .10 and abs(p.y) < .12}
                edges = [tuple(e.vertices) for e in mesh.edges
                         if e.vertices[0] in local or e.vertices[1] in local]
                row["evaluated_current_pose_sections"] = _sections(pts, edges)
            finally:
                evaluated_obj.to_mesh_clear()
        report["objects"][name] = row
    report["collar_bounds"] = {}
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.startswith(("Blouse_PeterPan_Collar", "Neck")):
            report["collar_bounds"][obj.name] = {
                "raw_bounds_m": _bounds([obj.matrix_world @ v.co for v in obj.data.vertices]),
                "render_visible": not obj.hide_render}
    if write_report:
        project = project_path()
        file = Path(bpy.data.filepath).resolve()
        try:
            file.relative_to((project / "work").resolve())
        except ValueError:
            raise RuntimeError("Write reports only beside a project work copy")
        path = file.parent / "neck-boundary-audit-r02.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("NECK_AUDIT_REPORT", path)
    # Keep inline output short; the return value retains exact candidate IDs.
    print(json.dumps({name: {"components": len(row.get("connected_components", [])),
                             "neck_loops": [r["vertex_count"] for r in row.get("local_boundary_loops", [])],
                             "cap_candidates": len(row.get("upward_or_downward_cap_candidate_faces", []))}
                      for name, row in report["objects"].items()}, ensure_ascii=False))
    return report
