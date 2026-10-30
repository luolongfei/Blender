"""Bounded fairing and skin-material continuity for the R03 joined neck.

Only the neck junction is affected. A fixed Basis mask and the same linear
Taubin smoothing operator are used for every key, preserving correspondence.
Vertices at Basis Z >= 1.315 m, face anatomy, topology, groups and drivers stay
unchanged. No modifiers are applied. run() writes a local reversible backup;
restore() restores only this pass. Neither function saves the Blend file.
"""
import bpy
import json
import math
import sys
from pathlib import Path
from mathutils import Vector

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

TAG = "r03_neck_fairing"
VERSION = 1
MIN_Z, MAX_Z = 1.245, 1.315


def _smooth(a, b, x):
    t = max(0., min(1., (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _guard():
    project = project_path()
    file = Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or file == (project / "work/character.blend").resolve():
        raise RuntimeError("Use a saved R03 joined working copy")
    try:
        file.relative_to((project / "work").resolve())
    except ValueError:
        raise RuntimeError("Expected a project work copy")
    obj = bpy.data.objects.get("R01 Head")
    if obj is None or not obj.get("r03_neck_join_report"):
        raise RuntimeError("Run only on the checked unified R03 head/body")
    if len(obj.data.vertices) != 26580 or not obj.data.shape_keys:
        raise RuntimeError("Topology differs from the reviewed unified R03 render")
    if max(abs(obj.matrix_world[i][j] - float(i == j)) for i in range(4) for j in range(4)) > 1e-6:
        raise RuntimeError("Expected the unified mesh's verified identity transform")
    if obj.data.users != 1 or obj.library or obj.data.library:
        raise RuntimeError("A local single-user unified mesh is required")
    if bpy.context.mode != "OBJECT":
        raise RuntimeError("Object Mode required")
    colour = obj.data.color_attributes.get("R01_region_color")
    if colour is None or colour.domain != "POINT":
        raise RuntimeError("Expected the preserved point-domain skin colour layer")
    # Use the same actual skin material already assigned to facial polygons.
    face_slots = {p.material_index for p in obj.data.polygons
                  if all(obj.data.vertices[i].co.z > 1.45 for i in p.vertices)}
    if len(face_slots) != 1:
        raise RuntimeError("Facial material is ambiguous; inspect before normalizing neck")
    material_index = next(iter(face_slots))
    material = obj.data.materials[material_index]
    if not material or not material.use_nodes:
        raise RuntimeError("Expected the current node-based head skin material")
    return project, file, obj, colour, material_index


def _fair(points, adjacency, selected, weights, pairs):
    """Small positive/negative diffusion pairs suppress ripples without the
    large shrinkage of repeated ordinary Laplacian smoothing. Z stays free
    inside the band to redistribute the narrow cut rings; band edges are fixed.
    """
    work = [p.copy() for p in points]
    for _ in range(pairs):
        for strength in (.45, -.47):
            next_points = {}
            for i in selected:
                neighbors = adjacency[i]
                avg = sum((work[j] for j in neighbors), Vector()) / len(neighbors)
                next_points[i] = work[i] + (avg - work[i]) * (strength * weights[i])
            for i, point in next_points.items():
                work[i] = point
    return work


def run(pairs=10, maximum_move=.006):
    project, file, obj, colours, skin_index = _guard()
    if obj.get(TAG):
        raise RuntimeError("This neck fairing is already applied; use restore() before trying different settings")
    if not isinstance(pairs, int) or not 1 <= pairs <= 20:
        raise ValueError("Use 1–20 bounded smoothing pairs")
    if not .001 <= maximum_move <= .010:
        raise ValueError("Displacement guard must be between 1 and 10 mm")
    mesh, keys = obj.data, obj.data.shape_keys.key_blocks
    basis = [v.co.copy() for v in keys[0].data]
    adjacency = [set() for _ in basis]
    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].add(b); adjacency[b].add(a)
    weights = {}
    for i, p in enumerate(basis):
        if p.z >= MAX_Z or p.z <= MIN_Z or abs(p.x) >= .080 or abs(p.y - .006) >= .080:
            continue
        if any(key.data[i].co.z >= MAX_Z for key in keys):
            continue  # Protect the upper limit in every stored key, not just Basis.
        envelope = _smooth(MIN_Z, 1.268, p.z) * (1 - _smooth(1.290, MAX_Z, p.z))
        envelope *= 1 - _smooth(.060, .080, abs(p.x))
        envelope *= 1 - _smooth(.060, .080, abs(p.y - .006))
        if envelope > 1e-8 and adjacency[i]:
            weights[i] = envelope
    selected = sorted(weights)
    if not 100 < len(selected) < 2500:
        raise RuntimeError("Unexpected neck selection size: " + str(len(selected)))
    plans, movements = {}, {}
    # Work on each key independently with exactly the same mask/operator. This
    # is a linear mapping, so key combinations remain coherent after fairing.
    for key in keys:
        points = [v.co.copy() for v in key.data]
        result = _fair(points, adjacency, selected, weights, pairs)
        distance = max((result[i] - points[i]).length for i in selected)
        if not math.isfinite(distance) or distance > maximum_move:
            raise RuntimeError(f"{key.name}: proposed move {distance:.6f} exceeds the guard; reduce pairs or review first")
        plans[key.name] = {i: result[i] for i in selected}
        movements[key.name] = distance
    # Inspect current neck skin colour immediately above the junction. Restrict
    # anchors to the neck, not lips, face or the default-colour appended body.
    anchors = [i for i, p in enumerate(basis) if 1.299 < p.z < 1.314
               and abs(p.x) < .065 and abs(p.y - .006) < .075]
    if len(anchors) < 8:
        raise RuntimeError("Cannot establish a local head-skin colour anchor")
    target_colour = tuple(sum(colours.data[i].color[c] for i in anchors) / len(anchors) for c in range(4))
    colour_changes = {}
    for i, p in enumerate(basis):
        if p.z >= MAX_Z or p.z <= 1.237 or abs(p.x) >= .085 or abs(p.y - .006) >= .085:
            continue
        blend = _smooth(1.237, 1.252, p.z) * (1 - _smooth(1.291, MAX_Z, p.z))
        blend *= 1 - _smooth(.070, .085, abs(p.x))
        blend *= 1 - _smooth(.070, .085, abs(p.y - .006))
        if blend > 1e-8:
            old = colours.data[i].color
            colour_changes[i] = tuple(old[c] * (1 - blend) + target_colour[c] * blend for c in range(4))
    material_changes = []
    for poly in mesh.polygons:
        points = [basis[i] for i in poly.vertices]
        if (poly.material_index != skin_index and
                all(1.242 < p.z < MAX_Z and abs(p.x) < .080 and abs(p.y - .006) < .080 for p in points)):
            material_changes.append(poly.index)
    if not material_changes:
        raise RuntimeError("No old-body neck material patch found; inspect current appearance before running")
    backup_path = file.parent / "neck-fair-r03-backup.json"
    number = 1
    while backup_path.exists():
        backup_path = file.parent / f"neck-fair-r03-backup-{number:03d}.json"
        number += 1
    backup = {"schema": 1, "version": VERSION, "object": obj.name,
              "vertices": len(basis), "keys": [k.name for k in keys],
              "coordinates": {key.name: {str(i): list(key.data[i].co) for i in selected} for key in keys},
              "colours": {str(i): list(colours.data[i].color) for i in colour_changes},
              "materials": {str(i): mesh.polygons[i].material_index for i in material_changes}}
    # Save rollback data before touching any key, colour or material assignment.
    backup_path.write_text(json.dumps(backup, ensure_ascii=False), encoding="utf-8")
    for key in keys:
        for i, point in plans[key.name].items():
            key.data[i].co = point
    # Keep raw mesh/Basis coherent for later topology and weight audits.
    for i in selected:
        mesh.vertices[i].co = keys[0].data[i].co
    for i, colour in colour_changes.items():
        colours.data[i].color = colour
    for i in material_changes:
        mesh.polygons[i].material_index = skin_index
    mesh.update()
    bpy.context.view_layer.update()
    report = {"version": VERSION, "source": str(file.relative_to(project)),
              "method": "Fixed-mask Taubin fairing applied to every relative key",
              "pairs": pairs, "geometry_vertices": len(selected),
              "maximum_key_moves_m": movements, "strict_protected_z": MAX_Z,
              "colour_vertices": len(colour_changes), "target_neck_colour": target_colour,
              "reassigned_neck_faces": len(material_changes),
              "skin_material": mesh.materials[skin_index].name,
              "backup": str(backup_path.relative_to(project)),
              "preserved": ["topology", "vertex order", "all shape keys and values",
                            "drivers", "actions", "weights", "UVs", "face coordinates above 1.315m"],
              "limitations": ["Local fairing proposal; rendered front/side/quarter review required",
                              "No new motion or collision acceptance", "No whole-body material replacement"]}
    obj[TAG] = json.dumps(report)
    path = file.parent / "neck-fair-r03-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report


def restore():
    project, file, obj, colours, skin_index = _guard()
    if not obj.get(TAG):
        raise RuntimeError("No applied neck fairing marker")
    report = json.loads(obj[TAG])
    path = project / report["backup"]
    backup = json.loads(path.read_text(encoding="utf-8"))
    keys = obj.data.shape_keys.key_blocks
    if backup["vertices"] != len(obj.data.vertices) or backup["keys"] != [k.name for k in keys]:
        raise RuntimeError("Geometry or key set changed; do not restore by outdated indices")
    for name, points in backup["coordinates"].items():
        for i, point in points.items():
            keys[name].data[int(i)].co = point
    for i in backup["coordinates"][keys[0].name]:
        index = int(i)
        obj.data.vertices[index].co = keys[0].data[index].co
    for i, colour in backup["colours"].items():
        colours.data[int(i)].color = colour
    for i, material in backup["materials"].items():
        obj.data.polygons[int(i)].material_index = material
    del obj[TAG]
    obj.data.update()
    bpy.context.view_layer.update()
    print("R03 neck fairing restored; backup retained", path)
