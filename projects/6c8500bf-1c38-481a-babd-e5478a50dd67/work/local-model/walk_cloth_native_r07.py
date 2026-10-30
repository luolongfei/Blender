"""Embed the existing R06 native cloth simulation as native mesh shape keys.

Integrator, in a fresh background Blender on reference-walk-cloth-r06.blend:
    run()                  # saves NEW reference-walk-native-cloth-r07.blend
Reopen that new file in a fresh background process, then verify_reopened().

No bake, cache free, live execution, invented dynamics, or master overwrite.
Only the already simulated 1..80 interval is embedded. Between integer samples
the mesh uses linear interpolation; this is playback, not a new cloth solver.
Original simulation objects, modifiers, collider setup and .bphys are retained.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

RECOVERY = runpy.run_path(str(Path(__file__).with_name("walk_cloth_cache_recovery_r07.py")))
API = runpy.run_path(str(Path(__file__).with_name("walk_cloth_r06.py")))
BASE = project_path() / "work/reconstruction-r02"
TARGET = BASE / "reference-walk-native-cloth-r07.blend"
OBJECT = "R07 Native Walk Cloth playback"
COLLECTION = "R07 Native Walk Cloth Playback"
ACTION = "R07 Cloth baked payload - CMU walk 1-80"
FRAMES = tuple(range(1, 81))
FRACTIONS = (1.25, 7.5, 15.75, 35.375, 61.5, 79.75)
SURFACE_FRAMES = (1, 15, 35, 61, 80)
TOLERANCE = 2e-6


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _matrix_error(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


def _topology(mesh):
    return API["_digest"]({"vertices": len(mesh.vertices),
                           "faces": [list(p.vertices) for p in mesh.polygons],
                           "materials": [p.material_index for p in mesh.polygons]})


def _uv_digest(mesh):
    return API["_digest"]({uv.name: [list(item.uv) for item in uv.data] for uv in mesh.uv_layers})


def _guard():
    if not bpy.app.background:
        raise RuntimeError("Run in an isolated background process; do not change live Blender")
    project, file, rig = API["_guard"]()
    source = bpy.data.objects.get(API["CLOTH"])
    if source is None or len(source.data.vertices) != 2176 or source.data.shape_keys:
        raise RuntimeError("Expected the inspected 2176-vertex simulation control")
    if rig.animation_data.action.name != API["ACTION"]:
        raise RuntimeError("The payload belongs only to the preserved forward-gaze walk with pre-roll")
    reference = json.loads(RECOVERY["REFERENCE"].read_text(encoding="utf-8"))
    if RECOVERY["_inventory"](RECOVERY["SOURCE"]) != reference["manifest"]:
        raise RuntimeError("Original native payload differs from decoded reference")
    if set(reference["positions_world"]) != {str(f) for f in FRAMES}:
        raise RuntimeError("Exactly 80 decoded playback frames required")
    return file, rig, source, reference


def _expected(reference, frame):
    low, high = math.floor(frame), math.ceil(frame)
    a, b = reference["positions_world"][str(low)], reference["positions_world"][str(high)]
    t = frame - low
    return [[v * (1 - t) + w * t for v, w in zip(x, y)] for x, y in zip(a, b)]


def _parity(obj, reference):
    flags = [(m, m.show_viewport) for m in obj.modifiers]
    errors = {}
    try:
        for modifier, flag in flags:
            modifier.show_viewport = False
        for frame in FRAMES + FRACTIONS:
            actual = API["_cloth_coordinates"](obj, (frame,))[str(frame)]
            expected = _expected(reference, frame)
            if len(actual) != 2176:
                raise RuntimeError("Native playback control changed topology")
            errors[str(frame)] = max(abs(v[c] - w[c]) for v, w in zip(actual, expected) for c in range(3))
    finally:
        for modifier, flag in flags:
            modifier.show_viewport = flag
        API["_set_frame"](1)
    if max(errors.values()) > TOLERANCE:
        raise RuntimeError("Embedded playback differs from source payload: " + repr(errors))
    return errors


def _shape_action(obj):
    keys = obj.data.shape_keys
    if not keys or len(keys.key_blocks) != 81 or not keys.animation_data or not keys.animation_data.action:
        raise RuntimeError("Expected Basis + 80 native animated shape keys")
    curves = API["_curves"](keys.animation_data.action)
    if len(curves) != 80 or any(curve.modifiers for curve in curves):
        raise RuntimeError("Unexpected native shape-key animation channels")
    return keys.animation_data.action


def run():
    file, rig, source, reference = _guard()
    if TARGET.exists() or bpy.data.objects.get(OBJECT) or bpy.data.actions.get(ACTION):
        raise RuntimeError("Native playback output already exists; preserve it")
    if [m.type for m in source.modifiers] != ["ARMATURE", "CLOTH", "SUBSURF", "SOLIDIFY"]:
        raise RuntimeError("Unreviewed original simulation modifier stack")
    source_digest = API["_digest"](API["_records"](rig.animation_data.action, False))
    simulation_visibility = {"hide_viewport": source.hide_viewport, "hide_render": source.hide_render,
                             "hide_set": source.hide_get()}
    simulation_cache = RECOVERY["_state"](next(m for m in source.modifiers if m.type == "CLOTH").point_cache)
    # Hide before frame stepping, so the unbaked source cannot resimulate or write
    # into its old internal directory while the native playback is checked.
    source.hide_viewport = True
    source.hide_render = True
    source.hide_set(True)
    matrices = {}
    for frame in FRAMES:
        API["_set_frame"](frame)
        matrices[frame] = rig.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
    world_motion = max(_matrix_error(matrices[1], value) for value in matrices.values())
    if world_motion > 1e-8:
        raise RuntimeError("Rig object transform changes; world-space interpolation needs explicit adaptation")
    # Only the rig object is the parent. No bone parent, Armature or Cloth
    # modifier is added: captured hip motion already exists in payload positions.
    mesh = source.data.copy()
    mesh.name = "R07 simulated cloth payload mesh"
    obj = bpy.data.objects.new(OBJECT, mesh)
    collection = bpy.data.collections.get(COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(COLLECTION)
        bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)
    obj.parent = rig
    obj.parent_type = "OBJECT"
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    for group in source.vertex_groups:
        created = obj.vertex_groups.new(name=group.name)
        created.lock_weight = group.lock_weight
    for index, slot in enumerate(source.material_slots):
        obj.material_slots[index].link = slot.link
        obj.material_slots[index].material = slot.material
    inverse = matrices[1].inverted()
    local = {frame: [list(inverse @ Vector(point)) for point in reference["positions_world"][str(frame)]]
             for frame in FRAMES}
    mesh.vertices.foreach_set("co", [v for point in local[1] for v in point])
    mesh.update()
    obj.shape_key_add(name="Basis", from_mix=False)
    for frame in FRAMES:
        key = obj.shape_key_add(name=f"R07 simulated frame {frame:03d}", from_mix=False)
        key.data.foreach_set("co", [v for point in local[frame] for v in point])
        key.slider_min, key.slider_max = 0., 1.
        for at, value in ((frame - 1, 0.), (frame, 1.), (frame + 1, 0.)):
            if 1 <= at <= 80:
                key.value = value
                key.keyframe_insert(data_path="value", frame=at, group="Recorded cloth simulation")
        key.value = 0.
    action = obj.data.shape_keys.animation_data.action
    action.name = ACTION
    for curve in API["_curves"](action):
        curve.extrapolation = "CONSTANT"
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"
        curve.update()
    obj.data.shape_keys.use_relative = True
    for original in source.modifiers:
        if original.type in {"SUBSURF", "SOLIDIFY"}:
            modifier = obj.modifiers.new(original.name, original.type)
            API["_copy_settings"](original, modifier, {"object"})
    obj.hide_viewport = False
    obj.hide_render = False
    obj.hide_set(False)
    obj.display_type = "TEXTURED"
    if _topology(mesh) != _topology(source.data) or _uv_digest(mesh) != _uv_digest(source.data):
        raise RuntimeError("Simulation topology, UVs or material assignments changed")
    _shape_action(obj)
    errors = _parity(obj, reference)
    setup = json.loads(source["r06_walk_cloth_report"])
    matrix_error = API["_matrix_error"](setup["source_matrices"], API["_matrices"](rig, API["CHECK_FRAMES"]))
    if matrix_error > 1e-6 or source_digest != API["_digest"](API["_records"](rig.animation_data.action, False)):
        raise RuntimeError("Body action or source matrices changed")
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 80
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 80
    surface = API["_cloth_coordinates"](obj, SURFACE_FRAMES)
    API["_set_frame"](1)
    report = {"source_blend": str(file.relative_to(project_path())), "source_blend_sha256": _sha(file),
              "target": str(TARGET.relative_to(project_path())), "object": OBJECT,
              "payload_reference": str(RECOVERY["REFERENCE"].relative_to(project_path())),
              "payload_reference_sha256": _sha(RECOVERY["REFERENCE"]),
              "payload_manifest": reference["manifest"], "source_cache": simulation_cache,
              "source_visibility_before": simulation_visibility,
              "source_simulation_preserved_hidden": source.name,
              "frames": FRAMES, "fractional_samples": FRACTIONS,
              "max_payload_error_m": max(errors.values()), "payload_errors_m": errors,
              "body_matrix_error": matrix_error, "body_action_sha256": source_digest,
              "rig_world_matrix_change": world_motion, "topology_sha256": _topology(mesh),
              "uv_sha256": _uv_digest(mesh), "shape_key_count": 81,
              "shape_action_sha256": API["_digest"](API["_records"](action, False)),
              "surface_reference": surface,
              "status": "Native embedded simulation playback saved; reopen and appearance checks pending",
              "scope": "Actual PBD cloth positions from the existing native bphys cache at integer frames 1..80; linear interpolation between samples. No new simulation or artificial oscillation. Playback is bound to this exact walk; changing body action does not generate matching cloth.",
              "known_limitations": ["Hidden pre-roll is retained in source action/cache, but only 1..80 is embedded", "Numerical parity does not establish collision-free or visually accepted cloth"]}
    obj["r07_native_cloth_report"] = json.dumps(report)
    action["r07_simulation_source"] = report["scope"]
    text = bpy.data.texts.new("R07 Cloth simulation provenance.json")
    text.write(json.dumps({key: value for key, value in report.items() if key != "surface_reference"}, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(TARGET))
    report["saved_sha256"] = _sha(TARGET)
    _write(BASE / "cloth-native-walk-r07.json", report)
    print(json.dumps({"saved": str(TARGET), "payload_max_error_m": max(errors.values()),
                      "body_matrix_error": matrix_error, "shape_keys": 81, "reopen_required": True}))
    return report


def verify_reopened(expected_relative=None):
    file, rig, source, reference = _guard()
    expected=(project_path()/expected_relative).resolve() if expected_relative else TARGET.resolve()
    if file != expected:
        raise RuntimeError("Open the exact saved native playback copy")
    obj = bpy.data.objects[OBJECT]
    report = json.loads(obj["r07_native_cloth_report"])
    if _sha(RECOVERY["REFERENCE"]) != report["payload_reference_sha256"]:
        raise RuntimeError("Payload reference identity changed")
    if [m.type for m in obj.modifiers] != ["SUBSURF", "SOLIDIFY"]:
        raise RuntimeError("Native playback has unexpected deformation modifiers")
    if _topology(obj.data) != report["topology_sha256"] or _uv_digest(obj.data) != report["uv_sha256"]:
        raise RuntimeError("Topology, UVs or materials changed")
    if API["_digest"](API["_records"](_shape_action(obj), False)) != report["shape_action_sha256"]:
        raise RuntimeError("Native shape action changed")
    if API["_digest"](API["_records"](rig.animation_data.action, False)) != report["body_action_sha256"]:
        raise RuntimeError("Body action changed")
    if not source.hide_viewport or not source.hide_render:
        raise RuntimeError("Preserved original simulation became visible")
    errors = _parity(obj, reference)
    surface_error = API["_coordinate_error"](report["surface_reference"], API["_cloth_coordinates"](obj, SURFACE_FRAMES))
    setup = json.loads(source["r06_walk_cloth_report"])
    body_error = API["_matrix_error"](setup["source_matrices"], API["_matrices"](rig, API["CHECK_FRAMES"]))
    if surface_error > TOLERANCE or body_error > 1e-6:
        raise RuntimeError("Reopened visible surface/body differs")
    result = {"source": str(file.relative_to(project_path())), "sha256": _sha(file),
              "payload_max_error_m": max(errors.values()), "payload_errors_m": errors,
              "surface_max_error_m": surface_error, "body_matrix_error": body_error,
              "frames_checked": 80, "fractional_frames_checked": FRACTIONS,
              "source_simulation_retained": source.name,
              "status": "Reopen numerics passed; render 1/15/35/61 and continuous motion for visible cloth review"}
    _write(BASE / "cloth-native-walk-r07-reopen.json", result)
    API["_set_frame"](1)
    print(json.dumps({k: result[k] for k in ("payload_max_error_m", "surface_max_error_m", "body_matrix_error", "status")}))
    return result
