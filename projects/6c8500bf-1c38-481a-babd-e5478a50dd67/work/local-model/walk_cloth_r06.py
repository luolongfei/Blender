"""Independent native-cloth walk with hidden pre-roll and unchanged mocap keys.

Integrator workflow, serially in Blender:
    setup()                 # creates/saves reference-walk-cloth-r06.blend
    bake()                  # explicit blocking bake, externalizes and saves
    # reopen that exact saved work copy, then:
    verify_reopened()       # external-cache + original-clip preservation checks

The hidden -60..0 interval is simulation settling and a transition into the
first captured pose, NOT a natural start animation. Playback stays 1..80.
No operation runs on import. No old action, cache payload or master is replaced.
"""
import hashlib
import json
import math
import re
import runpy
import shutil
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SOURCE_ACTION = "R06 CMU 07_01 walk - forward gaze"
ACTION = "R06 CMU 07_01 walk - cloth preroll"
SOURCE_CLOTH = "R02 Cloth Skirt control"
CLOTH = "R06 Walk cloth"
CACHE_NAME = "R06_Walk_Skirt"
TARGET = "work/reconstruction-r02/reference-walk-cloth-r06.blend"
CHECK_FRAMES = (1., 7., 15., 23., 35., 45., 61., 79., 79.75)
CACHE_FRAMES = (1, 15, 35, 61, 80)
POSE_KEYS = ("co", "handle_left", "handle_right", "handle_left_type", "handle_right_type",
             "interpolation", "easing", "amplitude", "back", "period", "type")
HELPER = runpy.run_path(str(Path(__file__).with_name("full_body_review_r03.py")))


def _value(value):
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return list(value)


def _key_record(key):
    return {name: _value(getattr(key, name)) for name in POSE_KEYS}


def _curves(action):
    return HELPER["_fcurves"](action)


def _records(action, interval=True):
    result = []
    for curve in sorted(_curves(action), key=lambda c: (c.data_path, c.array_index)):
        if curve.modifiers or curve.sampled_points:
            raise RuntimeError("Only the inspected ordinary keyframed walk curves are supported")
        result.append({"path": curve.data_path, "index": curve.array_index,
                       "extrapolation": curve.extrapolation, "mute": curve.mute,
                       "auto_smoothing": curve.auto_smoothing,
                       "group": curve.group.name if curve.group else None,
                       "keys": [_key_record(k) for k in curve.keyframe_points
                                if not interval or 1 - 1e-6 <= k.co.x <= 79.75 + 1e-6]})
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _semantic_records(action):
    """LINEAR segments depend on coordinates, not dormant Bezier handles.
    Keep every non-linear channel exact; never relabel a changed fingerprint.
    """
    rows = _records(action)
    for row in rows:
        if all(key["interpolation"] == "LINEAR" for key in row["keys"]):
            for key in row["keys"]:
                for name in ("handle_left", "handle_right", "handle_left_type", "handle_right_type"):
                    del key[name]
    return rows


def _handle_changes(source, candidate):
    changed = []
    for a, b in zip(_records(source), _records(candidate)):
        for ka, kb in zip(a["keys"], b["keys"]):
            if any(ka[name] != kb[name] for name in ("handle_left", "handle_right", "handle_left_type", "handle_right_type")):
                changed.append({"path": a["path"], "index": a["index"], "frame": ka["co"][0],
                                "interpolation": ka["interpolation"]})
    return changed


def _set_frame(frame):
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    bpy.context.view_layer.update()


def _matrices(rig, frames):
    result = {}
    for frame in frames:
        _set_frame(frame)
        result[str(frame)] = {bone.name: [list(row) for row in bone.matrix]
                              for bone in rig.pose.bones if bone.bone.use_deform}
    return result


def _matrix_error(a, b):
    return max(abs(a[frame][bone][i][j] - b[frame][bone][i][j])
               for frame in a for bone in a[frame] for i in range(4) for j in range(4))


def _guard(require_target=False):
    project = project_path()
    file = Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or file == (project / "work/character.blend").resolve():
        raise RuntimeError("Saved independent production copy required")
    file.relative_to((project / "work").resolve())
    if require_target and file != (project / TARGET).resolve():
        raise RuntimeError("Open the dedicated reference-walk-cloth-r06.blend first")
    rig = bpy.data.objects.get("Character_Rig")
    if rig is None or rig.type != "ARMATURE":
        raise RuntimeError("Character_Rig required")
    if bpy.context.mode != "OBJECT":
        raise RuntimeError("Object Mode required")
    if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):
        raise RuntimeError("Active NLA blending needs explicit integration")
    if abs(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base - 30) > 1e-6:
        raise RuntimeError("Preserve the validated 30 fps timebase")
    return project, file, rig


def _rest_value(curve):
    path, index = curve.data_path, curve.array_index
    if re.fullmatch(r'pose\.bones\["[^"\n]+"\]\.location', path):
        return 0.
    if re.fullmatch(r'pose\.bones\["[^"\n]+"\]\.scale', path):
        return 1.
    if re.fullmatch(r'pose\.bones\["[^"\n]+"\]\.rotation_quaternion', path):
        return 1. if index == 0 else 0.
    if re.fullmatch(r'pose\.bones\["[^"\n]+"\]\.rotation_euler', path):
        return 0.
    if re.fullmatch(r'pose\.bones\["[^"\n]+"\]\.rotation_axis_angle', path):
        return 1. if index == 2 else 0.
    if re.fullmatch(r'pose\.bones\["CTRL_R02_Master"\]\["ik_(leg|arm)_[LR]"\]', path):
        return 0.
    raise RuntimeError("Unreviewed pre-roll channel: " + path)


def _restore_original_keys(curve, original):
    # Insertion may recalculate AUTO handles even though the original segments
    # use LINEAR interpolation. Restore all stored original handle coordinates
    # and settings after the one required curve.update(), then compare exactly.
    lookup = {round(float(k.co.x), 6): k for k in curve.keyframe_points}
    for record in original:
        key = lookup[round(record["co"][0], 6)]
        for name in POSE_KEYS:
            if name not in {"co", "handle_left", "handle_right"}:
                setattr(key, name, record[name])
        key.co = record["co"]
        key.handle_left, key.handle_right = record["handle_left"], record["handle_right"]


def _preroll(action):
    for curve in _curves(action):
        original = [_key_record(k) for k in curve.keyframe_points]
        if not original or abs(original[0]["co"][0] - 1) > 1e-6 or abs(original[-1]["co"][0] - 79.75) > 1e-6:
            raise RuntimeError("Expected every source curve to span 1..79.75")
        rest, first = _rest_value(curve), original[0]["co"][1]
        for frame, value in ((-60., rest), (-30., rest), (0., first)):
            key = curve.keyframe_points.insert(frame, value, options={"FAST"})
            key.interpolation = "BEZIER" if frame == -30 else "CONSTANT"
            key.handle_left_type = key.handle_right_type = "FREE"
            key.handle_left = (frame - 10, value)
            key.handle_right = (frame + 10, value)
        curve.update()
        _restore_original_keys(curve, original)


def _copy_settings(source, destination, pointers=()):
    copied = {}
    for prop in source.bl_rna.properties:
        name = prop.identifier
        if name == "rna_type" or prop.is_readonly or prop.type == "COLLECTION":
            continue
        if prop.type == "POINTER" and name not in pointers:
            continue
        value = getattr(source, name)
        try:
            setattr(destination, name, value)
            copied[name] = value.name if prop.type == "POINTER" and value else _value(value)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(f"Cannot preserve {source.bl_rna.identifier}.{name}: {error}")
    return copied


def _cache_snapshot(cache):
    return {"name": cache.name, "path": cache.filepath, "external": cache.use_external,
            "start": cache.frame_start, "end": cache.frame_end, "index": cache.index,
            "disk": cache.use_disk_cache}


def setup():
    project, source_file, rig = _guard()
    target = project / TARGET
    if target.exists() or bpy.data.actions.get(ACTION) or bpy.data.objects.get(CLOTH):
        raise RuntimeError("Dedicated walk output/action/cloth already exists; do not overwrite it")
    original = bpy.data.objects.get(SOURCE_CLOTH)
    source_action = bpy.data.actions.get(SOURCE_ACTION)
    if original is None or source_action is None or len(original.data.vertices) != 2176:
        raise RuntimeError("Expected the 2176-point calibrated cloth and forward-gaze walk")
    if original.data.shape_keys:
        raise RuntimeError("Unexpected shape-keyed cloth control; inspect it before duplicating")
    if original.constraints or original.animation_data:
        raise RuntimeError("Cloth object constraints/animation need an explicit copy strategy")
    old_cloth = next((m for m in original.modifiers if m.type == "CLOTH"), None)
    if old_cloth is None or not old_cloth.point_cache.use_external:
        raise RuntimeError("Expected the preserved external diagnostic cloth cache")
    start_property = old_cloth.point_cache.bl_rna.properties["frame_start"]
    if start_property.hard_min > -60:
        raise RuntimeError(f"This Blender PointCache starts at {start_property.hard_min}; negative pre-roll is unsupported. No timeline or old cache changed.")
    if [m.type for m in original.modifiers] != ["ARMATURE", "CLOTH", "SUBSURF", "SOLIDIFY"]:
        raise RuntimeError("Unexpected calibrated cloth modifier order")
    collision_collection = old_cloth.collision_settings.collection
    if collision_collection is None:
        raise RuntimeError("Dedicated body-collision collection missing")
    for obj in collision_collection.all_objects:
        arms = [m for m in obj.modifiers if m.type == "ARMATURE"]
        if not arms or any(m.object != rig for m in arms):
            raise RuntimeError("Collider is frozen or bound elsewhere: " + obj.name)
    source_record = _records(source_action, False)
    for curve in _curves(source_action):
        _rest_value(curve)
    source_digest = _digest(source_record)
    source_semantic = _semantic_records(source_action)
    scene = bpy.context.scene
    old_cache = _cache_snapshot(old_cloth.point_cache)
    old_visibility = {name: {"render": bpy.data.objects[name].hide_render,
                              "viewport": bpy.data.objects[name].hide_get()}
                      for name in (SOURCE_CLOTH, "Skirt_Sixteen_Pleats", "Skirt_Turned_Hem_Stitch", "Skirt_Center_Back_Seam")}
    HELPER["_assign"](rig, source_action)
    rig.data.pose_position = "POSE"
    source_matrices = _matrices(rig, CHECK_FRAMES)
    created = source_action.copy()
    created.name = ACTION
    created.use_fake_user = True
    _preroll(created)
    if _semantic_records(created) != source_semantic or _digest(_records(source_action, False)) != source_digest:
        raise RuntimeError("Source interval evaluation or preserved source action changed; inspect the retained copy")
    HELPER["_assign"](rig, created)
    copied_matrices = _matrices(rig, CHECK_FRAMES)
    matrix_error = _matrix_error(source_matrices, copied_matrices)
    if matrix_error > 1e-6:
        raise RuntimeError(f"Original walk body matrices changed: {matrix_error}")
    _set_frame(0)
    first = source_matrices[str(1.)]
    zero_error = max(abs(rig.pose.bones[name].matrix[i][j] - first[name][i][j])
                     for name in first for i in range(4) for j in range(4))
    if zero_error > 1e-6:
        raise RuntimeError(f"Pre-roll frame 0 does not match original frame 1: {zero_error}")
    rest_checks = {}
    for frame in (-60, -30):
        _set_frame(frame)
        # Head gaze/eyes/jaw may carry independent drivers. These checks cover
        # the pelvis, trunk and legs that drive the collision experiment.
        names = ("hips", "spine", "chest", "upperChest", "leftUpperLeg", "rightUpperLeg",
                 "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot")
        error = max(abs(rig.pose.bones[name].matrix[i][j] - rig.data.bones[name].matrix_local[i][j])
                    for name in names for i in range(4) for j in range(4))
        rest_checks[str(frame)] = error
        if error > 1e-5:
            raise RuntimeError(f"Hidden settling interval is not body rest at {frame}: {error}")
    # Do not even transiently copy a baked Cloth modifier. A fresh object and
    # mesh copy preserve geometry/deform weights while every modifier/cache is
    # genuinely newly constructed, independent of the retained 301-frame cache.
    clone = bpy.data.objects.new(CLOTH, original.data.copy())
    clone.parent = original.parent
    clone.parent_type = original.parent_type
    clone.parent_bone = original.parent_bone
    clone.matrix_parent_inverse = original.matrix_parent_inverse.copy()
    clone.matrix_world = original.matrix_world.copy()
    bpy.context.scene.collection.objects.link(clone)
    for source_group in original.vertex_groups:
        if source_group.index < len(clone.vertex_groups):
            group = clone.vertex_groups[source_group.index]
            if group.name != source_group.name:
                raise RuntimeError('Copied cloth deform group index/name mismatch: ' + source_group.name)
        else:
            group = clone.vertex_groups.new(name=source_group.name)
        if group.index != source_group.index:
            raise RuntimeError('Copied cloth deform group index changed: ' + source_group.name)
        group.lock_weight = source_group.lock_weight
    if len(clone.vertex_groups) != len(original.vertex_groups):
        raise RuntimeError('Copied cloth gained unexpected deform groups')
    for source_slot, slot in zip(original.material_slots, clone.material_slots):
        slot.link = source_slot.link
        slot.material = source_slot.material
    settings_record = {}
    for source_modifier in original.modifiers:
        modifier = clone.modifiers.new("R06 Walk " + source_modifier.type, source_modifier.type)
        if source_modifier.type == "CLOTH":
            settings_record["cloth"] = _copy_settings(source_modifier.settings, modifier.settings)
            settings_record["collision"] = _copy_settings(source_modifier.collision_settings, modifier.collision_settings, {"collection"})
            _copy_settings(source_modifier.settings.effector_weights, modifier.settings.effector_weights, {"collection"})
            modifier.collision_settings.collision_quality = max(6, source_modifier.collision_settings.collision_quality)
            cache = modifier.point_cache
            cache.frame_start, cache.frame_end, cache.frame_step = -60, 80, 1
            if cache.frame_start != -60 or cache.frame_end != 80:
                raise RuntimeError("PointCache clamped the requested negative interval; stop without baking")
            cache.name = CACHE_NAME
            cache.use_external = False
            cache.use_disk_cache = True
            if cache.is_baked or cache.as_pointer() == source_modifier.point_cache.as_pointer():
                raise RuntimeError("New cloth did not receive a fresh independent cache")
        else:
            _copy_settings(source_modifier, modifier, {"object"})
    clone.hide_viewport = False
    clone.hide_set(False)
    clone.hide_render = False
    clone.display_type = "TEXTURED"
    for name in old_visibility:
        bpy.data.objects[name].hide_render = True
        bpy.data.objects[name].hide_set(True)
    # The previous embedded tail motion belongs to another action; retain its
    # data while leaving it disabled in this walk-only cloth review copy.
    hair_enabled = rig.get("R04_native_hair_enabled")
    if hair_enabled is not None:
        rig["R04_native_hair_enabled"] = 0.
    scene.frame_start, scene.frame_end = 1, 80
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 80
    if _cache_snapshot(old_cloth.point_cache) != old_cache:
        raise RuntimeError("The preserved old cache configuration changed")
    report = {"source": str(source_file.relative_to(project)),
              "source_saved_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
              "target": TARGET, "action": ACTION, "source_action": SOURCE_ACTION,
              "source_interval": [1, 79.75], "source_interval_sha256": _digest(source_semantic),
              "source_full_keyframes_sha256": source_digest,
              "copied_interval_full_keyframes_sha256": _digest(_records(created)),
              "source_key_count": sum(len(row["keys"]) for row in source_record),
              "original_interval_semantically_identical": True,
              "original_keyframes_and_handles_unchanged": not _handle_changes(source_action, created),
              "changed_inactive_handles": _handle_changes(source_action, created),
              "handle_note": "AUTO handle changes, if any, are recorded only for LINEAR source intervals; coordinates/interpolation/settings and sampled bone matrices remain equivalent.",
              "original_interval_matrix_max_error": matrix_error,
              "frame_zero_matches_one_max_error": zero_error,
              "rest_interval_matrix_errors": rest_checks,
              "check_frames": CHECK_FRAMES, "cache_interval": [-60, 80], "playback": [1, 80],
              "old_cache_preserved": old_cache, "old_visibility": old_visibility,
              "old_hair_enabled": hair_enabled, "settings": settings_record,
              "collision_quality": max(6, old_cloth.collision_settings.collision_quality), "control_vertices": len(clone.data.vertices),
              "source_matrices": source_matrices,
              "status": "Configured; not baked or visually accepted",
              "scope": "Negative pre-roll is hidden simulation settling, not a natural start. Old garment/collision reports do not certify this walk."}
    clone["r06_walk_cloth_report"] = json.dumps(report)
    created["r06_preroll_scope"] = report["scope"]
    _set_frame(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    (target.parent / "cloth-walk-r06-setup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(target), "original_interval_matrix_error": matrix_error,
                      "source_fingerprint": source_digest, "cache": [-60, 80], "playback": [1, 80]}))
    return report


def _cloth_coordinates(obj, frames):
    result = {}
    for frame in frames:
        _set_frame(frame)
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            result[str(frame)] = [list(evaluated.matrix_world @ v.co) for v in mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
    return result


def _coordinate_error(a, b):
    if set(a) != set(b) or any(len(a[k]) != len(b[k]) for k in a):
        raise RuntimeError("Cache verification changed topology/frame set")
    return max(abs(v[c] - w[c]) for k in a for v, w in zip(a[k], b[k]) for c in range(3))


def bake():
    project, file, rig = _guard(True)
    if bpy.data.is_dirty:
        raise RuntimeError("Save this dedicated work copy before baking")
    control = bpy.data.objects[CLOTH]
    cloth = next(m for m in control.modifiers if m.type == "CLOTH")
    cache = cloth.point_cache
    if cache.is_baked or cache.use_external:
        raise RuntimeError("A walk cache already exists; preserve it instead of overwriting")
    if rig.animation_data.action.name != ACTION or (cache.frame_start, cache.frame_end) != (-60, 80):
        raise RuntimeError("Action/cache interval mismatch")
    report = json.loads(control["r06_walk_cloth_report"])
    if _digest(_semantic_records(rig.animation_data.action)) != report["source_interval_sha256"]:
        raise RuntimeError("Walk source interval changed after setup")
    external = file.parent / "cloth-cache-walk-r06"
    if external.exists():
        raise RuntimeError("External destination already exists; preserve it")
    scene = bpy.context.scene
    # The solver's own range is independent of playback. Temporarily expose the
    # full range to ptcache, then restore normal playback after baking.
    scene.frame_start, scene.frame_end = -60, 80
    scene.use_preview_range = False
    _set_frame(-60)
    start = time.monotonic()
    with bpy.context.temp_override(scene=scene, object=control, active_object=control, point_cache=cache):
        result = bpy.ops.ptcache.bake(bake=True)
    elapsed = time.monotonic() - start
    if not cache.is_baked or cache.is_outdated:
        raise RuntimeError("Bake did not produce a current completed cache: " + str(result))
    directory = file.parent / ("blendcache_" + file.stem)
    files = sorted(directory.glob(CACHE_NAME + "_*.bphys"))
    parsed = {}
    for path in files:
        match = re.fullmatch(re.escape(CACHE_NAME) + r"_(-?\d+)_(\d+)\.bphys", path.name)
        if match:
            parsed[int(match.group(1))] = path
    missing = sorted(set(range(-60, 81)) - set(parsed))
    missing_playback = sorted(set(range(1, 81)) - set(parsed))
    if missing_playback or any(f not in {-60, 0} for f in missing):
        raise RuntimeError("Missing required baked frames; do not externalize: " + repr(missing))
    # This Blender build produced every playback frame but no disk payload for
    # the initial rest frame or frame zero. Keep that observed limitation
    # explicit; only 1..80 is the delivered playback interval. Actual evaluated
    # vertices are compared before and after switching to external playback.
    report['observed_absent_preroll_disk_frames'] = missing
    report['external_cache_playback_scope'] = [1, 80]
    reference = _cloth_coordinates(control, CACHE_FRAMES)
    external.mkdir()
    manifest = []
    for path in files:
        destination = external / path.name
        shutil.copy2(path, destination)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError("External cache copy checksum mismatch")
        manifest.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    cache.filepath = bpy.path.relpath(str(external))
    cache.use_external = True
    cache.name = CACHE_NAME
    control.update_tag()
    bpy.context.view_layer.update()
    external_coordinates = _cloth_coordinates(control, CACHE_FRAMES)
    difference = _coordinate_error(reference, external_coordinates)
    if difference > 1e-6:
        raise RuntimeError(f"Internal/external cloth mismatch: {difference} m")
    scene.frame_start, scene.frame_end = 1, 80
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 80
    report.update({"status": "Baked/externalized; reopen and visual acceptance pending",
                   "bake_seconds": elapsed, "bake_operator": sorted(result),
                   "external_cache": str(external.relative_to(project)), "cache_manifest": manifest,
                   "internal_external_max_difference_m": difference,
                   "cached_frames": sorted(parsed), "cloth_reference_frames": CACHE_FRAMES})
    (file.parent / "cloth-walk-r06-reference-vertices.json").write_text(json.dumps(reference), encoding="utf-8")
    control["r06_walk_cloth_report"] = json.dumps(report)
    _set_frame(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(file))
    report["saved_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    (file.parent / "cloth-walk-r06-bake.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bake_seconds": elapsed, "frames": len(parsed), "external_difference_m": difference,
                      "external": str(external), "reopen_required": True}))
    return report


def verify_reopened():
    project, file, rig = _guard(True)
    control = bpy.data.objects[CLOTH]
    report = json.loads(control["r06_walk_cloth_report"])
    cache = next(m for m in control.modifiers if m.type == "CLOTH").point_cache
    external = Path(bpy.path.abspath(cache.filepath)).resolve()
    if not cache.use_external or external != (project / report["external_cache"]).resolve():
        raise RuntimeError("Reopened source is not reading the expected external cache")
    for row in report["cache_manifest"]:
        path = external / row["file"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Reopened external cache payload differs: " + row["file"])
    if rig.animation_data.action.name != ACTION:
        raise RuntimeError("Reopened action differs")
    if _digest(_semantic_records(rig.animation_data.action)) != report["source_interval_sha256"]:
        raise RuntimeError("Reopened original interval evaluation changed")
    if _digest(_records(bpy.data.actions[SOURCE_ACTION], False)) != report["source_full_keyframes_sha256"]:
        raise RuntimeError("Preserved source action changed")
    matrix_error = _matrix_error(report["source_matrices"], _matrices(rig, CHECK_FRAMES))
    reference = json.loads((file.parent / "cloth-walk-r06-reference-vertices.json").read_text())
    geometry_error = _coordinate_error(reference, _cloth_coordinates(control, CACHE_FRAMES))
    old = next(m for m in bpy.data.objects[SOURCE_CLOTH].modifiers if m.type == "CLOTH").point_cache
    old_match = _cache_snapshot(old) == report["old_cache_preserved"]
    if matrix_error > 1e-6 or geometry_error > 1e-6 or not old_match:
        raise RuntimeError(f"Reopen mismatch: body={matrix_error}, cloth={geometry_error}, old cache={old_match}")
    result = {"source": str(file.relative_to(project)), "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
              "body_matrix_max_error": matrix_error, "cloth_vertex_max_error_m": geometry_error,
              "old_cache_configuration_preserved": old_match,
              "original_interval_semantically_identical": True,
              "changed_inactive_handles": _handle_changes(bpy.data.actions[SOURCE_ACTION], rig.animation_data.action),
              "reopened_interval_full_keyframes_sha256": _digest(_records(rig.animation_data.action)),
              "external_files_verified": len(report["cache_manifest"]),
              "status": "Reopen numerics passed; render 1/15/35/61 for visible cloth/body contact review",
              "acceptance": "No new cloth collision/appearance acceptance inferred from preserved body matrices"}
    (file.parent / "cloth-walk-r06-reopen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _set_frame(1)
    print(json.dumps(result, ensure_ascii=False))
    return result
