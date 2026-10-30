"""Recover an existing native cloth cache without baking or freeing any cache.

First run extract_payload() with a Python providing compression.zstd (3.14+).
Then in an isolated BACKGROUND Blender process opened on the saved walk copy:
    diagnose()             # isolated cache copies, no blend save; emits findings
    recover()              # only after payload parity, saves NEW recovery copy
Reopen that exact recovery file in a fresh background process:
    verify_reopened()

No automatic entry point and no bake/free operators. Never run in live Blender.
The payload reference is decoded from existing simulation files, not fabricated
motion. It cannot recover the lost in-memory reference from the failed process.
"""
import hashlib
import json
import math
import re
import runpy
import shutil
import struct
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

NAME = "R06_Walk_Skirt"
OBJECT = "R06 Walk cloth"
BASE = project_path() / "work/reconstruction-r02"
SOURCE = BASE / "cloth-cache-walk-r06"
REFERENCE = BASE / "cloth-walk-r07-payload-reference.json"
RECOVERED = BASE / "reference-walk-cloth-recovered-r07.blend"
API_PATH = Path(__file__).with_name("walk_cloth_r06.py")
FRAMES = tuple(range(1, 81))
PATTERN = re.compile(re.escape(NAME) + r"_(-?\d+)_(\d+)\.bphys")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(directory):
    result = []
    for path in sorted(directory.glob(NAME + "_*.bphys")):
        match = PATTERN.fullmatch(path.name)
        data = path.read_bytes()
        if not match or data[:8] != b"BPHYSICS" or len(data) < 20:
            raise RuntimeError("Invalid native cache header: " + str(path))
        typeflag, count, mask = struct.unpack("<III", data[8:20])
        if typeflag & 0xffff != 2 or count != 2176 or mask != 22:
            raise RuntimeError("Unreviewed cloth payload layout: " + str(path))
        result.append({"file": path.name, "frame": int(match[1]), "index": int(match[2]),
                       "bytes": len(data), "sha256": _sha(path), "typeflag": typeflag,
                       "points": count, "data_mask": mask})
    return result


def _locations(path):
    # Blender build e13b728c6880 pointcache.cc: location is first of mask 22;
    # compressed channel 3 is zstd byte transpose + per-byte delta coding.
    from compression import zstd
    data = path.read_bytes()
    typeflag, count, mask = struct.unpack("<III", data[8:20])
    if (typeflag, count, mask) != (65538, 2176, 22):
        raise RuntimeError("Only the inspected native compressed cloth layout is supported")
    compression, size = data[20], struct.unpack("<I", data[21:25])[0]
    if compression != 3:
        raise RuntimeError("Unreviewed point-cache compression")
    filtered = zstd.decompress(data[25:25 + size])
    if len(filtered) != count * 12:
        raise RuntimeError("Decoded location byte count mismatch")
    raw = bytearray(count * 12)
    for byte in range(12):
        previous = 0
        for item in range(count):
            previous = (previous + filtered[byte * count + item]) & 255
            raw[item * 12 + byte] = previous
    points = [list(point) for point in struct.iter_unpack("<fff", raw)]
    if any(not math.isfinite(v) or abs(v) > 100 for point in points for v in point):
        raise RuntimeError("Non-finite/unreasonable simulation coordinates")
    return points


def extract_payload():
    """Standard-library file inspection only; no Blender import or process."""
    manifest = _inventory(SOURCE)
    by_frame = {row["frame"]: row for row in manifest}
    if len(by_frame) != len(manifest) or {row["index"] for row in manifest} != {0}:
        raise RuntimeError("Duplicate frames or unexpected stack index")
    if not set(FRAMES).issubset(by_frame):
        raise RuntimeError("Original cache lacks required playback frames")
    result = {"source_directory": str(SOURCE.relative_to(project_path())),
              "manifest": manifest, "frames": FRAMES,
              "positions_world": {str(f): _locations(SOURCE / by_frame[f]["file"]) for f in FRAMES},
              "missing_preroll_frames": sorted(set(range(-60, 1)) - set(by_frame)),
              "scope": "Existing native simulation payload decoded directly; no resimulation or interpolation.",
              "source_code": "https://github.com/blender/blender/blob/e13b728c6880/source/blender/blenkernel/intern/pointcache.cc"}
    if REFERENCE.exists():
        old = json.loads(REFERENCE.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(result)):
            raise RuntimeError("Existing reference differs; retain it and use a new revision")
    else:
        REFERENCE.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"reference": str(REFERENCE), "files": len(manifest),
                      "frames_decoded": len(FRAMES), "missing_preroll": result["missing_preroll_frames"]}))
    return result


def _runtime():
    import bpy
    if not bpy.app.background:
        raise RuntimeError("Recovery diagnostics are background-only; leave live Blender untouched")
    api = runpy.run_path(str(API_PATH))
    project, file, rig = api["_guard"]()
    obj = bpy.data.objects[OBJECT]
    cloth = next(m for m in obj.modifiers if m.type == "CLOTH")
    if rig.animation_data.action.name != api["ACTION"] or len(obj.data.vertices) != 2176:
        raise RuntimeError("Wrong action/control mesh")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if _inventory(SOURCE) != reference["manifest"]:
        raise RuntimeError("Preserved original external payload has changed")
    return bpy, api, file, rig, obj, cloth, reference


def _state(cache):
    fields = ("name", "filepath", "index", "frame_start", "frame_end", "frame_step",
              "use_external", "use_disk_cache", "is_baked", "is_outdated", "is_frame_skip", "info")
    return {key: getattr(cache, key) for key in fields if hasattr(cache, key)}


def _probe_dir(label, reference, positive_only=False):
    path = BASE / label
    if path.exists():
        raise RuntimeError("Diagnostic output exists; preserve it: " + str(path))
    path.mkdir()
    for row in reference["manifest"]:
        if positive_only and row["frame"] <= 0:
            continue
        shutil.copy2(SOURCE / row["file"], path / row["file"])
        if _sha(path / row["file"]) != row["sha256"]:
            raise RuntimeError("Cache copy mismatch")
    return path


def _load(bpy, obj, cache, directory):
    # Crucial order: while INTERNAL, assigning filepath/name invokes Blender's
    # cache rename callback, even if name is unchanged. Enter external mode
    # before assigning a directory; this only reads/scans existing payloads.
    history = [{"step": "before", "cache": _state(cache)}]
    cache.use_external = True
    history.append({"step": "external_flag", "cache": _state(cache)})
    if cache.name != NAME:
        cache.name = NAME
    if cache.index != 0:
        cache.index = 0
    cache.filepath = bpy.path.relpath(str(directory))
    history.append({"step": "directory_scanned", "cache": _state(cache)})
    if not cache.is_baked or not cache.use_external or cache.frame_end != 80:
        raise RuntimeError("External cache did not load as baked through frame 80")
    obj.update_tag()
    bpy.context.view_layer.update()
    history.append({"step": "after_update", "cache": _state(cache)})
    return history


def _raw_parity(bpy, api, obj, cloth, reference, frames=FRAMES):
    # Native cloth stores world-space positions before surface modifiers.
    # Disable only post-cloth viewport modifiers temporarily, restore in finally.
    index = list(obj.modifiers).index(cloth)
    after = list(obj.modifiers)[index + 1:]
    flags = [(m, m.show_viewport) for m in after]
    old_frame, old_subframe = bpy.context.scene.frame_current, bpy.context.scene.frame_subframe
    errors = {}
    try:
        for modifier, flag in flags:
            modifier.show_viewport = False
        for frame in frames:
            coordinates = api["_cloth_coordinates"](obj, (frame,))[str(frame)]
            payload = reference["positions_world"][str(frame)]
            if len(coordinates) != len(payload):
                raise RuntimeError("Cloth-only vertex count differs from payload")
            errors[str(frame)] = max(abs(a[c] - b[c]) for a, b in zip(coordinates, payload) for c in range(3))
    finally:
        for modifier, flag in flags:
            modifier.show_viewport = flag
        bpy.context.scene.frame_set(old_frame, subframe=old_subframe)
        bpy.context.view_layer.update()
    return errors


def diagnose(label="cloth-cache-walk-r07-probe", positive_only=False):
    """No save, bake, free or source payload edits. Leaves scratch external active."""
    bpy, api, file, rig, obj, cloth, reference = _runtime()
    original = _state(cloth.point_cache)
    old_cache = _state(next(m for m in bpy.data.objects[api["SOURCE_CLOTH"]].modifiers if m.type == "CLOTH").point_cache)
    report = {"source_blend": str(file.relative_to(project_path())), "source_sha256": _sha(file),
              "cache_before": original, "old_cache_before": old_cache,
              "internal_files_before": _inventory(file.parent / ("blendcache_" + file.stem)),
              "positive_only": positive_only, "payload_file_count": len(reference["manifest"])}
    output = BASE / (label + ".json")
    if output.exists():
        raise RuntimeError("Existing diagnostic report preserved")
    try:
        directory = _probe_dir(label, reference, positive_only)
        report["cache_copy"] = str(directory.relative_to(project_path()))
        report["load_steps"] = _load(bpy, obj, cloth.point_cache, directory)
        report["errors_by_frame_m"] = _raw_parity(bpy, api, obj, cloth, reference)
        report["max_payload_error_m"] = max(report["errors_by_frame_m"].values())
        report["cache_after"] = _state(cloth.point_cache)
        report["status"] = "payload parity passed" if report["max_payload_error_m"] <= 2e-6 else "payload mismatch; do not save as recovered"
        report["old_cache_after"] = _state(next(m for m in bpy.data.objects[api["SOURCE_CLOTH"]].modifiers if m.type == "CLOTH").point_cache)
        report["note"] = "Positive-only scanning starts at 1; Blender cloth initializes at start instead of reading its cached shape. Do not silently accept frame 1 mismatch."
    except Exception as error:
        report["status"] = "diagnostic stopped"
        report["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "max_payload_error_m", "error", "cache_before", "cache_after") if key in report}))
    return report


def recover(label="cloth-cache-walk-r07-recovered"):
    """Use intact existing files; save a new work copy only after 80-frame parity."""
    if RECOVERED.exists():
        raise RuntimeError("Recovery copy already exists; do not overwrite")
    report = diagnose(label)
    if report.get("max_payload_error_m", 1.) > 2e-6:
        raise RuntimeError("Payload parity failed; no blend saved")
    bpy, api, file, rig, obj, cloth, reference = _runtime()
    setup = json.loads(obj["r06_walk_cloth_report"])
    if api["_digest"](api["_semantic_records"](rig.animation_data.action)) != setup["source_interval_sha256"]:
        raise RuntimeError("Original walk interval changed")
    matrix_error = api["_matrix_error"](setup["source_matrices"], api["_matrices"](rig, api["CHECK_FRAMES"]))
    if matrix_error > 1e-6:
        raise RuntimeError("Original body matrices changed")
    report["body_matrix_error"] = matrix_error
    report["payload_reference"] = str(REFERENCE.relative_to(project_path()))
    report["payload_reference_sha256"] = _sha(REFERENCE)
    report["recovered_file"] = str(RECOVERED.relative_to(project_path()))
    report["status"] = "existing native payload recovered; reopen and visual checks pending"
    report["scope"] = "This verifies existing disk simulation playback, not lost pre-switch memory parity or visual collision quality."
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 80
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 80
    api["_set_frame"](1)
    report["render_surface_reference"] = api["_cloth_coordinates"](obj, api["CACHE_FRAMES"])
    api["_set_frame"](1)
    obj["r07_cache_recovery_report"] = json.dumps(report)
    bpy.ops.wm.save_as_mainfile(filepath=str(RECOVERED))
    report["saved_sha256"] = _sha(RECOVERED)
    (BASE / "cloth-walk-r07-recovery.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def verify_reopened():
    bpy, api, file, rig, obj, cloth, reference = _runtime()
    if file != RECOVERED.resolve():
        raise RuntimeError("Reopen exact recovery copy first")
    report = json.loads(obj["r07_cache_recovery_report"])
    if _sha(REFERENCE) != report["payload_reference_sha256"]:
        raise RuntimeError("Reference identity changed")
    directory = Path(bpy.path.abspath(cloth.point_cache.filepath)).resolve()
    if directory != (project_path() / report["cache_copy"]).resolve() or not cloth.point_cache.use_external:
        raise RuntimeError("Reopened external path differs")
    if _inventory(directory) != reference["manifest"]:
        raise RuntimeError("Reopened cache payload differs")
    errors = _raw_parity(bpy, api, obj, cloth, reference)
    surface_error = api["_coordinate_error"](report["render_surface_reference"], api["_cloth_coordinates"](obj, api["CACHE_FRAMES"]))
    if max(errors.values()) > 2e-6 or surface_error > 2e-6:
        raise RuntimeError("Reopened geometry differs")
    result = {"source": str(file.relative_to(project_path())), "sha256": _sha(file),
              "cache": _state(cloth.point_cache), "raw_errors_m": errors,
              "surface_error_m": surface_error, "status": "reopen numerics passed; render 1/15/35/61 next"}
    (BASE / "cloth-walk-r07-recovery-reopen.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    api["_set_frame"](1)
    print(json.dumps({"maximum_raw_error_m": max(errors.values()), "surface_error_m": surface_error}))
    return result


def checkpoint_internal(label="reference-walk-cloth-internal-checkpoint-r07"):
    """Future bake hook: call immediately after bake, BEFORE any cache setters.

    Copy current payload to checkpoint-named internal directory, then save-copy.
    Do not call on the already failed/unbaked source; that state is not a bake.
    """
    bpy, api, file, rig, obj, cloth, reference = _runtime()
    cache = cloth.point_cache
    if not cache.is_baked or cache.use_external or cache.is_outdated:
        raise RuntimeError("Current completed internal bake required")
    target = BASE / (label + ".blend")
    folder = BASE / ("blendcache_" + label)
    source = file.parent / ("blendcache_" + file.stem)
    manifest = _inventory(source)
    if not set(FRAMES).issubset({row["frame"] for row in manifest}) or target.exists() or folder.exists():
        raise RuntimeError("Missing playback payload or checkpoint destination exists")
    folder.mkdir()
    for row in manifest:
        shutil.copy2(source / row["file"], folder / row["file"])
        if _sha(folder / row["file"]) != row["sha256"]:
            raise RuntimeError("Checkpoint copy checksum mismatch")
    coordinates = api["_cloth_coordinates"](obj, api["CACHE_FRAMES"])
    report = {"cache": _state(cache), "manifest": manifest, "coordinates": coordinates,
              "source": str(file.relative_to(project_path())), "checkpoint": str(target.relative_to(project_path()))}
    (BASE / (label + ".json")).write_text(json.dumps(report), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(target), copy=True)
    return report
