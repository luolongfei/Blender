"""Actual-mesh expression demonstration, rendered only in a disposable process.

Load work/expression-authoring/current.blend in background Blender, then use
--test-expression-video for one EEVEE test frame or --render-expression-video
after the final geometry has been saved. This never writes the source Blend.
"""
from pathlib import Path
import hashlib
import json
import math
import shutil
import subprocess
import time
import bpy
from mathutils import Vector

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / "work/expression-authoring"
FRAMES = WORK / "animation-frames"
PREVIEW = PROJECT / "previews/face-expression-demo.mp4"
FPS = 15
SECONDS = 13
SIZE = 640
VERSION = "actual-mesh-expression-demo-01"

# Times are seconds; all non-listed keys return to zero at each control point.
# Expression changes are separated by gentle neutral transitions. The speaking
# section uses real mouth keys without claiming synchronization to recorded audio.
CUES = [
    (0.00, {}, "自然"),
    (0.55, {}, "自然"),
    (10/FPS, {"blink": 1.0}, "眨眼"),
    (0.86, {}, "眨眼"),
    (1.12, {}, "自然"),
    (1.58, {"happy_eye": .35, "happy_mouth": .55, "brow_happy": .35}, "微笑"),
    (2.08, {"happy_eye": .85, "happy_mouth": .85, "brow_happy": .55}, "开心"),
    (2.48, {"happy_eye": .85, "happy_mouth": .85, "brow_happy": .55}, "开心"),
    (2.85, {}, "自然"),
    (3.22, {"sad_eye": .65, "sad_mouth": .75, "brow_sad": .75}, "难过"),
    (3.68, {"sad_eye": .65, "sad_mouth": .75, "brow_sad": .75}, "难过"),
    (4.02, {}, "自然"),
    (4.37, {"angry_eye": .55, "angry_mouth": .55, "brow_angry": .65}, "生气"),
    (4.80, {"angry_eye": .55, "angry_mouth": .55, "brow_angry": .65}, "生气"),
    (5.12, {}, "自然"),
    (5.47, {"surprised_eye": .70, "surprised_mouth": .70,
            "brow_surprised": .70}, "惊讶"),
    (5.93, {"surprised_eye": .70, "surprised_mouth": .70,
            "brow_surprised": .70}, "惊讶"),
    (6.25, {}, "自然"),
    (6.57, {"aa": .80}, "口型 aa"),
    (6.78, {"aa": .30}, "口型 aa"),
    (7.00, {"ih": .75}, "口型 ih"),
    (7.18, {}, "口型过渡"),
    (7.40, {"ou": .80}, "口型 ou"),
    (7.58, {}, "口型过渡"),
    (7.80, {"ee": .75}, "口型 ee"),
    (7.98, {}, "口型过渡"),
    (8.20, {"oh": .80}, "口型 oh"),
    (8.40, {}, "口型过渡"),
    (8.60, {"aa": .55}, "连续口型"),
    (8.80, {"ee": .50}, "连续口型"),
    (9.00, {"ou": .55}, "连续口型"),
    (9.20, {"ih": .50}, "连续口型"),
    (9.40, {"oh": .65}, "连续口型"),
    (9.65, {}, "自然"),
    (10.08, {"happy_eye": .30, "happy_mouth": .45, "brow_happy": .25}, "微笑"),
    (10.55, {"happy_eye": .30, "happy_mouth": .45, "brow_happy": .25}, "微笑"),
    (10.90, {"happy_mouth": .30, "brow_happy": .15}, "微笑"),
    (11.20, {"happy_mouth": .30, "brow_happy": .15}, "微笑"),
    (170/FPS, {"blink": 1.0, "happy_mouth": .30, "brow_happy": .15}, "眨眼"),
    (11.53, {"happy_mouth": .30, "brow_happy": .15}, "微笑"),
    (12.05, {}, "自然"),
    (13.00, {}, "自然"),
]


def values_at(seconds):
    if seconds <= CUES[0][0]:
        return dict(CUES[0][1])
    for first, second in zip(CUES, CUES[1:]):
        if first[0] <= seconds <= second[0]:
            alpha = (seconds-first[0])/(second[0]-first[0])
            alpha = alpha*alpha*(3-2*alpha)
            names = set(first[1]) | set(second[1])
            return {name: first[1].get(name, 0.0)*(1-alpha)
                    + second[1].get(name, 0.0)*alpha for name in names}
    return dict(CUES[-1][1])


def facial_keys():
    unique = {}
    objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.get("facial_role") and obj.data.shape_keys:
            keys = obj.data.shape_keys
            unique[keys.name] = keys
            objects.append(obj.name)
    if not unique:
        raise RuntimeError("No authored facial shape keys in the loaded model.")
    required = {name for _, values, _ in CUES for name in values}
    available = {block.name for keys in unique.values() for block in keys.key_blocks}
    if not required <= available:
        raise RuntimeError("Missing expression keys: " + repr(sorted(required-available)))
    return list(unique.values()), objects, required


def setup():
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    else:
        raise RuntimeError("This Blender build has no EEVEE renderer.")
    if hasattr(scene, "eevee"):
        for name in ("taa_render_samples", "taa_samples"):
            if hasattr(scene.eevee, name):
                setattr(scene.eevee, name, 32)
    scene.render.resolution_x = SIZE
    scene.render.resolution_y = 744
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = FPS*SECONDS
    camera = bpy.data.objects.get("Expression demonstration camera")
    if camera is None:
        data = bpy.data.cameras.new("Expression demonstration camera")
        camera = bpy.data.objects.new("Expression demonstration camera", data)
        scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = .350
    target = Vector((0, -.006, 1.429))
    camera.location = target + Vector((0, -3, 0))
    camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    if scene.world and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.type == "BACKGROUND":
                node.inputs["Strength"].default_value = .48
    for name, energy in [("Key · large softbox", 235), ("Fill · cool softbox", 80)]:
        light = bpy.data.objects.get(name)
        if light:
            light.data.energy = energy
    return scene


def prepare(save=False):
    scene = setup()
    groups, objects, used = facial_keys()
    for group in groups:
        group.animation_data_clear()
        for key in group.key_blocks:
            if key.name != "Basis":
                key.value = 0
    # Values are sampled at every actual render frame. This avoids interpolation
    # overshoot and preserves the 0..1 range even with layered action versions.
    ranges = {name: [1.0, 0.0] for name in used}
    for frame in range(scene.frame_start, scene.frame_end+1):
        seconds = (frame-1)/FPS
        values = values_at(seconds)
        for name in used:
            value = values.get(name, 0.0)
            if not 0 <= value <= 1:
                raise ValueError("Expression value outside 0..1.")
            ranges[name][0] = min(ranges[name][0], value)
            ranges[name][1] = max(ranges[name][1], value)
        for group in groups:
            for name in used:
                key = group.key_blocks.get(name)
                if key:
                    key.value = values.get(name, 0.0)
                    key.keyframe_insert("value", frame=frame, group="Facial demonstration")
    scene.frame_set(1)
    bpy.context.view_layer.update()
    report = {
        "version": VERSION, "source": bpy.data.filepath,
        "engine": scene.render.engine, "fps": FPS, "frames": scene.frame_end,
        "seconds": SECONDS, "resolution": [SIZE, 744],
        "objects": objects, "shape_key_data_blocks": len(groups),
        "animated_keys": sorted(used), "value_ranges": ranges,
        "cues": [{"seconds": t, "label": label, "values": values}
                 for t, values, label in CUES],
        "actual_mesh_render": True, "audio": False,
    }
    source = Path(bpy.data.filepath)
    if source.is_file():
        report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    WORK.mkdir(parents=True, exist_ok=True)
    if save:
        path = WORK / "expression-demo.blend"
        if source.resolve() == path.resolve():
            raise RuntimeError("Load current.blend, not the demonstration output.")
        scene.render.filepath = str(FRAMES / "frame_")
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        report["blend"] = str(path)
    return scene, report


def test_frame():
    scene, report = prepare(save=False)
    scene.frame_set(1)
    scene.render.filepath = str(WORK / "animation-test.png")
    started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    report["test_render_seconds"] = time.monotonic()-started
    report["test_frame"] = scene.render.filepath
    (WORK / "animation-test-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("EXPRESSION_ANIMATION_TEST", json.dumps({
        key: report[key] for key in ("engine", "test_render_seconds", "test_frame",
                                    "shape_key_data_blocks", "resolution")}), flush=True)


def benchmark():
    scene, report = prepare(save=False)
    report["timings"] = []
    if hasattr(scene, "eevee"):
        report["eevee_sampling"] = {
            name: getattr(scene.eevee, name) for name in
            ("taa_render_samples", "taa_samples", "use_shadow_jitter")
            if hasattr(scene.eevee, name)}
    cycles = bpy.context.preferences.addons.get("cycles")
    if cycles:
        preferences = cycles.preferences
        try:
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            report["cycles_devices"] = [{"name": d.name, "type": d.type}
                                        for d in preferences.devices]
        except Exception as error:
            report["cycles_device_error"] = str(error)
    print("ANIMATION_BENCHMARK_INFO", json.dumps({
        key: report.get(key) for key in ("eevee_sampling", "cycles_devices",
                                         "cycles_device_error")}), flush=True)
    for frame in (1, 11, 32):
        scene.frame_set(frame)
        scene.render.filepath = str(WORK / f"animation-benchmark-{frame:03d}.png")
        started = time.monotonic()
        bpy.ops.render.render(write_still=True)
        elapsed = time.monotonic()-started
        report["timings"].append({"frame": frame, "seconds": elapsed})
        print("ANIMATION_BENCHMARK_FRAME", frame, elapsed, flush=True)
    (WORK / "animation-benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


def render_demo():
    scene, report = prepare(save=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(FRAMES / "frame_")
    started = time.monotonic()
    bpy.ops.render.render(animation=True)
    report["render_seconds"] = time.monotonic()-started
    expected = [FRAMES/f"frame_{frame:04d}.png" for frame in range(1, scene.frame_end+1)]
    if not all(path.is_file() for path in expected):
        raise RuntimeError("Frame sequence is incomplete.")
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        raise RuntimeError("ffmpeg was not found.")
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(FPS), "-start_number", "1",
        "-i", str(FRAMES/"frame_%04d.png"),
        "-frames:v", str(scene.frame_end),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", str(PREVIEW),
    ], check=True)
    report["video"] = str(PREVIEW)
    report["video_bytes"] = PREVIEW.stat().st_size
    report["rendered_frames"] = len(expected)
    (WORK / "animation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("EXPRESSION_ANIMATION_COMPLETE", json.dumps({
        key: report[key] for key in ("video", "video_bytes", "rendered_frames",
                                    "render_seconds", "blend")}), flush=True)


if __name__ == "__main__":
    import sys
    if "--test-expression-video" in sys.argv:
        test_frame()
    elif "--benchmark-expression-video" in sys.argv:
        benchmark()
    elif "--render-expression-video" in sys.argv:
        render_demo()
