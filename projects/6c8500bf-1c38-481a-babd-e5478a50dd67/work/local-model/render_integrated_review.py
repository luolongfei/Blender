"""Render a bounded set of actual integrated-source views in a background copy."""
from pathlib import Path
import hashlib
import json
import runpy
import bpy
from mathutils import Vector

P = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = P / "previews" / "refined-current"


def run():
    if not bpy.app.background:
        raise RuntimeError("Render in a disposable background process.")
    source = Path(bpy.data.filepath)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    face = runpy.run_path(str(HERE / "face_expressions.py"))
    face["pose"]({})
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 6
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    data = bpy.data.cameras.new("Integrated review camera")
    camera = bpy.data.objects.new("Integrated review camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    rig = bpy.data.objects["Character_Rig"]
    hand = rig.matrix_world @ rig.pose.bones["leftHand"].head
    # Viewpoint, target, orthographic framing and image size are explicit.
    views = [
        ("hero", (2.4, -4.5, 2.0), (0, 0, .82), 1.85, 1000, 1250, {}),
        ("front", (0, -4.5, .85), (0, 0, .85), 1.84, 900, 1125, {}),
        ("back", (0, 4.5, .85), (0, 0, .85), 1.84, 900, 1125, {}),
        ("full-left", (-4.5, 0, .85), (0, 0, .85), 1.84, 900, 1125, {}),
        ("full-right", (4.5, 0, .85), (0, 0, .85), 1.84, 900, 1125, {}),
        ("full-back45", (3, 4.5, 1.4), (0, 0, .85), 1.84, 900, 1125, {}),
        ("portrait", (.8, -3, 1.47), (0, -.005, 1.422), .43, 900, 1000, {}),
        ("profile", (3, 0, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-front", (0, -3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-left", (-3, 0, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-left45", (-3, -3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-right45", (3, -3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-back", (0, 3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-back-left", (-3, 3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("head-back-right", (3, 3, 1.429), (0, -.005, 1.429), .36, 800, 900, {}),
        ("happy", (0, -3, 1.429), (0, -.005, 1.429), .36, 800, 900,
         face["presets"]()["happy"]),
        ("blink", (.85, -3, 1.429), (0, -.005, 1.429), .36, 800, 900,
         {"blink": 1}),
        ("blink-half", (.85, -3, 1.429), (0, -.005, 1.429), .36, 800, 900,
         {"blink": .5}),
        ("aa", (.8, -3, 1.429), (0, -.005, 1.429), .36, 800, 900, {"aa": 1}),
        ("aa-half", (1.5, -3, 1.429), (0, -.005, 1.429), .36, 800, 900,
         {"aa": .5}),
        ("hand", hand + Vector((1.2, -.7, .35)), hand + Vector((0, 0, -.04)),
         .29, 800, 800, {}),
    ]
    files = []
    for name, position, target, scale, width, height, expression in views:
        face["pose"](expression)
        camera.location = position
        camera.rotation_euler = (Vector(target) - camera.location).to_track_quat(
            "-Z", "Y").to_euler()
        camera.data.ortho_scale = scale
        scene.render.resolution_x, scene.render.resolution_y = width, height
        path = OUT / (name + ".png")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        files.append({
            "file": str(path.relative_to(P)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expression": expression,
        })
        print("INTEGRATED_RENDER", name, flush=True)
    face["pose"]({})
    # A clearly identified construction view proves that the sleeve contains
    # skinned arm geometry. This is only a disposable render override.
    sleeve_names = (
        "Blouse_Puff_Sleeve_L", "Sleeve_Shoulder_Seam_L",
        "Sleeve_Gather_Seam_L",
    )
    hidden = {name: bpy.data.objects[name].hide_render for name in sleeve_names}
    previous_pose_position = rig.data.pose_position
    try:
        for name in sleeve_names:
            bpy.data.objects[name].hide_render = True
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        camera.location = (1.1, -3, 1.65)
        target = Vector((.38, -.01, 1.16))
        camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.ortho_scale = .82
        scene.render.resolution_x, scene.render.resolution_y = 1000, 750
        path = OUT / "sleeve-inside.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        files.append({
            "file": str(path.relative_to(P)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expression": {},
            "technicalInspection": "Left sleeve hidden; original rest pose; body visible.",
        })
        print("INTEGRATED_RENDER sleeve-inside", flush=True)
    finally:
        rig.data.pose_position = previous_pose_position
        for name, value in hidden.items():
            bpy.data.objects[name].hide_render = value
    upper_garments = [
        obj for obj in bpy.data.collections["CHARACTER · Original geometry"].all_objects
        if obj.name.startswith(("Blouse_", "Sleeve_", "Cuff_", "Collar_", "Button_"))
    ]
    hidden = {obj.name: obj.hide_render for obj in upper_garments}
    try:
        for obj in upper_garments:
            obj.hide_render = True
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        camera.location = (1.3, -4, 1.48)
        target = Vector((0, 0, 1.18))
        camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.data.ortho_scale = 1.7
        scene.render.resolution_x, scene.render.resolution_y = 1200, 650
        path = OUT / "torso-inside.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        files.append({
            "file": str(path.relative_to(P)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "expression": {},
            "technicalInspection": "Upper garments hidden; original rest pose; complete torso and arms visible.",
        })
        print("INTEGRATED_RENDER torso-inside", flush=True)
    finally:
        rig.data.pose_position = previous_pose_position
        for name, value in hidden.items():
            bpy.data.objects[name].hide_render = value
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError("Source changed during rendering.")
    (OUT / "render-report.json").write_text(json.dumps({
        "sourceBlend": str(source), "sourceSha256": source_sha,
        "actualBlenderRender": True, "files": files,
    }, ensure_ascii=False, indent=2))
    print("INTEGRATED_RENDERS_COMPLETE", flush=True)


if __name__ == "__main__":
    run()
