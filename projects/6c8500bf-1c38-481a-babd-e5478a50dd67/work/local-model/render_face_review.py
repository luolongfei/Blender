"""Render the saved face refinement without changing the master Blend."""
import bpy
import hashlib
import json
import runpy
from pathlib import Path
from mathutils import Vector

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "previews"
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.threads_mode = "FIXED"
scene.render.threads = 6
scene.render.resolution_x = 1100
scene.render.resolution_y = 1300
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
camera_data = bpy.data.cameras.new("Face review orthographic")
camera = bpy.data.objects.new("Face review orthographic", camera_data)
scene.collection.objects.link(camera)
camera_data.type = "ORTHO"
camera_data.ortho_scale = .370
scene.camera = camera
target = Vector((0, -.006, 1.425))
views = {
    "side": Vector((3, -.006, 1.425)),
    "three-quarter": Vector((2.5, -2.506, 1.425)),
    "front": Vector((0, -3.006, 1.425)),
    "left": Vector((-3, -.006, 1.425)),
}
visibility = {o.name: o.hide_render for o in scene.objects}
for name, position in views.items():
    camera.location = position
    camera.rotation_euler = (target-position).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(OUT / f"face-{name}-refined.png")
    bpy.ops.render.render(write_still=True)
    print("FACE_RENDER", name, flush=True)

clay = bpy.data.materials.new("Face review clay")
clay.use_nodes = True
shader = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
shader.inputs["Base Color"].default_value = (.48, .43, .38, 1)
shader.inputs["Roughness"].default_value = .72
shader.inputs["Specular IOR Level"].default_value = .12
for obj in scene.objects:
    if obj.type == "MESH" and not obj.name.startswith(("Face ·", "Ear", "Neck")):
        obj.hide_render = True
scene.view_layers[0].material_override = clay
module = runpy.run_path(str(PROJECT / "work/local-model/face_refine.py"))
for label in ("after", "before"):
    if label == "before":
        module["restore"]()
    for name in ("side", "three-quarter"):
        position = views[name]
        camera.location = position
        camera.rotation_euler = (target-position).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(OUT / f"face-{name}-clay-{label}.png")
        bpy.ops.render.render(write_still=True)
        print("FACE_CLAY", label, name, flush=True)

face = bpy.data.objects["Face · sculpted planes & original illustration"]
report = {
    "sourceBlendSha256": hashlib.sha256((PROJECT/"work/character.blend").read_bytes()).hexdigest(),
    "geometryVersion": module["VERSION"],
    "defaultBlush": False,
    "textureVersion": face.get("neutral_face_texture_version"),
    "projectionImage": "textures/face-neutral-projection.png",
    "portableTexture": "textures/face-neutral-basecolor.png",
    "renderedViews": list(views),
    "clayComparisonViews": ["side", "three-quarter"],
    "masterSavedByRender": False,
}
(PROJECT/"work/local-model/face-render-report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print("FACE_REVIEW_RENDER_COMPLETE", flush=True)
