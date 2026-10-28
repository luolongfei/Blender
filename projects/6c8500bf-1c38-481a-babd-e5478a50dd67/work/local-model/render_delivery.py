"""Render the saved review model without modifying the master Blend."""
from pathlib import Path
import runpy
import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
cb = runpy.run_path(str(HERE/"character_builder.py"))
scene = bpy.context.scene
scene.render.threads_mode = "FIXED"
scene.render.threads = 6
for view, filename, resolution, samples in [
    ("three-quarter", "hero-refined.png", 2000, 48),
    ("portrait", "portrait-refined.png", 1600, 32),
    ("front", "front-refined.png", 1400, 24),
    ("back", "back-refined.png", 1400, 24),
    ("side", "side-refined.png", 1400, 24),
    ("detail shoes", "shoes-refined.png", 1200, 24),
]:
    print("DELIVERY_RENDER", cb["render"](view, filename, resolution, samples), flush=True)

camera = bpy.data.objects["Camera · portrait"]
rig = bpy.data.objects["Character_Rig"]
hand = rig.matrix_world @ rig.pose.bones["leftHand"].head
camera.location = hand + Vector((1.2, -.7, .35))
camera.rotation_euler = (hand+Vector((0, 0, -.04))-camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.ortho_scale = .29
print("DELIVERY_RENDER", cb["render"]("portrait", "hand-refined.png", 1200, 32), flush=True)
print("RENDERS_COMPLETE", flush=True)
