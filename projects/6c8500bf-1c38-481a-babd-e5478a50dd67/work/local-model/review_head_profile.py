"""Inspect actual head anatomy independent of the pending hair refinement."""
from pathlib import Path
import runpy
import bpy

P=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
runpy.run_path(str(HERE/"head_proportion_refine.py"))["apply"]()
render=runpy.run_path(str(HERE/"render_expressions.py"))
render["render"].__globals__["OUT"]=P/"work/realism-authoring"
scene=bpy.context.scene
scene.render.threads_mode="FIXED";scene.render.threads=4
for name,angle in (("balanced-front",0),("balanced-profile",90)):
    render["render"](name,{},angle,size=640,samples=12)
for obj in list(bpy.data.collections["CHARACTER · Original geometry"].all_objects):
    if obj.get("part")=="hair" or obj.name.startswith("Hair_"):
        obj.hide_render=True
clay=bpy.data.materials.new("Head anatomy clay")
clay.use_nodes=True
shader=clay.node_tree.nodes.get("Principled BSDF")
shader.inputs["Base Color"].default_value=(.38,.40,.42,1)
shader.inputs["Roughness"].default_value=.72
scene.view_layers[0].material_override=clay
for name,angle in (("balanced-profile-clay",90),("balanced-oblique-clay",45)):
    render["render"](name,{},angle,size=640,samples=12)
