"""Small background previews of the new proportions, before live integration."""
import json
import runpy
from pathlib import Path
import bpy

P=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=P/"work/qa-anatomy-material"
OUT.mkdir(parents=True,exist_ok=True)
result=runpy.run_path(str(HERE/"anatomy_material_refine.py"))["apply"](bake=True)
(OUT/"patch-report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"anatomy-material.blend"))
render=runpy.run_path(str(HERE/"render_expressions.py"))
render["render"].__globals__["OUT"]=OUT
scene=bpy.context.scene
scene.render.threads_mode="FIXED";scene.render.threads=5
for name,angle in (("front",0),("profile",90),("oblique",35)):
    render["render"](name,{},angle,size=640,samples=16)
render["render"]("blink",{"blink":1},35,size=640,samples=12)
render["render"]("happy",{"happy_eye":1,"happy_mouth":1,"brow_happy":.65},20,size=640,samples=12)
render["render"]("aa",{"aa":1},20,size=640,samples=12)
objects=list(bpy.data.collections["CHARACTER · Original geometry"].all_objects)
for obj in objects:
    if obj.get("part")=="hair" or obj.name.startswith("Hair_"):
        obj.hide_render=True
for name,angle in (("hairless-profile",90),("hairless-oblique",35)):
    render["render"](name,{},angle,size=640,samples=12)
