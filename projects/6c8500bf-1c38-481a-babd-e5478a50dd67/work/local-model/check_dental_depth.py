from pathlib import Path
import bpy,runpy
P=Path(__file__).resolve().parents[2]
lower=bpy.data.objects["Lower dental arch"]
for block in lower.data.shape_keys.key_blocks:
    for point in block.data:point.co.y+=.006
for point,basis in zip(lower.data.vertices,lower.data.shape_keys.key_blocks[0].data):
    point.co=basis.co
render=runpy.run_path(str(Path(__file__).with_name("render_expressions.py")))
render["render"].__globals__["OUT"]=P/"work/qa-anatomy-material"
bpy.context.scene.render.threads_mode="FIXED";bpy.context.scene.render.threads=4
render["render"]("aa-depth-corrected",{"aa":1},20,size=640,samples=16)
