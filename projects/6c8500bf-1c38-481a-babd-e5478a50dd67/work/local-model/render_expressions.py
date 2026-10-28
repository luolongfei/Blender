"""Review actual edited geometry. No image synthesis or master mutation."""
from pathlib import Path
import sys
import bpy
import runpy
from mathutils import Vector

P=Path(__file__).resolve().parents[2]
module=runpy.run_path(str(Path(__file__).with_name("face_expressions.py")))
OUT=P/"work/expression-authoring/review"
OUT.mkdir(parents=True,exist_ok=True)

def setup(size=900,samples=24):
    scene=bpy.context.scene
    cam=bpy.data.objects.get("Expression review camera")
    if cam is None:
        data=bpy.data.cameras.new("Expression review camera")
        cam=bpy.data.objects.new("Expression review camera",data)
        scene.collection.objects.link(cam)
    cam.data.type="ORTHO"
    cam.data.ortho_scale=.350
    scene.camera=cam
    scene.render.engine="CYCLES"
    scene.cycles.device="CPU"
    scene.cycles.samples=samples
    scene.cycles.use_denoising=True
    for node in scene.world.node_tree.nodes:
        if node.type=="BACKGROUND":
            node.inputs["Strength"].default_value=.48
    bpy.data.objects["Key · large softbox"].data.energy=235
    bpy.data.objects["Fill · cool softbox"].data.energy=80
    scene.render.resolution_x=size
    scene.render.resolution_y=int(size*1.16)
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    return scene,cam

def render(label,values=None,angle=0,size=900,samples=24):
    import math
    scene,cam=setup(size,samples)
    module["pose"](values)
    target=Vector((0,-.006,1.429))
    rad=math.radians(angle)
    cam.location=target+Vector((3*math.sin(rad),-3*math.cos(rad),0))
    cam.rotation_euler=(target-cam.location).to_track_quat("-Z","Y").to_euler()
    scene.render.filepath=str(OUT/(label+".png"))
    bpy.ops.render.render(write_still=True)
    print("EXPRESSION_RENDER",label,flush=True)
    return scene.render.filepath

def suite():
    presets=module["presets"]()
    for name in ("neutral","happy","blink","sad","angry","surprised","relaxed","aa","ih","ou","ee","oh","wink"):
        render(name,presets[name],size=750,samples=20)
    for name,values in (("neutral",{}),("blink",{"blink":1}),("half-blink",{"blink":.5}),("half-aa",{"aa":.5}),("happy",presets["happy"])):
        for angle in (45,-45,90):
            render(name+("-left45" if angle==-45 else "-45" if angle==45 else "-side"),values,angle,size=750,samples=20)
    module["pose"]()

if __name__=="__main__":
    suite()
