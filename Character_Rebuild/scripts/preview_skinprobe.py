import bpy,sys,math
from pathlib import Path
from mathutils import Vector,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,neutral_review
bpy.ops.wm.open_mainfile(filepath=str(OUT/'interactive/work/rigged-rest.blend'))
arm=bpy.data.objects['PreviewHumanoid']
for name,angle in [('LeftUpperArm',65),('RightUpperArm',-65)]:
    q=arm.data.bones[name].matrix_local.to_quaternion()
    arm.pose.bones[name].rotation_quaternion=q.inverted()@Quaternion(Vector((0,1,0)),math.radians(angle))@q
bpy.context.view_layer.update();scene=neutral_review();scene.render.resolution_x=640;scene.render.resolution_y=640;scene.eevee.taa_render_samples=32
scene.camera.location=(0,-5,0);scene.camera.rotation_euler=(Vector((0,0,0))-scene.camera.location).to_track_quat('-Z','Y').to_euler();scene.camera.data.ortho_scale=2.2
path=OUT/'interactive/work/drop-arm-probe.png';scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
print('SKIN_PROBE',path)
