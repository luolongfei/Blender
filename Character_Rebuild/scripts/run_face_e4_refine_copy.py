import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,neutral_review
import face_e4_refine_delta

bpy.ops.wm.open_mainfile(filepath=str(OUT/'stages/E4-final-refined.blend'))
report=face_e4_refine_delta.apply_e4_face_delta()
out=OUT/'work/face-e4-local-refine';out.mkdir(parents=True,exist_ok=True)
dest=out/'E4-face-local-candidate.blend'
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(dest))
report['candidatePath']=str(dest)
report['candidateSha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
report['deltaModuleSha256']=hashlib.sha256(Path(face_e4_refine_delta.__file__).read_bytes()).hexdigest()
(out/'report.json').write_text(json.dumps(report,indent=2))
scene=neutral_review();scene.render.resolution_percentage=60
cam=scene.camera
for name,eye in [('face-front',(0,-4,.78)),('face-side',(4,0,.78))]:
    cam.location=eye;cam.rotation_euler=(Vector((0,0,.78))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=.64
    scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
print('FACE_LOCAL_CANDIDATE_DONE',json.dumps(report))
