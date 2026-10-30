import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,neutral_review
from shoulder_refine_e4_v2 import apply_shoulder_refinement_v2
source=OUT/'stages/E4-final-refined.blend'
before_hash=hashlib.sha256(source.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(source))
report=apply_shoulder_refinement_v2(bpy.data.objects['Rebuild_Body_Clothing_SourceRetained'])
dest=OUT/'work/clothing-shoulder-e4-v2';dest.mkdir(parents=True,exist_ok=True)
candidate=dest/'shoulder-refined.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(candidate))
report['source']=str(source);report['source_sha256']=before_hash
report['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==before_hash
report['candidate']=str(candidate)
scene=neutral_review();scene.render.resolution_x=700;scene.render.resolution_y=700
cam=scene.camera;view=next(v for v in json.loads((OUT/'reports/stage-a.json').read_text())['reviewViews'] if v['name']=='face-front')
cam.location=view['eye'];cam.rotation_euler=(Vector(view['target'])-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=view['ortho']
scene.render.filepath=str(dest/'shoulder-after.png');bpy.ops.render.render(write_still=True)
report['review_image']=str(dest/'shoulder-after.png')
(dest/'modification-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
