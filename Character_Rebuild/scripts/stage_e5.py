import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,neutral_review
checks={
 'face_e4_refine_delta_v3.py':'bb6246849ac5ee20d792bd2c4488757e6f447264507e0132b79d7ddbef6a5e22',
 'hair_continuity_delta_v5b.py':'03c7c59ff39d210b04b0477a62f478a660b8c3212e9683042b21d706906b3683',
 'shoulder_refine_e4_v3.py':'063b1b932e426acfe32babd00db04c0fbc2b0d0725e16d209afe2cf3b651a04f',
 'shoulder_refine_e4.py':'10412344e9f02b13f9d7b6bede1dea2ac1321086ac4a58375417784c15fd5f9a',
 'ear_concha.py':'1fb6b2f6490c3b24b8c64d5a490bd9e13e265159e68acad45ba969b1bfe32a32',
}
for name,digest in checks.items():
    assert hashlib.sha256((OUT/'scripts'/name).read_bytes()).hexdigest()==digest,name+' is not frozen version'
source=OUT/'stages/E4-final-refined.blend'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='7a406fbbcf337e4098ebc98697766cdb3d88d8966286779c92a5e58e3e6928aa'
bpy.ops.wm.open_mainfile(filepath=str(source))
from face_e4_refine_delta_v3 import apply_e4_face_delta
from hair_continuity_delta_v5b import apply_hair_continuity_delta
from shoulder_refine_e4_v3 import apply_shoulder_refinement_v3
from ear_concha import apply as apply_ears
report={'frozenModules':checks,'sourceStage':source.name}
report['face']=apply_e4_face_delta()
report['fringe']=apply_hair_continuity_delta()
report['shoulder']=apply_shoulder_refinement_v3(bpy.data.objects['Rebuild_Body_Clothing_SourceRetained'])
report['ears']=apply_ears()
neutral_review();save('E5-integrated-refinements.blend')
(OUT/'reports/stage-e5.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
print('E5_GEOMETRY_INTEGRATED_REVIEW_AND_EXPORT_PENDING')
