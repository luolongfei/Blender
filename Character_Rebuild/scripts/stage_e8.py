import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,save,neutral_review
checks={
 'scripts/neck_join_delta.py':'d474118cccfec4b3e163aa370e7a560d53a7f76ee0288d061139dd1dc0c7b3a6',
 'scripts/face_e7_mouth_annulus_delta.py':'c384c564c37fb026dfba276b83eb664384ed9af43182e1ed30b26871b2365e80',
 'reports/e7-mouth-annulus-delta.json':'0227e1d56ff4bce8e8846052b446177c9009e3e86799657bc5c312f6048f36a0',
 'scripts/clothing_contact_e7.py':'ebf015bc9534e49a3efd98b1da470cc0805e36118763e398dc58b1dce2ebd77d',
 'scripts/cap_inner_diagonal_fix_e8.py':'c1adfddd26e2a2320e6a77869dfe71a438051a0011f3a352ff097e7253b57b28',
}
for name,digest in checks.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
source=OUT/'stages/E7-uv-normals-corrected.blend'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='fb285cbe33d0015f185cdf6a83e28e059e5043726b4640e347c2000117923cdf'
bpy.ops.wm.open_mainfile(filepath=str(source))
from neck_join_delta import apply as neck_fix
from face_e7_mouth_annulus_delta import apply_e7_mouth_annulus_delta
from clothing_contact_e7 import repair_clothing_contacts_e7
from cap_inner_diagonal_fix_e8 import apply_cap_inner_diagonal_fix
report={'frozenInputs':checks,'sourceStage':source.name}
report['neckJoin']=neck_fix()
report['mouthAnnulus']=apply_e7_mouth_annulus_delta()
report['clothing']=repair_clothing_contacts_e7()
report['crownInnerShell']=apply_cap_inner_diagonal_fix()
neutral_review();save('E8-delivery-source.blend')
(OUT/'reports/stage-e8.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
print('E8_INTEGRATED_CONTACT_AND_NECK_FIXES')
