"""Package only the verified deliverable, never rejected or intermediate stages."""
import json,hashlib,zipfile
from pathlib import Path

out=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=json.loads((out/'reports/final-glb-validation.json').read_text())
fresh=json.loads((out/'reports/clean-reimport.json').read_text())
visual=json.loads((out/'reports/final-visual-review.json').read_text())
assert binary['passed'] and fresh['structuralReimportPassed']
assert binary['sha256']==fresh['glbSha256']==visual['glbSha256']==sha(out/'character-rebuilt-textured.glb')
assert fresh['blendSha256']==sha(out/'character-rebuilt.blend')
assert visual['reviewCompleted'] and visual['bothHandsFiveFingersVisible']
paths=[out/n for n in ('character-rebuilt.blend','character-rebuilt-textured.glb','完成说明.txt','review.html')]
for folder in ('checks/final','checks/reimport','checks/before-matched'):
    paths.extend(p for p in (out/folder).iterdir() if p.suffix in ('.png','.json'))
paths.extend(out/'reports'/n for n in ('final-glb-validation.json','clean-reimport.json','final-source-geometry.json','final-topology-audit.json','final-visual-review.json','final-visual-agent-review.txt','matched-review-settings.json','delivery-manifest.json'))
paths.extend(out/'reports/reference-audit-images'/n for n in ('IMG_7223-original.png','IMG_7224-original.png'))
assert all(p.is_file() for p in paths)
destination=out/'Character_Rebuild_Delivery.zip'
if destination.exists():
    prior=out/'stages/delivery-history'/('Character_Rebuild_Delivery-'+sha(destination)[:16]+'.zip')
    prior.parent.mkdir(parents=True,exist_ok=True)
    if not prior.exists():
        import shutil
        shutil.copy2(destination,prior)
    assert sha(prior)==sha(destination)
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(paths):z.write(p,p.relative_to(out).as_posix())
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
    for p in paths:
        assert hashlib.sha256(z.read(p.relative_to(out).as_posix())).hexdigest()==sha(p)
report={'archive':destination.name,'sha256':sha(destination),'bytes':destination.stat().st_size,'entries':len(paths),'contentsByteVerified':True,'offMachineBackup':False}
(out/'reports/package-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
