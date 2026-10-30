"""Retain the local jaw/dental/skin changes; restore rejected fringe and try one iris optical correction."""
import hashlib, json, runpy
from pathlib import Path
import bpy

folder=Path(__file__).resolve().parent
eye=runpy.run_path(str(folder/'iris_single_reflection_r09.py'))
p=eye['project_path']()
source_relative='work/reconstruction-r02/reference-lower-jaw-curvature-r09.blend'
output_relative='work/reconstruction-r02/reference-production-refined-r09.blend'
source=p/source_relative; target=p/output_relative; report=target.with_suffix('.json')
if target.exists() or report.exists():raise RuntimeError('Preserve previous candidate/report')
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source.resolve():raise RuntimeError('Exact fresh jaw candidate required')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
source_sha=sha(source)
iris=eye['run'](source_relative=source_relative)
fringe=runpy.run_path(str(folder/'forelock_ellipse_candidate_r09.py'))['restore']()
if not fringe:raise RuntimeError('Expected the specifically rejected elliptical fringe layer')
if 'R09_forelock_ellipse_candidate' in bpy.data.objects['R03 Forelocks from crown']:
    raise RuntimeError('Rejected fringe marker remained active')
if abs(bpy.data.objects['R01 Head'].data.shape_keys.key_blocks['R09_lower_jaw_curvature'].value-1)>1e-6:
    raise RuntimeError('Static local jaw layer unexpectedly disabled')
if sha(source)!=source_sha:raise RuntimeError('Independent source changed on disk')
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result={'source':source_relative,'sourceSha256':source_sha,'output':output_relative,'outputSha256':sha(target),
        'iris':iris,'rejectedFringeRestore':fringe,'savedBlend':True,'appearanceAccepted':False,
        'scope':'Independent combination candidate; local jaw, dental, skin and sclera retained. Rejected ellipse groom restored to small-flow plus posterior PN source. Iris single reflection requires new actual images.'}
report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_APPEARANCE_REFINED_SAVED',result['outputSha256'],flush=True)
