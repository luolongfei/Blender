"""Save one explicit, reversible R09 appearance layer in a new work copy."""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import bpy

MODULES={'dental':'dental_contact_candidate_r09','eyes':'eye_tone_candidate_r09',
         'skin':'skin_optics_candidate_r09','forelock-ellipse':'forelock_ellipse_candidate_r09'}
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('phase',choices=MODULES);parser.add_argument('--source',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
api=runpy.run_path(str(Path(__file__).with_name(MODULES[args.phase]+'.py')))
p=api['project_path']();source=(p/args.source).resolve();target=(p/args.output).resolve()
source.relative_to((p/'work/reconstruction-r02').resolve());target.relative_to((p/'work/reconstruction-r02').resolve())
report=target.with_suffix('.json')
if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Exact independent input required')
if target==source or target.suffix!='.blend' or target.exists() or report.exists():raise RuntimeError('Use new independent output paths')
result=api['run'](source_relative=args.source)
bpy.ops.wm.save_as_mainfile(filepath=str(target))
result.update(savedBlend=True,output=args.output,outputSha256=hashlib.sha256(target.read_bytes()).hexdigest())
report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('R09_APPEARANCE_STAGE_SAVED '+json.dumps({'phase':args.phase,'output':args.output,'sha256':result['outputSha256']}),flush=True)
