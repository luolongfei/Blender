"""Compact actual radial-eye A/B, preserving selected roll and lower-face v2.

Wraps the camera-metadata-corrected review renderer. Only the exact five public
expression controls must be neutral; active static shape Keys are intentional.
No source save, geometry edit or independent Blender process is started here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

BASE='upper_lid_roll_review_r11.py'
BASE_SHA='a70b7dff532af7b5414c2d7c120d45033386e8fcb749cf66ddf881b5d7a15ff6'
BINDING='face_fibre_binding_r11_v2.py'
BINDING_SHA='47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49'
OUTER='R11 refined outer eye corners'
FIXED={'R11 blink L':0.,'R11 blink R':0.,'R11 gentle smile':0.,
       'R11 natural soft smile':0.,'R11 warm closed smile':0.,
       'R11 refined upper lid roll':1.,'R11 lower face contour v2':1.,
       'R11 lower face continuity':0.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import bpy
    folder=Path(__file__).resolve().parent
    if sha(folder/BASE)!=BASE_SHA or sha(folder/BINDING)!=BINDING_SHA:
        raise RuntimeError('Frozen metadata-correct renderer or combined binding validator changed')
    module=runpy.run_path(str(folder/BASE));project=module['P']
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    plan_path=(project/args.plan).resolve();plan_path.relative_to(project.resolve())
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    frames=plan['frames'];view=plan['view']
    expected=[0.,1.] if view=='eyes' else [1.] if view=='quarter' else None
    if expected is None or len(frames)!=len(expected) or plan.get('hideGroom') is not False:
        raise RuntimeError('Use the two-eye A/B or optional one-quarter plan with the saved groom visible')
    for row,strength in zip(frames,expected):
        wanted={**FIXED,OUTER:strength}
        if row['keys']!=wanted:
            raise RuntimeError('Only the new outer-eye strength may differ; explicit static and public controls must match the reviewed plan')
    if 'R11_refined_radial_outer_eye_candidate' not in bpy.context.scene:
        raise RuntimeError('Render the actually saved radial-eye candidate')
    keys=bpy.data.objects['R11 Female head study'].data.shape_keys.key_blocks
    for name,value in {**FIXED,OUTER:1.}.items():
        if name not in keys or abs(float(keys[name].value)-value)>1e-7:
            raise RuntimeError('Saved source differs from the selected static/public baseline: '+name)
    render=module['main'];runtime=render.__globals__;original_api=runtime['api']
    def current_api(name):
        return original_api('face_fibre_binding_r11_v2' if name=='face_fibre_binding_r11' else name)
    runtime['api']=current_api
    try:
        render()
    finally:
        runtime['api']=original_api
    report_path=(project/args.output/'render-report.json').resolve()
    report_path.relative_to((project/'previews/reconstruction-r02').resolve())
    report=json.loads(report_path.read_text(encoding='utf8'))
    report['reviewWrapper']={'file':Path(__file__).name,'sha256':sha(__file__),
        'renderer':BASE,'rendererSha256':BASE_SHA,'binding':BINDING,'bindingSha256':BINDING_SHA,
        'fixedControls':FIXED,'comparedControl':OUTER,
        'scope':'Matched source, pose, evaluated camera, lighting and selected static shape; only the radial outer-eye key changes.'}
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('R11_RADIAL_EYE_REVIEW_DONE '+json.dumps({'view':view,'frames':len(frames),'output':args.output}),flush=True)


if __name__=='__main__':
    main()
