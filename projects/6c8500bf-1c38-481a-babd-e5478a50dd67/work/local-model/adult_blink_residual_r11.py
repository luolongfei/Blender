"""Export actual fully closed lid/eye geometry for local corner correction.

Read-only, no binding recheck, no rendering, no source save. This is deliberately
a geometry observation, not a new acceptance gate. Existing controls are restored.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def main(source_relative, output_relative):
    import bpy
    helper = runpy.run_path(str(Path(__file__).with_name('adult_blink_recompute_r11.py')))
    p = project_path(); source = (p/source_relative).resolve(); output = (p/output_relative).resolve()
    for path in (source, output): path.relative_to((p/'work/reconstruction-r02').resolve())
    if output.suffix != '.json' or output.exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Fresh observation output required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Saved independent background source required')
    if 'R11_blink_closure_refinement' not in bpy.context.scene:
        raise RuntimeError('Observe the actual installed first closure refinement')
    source_sha = helper['_sha'](source)
    head = bpy.data.objects[helper['HEAD']]; keys = head.data.shape_keys.key_blocks
    names = ['R11 blink L', 'R11 blink R']; previous = {name: float(keys[name].value) for name in names}
    pose_api = helper['_api']('production_uv_r08'); pose = pose_api['_pose_contract']()
    arrays = {}; report = {'eyes': {}}
    try:
        for name in names: keys[name].value = 1.
        bpy.context.view_layer.update()
        with helper['_api']('face_contact_audit_r11')['rest_evaluation']():
            snap = helper['_snapshot'](head)
            for field in ('world', 'originalVertexIds', 'triangles', 'originalTriangleIds'):
                arrays['head_'+field] = snap[field]
            report['headMatrixWorld'] = snap['matrixWorld']
            arrays['currentMixedCageLocal'] = helper['_api']('adult_eye_proportion_r11')['_mixed'](head.data)
            for side in ('L', 'R'):
                eye = helper['_snapshot'](bpy.data.objects['R11 Adult sclera '+side])
                report['eyes'][side] = helper['_measure'](snap, eye)
                for field in ('world', 'originalVertexIds', 'triangles', 'originalTriangleIds'):
                    arrays['eye_'+side+'_'+field] = eye[field]
    finally:
        for name, value in previous.items(): keys[name].value = value
        bpy.context.view_layer.update()
    if pose_api['_pose_contract']() != pose or helper['_sha'](source) != source_sha:
        raise RuntimeError('Observation altered pose or source file')
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    report.update(source=source_relative, sourceSha256=source_sha,
                  helperSha256=helper['_sha'](__file__), arraysSha256=helper['_sha'](output.with_suffix('.npz')),
                  blinkValues={'L': 1., 'R': 1.}, sourceSaved=False,
                  scope='Actual residual visible globe and full-closed evaluated eye geometry; no physical or appearance acceptance claim.')
    with output.open('x', encoding='utf8') as stream: json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R11_BLINK_RESIDUAL_OBSERVED '+json.dumps({'output': output_relative, 'arraysSha256': report['arraysSha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    main(args.source, args.output)
