"""R11-compatible isolated hand pose review using actual current frame 1.

Reuses the frozen R10 measured hand axes, bounded thumb search, actual surface
checks and camera, without requiring R09 hair to remain visible. No model save.
The adapter changes only the old global-context entry; pose/rest/restore and
render checks stay exact. It never claims hair, cloth or body dynamics passed.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

FROZEN_REVIEW_SHA = '31264b7d2d04a3e43d88d89d59e15dac4336ae34e846b69208b3a3dffbf97889'
POSES = ('relaxed', 'half-grip', 'fist', 'opposition', 'wrist-flex', 'wrist-extend')
VIEWS = ('dorsal', 'palm', 'quarter', 'cuff')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _standing_context(bpy):
    """Observe the current compatible source; don't reset pose or old caches."""
    scene = bpy.context.scene; rig = bpy.data.objects['Character_Rig']
    if scene.frame_current != 1 or scene.frame_subframe != 0 or rig.data.pose_position != 'POSE':
        raise RuntimeError('Use the actual frame-one POSE source; no implicit reset permitted')
    action = rig.animation_data.action if rig.animation_data else None
    if not action: raise RuntimeError('Current standing body action is missing')
    context = {'frame': 1, 'bodyAction': action.name,
               'bodyActionSlot': getattr(getattr(rig.animation_data, 'action_slot', None), 'identifier', None),
               'framesPerSecond': scene.render.fps / scene.render.fps_base,
               'actionsCleared': False, 'staticHairOrClothSubstituted': False,
               'legacyHairCacheRequirementRemoved': True,
               'scope': 'Actual current frame-one body pose only; no old hair/cache parity, cloth or continuous-motion certification.'}
    return rig, context


def _api():
    path = Path(__file__).with_name('hand_asset_pose_review_r10.py')
    if _sha(path) != FROZEN_REVIEW_SHA:
        raise RuntimeError('Frozen hand-review helper changed; inspect before updating the adapter')
    api = runpy.run_path(str(path)); original_loader = api['_api']
    def load(name):
        result = original_loader(name)
        if name == 'final_detail_review_r09':
            result = dict(result); result['_matched_standing'] = _standing_context
        return result
    # runpy functions retain their original module globals. Replace only this
    # dependency loader; never monkeypatch Blender state or render operators.
    api['run'].__globals__['_api'] = load
    return api


def run(source, side, pose, view, label, samples=32):
    api = _api(); result = api['run'](source, side, pose, view, label, samples)
    report_path = (project_path() / result['report']).resolve()
    report_path.relative_to(project_path().resolve())
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['r11Adapter'] = {'helperSha256': _sha(__file__), 'frozenR10ReviewSha256': FROZEN_REVIEW_SHA,
                          'oldHairOrClothParityClaimed': False,
                          'change': 'Observe actual R11 current standing context; retain measured hand pose, surface checks and exact restoration.'}
    report['scope'] = 'Temporary local hand/finger offsets on the unchanged current R11 frame 1; actual image plus finite hand/cuff surface diagnostics.'
    # The file was created by the above single invocation. Existing files are
    # rejected by the frozen renderer before any rendering or modification.
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--side', choices=('L', 'R'), required=True)
    parser.add_argument('--pose', choices=POSES, required=True); parser.add_argument('--view', choices=VIEWS, required=True)
    parser.add_argument('--label', required=True); parser.add_argument('--samples', type=int, choices=(24, 32), default=32)
    args = parser.parse_args(argv)
    print('R11_HAND_POSE_REVIEW ' + json.dumps(run(args.source, args.side, args.pose, args.view, args.label, args.samples), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
