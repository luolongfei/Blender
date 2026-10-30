"""Frozen balanced two-light review profile, using the verified v2 renderer.

This wrapper does not edit the original or v2 helper. It imports a private
module instance, adds one profile there and delegates the proven temporary
light/world/neutral-expression contexts. No Blend is saved. make_plan() and
diagnostic_lights() are also available to same-source skin comparison drivers.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

BASE_NAME = 'adult_neutral_lighting_review_r11_v2.py'
BASE_SHA256 = '5378ca57e92d6edbdb2554723e36f8bc1cfd0ed1f85f57cb1d210ceeb473521e'
BASELINE_SPEC = 'previews/reconstruction-r02/adult-full-groom-v5-r11/full-groom-front.json'
PROFILE = {
    'worldColour': [.5, .5, .5, 1.],
    'worldStrength': .16,
    'lights': [
        {'name': 'balanced broad key', 'offset': [-.48, -.72, .36], 'watts': 12., 'size': .80},
        {'name': 'balanced broad fill', 'offset': [.60, -.80, .12], 'watts': 4.5, 'size': 1.00}
    ],
    'purpose': 'Balanced neutral form/skin diagnostic after the actual 18W/.45m trial produced excessive contrast and blocked dark hair; not a beauty-light or equal-exposure calibration.'
}
_PRIVATE_BASE = None


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _base():
    global _PRIVATE_BASE
    path = Path(__file__).with_name(BASE_NAME)
    if _sha(path) != BASE_SHA256:
        raise RuntimeError('Frozen v2 lighting dependency changed; re-review before use')
    if _PRIVATE_BASE is None:
        spec = importlib.util.spec_from_file_location('_r11_balanced_private_v2', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Private imported module only. No file or shared module is modified.
        module.PROFILES = copy.deepcopy(module.PROFILES)
        module.PROFILES['balanced'] = copy.deepcopy(PROFILE)
        _PRIVATE_BASE = module
    if _hash(_PRIVATE_BASE.PROFILES['balanced']) != _hash(PROFILE):
        raise RuntimeError('Private balanced profile was modified after import')
    return _PRIVATE_BASE


def profile_plan(target):
    """Pure plan in metres. Caller fixes this once across all material cases."""
    plan = _base().profile_plan('balanced', target)
    plan['profile'] = 'balanced'
    plan['balancedRecipeSha256'] = _hash(PROFILE)
    plan['balancedHelperSha256'] = _sha(__file__)
    plan['frozenV2HelperSha256'] = BASE_SHA256
    plan['thirdLightEnabled'] = False
    return plan


def make_plan(bpy):
    """Measure actual current eyes once; no scene changes and no pose reset."""
    bpy.context.view_layer.update()
    target, eyes = _base()._actual_face_target(bpy)
    plan = profile_plan(target)
    plan['actualEyeCenters'] = [eye.tolist() for eye in eyes]
    return plan


def _validate_plan(plan):
    if plan.get('profile') != 'balanced' or plan.get('balancedRecipeSha256') != _hash(PROFILE):
        raise RuntimeError('Explicit frozen balanced profile required')
    if len(plan.get('lights', [])) != 2:
        raise RuntimeError('Balanced diagnostic uses exactly two lights')
    expected = profile_plan(plan['lights'][0]['target'])
    for key in ('worldColour', 'worldStrength', 'lights'):
        if plan[key] != expected[key]:
            raise RuntimeError('Frozen balanced light recipe changed: ' + key)


@contextlib.contextmanager
def diagnostic_lights(bpy, plan):
    """Temporary balanced light/world only; caller owns material comparison.

Capture plan before the first image and reuse it unchanged for every material
and camera view. Do not call make_plan again after changing pose or camera.
The original source background, exposure, material and geometry are untouched.
"""
    source = Path(bpy.data.filepath).resolve()
    source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if not bpy.app.background or source.suffix != '.blend':
        raise RuntimeError('Explicit independent background source required')
    _validate_plan(plan)
    with _base().diagnostic_lights(bpy, plan) as audit:
        audit['balancedRecipeSha256'] = _hash(PROFILE)
        audit['balancedHelperSha256'] = _sha(__file__)
        audit['frozenV2HelperSha256'] = BASE_SHA256
        audit['thirdLightEnabled'] = False
        yield audit


def review(source_relative, expected_sha, output_relative, camera_spec=BASELINE_SPEC,
           views=('front',), height=560, samples=24):
    """Original/balanced small A/B with v2's exact neutral-session handling."""
    output = (project_path() / output_relative).resolve()
    output.relative_to((project_path() / 'previews/reconstruction-r02').resolve())
    manifest_path = output / 'balanced-review-manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Preserve the previous balanced review manifest')
    result = _base().review(source_relative, expected_sha, output_relative, camera_spec,
                            views, 'balanced', height, samples)
    # Keep the original v2 report intact. Record this wrapper's role separately.
    report_path = output / 'lighting-comparison.json'
    manifest = {
        'wrapper': str(Path(__file__).resolve().relative_to(project_path().resolve())).replace('\\', '/'),
        'wrapperSha256': _sha(__file__),
        'baseHelper': 'work/local-model/' + BASE_NAME, 'baseHelperSha256': BASE_SHA256,
        'recipe': copy.deepcopy(PROFILE), 'recipeSha256': _hash(PROFILE),
        'source': source_relative, 'sourceSha256': expected_sha,
        'v2Report': str(report_path.relative_to(project_path())).replace('\\', '/'),
        'v2ReportSha256': _sha(report_path),
        'sourceAndNeutralSessionRestored': bool(result.get('neutralSessionRestoredBeforeReport')),
        'originalLightingRestored': bool(result.get('originalLightingRestored')),
        'thirdLightEnabled': False, 'savedBlend': False, 'appearanceAccepted': False
    }
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print('R11_BALANCED_LIGHTING_REVIEW_DONE ' + json.dumps({
        'output': output_relative, 'sourceSha256': expected_sha,
        'wrapperSha256': manifest['wrapperSha256']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--camera-spec', default=BASELINE_SPEC)
    parser.add_argument('--views', nargs='+', choices=('front', 'quarter'), default=['front'])
    parser.add_argument('--height', type=int, default=560)
    parser.add_argument('--samples', type=int, default=24)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source, args.expected_source_sha256, args.output, args.camera_spec,
           tuple(args.views), args.height, args.samples)
