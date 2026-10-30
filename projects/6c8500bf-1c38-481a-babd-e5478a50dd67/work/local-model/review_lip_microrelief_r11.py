"""Serial same-source mouth macro: untouched / nominal 5um / nominal 10um.

Does not save a Blend. Creates one camera from actual evaluated vermilion and
keeps it, the source lights/world, exposure, pose and render seed identical.
The independent integration helper is applied/restored in memory per case.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def review(source_relative, output_relative, expected_sha, samples=128):
    import bpy
    import numpy as np
    from mathutils import Vector
    p = project_path().resolve()
    source = (p / source_relative).resolve()
    output = (p / output_relative).resolve()
    source.relative_to(p / 'work/reconstruction-r02')
    output.relative_to(p / 'previews/reconstruction-r02')
    if source.suffix != '.blend' or Path(bpy.data.filepath).resolve() != source or not bpy.app.background or bpy.data.is_dirty:
        raise RuntimeError('Fresh exact independent source required')
    if _sha(source) != expected_sha or not 32 <= samples <= 256:
        raise RuntimeError('Source hash or bounded samples mismatch')
    names = ('original', '5um', '10um')
    if any((output / (name + '.png')).exists() for name in names) or (output / 'comparison.json').exists():
        raise RuntimeError('Preserve previous review outputs')
    module = runpy.run_path(str(Path(__file__).with_name('lip_microrelief_candidate_r11.py')))
    if module['MARKER'] in bpy.context.scene:
        raise RuntimeError('Comparison source must have the unmodified original skin assignment')
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.update()
    head = bpy.data.objects[module['HEAD']]
    keys = head.data.shape_keys
    if keys and any(abs(k.value) > 1e-7 for k in keys.key_blocks[1:]):
        raise RuntimeError('Neutral face required for initial original/5um/10um comparison')
    ev = head.evaluated_get(bpy.context.evaluated_depsgraph_get())
    support = ev.data.attributes.get(module['MASK'])
    if not support or support.domain != 'POINT' or support.data_type != 'FLOAT':
        raise RuntimeError('Evaluated true vermilion support missing')
    mask = np.empty(len(support.data), np.float32)
    support.data.foreach_get('value', mask)
    ids = np.flatnonzero(mask > .1)
    if len(ids) < 40:
        raise RuntimeError('Insufficient evaluated mouth for trustworthy framing')
    points = np.asarray([ev.matrix_world @ ev.data.vertices[int(i)].co for i in ids], dtype=float)
    lo, hi = points.min(0), points.max(0)
    target = (lo + hi) * .5
    width = float(hi[0] - lo[0])
    if not .020 < width < .080:
        raise RuntimeError('Actual evaluated mouth bounds unexpected')
    camera = bpy.data.objects.new('R11 lip microrelief fixed comparison', bpy.data.cameras.new('R11 lip comparison camera'))
    scene.collection.objects.link(camera)
    camera.location = Vector(target + np.array([0., -2., 0.]))
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max(.085, width * 1.7)
    camera.data.dof.use_dof = False
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.seed = 41
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 680
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    output.mkdir(parents=True, exist_ok=True)
    detail = runpy.run_path(str(Path(__file__).with_name('final_detail_review_r09.py')))

    def setting_contract():
        return {
            'frame': scene.frame_current, 'subframe': scene.frame_subframe,
            'cameraMatrix': [list(row) for row in camera.matrix_world],
            'orthographicScale': camera.data.ortho_scale, 'depthOfField': camera.data.dof.use_dof,
            'resolution': [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
            'samples': scene.cycles.samples, 'seed': scene.cycles.seed, 'animatedSeed': scene.cycles.use_animated_seed,
            'denoising': scene.cycles.use_denoising,
            'lights': detail['_lights'](scene), 'world': detail['_world'](scene),
            'colourManagement': {'view': scene.view_settings.view_transform, 'look': scene.view_settings.look,
                                 'exposure': scene.view_settings.exposure, 'gamma': scene.view_settings.gamma}
        }

    bpy.context.view_layer.update()
    fixed = setting_contract()
    result = {
        'recordedAtUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': source.relative_to(p).as_posix(), 'sourceSha256': expected_sha,
        'reviewHelperSha256': _sha(__file__),
        'materialHelperSha256': _sha(Path(__file__).with_name('lip_microrelief_candidate_r11.py')),
        'fixedSettings': fixed, 'evaluatedMouthBounds': [lo.tolist(), hi.tolist()], 'cases': [],
        'scope': 'One actual neutral pose and one macro camera only; not expression/contact or final appearance acceptance.',
        'blendSaved': False, 'appearanceAccepted': False
    }
    for name, microns in zip(names, (None, 5., 10.)):
        installed = False
        row = {'case': name, 'micronsNominalNewHeightField': microns, 'isOriginalUnmodifiedMaterial': microns is None}
        try:
            if microns is not None:
                row['candidate'] = module['apply'](microns)
                installed = True
            bpy.context.view_layer.update()
            if setting_contract() != fixed:
                raise RuntimeError('Camera/pose/light/exposure/settings changed between comparisons')
            path = output / (name + '.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            if not path.is_file() or _sha(source) != expected_sha:
                raise RuntimeError('Render or source-file identity check failed')
            row.update(image=path.relative_to(p).as_posix(), imageSha256=_sha(path))
        finally:
            if installed:
                row['restore'] = module['restore']()
        result['cases'].append(row)
    result['sameSettingsAllCases'] = setting_contract() == fixed
    result['sourceFileUnchanged'] = _sha(source) == expected_sha
    if not result['sameSettingsAllCases'] or not result['sourceFileUnchanged']:
        raise RuntimeError('Final comparison contract failed')
    with (output / 'comparison.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('R11_LIP_MICRORELIEF_REVIEW_DONE ' + json.dumps({'output': output.relative_to(p).as_posix(), 'sourceSha256': expected_sha}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source, args.output, args.expected_source_sha256, args.samples)
