"""Actual front/quarter single-eye optical review; source/light unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def review(source_relative, output_relative, view='front', samples=128):
    import bpy
    from mathutils import Vector
    source = (project_path() / source_relative).resolve(); out = (project_path() / output_relative).resolve()
    source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    out.relative_to((project_path() / 'previews/reconstruction-r02').resolve())
    if view not in ('front', 'quarter') or samples not in (64, 128, 256): raise ValueError('Explicit bounded actual review settings required')
    png = out / ('eye-' + view + '.png'); report_path = png.with_suffix('.json')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source or png.exists() or report_path.exists():
        raise RuntimeError('Fresh exact saved source and unused review filenames required')
    out.mkdir(parents=True, exist_ok=True); source_sha = sha(source); scene = bpy.context.scene
    scene.frame_set(1); bpy.context.view_layer.update()
    pivot = bpy.data.objects['R11 Adult eye pivot L']; rotation = pivot.matrix_world.to_quaternion()
    target = pivot.matrix_world.translation + rotation @ Vector((0, 0, .002))
    direction = rotation @ Vector((0, -1, 0) if view == 'front' else (.6, -1, .05)).normalized()
    camera = bpy.data.objects.new('TEMP actual R11 eye optics ' + view, bpy.data.cameras.new('TEMP eye optics'))
    scene.collection.objects.link(camera); camera.location = target + direction * 2
    camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'; camera.data.ortho_scale = .045; camera.data.dof.use_dof = False; scene.camera = camera
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = samples
    scene.cycles.seed = 29; scene.cycles.use_denoising = True
    scene.render.resolution_x = 640; scene.render.resolution_y = 480; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'; scene.render.filepath = str(png)
    bpy.context.view_layer.update()
    detail = runpy.run_path(str(Path(__file__).with_name('final_detail_review_r09.py')))
    report = dict(source=source_relative, sourceSha256=source_sha, helperSha256=sha(__file__), view=view,
        actualCameraMatrix=[list(row) for row in camera.matrix_world], orthographicScale=.045,
        resolution=[640, 480], samples=samples, seed=29, denoising=True, depthOfField=False,
        lights=detail['_lights'](scene), world=detail['_world'](scene),
        colourManagement=dict(view=scene.view_settings.view_transform, look=scene.view_settings.look,
            exposure=scene.view_settings.exposure, gamma=scene.view_settings.gamma),
        eyeDepthCandidatePresent='R11_eye_depth_candidate' in scene,
        sourceSaved=False, appearanceAccepted=False,
        scope='Actual saved single-eye rendering; no source geometry, material, lights or visibility edits. Not a completed blink/gaze acceptance.')
    bpy.ops.render.render(write_still=True)
    if not png.is_file() or sha(source) != source_sha: raise RuntimeError('Actual render/source identity check failed')
    report['imageSha256'] = sha(png)
    with report_path.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_EYE_OPTICS_REVIEW ' + json.dumps(dict(image=str(png), sourceSha256=source_sha)), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--view', choices=('front', 'quarter'), default='front'); p.add_argument('--samples', type=int, default=128)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]); review(a.source, a.output, a.view, a.samples)
