"""Actual same-camera isolation of fibre highlights versus pixel coverage.

Three optional temporary cases: unchanged source; zero Principled specular;
1.6x physical radius. Nothing is saved to Blend and binding contracts are NOT
relaxed. Radius images are diagnostics, not adopted or verified new geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

NAMES = ['R11 Adult natural brow ' + s for s in ('L', 'R')] + [
    'R11 Adult lashes ' + p + ' ' + s for p in ('upper', 'lower') for s in ('L', 'R')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _materials(obj):
    def detail(mat):
        if mat is None: return None
        nodes = []
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if 'BSDF' not in node.type and node.type != 'OUTPUT_MATERIAL': continue
                inputs = {}
                for name in ('Color', 'Base Color', 'Roughness', 'Radial Roughness', 'IOR',
                    'Specular IOR Level', 'Transmission Weight', 'Coat Weight', 'Alpha'):
                    socket = node.inputs.get(name)
                    if socket is None: continue
                    value = socket.default_value
                    inputs[name] = {'value': list(value) if hasattr(value, '__len__') else value,
                        'linked': socket.is_linked}
                nodes.append(dict(name=node.name, type=node.type, inputs=inputs))
        return dict(name=mat.name, nodes=nodes)
    return [detail(m) for m in obj.data.materials]


def _measure(obj, camera, scene):
    import bpy
    import numpy as np
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()); data = ev.data
    radii = np.empty(len(data.attributes['radius'].data), np.float32)
    data.attributes['radius'].data.foreach_get('value', radii)
    frame = camera.data.view_frame(scene=scene)
    width = max(v.x for v in frame) - min(v.x for v in frame)
    scale = max(abs(v) for v in ev.matrix_world.to_scale())
    pixels = 2 * radii * scale * scene.render.resolution_x / width
    counts = np.array([c.points_length for c in data.curves], np.int32)
    roots = np.r_[0, np.cumsum(counts)[:-1]]
    return dict(curves=len(counts), points=len(radii), materials=_materials(ev),
        radiusMicrometerQuantiles=np.quantile(radii * scale * 1e6, [0, .1, .5, .9, 1]).tolist(),
        diameterPixelQuantiles=np.quantile(pixels, [0, .1, .5, .9, 1]).tolist(),
        rootDiameterPixelQuantiles=np.quantile(pixels[roots], [0, .5, 1]).tolist(),
        fractionControlPointsBelowQuarterPixel=float(np.mean(pixels < .25)),
        cameraFrameWidthMeters=float(width), pixelWidthMeaning='Orthographic projected tube diameter; actual visibility also depends on background, shadows, strand overlap and lighting.')


def run(source_relative, output_relative, cases=('source', 'no-specular', 'radius-16'), samples=64):
    import bpy
    import numpy as np
    from mathutils import Vector
    source = (project_path() / source_relative).resolve(); out = (project_path() / output_relative).resolve()
    source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    out.relative_to((project_path() / 'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source or out.exists():
        raise RuntimeError('Fresh exact saved source and new diagnostic output directory required')
    if samples not in (64, 128) or not cases or set(cases) - {'source', 'no-specular', 'radius-16'}:
        raise ValueError('Use explicit bounded diagnostic cases and 64/128 samples')
    out.mkdir(parents=True); source_sha = sha(source)
    binding = runpy.run_path(str(Path(__file__).with_name('face_fibre_binding_r11.py')))
    validation_before = binding['validate'](neutral=False)
    objects = [bpy.data.objects[name] for name in NAMES]
    original = {obj.name: obj.data for obj in objects}
    originals_contract = {obj.name: binding['_curve_contract'](obj.data) for obj in objects}
    scene = bpy.context.scene; scene.frame_set(1); bpy.context.view_layer.update()
    ev = bpy.data.objects['R11 Adult sclera L'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    co = np.asarray([ev.matrix_world @ v.co for v in ev.data.vertices])
    target = (co.min(0) + co.max(0)) * .5; target[2] += .012
    camera = bpy.data.objects.new('TEMP R11 fibre visibility camera', bpy.data.cameras.new('TEMP R11 fibre visibility camera'))
    scene.collection.objects.link(camera); camera.location = Vector(target + np.array([0., -2., 0.]))
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'; camera.data.ortho_scale = .055; camera.data.dof.use_dof = False; scene.camera = camera
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'; scene.cycles.samples = samples
    scene.cycles.seed = 29; scene.cycles.use_denoising = True
    scene.render.resolution_x = 640; scene.render.resolution_y = 640; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    detail = runpy.run_path(str(Path(__file__).with_name('final_detail_review_r09.py')))
    report = dict(source=source_relative, sourceSha256=source_sha, helperSha256=sha(__file__), cases=[],
        cameraScale=.055, resolution=[640, 640], samples=samples, denoising=True, seed=29,
        lights=detail['_lights'](scene), world=detail['_world'](scene),
        bindingBefore=validation_before, sourceSaved=False, diagnosticOnly=True, appearanceAccepted=False,
        scope='Temporary isolated material/radius tests only. No head/eye/light edits, no new production binding contract.')
    # Predict diameter in the exact full-face orthographic framing, without rendering it.
    camera.data.ortho_scale = .365; scene.render.resolution_x = 960; scene.render.resolution_y = 1080
    report['fullFaceCoverage'] = {obj.name: _measure(obj, camera, scene) for obj in objects}
    camera.data.ortho_scale = .055; scene.render.resolution_x = scene.render.resolution_y = 640
    try:
        for case in cases:
            copied_data = []; copied_materials = {}; edits = []
            try:
                if case != 'source':
                    for obj in objects:
                        data = original[obj.name].copy(); copied_data.append(data); obj.data = data
                        if case == 'radius-16':
                            attr = data.attributes['radius']; radii = np.empty(len(attr.data), np.float32)
                            attr.data.foreach_get('value', radii); attr.data.foreach_set('value', radii * 1.6)
                            edits.append(dict(object=obj.name, radiusMultiplier=1.6))
                        else:
                            for index, old in enumerate(list(data.materials)):
                                if old.name not in copied_materials:
                                    new = old.copy(); copied_materials[old.name] = new
                                    changed = []
                                    for node in new.node_tree.nodes:
                                        if node.type != 'BSDF_PRINCIPLED': continue
                                        for socket_name in ('Specular IOR Level', 'Coat Weight'):
                                            socket = node.inputs.get(socket_name)
                                            if socket is not None and not socket.is_linked:
                                                socket.default_value = 0.; changed.append(socket_name)
                                    edits.append(dict(material=old.name, disabledInputs=changed,
                                        unsupportedHairShaderUnchanged=not bool(changed)))
                                data.materials[index] = copied_materials[old.name]
                bpy.context.view_layer.update()
                row = dict(case=case, edits=edits, fibres={obj.name: _measure(obj, camera, scene) for obj in objects})
                png = out / (case + '.png'); scene.render.filepath = str(png)
                bpy.ops.render.render(write_still=True)
                if not png.is_file(): raise RuntimeError('Diagnostic render not written')
                row['image'] = str(png.relative_to(project_path())); row['imageSha256'] = sha(png); report['cases'].append(row)
                print('R11_FIBRE_VISIBILITY_CASE ' + case, flush=True)
            finally:
                for obj in objects: obj.data = original[obj.name]
                bpy.context.view_layer.update()
                for data in copied_data:
                    if data.users == 0: bpy.data.hair_curves.remove(data)
                for material in copied_materials.values():
                    if material.users == 0: bpy.data.materials.remove(material)
    finally:
        for obj in objects: obj.data = original[obj.name]
        bpy.context.view_layer.update()
    for obj in objects:
        if binding['_curve_contract'](obj.data) != originals_contract[obj.name]:
            raise RuntimeError('Diagnostic failed to restore original fibres')
    report['bindingAfterRestore'] = binding['validate'](neutral=False)
    if sha(source) != source_sha: raise RuntimeError('Source file changed')
    with (out / 'report.json').open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--cases', default='source,no-specular,radius-16')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    run(args.source, args.output, tuple(args.cases.split(',')), args.samples)
