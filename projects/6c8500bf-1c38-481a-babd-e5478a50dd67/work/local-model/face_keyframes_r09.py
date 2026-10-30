"""Five explicit real portrait checks of a saved current R09 candidate.

No new shape keys, animation, grooming, materials, cache bake or blend save.
Each fresh background invocation selects ONE case: aa42, ee63, ou98, blink-smile,
headpitch. --source is mandatory so a historical source is never picked by age.
AA/EE/OU use the same authored body/KEY actions and camera as the R08 sparse images;
the other two isolate full blink+smile and a calibrated 15 degree downward head
pitch with jaw .2. Old walk/performance hair and cloth playback are disabled:
these are static expression/attachment reviews, never dynamic cache validation.
"""
import argparse
import json
import math
import re
import runpy
import sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CASES = ('aa42', 'ee63', 'ou98', 'blink-smile', 'headpitch')
CONTROLS = ('R02_blink_L', 'R02_blink_R', 'R02_jaw_open', 'R02_mouth_wide',
            'R02_pucker', 'R02_lip_press', 'R02_smile', 'R02_frown',
            'R02_brow_raise', 'R02_brow_furrow', 'R03_squint_L', 'R03_squint_R')
ORAL = ('R02 Upper teeth', 'R02 Lower teeth', 'R02 Tongue', 'R03 Upper gingiva', 'R03 Lower gingiva')
# performance_review_r03.py: center=42,width=12 pulse is exactly 1 at frame 42;
# other pulses and all blink windows are zero there, with baseline smile .45.
AA42_EXPECTED = dict.fromkeys(CONTROLS, 0.)
AA42_EXPECTED.update({'R02_jaw_open':.65, 'R02_mouth_wide':.15, 'R02_smile':.45})


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _oral_observation(head):
    """Report actual evaluated geometry and active drivers; not collision proof."""
    graph = bpy.context.evaluated_depsgraph_get()
    opening = float(head.data.shape_keys.key_blocks['R02_jaw_open'].value)
    rows = []
    for name in ORAL:
        obj = bpy.data.objects[name]
        if obj.type != 'MESH' or obj.hide_render:
            raise RuntimeError('Visible R09 mapped oral mesh expected: '+name)
        keys = obj.data.shape_keys
        driven = {}
        if keys:
            for key in keys.key_blocks:
                if not key.name.startswith('R09_jaw_map_'):
                    continue
                knot = float(key.name.rsplit('_', 1)[1])
                expected = max(0., 1.-abs(opening-knot)*8.)
                value = float(key.value)
                if abs(value-expected) > 2e-5:
                    raise RuntimeError('R09 oral driver does not evaluate: '+name+' / '+key.name)
                if value > 1e-7:
                    driven[key.name] = value
        evaluated = obj.evaluated_get(graph); mesh = evaluated.to_mesh()
        try:
            coords = np.array([evaluated.matrix_world@v.co for v in mesh.vertices], dtype=float)
        finally:
            evaluated.to_mesh_clear()
        if not len(coords) or not np.isfinite(coords).all():
            raise RuntimeError('Invalid evaluated oral geometry: '+name)
        rows.append({'object':name, 'parentBone':obj.parent_bone,
                     'activeJawMapKeys':driven, 'evaluatedVertices':len(coords),
                     'worldBounds':[coords.min(axis=0).tolist(), coords.max(axis=0).tolist()]})
    return rows


def _lights(scene):
    rows = []
    for obj in scene.objects:
        if obj.type != 'LIGHT':
            continue
        data = obj.data
        rows.append({'object':obj.name, 'type':data.type, 'hidden':obj.hide_render,
            'energy':float(data.energy), 'color':list(data.color),
            'worldMatrix':[list(row) for row in obj.matrix_world],
            'size':float(data.size) if hasattr(data, 'size') else None})
    return rows


def run(case, source_relative, label='face-keyframes-r09', samples=24):
    if bpy is None or not bpy.app.background:
        raise RuntimeError('Use a fresh disposable background Blender process')
    if case not in CASES or samples not in (16, 24, 32):
        raise ValueError('One explicit case and 16/24/32 samples required')
    if not re.fullmatch(r'[a-z0-9-]+', label):
        raise ValueError('Safe lowercase output label required')
    p = project_path(); source = (p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if (source != Path(bpy.data.filepath).resolve() or not source.is_file()
            or bpy.data.is_dirty or bpy.context.mode != 'OBJECT'):
        raise RuntimeError('Exact fresh saved source_relative in Object Mode required')
    for handlers in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        if any(getattr(h, '_r02_secondary_hair', False) for h in handlers):
            raise RuntimeError('Old Python hair playback handler present')
    head = bpy.data.objects['R01 Head']; rig = bpy.data.objects['Character_Rig']
    if not head.get('R09_lower_face_candidate') or 'R08 ProductionUV' not in head.data.uv_layers:
        raise RuntimeError('Current R09 proportion candidate with production UV required')
    if 'R06_head_pivot_calibration' not in rig:
        raise RuntimeError('Calibrated head pivot required')
    if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4))) > 1e-6:
        raise RuntimeError('Expected identity rig object transform')
    out = p/'previews/reconstruction-r02'/label
    png = out/(case+'.png'); report_path = out/(case+'-report.json')
    if png.exists() or report_path.exists():
        raise RuntimeError('Existing case image/report retained; select a new label')
    stills = _api('reference_final_stills_r08')
    performance = _api('face_performance_r08')
    poses = _api('head_neck_pose_review_r07')
    source_sha = stills['_sha'](source)
    setup = stills['_neutral'](rig)
    keys = head.data.shape_keys.key_blocks
    actions = []; pose = {'name':'neutral'}
    if case in ('aa42', 'ee63', 'ou98'):
        # These are the old authored actions, evaluated on the current R09
        # geometry. Do not rebuild their keys or zero new static R09 sculpts.
        actions = [performance['_bind'](rig, performance['BODY_ACTION'], 'OBJECT'),
                   performance['_bind'](head.data.shape_keys, performance['FACE_ACTION'], 'KEY')]
        frame = 42 if case == 'aa42' else 63 if case == 'ee63' else 98
        performance['_frame'](frame)
        expected = {'R02_jaw_open':.2, 'R02_smile':.45,
                    'R02_mouth_wide':.75 if case == 'ee63' else 0.,
                    'R02_pucker':.85 if case == 'ou98' else 0.}
        if case == 'aa42':
            expected = AA42_EXPECTED.copy()
        performance['_check_actions']()
        pose = {'name':'Original R03 authored performance frame '+str(frame),
                'headPoseMatrix':[list(row) for row in rig.pose.bones['head'].matrix]}
    else:
        frame = 1
        expected = {'R02_blink_L':1., 'R02_blink_R':1., 'R02_smile':1.} if case == 'blink-smile' else {
            'R02_jaw_open':.2, 'R02_smile':.45}
        for name, value in expected.items():
            keys[name].value = value
        head.data.update_tag(); bpy.context.view_layer.update()
        if case == 'headpitch':
            before = rig.matrix_world@rig.pose.bones['head'].head
            pose = poses['_pose'](rig, 'look_down')
            after = rig.matrix_world@rig.pose.bones['head'].head
            drift = float((after-before).length)
            if drift > 2e-6:
                raise RuntimeError('Head pitch moved calibrated pivot')
            pose['pivotDriftMeters'] = drift
    head.data.update_tag(); rig.update_tag(); bpy.context.view_layer.update()
    for name, value in expected.items():
        if abs(keys[name].value-value) > 2e-5:
            raise RuntimeError('Requested expression did not evaluate: '+name)
    for row in setup['temporarilyDisabledHairModifiers']:
        modifier = bpy.data.objects[row['object']].modifiers[row['modifier']]
        if modifier.show_viewport or modifier.show_render:
            raise RuntimeError('Mismatched native hair playback re-enabled')
    oral = _oral_observation(head)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
    scene.cycles.samples = samples; scene.cycles.use_denoising = True
    scene.render.threads_mode = 'FIXED'; scene.render.threads = 6
    scene.render.use_persistent_data = False
    scene.render.resolution_x = 640; scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.use_sequencer = False; scene.render.use_file_extension = True
    scene.render.use_overwrite = False; scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'IMAGE'
    camera = bpy.data.objects.new('R09 expression check camera', bpy.data.cameras.new('R09 expression check camera'))
    scene.collection.objects.link(camera); scene.camera = camera
    camera.data.type = 'ORTHO'; camera.data.ortho_scale = .36
    camera.data.dof.use_dof = False
    camera.location = (3., 0., 1.429) if case == 'headpitch' else (0., -3., 1.429)
    target = (0., -.005, 1.429)
    camera.rotation_euler = (Vector(target)-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(png); bpy.context.view_layer.update()
    report = {'source':str(source.relative_to(p)), 'sourceSha256':source_sha,
        'helperSha256':stills['_sha'](__file__), 'case':case, 'frame':frame,
        'actions':actions, 'pose':pose, 'neutralSetup':setup,
        'expressionValues':{name:float(keys[name].value) for name in CONTROLS},
        'oralEvaluatedObservation':oral,
        'camera':{'position':list(camera.location), 'target':target, 'orthoScale':.36,
                  'view':'side' if case == 'headpitch' else 'front', 'dof':False},
        'resolution':[640,720], 'samples':samples, 'engine':'Cycles CPU',
        'lightsUnchanged':True, 'lights':_lights(scene),
        'viewSettings':{'transform':scene.view_settings.view_transform,
                        'look':scene.view_settings.look, 'exposure':scene.view_settings.exposure},
        'sourceSavedByRender':False, 'accepted':False,
        'scope':'Actual R09 expression/head pose still. Rest native tail shapes follow their existing attachments; old caches disabled. No new geometry or material edits, no animation/cache quality claim.',
        'visualReview':['AA/EE/OU: squared chin/cheek shading, mouth-corner continuity, lip thickness and tooth exposure.',
                        'Blink+smile: complete lid contact, lower-lid/cheek compression and mouth-corner volume.',
                        'Headpitch: jaw/teeth alignment, chin-neck transition, fixed head pivot and attached hair.']}
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    if not png.is_file() or not png.stat().st_size:
        raise RuntimeError('Actual render image is missing')
    if stills['_sha'](source) != source_sha:
        raise RuntimeError('Saved source changed while rendering')
    report['image'] = str(png.relative_to(p)); report['imageSha256'] = stills['_sha'](png)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R09_FACE_CASE_RENDERED '+json.dumps({'case':case, 'image':report['image'], 'sourceSha256':source_sha}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Exact project-relative saved current R09 source')
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--label', default='face-keyframes-r09')
    parser.add_argument('--samples', type=int, choices=(16,24,32), default=24)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    run(args.case, args.source, args.label, args.samples)
