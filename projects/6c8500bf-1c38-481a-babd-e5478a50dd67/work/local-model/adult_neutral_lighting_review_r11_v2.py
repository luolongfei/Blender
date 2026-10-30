"""V2 small lighting A/B preserving permanent sculpt keys during neutral review.

Preserves the original material, pose, camera framing, colour management and
camera-ray world background. Only diagnostic lighting changes. The new lights
stay in character/world space when switching front/quarter cameras. Neutral is
the first trial; side/back are separate diagnostic profiles, not beauty fixes.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
DIRECTIONS = {'front': (0., -1., 0.), 'quarter': (.6, -.8, 0.)}
BASELINE_SPEC = 'previews/reconstruction-r02/adult-full-groom-v5-r11/full-groom-front.json'
PROFILES = {
    'neutral': {
        'worldColour': [.5, .5, .5, 1.], 'worldStrength': .06,
        'lights': [
            {'name': 'neutral key', 'offset': [-.48, -.72, .36], 'watts': 18., 'size': .45},
            {'name': 'neutral fill', 'offset': [.60, -.80, .12], 'watts': 3., 'size': .70}],
        'purpose': 'First low-cost neutral form/colour diagnostic; no rim enhancement.'},
    'side': {
        'worldColour': [.5, .5, .5, 1.], 'worldStrength': .02,
        'lights': [
            {'name': 'side structure', 'offset': [-.80, -.15, .28], 'watts': 16., 'size': .30},
            {'name': 'side weak front fill', 'offset': [.25, -.90, .10], 'watts': 1., 'size': .60}],
        'purpose': 'Separate later inspection of nose/lid/lip/jaw relief and grazing highlights; not neutral skin-colour judgement.'},
    'back': {
        'worldColour': [.5, .5, .5, 1.], 'worldStrength': .02,
        'lights': [
            {'name': 'back transmission', 'offset': [.35, .70, .28], 'watts': 18., 'size': .35},
            {'name': 'back weak front fill', 'offset': [0., -.90, .10], 'watts': 1.5, 'size': .65}],
        'purpose': 'Separate later hair silhouette/transmission/ear and edge diagnostic; does not certify the frontal face.'}
}


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def profile_plan(profile, target):
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Finite actual face target required')
    spec = json.loads(json.dumps(PROFILES[profile]))
    for row in spec['lights']:
        offset = np.asarray(row['offset'], float)
        distance = float(np.linalg.norm(offset))
        row['position'] = (target + offset).tolist()
        row['target'] = target.tolist()
        row['distanceMeters'] = distance
        row['angularWidthDegreesApprox'] = math.degrees(2 * math.atan(row['size'] / (2 * distance)))
        row['colour'] = [1., 1., 1.]
        row['frontalPointPowerProxy'] = row['watts'] / distance ** 2 * max(0., -float(offset[1]) / distance)
    spec['scope'] = 'P/d^2*cos is only a rough direct-light comparison, not measured lux or equal-exposure calibration.'
    return spec


def _colour(scene):
    return {k: getattr(scene.view_settings, k) for k in ('view_transform', 'look', 'exposure', 'gamma')}


def _light_records(scene):
    detail = _api('final_detail_review_r09')
    face = _api('runtime_face_refresh_r08')
    result = detail['_lights'](scene)
    by_name = {o.name: o for o in scene.objects if o.type == 'LIGHT'}
    for row in result:
        data = by_name[row['name']].data
        row['shape'] = getattr(data, 'shape', None)
        row['normalize'] = getattr(data, 'normalize', None)
        row['nodeTree'] = face['_tree_contract'](data.node_tree) if data.use_nodes and data.node_tree else None
    return result


def _world_record(world):
    face = _api('runtime_face_refresh_r08')
    return None if world is None else {
        'name': world.name, 'colour': list(world.color), 'useNodes': world.use_nodes,
        'nodeTree': face['_tree_contract'](world.node_tree) if world.use_nodes and world.node_tree else None}


def _nonlight_contract(scene, head):
    eye = _api('eye_margin_refine_r11')
    face = _api('runtime_face_refresh_r08')
    objects = []
    for obj in scene.objects:
        if obj.type in {'LIGHT', 'CAMERA'}:
            continue
        objects.append({
            'name': obj.name, 'type': obj.type, 'data': getattr(getattr(obj, 'data', None), 'as_pointer', lambda: None)(),
            'matrixWorld': [list(r) for r in obj.matrix_world], 'hideRender': obj.hide_render,
            'materials': [s.material.as_pointer() if s.material else None for s in obj.material_slots],
            'modifiers': [(m.name, m.type, m.show_viewport, m.show_render) for m in obj.modifiers],
            'keyValues': [(k.name, k.value) for k in obj.data.shape_keys.key_blocks]
                         if obj.type == 'MESH' and obj.data.shape_keys else None,
            'pose': [(b.name, [list(r) for r in b.matrix_basis]) for b in obj.pose.bones] if obj.type == 'ARMATURE' else None})
    return _hash({'objects': objects, 'head': eye['_state'](head, 0),
                  'headMaterials': [face['_material_contract'](s.material) for s in head.material_slots if s.material],
                  'frame': scene.frame_current, 'subframe': scene.frame_subframe})


def _actual_face_target(bpy):
    graph = bpy.context.evaluated_depsgraph_get()
    centers = []
    for side in ('L', 'R'):
        eye = bpy.data.objects['R11 Adult sclera ' + side].evaluated_get(graph)
        points = np.asarray([eye.matrix_world @ v.co for v in eye.data.vertices], dtype=float)
        if not len(points) or not np.isfinite(points).all():
            raise RuntimeError('Actual adult eye geometry unavailable')
        centers.append((points.min(0) + points.max(0)) * .5)
    if not .040 < float(np.linalg.norm(centers[1] - centers[0])) < .080:
        raise RuntimeError('Actual head scale differs from metre lighting recipe')
    target = np.mean(centers, axis=0)
    target += np.array([0., -.012, -.025])
    return target, centers


@contextlib.contextmanager
def diagnostic_lights(bpy, plan):
    from mathutils import Vector
    scene = bpy.context.scene
    original_world = scene.world
    if not original_world or not original_world.use_nodes:
        raise RuntimeError('Explicit existing node world needed to preserve camera background')
    old_lights = [(obj, obj.hide_render) for obj in scene.objects if obj.type == 'LIGHT']
    old_light_record = _light_records(scene)
    old_world_record = _world_record(original_world)
    copy_world = None
    created = []
    try:
        copy_world = original_world.copy()
        copy_world.name = 'R11 diagnostic world / ' + plan['profile']
        nt = copy_world.node_tree
        outputs = [n for n in nt.nodes if n.type == 'OUTPUT_WORLD' and n.is_active_output]
        if len(outputs) != 1 or len(outputs[0].inputs['Surface'].links) != 1:
            raise RuntimeError('Unambiguous original world surface required')
        output = outputs[0]
        original_surface = output.inputs['Surface'].links[0].from_socket
        path = nt.nodes.new('ShaderNodeLightPath')
        path.name = 'R11 diagnostic camera-ray separation'
        environment = nt.nodes.new('ShaderNodeBackground')
        environment.name = 'R11 diagnostic neutral environment'
        environment.inputs['Color'].default_value = plan['worldColour']
        environment.inputs['Strength'].default_value = plan['worldStrength']
        mix = nt.nodes.new('ShaderNodeMixShader')
        mix.name = 'R11 preserve original camera background'
        nt.links.new(path.outputs['Is Camera Ray'], mix.inputs[0])
        nt.links.new(environment.outputs[0], mix.inputs[1])
        nt.links.new(original_surface, mix.inputs[2])
        nt.links.new(mix.outputs[0], output.inputs['Surface'])
        for obj, _ in old_lights:
            obj.hide_render = True
        scene.world = copy_world
        for row in plan['lights']:
            data = bpy.data.lights.new('R11 diagnostic / ' + row['name'], type='AREA')
            obj = bpy.data.objects.new(data.name, data)
            created.append(obj)
            scene.collection.objects.link(obj)
            data.use_nodes = False
            if hasattr(data, 'normalize'):
                data.normalize = True
            data.energy = row['watts']
            data.color = row['colour']
            data.shape = 'SQUARE'
            data.size = row['size']
            obj.location = row['position']
            obj.rotation_euler = (Vector(row['target']) - obj.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        yield {'beforeLights': old_light_record, 'beforeWorld': old_world_record,
               'afterLights': _light_records(scene), 'afterWorld': _world_record(copy_world),
               'cameraBackgroundPreservedByWorldBranch': True,
               'reflectionTransmissionAndDiffuseEnvironmentChanged': True,
               'originalLightDataUnedited': True}
    finally:
        scene.world = original_world
        for obj, hidden in old_lights:
            obj.hide_render = hidden
        for obj in reversed(created):
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                bpy.data.lights.remove(data)
        if copy_world is not None and copy_world.users == 0:
            bpy.data.worlds.remove(copy_world)
        bpy.context.view_layer.update()
        if _light_records(scene) != old_light_record or _world_record(original_world) != old_world_record:
            raise RuntimeError('Original light/world restoration mismatch')


def _render_stats(bpy, path):
    result = bpy.data.images.get('Render Result')
    linear = None
    if result is not None and len(result.pixels):
        pixels = np.empty(len(result.pixels), np.float32)
        result.pixels.foreach_get(pixels)
        rgba = pixels.reshape(-1, 4)
        rgb = rgba[rgba[:, 3] > .5, :3]
        rgb = rgb[np.isfinite(rgb).all(axis=1)]
        if len(rgb):
            linear = {'rgbPercentiles1_50_95_99': np.percentile(rgb, [1, 50, 95, 99], axis=0).tolist(),
                      'maxRGB': float(rgb.max()), 'fractionAnyRGBAbove1': float(np.mean(np.max(rgb, axis=1) > 1)),
                      'scope': 'Scene-linear output diagnostic; above1 is not proof of display clipping.'}
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        values = np.empty(len(image.pixels), np.float32)
        image.pixels.foreach_get(values)
        rgba = values.reshape(-1, 4)
        rgb = rgba[rgba[:, 3] > .5, :3]
        rgb = rgb[np.isfinite(rgb).all(axis=1)]
        encoded = None if not len(rgb) else {
            'fractionAllRGBAbove098': float(np.mean(np.all(rgb > .98, axis=1))),
            'fractionAllRGBBelow002': float(np.mean(np.all(rgb < .02, axis=1))),
            'rgbPercentiles1_50_95_99': np.percentile(rgb, [1, 50, 95, 99], axis=0).tolist(),
            'scope': 'Whole-image encoded PNG values; includes hair/background/white collar, not a skin segmentation or acceptance score.'}
    finally:
        bpy.data.images.remove(image)
    return {'sceneLinear': linear, 'displayPNG': encoded, 'automaticExposureAdjustment': False}


def _review_neutral_core(source_relative, expected_sha, output_relative, camera_spec,
                         views, profile, height, samples, neutral_record):
    import bpy
    from mathutils import Vector
    p = project_path().resolve()
    source = (p / source_relative).resolve()
    source.relative_to(p / 'work/reconstruction-r02')
    output = (p / output_relative).resolve()
    output.relative_to(p / 'previews/reconstruction-r02')
    spec_path = (p / camera_spec).resolve()
    spec_path.relative_to(p / 'previews/reconstruction-r02')
    if source.suffix != '.blend' or not bpy.app.background or Path(bpy.data.filepath).resolve() != source or _sha(source) != expected_sha:
        raise RuntimeError('Exact independent source required after explicit neutral-session setup')
    if not views or len(set(views)) != len(views) or any(v not in DIRECTIONS for v in views):
        raise ValueError('Distinct valid camera views required')
    if height not in (350, 420, 490, 560, 630) or not 16 <= samples <= 64:
        raise ValueError('Small 6:7 portrait and 16..64 samples required')
    reference = json.loads(spec_path.read_text(encoding='utf-8'))
    camera_data = reference['camera']
    if camera_data['sensorFit'] != 'VERTICAL' or camera_data['depthOfField'] is not False or camera_data['resolution'] != [1200, 1400]:
        raise RuntimeError('Expected shared full-groom portrait camera specification')
    expected_files = [output / (v + '-' + label + '.png') for v in views for label in ('original', profile)]
    if any(f.exists() for f in expected_files) or (output / 'lighting-comparison.json').exists():
        raise RuntimeError('Preserve previous lighting trial outputs')
    scene = bpy.context.scene
    if scene.frame_current != 1 or abs(scene.frame_subframe) > 1e-8:
        raise RuntimeError('Neutral-session frame one required')
    bpy.context.view_layer.update()
    head = bpy.data.objects[HEAD]
    neutral_api = _api('neutral_expression_context_r11')
    neutral_api['assert_neutral'](neutral_api['_values'](head.data.shape_keys),
                                 neutral_record['preservedValues'], head.data.shape_keys.reference_key.name)
    target, eyes = _actual_face_target(bpy)
    plan = profile_plan(profile, target)
    plan['profile'] = profile
    protected = _nonlight_contract(scene, head)
    colour = _colour(scene)
    lights = _light_records(scene)
    world = _world_record(scene.world)
    camera = bpy.data.objects.new('R11 lighting A-B camera', bpy.data.cameras.new('R11 lighting A-B camera'))
    scene.collection.objects.link(camera)
    camera.data.type = 'ORTHO'
    camera.data.sensor_fit = 'VERTICAL'
    camera.data.ortho_scale = camera_data['orthographicScale']
    camera.data.dof.use_dof = False
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.seed = 67
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = True
    scene.render.resolution_x = height * 6 // 7
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'recordedAtUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': source.relative_to(p).as_posix(), 'sourceSha256': expected_sha, 'helperSha256': _sha(__file__),
        'cameraReference': {'path': camera_spec, 'sha256': _sha(spec_path), 'sourceSha256': reference.get('sourceSha256')},
        'framingScope': 'Reuses the recorded shared portrait frame. New full dense-hair bounds are not remeasured here; compare visible framing and use the dedicated groom review if hairstyle extents changed.',
        'camera': camera_data, 'resolution': [scene.render.resolution_x, height],
        'samples': samples, 'seed': 67, 'denoising': True, 'colourManagementUnchanged': colour,
        'actualFaceTarget': target.tolist(), 'actualEyeCenters': [e.tolist() for e in eyes],
        'profile': plan, 'originalLights': lights, 'originalWorld': world, 'cases': [],
        'savedBlend': False, 'appearanceAccepted': False,
        'scope': 'Small actual-source lighting/form diagnostic. Not equal-brightness calibration, skin reflectance measurement, microdetail acceptance or a geometry fix.'}
    framing_target = Vector(camera_data['target'])
    for view in views:
        direction = Vector(DIRECTIONS[view])
        camera.location = framing_target + direction * camera_data['distance']
        camera.rotation_euler = (framing_target - camera.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.context.view_layer.update()
        matrix = [list(r) for r in camera.matrix_world]
        for label in ('original', profile):
            context = contextlib.nullcontext(None) if label == 'original' else diagnostic_lights(bpy, plan)
            with context as lighting:
                if _colour(scene) != colour or [list(r) for r in camera.matrix_world] != matrix or _nonlight_contract(scene, head) != protected:
                    raise RuntimeError('Pose, material, geometry, camera or colour management changed')
                path = output / (view + '-' + label + '.png')
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                if not path.is_file() or _sha(source) != expected_sha:
                    raise RuntimeError('Render or unchanged-source check failed')
                result['cases'].append({'view': view, 'lighting': label, 'cameraMatrix': matrix,
                    'lightChange': lighting, 'image': path.relative_to(p).as_posix(), 'imageSha256': _sha(path),
                    'imageStats': _render_stats(bpy, path)})
    if _light_records(scene) != lights or _world_record(scene.world) != world or _colour(scene) != colour or _nonlight_contract(scene, head) != protected or _sha(source) != expected_sha:
        raise RuntimeError('Final source content or original lighting restoration mismatch')
    result['originalLightingRestored'] = True
    result['protectedHeadAndNonlightSettingsExact'] = True
    return result


def review(source_relative, expected_sha, output_relative, camera_spec=BASELINE_SPEC,
           views=('front',), profile='neutral', height=560, samples=24):
    import bpy
    p = project_path().resolve()
    source = (p / source_relative).resolve()
    source.relative_to(p / 'work/reconstruction-r02')
    output = (p / output_relative).resolve()
    output.relative_to(p / 'previews/reconstruction-r02')
    if source.suffix != '.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source or _sha(source) != expected_sha:
        raise RuntimeError('Fresh exact independent saved source required before neutral-session setup')
    audit_path = output / 'neutral-expression-session.json'
    if audit_path.exists() or (output / 'lighting-comparison.json').exists():
        raise RuntimeError('Preserve prior v2 session/review records')
    neutral_api = _api('neutral_expression_context_r11')
    audit = {'source': source_relative, 'sourceSha256': expected_sha,
             'contextHelperSha256': _sha(Path(__file__).with_name('neutral_expression_context_r11.py')),
             'lightingHelperSha256': _sha(__file__), 'blendSaved': False}
    try:
        with neutral_api['neutral_session'](bpy, audit):
            result = _review_neutral_core(source_relative, expected_sha, output_relative,
                                          camera_spec, views, profile, height, samples, audit)
    except BaseException as exc:
        audit['reviewError'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        audit['sourceFileUnchanged'] = _sha(source) == expected_sha
        output.mkdir(parents=True, exist_ok=True)
        with audit_path.open('x', encoding='utf-8') as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
    if not audit.get('restored') or not audit['sourceFileUnchanged']:
        raise RuntimeError('Neutral-session/source restoration incomplete')
    result['neutralExpressionSession'] = audit
    result['neutralSessionRestoredBeforeReport'] = True
    with (output / 'lighting-comparison.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('R11_LIGHTING_REVIEW_V2_DONE ' + json.dumps({'output': output.relative_to(p).as_posix(), 'sourceSha256': expected_sha}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--camera-spec', default=BASELINE_SPEC)
    parser.add_argument('--views', nargs='+', choices=tuple(DIRECTIONS), default=['front'])
    parser.add_argument('--profile', choices=tuple(PROFILES), default='neutral')
    parser.add_argument('--height', type=int, default=560)
    parser.add_argument('--samples', type=int, default=24)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source, args.expected_source_sha256, args.output, args.camera_spec,
           tuple(args.views), args.profile, args.height, args.samples)
