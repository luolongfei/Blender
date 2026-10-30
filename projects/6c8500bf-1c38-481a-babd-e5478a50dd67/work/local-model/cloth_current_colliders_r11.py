"""Current evaluated body/hand collision proxies, integrator-only Blender stage.

probe creates three temporary native Geometry Nodes proxies, verifies them
against a fresh cloth_current_inputs_capture_r11 report, and removes them.
prepare additionally requires explicit collision settings and saves a new work
copy. It neither attaches cloth nor changes old colliders, caches or actions.
These three inputs do NOT include separate legs or the new head/neck bridge.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

CAPTURE_FILE = 'cloth_current_inputs_capture_r11.py'
CAPTURE_SHA = 'c2a3bffb94a2231bac4a96337bc10094d5e4e0367ce48a35703361e3b2e453ad'
PREFIX = 'R11 Current body-hand contact '
COLLECTION = PREFIX + 'surfaces'
MARKER = 'R11_current_body_hand_collision_proxies'
PARITY_TOLERANCE = 2e-6
SETTING_KEYS = ('thickness_outer', 'thickness_inner', 'cloth_friction', 'damping', 'use_culling')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def project_file(relative, suffix=None):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Explicit project-relative reconstruction path required')
    path = (project_path() / relative).resolve()
    path.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if suffix and path.suffix != suffix:
        raise ValueError('Unexpected extension')
    return path


def plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def stable(value):
    """Drop only known session-address fields, preserving semantic identities."""
    if isinstance(value, dict):
        if 'pointer' in value:
            keys = set(value)
            if keys != {'id', 'pointer'} and keys != {'name', 'pointer', 'curvesSha256', 'custom', 'fakeUser'}:
                raise RuntimeError('Unrecognized pointer-bearing dependency; explicit review required')
        return {key: stable(item) for key, item in value.items() if key != 'pointer'}
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    return value


def actual_pose_matches(rig, sources, state):
    if rig.data.pose_position != state['posePosition']:
        raise RuntimeError('Actual rig pose mode differs from capture')
    current = {'rigMatrixWorld': np.asarray(rig.matrix_world).tolist(),
               'bonePoseWorld': {b.name: np.asarray(rig.matrix_world @ b.matrix).tolist() for b in rig.pose.bones},
               'shapeValues': {obj.name: {k.name: float(k.value) for k in obj.data.shape_keys.key_blocks}
                               if obj.data.shape_keys else {} for obj in sources}}
    for key in current:
        if plain(current[key]) != state[key]:
            raise RuntimeError('Actual semantic pose differs from capture: ' + key)


def load_capture(relative, expected_sha):
    helper = HERE / CAPTURE_FILE
    if sha(helper) != CAPTURE_SHA:
        raise RuntimeError('Frozen capture helper changed')
    api = runpy.run_path(str(helper))
    report_path = project_file(relative, '.json')
    if sha(report_path) != expected_sha:
        raise RuntimeError('Explicit capture report SHA mismatch')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'captured-current-surfaces'
            or not report.get('sourceAndScenePreserved')
            or report.get('helperSha256') != CAPTURE_SHA
            or report.get('sourceObjects') != list(api['SOURCES'])):
        raise RuntimeError('Successful current body/hand capture required')
    archive_path = project_file(report['arrays'], '.npz')
    if sha(archive_path) != report['arraysSha256']:
        raise RuntimeError('Captured arrays changed')
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return api, report, arrays


def settings_from_file(relative):
    if relative is None:
        return None
    path = project_file(relative, '.json')
    record = json.loads(path.read_text(encoding='utf-8'))
    values = record.get('collision')
    if not isinstance(values, dict) or set(values) != set(SETTING_KEYS):
        raise ValueError('Explicit collision settings require exactly ' + repr(SETTING_KEYS))
    if not record.get('rationale') or not isinstance(record['rationale'], str):
        raise ValueError('Record the physical/contact rationale; no implicit legacy padding')
    if not isinstance(values['use_culling'], bool):
        raise ValueError('use_culling must be an explicit boolean')
    for name in SETTING_KEYS[:-1]:
        if isinstance(values[name], bool) or not isinstance(values[name], (int, float)) or not math.isfinite(values[name]) or values[name] < 0:
            raise ValueError('Nonnegative finite collision setting required: ' + name)
    return {'path': relative, 'sha256': sha(path), 'rationale': record['rationale'], 'collision': values}


def new_proxy(bpy, source, collection, created):
    from mathutils import Matrix
    mesh = bpy.data.meshes.new(PREFIX + source.name + ' empty input')
    created['meshes'].append(mesh)
    proxy = bpy.data.objects.new(PREFIX + source.name, mesh)
    created['objects'].append(proxy)
    collection.objects.link(proxy)
    proxy.matrix_world = Matrix.Identity(4)
    proxy.hide_render = True
    proxy.hide_viewport = False
    proxy.hide_set(False)
    proxy.display_type = 'WIRE'
    tree = bpy.data.node_groups.new(PREFIX + source.name + ' evaluated reference', 'GeometryNodeTree')
    created['node_groups'].append(tree)
    tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    info = tree.nodes.new('GeometryNodeObjectInfo')
    info.name = 'Current fully evaluated production surface'
    info.transform_space = 'RELATIVE'
    info.inputs['Object'].default_value = source
    info.inputs['As Instance'].default_value = False
    out = tree.nodes.new('NodeGroupOutput')
    tree.links.new(info.outputs['Geometry'], out.inputs['Geometry'])
    modifier = proxy.modifiers.new(PREFIX + 'live surface', 'NODES')
    modifier.node_group = tree
    if (info.transform_space != 'RELATIVE' or info.inputs['Object'].default_value != source
            or info.inputs['As Instance'].default_value or proxy.parent is not None
            or not np.array_equal(np.asarray(proxy.matrix_world), np.eye(4))):
        raise RuntimeError('Native source/transform contract did not read back')
    proxy['r11_current_collision_source'] = source.name
    proxy['r11_contact_scope'] = 'body and hands only; not full-body/neck collision acceptance'
    return proxy


def cleanup(bpy, created):
    # Only objects/data created by this invocation are touched.
    for obj in reversed(created['objects']):
        bpy.data.objects.remove(obj, do_unlink=True)
    for kind in ('node_groups', 'meshes', 'collections'):
        for data in reversed(created[kind]):
            getattr(bpy.data, kind).remove(data)


def run(capture_relative, capture_sha, output_relative, mode='probe', settings_relative=None):
    import bpy
    api, capture, archive = load_capture(capture_relative, capture_sha)
    source = project_file(capture['source'], '.blend')
    output = project_file(output_relative, '.json' if mode == 'probe' else '.blend')
    report_path = output if mode == 'probe' else output.with_suffix('.collision-proxies.json')
    if output.exists() or report_path.exists():
        raise FileExistsError('Choose unused candidate/report paths')
    if (not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source
            or sha(source) != capture['sourceSha256']):
        raise RuntimeError('Open exact captured saved source in fresh background Blender')
    if bpy.data.collections.get(COLLECTION) or MARKER in bpy.context.scene:
        raise RuntimeError('Existing current proxies require an explicit successor; preserve them')
    collision_settings = settings_from_file(settings_relative)
    if mode == 'prepare' and collision_settings is None:
        raise ValueError('prepare needs explicit settings; geometry parity cannot choose contact padding')
    if mode == 'probe' and collision_settings is not None:
        raise ValueError('probe only tests geometry, without collision modifiers')
    scene = bpy.context.scene
    if any(m.show_viewport and m.type in {'CLOTH', 'SOFT_BODY', 'FLUID', 'PARTICLE_SYSTEM'}
           for obj in scene.objects for m in obj.modifiers):
        raise RuntimeError('Refuse active simulation; do not touch its cache')
    if any(getattr(h, '_r02_secondary_hair', False) for h in bpy.app.handlers.frame_change_post):
        raise RuntimeError('Retire the known legacy hair callback separately before sampling')
    garment, uv, audit = (api['api'](name) for name in
                         ('garment_neck_observe_r11', 'production_uv_r08', 'foot_contact_audit_r03'))
    rig = bpy.data.objects[api['RIG']]
    sources = [bpy.data.objects[name] for name in capture['sourceObjects']]
    dependencies = lambda: plain({obj.name: api['object_dependencies'](obj, garment, uv) for obj in sources})
    before = dependencies()
    if stable(before) != stable(capture['sourceDependencies']):
        raise RuntimeError('Captured source dependencies changed; recapture current source')
    original_pose = uv['_pose_contract']()
    if (scene.frame_current != capture['originalPose']['frame']
            or scene.frame_subframe != capture['originalPose']['subframe']):
        raise RuntimeError('Saved frame differs from capture origin')
    if plain({b.name: np.asarray(b.matrix_local).tolist() for b in rig.data.bones}) != capture['rigRestBones']:
        raise RuntimeError('Current rig rest matrices differ from capture')
    action = rig.animation_data.action if rig.animation_data else None
    action_record = {'name': action.name, 'sha256': audit['_action_hash'](action)} if action else None
    if action_record != capture['activeAction']:
        raise RuntimeError('Current active action changed')
    original_frame = (scene.frame_current, scene.frame_subframe)
    original_ids = api['scene_ids'](bpy)
    original_inventory = api['old_cloth_inventory'](bpy)
    created = {key: [] for key in ('objects', 'meshes', 'node_groups', 'collections')}
    checks, error = [], None
    collection = bpy.data.collections.new(COLLECTION)
    created['collections'].append(collection)
    scene.collection.children.link(collection)
    try:
        proxies = [new_proxy(bpy, obj, collection, created) for obj in sources]
        if collision_settings:
            for proxy in proxies:
                proxy.modifiers.new(PREFIX + 'collision', 'COLLISION')
                for name, value in collision_settings['collision'].items():
                    setattr(proxy.collision, name, value)
                    actual = getattr(proxy.collision, name)
                    if isinstance(value, bool):
                        if actual != value:
                            raise RuntimeError('Collision boolean did not read back: ' + name)
                    elif abs(actual - value) > max(1e-9, abs(value) * 1e-6):
                        raise RuntimeError('Collision setting clamped or changed: ' + name)
        for state in capture['states']:
            frame = state['frame']
            scene.frame_set(math.floor(frame), subframe=frame % 1.)
            bpy.context.view_layer.update()
            actual_pose_matches(rig, sources, state)
            graph = bpy.context.evaluated_depsgraph_get()
            if len(state['surfaces']) != len(sources):
                raise RuntimeError('Captured surface list must exactly match three current sources')
            for source_object, proxy, recorded in zip(sources, proxies, state['surfaces']):
                if source_object.name != recorded['sourceObject']:
                    raise RuntimeError('Captured surface ordering changed')
                current_arrays, proxy_arrays = {}, {}
                current = api['surface'](source_object, graph, current_arrays, 'source')
                observed = api['surface'](proxy, graph, proxy_arrays, 'proxy')
                if current['topology'] != recorded['topology'] or observed['topology'] != current['topology']:
                    raise RuntimeError('Current/proxy/captured topology mismatch: ' + source_object.name)
                a = current_arrays['source_world'].astype(float)
                b = proxy_arrays['proxy_world'].astype(float)
                saved = archive[recorded['arrayPrefix'] + '_world'].astype(float)
                source_error = float(np.linalg.norm(a - saved, axis=1).max(initial=0.))
                proxy_error = float(np.linalg.norm(a - b, axis=1).max(initial=0.))
                if max(source_error, proxy_error) > PARITY_TOLERANCE:
                    raise RuntimeError('Source/proxy parity exceeds numerical tolerance: ' + source_object.name)
                if not np.array_equal(current_arrays['source_triangles'], proxy_arrays['proxy_triangles']):
                    raise RuntimeError('Proxy triangulation differs from current production input')
                checks.append({'frame': frame, 'source': source_object.name, 'proxy': proxy.name,
                               'vertices': len(a), 'capturedSourceMaximumErrorMeters': source_error,
                               'liveProxyMaximumErrorMeters': proxy_error})
    except Exception as exc:
        error = {'error': str(exc), 'traceback': traceback.format_exc()}
    finally:
        scene.frame_set(original_frame[0], subframe=original_frame[1])
        bpy.context.view_layer.update()
    preserved = (dependencies() == before
                 and uv['_pose_contract']() == original_pose
                 and api['old_cloth_inventory'](bpy) == original_inventory
                 and sha(source) == capture['sourceSha256'])
    if not preserved and error is None:
        error = {'error': 'Input/source/old-cloth preservation check failed'}
    if error or mode == 'probe':
        cleanup(bpy, created)
        bpy.context.view_layer.update()
        if api['scene_ids'](bpy) != original_ids:
            error = error or {'error': 'Temporary proxy cleanup did not restore original IDs'}
    report = {'schemaVersion': 1, 'mode': mode, 'helperSha256': sha(__file__),
              'source': capture['source'], 'sourceSha256': capture['sourceSha256'],
              'capture': capture_relative, 'captureSha256': capture_sha,
              'scope': 'R01 current body plus refined hands only',
              'sources': capture['sourceObjects'], 'checks': checks,
              'parityToleranceMeters': PARITY_TOLERANCE,
              'collisionSettings': collision_settings, 'inputsAndOldClothPreserved': preserved,
              'simulationRun': False, 'oldCacheChanged': False, 'garmentCollisionCollectionAssigned': False,
              'savedBlend': False, 'appearanceAccepted': False, 'surfaceCollisionAccepted': False,
              'limitations': ['Actual VIEWPORT native surface parity only; no full RENDER dependency-graph parity.',
                             'Separate legs, feet, new head and neck bridge are absent from this three-surface collection.',
                             'No garment is attached, and no contact margin or fabric behavior is physically validated.',
                             'World-space geometry receives no extra rig modifier, parent or offset.',
                             'Future cloth must explicitly use this dedicated collection and add verified missing surfaces.']}
    if error:
        report.update(error)
    elif mode == 'prepare':
        scene[MARKER] = json.dumps({key: report[key] for key in ('sourceSha256', 'capture', 'captureSha256', 'sources', 'collisionSettings')})
        output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=True)
        report.update(savedBlend=True, output=output_relative, outputSha256=sha(output))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if error:
        raise RuntimeError('Current collision proxy preparation failed: ' + str(report_path))
    return {'report': str(report_path.relative_to(project_path())), 'reportSha256': sha(report_path),
            'savedBlend': report['savedBlend'], 'checks': len(checks)}


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('probe', 'prepare'), default='probe')
    parser.add_argument('--capture', required=True)
    parser.add_argument('--capture-sha', required=True)
    parser.add_argument('--settings')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print('R11_CURRENT_COLLIDERS ' + json.dumps(run(args.capture, args.capture_sha, args.output, args.mode, args.settings)), flush=True)
