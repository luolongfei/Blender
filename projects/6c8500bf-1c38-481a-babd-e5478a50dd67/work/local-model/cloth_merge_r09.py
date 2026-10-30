"""Package only R09 native cloth; append it into an explicit final R09 source.

Integrator-only background modes, no action creation, bake or render:
  --mode package --source work/reconstruction-r02/body-cloth-native-r09.blend
  --mode combine --source <final-face-groom-matched-hair.blend> --output <new.blend>
  --mode verify --source <new.blend> --combined-report <new.blend.merge.json>
The package contains one unparented identity-transform mesh, its shape action,
materials and their normal image/node dependencies. It contains no body rig.
Combine requires the EXISTING active exact R09 action and matching native hair.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

C = runpy.run_path(str(Path(__file__).with_name('cloth_body_r09.py')))
H = runpy.run_path(str(Path(__file__).with_name('native_hair_playback_r04.py')))
project_path = C['project_path']
PACKAGE_OBJECT = 'R09 Cloth independent payload'
IDENTITY = np.eye(4)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _new_report(path):
    if path.exists():
        raise RuntimeError('Preserve existing evidence; use a new report path')
    path.parent.mkdir(parents=True, exist_ok=True)


def _identity(matrix):
    return float(np.max(np.abs(np.asarray(matrix, dtype=float) - IDENTITY))) <= 1e-8


def modifier_record(obj):
    result = []
    if [m.type for m in obj.modifiers] != ['SUBSURF', 'SOLIDIFY']:
        raise RuntimeError('Only the captured subdivision and thickness are allowed; no Armature or Cloth')
    for modifier in obj.modifiers:
        row = {}
        for prop in modifier.bl_rna.properties:
            if prop.identifier == 'rna_type' or prop.is_readonly or prop.type == 'COLLECTION':
                continue
            value = getattr(modifier, prop.identifier)
            if prop.type == 'POINTER':
                if value is not None:
                    raise RuntimeError('Unexpected modifier ID dependency: ' + prop.identifier)
                row[prop.identifier] = None
            else:
                row[prop.identifier] = sorted(value) if isinstance(value, set) else C['API']['_value'](value)
        result.append(row)
    return result


def verify_loaded(data_dir, obj=None, require_body=True, readonly_master=False):
    """Read-only evidence, finite evaluated parity; writes no existing report."""
    capture, points, matrix = C['_load'](data_dir)
    native = _read(data_dir / 'native-report.json')
    obj = obj or bpy.data.objects[C['NATIVE']]
    if len(obj.data.vertices) != C['COUNT'] or C['_topology'](obj.data) != native['topologySha256']:
        raise RuntimeError('Cloth point count or topology differs from capture')
    if C['_uv_hash'](obj.data) != native['uvSha256']:
        raise RuntimeError('Cloth UVs/material indices differ from capture')
    modifiers = modifier_record(obj)
    keys = obj.data.shape_keys
    if not keys or len(keys.key_blocks) != C['END'] + 1 or not keys.animation_data or not keys.animation_data.action:
        raise RuntimeError('Expected Basis plus all 301 simulated samples')
    if keys.animation_data.drivers or any(not t.mute for t in keys.animation_data.nla_tracks):
        raise RuntimeError('Cloth shape playback has unexpected drivers or active NLA')
    if obj.animation_data or obj.constraints:
        raise RuntimeError('Cloth object must not have independent animation/constraints')
    if C['_action_hash'](keys.animation_data.action) != native['shapeActionSha256']:
        raise RuntimeError('Cloth shape action differs from verified capture')
    if not _identity(matrix) or not _identity(obj.matrix_world):
        raise RuntimeError('This bounded merge supports the observed identity world transform only')
    if require_body:
        rig = C['_rig'](readonly_master=readonly_master)
        if not rig.animation_data or not rig.animation_data.action or rig.animation_data.action.name != C['ACTION']:
            raise RuntimeError('The matching R09 body action must already be active; no old action is selected automatically')
        if C['_action_hash'](rig.animation_data.action) != capture['actionSha256'] or not _identity(rig.matrix_world):
            raise RuntimeError('Body action or rig object matrix differs from simulation')
    errors = C['_parity'](obj, points)
    reference = C['_path'](native['surfaceReference'])
    if C['_sha'](reference) != native['surfaceReferenceSha256']:
        raise RuntimeError('Downstream surface reference identity changed')
    surface_errors = {}
    with np.load(reference, allow_pickle=False) as stored:
        for frame, actual in C['_surface_samples'](obj).items():
            expected = stored[frame]
            if actual.shape != expected.shape:
                raise RuntimeError('Downstream viewport point count changed')
            surface_errors[frame] = float(np.max(np.abs(actual - expected)))
    if max(surface_errors.values()) > C['TOLERANCE']:
        raise RuntimeError('Downstream viewport surface differs from captured native result')
    C['_set_frame'](1)
    return {'actionSha256': capture['actionSha256'], 'shapeActionSha256': native['shapeActionSha256'],
        'topologySha256': native['topologySha256'], 'uvSha256': native['uvSha256'],
        'payloadSha256': capture['payloadSha256'], 'modifiers': modifiers,
        'pointErrorsMeters': errors, 'viewportSurfaceErrorsMeters': surface_errors,
        'rigMatrix': matrix.tolist(), 'scope': 'Finite native cloth payload/surface parity, not visual or collision acceptance'}


def _check_native_source(source, data_dir):
    native = _read(data_dir / 'native-report.json')
    if source != C['_path'](native['targetBlend']) or C['_sha'](source) != native['targetSha256']:
        raise RuntimeError('The exact newly captured native cloth source is required')


def _hide_other_cloth(keep):
    records = []
    for obj in bpy.context.scene.objects:
        dynamic = any(m.type == 'CLOTH' for m in obj.modifiers)
        native = 'Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name
        if obj != keep and (dynamic or native or obj.name in C['BASE']['ORIGINALS']):
            records.append(obj.name)
            obj.hide_viewport = obj.hide_render = True
            obj.hide_set(True)
        for modifier in obj.modifiers:
            if modifier.type == 'CLOTH':
                modifier.show_viewport = modifier.show_render = False
    keep.hide_viewport = keep.hide_render = False
    keep.hide_set(False)
    return records


def _matched_hair(rig):
    if rig.get(H['ENABLE'], 0) != 1:
        raise RuntimeError('Final source must already have matching R09 native hair enabled')
    guide = H['_active_guide'](rig)
    record = json.loads(guide[H['RECORD']])
    meta = record['cacheMetadata']
    action = rig.animation_data.action
    if meta['action'] != C['ACTION'] or H['_action_fingerprint'](action) != meta['actionSha256']:
        raise RuntimeError('Final source hair cache belongs to another body action')
    frames = np.asarray(guide['r04_cache_frames'], dtype=int)
    if not np.array_equal(frames, np.arange(1, 302)):
        raise RuntimeError('Expected matching 301-frame R09 native hair')
    C['_set_frame'](1)
    checked = H['_recorded_state'](guide, rig, check_geometry=True)
    return {'guide': guide.name, 'cacheSha256': record['cacheSha256'],
        'action': meta['action'], 'hairActionFingerprint': meta['actionSha256'],
        'geometryHashes': record['geometryHashes'], 'frameOneAttachmentCheck': checked['currentStateValidation'],
        'scope': 'Existing source cache identity, groom and frame-one attachment checked; not a new hair bake or full hair parity audit'}


def package(source, target, data_dir, report_path):
    if target.exists() or source == target:
        raise RuntimeError('A new independent library output is required')
    _new_report(report_path)
    _check_native_source(source, data_dir)
    C['_open'](source)
    original = bpy.data.objects[C['NATIVE']]
    check = verify_loaded(data_dir, original)
    if bpy.data.objects.get(PACKAGE_OBJECT):
        raise RuntimeError('Package object already exists')
    mesh = original.data.copy()
    mesh.name = 'R09 independent cloth payload mesh'
    obj = bpy.data.objects.new(PACKAGE_OBJECT, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = Matrix.Identity(4)
    # Coordinates already equal world coordinates because source rig is identity.
    # No original object copy: parent, groups, animation and constraints cannot leak.
    for old in original.modifiers:
        new = obj.modifiers.new(old.name, old.type)
        C['API']['_copy_settings'](old, new, {'object'})
    if mesh.shape_keys == original.data.shape_keys:
        raise RuntimeError('Mesh copy unexpectedly shares its shape-key datablock')
    # A copied mesh may share the same action ID; that action is precisely the
    # intended dependency and has only shape-key value curves, never a rig slot.
    copied = verify_loaded(data_dir, obj)
    if copied['modifiers'] != check['modifiers']:
        raise RuntimeError('Downstream modifier settings changed while packaging')
    obj['r09_cloth_package'] = json.dumps({'bodyAction': C['ACTION'], 'bodyActionSha256': check['actionSha256'],
        'shapeActionSha256': check['shapeActionSha256'], 'payloadSha256': check['payloadSha256'],
        'topologySha256': check['topologySha256'], 'modifierRecord': check['modifiers']})
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.libraries.write(str(target), {obj}, path_remap='RELATIVE', fake_user=True, compress=True)
    # Read library inventory without loading. A dependency must not silently
    # drag another object/rig into the supposedly independent package.
    with bpy.data.libraries.load(str(target), link=False) as (available, requested):
        inventory = {'objects': list(available.objects), 'armatures': list(available.armatures),
                     'actions': list(available.actions), 'materials': list(available.materials)}
    if inventory['objects'] != [PACKAGE_OBJECT] or inventory['armatures']:
        raise RuntimeError('Package includes an unexpected object or rig dependency; do not use it')
    result = {'sourceBlend': C['_relative'](source), 'sourceSha256': C['_sha'](source),
        'packageBlend': C['_relative'](target), 'packageSha256': C['_sha'](target),
        'inventory': inventory, 'verification': copied, 'status': 'independent-cloth-library-created',
        'scope': 'One parentless identity mesh with native shape action/material dependencies; no rig or body action imported'}
    C['_write'](report_path, result)
    print('R09_CLOTH_PACKAGE ' + json.dumps({'path': str(target), 'objects': inventory['objects']}), flush=True)
    return result


def combine(source, target, payload, package_report, data_dir, report_path):
    if target.exists() or source == target or payload == target:
        raise RuntimeError('Use a new independent combined output')
    _new_report(report_path)
    package_record = _read(package_report)
    if payload != C['_path'](package_record['packageBlend']) or C['_sha'](payload) != package_record['packageSha256']:
        raise RuntimeError('Exact checked independent cloth library required')
    source_sha = C['_sha'](source)
    C['_open'](source)
    rig = C['_rig']()
    capture, _, _ = C['_load'](data_dir)
    if not rig.animation_data or not rig.animation_data.action or rig.animation_data.action.name != C['ACTION']:
        raise RuntimeError('Final source must already have R09 active; this merge never imports or selects a body action')
    if C['_action_hash'](rig.animation_data.action) != capture['actionSha256'] or not _identity(rig.matrix_world):
        raise RuntimeError('Final source body action or rig matrix differs from captured cloth')
    hair = _matched_hair(rig)
    if bpy.data.objects.get(C['NATIVE']) or bpy.data.objects.get(PACKAGE_OBJECT):
        raise RuntimeError('Final source already has this cloth payload; preserve it instead of duplicating')
    objects_before = set(bpy.data.objects.keys())
    rigs_before = set(bpy.data.armatures.keys())
    body_actions_before = {action.name: C['_action_hash'](action) for action in [rig.animation_data.action]}
    with bpy.data.libraries.load(str(payload), link=False) as (available, requested):
        if available.objects != [PACKAGE_OBJECT] or available.armatures:
            raise RuntimeError('Package inventory is no longer independent')
        requested.objects = [PACKAGE_OBJECT]
    obj = requested.objects[0]
    added = set(bpy.data.objects.keys()) - objects_before
    if added != {obj.name} or set(bpy.data.armatures.keys()) != rigs_before:
        raise RuntimeError('Appending cloth unexpectedly introduced a rig or another object')
    if obj.parent or obj.constraints or obj.animation_data or not _identity(obj.matrix_world):
        raise RuntimeError('Package object is not parentless identity playback')
    collection = bpy.data.collections.new('R09 Combined Native Cloth')
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(obj)
    obj.name = C['NATIVE']
    obj.parent = rig
    obj.parent_type = 'OBJECT'
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix.Identity(4)
    hidden = _hide_other_cloth(obj)
    check = verify_loaded(data_dir, obj)
    if check['modifiers'] != package_record['verification']['modifiers']:
        raise RuntimeError('Downstream modifier settings differ from the package')
    if any(C['_action_hash'](bpy.data.actions[name]) != digest for name, digest in body_actions_before.items()):
        raise RuntimeError('Existing body action changed during append')
    if C['_sha'](source) != source_sha:
        raise RuntimeError('Input source was externally modified during merge')
    C['_set_frame'](1)
    hair_after = _matched_hair(rig)
    if hair_after['cacheSha256'] != hair['cacheSha256']:
        raise RuntimeError('Existing matched hair cache changed')
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 301
    scene.use_preview_range = True
    scene.frame_preview_start, scene.frame_preview_end = 1, 301
    result = {'sourceBlend': C['_relative'](source), 'sourceSha256': source_sha,
        'targetBlend': C['_relative'](target), 'packageBlend': C['_relative'](payload),
        'packageSha256': package_record['packageSha256'], 'verification': check,
        'matchedHair': hair_after, 'hiddenOldClothObjects': hidden,
        'armatureCountBefore': len(rigs_before), 'armatureCountAfter': len(bpy.data.armatures),
        'bodyActionImportedOrReplaced': False, 'status': 'combined-saved; fresh-reopen-and-visible-review-pending'}
    obj['r09_combined_cloth_provenance'] = json.dumps(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    result['targetSha256'] = C['_sha'](target)
    C['_write'](report_path, result)
    print('R09_CLOTH_COMBINED ' + json.dumps({'target': str(target), 'report': str(report_path)}), flush=True)
    return result


def verify_combined(source, combined_report, data_dir, report_path):
    _new_report(report_path)
    record = _read(combined_report)
    if source != C['_path'](record['targetBlend']) or C['_sha'](source) != record['targetSha256']:
        raise RuntimeError('Exact saved combined source required')
    C['_open'](source)
    obj = bpy.data.objects[C['NATIVE']]
    rig = C['_rig']()
    if obj.parent != rig or obj.parent_type != 'OBJECT' or not _identity(obj.matrix_basis):
        raise RuntimeError('Combined cloth parenting differs from the intended rig')
    if obj.hide_viewport or obj.hide_render or obj.hide_get():
        raise RuntimeError('The new cloth is hidden in the saved combined source')
    for name in record['hiddenOldClothObjects']:
        old = bpy.data.objects.get(name)
        if old and (not old.hide_render or not old.hide_viewport):
            raise RuntimeError('Old/static cloth became visible: ' + name)
    check = verify_loaded(data_dir, obj)
    if check['modifiers'] != record['verification']['modifiers']:
        raise RuntimeError('Downstream modifier settings changed after reopen')
    hair = _matched_hair(rig)
    if hair['cacheSha256'] != record['matchedHair']['cacheSha256']:
        raise RuntimeError('Matched hair identity changed')
    result = {'sourceBlend': C['_relative'](source), 'sourceSha256': C['_sha'](source),
        'verification': check, 'matchedHair': hair, 'savedBlend': False,
        'status': 'finite-combined-reopen-parity-passed; visual-review-pending'}
    C['_write'](report_path, result)
    print('R09_CLOTH_COMBINED_REOPEN ' + json.dumps({'report': str(report_path)}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('package', 'combine', 'verify'), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output')
    parser.add_argument('--data-dir', default='work/reconstruction-r02/cloth-body-r09')
    parser.add_argument('--payload-blend', default='work/reconstruction-r02/body-cloth-payload-r09.blend')
    parser.add_argument('--package-report', default='work/reconstruction-r02/body-cloth-payload-r09.json')
    parser.add_argument('--combined-report')
    parser.add_argument('--report')
    args = parser.parse_args(argv)
    if not bpy.app.background:
        raise RuntimeError('Explicit serial background Blender required')
    source, data_dir, payload, package_report = map(C['_path'],
        (args.source, args.data_dir, args.payload_blend, args.package_report))
    if args.mode == 'package':
        return package(source, payload, data_dir, C['_path'](args.report) if args.report else package_report)
    if args.mode == 'combine':
        if not args.output:
            raise RuntimeError('Supply an explicit new --output combined candidate')
        target = C['_path'](args.output)
        report = C['_path'](args.report) if args.report else Path(str(target) + '.merge.json')
        return combine(source, target, payload, package_report, data_dir, report)
    if not args.combined_report:
        raise RuntimeError('Supply the --combined-report written by combine')
    report = C['_path'](args.report) if args.report else Path(str(source) + '.reopen.json')
    return verify_combined(source, C['_path'](args.combined_report), data_dir, report)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
