"""Read-only, no-render replay of the completed neutral-lash preview transaction.

Reuses the exact successful proposed paths from its retained actual arrays.
Separates authored controls, derived pose, the detailed external contract, and
actual evaluated geometry. Never writes a Blend, old helper or pose matrix.
An exact replay without rendering cannot by itself explain a render-only fault.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
import numpy as np

PINS = {
    'adult_upper_lash_neutral_preview_r11.py': 'fa2f8a246e37678eb2b00208112ec4de03bef4d5b9c916a3d82091fb5254f478',
    'adult_upper_lash_regroom_r11_v4.py': '3b573fc79c69057e0123f8831373c850ac2b1700aae5837f9dfd28986b4e7b76',
    'face_fibre_binding_r11_v2.py': '47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49',
    'chin_throat_contour_preview_r11_v2.py': '0a2bdd12fecaa137eb6bd7be28f79237e4c71e9c6d6039e200d595f80fc9e50b',
    'chin_throat_contour_preview_r11.py': '7e0777082dacf1069c4ae195517e3ee2c121376ce0cee99cb7cb6208b5575e90',
    'head_neck_posture_observe_r11_v2.py': '854636345a31a97f9eb9b9edbdafc4a0c3431fba422eab6ae3aee610bd79686d',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def external_rows(bpy, binding):
    """Unhashed rows matching frozen regroom.external_contract, without exemptions."""
    if 'R11_upper_lash_late_surface' in bpy.context.scene:
        raise RuntimeError('This diagnostic requires the exact old-lash source without a late field')
    rows = []
    for obj in bpy.context.scene.objects:
        row = {'name': obj.name, 'object': obj.as_pointer(),
            'data': None if obj.name in binding['OBJECTS'] else getattr(getattr(obj, 'data', None), 'as_pointer', lambda: None)(),
            'parent': obj.parent.as_pointer() if obj.parent else None,
            'world': [list(r) for r in obj.matrix_world],
            'materials': [slot.material.as_pointer() if slot.material else None for slot in obj.material_slots],
            'modifiers': [(m.name, m.type) for m in obj.modifiers if m.name != binding['MODIFIER']]}
        if obj.type == 'ARMATURE':
            row['pose'] = [(bone.name, [list(r) for r in bone.matrix_basis]) for bone in obj.pose.bones]
        rows.append(row)
    return {'objects': rows, 'actions': [(a.name, a.as_pointer()) for a in bpy.data.actions],
        'frame': [bpy.context.scene.frame_current, bpy.context.scene.frame_subframe]}


def observe(source_relative, source_sha, prior_relative, output_relative):
    import bpy
    from mathutils import Matrix
    for filename, expected in PINS.items():
        if sha(Path(__file__).with_name(filename)) != expected:
            raise RuntimeError('Frozen diagnostic dependency changed: '+filename)
    preview = api('adult_upper_lash_neutral_preview_r11')
    for filename, expected in preview['PINS'].items():
        if sha(Path(__file__).with_name(filename)) != expected:
            raise RuntimeError('Frozen preview dependency changed: '+filename)
    project = preview['project_path']().resolve()
    source, prior_path, output = [(project/r).resolve() for r in (source_relative, prior_relative, output_relative)]
    source.relative_to(project/'work/reconstruction-r02')
    output.relative_to(project/'work/reconstruction-r02')
    prior_path.relative_to(project/'previews/reconstruction-r02')
    if source.suffix != '.blend' or output.suffix != '.json' or output.exists():
        raise RuntimeError('Independent source and unused diagnostic JSON required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh explicit background source required')
    if sha(source) != source_sha.lower():
        raise RuntimeError('Explicit source hash mismatch')
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    if prior['sourceSha256'] != source_sha.lower() or not prior.get('localNeutralChecksPassed'):
        raise RuntimeError('Exact source and previously successful neutral checks required')
    array_path = prior_path.parent/prior['actualArrays']['path']
    if sha(array_path) != prior['actualArrays']['sha256']:
        raise RuntimeError('Retained actual arrays changed')
    data = np.load(array_path)
    binding = api('face_fibre_binding_r11_v2')
    regroom = api('adult_upper_lash_regroom_r11_v4')
    qa = api('face_contact_audit_r11')
    three = api('chin_throat_contour_preview_r11_v2')
    uv = api('production_uv_r08')
    diff = api('head_neck_posture_observe_r11_v2')['leaf_diff']
    scene = bpy.context.scene
    head = bpy.data.objects[binding['HEAD']]
    if regroom['MARKER'] in scene or 'R11_upper_lash_late_surface' in scene:
        raise RuntimeError('Retained old upper-lash source required')
    keys = head.data.shape_keys.key_blocks
    if {k.name: float(k.value) for k in keys} != prior['savedKeyValues']:
        raise RuntimeError('Saved sculpt/expression controls differ from completed preview')
    old_marker = str(scene[binding['MARKER']])
    if hashlib.sha256(old_marker.encode()).hexdigest() != prior['oldBindingMarkerSha256']:
        raise RuntimeError('Saved binding marker differs from completed preview')
    record = copy.deepcopy(json.loads(old_marker))
    old_data = {n: bpy.data.objects[n].data for n in regroom['NAMES']}
    old_records = {n: {'data': bpy.data.objects[n].data,
        'fakeUser': bpy.data.objects[n].data.use_fake_user,
        'contract': binding['_curve_contract'](bpy.data.objects[n].data),
        'mapping': binding['_mapping_hash'](binding['_mapping'](bpy.data.objects[n].data))}
        for n in binding['OBJECTS']}
    old_camera = scene.camera
    old_render = {n: getattr(scene.render, n) for n in ('engine', 'resolution_x', 'resolution_y', 'resolution_percentage', 'filepath')}
    old_cycles = {n: getattr(scene.cycles, n) for n in ('device', 'samples', 'use_denoising')}
    old_format = scene.render.image_settings.file_format
    created = []
    camera = camera_data = None
    report = {'source': source_relative, 'sourceSha256': source_sha.lower(), 'helperSha256': sha(__file__),
        'pinnedHelpers': PINS, 'priorAttempt': prior_relative, 'priorAttemptSha256': sha(prior_path),
        'priorArraysSha256': sha(array_path), 'sourceSaved': False, 'rendersPerformed': 0,
        'technicalAcceptance': False, 'appearanceAccepted': False, 'stages': {}, 'errors': [],
        'scope': 'Read-only no-render transaction replay. Reuses exact successful proposed paths, rebuilds their actual current maps, replays binding/native sampling and restores originals. Original render images are preserved. A non-reproduced failure remains unresolved.'}
    def persist(label):
        report['lastCompletedStep'] = label
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print('R11_LASH_RESTORE_DIAGNOSTIC '+label, flush=True)
    def capture(label, geometry=False):
        row = three['state'](head, uv, excluded_key='__no_original_key_excluded__')
        row['externalContractRows'] = external_rows(bpy, binding)
        row['externalRowsMatchFrozenHash'] = binding['_hash'](row['externalContractRows']) == regroom['external_contract'](bpy, binding)
        if not row['externalRowsMatchFrozenHash']:
            raise RuntimeError('Diagnostic rows do not reproduce original external contract')
        if geometry:
            row['actualGeometry'] = three['external_geometry'](set())
            row['stateAfterActualGeometryCapture'] = three['state'](head, uv, excluded_key='__no_original_key_excluded__')
            row['actualCaptureStateDiff'] = diff({k: row[k] for k in ('author', 'derived')}, row['stateAfterActualGeometryCapture'])
        report['stages'][label] = row
        persist(label)
        return row
    def actual_points(name):
        obj = bpy.data.objects[name]
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return binding['_transform'](binding['_array'](ev.data.position_data, 'vector'), ev.matrix_world)
    initial = capture('before_explicit_update')
    bpy.context.view_layer.update()
    baseline = capture('baseline_after_update', geometry=True)
    report['initialToBaselineAuthorDiff'] = diff(initial['author'], baseline['author'])
    report['initialToBaselineDerivedDiff'] = diff(initial['derived'], baseline['derived'])
    report['initialToBaselineExternalDiff'] = diff(initial['externalContractRows'], baseline['externalContractRows'])
    protected = binding['_protected'](head)
    failure = None
    try:
        with qa['rest_evaluation']():
            capture('rest_entered')
            report['bindingBefore'] = binding['validate'](neutral=False)
            capture('rest_binding_validated')
            hd = binding['_head_mesh'](head, triangulate=True)
            if hd['topologySha256'] != prior['actualCurrentCapture']['headTopologySha256'] or binding['_ah'](hd['world']) != prior['actualCurrentCapture']['headWorldSha256']:
                raise RuntimeError('Current actual REST head differs from original preview')
            skin = regroom['bvh'](hd)
            for side in ('L', 'R'):
                name = 'R11 Adult lashes upper '+side
                if not np.array_equal(actual_points(name), data[side+'_beforeWorld']):
                    raise RuntimeError('Current old upper-lash output differs from actual preview: '+side)
            camera_data = bpy.data.cameras.new('TEMP R11 no-render lash restoration diagnosis')
            camera = bpy.data.objects.new(camera_data.name, camera_data)
            scene.collection.objects.link(camera)
            camera.matrix_world = Matrix(prior['camera']['matrixWorld'])
            camera_data.type = 'ORTHO'
            camera_data.ortho_scale = prior['camera']['orthographicScale']
            camera_data.dof.use_dof = False
            scene.camera = camera
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'CPU'
            scene.cycles.samples = prior['samples']
            scene.cycles.use_denoising = True
            scene.render.resolution_x, scene.render.resolution_y = prior['resolution']
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = 'PNG'
            bpy.context.view_layer.update()
            capture('before_render_deliberately_omitted')
            for side in ('L', 'R'):
                name = 'R11 Adult lashes upper '+side
                obj = bpy.data.objects[name]
                prior_item = copy.deepcopy(record['objects'][name])
                reference = bpy.data.hair_curves[prior_item['originalData']]
                world = data[side+'_proposedWorld']
                matrix = np.asarray(obj.matrix_world)
                new_reference = reference.copy()
                created.append(new_reference)
                new_reference.name = 'TEMP R11 lash restore reference '+side
                new_reference.use_fake_user = True
                new_reference.position_data.foreach_set('vector', np.asarray(binding['_transform'](world, np.linalg.inv(matrix)), np.float32).ravel())
                new_reference.materials.clear()
                for material in obj.data.materials:
                    new_reference.materials.append(material)
                new_bound = new_reference.copy()
                created.append(new_bound)
                new_bound.name = 'TEMP R11 lash restore bound '+side
                obj.data = new_reference
                mapping, plan = binding['_plan'](obj, hd, skin)
                for i in range(3):
                    for label, values in (('vertex', mapping['vertices']), ('corner', mapping['corners'])):
                        new_bound.attributes.new(binding['P']+label+'_'+str(i), 'INT', 'CURVE').data.foreach_set('value', np.ascontiguousarray(values[:, i]))
                new_bound.attributes.new(binding['P']+'barycentric', 'FLOAT_VECTOR', 'CURVE').data.foreach_set('vector', mapping['weights'].ravel())
                new_bound.attributes.new(binding['P']+'frame_offset', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', mapping['offsets'].ravel())
                obj.data = new_bound
                record['objects'][name] = {**plan, 'originalData': new_reference.name, 'candidateData': new_bound.name,
                    'originalFakeUser': True, 'originalContractSha256': binding['_hash'](binding['_curve_contract'](new_reference))}
            scene[binding['MARKER']] = json.dumps(record, ensure_ascii=False)
            bpy.context.view_layer.update()
            report['bindingAfter'] = binding['validate'](neutral=False)
            capture('temporary_upper_lashes_validated')
            report['replayedNative33Exact'] = {}
            for side in ('L', 'R'):
                actual, radii, counts = regroom['sample_actual'](bpy.data.objects['R11 Adult lashes upper '+side], binding)
                exact = all(np.array_equal(a, data[side+n]) for a, n in ((actual, '_actualNative33World'), (radii, '_actualNative33Radii'), (counts, '_actualNative33Counts')))
                report['replayedNative33Exact'][side] = exact
                if not exact:
                    raise RuntimeError('Replayed actual native candidate differs from completed preview: '+side)
            capture('after_render_deliberately_omitted')
        capture('pose_restored_before_lash_cleanup')
    except BaseException as exc:
        failure = exc
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    finally:
        for name, datablock in old_data.items():
            bpy.data.objects[name].data = datablock
        scene[binding['MARKER']] = old_marker
        for datablock in reversed(created):
            datablock.use_fake_user = False
            if datablock.users == 0:
                bpy.data.hair_curves.remove(datablock)
        scene.camera = old_camera
        if camera is not None:
            bpy.data.objects.remove(camera, do_unlink=True)
        if camera_data is not None and camera_data.users == 0:
            bpy.data.cameras.remove(camera_data)
        for name, value in old_render.items():
            setattr(scene.render, name, value)
        for name, value in old_cycles.items():
            setattr(scene.cycles, name, value)
        scene.render.image_settings.file_format = old_format
        capture('cleanup_before_final_update')
        bpy.context.view_layer.update()
        final = capture('cleanup_after_final_update', geometry=True)
        report['authorDiff'] = diff(baseline['author'], final['author'])
        report['derivedDiff'] = diff(baseline['derived'], final['derived'])
        report['externalContractDiff'] = diff(baseline['externalContractRows'], final['externalContractRows'])
        report['actualGeometryDiff'] = diff(baseline['actualGeometry'], final['actualGeometry'])
        report['initialToFinalDerivedDiff'] = diff(initial['derived'], final['derived'])
        report['initialToFinalExternalDiff'] = diff(initial['externalContractRows'], final['externalContractRows'])
        report['oldFibreDataExact'] = {}
        for name, row in old_records.items():
            current = bpy.data.objects[name].data
            report['oldFibreDataExact'][name] = (current == row['data'] and current.use_fake_user == row['fakeUser'] and
                binding['_curve_contract'](current) == row['contract'] and binding['_mapping_hash'](binding['_mapping'](current)) == row['mapping'])
        report['protectedHeadExact'] = binding['_protected'](head) == protected
        report['oldMarkerExact'] = str(scene[binding['MARKER']]) == old_marker
        report['sourceFileExact'] = sha(source) == source_sha.lower()
        report['nonRenderFailureReproduced'] = bool(report['initialToFinalDerivedDiff'] or report['initialToFinalExternalDiff'])
        report['classification'] = 'unresolved'
        author_and_geometry_exact = (not report['authorDiff'] and not report['actualGeometryDiff'] and
            all(report['oldFibreDataExact'].values()) and report['protectedHeadExact'] and report['oldMarkerExact'] and report['sourceFileExact'])
        if author_and_geometry_exact:
            report['classification'] = ('non-render replay has exact authored inputs and actual geometry, but contract differences retained' if report['nonRenderFailureReproduced'] else
                'non-render replay exact; original render-associated contract failure not reproduced and remains unresolved')
        report['diagnosticCompleted'] = failure is None
        report['originalPreviewAcceptedAsSuccessfulTransaction'] = False
        persist('final_differences_persisted')
    if failure:
        raise failure
    if not author_and_geometry_exact:
        raise RuntimeError('No-render replay changed authored inputs, actual geometry or protected source; exact diagnostics retained')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--prior', default='previews/reconstruction-r02/upper-lash-neutral-preview-r11/attempt-report.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.source_sha, args.prior, args.output)
