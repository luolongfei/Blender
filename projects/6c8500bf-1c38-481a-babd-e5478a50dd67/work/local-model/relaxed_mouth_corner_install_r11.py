"""Save the reviewed resting mouth-corner field in an independent candidate.

No render, frame change, bone-channel assignment, or production-source write.
The neutral actual support is matched exactly; unrelated source work is allowed.
Warm 0/.5/1 are current-source combinations, not new expression acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
KEY = 'R11 relaxed mouth corners'
WARM = 'R11 warm closed smile'
MARKER = 'R11_relaxed_mouth_corners_candidate'
PLAN = 'work/reconstruction-r02/relaxed-rest-mouth-corners-plan-r11.json'
PLAN_SHA = '929509d98703f4c23689c7271714e5e79df8a369ab8b4a34a0b401c702b6084a'
ARRAYS_SHA = 'eeeba21987e9e39839d1120ca5ec2606c407bd04615bd7d42fe7e4af988d8c36'
PINNED = {
    'relaxed_mouth_corners_plan_r11.py': '229db23142ab5a704fa56be5d5696fc0a985b296b26882d1c42026bb2b07a918',
    'relaxed_mouth_corner_nodes_r11.py': '2de3071e3d17f777e20523e3c4f35bd3c44bc280a8f8ca5fa789f26da195f2fb',
    'relaxed_mouth_corner_preview_r11.py': '8ee286cba24272c586709ecceb56acfb33690a8a714f5609cbec8f6f3e7c5572',
    'chin_throat_contour_preview_r11_v2.py': '0a2bdd12fecaa137eb6bd7be28f79237e4c71e9c6d6039e200d595f80fc9e50b',
    'chin_throat_contour_preview_r11.py': '7e0777082dacf1069c4ae195517e3ee2c121376ce0cee99cb7cb6208b5575e90',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def api(name):
    path = Path(__file__).with_name(name+'.py')
    if path.name in PINNED and sha(path) != PINNED[path.name]:
        raise RuntimeError('Frozen dependency changed: '+path.name)
    return runpy.run_path(str(path))


def artifact(relative, suffix):
    path = (project_path()/relative).resolve()
    path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix != suffix:
        raise ValueError('Explicit independent reconstruction artifact required')
    return path


def all_rig_channels(bpy, uv):
    """Include inactive rotations; never restore a bone by decomposing a matrix."""
    names = ('location', 'rotation_mode', 'rotation_quaternion',
             'rotation_euler', 'rotation_axis_angle', 'scale')
    def channels(block):
        return {name: uv['_plain'](getattr(block, name)) for name in names}
    result = {}
    for rig in bpy.context.scene.objects:
        if rig.type != 'ARMATURE':
            continue
        result[rig.name] = {
            'object': channels(rig),
            'delta': {name: uv['_plain'](getattr(rig, name)) for name in
                      ('delta_location', 'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale')},
            'bones': {bone.name: channels(bone) for bone in rig.pose.bones},
        }
    return result


def head_contract(head, uv, natural, groom, face, shared, owned_modifier=None):
    keys = natural['_key_records'](head.data, uv)
    own_key = keys['blocks'].pop(KEY, None)
    return {
        'originalKeys': keys,
        'keyOrder': [key.name for key in head.data.shape_keys.key_blocks if key.name != KEY],
        'keySettings': uv['_scalar_rna'](head.data.shape_keys),
        'rawDataSha256': groom['_data_hash'](head.data),
        'uvOrder': [layer.name for layer in head.data.uv_layers],
        'uvLayers': {layer.name: sha_array(uv['_values'](layer.data, 'uv', 2))
                     for layer in head.data.uv_layers},
        'uvSettings': uv['_uv_settings'](head.data),
        'materials': [{'link': slot.link, 'pointer': slot.material.as_pointer() if slot.material else None,
                       'name': slot.material.name if slot.material else None,
                       'material': face['_material_contract'](slot.material) if slot.material else None}
                      for slot in head.material_slots],
        'modifiers': shared['modifier_contract'](head, uv, owned_modifier),
        'headCustom': uv['_custom'](head), 'meshCustom': uv['_custom'](head.data),
        'activeShapeKeyIndex': head.active_shape_key_index,
    }, own_key


def sha_array(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()


def original_modifier_contract(bpy, uv, shared, excluded_object=None, excluded_modifier=None):
    """Original modifier order, settings, pointers, GN inputs and tree semantics."""
    capture_group = api('upper_lash_corrective_validation_r11')['capture_group']
    objects, groups = {}, {}
    def capture(group):
        if group.name in groups:
            return
        groups[group.name] = {'tree': capture_group(group), 'custom': uv['_custom'](group),
                              'animation': uv['_animation'](group)}
        for node in group.nodes:
            if getattr(node, 'node_tree', None) is not None:
                capture(node.node_tree)
    for obj in bpy.context.scene.objects:
        if obj == excluded_object:
            continue
        rows = []
        for mod in obj.modifiers:
            if mod == excluded_modifier:
                continue
            pointers = {}
            for prop in mod.bl_rna.properties:
                if prop.identifier != 'rna_type' and not prop.is_readonly and prop.type == 'POINTER':
                    pointers[prop.identifier] = uv['_plain'](getattr(mod, prop.identifier))
            try:
                custom = uv['_custom'](mod)
            except TypeError:
                if mod.type == 'NODES':
                    raise
                custom = {'idPropertiesSupported': False}
            rows.append({'name': mod.name, 'pointers': pointers, 'custom': custom})
            group = getattr(mod, 'node_group', None)
            if group is not None:
                capture(group)
        objects[obj.name] = {'modifiers': shared['modifier_contract'](obj, uv, excluded_modifier), 'inputs': rows}
    return {'objects': objects, 'groups': groups}


def prepare(source_relative, output_relative):
    import bpy
    p = project_path()
    source, output = artifact(source_relative, '.blend'), artifact(output_relative, '.blend')
    record = output.with_suffix('.mouth-corners.json')
    attempt = output.with_suffix('.mouth-corners-attempt.json')
    payload = output.with_suffix('.mouth-corners.npz')
    if source == output or any(path.exists() for path in (output, record, attempt, payload)):
        raise RuntimeError('Preserve source and all existing candidate/diagnostic outputs')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh saved background source required')
    scene = bpy.context.scene
    if (scene.frame_current, scene.frame_subframe) != (1, 0.):
        raise RuntimeError('Open the saved frame-one source; this installer never changes frames')
    head = bpy.data.objects[HEAD]
    if not head.data.shape_keys or not head.data.shape_keys.use_relative:
        raise RuntimeError('Existing relative head controls required')
    keys = head.data.shape_keys.key_blocks
    if KEY in keys or MARKER in scene or WARM not in keys:
        raise RuntimeError('Fresh corner candidate and existing warm control required')
    for name in ('R11 blink L', 'R11 blink R', 'R11 gentle smile', 'R11 natural soft smile', WARM, 'R11 jaw open'):
        if name in keys and float(keys[name].value) != 0.:
            raise RuntimeError('Original default must be neutral: '+name)
    for name in ('R11 relaxed lip seal', 'R11 lower face contour v2'):
        if name not in keys or float(keys[name].value) != 1.:
            raise RuntimeError('Retain the current accepted default: '+name)
    if float(head.get('R11 lower face continuity', 0.)) != 0.:
        raise RuntimeError('Rejected lower continuity must remain disabled')
    if 'R11 upper oral clearance' in keys and float(keys['R11 upper oral clearance'].value) != 0.:
        raise RuntimeError('Unmeasured earlier oral-clearance layer is active')
    for name, expected in PINNED.items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise RuntimeError('Frozen dependency changed: '+name)
    plan_path = artifact(PLAN, '.json')
    if sha(plan_path) != PLAN_SHA:
        raise RuntimeError('Frozen corner-plan report changed')
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    plan_arrays = artifact(plan['arrays'], '.npz')
    if sha(plan_arrays) != ARRAYS_SHA or plan['arraysSha256'] != ARRAYS_SHA:
        raise RuntimeError('Frozen corner field changed')
    if plan['remainingCrossingPairs'] or not plan['centralExact']:
        raise RuntimeError('Frozen neutral plan is not complete')
    with np.load(plan_arrays) as loaded:
        frozen = {name: loaded[name].copy() for name in loaded.files}
    preview = api('relaxed_mouth_corner_preview_r11')
    snapshot, external_geometry, array = (preview[name] for name in ('snapshot', 'external_geometry', 'array'))
    states = api('chin_throat_contour_preview_r11_v2')
    uv = api('production_uv_r08')
    natural, groom = api('natural_smile_adult_r11'), api('posterior_groom_r09')
    face, shared = api('runtime_face_refresh_r08'), api('upper_lid_relief_install_r11')
    qa = api('face_contact_audit_r11')
    quality_fn = api('upper_lid_roll_install_r11')['quality']
    cross = api('adult_blink_midpath_contact_r11')['changed_intersections']
    source_sha = sha(source)
    old_active, old_warm = head.active_shape_key_index, float(keys[WARM].value)
    saved_state = states['state'](head, uv, excluded_key=KEY)
    saved_rig_channels = all_rig_channels(bpy, uv)
    saved_head, _ = head_contract(head, uv, natural, groom, face, shared)
    saved_external = shared['external_contract'](bpy, head)
    saved_modifiers = original_modifier_contract(bpy, uv, shared)
    saved_scene_custom = uv['_custom'](scene)
    modifier = group = cache = mesh = key = None
    success = False
    offset = np.asarray(frozen['offsets'], float)
    active = np.any(offset != 0., axis=1)
    ids = np.flatnonzero(active).astype(np.int32)
    deps = frozen['dependencyOriginalVertexIds']
    diagnostic = {name: frozen[name] for name in ('offsets', 'commonOffsets', 'contactOffsets',
                  'dependencyOriginalVertexIds', 'protectedCentral', 'contactRegion', 'upperExclusive', 'lowerExclusive')}
    report = {
        'source': source_relative, 'sourceSha256': source_sha,
        'helperSha256': sha(__file__), 'frozenPlan': PLAN, 'frozenPlanSha256': PLAN_SHA,
        'frozenPlanArraysSha256': ARRAYS_SHA, 'actualChecks': [], 'guardErrors': [],
        'candidateSaved': False, 'technicalAcceptance': False, 'appearanceAccepted': False,
        'control': KEY, 'defaultWarm': 0., 'defaultCorner': 1.,
        'scope': 'Reviewed neutral corner release; current warm 0/.5/1 local combination checks. No jaw/dental/full-expression or complete-character acceptance.',
        'savedState': saved_state, 'savedRigChannels': saved_rig_channels,
        'savedHeadContract': saved_head,
        'helperHashes': {name+'.py': sha(Path(__file__).with_name(name+'.py')) for name in
            ('relaxed_mouth_corner_preview_r11', 'relaxed_mouth_corner_nodes_r11', 'relaxed_mouth_corners_plan_r11',
             'chin_throat_contour_preview_r11_v2', 'chin_throat_contour_preview_r11', 'production_uv_r08',
             'natural_smile_adult_r11', 'posterior_groom_r09', 'runtime_face_refresh_r08',
             'upper_lid_relief_install_r11', 'upper_lid_roll_install_r11', 'face_contact_audit_r11',
             'adult_blink_midpath_contact_r11', 'upper_lash_corrective_validation_r11')},
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        np.savez_compressed(payload, **diagnostic)
        report.update(arrays=str(payload.relative_to(p)), arraysSha256=sha(payload))
        attempt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    def values(warm, strength=None):
        keys[WARM].value = float(warm)
        if strength is not None:
            keys[KEY].value = float(strength)
        head.update_tag()
        bpy.context.view_layer.update()

    def store_snapshot(prefix, snap):
        diagnostic.update({prefix+name[0].upper()+name[1:]: value for name, value in snap.items()})

    def exact_originals(label):
        current_state = states['state'](head, uv, excluded_key=KEY)
        current_head, own = head_contract(head, uv, natural, groom, face, shared, modifier)
        diffs = {
            'authorAndPose': states['exact_diff'](saved_state, current_state),
            'allRigChannels': states['exact_diff'](saved_rig_channels, all_rig_channels(bpy, uv)),
            'head': states['exact_diff'](saved_head, current_head),
            'external': states['exact_diff'](saved_external, shared['external_contract'](bpy, head, cache)),
            'originalModifiers': states['exact_diff'](saved_modifiers, original_modifier_contract(bpy, uv, shared, cache, modifier)),
        }
        scene_custom = uv['_custom'](scene)
        scene_custom.pop(MARKER, None)
        diffs['sceneCustom'] = states['exact_diff'](saved_scene_custom, scene_custom)
        report[label] = {'diffs': diffs, 'state': current_state, 'sourceFileExact': sha(source) == source_sha}
        basis = saved_head['originalKeys']['blocks'][head.data.shape_keys.reference_key.name]['coordinatesSha256']
        report[label]['ownKeyZeroRawDelta'] = bool(own and own['coordinatesSha256'] == basis)
        persist()
        if any(diffs.values()) or not report[label]['sourceFileExact'] or not report[label]['ownKeyZeroRawDelta']:
            raise RuntimeError('Original author/pose/raw keys/UV/material/modifiers/source changed: '+label)

    try:
        bpy.context.view_layer.update()
        report['initialUpdateStateDiff'] = states['exact_diff'](saved_state, states['state'](head, uv, excluded_key=KEY))
        report['initialUpdateRigChannelDiff'] = states['exact_diff'](saved_rig_channels, all_rig_channels(bpy, uv))
        persist()
        if report['initialUpdateStateDiff'] or report['initialUpdateRigChannelDiff']:
            raise RuntimeError('Initial ordinary update changed original source author/pose channels')
        baseline = {}
        with qa['rest_evaluation']():
            for warm in (0., .5, 1.):
                values(warm)
                baseline[warm] = now = snapshot(head)
                store_snapshot('beforeWarm'+str(warm), now)
                if not np.isfinite(now['world']).all() or not np.isfinite(now['matrix']).all():
                    persist()
                    raise RuntimeError('Current actual source contains nonfinite coordinates')
                if not np.array_equal(now['faces'], frozen['faces']) or not np.array_equal(now['groups'], frozen['groups']):
                    persist()
                    raise RuntimeError('Current complete topology/anatomical correspondence differs')
                if warm == 0.:
                    error = np.linalg.norm(now['world'][deps]-frozen['fullWorldBefore'][deps], axis=1)
                    report['neutralDependency'] = {'vertices': len(deps), 'exact': bool(np.array_equal(now['world'][deps], frozen['fullWorldBefore'][deps])),
                        'maximumMeters': float(error.max())}
                    persist()
                    if not report['neutralDependency']['exact']:
                        raise RuntimeError('Current actual 20111-point corner dependency changed')
                if offset.shape != now['world'].shape or not np.isfinite(offset).all():
                    persist()
                    raise RuntimeError('Finite complete fixed-L3 field required')
            values(0.)
            rest_external = external_geometry({head})
            key = head.shape_key_add(name=KEY, from_mix=False)
            key.relative_key = head.data.shape_keys.reference_key
            key.data.foreach_set('co', array(key.relative_key.data, 'co', 3, np.float32).ravel())
            key.slider_min, key.slider_max, key.value = 0., 1., 0.
            modifier, group, cache, mesh = api('relaxed_mouth_corner_nodes_r11')['build'](
                head, len(offset), ids, offset[ids], baseline[0.]['matrix'])
            for warm in (0., .5, 1.):
                before = baseline[warm]
                values(warm, 0.)
                zero = snapshot(head)
                store_snapshot('zeroWarm'+str(warm), zero)
                zero_exact = all(np.array_equal(before[name], zero[name]) for name in
                    ('world', 'faces', 'groups', 'matrix', 'triangles', 'trianglePolygonIds'))
                values(warm, 1.)
                after = snapshot(head)
                store_snapshot('afterWarm'+str(warm), after)
                if not np.isfinite(after['world']).all() or not np.isfinite(after['matrix']).all():
                    persist()
                    raise RuntimeError('Current actual corner candidate contains nonfinite coordinates')
                delta = after['world']-before['world']
                topology = all(np.array_equal(before[name], after[name]) for name in ('faces', 'groups', 'matrix', 'trianglePolygonIds'))
                protected = {
                    'centralSealAndCentralLipChin': np.flatnonzero((abs(before['world'][:, 0]) <= .010) | frozen['protectedCentral']),
                    'eyesEars': np.unique(before['faces'][np.isin(before['groups'], [2, 3, 4, 5])]),
                    'lowNeck': np.flatnonzero(before['world'][:, 2] < 1.335),
                    'outsideContinuousSupport': np.flatnonzero(~active),
                }
                protections = {name: {'vertices': len(v), 'exact': bool(np.array_equal(before['world'][v], after['world'][v])),
                    'maximumMeters': float(np.linalg.norm(delta[v], axis=1).max()) if len(v) else 0.}
                    for name, v in protected.items()}
                quality = quality_fn(before['world'], after['world'], after['triangles'], active)
                row = {'warm': warm, 'corners': 1., 'zeroExact': zero_exact, 'topologyExact': topology,
                    'expectedMaximumErrorMeters': float(np.linalg.norm(delta-offset, axis=1).max()),
                    'maximumActualMeters': float(np.linalg.norm(delta, axis=1).max()), 'protected': protections,
                    'quality': quality, 'changedTriangleRows': int(np.count_nonzero(np.any(before['triangles'] != after['triangles'], axis=1)))}
                report['actualChecks'].append(row)
                persist()
                if not zero_exact or not topology or any(not r['exact'] for r in protections.values()) or row['expectedMaximumErrorMeters'] > 3e-7:
                    raise RuntimeError('Actual corner field differs or moves protected anatomy at warm '+str(warm))
                if not np.isfinite([quality['minimumNormalAlignment'], *quality['areaRatioRange'], row['expectedMaximumErrorMeters']]).all():
                    raise RuntimeError('Actual corner field has nonfinite differential geometry at warm '+str(warm))
                if quality['nonpositiveNormals'] or quality['minimumNormalAlignment'] < .75 or quality['areaRatioRange'][0] < .50 or quality['areaRatioRange'][1] > 1.8:
                    raise RuntimeError('Actual corner field folds/compresses excessively at warm '+str(warm))
                # Independently use each surface's actual triangulation. A zero
                # final set cannot hide an old pair by a diagonal reassignment.
                old_cross = cross(before['world'], before['triangles'], active)
                new_cross = cross(after['world'], after['triangles'], active)
                diagnostic['beforeWarm'+str(warm)+'CrossingPairs'] = np.asarray(sorted(old_cross), np.int32).reshape(-1, 2)
                diagnostic['afterWarm'+str(warm)+'CrossingPairs'] = np.asarray(sorted(new_cross), np.int32).reshape(-1, 2)
                row.update(oldActualCrossings=len(old_cross), newActualCrossings=len(new_cross),
                           crossingScope='Active triangles against complete current head, nonadjacent strict intersections, actual per-state triangulation')
                persist()
                if new_cross:
                    raise RuntimeError('Actual corner intersection remains at warm '+str(warm))
                print('R11_RELAXED_CORNERS_ACTUAL '+json.dumps({'warm': warm, 'crossings': 0, 'maximumMeters': row['maximumActualMeters']}), flush=True)
            values(old_warm, 1.)
            head.active_shape_key_index = old_active
            final_rest = snapshot(head)
            store_snapshot('defaultRest', final_rest)
            report['defaultRestRepeatExact'] = all(np.array_equal(final_rest[name], diagnostic['afterWarm0.0'+name[0].upper()+name[1:]])
                for name in ('world', 'faces', 'groups', 'matrix', 'triangles', 'trianglePolygonIds'))
            report['defaultRestExternalGeometryDiff'] = states['exact_diff'](rest_external, external_geometry({head, cache}))
            persist()
            if not report['defaultRestRepeatExact'] or report['defaultRestExternalGeometryDiff']:
                raise RuntimeError('Restoring default expression changed actual head or external geometry')
        head.active_shape_key_index = old_active
        exact_originals('beforeSave')
        group['R11_scope'] = 'Reviewed static mouth-corner candidate; actual current-source warm 0/.5/1 checks passed. Jaw/dental/motion and overall appearance not accepted.'
        report.update(actualLocalGeometryChecksPassed=True, technicalAcceptance=True,
            controlType='Zero-raw-displacement Key driving native fixed-L3 XYZ field',
            modifier=modifier.name, nodeGroup=group.name, lookupObject=cache.name,
            limitations=['Only local current-source neutral and warm 0/.5/1 geometry is checked.',
                         'Default warm remains exactly zero; warm one is a full smile, not the intended resting expression.',
                         'Jaw opening, dental contact and full animation need new combined validation.',
                         'Whole-face final appearance remains unaccepted.'])
        scene[MARKER] = json.dumps({'source': source_relative, 'sourceSha256': source_sha,
            'report': str(record.relative_to(p)), 'control': KEY, 'defaultWarm': 0.,
            'fieldArraysSha256': ARRAYS_SHA, 'localChecksPassed': True, 'appearanceAccepted': False})
        persist()
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False, relative_remap=True):
            raise RuntimeError('Independent mouth-corner save failed')
        report.update(candidateSaved=True, output=str(output.relative_to(p)), outputSha256=sha(output))
        exact_originals('afterSave')
        report['complete'] = True
        record.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        persist()
        success = True
        print('R11_RELAXED_CORNERS_SAVED '+json.dumps({'output': report['output'], 'sha256': report['outputSha256']}), flush=True)
        return report
    except BaseException as exc:
        report['technicalAcceptance'] = False
        report['guardErrors'].append(str(exc))
        if output.exists():
            report.update(candidateFileRetainedAfterFailure=True, output=str(output.relative_to(p)), outputSha256=sha(output))
        # Retain raw pose evidence before any cleanup; no bone matrix rewrite.
        report['failureState'] = states['state'](head, uv, excluded_key=KEY)
        report['failureAllRigChannels'] = all_rig_channels(bpy, uv)
        persist()
        raise
    finally:
        if not success:
            keys[WARM].value = old_warm
            head.active_shape_key_index = old_active
            if modifier is not None:
                head.modifiers.remove(modifier)
            if key is not None:
                head.shape_key_remove(key)
            if MARKER in scene:
                del scene[MARKER]
            if cache is not None:
                bpy.data.objects.remove(cache, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
            if group is not None and group.users == 0:
                bpy.data.node_groups.remove(group)
            head.update_tag()
            bpy.context.view_layer.update()
            report['rollbackStateDiff'] = states['exact_diff'](saved_state, states['state'](head, uv, excluded_key=KEY))
            report['rollbackRigChannelDiff'] = states['exact_diff'](saved_rig_channels, all_rig_channels(bpy, uv))
            persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.output)
