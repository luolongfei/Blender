"""Install the verified R10 v2 HBM hand payload into an independent R11 copy.

This is a reversible geometry/weight integration, not an appearance approval.
The payload is the same actual evaluated donor geometry as R10 v2; it is not
regenerated from primitives. No rig, action, head, body, garment or hair edits.
Run only in a disposable background Blender; only prepare() saves a new copy.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

MARKER = 'R11_hbm_hands_integrated'
PLAN = 'work/reconstruction-r02/hand-asset-fit-plan-r10-v3.json'
PLAN_SHA = '9296d025a6b6559842a35ab222d8af86bf86e1bf296865a5fb26e7ea169df94c'
FIT_SHA = 'b5ef17043b22dcad3a148cda8d0cd61e0325b38b5e471f470635b69e8aac5f3c'
DONOR_CANDIDATE = 'work/reconstruction-r02/reference-asset-hands-r10-v2.blend'
DONOR_SHA = '7f45d5a151c8b5efb61afda46d482d97dc60ed5774a988847d82cd6447dd9b4b'
HANDS = ('Hand L · continuous anatomy', 'Hand R · continuous anatomy')
FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Little')


def _path(relative):
    p = (project_path() / relative).resolve()
    p.relative_to(project_path().resolve())
    return p


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _fit():
    path = Path(__file__).with_name('hand_asset_fit_r10.py')
    if _sha(path) != FIT_SHA:
        raise RuntimeError('Frozen hand fit helper changed; inspect before updating this integration')
    return runpy.run_path(str(path))


def _recipe():
    if _sha(_path(PLAN)) != PLAN_SHA:
        raise RuntimeError('Verified v3 hand plan changed')
    recipe = json.loads(_path(PLAN).read_text(encoding='utf-8'))
    if not recipe['candidateReady'] or _sha(_path(recipe['payload'])) != recipe['payloadSha256']:
        raise RuntimeError('Verified v3 hand payload changed or is not ready')
    obs_path = _path(recipe['observation'])
    if _sha(obs_path) != recipe['observationSha256']:
        raise RuntimeError('Observed hand baseline changed')
    obs = json.loads(obs_path.read_text(encoding='utf-8'))
    if _sha(_path(obs['arrays'])) != recipe['observedArraysSha256']:
        raise RuntimeError('Observed hand arrays changed')
    return recipe, obs


def _groups(obj):
    return [{'name': g.name, 'index': g.index, 'lockWeight': g.lock_weight} for g in obj.vertex_groups]


def _mesh_contract(obj):
    """Persist raw topology/UV/material/weights, not evaluated pose-dependent data."""
    mesh = obj.data
    h = hashlib.sha256()
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get('co', co.ravel())
    h.update(co.tobytes())
    loops = np.empty(len(mesh.loops), np.int32)
    mesh.loops.foreach_get('vertex_index', loops)
    h.update(loops.tobytes())
    for poly in mesh.polygons:
        h.update(np.asarray((poly.loop_start, poly.loop_total, poly.material_index, poly.use_smooth), np.int32).tobytes())
    uvs = []
    for uv in mesh.uv_layers:
        values = np.empty((len(uv.data), 2), np.float32)
        uv.data.foreach_get('uv', values.ravel())
        h.update(uv.name.encode('utf-8')); h.update(values.tobytes()); uvs.append(uv.name)
    attr = mesh.attributes.get('R10_hbm_face_group')
    if attr:
        groups = np.empty(len(attr.data), np.int32); attr.data.foreach_get('value', groups); h.update(groups.tobytes())
    weights = np.zeros((len(mesh.vertices), len(obj.vertex_groups)), np.float32)
    for vertex in mesh.vertices:
        for group in vertex.groups:
            weights[vertex.index, group.group] = group.weight
    h.update(weights.tobytes())
    definition = _groups(obj)
    h.update(json.dumps(definition, sort_keys=True).encode('utf-8'))
    return {'sha256': h.hexdigest(), 'vertices': len(mesh.vertices), 'polygons': len(mesh.polygons),
            'uvLayers': uvs, 'groups': definition, 'materials': [m.name if m else None for m in mesh.materials],
            'shapeKeys': mesh.shape_keys.name if mesh.shape_keys else None}


def _protected_objects(bpy, hands):
    return {o.as_pointer(): (o.name, o.data.as_pointer() if o.data else None,
            o.parent.as_pointer() if o.parent else None, o.parent_type, o.parent_bone,
            np.asarray(o.matrix_parent_inverse).tolist(), np.asarray(o.matrix_basis).tolist())
            for o in bpy.data.objects if o not in hands}


def validate():
    import bpy
    record = json.loads(bpy.context.scene[MARKER])
    if record['planSha256'] != PLAN_SHA:
        raise RuntimeError('A different hand integration is already present')
    fit = _fit()
    for row in record['changes']:
        obj = bpy.data.objects[row['hand']]
        if _mesh_contract(obj) != row['installedContract']:
            raise RuntimeError('Integrated hand has changed; preserve it rather than reinstall: ' + obj.name)
        original = bpy.data.objects[row['backupObject']]
        if _mesh_contract(original) != row['originalContract']:
            raise RuntimeError('Retained original hand no longer matches its integration snapshot')
        rig = bpy.data.objects['Character_Rig']
        armatures = [m for m in obj.modifiers if m.type == 'ARMATURE']
        if len(armatures) != 1 or armatures[0].object != rig or not armatures[0].use_vertex_groups:
            raise RuntimeError('Integrated hand rig attachment changed')
    return {'alreadyInstalled': True, 'rawContractsExact': True, 'appearanceAccepted': False,
            'actualNailBoneAttachment': [fit['hand_camera'](None, bpy.data.objects['Character_Rig'], side, None,
                                        diagnostics_only=True) for side in ('L', 'R')]}


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Disposable background Object Mode required')
    scene = bpy.context.scene
    if MARKER in scene:
        return validate()
    recipe, obs = _recipe(); fit = _fit()
    if scene.frame_current != 1 or scene.frame_subframe != 0:
        raise RuntimeError('Keep actual frame 1; this integration never resets the body pose')
    rig = bpy.data.objects['Character_Rig']; rig_before = fit['_rig_contract'](rig)
    if rig.data.pose_position != 'POSE':
        raise RuntimeError('Current standing pose required for actual hand attachment verification')
    for row in obs['current'][-1]['bones']:
        bone = rig.data.bones.get(row['name'])
        if bone is None or not np.allclose(np.asarray(bone.matrix_local), row['restMatrixLocal'], atol=2e-7, rtol=0):
            raise RuntimeError('Target rig rest differs from hand fit: ' + row['name'])
    hands = [bpy.data.objects[name] for name in HANDS]
    observed = {o['name']: o for o in obs['current'][:2]}
    columns = {}; old_data = []; original_contract = {}
    with np.load(_path(obs['arrays'])) as arrays:
        for obj, side in zip(hands, ('left', 'right')):
            if obj.data.shape_keys or 'R10_hbm_hand_provenance' in obj:
                raise RuntimeError('Existing modified hand requires explicit reconciliation, not replacement: ' + obj.name)
            if hashlib.sha256(fit['_coordinates'](obj.data).tobytes()).hexdigest() != recipe['sourceHandCoordinatesSha256'][obj.name]:
                raise RuntimeError('Current raw hand differs from fitted baseline: ' + obj.name)
            if not np.allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-7, rtol=0):
                raise RuntimeError('Target hand coordinate space changed')
            columns[obj.name] = fit['_group_columns'](_groups(obj), observed[obj.name]['vertexGroups'],
                                                      recipe['weightTransfer']['sourceGroups'], side)
            fit['_source_weights_unchanged'](obj, observed[obj.name], arrays)
            original_contract[obj.name] = _mesh_contract(obj); old_data.append(obj.data)
    protected = _protected_objects(bpy, hands)
    nails = [bpy.data.objects['Nail ' + side + ' ' + finger + ' · polished'] for side in ('L', 'R') for finger in FINGERS]
    nail_state = {o.name: {'render': o.hide_render, 'viewport': o.hide_viewport, 'hideGet': o.hide_get()} for o in nails}
    backups = []; created_meshes = []; success = False
    collection = bpy.data.collections.new('R11 retained original hands')
    scene.collection.children.link(collection); collection.hide_render = True; collection.hide_viewport = True
    result = {'schemaVersion': 1, 'plan': PLAN, 'planSha256': PLAN_SHA, 'payloadSha256': recipe['payloadSha256'],
              'matchingR10Candidate': DONOR_CANDIDATE, 'matchingR10CandidateSha256': DONOR_SHA,
              'sourceIsActualEvaluatedHBMAsset': True, 'assetName': recipe['assetName'],
              'assetMetadata': recipe['assetMetadata'], 'bundleSha256': recipe['bundleSha256'],
              'changes': [], 'originalNailVisibility': nail_state, 'masterChanged': False,
              'appearanceAccepted': False, 'limits': [recipe['wristScope'],
              'Visible wrist overlap and flat nail shading remain open issues; this is a starting point for local corrections.',
              'Flexed fingers, palm, wrist/cuff contact and continuous hand movement need actual inspection.']}
    try:
        with np.load(_path(recipe['payload'])) as payload:
            loops = payload['loopVertex']; starts = payload['polygonStart']; sizes = payload['polygonSize']
            groups = payload['faceGroup']
            for obj, side, previous in zip(hands, ('L', 'R'), old_data):
                backup = obj.copy(); backup.name = 'R11 retained ' + obj.name; collection.objects.link(backup)
                backup.hide_render = True; backup.hide_viewport = True; backups.append(backup)
                mesh = bpy.data.meshes.new('R11 integrated HBM hand ' + side); created_meshes.append(mesh)
                mesh.use_fake_user = True
                faces = [loops[a:a+b].tolist() for a, b in zip(starts, sizes)]
                if side == 'L': faces = [list(reversed(face)) for face in faces]
                mesh.from_pydata(payload['left' if side == 'L' else 'right'].tolist(), [], faces); mesh.update()
                uv = mesh.uv_layers.new(name='R10 HBM Hand UV'); values = np.empty((len(mesh.loops), 2), np.float32)
                for polygon, start, size in zip(mesh.polygons, starts, sizes):
                    source = payload['uv'][start:start+size]
                    values[polygon.loop_start:polygon.loop_start+polygon.loop_total] = source[::-1] if side == 'L' else source
                uv.data.foreach_set('uv', values.ravel())
                mesh.attributes.new('R10_hbm_face_group', 'INT', 'FACE').data.foreach_set('value', groups)
                mesh.materials.append(previous.materials[0])
                mesh.materials.append(bpy.data.objects['Nail ' + side + ' Index · polished'].data.materials[0])
                for polygon, group in zip(mesh.polygons, groups):
                    polygon.material_index = 1 if group == 10 else 0; polygon.use_smooth = True
                obj.data = mesh
                if len(obj.vertex_groups) == 0:
                    for row in observed[obj.name]['vertexGroups']:
                        group = obj.vertex_groups.new(name=row['name']); group.lock_weight = row['lockWeight']
                        if group.index != row['index']: raise RuntimeError('Actual group index changed during installation')
                fit['_group_columns'](_groups(obj), observed[obj.name]['vertexGroups'],
                                      recipe['weightTransfer']['sourceGroups'], 'left' if side == 'L' else 'right')
                expected = np.zeros((len(mesh.vertices), len(obj.vertex_groups)), np.float32)
                for group in obj.vertex_groups:
                    values = payload['weights'][:, columns[obj.name][group.index]]; expected[:, group.index] = values
                    for index in np.flatnonzero(values > 0): group.add([int(index)], float(values[index]), 'REPLACE')
                weight_check = fit['_installed_weights'](obj, expected)
                obj['R10_hbm_hand_original_object'] = backup.name
                obj['R10_hbm_hand_provenance'] = json.dumps({'asset': recipe['assetName'], 'assetMetadata': recipe['assetMetadata'],
                    'bundleSha256': recipe['bundleSha256'], 'observationSha256': recipe['observationSha256'],
                    'fitPayloadSha256': recipe['payloadSha256'], 'faceGroupAttribute': 'R10_hbm_face_group',
                    'nailFaceGroup': 10, 'side': side}, ensure_ascii=False)
                result['changes'].append({'hand': obj.name, 'backupObject': backup.name,
                    'originalContract': original_contract[obj.name], 'installedContract': _mesh_contract(obj),
                    'installedWeights': weight_check})
        for obj in nails: obj.hide_render = True; obj.hide_viewport = True; obj.hide_set(True)
        bpy.context.view_layer.update()
        result['actualArmatureMotion'] = [fit['_actual_armature_motion'](bpy, obj, rig) for obj in hands]
        result['actualNailBoneAttachment'] = [fit['hand_camera'](None, rig, side, None, diagnostics_only=True) for side in ('L', 'R')]
        if any(max(n['centerTipDistance'] for n in row['nails']) > .020 for row in result['actualNailBoneAttachment']):
            raise RuntimeError('Integrated hand does not follow actual posed finger bones')
        if fit['_rig_contract'](rig) != rig_before:
            raise RuntimeError('Hand integration changed rig/action/rest/pose')
        if any(_protected_objects(bpy, hands).get(key) != value for key, value in protected.items()):
            raise RuntimeError('A pre-existing non-hand object/data/parent/transform changed')
        result['rigActionRestPoseUnchanged'] = True; result['otherObjectDataParentsTransformsUnchanged'] = True
        scene[MARKER] = json.dumps(result, ensure_ascii=False); success = True
        return result
    finally:
        if not success:
            for obj, data in zip(hands, old_data):
                obj.data = data
                for key in ('R10_hbm_hand_original_object', 'R10_hbm_hand_provenance'):
                    if key in obj: del obj[key]
            for obj in nails:
                row = nail_state[obj.name]; obj.hide_render = row['render']; obj.hide_viewport = row['viewport']; obj.hide_set(row['hideGet'])
            for obj in backups: bpy.data.objects.remove(obj, do_unlink=True)
            for mesh in created_meshes: bpy.data.meshes.remove(mesh)
            bpy.data.collections.remove(collection)
            if MARKER in scene: del scene[MARKER]
            bpy.context.view_layer.update()


def restore():
    """Restore only this unchanged in-memory installation; retain both meshes."""
    import bpy
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Use a disposable background copy for restoration')
    scene = bpy.context.scene
    if MARKER not in scene:
        return {'alreadyRestored': True, 'savedBlend': False}
    # Never discard a subsequent sculpt, UV edit or weight edit implicitly.
    validate()
    record = json.loads(scene[MARKER]); restored = []
    for row in record['changes']:
        obj = bpy.data.objects[row['hand']]; backup = bpy.data.objects[row['backupObject']]
        obj.data.use_fake_user = True; obj.data = backup.data
        for key in ('R10_hbm_hand_original_object', 'R10_hbm_hand_provenance'):
            if key in obj: del obj[key]
        if _mesh_contract(obj) != row['originalContract']:
            raise RuntimeError('Original hand groups/geometry were not exactly restored')
        restored.append(obj.name)
    for name, row in record['originalNailVisibility'].items():
        obj = bpy.data.objects[name]; obj.hide_render = row['render']; obj.hide_viewport = row['viewport']; obj.hide_set(row['hideGet'])
    del scene[MARKER]; bpy.context.view_layer.update()
    return {'restored': restored, 'savedBlend': False, 'newMeshesRetained': True}


def prepare(source_relative, output_relative, source_sha256):
    import bpy
    source = _path(source_relative); output = _path(output_relative); report = output.with_suffix('.hands-r11.json')
    if Path(bpy.data.filepath).resolve() != source or _sha(source) != source_sha256:
        raise RuntimeError('Explicit source identity mismatch')
    if output == source or not output.is_relative_to(_path('work/reconstruction-r02')):
        raise RuntimeError('Save only a new independent reconstruction candidate')
    if output.exists() or report.exists(): raise FileExistsError('Preserve previous candidates and reports')
    result = apply()
    if _sha(source) != source_sha256: raise RuntimeError('Source changed on disk during hand integration')
    result.update({'source': source_relative, 'sourceSha256': source_sha256,
                   'helperSha256': _sha(__file__), 'sourceUnchangedOnDisk': True})
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False, relative_remap=True)
    result.update({'candidate': output_relative, 'candidateSha256': _sha(output)})
    with report.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(argv)
    result = prepare(args.source, args.output, args.source_sha)
    print('R11_HAND_INTEGRATION ' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
