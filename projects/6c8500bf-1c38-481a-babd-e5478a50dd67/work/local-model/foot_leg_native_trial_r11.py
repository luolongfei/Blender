"""Disposable native unified-leg/foot and provisional joint-axis trial.

Only the integrator runs this in background Blender. All armature, pose,
constraint and mesh edits are on newly copied datablocks. No .blend is saved.
The input foot length and surface-derived axes are artistic candidates, not
population standards or certified anatomical joint centres. This isolated FK
probe deliberately does not reuse the source action or validate production IK.
"""
import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

STAGE = 'work/reconstruction-r02/'
PINS = {
    STAGE + 'foot-leg-unified-candidate-r11-v2.json': '8977e539f122dc0cdf501b3a8f3fef2f72acab6ef94337aadc8229046a178a11',
    STAGE + 'foot-leg-unified-candidate-r11-v2.npz': 'f3f245b3ee2d646af28dee7140009eb44baba60e9b10ae2ad2e8b090bd091129',
    STAGE + 'foot-leg-join-capture-r11.json': 'adb90f20b6369b8228d0a19ddc634d0598ab8b151ea3e1ca856d505efa1f1578',
    STAGE + 'foot-leg-join-capture-r11.npz': '18aae98b8a9c7d054985c772e4bf52d019c96cf4d53133742fb67fd214d46198',
    STAGE + 'foot-225-placement-plan-r11.json': '52e7e5f3a4b45abc2e4857c2c05adb9b1b9b4ede95a4575cdeffe56d7407d4f2',
    STAGE + 'foot-225-placement-plan-r11.npz': '89cf1cee619c639628552e13c6bf23b1b229200e6fc39417d7f653e2bda9d776',
    'work/local-model/hand_asset_observe_r10.py': 'b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c',
}
CASE_SPECS = (
    ('REST', None, 0.0),
    ('neutral_fk', None, 0.0),
    ('ankle_dorsiflex_15', 'Foot', -15.0),
    ('ankle_plantarflex_25', 'Foot', 25.0),
    ('knee_flex_45', 'LowerLeg', 45.0),
    ('toe_flex_25', 'Toes', 25.0),
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def path_for(relative):
    path = (project_path() / relative.replace('\\', '/')).resolve()
    path.relative_to(project_path().resolve())
    return path


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-9:
        raise RuntimeError('Degenerate provisional axis')
    return vector / length


def world_points(local, matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray(local) @ matrix[:3, :3].T + matrix[:3, 3]


def axis_plan(side, plan, candidate):
    row = plan['sides'][side]
    roots = np.asarray(row['toeRootCentersWorld'], dtype=float)
    ankle = np.asarray(row['provisionalAnkleAxisCenterWorld'], dtype=float)
    toe = np.asarray(row['provisionalCommonToeFlexZoneWorld'], dtype=float)
    malleoli = np.asarray(row['malleolarSurfaceAxisWorld'], dtype=float)
    ankle_axis = unit(malleoli[1] - malleoli[0])
    if ankle_axis[0] < 0:
        ankle_axis *= -1
    toe_axis = unit(roots[-1] - roots[0])
    if toe_axis[0] < 0:
        toe_axis *= -1
    # Actual distal surface centroid determines a trial bone endpoint. It is
    # not asserted to be a distal phalanx centre or a multi-toe anatomical rig.
    co = candidate['legFoot_' + side + '_positions']
    central_x = np.sort(roots[1:3, 0])
    distal = co[(co[:, 1] < toe[1] - .02) & (co[:, 2] < .08)
                & (co[:, 0] >= central_x[0] - .004) & (co[:, 0] <= central_x[1] + .004)]
    if len(distal) < 10:
        raise RuntimeError('Actual distal toe surface missing')
    tip = distal[distal[:, 1] < np.quantile(distal[:, 1], .15)].mean(axis=0)
    if tip[1] >= toe[1] - .01:
        raise RuntimeError('Trial toe endpoint does not extend forward')
    forward = tip - ankle
    dorsal_dz = np.cross(ankle_axis, forward)[2] * -1
    if dorsal_dz <= 0:
        raise RuntimeError('Dorsiflexion sign does not lift the distal surface')
    return {
        'ankleCentreWorld': ankle.tolist(), 'toeRootProxyWorld': toe.tolist(),
        'toeDistalSurfaceEndpointWorld': tip.tolist(),
        'toeEndpointSelection': 'Distal surface of the central two actual toe-root X lanes; excludes the longest big-toe endpoint bias.',
        'ankleAxisWorld': ankle_axis.tolist(), 'toeAxisWorld': toe_axis.tolist(),
        'kneeTrialAxisWorld': [1.0, 0.0, 0.0],
        'ankleDeltaFromSourceHeadWorld': row['surfaceCenterMinusOldRigHead'],
        'bareSoleDatumWorldZ': row['targetBareFootMinimumZ'],
        'sourceSurfaceMalleoliWorld': row['malleolarSurfaceAxisWorld'],
        'actualToeRootLoopCentresWorld': row['toeRootCentersWorld'],
        'functionalJointCentreCertified': False,
        'meaning': 'Finite FK surface-axis experiment; knee uses world X. Common toe proxy does not validate individual toe joints. Bare sole datum is not the floor or shoe outsole.',
    }


def load_inputs():
    for relative, digest in PINS.items():
        if sha(path_for(relative)) != digest:
            raise RuntimeError('Frozen dependency changed: ' + relative)
    candidate = json.loads(path_for(STAGE + 'foot-leg-unified-candidate-r11-v2.json').read_text(encoding='utf8'))
    capture = json.loads(path_for(STAGE + 'foot-leg-join-capture-r11.json').read_text(encoding='utf8'))
    plan = json.loads(path_for(STAGE + 'foot-225-placement-plan-r11.json').read_text(encoding='utf8'))
    for row in (candidate, capture, plan):
        array_relative = row['arrays'].replace('\\', '/')
        if array_relative not in PINS or row['arraysSha256'] != PINS[array_relative]:
            raise RuntimeError('JSON/NPZ dependency identity differs')
    with np.load(path_for(candidate['arrays']), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    with np.load(path_for(capture['arrays']), allow_pickle=False) as loaded:
        captured = {key: loaded[key] for key in loaded.files}
    rows = {row['name']: row for row in capture['legObjects']}
    validation = {}
    for side, word in (('L', 'left'), ('R', 'right')):
        prefix = 'legFoot_' + side
        old = 'leg_' + side
        row = rows['Leg ' + side + ' · knee calf ankle']
        co, loops = arrays[prefix + '_positions'], arrays[prefix + '_loops']
        starts, sizes = arrays[prefix + '_polygonStarts'], arrays[prefix + '_polygonSizes']
        weights = arrays[prefix + '_weights']
        if co.shape != (10239, 3) or len(starts) != 10314 or weights.shape != (len(co), 4):
            raise RuntimeError('Unexpected frozen candidate dimensions')
        if not np.all(np.isfinite(co)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise RuntimeError('Nonfinite or negative candidate data')
        if np.max(abs(weights.sum(axis=1) - 1)) > 2e-6:
            raise RuntimeError('Candidate weight normalization changed')
        if np.any(sizes < 3) or not np.array_equal(starts, np.r_[0, np.cumsum(sizes[:-1])]) or int(sizes.sum()) != len(loops):
            raise RuntimeError('Invalid candidate polygon layout')
        if loops.min() < 0 or loops.max() >= len(co):
            raise RuntimeError('Invalid candidate vertex reference')
        original_ids, candidate_ids = arrays[prefix + '_legOriginalIds'], arrays[prefix + '_legCandidateIds']
        source_world = world_points(captured[old + '_raw_positions'], row['matrixWorld'])
        if not np.array_equal(co[candidate_ids], source_world[original_ids]):
            raise RuntimeError('Retained original upper leg is not exact')
        source_weights = np.zeros((len(source_world), 2))
        source_weights[captured[old + '_weight_vertex'], captured[old + '_weight_group']] = captured[old + '_weight_value']
        if not np.array_equal(weights[candidate_ids, :2], source_weights[original_ids]) or np.any(weights[candidate_ids, 2:] != 0):
            raise RuntimeError('Retained original leg weights are not exact')
        original_by_candidate = np.full(len(co), -1, dtype=np.int32)
        original_by_candidate[candidate_ids] = original_ids
        source_loops = captured[old + '_raw_loopVertex']
        source_starts, source_sizes = captured[old + '_raw_polygonStart'], captured[old + '_raw_polygonSize']
        source_uv = captured[old + '_raw_uv_0']
        source_faces = {tuple(source_loops[s:s + n]): source_uv[s:s + n]
                        for s, n in zip(source_starts, source_sizes)}
        preserved_faces = np.flatnonzero(arrays[prefix + '_faceRegion'] == 2)
        for face in preserved_faces:
            start, size = starts[face], sizes[face]
            ids = tuple(original_by_candidate[loops[start:start + size]])
            if ids not in source_faces or not np.array_equal(arrays[prefix + '_uvSource'][start:start + size], source_faces[ids]):
                raise RuntimeError('Retained original leg face/corner UV changed')
        validation[side] = {'retainedOriginalRawVerticesExact': len(original_ids),
                            'retainedOriginalRawWeightsExact': True,
                            'retainedOriginalFaceCornerUVsExact': len(preserved_faces),
                            'groups': [word + name for name in ('LowerLeg', 'UpperLeg', 'Foot', 'Toes')],
                            'axes': axis_plan(side, plan, arrays)}
    return candidate, capture, plan, arrays, captured, validation


def source_checks(bpy, api, capture, captured):
    rig = bpy.data.objects.get(capture['rigCurrentPose']['name'])
    if rig is None or rig.type != 'ARMATURE':
        raise RuntimeError('Exact captured source rig missing')
    if not np.array_equal(np.asarray(rig.matrix_world), capture['rigCurrentPose']['matrixWorld']):
        raise RuntimeError('Source rig transform changed; recapture required')
    expected_bones = {r['name']: r for r in capture['rigCurrentPose']['bones']}
    # All leg-chain rest transforms are needed, not only the foot bone.
    for name in ('hips', 'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg', 'rightLowerLeg', 'leftFoot', 'rightFoot', 'leftToes', 'rightToes'):
        old = expected_bones[name]
        bone = rig.data.bones.get(name)
        if bone is None or not np.array_equal(np.asarray(bone.matrix_local), old['restMatrixLocal']):
            raise RuntimeError('Source lower-limb rest contract changed: ' + name)
        if (bone.parent.name if bone.parent else None) != old['parent'] or bone.use_connect != old['useConnect']:
            raise RuntimeError('Source lower-limb hierarchy changed: ' + name)
        if bone.use_deform != old['useDeform'] or bone.inherit_scale != old['inheritScale']:
            raise RuntimeError('Source lower-limb deformation settings changed: ' + name)
    legs = {}
    for side, row in zip(('L', 'R'), capture['legObjects']):
        if row['name'] != 'Leg ' + side + ' · knee calf ankle':
            raise RuntimeError('Captured leg ordering changed')
        obj = bpy.data.objects.get(row['name'])
        if obj is None or obj.type != 'MESH' or obj.data.shape_keys:
            raise RuntimeError('Captured source Leg missing or newly keyed')
        if not np.array_equal(np.asarray(obj.matrix_world), row['matrixWorld']):
            raise RuntimeError('Source Leg transform changed')
        actual = {}
        api['_mesh'](obj.data, 'actual', actual)
        for key in ('positions', 'loopVertex', 'polygonStart', 'polygonSize', 'uv_0', 'materialIndex', 'smoothFace'):
            if not np.array_equal(actual['actual_' + key], captured['leg_' + side + '_raw_' + key]):
                raise RuntimeError('Source Leg mesh/UV changed: ' + side + '/' + key)
        groups = [{'name': g.name, 'index': g.index, 'lockWeight': g.lock_weight} for g in obj.vertex_groups]
        if groups != row['vertexGroups'] or [api['_rna'](m) for m in obj.modifiers] != row['modifiers']:
            raise RuntimeError('Source Leg groups/modifier order/settings changed')
        weights = [(v.index, g.group, g.weight) for v in obj.data.vertices for g in v.groups]
        for column, key in enumerate(('vertex', 'group', 'value')):
            if not np.array_equal(np.asarray([x[column] for x in weights]), captured['leg_' + side + '_weight_' + key]):
                raise RuntimeError('Source Leg weights changed')
        if [m.type for m in obj.modifiers] != ['ARMATURE', 'SUBSURF'] or obj.modifiers[0].object != rig:
            raise RuntimeError('Required native Armature then Subdivision stack missing')
        legs[side] = obj
    return rig, legs


def copy_trial_rig(bpy, source, axes):
    from mathutils import Matrix, Vector
    copied = source.copy()
    copied.data = source.data.copy()
    copied.name = 'R11_NATIVE_LEG_TRIAL_RIG'
    copied.data.name = 'R11_NATIVE_LEG_TRIAL_ARMATURE'
    bpy.context.scene.collection.objects.link(copied)
    world = source.matrix_world.copy()
    copied.parent = None
    copied.matrix_world = world
    copied.animation_data_clear()
    copied.data.animation_data_clear()
    for constraint in list(copied.constraints):
        copied.constraints.remove(constraint)
    removed_constraints = 0
    for bone in copied.pose.bones:
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
            removed_constraints += 1
        bone.matrix_basis = Matrix.Identity(4)
    copied.hide_viewport = False
    copied.hide_set(False)
    bpy.ops.object.select_all(action='DESELECT')
    copied.select_set(True)
    bpy.context.view_layer.objects.active = copied
    before = {b.name: np.asarray(b.matrix_local).copy() for b in copied.data.bones}
    before_upper = {n: (tuple(copied.data.bones[n].head_local), tuple(copied.data.bones[n].tail_local))
                    for n in ('leftUpperLeg', 'rightUpperLeg')}
    inv = copied.matrix_world.inverted()
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        for side, word in (('L', 'left'), ('R', 'right')):
            a = axes[side]['axes']
            lower, foot, toes = [copied.data.edit_bones[word + n] for n in ('LowerLeg', 'Foot', 'Toes')]
            # Disconnect the copied children during editing, then restore the
            # original connected hierarchy at the coincident candidate joints.
            connected = (foot.use_connect, toes.use_connect)
            foot.use_connect = False
            toes.use_connect = False
            ankle = inv @ Vector(a['ankleCentreWorld'])
            toe = inv @ Vector(a['toeRootProxyWorld'])
            tip = inv @ Vector(a['toeDistalSurfaceEndpointWorld'])
            lower.tail = ankle
            foot.head, foot.tail = ankle, toe
            toes.head, toes.tail = toe, tip
            foot.use_connect, toes.use_connect = connected
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    changed = []
    for bone in copied.data.bones:
        if not np.allclose(np.asarray(bone.matrix_local), before[bone.name], atol=1e-7, rtol=0):
            changed.append(bone.name)
    permitted = {word + n for word in ('left', 'right') for n in ('LowerLeg', 'Foot', 'Toes')}
    if not set(changed) <= permitted:
        raise RuntimeError('Unrelated copied rest bones changed: ' + repr(set(changed) - permitted))
    for name, endpoints in before_upper.items():
        bone = copied.data.bones[name]
        if (tuple(bone.head_local), tuple(bone.tail_local)) != endpoints:
            raise RuntimeError('Copied upper-leg rest endpoints changed')
    copied.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    return copied, {'removedCopiedPoseConstraints': removed_constraints, 'changedCopiedRestBones': changed,
                    'copiedActionAndDriversCleared': copied.animation_data is None,
                    'sourceActionNotRetargeted': True, 'productionControlAdaptationValidated': False}


def candidate_object(bpy, source, rig, side, arrays, validation, api):
    from mathutils import Matrix
    prefix = 'legFoot_' + side
    co, flat = arrays[prefix + '_positions'], arrays[prefix + '_loops']
    starts, sizes = arrays[prefix + '_polygonStarts'], arrays[prefix + '_polygonSizes']
    faces = [flat[start:start + size].tolist() for start, size in zip(starts, sizes)]
    mesh = bpy.data.meshes.new('R11_NATIVE_LEG_TRIAL_MESH_' + side)
    mesh.from_pydata(co.tolist(), [], faces)
    mesh.update()
    if mesh.validate(verbose=False, clean_customdata=False):
        raise RuntimeError('Candidate mesh required repair; frozen topology must be investigated')
    mesh.polygons.foreach_set('use_smooth', np.ones(len(faces), dtype=np.bool_))
    mesh.polygons.foreach_set('material_index', arrays[prefix + '_materialIndex'])
    layer = mesh.uv_layers.new(name='SourceUV_UnpackedTrial')
    layer.data.foreach_set('uv', arrays[prefix + '_uvSource'].astype(np.float32).ravel())
    # Materials are carried for data continuity only: neither nails nor final
    # texture packing are accepted by this geometry-only, non-rendering trial.
    for material in source.data.materials:
        mesh.materials.append(material)
    while len(mesh.materials) < 2 and len(mesh.materials):
        mesh.materials.append(mesh.materials[0])
    obj = source.copy()
    obj.data = mesh
    obj.name = 'R11_NATIVE_LEG_TRIAL_' + side
    bpy.context.scene.collection.objects.link(obj)
    obj.animation_data_clear()
    obj.parent = None
    obj.matrix_world = Matrix.Identity(4)
    for constraint in list(obj.constraints):
        obj.constraints.remove(constraint)
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.vertex_groups.clear()
    weights = arrays[prefix + '_weights']
    for col, name in enumerate(validation[side]['groups']):
        group = obj.vertex_groups.new(name=name)
        ids = np.flatnonzero(weights[:, col] > 0)
        # Group exact repeated values to retain upper-leg source weights.
        for value in np.unique(weights[ids, col]):
            group.add(ids[weights[ids, col] == value].tolist(), float(value), 'REPLACE')
    obj.modifiers[0].object = rig
    old = [api['_rna'](m) for m in source.modifiers]
    new = [api['_rna'](m) for m in obj.modifiers]
    new[0]['object'] = old[0]['object']
    # Read-only identity is allowed to differ when Blender copies a modifier;
    # every captured setting and the stack order still must match exactly.
    for values in old + new:
        values.pop('persistent_uid', None)
    if new != old:
        raise RuntimeError('Copied native modifier settings changed beyond armature target')
    readback = np.empty(co.shape, dtype=np.float32)
    mesh.vertices.foreach_get('co', readback.ravel())
    if not np.array_equal(readback, co.astype(np.float32)):
        raise RuntimeError('Candidate coordinate readback differs')
    uv_readback = np.empty(arrays[prefix + '_uvSource'].shape, dtype=np.float32)
    layer.data.foreach_get('uv', uv_readback.ravel())
    if not np.array_equal(uv_readback, arrays[prefix + '_uvSource'].astype(np.float32)):
        raise RuntimeError('Candidate source UV readback differs')
    actual_weights = np.zeros(weights.shape, dtype=np.float32)
    for vertex in mesh.vertices:
        for group in vertex.groups:
            actual_weights[vertex.index, group.group] = group.weight
    if not np.array_equal(actual_weights, weights.astype(np.float32)):
        raise RuntimeError('Actual candidate weight readback differs')
    return obj


def surface(bpy, obj):
    graph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(graph)
    mesh = ev.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        mesh.calc_loop_triangles()
        if len(mesh.vertices) > 100000 or len(mesh.loop_triangles) > 200000:
            raise RuntimeError('Native trial mesh exceeds bounded scope')
        vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get('co', vertices.ravel())
        triangles = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles.ravel())
        return world_points(vertices, ev.matrix_world), triangles
    finally:
        ev.to_mesh_clear()


def local_candidates(points, triangles, roi, limit=2000000):
    from mathutils.bvhtree import BVHTree
    local = triangles[roi]
    tree = BVHTree.FromPolygons(points.tolist(), local.tolist(), all_triangles=True, epsilon=1e-6)
    overlaps = tree.overlap(tree)
    if len(overlaps) > limit:
        raise RuntimeError('Local BVH self-overlap budget exceeded; narrow capture in a separate trial')
    pairs = np.asarray(overlaps, dtype=np.int32).reshape(-1, 2)
    raw_count = len(pairs)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if len(pairs):
        pairs = np.unique(pairs, axis=0)
        shared = np.any(local[pairs[:, 0], :, None] == local[pairs[:, 1], None, :], axis=(1, 2))
        adjacency = int(shared.sum())
        pairs = pairs[~shared]
    else:
        adjacency = 0
    return roi[pairs], {'rawBvhPairs': raw_count, 'sharedVertexPairsExcluded': adjacency,
                        'nonAdjacentCandidatePairs': len(pairs), 'epsilonMeters': 1e-6,
                        'confirmedIntersections': None, 'continuousContactChecked': False}


def pose_case(bpy, rig, case, validation):
    from mathutils import Matrix, Quaternion, Vector
    label, suffix, degrees = case
    rig.data.pose_position = 'REST' if label == 'REST' else 'POSE'
    for bone in rig.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    if suffix:
        for side, word in (('L', 'left'), ('R', 'right')):
            record = validation[side]['axes']
            key = {'Foot': 'ankleAxisWorld', 'Toes': 'toeAxisWorld', 'LowerLeg': 'kneeTrialAxisWorld'}[suffix]
            bone = rig.pose.bones[word + suffix]
            # Convert a rest-world axis into this bone's local basis. In these
            # isolated cases all ancestors are neutral, so no posed-parent
            # conversion or action-derived matrix is silently reused.
            axis = bone.bone.matrix_local.to_3x3().inverted() @ rig.matrix_world.to_3x3().inverted() @ Vector(record[key])
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion(axis.normalized(), math.radians(degrees))
    rig.update_tag(refresh={'OBJECT'})
    bpy.context.view_layer.update()
    if rig.animation_data is not None or any(b.constraints for b in rig.pose.bones):
        raise RuntimeError('Isolated trial unexpectedly retained animation or constraints')


def extra_rig_contract(rig):
    return {'world': [list(r) for r in rig.matrix_world], 'posePosition': rig.data.pose_position,
            'dataAnimation': bool(rig.data.animation_data),
            'bones': [{'name': b.name, 'head': list(b.head_local), 'tail': list(b.tail_local),
                       'connect': b.use_connect, 'deform': b.use_deform,
                       'pose': [list(r) for r in rig.pose.bones[b.name].matrix]}
                      for b in rig.data.bones]}


def run(source_relative, source_sha, output_relative):
    import bpy
    source, output = path_for(source_relative), path_for(output_relative)
    output_arrays = output.with_suffix('.npz')
    if output.suffix != '.json' or output.exists() or output_arrays.exists():
        raise FileExistsError('Use a new JSON/NPZ output pair; preserve previous probes')
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Exact source in disposable background Object Mode required')
    if len(source_sha) != 64 or sha(source) != source_sha:
        raise RuntimeError('Explicit current source SHA-256 mismatch')
    if bpy.context.scene.frame_current != 1 or bpy.context.scene.frame_subframe != 0:
        raise RuntimeError('This trial requires the unmodified source frame one')
    candidate, capture, plan, data, captured, validation = load_inputs()
    api = runpy.run_path(str(path_for('work/local-model/hand_asset_observe_r10.py')))
    source_rig, source_legs = source_checks(bpy, api, capture, captured)
    guarded = [source_rig] + list(source_legs.values())
    contract = api['_source_contract'](guarded)
    extra = extra_rig_contract(source_rig)
    old_ids = set(api['_ids'](bpy))
    active = bpy.context.view_layer.objects.active
    selected = list(bpy.context.selected_objects)
    scene = bpy.context.scene
    scene_contract = (scene.frame_current, scene.frame_subframe, scene.camera, scene.render.filepath)
    arrays, result = {}, {
        'schemaVersion': 1, 'mode': 'actual-native-isolated-FK-trial', 'source': source_relative,
        'sourceSha256': source_sha, 'helperSha256': sha(__file__), 'dependencies': PINS,
        'candidateGeometrySource': candidate['source'], 'candidateGeometrySourceSha256': candidate['sourceSha256'],
        'sourceAction': api['_animation'](source_rig), 'sides': validation, 'cases': [],
        'savedBlend': False, 'sourceGeometryEdited': False, 'productionRigInstalled': False,
        'productionActionsRetargeted': False, 'appearanceAccepted': False, 'deliveryReady': False,
        'limitations': ['Finite isolated FK snapshots only; no between-frame collision proof.',
                       'Surface malleoli and toe-root loops are provisional joint proxies.',
                       'New Foot/Toes weights are candidate values; source upper-leg raw weights retained.',
                       'Native Subdivision additionally smooths the donor evaluated base; old donor evaluation is not identical.',
                       'BVH pairs require separate continuous triangle-pair analysis; no zero-intersection claim is made here.',
                       'No shoes/socks, floor constraints, IK controls, old action retargeting or final UV/material validation.'],
    }
    try:
        rig, result['copiedRig'] = copy_trial_rig(bpy, source_rig, validation)
        objects = {side: candidate_object(bpy, source_legs[side], rig, side, data, validation, api) for side in ('L', 'R')}
        rest = {}
        for case in CASE_SPECS:
            pose_case(bpy, rig, case, validation)
            case_row = {'name': case[0], 'rotationDegrees': case[2], 'boneSuffix': case[1], 'sides': {}}
            for side, word in (('L', 'left'), ('R', 'right')):
                points, triangles = surface(bpy, objects[side])
                prefix = case[0] + '_' + side
                if case[0] == 'REST':
                    roi = np.flatnonzero(np.min(points[triangles, 2], axis=1) < .355).astype(np.int32)
                    rest[side] = (points, triangles, roi)
                    arrays['triangles_' + side] = triangles
                    arrays['roiTriangleIds_' + side] = roi
                base, base_triangles, roi = rest[side]
                if points.shape != base.shape or not np.array_equal(triangles, base_triangles):
                    raise RuntimeError('Native topology changed between finite pose states')
                if not np.all(np.isfinite(points)):
                    raise RuntimeError('Native evaluated coordinates are not finite')
                if case[0] == 'neutral_fk' and np.max(np.linalg.norm(points - base, axis=1)) > 2e-6:
                    raise RuntimeError('Neutral FK differs from native REST beyond float tolerance')
                pairs, collision = local_candidates(points, triangles, roi)
                arrays[prefix + '_world'] = points
                arrays[prefix + '_candidatePairs'] = pairs
                mesh_tri = points[triangles[roi]]
                old_tri = base[triangles[roi]]
                areas = np.linalg.norm(np.cross(mesh_tri[:, 1] - mesh_tri[:, 0], mesh_tri[:, 2] - mesh_tri[:, 0]), axis=1)
                old_areas = np.linalg.norm(np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0]), axis=1)
                ratios = areas[old_areas > 1e-16] / old_areas[old_areas > 1e-16]
                pose_bones = {}
                for suffix in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes'):
                    pb = rig.pose.bones[word + suffix]
                    pose_bones[word + suffix] = {'headWorld': list(rig.matrix_world @ pb.head),
                                                'tailWorld': list(rig.matrix_world @ pb.tail),
                                                'matrixBasis': [list(r) for r in pb.matrix_basis],
                                                'poseMatrixRigSpace': [list(r) for r in pb.matrix]}
                case_row['sides'][side] = {
                    'arrayPrefix': prefix, 'vertices': len(points), 'triangles': len(triangles),
                    'roiTriangles': len(roi), 'roiRule': 'Fixed REST triangles with any vertex below world Z .355 m, including foot, bridge and join margin.',
                    'boundsWorld': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                    'minimumZMinusBareSoleDatum': float(points[:, 2].min() - validation[side]['axes']['bareSoleDatumWorldZ']),
                    'minimumZIsNotAContactOrGroundValidation': True,
                    'roiDegenerateTriangles': int(np.count_nonzero(areas < 1e-16)),
                    'roiAreaRatioToRestQuantiles': np.quantile(ratios, [0, .01, .5, .99, 1]).tolist(),
                    'localSurfaceCandidates': collision, 'actualPoseBones': pose_bones,
                }
            result['cases'].append(case_row)
            print('R11_NATIVE_LEG_TRIAL_CAPTURE ' + case[0], flush=True)
    finally:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        added = [item for pointer, item in api['_ids'](bpy).items() if pointer not in old_ids]
        if added:
            bpy.data.batch_remove(ids=added)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active
        bpy.context.view_layer.update()
        if (api['_source_contract'](guarded) != contract or extra_rig_contract(source_rig) != extra
                or set(api['_ids'](bpy)) != old_ids or sha(source) != source_sha
                or (scene.frame_current, scene.frame_subframe, scene.camera, scene.render.filepath) != scene_contract
                or bpy.context.mode != 'OBJECT' or set(bpy.context.selected_objects) != set(selected)
                or bpy.context.view_layer.objects.active != active):
            raise RuntimeError('Native trial failed complete source/session restoration')
    result['restorationExact'] = True
    result['sourceContractBeforeAndAfter'] = contract
    output.parent.mkdir(parents=True, exist_ok=True)
    with output_arrays.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result['arrays'] = output_arrays.relative_to(project_path()).as_posix()
    result['arraysSha256'] = sha(output_arrays)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'report': output.relative_to(project_path()).as_posix(), 'reportSha256': sha(output),
            'arraysSha256': result['arraysSha256'], 'savedBlend': False, 'restorationExact': True}


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--source-sha')
    parser.add_argument('--output', default=STAGE + 'foot-leg-native-trial-r11.json')
    parser.add_argument('--offline-check', action='store_true')
    args = parser.parse_args(argv)
    if args.offline_check:
        _, _, _, _, _, validation = load_inputs()
        print(json.dumps({'helperSha256': sha(__file__), 'offlineDependencyAndPayloadCheck': True,
                          'blenderExecuted': False, 'nativeTrialValidated': False, 'sides': validation}, ensure_ascii=False, indent=2))
    else:
        if not args.source or not args.source_sha:
            parser.error('--source and --source-sha are required for an actual trial')
        print('R11_NATIVE_LEG_TRIAL ' + json.dumps(run(args.source, args.source_sha, args.output)), flush=True)


if __name__ == '__main__':
    main()
