"""Reversible R11 head/neck conversion after a final static-source refresh.

Requires a newly observed exact input Blend and pinned anatomical weight plan.
Supports gaze schema 3 affine-driver stages, plus retained schema 2 records.
Adds explicit native sclera-raycast space migration before cervical skinning.
V7 additionally adapts the reviewed radial outer-corner before/after eye rays.
The retained v6 geometry, weighting, fibre preflight and rollback logic is unchanged.
All current expression keys, groom, jaw attachments and driver definitions are
captured from the input being executed. No asset is restored from an older Blend.

No mesh/key datablock is replaced; existing shape-key driver IDs stay valid.
Only the integrator runs this in a fresh background process and saves a new copy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
import bpy
from mathutils import Matrix

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
BRIDGE = 'R11 Continuous neck bridge'
BODY = 'R01 Head'
RIG = 'Character_Rig'
CARRIER = 'R11 Cranial attachment space'
MODIFIER = 'R11 anatomical cervical skin'
MARKER = 'R11_head_neck_skin_bind'
EYE_SPACE_SHA = '5c22eaf1fd19b6574c189299b413c8eb8a325f407fe550fa9c6f5b2b13fe6eda'
GAZE_HELPERS = {
    2: ('adult_gaze_control_r11_v2', '7d064176ec7fffc9c0126f149b67a72e4a6396bc3ddc4eb74b80b9515c66f55d'),
    3: ('adult_gaze_control_r11_v3', '29eae16db0f046f2fac2f80790879fc1684a5750c97d481090cea25baf1b3493')}


FIBRE_PINS = {
    'face_fibre_binding_r11_v2.py': '47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49',
    'upper_lash_corrective_validation_r11.py': '36a8713e93a962504b6f1546fb466afd70e3a7af394c449b40176b32999e3c0a',
}


def api(name):
    if name == 'face_fibre_binding_r11':
        for filename, expected in FIBRE_PINS.items():
            if sha(Path(__file__).with_name(filename)) != expected:
                raise RuntimeError('Frozen combined fibre validation dependency changed: ' + filename)
        name = 'face_fibre_binding_r11_v2'
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def path(relative, suffix=None):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Project-relative path required')
    result = (project_path() / relative).resolve()
    result.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if suffix and result.suffix != suffix:
        raise ValueError('Unexpected suffix')
    return result


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in ('pointer', 'is_active')}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def matrix(m):
    return [list(row) for row in m]


def attachment(obj):
    return {'parent': obj.parent.name if obj.parent else None, 'parentType': obj.parent_type,
            'parentBone': obj.parent_bone, 'inverse': matrix(obj.matrix_parent_inverse),
            'basis': matrix(obj.matrix_basis), 'rotationMode': obj.rotation_mode}


def assign_attachment(obj, record):
    obj.parent = bpy.data.objects.get(record['parent']) if record['parent'] else None
    obj.parent_type = record['parentType']
    obj.parent_bone = record['parentBone']
    obj.rotation_mode = record['rotationMode']
    obj.matrix_parent_inverse = Matrix(record['inverse'])
    obj.matrix_basis = Matrix(record['basis'])


def group_record(obj):
    return [(g.name, g.index, bool(g.lock_weight)) for g in obj.vertex_groups]


def set_weights(obj, names, values):
    values = np.asarray(values, np.float32)
    if values.shape != (len(obj.data.vertices), len(names)):
        raise RuntimeError('Actual skin matrix dimensions differ')
    ids = list(range(len(obj.data.vertices)))
    for column, name in enumerate(names):
        group = obj.vertex_groups.get(name)
        if group is None:
            group = obj.vertex_groups.new(name=name)
        group.remove(ids)
        weight = values[:, column]
        full = np.flatnonzero(weight == 1.)
        if len(full):
            group.add(full.tolist(), 1., 'REPLACE')
        for i in np.flatnonzero((weight > 0.) & (weight != 1.)):
            group.add([int(i)], float(weight[i]), 'REPLACE')


def protected_geometry(obj):
    record = api('adult_head_bind_observe_r11')['data_contract'](obj)
    for key in ('weights', 'groups', 'modifiers'):
        record.pop(key)
    return digest(record)


def cranial_data_contract(names):
    """All current descendants, including gaze/jaw Empty driver definitions."""
    groom = api('posterior_groom_r09')
    uv = api('production_uv_r08')
    rows = {}
    for name in names:
        obj = bpy.data.objects[name]
        animation = uv['_animation'](obj)
        driven = {c.data_path for c in obj.animation_data.drivers} if obj.animation_data else set()
        custom = {k: v for k, v in uv['_custom'](obj).items() if '["'+k+'"]' not in driven}
        # Driver outputs are derived cached values, not authored definitions.
        # Preserve non-driven controls, all driver equations and ID relations.
        rows[name] = {'type': obj.type, 'animation': animation, 'custom': custom,
            'constraints': [{'name': c.name, 'type': c.type, 'settings': uv['_scalar_rna'](c),
                'pointers': {p.identifier: uv['_plain'](getattr(c, p.identifier))
                    for p in c.bl_rna.properties if p.type == 'POINTER' and p.identifier != 'rna_type'}}
                for c in obj.constraints],
            'driverTargetTypes': [[c.data_path, v.name, t.id_type,
                getattr(getattr(t, 'id', None), 'name', None)]
                for c in obj.animation_data.drivers for v in c.driver.variables for t in v.targets]
                if obj.animation_data else []}
        if obj.type not in ('MESH', 'CURVES'):
            continue
        keys = getattr(obj.data, 'shape_keys', None)
        rows[name].update({'data': obj.data.name, 'geometryAttributes': groom['_data_hash'](obj.data),
            'materials': [m.name if m else None for m in obj.data.materials],
            'keys': [(k.name, api('adult_head_bind_observe_r11')['ah'](
                api('adult_head_bind_observe_r11')['array'](k.data, 'co'))) for k in keys.key_blocks] if keys else [],
            'keyAnimation': uv['_animation'](keys) if keys else None,
            'modifiers': [(m.name, m.type, clean(uv['_scalar_rna'](m)),
                           getattr(getattr(m, 'node_group', None), 'name', None)) for m in obj.modifiers]})
    return digest(rows)


def gaze_api(schema):
    if schema not in GAZE_HELPERS:
        raise RuntimeError('Unsupported current gaze schema; preserve its installation')
    helper, expected = GAZE_HELPERS[schema]
    if sha(Path(__file__).with_name(helper + '.py')) != expected:
        raise RuntimeError('Gaze compatibility helper changed; review before migrating')
    return api(helper), {'schema': schema, 'helper': helper, 'helperSha256': expected}


def eye_space_api():
    if sha(Path(__file__).with_name('adult_prearm_eye_space_r11_v4.py')) != EYE_SPACE_SHA:
        raise RuntimeError('Native pre-Armature eye-space helper changed; review before migrating')
    return api('adult_prearm_eye_space_r11_v4')


def rig_definition(rig):
    uv = api('production_uv_r08')
    return digest({'matrix': matrix(rig.matrix_world), 'animation': uv['_animation'](rig),
        'bones': {b.name: {'rest': matrix(b.matrix_local), 'head': list(b.head_local),
            'tail': list(b.tail_local), 'deform': b.use_deform,
            'parent': b.parent.name if b.parent else None} for b in rig.data.bones},
        'constraints': {b.name: [{'name': c.name, 'type': c.type, 'settings': uv['_scalar_rna'](c)}
                               for c in b.constraints] for b in rig.pose.bones}})


def candidate_contract(record):
    obs = api('adult_head_bind_observe_r11')
    names = [HEAD, BRIDGE]
    return digest({'meshes': {n: obs['data_contract'](bpy.data.objects[n]) for n in names},
        'attachments': {n: attachment(bpy.data.objects[n]) for n in [HEAD, CARRIER] + record['directChildren']},
        'carrierTag': bpy.data.objects[CARRIER].get('R11_original_head_space_source'),
        'carrierDirectChildren': sorted(o.name for o in bpy.data.objects[CARRIER].children),
        'currentCranialDescendantContract': cranial_data_contract(record['allCranialDescendants']),
        'gazeRecord': bpy.context.scene.get('R11_gaze_control'),
        'gazeMigrationRecord': bpy.context.scene.get('R11_gaze_space_upgrade'),
        'prearmatureEyeSpace': eye_space_api()['state'](record['prearmatureEyeMigration']) if record.get('prearmatureEyeMigration') else None,
        'rig': rig_definition(bpy.data.objects[RIG])})


def _restore(record, remove_marker=True):
    # Resolve/pin the migration dependency before any scene mutation. A later
    # helper edit must not leave restored geometry with a stale gaze marker.
    rollback_gaze = gaze_api(record['gazeCompatibility']['schema'])[0] if record.get('gazeSpaceUpgraded') else None
    rollback_eye = eye_space_api() if record.get('prearmatureEyeMigration') else None
    if rollback_eye is not None:
        rollback_eye['validate_restore'](record['prearmatureEyeMigration'])
    # Refuse later attachment edits before the first destructive restoration.
    # This also applies to automatic rollback of a partially completed install.
    carrier = bpy.data.objects.get(CARRIER)
    expected_children = set(record['originalChildAttachments'])
    if carrier is not None and any(o.name not in expected_children for o in carrier.children):
        raise RuntimeError('Carrier has unrecorded children; preserve subsequent attachment edits')
    missing = expected_children - set(bpy.data.objects.keys())
    if missing:
        raise RuntimeError('Recorded cranial children are missing; preserve subsequent edits: ' + str(sorted(missing)))
    head = bpy.data.objects[HEAD]
    bridge = bpy.data.objects[BRIDGE]
    modifier = head.modifiers.get(MODIFIER)
    if modifier is not None:
        head.modifiers.remove(modifier)
    for name in reversed(record['addedHeadGroups']):
        group = head.vertex_groups.get(name)
        if group is not None:
            head.vertex_groups.remove(group)
    set_weights(bridge, record['bridgeGroupNames'], record['originalBridgeWeights'])
    assign_attachment(head, record['originalHeadAttachment'])
    for name, item in record['originalChildAttachments'].items():
        assign_attachment(bpy.data.objects[name], item)
    if rollback_eye is not None:
        rollback_eye['restore'](record['prearmatureEyeMigration'])
    if carrier is not None:
        if carrier.children:
            raise RuntimeError('Carrier has unrecorded children; preserve subsequent attachment edits')
        bpy.data.objects.remove(carrier, do_unlink=True)
    bpy.context.view_layer.update()
    if rollback_gaze is not None:
        rollback_gaze['rollback_space_record']()
    if remove_marker and MARKER in bpy.context.scene:
        del bpy.context.scene[MARKER]
    bpy.context.view_layer.update()


def _pose_preservation(rig, children, neutral_matrix, expected=None, head_reference=None, rigid_ids=None, prearmature_record=None):
    motion = api('adult_motion_review_r11')
    observer = api('adult_head_bind_observe_r11')
    rows = []
    for label, yaw, pitch in [('neutral', 0., 0.), ('turn-left', 30., 0.), ('look-up', 0., 15.)]:
        motion['_neck_pose'](rig, neutral_matrix, yaw, pitch)
        row = {'label': label, 'matrices': {n: matrix(bpy.data.objects[n].matrix_world) for n in children}}
        if expected is not None:
            old = next(r for r in expected if r['label'] == label)
            errors = {n: float(np.max(abs(np.asarray(row['matrices'][n]) - old['matrices'][n]))) for n in children}
            if max(errors.values(), default=0.) > 3e-6:
                raise RuntimeError('Cranial child matrix changed under head articulation: ' + json.dumps(errors))
            row['maximumAttachmentMatrixError'] = max(errors.values(), default=0.)
        h = api('adult_motion_review_r11')['_evaluated'](HEAD)
        row['headWorld'] = h['world']
        if head_reference is not None and rigid_ids is not None:
            reference = next(r for r in head_reference if r['label'] == label)['headWorld']
            error = float(np.linalg.norm(h['world'][rigid_ids] - reference[rigid_ids], axis=1).max())
            if error > 3e-6:
                raise RuntimeError('Rigid cranial surface changed from original head motion: ' + str(error))
            row['rigidCranialSurfaceErrorMeters'] = error
            row['actualFibreValidation'] = api('face_fibre_binding_r11')['validate'](neutral=False)
        if prearmature_record is not None:
            row['actualPreArmatureEyeMapping'] = eye_space_api()['probe_current_mapping'](prearmature_record)
        rows.append(row)
    return rows


def apply(source_relative, plan_relative):
    source = path(source_relative, '.blend')
    plan_path = path(plan_relative, '.json')
    scene = bpy.context.scene
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh exact current independent source required')
    if MARKER in scene or bpy.data.objects.get(CARRIER):
        raise RuntimeError('Existing head binding/carrier must be preserved')
    head, bridge, rig = (bpy.data.objects[n] for n in (HEAD, BRIDGE, RIG))
    if head.parent != rig or head.parent_type != 'BONE' or head.parent_bone != 'head':
        raise RuntimeError('Expected current rigid head parent')
    if head.animation_data or head.constraints or any(m.type == 'ARMATURE' for m in head.modifiers):
        raise RuntimeError('Unexpected independent head animation/constraints/skinning')
    if any(head.vertex_groups.get(n) is not None for n in ('head', 'neck')):
        raise RuntimeError('Existing anatomical skin groups need their own explicit conversion')
    if any(abs(v) > 1e-8 for v in head.delta_location) or any(abs(v) > 1e-8 for v in head.delta_rotation_euler) or any(abs(v - 1.) > 1e-8 for v in head.delta_scale):
        raise RuntimeError('Head delta transforms require explicit carrier adaptation')
    if rig.pose.bones['head'].constraints or rig.animation_data is None:
        raise RuntimeError('Expected recorded rig action without head constraints')
    if any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA requires explicit composition')
    old_bridge_arm = [m for m in bridge.modifiers if m.type == 'ARMATURE']
    if len(old_bridge_arm) != 1 or old_bridge_arm[0].object != rig or old_bridge_arm[0].use_deform_preserve_volume:
        raise RuntimeError('Keep matched existing bridge LBS skinning')
    proposal = json.loads(plan_path.read_text(encoding='utf-8'))
    if not proposal.get('anatomicalProtection', {}).get('fullyProtectedVertexIds'):
        raise RuntimeError('Use the v2 plan with explicit actual oral/eye topology protection')
    if not proposal.get('actualCageMatchesAnatomicalIndexSpace'):
        raise RuntimeError('Use the refreshed v3 plan with actual author topology validation')
    observation_path = path(proposal['observation'], '.json')
    observation = json.loads(observation_path.read_text(encoding='utf-8'))
    if source != path(observation['source'], '.blend') or sha(source) != observation['sourceSha256']:
        raise RuntimeError('Refresh observation and plan on the exact final static input before binding')
    observation_arrays = path(observation['arrays'], '.npz')
    proposal_arrays = path(proposal['arrays'], '.npz')
    if sha(observation_path) != proposal['observationSha256'] or sha(observation_arrays) != proposal['observationArraysSha256'] or sha(proposal_arrays) != proposal['arraysSha256']:
        raise RuntimeError('Pinned observation/weight plan has changed')
    api('face_fibre_binding_r11')  # Pin both composed validators before any scene mutation.
    observer = api('adult_head_bind_observe_r11')
    refs = observer['references'](head)
    prearm_plan = eye_space_api()['prepare'](head)
    scalar_refs = eye_space_api()['allowed_head_property_references'](head)
    blocked_refs = [ref for ref in refs['driverTargets']
                    if not any(all(ref.get(k) == v for k, v in allowed.items()) for allowed in scalar_refs)]
    if blocked_refs or refs['constraints']:
        raise RuntimeError('Direct head-transform driver/constraint references need explicit migration: ' + json.dumps(
            {'driverTargets': blocked_refs, 'constraints': refs['constraints']}))
    direct = sorted([o for o in bpy.data.objects if o.parent == head], key=lambda o: o.name)
    if not direct or any(o.parent_type != 'OBJECT' for o in direct):
        raise RuntimeError('Current cranial direct children must use ordinary OBJECT parenting')
    descendants = sorted([o.name for o in bpy.data.objects if observer['descendant'](o, head)])
    gaze_compatibility = None
    if 'R11_gaze_control' in scene:
        gaze_record = json.loads(scene['R11_gaze_control'])
        gaze_module, gaze_compatibility = gaze_api(gaze_record.get('schema'))
        if 'R11_gaze_space_upgrade' in scene or gaze_record.get('gazeSpace', HEAD) != HEAD:
            raise RuntimeError('Existing gaze migration history must be preserved')
        if gaze_record.get('schema') == 3:
            stages = gaze_record.get('driverStages', [])
            if len(stages) != 6 or len(set(stages)) != 6:
                raise RuntimeError('Expected six distinct existing affine gaze driver stages')
            for name in stages:
                stage = bpy.data.objects.get(name)
                if stage is None or stage.type != 'EMPTY' or stage not in direct:
                    raise RuntimeError('Affine gaze stage must remain a current direct head Empty: ' + name)
    cranial_before = cranial_data_contract(descendants)
    body_before = digest(observer['data_contract'](bpy.data.objects[BODY]))
    head_geometry = protected_geometry(head)
    bridge_geometry = protected_geometry(bridge)
    head_data = head.data
    key_data = head.data.shape_keys
    head_weights_before = observer['weights'](head)
    head_groups_before = group_record(head)
    bridge_weights_before = observer['weights'](bridge)
    bridge_groups_before = group_record(bridge)
    if any(n not in rig.data.bones for n, _, _ in bridge_groups_before):
        raise RuntimeError('Bridge has non-skin groups; preserve separately before conversion')
    existing_head_modifiers = [(m.name, m.type, clean(api('production_uv_r08')['_scalar_rna'](m))) for m in head.modifiers]
    uv = api('production_uv_r08')
    pose_before = uv['_pose_contract']()
    rig_before = rig_definition(rig)
    source_sha = sha(source)
    old_frame = (scene.frame_current, scene.frame_subframe)
    old_pose = rig.data.pose_position
    old_action = rig.animation_data.action
    old_slot = rig.animation_data.action_slot
    old_basis = {b.name: b.matrix_basis.copy() for b in rig.pose.bones}
    record = {'source': source_relative, 'sourceSha256': source_sha, 'helperSha256': sha(__file__),
        'plan': plan_relative, 'planSha256': sha(plan_path), 'observationSourceSha256': observation['sourceSha256'],
        'exactFinalObservationSource': True, 'gazeCompatibility': gaze_compatibility,
        'originalHeadAttachment': attachment(head), 'originalChildAttachments': {o.name: attachment(o) for o in direct},
        'directChildren': [o.name for o in direct], 'allCranialDescendants': descendants,
        'originalHeadGroups': head_groups_before, 'addedHeadGroups': [],
        'bridgeGroupNames': [n for n, _, _ in bridge_groups_before],
        'originalBridgeWeights': bridge_weights_before.tolist(), 'originalBridgeGroups': bridge_groups_before,
        'gazeSpaceUpgraded': False, 'savedBlend': False, 'appearanceAccepted': False,
        'prearmatureEyePlan': prearm_plan, 'preservedHeadScalarReferences': scalar_refs,
        'prearmatureEyeMigration': None, 'prearmatureEyeHelperSha256': EYE_SPACE_SHA}
    success = False
    try:
        scene.frame_set(1)
        rig.data.pose_position = 'POSE'
        bpy.context.view_layer.update()
        neutral_matrix = rig.pose.bones['head'].matrix.copy()
        rig.animation_data.action = None
        pose_reference = _pose_preservation(rig, descendants, neutral_matrix)
        api('adult_motion_review_r11')['_neck_pose'](rig, neutral_matrix, 0., 0.)
        with api('face_contact_audit_r11')['rest_evaluation']():
            mix, mix_metadata = uv['current_mix'](head)
            current_head = api('adult_motion_review_r11')['_evaluated'](HEAD)
            current_bridge = api('adult_motion_review_r11')['_evaluated'](BRIDGE)
            rest_world = head.matrix_world.copy()
            with np.load(observation_arrays, allow_pickle=False) as z:
                mix_error = float(np.linalg.norm(observer['world'](mix, rest_world) - z['head_cage_current_world'], axis=1).max())
                head_error = float(np.linalg.norm(current_head['world'] - z['head_evaluated_rest_world'], axis=1).max())
                bridge_error = float(np.linalg.norm(current_bridge['world'] - z['bridge_evaluated_rest_world'], axis=1).max())
                if max(mix_error, head_error, bridge_error) > 2e-6:
                    raise RuntimeError('Static geometry changed since binding plan; recompute actual observation: ' + json.dumps([mix_error, head_error, bridge_error]))
                if not np.array_equal(current_head['loops'], z['head_evaluated_rest_loops']) or not np.array_equal(current_bridge['loops'], z['bridge_evaluated_rest_loops']):
                    raise RuntimeError('Head/bridge topology changed since binding observation')
            bones = observation['rig']['bones']
            for name in ('neck', 'head', 'upperChest'):
                if np.max(abs(np.asarray(rig.data.bones[name].matrix_local) - bones[name]['matrixLocal'])) > 2e-7:
                    raise RuntimeError('Cervical rest bone changed from the actual plan')
            with np.load(proposal_arrays, allow_pickle=False) as z:
                cage_weights = z['proposed_head_neck_weights'].copy()
            if cage_weights.shape != (len(head.data.vertices), 2):
                raise RuntimeError('Proposed current head cage size differs')
            protected_ids = np.asarray(proposal['anatomicalProtection']['fullyProtectedVertexIds'], np.int32)
            if np.max(cage_weights[protected_ids, 1]) != 0.:
                raise RuntimeError('Protected mouth/eye cage-support points have neck influence')
            carrier = bpy.data.objects.new(CARRIER, None)
            scene.collection.objects.link(carrier)
            assign_attachment(carrier, record['originalHeadAttachment'])
            carrier['R11_original_head_space_source'] = HEAD
            carrier['R11_skin_bind_source_sha256'] = source_sha
            carrier.empty_display_type = 'PLAIN_AXES'
            carrier.empty_display_size = .03
            bpy.context.view_layer.update()
            if np.max(abs(np.asarray(carrier.matrix_world) - rest_world)) > 2e-7:
                raise RuntimeError('Carrier does not reproduce old rigid head REST matrix')
            for child in direct:
                old = record['originalChildAttachments'][child.name]
                child.parent = carrier
                child.matrix_parent_inverse = Matrix(old['inverse'])
                child.matrix_basis = Matrix(old['basis'])
            head.parent = rig
            head.parent_type = 'OBJECT'
            head.parent_bone = ''
            head.matrix_parent_inverse = Matrix.Identity(4)
            head.matrix_basis = rig.matrix_world.inverted() @ rest_world
            bpy.context.view_layer.update()
            record['prearmatureEyeMigration'] = eye_space_api()['migrate'](record['prearmatureEyePlan'], head, carrier)
            for name in ('head', 'neck'):
                head.vertex_groups.new(name=name)
                record['addedHeadGroups'].append(name)
            set_weights(head, ['head', 'neck'], cage_weights)
            arm = head.modifiers.new(MODIFIER, 'ARMATURE')
            arm.object = rig
            arm.use_vertex_groups = True
            arm.use_bone_envelopes = False
            arm.use_deform_preserve_volume = False
            arm.show_viewport = arm.show_render = True
            bpy.context.view_layer.update()
            actual_head = api('adult_motion_review_r11')['_evaluated'](HEAD)
            if not api('adult_motion_review_r11')['_same_topology'](current_head, actual_head):
                raise RuntimeError('Armature altered current L3 topology')
            neutral_error = float(np.linalg.norm(actual_head['world'] - current_head['world'], axis=1).max())
            if neutral_error > 2e-6:
                raise RuntimeError('REST head surface changed while installing skin: ' + str(neutral_error))
            ev = head.evaluated_get(bpy.context.evaluated_depsgraph_get())
            hw = observer['weights'](head, ev.data.vertices)
            skin_indices = [head.vertex_groups[n].index for n in ('head', 'neck')]
            if np.max(abs(hw[:, skin_indices].sum(1) - 1.)) > 2e-5:
                raise RuntimeError('Actual interpolated L3 skin weights are not normalized')
            face_sets = ev.data.attributes.get('.sculpt_face_set')
            if face_sets is None or face_sets.domain != 'FACE':
                raise RuntimeError('Actual author oral face-set layer is required for protected skin validation')
            labels = observer['array'](face_sets.data, 'value', 1, np.int32).ravel()
            oral_polygons = np.flatnonzero(np.isin(labels, (7, 8)))
            if not len(oral_polygons):
                raise RuntimeError('Actual upper/lower oral face regions disappeared')
            oral_ids = np.unique(np.concatenate([np.asarray(ev.data.polygons[int(i)].vertices, np.int32) for i in oral_polygons]))
            oral_neck = float(hw[oral_ids, skin_indices[1]].max())
            if oral_neck > 1e-7:
                raise RuntimeError('Actual evaluated oral bag has nonzero cervical influence: ' + str(oral_neck))
            rigid_ids = np.flatnonzero(hw[:, skin_indices[0]] >= 1. - 1e-7)
            reference, mapping = api('adult_motion_review_r11')['_seam_map'](rig)
            ti, hi, bi, oi = mapping
            body_obj = bpy.data.objects[BODY]
            body_ev = body_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            body_weights = observer['weights'](body_obj, body_ev.data.vertices)[oi]
            names = record['bridgeGroupNames']
            boundary = np.zeros((len(ti) + len(bi), len(names)), np.float64)
            for name in ('head', 'neck'):
                boundary[:len(ti), names.index(name)] = hw[hi, head.vertex_groups[name].index]
            for i, name in enumerate(names):
                g = body_obj.vertex_groups.get(name)
                if g is not None:
                    boundary[len(ti):, i] = body_weights[:, g.index]
            loops = current_bridge['loops']
            nxt = np.arange(len(loops)) + 1
            nxt[current_bridge['face_starts'] + current_bridge['face_sizes'] - 1] = current_bridge['face_starts']
            edges = np.unique(np.sort(np.column_stack((loops, loops[nxt])), axis=1), axis=0)
            bw, solver = api('adult_head_bind_plan_r11_v3')['harmonic_bridge_weights'](
                current_bridge['world'], edges, np.r_[ti, bi], boundary)
            set_weights(bridge, names, bw)
            bpy.context.view_layer.update()
            after_bridge = api('adult_motion_review_r11')['_evaluated'](BRIDGE)
            bridge_rest_error = float(np.linalg.norm(after_bridge['world'] - current_bridge['world'], axis=1).max())
            if bridge_rest_error > 2e-6:
                raise RuntimeError('REST bridge surface changed when rebinding')
            record.update(currentMixMetadata=mix_metadata, observationParityMeters={'cage': mix_error, 'head': head_error, 'bridge': bridge_error},
                actualHeadRestErrorMeters=neutral_error, actualBridgeRestErrorMeters=bridge_rest_error,
                actualBridgeCount=len(current_bridge['world']), boundaryCounts=[len(ti), len(bi)],
                bridgeWeightSolver=solver, actualRigidCranialVertexCount=len(rigid_ids),
                actualProtectedOralVertices=len(oral_ids), actualMaximumOralNeckWeight=oral_neck,
                actualHeadBoundaryWeightRanges={name: [float(hw[hi, head.vertex_groups[name].index].min()),
                    float(hw[hi, head.vertex_groups[name].index].max())] for name in ('head', 'neck')})
        if 'R11_gaze_control' in scene:
            record['gazeSpaceUpgrade'] = gaze_api(gaze_compatibility['schema'])[0]['upgrade_existing_space']()
            record['gazeSpaceUpgraded'] = True
        poses = _pose_preservation(rig, descendants, neutral_matrix, expected=pose_reference,
                                   head_reference=pose_reference, rigid_ids=rigid_ids,
                                   prearmature_record=record['prearmatureEyeMigration'])
        for row in poses:
            row.pop('headWorld')
        record['finiteAttachmentAndCranialPoseChecks'] = poses
        record['actualFibreBindingValidation'] = api('face_fibre_binding_r11')['validate'](neutral=False)
        success = True
    finally:
        try:
            if not success:
                _restore(record)
        finally:
            rig.animation_data.action = old_action
            if old_slot is not None:
                rig.animation_data.action_slot = old_slot
            rig.data.pose_position = old_pose
            for name, basis in old_basis.items():
                rig.pose.bones[name].matrix_basis = basis
            scene.frame_set(old_frame[0], subframe=old_frame[1])
            bpy.context.view_layer.update()
    try:
        if head.data != head_data or head.data.shape_keys != key_data:
            raise RuntimeError('Original Mesh/Key datablock identity changed')
        if cranial_data_contract(descendants) != cranial_before:
            raise RuntimeError('Current-source eye/groom/fibre raw data, keys or modifier settings changed')
        if protected_geometry(head) != head_geometry or protected_geometry(bridge) != bridge_geometry:
            raise RuntimeError('Current source raw geometry/keys/UV/materials changed')
        now = observer['weights'](head)
        if group_record(head)[:len(head_groups_before)] != head_groups_before or not np.array_equal(now[:, :head_weights_before.shape[1]], head_weights_before):
            raise RuntimeError('Existing head mask/groups/weights changed')
        kept = [(m.name, m.type, clean(uv['_scalar_rna'](m))) for m in head.modifiers if m.name != MODIFIER]
        if kept != existing_head_modifiers:
            raise RuntimeError('Original head modifier settings changed')
        if group_record(bridge) != bridge_groups_before:
            raise RuntimeError('Existing bridge group mapping changed')
        if digest(observer['data_contract'](bpy.data.objects[BODY])) != body_before or rig_definition(rig) != rig_before or uv['_pose_contract']() != pose_before:
            raise RuntimeError('Current body/action/rig pose changed')
        if sha(source) != source_sha:
            raise RuntimeError('Input source file changed')
        record.update(originalMeshAndKeyIdsPreserved=True, allCurrentKeysAndMaterialsPreserved=True,
            currentCranialGeometryAndModifiersUnchanged=True, currentCranialContractSha256=cranial_before,
            originalHeadMaskAndGroupsPreserved=True, bodyAndRigUnchanged=True,
            originalHeadKeyNames=[k.name for k in key_data.key_blocks],
            newModifierIndex=list(head.modifiers).index(head.modifiers[MODIFIER]),
            limits=['Initial linear skinning only; actual neck shape/volume under combined motion still needs visible review.',
                'Finite attachment checks preserve original cranial frames; they do not certify face/lid/oral contact.',
                'Marked eye raycasts are evaluated in pre-Armature cranial space at neutral/yaw30/pitch15; combined gaze/blink contact remains separate.',
                'Input must exactly match its final fresh observation; all current expression/groom/Empty drivers are protected.',
                'New head/neck deformation invalidates prior hair/garment contact envelopes and tail source-contract caches.'])
        record['candidateContractSha256'] = candidate_contract(record)
        scene[MARKER] = json.dumps(record, ensure_ascii=False)
    except Exception:
        _restore(record)
        raise
    return record


def restore():
    if MARKER not in bpy.context.scene:
        return False
    record = json.loads(bpy.context.scene[MARKER])
    if candidate_contract(record) != record['candidateContractSha256']:
        raise RuntimeError('Preserve subsequent skin/key/attachment work before rollback')
    _restore(record)
    return {'restored': True, 'savedBlend': False}


def prepare(source_relative, output_relative, plan_relative):
    output = path(output_relative, '.blend')
    report_path = output.with_suffix('.head-skin.json')
    if output.exists() or report_path.exists() or output == path(source_relative, '.blend'):
        raise RuntimeError('New independent output required')
    record = apply(source_relative, plan_relative)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
        raise RuntimeError('Independent skin source save failed')
    record.update(savedBlend=True, output=output_relative, outputSha256=sha(output))
    with report_path.open('x', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print('R11_HEAD_SKIN_BOUND ' + json.dumps({'output': output_relative, 'sha256': record['outputSha256']}), flush=True)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plan', required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    prepare(a.source, a.output, a.plan)
