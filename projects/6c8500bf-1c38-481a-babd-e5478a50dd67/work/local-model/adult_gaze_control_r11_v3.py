"""Full-affine metric binocular gaze for the adult R11 eyes (independent v3).

No generated geometry, eye reparenting, old-head drivers, or eyelid deformation.
Persisted controls use ordinary Blender expression drivers, not Python handlers.
Default influence zero exactly retains the source neutral forward orientation.
Positive yaw looks toward orthonormal cranial +X; pitch toward cranial +Z.
The exposed per-eye angles already include influence. Blink/lid-follow consumers
must not multiply them by influence again. Actual pose/contact imagery is pending.
The anatomical head mesh may have an Armature modifier and a stable object
matrix; rotation space then comes from the explicitly identified cranial carrier.
The real parent matrix maps local target differences into physical meters.
Its inverse metric maps bounded physical directions back into pivot Euler axes.
Actual eye vertices are checked against world rigid rotation, so severe parent
anisotropy cannot silently distort the globe while its gaze ray looks correct.
Only v3 installations may use this version's metadata-only carrier migration.
This module never changes existing parenting, eye data, optics or the head rig.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
CONTROL = 'R11 Gaze control'
TARGET = 'R11 Gaze target'
MARKER = 'R11_gaze_control'
CRANIAL = 'R11 Cranial attachment space'
SPACE_UPGRADE = 'R11_gaze_space_upgrade'
SOURCE_TAG = 'R11_original_head_space_source'
INFLUENCE = 'R11_gaze_influence'
LIMITS = dict(yaw=25.0, pitchUp=18.0, pitchDown=20.0, minimumForwardMeters=.25)
PIVOTS = {s: 'R11 Adult eye pivot ' + s for s in ('L', 'R')}
EYES = ['R11 Adult ' + p + ' ' + s for s in ('L', 'R') for p in ('sclera', 'iris')]
STAGES = {s: {role: 'R11 Gaze ' + role + ' ' + s for role in ('metric', 'forward', 'local')} for s in ('L', 'R')}
RIGID_TOLERANCE_METERS = 2e-6


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metric_frame(matrix):
    """Full affine map and an orthonormal physical frame; no scalar scale."""
    import numpy as np
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Finite 4x4 affine matrix required')
    if np.max(np.abs(matrix[3] - [0, 0, 0, 1])) > 1e-10:
        raise ValueError('Affine last row required')
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if np.linalg.det(linear) <= 0 or min(singular) < 1e-9 or max(singular) / min(singular) > 1e5:
        raise ValueError('Non-singular right-handed cranial transform required')
    back = linear[:, 1] / np.linalg.norm(linear[:, 1])
    right = linear[:, 0] - back * np.dot(back, linear[:, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, back)
    basis = np.column_stack((right, back, up))
    metric = basis.T @ linear
    return dict(worldMatrix=matrix.tolist(), worldBasisColumns=basis.tolist(),
        localToMetricLinear=metric.tolist(), metricToLocalLinear=np.linalg.inv(metric).tolist(),
        singularValues=singular.tolist(), determinant=float(np.linalg.det(linear)))


def solve_angles(center, target, influence=1.0, metric=None):
    """Pure reference solver: local inputs, physical angles, local Euler output."""
    import numpy as np
    values = [*center, *target, influence]
    if len(center) != 3 or len(target) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Finite 3D centers, target and influence required')
    if not 0 <= influence <= 1:
        raise ValueError('Influence must be 0..1')
    metric = np.eye(3) if metric is None else np.asarray(metric, dtype=np.float64)
    if metric.shape != (3, 3) or not np.isfinite(metric).all():
        raise ValueError('Finite 3x3 metric required')
    x, dy, z = metric @ (np.asarray(target) - center)
    minimum = LIMITS['minimumForwardMeters']
    depth = max(minimum, -dy)
    yaw_raw = math.degrees(math.atan2(x, depth))
    pitch_raw = math.degrees(math.atan2(z, math.hypot(x, depth)))
    yaw = min(LIMITS['yaw'], max(-LIMITS['yaw'], yaw_raw)) * influence
    pitch = min(LIMITS['pitchUp'], max(-LIMITS['pitchDown'], pitch_raw)) * influence
    y, p = map(math.radians, (yaw, pitch))
    forward = np.array([math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)])
    local = np.linalg.solve(metric, forward)
    local /= np.linalg.norm(local)
    euler = [-math.atan2(local[2], math.hypot(local[0], local[1])), 0, math.atan2(local[0], -local[1])]
    return dict(yawDegrees=yaw, pitchDegrees=pitch, forward=forward.tolist(),
                localForward=local.tolist(), localEulerXYZRadians=euler,
                depthClamped=bool(-dy < minimum), angularClamped=bool(abs(yaw_raw) > LIMITS['yaw'] or
                    pitch_raw > LIMITS['pitchUp'] or pitch_raw < -LIMITS['pitchDown']))


def driver_plan(center, metric, inverse, side):
    """Topologically ordered small expression drivers, independently auditable."""
    plan = []
    def add(role, path, expr, variables, index=None):
        if len(expr) > 255: raise RuntimeError('Oversized planned driver')
        plan.append(dict(role=role, path=path, expression=expr, variables=variables, index=index))
    def prop(name): return '["' + name + '"]'
    for axis, row in zip('xyz', metric):
        expr = '+'.join('({:.17g})*(t{}-({:.17g}))'.format(c, i, center[i]) for i, c in enumerate(row))
        add('metric', prop(axis), expr, [('t'+str(i), 'target', 'location['+str(i)+']') for i in range(3)])
    variables = [(axis, 'metric', prop(axis)) for axis in 'xyz'] + [('w', 'control', prop(INFLUENCE))]
    add('control', prop('R11_gaze_yaw_'+side+'_deg'),
        'min(25,max(-25,atan2(x,max(.25,-y))*180/pi))*min(1,max(0,w))', variables)
    add('control', prop('R11_gaze_pitch_'+side+'_deg'),
        'min(18,max(-20,atan2(z,sqrt(x*x+max(.25,-y)**2))*180/pi))*min(1,max(0,w))', variables)
    variables = [('a', 'control', prop('R11_gaze_yaw_'+side+'_deg')), ('p', 'control', prop('R11_gaze_pitch_'+side+'_deg'))]
    for axis, expr in zip('xyz', ('sin(a*pi/180)*cos(p*pi/180)', '-cos(a*pi/180)*cos(p*pi/180)', 'sin(p*pi/180)')):
        add('forward', prop(axis), expr, variables)
    for axis, row in zip('xyz', inverse):
        expr = '+'.join('({:.17g})*{}'.format(c, a) for a, c in zip('xyz', row))
        add('local', prop(axis), expr, [(a, 'forward', prop(a)) for a in 'xyz'])
    add('pivot', 'rotation_euler', '0 if w<=0 else -atan2(z,sqrt(x*x+y*y))',
        [(a, 'local', prop(a)) for a in 'xyz'] + [('w', 'control', prop(INFLUENCE))], 0)
    add('pivot', 'rotation_euler', '0 if w<=0 else atan2(x,-y)',
        [(a, 'local', prop(a)) for a in 'xy'] + [('w', 'control', prop(INFLUENCE))], 2)
    return plan


def _matrix(matrix):
    return [[float(v) for v in row] for row in matrix]


def _max_matrix_error(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


def gaze_space(record=None, *, allow_unmigrated=False):
    """Resolve the actual bone-following frame, never infer it from mesh pose."""
    import bpy
    scene = bpy.context.scene
    if record is None and MARKER in scene: record = json.loads(scene[MARKER])
    carrier = bpy.data.objects.get(CRANIAL)
    if carrier is not None:
        if carrier.type != 'EMPTY' or carrier.get(SOURCE_TAG) != HEAD:
            raise RuntimeError('Named cranial object lacks the explicit original-head-space provenance')
        if carrier.parent_type != 'BONE' or not carrier.parent or carrier.parent.type != 'ARMATURE':
            raise RuntimeError('Cranial carrier must preserve the actual head bone parenting')
    expected = record.get('gazeSpace') if record else None
    if record and not expected:
        if carrier is not None and not allow_unmigrated:
            raise RuntimeError('Existing gaze needs explicit upgrade_existing_space() after cranial reparenting')
        expected = carrier.name if carrier is not None and allow_unmigrated else HEAD
    if expected is None: expected = carrier.name if carrier is not None else HEAD
    if expected not in (HEAD, CRANIAL) or expected not in bpy.data.objects:
        raise RuntimeError('Recorded gaze space is missing or unsupported: ' + str(expected))
    if expected == HEAD and carrier is not None and allow_unmigrated:
        expected = CRANIAL
    if expected == HEAD and carrier is not None:
        raise RuntimeError('Gaze still names the deforming head; migrate explicitly to the cranial carrier')
    return bpy.data.objects[expected]


def _check_space_parents(space, record):
    """Both angle inputs and pivots must express coordinates in the same frame."""
    import bpy
    import numpy as np
    from mathutils import Matrix
    if record.get('schema') != 3 or record.get('spaceSchema') != 3:
        raise RuntimeError('This module requires v3 drivers; legacy v1/v2 must be explicitly restored before fresh install')
    for name in (CONTROL, TARGET, *PIVOTS.values(), *record['driverStages']):
        obj = bpy.data.objects[name]
        if obj.parent != space or obj.parent_type != 'OBJECT' or obj.parent_bone:
            raise RuntimeError('Gaze attachment has a different parent space: ' + name)
        if _max_matrix_error(obj.matrix_parent_inverse, Matrix.Identity(4)) > 2e-6:
            raise RuntimeError('Unexpected parent inverse in gaze coordinate contract: ' + name)
    current = metric_frame(_matrix(space.matrix_world))
    metric = np.asarray(record['calibration']['localToMetricLinear'])
    delta = float(np.max(np.abs(np.asarray(current['localToMetricLinear']) - metric)))
    if delta > 2e-6:
        raise RuntimeError('Cranial affine metric changed; recalibration required, max coefficient difference ' + str(delta))
    return current


def _gaze_snapshot():
    import bpy
    record = json.loads(bpy.context.scene[MARKER])
    names = (CONTROL, TARGET, *PIVOTS.values(), *EYES, *record['driverStages'])
    return {name: dict(world=_matrix(bpy.data.objects[name].matrix_world),
        basis=_matrix(bpy.data.objects[name].matrix_basis),
        parent=bpy.data.objects[name].parent.name if bpy.data.objects[name].parent else None,
        inverse=_matrix(bpy.data.objects[name].matrix_parent_inverse)) for name in names}


def upgrade_existing_space():
    """Record a separately verified cranial reparenting; do not move any object.

    Call after the head-rig owner copied the original head transform to the
    carrier and moved every direct child while preserving all local matrices.
    This separately tests target/world conversion, drivers and eye convergence.
    """
    import bpy
    scene = bpy.context.scene
    if not bpy.app.background or MARKER not in scene or SPACE_UPGRADE in scene:
        raise RuntimeError('One existing independent v3 gaze installation required')
    previous = str(scene[MARKER]); record = json.loads(previous)
    if record.get('schema') != 3: raise RuntimeError('Do not migrate legacy driver equations with v3 metadata')
    if record.get('gazeSpace') not in (None, HEAD):
        raise RuntimeError('Gaze already has another explicit space; preserve its migration history')
    carrier = gaze_space(record, allow_unmigrated=True)
    if carrier.name != CRANIAL: raise RuntimeError('Install the verified cranial carrier first')
    _check_space_parents(carrier, record)
    before = _gaze_snapshot()
    eye_contracts = {name: _mesh_contract(bpy.data.objects[name]) for name in EYES}
    for name, contract in record['eyeContracts'].items():
        if eye_contracts[name] != contract: raise RuntimeError('Pre-upgrade eye contract mismatch: ' + name)
    record.update(gazeSpace=carrier.name, spaceSchema=3, anatomicalHead=HEAD,
        spaceUpgradeHelperSha256=_sha(__file__),
        spaceUpgradeScope='Only the space record changed; the head-rig owner separately preserved child transforms and anatomy.')
    try:
        scene[MARKER] = json.dumps(record, ensure_ascii=False)
        report = validate()
        after = _gaze_snapshot()
        for name, old in before.items():
            if old['parent'] != after[name]['parent'] or old['inverse'] != after[name]['inverse']:
                raise RuntimeError('Space upgrade changed attachment parent or inverse')
            if _max_matrix_error(old['world'], after[name]['world']) > 2e-6 or _max_matrix_error(old['basis'], after[name]['basis']) > 2e-6:
                raise RuntimeError('Space upgrade changed restored gaze transform: ' + name)
        scene[SPACE_UPGRADE] = json.dumps(dict(previousRecord=previous, targetSpace=carrier.name,
            helperSha256=_sha(__file__), validation=report, sourceSaved=False), ensure_ascii=False)
        return dict(gazeSpace=carrier.name, anatomyOrParentingChanged=False, eyeContractsUnchanged=True,
            validation=report, sourceSaved=False, appearanceAccepted=False)
    except Exception:
        scene[MARKER] = previous
        raise


def rollback_space_record():
    """After the rig owner restores old parenting, restore the original record.

    Does not undo Armature/carrier work. It is safe to call only after control,
    target and both pivots again have the original head as their direct parent,
    and the carrier has been removed or renamed by its owning module.
    """
    import bpy
    scene = bpy.context.scene
    if SPACE_UPGRADE not in scene: raise RuntimeError('No gaze-space migration to roll back')
    backup = json.loads(scene[SPACE_UPGRADE]); previous = backup['previousRecord']
    current = str(scene[MARKER]); record = json.loads(previous)
    if bpy.data.objects.get(CRANIAL) is not None:
        raise RuntimeError('Head-rig owner must restore attachment hierarchy/remove its carrier first')
    _check_space_parents(bpy.data.objects[HEAD], record)
    try:
        scene[MARKER] = previous
        result = validate()
        del scene[SPACE_UPGRADE]
        return result
    except Exception:
        scene[MARKER] = current
        raise


def _mesh_contract(obj):
    import numpy as np
    m = obj.data
    co = np.empty((len(m.vertices), 3), np.float32); m.vertices.foreach_get('co', co.ravel())
    h = hashlib.sha256(co.tobytes())
    loops = np.empty(len(m.loops), np.int32); m.loops.foreach_get('vertex_index', loops)
    h.update(loops.tobytes())
    for uv in m.uv_layers:
        values = np.empty((len(uv.data), 2), np.float32); uv.data.foreach_get('uv', values.ravel())
        h.update(uv.name.encode()); h.update(values.tobytes())
    return dict(data=m.name, shapeKeys=bool(m.shape_keys),
        geometryUvSha256=h.hexdigest(), materials=[mat.name if mat else None for mat in m.materials],
        parent=obj.parent.name if obj.parent else None, parentType=obj.parent_type, parentBone=obj.parent_bone,
        matrixParentInverse=_matrix(obj.matrix_parent_inverse), matrixBasis=_matrix(obj.matrix_basis),
        modifierNames=[[mod.name, mod.type] for mod in obj.modifiers])


def _prop(obj, name, value, *, low=None, high=None, description=''):
    obj[name] = float(value)
    settings = dict(description=description)
    if low is not None: settings.update(min=low, soft_min=low)
    if high is not None: settings.update(max=high, soft_max=high)
    obj.id_properties_ui(name).update(**settings)


def _driver(obj, path, expression, variables, index=None):
    if len(expression) > 255:
        raise RuntimeError('Driver expression must fit the persisted bounded expression field')
    fc = obj.driver_add(path) if index is None else obj.driver_add(path, index)
    driver = fc.driver; driver.type = 'SCRIPTED'
    for name, target, data_path in variables:
        var = driver.variables.new(); var.name = name; var.type = 'SINGLE_PROP'
        var.targets[0].id_type = 'OBJECT'; var.targets[0].id = target; var.targets[0].data_path = data_path
    driver.expression = expression
    if driver.expression != expression:
        raise RuntimeError('Blender did not retain the complete gaze driver expression')
    return fc


def _update():
    import bpy
    bpy.data.objects[CONTROL].update_tag()
    bpy.data.objects[TARGET].update_tag()
    for name in PIVOTS.values(): bpy.data.objects[name].update_tag()
    for stages in STAGES.values():
        for name in stages.values():
            obj = bpy.data.objects.get(name)
            if obj: obj.update_tag()
    bpy.context.view_layer.update()


def set_target_head_local(coordinates, influence=1.0):
    """Compatibility name: coordinates are in gaze_space(), now cranial local."""
    import bpy
    if len(coordinates) != 3 or not all(math.isfinite(x) for x in coordinates):
        raise ValueError('Finite head-local 3D target required')
    if not 0 <= influence <= 1: raise ValueError('Influence must be 0..1')
    _check_space_parents(gaze_space(), json.loads(bpy.context.scene[MARKER]))
    bpy.data.objects[TARGET].location = coordinates
    bpy.data.objects[CONTROL][INFLUENCE] = float(influence)
    _update()


def set_target_cranial_local(coordinates, influence=1.0):
    return set_target_head_local(coordinates, influence)


def set_target_world(coordinates, influence=1.0):
    """Map this world point now; target remains cranial-attached afterwards.

    Continuous world-locked fixation needs external animation of this control.
    """
    import bpy
    from mathutils import Vector
    space = gaze_space()
    set_target_head_local(space.matrix_world.inverted() @ Vector(coordinates), influence)


def neutral():
    import bpy
    bpy.data.objects[CONTROL][INFLUENCE] = 0.0
    _update()


def _world_vertices(obj):
    import numpy as np
    coords = np.empty((len(obj.data.vertices), 3), np.float64)
    obj.data.vertices.foreach_get('co', coords.ravel())
    matrix = np.asarray(_matrix(obj.matrix_world))
    return coords @ matrix[:3, :3].T + matrix[:3, 3]


def _metric_rotation(yaw_degrees, pitch_degrees):
    """The same world rigid XYZ rotation that a genuinely spherical eye uses."""
    import numpy as np
    y, p = map(math.radians, (yaw_degrees, pitch_degrees))
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    return rz @ rx


def install():
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    scene = bpy.context.scene
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Use an independent background candidate in Object Mode')
    source = Path(bpy.data.filepath).resolve(); source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    reserved = [CONTROL, TARGET, *(name for stages in STAGES.values() for name in stages.values())]
    if scene.get(MARKER) or any(bpy.data.objects.get(name) for name in reserved):
        raise RuntimeError('Existing gaze installation must be preserved; no stacking')
    if 'R11_adult_eye_proportion' not in scene:
        raise RuntimeError('Actual current eye-proportion source required')
    head = bpy.data.objects[HEAD]
    if abs(head.data.shape_keys.key_blocks['R11 adult eye proportion'].value - 1) > 1e-7:
        raise RuntimeError('Keep new globe and corresponding static eyelid proportion together')
    head = gaze_space()
    old = {}; centers = {}; made = []
    eye_contracts = {name: _mesh_contract(bpy.data.objects[name]) for name in EYES}
    source_sha = _sha(source)
    before_pose = _api('production_uv_r08')['_pose_contract']()
    with _api('face_contact_audit_r11')['rest_evaluation']():
        before_world = {name: _matrix(bpy.data.objects[name].matrix_world) for name in EYES}
        calibration = metric_frame(_matrix(head.matrix_world))
        metric = np.asarray(calibration['localToMetricLinear'])
        inverse = np.asarray(calibration['metricToLocalLinear'])
        print('R11_GAZE_ACTUAL_AFFINE ' + json.dumps(calibration), flush=True)
        for side, name in PIVOTS.items():
            pivot = bpy.data.objects[name]
            if pivot.parent != head or pivot.parent_type != 'OBJECT' or pivot.constraints or pivot.animation_data:
                raise RuntimeError('Inspect unexpected eye-pivot hierarchy, constraints or animation: ' + name)
            local = head.matrix_world.inverted() @ pivot.matrix_world
            # The actual adult-eye builder uses an identity local orientation.
            # Reject another basis rather than silently applying wrong axes.
            unit = Matrix.Identity(4); unit.translation = local.translation
            if _max_matrix_error(local, unit) > 2e-6 or _max_matrix_error(pivot.matrix_parent_inverse, Matrix.Identity(4)) > 2e-6:
                raise RuntimeError('Eye pivot is not the documented neutral local translation: ' + name)
            centers[side] = list(local.translation)
            old[side] = dict(rotationMode=pivot.rotation_mode, matrixBasis=_matrix(pivot.matrix_basis),
                matrixParentInverse=_matrix(pivot.matrix_parent_inverse), parent=pivot.parent.name,
                parentType=pivot.parent_type, parentBone=pivot.parent_bone)
        try:
            collection = head.users_collection[0]
            for name in reserved:
                obj = bpy.data.objects.new(name, None); collection.objects.link(obj); made.append(obj)
                obj.parent = head; obj.matrix_parent_inverse = Matrix.Identity(4); obj.matrix_basis = Matrix.Identity(4)
                obj.empty_display_type = 'SPHERE' if name == TARGET else 'PLAIN_AXES'
                obj.empty_display_size = .012 / max(calibration['singularValues']); obj.hide_render = True
                if name not in (CONTROL, TARGET): obj.hide_set(True)
            control, target = made[:2]
            middle = [(centers['L'][i] + centers['R'][i]) * .5 for i in range(3)]
            control.location = middle
            target.location = (np.asarray(middle) + inverse @ [0, -2.0, 0]).tolist()
            _prop(control, INFLUENCE, 0, low=0, high=1, description='0: exact source neutral; 1: binocular target gaze')
            for side, center in centers.items():
                for axis in ('yaw', 'pitch'):
                    prop = 'R11_gaze_' + axis + '_' + side + '_deg'
                    _prop(control, prop, 0, low=-25, high=25, description='Physical cranial degrees including influence; positive ' + ('+X' if axis == 'yaw' else '+Z'))
                pivot = bpy.data.objects[PIVOTS[side]]
                pivot.rotation_mode = 'XYZ'
                objects = dict(target=target, control=control, pivot=pivot,
                    **{role: bpy.data.objects[name] for role, name in STAGES[side].items()})
                for role in STAGES[side]:
                    for axis in 'xyz': _prop(objects[role], axis, 0, description='Internal affine gaze stage; do not edit')
                for entry in driver_plan(center, metric, inverse, side):
                    variables = [(n, objects[role], path) for n, role, path in entry['variables']]
                    _driver(objects[entry['role']], entry['path'], entry['expression'], variables, entry['index'])
            control['R11_gaze_usage'] = 'Move R11 Gaze target; influence 0 is exact neutral. Angles already include influence. No lid follow installed.'
            scene[MARKER] = json.dumps(dict(schema=3, head=HEAD, anatomicalHead=HEAD, gazeSpace=head.name, spaceSchema=3, control=CONTROL, target=TARGET,
                centersHeadLocal=centers, calibration=calibration, originalPivots=old, limits=LIMITS,
                driverStages=reserved[2:], rigidVertexToleranceMeters=RIGID_TOLERANCE_METERS,
                eyeContracts=eye_contracts, source=str(source.relative_to(project_path())), sourceSha256=source_sha,
                helperSha256=_sha(__file__), eyelidFollowInstalled=False), ensure_ascii=False)
            _update()
            neutral_error = max(_max_matrix_error(bpy.data.objects[n].matrix_world, m) for n, m in before_world.items())
            if neutral_error > 2e-6: raise RuntimeError('Neutral eye transform changed: ' + str(neutral_error))
            report = validate()
            report['neutralWorldMatrixMaxError'] = neutral_error
            for name, contract in eye_contracts.items():
                if _mesh_contract(bpy.data.objects[name]) != contract:
                    raise RuntimeError('Eye geometry, UV, material or child transform changed: ' + name)
        except Exception:
            if MARKER in scene: restore()
            else:
                for side, data in old.items():
                    pivot = bpy.data.objects[PIVOTS[side]]
                    if pivot.animation_data: pivot.animation_data_clear()
                    pivot.rotation_mode = data['rotationMode']; pivot.matrix_basis = Matrix(data['matrixBasis'])
                for obj in reversed(made): bpy.data.objects.remove(obj, do_unlink=True)
            raise
    if _api('production_uv_r08')['_pose_contract']() != before_pose or _sha(source) != source_sha:
        restore(); raise RuntimeError('Gaze installation changed original rig pose or source file')
    report.update(sourceSaved=False, appearanceAccepted=False, helperSha256=_sha(__file__))
    return report


def validate():
    """Single-update real drivers, metric convergence and world rigid geometry."""
    import bpy
    import numpy as np
    from mathutils import Vector
    scene = bpy.context.scene; data = json.loads(scene[MARKER])
    head = gaze_space(data); frame = _check_space_parents(head, data)
    control = bpy.data.objects[CONTROL]; target = bpy.data.objects[TARGET]
    target_old = target.location.copy(); influence_old = float(control[INFLUENCE])
    centers = data['centersHeadLocal']
    metric = np.asarray(data['calibration']['localToMetricLinear'])
    inverse = np.asarray(data['calibration']['metricToLocalLinear'])
    basis = np.asarray(frame['worldBasisColumns'])
    world_matrix = np.asarray(_matrix(head.matrix_world))
    middle = (np.asarray(centers['L']) + centers['R']) * .5
    middle_world = world_matrix[:3, :3] @ middle + world_matrix[:3, 3]
    world_centers = {side: world_matrix[:3, :3] @ center + world_matrix[:3, 3] for side, center in centers.items()}
    cases = [('neutral', (0, -2, 0), 0), ('forward', (0, -2, 0), 1), ('near', (0, -.4, 0), 1),
        ('left', (.4, -2, 0), 1), ('right', (-.4, -2, 0), 1), ('up', (0, -2, .35), 1),
        ('down', (0, -2, -.35), 1), ('half', (.4, -2, .2), .5),
        ('limited-up-left', (3, -.1, 3), 1), ('limited-down-right', (-3, -.1, -3), 1),
        ('limited-up-right', (-3, -.1, 3), 1), ('limited-down-left', (3, -.1, -3), 1),
        ('behind', (0, .2, 0), 1), ('reverse-up', (0, -2, .35), 1),
        ('reverse-right', (-.4, -2, 0), 1), ('reverse-near', (0, -.4, 0), 1),
        ('reverse-neutral', (3, -.1, -3), 0)]
    rows = []
    try:
        neutral()
        neutral_vertices = {name: _world_vertices(bpy.data.objects[name]) for name in EYES}
        for label, offset, influence in cases:
            position = middle + inverse @ offset
            target_world = middle_world + basis @ offset
            set_target_world(target_world.tolist(), influence)
            actual_target_world = np.asarray(head.matrix_world @ target.location)
            roundtrip = float(np.linalg.norm(actual_target_world - target_world))
            if roundtrip > 2e-6: raise RuntimeError('World target did not map back into the cranial frame')
            row = dict(case=label, targetHeadLocal=position.tolist(), targetWorld=target_world.tolist(), influence=influence,
                cranialWorldTargetRoundtripErrorMeters=roundtrip, eyes={})
            for side, center in centers.items():
                pivot = bpy.data.objects[PIVOTS[side]]
                actual_forward = np.asarray((pivot.matrix_world.to_3x3() @ Vector((0, -1, 0))).normalized())
                expected = solve_angles(center, target.location, influence, metric)
                expected_world = basis @ expected['forward']
                direction_error = float(np.linalg.norm(actual_forward - expected_world))
                center_error = float(np.linalg.norm(np.asarray(pivot.matrix_world.translation) - world_centers[side]))
                angle_error = max(abs(float(control['R11_gaze_' + axis + '_' + side + '_deg']) - expected[axis + 'Degrees']) for axis in ('yaw', 'pitch'))
                if center_error > 2e-6 or direction_error > 3e-6 or angle_error > 1e-4:
                    raise RuntimeError('Actual gaze driver evaluation mismatch: ' + json.dumps(dict(case=label, side=side, centerError=center_error, directionError=direction_error, angleError=angle_error)))
                delta = actual_target_world - world_centers[side]
                ray_miss = float(np.linalg.norm(np.cross(actual_forward, delta)))
                target_exact = influence == 1 and not expected['depthClamped'] and not expected['angularClamped']
                if target_exact and (ray_miss > 1e-5 or float(actual_forward @ delta) <= 0):
                    raise RuntimeError('Unclamped eyes fail to converge on common target')
                rigid_rotation = basis @ _metric_rotation(expected['yawDegrees'], expected['pitchDegrees']) @ basis.T
                geometry_errors = {}
                for part in ('sclera', 'iris'):
                    name = 'R11 Adult ' + part + ' ' + side
                    reference = (neutral_vertices[name] - world_centers[side]) @ rigid_rotation.T + world_centers[side]
                    geometry_errors[name] = float(np.linalg.norm(_world_vertices(bpy.data.objects[name]) - reference, axis=1).max())
                if max(geometry_errors.values()) > RIGID_TOLERANCE_METERS:
                    raise RuntimeError('Affine pivot motion distorts the actual eye beyond physical tolerance: ' + json.dumps(geometry_errors))
                row['eyes'][side] = dict(**expected, centerErrorMeters=center_error, directionError=direction_error,
                    evaluatedAngleErrorDegrees=angle_error, targetRayMissMeters=ray_miss, targetConvergenceExpected=target_exact,
                    actualWorldRigidVertexErrorMeters=geometry_errors)
            rows.append(row)
        for name, contract in data['eyeContracts'].items():
            if _mesh_contract(bpy.data.objects[name]) != contract:
                raise RuntimeError('Eye contract mismatch after gaze tests: ' + name)
        driven = [control, *(bpy.data.objects[n] for n in PIVOTS.values()), *(bpy.data.objects[n] for n in data['driverStages'])]
        invalid = [obj.name + ':' + fc.data_path for obj in driven for fc in obj.animation_data.drivers if not fc.driver.is_valid]
        if invalid: raise RuntimeError('Invalid persisted gaze drivers: ' + str(invalid))
    finally:
        set_target_head_local(target_old, influence_old)
    return dict(cases=rows, controls=dict(object=CONTROL, target=TARGET, influence=INFLUENCE,
        angleConvention='yaw +X, pitch +Z, cranial-space degrees, already multiplied by influence', gazeSpace=head.name),
        limits=LIMITS, calibration=data['calibration'], evaluatedFrame=frame,
        originalParentsAndEyeGeometryPreserved=True, eyelidFollowInstalled=False,
        rigidVertexToleranceMeters=RIGID_TOLERANCE_METERS,
        maximumActualWorldRigidVertexErrorMeters=max(max(eye['actualWorldRigidVertexErrorMeters'].values()) for row in rows for eye in row['eyes'].values()),
        limitsOfEvidence=['This verifies actual eye transforms and convergence, not eyelid contact or appearance.',
            'Up/down/side gaze must be combined with the current-size eyelid follow and blink, then rendered.',
            'Current source eye geometry, UVs and optical materials are preserved, including any recessed iris.',
            'Rigid head motion is supported. Animated changes of the cranial affine metric require explicit recalibration.',
            'The target is cranial-attached; set_target_world maps a point at call time, not a persistent world lock.',
            'Fresh-process validate() after saving is required to establish persistent driver evaluation.'])


def restore():
    import bpy
    from mathutils import Matrix
    scene = bpy.context.scene; data = json.loads(scene[MARKER])
    for side, old in data['originalPivots'].items():
        pivot = bpy.data.objects[PIVOTS[side]]
        for index in (0, 2): pivot.driver_remove('rotation_euler', index)
        if pivot.animation_data and not pivot.animation_data.drivers and not pivot.animation_data.action and not pivot.animation_data.nla_tracks:
            pivot.animation_data_clear()
        pivot.rotation_mode = old['rotationMode']; pivot.matrix_basis = Matrix(old['matrixBasis'])
    for name in (*data['driverStages'], TARGET, CONTROL): bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    del scene[MARKER]
    if SPACE_UPGRADE in scene: del scene[SPACE_UPGRADE]
    bpy.context.view_layer.update()


def prepare(source_relative, output_relative):
    import bpy
    root = (project_path() / 'work/reconstruction-r02').resolve()
    source = (project_path() / source_relative).resolve(); output = (project_path() / output_relative).resolve()
    for path in (source, output): path.relative_to(root)
    report_path = output.with_suffix('.gaze.json')
    if source.suffix != '.blend' or output.suffix != '.blend' or output.exists() or report_path.exists():
        raise RuntimeError('Fresh independent .blend and .gaze.json output paths required')
    if bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the exact saved independent source before prepare')
    report = install()
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    report.update(output=str(output.relative_to(project_path())), outputSha256=_sha(output), outputSaved=True, sourceSaved=False)
    with report_path.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_GAZE_PREPARED ' + json.dumps(dict(output=str(output), sha256=report['outputSha256'])), flush=True)
    return report


def verify_saved(source_relative, output_relative):
    """Fresh-process numeric replay only; never save or render a Blend."""
    import bpy
    root = (project_path() / 'work/reconstruction-r02').resolve()
    source = (project_path() / source_relative).resolve(); output = (project_path() / output_relative).resolve()
    for path in (source, output): path.relative_to(root)
    if source.suffix != '.blend' or output.suffix != '.json' or output.exists():
        raise RuntimeError('Existing Blend and fresh JSON report path required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the exact saved independent source in a fresh background process')
    before_sha = _sha(source)
    report = validate()
    if _sha(source) != before_sha: raise RuntimeError('Fresh-process numeric replay changed source file')
    report.update(source=str(source.relative_to(project_path())), sourceSha256=before_sha,
        helperSha256=_sha(__file__), freshProcessReplay=True, sourceSaved=False, appearanceAccepted=False)
    with output.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_GAZE_RELOADED ' + json.dumps(dict(source=str(source), sourceSha256=before_sha)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'verify'), default='prepare')
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    (prepare if args.mode == 'prepare' else verify_saved)(args.source, args.output)
