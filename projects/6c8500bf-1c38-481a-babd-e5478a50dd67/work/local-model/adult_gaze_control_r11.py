"""Independent binocular target control for the actual adult R11 eye pivots.

No generated geometry, eye reparenting, old-head drivers, or eyelid deformation.
Persisted controls use ordinary Blender expression drivers, not Python handlers.
Default influence zero exactly retains the source neutral forward orientation.
Positive yaw looks toward head-local +X; positive pitch toward head-local +Z.
The exposed per-eye angles already include influence. Blink/lid-follow consumers
must not multiply them by influence again. Actual pose/contact imagery is pending.
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
INFLUENCE = 'R11_gaze_influence'
LIMITS = dict(yaw=25.0, pitchUp=18.0, pitchDown=20.0, minimumForwardMeters=.25)
PIVOTS = {s: 'R11 Adult eye pivot ' + s for s in ('L', 'R')}
EYES = ['R11 Adult ' + p + ' ' + s for s in ('L', 'R') for p in ('sclera', 'iris')]


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def solve_angles(center, target, influence=1.0, head_scale=1.0):
    """Pure reference calculation, in head-local units. No Blender imports."""
    values = [*center, *target, influence, head_scale]
    if len(center) != 3 or len(target) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('Finite 3D centers, target, influence and scale required')
    if not 0 <= influence <= 1 or head_scale <= 0:
        raise ValueError('Influence must be 0..1; scale must be positive')
    x, dy, z = [target[i] - center[i] for i in range(3)]
    minimum = LIMITS['minimumForwardMeters'] / head_scale
    depth = max(minimum, -dy)
    yaw_raw = math.degrees(math.atan2(x, depth))
    pitch_raw = math.degrees(math.atan2(z, math.hypot(x, depth)))
    yaw = min(LIMITS['yaw'], max(-LIMITS['yaw'], yaw_raw)) * influence
    pitch = min(LIMITS['pitchUp'], max(-LIMITS['pitchDown'], pitch_raw)) * influence
    y, p = map(math.radians, (yaw, pitch))
    return dict(yawDegrees=yaw, pitchDegrees=pitch,
                forward=[math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)],
                depthClamped=(-dy < minimum), angularClamped=(abs(yaw_raw) > LIMITS['yaw'] or
                    pitch_raw > LIMITS['pitchUp'] or pitch_raw < -LIMITS['pitchDown']))


def _matrix(matrix):
    return [[float(v) for v in row] for row in matrix]


def _max_matrix_error(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


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
    bpy.context.view_layer.update()


def set_target_head_local(coordinates, influence=1.0):
    import bpy
    if len(coordinates) != 3 or not all(math.isfinite(x) for x in coordinates):
        raise ValueError('Finite head-local 3D target required')
    if not 0 <= influence <= 1: raise ValueError('Influence must be 0..1')
    bpy.data.objects[TARGET].location = coordinates
    bpy.data.objects[CONTROL][INFLUENCE] = float(influence)
    _update()


def set_target_world(coordinates, influence=1.0):
    import bpy
    from mathutils import Vector
    head = bpy.data.objects[HEAD]
    set_target_head_local(head.matrix_world.inverted() @ Vector(coordinates), influence)


def neutral():
    import bpy
    bpy.data.objects[CONTROL][INFLUENCE] = 0.0
    _update()


def install():
    import bpy
    from mathutils import Matrix, Vector
    scene = bpy.context.scene
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Use an independent background candidate in Object Mode')
    source = Path(bpy.data.filepath).resolve(); source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if scene.get(MARKER) or bpy.data.objects.get(CONTROL) or bpy.data.objects.get(TARGET):
        raise RuntimeError('Existing gaze installation must be preserved; no stacking')
    if 'R11_adult_eye_proportion' not in scene:
        raise RuntimeError('Actual current eye-proportion source required')
    head = bpy.data.objects[HEAD]
    if abs(head.data.shape_keys.key_blocks['R11 adult eye proportion'].value - 1) > 1e-7:
        raise RuntimeError('Keep new globe and corresponding static eyelid proportion together')
    old = {}; centers = {}; made = []
    eye_contracts = {name: _mesh_contract(bpy.data.objects[name]) for name in EYES}
    source_sha = _sha(source)
    before_pose = _api('production_uv_r08')['_pose_contract']()
    with _api('face_contact_audit_r11')['rest_evaluation']():
        before_world = {name: _matrix(bpy.data.objects[name].matrix_world) for name in EYES}
        head_scale = float(head.matrix_world.to_scale().x)
        if max(abs(v - head_scale) for v in head.matrix_world.to_scale()) > 1e-7 or head_scale <= 0:
            raise RuntimeError('Positive uniform head transform required')
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
            for name in (CONTROL, TARGET):
                obj = bpy.data.objects.new(name, None); collection.objects.link(obj); made.append(obj)
                obj.parent = head; obj.matrix_parent_inverse = Matrix.Identity(4); obj.matrix_basis = Matrix.Identity(4)
                obj.empty_display_type = 'SPHERE' if name == TARGET else 'PLAIN_AXES'
                obj.empty_display_size = .012 / head_scale; obj.hide_render = True
            control, target = made
            middle = [(centers['L'][i] + centers['R'][i]) * .5 for i in range(3)]
            control.location = middle
            target.location = [middle[0], middle[1] - 2.0 / head_scale, middle[2]]
            _prop(control, INFLUENCE, 0, low=0, high=1, description='0: exact source neutral; 1: binocular target gaze')
            for side, center in centers.items():
                x = '(tx-({:.17g}))'.format(center[0])
                depth = 'max({:.17g},({:.17g})-ty)'.format(LIMITS['minimumForwardMeters'] / head_scale, center[1])
                z = '(tz-({:.17g}))'.format(center[2])
                variables = [(n, target, 'location[' + str(i) + ']') for i, n in enumerate(('tx', 'ty', 'tz'))]
                variables += [('weight', control, '["' + INFLUENCE + '"]')]
                expressions = {
                    'yaw': 'min(25,max(-25,atan2(' + x + ',' + depth + ')*180/pi))*min(1,max(0,weight))',
                    'pitch': 'min(18,max(-20,atan2(' + z + ',sqrt(' + x + '*' + x + '+' + depth + '*' + depth + '))*180/pi))*min(1,max(0,weight))',
                }
                for axis, expression in expressions.items():
                    prop = 'R11_gaze_' + axis + '_' + side + '_deg'
                    _prop(control, prop, 0, low=-25, high=25, description='Evaluated per-eye angle, including influence; positive ' + ('+X' if axis == 'yaw' else '+Z'))
                    _driver(control, '["' + prop + '"]', expression, variables)
                pivot = bpy.data.objects[PIVOTS[side]]
                pivot.rotation_mode = 'XYZ'
                _driver(pivot, 'rotation_euler', '-pitch*pi/180', [('pitch', control, '["R11_gaze_pitch_' + side + '_deg"]')], 0)
                _driver(pivot, 'rotation_euler', 'yaw*pi/180', [('yaw', control, '["R11_gaze_yaw_' + side + '_deg"]')], 2)
            control['R11_gaze_usage'] = 'Move R11 Gaze target; influence 0 is exact neutral. Angles already include influence. No lid follow installed.'
            scene[MARKER] = json.dumps(dict(schema=1, head=HEAD, control=CONTROL, target=TARGET,
                centersHeadLocal=centers, headUniformScale=head_scale, originalPivots=old, limits=LIMITS,
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
    """Evaluate real driver directions, center invariance and shared-ray convergence."""
    import bpy
    from mathutils import Vector
    scene = bpy.context.scene; data = json.loads(scene[MARKER])
    head = bpy.data.objects[HEAD]; control = bpy.data.objects[CONTROL]; target = bpy.data.objects[TARGET]
    target_old = target.location.copy(); influence_old = float(control[INFLUENCE])
    centers = data['centersHeadLocal']; scale = data['headUniformScale']
    middle = [(centers['L'][i] + centers['R'][i]) * .5 for i in range(3)]
    cases = [('neutral', (0, -2, 0), 0), ('forward', (0, -2, 0), 1), ('near', (0, -.4, 0), 1),
        ('left', (.4, -2, 0), 1), ('right', (-.4, -2, 0), 1), ('up', (0, -2, .35), 1),
        ('down', (0, -2, -.35), 1), ('half', (.4, -2, .2), .5),
        ('limited', (3, -.1, 3), 1), ('behind', (0, .2, 0), 1)]
    rows = []
    try:
        for label, offset, influence in cases:
            position = [middle[i] + offset[i] / scale for i in range(3)]
            set_target_head_local(position, influence)
            row = dict(case=label, targetHeadLocal=position, influence=influence, eyes={})
            for side, center in centers.items():
                pivot = bpy.data.objects[PIVOTS[side]]
                local = head.matrix_world.inverted() @ pivot.matrix_world
                actual_forward = (local.to_3x3() @ Vector((0, -1, 0))).normalized()
                expected = solve_angles(center, position, influence, scale)
                direction_error = (actual_forward - Vector(expected['forward'])).length
                center_error = (local.translation - Vector(center)).length * scale
                angle_error = max(abs(float(control['R11_gaze_' + axis + '_' + side + '_deg']) - expected[axis + 'Degrees']) for axis in ('yaw', 'pitch'))
                if center_error > 2e-6 or direction_error > 3e-6 or angle_error > 1e-4:
                    raise RuntimeError('Actual gaze driver evaluation mismatch: ' + json.dumps(dict(case=label, side=side, centerError=center_error, directionError=direction_error, angleError=angle_error)))
                delta = Vector(position) - Vector(center)
                ray_miss = actual_forward.cross(delta).length * scale
                target_exact = influence == 1 and not expected['depthClamped'] and not expected['angularClamped']
                if target_exact and (ray_miss > 1e-5 or actual_forward.dot(delta) <= 0):
                    raise RuntimeError('Unclamped eyes fail to converge on common target')
                row['eyes'][side] = dict(**expected, centerErrorMeters=center_error, directionError=direction_error,
                    evaluatedAngleErrorDegrees=angle_error, targetRayMissMeters=ray_miss, targetConvergenceExpected=target_exact)
            rows.append(row)
        for name, contract in data['eyeContracts'].items():
            if _mesh_contract(bpy.data.objects[name]) != contract:
                raise RuntimeError('Eye contract mismatch after gaze tests: ' + name)
        invalid = [fc.data_path for obj in (control, *(bpy.data.objects[n] for n in PIVOTS.values()))
                   for fc in obj.animation_data.drivers if not fc.driver.is_valid]
        if invalid: raise RuntimeError('Invalid persisted gaze drivers: ' + str(invalid))
    finally:
        set_target_head_local(target_old, influence_old)
    return dict(cases=rows, controls=dict(object=CONTROL, target=TARGET, influence=INFLUENCE,
        angleConvention='yaw +X, pitch +Z, head-local degrees, already multiplied by influence'),
        limits=LIMITS, originalParentsAndEyeGeometryPreserved=True, eyelidFollowInstalled=False,
        limitsOfEvidence=['This verifies actual eye transforms and convergence, not eyelid contact or appearance.',
            'Up/down/side gaze must be combined with the current-size eyelid follow and blink, then rendered.',
            'Current opaque iris/clearcoat optics are unchanged; this is not a new corneal model.'])


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
    for name in (TARGET, CONTROL): bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    del scene[MARKER]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.output)
