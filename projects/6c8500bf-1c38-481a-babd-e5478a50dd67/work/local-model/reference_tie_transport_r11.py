"""Reference-led rigid twin-tail transport with explicit native gather fields.

No import-time Blender edits. Caller must supply a measured plan and enter
REST. Whole tails, all pinned points and both rings share one proper rigid
transform per side. Raw donor guides remain unchanged; the explicit editable
gather stage transports their final dense paths before existing head contact.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

MAIN = 'R11 Donor / long hair main'
GROUP = 'R11 groom interpolation group'
STAGE = 'R11 reference gather tie transport'
START, END = .45, .84
TAILS = {'positive': 'R02 Strands / R02 Twin tail L',
         'negative': 'R02 Strands / R02 Twin tail R'}
TIES = {side: ['R02 Tie %d %d' % (sign, ring) for ring in (-1, 1)]
        for side, sign in (('positive', 1), ('negative', -1))}


def transform(points, specification):
    r = np.asarray(specification['rotationWorld'], np.float64)
    origin = np.asarray(specification['originWorld'], np.float64)
    target = np.asarray(specification['targetWorld'], np.float64)
    return (np.asarray(points, np.float64) - origin) @ r.T + target


def validate_transform(value):
    r = np.asarray(value['rotationWorld'], np.float64)
    origin = np.asarray(value['originWorld'], np.float64)
    target = np.asarray(value['targetWorld'], np.float64)
    if r.shape != (3, 3) or origin.shape != (3,) or target.shape != (3,):
        raise RuntimeError('Rigid transform shape mismatch')
    if not np.isfinite(np.r_[r.ravel(), origin, target]).all():
        raise RuntimeError('Nonfinite rigid transform')
    if np.max(np.abs(r.T @ r - np.eye(3))) > 1e-9 or abs(np.linalg.det(r) - 1.) > 1e-9:
        raise RuntimeError('Tail transform must be a proper rotation, without scale/reflection')
    angle = float(np.degrees(np.arccos(np.clip((np.trace(r)-1.)*.5, -1., 1.))))
    # These are explicit scope bounds, not a physical-clearance acceptance.
    if np.linalg.norm(target-origin) > .085 or angle > 20.:
        raise RuntimeError('Proposed transport exceeds reviewed small rigid-move scope')
    return {'rotationDegrees': angle, 'originWorld': origin.tolist(),
            'targetWorld': target.tolist(), 'originDisplacementWorld': (target-origin).tolist(),
            'rotationWorld': r.tolist(), 'determinant': float(np.linalg.det(r))}


def raw_attribute_contract(data):
    """Preserve all copied non-position scalar/vector attributes, including absence."""
    result = []
    for attr in data.attributes:
        if attr.name == 'position':
            continue
        if not len(attr.data):
            payload = b''
        elif attr.data_type == 'STRING':
            payload = json.dumps([item.value for item in attr.data], ensure_ascii=False).encode()
        else:
            # Meshes expose built-in INT32_2D edge indices as well as UV vectors.
            # Discover the actual RNA value width without inventing defaults.
            item = attr.data[0]
            field = next((key for key in ('value', 'vector', 'color') if hasattr(item, key)), None)
            if field is None:
                raise RuntimeError('Unreadable copied attribute requires explicit preservation: ' + attr.name)
            sample = np.asarray(getattr(item, field))
            width = int(sample.size)
            dtype = np.int32 if attr.data_type.startswith('INT') or attr.data_type == 'BOOLEAN' else np.float32
            a = np.empty(len(attr.data)*width, dtype); attr.data.foreach_get(field, a)
            payload = a.tobytes()
        result.append({'name': attr.name, 'type': attr.data_type, 'domain': attr.domain,
                       'elements': len(attr.data), 'sha256': hashlib.sha256(payload).hexdigest()})
    return result


def math(tree, operation, a, b):
    node = tree.nodes.new('ShaderNodeMath'); node.operation = operation
    for index, value in enumerate((a, b)):
        if isinstance(value, (int, float, np.integer, np.floating)):
            node.inputs[index].default_value = float(value)
        else:
            tree.links.new(value, node.inputs[index])
    return node.outputs[0]


def geometry_socket(sockets):
    candidates = [s for s in sockets if s.type == 'GEOMETRY']
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one geometry socket')
    return candidates[0]


def install_gather(obj, transforms):
    import bpy
    mods = [m for m in obj.modifiers if m.type == 'NODES' and m.show_render]
    if len(mods) != 1:
        raise RuntimeError('Expected one actual render styling modifier')
    modifier = mods[0]; old = modifier.node_group
    if STAGE in old.nodes:
        raise RuntimeError('Tie transport already installed')
    old.use_fake_user = True
    tree = old.copy(); tree.name = 'R11 original-reference low ties rigid transport'
    modifier.node_group = tree
    contact = tree.nodes['R11 full head clearance']
    destination = geometry_socket(contact.inputs)
    if len(destination.links) != 1 or destination.links[0].from_node.name != 'R11 reference local front detail':
        raise RuntimeError('Actual saved v10 before-contact stage is required')
    source = destination.links[0].from_socket
    tree.links.remove(destination.links[0])
    attr = tree.nodes.new('GeometryNodeInputNamedAttribute'); attr.data_type = 'INT'
    attr.inputs['Name'].default_value = GROUP
    roles = math(tree, 'MODULO', attr.outputs['Attribute'], 256.)
    matrix = np.asarray(obj.matrix_world, np.float64)
    a = matrix[:3, :3]; b = matrix[:3, 3]; inv = np.linalg.inv(a)
    rows = []
    for side, role in (('positive', 40), ('negative', 41)):
        specification = transforms[side]
        r = np.asarray(specification['rotationWorld'], np.float64)
        origin = np.asarray(specification['originWorld'], np.float64)
        target = np.asarray(specification['targetWorld'], np.float64)
        local_r = inv @ r @ a
        local_t = inv @ (r @ (b-origin) + target-b)
        position = tree.nodes.new('GeometryNodeInputPosition')
        split = tree.nodes.new('ShaderNodeSeparateXYZ')
        tree.links.new(position.outputs['Position'], split.inputs['Vector'])
        combined = tree.nodes.new('ShaderNodeCombineXYZ')
        for axis, letter in enumerate('XYZ'):
            terms = [math(tree, 'MULTIPLY', split.outputs[c], local_r[axis, j])
                     for j, c in enumerate('XYZ')]
            value = math(tree, 'ADD', math(tree, 'ADD', terms[0], terms[1]), terms[2])
            value = math(tree, 'ADD', value, local_t[axis])
            tree.links.new(value, combined.inputs[letter])
        difference = tree.nodes.new('ShaderNodeVectorMath'); difference.operation = 'SUBTRACT'
        tree.links.new(combined.outputs['Vector'], difference.inputs[0])
        tree.links.new(position.outputs['Position'], difference.inputs[1])
        parameter = tree.nodes.new('GeometryNodeSplineParameter')
        ramp = tree.nodes.new('ShaderNodeMapRange'); ramp.interpolation_type = 'SMOOTHERSTEP'; ramp.clamp = True
        ramp.inputs['From Min'].default_value = START; ramp.inputs['From Max'].default_value = END
        ramp.inputs['To Min'].default_value = 0.; ramp.inputs['To Max'].default_value = 1.
        tree.links.new(parameter.outputs['Factor'], ramp.inputs['Value'])
        first = tree.nodes.new('GeometryNodeCurveEndpointSelection')
        last = tree.nodes.new('GeometryNodeCurveEndpointSelection')
        for node, first_count, last_count in ((first, 3, 0), (last, 0, 3)):
            node.inputs['Start Size'].default_value = first_count
            node.inputs['End Size'].default_value = last_count
            if node.inputs['Start Size'].default_value != first_count or node.inputs['End Size'].default_value != last_count:
                raise RuntimeError('Native endpoint selection count did not persist')
        weight = math(tree, 'MAXIMUM', ramp.outputs['Result'], last.outputs['Selection'])
        weight = math(tree, 'MULTIPLY', weight, math(tree, 'SUBTRACT', 1., first.outputs['Selection']))
        offset = tree.nodes.new('ShaderNodeVectorMath'); offset.operation = 'SCALE'
        tree.links.new(difference.outputs['Vector'], offset.inputs[0]); tree.links.new(weight, offset.inputs['Scale'])
        selection = tree.nodes.new('ShaderNodeMath'); selection.operation = 'COMPARE'
        tree.links.new(roles, selection.inputs[0]); selection.inputs[1].default_value = role; selection.inputs[2].default_value = .1
        change = tree.nodes.new('GeometryNodeSetPosition'); change.name = 'R11 rigid gather transport ' + side
        tree.links.new(source, change.inputs['Geometry']); tree.links.new(selection.outputs[0], change.inputs['Selection'])
        tree.links.new(offset.outputs['Vector'], change.inputs['Offset']); source = change.outputs['Geometry']
        rows.append({'side': side, 'role': role, 'localRotation': local_r.tolist(), 'localTranslation': local_t.tolist()})
    change.name = STAGE; tree.links.new(source, destination)
    tree.update_tag(); obj.update_tag(); bpy.context.view_layer.update()
    return {'oldTreeRetained': old.name, 'newTree': tree.name, 'stage': STAGE,
            'beforeExistingHeadContact': True, 'firstThreePointsWeight': 0., 'lastThreePointsWeight': 1.,
            'interiorNativeSplineFactorRamp': [START, END], 'interpolation': 'SMOOTHERSTEP',
            'rows': rows, 'rawDonorGuidesUnchanged': True}


def apply(transforms, observer, arrays):
    import bpy
    if bpy.data.objects['Character_Rig'].data.pose_position != 'REST':
        raise RuntimeError('Rigid installation requires REST')
    donor = bpy.data.objects[MAIN]
    if not donor.get('R11_reference_detail_v10') or donor.get('R11_reference_tie_transport'):
        raise RuntimeError('A fresh saved v10 groom without tie transport is required')
    rows = []
    for side in ('positive', 'negative'):
        specification = transforms[side]; validate_transform(specification)
        names = [TAILS[side]] + TIES[side]
        for index, name in enumerate(names):
            obj = bpy.data.objects[name]
            if getattr(obj.data, 'shape_keys', None):
                raise RuntimeError('Tail/ring shape keys need explicit transport: ' + name)
            if any(m.show_render or m.show_viewport for m in obj.modifiers):
                raise RuntimeError('Active tail/ring deformation must be retired explicitly first: ' + name)
            prefix = 'tail_' + side if index == 0 else 'tie_%s_%s' % (side, 'a' if index == 1 else 'b')
            old = obj.data; local = observer['coordinates'](obj)
            matrix = np.asarray(obj.matrix_world, np.float64)
            world = observer['world_positions'](local, matrix)
            desired = transform(world, specification)
            result = observer['world_positions'](desired, np.asarray(obj.matrix_world.inverted())).astype(np.float32)
            attrs = raw_attribute_contract(old)
            old.use_fake_user = True; obj.data = old.copy(); obj.data.name = old.name + ' reference tie transport'
            items, field = (obj.data.position_data, 'vector') if obj.type == 'CURVES' else (obj.data.vertices, 'co')
            items.foreach_set(field, result.ravel()); obj.data.update_tag(); obj.update_tag()
            if raw_attribute_contract(obj.data) != attrs:
                raise RuntimeError('Non-position attribute changed during rigid transport: ' + name)
            actual = observer['world_positions'](observer['coordinates'](obj), obj.matrix_world)
            residual = float(np.linalg.norm(actual-desired, axis=1).max())
            if residual > 2e-6:
                raise RuntimeError('Actual stored rigid transport differs from plan: ' + name)
            arrays[prefix+'_rigid_before_world'] = world
            arrays[prefix+'_rigid_desired_world'] = desired
            arrays[prefix+'_rigid_after_world'] = actual
            row = {'object': name, 'retainedData': old.name, 'newData': obj.data.name,
                   'maximumRigidResidualMeters': residual, 'preservedNonPositionAttributes': attrs}
            if obj.type == 'CURVES':
                lengths = np.asarray([c.points_length for c in obj.data.curves], np.int32)
                starts = np.asarray([c.first_point_index for c in obj.data.curves], np.int32)
                if len(starts) != 2480 or not np.all(lengths == 35):
                    raise RuntimeError('Tail topology changed')
                old_segments = np.diff(world.reshape(-1, 35, 3), axis=1)
                new_segments = np.diff(actual.reshape(-1, 35, 3), axis=1)
                length_error = float(np.abs(np.linalg.norm(old_segments, axis=2)-np.linalg.norm(new_segments, axis=2)).max())
                expected_segments = old_segments @ np.asarray(specification['rotationWorld']).T
                vector_error = float(np.linalg.norm(new_segments-expected_segments, axis=2).max())
                if max(length_error, vector_error) > 3e-6:
                    raise RuntimeError('Rigid tail lengths/pin-relative vectors changed')
                row.update(curves=len(starts), everyNativePointRigidlyMoved=True,
                           firstThreePinsRigidlyMoved=True, maximumSegmentLengthChangeMeters=length_error,
                           maximumSegmentVectorRotationResidualMeters=vector_error)
                obj['R11_tail_dynamics_valid'] = False
                if obj.get('R11_tail_rest_shape'):
                    obj['R11_tail_rest_shape_before_reference_tie_transport'] = obj['R11_tail_rest_shape']
                obj['R11_tail_rest_shape'] = 'Existing C-tail shape rigidly transported to original-reference low tie; all first-three pins transported together. Previous dynamics/cache invalid.'
            rows.append(row)
    bpy.context.view_layer.update()
    stage = install_gather(donor, transforms)
    donor['R11_reference_tie_transport'] = True
    bpy.context.scene['R11_tail_dynamics_valid'] = False
    return {'rigidObjects': rows, 'gatherStage': stage, 'dynamicsValid': False,
            'oldCachesRetainedButNotReenabled': True,
            'transformValidation': {s: validate_transform(transforms[s]) for s in transforms},
            'appearanceAccepted': False}
