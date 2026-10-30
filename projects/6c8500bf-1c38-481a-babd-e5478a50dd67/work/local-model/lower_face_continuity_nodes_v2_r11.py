"""Native fixed-L3 lookup builder for the R11 lower-face continuity field.

The caller validates the source, arrays, protected surfaces and topology,
creates the public zero-displacement shape key, and owns evaluation, save and rollback
after a successful return. This module never evaluates a mesh or saves a file.
Existing shape keys, node trees and modifier properties are not edited.
"""
import ast
import json
from pathlib import Path
import runpy

import numpy as np


GROUP = 'R11 lower face profile continuity v2'
ATTRIBUTE = 'R11_lower_face_continuity_v2_local_offset'
CONTROL = 'R11 lower face contour v2'
OFF_AXIS_TOLERANCE_WORLD = 2e-9


def _constant(filename, name):
    """Read actual helper names without running any helper's installation."""
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets):
            value = ast.literal_eval(statement.value)
            if not isinstance(value, str): break
            return value
    raise RuntimeError('Expected explicit helper constant ' + filename + '/' + name)


def _name_matches(name, expected):
    return name == expected or name.startswith(expected + '.')


def _stack_plan(head):
    """Use only the reviewed neutral stack; reject unknown lower-face inputs."""
    jaw_name = _constant('adult_jaw_open_r11.py', 'MODIFIER')
    jaw_key = _constant('adult_jaw_open_r11.py', 'KEY')
    known_groups = {
        _constant('eye_margin_refine_r11.py', 'PREFIX') + ' attributes only': 'eye attributes',
        _constant('upper_lid_relief_nodes_r11.py', 'GROUP'): 'upper-lid relief',
        _constant('adult_blink_corner_nodes_r11.py', 'GROUP'): 'canthal correction',
        jaw_name: 'jaw opening',
        _constant('lower_face_continuity_nodes_r11.py', 'GROUP'): 'preserved disabled v1 continuity',
        _constant('adult_blink_midpath_nodes_r11.py', 'GROUP'): 'middle blink inner contact',
        _constant('upper_lid_roll_nodes_r11.py', 'GROUP'): 'neutral upper-lid roll',
        _constant('adult_neutral_eye_outerends_nodes_r11.py', 'GROUP'): 'neutral outer eye ends',
    }
    stack = list(head.modifiers)
    multires = [i for i, item in enumerate(stack) if item.type == 'MULTIRES']
    if len(multires) != 1:
        raise RuntimeError('Exactly one reviewed L3 Multires input is required')
    multi_index = multires[0]; multi = stack[multi_index]
    if multi.levels != 3 or multi.render_levels != 3 or not multi.show_viewport or not multi.show_render:
        raise RuntimeError('Viewport and render must both use the verified L3 surface')
    if any(item.type in {'NODES', 'ARMATURE'} for item in stack[:multi_index]):
        raise RuntimeError('Unverified geometry/armature dependency precedes the L3 input')
    candidates = [i for i, item in enumerate(stack) if i > multi_index and item.type in {'NODES', 'ARMATURE'}]
    insertion_index = min(candidates) if candidates else len(stack)
    # The observed IDs are those of the displayed head after its existing Mask.
    # Other surface deformers here would become an unreviewed upstream input.
    for item in stack[multi_index + 1:insertion_index]:
        if item.type != 'MASK' or not item.show_viewport or not item.show_render:
            raise RuntimeError('Only the verified active head Mask may precede the new L3 field: ' + item.name)
    jaw_modifiers = []
    for index, item in enumerate(stack):
        if item.type == 'NODES':
            group = item.node_group
            if group is None:
                raise RuntimeError('Existing geometry modifier has no reviewed node group: ' + item.name)
            if _name_matches(group.name, GROUP):
                raise RuntimeError('Retain the existing lower-face continuity candidate before rebuilding')
            roles = [role for name, role in known_groups.items() if _name_matches(group.name, name)]
            if len(roles) != 1:
                raise RuntimeError('Unverified GN may already change the observed lower face; validate its dependency first: ' + group.name)
            if index < insertion_index:
                raise RuntimeError('An existing GN dependency would precede the new continuity field')
            if roles[0] == 'jaw opening': jaw_modifiers.append(item)
        elif index >= insertion_index and item.type not in {'ARMATURE', 'WEIGHTED_NORMAL', 'NORMAL_EDIT'}:
            raise RuntimeError('Unverified downstream topology/surface modifier: ' + item.name)
    if jaw_modifiers:
        keys = head.data.shape_keys
        if keys is None or jaw_key not in keys.key_blocks or abs(keys.key_blocks[jaw_key].value) > 1e-7:
            raise RuntimeError('The actual jaw control must be neutral before installing the observed field')
    return insertion_index, jaw_modifiers, jaw_name


def build(head, full_vertex_count, original_ids, world_offsets, head_matrix_world):
    """Return (modifier, group, lookup_object, lookup_mesh), without evaluation.

    ``world_offsets`` has shape (len(original_ids), 3) in the supplied observed
    head frame. Conversion must produce only local-Y motion; substantive X/Z
    components are rejected instead of being silently discarded. The caller
    must create ``head.data.shape_keys.key_blocks[CONTROL]`` before calling; the driver clamps it to [0,1].
    """
    import bpy

    if head.type != 'MESH' or head.library or not head.data.shape_keys or CONTROL not in head.data.shape_keys.key_blocks:
        raise RuntimeError('A local mesh head and caller-created zero-displacement continuity shape key are required')
    if not np.isfinite(float(head.data.shape_keys.key_blocks[CONTROL].value)):
        raise ValueError('Continuity control must be finite')
    if isinstance(full_vertex_count, bool) or int(full_vertex_count) != full_vertex_count or full_vertex_count <= 0:
        raise ValueError('A positive complete L3 vertex count is required')
    full_vertex_count = int(full_vertex_count)
    ids = np.asarray(original_ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Original evaluated vertex IDs must be a nonempty integer vector')
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= full_vertex_count or len(np.unique(ids)) != len(ids):
        raise ValueError('Original evaluated vertex IDs must be unique and in range')
    offsets = np.asarray(world_offsets, dtype=float)
    if offsets.shape != (len(ids), 3) or not np.isfinite(offsets).all():
        raise ValueError('One finite world-space offset vector per original vertex ID is required')
    matrix = np.asarray(head_matrix_world, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-10):
        raise ValueError('A finite affine observed head matrix is required')
    if not np.allclose(np.asarray(head.matrix_world, float), matrix, rtol=0., atol=2e-7):
        raise RuntimeError('Current head frame differs from the supplied verified frame')
    linear = matrix[:3, :3]
    if abs(np.linalg.det(linear)) < 1e-12:
        raise ValueError('Observed head frame is singular')
    local = offsets @ np.linalg.inv(linear).T
    rejected = local.copy(); rejected[:, 1] = 0.
    off_axis_error = float(np.linalg.norm(rejected @ linear.T, axis=1).max())
    if off_axis_error > OFF_AXIS_TOLERANCE_WORLD:
        raise RuntimeError('Continuity field contains non-Y local motion: ' + str(off_axis_error))
    local[:, [0, 2]] = 0.
    local = local.astype(np.float32)
    conversion_error = float(np.linalg.norm(local.astype(float) @ linear.T - offsets, axis=1).max())
    if conversion_error > OFF_AXIS_TOLERANCE_WORLD:
        raise RuntimeError('Stored local-Y lookup does not reproduce the verified vectors: ' + str(conversion_error))
    if not np.any(local[:, 1] != 0.):
        raise ValueError('Continuity field has no nonzero local-Y offsets')
    insertion_index, jaw_modifiers, jaw_name = _stack_plan(head)
    dense = np.zeros((full_vertex_count, 3), np.float32); dense[ids] = local
    socket_api = runpy.run_path(str(Path(__file__).with_name('donor_groom_finish_v3_r11.py')))
    geometry_socket = socket_api['geometry_socket']
    modifier = group = cache = mesh = None
    try:
        mesh = bpy.data.meshes.new(GROUP + ' lookup')
        mesh.vertices.add(full_vertex_count)
        mesh.vertices.foreach_set('co', np.zeros(full_vertex_count * 3, np.float32))
        attribute = mesh.attributes.new(ATTRIBUTE, 'FLOAT_VECTOR', 'POINT')
        attribute.data.foreach_set('vector', dense.ravel()); mesh.update()
        cache = bpy.data.objects.new(GROUP + ' lookup', mesh)
        bpy.context.scene.collection.objects.link(cache)
        cache.hide_render = True; cache.hide_set(True)
        group = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        nodes = group.nodes; links = group.links
        source = nodes.new('NodeGroupInput'); output = nodes.new('NodeGroupOutput')
        geometry = geometry_socket(source.outputs)
        info = nodes.new('GeometryNodeObjectInfo'); info.transform_space = 'RELATIVE'
        info.inputs['Object'].default_value = cache; info.inputs['As Instance'].default_value = False
        named = nodes.new('GeometryNodeInputNamedAttribute'); named.data_type = 'FLOAT_VECTOR'
        named.inputs['Name'].default_value = ATTRIBUTE
        index = nodes.new('GeometryNodeInputIndex')
        sample = nodes.new('GeometryNodeSampleIndex'); sample.data_type = 'FLOAT_VECTOR'
        sample.domain = 'POINT'; sample.clamp = False
        links.new(info.outputs['Geometry'], geometry_socket(sample.inputs))
        links.new(named.outputs['Attribute'], sample.inputs['Value'])
        links.new(index.outputs['Index'], sample.inputs['Index'])
        control = nodes.new('ShaderNodeValue'); control.name = 'R11 lower-face continuity normalized control'
        control.outputs[0].default_value = 0.
        driver = control.outputs[0].driver_add('default_value').driver; driver.type = 'SCRIPTED'
        driver.expression = 'min(max(c,0.0),1.0)'
        variable = driver.variables.new(); variable.name = 'c'; variable.type = 'SINGLE_PROP'
        target = variable.targets[0]; target.id_type = 'KEY'; target.id = head.data.shape_keys
        target.data_path = head.data.shape_keys.key_blocks[CONTROL].path_from_id('value')
        offset = nodes.new('ShaderNodeVectorMath'); offset.operation = 'SCALE'
        links.new(sample.outputs['Value'], offset.inputs[0])
        links.new(control.outputs[0], offset.inputs['Scale'])
        length = nodes.new('ShaderNodeVectorMath'); length.operation = 'LENGTH'
        links.new(sample.outputs['Value'], length.inputs[0])
        selected = nodes.new('ShaderNodeMath'); selected.operation = 'GREATER_THAN'; selected.inputs[1].default_value = 0.
        links.new(length.outputs['Value'], selected.inputs[0])
        moved = nodes.new('GeometryNodeSetPosition'); moved.name = 'R11 fixed-L3 local-Y lower-face continuity'
        links.new(geometry, geometry_socket(moved.inputs))
        links.new(selected.outputs[0], moved.inputs['Selection'])
        links.new(offset.outputs['Vector'], moved.inputs['Offset'])
        enabled = nodes.new('ShaderNodeMath'); enabled.operation = 'GREATER_THAN'; enabled.inputs[1].default_value = 0.
        links.new(control.outputs[0], enabled.inputs[0])
        passthrough = nodes.new('GeometryNodeSwitch'); passthrough.input_type = 'GEOMETRY'
        links.new(enabled.outputs[0], passthrough.inputs['Switch'])
        links.new(geometry, passthrough.inputs['False'])
        links.new(geometry_socket(moved.outputs), passthrough.inputs['True'])
        links.new(passthrough.outputs['Output'], geometry_socket(output.inputs))
        group['R11_full_vertex_count'] = full_vertex_count
        group['R11_local_axis'] = 'Y only'
        group['R11_vector_conversion_error_world_meters'] = conversion_error
        group['R11_expected_jaw_group'] = jaw_name
        modifier = head.modifiers.new(GROUP, 'NODES'); modifier.node_group = group
        if insertion_index < len(head.modifiers) - 1:
            head.modifiers.move(len(head.modifiers) - 1, insertion_index)
        final_stack = list(head.modifiers); own_index = final_stack.index(modifier)
        if any(i >= own_index for i, item in enumerate(final_stack) if item.type == 'MULTIRES'):
            raise RuntimeError('Continuity field must follow Multires')
        if any(i <= own_index for i, item in enumerate(final_stack) if item.type == 'ARMATURE' or item in jaw_modifiers):
            raise RuntimeError('Continuity field must precede all actual jaw-opening nodes and armatures')
        if any(item.type == 'NODES' for item in final_stack[:own_index]):
            raise RuntimeError('Unreviewed upstream GN dependency before lower-face continuity')
        group.update_tag(); head.update_tag()
        return modifier, group, cache, mesh
    except Exception:
        # The caller cannot roll back handles until a build has returned.
        if modifier is not None: head.modifiers.remove(modifier)
        if group is not None and group.users == 0: bpy.data.node_groups.remove(group)
        if cache is not None: bpy.data.objects.remove(cache, do_unlink=True)
        if mesh is not None and mesh.users == 0: bpy.data.meshes.remove(mesh)
        raise
