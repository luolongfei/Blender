"""Native two-layer GN builder for the current unbound REST eyelid candidate.

Arc uses measured world-delta knots with exact zero endpoints. Inner contact
queries the actual globe and the same arc-stage head geometry, limits movement
to inner-facing masked points, and retains an explicit outer-wall allowance.
The caller owns actual validation, rollback after return, and candidate save.
"""
import json
from pathlib import Path
import runpy

import numpy as np

GROUP = 'R11 middle blink arc and inner contact'
KNOTS = (0., .2, .5, .8, .9, 1.)
MASK_ATTRIBUTE = 'R11_midpath_inner_masks_L_R'
CONTACT_STRENGTH_NODE = 'R11 inner contact strength'
QUERY_SPACE = 'unbound_rest_head_object'
PRIOR_GROUPS = ('R11 upper lid surface relief', 'R11 actual canthal surface path')


def _compatible_geometry_socket():
    module = runpy.run_path(str(Path(__file__).with_name('donor_groom_finish_v3_r11.py')))
    return module['geometry_socket']


def _named_socket(collection, names, socket_type=None, context=''):
    if isinstance(names, str):
        names = (names,)
    found = [s for s in collection if s.name in names and (socket_type is None or s.type == socket_type)]
    if len(found) != 1:
        available = [(s.name, s.type) for s in collection]
        raise RuntimeError('Expected one compatible socket '+context+'/'+repr(tuple(names))+'; got '+repr(available))
    return found[0]


def _ray_source(node):
    return _named_socket(node.inputs, ('Source Position', 'Position', 'Ray Origin'), 'VECTOR', node.bl_idname)


def _hat_expression(knot_index):
    if knot_index not in (1, 2, 3, 4):
        raise ValueError('Only nonzero interior arc knots have a driver')
    a, b, c = KNOTS[knot_index-1:knot_index+2]
    rise = 'max(0.0,min(1.0,(b-'+repr(a)+')/'+repr(b-a)+'))'
    fall = 'max(0.0,min(1.0,('+repr(c)+'-b)/'+repr(c-b)+'))'
    return '('+rise+')**2*(3.0-2.0*('+rise+'))*('+fall+')**2*(3.0-2.0*('+fall+'))'


def _group_matches(modifier, name):
    return modifier.type == 'NODES' and modifier.node_group is not None and (
        modifier.node_group.name == name or modifier.node_group.name.startswith(name+'.'))


def build(head, full_vertex_count, original_ids, arc_fields, inner_masks,
          centers_world, matrix_world, clearance=.00006, minimum_wall=.00008,
          query_eyes=None):
    """Return modifier, group, lookup object and mesh; never save or evaluate.

    arc_fields[side]: (6, ROI point count, 3) world-space delta array.
    inner_masks[side]: Boolean ROI point mask; the two sides must not overlap.
    query_eyes optionally names explicit mesh objects in the same current
    unbound head-object query frame. This version still rejects every head
    ARMATURE/recorded skin-bind migration; it does not guess posed mappings.
    """
    import bpy

    if isinstance(full_vertex_count, bool) or int(full_vertex_count) != full_vertex_count or full_vertex_count <= 0:
        raise ValueError('Positive complete evaluated vertex count required')
    full_vertex_count = int(full_vertex_count)
    ids = np.asarray(original_ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Nonempty integer original evaluated point IDs required')
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= full_vertex_count or len(np.unique(ids)) != len(ids):
        raise ValueError('Original evaluated point IDs must be unique and in range')
    matrix = np.asarray(matrix_world, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], (0., 0., 0., 1.), rtol=0., atol=1e-10):
        raise ValueError('Finite affine head world matrix required')
    scale = float(np.linalg.norm(matrix[:3, 0]))
    if scale <= 0. or not np.allclose(matrix[:3, :3], np.eye(3)*scale, rtol=0., atol=1e-7):
        raise RuntimeError('Observed upright positive uniform REST head transform required')
    if not np.allclose(np.asarray(head.matrix_world, float), matrix, rtol=0., atol=1e-7):
        raise RuntimeError('Supplied head transform does not match the current object')
    clearance, minimum_wall = float(clearance), float(minimum_wall)
    if not np.isfinite((clearance, minimum_wall)).all() or clearance <= 0. or minimum_wall <= 0.:
        raise ValueError('Positive finite actual-globe clearance and outer-wall allowance required')
    if head.type != 'MESH' or head.data.shape_keys is None:
        raise RuntimeError('Actual head mesh and existing blink keys required')
    if any(m.type == 'ARMATURE' for m in head.modifiers) or 'R11_head_neck_skin_bind' in bpy.context.scene:
        raise RuntimeError('This builder supports only the unbound REST head object; mapped pre-armature eye query geometry is not implemented')
    if any(_group_matches(m, GROUP) for m in head.modifiers):
        raise RuntimeError('Retain the existing middle blink candidate before rebuilding')
    keys = head.data.shape_keys
    fields, masks, centers, eyes = {}, {}, {}, {}
    if query_eyes is not None and set(query_eyes) != {'L', 'R'}:
        raise ValueError('Explicit query eyes must provide L and R mesh objects')
    for side in ('L', 'R'):
        if 'R11 blink '+side not in keys.key_blocks:
            raise RuntimeError('Missing actual blink control '+side)
        values = np.asarray(arc_fields[side], float)
        if values.shape != (len(KNOTS), len(ids), 3) or not np.isfinite(values).all():
            raise ValueError('Six finite ROI world-delta arrays required for '+side)
        if np.any(values[0] != 0.) or np.any(values[-1] != 0.):
            raise ValueError('Neutral and full-closed Arc fields must be exactly zero for '+side)
        fields[side] = values
        mask = np.asarray(inner_masks[side])
        if mask.shape != ids.shape or mask.dtype.kind != 'b':
            raise ValueError('One Boolean inner-contact ROI mask required for '+side)
        masks[side] = mask
        center = np.asarray(centers_world[side], float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Finite actual eye centre required for '+side)
        centers[side] = (center-matrix[:3, 3])/scale
        eye = query_eyes[side] if query_eyes is not None else bpy.data.objects.get('R11 Adult sclera '+side)
        if isinstance(eye, str):
            eye = bpy.data.objects.get(eye)
        if eye is None or eye.type != 'MESH' or eye == head:
            raise RuntimeError('Independent actual eye-query mesh required for '+side)
        eyes[side] = eye
    if np.any(masks['L'] & masks['R']):
        raise ValueError('Left and right inner-contact masks must not overlap')
    if eyes['L'] == eyes['R']:
        raise RuntimeError('Distinct actual left and right query meshes required')
    original_modifiers = list(head.modifiers)
    # Current supported input has no Armature. Appending puts this after the
    # already installed relief/canthal fields without moving any old modifier.
    insertion_index = len(original_modifiers)
    geometry_socket = _compatible_geometry_socket()
    modifier = group = cache = mesh = None
    try:
        mesh = bpy.data.meshes.new(GROUP+' lookup')
        mesh.vertices.add(full_vertex_count)
        mesh.vertices.foreach_set('co', np.zeros(full_vertex_count*3, np.float32))
        attribute_names = {}
        for side in ('L', 'R'):
            for knot in range(1, len(KNOTS)-1):
                name = 'R11_midpath_arc_'+side+'_k'+str(knot)
                dense = np.zeros((full_vertex_count, 3), np.float32)
                dense[ids] = (fields[side][knot]/scale).astype(np.float32)
                if not np.isfinite(dense).all():
                    raise ValueError('Arc field overflows native float storage')
                mesh.attributes.new(name, 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', dense.ravel())
                attribute_names[(side, knot)] = name
        dense_masks = np.zeros((full_vertex_count, 3), np.float32)
        dense_masks[ids, 0] = masks['L']
        dense_masks[ids, 1] = masks['R']
        mesh.attributes.new(MASK_ATTRIBUTE, 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', dense_masks.ravel())
        mesh.update()
        cache = bpy.data.objects.new(GROUP+' lookup', mesh)
        bpy.context.scene.collection.objects.link(cache)
        cache.hide_render = True
        cache.hide_set(True)
        group = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        nodes, links = group.nodes, group.links
        source = nodes.new('NodeGroupInput')
        output = nodes.new('NodeGroupOutput')
        input_geometry = geometry_socket(source.outputs)
        index = nodes.new('GeometryNodeInputIndex')

        def math(operation, a, b=None, label=None):
            node = nodes.new('ShaderNodeMath')
            node.operation = operation
            if label:
                node.label = label
            for position, value in enumerate((a, b)):
                if value is None:
                    continue
                if isinstance(value, (float, int)):
                    node.inputs[position].default_value = value
                else:
                    links.new(value, node.inputs[position])
            return node.outputs[0]

        def vector_math(operation, a, b=None, scale_socket=None, label=None):
            node = nodes.new('ShaderNodeVectorMath')
            node.operation = operation
            if label:
                node.label = label
            for position, value in enumerate((a, b)):
                if value is None:
                    continue
                if isinstance(value, (tuple, list, np.ndarray)):
                    node.inputs[position].default_value = value
                else:
                    links.new(value, node.inputs[position])
            if scale_socket is not None:
                links.new(scale_socket, _named_socket(node.inputs, 'Scale', 'VALUE', 'VectorMath/SCALE'))
            return _named_socket(node.outputs, 'Vector', 'VECTOR', 'VectorMath')

        lookup = nodes.new('GeometryNodeObjectInfo')
        lookup.transform_space = 'RELATIVE'
        lookup.inputs['Object'].default_value = cache
        lookup.inputs['As Instance'].default_value = False
        lookup_geometry = geometry_socket(lookup.outputs)

        def cached_vector(name, label):
            named = nodes.new('GeometryNodeInputNamedAttribute')
            named.data_type = 'FLOAT_VECTOR'
            named.inputs['Name'].default_value = name
            sample = nodes.new('GeometryNodeSampleIndex')
            sample.data_type = 'FLOAT_VECTOR'
            sample.domain = 'POINT'
            sample.clamp = False
            sample.label = label
            links.new(lookup_geometry, geometry_socket(sample.inputs))
            links.new(named.outputs['Attribute'], sample.inputs['Value'])
            links.new(index.outputs['Index'], sample.inputs['Index'])
            return sample.outputs['Value']

        arc_vectors = []
        for side in ('L', 'R'):
            for knot in range(1, len(KNOTS)-1):
                value = nodes.new('ShaderNodeValue')
                value.name = 'R11 Arc '+side+' knot '+str(KNOTS[knot])
                value.label = 'Arc / '+side+' measured knot '+str(KNOTS[knot])
                value.outputs[0].default_value = 0.
                driver = value.outputs[0].driver_add('default_value').driver
                driver.type = 'SCRIPTED'
                variable = driver.variables.new()
                variable.name = 'b'
                variable.type = 'SINGLE_PROP'
                variable.targets[0].id_type = 'KEY'
                variable.targets[0].id = keys
                variable.targets[0].data_path = keys.key_blocks['R11 blink '+side].path_from_id('value')
                driver.expression = _hat_expression(knot)
                field = cached_vector(attribute_names[(side, knot)], 'Arc / cached local delta '+side)
                arc_vectors.append(vector_math('SCALE', field, scale_socket=value.outputs[0]))
        arc_delta = arc_vectors[0]
        for value in arc_vectors[1:]:
            arc_delta = vector_math('ADD', arc_delta, value)
        arc = nodes.new('GeometryNodeSetPosition')
        arc.name = 'R11 Arc stage'
        arc.label = 'Arc / measured intermediate aperture'
        links.new(input_geometry, geometry_socket(arc.inputs))
        links.new(arc_delta, arc.inputs['Offset'])
        stage_one = geometry_socket(arc.outputs)

        # Position and point normals are explicitly sampled from the same
        # stage-one geometry that both eyes' outer-wall rays will query.
        position_field = nodes.new('GeometryNodeInputPosition')
        normal_field = nodes.new('GeometryNodeInputNormal')

        def stage_vector(field, label):
            sample = nodes.new('GeometryNodeSampleIndex')
            sample.data_type = 'FLOAT_VECTOR'
            sample.domain = 'POINT'
            sample.clamp = False
            sample.label = label
            links.new(stage_one, geometry_socket(sample.inputs))
            links.new(field, sample.inputs['Value'])
            links.new(index.outputs['Index'], sample.inputs['Index'])
            return sample.outputs['Value']

        current_position = stage_vector(position_field.outputs['Position'], 'Inner contact / Arc-stage position')
        current_normal = stage_vector(normal_field.outputs['Normal'], 'Inner contact / Arc-stage point normal')
        unit_normal = vector_math('NORMALIZE', current_normal)
        position_split = nodes.new('ShaderNodeSeparateXYZ')
        normal_split = nodes.new('ShaderNodeSeparateXYZ')
        mask_split = nodes.new('ShaderNodeSeparateXYZ')
        links.new(current_position, position_split.inputs[0])
        links.new(unit_normal, normal_split.inputs[0])
        links.new(cached_vector(MASK_ATTRIBUTE, 'Inner contact / disjoint masks'), mask_split.inputs[0])
        strength_node = nodes.new('ShaderNodeValue')
        strength_node.name = CONTACT_STRENGTH_NODE
        strength_node.label = 'Inner contact / installer may set 0 for actual Arc-only capture'
        strength_node.outputs[0].default_value = 1.
        strength = math('MINIMUM', math('MAXIMUM', strength_node.outputs[0], 0.), 1.)
        inner_facing = math('GREATER_THAN', normal_split.outputs['Y'], .1)

        def cast(geometry, origin, label, tagged_side=None):
            node = nodes.new('GeometryNodeRaycast')
            node.data_type = 'FLOAT'
            node.label = label
            if tagged_side is not None:
                node.name = 'R11 Inner contact actual sclera Raycast '+tagged_side
                node['R11_prearm_sclera_raycast'] = tagged_side
            links.new(geometry, geometry_socket(node.inputs))
            links.new(origin, _ray_source(node))
            _named_socket(node.inputs, ('Ray Direction', 'Direction'), 'VECTOR', node.bl_idname).default_value = (0., 1., 0.)
            _named_socket(node.inputs, ('Ray Length', 'Length'), 'VALUE', node.bl_idname).default_value = .08/scale
            hit_position = _named_socket(node.outputs, ('Hit Position', 'Position'), 'VECTOR', node.bl_idname)
            hit = _named_socket(node.outputs, ('Is Hit', 'Hit'), 'BOOLEAN', node.bl_idname)
            split = nodes.new('ShaderNodeSeparateXYZ')
            links.new(hit_position, split.inputs[0])
            return split.outputs['Y'], hit

        y_deltas = []
        for side, channel in (('L', 'X'), ('R', 'Y')):
            origin = nodes.new('ShaderNodeCombineXYZ')
            origin.label = 'Inner contact / front ray origin '+side
            links.new(position_split.outputs['X'], origin.inputs['X'])
            origin.inputs['Y'].default_value = float(centers[side][1]-.04/scale)
            links.new(position_split.outputs['Z'], origin.inputs['Z'])
            globe = nodes.new('GeometryNodeObjectInfo')
            globe.transform_space = 'RELATIVE'
            globe.name = 'R11 Inner contact query sclera '+side
            globe['R11_prearm_sclera_side'] = side
            globe.label = 'Inner contact / actual query eye in unbound head-local space '+side
            globe.inputs['Object'].default_value = eyes[side]
            globe.inputs['As Instance'].default_value = False
            globe_y, globe_hit = cast(geometry_socket(globe.outputs), origin.outputs['Vector'], 'Inner contact / actual globe '+side, tagged_side=side)
            outer_y, outer_hit = cast(stage_one, origin.outputs['Vector'], 'Inner contact / same Arc-stage outer skin '+side)
            requested_y = math('SUBTRACT', globe_y, clearance/scale)
            outer_limit = math('ADD', outer_y, minimum_wall/scale)
            bounded_y = math('MINIMUM', position_split.outputs['Y'], math('MAXIMUM', requested_y, outer_limit))
            y_delta = math('SUBTRACT', bounded_y, position_split.outputs['Y'])
            selected = math('MULTIPLY', math('GREATER_THAN', mask_split.outputs[channel], .5), inner_facing)
            selected = math('MULTIPLY', selected, math('LESS_THAN', position_split.outputs['Y'], float(centers[side][1])))
            selected = math('MULTIPLY', selected, math('MULTIPLY', globe_hit, outer_hit))
            y_deltas.append(math('MULTIPLY', math('MULTIPLY', y_delta, selected), strength))
        total_y = math('ADD', y_deltas[0], y_deltas[1])
        contact_offset = nodes.new('ShaderNodeCombineXYZ')
        links.new(total_y, contact_offset.inputs['Y'])
        contact = nodes.new('GeometryNodeSetPosition')
        contact.name = 'R11 Inner contact stage'
        contact.label = 'Inner contact / both sides query unchanged Arc stage; Y only'
        links.new(stage_one, geometry_socket(contact.inputs))
        links.new(math('LESS_THAN', total_y, 0.), contact.inputs['Selection'])
        links.new(contact_offset.outputs['Vector'], contact.inputs['Offset'])
        links.new(geometry_socket(contact.outputs), geometry_socket(output.inputs))
        group['R11_layer_labels'] = json.dumps(['Arc', 'Inner contact'])
        group['R11_arc_knots'] = json.dumps(list(KNOTS))
        group['R11_arc_zero_endpoints'] = True
        group['R11_contact_strength_node'] = CONTACT_STRENGTH_NODE
        group['R11_query_space'] = QUERY_SPACE
        group['R11_query_eye_objects'] = json.dumps({side: eyes[side].name for side in ('L', 'R')})
        group['R11_explicit_query_eyes'] = query_eyes is not None
        group['R11_current_armature_support'] = False
        group['R11_full_vertex_count'] = full_vertex_count
        group['R11_roi_vertex_count'] = len(ids)
        group['R11_clearance_world_meters'] = clearance
        group['R11_minimum_wall_world_meters'] = minimum_wall
        group['R11_inner_normal_y_threshold'] = .1
        group['R11_rest_head_matrix_world'] = json.dumps(matrix.tolist())
        group['R11_inner_contact_axes'] = 'Head-local X/Z unchanged; only nonpositive Y offset'
        modifier = head.modifiers.new(GROUP, 'NODES')
        modifier.node_group = group
        if list(head.modifiers)[:insertion_index] != original_modifiers or list(head.modifiers)[-1] != modifier:
            raise RuntimeError('Existing modifier relative order changed while appending the new candidate')
        if any(m.type == 'ARMATURE' for m in head.modifiers):
            raise RuntimeError('Head armature appeared during the unbound REST build')
        new_index = list(head.modifiers).index(modifier)
        if any(i >= new_index for i, item in enumerate(head.modifiers) if any(_group_matches(item, name) for name in PRIOR_GROUPS)):
            raise RuntimeError('Middle blink correction must follow existing relief and canthal geometry')
        group.update_tag()
        head.update_tag()
        return modifier, group, cache, mesh
    except Exception:
        # Until a successful return, only this builder owns these handles.
        if modifier is not None:
            head.modifiers.remove(modifier)
        if group is not None and group.users == 0:
            bpy.data.node_groups.remove(group)
        if cache is not None:
            bpy.data.objects.remove(cache, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise
