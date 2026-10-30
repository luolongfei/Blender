"""Native GN builder for the open-eye R11 upper-lid roll candidate.

The caller supplies a verified fixed-topology ROI, creates the public zero-delta control
key, and owns actual geometry validation, rollback after return and save.
This module neither evaluates Blender geometry nor changes existing keys. A
single Set Position keeps all queries on the unmodified modifier input.
"""
import json
from pathlib import Path
import runpy

import numpy as np


GROUP = 'R11 refined upper lid roll surface'
ATTRIBUTE = 'R11_refined_upper_roll_weights_L_R'
CONTROL = 'R11 refined upper lid roll'
CORNER_GROUP = 'R11 actual canthal surface path'


def _compatible_sockets():
    api = runpy.run_path(str(Path(__file__).with_name('donor_groom_finish_v3_r11.py')))
    return api['geometry_socket'], api['sample_position']


def _is_corner(modifier):
    if modifier.type != 'NODES' or modifier.node_group is None:
        return False
    return (modifier.node_group.name == CORNER_GROUP or
            modifier.node_group.name.startswith(CORNER_GROUP + '.'))


def build(head, full_vertex_count, original_ids, weights, scale,
          maximum=.00065, protected_gap=.00045, fade_end=.5, minimum_wall=.00020):
    """Return (modifier, group, cache_object, cache_mesh), with no save.

    ``weights[side]`` is one finite [0, 1] weight per ``original_ids`` entry.
    IDs address the complete evaluated mesh entering this modifier. The dense
    point attribute stores L in X and R in Y; its Z component stays zero.
    World distances are converted into the uniformly scaled head local space.
    """
    import bpy

    if isinstance(full_vertex_count, bool) or int(full_vertex_count) != full_vertex_count or full_vertex_count <= 0:
        raise ValueError('A positive complete evaluated vertex count is required')
    full_vertex_count = int(full_vertex_count)
    scale = float(scale); maximum = float(maximum)
    protected_gap = float(protected_gap); fade_end = float(fade_end)
    if not np.isfinite([scale, maximum, protected_gap, fade_end, minimum_wall]).all() or scale <= 0. or maximum <= 0. or protected_gap < 0. or minimum_wall <= 0. or not 0. < fade_end <= 1.:
        raise ValueError('Finite positive scale/maximum and bounded fade are required')
    ids = np.asarray(original_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids):
        raise ValueError('Original evaluated vertex IDs must be a nonempty integer vector')
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= full_vertex_count or len(np.unique(ids)) != len(ids):
        raise ValueError('Original evaluated vertex IDs must be unique and in range')
    dense = np.zeros((full_vertex_count, 3), np.float32)
    for channel, side in enumerate(('L', 'R')):
        values = np.asarray(weights[side], dtype=float)
        if values.shape != ids.shape or not np.isfinite(values).all() or np.any(values < 0.) or np.any(values > 1.):
            raise ValueError('One finite [0, 1] ROI weight per ID is required for ' + side)
        dense[ids, channel] = values.astype(np.float32)
    if np.any((dense[:, 0] > 0.) & (dense[:, 1] > 0.)):
        raise ValueError('Left and right relief domains must be disjoint')
    if not np.any(dense[:, :2] > 0.):
        raise ValueError('Upper-lid relief ROI contains no active vertices')
    if head.type != 'MESH' or not head.data.shape_keys or CONTROL not in head.data.shape_keys.key_blocks:
        raise RuntimeError('Actual mesh, installed blink keys and caller-created zero-delta roll control key required')
    if not np.isfinite(float(head.data.shape_keys.key_blocks[CONTROL].value)) or float(head.data.shape_keys.key_blocks[CONTROL].value) < 0.:
        raise RuntimeError('Relief control must be a finite nonnegative value')
    keys = head.data.shape_keys
    eyes = {}
    for side in ('L', 'R'):
        if 'R11 blink ' + side not in keys.key_blocks:
            raise RuntimeError('Missing actual blink control ' + side)
        eyes[side] = bpy.data.objects.get('R11 Adult sclera ' + side)
        if eyes[side] is None or eyes[side].type != 'MESH' or eyes[side] == head:
            raise RuntimeError('Actual sclera mesh required for ' + side)
    if any(m.type == 'NODES' and m.node_group and
           (m.node_group.name == GROUP or m.node_group.name.startswith(GROUP + '.'))
           for m in head.modifiers):
        raise RuntimeError('Retain the existing upper-lid relief before rebuilding')
    scales = np.linalg.svd(np.asarray(head.matrix_world, float)[:3, :3], compute_uv=False)
    if not np.allclose(scales, scale, rtol=1e-6, atol=1e-8):
        raise RuntimeError('Supplied scale must match the uniform current head transform')
    armatures = [i for i, item in enumerate(head.modifiers) if item.type == 'ARMATURE']
    corners = [i for i, item in enumerate(head.modifiers) if _is_corner(item)]
    if armatures and any(i >= armatures[0] for i in corners):
        raise RuntimeError('Existing canthal correction must already precede the first armature')
    insertion_index = corners[0] if corners else (armatures[0] if armatures else len(head.modifiers))
    if any(i >= insertion_index for i, item in enumerate(head.modifiers) if item.type == 'MULTIRES'):
        raise RuntimeError('Upper-lid relief lookup requires Multires before the insertion point')
    geometry_socket, sample_position = _compatible_sockets()
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
        input_geometry = geometry_socket(source.outputs)
        index = nodes.new('GeometryNodeInputIndex')
        position = nodes.new('GeometryNodeInputPosition')

        def math(operation, a, b=None, label=None):
            node = nodes.new('ShaderNodeMath'); node.operation = operation
            if label: node.label = label
            for number, value in enumerate((a, b)):
                if value is None: continue
                if isinstance(value, (float, int)): node.inputs[number].default_value = value
                else: links.new(value, node.inputs[number])
            return node.outputs[0]

        lookup = nodes.new('GeometryNodeObjectInfo'); lookup.transform_space = 'RELATIVE'
        lookup.inputs['Object'].default_value = cache; lookup.inputs['As Instance'].default_value = False
        named = nodes.new('GeometryNodeInputNamedAttribute'); named.data_type = 'FLOAT_VECTOR'
        named.inputs['Name'].default_value = ATTRIBUTE
        sample = nodes.new('GeometryNodeSampleIndex'); sample.data_type = 'FLOAT_VECTOR'
        sample.domain = 'POINT'; sample.clamp = False
        links.new(lookup.outputs['Geometry'], geometry_socket(sample.inputs))
        links.new(named.outputs['Attribute'], sample.inputs['Value'])
        links.new(index.outputs['Index'], sample.inputs['Index'])
        split = nodes.new('ShaderNodeSeparateXYZ'); links.new(sample.outputs['Value'], split.inputs[0])

        # Resolve the head normal explicitly on the original input geometry,
        # independently of the Normal field sampled on each sclera below.
        head_normal = nodes.new('GeometryNodeInputNormal'); head_normal.label = 'Original input outer-surface normal'
        normal_sample = nodes.new('GeometryNodeSampleIndex'); normal_sample.data_type = 'FLOAT_VECTOR'
        normal_sample.domain = 'POINT'; normal_sample.clamp = False
        links.new(input_geometry, geometry_socket(normal_sample.inputs))
        links.new(head_normal.outputs['Normal'], normal_sample.inputs['Value'])
        links.new(index.outputs['Index'], normal_sample.inputs['Index'])
        unit_normal = nodes.new('ShaderNodeVectorMath'); unit_normal.operation = 'NORMALIZE'
        links.new(normal_sample.outputs['Value'], unit_normal.inputs[0])
        normal_split = nodes.new('ShaderNodeSeparateXYZ'); links.new(unit_normal.outputs['Vector'], normal_split.inputs[0])
        front_facing = math('LESS_THAN', normal_split.outputs['Y'], -.10)

        amplitudes = []
        for side, channel in (('L', 'X'), ('R', 'Y')):
            influence = nodes.new('ShaderNodeValue'); influence.name = 'R11 refined upper-roll open-eye influence ' + side
            influence.outputs[0].default_value = 0.
            driver = influence.outputs[0].driver_add('default_value').driver; driver.type = 'SCRIPTED'
            variable = driver.variables.new(); variable.name = 'r'; variable.type = 'SINGLE_PROP'
            target = variable.targets[0]; target.id_type = 'KEY'; target.id = keys
            target.data_path = keys.key_blocks[CONTROL].path_from_id('value')
            variable = driver.variables.new(); variable.name = 'b'; variable.type = 'SINGLE_PROP'
            target = variable.targets[0]; target.id_type = 'KEY'; target.id = keys
            target.data_path = keys.key_blocks['R11 blink ' + side].path_from_id('value')
            t = 'max(0.0,min(1.0,1.0-b/' + repr(fade_end) + '))'
            driver.expression = ('min(max(r,0.0),1.0)*(' + t + ')**2*(3.0-2.0*(' + t + '))'
                                 ' if b < ' + repr(fade_end) + ' else 0.0')
            requested = math('MULTIPLY', math('MULTIPLY', split.outputs[channel], influence.outputs[0]), maximum / scale)

            globe = nodes.new('GeometryNodeObjectInfo'); globe.transform_space = 'RELATIVE'
            globe.label = 'Actual sclera in head local space ' + side
            globe.inputs['Object'].default_value = eyes[side]; globe.inputs['As Instance'].default_value = False
            nearest = nodes.new('GeometryNodeProximity'); nearest.target_element = 'FACES'
            links.new(globe.outputs['Geometry'], geometry_socket(nearest.inputs))
            links.new(position.outputs['Position'], sample_position(nearest))
            sphere_normal = nodes.new('GeometryNodeInputNormal')
            normal_at_surface = nodes.new('GeometryNodeSampleNearestSurface'); normal_at_surface.data_type = 'FLOAT_VECTOR'
            links.new(globe.outputs['Geometry'], geometry_socket(normal_at_surface.inputs))
            links.new(sphere_normal.outputs['Normal'], normal_at_surface.inputs['Value'])
            links.new(position.outputs['Position'], sample_position(normal_at_surface))
            difference = nodes.new('ShaderNodeVectorMath'); difference.operation = 'SUBTRACT'
            links.new(position.outputs['Position'], difference.inputs[0])
            links.new(nearest.outputs['Position'], difference.inputs[1])
            signed = nodes.new('ShaderNodeVectorMath'); signed.operation = 'DOT_PRODUCT'
            links.new(difference.outputs['Vector'], signed.inputs[0])
            links.new(normal_at_surface.outputs['Value'], signed.inputs[1])
            outside = math('GREATER_THAN', signed.outputs['Value'], 0., 'Move only original outside points ' + side)
            available = math('MAXIMUM', math('SUBTRACT', nearest.outputs['Distance'], protected_gap / scale), 0.)
            bounded = math('MINIMUM', requested, available, 'Dynamic sclera clearance cap ' + side)
            # Query the same unmodified head input from just inside the outer
            # point. Its first +Y exit must be an inner-facing actual surface.
            origin = nodes.new('ShaderNodeVectorMath'); origin.operation = 'ADD'
            links.new(position.outputs['Position'], origin.inputs[0]); origin.inputs[1].default_value = (0., .000001 / scale, 0.)
            ray = nodes.new('GeometryNodeRaycast'); ray.data_type = 'FLOAT'
            links.new(input_geometry, geometry_socket(ray.inputs))
            ray_source = [s for s in ray.inputs if s.name in ('Source Position', 'Position', 'Ray Origin') and s.type == 'VECTOR']
            if len(ray_source) != 1: raise RuntimeError('Actual inner-wall Raycast source socket is ambiguous')
            links.new(origin.outputs['Vector'], ray_source[0])
            ray.inputs['Ray Direction'].default_value = (0., 1., 0.)
            ray.inputs['Ray Length'].default_value = .020 / scale
            hit_normal = nodes.new('ShaderNodeSeparateXYZ'); links.new(ray.outputs['Hit Normal'], hit_normal.inputs[0])
            inner_facing = math('GREATER_THAN', hit_normal.outputs['Y'], .10)
            valid_wall = math('MULTIPLY', ray.outputs['Is Hit'], inner_facing)
            wall_available = math('MAXIMUM', math('SUBTRACT', math('ADD', ray.outputs['Hit Distance'], .000001 / scale), minimum_wall / scale), 0.)
            bounded = math('MINIMUM', bounded, math('MULTIPLY', wall_available, valid_wall), 'Current actual inner-wall cap ' + side)
            amplitudes.append(math('MULTIPLY', math('MULTIPLY', bounded, outside), front_facing))

        amplitude = math('ADD', amplitudes[0], amplitudes[1])
        offset = nodes.new('ShaderNodeVectorMath'); offset.operation = 'SCALE'
        offset.inputs[0].default_value = (0., 1., 0.)
        links.new(amplitude, offset.inputs['Scale'])
        moved = nodes.new('GeometryNodeSetPosition'); moved.name = 'R11 bounded upper-lid posterior roll reduction'
        links.new(input_geometry, geometry_socket(moved.inputs))
        links.new(math('GREATER_THAN', amplitude, 0.), moved.inputs['Selection'])
        links.new(offset.outputs['Vector'], moved.inputs['Offset'])
        links.new(geometry_socket(moved.outputs), geometry_socket(output.inputs))
        group['R11_maximum_world_meters'] = maximum
        group['R11_protected_gap_world_meters'] = protected_gap
        group['R11_blink_fade_end'] = fade_end
        group['R11_full_vertex_count'] = full_vertex_count
        group['R11_minimum_inner_wall_meters'] = minimum_wall
        modifier = head.modifiers.new(GROUP, 'NODES'); modifier.node_group = group
        # Relief is an early/open-eye correction. At b >= fade_end it is
        # exactly zero, preserving the input of the later canthal correction.
        # Move only our new modifier; existing modifier properties are intact.
        if insertion_index < len(head.modifiers) - 1:
            head.modifiers.move(len(head.modifiers) - 1, insertion_index)
        position_in_stack = list(head.modifiers).index(modifier)
        if any(i <= position_in_stack for i, item in enumerate(head.modifiers)
               if _is_corner(item) or item.type == 'ARMATURE'):
            raise RuntimeError('Upper-lid relief must precede canthal nodes and armatures')
        if any(i >= position_in_stack for i, item in enumerate(head.modifiers) if item.type == 'MULTIRES'):
            raise RuntimeError('Upper-lid relief must follow the evaluated Multires surface')
        group.update_tag(); head.update_tag()
        return modifier, group, cache, mesh
    except Exception:
        # A failed build has not returned handles to the caller yet.
        if modifier is not None: head.modifiers.remove(modifier)
        if group is not None and group.users == 0: bpy.data.node_groups.remove(group)
        if cache is not None: bpy.data.objects.remove(cache, do_unlink=True)
        if mesh is not None and mesh.users == 0: bpy.data.meshes.remove(mesh)
        raise
