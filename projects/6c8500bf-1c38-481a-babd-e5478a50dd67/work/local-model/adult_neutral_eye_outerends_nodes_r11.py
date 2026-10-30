"""REJECTED research-only front-depth outer-eye taper; never install.

The stored field contains only a neutral world-Z displacement. Two rays on the
actual sclera transport each affected point by the change in front depth. The
existing front-axis tissue gap is retained; this is not an inner contact repair.
Both offline proposals failed surface quality at the globe projection tangent:
the first crossed adjacent faces; the second had severe local area/normal
distortion despite zero detected crossings. This preserved builder documents
the attempted matching native formula and is deliberately disabled.
"""
from pathlib import Path
import runpy

import numpy as np

GROUP = 'R11 refined outer eye corners'
CONTROL = GROUP
CORNER_GROUP = 'R11 actual canthal surface path'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def build(head, full_vertex_count, original_ids, fields, scale, centers_world=None):
    """Return modifier, group, lookup object and mesh; no save or source edit.

    fields[side] is (ROI count, 3), only Z nonzero, measured in world metres.
    A caller-created zero-displacement Key CONTROL supplies strength [0, 1].
    This builder requires the unbound, upright uniformly scaled REST source.
    """
    raise RuntimeError('Rejected front-depth outer-eye design: offline surface quality failed; do not install this research candidate')
    import bpy

    n = int(full_vertex_count)
    if isinstance(full_vertex_count, bool) or n != full_vertex_count or n <= 0:
        raise ValueError('Positive actual evaluated point count required')
    ids = np.asarray(original_ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Nonempty original evaluated point IDs required')
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= n or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique in-range evaluated IDs required')
    matrix = np.asarray(head.matrix_world, float)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0. or not np.allclose(matrix[:3, :3], np.eye(3)*scale, rtol=0., atol=1e-7):
        raise RuntimeError('Upright positive uniformly scaled head required')
    if head.type != 'MESH' or head.data.shape_keys is None:
        raise RuntimeError('Current head and blink Keys required')
    if any(m.type == 'ARMATURE' for m in head.modifiers) or 'R11_head_neck_skin_bind' in bpy.context.scene:
        raise RuntimeError('Install on unbound REST source; later pose needs explicit cranial query migration')
    keys = head.data.shape_keys
    if CONTROL not in keys.key_blocks:
        raise RuntimeError('Caller must create the zero-displacement strength Key '+CONTROL)
    if any(m.type == 'NODES' and m.node_group and m.node_group.name.split('.')[0] == GROUP for m in head.modifiers):
        raise RuntimeError('Existing outer-eye candidate must be retained before rebuilding')
    corners = [i for i, m in enumerate(head.modifiers) if m.type == 'NODES' and m.node_group and
               (m.node_group.name == CORNER_GROUP or m.node_group.name.startswith(CORNER_GROUP+'.'))]
    if len(corners) != 1:
        raise RuntimeError('One installed late canthal modifier required')
    insertion_index = corners[0]
    if any(m.type == 'MULTIRES' for m in list(head.modifiers)[insertion_index:]):
        raise RuntimeError('Evaluated-index field must follow Multires')
    data, eyes, centers = {}, {}, {}
    for side in ('L', 'R'):
        values = np.asarray(fields[side], float)
        if values.shape != (len(ids), 3) or not np.isfinite(values).all() or np.any(values[:, :2]):
            raise ValueError('Finite Z-only world displacement field required for '+side)
        if 'R11 blink '+side not in keys.key_blocks:
            raise RuntimeError('Missing blink control '+side)
        data[side] = values
        eye = bpy.data.objects.get('R11 Adult sclera '+side)
        if eye is None or eye.type != 'MESH' or eye == head:
            raise RuntimeError('Independent actual sclera required for '+side)
        eyes[side] = eye
        if centers_world is None:
            evaluated = eye.evaluated_get(bpy.context.evaluated_depsgraph_get())
            points = np.empty(len(evaluated.data.vertices)*3, np.float32)
            evaluated.data.vertices.foreach_get('co', points)
            em = np.asarray(evaluated.matrix_world, float)
            world = points.reshape(-1, 3).astype(float) @ em[:3, :3].T + em[:3, 3]
            center = .5*(world.min(0)+world.max(0))
        else:
            center = np.asarray(centers_world[side], float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Finite actual eye center required')
        centers[side] = (center-matrix[:3, 3])/scale
    if np.any((data['L'][:, 2] != 0.) & (data['R'][:, 2] != 0.)):
        raise ValueError('Independent left and right outer-eye fields must be disjoint')
    compat = api('adult_blink_midpath_nodes_r11')
    geometry_socket = compat['_compatible_geometry_socket']()
    ray_source = compat['_ray_source']
    modifier = group = cache = mesh = None
    try:
        mesh = bpy.data.meshes.new(GROUP+' lookup')
        mesh.vertices.add(n)
        mesh.vertices.foreach_set('co', np.zeros(n*3, np.float32))
        attribute = 'R11_outer_eye_delta_z_L_R'
        dense = np.zeros((n, 3), np.float32)
        for channel, side in enumerate(('L', 'R')):
            dense[ids, channel] = (data[side][:, 2]/scale).astype(np.float32)
        mesh.attributes.new(attribute, 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', dense.ravel())
        mesh.update()
        cache = bpy.data.objects.new(GROUP+' lookup', mesh)
        bpy.context.scene.collection.objects.link(cache)
        cache.hide_render = True
        cache.hide_set(True)
        group = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        nodes, links = group.nodes, group.links
        source, output = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
        input_geometry = geometry_socket(source.outputs)
        index = nodes.new('GeometryNodeInputIndex')

        def math(operation, a, b=None):
            node = nodes.new('ShaderNodeMath')
            node.operation = operation
            for j, value in enumerate((a, b)):
                if value is None:
                    continue
                if isinstance(value, (float, int)):
                    node.inputs[j].default_value = value
                else:
                    links.new(value, node.inputs[j])
            return node.outputs[0]

        def split(vector):
            node = nodes.new('ShaderNodeSeparateXYZ')
            links.new(vector, node.inputs[0])
            return node.outputs

        def combine(x, y, z):
            node = nodes.new('ShaderNodeCombineXYZ')
            for name, value in zip(('X', 'Y', 'Z'), (x, y, z)):
                if isinstance(value, (float, int)):
                    node.inputs[name].default_value = value
                else:
                    links.new(value, node.inputs[name])
            return node.outputs[0]

        position = nodes.new('GeometryNodeInputPosition')
        position_sample = nodes.new('GeometryNodeSampleIndex')
        position_sample.data_type, position_sample.domain, position_sample.clamp = 'FLOAT_VECTOR', 'POINT', False
        links.new(input_geometry, geometry_socket(position_sample.inputs))
        links.new(position.outputs['Position'], position_sample.inputs['Value'])
        links.new(index.outputs['Index'], position_sample.inputs['Index'])
        components = split(position_sample.outputs['Value'])
        lookup = nodes.new('GeometryNodeObjectInfo')
        lookup.transform_space = 'RELATIVE'
        lookup.inputs['Object'].default_value = cache
        lookup.inputs['As Instance'].default_value = False
        named = nodes.new('GeometryNodeInputNamedAttribute')
        named.data_type = 'FLOAT_VECTOR'
        named.inputs['Name'].default_value = attribute
        sample = nodes.new('GeometryNodeSampleIndex')
        sample.data_type, sample.domain, sample.clamp = 'FLOAT_VECTOR', 'POINT', False
        links.new(lookup.outputs['Geometry'], geometry_socket(sample.inputs))
        links.new(named.outputs['Attribute'], sample.inputs['Value'])
        links.new(index.outputs['Index'], sample.inputs['Index'])
        field_components = split(sample.outputs['Value'])
        offsets, selections = [], []
        for side, channel in (('L', 'X'), ('R', 'Y')):
            value = nodes.new('ShaderNodeValue')
            value.name = 'R11 outer-eye early blink influence '+side
            driver = value.outputs[0].driver_add('default_value').driver
            driver.type = 'SCRIPTED'
            for name, control in (('r', CONTROL), ('b', 'R11 blink '+side)):
                variable = driver.variables.new()
                variable.name, variable.type = name, 'SINGLE_PROP'
                target = variable.targets[0]
                target.id_type, target.id = 'KEY', keys
                target.data_path = keys.key_blocks[control].path_from_id('value')
            t = 'max(0.0,min(1.0,1.0-b/0.5))'
            driver.expression = 'min(max(r,0.0),1.0)*('+t+')**2*(3.0-2.0*('+t+')) if b < 0.5 else 0.0'
            dz = math('MULTIPLY', field_components[channel], value.outputs[0])
            globe = nodes.new('GeometryNodeObjectInfo')
            globe.name = 'R11 outer-eye actual sclera '+side
            globe.transform_space = 'RELATIVE'
            globe.inputs['Object'].default_value = eyes[side]
            globe.inputs['As Instance'].default_value = False
            globe['R11_prearm_outerend_sclera_side'] = side
            rays = []
            for stage, z in (('before', components['Z']), ('after', math('ADD', components['Z'], dz))):
                ray = nodes.new('GeometryNodeRaycast')
                ray.data_type, ray.mapping = 'FLOAT', 'INTERPOLATED'
                ray.name = 'R11 outer-eye actual sclera ray '+side+' '+stage
                ray['R11_prearm_outerend_sclera_raycast'] = side
                ray['R11_outerend_ray_stage'] = stage
                links.new(globe.outputs['Geometry'], geometry_socket(ray.inputs))
                origin = combine(components['X'], float(centers[side][1]-.04/scale), z)
                links.new(origin, ray_source(ray))
                ray.inputs['Ray Direction'].default_value = (0., 1., 0.)
                ray.inputs['Ray Length'].default_value = .08/scale
                rays.append(ray)
            hits = math('MULTIPLY', rays[0].outputs['Is Hit'], rays[1].outputs['Is Hit'])
            active = math('MULTIPLY', hits, math('GREATER_THAN', math('ABSOLUTE', dz), 0.))
            before_y = split(rays[0].outputs['Hit Position'])['Y']
            after_y = split(rays[1].outputs['Hit Position'])['Y']
            dy = math('MULTIPLY', math('SUBTRACT', after_y, before_y), active)
            offsets.append(combine(0., dy, math('MULTIPLY', dz, active)))
            selections.append(active)
        total = nodes.new('ShaderNodeVectorMath')
        total.operation = 'ADD'
        links.new(offsets[0], total.inputs[0])
        links.new(offsets[1], total.inputs[1])
        moved = nodes.new('GeometryNodeSetPosition')
        moved.name = 'R11 early outer-eye true-globe tangent taper'
        links.new(input_geometry, geometry_socket(moved.inputs))
        links.new(math('GREATER_THAN', math('ADD', selections[0], selections[1]), 0.), moved.inputs['Selection'])
        links.new(total.outputs['Vector'], moved.inputs['Offset'])
        links.new(geometry_socket(moved.outputs), geometry_socket(output.inputs))
        group['R11_query_space'] = 'unbound_rest_head_object'
        group['R11_current_armature_support'] = False
        group['R11_blink_fade_end'] = .5
        group['R11_full_vertex_count'] = n
        group['R11_offset_semantics'] = 'Neutral Z field; actual globe front-depth transport; existing gap retained; not contact repair'
        modifier = head.modifiers.new(GROUP, 'NODES')
        modifier.node_group = group
        head.modifiers.move(len(head.modifiers)-1, insertion_index)
        return modifier, group, cache, mesh
    except Exception:
        if modifier is not None:
            head.modifiers.remove(modifier)
        if cache is not None:
            bpy.data.objects.remove(cache, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        if group is not None and group.users == 0:
            bpy.data.node_groups.remove(group)
        raise
