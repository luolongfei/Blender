"""Native angular outer-eye refinement with bounded real-section transport.

Matches the independently checked radial v3 reference. The true sclera query
continues onto an actual safe section; extra radial transport fades out at the
real globe end, preserving the thin outer bridge's radius. This layer is an
appearance candidate, not the later inner-lining contact correction.
"""
from pathlib import Path
import runpy

import numpy as np

GROUP = 'R11 refined outer eye corners radial'
CONTROL = 'R11 refined outer eye corners'
CORNER_GROUP = 'R11 actual canthal surface path'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def build(head, full_vertex_count, original_ids, fields, scale, metadata):
    """Return modifier, group, lookup object and mesh; no save or source edit.

    fields[side] is (ROI count, 3): xEnvelope, thetaMid, angularCoreHalf.
    A caller-created zero-displacement Key CONTROL supplies strength [0, 1].
    This builder requires the unbound, upright uniformly scaled REST source.
    """
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
        if values.shape != (len(ids), 3) or not np.isfinite(values).all() or np.any(values[:, 0]<0.) or np.any(values[:, 0]>1.) or np.any(values[:, 2]<=0.):
            raise ValueError('Finite angular channel field required for '+side)
        if 'R11 blink '+side not in keys.key_blocks:
            raise RuntimeError('Missing blink control '+side)
        data[side] = values
        eye = bpy.data.objects.get('R11 Adult sclera '+side)
        if eye is None or eye.type != 'MESH' or eye == head:
            raise RuntimeError('Independent actual sclera required for '+side)
        eyes[side] = eye
        center = np.asarray(metadata['sides'][side]['centerWorld'], float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Finite actual eye center required')
        centers[side] = (center-matrix[:3, 3])/scale
    if np.any((data['L'][:, 0] != 0.) & (data['R'][:, 0] != 0.)):
        raise ValueError('Independent left and right outer-eye fields must be disjoint')
    if metadata['blinkFadeEnd'] != .5 or not 0. < metadata['compression'] <= .75:
        raise ValueError('Checked half-blink fade and bounded angular compression required')
    compat = api('adult_blink_midpath_nodes_r11')
    geometry_socket = compat['_compatible_geometry_socket']()
    ray_source = compat['_ray_source']
    modifier = group = cache = mesh = None
    try:
        mesh = bpy.data.meshes.new(GROUP+' lookup')
        mesh.vertices.add(n)
        mesh.vertices.foreach_set('co', np.zeros(n*3, np.float32))
        attributes = {}
        for side in ('L', 'R'):
            attribute = 'R11_outer_eye_radial_channels_'+side
            dense = np.zeros((n, 3), np.float32)
            dense[ids] = data[side].astype(np.float32)
            mesh.attributes.new(attribute, 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', dense.ravel())
            attributes[side] = attribute
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

        def smooth(value):
            t = math('MINIMUM', math('MAXIMUM', value, 0.), 1.)
            return math('MULTIPLY', math('MULTIPLY', t, t), math('SUBTRACT', 3., math('MULTIPLY', 2., t)))

        def product(*values):
            result = values[0]
            for value in values[1:]:
                result = math('MULTIPLY', result, value)
            return result

        def select(condition, if_true, if_false):
            switch = nodes.new('GeometryNodeSwitch')
            switch.input_type = 'FLOAT'
            links.new(condition, switch.inputs['Switch'])
            for name, value in (('True', if_true), ('False', if_false)):
                if isinstance(value, (float, int)):
                    switch.inputs[name].default_value = value
                else:
                    links.new(value, switch.inputs[name])
            return switch.outputs['Output']

        offsets, selections = [], []
        for side in ('L', 'R'):
            details = metadata['sides'][side]
            named = nodes.new('GeometryNodeInputNamedAttribute')
            named.data_type = 'FLOAT_VECTOR'
            named.inputs['Name'].default_value = attributes[side]
            sample = nodes.new('GeometryNodeSampleIndex')
            sample.data_type, sample.domain, sample.clamp = 'FLOAT_VECTOR', 'POINT', False
            links.new(lookup.outputs['Geometry'], geometry_socket(sample.inputs))
            links.new(named.outputs['Attribute'], sample.inputs['Value'])
            links.new(index.outputs['Index'], sample.inputs['Index'])
            field = split(sample.outputs['Value'])
            value = nodes.new('ShaderNodeValue')
            value.name = 'R11 radial outer-eye early blink influence '+side
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
            dy = math('SUBTRACT', components['Y'], float(centers[side][1]))
            dz = math('SUBTRACT', components['Z'], float(centers[side][2]))
            theta = math('ARCTAN2', dz, math('MULTIPLY', dy, -1.))
            rho = math('SQRT', math('ADD', math('MULTIPLY', dy, dy), math('MULTIPLY', dz, dz)))
            angle = math('SUBTRACT', theta, field['Y'])
            angular = math('SUBTRACT', 1., smooth(math('DIVIDE', math('SUBTRACT', math('ABSOLUTE', angle), field['Z']), float(metadata['angularFadeRadians']))))
            sign = float(details['outwardSign'])
            safe = sign*(float(details['outwardActualAxisExitX'])-matrix[0, 3])/scale-float(metadata['querySafeInsetMeters'])/scale
            span = float(metadata['queryClampTransitionMeters'])/scale
            start = safe-span*.5
            u = math('MULTIPLY', components['X'], sign)
            q_t = math('MINIMUM', math('MAXIMUM', math('DIVIDE', math('SUBTRACT', u, start), span), 0.), 1.)
            q_t2 = math('MULTIPLY', q_t, q_t)
            q_t3 = math('MULTIPLY', q_t2, q_t)
            q_t4 = math('MULTIPLY', q_t2, q_t2)
            mapped = math('ADD', start, math('MULTIPLY', span, math('ADD', math('SUBTRACT', q_t, q_t3), math('MULTIPLY', .5, q_t4))))
            mapped = select(math('GREATER_THAN', u, start+span), safe, mapped)
            mapped = select(math('LESS_THAN', u, start), u, mapped)
            query_x = math('MULTIPLY', mapped, sign)
            globe = nodes.new('GeometryNodeObjectInfo')
            globe.name = 'R11 radial outer-eye actual sclera '+side
            globe.transform_space = 'RELATIVE'
            globe.inputs['Object'].default_value = eyes[side]
            globe.inputs['As Instance'].default_value = False
            globe['R11_prearm_radial_outerend_sclera_side'] = side

            def ray_for(angle_socket, stage):
                ray = nodes.new('GeometryNodeRaycast')
                ray.data_type = 'FLOAT'
                # Blender 5.2 exposes interpolation through a menu socket.
                ray.inputs['Interpolation'].default_value = 'Interpolated'
                if ray.inputs['Interpolation'].default_value != 'Interpolated':
                    raise RuntimeError('Actual Raycast interpolation menu readback failed')
                ray.name = 'R11 radial outer-eye sclera ray '+side+' '+stage
                ray['R11_prearm_radial_outerend_sclera_raycast'] = side
                ray['R11_outerend_ray_stage'] = stage
                links.new(globe.outputs['Geometry'], geometry_socket(ray.inputs))
                origin = combine(query_x, float(centers[side][1]), float(centers[side][2]))
                links.new(origin, ray_source(ray))
                direction = combine(0., math('MULTIPLY', -1., math('COSINE', angle_socket)), math('SINE', angle_socket))
                links.new(direction, ray.inputs['Ray Direction'])
                ray.inputs['Ray Length'].default_value = .08/scale
                return ray

            old_ray = ray_for(theta, 'before')
            gap = math('SUBTRACT', rho, old_ray.outputs['Hit Distance'])
            positive = math('SUBTRACT', 1., smooth(math('DIVIDE', math('SUBTRACT', gap, float(metadata['gapCoreMeters'])/scale), float(metadata['gapFadeMeters'])/scale)))
            negative = smooth(math('DIVIDE', math('SUBTRACT', gap, float(metadata['negativeGapFadeStartMeters'])/scale),
                                   float(metadata['negativeGapFadeEndMeters']-metadata['negativeGapFadeStartMeters'])/scale))
            amount = product(float(metadata['compression']), field['X'], angular, positive, negative, value.outputs[0])
            target_theta = math('SUBTRACT', theta, math('MULTIPLY', amount, angle))
            new_ray = ray_for(target_theta, 'after')
            outer_x = (float(details['actualGlobeOuterX'])-matrix[0, 3])/scale
            transport = smooth(math('DIVIDE', math('MULTIPLY', sign, math('SUBTRACT', outer_x, components['X'])), float(metadata['radialTransportFadeMeters'])/scale))
            target_rho = math('ADD', rho, math('MULTIPLY', transport, math('SUBTRACT', new_ray.outputs['Hit Distance'], old_ray.outputs['Hit Distance'])))
            axis = (np.asarray(details['axisInsideXRange'], float)-matrix[0, 3])/scale
            minimum_radius = float(metadata['minimumRadiusMeters'])/scale
            # The reference extent is only an early-out. Its native equivalent
            # is redundant once both actual hits and positive gap taper pass:
            # gap < core+fade bounds rho by the real queried radius plus fade.
            active = product(math('GREATER_THAN', field['X'], 0.), math('GREATER_THAN', angular, 0.),
                             math('GREATER_THAN', rho, minimum_radius), old_ray.outputs['Is Hit'], new_ray.outputs['Is Hit'],
                             math('GREATER_THAN', old_ray.outputs['Hit Distance'], 0.), math('GREATER_THAN', new_ray.outputs['Hit Distance'], 0.),
                             math('GREATER_THAN', query_x, float(axis[0]+metadata['axisInsetMeters']/scale)),
                             math('LESS_THAN', query_x, float(axis[1]-metadata['axisInsetMeters']/scale)),
                             math('GREATER_THAN', target_rho, minimum_radius), math('GREATER_THAN', amount, 1e-12))
            target_y = math('SUBTRACT', float(centers[side][1]), math('MULTIPLY', target_rho, math('COSINE', target_theta)))
            target_z = math('ADD', float(centers[side][2]), math('MULTIPLY', target_rho, math('SINE', target_theta)))
            offset_y = math('MULTIPLY', math('SUBTRACT', target_y, components['Y']), active)
            offset_z = math('MULTIPLY', math('SUBTRACT', target_z, components['Z']), active)
            offsets.append(combine(0., offset_y, offset_z))
            selections.append(active)
        total = nodes.new('ShaderNodeVectorMath')
        total.operation = 'ADD'
        links.new(offsets[0], total.inputs[0])
        links.new(offsets[1], total.inputs[1])
        moved = nodes.new('GeometryNodeSetPosition')
        moved.name = 'R11 early outer-eye angular taper with bridge radius protection'
        links.new(input_geometry, geometry_socket(moved.inputs))
        links.new(math('GREATER_THAN', math('ADD', selections[0], selections[1]), 0.), moved.inputs['Selection'])
        links.new(total.outputs['Vector'], moved.inputs['Offset'])
        links.new(geometry_socket(moved.outputs), geometry_socket(output.inputs))
        group['R11_query_space'] = 'unbound_rest_head_object'
        group['R11_current_armature_support'] = False
        group['R11_blink_fade_end'] = .5
        group['R11_full_vertex_count'] = n
        group['R11_offset_semantics'] = 'Monotone theta field; safe actual-section radial transport fading to pure rotation at globe end; not inner contact repair'
        group['R11_radial_reference_helper'] = 'adult_neutral_eye_outerends_radial_r11_v3.py'
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
