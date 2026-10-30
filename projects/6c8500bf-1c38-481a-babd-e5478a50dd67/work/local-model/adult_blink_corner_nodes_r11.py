"""Geometry-node builder for the bounded R11 late canthal surface correction.

Internal component, not a command-line installer. A dense point-domain lookup
keeps final L3 vertex order intact. Angles are applied about the actual eye
center; signed actual-globe radial gap corrections are separate. The caller
owns source guards, actual contact checks, rollback and save.
"""
import json
import numpy as np

GROUP = 'R11 actual canthal surface path'
ATTRIBUTE = 'R11_canthal_surface_angle_radius_side'


def build(head, full_vertex_count, original_ids, fields, centers_world, matrix_world,
          onset=.8):
    import bpy
    matrix = np.asarray(matrix_world, float)
    scales = np.linalg.norm(matrix[:3, :3], axis=0)
    if max(scales)-min(scales)>1e-7 or np.max(abs(matrix[:3, :3]-np.eye(3)*scales[0]))>1e-7:
        raise RuntimeError('Current candidate requires the observed uniform head transform')
    dense = np.zeros((full_vertex_count, 3), np.float32)
    for sign, side in ((1., 'L'), (-1., 'R')):
        angle = np.asarray(fields[side]['angle']); radius = np.asarray(fields[side]['radiusChange'])
        selected = (abs(angle)>1e-10) | (abs(radius)>1e-10)
        ids = np.asarray(original_ids)[selected]
        if np.any(dense[ids, 2]): raise RuntimeError('Canthal fields overlap across the two eyes')
        dense[ids, 0] = angle[selected]
        dense[ids, 1] = radius[selected]/scales[0]
        dense[ids, 2] = sign
    mesh = bpy.data.meshes.new(GROUP+' lookup')
    mesh.vertices.add(full_vertex_count)
    mesh.vertices.foreach_set('co', np.zeros(full_vertex_count*3, np.float32))
    attr = mesh.attributes.new(ATTRIBUTE, 'FLOAT_VECTOR', 'POINT')
    attr.data.foreach_set('vector', dense.ravel()); mesh.update()
    cache = bpy.data.objects.new(GROUP+' lookup', mesh)
    bpy.context.scene.collection.objects.link(cache); cache.hide_render = True; cache.hide_set(True)
    group = bpy.data.node_groups.new(GROUP, 'GeometryNodeTree')
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes = group.nodes; links = group.links
    source = nodes.new('NodeGroupInput'); output = nodes.new('NodeGroupOutput')
    info = nodes.new('GeometryNodeObjectInfo'); info.inputs['Object'].default_value = cache
    info.inputs['As Instance'].default_value = False
    named = nodes.new('GeometryNodeInputNamedAttribute'); named.data_type = 'FLOAT_VECTOR'
    named.inputs['Name'].default_value = ATTRIBUTE
    index = nodes.new('GeometryNodeInputIndex')
    sample = nodes.new('GeometryNodeSampleIndex'); sample.data_type = 'FLOAT_VECTOR'; sample.domain = 'POINT'; sample.clamp = False
    links.new(info.outputs['Geometry'], sample.inputs['Geometry'])
    links.new(named.outputs['Attribute'], sample.inputs['Value']); links.new(index.outputs['Index'], sample.inputs['Index'])
    split = nodes.new('ShaderNodeSeparateXYZ'); links.new(sample.outputs['Value'], split.inputs[0])

    def math(operation, a, b=None):
        node = nodes.new('ShaderNodeMath'); node.operation = operation
        for input_index, value in enumerate((a, b)):
            if value is None: continue
            if isinstance(value, (float, int)): node.inputs[input_index].default_value = value
            else: links.new(value, node.inputs[input_index])
        return node.outputs[0]

    geometry = source.outputs['Geometry']
    for sign, side in ((1., 'L'), (-1., 'R')):
        center = (np.asarray(centers_world[side])-matrix[:3, 3])/scales[0]
        influence = nodes.new('ShaderNodeValue'); influence.name = 'Last-phase canthal closure '+side
        influence.outputs[0].default_value = 0.
        driver = influence.outputs[0].driver_add('default_value').driver; driver.type = 'SCRIPTED'
        variable = driver.variables.new(); variable.name = 'b'; variable.type = 'SINGLE_PROP'
        variable.targets[0].id_type = 'KEY'; variable.targets[0].id = head.data.shape_keys
        variable.targets[0].data_path = 'key_blocks['+json.dumps('R11 blink '+side)+'].value'
        expression = 'max(0.0,min(1.0,(b-'+repr(onset)+')/'+repr(1.-onset)+'))'
        driver.expression = '('+expression+')**2*(3.0-2.0*('+expression+'))'
        angle = math('MULTIPLY', math('MULTIPLY', split.outputs['X'], influence.outputs[0]), -1.)
        radius = math('MULTIPLY', split.outputs['Y'], influence.outputs[0])
        choose = math('GREATER_THAN', math('MULTIPLY', split.outputs['Z'], sign), .5)
        position = nodes.new('GeometryNodeInputPosition')
        rotate = nodes.new('ShaderNodeVectorRotate'); rotate.rotation_type = 'AXIS_ANGLE'
        rotate.inputs['Center'].default_value = center; rotate.inputs['Axis'].default_value = (1., 0., 0.)
        links.new(position.outputs['Position'], rotate.inputs['Vector']); links.new(angle, rotate.inputs['Angle'])
        relative = nodes.new('ShaderNodeVectorMath'); relative.operation = 'SUBTRACT'
        links.new(rotate.outputs['Vector'], relative.inputs[0]); relative.inputs[1].default_value = center
        yz = nodes.new('ShaderNodeVectorMath'); yz.operation = 'MULTIPLY'
        links.new(relative.outputs['Vector'], yz.inputs[0]); yz.inputs[1].default_value = (0., 1., 1.)
        normal = nodes.new('ShaderNodeVectorMath'); normal.operation = 'NORMALIZE'; links.new(yz.outputs['Vector'], normal.inputs[0])
        scale = nodes.new('ShaderNodeVectorMath'); scale.operation = 'SCALE'; links.new(normal.outputs['Vector'], scale.inputs[0]); links.new(radius, scale.inputs['Scale'])
        moved = nodes.new('ShaderNodeVectorMath'); moved.operation = 'ADD'; links.new(rotate.outputs['Vector'], moved.inputs[0]); links.new(scale.outputs['Vector'], moved.inputs[1])
        set_position = nodes.new('GeometryNodeSetPosition'); links.new(geometry, set_position.inputs['Geometry'])
        links.new(choose, set_position.inputs['Selection']); links.new(moved.outputs['Vector'], set_position.inputs['Position'])
        geometry = set_position.outputs['Geometry']
    links.new(geometry, output.inputs['Geometry'])
    modifier = head.modifiers.new(GROUP, 'NODES'); modifier.node_group = group
    # Preserve REST surface vertex order and let any later cranial armature
    # deform the corrected tissue. Do not append this after a posed armature.
    armatures = [i for i, candidate in enumerate(head.modifiers) if candidate.type == 'ARMATURE']
    if armatures:
        head.modifiers.move(len(head.modifiers)-1, min(armatures))
    return modifier, group, cache, mesh
