"""Native reversible fixed-L3 general-vector field for a static mouth-corner preview.

The caller creates the zero-delta public Key, checks the actual complete input
topology/ID correspondence, and owns actual evaluation, visual review and
rollback. This builder never evaluates, renders, saves or edits existing GN.
It does not claim warm-smile/jaw/motion compatibility. All existing GN precedes this new
field, and all Armature modifiers follow it.
"""
import hashlib
import numpy as np

GROUP = 'R11 relaxed mouth corners preview'
ATTRIBUTE = 'R11_relaxed_mouth_corners_local_offset'
CONTROL = 'R11 relaxed mouth corners'
CONVERSION_TOLERANCE_WORLD = 2e-9


def geometry_socket(sockets):
    choices = [s for s in sockets if s.type == 'GEOMETRY']
    if len(choices) != 1:
        raise RuntimeError('Expected one unambiguous geometry socket')
    return choices[0]


def _stack_plan(head):
    stack = list(head.modifiers)
    multires = [i for i, mod in enumerate(stack) if mod.type == 'MULTIRES']
    if len(multires) != 1:
        raise RuntimeError('Exactly one actual L3 Multires is required')
    mi = multires[0]
    multi = stack[mi]
    if multi.levels != 3 or multi.render_levels != 3 or not multi.show_viewport or not multi.show_render:
        raise RuntimeError('Current preview and render must use active L3 Multires')
    if any(mod.type in {'ARMATURE', 'NODES'} for mod in stack[:mi]):
        raise RuntimeError('Geometry nodes or Armature before Multires require a different dependency contract')
    armatures = [i for i, mod in enumerate(stack) if mod.type == 'ARMATURE']
    insertion = min(armatures) if armatures else len(stack)
    if mi >= insertion:
        raise RuntimeError('Armature precedes the required L3 input')
    for i, mod in enumerate(stack):
        if mod.type == 'NODES':
            if mod.node_group is None:
                raise RuntimeError('Existing GN has no node group: '+mod.name)
            if mod.node_group.name == GROUP or mod.node_group.name.startswith(GROUP+'.'):
                raise RuntimeError('Existing relaxed-mouth-corner preview must be retained or explicitly removed by its owner')
            if i >= insertion:
                raise RuntimeError('An existing GN follows Armature; cannot preserve the requested static input order')
        if mi < i < insertion and mod.type not in {'MASK', 'NODES', 'WEIGHTED_NORMAL', 'NORMAL_EDIT'}:
            raise RuntimeError('Unverified pre-field modifier may change the observed L3 topology: '+mod.name)
        if i >= insertion and mod.type not in {'ARMATURE', 'WEIGHTED_NORMAL', 'NORMAL_EDIT'}:
            raise RuntimeError('Downstream topology change is incompatible with fixed evaluated IDs: '+mod.name)
    return insertion, tuple(stack)


def build(head, N, ids, world_offsets, matrix_world):
    """Return (modifier, group, lookup_object, lookup_mesh), without evaluation.

    N and ids refer to caller-verified actual complete L3 geometry after the
    current existing GN. Counts alone do not establish vertex correspondence;
    the caller must compare actual polygon/loop IDs before/after this field.
    All XYZ world offsets are transformed by the inverse observed linear frame.
    The public zero-delta Key is clamped to [0,1]. Key0 returns the unchanged
    input Geometry through a native Switch, without evaluating Set Position.
    """
    import bpy
    if head.type != 'MESH' or head.library or head.data.library or not head.data.shape_keys:
        raise RuntimeError('A local mesh head with caller-created shape keys is required')
    keys = head.data.shape_keys
    if CONTROL not in keys.key_blocks or not keys.use_relative:
        raise RuntimeError('Caller-created relative zero-delta relaxed-mouth-corner control Key required')
    key = keys.key_blocks[CONTROL]
    if not np.isfinite(float(key.value)) or key.mute:
        raise RuntimeError('Mouth-corner preview control must be finite and unmuted')
    co = np.empty((len(key.data), 3), np.float32)
    ref = np.empty_like(co)
    key.data.foreach_get('co', co.ravel())
    key.relative_key.data.foreach_get('co', ref.ravel())
    if not np.array_equal(co, ref):
        raise RuntimeError('The public relaxed-mouth-corner control must have exactly zero raw displacement')
    if isinstance(N, bool) or int(N) != N or N <= 0:
        raise ValueError('Positive complete actual L3 point count required')
    N = int(N)
    ids = np.asarray(ids)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Nonempty original evaluated integer ID vector required')
    ids = ids.astype(np.int64, copy=False)
    if ids.min() < 0 or ids.max() >= N or len(np.unique(ids)) != len(ids):
        raise ValueError('Evaluated IDs must be unique and inside the supplied actual topology')
    offsets = np.asarray(world_offsets, float)
    if offsets.shape != (len(ids), 3) or not np.isfinite(offsets).all():
        raise ValueError('Finite world-space displacement per selected ID required')
    M = np.asarray(matrix_world, float)
    if M.shape != (4, 4) or not np.isfinite(M).all() or not np.allclose(M[3], [0., 0., 0., 1.], atol=1e-10, rtol=0.):
        raise ValueError('Finite affine observed head transform required')
    if not np.allclose(np.asarray(head.matrix_world, float), M, atol=2e-7, rtol=0.):
        raise RuntimeError('Current head transform differs from the supplied observed transform')
    linear = M[:3, :3]
    if abs(np.linalg.det(linear)) < 1e-12:
        raise ValueError('Head transform is singular')
    local = offsets@np.linalg.inv(linear).T
    local = local.astype(np.float32)
    conversion_error = float(np.linalg.norm(local.astype(float)@linear.T-offsets, axis=1).max())
    if conversion_error > CONVERSION_TOLERANCE_WORLD:
        raise RuntimeError('Float32 general-vector field differs from verified world offsets')
    if not np.any(local != 0.):
        raise ValueError('Mouth-corner candidate has no nonzero offset')
    insertion, original_stack = _stack_plan(head)
    dense = np.zeros((N, 3), np.float32)
    dense[ids] = local
    modifier = group = cache = mesh = None
    try:
        mesh = bpy.data.meshes.new(GROUP+' lookup')
        mesh.vertices.add(N)
        mesh.vertices.foreach_set('co', np.zeros(N*3, np.float32))
        attr = mesh.attributes.new(ATTRIBUTE, 'FLOAT_VECTOR', 'POINT')
        attr.data.foreach_set('vector', dense.ravel())
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
        geometry = geometry_socket(source.outputs)
        info = nodes.new('GeometryNodeObjectInfo')
        info.transform_space = 'RELATIVE'
        info.inputs['Object'].default_value = cache
        info.inputs['As Instance'].default_value = False
        named = nodes.new('GeometryNodeInputNamedAttribute')
        named.data_type = 'FLOAT_VECTOR'
        named.inputs['Name'].default_value = ATTRIBUTE
        index = nodes.new('GeometryNodeInputIndex')
        sample = nodes.new('GeometryNodeSampleIndex')
        sample.data_type = 'FLOAT_VECTOR'
        sample.domain = 'POINT'
        sample.clamp = False
        links.new(info.outputs['Geometry'], geometry_socket(sample.inputs))
        links.new(named.outputs['Attribute'], sample.inputs['Value'])
        links.new(index.outputs['Index'], sample.inputs['Index'])
        control = nodes.new('ShaderNodeValue')
        control.name = 'R11 relaxed mouth corners normalized control'
        control.outputs[0].default_value = 0.
        driver = control.outputs[0].driver_add('default_value').driver
        driver.type = 'SCRIPTED'
        driver.expression = 'min(max(c,0.0),1.0)'
        variable = driver.variables.new()
        variable.name = 'c'
        variable.type = 'SINGLE_PROP'
        target = variable.targets[0]
        target.id_type = 'KEY'
        target.id = keys
        target.data_path = key.path_from_id('value')
        offset = nodes.new('ShaderNodeVectorMath')
        offset.operation = 'SCALE'
        links.new(sample.outputs['Value'], offset.inputs[0])
        links.new(control.outputs[0], offset.inputs['Scale'])
        length = nodes.new('ShaderNodeVectorMath')
        length.operation = 'LENGTH'
        links.new(sample.outputs['Value'], length.inputs[0])
        selected = nodes.new('ShaderNodeMath')
        selected.operation = 'GREATER_THAN'
        selected.inputs[1].default_value = 0.
        links.new(length.outputs['Value'], selected.inputs[0])
        moved = nodes.new('GeometryNodeSetPosition')
        moved.name = 'R11 fixed-L3 general-vector relaxed mouth corners'
        links.new(geometry, geometry_socket(moved.inputs))
        links.new(selected.outputs[0], moved.inputs['Selection'])
        links.new(offset.outputs['Vector'], moved.inputs['Offset'])
        enabled = nodes.new('ShaderNodeMath')
        enabled.operation = 'GREATER_THAN'
        enabled.inputs[1].default_value = 0.
        links.new(control.outputs[0], enabled.inputs[0])
        switch = nodes.new('GeometryNodeSwitch')
        switch.input_type = 'GEOMETRY'
        links.new(enabled.outputs[0], switch.inputs['Switch'])
        links.new(geometry, switch.inputs['False'])
        links.new(geometry_socket(moved.outputs), switch.inputs['True'])
        links.new(switch.outputs['Output'], geometry_socket(output.inputs))
        group['R11_full_vertex_count'] = N
        group['R11_local_axis'] = 'XYZ, transformed from observed world vectors'
        group['R11_vector_conversion_error_world_meters'] = conversion_error
        group['R11_lookup_sha256'] = hashlib.sha256(dense.tobytes()).hexdigest()
        group['R11_scope'] = 'Static in-memory preview only; caller verifies actual topology and local/external surfaces. No warm-smile/jaw/motion compatibility claim.'
        modifier = head.modifiers.new(GROUP, 'NODES')
        modifier.node_group = group
        if insertion < len(head.modifiers)-1:
            head.modifiers.move(len(head.modifiers)-1, insertion)
        final = list(head.modifiers)
        own = final.index(modifier)
        if tuple(mod for mod in final if mod != modifier) != original_stack:
            raise RuntimeError('Existing modifier order changed during insertion')
        if any(mod.type == 'ARMATURE' for mod in final[:own]):
            raise RuntimeError('Armature illegally precedes relaxed-mouth-corner field')
        if any(mod.type == 'NODES' for mod in final[own+1:]):
            raise RuntimeError('Existing GN illegally follows relaxed-mouth-corner field')
        if any(mod.type == 'MULTIRES' for mod in final[own+1:]):
            raise RuntimeError('L3 Multires must precede relaxed-mouth-corner field')
        group.update_tag()
        head.update_tag()
        return modifier, group, cache, mesh
    except Exception:
        if modifier is not None:
            head.modifiers.remove(modifier)
        if group is not None and group.users == 0:
            bpy.data.node_groups.remove(group)
        if cache is not None:
            bpy.data.objects.remove(cache, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise
