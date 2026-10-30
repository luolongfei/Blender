"""Explicit native eye-raycast space migration for R11 cervical skinning.

The unbound lid builder tags its existing RELATIVE sclera ObjectInfo and
Raycast nodes. Only the binder calls migrate(), in actual REST, after creating
the cranial carrier. No head/eye mesh, driver, frame or file is changed here.

Blender 5.2.3 native contract: ORIGINAL Geometry is evaluated eye-local geometry;
ORIGINAL Transform is eye-world. Two complete matrix transforms yield
C_pose^-1 E_world, which equals the head pre-Armature space because this binder
explicitly preserves H_rest == C_rest. Do not decompose scale/rotation.

V2 additionally recognizes the exact lower-face continuity scalar driver.
This is a read-only exception for its authored strength, never its transform.
V4 also migrates each radial outer-corner eye's two existing before/after rays.
Both consumers share the same native full-matrix eye-to-cranial conversion.
"""
import hashlib
import json

HEAD = 'R11 Female head study'
RIG = 'Character_Rig'
CARRIER = 'R11 Cranial attachment space'
EYE_TAG = 'R11_prearm_sclera_side'
RAY_TAG = 'R11_prearm_sclera_raycast'
FRAME_TAG = 'R11_prearm_cranial_frame'
OWNER_TAG = 'R11_prearm_eye_space_owner'
OWNER = 'r11-native-eye-space-v1'
PREFIX = 'R11 prearm eye / '
RELIEF_GROUP = 'R11 upper lid surface relief'
RELIEF_CONTROL = 'R11 upper lid relief'
LOWER_FACE_GROUP = 'R11 lower face surface continuity'
LOWER_FACE_CONTROL = 'R11 lower face continuity'
RADIAL_GROUP = 'R11 refined outer eye corners radial'
RADIAL_EYE_TAG = 'R11_prearm_radial_outerend_sclera_side'
RADIAL_RAY_TAG = 'R11_prearm_radial_outerend_sclera_raycast'
RADIAL_STAGE_TAG = 'R11_outerend_ray_stage'


def _geometry_socket(sockets):
    found = [s for s in sockets if s.type == 'GEOMETRY']
    if len(found) != 1:
        raise RuntimeError('Expected one geometry socket')
    return found[0]


def _groups(head):
    found = {}
    def visit(group):
        if group.name in found:
            return
        found[group.name] = group
        for node in group.nodes:
            child = getattr(node, 'node_tree', None)
            if child is not None:
                visit(child)
    for mod in head.modifiers:
        if mod.type == 'NODES' and mod.node_group is not None:
            visit(mod.node_group)
    return found


def _edge(link):
    return [link.from_node.name, link.from_socket.identifier,
            link.to_node.name, link.to_socket.identifier]


def _drivers(group):
    animation = group.animation_data
    return [{'path': f.data_path, 'index': f.array_index, 'mute': f.mute,
             'type': f.driver.type, 'expression': f.driver.expression,
             'variables': [[v.name, v.type, [[t.id_type, getattr(t.id, 'name', None), t.data_path]
                              for t in v.targets]] for v in f.driver.variables]}
            for f in animation.drivers] if animation else []


def allowed_head_property_references(head):
    """Allow exact authored relief/continuity scalar reads, never transform reads.

    Mid-blink arc drivers reference the retained Key datablock, not the head
    Object; its inner-contact strength is an undriven Value node. Neither
    requires a new head-reference exception.
    """
    groups = _groups(head)
    result = _lower_face_property_reference(head, groups.get(LOWER_FACE_GROUP))
    group = groups.get(RELIEF_GROUP)
    if group is None:
        return result
    if RELIEF_CONTROL not in head:
        raise RuntimeError('Existing relief head property disappeared')
    for side in ('L', 'R'):
        node = group.nodes.get('R11 upper-lid open-eye influence ' + side)
        if node is None or node.bl_idname != 'ShaderNodeValue':
            raise RuntimeError('Existing relief influence node changed')
        path = node.outputs[0].path_from_id('default_value')
        curves = [f for f in group.animation_data.drivers if f.data_path == path] if group.animation_data else []
        if len(curves) != 1:
            raise RuntimeError('Expected one existing relief influence driver per side')
        variables = [v for v in curves[0].driver.variables if v.name == 'r']
        if len(variables) != 1 or variables[0].type != 'SINGLE_PROP' or len(variables[0].targets) != 1:
            raise RuntimeError('Existing relief scalar driver changed')
        target = variables[0].targets[0]
        if target.id != head or target.id_type != 'OBJECT' or target.data_path != '[' + json.dumps(RELIEF_CONTROL) + ']':
            raise RuntimeError('Only the exact existing head relief property is allowed')
        result.append({'id': group.name, 'dataPath': path, 'variable': 'r',
                       'variableType': 'SINGLE_PROP', 'targetPath': target.data_path})
    return result


def _lower_face_property_reference(head, group):
    """Match the one known continuity control exactly; return no wildcards."""
    if group is None:
        return []
    if group.library or group.override_library or LOWER_FACE_CONTROL not in head:
        raise RuntimeError('Existing local lower-face continuity property disappeared or is linked')
    node = group.nodes.get('R11 lower-face continuity normalized control')
    if node is None or node.bl_idname != 'ShaderNodeValue' or len(node.outputs) != 1:
        raise RuntimeError('Existing lower-face continuity influence node changed')
    path = node.outputs[0].path_from_id('default_value')
    curves = [f for f in group.animation_data.drivers if f.data_path == path] if group.animation_data else []
    if len(curves) != 1:
        raise RuntimeError('Expected one existing lower-face continuity driver')
    curve = curves[0]
    driver = curve.driver
    if curve.array_index != 0 or curve.mute or driver.type != 'SCRIPTED' or driver.expression != 'min(max(c,0.0),1.0)':
        raise RuntimeError('Existing lower-face continuity driver definition changed')
    variables = list(driver.variables)
    if len(variables) != 1 or variables[0].name != 'c' or variables[0].type != 'SINGLE_PROP' or len(variables[0].targets) != 1:
        raise RuntimeError('Expected the exact lower-face continuity scalar variable')
    target = variables[0].targets[0]
    if target.id != head or target.id_type != 'OBJECT' or target.data_path != '[' + json.dumps(LOWER_FACE_CONTROL) + ']':
        raise RuntimeError('Only the exact existing lower-face continuity property is allowed')
    return [{'id': group.name, 'dataPath': path, 'variable': 'c',
             'variableType': 'SINGLE_PROP', 'targetPath': target.data_path}]


def prepare(head):
    """Read-only exact migration plan; reject unadapted consumers, no guessing."""
    rows = []
    for group in _groups(head).values():
        actual_eye_nodes = [n for n in group.nodes if n.bl_idname == 'GeometryNodeObjectInfo'
                            and getattr(n.inputs['Object'].default_value, 'name', None)
                            in ('R11 Adult sclera L', 'R11 Adult sclera R') and n.outputs['Geometry'].is_linked]
        relief = group.name == RELIEF_GROUP
        radial = group.name == RADIAL_GROUP
        radial_tags = any(RADIAL_EYE_TAG in n or RADIAL_RAY_TAG in n or RADIAL_STAGE_TAG in n for n in group.nodes)
        if radial_tags and not radial:
            raise RuntimeError('Radial eye-query tags occur outside the exact reviewed group: ' + group.name)
        if radial and any(EYE_TAG in n or RAY_TAG in n for n in group.nodes):
            raise RuntimeError('Radial and single-ray migration tags must not be mixed')
        eye_tag = RADIAL_EYE_TAG if radial else EYE_TAG
        ray_tag = RADIAL_RAY_TAG if radial else RAY_TAG
        eyes = actual_eye_nodes if relief else [n for n in group.nodes if eye_tag in n]
        rays = [n for n in group.nodes if ray_tag in n]
        if not eyes and not rays:
            if actual_eye_nodes or radial:
                raise RuntimeError('Unadapted real-eye geometry reader: ' + group.name)
            continue
        if set(actual_eye_nodes) != set(eyes):
            raise RuntimeError('Additional unadapted real-eye reader in ' + group.name)
        if group.library or group.override_library:
            raise RuntimeError('Keep externally linked lid node groups unchanged')
        if any(n.get(OWNER_TAG) or n.get(FRAME_TAG) for n in group.nodes):
            raise RuntimeError('Existing pre-Armature eye space must be preserved')
        sides = [n.inputs['Object'].default_value.name[-1] if relief else n[eye_tag] for n in eyes]
        ray_sides = [n[ray_tag] for n in rays]
        if sorted(sides) != ['L', 'R'] or (not relief and not radial and sorted(ray_sides) != ['L', 'R']) or (relief and rays):
            raise RuntimeError('Expected exactly one marked sclera and raycast per side')
        if radial:
            expected = [('L', 'after'), ('L', 'before'), ('R', 'after'), ('R', 'before')]
            if sorted((n[ray_tag], n.get(RADIAL_STAGE_TAG, '')) for n in rays) != expected:
                raise RuntimeError('Radial eyes require exactly one before and after ray per side')
        row = {'group': group.name, 'branch': 'existing-relief' if relief else ('radial-two-rays' if radial else 'marked-raycast'),
               'sides': [], 'addedNodes': [], 'originalDrivers': _drivers(group),
               'originalSpaceProperties': {k: group[k] for k in
                   ('R11_query_space', 'R11_current_armature_support') if k in group}}
        for side in ('L', 'R'):
            eye = next(n for n in eyes if (n.inputs['Object'].default_value.name[-1] if relief else n[eye_tag]) == side)
            if eye.bl_idname != 'GeometryNodeObjectInfo' or eye.transform_space != 'RELATIVE':
                raise RuntimeError('Only documented unbound RELATIVE eye info can migrate')
            obj = eye.inputs['Object'].default_value
            if obj is None or obj.name != 'R11 Adult sclera ' + side or obj.type != 'MESH':
                raise RuntimeError('Marked node does not read the actual R11 sclera')
            if eye.inputs['As Instance'].is_linked or eye.inputs['As Instance'].default_value:
                raise RuntimeError('Expected evaluated non-instance sclera geometry')
            if eye.inputs['Object'].is_linked or any(s.is_linked for s in eye.outputs if s.name != 'Geometry'):
                raise RuntimeError('Additional eye transform consumers need explicit adaptation')
            output = eye.outputs['Geometry']
            if relief:
                consumers = [link.to_node for link in output.links]
                if sorted(n.bl_idname for n in consumers) != ['GeometryNodeProximity', 'GeometryNodeSampleNearestSurface']:
                    raise RuntimeError('Existing relief must retain its two known geometry consumers')
            elif radial:
                consumers = [n for n in rays if n[ray_tag] == side]
                if len(consumers) != 2 or any(n.bl_idname != 'GeometryNodeRaycast' for n in consumers) or len(output.links) != 2:
                    raise RuntimeError('Radial eye must feed only its two existing native Raycasts')
                for ray in consumers:
                    if ray.data_type != 'FLOAT' or ray.inputs['Interpolation'].is_linked or ray.inputs['Interpolation'].default_value != 'Interpolated':
                        raise RuntimeError('Reviewed radial Raycast interpolation changed')
            else:
                consumers = [next(n for n in rays if n[RAY_TAG] == side)]
                if consumers[0].bl_idname != 'GeometryNodeRaycast' or len(output.links) != 1:
                    raise RuntimeError('Marked eye must feed only its native Raycast')
            targets = []
            for consumer in consumers:
                target = _geometry_socket(consumer.inputs)
                if len(target.links) != 1 or target.links[0].from_socket != output:
                    raise RuntimeError('Existing eye target link changed')
                target_row = {'node': consumer.name, 'socket': target.identifier, 'originalLink': _edge(target.links[0])}
                if radial: target_row['rayStage'] = consumer[RADIAL_STAGE_TAG]
                targets.append(target_row)
            rowside = {'side': side, 'eyeNode': eye.name, 'eyeObject': obj.name,
                       'targets': sorted(targets, key=lambda t: t['node'])}
            row['sides'].append(rowside)
        reserved = [PREFIX + x for x in ('cranial frame', 'inverse cranial',
                     'eye world L', 'eye rest L', 'eye world R', 'eye rest R')]
        if any(group.nodes.get(n) for n in reserved):
            raise RuntimeError('Reserved matrix conversion nodes already exist')
        row['reservedNodes'] = reserved
        rows.append(row)
    return {'schema': 1, 'groups': sorted(rows, key=lambda r: r['group']),
            'spaceFormula': 'C_pose^-1 * E_world; requires H_rest == C_rest',
            'applied': False}


def state(record):
    """Only this adapter's structure; excludes changing expression driver values."""
    import bpy
    rows = []
    for row in record['groups']:
        group = bpy.data.node_groups[row['group']]
        names = set(row['addedNodes']) | {s['eyeNode'] for s in row['sides']} | {t['node'] for s in row['sides'] for t in s['targets']}
        node_rows = []
        for name in sorted(names):
            node = group.nodes.get(name)
            if node is None:
                node_rows.append([name, None]); continue
            inputs = {}
            for socket in node.inputs:
                if socket.type == 'OBJECT':
                    inputs[socket.identifier] = getattr(socket.default_value, 'name', None)
                elif socket.name in ('As Instance', 'Mode', 'Interpolation'):
                    inputs[socket.identifier] = socket.default_value
            node_rows.append([name, node.bl_idname, getattr(node, 'transform_space', None),
                              node.get(OWNER_TAG), node.get(FRAME_TAG), node.get(EYE_TAG), node.get(RAY_TAG),
                              node.get(RADIAL_EYE_TAG), node.get(RADIAL_RAY_TAG), node.get(RADIAL_STAGE_TAG),
                              getattr(node, 'data_type', None), inputs])
        links = sorted([_edge(link) for link in group.links
                        if link.from_node.name in names or link.to_node.name in names])
        rows.append({'group': row['group'], 'nodes': node_rows, 'links': links, 'drivers': _drivers(group),
                     'spaceProperties': {k: group[k] for k in
                         ('R11_query_space', 'R11_current_armature_support') if k in group}})
    return rows


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _undo(record):
    import bpy
    for row in record['groups']:
        group = bpy.data.node_groups[row['group']]
        for name in reversed(row['addedNodes']):
            node = group.nodes.get(name)
            if node is not None:
                group.nodes.remove(node)
        for item in row['sides']:
            eye = group.nodes[item['eyeNode']]
            eye.transform_space = 'RELATIVE'
            for dest in item['targets']:
                target = next(s for s in group.nodes[dest['node']].inputs if s.identifier == dest['socket'])
                for link in list(target.links):
                    group.links.remove(link)
                group.links.new(eye.outputs['Geometry'], target)
        for key in ('R11_query_space', 'R11_current_armature_support'):
            if key in row['originalSpaceProperties']:
                group[key] = row['originalSpaceProperties'][key]
            elif key in group:
                del group[key]
        row['addedNodes'] = []
    record['applied'] = False


def migrate(plan, head, carrier):
    """Bind-time reversible edit. Caller stays in actual rig REST throughout."""
    import bpy
    import numpy as np
    if prepare(head) != plan:
        raise RuntimeError('Tagged eye/raycast inputs changed after the binding preflight')
    record = json.loads(json.dumps(plan))
    rig = bpy.data.objects[RIG]
    if head.name != HEAD or head.parent != rig or head.parent_type != 'OBJECT':
        raise RuntimeError('Run after converting the head object to the rig REST frame')
    if carrier.name != CARRIER or carrier.type != 'EMPTY' or carrier.instance_collection:
        raise RuntimeError('Expected a non-instancing native cranial Empty')
    if carrier.get('R11_original_head_space_source') != HEAD or carrier.parent != rig or carrier.parent_type != 'BONE' or carrier.parent_bone != 'head':
        raise RuntimeError('Cranial attachment provenance does not match the binder')
    if rig.data.pose_position != 'REST':
        raise RuntimeError('Binding eye-space migration requires actual REST evaluation')
    h, c = np.asarray(head.matrix_world, float), np.asarray(carrier.matrix_world, float)
    if not np.isfinite(h).all() or not np.isfinite(c).all() or abs(np.linalg.det(c[:3, :3])) < 1e-10:
        raise RuntimeError('Invalid or singular cranial matrix')
    error = float(abs(h - c).max())
    if error > 2e-7:
        raise RuntimeError('Head and cranial REST local spaces differ: ' + str(error))
    record['headRestWorld'] = h.tolist()
    record['cranialRestWorld'] = c.tolist()
    record['actualRestFrameMaxError'] = error
    try:
        for row in record['groups']:
            group = bpy.data.node_groups[row['group']]
            def new(kind, name):
                node = group.nodes.new(kind)
                node.name = PREFIX + name
                node[OWNER_TAG] = OWNER
                row['addedNodes'].append(node.name)
                return node
            frame = new('GeometryNodeObjectInfo', 'cranial frame')
            frame.transform_space = 'ORIGINAL'
            frame.inputs['Object'].default_value = carrier
            frame.inputs['As Instance'].default_value = False
            frame[FRAME_TAG] = 1
            inverse = new('FunctionNodeInvertMatrix', 'inverse cranial')
            group.links.new(frame.outputs['Transform'], inverse.inputs['Matrix'])
            for item in row['sides']:
                eye = group.nodes[item['eyeNode']]
                eye.transform_space = 'ORIGINAL'
                world = new('GeometryNodeTransform', 'eye world ' + item['side'])
                rest = new('GeometryNodeTransform', 'eye rest ' + item['side'])
                for node in (world, rest):
                    node.inputs['Mode'].default_value = 'Matrix'
                    if node.inputs['Mode'].default_value != 'Matrix':
                        raise RuntimeError('Native Transform Geometry Matrix mode did not apply')
                group.links.new(eye.outputs['Geometry'], world.inputs['Geometry'])
                group.links.new(eye.outputs['Transform'], world.inputs['Transform'])
                group.links.new(world.outputs['Geometry'], rest.inputs['Geometry'])
                group.links.new(inverse.outputs['Matrix'], rest.inputs['Transform'])
                for dest in item['targets']:
                    target = next(s for s in group.nodes[dest['node']].inputs if s.identifier == dest['socket'])
                    for link in list(target.links):
                        group.links.remove(link)
                    group.links.new(rest.outputs['Geometry'], target)
            group['R11_query_space'] = 'head pre-Armature; full native C_pose^-1 E_world'
            group['R11_current_armature_support'] = True
        record['applied'] = True
        record['installedStateSha256'] = _hash(state(record))
        return record
    except Exception:
        _undo(record)
        raise


def validate_restore(record):
    if record and record.get('applied') and _hash(state(record)) != record['installedStateSha256']:
        raise RuntimeError('Preserve subsequent eye-space graph edits before rollback')


def restore(record):
    if not record or not record.get('applied'):
        return
    validate_restore(record)
    _undo(record)


def eye_to_prearmature(eye_world, carrier_world, eye_local_points):
    """Pure NumPy reference for independent pose/gaze checks, not native proof."""
    import numpy as np
    m = np.linalg.inv(np.asarray(carrier_world, float)) @ np.asarray(eye_world, float)
    p = np.asarray(eye_local_points, float)
    return p @ m[:3, :3].T + m[:3, 3]


def probe_current_mapping(record, tolerance=2e-6):
    """Evaluate copies of the installed native target branches at CURRENT pose/gaze.

    No action, pose, driver or production graph is edited. Temporary empty host
    meshes and copied node groups are removed in finally. Call across actual
    head/neck poses AND gaze cases; this establishes coordinates, not lid contact.
    """
    import bpy
    import numpy as np
    validate_restore(record)
    if not record or not record.get('applied'):
        return {'groups': [], 'scope': 'No marked raycast target to migrate'}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    carrier = bpy.data.objects[CARRIER].evaluated_get(depsgraph)
    matrix = np.asarray(carrier.matrix_world, float)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-10:
        raise RuntimeError('Current cranial matrix cannot be inverted')
    results = []
    for row in record['groups']:
        original = bpy.data.node_groups[row['group']]
        for item in row['sides']:
            copied = host = mesh = evaluated = None
            try:
                copied = original.copy()
                outputs = [n for n in copied.nodes if n.bl_idname == 'NodeGroupOutput' and n.is_active_output]
                if len(outputs) != 1:
                    raise RuntimeError('Expected one active native lid group output')
                target = _geometry_socket(outputs[0].inputs)
                for link in list(target.links):
                    copied.links.remove(link)
                result = copied.nodes[PREFIX + 'eye rest ' + item['side']].outputs['Geometry']
                # Capture the actual native Invertible output on the temporary
                # target points. An invalid inverse otherwise silently gives I.
                store = copied.nodes.new('GeometryNodeStoreNamedAttribute')
                store.data_type = 'BOOLEAN'; store.domain = 'POINT'
                store.inputs['Name'].default_value = 'R11_probe_inverse_ok'
                copied.links.new(result, store.inputs['Geometry'])
                copied.links.new(copied.nodes[PREFIX + 'inverse cranial'].outputs['Invertible'], store.inputs['Value'])
                copied.links.new(store.outputs['Geometry'], target)
                mesh = bpy.data.meshes.new('R11 temporary eye space points')
                host = bpy.data.objects.new('R11 temporary eye space probe', mesh)
                bpy.context.scene.collection.objects.link(host)
                host.hide_render = True
                mod = host.modifiers.new('R11 temporary actual mapping', 'NODES')
                mod.node_group = copied
                bpy.context.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
                evaluated = host.evaluated_get(depsgraph)
                observed = evaluated.to_mesh()
                p = np.empty((len(observed.vertices), 3), float)
                observed.vertices.foreach_get('co', p.ravel())
                flags = observed.attributes.get('R11_probe_inverse_ok')
                if flags is None or flags.domain != 'POINT' or not len(p):
                    raise RuntimeError('Actual inverse proof or eye points missing')
                ok = np.empty(len(p), bool); flags.data.foreach_get('value', ok)
                if not ok.all():
                    raise RuntimeError('Actual native cranial inverse returned false')
                eye = bpy.data.objects[item['eyeObject']].evaluated_get(depsgraph)
                eye_mesh = eye.to_mesh()
                try:
                    q = np.empty((len(eye_mesh.vertices), 3), float)
                    eye_mesh.vertices.foreach_get('co', q.ravel())
                    expected = eye_to_prearmature(eye.matrix_world, matrix, q)
                finally:
                    eye.to_mesh_clear()
                if p.shape != expected.shape or not np.isfinite(p).all() or not np.isfinite(expected).all():
                    raise RuntimeError('Actual target/source eye topology differs')
                error = float(np.linalg.norm(p - expected, axis=1).max())
                if error > tolerance:
                    raise RuntimeError('Actual native eye/prearm mapping mismatch: ' + str(error))
                results.append({'group': row['group'], 'side': item['side'], 'points': len(p),
                                'maximumErrorMeters': error, 'nativeInvertibleAllTrue': True})
            finally:
                if evaluated is not None:
                    evaluated.to_mesh_clear()
                if host is not None:
                    bpy.data.objects.remove(host, do_unlink=True)
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
                if copied is not None and copied.users == 0:
                    bpy.data.node_groups.remove(copied)
    return {'groups': results, 'toleranceMeters': tolerance,
            'scope': 'Actual installed native mapping at current pose/gaze only; no contact or appearance approval'}
