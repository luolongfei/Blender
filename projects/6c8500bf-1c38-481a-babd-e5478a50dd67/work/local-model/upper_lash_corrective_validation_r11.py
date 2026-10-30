"""Exact contracts and float32 reference for explicit late upper-lash fields.

No Blender work runs at import. No source, data, node, driver or marker is
modified here. Callers author the eight point fields and record these stable
contracts. A missing marker/entry leaves the original binding reference exact.
"""
import hashlib
import json
import math
import numpy as np

MARKER = 'R11_upper_lash_late_surface'
MODIFIER = 'R11 upper lash late surface'
NAMES = ('R11 Adult lashes upper L', 'R11 Adult lashes upper R')
ANCHORS = (.65, .8, .9, 1.)
FIELD_NAMES = tuple('r11_lash_late_'+kind+'_'+str(i) for kind in ('n', 'w') for i in range(4))
SOCKET_KEYS = tuple('weight'+str(i) for i in range(4))+('warm',)


def contract_hash(value):
    """Same serialization as face_fibre_binding_r11._hash, without importing it."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_hash(value):
    """Same shape/dtype/bytes contract as face_fibre_binding_r11._ah."""
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode()+value.dtype.str.encode()+value.tobytes()).hexdigest()


def driver_expressions():
    """Use variable b for this eye's blink and w for the warm smile key."""
    return {
        'weight0': 'max(0.0,min(1.0,(b-0.5)/(0.65-0.5),(0.8-b)/(0.8-0.65)))',
        'weight1': 'max(0.0,min(1.0,(b-0.65)/(0.8-0.65),(0.9-b)/(0.9-0.8)))',
        'weight2': 'max(0.0,min(1.0,(b-0.8)/(0.9-0.8),(1.0-b)/(1.0-0.9)))',
        'weight3': 'max(0.0,min(1.0,(b-0.9)/(1.0-0.9)))',
        'warm': 'max(0.0,min(1.0,w))',
    }


def hat_weights(blink):
    """Python-double driver arithmetic followed by actual socket float32."""
    b = float(blink)
    if not math.isfinite(b): raise RuntimeError('Nonfinite blink control')
    return np.asarray((
        max(0., min(1., (b-.5)/(.65-.5), (.8-b)/(.8-.65))),
        max(0., min(1., (b-.65)/(.8-.65), (.9-b)/(.9-.8))),
        max(0., min(1., (b-.8)/(.9-.8), (1.-b)/(1.-.9))),
        max(0., min(1., (b-.9)/(1.-.9))),
    ), dtype=np.float32)


def warm_weight(value):
    value = float(value)
    if not math.isfinite(value): raise RuntimeError('Nonfinite warm control')
    return np.float32(max(0., min(1., value)))


def apply_fields(base_local, fields, weights, warm):
    """Mirror four sequential VectorMath MULTIPLY/ADD/SCALE/ADD chains."""
    position = np.asarray(base_local, np.float32).copy()
    if position.ndim != 2 or position.shape[1] != 3 or not np.isfinite(position).all():
        raise RuntimeError('Finite N x 3 actual binding coordinates required')
    weights = np.asarray(weights, np.float32)
    warm = np.float32(warm)
    if weights.shape != (4,) or not np.isfinite(weights).all() or not np.isfinite(warm):
        raise RuntimeError('Four finite driver weights and warm input required')
    for i in range(4):
        neutral = np.asarray(fields['r11_lash_late_n_'+str(i)], np.float32)
        difference = np.asarray(fields['r11_lash_late_w_'+str(i)], np.float32)
        if neutral.shape != position.shape or difference.shape != position.shape:
            raise RuntimeError('Late-lash field shape differs from original point order')
        if not np.isfinite(neutral).all() or not np.isfinite(difference).all():
            raise RuntimeError('Nonfinite late-lash field')
        term = (difference*warm).astype(np.float32)
        term = (neutral+term).astype(np.float32)
        term = (term*weights[i]).astype(np.float32)
        position = (position+term).astype(np.float32)
    return position


def _id(value):
    if value is None: return None
    return {'idType': value.bl_rna.identifier, 'name': value.name_full,
            'library': value.library.filepath if getattr(value, 'library', None) else None}


def _plain(value):
    if value is None or isinstance(value, (str, bool, int)): return value
    if isinstance(value, float):
        if not math.isfinite(value): raise RuntimeError('Nonfinite contract value')
        return value
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, set): return sorted(_plain(v) for v in value)
    if hasattr(value, 'name_full') and hasattr(value, 'bl_rna'): return _id(value)
    if hasattr(value, 'to_list'): return _plain(value.to_list())
    return [_plain(v) for v in value]


def _socket(socket, index):
    row = {'index': index, 'identifier': socket.identifier, 'name': socket.name,
           'type': socket.bl_idname, 'enabled': bool(socket.enabled)}
    if hasattr(socket, 'default_value'): row['default'] = _plain(socket.default_value)
    return row


def capture_group(group, _visited=None):
    """Pure semantic node/interface/link contract; no memory pointers/UI layout."""
    if group is None or group.bl_idname != 'GeometryNodeTree': raise RuntimeError('Geometry node group required')
    visited = set() if _visited is None else set(_visited)
    if group.name_full in visited: raise RuntimeError('Recursive late-lash group is unsupported')
    visited.add(group.name_full)
    interface = []
    for item in group.interface.items_tree:
        if item.item_type != 'SOCKET': continue
        row = {'name': item.name, 'identifier': item.identifier, 'inOut': item.in_out,
               'socketType': item.socket_type}
        for name in ('default_value', 'min_value', 'max_value', 'attribute_domain',
                     'default_attribute_name', 'hide_value', 'force_non_field', 'structure_type'):
            if hasattr(item, name): row[name] = _plain(getattr(item, name))
        interface.append(row)
    nodes = []
    for node in group.nodes:
        row = {'name': node.name, 'blIdname': node.bl_idname, 'mute': bool(node.mute),
               'inputs': [_socket(socket, i) for i, socket in enumerate(node.inputs)],
               'outputs': [_socket(socket, i) for i, socket in enumerate(node.outputs)]}
        for name in ('operation', 'data_type', 'domain', 'clamp', 'transform_space',
                     'input_type', 'mode', 'component', 'interpolation_type', 'is_active_output'):
            if hasattr(node, name): row[name] = _plain(getattr(node, name))
        if getattr(node, 'node_tree', None) is not None:
            row['group'] = capture_group(node.node_tree, visited)
        nodes.append(row)
    links = []
    for link in group.links:
        links.append({'fromNode': link.from_node.name, 'fromSocket': link.from_socket.identifier,
                      'fromIndex': list(link.from_node.outputs).index(link.from_socket),
                      'toNode': link.to_node.name, 'toSocket': link.to_socket.identifier,
                      'toIndex': list(link.to_node.inputs).index(link.to_socket),
                      'muted': bool(link.is_muted)})
    return {'name': group.name_full, 'blIdname': group.bl_idname, 'interface': interface,
            'nodes': sorted(nodes, key=lambda r: r['name']),
            'links': sorted(links, key=lambda r: (r['fromNode'], r['fromIndex'], r['toNode'], r['toIndex']))}


def _driver_path(modifier, identifier):
    return modifier.path_from_id()+'['+json.dumps(identifier)+']'


def _driver_curves(obj, modifier, socket_ids):
    if set(socket_ids) != set(SOCKET_KEYS) or len(set(socket_ids.values())) != 5:
        raise RuntimeError('Five distinct explicit late-lash interface identifiers required')
    drivers = obj.animation_data.drivers if obj.animation_data else []
    result = {}
    for key in SOCKET_KEYS:
        path = _driver_path(modifier, socket_ids[key]); matches = [f for f in drivers if f.data_path == path]
        if len(matches) != 1: raise RuntimeError('One exact late-lash driver required for '+key)
        result[key] = matches[0]
    return result


def capture_drivers(obj, modifier, socket_ids):
    """Stable equations and targets, deliberately excluding pose-varying values."""
    curves = _driver_curves(obj, modifier, socket_ids); rows = {}
    for key in SOCKET_KEYS:
        curve = curves[key]; driver = curve.driver
        if len(curve.modifiers): raise RuntimeError('Late-lash input driver modifiers are unsupported')
        variables = []
        for variable in driver.variables:
            targets = []
            for target in variable.targets:
                row = {'id': _id(target.id), 'dataPath': target.data_path}
                for name in ('id_type', 'bone_target', 'transform_type', 'transform_space',
                             'rotation_mode', 'context_property', 'use_fallback_value', 'fallback_value'):
                    if hasattr(target, name): row[name] = _plain(getattr(target, name))
                targets.append(row)
            variables.append({'name': variable.name, 'type': variable.type, 'targets': targets})
        rows[key] = {'path': curve.data_path, 'arrayIndex': curve.array_index, 'mute': bool(curve.mute),
                     'driverType': driver.type, 'expression': driver.expression, 'useSelf': bool(driver.use_self),
                     'variables': variables}
    return {'object': obj.name_full, 'modifier': modifier.name, 'interfaceIdentifiers': dict(socket_ids), 'drivers': rows}


def _validate_controls(obj, modifier, socket_ids, head):
    curves = _driver_curves(obj, modifier, socket_ids); expressions = driver_expressions()
    side = obj.name[-1]; keys = head.data.shape_keys
    if keys is None: raise RuntimeError('Current head facial controls missing')
    blink_name = 'R11 blink '+side; warm_name = 'R11 warm closed smile'
    blink = float(keys.key_blocks[blink_name].value); warm = warm_weight(keys.key_blocks[warm_name].value)
    expected = dict(zip(SOCKET_KEYS[:4], hat_weights(blink))); expected['warm'] = warm
    evaluated = obj.evaluated_get(__import__('bpy').context.evaluated_depsgraph_get())
    actual_modifier = evaluated.modifiers.get(modifier.name)
    if actual_modifier is None: raise RuntimeError('Evaluated late-lash modifier missing')
    for key in SOCKET_KEYS:
        curve = curves[key]; driver = curve.driver; variable_name = 'w' if key == 'warm' else 'b'
        target_key = warm_name if key == 'warm' else blink_name
        path = 'key_blocks['+json.dumps(target_key)+'].value'
        if curve.mute or driver.type != 'SCRIPTED' or driver.use_self or driver.expression != expressions[key]:
            raise RuntimeError('Late-lash driver equation differs from recorded interpolation '+key)
        if len(driver.variables) != 1: raise RuntimeError('One direct current facial input required '+key)
        variable = driver.variables[0]
        if variable.name != variable_name or variable.type != 'SINGLE_PROP' or len(variable.targets) != 1:
            raise RuntimeError('Incorrect direct late-lash control variable '+key)
        target = variable.targets[0]
        if target.id != keys or target.data_path != path:
            raise RuntimeError('Late-lash driver is not bound to the current head key '+key)
        actual = float(actual_modifier[socket_ids[key]])
        if not math.isfinite(actual) or abs(actual-float(expected[key])) > 1e-6:
            raise RuntimeError('Evaluated late-lash driver input differs from current controls '+key)
    return hat_weights(blink), warm


def expected_final(obj, base_local, binding_record):
    """Validate the complete optional correction and return its F32 reference."""
    import bpy
    scene = bpy.context.scene
    if MARKER not in scene: return base_local
    record = json.loads(scene[MARKER])
    if record.get('schemaVersion') != 1 or not isinstance(record.get('objects'), dict):
        raise RuntimeError('Unsupported late upper-lash corrective record')
    if obj.name not in record['objects']: return base_local
    if obj.name not in NAMES: raise RuntimeError('Late-lash record targets a non-upper-lash object')
    entry = record['objects'][obj.name]
    if entry['modifier'] != MODIFIER: raise RuntimeError('Unexpected late-lash modifier name')
    modifier = obj.modifiers.get(MODIFIER)
    if modifier is None or modifier.type != 'NODES' or not modifier.show_viewport or not modifier.show_render:
        raise RuntimeError('Recorded late-lash modifier is not active')
    active = [m for m in obj.modifiers if m.show_viewport or m.show_render]
    if active[-1] != modifier: raise RuntimeError('Late-lash correction must be the final active modifier')
    group = modifier.node_group
    if group is None or group.name != entry['nodeGroup'] or contract_hash(capture_group(group)) != entry['nodeGroupContractSha256']:
        raise RuntimeError('Late-lash node group contract changed')
    socket_ids = entry['interfaceIdentifiers']
    if contract_hash(capture_drivers(obj, modifier, socket_ids)) != entry['driverContractSha256']:
        raise RuntimeError('Late-lash driver contract changed')
    lookup = bpy.data.objects.get(entry['lookupObject'])
    expected_count = len(base_local)
    if lookup is None or lookup.type != 'MESH' or len(lookup.data.vertices) != expected_count or entry['lookupVertexCount'] != expected_count:
        raise RuntimeError('Late-lash lookup must have one vertex per original curve point')
    if lookup.data.shape_keys is not None or any(m.show_viewport or m.show_render for m in lookup.modifiers):
        raise RuntimeError('Late-lash lookup fields must be immutable raw mesh attributes')
    if set(entry['fieldsSha256']) != set(FIELD_NAMES): raise RuntimeError('Exactly eight late-lash fields required')
    fields = {}
    for name in FIELD_NAMES:
        attribute = lookup.data.attributes.get(name)
        if attribute is None or attribute.domain != 'POINT' or attribute.data_type != 'FLOAT_VECTOR' or len(attribute.data) != expected_count:
            raise RuntimeError('Incorrect late-lash lookup attribute '+name)
        values = np.empty((expected_count, 3), np.float32); attribute.data.foreach_get('vector', values.ravel())
        if not np.isfinite(values).all() or array_hash(values) != entry['fieldsSha256'][name]:
            raise RuntimeError('Late-lash lookup field changed '+name)
        fields[name] = values
    head = bpy.data.objects[binding_record['head']]
    weights, warm = _validate_controls(obj, modifier, socket_ids, head)
    return apply_fields(base_local, fields, weights, warm)
