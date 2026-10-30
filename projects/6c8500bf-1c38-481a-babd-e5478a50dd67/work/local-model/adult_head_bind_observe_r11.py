"""Read-only current R11 head/neck binding conversion observation.

Fresh background source only. Saves JSON/NPZ observations, never a Blend.
Exports current cage mix, mask, all original weights, actual evaluated neck
boundary correspondence, rig rest/pose, direct and nested cranial attachments,
and references to the head object. Does not install weights or change parents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
BRIDGE = 'R11 Continuous neck bridge'
BODY = 'R01 Head'
RIG = 'Character_Rig'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def path(relative, suffix):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Project-relative reconstruction path required')
    p = (project_path() / relative).resolve()
    p.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if p.suffix != suffix:
        raise ValueError('Wrong output/source suffix')
    return p


def array(items, field, width=3, dtype=np.float32):
    a = np.empty((len(items), width), dtype=dtype)
    items.foreach_get(field, a.ravel())
    return a


def matrix(value):
    return [list(row) for row in value]


def ah(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def world(points, transform):
    m = np.asarray(transform, float)
    return np.asarray(points) @ m[:3, :3].T + m[:3, 3]


def weights(obj, vertices=None):
    vertices = obj.data.vertices if vertices is None else vertices
    result = np.zeros((len(vertices), len(obj.vertex_groups)), np.float32)
    for i, vertex in enumerate(vertices):
        for group in vertex.groups:
            if group.group >= result.shape[1]:
                raise RuntimeError('Vertex group index exceeds actual group table: ' + obj.name)
            result[i, group.group] = group.weight
    return result


def attachment(obj):
    uv = api('production_uv_r08')
    return {'name': obj.name, 'type': obj.type, 'data': getattr(obj.data, 'name', None),
            'parent': obj.parent.name if obj.parent else None, 'parentType': obj.parent_type,
            'parentBone': obj.parent_bone, 'matrixParentInverse': matrix(obj.matrix_parent_inverse),
            'matrixBasis': matrix(obj.matrix_basis), 'matrixWorld': matrix(obj.matrix_world),
            'animation': uv['_animation'](obj),
            'constraints': [{'name': c.name, 'type': c.type, 'settings': uv['_scalar_rna'](c),
                             'target': getattr(getattr(c, 'target', None), 'name', None)} for c in obj.constraints]}


def data_contract(obj):
    uv = api('production_uv_r08')
    keys = obj.data.shape_keys
    return {'data': obj.data.name, 'coordinates': ah(array(obj.data.vertices, 'co')),
            'loops': ah(array(obj.data.loops, 'vertex_index', 1, np.int32)),
            'groups': [(g.name, g.index) for g in obj.vertex_groups], 'weights': ah(weights(obj)),
            'keys': [{'name': k.name, 'coordinates': ah(array(k.data, 'co')),
                      'relative': k.relative_key.name, 'vertexGroup': k.vertex_group,
                      'value': float(k.value), 'mute': bool(k.mute)} for k in keys.key_blocks] if keys else [],
            'keyAnimation': uv['_animation'](keys) if keys else None,
            'modifiers': [{'name': m.name, 'type': m.type, 'settings': uv['_scalar_rna'](m),
                           'object': getattr(getattr(m, 'object', None), 'name', None),
                           'nodeGroup': getattr(getattr(m, 'node_group', None), 'name', None)} for m in obj.modifiers],
            'uvs': {v.name: ah(array(v.data, 'uv', 2)) for v in obj.data.uv_layers},
            'materials': [m.name if m else None for m in obj.data.materials]}


def descendant(obj, root):
    current = obj.parent
    visited = set()
    while current is not None:
        if current == root:
            return True
        if current.name in visited:
            raise RuntimeError('Object parent cycle')
        visited.add(current.name)
        current = current.parent
    return False


def references(head):
    import bpy
    found = {'driverTargets': [], 'constraints': [], 'nodeObjectInputs': []}
    ids = list(bpy.data.objects) + list(bpy.data.shape_keys) + list(bpy.data.node_groups)
    for block in ids:
        animation = getattr(block, 'animation_data', None)
        for curve in animation.drivers if animation else []:
            for var in curve.driver.variables:
                for target in var.targets:
                    if target.id == head:
                        found['driverTargets'].append({'id': block.name, 'dataPath': curve.data_path,
                            'variable': var.name, 'variableType': var.type, 'targetPath': target.data_path,
                            'transformSpace': target.transform_space, 'transformType': target.transform_type})
    for obj in bpy.data.objects:
        for constraint in obj.constraints:
            if getattr(constraint, 'target', None) == head:
                found['constraints'].append({'object': obj.name, 'name': constraint.name, 'type': constraint.type})
    for group in bpy.data.node_groups:
        for node in group.nodes:
            for socket in node.inputs:
                if getattr(socket, 'type', None) == 'OBJECT' and getattr(socket, 'default_value', None) == head:
                    found['nodeObjectInputs'].append({'group': group.name, 'node': node.name,
                        'socket': socket.name, 'nodeType': node.bl_idname,
                        'transformSpace': getattr(node, 'transform_space', None)})
    return found


def evaluated(obj, prefix, arrays, include_mesh=False):
    import bpy
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = ev.to_mesh()
    try:
        data = api('adult_neck_bridge_r11')['_mesh_arrays'](mesh, ev.matrix_world)
        w = weights(obj, mesh.vertices)
        if include_mesh:
            for key in ('world', 'loops', 'face_starts', 'face_sizes', 'triangles'):
                arrays[prefix + '_' + key] = data[key]
        arrays[prefix + '_weights'] = w
        row = {'prefix': prefix, 'vertices': len(mesh.vertices), 'faces': len(mesh.polygons),
               'matrixWorld': matrix(ev.matrix_world),
               'groups': [(g.name, g.index) for g in obj.vertex_groups],
               'weightShape': list(w.shape), 'bounds': [data['world'].min(0).tolist(), data['world'].max(0).tolist()]}
        return row, data, w
    finally:
        ev.to_mesh_clear()


def observe(source_relative, output_relative):
    import bpy
    source = path(source_relative, '.blend')
    output = path(output_relative, '.json')
    npz = output.with_suffix('.npz')
    if output.exists() or npz.exists():
        raise RuntimeError('Preserve existing observations')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh saved explicit R11 source required')
    scene = bpy.context.scene
    rig = bpy.data.objects[RIG]
    head = bpy.data.objects[HEAD]
    bridge = bpy.data.objects[BRIDGE]
    if head.parent != rig or head.parent_type != 'BONE' or head.parent_bone != 'head':
        raise RuntimeError('This observation specifically scopes the existing rigid-head conversion')
    if any(m.type == 'ARMATURE' for m in head.modifiers):
        raise RuntimeError('Head already has skinning; use its explicit conversion history')
    source_sha = sha(source)
    uv = api('production_uv_r08')
    source_contract = {name: data_contract(bpy.data.objects[name]) for name in (HEAD, BRIDGE, BODY)}
    pose_before = uv['_pose_contract']()
    children = sorted((o for o in bpy.data.objects if descendant(o, head)), key=lambda o: o.name)
    attachment_before = {o.name: attachment(o) for o in [head] + children}
    arrays = {}
    report = {'source': source_relative, 'sourceSha256': source_sha, 'helperSha256': sha(__file__),
              'frame': [scene.frame_current, scene.frame_subframe], 'sourceContracts': source_contract,
              'currentAttachments': attachment_before, 'directChildren': [o.name for o in children if o.parent == head],
              'headReferences': references(head), 'rig': {'matrixWorld': matrix(rig.matrix_world),
                  'posePosition': rig.data.pose_position, 'animation': uv['_animation'](rig),
                  'bones': {b.name: {'parent': b.parent.name if b.parent else None,
                      'head': list(b.head_local), 'tail': list(b.tail_local), 'matrixLocal': matrix(b.matrix_local),
                      'useDeform': b.use_deform, 'inheritScale': b.inherit_scale,
                      'poseMatrix': matrix(rig.pose.bones[b.name].matrix),
                      'matrixBasis': matrix(rig.pose.bones[b.name].matrix_basis)} for b in rig.data.bones}},
              'savedBlend': False, 'weightsApplied': False,
              'scope': 'Read-only current geometry, skeleton and dependency evidence for a reversible head/neck skin conversion. No binding, pose quality or client acceptance.'}
    with api('face_contact_audit_r11')['rest_evaluation']():
        report['restAttachments'] = {o.name: attachment(o) for o in [head] + children}
        mix, mix_metadata = uv['current_mix'](head)
        arrays['head_cage_current_local'] = mix
        arrays['head_cage_current_world'] = world(mix, head.matrix_world)
        arrays['head_cage_edges'] = array(head.data.edges, 'vertices', 2, np.int32)
        arrays['head_cage_weights'] = weights(head)
        arrays['bridge_cage_weights'] = weights(bridge)
        report['currentMixMetadata'] = mix_metadata
        report['headMaskModifiers'] = [{'name': m.name, 'group': m.vertex_group,
            'threshold': m.threshold, 'invert': m.invert_vertex_group} for m in head.modifiers if m.type == 'MASK']
        head_row, head_data, head_weights = evaluated(head, 'head_evaluated_rest', arrays, True)
        bridge_row, bridge_data, bridge_weights = evaluated(bridge, 'bridge_evaluated_rest', arrays, True)
        body_row, body_data, body_weights = evaluated(bpy.data.objects[BODY], 'body_evaluated_rest', arrays)
        core = api('adult_neck_bridge_r11')
        top = core['boundary_ring'](head_data)
        bottom = core['boundary_ring'](body_data, neck_only=True)
        nt, nb = int(bridge['R11_boundary_top_count']), int(bridge['R11_boundary_bottom_count'])
        if len(top['ids']) != nt or len(bottom['ids']) != nb:
            raise RuntimeError('Actual boundary count differs from bridge metadata')
        ti = np.arange(nt, dtype=np.int32)
        bi = np.arange(len(bridge_data['world']) - nb, len(bridge_data['world']), dtype=np.int32)
        matcher = api('adult_motion_review_r11')['_match']
        hi = top['ids'][matcher(bridge_data['world'][ti], top['points'])]
        oi = bottom['ids'][matcher(bridge_data['world'][bi], bottom['points'])]
        for key, value in {'bridge_top_ids': ti, 'head_top_ids': hi, 'bridge_bottom_ids': bi,
                           'body_bottom_ids': oi, 'body_bottom_world': body_data['world'][oi],
                           'body_bottom_weights': body_weights[oi], 'head_top_world': head_data['world'][hi],
                           'head_top_weights': head_weights[hi]}.items():
            arrays[key] = value
        report['restEvaluated'] = {HEAD: head_row, BRIDGE: bridge_row, BODY: body_row}
        report['seamCounts'] = {'top': nt, 'bottom': nb, 'bridgeTotal': len(bridge_data['world'])}
        # Show actual anatomical extents around the existing neck/head joint.
        neck = rig.data.bones['neck']
        head_bone = rig.data.bones['head']
        report['cervicalRigWorld'] = {'neckHead': list(rig.matrix_world @ neck.head_local),
            'neckTail': list(rig.matrix_world @ neck.tail_local),
            'headHead': list(rig.matrix_world @ head_bone.head_local),
            'headTail': list(rig.matrix_world @ head_bone.tail_local),
            'neckLengthMeters': float((rig.matrix_world.to_3x3() @ (neck.tail_local - neck.head_local)).length)}
    if ({name: data_contract(bpy.data.objects[name]) for name in source_contract} != source_contract or
        {o.name: attachment(o) for o in [head] + children} != attachment_before or
        uv['_pose_contract']() != pose_before or sha(source) != source_sha):
        raise RuntimeError('Read-only observation changed raw geometry, weights, attachments, keys, source or rig pose')
    np.savez_compressed(npz, **arrays)
    report.update(arrays=npz.relative_to(project_path()).as_posix(), arraysSha256=sha(npz),
                  sourceContractsAndPoseUnchanged=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_HEAD_BIND_OBSERVED ' + json.dumps({'output': output_relative,
          'sourceSha256': source_sha, 'arraysSha256': report['arraysSha256'],
          'directChildren': report['directChildren'], 'seamCounts': report['seamCounts']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.output)
