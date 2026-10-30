"""Read-only hand observations; no Blender launch, render, save or source edits.

First inventory the installed bundle, then explicitly choose its exact object
names for observe. All appended datablocks are removed in finally. Numeric
arrays preserve raw and evaluated geometry as separate coordinate spaces.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

BUNDLE = 'assets/human-base-meshes-v1.4.1/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend'
HANDS = tuple('Hand '+side+' · continuous anatomy' for side in ('L', 'R'))
NAILS = tuple('Nail '+side+' '+finger+' · polished' for side in ('L', 'R')
              for finger in ('Thumb', 'Index', 'Middle', 'Ring', 'Little'))


def _path(relative):
    path = (project_path()/relative).resolve()
    path.relative_to(project_path().resolve())
    return path


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def _write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)


def _ids(bpy):
    result = {}
    for prop in bpy.data.bl_rna.properties:
        if prop.type != 'COLLECTION': continue
        collection = getattr(bpy.data, prop.identifier)
        for item in collection:
            if isinstance(item, bpy.types.ID): result[item.as_pointer()] = item
    return result


def inventory(report_relative='work/reconstruction-r02/hand-asset-inventory-r10.json',
              bundle_relative=BUNDLE):
    import bpy
    if not bpy.app.background: raise RuntimeError('Run the observation in a disposable background process')
    path = _path(bundle_relative); output = _path(report_relative)
    if output.exists(): raise FileExistsError(output)
    before = _ids(bpy)
    # Deliberately load no object. These are the library's actual stored names.
    with bpy.data.libraries.load(str(path), link=False) as (available, requested):
        names = {kind: sorted(list(getattr(available, kind)))
                 for kind in ('objects', 'collections', 'meshes', 'armatures', 'materials')}
    if set(_ids(bpy)) != set(before): raise RuntimeError('Inventory unexpectedly changed datablock identity')
    result = {'schemaVersion': 1, 'mode': 'inventory', 'bundle': bundle_relative,
              'bundleSha256': _sha(path), 'libraryNames': names,
              'handNameCandidatesOnly': [name for name in names['objects'] if 'hand' in name.lower()],
              'objectsLoaded': 0, 'savedBlend': False, 'sourceGeometryEdited': False,
              'selectionPolicy': 'Choose exact object names from libraryNames; candidates are not automatically selected.'}
    _write(output, result)
    return {'report': str(output.relative_to(project_path())),
            'handNameCandidatesOnly': result['handNameCandidatesOnly'], 'bundleSha256': result['bundleSha256']}


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if hasattr(value, 'name_full') and hasattr(value, 'as_pointer'):
        return {'idType': value.bl_rna.identifier, 'name': value.name_full,
                'library': value.library.filepath if value.library else None}
    if hasattr(value, 'to_list'): return value.to_list()
    if hasattr(value, 'items'):
        try: return {str(k): _plain(v) for k, v in value.items()}
        except TypeError:
            # Some non-ID RNA settings expose items() but do not support ID
            # properties. Preserve their actual readable RNA fields instead.
            if hasattr(value, 'bl_rna'):
                return {'rnaType': value.bl_rna.identifier, 'properties': _rna(value)}
            raise
    try: return [_plain(v) for v in value]
    except TypeError: return repr(value)


def _props(value):
    try: keys = list(value.keys())
    except TypeError: return {}
    return {str(key): _plain(value[key]) for key in keys}


def _rna(value):
    result = {}
    for prop in value.bl_rna.properties:
        if prop.identifier in {'rna_type', 'execution_time', 'is_updated', 'is_updated_data'} or prop.type == 'COLLECTION': continue
        try: result[prop.identifier] = _plain(getattr(value, prop.identifier))
        except (AttributeError, RuntimeError): continue
    return result


def _animation(owner):
    anim = owner.animation_data
    if not anim: return None
    return {'action': _plain(anim.action),
            'slot': getattr(getattr(anim, 'action_slot', None), 'identifier', None),
            'nla': [{'name': t.name, 'mute': t.mute,
                     'strips': [{'name': s.name, 'action': _plain(s.action),
                                 'frameStart': s.frame_start, 'frameEnd': s.frame_end}
                                for s in t.strips]} for t in anim.nla_tracks],
            'drivers': [{'path': f.data_path, 'index': f.array_index,
                         'valid': f.is_valid, 'type': f.driver.type,
                         'expression': f.driver.expression,
                         'variables': [{'name': v.name, 'type': v.type,
                                        'targets': [_rna(t) for t in v.targets]}
                                       for v in f.driver.variables]} for f in anim.drivers]}


def _material(mat):
    if not mat: return None
    import bpy
    result = {'name': mat.name, 'properties': _rna(mat), 'custom': _props(mat),
              'nodes': [], 'links': [], 'images': []}
    if not mat.use_nodes: return result
    for node in mat.node_tree.nodes:
        row = {'name': node.name, 'type': node.bl_idname, 'properties': _rna(node), 'inputs': []}
        for socket in node.inputs:
            row['inputs'].append({'name': socket.name, 'identifier': socket.identifier,
                                  'linked': socket.is_linked,
                                  'value': _plain(getattr(socket, 'default_value', None))})
        result['nodes'].append(row)
        image = getattr(node, 'image', None)
        if image:
            resolved = Path(bpy.path.abspath(image.filepath, library=image.library)) if image.filepath else None
            result['images'].append({'name': image.name, 'filepath': image.filepath,
                'resolvedPath': str(resolved) if resolved else None, 'size': list(image.size),
                'packed': bool(image.packed_file or image.packed_files),
                'fileSha256': _sha(resolved) if resolved and resolved.is_file() else None,
                'colorspace': image.colorspace_settings.name})
    result['links'] = [[link.from_node.name, link.from_socket.identifier,
                        link.to_node.name, link.to_socket.identifier] for link in mat.node_tree.links]
    return result


def _array(arrays, key, value):
    import numpy as np
    value = np.ascontiguousarray(value)
    arrays[key] = value
    return {'array': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def _foreach(collection, field, count, width, dtype):
    import numpy as np
    result = np.empty((count, width), dtype=dtype) if width > 1 else np.empty(count, dtype=dtype)
    collection.foreach_get(field, result.ravel())
    return result


def _mesh(mesh, prefix, arrays):
    import numpy as np
    count = len(mesh.vertices)
    if count > 1000000: raise RuntimeError('Read-only mesh observation exceeds bounded one-million-point limit')
    mesh.calc_loop_triangles()
    data = {
        'positions': _foreach(mesh.vertices, 'co', count, 3, np.float32),
        'normals': _foreach(mesh.vertices, 'normal', count, 3, np.float32),
        'edges': _foreach(mesh.edges, 'vertices', len(mesh.edges), 2, np.int32),
        'loopVertex': _foreach(mesh.loops, 'vertex_index', len(mesh.loops), 1, np.int32),
        'polygonStart': _foreach(mesh.polygons, 'loop_start', len(mesh.polygons), 1, np.int32),
        'polygonSize': _foreach(mesh.polygons, 'loop_total', len(mesh.polygons), 1, np.int32),
        'materialIndex': _foreach(mesh.polygons, 'material_index', len(mesh.polygons), 1, np.int32),
        'smoothFace': _foreach(mesh.polygons, 'use_smooth', len(mesh.polygons), 1, np.bool_),
        'faceNormals': _foreach(mesh.polygons, 'normal', len(mesh.polygons), 3, np.float32),
        'triangles': _foreach(mesh.loop_triangles, 'vertices', len(mesh.loop_triangles), 3, np.int32),
        'trianglePolygon': _foreach(mesh.loop_triangles, 'polygon_index', len(mesh.loop_triangles), 1, np.int32),
        'cornerNormals': _foreach(mesh.corner_normals, 'vector', len(mesh.corner_normals), 3, np.float32),
    }
    edge_use = {}
    for face in mesh.polygons:
        indices = list(face.vertices)
        for a, b in zip(indices, indices[1:]+indices[:1]):
            edge = (min(a, b), max(a, b)); edge_use[edge] = edge_use.get(edge, 0)+1
    boundary = np.asarray([e for e, n in edge_use.items() if n == 1], np.int32).reshape(-1, 2)
    data['boundaryEdges'] = boundary
    result = {'vertexCount': count, 'polygonCount': len(mesh.polygons),
              'triangleCount': len(mesh.loop_triangles), 'boundaryEdgeCount': len(boundary),
              'nonManifoldFaceEdgeCount': sum(n > 2 for n in edge_use.values()),
              'arrays': {k: _array(arrays, prefix+'_'+k, v) for k, v in data.items()},
              'uvLayers': [], 'attributes': [], 'custom': _props(mesh)}
    for i, layer in enumerate(mesh.uv_layers):
        uv = _foreach(layer.data, 'uv', len(layer.data), 2, np.float32)
        result['uvLayers'].append({'name': layer.name, 'active': layer == mesh.uv_layers.active,
            'render': layer.active_render, 'data': _array(arrays, prefix+'_uv_'+str(i), uv)})
    attr_types = {'FLOAT': ('value', 1, np.float32), 'INT': ('value', 1, np.int32),
        'INT8': ('value', 1, np.int8), 'BOOLEAN': ('value', 1, np.bool_),
        'FLOAT_VECTOR': ('vector', 3, np.float32), 'FLOAT2': ('vector', 2, np.float32),
        'FLOAT_COLOR': ('color', 4, np.float32), 'BYTE_COLOR': ('color', 4, np.float32),
        'QUATERNION': ('value', 4, np.float32), 'INT32_2D': ('value', 2, np.int32)}
    for i, attr in enumerate(mesh.attributes):
        row = {'name': attr.name, 'domain': attr.domain, 'dataType': attr.data_type, 'count': len(attr.data)}
        if attr.data_type in attr_types:
            field, width, dtype = attr_types[attr.data_type]
            values = _foreach(attr.data, field, len(attr.data), width, dtype)
            row['data'] = _array(arrays, prefix+'_attr_'+str(i), values)
            if width == 1 and len(values) and attr.domain == 'FACE':
                unique, counts = np.unique(values, return_counts=True)
                if len(unique) <= 256: row['faceGroups'] = [[_plain(v.item()), int(n)] for v, n in zip(unique, counts)]
        else: row['notExported'] = 'Unsupported non-numeric attribute; retained in source and named here'
        result['attributes'].append(row)
    return result


def _object(obj, prefix, arrays, depsgraph):
    import numpy as np
    result = {'name': obj.name, 'type': obj.type, 'dataName': obj.data.name if obj.data else None,
              'matrixWorld': _plain(obj.matrix_world), 'matrixLocal': _plain(obj.matrix_local),
              'matrixBasis': _plain(obj.matrix_basis), 'matrixParentInverse': _plain(obj.matrix_parent_inverse),
              'parent': _plain(obj.parent), 'parentType': obj.parent_type, 'parentBone': obj.parent_bone,
              'children': [child.name for child in obj.children],
              'instanceType': obj.instance_type, 'instanceCollection': _plain(obj.instance_collection),
              'custom': _props(obj), 'modifiers': [_rna(m) for m in obj.modifiers],
              'constraints': [_rna(c) for c in obj.constraints], 'animation': _animation(obj),
              'hideRender': obj.hide_render, 'hideViewport': obj.hide_viewport,
              'materialSlots': [{'link': s.link, 'material': _material(s.material)} for s in obj.material_slots]}
    if obj.asset_data: result['assetMetadata'] = _rna(obj.asset_data)
    if obj.type == 'MESH':
        result['raw'] = _mesh(obj.data, prefix+'_raw', arrays)
        result['vertexGroups'] = [{'name': g.name, 'index': g.index, 'lockWeight': g.lock_weight}
                                 for g in obj.vertex_groups]
        weights = [(v.index, g.group, g.weight) for v in obj.data.vertices for g in v.groups]
        result['weights'] = {
            'vertex': _array(arrays, prefix+'_weight_vertex', np.asarray([r[0] for r in weights], np.int32)),
            'group': _array(arrays, prefix+'_weight_group', np.asarray([r[1] for r in weights], np.int32)),
            'value': _array(arrays, prefix+'_weight_value', np.asarray([r[2] for r in weights], np.float32))}
        keys = obj.data.shape_keys
        result['shapeKeys'] = []
        if keys:
            result['shapeKeyAnimation'] = _animation(keys)
            result['shapeKeyMode'] = {'useRelative': keys.use_relative, 'evalTime': keys.eval_time}
            for i, key in enumerate(keys.key_blocks):
                values = _foreach(key.data, 'co', len(key.data), 3, np.float32)
                result['shapeKeys'].append({'name': key.name, 'relative': key.relative_key.name,
                    'value': key.value, 'mute': key.mute, 'vertexGroup': key.vertex_group,
                    'interpolation': key.interpolation, 'sliderMin': key.slider_min, 'sliderMax': key.slider_max,
                    'data': _array(arrays, prefix+'_key_'+str(i), values)})
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            result['evaluated'] = _mesh(mesh, prefix+'_evaluated', arrays)
            result['evaluatedMatrixWorld'] = _plain(evaluated.matrix_world)
            result['evaluatedCoordinateMeaning'] = 'Object-local after active shape mix and evaluated modifiers; use evaluatedMatrixWorld for world coordinates. Not rest mesh.'
        finally: evaluated.to_mesh_clear()
    if obj.type == 'ARMATURE':
        result['posePosition'] = obj.data.pose_position
        result['bones'] = [{'name': b.name, 'parent': b.parent.name if b.parent else None,
            'headLocal': _plain(b.head_local), 'tailLocal': _plain(b.tail_local),
            'restMatrixLocal': _plain(b.matrix_local), 'useDeform': b.use_deform,
            'inheritScale': b.inherit_scale, 'useConnect': b.use_connect,
            'collections': [c.name for c in b.collections], 'custom': _props(b),
            'poseMatrix': _plain(obj.pose.bones[b.name].matrix),
            'poseMatrixBasis': _plain(obj.pose.bones[b.name].matrix_basis),
            'constraints': [_rna(c) for c in obj.pose.bones[b.name].constraints]}
            for b in obj.data.bones]
    return result


def _source_contract(objects):
    """Local source data only; no evaluated caches or external dependencies mutated."""
    import numpy as np
    h = hashlib.sha256()
    for obj in objects:
        row = {'name': obj.name, 'type': obj.type, 'matrixBasis': _plain(obj.matrix_basis),
               'matrixParentInverse': _plain(obj.matrix_parent_inverse), 'parent': _plain(obj.parent),
               'parentBone': obj.parent_bone, 'modifiers': [_rna(m) for m in obj.modifiers],
               'animation': _animation(obj), 'custom': _props(obj)}
        if obj.type == 'MESH':
            arrays = {}; row['mesh'] = _mesh(obj.data, 'guard', arrays)
            row['materials'] = [_material(slot.material) for slot in obj.material_slots]
            row['groups'] = [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups]
            row['weights'] = [(v.index, g.group, g.weight) for v in obj.data.vertices for g in v.groups]
            if obj.data.shape_keys:
                row['shapeAnimation'] = _animation(obj.data.shape_keys)
                for key in obj.data.shape_keys.key_blocks:
                    h.update(key.name.encode('utf-8'))
                    h.update(_foreach(key.data, 'co', len(key.data), 3, np.float32).tobytes())
                    h.update(json.dumps([key.value, key.relative_key.name, key.mute]).encode())
        elif obj.type == 'ARMATURE':
            row['bones'] = [{'name': b.name, 'parent': b.parent.name if b.parent else None,
                'matrixLocal': _plain(b.matrix_local), 'poseBasis': _plain(obj.pose.bones[b.name].matrix_basis),
                'constraints': [_rna(c) for c in obj.pose.bones[b.name].constraints]}
                for b in obj.data.bones]
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    return h.hexdigest()


def observe(source_relative, source_sha256, asset_names,
            inventory_relative='work/reconstruction-r02/hand-asset-inventory-r10.json',
            report_relative='work/reconstruction-r02/hand-observation-r10.json', frame=1):
    import bpy
    import numpy as np
    source = _path(source_relative); output = _path(report_relative); npz = output.with_suffix('.npz')
    if output.exists() or npz.exists(): raise FileExistsError('Keep the previous observation; choose a new report path')
    if not bpy.app.background or bpy.context.mode != 'OBJECT': raise RuntimeError('Disposable background Object Mode required')
    if Path(bpy.data.filepath).resolve() != source or _sha(source) != source_sha256:
        raise RuntimeError('Open the explicitly identified source and supply its actual SHA-256')
    catalog_path = _path(inventory_relative)
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    bundle = _path(catalog['bundle'])
    if catalog['mode'] != 'inventory' or _sha(bundle) != catalog['bundleSha256']:
        raise RuntimeError('Library inventory no longer identifies this exact bundle')
    requested_names = tuple(asset_names)
    if not requested_names or len(requested_names) > 8 or len(set(requested_names)) != len(requested_names):
        raise ValueError('Choose between one and eight unique exact asset object names from inventory')
    if any(name not in catalog['libraryNames']['objects'] for name in requested_names):
        raise ValueError('Asset name absent from actual inventory; do not infer a guessed name')
    required = HANDS+NAILS+('Character_Rig',)
    missing = [name for name in required if name not in bpy.data.objects]
    if missing: raise RuntimeError('Required current hand/nail/rig not present: '+repr(missing))
    source_objects = [bpy.data.objects[name] for name in required]
    originals = _ids(bpy); arrays = {}; scene = bpy.context.scene
    old_frame = (scene.frame_current, scene.frame_subframe)
    original_collection = set(c.as_pointer() for c in scene.collection.children)
    result = {'schemaVersion': 1, 'mode': 'observe', 'source': source_relative,
        'sourceSha256': source_sha256, 'bundle': catalog['bundle'], 'bundleSha256': catalog['bundleSha256'],
        'inventory': inventory_relative, 'inventorySha256': _sha(catalog_path),
        'frame': frame, 'current': [], 'asset': [], 'requestedAssetNames': list(requested_names),
        'savedBlend': False, 'sourceGeometryEdited': False, 'assetAdaptationOrAcceptanceClaimed': False,
        'scope': 'Raw/evaluated hands+nails, all existing rig rest and actual pose matrices, material nodes, UV, weights, shape keys and numeric attributes. Assets appended temporarily without pose fitting, subdivision application, remeshing or transfer.'}
    cleanup = {}; guard = None
    try:
        scene.frame_set(int(frame), subframe=float(frame)-int(frame)); bpy.context.view_layer.update()
        guard = _source_contract(source_objects)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for i, obj in enumerate(source_objects):
            result['current'].append(_object(obj, 'current_'+str(i), arrays, depsgraph))
        with bpy.data.libraries.load(str(bundle), link=False) as (available, requested):
            if any(name not in available.objects for name in requested_names): raise RuntimeError('Inventory and load disagree')
            # Blender replaces this list's strings with IDs; never alias immutable requested_names.
            requested.objects = list(requested_names)
        mapping = dict(zip(requested_names, requested.objects))
        if any(obj is None for obj in mapping.values()): raise RuntimeError('Requested asset could not be appended')
        temporary = bpy.data.collections.new('R10 temporary hand observation - no save')
        scene.collection.children.link(temporary)
        new_ids = {pointer: item for pointer, item in _ids(bpy).items() if pointer not in originals}
        asset_objects = sorted([item for item in new_ids.values() if isinstance(item, bpy.types.Object)], key=lambda x:x.name)
        if not any(obj.type == 'MESH' for obj in asset_objects):
            raise RuntimeError('Selected names and their actual appended dependencies contain no mesh: '+
                               repr([(name, obj.type) for name, obj in mapping.items()]))
        for obj in asset_objects:
            temporary.objects.link(obj)
            if obj.type == 'MESH' and len(obj.data.vertices) > 250000:
                raise RuntimeError('Selected asset unexpectedly exceeds 250,000 raw vertices; inspect inventory selection')
        bpy.context.view_layer.update(); depsgraph = bpy.context.evaluated_depsgraph_get()
        for i, obj in enumerate(asset_objects):
            row = _object(obj, 'asset_'+str(i), arrays, depsgraph)
            row['requestedLibraryNames'] = [name for name, selected in mapping.items() if selected == obj]
            row['automaticallyAppendedDependency'] = not row['requestedLibraryNames']
            result['asset'].append(row)
        result['temporaryDatablocks'] = [{'type': item.bl_rna.identifier, 'name': item.name_full} for item in new_ids.values()]
    finally:
        created = [item for pointer, item in _ids(bpy).items() if pointer not in originals]
        cleanup['newDatablocksBeforeRemoval'] = len(created)
        if created: bpy.data.batch_remove(ids=created)
        bpy.context.view_layer.update()
        after = _ids(bpy)
        cleanup['newDatablocksRemaining'] = len(set(after)-set(originals))
        cleanup['originalDatablocksMissing'] = len(set(originals)-set(after))
        cleanup['sceneCollectionsRestored'] = set(c.as_pointer() for c in scene.collection.children) == original_collection
        if guard is not None:
            cleanup['sourceDataContractBefore'] = guard
            cleanup['sourceDataContractAfter'] = _source_contract(source_objects)
            cleanup['sourceDataContractUnchanged'] = cleanup['sourceDataContractAfter'] == guard
        scene.frame_set(old_frame[0], subframe=old_frame[1]); bpy.context.view_layer.update()
        cleanup['frameRestored'] = [scene.frame_current, scene.frame_subframe] == list(old_frame)
        cleanup['sourceUnchangedOnDisk'] = _sha(source) == source_sha256
        result['cleanup'] = cleanup
        if (cleanup['newDatablocksRemaining'] or cleanup['originalDatablocksMissing']
                or not cleanup['sceneCollectionsRestored'] or not cleanup['sourceUnchangedOnDisk']
                or cleanup.get('sourceDataContractUnchanged') is False):
            raise RuntimeError('Read-only cleanup verification failed: '+json.dumps(cleanup))
    output.parent.mkdir(parents=True, exist_ok=True)
    with npz.open('xb') as stream: np.savez_compressed(stream, **arrays)
    result['arrays'] = str(npz.relative_to(project_path())); result['arraysSha256'] = _sha(npz)
    result['arrayCount'] = len(arrays)
    _write(output, result)
    return {'report': str(output.relative_to(project_path())), 'arrays': result['arrays'],
            'arraysSha256': result['arraysSha256'], 'currentObjects': len(result['current']),
            'assetObjects': len(result['asset']), 'cleanup': cleanup}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--report', default='work/reconstruction-r02/hand-asset-inventory-r10.json')
    inv.add_argument('--bundle', default=BUNDLE)
    obs = sub.add_parser('observe')
    obs.add_argument('--source', required=True)
    obs.add_argument('--source-sha', required=True)
    obs.add_argument('--asset-object', required=True, action='append')
    obs.add_argument('--inventory', default='work/reconstruction-r02/hand-asset-inventory-r10.json')
    obs.add_argument('--report', default='work/reconstruction-r02/hand-observation-r10.json')
    obs.add_argument('--frame', type=float, default=1)
    args = parser.parse_args(argv)
    if args.mode == 'inventory': result = inventory(args.report, args.bundle)
    else: result = observe(args.source, args.source_sha, args.asset_object, args.inventory, args.report, args.frame)
    print('R10_HAND_OBSERVATION '+json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
