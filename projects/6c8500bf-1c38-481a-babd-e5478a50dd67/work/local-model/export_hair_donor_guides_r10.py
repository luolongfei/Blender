"""Export stored straight-hair guides and node metadata from the exact donor.

Integrator-only entry point: open the donor in a fresh background Blender with
--disable-autoexec, then run this script. Does not evaluate Geometry Nodes,
change visibility/selection/geometry, save a .blend, import into the character,
or render. Writes only a new NPZ and JSON observation in work/reconstruction-r02.
"""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

DONOR = 'assets/blender-hair-styles-bystedt-2023/hair_nodes-female_hair_styles.blend'
DONOR_SHA = '1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48'
GUIDES = ('long hair main', 'long hair strands')
GROWTH = 'long hair growth mesh'
LAYOUT = 'Long hair head'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def values(collection, prop, width=1, dtype=np.float64):
    result = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(prop, result)
    return result.reshape(-1, width) if width > 1 else result


def json_value(value, depth=0):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bpy.types.ID):
        return {'idType': value.bl_rna.identifier, 'name': value.name,
                'library': value.library.filepath if value.library else None}
    if depth > 5:
        return {'unsupportedType': type(value).__name__}
    if hasattr(value, 'keys'):
        try:
            keys = list(value.keys())
        except TypeError:
            if not hasattr(value, 'bl_rna'):
                raise
            return {'rnaType': value.bl_rna.identifier,
                    'rnaValues': {prop.identifier: json_value(getattr(value, prop.identifier), depth + 1)
                                  for prop in value.bl_rna.properties
                                  if prop.identifier != 'rna_type' and prop.type != 'COLLECTION'}}
        return {str(key): json_value(value[key], depth + 1) for key in keys}
    try:
        return [json_value(v, depth + 1) for v in value]
    except TypeError:
        return {'unsupportedType': type(value).__name__}


def world(points, matrix):
    m = np.asarray(matrix, dtype=np.float64)
    return points @ m[:3, :3].T + m[:3, 3]


def object_metadata(obj):
    return {'name': obj.name, 'type': obj.type,
            'parent': obj.parent.name if obj.parent else None,
            'matrixWorld': json_value(obj.matrix_world),
            'matrixLocal': json_value(obj.matrix_local),
            'matrixParentInverse': json_value(obj.matrix_parent_inverse),
            'collections': [c.name for c in obj.users_collection],
            'inCurrentScene': obj.name in bpy.context.scene.objects,
            'users': obj.users, 'hideRender': obj.hide_render,
            'hideViewport': obj.hide_viewport,
            'instanceType': obj.instance_type,
            'instanceCollection': obj.instance_collection.name if obj.instance_collection else None,
            'instanceCollectionOffset': json_value(obj.instance_collection.instance_offset)
                if obj.instance_collection else None,
            'materials': [s.material.name if s.material else None for s in obj.material_slots]}


def connected_components(vertex_count, edges):
    adjacency = [[] for _ in range(vertex_count)]
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    labels = np.full(vertex_count, -1, dtype=np.int32)
    sizes = []
    for seed in range(vertex_count):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = len(sizes)
        count = 0
        while stack:
            a = stack.pop()
            count += 1
            for b in adjacency[a]:
                if labels[b] < 0:
                    labels[b] = len(sizes)
                    stack.append(b)
        sizes.append(count)
    return labels, sizes


def node_metadata(group):
    interface = []
    if hasattr(group, 'interface'):
        for socket in group.interface.items_tree:
            item = {'name': socket.name, 'itemType': socket.item_type}
            for prop in ('identifier', 'in_out', 'socket_type', 'default_value'):
                if hasattr(socket, prop):
                    item[prop] = json_value(getattr(socket, prop))
            interface.append(item)
    nodes = []
    for node in group.nodes:
        item = {'name': node.name, 'type': node.bl_idname, 'label': node.label,
                'mute': node.mute, 'parentFrame': node.parent.name if node.parent else None,
                'inputs': [], 'nodeGroup': getattr(getattr(node, 'node_tree', None), 'name', None)}
        for prop in ('operation', 'data_type', 'domain', 'mode', 'distribution',
                     'attribute_name', 'interpolation', 'transform_space', 'image'):
            if hasattr(node, prop):
                item[prop] = json_value(getattr(node, prop))
        for socket in node.inputs:
            socket_item = {'name': socket.name, 'identifier': socket.identifier,
                           'type': socket.bl_idname, 'isLinked': socket.is_linked}
            if hasattr(socket, 'default_value'):
                socket_item['default'] = json_value(socket.default_value)
            item['inputs'].append(socket_item)
        nodes.append(item)
    return {'name': group.name, 'type': group.bl_idname, 'interface': interface,
            'nodes': nodes,
            'links': [{'fromNode': link.from_node.name, 'fromSocket': link.from_socket.identifier,
                       'toNode': link.to_node.name, 'toSocket': link.to_socket.identifier}
                      for link in group.links]}


def export(output_relative='work/reconstruction-r02/hair-donor-guides-r10'):
    project = project_path()
    source = (project / DONOR).resolve()
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Only the exact donor in a fresh background process is allowed')
    if '--disable-autoexec' not in sys.argv:
        raise RuntimeError('Start Blender with --disable-autoexec before opening the donor')
    if sha(source) != DONOR_SHA:
        raise RuntimeError('Donor SHA does not match the recorded unmodified official file')
    output = (project / output_relative).resolve()
    output.relative_to((project / 'work/reconstruction-r02').resolve())
    report_path, arrays_path = output.with_suffix('.json'), output.with_suffix('.npz')
    if output.suffix or report_path.exists() or arrays_path.exists():
        raise FileExistsError('Use a new output stem; preserve previous observations')
    initial_dirty = bpy.data.is_dirty
    arrays, objects, modifiers = {}, {}, {}
    layout = bpy.data.objects[LAYOUT]
    layout_inverse = layout.matrix_world.inverted()
    graph_groups = []
    for index, name in enumerate(GUIDES):
        obj = bpy.data.objects[name]
        if obj.type != 'CURVES' or obj.data.surface != bpy.data.objects[GROWTH]:
            raise RuntimeError('Unexpected guide type or root surface: ' + name)
        prefix = 'guide_' + str(index)
        raw = values(obj.data.position_data, 'vector', 3)
        starts = np.asarray([c.first_point_index for c in obj.data.curves], dtype=np.int32)
        lengths = np.asarray([c.points_length for c in obj.data.curves], dtype=np.int32)
        if int(lengths.sum()) != len(raw) or np.any(lengths < 2):
            raise RuntimeError('Unexpected native curve topology: ' + name)
        canonical_matrix = layout_inverse @ obj.matrix_world
        arrays[prefix + '_local'] = raw
        arrays[prefix + '_world'] = world(raw, obj.matrix_world)
        arrays[prefix + '_layout_local'] = world(raw, canonical_matrix)
        arrays[prefix + '_starts'] = starts
        arrays[prefix + '_lengths'] = lengths
        arrays[prefix + '_matrix'] = np.asarray(obj.matrix_world)
        arrays[prefix + '_layout_matrix'] = np.asarray(canonical_matrix)
        attributes = []
        formats = {'FLOAT': ('value', 1, np.float64), 'INT': ('value', 1, np.int32),
                   'BOOLEAN': ('value', 1, np.int32), 'FLOAT2': ('vector', 2, np.float64),
                   'FLOAT_VECTOR': ('vector', 3, np.float64),
                   'FLOAT_COLOR': ('color', 4, np.float64),
                   'BYTE_COLOR': ('color', 4, np.float64)}
        for attr_index, attr in enumerate(obj.data.attributes):
            row = {'name': attr.name, 'domain': attr.domain, 'type': attr.data_type}
            if attr.name != 'position' and attr.data_type in formats:
                prop, width, dtype = formats[attr.data_type]
                key = prefix + '_attribute_' + str(attr_index)
                arrays[key] = values(attr.data, prop, width, dtype)
                row['array'] = key
            attributes.append(row)
        row = object_metadata(obj)
        row.update(prefix=prefix, curves=len(starts), points=len(raw),
                   surface=obj.data.surface.name, surfaceUV=obj.data.surface_uv_map,
                   attributes=attributes, evaluated=False)
        objects[name] = row
        modifiers[name] = []
        for mod in obj.modifiers:
            group = getattr(mod, 'node_group', None)
            modifiers[name].append({'name': mod.name, 'type': mod.type,
                                    'nodeGroup': group.name if group else None,
                                    'showViewport': mod.show_viewport, 'showRender': mod.show_render,
                                    'properties': json_value(mod)})
            if group:
                graph_groups.append(group)
    for name, prefix in ((GROWTH, 'growth'), ('head', 'reference_head'), ('eyes', 'reference_eyes')):
        obj = bpy.data.objects[name]
        mesh = obj.data
        raw = values(mesh.vertices, 'co', 3)
        arrays[prefix + '_local'] = raw
        arrays[prefix + '_matrix'] = np.asarray(obj.matrix_world)
        arrays[prefix + '_loop_vertex'] = values(mesh.loops, 'vertex_index', dtype=np.int32)
        arrays[prefix + '_polygon_start'] = values(mesh.polygons, 'loop_start', dtype=np.int32)
        arrays[prefix + '_polygon_size'] = values(mesh.polygons, 'loop_total', dtype=np.int32)
        row = object_metadata(obj)
        row.update(prefix=prefix, vertices=len(mesh.vertices), polygons=len(mesh.polygons))
        if name == GROWTH:
            arrays[prefix + '_world'] = world(raw, obj.matrix_world)
            arrays[prefix + '_layout_local'] = world(raw, layout_inverse @ obj.matrix_world)
            edges = values(mesh.edges, 'vertices', 2, np.int32)
            arrays[prefix + '_edges'] = edges
            arrays[prefix + '_component'], row['componentSizes'] = connected_components(len(raw), edges)
            row['vertexGroups'] = []
            for group in obj.vertex_groups:
                key = prefix + '_vertex_group_' + str(group.index)
                weights = np.zeros(len(raw), dtype=np.float64)
                for vertex in mesh.vertices:
                    for assignment in vertex.groups:
                        if assignment.group == group.index:
                            weights[vertex.index] = assignment.weight
                arrays[key] = weights
                row['vertexGroups'].append({'name': group.name, 'array': key})
            row['uvLayers'] = []
            for uv_index, uv_layer in enumerate(mesh.uv_layers):
                key = prefix + '_uv_' + str(uv_index)
                if hasattr(uv_layer, 'uv'):
                    arrays[key] = values(uv_layer.uv, 'vector', 2)
                else:
                    arrays[key] = values(uv_layer.data, 'uv', 2)
                row['uvLayers'].append({'name': uv_layer.name, 'array': key})
        objects[name] = row
    graph, seen = [], set()
    while graph_groups:
        group = graph_groups.pop()
        if group.name in seen:
            continue
        seen.add(group.name)
        graph.append(node_metadata(group))
        for node in group.nodes:
            nested = getattr(node, 'node_tree', None)
            if nested is not None:
                graph_groups.append(nested)
    materials = []
    for material_name in sorted({s.material.name for n in GUIDES
                                 for s in bpy.data.objects[n].material_slots if s.material}):
        mat = bpy.data.materials[material_name]
        materials.append({'name': mat.name, 'useNodes': mat.use_nodes,
                          'graph': node_metadata(mat.node_tree) if mat.node_tree else None})
    report = {
        'schemaVersion': 1, 'source': DONOR, 'sourceSha256': DONOR_SHA,
        'blenderVersion': bpy.app.version_string, 'helperSha256': sha(__file__),
        'scope': 'Stored, unevaluated guide positions; no GN interpolation/output is expanded.',
        'objects': objects, 'layout': object_metadata(layout), 'modifiers': modifiers,
        'nodeGroups': graph, 'materials': materials,
        'otherPrototype': object_metadata(bpy.data.objects['new hair'])
                          if 'new hair' in bpy.data.objects else None,
        'texts': [{'name': text.name, 'content': text.as_string()} for text in bpy.data.texts],
        'images': [{'name': image.name, 'size': list(image.size), 'filepath': image.filepath,
                    'packed': bool(image.packed_file)} for image in bpy.data.images],
        'sceneUnits': {'system': bpy.context.scene.unit_settings.system,
                       'scaleLength': bpy.context.scene.unit_settings.scale_length},
        'arrayManifest': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                          for key, value in arrays.items()},
        'coordinateNotes': [
            'guide_*_local are the exact saved curve datablock coordinates.',
            'guide_*_world include the demo layout object transforms.',
            'guide_*_layout_local remove Long hair head world transform for a shared donor frame.',
            'Reference head/eyes are raw meshes from their source collection. Inspect layout.instanceCollection and its instance offset before comparing them to guide layout coordinates.',
            'Stored surface_uv_coordinate values belong to donor UVMap; they cannot be reused as target scalp UV binding without reattachment.',
            'Guide counts are not final interpolated strand counts. Growth mesh is a root surface, not a proposed inflated hair shell.'
        ],
        'sourceDataEdited': False, 'savedBlend': False, 'rendered': False,
        'evaluatedGeometryNodes': False,
    }
    if bpy.data.is_dirty != initial_dirty or sha(source) != DONOR_SHA:
        raise RuntimeError('Read-only observation changed the donor state or file')
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['arrays'] = {'path': arrays_path.relative_to(project).as_posix(), 'sha256': sha(arrays_path)}
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('HAIR_DONOR_GUIDES_EXPORTED=' + json.dumps({
        'report': str(report_path), 'arrays': str(arrays_path),
        'mainGuides': objects[GUIDES[0]]['curves'], 'secondaryGuides': objects[GUIDES[1]]['curves'],
        'nodeGroups': len(graph), 'sourceDataEdited': False}), flush=True)
    return report


if __name__ == '__main__':
    export()
