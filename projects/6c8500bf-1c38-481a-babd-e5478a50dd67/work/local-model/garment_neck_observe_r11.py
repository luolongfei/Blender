"""Read-only evaluated neckline/collar observation for an explicit R11 source.

No simulation, save, remesh, key/material edit or automatic repair.  The current
pose and temporary REST are both exported.  Distances to open neck surfaces are
local signed-normal observations, not a proof of inside/outside or no collision.
Run serially in background Blender through the workspace's configured binary:
  --python garment_neck_observe_r11.py -- --source work/<candidate>.blend
    --output work/reconstruction-r02/garment-neck-current-r11.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SKIN = ('R11 Female head study', 'R11 Continuous neck bridge', 'R01 Head')
CLOTH = ('Blouse_PeterPan_Collar_L', 'Blouse_PeterPan_Collar_R', 'Blouse_Tucked_Bodice')
SEAMS = ('Collar_Edge_Stitch_L', 'Collar_Edge_Stitch_R')
ROI_LOW = np.array((-.18, -.15, 1.18), float)
ROI_HIGH = np.array((.18, .15, 1.39), float)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _array(data, field, width=3, dtype=np.float32):
    result = np.empty((len(data), width), dtype)
    data.foreach_get(field, result.ravel())
    return result


def _digest_array(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()


def _value(value):
    if isinstance(value, (str, int, bool, float)) or value is None:
        return value
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, bpy.types.ID):
        return {'id': value.name, 'type': value.bl_rna.identifier}
    try:
        return list(value)
    except TypeError:
        return str(value)


def _material(mat):
    if mat is None:
        return None
    nodes = []
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            nodes.append({'name': node.name, 'type': node.bl_idname,
                          'image': getattr(getattr(node, 'image', None), 'name', None),
                          'attribute': getattr(node, 'attribute_name', None),
                          'inputs': [{'name': s.name, 'linked': s.is_linked,
                                      'value': _value(s.default_value)}
                                     for s in node.inputs if hasattr(s, 'default_value')]})
    return {'name': mat.name, 'nodes': nodes,
            'links': [(l.from_node.name, l.from_socket.name, l.to_node.name, l.to_socket.name)
                      for l in mat.node_tree.links] if mat.use_nodes else []}


def _modifiers(obj):
    result = []
    for modifier in obj.modifiers:
        row = {'type': modifier.type, 'name': modifier.name}
        for prop in modifier.bl_rna.properties:
            if prop.identifier == 'rna_type' or prop.type == 'COLLECTION' or prop.is_readonly:
                continue
            row[prop.identifier] = _value(getattr(modifier, prop.identifier))
        result.append(row)
    return result


def _contract(objects):
    result = {}
    for obj in objects:
        data = obj.data
        row = {'data': data.name, 'matrix': [list(r) for r in obj.matrix_world],
               'parent': obj.parent.name if obj.parent else None,
               'parentType': obj.parent_type, 'parentBone': obj.parent_bone,
               'visibility': [obj.hide_render, obj.hide_viewport, obj.hide_get()],
               'modifiers': _modifiers(obj), 'materials': [_material(m) for m in data.materials]}
        if obj.type == 'MESH':
            row['vertices'] = _digest_array(_array(data.vertices, 'co'))
            row['indices'] = _digest_array(_array(data.loops, 'vertex_index', 1, np.int32))
            row['keys'] = {key.name: {'coordinates': _digest_array(_array(key.data, 'co')),
                                     'value': key.value, 'mute': key.mute,
                                     'relative': key.relative_key.name}
                           for key in data.shape_keys.key_blocks} if data.shape_keys else {}
        result[obj.name] = row
    return result


def _capture(obj, arrays, prefix):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        local = _array(mesh.vertices, 'co').astype(float)
        matrix = np.asarray(evaluated.matrix_world, float)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        mesh.calc_loop_triangles()
        triangles = _array(mesh.loop_triangles, 'vertices', 3, np.int32)
        # Keep every triangle with an AABB that meets this documented finite ROI,
        # including crossing faces whose vertices all lie outside the ROI.
        tri_points = world[triangles]
        keep = ((tri_points.max(1) >= ROI_LOW).all(1)
                & (tri_points.min(1) <= ROI_HIGH).all(1))
        actual_triangle_ids = np.flatnonzero(keep)
        kept_triangles = triangles[keep]
        vertex_ids = np.unique(kept_triangles)
        if not len(vertex_ids):
            raise RuntimeError('No actual evaluated neckline geometry: ' + obj.name)
        remap = np.full(len(world), -1, np.int32)
        remap[vertex_ids] = np.arange(len(vertex_ids))
        points = world[vertex_ids]
        tris = remap[kept_triangles]
        normals = _array(mesh.vertices, 'normal').astype(float)
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T
        normals = normals @ normal_matrix.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        arrays[prefix + '_world'] = points.astype(np.float32)
        arrays[prefix + '_triangles'] = tris
        arrays[prefix + '_vertex_ids'] = vertex_ids.astype(np.int32)
        arrays[prefix + '_triangle_ids'] = actual_triangle_ids.astype(np.int32)
        arrays[prefix + '_normals'] = normals[vertex_ids].astype(np.float32)
        arrays[prefix + '_triangle_materials'] = np.array(
            [mesh.polygons[mesh.loop_triangles[int(i)].polygon_index].material_index
             for i in actual_triangle_ids], np.int32)
        return {'object': obj.name, 'prefix': prefix, 'evaluatedVerticesTotal': len(world),
                'roiVertices': len(points), 'roiTriangles': len(tris),
                'worldBounds': [points.min(0).tolist(), points.max(0).tolist()],
                'materials': [_material(m) for m in mesh.materials]}, points, tris
    finally:
        evaluated.to_mesh_clear()


def _tree(points, triangles):
    return BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)


def _nearest(points, bvh):
    distances = np.empty(len(points), np.float32)
    signed = np.empty(len(points), np.float32)
    face_ids = np.empty(len(points), np.int32)
    for i, point in enumerate(points):
        location, normal, face_id, distance = bvh.find_nearest(Vector(point))
        if face_id is None:
            raise RuntimeError('Cannot measure actual nearest surface')
        distances[i] = distance
        signed[i] = (Vector(point) - location).dot(normal)
        face_ids[i] = face_id
    return distances, signed, face_ids


def _edge_crossings(points, triangles, other_tree):
    edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    hits = []
    for a, b in edges:
        start, finish = Vector(points[a]), Vector(points[b])
        direction = finish - start
        length = direction.length
        if length <= 2e-7:
            continue
        direction /= length
        hit, normal, face, distance = other_tree.ray_cast(start + direction * 1e-7,
                                                         direction, length - 2e-7)
        if face is not None:
            hits.append((int(a), int(b), int(face), float(distance)))
    return np.asarray(hits, np.float64).reshape((-1, 4)), len(edges)


def observe(source, output):
    if not bpy.app.background:
        raise RuntimeError('Integrator-only isolated background observation')
    if output.exists() or output.with_suffix('.npz').exists():
        raise RuntimeError('Preserve earlier evidence; provide a new output path')
    if Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('The exact explicit source must already be loaded')
    source_sha = _sha(source)
    objects = [bpy.data.objects[name] for name in SKIN + CLOTH + SEAMS]
    for obj in bpy.context.scene.objects:
        if not obj.hide_viewport and not obj.hide_get():
            if any(m.type == 'CLOTH' and m.show_viewport for m in obj.modifiers):
                raise RuntimeError('Active simulation is forbidden in this read-only observer: ' + obj.name)
    rig = bpy.data.objects['Character_Rig']
    scene = bpy.context.scene
    original_pose = rig.data.pose_position
    frame = (scene.frame_current, scene.frame_subframe)
    contract = _contract(objects)
    arrays = {}
    states = []
    try:
        for label, pose in (('current', original_pose), ('rest', 'REST')):
            rig.data.pose_position = pose
            bpy.context.view_layer.update()
            rows = []
            surfaces = {}
            for i, obj in enumerate(objects):
                row, points, triangles = _capture(obj, arrays, f'{label}_o{i}')
                rows.append(row)
                surfaces[obj.name] = (points, triangles)
            distances = []
            for ci, cloth_name in enumerate(CLOTH):
                cp, ct = surfaces[cloth_name]
                cbvh = _tree(cp, ct)
                for si, skin_name in enumerate(SKIN):
                    sp, st = surfaces[skin_name]
                    sbvh = _tree(sp, st)
                    distance, signed, face_ids = _nearest(cp, sbvh)
                    prefix = f'{label}_c{ci}_s{si}'
                    arrays[prefix + '_distance'] = distance
                    arrays[prefix + '_signed_normal_distance'] = signed
                    arrays[prefix + '_nearest_skin_triangle'] = face_ids
                    forward, forward_count = _edge_crossings(cp, ct, sbvh)
                    reverse, reverse_count = _edge_crossings(sp, st, cbvh)
                    arrays[prefix + '_cloth_edges_crossing_skin'] = forward
                    arrays[prefix + '_skin_edges_crossing_cloth'] = reverse
                    distances.append({'cloth': cloth_name, 'skin': skin_name, 'prefix': prefix,
                        'minDistanceMeters': float(distance.min()),
                        'distanceQuantilesMeters': np.quantile(distance, (0, .01, .5, .99, 1)).tolist(),
                        'signedNormalDistanceQuantilesMeters': np.quantile(signed, (0, .01, .5, .99, 1)).tolist(),
                        'belowMinusHalfMillimeter': int((signed < -.0005).sum()),
                        'aboveTenMillimeters': int((distance > .01).sum()),
                        'testedClothEdges': forward_count, 'testedSkinEdges': reverse_count,
                        'clothEdgesCrossingSkin': len(forward), 'skinEdgesCrossingCloth': len(reverse)})
            states.append({'state': label, 'posePosition': pose, 'objects': rows, 'surfaceObservations': distances})
        # Full small raw collar cages/UV/group maps allow offline diagnosis of
        # folded rows and smooth shading without recreating or deleting a collar.
        raw = []
        for i, name in enumerate(CLOTH[:2]):
            obj = bpy.data.objects[name]
            mesh = obj.data
            prefix = f'raw_collar_{i}'
            arrays[prefix + '_local'] = _array(mesh.vertices, 'co')
            arrays[prefix + '_matrix_world_rest'] = np.asarray(obj.matrix_world, np.float64)
            arrays[prefix + '_loop_vertices'] = _array(mesh.loops, 'vertex_index', 1, np.int32).ravel()
            arrays[prefix + '_face_starts'] = _array(mesh.polygons, 'loop_start', 1, np.int32).ravel()
            arrays[prefix + '_face_counts'] = _array(mesh.polygons, 'loop_total', 1, np.int32).ravel()
            arrays[prefix + '_face_smooth'] = _array(mesh.polygons, 'use_smooth', 1, np.bool_).ravel()
            raw.append({'object': name, 'prefix': prefix, 'groups': [(g.name, g.index) for g in obj.vertex_groups],
                        'weights': [[(g.group, g.weight) for g in v.groups] for v in mesh.vertices],
                        'uvLayers': [uv.name for uv in mesh.uv_layers]})
    finally:
        rig.data.pose_position = original_pose
        scene.frame_set(frame[0], subframe=frame[1])
        bpy.context.view_layer.update()
    if _contract(objects) != contract:
        raise RuntimeError('Source-scene contract changed; observation is invalid')
    if _sha(source) != source_sha:
        raise RuntimeError('Source file changed during read-only observation')
    output.parent.mkdir(parents=True, exist_ok=True)
    npz = output.with_suffix('.npz')
    np.savez_compressed(npz, **arrays)
    result = {'schemaVersion': 1, 'sourceBlend': source.relative_to(project_path()).as_posix(),
              'sourceSha256': source_sha, 'blender': bpy.app.version_string, 'frame': frame,
              'helperSha256': _sha(Path(__file__)), 'roiWorldMeters': [ROI_LOW.tolist(), ROI_HIGH.tolist()],
              'states': states, 'rawCollars': raw, 'sourceContract': contract,
              'payload': npz.relative_to(project_path()).as_posix(), 'payloadSha256': _sha(npz),
              'sourceAndScenePreserved': True, 'savedBlend': False,
              'scope': 'Finite evaluated neck/collar/bodice ROI at current saved frame and temporary REST.',
              'limitations': ['Open surfaces: signed normal distances are local observations, not a reliable global inside test.',
                  'Bidirectional edge rays detect non-coplanar crossings; tangent/coplanar overlap and full containment are not certified.',
                  'No continuous motion, dynamic cloth simulation, cuff/skirt test or visual acceptance is implied.',
                  'Original R09 cloth payload and body action are untouched; their old reports cannot prove R11 compatibility.']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('R11_GARMENT_NECK_OBSERVED ' + json.dumps({'output': str(output), 'states': len(states)}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    source = (project_path() / args.source).resolve()
    output = (project_path() / args.output).resolve()
    source.relative_to((project_path() / 'work').resolve())
    output.relative_to((project_path() / 'work').resolve())
    if source == (project_path() / 'work/character.blend').resolve():
        raise RuntimeError('Use an explicit independent R11 candidate for this finite observer')
    return observe(source, output)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
