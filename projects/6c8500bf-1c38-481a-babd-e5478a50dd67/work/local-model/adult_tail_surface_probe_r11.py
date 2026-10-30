"""Read-only, bounded actual mesh/strand contact probe for an R11 tail trial.

Samples explicit cached frames, real representative strands and their segment
midpoints against evaluated head/neck/blouse/sleeve surfaces. Segment ray hits
detect crossings between samples. Nearest-normal signs are diagnostics, not a
closed-manifold inside test. No simulation, rendering, model edits or saving.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

TAILS = ('R02 Strands / R02 Twin tail L', 'R02 Strands / R02 Twin tail R')
SURFACES = ('R11 Female head study', 'R11 Continuous neck bridge', 'R01 Head',
            'Blouse_Tucked_Bodice', 'Blouse_Puff_Sleeve_L', 'Blouse_Puff_Sleeve_R')


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def path(relative, suffix=None):
    q = (project_path() / relative).resolve()
    q.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if suffix and q.suffix != suffix:
        raise ValueError('Unexpected file type: ' + str(q))
    return q


def world(a, matrix):
    m = np.asarray(matrix, float)
    return a @ m[:3, :3].T + m[:3, 3]


def array(items, prop):
    a = np.empty((len(items), 3), np.float32)
    items.foreach_get(prop, a.ravel())
    return a


def trees(bpy):
    from mathutils.bvhtree import BVHTree
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    for name in SURFACES:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'MESH':
            raise RuntimeError('Explicit current collision surface missing: ' + name)
        ev = obj.evaluated_get(graph)
        mesh = ev.to_mesh()
        try:
            mesh.calc_loop_triangles()
            verts = world(array(mesh.vertices, 'co'), ev.matrix_world)
            tris = np.asarray([list(t.vertices) for t in mesh.loop_triangles], np.int32)
            if not len(tris):
                raise RuntimeError('Empty evaluated collision surface: ' + name)
            result.append((name, BVHTree.FromPolygons(verts.tolist(), tris.tolist(), all_triangles=True),
                {'object': name, 'evaluatedVertices': len(verts), 'triangles': len(tris),
                 'geometrySha256': hashlib.sha256(verts.tobytes()+tris.tobytes()).hexdigest(),
                 'worldBounds': [verts.min(0).tolist(), verts.max(0).tolist()],
                 'renderHidden': bool(obj.hide_render)}))
        finally:
            ev.to_mesh_clear()
    return result


def strand_selection(rest, guide_count=24):
    secondary = api('secondary_hair_r02')
    guides, _, _, _ = secondary['_guide_map'](rest, guide_count)
    original_ids = [int(np.argmin(np.sum((rest-g[None])**2, axis=(1, 2)))) for g in guides]
    # Uniform coverage samples original clump topology; extremal strands catch
    # a small number of thin outermost hairs that the guide interpolation omits.
    uniform = np.linspace(0, len(rest)-1, 64, dtype=int)
    extrema = []
    for at in (3, rest.shape[1]//2, rest.shape[1]-1):
        for axis in range(3):
            extrema += [int(np.argmin(rest[:, at, axis])), int(np.argmax(rest[:, at, axis]))]
    ids = np.unique(np.r_[uniform, original_ids, extrema]).astype(np.int32)
    if len(ids) > 106:
        raise RuntimeError('Unexpected contact sample budget')
    return ids


def probe(source_relative, output_relative, frames):
    import bpy
    from mathutils import Vector
    source = path(source_relative, '.blend')
    out = path(output_relative, '.json')
    npz = out.with_suffix('.npz')
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open exact independent source in background Object Mode')
    if out.exists() or npz.exists():
        raise FileExistsError('Retain previous observation files')
    if not 1 <= len(frames) <= 7 or not all(math.isfinite(f) for f in frames):
        raise ValueError('Choose 1..7 explicit finite frames; this is a sparse diagnostic')
    source_sha = sha(source)
    native = api('native_hair_playback_r04')
    secondary = api('secondary_hair_r02')
    rig = bpy.data.objects['Character_Rig']
    rows, arrays, selections, records = [], {}, {}, {}
    for i, name in enumerate(TAILS):
        obj = bpy.data.objects[name]
        if not obj.get('R11_tail_dynamics_valid') or 'R11_tail_cache_record' not in obj:
            raise RuntimeError('No new R11 matched tail cache: ' + name)
        records[name] = json.loads(obj['R11_tail_cache_record'])
        rest = array(obj.data.position_data, 'vector').reshape(len(obj.data.curves), -1, 3)
        if rest.shape != (2480, 35, 3):
            raise RuntimeError('Expected actual R11 retained 2480x35 strand topology')
        selections[name] = strand_selection(rest)
        arrays['tail_'+str(i)+'_strand_ids'] = selections[name]
        if secondary['_digest'](rest) != records[name]['staticRest'][name]['restSha256']:
            raise RuntimeError('New static tail changed after R11 bake: ' + name)
    if records[TAILS[0]] != records[TAILS[1]]:
        raise RuntimeError('The two tails record different cache inputs')
    record = records[TAILS[0]]
    cache_path = path(record['cache'], '.npz')
    if sha(cache_path) != record['cacheSha256']:
        raise RuntimeError('External cache changed from the new R11 native recording')
    if min(frames) < record['frameRange'][0] or max(frames) > record['frameRange'][1]:
        raise ValueError('Probe outside cache range')
    old_action = rig.animation_data.action if rig.animation_data else None
    old_slot = rig.animation_data.action_slot if rig.animation_data else None
    old_pose = rig.data.pose_position
    scene = bpy.context.scene
    old_frame, old_subframe = scene.frame_current, scene.frame_subframe
    old_enable = rig.get(native['ENABLE'], 0.)
    try:
        for frame_index, frame in enumerate(frames):
            native['enable_for_action'](action_name=record['action'], select=True, frame=frame)
            if frame_index == 0:
                api('adult_tail_dynamics_r11')['check_current_cache']()
            actual_surfaces = trees(bpy)
            graph = bpy.context.evaluated_depsgraph_get()
            row = {'frame': frame, 'surfaces': [r for _, _, r in actual_surfaces], 'tails': {}}
            for tail_index, name in enumerate(TAILS):
                obj = bpy.data.objects[name].evaluated_get(graph)
                selected = world(array(obj.data.position_data, 'vector'), obj.matrix_world).reshape(2480, 35, 3)[selections[name]]
                # Include root-adjacent free point and all later segment halves.
                free = selected[:, 3:]
                mids = (selected[:, 2:-1] + selected[:, 3:]) * .5
                samples = np.concatenate([free, mids], axis=1)
                prefix = 'frame_'+str(frame_index)+'_tail_'+str(tail_index)
                arrays[prefix+'_world_points'] = selected.astype(np.float32)
                sample_flat = samples.reshape(-1, 3)
                signed = np.empty((len(sample_flat), len(actual_surfaces)), np.float32)
                distance = np.empty_like(signed)
                contact_rows = []
                for surface_index, (surface_name, tree, _) in enumerate(actual_surfaces):
                    for point_index, p in enumerate(sample_flat):
                        hit, normal, triangle, d = tree.find_nearest(Vector(p))
                        if hit is None:
                            raise RuntimeError('Actual surface nearest query returned no result')
                        distance[point_index, surface_index] = d
                        signed[point_index, surface_index] = (Vector(p)-hit).dot(normal)
                    crossings = []
                    for strand_index, strand in enumerate(selected):
                        for segment_index in range(2, len(strand)-1):
                            a, b = Vector(strand[segment_index]), Vector(strand[segment_index+1])
                            direction = b-a
                            length = direction.length
                            if length < 1e-7:
                                raise RuntimeError('Degenerate evaluated tail segment')
                            hit, normal, triangle, d = tree.ray_cast(a, direction.normalized(), length)
                            if hit is not None and d > 1e-6 and d < length-1e-6:
                                crossings.append({'strandId': int(selections[name][strand_index]),
                                    'segment': segment_index, 'triangle': int(triangle),
                                    'distanceFromSegmentStartMeters': float(d), 'point': list(hit)})
                    near = distance[:, surface_index] < .0015
                    negative = signed[:, surface_index] < -.0005
                    contact_rows.append({'surface': surface_name,
                        'minimumUnsignedDistanceMeters': float(distance[:, surface_index].min()),
                        'minimumNearestNormalOffsetMeters': float(signed[:, surface_index].min()),
                        'samplesWithin1_5mm': int(near.sum()),
                        'samplesNegativeNormalOver0_5mm': int(negative.sum()),
                        'sampledSegmentCrossings': len(crossings), 'crossingEvidence': crossings[:120],
                        'crossingEvidenceTruncated': len(crossings) > 120,
                        'interpretation': 'Crossings are actual segment hits. Normal-sign samples alone are not a closed-surface penetration proof.'})
                arrays[prefix+'_nearest_distance'] = distance
                arrays[prefix+'_normal_offset'] = signed
                row['tails'][name] = {'sampledStrands': len(selected), 'allStrands': 2480,
                    'freePointAndMidpointSamples': len(sample_flat), 'actualSurfaceContact': contact_rows}
            rows.append(row)
            print('R11_TAIL_SURFACE_FRAME ' + str(frame), flush=True)
    finally:
        secondary['_assign_action'](rig, old_action)
        if old_slot is not None:
            rig.animation_data.action_slot = old_slot
        rig.data.pose_position = old_pose
        rig[native['ENABLE']] = old_enable
        scene.frame_set(old_frame, subframe=old_subframe)
        bpy.context.view_layer.update()
    if sha(source) != source_sha:
        raise RuntimeError('Source file changed during readonly observation')
    out.parent.mkdir(parents=True, exist_ok=True)
    with npz.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    result = {'schemaVersion': 1, 'source': source_relative, 'sourceSha256': source_sha,
        'helperSha256': sha(__file__), 'cache': record['cache'], 'cacheSha256': record['cacheSha256'],
        'action': record['action'], 'actionSha256': record['actionSha256'], 'frames': rows,
        'arrayFile': npz.name, 'arraySha256': sha(npz), 'savedBlend': False,
        'fullCollisionAccepted': False, 'appearanceAccepted': False,
        'scope': 'Actual evaluated surface and sparse strand/segment contact; excludes pinned points 0..2 from point tests.',
        'limitations': ['At most 106 of 2480 strands per side and at most seven frames; not a full continuous collision audit.',
                        'No hair self collision, donor fringe motion, cloth simulation or visual naturalness evidence.',
                        'Near/negative nearest-normal results must be interpreted with actual surface orientation and render; they are not automatic inside tests.']}
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {'report': output_relative, 'reportSha256': sha(out), 'frames': frames,
            'sourceChanged': False, 'fullCollisionAccepted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', required=True, help='Comma-separated explicit integer/fractional cached frames')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    frames = [float(f.strip()) for f in args.frames.split(',')]
    print('R11_TAIL_SURFACE_PROBE=' + json.dumps(probe(args.source, args.output, frames)), flush=True)


if __name__ == '__main__':
    main()
