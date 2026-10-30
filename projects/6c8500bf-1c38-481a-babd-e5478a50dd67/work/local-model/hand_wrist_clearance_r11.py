"""Bounded reversible correction of the actual old forearm/new-hand overlap.

The R10 v2 observation proves the old closed forearm cap protrudes dorsally
through the fitted HBM hand. Only that cap is fitted inside the actual hand;
head, fingers, nails, rig, weights and original shape-key deltas stay intact.
The retained internal overlap is explicitly not a welded anatomical wrist.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
OBS = 'work/reconstruction-r02/hand-junction-r11.json'
ARRAYS_SHA = '9f7bba9b423336d64ab07f3dbc234e5ad866d13f65ff90d04d6b76fc1ed48179'
HAND_PLAN = 'work/reconstruction-r02/hand-asset-fit-plan-r10-v3.json'
HAND_PAYLOAD_SHA = 'c86aaacc0fde9a15d6d9009531727b8a3e7d8ecff931fea81f38572b70ad6bbc'
KEY = 'R11 wrist overlap clearance'
MARKER = 'R11_wrist_overlap_clearance'
DEFAULT_PLAN = 'work/reconstruction-r02/hand-wrist-clearance-plan-r11.json'


def _path(relative):
    value = (project_path() / relative).resolve(); value.relative_to(project_path().resolve()); return value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _smooth(a, b, value):
    x = np.clip((value-a)/(b-a), 0, 1); return x*x*(3-2*x)


def _section_segments(vertices, triangles, x):
    """Real hand triangle/plane intersections; no ellipse approximation."""
    tri = vertices[triangles]; mask = (tri[:, :, 0].min(1) < x) & (tri[:, :, 0].max(1) > x)
    tri = tri[mask]; segments = []
    for points in tri:
        hits = []
        for i in range(3):
            a = points[i]; b = points[(i+1) % 3]
            if (a[0]-x)*(b[0]-x) < 0:
                hits.append((a+(b-a)*((x-a[0])/(b[0]-a[0])))[1:])
        if len(hits) == 2: segments.append(hits)
    if len(segments) < 30: raise RuntimeError('Actual hand cross-section is incomplete at ' + str(x))
    return np.asarray(segments)


def _radial_limit(segments, point):
    # The midpoint of this measured wrist section lies inside its loop. Check
    # two opposite ray crossings explicitly before using its radial envelope.
    points = segments.reshape((-1, 2)); center = (points.min(0)+points.max(0))*.5
    direction = point-center; radius = np.linalg.norm(direction)
    if radius < 1e-8: return point.copy(), 0., 0.
    direction /= radius; edge = segments[:, 1]-segments[:, 0]; q = segments[:, 0]-center
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    denominator = cross(direction, edge); stable = np.abs(denominator) > 1e-12
    t = np.full(len(edge), np.nan); u = np.full(len(edge), np.nan)
    t[stable] = cross(q[stable], edge[stable])/denominator[stable]
    u[stable] = cross(q[stable], direction)/denominator[stable]
    hits = t[stable & (u >= -1e-7) & (u <= 1+1e-7)]
    positive = hits[hits > 1e-7]; negative = hits[hits < -1e-7]
    if not len(positive) or not len(negative): raise RuntimeError('Measured section center is not bracketed by the real hand')
    limit = float(positive.min())
    target = min(radius, max(.003, limit-.00125))
    return center+direction*target, float(radius), limit


def _triangle_metrics(old, new, triangles):
    edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    a = old[triangles]; b = new[triangles]
    na = np.cross(a[:, 1]-a[:, 0], a[:, 2]-a[:, 0]); nb = np.cross(b[:, 1]-b[:, 0], b[:, 2]-b[:, 0])
    length_a = np.linalg.norm(na, axis=1); length_b = np.linalg.norm(nb, axis=1)
    cosine = np.einsum('ij,ij->i', na, nb)/np.maximum(length_a*length_b, 1e-20)
    ratios = np.linalg.norm(new[edges[:, 0]]-new[edges[:, 1]], axis=1)/np.maximum(np.linalg.norm(old[edges[:, 0]]-old[edges[:, 1]], axis=1), 1e-12)
    valid = length_a > 1e-12
    return {'minimumNormalCosine': float(cosine[valid].min()), 'flippedTriangles': int(np.sum(cosine[valid] <= 0)),
            'edgeLengthRatioRange': [float(ratios.min()), float(ratios.max())]}


def plan(output_relative=DEFAULT_PLAN):
    output = _path(output_relative); archive = output.with_suffix('.npz')
    if output.exists() or archive.exists(): raise FileExistsError('Preserve existing plans')
    observation = json.loads(_path(OBS).read_text(encoding='utf-8'))
    if observation['arraysSha256'] != ARRAYS_SHA or _sha(_path(observation['arrays'])) != ARRAYS_SHA:
        raise RuntimeError('Actual wrist observation changed')
    recipe = json.loads(_path(HAND_PLAN).read_text(encoding='utf-8'))
    if recipe['payloadSha256'] != HAND_PAYLOAD_SHA or _sha(_path(recipe['payload'])) != HAND_PAYLOAD_SHA:
        raise RuntimeError('Actual fitted hand payload changed')
    payload = {}; summaries = {}; all_ids = []; all_delta = []
    with np.load(_path(observation['arrays'])) as observed, np.load(_path(recipe['payload'])) as hand_data:
        loops = hand_data['loopVertex']; starts = hand_data['polygonStart']; sizes = hand_data['polygonSize']
        faces = [loops[a:a+b] for a, b in zip(starts, sizes)]
        triangles = np.asarray([[int(f[0]), int(f[i]), int(f[i+1])] for f in faces for i in range(1, len(f)-1)])
        edge_count = {}
        for face in faces:
            for a, b in zip(face, np.roll(face, -1)):
                key = tuple(sorted((int(a), int(b)))); edge_count[key] = edge_count.get(key, 0)+1
        if any(count != 2 for count in edge_count.values()): raise RuntimeError('Actual hand is not closed/manifold for the overlap plan')
        for side in ('L', 'R'):
            prefix = side+'_raw_1'; old = observed[prefix+'_positions'].copy(); normalized = old.copy(); normalized[:, 0] = np.abs(normalized[:, 0])
            hand = hand_data['left' if side == 'L' else 'right'].copy(); hand[:, 0] = np.abs(hand[:, 0])
            delta = np.zeros_like(old); section_checks = []
            for i, point in enumerate(normalized):
                if not .590 < point[0] < .631: continue
                segments = _section_segments(hand, triangles, float(point[0]))
                target, radius, limit = _radial_limit(segments, point[1:])
                delta[i, 1:] = (target-point[1:])*_smooth(.590, .600, point[0])
                section_checks.append((i, point[0], radius, limit))
            new = old+delta; changed = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-10)
            ids = observed[prefix+'_originalVertexIds']; metrics = _triangle_metrics(old, new, observed[prefix+'_triangles'])
            maximum = float(np.linalg.norm(delta, axis=1).max())
            if maximum > .0045 or metrics['flippedTriangles'] or metrics['edgeLengthRatioRange'][0] < .4:
                raise RuntimeError('Wrist correction exceeded bounded geometry guard: '+json.dumps({'side': side, 'maximum': maximum, **metrics}))
            payload[side+'_observedIds'] = ids; payload[side+'_observedMix'] = old
            payload[side+'_observedBase'] = observed[prefix+'_basePositionsLocal']
            payload[side+'_delta'] = delta; payload[side+'_sectionSamples'] = np.asarray(section_checks)
            all_ids.extend(ids[changed].tolist()); all_delta.extend(delta[changed].tolist())
            summaries[side] = {'changedCagePoints': len(changed), 'maximumDisplacementMeters': maximum,
                               'changedAbsoluteXRange': [float(normalized[changed, 0].min()), float(normalized[changed, 0].max())],
                               'triangleDiagnostics': metrics, 'noLongitudinalDisplacement': bool(np.all(delta[:, 0] == 0))}
    if len(all_ids) != len(set(all_ids)): raise RuntimeError('Left/right correction domains overlap')
    payload['changedIds'] = np.asarray(all_ids, np.int32); payload['delta'] = np.asarray(all_delta, np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream: np.savez_compressed(stream, **payload)
    result = {'schemaVersion': 1, 'observation': OBS, 'observationSha256': _sha(_path(OBS)),
              'observedArraysSha256': ARRAYS_SHA, 'observedSourceSha256': observation['sourceSha256'],
              'handPayloadSha256': HAND_PAYLOAD_SHA, 'arrays': str(archive.relative_to(project_path())),
              'arraysSha256': _sha(archive), 'sides': summaries, 'targetHandInsideMarginMeters': .00125,
              'method': 'Radial envelope from real hand triangle/plane intersections; only protruding old wrist cap reduced with proximal smooth fade.',
              'notClaimed': ['Welded wrist topology', 'Natural appearance', 'Actual posed containment before Blender validation', 'All motion contact']}
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def _co(data):
    values = np.empty((len(data), 3), np.float64); data.foreach_get('co', values.ravel()); return values


def _mix(obj):
    blocks = obj.data.shape_keys.key_blocks; result = _co(blocks[0].data)
    for block in blocks[1:]:
        if block.value and not block.mute: result += block.value*(_co(block.data)-_co(block.relative_key.data))
    return result


def _key_contract(obj, exclude=()):
    rows = []
    for block in obj.data.shape_keys.key_blocks:
        if block.name in exclude: continue
        rows.append({'name': block.name, 'relative': block.relative_key.name,
                     'sha256': hashlib.sha256(_co(block.data).astype(np.float32).tobytes()).hexdigest(),
                     'value': float(block.value), 'mute': block.mute, 'vertexGroup': block.vertex_group})
    return rows


def _inside_samples(points, hand):
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    vertices = hand['vertices']; triangles = hand['triangles']
    edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2): raise RuntimeError('Actual evaluated hand is not closed; inside test is inapplicable')
    tree = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    directions = [Vector(v).normalized() for v in ((.813, .347, .468), (-.271, .932, .239), (.413, -.233, .881))]
    votes = []; distances = []; disagreements = []
    for index, point in enumerate(points):
        origin = Vector(point); _, _, _, nearest = tree.find_nearest(origin)
        distances.append(float(nearest)); parities = []
        for direction in directions:
            cursor = origin.copy(); count = 0
            while count < 64:
                hit, _, _, _ = tree.ray_cast(cursor, direction, 2.)
                if hit is None: break
                count += 1; cursor = hit+direction*.000002
            if count == 64: raise RuntimeError('Containment ray exceeded bounded crossing count')
            parities.append(count % 2)
        votes.append(sum(parities) >= 2)
        if len(set(parities)) > 1: disagreements.append(index)
    outside = np.flatnonzero(~np.asarray(votes, bool))
    return {'samples': len(points), 'outsideByThreeRayMajority': len(outside),
            'outsideSampleIndices': outside.tolist(), 'rayParityDisagreementIndices': disagreements,
            'minimumSurfaceDistanceMeters': min(distances) if distances else None,
            'method': 'Actual closed evaluated hand, three non-axial repeated BVH ray parity counts; nearest distance recorded separately.'}


def _actual_overlap(bpy, body, side, api):
    armatures = [m for m in body.modifiers if m.type == 'ARMATURE']
    if len(armatures) != 1 or not armatures[0].show_viewport:
        raise RuntimeError('Expected one active body Armature modifier')
    armature = armatures[0]
    try:
        armature.show_viewport = False; bpy.context.view_layer.update()
        unposed = api['_surface'](bpy, body)
    finally:
        armature.show_viewport = True; bpy.context.view_layer.update()
    actual = api['_surface'](bpy, body)
    if not np.array_equal(unposed['triangles'], actual['triangles']):
        raise RuntimeError('Disabling only body Armature changed evaluated topology')
    p = unposed['vertices']; sign = 1 if side == 'L' else -1
    core = (p[:, 0]*sign >= .602) & (p[:, 0]*sign <= .631) & (np.abs(p[:, 1]) < .06) & (np.abs(p[:, 2]-1.235) < .06)
    if not 50 < int(core.sum()) < 3000: raise RuntimeError('Unexpected actual wrist cap sample count: '+str(core.sum()))
    hand = api['_surface'](bpy, bpy.data.objects['Hand '+side+' · continuous anatomy'])
    row = _inside_samples(actual['vertices'][core], hand); row['side'] = side
    row['actualBodyVertexIndices'] = np.flatnonzero(core).tolist()
    row['scope'] = 'Old forearm core distal to absolute rest X=.602 m; proximal fade and other hand contacts remain outside this check.'
    return row


def apply(plan_relative=DEFAULT_PLAN):
    import bpy
    from mathutils import Matrix
    if not bpy.app.background or bpy.context.mode != 'OBJECT': raise RuntimeError('Disposable background Object Mode required')
    scene = bpy.context.scene; body = bpy.data.objects['R01 Head']; rig = bpy.data.objects['Character_Rig']
    if scene.frame_current != 1 or scene.frame_subframe != 0 or rig.data.pose_position != 'POSE':
        raise RuntimeError('Keep current frame-one standing pose')
    if not np.allclose(np.asarray(body.matrix_world), np.eye(4), atol=1e-7):
        raise RuntimeError('Observed body coordinate space changed')
    if MARKER in scene or KEY in body.data.shape_keys.key_blocks:
        raise RuntimeError('Wrist correction already present; validate or preserve it rather than reapplying')
    recipe = json.loads(_path(plan_relative).read_text(encoding='utf-8'))
    if recipe['observedArraysSha256'] != ARRAYS_SHA or _sha(_path(recipe['arrays'])) != recipe['arraysSha256']:
        raise RuntimeError('Wrist plan no longer matches actual observations')
    api = runpy.run_path(str(Path(__file__).with_name('hand_asset_pose_review_r11.py')))['_api']()
    fit = runpy.run_path(str(Path(__file__).with_name('hand_asset_integrate_r11.py')))['_fit']()
    rig_before = fit['_rig_contract'](rig); keys_before = _key_contract(body); current = _mix(body)
    with np.load(_path(recipe['arrays'])) as payload:
        for side in ('L', 'R'):
            ids = payload[side+'_observedIds']
            if np.max(np.abs(current[ids]-payload[side+'_observedMix'])) > 2e-7:
                raise RuntimeError('Current wrist geometry changed since the actual observation: '+side)
        changed_ids = payload['changedIds'].copy(); delta = payload['delta'].copy()
    if not np.all((np.abs(current[changed_ids, 0]) > .590) & (np.abs(current[changed_ids, 0]) < .631)):
        raise RuntimeError('Correction would leave observed wrist domain')
    for side in ('L', 'R'):
        hand = bpy.data.objects['Hand '+side+' · continuous anatomy']
        record = json.loads(hand['R10_hbm_hand_provenance'])
        if record['fitPayloadSha256'] != HAND_PAYLOAD_SHA: raise RuntimeError('Actual hand fit changed')
    # Capture current body surface before the additive key for direct locality
    # verification; no old head geometry is removed or reauthored.
    before = api['_surface'](bpy, body)
    before_overlap = [_actual_overlap(bpy, body, side, api) for side in ('L', 'R')]
    key = body.shape_key_add(name=KEY, from_mix=False); key.relative_key = body.data.shape_keys.key_blocks[0]
    values = _co(key.relative_key.data); values[changed_ids] += delta; key.data.foreach_set('co', values.ravel()); key.value = 1.
    success = False; frames = {}; snapshots = {}
    try:
        bpy.context.view_layer.update(); after = api['_surface'](bpy, body)
        if not np.array_equal(before['triangles'], after['triangles']): raise RuntimeError('Wrist key changed evaluated body topology')
        # Identify protected regions by their actual current distance to both
        # wrists. This covers all visible head/neck/torso and the other limbs.
        wrists = np.asarray([list(rig.matrix_world@rig.pose.bones[p+'Hand'].head) for p in ('left', 'right')])
        distance = np.linalg.norm(before['vertices'][:, None]-wrists[None], axis=2).min(1)
        protected = distance > .10
        protected_error = float(np.linalg.norm(after['vertices'][protected]-before['vertices'][protected], axis=1).max())
        if protected_error > 2e-6: raise RuntimeError('Wrist key moved protected head/body region: '+str(protected_error))
        for side in ('L', 'R'):
            frame = api['_rest_frame'](bpy.data.objects['Hand '+side+' · continuous anatomy'], rig, side)
            frames[side] = frame
            snapshots[side] = {name: {'basis': np.asarray(rig.pose.bones[name].matrix_basis).tolist(),
                                     'rotationMode': rig.pose.bones[name].rotation_mode} for name in frame['axes']}
        poses = []
        for pose in ('relaxed', 'wrist-flex', 'wrist-extend'):
            for side in ('L', 'R'):
                api['_apply_pose'](bpy, rig, frames[side], pose, snapshots[side])
            rows = [_actual_overlap(bpy, body, side, api) for side in ('L', 'R')]
            poses.append({'pose': pose, 'sides': rows})
            failures = [row for row in rows if row['outsideByThreeRayMajority'] or row['minimumSurfaceDistanceMeters'] < .00015]
            if failures: raise RuntimeError('Actual corrected wrist cap not safely inside hand: '+json.dumps({'pose': pose, 'failures': failures}))
        if _key_contract(body, (KEY,)) != keys_before: raise RuntimeError('Original body/face shape keys changed')
        result = {'plan': plan_relative, 'planSha256': _sha(_path(plan_relative)), 'key': KEY, 'defaultValue': 1.,
                  'changedCagePoints': len(changed_ids), 'maximumDisplacementMeters': float(np.linalg.norm(delta, axis=1).max()),
                  'originalKeyContract': keys_before, 'newKeySha256': hashlib.sha256(values.astype(np.float32).tobytes()).hexdigest(),
                  'actualNeutralBefore': before_overlap, 'actualFinitePoseAfter': poses,
                  'protectedHeadBodyMaximumErrorMeters': protected_error, 'rigAndWeightsEdited': False,
                  'topologyEdited': False, 'appearanceAccepted': False,
                  'limits': ['Closed old cap is retained inside the actual hand; no welded wrist topology claimed.',
                             'Only relaxed, 25-degree wrist flexion and 20-degree extension were checked numerically.',
                             'Visible wrist/cuff closeups and other hand pose combinations remain required.']}
        success = True
    finally:
        for side, snapshot in snapshots.items():
            for name, row in snapshot.items():
                pb = rig.pose.bones[name]; pb.rotation_mode = row['rotationMode']; pb.matrix_basis = Matrix(row['basis'])
        bpy.context.view_layer.update()
        restored = fit['_rig_contract'](rig)
        pose_error = max(float(np.max(np.abs(np.asarray(a['pose'])-np.asarray(b['pose']))))
                         for a, b in zip(restored['bones'], rig_before['bones']))
        # Matrix assignment decomposes quaternion channels in Blender. Permit
        # only float roundoff in pose matrices; rest, parent, action and slot
        # identities remain exact.
        comparable = json.loads(json.dumps(restored))
        for row, original in zip(comparable['bones'], rig_before['bones']): row['pose'] = original['pose']
        if comparable != rig_before or pose_error > 2e-6:
            success = False
            if key.name in body.data.shape_keys.key_blocks: body.shape_key_remove(key)
            raise RuntimeError('Temporary wrist pose failed rest/action/pose restoration: '+str(pose_error))
        if not success and key.name in body.data.shape_keys.key_blocks: body.shape_key_remove(key)
    result['restoredPoseMaximumMatrixError'] = pose_error
    scene[MARKER] = json.dumps(result, ensure_ascii=False)
    return result


def prepare(source_relative, output_relative, source_sha256, plan_relative=DEFAULT_PLAN):
    import bpy
    source = _path(source_relative); output = _path(output_relative); report = output.with_suffix('.wrist-clearance.json')
    if Path(bpy.data.filepath).resolve() != source or _sha(source) != source_sha256:
        raise RuntimeError('Explicit source identity mismatch')
    if output == source or not output.is_relative_to(_path('work/reconstruction-r02')):
        raise RuntimeError('Save only a new independent reconstruction candidate')
    if output.exists() or report.exists(): raise FileExistsError('Preserve previous wrist candidates')
    result = apply(plan_relative)
    if _sha(source) != source_sha256: raise RuntimeError('Source changed during wrist preparation')
    result.update({'source': source_relative, 'sourceSha256': source_sha256, 'helperSha256': _sha(__file__),
                   'sourceUnchangedOnDisk': True, 'masterChanged': False})
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False, relative_remap=True)
    result.update({'candidate': output_relative, 'candidateSha256': _sha(output)})
    with report.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    offline = modes.add_parser('plan'); offline.add_argument('--plan', default=DEFAULT_PLAN)
    build = modes.add_parser('prepare'); build.add_argument('--plan', default=DEFAULT_PLAN)
    build.add_argument('--source', required=True); build.add_argument('--source-sha', required=True); build.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = plan(args.plan) if args.mode == 'plan' else prepare(args.source, args.output, args.source_sha, args.plan)
    print('R11_WRIST_CLEARANCE '+json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
