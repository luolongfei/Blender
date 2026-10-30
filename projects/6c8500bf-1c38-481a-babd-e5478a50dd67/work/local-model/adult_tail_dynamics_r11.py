"""Explicit R11 tail cache plan/bake on a saved independent adopted groom.

No import-time Blender work. The plan measures collision-envelope candidates
from the current evaluated REST meshes; it is not a collision acceptance. Bake
uses the exact supplied plan, rebuilds all representative guides and embeds the
result through the existing native playback implementation. Original source,
old cache directories, rig rest, editable geometry and actions are preserved.

Run plan first, inspect the actual envelope and choose a short action interval.
Only the root integrator should execute Blender jobs, one at a time.
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
HEAD = 'R11 Female head study'
BRIDGE = 'R11 Continuous neck bridge'
RECORD = 'R11_tail_cache_record'
VALID = 'R11_tail_dynamics_valid'
PARAMETERS = dict(guides_per_tail=24, substeps=6, iterations=8,
                  pinned_points=3, gravity=[0., 0., -9.81],
                  damping_per_second=9., root_frequency_hz=12.,
                  tip_frequency_hz=4., bend_stiffness=.32,
                  collision_margin=.0015, contact_friction=.18,
                  warmup_seconds=.6, loop_preroll_cycles=0)
LIMITATIONS = [
    'Only the two tails use this cache; donor crown/fringe/temple guides are not dynamically simulated.',
    'PBD elastic guide model and sphere envelopes are approximations, not measured hair material or actual garment mesh collision.',
    'New static rest, head/neck shapes, garment changes, body action, constraints or attachment motion invalidate this cache.',
    'Native parity and input fingerprints do not prove appearance, surface contact, self collision or a natural continuous performance.',
    'Actual sampled mesh contact, full continuous video, source reopen, and separate realtime adaptation remain necessary.'
]


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def path(relative, suffix=None):
    p = project_path()
    q = (p / relative).resolve()
    q.relative_to((p / 'work/reconstruction-r02').resolve())
    if suffix and q.suffix != suffix:
        raise ValueError('Unexpected file type: ' + str(q))
    return q


def write_new(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def points(obj):
    items, field = (obj.data.position_data, 'vector') if obj.type == 'CURVES' else (obj.data.vertices, 'co')
    a = np.empty((len(items), 3), np.float32)
    items.foreach_get(field, a.ravel())
    return a


def world(a, matrix):
    m = np.asarray(matrix, float)
    return a @ m[:3, :3].T + m[:3, 3]


def mesh_world(obj, graph):
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        a = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get('co', a.ravel())
        result = world(a, evaluated.matrix_world)
        if not len(result) or not np.isfinite(result).all():
            raise RuntimeError('Missing/non-finite evaluated mesh: ' + obj.name)
        return result
    finally:
        evaluated.to_mesh_clear()


def guard(source_relative, require_static=True):
    import bpy
    source = path(source_relative, '.blend')
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the exact independent source in background Object Mode')
    rig = bpy.data.objects.get('Character_Rig')
    if rig is None or rig.type != 'ARMATURE' or rig.library:
        raise RuntimeError('Production Character_Rig is required')
    if len([o for o in bpy.context.scene.objects if o.type == 'ARMATURE']) != 1:
        raise RuntimeError('Unexpected additional rig')
    if abs(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base - 30.) > 1e-6:
        raise RuntimeError('Current action/cache timebase must be 30 fps')
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA must be resolved before explicit action caching')
    if bpy.data.objects.get(HEAD) is None or bpy.data.objects.get(BRIDGE) is None:
        raise RuntimeError('Actual R11 head and continuous neck bridge required')
    for name in TAILS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'CURVES' or not obj.get('R11_tail_rest_shape'):
            raise RuntimeError('Adopted R11 static tail marker required: ' + name)
        if require_static and (obj.get(VALID) or obj.modifiers.get('R04 Native hair playback')):
            raise RuntimeError('Plan/bake only the adopted static tail copy; preserve existing dynamic source: ' + name)
    return bpy, source, rig


def contract(bpy, rig):
    """Raw source, all current keys/constraints, and attachment rest, not a pose image."""
    body = api('body_hair_driver_r09')
    uv = api('production_uv_r08')
    native = api('native_hair_playback_r04')
    # _geometry intentionally excludes only the new native mapping attributes.
    geometry = {o.name: body['_geometry'](o) for o in bpy.context.scene.objects
                if o.type in {'MESH', 'CURVES'} and not o.get('r04_native_hair_active')}
    keys = {o.name: {k.name: float(k.value) for k in o.data.shape_keys.key_blocks}
            for o in bpy.context.scene.objects if o.type == 'MESH' and o.data.shape_keys
            and not o.get('r04_native_hair_active')}
    animated_paths = {curve.data_path for action in bpy.data.actions for curve in native['_fcurves'](action)}
    if rig.animation_data:
        animated_paths.update(curve.data_path for curve in rig.animation_data.drivers)

    def constraint_settings(constraint):
        result = uv['_scalar_rna'](constraint)
        for name in list(result):
            channel = constraint.path_from_id(name)
            if channel in animated_paths:
                result[name] = {'animatedChannel': channel}
        return result

    constraints = {bone.name: [{'settings': constraint_settings(c),
                                'target': uv['_plain'](getattr(c, 'target', None))}
                               for c in bone.constraints] for bone in rig.pose.bones if bone.constraints}
    return {'geometry': geometry, 'rigRest': body['_rig_rest'](rig),
            'drivers': body['_stable_ids']((uv['_animation'](rig) or {}).get('drivers', [])),
            'constraints': body['_stable_ids'](constraints),
            'shapeValues': keys,
            'actions': {a.name: native['_action_fingerprint'](a) for a in bpy.data.actions if native['_fcurves'](a)},
            'tailAttachment': {name: {'parent': bpy.data.objects[name].parent.name,
                'parentBone': bpy.data.objects[name].parent_bone,
                'parentInverse': [list(r) for r in bpy.data.objects[name].matrix_parent_inverse],
                'basis': [list(r) for r in bpy.data.objects[name].matrix_basis]} for name in TAILS}}


def check_current_cache():
    """Read-only structural gate for reviewers; native checks pose/action too.

    This does not mark surface contact or continuous motion accepted. Shape-key
    values may animate, but edited rest/key geometry, driver definitions, rig
    rest, parent attachments or any original action invalidate the recorded
    source. Existing native _recorded_state additionally checks the current
    active clip, sampled body matrices, frame, and original tail rest hashes.
    """
    import bpy
    rig = bpy.data.objects['Character_Rig']
    records = []
    for name in TAILS:
        obj = bpy.data.objects[name]
        if not obj.get(VALID) or RECORD not in obj:
            raise RuntimeError('R11 tail has no new cache marker: ' + name)
        records.append(json.loads(obj[RECORD]))
    if records[0] != records[1]:
        raise RuntimeError('The two R11 tail cache records differ')
    record = records[0]
    expected = record['inputContract']
    actual = contract(bpy, rig)
    changed = []
    for category in ('geometry', 'actions'):
        changed += [category + ':' + name for name, value in expected[category].items()
                    if actual[category].get(name) != value]
    for category in ('rigRest', 'drivers', 'constraints', 'tailAttachment'):
        if actual[category] != expected[category]:
            changed.append(category)
    if changed:
        raise RuntimeError('R11 tail cache input changes require a new stage: ' + json.dumps(changed))
    native = api('native_hair_playback_r04')
    guide = native['_active_guide'](rig)
    native_record = native['_recorded_state'](guide, rig)
    if native_record['cacheSha256'] != record['cacheSha256']:
        raise RuntimeError('Active native cache differs from the R11 recorded cache')
    cache_path, meta, arrays, _, _ = native['load_cache'](record['cache'])
    if sha(cache_path) != record['cacheSha256']:
        raise RuntimeError('External R11 cache differs from the embedded recording')
    # The legacy native check covers head attachments. Also verify every
    # body-following collision sphere so edited arm/shoulder control state does
    # not accidentally pass just because the head transform remained equal.
    scene = bpy.context.scene
    old_frame, old_subframe = scene.frame_current, scene.frame_subframe
    frame = old_frame + old_subframe
    anchors = sorted(set([int(math.floor(frame)), int(math.ceil(frame))]))
    collider_checks = []
    try:
        for anchor in anchors:
            scene.frame_set(anchor)
            bpy.context.view_layer.update()
            index = anchor - int(arrays['frames'][0])
            current = []
            for spec in meta['colliders']:
                m = np.asarray(rig.matrix_world @ rig.pose.bones[spec['bone']].matrix, float)
                current.append(m[:3, :3] @ np.asarray(spec['localCenter']) + m[:3, 3])
            error = np.linalg.norm(np.asarray(current) - arrays['collider_centers'][index], axis=1)
            collider_checks.append({'frame': anchor, 'maxCenterErrorMeters': float(error.max())})
            if error.max() > 2e-4:
                raise RuntimeError('Body/garment collider attachment changed from the recorded motion: ' + json.dumps(collider_checks))
    finally:
        scene.frame_set(old_frame, subframe=old_subframe)
        bpy.context.view_layer.update()
    return {'technicalCacheMatches': True, 'action': record['action'],
            'frameRange': record['frameRange'], 'cacheSha256': record['cacheSha256'],
            'colliderAttachmentChecks': collider_checks,
            'surfaceCollisionAccepted': False, 'appearanceAccepted': False}


def rest_envelopes(bpy, rig):
    """Measured sphere *proposal*, no use of R09 hard-coded head center/radius.

    Band spheres deliberately enclose all measured points of the selected
    current surface band. That can be overly conservative; the report retains
    every fitting region and overshoot instead of silently shrinking a radius.
    """
    from mathutils import Vector
    old_pose = rig.data.pose_position
    old_frame, old_subframe = bpy.context.scene.frame_current, bpy.context.scene.frame_subframe
    specs, rows = [], []

    def sphere(name, bone, selected, method):
        if len(selected) < 12:
            raise RuntimeError('Insufficient actual points for ' + name)
        center = (selected.min(0) + selected.max(0)) / 2.
        radial = np.linalg.norm(selected - center, axis=1)
        radius = float(radial.max() + .001)
        local = (rig.matrix_world @ rig.data.bones[bone].matrix_local).inverted() @ Vector(center)
        spec = {'name': name, 'bone': bone, 'localCenter': list(local), 'radius': radius}
        specs.append(spec)
        rows.append({'collider': spec, 'method': method, 'selectedVertices': len(selected),
                     'actualWorldBounds': [selected.min(0).tolist(), selected.max(0).tolist()],
                     'worldCenter': center.tolist(), 'radialP50P95Max': np.percentile(radial, [50, 95, 100]).tolist(),
                     'addedMarginMeters': .001, 'radiusClamped': False})

    try:
        rig.data.pose_position = 'REST'
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        head = mesh_world(bpy.data.objects[HEAD], graph)
        bridge = mesh_world(bpy.data.objects[BRIDGE], graph)
        eyes = [mesh_world(bpy.data.objects['R11 Adult sclera ' + side], graph) for side in ('L', 'R')]
        eye_center = np.mean([(v.min(0) + v.max(0)) / 2. for v in eyes], axis=0)
        head_bottom = max(float(head[:, 2].min()), float(eye_center[2] - .065))
        edges = np.linspace(head_bottom, float(head[:, 2].max()) + 1e-7, 5)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            chosen = head[(head[:, 2] >= a) & (head[:, 2] < b)]
            sphere('R11 measured head band ' + str(i + 1), 'head', chosen,
                   {'source': HEAD, 'worldZBand': [a, b], 'selection': 'All evaluated points in band; includes ear/nose extent where present'})
        neck = np.concatenate([head[head[:, 2] < head_bottom], bridge])
        neck_edges = np.linspace(float(neck[:, 2].min()), float(neck[:, 2].max()) + 1e-7, 4)
        for i, (a, b) in enumerate(zip(neck_edges[:-1], neck_edges[1:])):
            chosen = neck[(neck[:, 2] >= a) & (neck[:, 2] < b)]
            sphere('R11 measured neck band ' + str(i + 1), 'neck', chosen,
                   {'sources': [HEAD, BRIDGE], 'worldZBand': [a, b], 'selection': 'All actual lower neck and bridge points in band'})
        # Shoulder/chest envelope is explicitly the retained rig approximation.
        secondary = api('secondary_hair_r02')
        for spec in secondary['_collider_definitions'](rig):
            if spec['name'] not in {'head', 'neck'}:
                specs.append(spec)
                rows.append({'collider': spec, 'method': 'Retained rig-anatomy sphere approximation, not actual garment fit'})
        # Full sleeve local sections, no historical 85th-percentile radius cap.
        for side, suffix in (('left', 'L'), ('right', 'R')):
            name = 'Blouse_Puff_Sleeve_' + suffix
            sleeve = mesh_world(bpy.data.objects[name], graph)
            bone = rig.data.bones[side + 'UpperArm']
            inv = np.asarray((rig.matrix_world @ bone.matrix_local).inverted(), float)
            local = world(sleeve, inv)
            for i, (a, b) in enumerate(zip(np.linspace(local[:, 1].min(), local[:, 1].max() + 1e-7, 4)[:-1],
                                         np.linspace(local[:, 1].min(), local[:, 1].max() + 1e-7, 4)[1:])):
                chosen = sleeve[(local[:, 1] >= a) & (local[:, 1] < b)]
                sphere('R11 measured ' + side + ' sleeve ' + str(i + 1), bone.name, chosen,
                       {'source': name, 'boneLocalYBand': [float(a), float(b)], 'selection': 'All evaluated sleeve points in axial band'})
    finally:
        rig.data.pose_position = old_pose
        bpy.context.scene.frame_set(old_frame, subframe=old_subframe)
        bpy.context.view_layer.update()
    return specs, rows


def plan(source_relative, output_relative):
    bpy, source, rig = guard(source_relative)
    out = path(output_relative, '.json')
    if out.exists():
        raise FileExistsError('Preserve previous plan: ' + str(out))
    source_sha = sha(source)
    before = contract(bpy, rig)
    secondary = api('secondary_hair_r02')
    maps = {}
    for name in TAILS:
        obj, rest = secondary['_read_tail'](name)
        guides, ids, weights, padding = secondary['_guide_map'](rest, PARAMETERS['guides_per_tail'])
        maps[name] = {'restSha256': secondary['_digest'](rest), 'strands': len(rest),
                      'pointsPerStrand': rest.shape[1], 'guideSha256': secondary['_digest'](guides),
                      'mapSha256': secondary['_digest'](ids, weights), 'paddingMeters': padding.tolist(),
                      'newGuideCount': len(guides), 'oldCacheInvalid': True}
    specs, fit = rest_envelopes(bpy, rig)
    if contract(bpy, rig) != before or sha(source) != source_sha:
        raise RuntimeError('Read-only planning changed source contract')
    result = {'schemaVersion': 1, 'source': source_relative, 'sourceSha256': source_sha,
              'helperSha256': sha(__file__), 'contract': before, 'contractSha256': digest(before),
              'newGuideMaps': maps, 'colliders': specs, 'colliderMeasurements': fit,
              'simulationParameters': PARAMETERS, 'automaticEnvelopeProposal': True,
              'savedBlend': False, 'appearanceAccepted': False, 'collisionAccepted': False,
              'limitations': LIMITATIONS,
              'next': 'Inspect measured sphere envelope and run a bounded 31-61 frame cache before full action. Use new stage paths.'}
    write_new(out, result)
    return {'plan': output_relative, 'sha256': sha(out), 'colliders': len(specs), 'oldCachesInvalid': True}


def bake(source_relative, plan_relative, action_name, start, end, cache_directory, output_relative):
    bpy, source, rig = guard(source_relative)
    target = path(output_relative, '.blend')
    report_path = target.with_suffix('.tail-dynamics.json')
    folder = path(cache_directory)
    if target == source or target.exists() or report_path.exists() or folder.exists():
        raise RuntimeError('Use a new independent target, report and cache directory; retain previous stages')
    if not 2 <= end - start + 1 <= 600:
        raise ValueError('Require 2..600 consecutive frames; first trial should be 31-61 frames')
    plan_path = path(plan_relative, '.json')
    proposal = json.loads(plan_path.read_text(encoding='utf-8'))
    source_sha = sha(source)
    before = contract(bpy, rig)
    if proposal['sourceSha256'] != source_sha or proposal['source'] != source_relative or proposal['contract'] != before:
        raise RuntimeError('Envelope/rest/action plan is stale; create a new plan from this exact source')
    native = api('native_hair_playback_r04')
    secondary = api('secondary_hair_r02')
    action = bpy.data.actions.get(action_name)
    if action is None or not native['_fcurves'](action):
        raise RuntimeError('Explicit existing action required: ' + action_name)
    action_range = list(action.frame_range)
    if start < math.floor(action_range[0]) or end > math.ceil(action_range[1]):
        raise ValueError('Requested frames extend beyond the recorded action')
    for curve in native['_fcurves'](action):
        rig.path_resolve(curve.data_path)
    params = proposal['simulationParameters']
    if params != PARAMETERS:
        raise RuntimeError('Plan parameters differ from this audited helper; revise helper/plan explicitly')
    rig.data.pose_position = 'POSE'
    secondary['_assign_action'](rig, action)
    bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()
    folder.mkdir(parents=True)
    write_new(folder / 'inputs-r11.json', {'source': source_relative, 'sourceSha256': source_sha,
        'plan': plan_relative, 'planSha256': sha(plan_path), 'action': action_name,
        'actionSha256': native['_action_fingerprint'](action), 'frameRange': [start, end],
        'contractSha256': digest(before), 'parameters': params, 'colliders': proposal['colliders']})
    baked = secondary['bake_actions'](actions=[action_name], frame_ranges={action_name: (start, end)},
        output=cache_directory, allow_authored_ik=True, colliders=proposal['colliders'], **params)
    cache = baked['actions'][0]['cache'].replace('\\', '/')
    metrics = baked['actions'][0]['metrics']
    # Fail prior to installing or saving. Keep the generated failed trial files.
    if metrics['maxRootDriftMeters'] > 2e-6 or metrics['maxSegmentRelativeStretch'] > .15:
        raise RuntimeError('New tail trial root/stretch gate failed: ' + json.dumps(metrics))
    build = native['run'](cache_relative=cache)
    native['enable_for_action'](action_name=action_name, select=True, frame=start)
    check_frames = sorted(set([float(start), start + (end-start)*.25 + .375,
                               start + (end-start)*.5, start + (end-start)*.75, float(end)]))
    check_frames = [f for f in check_frames if f <= end]
    parity = native['validate_evaluated'](cache_relative=cache, frames=check_frames)
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    bpy.context.scene.frame_set(start)
    bpy.context.view_layer.update()
    after = contract(bpy, rig)
    # Native playback adds a new hidden guide mesh/action; original entries
    # must remain byte-for-byte equivalent. New animation pose is deliberate.
    for category in ('geometry', 'rigRest', 'drivers', 'constraints', 'tailAttachment'):
        if category == 'geometry':
            if any(after[category].get(k) != v for k, v in before[category].items()):
                raise RuntimeError('Original editable geometry changed during tail bake')
        elif after[category] != before[category]:
            raise RuntimeError('Original contract changed: ' + category)
    if any(after['actions'].get(k) != v for k, v in before['actions'].items()):
        raise RuntimeError('Existing action data changed during tail bake')
    record = {'schemaVersion': 1, 'source': source_relative, 'sourceSha256': source_sha,
              'plan': plan_relative, 'planSha256': sha(plan_path), 'action': action_name,
              'actionSha256': native['_action_fingerprint'](action), 'frameRange': [start, end], 'fps': 30,
              'cache': cache, 'cacheSha256': sha(project_path()/cache),
              'inputContract': before, 'inputContractSha256': digest(before),
              'staticRest': proposal['newGuideMaps'], 'parameters': params,
              'nativeBuild': build, 'actualNativeParity': parity, 'solverMetrics': metrics,
              'technicalCacheMatches': True, 'surfaceCollisionAccepted': False,
              'appearanceAccepted': False, 'continuousVideoReviewed': False, 'limitations': LIMITATIONS}
    for name in TAILS:
        bpy.data.objects[name][RECORD] = json.dumps(record, ensure_ascii=False)
        bpy.data.objects[name][VALID] = True
    if sha(source) != source_sha:
        raise RuntimeError('Original source changed on disk')
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    record.update(target=output_relative, targetSha256=sha(target), savedBlend=True,
                  requiresFreshReopen=True, helperSha256=sha(__file__))
    write_new(report_path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('plan', 'bake'), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--action')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--cache-directory')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.mode == 'plan':
        result = plan(args.source, args.output)
    else:
        if None in (args.plan, args.action, args.start, args.end, args.cache_directory):
            parser.error('bake requires --plan --action --start --end --cache-directory')
        result = bake(args.source, args.plan, args.action, args.start, args.end, args.cache_directory, args.output)
    print('R11_TAIL_DYNAMICS=' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
