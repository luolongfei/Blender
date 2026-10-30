"""Read-only actual R11 neck/body motion review in one fresh background Blender.

--mode neck: retain frame-1 body pose; inspect neutral, +/-15/30 yaw and
             +/-7.5/15 pitch. No action is authored or saved.
--mode sequence: evaluate the currently selected, explicitly named body action.
--render none/sparse/continuous: continuous is only for the unmodified sequence.
The output directory is new and never overwrites previous visual evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

FOLDER = Path(__file__).resolve().parent
HEAD = 'R11 Female head study'
BRIDGE = 'R11 Continuous neck bridge'
BODY = 'R01 Head'
RIG = 'Character_Rig'
POSES = [('neutral', 0., 0.), ('left-half', 15., 0.), ('left', 30., 0.),
         ('right-half', -15., 0.), ('right', -30., 0.),
         ('down-half', 0., -7.5), ('down', 0., -15.),
         ('up-half', 0., 7.5), ('up', 0., 15.)]


def _api(name):
    return runpy.run_path(str(FOLDER / (name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _path(relative, category):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Use a project-relative path without parent traversal')
    path = (project_path() / relative).resolve()
    path.relative_to((project_path() / category).resolve())
    return path


def _set_frame(frame):
    integer = math.floor(frame)
    bpy.context.scene.frame_set(integer, subframe=frame - integer)
    bpy.context.view_layer.update()


def _evaluated(name):
    obj = bpy.data.objects[name].evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = obj.to_mesh()
    try:
        return _api('adult_neck_bridge_r11')['_mesh_arrays'](mesh, obj.matrix_world)
    finally:
        obj.to_mesh_clear()


def _same_topology(a, b):
    return len(a['world']) == len(b['world']) and all(
        np.array_equal(a[key], b[key]) for key in ('loops', 'face_starts', 'face_sizes'))


def _match(points, target):
    # Only the two explicitly extracted boundary cycles, not global nearest-skin.
    distance = np.linalg.norm(points[:, None, :] - target[None, :, :], axis=2)
    match = distance.argmin(axis=1)
    if len(np.unique(match)) != len(points):
        raise RuntimeError('REST seam correspondence is not one-to-one')
    error = distance[np.arange(len(points)), match]
    if error.max() > 2e-6:
        raise RuntimeError('Actual REST seam is already open: ' + str(float(error.max())))
    return match


def _seam_map(rig):
    old = rig.data.pose_position
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        data = {name: _evaluated(name) for name in (HEAD, BRIDGE, BODY)}
        core = _api('adult_neck_bridge_r11')
        top = core['boundary_ring'](data[HEAD])
        bottom = core['boundary_ring'](data[BODY], neck_only=True)
        bridge = bpy.data.objects[BRIDGE]
        nt = int(bridge['R11_boundary_top_count'])
        nb = int(bridge['R11_boundary_bottom_count'])
        if len(top['ids']) != nt or len(bottom['ids']) != nb:
            raise RuntimeError('Current seam counts do not match explicit bridge record')
        points = data[BRIDGE]['world']
        ti = np.arange(nt, dtype=np.int32)
        bi = np.arange(len(points) - nb, len(points), dtype=np.int32)
        hi = top['ids'][_match(points[ti], top['points'])]
        oi = bottom['ids'][_match(points[bi], bottom['points'])]
        return data, (ti, hi, bi, oi)
    finally:
        rig.data.pose_position = old
        bpy.context.view_layer.update()


def _seam_sample(reference, mapping, baseline=None):
    current = {name: _evaluated(name) for name in (HEAD, BRIDGE, BODY)}
    for name in current:
        if not _same_topology(reference[name], current[name]):
            raise RuntimeError('Actual evaluated topology changed during pose: ' + name)
    ti, hi, bi, oi = mapping
    p = current[BRIDGE]['world']
    # Fixed REST triangulation gives comparable areas even if Blender chooses a
    # different display diagonal for a deformed quad. Original polygons match.
    triangles = reference[BRIDGE]['triangles']
    tp = p[triangles]
    area = np.linalg.norm(np.cross(tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]), axis=1) / 2
    edges = np.unique(np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(p[edges[:, 1]] - p[edges[:, 0]], axis=1)
    row = {'headBridgeGapMaximumMeters': float(np.linalg.norm(p[ti] - current[HEAD]['world'][hi], axis=1).max()),
           'bodyBridgeGapMaximumMeters': float(np.linalg.norm(p[bi] - current[BODY]['world'][oi], axis=1).max()),
           'bridgeMinimumTriangleAreaMetersSquared': float(area.min()),
           'headWorldBounds': [current[HEAD]['world'].min(0).tolist(), current[HEAD]['world'].max(0).tolist()]}
    if baseline is not None:
        ar = area / np.maximum(baseline[0], 1e-15)
        er = lengths / np.maximum(baseline[1], 1e-15)
        row['bridgeAreaRatioToNeutral'] = [float(ar.min()), float(ar.max())]
        row['bridgeEdgeRatioToNeutral'] = [float(er.min()), float(er.max())]
    return row, (area, lengths)


def _neck_pose(rig, neutral_matrix, yaw, pitch):
    bone = rig.pose.bones['head']
    rz = Matrix.Rotation(math.radians(yaw), 3, 'Z')
    right = Vector((0., 0., 1.)).cross(rz @ Vector((0., -1., 0.))).normalized()
    world_rotation = Matrix.Rotation(-math.radians(pitch), 3, right) @ rz
    object_rotation = rig.matrix_world.to_3x3().inverted() @ world_rotation @ rig.matrix_world.to_3x3()
    target = neutral_matrix.copy()
    target_rotation = object_rotation @ neutral_matrix.to_3x3()
    for i in range(3):
        for j in range(3):
            target[i][j] = target_rotation[i][j]
    bone.matrix_basis = bone.bone.convert_local_to_pose(target, bone.bone.matrix_local,
        parent_matrix=bone.parent.matrix, parent_matrix_local=bone.parent.bone.matrix_local, invert=True)
    rig.update_tag()
    bpy.context.view_layer.update()
    error = max(abs(bone.matrix[i][j] - target[i][j]) for i in range(4) for j in range(4))
    if error > 3e-5:
        raise RuntimeError('Actual head pose differs from requested target: ' + str(error))


def _hair_status():
    rows = []
    for obj in bpy.data.objects:
        if 'R11_tail_rest_shape' in obj:
            record = obj.get('R11_tail_cache_record')
            try:
                record = json.loads(record) if isinstance(record, str) else str(record)
            except ValueError:
                record = {'unparsed': str(record)}
            rows.append({'object': obj.name, 'restShape': obj['R11_tail_rest_shape'],
                         'dynamicsValidFlag': bool(obj.get('R11_tail_dynamics_valid', False)),
                         'cacheRecord': record, 'visible': not obj.hide_render})
    return {'newTailRecords': rows,
            'allNewTailsClaimMatchingCache': bool(rows) and all(r['dynamicsValidFlag'] and r['cacheRecord'] for r in rows),
            'scope': 'Inventory only. Cache flags are not fresh parity or collision proof.'}


def _shoe_sample(audit, selection, ground, previous=None, dt=None):
    sample = audit['sample_sole_points'](selection)
    result = {}
    for side in ('L', 'R'):
        names = sorted(name for name in sample if name.endswith('_' + side))
        p = np.concatenate([np.asarray(sample[name], float) for name in names])
        h = p[:, 2] - ground
        row = {'lowestSelectedSoleHeightMeters': float(h.min()),
               'selectedPointCount': len(p)}
        if previous is not None:
            old = np.concatenate([np.asarray(previous[name], float) for name in names])
            contact = (h <= .003) & (old[:, 2] - ground <= .003)
            row['candidateContactPointCount'] = int(contact.sum())
            row['maximumCandidateContactXYSpeedMetersPerSecond'] = (
                float(np.linalg.norm(p[contact, :2] - old[contact, :2], axis=1).max() / dt)
                if contact.any() else None)
        result[side] = row
    return result, sample


def run(source_relative, output_relative, mode='neck', rendering='none',
        action_name=None, first=1, last=301, step=1., samples=24, view='quarter',
        validate_cloth=False):
    source = _path(source_relative, 'work')
    out = _path(output_relative, 'previews/reconstruction-r02')
    if source == (project_path() / 'work/character.blend').resolve():
        raise RuntimeError('Use a saved independent work copy')
    if not bpy.app.background or Path(bpy.data.filepath).resolve() != source or bpy.data.is_dirty:
        raise RuntimeError('Open the exact saved source in a fresh background Blender')
    if out.exists():
        raise RuntimeError('Preserve prior output directory')
    if mode not in ('neck', 'sequence') or rendering not in ('none', 'sparse', 'continuous'):
        raise ValueError('Unknown mode')
    if mode == 'neck' and rendering == 'continuous':
        raise ValueError('Finite diagnostic neck poses are not a continuous performance')
    if rendering == 'continuous' and step != 1:
        raise ValueError('Continuous video must render every integer frame')
    if not 1 <= first <= last <= 5000 or step not in (.25, .5, 1.) or not 8 <= samples <= 128:
        raise ValueError('Invalid bounded frame/sample parameters')
    if rendering != 'none' and step != 1:
        raise ValueError('Fractional-frame probes are measurements only')
    rig = bpy.data.objects[RIG]
    scene = bpy.context.scene
    if rig.type != 'ARMATURE' or HEAD not in bpy.data.objects or BRIDGE not in bpy.data.objects:
        raise RuntimeError('Current R11 head/bridge/rig required')
    if rig.pose.bones['head'].constraints:
        raise RuntimeError('Head constraints need a separately scoped pose review')
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA needs explicit composition review')
    for obj in scene.objects:
        if not obj.hide_viewport and not obj.hide_get() and any(m.type == 'CLOTH' and m.show_viewport for m in obj.modifiers):
            raise RuntimeError('Use embedded playback; this review must not launch a live cloth simulation: ' + obj.name)
    action = rig.animation_data.action if rig.animation_data else None
    if mode == 'sequence' and (action is None or action.name != action_name):
        raise RuntimeError('Explicit active action must match; no automatic old-action selection')
    if mode == 'sequence' and (first < action.frame_range[0] or last > action.frame_range[1]):
        raise RuntimeError('Requested range exceeds active action keys')
    fps = scene.render.fps / scene.render.fps_base
    if abs(fps - 30.) > 1e-6:
        raise RuntimeError('Keep current 30 fps source timebase')
    hair = _hair_status()
    if rendering == 'continuous' and not hair['allNewTailsClaimMatchingCache']:
        raise RuntimeError('Current R11 tail rest has no matching cache flag/record; run sparse diagnostic first and rebuild hair')
    source_sha = _sha(source)
    action_api = _api('walk_cloth_r06')
    action_contract = _hash(action_api['_records'](action, False)) if action else None
    action_inventory = sorted(a.name for a in bpy.data.actions)
    report_action = {'name': action.name, 'sha256': action_contract} if action else None
    old_frame = (scene.frame_current, scene.frame_subframe)
    old_pose = rig.data.pose_position
    old_slot = rig.animation_data.action_slot if rig.animation_data else None
    old_basis = {b.name: b.matrix_basis.copy() for b in rig.pose.bones}
    old_cache = rig.get('R04_native_hair_enabled')
    hidden = []
    camera = None
    camera_data = None
    render_attrs = [(scene, 'camera'), (scene.render, 'engine'), (scene.render, 'resolution_x'),
                    (scene.render, 'resolution_y'), (scene.render, 'resolution_percentage'),
                    (scene.render, 'filepath'), (scene.render, 'use_sequencer'),
                    (scene.render, 'threads_mode'), (scene.render, 'threads'),
                    (scene.render.image_settings, 'file_format'),
                    (scene.cycles, 'device'), (scene.cycles, 'samples'), (scene.cycles, 'use_denoising')]
    settings = [(owner, attr, getattr(owner, attr)) for owner, attr in render_attrs]
    report = {'status': 'running', 'source': source_relative, 'sha256': source_sha,
              'helperSha256': _sha(__file__), 'mode': mode, 'rendering': rendering,
              'fps': fps, 'activeBodyAction': report_action,
              'headParent': {'object': bpy.data.objects[HEAD].parent.name,
                  'type': bpy.data.objects[HEAD].parent_type, 'bone': bpy.data.objects[HEAD].parent_bone},
              'hairCacheInventory': hair, 'measurements': [], 'frames': [],
              'savedBlend': False, 'appearanceAccepted': False,
              'scope': 'Actual R11 evaluated motion and finite seam/shoe probes. No full collision, physical balance, natural-performance or client acceptance.'}
    out.mkdir(parents=True)
    report_path = out / 'render-report.json'
    try:
        _set_frame(1 if mode == 'neck' else first)
        if validate_cloth:
            report['freshR09ClothPayloadParity'] = _api('cloth_merge_r09')['verify_loaded'](
                _path('work/reconstruction-r02/cloth-body-r09', 'work'))
            _set_frame(1 if mode == 'neck' else first)
        if rendering == 'continuous':
            cache_check = _api('adult_tail_dynamics_r11')['check_current_cache']
            checks = []
            for check_frame in sorted({first, (first + last) // 2, last}):
                _set_frame(check_frame)
                checks.append({'frame': check_frame, 'check': cache_check()})
            report['freshTailSourceAndColliderAnchorChecks'] = checks
            _set_frame(first)
        reference, mapping = _seam_map(rig)
        report['seamIndices'] = {name: ids.tolist() for name, ids in zip(
            ('bridgeTop', 'headTop', 'bridgeBottom', 'bodyBottom'), mapping)}
        shoe_api = _api('foot_contact_audit_r03')
        selection = shoe_api['select_sole_vertices'](objects=shoe_api['DEFAULT_OBJECTS'])
        ground = shoe_api['resolve_ground_z']()
        report['shoeProbes'] = shoe_api['probe_description'](selection)
        report['groundZ'] = ground
        if mode == 'neck':
            neutral_matrix = rig.pose.bones['head'].matrix.copy()
            rig.animation_data.action = None
            if old_cache is not None:
                rig['R04_native_hair_enabled'] = 0.
            for obj in scene.objects:
                if obj.hide_render:
                    continue
                name = obj.name.lower()
                if obj.name.startswith(('R11 Donor / ', 'R11 Recombed ')) or (
                    not obj.name.startswith('R11 ') and any(t in name for t in
                    ('hair', 'forelock', 'scalp', 'gather', 'cheek wisps', 'tail', 'tie ring', 'r02 tie '))):
                    if obj.type in ('MESH', 'CURVES', 'CURVE'):
                        hidden.append(obj)
                        obj.hide_render = True
            targets = [(i + 1, name, yaw, pitch) for i, (name, yaw, pitch) in enumerate(POSES)]
        else:
            targets = [(float(f), 'source-action', None, None) for f in
                       np.arange(first, last + step * .1, step)]
        if rendering != 'none':
            camera_data = bpy.data.cameras.new('R11 motion review camera')
            camera = bpy.data.objects.new(camera_data.name, camera_data)
            scene.collection.objects.link(camera)
            scene.camera = camera
            camera_data.type = 'ORTHO'
            camera_data.dof.use_dof = False
            camera_data.ortho_scale = .43 if mode == 'neck' else 1.90
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'CPU'
            scene.cycles.samples = samples
            scene.cycles.use_denoising = True
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = 6
            scene.render.resolution_x, scene.render.resolution_y = ((640, 720) if mode == 'neck' else (480, 640))
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = 'PNG'
            scene.render.use_sequencer = False
        # Added from the actual retained cloth payload extrema, not guessed
        # action peaks: 31..39 acceleration and 133/181 waist stretch context.
        sparse = {first, last, 31, 35, 39, 55, 133, 151, 181, 247} if last == 301 else {first, last, int((first + last) / 2)}
        baseline = None
        previous = None
        for frame, label, yaw, pitch in targets:
            if mode == 'neck':
                _neck_pose(rig, neutral_matrix, yaw, pitch)
            else:
                _set_frame(frame)
            row, metrics = _seam_sample(reference, mapping, baseline)
            if baseline is None:
                baseline = metrics
            row.update(frame=frame, label=label, yaw=yaw, pitch=pitch)
            row['shoeContact'], sampled = _shoe_sample(shoe_api, selection, ground,
                previous if mode == 'sequence' else None, step / fps)
            previous = sampled
            row['headPivotWorld'] = list(rig.matrix_world @ rig.pose.bones['head'].head)
            row['neckPivotWorld'] = list(rig.matrix_world @ rig.pose.bones['neck'].head)
            report['measurements'].append(row)
            do_render = rendering == 'continuous' or (rendering == 'sparse' and (mode == 'neck' or frame in sparse))
            if do_render:
                direction = Vector({'front': (0., -1., 0.), 'quarter': (.6, -.8, 0.), 'side': (1., 0., 0.)}[view])
                if mode == 'neck':
                    lo, hi = [np.asarray(v) for v in report['measurements'][0]['headWorldBounds']]
                    target = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] - .165))
                else:
                    hip = rig.matrix_world @ rig.pose.bones['hips'].head
                    target = Vector((hip.x, hip.y, .83))
                camera.location = target + direction * 3
                camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
                image_path = out / f'frame-{int(frame):04d}.png'
                scene.render.filepath = str(image_path)
                bpy.ops.render.render(write_still=True)
                report['frames'].append({'frame': int(frame), 'label': label,
                    'path': image_path.relative_to(project_path()).as_posix(), 'sha256': _sha(image_path)})
            print('R11_MOTION_SAMPLE ' + json.dumps({'frame': frame, 'label': label,
                'headGap': row['headBridgeGapMaximumMeters'], 'bodyGap': row['bodyBridgeGapMaximumMeters']}), flush=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if (sorted(a.name for a in bpy.data.actions) != action_inventory or
            (_hash(action_api['_records'](action, False)) if action else None) != action_contract):
            raise RuntimeError('Body action or action inventory changed during read-only review')
        if _sha(source) != source_sha:
            raise RuntimeError('Saved source changed during review')
        report.update(status='completed-probes-await-visible-review', activeBodyActionUnchanged=True,
            actionInventoryUnchanged=True,
            sourceUnchanged=True, hiddenHairForNeck=[o.name for o in hidden],
            limits=['R11 head may remain rigidly parented; endpoint continuity is not natural neck deformation.',
                'Seam and bridge strain are finite geometric diagnostics, not full surface collision.',
                'Shoe probes are selected original rigid cage points, excluding bevel-generated surface.',
                'Low/speed contact flags are geometric candidates, not ground force or full no-slide proof.',
                'Neck poses keep body frame 1 and hide hair for structure visibility; old tail cache does not follow these temporary head poses.',
                'No old or new hair/cloth report alone establishes new full-character physical performance.'])
    except Exception as exc:
        report.update(status='failed-evidence-retained', error=repr(exc))
        raise
    finally:
        if rig.animation_data:
            rig.animation_data.action = action
            if old_slot is not None:
                rig.animation_data.action_slot = old_slot
        rig.data.pose_position = old_pose
        if old_cache is not None:
            rig['R04_native_hair_enabled'] = old_cache
        for name, value in old_basis.items():
            rig.pose.bones[name].matrix_basis = value
        for obj in hidden:
            obj.hide_render = False
        for owner, attr, value in settings:
            setattr(owner, attr, value)
        if camera is not None:
            bpy.data.objects.remove(camera, do_unlink=True)
        if camera_data is not None and camera_data.users == 0:
            bpy.data.cameras.remove(camera_data)
        scene.frame_set(old_frame[0], subframe=old_frame[1])
        bpy.context.view_layer.update()
        report['originalActionPoseFrameAndRenderSettingsRestored'] = True
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('neck', 'sequence'), default='neck')
    parser.add_argument('--render', choices=('none', 'sparse', 'continuous'), default='none')
    parser.add_argument('--action')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, default=301)
    parser.add_argument('--step', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--view', choices=('front', 'quarter', 'side'), default='quarter')
    parser.add_argument('--validate-cloth-r09', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    run(args.source, args.output, args.mode, args.render, args.action, args.first,
        args.last, args.step, args.samples, args.view, args.validate_cloth_r09)
