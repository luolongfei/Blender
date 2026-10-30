"""One coherent, reversible R10 crown-volume work candidate; never renders.

Use observe() first on the exact R09 refined source. It exports current actual
PN root/scalp/forelock/gather data plus render-level head surface, then restores
temporary pose/subdivision settings. No saved source is altered by observation.
Candidate construction is explicit; nothing executes on import.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SOURCE = 'work/reconstruction-r02/reference-production-refined-r09.blend'
SOURCE_SHA = '1c0b0210114170f0ce6e09b266b4a9ded58f29bf1df45a6e2be13e9b5c77aa72'
OBSERVATION = 'work/reconstruction-r02/crown-observation-r10.json'
OBSERVATION_V2 = 'work/reconstruction-r02/crown-observation-r10-v2.json'
SHELL = 'R06 attached scalp root substrate'
SCALP = 'R06 continuous scalp flow'
FORELOCK = 'R03 Forelocks from crown'
GATHERS = ('R06 full gathered roots L', 'R06 full gathered roots R')
TAILS = ('R02 Strands / R02 Twin tail L', 'R02 Strands / R02 Twin tail R')
NAMES = (SHELL, SCALP, FORELOCK)+GATHERS
MARKER = 'R10_crown_volume_candidate'
RECORD = 'R10_crown_volume_restore_record'
BACKUP = 'R10_crown_original_data'
PARAMETERS = {'topLiftMeters': .016, 'sideExpansionMeters': .016, 'depthExpansionMeters': .006,
              'anchorFixedRadiusMeters': .0015, 'anchorFullDistanceMeters': .033,
              'heightZeroAboveEyeMeters': .017, 'heightFullAboveEyeMeters': .057,
              'verticalLiftStartsAboveEyeMeters': .045,
              'maximumVectorDisplacementMeters': .026}


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _path(relative, suffix):
    path = (project_path()/relative).resolve()
    path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix != suffix:
        raise ValueError('Unexpected file suffix: '+str(path))
    return path


def _sha(path):
    return _api('posterior_groom_r09')['_sha'](Path(path))


def _guard(source_relative):
    import bpy
    source = _path(source_relative, '.blend')
    if (not bpy.app.background or bpy.context.mode != 'OBJECT' or bpy.data.is_dirty
            or Path(bpy.data.filepath).resolve() != source or _sha(source) != SOURCE_SHA):
        raise RuntimeError('Fresh exact R09 refined 1c0b0210 source required; never use the rejected volume/ellipse source')
    objects = {name: bpy.data.objects[name] for name in NAMES}
    for name, obj in objects.items():
        if obj.library or obj.data.library or obj.data.users != 1:
            raise RuntimeError('Expected local single-user data: '+name)
        if any(m.show_viewport or m.show_render for m in obj.modifiers):
            raise RuntimeError('Unexpected live grooming modifier: '+name)
        if any(key in obj for key in (MARKER, RECORD, BACKUP)):
            raise RuntimeError('Preserve or restore the existing R10 candidate before another run')
    if not objects[SHELL].get('R09_posterior_surface_transport'):
        raise RuntimeError('Current adopted PN posterior root layer is missing')
    if not objects[FORELOCK].get('R09_forelock_flow') or any(
            k in objects[FORELOCK] for k in ('R09_forelock_volume_candidate', 'R09_forelock_ellipse_candidate')):
        raise RuntimeError('Expected the retained small-flow forelocks')
    if objects[SHELL].data.shape_keys:
        raise RuntimeError('Root shell unexpectedly has shape keys')
    return source, objects


def _protected(bpy, groom, uv, excluded=None):
    bpy.context.view_layer.update()
    rows = groom['_protected'](bpy, uv)
    for name in NAMES if excluded is None else excluded:
        rows.pop(name, None)
    oral = ('R02 Upper teeth', 'R02 Lower teeth', 'R02 Tongue', 'R03 Upper gingiva', 'R03 Lower gingiva')
    for name in oral+('R01 Sclera L', 'R01 Sclera R', 'R01 Textured iris L', 'R01 Textured iris R'):
        obj = bpy.data.objects[name]; keys = getattr(obj.data, 'shape_keys', None)
        rows[name] = {'data': groom['_data_hash'](obj.data),
            'keys': {k.name: groom['_array_hash'](groom['_values'](k.data, 'co')) for k in keys.key_blocks} if keys else None,
            'keyAnimation': uv['_animation'](keys) if keys else None,
            'attachment': _api('root_surface_transport_r09')['_attachment'](obj, uv),
            'animation': uv['_animation'](obj),
            'materials': [(s.material.name, s.material.as_pointer()) if s.material else None for s in obj.material_slots]}
    return rows


def observe(source_relative=SOURCE, output_relative=OBSERVATION):
    """Export current source arrays only; no reuse of pre-PN observations."""
    import bpy
    source, objects = _guard(source_relative)
    report_path = _path(output_relative, '.json'); arrays_path = report_path.with_suffix('.npz')
    if report_path.exists() or arrays_path.exists():
        raise RuntimeError('Retain prior measurements; choose a new output path')
    groom = _api('posterior_groom_r09'); diag = _api('root_surface_diagnostic_r09')
    uv = _api('production_uv_r08'); transport = _api('root_surface_transport_r09')
    render_surface = _api('forelock_volume_candidate_r09')['_render_surface']
    rig = bpy.data.objects['Character_Rig']; head = bpy.data.objects['R01 Head']
    pose = rig.data.pose_position; protected = _protected(bpy, groom, uv)
    source_hashes = {name: groom['_data_hash'](obj.data) for name, obj in objects.items()}
    arrays = {}; rows = {}
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        root, triangles, normals, polygons, root_summary = diag['_mesh_surface'](objects[SHELL])
        boundary = diag['_boundary'](polygons)
        arrays.update(root_world=root, root_triangles=triangles, root_boundary=boundary)
        for index, (name, obj) in enumerate(objects.items()):
            prefix = 'object_'+str(index)
            raw = groom['_values'](obj.data.vertices, 'co') if name == SHELL else groom['_values'](obj.data.position_data, 'vector')
            world = groom['_world'](raw, obj.matrix_world)
            arrays[prefix+'_local'] = raw; arrays[prefix+'_world'] = world
            arrays[prefix+'_matrix'] = np.asarray(obj.matrix_world, dtype=float)
            if name != SHELL:
                arrays[prefix+'_starts'] = np.asarray([c.first_point_index for c in obj.data.curves], np.int32)
                arrays[prefix+'_lengths'] = np.asarray([c.points_length for c in obj.data.curves], np.int32)
                arrays[prefix+'_radii'] = groom['_values'](obj.data.attributes['radius'].data, 'value', 1).ravel()
            rows[name] = {'prefix': prefix, 'dataName': obj.data.name, 'dataSha256': source_hashes[name],
                'pointCount': len(raw), 'curveCount': len(obj.data.curves) if name != SHELL else None,
                'worldBounds': [world.min(0).tolist(), world.max(0).tolist()],
                'nonPosition': transport['_nonposition'](obj.data, groom),
                'attachment': transport['_attachment'](obj, uv),
                'runtimeGroups': obj.get('RT09_runtime_group_counts'),
                'runtimeProvenance': obj.get('RT09_runtime_group_provenance')}
        tail_pins = []
        for name in TAILS:
            obj = bpy.data.objects[name]
            sizes = [c.points_length for c in obj.data.curves]
            if len(sizes) != 2480 or set(sizes) != {35}:
                raise RuntimeError('Unexpected original tail topology')
            world = groom['_world'](groom['_values'](obj.data.position_data, 'vector'), obj.matrix_world).reshape(2480, 35, 3)
            tail_pins.append(world[:, :3].reshape(-1, 3))
        arrays['tail_pins_world'] = np.concatenate(tail_pins)
        eyes = []
        for name in ('R01 Sclera L', 'R01 Sclera R'):
            obj = bpy.data.objects[name]
            world = groom['_world'](groom['_values'](obj.data.vertices, 'co'), obj.matrix_world)
            eyes.append((world.min(0)+world.max(0))*.5)
        arrays['eye_centers_world'] = np.asarray(eyes)
        with render_surface(head) as settings:
            surface = diag['_mesh_surface'](head, bpy.context.evaluated_depsgraph_get())
            arrays['render_head_world'], arrays['render_head_triangles'], arrays['render_head_normals'] = surface[:3]
            head_summary = surface[-1]
        report = {'source': str(source.relative_to(project_path())), 'sourceSha256': SOURCE_SHA,
            'helperSha256AtObservation': _sha(Path(__file__)), 'objects': rows,
            'root': root_summary, 'renderHead': head_summary, 'renderSubdivisionSettings': settings,
            'rootBoundaryCount': len(boundary), 'tailPinCount': len(arrays['tail_pins_world']),
            'eyeCentersWorld': arrays['eye_centers_world'].tolist(),
            'sourceDataHashes': source_hashes, 'savedBlend': False, 'geometryEdited': False,
            'scope': 'Actual current R09 refined 1c source in temporary REST; actual head render subdivision levels. No old PN hypothesis or old forelock data is treated as current.'}
    finally:
        rig.data.pose_position = pose; bpy.context.view_layer.update()
    if _protected(bpy, groom, uv) != protected or any(groom['_data_hash'](objects[n].data) != h for n, h in source_hashes.items()):
        raise RuntimeError('Observation failed to restore the source state')
    if _sha(source) != SOURCE_SHA:
        raise RuntimeError('Saved source changed during observation')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    report['arrays'] = {'path': str(arrays_path.relative_to(project_path())), 'sha256': _sha(arrays_path)}
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R10_CROWN_OBSERVED '+json.dumps({'report': str(report_path), 'arraysSha256': report['arrays']['sha256']}), flush=True)
    return report


def _smooth(value):
    """Quintic transition, with zero first/second derivatives at its endpoints."""
    t = np.clip(value, 0., 1.)
    return t*t*t*(10.-15.*t+6.*t*t)


def _load_observation(relative):
    path = _path(relative, '.json')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['sourceSha256'] != SOURCE_SHA or not set(NAMES).issubset(record['objects']):
        raise RuntimeError('Observation does not contain the current R09 core crown objects')
    payload = _path(record['arrays']['path'], '.npz')
    if _sha(payload) != record['arrays']['sha256']:
        raise RuntimeError('Current-source observation arrays changed')
    with np.load(payload, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    return path, record, arrays


def _is_head_attached(obj, rig):
    current = obj
    while current.parent is not None:
        if current.parent == rig:
            return current.parent_type == 'BONE' and current.parent_bone == 'head'
        current = current.parent
    return False


def observe_v2(source_relative=SOURCE, base_relative=OBSERVATION, output_relative=OBSERVATION_V2):
    """Add an actual visible, head-attached CURVES inventory; preserve v1.

    The existing field/anchors are unchanged. Every additional static object
    with any point inside that field is recorded for the same transport. No
    geometry is changed here, including the originally missed cheek wisps.
    """
    import bpy
    source, core = _guard(source_relative)
    base_path, base, arrays = _load_observation(base_relative)
    if set(base['objects']) != set(NAMES) or base.get('observationVersion', 1) != 1:
        raise RuntimeError('v2 supplements the preserved original five-object observation only')
    output = _path(output_relative, '.json'); payload = output.with_suffix('.npz')
    if output.exists() or payload.exists() or output == base_path:
        raise RuntimeError('Use new v2 observation names; preserve the original eeb303 payload')
    groom = _api('posterior_groom_r09'); uv = _api('production_uv_r08')
    transport = _api('root_surface_transport_r09'); rig = bpy.data.objects['Character_Rig']
    protected = _protected(bpy, groom, uv); pose = rig.data.pose_position
    record = json.loads(json.dumps(base)); inventory = []
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        for name, obj in core.items():
            if groom['_data_hash'](obj.data) != base['sourceDataHashes'][name]:
                raise RuntimeError('Current core differs from preserved v1 observation')
            prefix = base['objects'][name]['prefix']
            raw = groom['_values'](obj.data.vertices, 'co') if name == SHELL else groom['_values'](obj.data.position_data, 'vector')
            if np.max(np.abs(groom['_world'](raw, obj.matrix_world)-arrays[prefix+'_world'])) > 2e-7:
                raise RuntimeError('Current core REST placement differs from v1')
        setup, tree, _ = _field_setup(base, arrays)
        visible = [o for o in bpy.context.scene.objects if o.type == 'CURVES' and not o.hide_render
                   and _is_head_attached(o, rig)]
        for index, obj in enumerate(sorted(visible, key=lambda o: o.name)):
            name = obj.name
            if name in NAMES or name in TAILS:
                inventory.append({'object': name, 'status': 'existing common-field core' if name in NAMES else 'protected native tail',
                                  'selectedAdditionalTransport': False})
                continue
            prefix = 'inventory_'+str(index)
            raw = groom['_values'](obj.data.position_data, 'vector')
            world = groom['_world'](raw, obj.matrix_world)
            starts = np.asarray([c.first_point_index for c in obj.data.curves], np.int32)
            lengths = np.asarray([c.points_length for c in obj.data.curves], np.int32)
            radii = groom['_values'](obj.data.attributes['radius'].data, 'value', 1).ravel()
            delta = _field(world, setup, tree); amount = np.linalg.norm(delta, axis=1)
            selected = bool(np.any(amount > 1e-8))
            modifiers = [m.name for m in obj.modifiers if m.show_viewport or m.show_render]
            ad = obj.animation_data
            own_animation = bool(ad and (ad.action or any(not t.mute for t in ad.nla_tracks) or len(ad.drivers)))
            if selected and (modifiers or own_animation or obj.library or obj.data.library or obj.data.users != 1):
                raise RuntimeError('Additional affected hair is not static local source data; explicit integration required: '+name)
            arrays.update({prefix+'_local': raw, prefix+'_world': world,
                prefix+'_matrix': np.asarray(obj.matrix_world, dtype=float), prefix+'_starts': starts,
                prefix+'_lengths': lengths, prefix+'_radii': radii})
            ends = starts+lengths-1
            row = {'object': name, 'prefix': prefix, 'dataName': obj.data.name,
                'dataSha256': groom['_data_hash'](obj.data), 'pointCount': len(raw), 'curveCount': len(starts),
                'worldBounds': [world.min(0).tolist(), world.max(0).tolist()],
                'nonPosition': transport['_nonposition'](obj.data, groom),
                'attachment': transport['_attachment'](obj, uv),
                'runtimeGroups': obj.get('RT09_runtime_group_counts'),
                'runtimeProvenance': obj.get('RT09_runtime_group_provenance'),
                'selectedAdditionalTransport': selected, 'movingPoints': int((amount > 1e-8).sum()),
                'movingCurveRoots': int((amount[starts] > 1e-8).sum()),
                'movingCurveTips': int((amount[ends] > 1e-8).sum()),
                'maximumProposedDisplacementMeters': float(amount.max()),
                'tipMaximumProposedDisplacementMeters': float(amount[ends].max()),
                'visibleRender': not obj.hide_render, 'hiddenViewport': obj.hide_viewport,
                'activeModifiers': modifiers, 'ownAnimation': own_animation,
                'status': 'same existing field transport required' if selected else 'common field is zero; remains protected'}
            inventory.append(row)
            if selected:
                record['objects'][name] = row
                record['sourceDataHashes'][name] = row['dataSha256']
        if 'R03 curved cheek wisps' not in record['objects']:
            raise RuntimeError('The actually rejected exposed cheek-wisp roots were not captured for common-field transport')
        record.update(observationVersion=2, baseObservation=str(base_path.relative_to(project_path())),
            baseObservationSha256=_sha(base_path), baseArraysSha256=base['arrays']['sha256'],
            helperSha256AtObservation=_sha(Path(__file__)),
            additionalVisibleHeadCurveInventory=inventory,
            fieldAnchorSourceObjects=list(NAMES), fieldUnchangedFromV1=True,
            additionalTransportObjects=[n for n in record['objects'] if n not in NAMES],
            scope='Actual refined 1c visible head-attached native CURVES inventory appended to preserved v1 arrays. Additional static fibres use the IDENTICAL existing field; no reduced lift, new anchors or relaxed burial threshold.')
    finally:
        rig.data.pose_position = pose; bpy.context.view_layer.update()
    if _protected(bpy, groom, uv) != protected or _sha(source) != SOURCE_SHA:
        raise RuntimeError('Supplemental observation changed the source state')
    for name, obj in core.items():
        if groom['_data_hash'](obj.data) != base['sourceDataHashes'][name]:
            raise RuntimeError('Core source changed during supplemental observation')
    with payload.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    record['arrays'] = {'path': str(payload.relative_to(project_path())), 'sha256': _sha(payload)}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    print('R10_CROWN_V2_OBSERVED '+json.dumps({'report': str(output), 'arraysSha256': record['arrays']['sha256'],
          'additionalTransportObjects': record['additionalTransportObjects']}), flush=True)
    return record


def _fixed_points(record, arrays):
    """These physical points constrain ONE field, rather than separate layers."""
    fixed = {SHELL: np.asarray(arrays['root_boundary'], np.int32)}
    clouds = [arrays['root_world'][fixed[SHELL]], arrays['tail_pins_world']]
    for name in (n for n in record['objects'] if n != SHELL):
        prefix = record['objects'][name]['prefix']; starts = arrays[prefix+'_starts']; lengths = arrays[prefix+'_lengths']
        indices = []
        if name == FORELOCK or name == 'R03 curved cheek wisps':
            # Preserve the existing lower 20% / eye-framing terminal contour.
            for start, count in zip(starts, lengths):
                indices.extend(range(int(start)+int(np.ceil(.8*(int(count)-1))), int(start+count)))
        elif name in GATHERS:
            for start, count in zip(starts, lengths):
                indices.extend(range(int(start+count)-4, int(start+count)))
        fixed[name] = np.asarray(indices, np.int32)
        # The added cheek-wisp lower contour must ALREADY be stationary in the
        # original field. Do not add its points as anchors and reduce the cage.
        if len(indices) and name in NAMES:
            clouds.append(arrays[prefix+'_world'][fixed[name]])
    return fixed, np.concatenate(clouds)


def _field_setup(record, arrays, tree_factory=None):
    eye = arrays['eye_centers_world'].mean(0)
    head = arrays['render_head_world']
    upper = head[head[:, 2] > eye[2]+.03]
    if not len(upper):
        raise RuntimeError('Render-level head does not contain the expected crown')
    low, high = upper.min(0), upper.max(0)
    center = np.array([(low[0]+high[0])*.5, (low[1]+high[1])*.5, eye[2]+.035])
    radius = (high-low)*.5
    if not (.06 < radius[0] < .10 and .075 < radius[1] < .13 and .09 < high[2]-eye[2] < .16):
        raise RuntimeError('Actual head bounds differ from the measured R09 character; do not scale a different source')
    fixed, anchors = _fixed_points(record, arrays)
    if tree_factory is None:
        from mathutils import Vector
        from mathutils.kdtree import KDTree
        tree = KDTree(len(anchors))
        for index, point in enumerate(anchors):
            tree.insert(Vector(point), index)
        tree.balance()
    else:
        # An offline numerical caller may supply an exact bounded-distance
        # query implementation. Production Blender uses its native KDTree.
        tree = tree_factory(anchors)
    setup = {'centerWorld': center.tolist(), 'eyeCenterWorld': eye.tolist(),
             'headRenderCrownTopZ': float(high[2]), 'headRenderUpperBounds': [low.tolist(), high.tolist()],
             'xRadius': float(radius[0]), 'yRadius': float(radius[1]),
             'anchorPointCount': len(anchors), 'parameters': dict(PARAMETERS)}
    return setup, tree, fixed


def _field(points, setup, tree):
    """Continuous spatial map shared by the shell and EVERY selected fibre.

    No per-cohort rank, layer flattening, nearest-surface projection, resampling
    or per-object normal. The same world point always receives the same delta.
    The lower/edge anchor neighbourhoods are exactly stationary.
    """
    points = np.asarray(points, dtype=float); delta = np.zeros_like(points)
    p = setup['parameters']; eye_z = setup['eyeCenterWorld'][2]
    active = np.flatnonzero(points[:, 2] > eye_z+p['heightZeroAboveEyeMeters'])
    if not len(active):
        return delta
    q = points[active]; center = np.asarray(setup['centerWorld'])
    if hasattr(tree, 'distances'):
        distance = tree.distances(q)
    else:
        from mathutils import Vector
        distance = np.fromiter((tree.find(Vector(point))[2] for point in q), dtype=float, count=len(q))
    edge = _smooth((distance-p['anchorFixedRadiusMeters'])/
                   (p['anchorFullDistanceMeters']-p['anchorFixedRadiusMeters']))
    height = _smooth((q[:, 2]-eye_z-p['heightZeroAboveEyeMeters'])/
                     (p['heightFullAboveEyeMeters']-p['heightZeroAboveEyeMeters']))
    upright = _smooth((q[:, 2]-eye_z-p['verticalLiftStartsAboveEyeMeters'])/
                      (setup['headRenderCrownTopZ']-eye_z-p['verticalLiftStartsAboveEyeMeters']))
    move = np.empty_like(q)
    move[:, 0] = p['sideExpansionMeters']*np.tanh((q[:, 0]-center[0])/(.55*setup['xRadius']))*height
    move[:, 1] = p['depthExpansionMeters']*np.tanh((q[:, 1]-center[1])/(.65*setup['yRadius']))*height
    move[:, 2] = p['topLiftMeters']*upright
    delta[active] = move*edge[:, None]
    if np.linalg.norm(delta, axis=1).max() > p['maximumVectorDisplacementMeters']+1e-9:
        raise RuntimeError('Common crown field exceeded the bounded vector displacement')
    return delta


def _curve_segments(old, new, starts, lengths):
    valid = np.ones(len(old)-1, bool)
    valid[np.asarray(starts[1:], np.int64)-1] = False
    before = np.linalg.norm(np.diff(old, axis=0), axis=1)[valid]
    after = np.linalg.norm(np.diff(new, axis=0), axis=1)[valid]
    meaningful = before > .0001
    ratio = after[meaningful]/before[meaningful]
    extra = after-before
    result = {'maximumStepBeforeMeters': float(before.max()), 'maximumStepAfterMeters': float(after.max()),
              'maximumExtraStepMeters': float(extra.max()), 'minimumLengthRatio': float(ratio.min()),
              'maximumLengthRatio': float(ratio.max())}
    if result['maximumExtraStepMeters'] > .005 or ratio.min() < .30 or ratio.max() > 2.5:
        raise RuntimeError('Continuous field produced excessive local strand compression/stretch: '+json.dumps(result))
    return result


def _jacobian_probe(points, setup, tree):
    points = np.asarray(points, dtype=float)
    points = points[np.linalg.norm(_field(points, setup, tree), axis=1) > .0001]
    if len(points) > 6000:
        points = points[np.linspace(0, len(points)-1, 6000, dtype=np.int64)]
    if not len(points):
        raise RuntimeError('Crown proposal unexpectedly has no meaningful movement')
    epsilon = .00015; jacobian = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
    for axis in range(3):
        step = np.zeros(3); step[axis] = epsilon
        jacobian[:, :, axis] += (_field(points+step, setup, tree)-_field(points-step, setup, tree))/(2*epsilon)
    determinant = np.linalg.det(jacobian); singular = np.linalg.svd(jacobian, compute_uv=False)
    result = {'sampleCount': len(points), 'finiteDifferenceMeters': epsilon,
              'minimumDeterminant': float(determinant.min()), 'maximumDeterminant': float(determinant.max()),
              'minimumSingularValue': float(singular.min()), 'maximumSingularValue': float(singular.max())}
    if determinant.min() < .20 or singular.min() < .25 or singular.max() > 2.75:
        raise RuntimeError('Shared cage field folds or severely distorts its volume: '+json.dumps(result))
    return result


def propose(record, arrays, tree_factory=None):
    """Numerical proposal on source-hashed CURRENT arrays, not on an old groom."""
    setup, tree, fixed = _field_setup(record, arrays, tree_factory=tree_factory)
    local = {}; world = {}; rows = {}
    for name in record['objects']:
        prefix = record['objects'][name]['prefix']; old = arrays[prefix+'_world'].astype(float)
        raw = arrays[prefix+'_local']; matrix = arrays[prefix+'_matrix']; inverse = np.linalg.inv(matrix)
        delta = _field(old, setup, tree)
        if len(fixed[name]) and np.max(np.abs(delta[fixed[name]])) > 1e-12:
            raise RuntimeError('A protected point did not receive exactly zero common-field motion: '+name)
        target = old+delta
        output = (target@inverse[:3, :3].T+inverse[:3, 3]).astype(np.float32)
        unchanged = np.all(delta == 0, axis=1)
        output[unchanged] = raw[unchanged]
        actual = output.astype(float)@matrix[:3, :3].T+matrix[:3, 3]
        rounding = float(np.max(np.abs(actual-target)))
        if rounding > 3e-7 or not np.array_equal(output[fixed[name]], raw[fixed[name]]):
            raise RuntimeError('Float32 output lost protected coordinates or exceeded rounding tolerance')
        local[name] = output; world[name] = actual
        rows[name] = {'changedPoints': int((np.linalg.norm(actual-old, axis=1) > 1e-8).sum()),
            'pointCount': len(old), 'maximumDisplacementMeters': float(np.linalg.norm(actual-old, axis=1).max()),
            'componentDisplacementRangesMeters': [np.min(actual-old, axis=0).tolist(), np.max(actual-old, axis=0).tolist()],
            'beforeWorldBounds': [old.min(0).tolist(), old.max(0).tolist()],
            'afterWorldBounds': [actual.min(0).tolist(), actual.max(0).tolist()],
            'fixedPointCount': len(fixed[name]), 'fixedPointCoordinatesExact': True,
            'floatRoundingMaximumMeters': rounding}
        if name != SHELL:
            rows[name]['segmentContinuity'] = _curve_segments(old, actual, arrays[prefix+'_starts'], arrays[prefix+'_lengths'])
        print('R10_CROWN_PROPOSED '+name+' '+json.dumps(rows[name]), flush=True)
    old = arrays['root_world']; new = world[SHELL]; triangles = arrays['root_triangles']
    before = np.cross(old[triangles[:, 1]]-old[triangles[:, 0]], old[triangles[:, 2]]-old[triangles[:, 0]])
    after = np.cross(new[triangles[:, 1]]-new[triangles[:, 0]], new[triangles[:, 2]]-new[triangles[:, 0]])
    area_before = np.linalg.norm(before, axis=1); area_after = np.linalg.norm(after, axis=1)
    valid = area_before > 1e-14
    orientation = np.sum(before[valid]*after[valid], axis=1)/(area_before[valid]*area_after[valid])
    if np.any(area_after[valid] <= 1e-14) or np.any(orientation <= .25):
        raise RuntimeError('Crown shell has degenerate/flipped or excessively rotated triangles')
    report = {'field': setup, 'objects': rows, 'jacobian': _jacobian_probe(old, setup, tree),
              'rootMinimumNormalDot': float(orientation.min()),
              'rootMinimumAreaRatio': float((area_after[valid]/area_before[valid]).min()),
              'rootBoundaryExact': True, 'forelockLowerTwentyPercentExact': True,
              'gatherFinalFourSamplesExact': True, 'tailPinsOutsideEditScope': True}
    return local, world, report, setup, tree


def _sample_indices(record, arrays, name, maximum=16000):
    prefix = record['objects'][name]['prefix']; count = len(arrays[prefix+'_world'])
    if name == SHELL:
        return np.arange(count, dtype=np.int64)
    starts = arrays[prefix+'_starts']; lengths = arrays[prefix+'_lengths']
    # Each strand root and tip, plus a deterministic sample through its interior.
    return np.unique(np.concatenate((starts, starts+lengths-1,
                                    np.linspace(0, count-1, min(count, maximum), dtype=np.int64))))


def _clearance(old, new, radii, old_tree, new_tree, center, label, opaque_faces=None):
    from mathutils import Vector
    before = []; after = []; eligible = []
    for source, target, radius in zip(old, new, radii):
        row = []; distances = []; opacity = []
        for point, tree in ((source, old_tree), (target, new_tree)):
            hit, normal, index, distance = tree.find_nearest(Vector(point))
            if hit is None:
                raise RuntimeError('Actual surface query failed: '+label)
            h = np.asarray(hit); n = np.asarray(normal)
            if np.dot(n, h-center) < 0:
                n = -n
            row.append(float(np.dot(point-h, n)-radius))
            distances.append(float(distance))
            opacity.append(True if opaque_faces is None else bool(opaque_faces[index]))
        before.append(row[0]); after.append(row[1])
        eligible.append(True if opaque_faces is None else all(opacity) and min(distances) < .012)
    before = np.asarray(before); after = np.asarray(after)
    # Points already embedded in the opaque underlayer are reported honestly.
    # This pass must not bury source fibres that were clearly exposed.
    eligible = np.asarray(eligible, bool)
    new_penetration = eligible & (before >= .00010) & (after < -.00010)
    result = {'sampleCount': len(old), 'beforeMinimumMeters': float(before.min()),
              'afterMinimumMeters': float(after.min()), 'negativeSignedClearanceBefore': int((before < 0).sum()),
              'eligibleOpaqueNearbySamples': int(eligible.sum()),
              'newlyBuriedPreviouslyExposedPoints': int(new_penetration.sum()),
              'classificationToleranceMeters': .00010}
    if new_penetration.any():
        indices = np.flatnonzero(new_penetration)[:8]
        result['examples'] = [{'sample': int(i), 'oldWorld': old[i].tolist(), 'newWorld': new[i].tolist(),
                               'beforeMeters': float(before[i]), 'afterMeters': float(after[i])} for i in indices]
        raise RuntimeError('Common-field proposal buried exposed geometry against '+label+': '+json.dumps(result))
    return result


def _surface_checks(bpy, record, arrays, world, setup):
    groom = _api('posterior_groom_r09'); diag = _api('root_surface_diagnostic_r09')
    old_root = diag['_tree'](arrays['root_world'], arrays['root_triangles'])
    new_root = diag['_tree'](world[SHELL], arrays['root_triangles'])
    head_tree = diag['_tree'](arrays['render_head_world'], arrays['render_head_triangles'])
    alpha = groom['_values'](bpy.data.objects[SHELL].data.attributes['R06_root_edge_alpha'].data, 'value', 1).ravel()
    opaque_faces = alpha[arrays['root_triangles']].min(axis=1) >= .95
    center = np.asarray(setup['centerWorld']); rows = {}
    for name in record['objects']:
        prefix = record['objects'][name]['prefix']; ids = _sample_indices(record, arrays, name)
        old = arrays[prefix+'_world'][ids]; new = world[name][ids]
        radius = np.zeros(len(ids)) if name == SHELL else arrays[prefix+'_radii'][ids]
        rows[name] = {'renderHead': _clearance(old, new, radius, head_tree, head_tree, center, name+' / render-level head')}
        if name != SHELL:
            rows[name]['rootUnderlayer'] = _clearance(old, new, radius, old_root, new_root, center, name+' / shared root underlayer', opaque_faces)
        print('R10_CROWN_SURFACE_CHECK '+name, flush=True)
    other = []
    low = world[SHELL].min(0)-.006; high = world[SHELL].max(0)+.006
    for obj in bpy.context.scene.objects:
        if obj.type != 'CURVES' or obj.name in record['objects'] or obj.name in TAILS or obj.hide_render:
            continue
        if not _is_head_attached(obj, bpy.data.objects['Character_Rig']):
            continue
        points = groom['_world'](groom['_values'](obj.data.position_data, 'vector'), obj.matrix_world)
        ids = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
        if not len(ids):
            continue
        if len(ids) > 16000:
            ids = ids[np.linspace(0, len(ids)-1, 16000, dtype=np.int64)]
        radius = groom['_values'](obj.data.attributes['radius'].data, 'value', 1).ravel()[ids]
        row = _clearance(points[ids], points[ids], radius, old_root, new_root, center, 'unchanged '+obj.name, opaque_faces)
        row['object'] = obj.name; other.append(row)
    return {'selectedObjects': rows, 'otherUnchangedVisibleHeadHair': other,
            'scope': 'Actual render-level head surface and actual old/new root triangles. All root vertices, every selected fibre root/tip and deterministic interior samples; not a full continuous-fibre or collision-solver proof.'}


def run(source_relative=SOURCE, observation_relative=OBSERVATION_V2, diagnostics_only=False):
    """Build one candidate in memory, with rollback on any failed guard."""
    import bpy
    source, objects = _guard(source_relative)
    observation_path, record, arrays = _load_observation(observation_relative)
    if record.get('observationVersion', 1) < 2 or 'R03 curved cheek wisps' not in record['objects']:
        raise RuntimeError('Use the new actual v2 visible-head-hair inventory; original five-object coverage is incomplete')
    objects = {name: bpy.data.objects[name] for name in record['objects']}
    for name, obj in objects.items():
        if obj.library or obj.data.library or obj.data.users != 1 or any(
                m.show_viewport or m.show_render for m in obj.modifiers):
            raise RuntimeError('Observed transport object is not unchanged static local data: '+name)
    groom = _api('posterior_groom_r09'); uv = _api('production_uv_r08')
    transport = _api('root_surface_transport_r09')
    for name, obj in objects.items():
        if groom['_data_hash'](obj.data) != record['sourceDataHashes'][name]:
            raise RuntimeError('Current source data differs from the actual R10 observation: '+name)
    original = {name: obj.data for name, obj in objects.items()}
    fake = {name: data.use_fake_user for name, data in original.items()}
    contracts = {name: transport['_nonposition'](obj.data, groom) for name, obj in objects.items()}
    attachments = {name: transport['_attachment'](obj, uv) for name, obj in objects.items()}
    protected = _protected(bpy, groom, uv, objects); rig = bpy.data.objects['Character_Rig']; pose = rig.data.pose_position
    prepared = {}; committed = False
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        for name, obj in objects.items():
            prefix = record['objects'][name]['prefix']
            raw = groom['_values'](obj.data.vertices, 'co') if name == SHELL else groom['_values'](obj.data.position_data, 'vector')
            if not np.array_equal(raw, arrays[prefix+'_local']):
                raise RuntimeError('Current raw data differs from observed actual positions')
            if np.max(np.abs(groom['_world'](raw, obj.matrix_world)-arrays[prefix+'_world'])) > 2e-7:
                raise RuntimeError('Current REST attachment differs from observation')
        local, world, report, setup, tree = propose(record, arrays)
        report['surfaces'] = _surface_checks(bpy, record, arrays, world, setup)
        for name, obj in objects.items():
            candidate = obj.data.copy(); candidate.name = 'R10 unified crown / '+obj.data.name
            prepared[name] = candidate
            if name == SHELL:
                candidate.vertices.foreach_set('co', local[name].ravel()); candidate.update()
            else:
                candidate.position_data.foreach_set('vector', local[name].ravel()); candidate.update_tag()
            if transport['_nonposition'](candidate, groom) != contracts[name]:
                raise RuntimeError('Crown edit changed non-position attributes/topology/materials: '+name)
        rig.data.pose_position = pose; bpy.context.view_layer.update()
        if _protected(bpy, groom, uv, objects) != protected:
            raise RuntimeError('Protected face, oral, tail, rig or cache data changed')
        if any(groom['_data_hash'](original[n]) != record['sourceDataHashes'][n] for n in objects):
            raise RuntimeError('Retained original geometry changed')
        report.update(source=str(source.relative_to(project_path())), sourceSha256=SOURCE_SHA,
            observation=str(observation_path.relative_to(project_path())), observationSha256=_sha(observation_path),
            arraysSha256=record['arrays']['sha256'], sourceDataHashes=record['sourceDataHashes'],
            candidateDataHashes={name: groom['_data_hash'](data) for name, data in prepared.items()},
            originalData={name: data.name for name, data in original.items()},
            curveOrderCountRadiusMaterialsUvAlphaUnchanged=True, headAndRigAndTailCacheUnchanged=True,
            runtimeGroupsUnchanged=True, runtimeStaticHairRefreshRequired=True,
            additionalTransportObjects=record.get('additionalTransportObjects', []),
            fieldUnchangedFromFiveObjectAttempt=True,
            savedBlend=False, writtenInMemory=not diagnostics_only, appearanceAccepted=False,
            method='One shared anchored spatial volume map on actual current PN root/scalp/forelock/gather. No per-cohort flattening, new fibres, new materials or independent-layer projection.',
            limits=['Work candidate only. Numerical continuity/clearance does not establish visual improvement.',
                    'Compare actual front, quarter and side against R09 small-flow under identical light before adoption.',
                    'The deliberately fixed forehead/ear boundary and lower forelock contour limit the expansion.',
                    'Original double-tail rest, three-point pinning and existing tail cache are unchanged; long/hard tail silhouette remains a separate visible limitation.',
                    'If adopted, root/scalp/fringe/gather runtime geometry and projected atlases require explicit refresh; old R09 payloads are not corresponding output.',
                    'Restore R10 before restoring any underlying R09 posterior or forelock layer.'])
        if _sha(source) != SOURCE_SHA:
            raise RuntimeError('Saved source changed while preparing the candidate')
        if diagnostics_only:
            return report
        restore_record = {'sourceSha256': SOURCE_SHA, 'observationSha256': report['observationSha256'],
            'sourceDataHashes': record['sourceDataHashes'], 'candidateDataHashes': report['candidateDataHashes'],
            'objects': {name: {'originalData': original[name].name, 'fakeUser': fake[name]} for name in objects}}
        for name, obj in objects.items():
            original[name].use_fake_user = True; obj.data = prepared[name]
            obj[MARKER] = SOURCE_SHA; obj[BACKUP] = original[name].name
            obj.data.update_tag(); obj.update_tag()
        objects[SHELL][RECORD] = json.dumps(restore_record, ensure_ascii=False)
        bpy.context.view_layer.update()
        if _protected(bpy, groom, uv, objects) != protected:
            raise RuntimeError('Candidate assignment changed protected source data')
        if any(transport['_attachment'](obj, uv) != attachments[name] for name, obj in objects.items()):
            raise RuntimeError('Candidate assignment changed original attachment transforms')
        committed = True
        return report
    finally:
        rig.data.pose_position = pose
        if not committed:
            for name, obj in objects.items():
                if obj.data != original[name]:
                    obj.data = original[name]
                original[name].use_fake_user = fake[name]
                for key in (MARKER, RECORD, BACKUP):
                    if key in obj:
                        del obj[key]
            for name, data in prepared.items():
                if data.users == 0:
                    (bpy.data.meshes if name == SHELL else bpy.data.hair_curves).remove(data)
        bpy.context.view_layer.update()


def apply_and_save(source_relative=SOURCE, observation_relative=OBSERVATION_V2,
                   output_relative='work/reconstruction-r02/reference-crown-volume-r10.blend',
                   report_relative='work/reconstruction-r02/crown-volume-r10.json'):
    import bpy
    output = _path(output_relative, '.blend'); report_path = _path(report_relative, '.json')
    if output.exists() or report_path.exists() or output == _path(source_relative, '.blend'):
        raise RuntimeError('Explicit NEW candidate and report paths required')
    report = run(source_relative, observation_relative)
    if output.exists():
        raise RuntimeError('Candidate output appeared during preparation; do not overwrite it')
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    report.update(output=str(output.relative_to(project_path())), outputSha256=_sha(output), savedBlend=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R10_CROWN_SAVED '+json.dumps({'output': str(output), 'sha256': report['outputSha256']}), flush=True)
    return report


def restore():
    """Restore only this layer, keeping the rejected candidate data as history."""
    import bpy
    shell = bpy.data.objects[SHELL]
    if RECORD not in shell:
        return False
    groom = _api('posterior_groom_r09'); record = json.loads(shell[RECORD]); old = {}
    for name, row in record['objects'].items():
        data = (bpy.data.meshes if name == SHELL else bpy.data.hair_curves).get(row['originalData'])
        if data is None or groom['_data_hash'](data) != record['sourceDataHashes'][name]:
            raise RuntimeError('Exact original R09 data is missing or changed: '+name)
        if groom['_data_hash'](bpy.data.objects[name].data) != record['candidateDataHashes'][name]:
            raise RuntimeError('Current R10 layer was edited; refuse an unaudited rollback: '+name)
        old[name] = data
    retained = {}
    for name, data in old.items():
        obj = bpy.data.objects[name]; retained[name] = obj.data.name; obj.data.use_fake_user = True
        obj.data = data; data.use_fake_user = record['objects'][name]['fakeUser']
        for key in (MARKER, RECORD, BACKUP):
            if key in obj:
                del obj[key]
        obj.data.update_tag(); obj.update_tag()
    bpy.context.view_layer.update()
    return {'restoredExactR09Data': list(old), 'retainedCandidateData': retained, 'savedBlend': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('observe', 'observe-v2', 'prepare', 'diagnose'), required=True)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--observation')
    parser.add_argument('--base-observation', default=OBSERVATION)
    parser.add_argument('--output', default='work/reconstruction-r02/reference-crown-volume-r10.blend')
    parser.add_argument('--report', default='work/reconstruction-r02/crown-volume-r10.json')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if args.mode == 'observe':
        observe(args.source, args.observation or OBSERVATION)
    elif args.mode == 'observe-v2':
        observe_v2(args.source, args.base_observation, args.observation or OBSERVATION_V2)
    elif args.mode == 'prepare':
        apply_and_save(args.source, args.observation or OBSERVATION_V2, args.output, args.report)
    else:
        print(json.dumps(run(args.source, args.observation or OBSERVATION_V2, diagnostics_only=True), ensure_ascii=False, indent=2))
