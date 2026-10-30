"""One substantive, reversible anterior-volume candidate on existing 7x480 hair.

Explicit run() in the current saved independent R09 source; no automatic
execution, save, rendering or new fibres. Actual paths define seven main bows;
roots stagger on the actual render-level head, broad cross-sections replace the
old eight depth rails, and the last 20% of every original path is exact. Tail,
posterior scalp, rig, face, radii, materials and runtime group order are untouched.
Restore this layer BEFORE restoring the preceding forelock_flow_r09 layer.
"""
import contextlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SOURCE = 'work/reconstruction-r02/reference-forelock-flow-r09.blend'
OBJECT = 'R03 Forelocks from crown'
MARKER = 'R09_forelock_volume_candidate'
BACKUP = 'R09_forelock_volume_original_data'
META = 'R09_forelock_volume_restore_record'
ROOT_SIDE = np.array([-.0025, .0004, -.0017, .0013, -.0006, .0025, .0005])
ROOT_ADVANCE = np.array([.010, .003, .007, -.004, .008, .001, .006])
LIFT = np.array([.0065, .010, .008, .011, .0075, .0095, .0065])
SWEEP = np.array([-.003, -.0055, -.002, .003, .0055, .003, .004])
WIDTH = np.array([1.15, 1.28, 1.18, 1.30, 1.12, 1.24, 1.14])
ROOT_WIDTH = np.array([1.10, .94, 1.12, .98, 1.10, 1.00, 1.12])


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def _unit(a):
    return a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def _sample(path, t):
    u = np.clip(t, 0., 1.)*(len(path)-1)
    i = np.minimum(u.astype(np.int32), len(path)-2); f = u-i
    return path[i]*(1.-f[:, None])+path[i+1]*f[:, None]


def _frame(core, center):
    tangent = _unit(np.gradient(core, axis=0))
    across = np.tile([1., 0., 0.], (len(core), 1))-tangent*tangent[:, 0, None]
    if np.linalg.norm(across, axis=1).min() < .12:
        raise RuntimeError('Main lock lateral frame becomes degenerate')
    across = _unit(across); normal = _unit(np.cross(tangent, across))
    normal *= np.where(np.sum(normal*(core-center), axis=1) < 0., -1., 1.)[:, None]
    if np.min(np.sum(normal[1:]*normal[:-1], axis=1)) < .4:
        raise RuntimeError('Main lock outward frame flips; inspect source guide')
    return tangent, across, normal


def propose(points, starts, sizes, center):
    """Pure numerical main-shape proposal; surface attachment follows separately."""
    if len(starts) != 3360 or sizes.min() < 40 or sizes.max() > 60:
        raise RuntimeError('Expected current seven consecutive 480-strand forelocks')
    points = np.asarray(points, dtype=float); result = points.copy()
    fixed = np.ones(len(points), dtype=bool)
    root_clearance = np.zeros(len(points), dtype=float)
    t_values = np.zeros(len(points), dtype=float)
    grid = np.linspace(0., 1., 81); rows = []
    # One broad bow, C1 zero at both ends. This changes the primary volume,
    # instead of layering small independent oscillations onto every fibre.
    u = np.clip(grid/.80, 0., 1.); bow = 16.*u*u*(1.-u)*(1.-u)
    root_fade = 1.-_smooth(grid/.46)
    cross_fade = 1.-_smooth((grid-.50)/.30)
    for group in range(7):
        ids = list(range(group*480, (group+1)*480))
        sampled = np.stack([_sample(points[starts[i]:starts[i]+sizes[i]], grid) for i in ids])
        core = sampled.mean(axis=0)
        tangent, across, normal = _frame(core, center)
        # Move the actual source roots along their local scalp direction.
        # Individual root projection later seats them on the evaluated surface.
        root_delta = tangent[0]*ROOT_ADVANCE[group]+across[0]*ROOT_SIDE[group]
        altered_core = (core+root_delta[None, :]*root_fade[:, None]
                        +across*(SWEEP[group]*bow)[:, None]
                        +normal*(LIFT[group]*bow)[:, None])
        _, new_across, new_normal = _frame(altered_core, center)
        relative = sampled-core[None, :, :]
        x = np.sum(relative*across[None, :, :], axis=2)
        width = np.quantile(np.abs(x), .975, axis=0)
        width = np.maximum(width, .0002)
        # Coherent source root rank is retained along each path. A deterministic
        # fine-depth distribution removes eight discrete depth plateaus without
        # subdividing the seven main locks into more visible ribbons.
        order = np.argsort(x[:, 0], kind='stable'); rank = np.empty(480)
        rank[order] = (np.arange(480)+.5)/480.
        across_rank = rank*2.-1.
        depth_rank = (np.arange(480)*.618033988749895+group*.371) % 1.
        width_factor = 1.+(WIDTH[group]-1.)*bow+(ROOT_WIDTH[group]-1.)*root_fade
        for within, i in enumerate(ids):
            start = int(starts[i]); count = int(sizes[i]); t = np.linspace(0., 1., count)
            blend = np.interp(t, grid, cross_fade)
            old = points[start:start+count]
            old_core = _sample(core, t); new_core = _sample(altered_core, t)
            a = _unit(_sample(new_across, t)); n = _unit(_sample(new_normal, t))
            halfwidth = np.interp(t, grid, width*width_factor)
            q = across_rank[within]; d = depth_rank[within]
            # A gently convex major cross-section, not a flat card. Sub-mm
            # fibre depth remains, but no repeatable millimetre-scale rails.
            barrel = .0013*(1.-q*q)*np.interp(t, grid, bow)
            depth = (.00018+.00052*d)+barrel
            transverse = halfwidth*q
            fine_wave = .00015*np.sin((within*.754877666+group)*np.pi*2.+t*4.)
            transverse += fine_wave*np.interp(t, grid, bow)
            new_cross = a*transverse[:, None]+n*depth[:, None]
            old_cross = old-old_core
            new = new_core+old_cross*(1.-blend[:, None])+new_cross*blend[:, None]
            untouched = t >= .80
            new[untouched] = old[untouched]
            result[start:start+count] = new
            fixed[start:start+count] = untouched
            t_values[start:start+count] = t
            root_clearance[start:start+count] = .00012+.00030*d
        rows.append({'group':group, 'strands':480,
            'sourceRootCenter':core[0].tolist(), 'proposedRootCenter':altered_core[0].tolist(),
            'rootSideShiftMm':float(ROOT_SIDE[group]*1000.),
            'rootAlongFlowShiftMm':float(ROOT_ADVANCE[group]*1000.),
            'peakMainBowLiftMm':float(LIFT[group]*1000.), 'peakSweepMm':float(SWEEP[group]*1000.),
            'peakWidthFactor':float(WIDTH[group]), 'rootWidthFactor':float(ROOT_WIDTH[group]),
            'sourceGuideSamples':core[::10].tolist(), 'proposedGuideSamples':altered_core[::10].tolist()})
    return result, fixed, t_values, root_clearance, rows


@contextlib.contextmanager
def _render_surface(head):
    """Use render subdivision for BVH evaluation, then restore preview settings."""
    states = []; report = []
    try:
        for modifier in head.modifiers:
            if modifier.type != 'SUBSURF':
                continue
            states.append((modifier, modifier.levels, modifier.show_viewport))
            report.append({'modifier':modifier.name, 'viewportLevelsBefore':modifier.levels,
                           'renderLevels':modifier.render_levels, 'renderEnabled':modifier.show_render})
            modifier.levels = modifier.render_levels
            modifier.show_viewport = modifier.show_render
        if not states:
            raise RuntimeError('Expected explicit production head subdivision settings')
        head.update_tag()
        import bpy
        bpy.context.view_layer.update()
        yield report
    finally:
        for modifier, levels, visible in states:
            modifier.levels = levels; modifier.show_viewport = visible
        head.update_tag()
        if states:
            import bpy
            bpy.context.view_layer.update()


def _seat_and_check(old, proposed, fixed, t, root_clearance, nearest):
    result = proposed.copy(); root_errors = []; maximum_push = 0.; pushes = 0
    # Root-to-bow transition is continuous. It uses only the new root area, not
    # a camera-direction projection of all forelocks onto the forehead.
    for i in np.flatnonzero(~fixed):
        point = result[i]
        hit, normal, clearance = nearest(point)
        root_weight = 1.-float(_smooth((t[i]-.025)/.14))
        if root_weight > 0.:
            seated = hit+normal*root_clearance[i]
            if np.linalg.norm(seated-point) > .018:
                raise RuntimeError('Root projection exceeds 18 mm; inspect current head/path at point '+str(i))
            point = point*(1.-root_weight)+seated*root_weight
        # Enforce the actual fine render surface. Existing free ends are outside
        # the changed range. Two local corrections are enough; no iterative bake.
        correction = 0.
        for unused in range(2):
            hit, normal, clearance = nearest(point)
            minimum = min(.00020, root_clearance[i])
            if clearance >= minimum-.000015:
                break
            step = minimum-clearance+.000003
            correction += step; point += normal*step
        if clearance < min(.00020, root_clearance[i])-.000015:
            hit, normal, clearance = nearest(point)
        if clearance < min(.00020, root_clearance[i])-.000015 or correction > .004:
            raise RuntimeError('Candidate contact correction escaped its bound: '+json.dumps(
                {'point':int(i), 't':float(t[i]), 'clearanceMm':clearance*1000., 'pushMm':correction*1000.}))
        if correction:
            pushes += 1; maximum_push = max(maximum_push, correction)
        result[i] = point
        if t[i] == 0.:
            root_errors.append(clearance-root_clearance[i])
    result[fixed] = old[fixed]
    return result, {'rootCount':len(root_errors), 'maximumRootClearanceErrorMm':float(np.max(np.abs(root_errors)))*1000.,
                    'extraContactPushPoints':pushes, 'maximumExtraPushMm':maximum_push*1000.,
                    'rootClearanceRangeMm':[float(root_clearance.min()*1000.), float(root_clearance.max()*1000.)]}


def _eyes(bpy, util, old, new, fixed):
    rows = []; active = ~fixed
    for side in ('L', 'R'):
        obj = bpy.data.objects['R01 Sclera '+side]
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = evaluated.to_mesh()
        try:
            points = util['_world'](util['_values'](mesh.vertices, 'co'), evaluated.matrix_world)
        finally:
            evaluated.to_mesh_clear()
        low = points.min(axis=0); high = points.max(axis=0)
        center = (low+high)*.5; radius = float(np.max(high-low)*.5)
        old_gap = np.linalg.norm(old[active]-center, axis=1)-radius
        new_gap = np.linalg.norm(new[active]-center, axis=1)-radius
        if np.any(new_gap < np.minimum(old_gap, .004)-.00002):
            raise RuntimeError('New main forelock intrudes on the actual eye safety envelope: '+side)
        # Preserve a clear frontal aperture as well as non-intersection. Do not
        # let a new hair point fall in front of the visible eye's central disc.
        for cloud, tag in ((old[active], 'old'), (new[active], 'new')):
            visible = (((cloud[:,0]-center[0])/(radius*.95))**2
                       +((cloud[:,2]-center[2])/(radius*.65))**2 < 1.) & (cloud[:,1] < center[1])
            if tag == 'old':
                old_visible = visible
            elif np.any(visible & ~old_visible):
                raise RuntimeError('New frontal eye occlusion from modified forelock: '+side)
        rows.append({'eye':obj.name, 'centerWorld':center.tolist(), 'boundingRadiusMm':radius*1000.,
                     'oldMinimumChangedPointGapMm':float(old_gap.min()*1000.),
                     'newMinimumChangedPointGapMm':float(new_gap.min()*1000.), 'newCentralEyeOcclusionPoints':0})
    return rows


def _geometry_checks(old, new, starts, sizes, fixed):
    if not np.isfinite(new).all() or not np.array_equal(old[fixed], new[fixed]):
        raise RuntimeError('Finite/tip-preservation invariant failed')
    displacement = np.linalg.norm(new-old, axis=1)
    if displacement.max() > .022:
        raise RuntimeError('Main shape correction exceeds 22 mm: '+str(float(displacement.max()*1000.)))
    lengths = []; worst_step = 0.; extra_step = 0.; added_turn = 0.
    for i, (start, count) in enumerate(zip(starts, sizes)):
        a = old[start:start+count]; b = new[start:start+count]
        da = np.diff(a, axis=0); db = np.diff(b, axis=0)
        la = np.linalg.norm(da, axis=1); lb = np.linalg.norm(db, axis=1)
        j = int(np.argmax(lb-la))
        old_turn = np.degrees(np.arccos(np.clip(np.sum(_unit(da)[1:]*_unit(da)[:-1], axis=1), -1, 1)))
        new_turn = np.degrees(np.arccos(np.clip(np.sum(_unit(db)[1:]*_unit(db)[:-1], axis=1), -1, 1)))
        if lb.max() > .012 or (lb-la).max() > .004 or np.any(new_turn > np.maximum(old_turn+40., 70.)):
            raise RuntimeError('Main bow continuity guard: '+json.dumps({'strand':int(i), 'segment':j,
                'oldStepMm':float(la[j]*1000.), 'newStepMm':float(lb[j]*1000.),
                'largestTurnDegrees':float(new_turn.max()), 'oldPoints':a[j:j+2].tolist(), 'newPoints':b[j:j+2].tolist()}))
        ratio = float(lb.sum()/la.sum())
        if not .76 <= ratio <= 1.30:
            raise RuntimeError('Main bow path length escaped 0.76..1.30: '+str((i, ratio)))
        lengths.append(ratio); worst_step = max(worst_step, float(lb.max()))
        extra_step = max(extra_step, float((lb-la).max()))
        added_turn = max(added_turn, float((new_turn-old_turn).max()))
    return {'maximumDisplacementMm':float(displacement.max()*1000.),
            'changedPoints':int((displacement > 1e-9).sum()), 'exactTipPoints':int(fixed.sum()),
            'maximumSegmentMm':worst_step*1000., 'maximumExtraSegmentMm':extra_step*1000.,
            'maximumAdditionalTurnDegrees':added_turn, 'lengthRatioRange':[min(lengths), max(lengths)],
            'oldBounds':[old.min(axis=0).tolist(), old.max(axis=0).tolist()],
            'newBounds':[new.min(axis=0).tolist(), new.max(axis=0).tolist()]}


def run(source_relative=SOURCE, diagnostics_only=False):
    import bpy
    p = project_path(); source = (p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source or bpy.data.is_dirty:
        raise RuntimeError('Exact fresh saved independent source in background Object Mode required')
    obj = bpy.data.objects[OBJECT]; rig = bpy.data.objects['Character_Rig']; head = bpy.data.objects['R01 Head']
    if not obj.get('R09_forelock_flow') or not head.get('R09_lower_face_candidate'):
        raise RuntimeError('Use the accepted current R09 forelock-flow and proportion source')
    if any(key in obj for key in (MARKER, BACKUP, META)):
        raise RuntimeError('Volume candidate already exists; preserve or restore before retrying')
    if obj.type != 'CURVES' or obj.library or obj.data.library or obj.data.users != 1:
        raise RuntimeError('Expected local single-user native forelocks')
    if any(m.show_render or m.show_viewport for m in obj.modifiers) or obj.animation_data or obj.data.animation_data:
        raise RuntimeError('Unexpected forelock animation/modifiers require explicit review')
    flow = _api('forelock_flow_r09'); util = _api('posterior_groom_r09'); uv = _api('production_uv_r08')
    before = flow['_protected'](bpy, util, uv)
    original = obj.data; original_hash = util['_data_hash'](original); source_sha = util['_sha'](source)
    attachment = {'parent':uv['_plain'](obj.parent), 'parentType':obj.parent_type, 'parentBone':obj.parent_bone,
                  'basis':uv['_plain'](obj.matrix_basis), 'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
    restore_record = {'originalHash':original_hash, 'originalFakeUser':original.use_fake_user}
    pose = rig.data.pose_position; prepared = None; committed = False
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        local = util['_values'](original.position_data, 'vector'); matrix = obj.matrix_world.copy()
        world = util['_world'](local, matrix)
        starts = np.array([c.first_point_index for c in original.curves], dtype=np.int32)
        sizes = np.array([c.points_length for c in original.curves], dtype=np.int32)
        with _render_surface(head) as subdivision:
            nearest, top = flow['_surface'](bpy, util)
        center = np.array([0., .003, top-.10])
        proposal, fixed, t, clearance, groups = propose(world, starts, sizes, center)
        candidate, contact = _seat_and_check(world, proposal, fixed, t, clearance, nearest)
        numeric = _geometry_checks(world, candidate, starts, sizes, fixed)
        eyes = _eyes(bpy, util, world, candidate, fixed)
        new_local = util['_world'](candidate, matrix.inverted()).astype(np.float32)
        new_local[fixed] = local[fixed]
        if np.abs(util['_world'](new_local, matrix)-candidate).max() > 2e-6:
            raise RuntimeError('World/local conversion exceeded two micrometres')
        prepared = original.copy(); prepared.name = 'R09 seven bowed forelocks candidate'
        prepared.position_data.foreach_set('vector', new_local.ravel())
        old_attrs = util['_attributes'](original); new_attrs = util['_attributes'](prepared)
        for name, row in old_attrs.items():
            other = new_attrs.get(name)
            if other is None or row['domain'] != other['domain'] or row['type'] != other['type']:
                raise RuntimeError('Unexpected attribute change: '+name)
            if name != 'position' and not np.array_equal(row['values'], other['values']):
                raise RuntimeError('Non-position attribute changed: '+name)
        if not np.array_equal(sizes, [c.points_length for c in prepared.curves]):
            raise RuntimeError('Curve order/point counts changed')
        rig.data.pose_position = pose; bpy.context.view_layer.update()
        if flow['_protected'](bpy, util, uv) != before:
            raise RuntimeError('Protected face/tail/posterior/rig data changed')
        report = {'source':str(source.relative_to(p)), 'sourceSha256':source_sha,
            'object':OBJECT, 'sourceData':original.name, 'sourceDataSha256':original_hash,
            'candidateDataSha256':util['_data_hash'](prepared), 'groups':groups,
            'geometry':numeric, 'contact':contact, 'eyes':eyes, 'headSurface':subdivision,
            'surfaceMethod':'Actual render-level evaluated head triangles and barycentric smooth normals; original viewport subdivision settings restored.',
            'scope':'Rearrange existing seven anterior main locks: root band, crown volume and broad continuous cross-sections; last 20% of each strand exact.',
            'strandsAdded':0, 'radiiMaterialsAndCohortOrderChanged':False,
            'posteriorAndTailsUnchanged':True, 'savedBlend':False, 'writtenInMemory':not diagnostics_only,
            'rollbackOrder':'Restore this volume layer first, then optionally forelock_flow_r09.',
            'backupMetadataKey':BACKUP, 'runtimeConsequence':'If visually adopted, refresh seven fringe cohorts using this layer backup as exact prior source. Tail cache and scalp groups remain unchanged.',
            'requiredEvidence':['One same-light front and quarter candidate pair versus forelock-flow-r09.',
                                'Check broad bows and overlap, crown silhouette, root exposure, eye clearance and absence of new sheet-like plates.'],
            'appearanceAccepted':False}
        if diagnostics_only:
            return report
        original.use_fake_user = True; obj[BACKUP] = original.name; obj[META] = json.dumps(restore_record)
        obj.data = prepared; obj[MARKER] = json.dumps(report)
        obj.data.update_tag(); obj.update_tag(); bpy.context.view_layer.update()
        after_attachment = {'parent':uv['_plain'](obj.parent), 'parentType':obj.parent_type, 'parentBone':obj.parent_bone,
                            'basis':uv['_plain'](obj.matrix_basis), 'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
        if after_attachment != attachment or flow['_protected'](bpy, util, uv) != before:
            raise RuntimeError('Protected data/attachment changed during candidate swap')
        if util['_data_hash'](original) != original_hash or util['_sha'](source) != source_sha:
            raise RuntimeError('Retained source data or saved source changed')
        report['protectedDataUnchanged'] = True; committed = True
        return report
    finally:
        rig.data.pose_position = pose
        if not committed:
            if obj.data != original:
                obj.data = original
            original.use_fake_user = restore_record['originalFakeUser']
            for key in (MARKER, BACKUP, META):
                if key in obj:
                    del obj[key]
            if prepared is not None and prepared.users == 0:
                bpy.data.hair_curves.remove(prepared)
        bpy.context.view_layer.update()


def restore():
    import bpy
    obj = bpy.data.objects[OBJECT]
    if not obj.get(MARKER):
        return False
    util = _api('posterior_groom_r09'); record = json.loads(obj[META])
    original = bpy.data.hair_curves.get(obj.get(BACKUP, ''))
    if original is None or util['_data_hash'](original) != record['originalHash']:
        raise RuntimeError('Retained original forelocks missing or changed; no blind restore')
    candidate = obj.data; candidate.use_fake_user = True
    obj.data = original; original.use_fake_user = record['originalFakeUser']
    for key in (MARKER, BACKUP, META):
        del obj[key]
    bpy.context.view_layer.update()
    return {'restored':original.name, 'retainedCandidate':candidate.name, 'savedBlend':False}


if __name__ == '__main__':
    print('Loaded only; call run() explicitly on the saved current forelock-flow R09 candidate.')
