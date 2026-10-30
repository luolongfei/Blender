"""Bounded, reversible R09 anterior forelock correction; never save or render.

Call run(source_relative=...) in a fresh background production COPY. Only the
existing 7 x 480 native forelocks change. The first three points and t >= .40
remain exact. No strands, radii, materials, runtime groups, posterior fibres,
tail pins, face keys or rig data are added or altered. inspect is numeric only.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

OBJECT = 'R03 Forelocks from crown'
MARKER = 'R09_forelock_flow'
BACKUP = 'R09_forelock_original_data'
META = 'R09_forelock_restore_record'
SOURCE = 'work/reconstruction-r02/reference-lower-face-candidate-r09.blend'
WIDTH = np.array([.92, 1.06, 1.10, .90, 1.05, .94, 1.04])
SHIFT = np.array([-.0012, -.0005, .0014, .0008, -.0010, .0012, -.0013])


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)


def envelope(t):
    """C1 zero boundaries, narrow transition from fixed roots to altered crown."""
    return _smooth((np.asarray(t)-.06)/.08)*(1.-_smooth((np.asarray(t)-.25)/.15))


def _sample(path, t):
    u = np.clip(t, 0., 1.)*(len(path)-1)
    i = np.minimum(u.astype(np.int32), len(path)-2)
    f = u-i
    return path[i]*(1.-f[:, None])+path[i+1]*f[:, None]


def _unit(a):
    return a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def propose(points, starts, sizes, center, strength=1.):
    """Pure NumPy proposal, using actual input paths rather than old guides."""
    if len(starts) != 3360 or len(sizes) != 3360:
        raise RuntimeError('Expected seven consecutive existing 480-fibre cohorts')
    if not 0. <= strength <= 1.:
        raise ValueError('strength must stay between zero and one')
    points = np.asarray(points, dtype=np.float64)
    result = points.copy()
    protected = np.ones(len(points), bool)
    groups = []
    grid = np.linspace(0., 1., 61)
    for group in range(7):
        indices = range(group*480, (group+1)*480)
        samples = np.stack([_sample(points[starts[i]:starts[i]+sizes[i]], grid) for i in indices])
        core = samples.mean(axis=0)
        tangent = _unit(np.gradient(core, axis=0))
        across = np.tile([1., 0., 0.], (len(grid), 1))-tangent*tangent[:, 0, None]
        if np.linalg.norm(across, axis=1).min() < .12:
            raise RuntimeError('Forelock path needs an explicit lateral-frame review: '+str(group))
        across = _unit(across)
        normal = _unit(np.cross(tangent, across))
        normal *= np.where(np.sum(normal*(core-center), axis=1) < 0., -1., 1.)[:, None]
        offset = samples-core[None, :, :]
        x = np.sum(offset*across[None, :, :], axis=2)
        depth = np.sum(offset*normal[None, :, :], axis=2)
        # Fit one smooth cross-section per main lock. Fixed eight-bin depth
        # steps from R03 are residuals; a quadratic retains the broad arch.
        fitted = np.empty_like(depth)
        for j in range(len(grid)):
            scale = max(float(np.quantile(np.abs(x[:, j]), .95)), .001)
            u = x[:, j]/scale
            design = np.column_stack((np.ones(480), u, u*u))
            fitted[:, j] = design@np.linalg.lstsq(design, depth[:, j], rcond=None)[0]
        residual = depth-fitted
        # Smooth the residual correction longitudinally; never move each point
        # independently or introduce extra clumps/fibres to hide the ridges.
        correction = -.65*.0015*np.tanh(residual/.0015)
        for unused in range(2):
            correction[:, 1:-1] = (.25*correction[:, :-2]+.5*correction[:, 1:-1]
                                    +.25*correction[:, 2:])
        changed_depth = []
        for local, i in enumerate(indices):
            begin, count = int(starts[i]), int(sizes[i])
            t = np.linspace(0., 1., count)
            weight = envelope(t)*strength
            weight[:3] = 0.
            a = _unit(_sample(across, t)); n = _unit(_sample(normal, t))
            c = _sample(core, t)
            old = points[begin:begin+count]
            transverse = np.sum((old-c)*a, axis=1)
            side = (SHIFT[group]+transverse*(WIDTH[group]-1.))*weight
            dep = np.interp(t, grid, correction[local])*weight
            result[begin:begin+count] = old+a*side[:, None]+n*dep[:, None]
            active = weight > 0.
            protected[begin:begin+count] = ~active
            changed_depth.extend(np.abs(dep[active]).tolist())
        active_grid = (grid > .10) & (grid < .34)
        groups.append({'group':group, 'strands':480,
            'rootCenterWorld':core[0].tolist(), 'tipCenterWorld':core[-1].tolist(),
            'rootWidth95Mm':float(np.quantile(x[:, 0], .975)-np.quantile(x[:, 0], .025))*1000.,
            'initialCrossSectionResidualRmsMm':float(np.sqrt(np.mean(residual[:, active_grid]**2)))*1000.,
            'maximumNormalCorrectionMm':float(max(changed_depth, default=0.))*1000.,
            'peakWidthFactor':float(1.+(WIDTH[group]-1.)*strength),
            'peakLateralShiftMm':float(SHIFT[group]*strength*1000.)})
    result[protected] = points[protected]
    return result, protected, groups


def _surface(bpy, util):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    evaluated = bpy.data.objects['R01 Head'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = util['_world'](util['_values'](mesh.vertices, 'co'), evaluated.matrix_world)
        triangles = np.array([t.vertices[:] for t in mesh.loop_triangles], dtype=np.int32)
        normals = util['_values'](mesh.vertices, 'normal').astype(float)
        normals = _unit(normals@np.linalg.inv(np.array(evaluated.matrix_world)[:3, :3]))
    finally:
        evaluated.to_mesh_clear()
    tree = BVHTree.FromPolygons([Vector(v) for v in vertices], triangles.tolist(), all_triangles=True)

    def nearest(point):
        hit, face_normal, index, distance = tree.find_nearest(Vector(point), .08)
        if hit is None:
            raise RuntimeError('Anterior forelock contact query missed actual evaluated head')
        hit = np.asarray(hit); ids = triangles[index]; a, b, c = vertices[ids]
        e0 = b-a; e1 = c-a; d = hit-a
        d00 = e0@e0; d01 = e0@e1; d11 = e1@e1; den = d00*d11-d01*d01
        if abs(den) < 1e-20:
            raise RuntimeError('Degenerate nearest head triangle')
        v = (d11*(d@e0)-d01*(d@e1))/den
        w = (d00*(d@e1)-d01*(d@e0))/den
        n = _unit(normals[ids[0]]*(1-v-w)+normals[ids[1]]*v+normals[ids[2]]*w)
        if n@np.asarray(face_normal) < 0.:
            n = -n
        return hit, n, float((point-hit)@n)
    return nearest, float(vertices[:, 2].max())


def _contact(points, candidate, protected, nearest):
    out = candidate.copy(); touched = 0; maximum = 0.; minimum_gain = 1.
    for i in np.flatnonzero(~protected):
        old_hit, old_n, old_clearance = nearest(points[i])
        hit, normal, clearance = nearest(out[i])
        # Existing contact remains the reference. Do not lift an old partially
        # buried root globally, and do not project free forelocks onto the face.
        target = min(.00025, old_clearance)
        needed = max(0., target-clearance)
        if needed > .0009:
            raise RuntimeError('New contact correction exceeds 0.9 mm: '+json.dumps(
                {'point':int(i), 'requiredMm':needed*1000., 'old':points[i].tolist(),
                 'proposed':out[i].tolist(), 'oldClearanceMm':old_clearance*1000.}))
        if needed > 0.:
            out[i] += normal*(needed+.000003)
            touched += 1; maximum = max(maximum, needed+.000003)
            hit, normal, clearance = nearest(out[i])
        minimum_gain = min(minimum_gain, clearance-target)
        if clearance < target-.000015:
            raise RuntimeError('Actual head surface contact did not converge at point '+str(i))
    return out, {'correctedPoints':touched, 'maximumCorrectionMm':maximum*1000.,
                 'minimumClearanceAbovePreservedTargetMm':minimum_gain*1000.}


def _continuity(old, new, starts, sizes, protected):
    if not np.isfinite(new).all() or not np.array_equal(old[protected], new[protected]):
        raise RuntimeError('Finite/fixed point invariant failed')
    displacement = np.linalg.norm(new-old, axis=1)
    if displacement.max() > .0036:
        raise RuntimeError('Forelock correction exceeds 3.6 mm: '+str(float(displacement.max()*1000.)))
    maximum_step = 0.; maximum_extra = 0.
    for i, (start, count) in enumerate(zip(starts, sizes)):
        a = old[start:start+count]; b = new[start:start+count]
        oldstep = np.linalg.norm(np.diff(a, axis=0), axis=1)
        step = np.linalg.norm(np.diff(b, axis=0), axis=1)
        extra = step-oldstep
        j = int(np.argmax(extra)); maximum_step = max(maximum_step, float(step.max()))
        maximum_extra = max(maximum_extra, float(extra[j]))
        if step.max() > .020 or extra[j] > .0012:
            raise RuntimeError('Forelock continuity guard: '+json.dumps({'strand':int(i),
                'segment':j, 'oldStepMm':float(oldstep[j]*1000.), 'newStepMm':float(step[j]*1000.),
                'oldPoints':a[j:j+2].tolist(), 'newPoints':b[j:j+2].tolist()}))
    return {'maximumDisplacementMm':float(displacement.max()*1000.),
            'changedPoints':int((displacement > 1e-9).sum()), 'exactFixedPoints':int(protected.sum()),
            'maximumSegmentMm':maximum_step*1000., 'maximumExtraSegmentMm':maximum_extra*1000.}


def _protected(bpy, util, uv):
    result = util['_protected'](bpy, uv)
    result.pop(OBJECT, None)
    obj = bpy.data.objects[util['SCALP']]
    result['posteriorScalpExplicit'] = {'data':obj.data.name, 'hash':util['_data_hash'](obj.data),
        'matrix':uv['_plain'](obj.matrix_world),
        'metadata':{key:obj.get(key) for key in (util['GROUPS'], util['PROVENANCE'],
                                               util['MARKER'], util['BACKUP'], util['META'])}}
    return result


def run(source_relative=SOURCE, strength=1., diagnostics_only=False):
    import bpy
    p = project_path(); source = (p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the explicitly selected independent source in background Object Mode')
    obj = bpy.data.objects[OBJECT]; rig = bpy.data.objects['Character_Rig']
    if any(key in obj for key in (MARKER, BACKUP, META)):
        raise RuntimeError('A forelock candidate already exists; preserve or restore before retrying')
    if obj.type != 'CURVES' or obj.library or obj.data.library or obj.data.users != 1:
        raise RuntimeError('Expected local single-user native forelocks')
    if any(m.show_viewport or m.show_render for m in obj.modifiers) or obj.animation_data or obj.data.animation_data:
        raise RuntimeError('Static forelock data required; do not remove unknown modifiers/animation')
    if not obj.get('R03_softening_v1') or not obj.get('R05_reference_groom'):
        raise RuntimeError('Expected accepted R03 softened and R05 widened forelock provenance')
    util = _api('posterior_groom_r09'); uv = _api('production_uv_r08')
    source_sha = util['_sha'](source); original = obj.data
    original_hash = util['_data_hash'](original)
    restore_record = {'originalHash':original_hash, 'originalFakeUser':original.use_fake_user}
    attachment = {'parent':uv['_plain'](obj.parent), 'type':obj.parent_type, 'bone':obj.parent_bone,
                  'basis':uv['_plain'](obj.matrix_basis), 'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
    before = _protected(bpy, util, uv); pose = rig.data.pose_position
    prepared = None; committed = False
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        local = util['_values'](original.position_data, 'vector')
        matrix = obj.matrix_world.copy(); world = util['_world'](local, matrix)
        starts = np.array([c.first_point_index for c in original.curves], dtype=np.int32)
        sizes = np.array([c.points_length for c in original.curves], dtype=np.int32)
        if len(starts) != 3360 or sizes.min() < 40 or sizes.max() > 60:
            raise RuntimeError('Reviewed 7 x 480 variable-length forelock topology changed')
        nearest, top = _surface(bpy, util)
        center = np.array([float(world[:, 0].mean()), .003, top-.10])
        proposal, protected, groups = propose(world, starts, sizes, center, strength)
        candidate, contact = _contact(world, proposal, protected, nearest)
        candidate[protected] = world[protected]
        numeric = _continuity(world, candidate, starts, sizes, protected)
        new_local = util['_world'](candidate, matrix.inverted()).astype(np.float32)
        new_local[protected] = local[protected]
        roundtrip = util['_world'](new_local, matrix)
        if np.abs(roundtrip-candidate).max() > 2e-6:
            raise RuntimeError('Forelock world/local conversion exceeded 2 micrometres')
        prepared = original.copy(); prepared.name = 'R09 anterior forelock flow'
        prepared.position_data.foreach_set('vector', new_local.ravel())
        old_attributes = util['_attributes'](original); new_attributes = util['_attributes'](prepared)
        for name, row in old_attributes.items():
            other = new_attributes.get(name)
            if other is None or row['domain'] != other['domain'] or row['type'] != other['type']:
                raise RuntimeError('Forelock attributes changed unexpectedly: '+name)
            if name != 'position' and not np.array_equal(row['values'], other['values']):
                raise RuntimeError('Non-position forelock attribute changed: '+name)
        if not np.array_equal(sizes, [c.points_length for c in prepared.curves]):
            raise RuntimeError('Forelock curve topology/order changed')
        rig.data.pose_position = pose; bpy.context.view_layer.update()
        if _protected(bpy, util, uv) != before:
            raise RuntimeError('Protected face/posterior/tail/rig data changed during preparation')
        report = {'source':str(source.relative_to(p)), 'sourceSha256':source_sha,
            'object':OBJECT, 'sourceData':original.name, 'sourceDataSha256':original_hash,
            'candidateDataSha256':util['_data_hash'](prepared), 'strength':strength,
            'scope':'Only original forelocks at normalized t=.06..40; first three points and t>=.40 exact.',
            'groupStats':groups, 'continuity':numeric, 'contact':contact,
            'strandsAdded':0, 'radiiMaterialsUVChanged':False, 'existingCohorts':[480]*7,
            'posteriorCompatibility':'No R06 scalp data/RT09 group metadata changes; apply before or after posterior append.',
            'runtimeConsequence':'If adopted, refresh these seven existing fringe cards/atlas groups; tail cache unchanged.',
            'savedBlend':False, 'writtenInMemory':not diagnostics_only, 'appearanceAccepted':False,
            'requiredEvidence':['Matched front and quarter with unchanged lighting.',
                                'Crown close-up: fewer regular ridges, no new skin gap, unchanged tip silhouette.']}
        if diagnostics_only:
            return report
        original.use_fake_user = True
        obj[BACKUP] = original.name; obj[META] = json.dumps(restore_record)
        obj.data = prepared; obj[MARKER] = json.dumps(report)
        obj.data.update_tag(); obj.update_tag(); bpy.context.view_layer.update()
        after_attachment = {'parent':uv['_plain'](obj.parent), 'type':obj.parent_type, 'bone':obj.parent_bone,
                            'basis':uv['_plain'](obj.matrix_basis), 'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
        if after_attachment != attachment or _protected(bpy, util, uv) != before:
            raise RuntimeError('Protected data or forelock binding changed after assignment')
        if util['_data_hash'](original) != original_hash or util['_sha'](source) != source_sha:
            raise RuntimeError('Retained original data or saved source changed')
        report['protectedDataUnchanged'] = True
        committed = True
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
    util = _api('posterior_groom_r09')
    record = json.loads(obj[META]); original = bpy.data.hair_curves.get(obj.get(BACKUP, ''))
    if original is None or util['_data_hash'](original) != record['originalHash']:
        raise RuntimeError('Retained original forelock data missing or changed; refuse blind restore')
    candidate = obj.data; candidate.use_fake_user = True
    obj.data = original; original.use_fake_user = record['originalFakeUser']
    for key in (MARKER, BACKUP, META):
        del obj[key]
    bpy.context.view_layer.update()
    return {'object':OBJECT, 'restored':original.name, 'retainedCandidate':candidate.name, 'savedBlend':False}


if __name__ == '__main__':
    print('Loaded only. Call run(source_relative=..., diagnostics_only=True), then run() in an independent copy.')
