"""Soften the existing R03 native groom in a saved candidate.

No save, render, simulation, or live connection is performed by this module.
The four existing curve objects retain their point counts and attachment.
Twin tails, ties, expressions and the scalp root substrate are untouched.
Call run() once; restore() switches the originals back to retained data copies.
All geometric work is done in the rig rest pose and evaluated-head world space.
"""
import math
import random
import json
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

NAMES = ('R03 Scalp gather guides', 'R03 Crown soft coverage',
         'R03 Forelocks from crown', 'R03 curved cheek wisps')
SECONDARY = 'R03 soft transition fibres'
MARKER = 'R03_softening_v1'


def _smooth(t):
    t = max(0.0, min(1.0, float(t)))
    return t*t*(3.0-2.0*t)


def _mix(a, b, t):
    return a*(1.0-t)+b*t


def _read(obj):
    data = obj.data
    starts = [c.first_point_index for c in data.curves]
    sizes = [c.points_length for c in data.curves]
    raw = np.empty(len(data.position_data)*3, dtype=np.float64)
    data.position_data.foreach_get('vector', raw)
    points = raw.reshape(-1, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    points = points @ mw[:3,:3].T + mw[:3,3]
    radius = np.empty(len(points), dtype=np.float64)
    if not data.attributes.get('radius'):
        raise RuntimeError('Existing strand radius is required: '+obj.name)
    data.attributes['radius'].data.foreach_get('value', radius)
    return points, radius, starts, sizes, mw


def _sample(points, t):
    u = max(0.0, min(1.0, float(t)))*(len(points)-1)
    i = min(int(u), len(points)-2)
    return _mix(points[i], points[i+1], u-i)


def _head_surface(head):
    evaluated = head.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        vertices = [evaluated.matrix_world @ v.co for v in mesh.vertices]
        faces = [tuple(p.vertices) for p in mesh.polygons]
        top = max(v.z for v in vertices)
        return BVHTree.FromPolygons(vertices, faces), top-1.553
    finally:
        evaluated.to_mesh_clear()


def _new_material(source):
    material = source.copy()
    material.name = 'R03 softer charcoal fibres'
    for node in material.node_tree.nodes:
        if node.type != 'BSDF_HAIR_PRINCIPLED':
            continue
        for key, value in [('Roughness', .64), ('Radial Roughness', .55),
                           ('Coat', 0.0), ('Random Roughness', .13)]:
            if node.inputs.get(key):
                node.inputs[key].default_value = value
    return material


def run(add_secondary=True, diagnostics_only=False):
    if bpy.data.filepath.replace('\\','/').endswith('/work/character.blend'):
        raise RuntimeError('Use the existing saved R03 candidate, not the master')
    objects = [bpy.data.objects.get(name) for name in NAMES]
    if any(o is None or o.type != 'CURVES' for o in objects):
        raise RuntimeError('All four existing R03 native groom objects are required')
    if any(o.get(MARKER) for o in objects) or bpy.data.objects.get(SECONDARY):
        raise RuntimeError('Softening already applied; use restore() before another pass')
    if any(o.data.users != 1 or o.library or o.data.library for o in objects):
        raise RuntimeError('Use local single-user candidate curves')
    if any(any(m.show_viewport or m.show_render for m in o.modifiers) for o in objects):
        raise RuntimeError('Existing groom modifiers need an explicit integration path')
    rig = bpy.data.objects.get('Character_Rig')
    head = bpy.data.objects.get('R01 Head')
    if not head:
        raise RuntimeError('The current evaluated R01 Head is required')
    pose = rig.data.pose_position if rig else None
    if rig:
        rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    try:
        bvh, shift = _head_surface(head)
        if not -.045 < shift < .025:
            raise RuntimeError('Unexpected rest crown height; check the candidate')
        center = Vector((0, .003, 1.445+shift))
        rng = random.Random(303041)
        prepared = []
        radial_misses = 0
        continuity_summary = []

        def contact(q, lift):
            nonlocal radial_misses
            q = Vector(q)
            direction = (q-center).normalized()
            hit, normal, _, _ = bvh.ray_cast(center+direction*.36, -direction)
            if hit is None:
                radial_misses += 1
                raise RuntimeError('Head surface control ray missed; no height fallback')
            return np.array(hit+normal*lift), np.array(normal)

        def smooth_path(path):
            out = path.copy()
            for j in range(1, len(path)-1):
                out[j] = .7*path[j]+.15*(path[j-1]+path[j+1])
            return out

        # Contact follows existing coarse guides, not camera-facing ray scans.
        # Cohorts are known from the R03 builder: 34 scalp / 20 crown fibres.
        for oi, obj in enumerate(objects):
            points, radii, starts, sizes, mw = _read(obj)
            result = points.copy()
            new_radius = radii.copy()
            guide_trace = {}
            if oi in (0, 1):
                cohort = 34 if oi == 0 else 20
                expected = 5848 if oi == 0 else 720
                if len(starts) != expected or len(set(sizes)) != 1:
                    raise RuntimeError('The R03 surface strand ordering changed: '+obj.name)
                samples = sizes[0]
                for first in range(0, len(starts), cohort):
                    ids = range(first, first+cohort)
                    original = np.array([points[starts[i]:starts[i]+samples] for i in ids])
                    guide = original.mean(axis=0)
                    altered = guide.copy()
                    normals = []
                    routed_points = []
                    projected_points = []
                    contact_weights = []
                    # Crown roots occupy an oval patch rather than one top pole.
                    angle = (first/cohort)*2.3999632297
                    spread = (.0075 if oi == 1 else .0028)
                    root_shift = np.array([math.cos(angle)*spread,
                                           math.sin(angle)*spread*1.25, 0.0])
                    for j, q0 in enumerate(guide):
                        t = j/(samples-1)
                        q = q0+root_shift*(1-_smooth(t/.28))
                        z = q[2]-shift
                        # Ears span |X| .060-.086, Y -.010..+.025,
                        # Z 1.406..1.451 in the unlowered reference. Route the
                        # late gather behind them before the last tie approach.
                        ear = (_smooth((abs(q[0])-.052)/.023)
                               *_smooth((1.468-z)/.027)
                               *_smooth((.038-q[1])/.046))
                        route = ear*(1-_smooth((t-.87)/.13))
                        q[1] += .042*route
                        q[2] += .0055*route*(.75+.25*math.sin(angle))
                        # C1 fade from surface contact to the original free tie
                        # approach. There is no branch at t=.88: contact and its
                        # normal smoothly lose influence over fourteen samples.
                        contact_weight = 1-_smooth((t-.72)/.23)
                        radial_normal = np.array((Vector(q)-center).normalized())
                        routed_points.append(q.copy())
                        if contact_weight > 0:
                            lift = .0015+.00075*(.5+.5*math.sin(angle+t*3.5))
                            if oi == 1:
                                lift += .0010*(1-_smooth(t/.32))
                            surface_q, surface_normal = contact(q, lift)
                            projected_points.append(surface_q.copy())
                            q = _mix(q, surface_q, contact_weight)
                            normal = _mix(radial_normal, surface_normal, contact_weight)
                            normal /= max(float(np.linalg.norm(normal)), 1e-9)
                        else:
                            projected_points.append(None)
                            normal = radial_normal
                        contact_weights.append(contact_weight)
                        altered[j] = q
                        normals.append(normal)
                    raw_altered = altered.copy()
                    altered = smooth_path(altered)
                    normals = np.array(normals)
                    # Continuous normal field prevents a BVH triangle or ear
                    # rim normal change from swinging a whole bundle at once.
                    normals = smooth_path(normals)
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
                    guide_trace[first//cohort] = {
                        'original_guide':guide, 'routed_guide':routed_points,
                        'projected_guide':projected_points,
                        'contact_weights':contact_weights,
                        'raw_altered_guide':raw_altered, 'smoothed_guide':altered}
                    # Existing transverse offsets remain, with modest coherent
                    # depth separation instead of dense flat ribbons.
                    for k, i in enumerate(ids):
                        begin = starts[i]
                        src = original[k]
                        phase = k*.61803398875*math.tau+angle
                        lift = .00045+.00080*((k*.754877666)%1)
                        delta = src-guide
                        for j in range(samples):
                            t = j/(samples-1)
                            spread_t = 1.12-.12*_smooth(t/.22)
                            normal_component = float(np.dot(delta[j], normals[j]))
                            tangent_delta = delta[j]-normals[j]*normal_component
                            flutter = .00022*math.sin(phase+t*4)*math.sin(math.pi*t)
                            candidate = (altered[j]+tangent_delta*spread_t
                                         +normals[j]*(lift+flutter))
                            # The last sample is the exact existing per-strand
                            # tie point; zero-derivative blending also preserves
                            # the original arrival tangent as the weight reaches 1.
                            tie_weight = _smooth((t-.88)/.12)
                            result[begin+j] = _mix(candidate, src[j], tie_weight)
                            root_fade = .58+.42*_smooth(t/.12)
                            taper = max(.035, (1-t**1.8)**.65)
                            factor = .72 if oi == 0 else .68
                            new_radius[begin+j] = radii[begin]*factor*root_fade*taper
            elif oi == 2:
                if len(starts) != 3360:
                    raise RuntimeError('Expected the seven existing 480-strand forelocks')
                # Eight transverse sub-bundles within every existing broad lock.
                # Separate depth and terminal lengths while retaining most length.
                retain = [.947, .992, .965, 1.0, .954, .984, .973, .997]
                layer = [.0007, .0030, -.0006, .0041, .0015, .0035, .0002, .0023]
                for group in range(7):
                    ids = range(group*480, (group+1)*480)
                    roots = np.array([points[starts[i]] for i in ids])
                    xmin, xmax = roots[:,0].min(), roots[:,0].max()
                    root_back = [.003, .007, .001, .005, .000, .006, .002][group]
                    for i in ids:
                        begin, count = starts[i], sizes[i]
                        source = points[begin:begin+count]
                        band = min(7, int(8*(source[0,0]-xmin)/max(xmax-xmin, 1e-8)))
                        individuality = ((i*.754877666)%1)-.5
                        length = min(1.0, retain[band]+individuality*.012)
                        phase = group*1.7+band*.9
                        for j in range(count):
                            t = j/(count-1)
                            q = _sample(source, t*length).copy()
                            root_w = 1-_smooth(t/.34)
                            q[1] += root_back*root_w
                            q[0] += .0018*math.sin(phase)*root_w
                            # Free forelocks occupy different depth layers;
                            # their lower portion is never projected onto the face.
                            q[1] -= layer[band]*math.sin(math.pi*t*.85)
                            q[0] += (.00065*math.sin(phase+t*4.0)
                                     +individuality*.0005)*math.sin(math.pi*t)
                            if t < .28:
                                surface, _ = contact(q, .0022+.0018*((band+group)%3)/2)
                                q = _mix(q, surface, 1-_smooth((t-.12)/.16))
                            result[begin+j] = q
                            taper = max(.026, (1-t**1.45)**.72)
                            root_fade = .52+.48*_smooth(t/.12)
                            new_radius[begin+j] = radii[begin]*(.48+.08*((i*.618)%1))*taper*root_fade
            else:
                if len(starts) != 630:
                    raise RuntimeError('Expected the six existing curved wisps')
                # Keep the gesture but feather the ends and split the dark rods.
                for i, (begin, count) in enumerate(zip(starts, sizes)):
                    source = points[begin:begin+count]
                    phase = (i//105)*1.23+(i%7)*.31
                    length = .91+.09*((i*.61803398875)%1)
                    side = 1 if source[0,0] > 0 else -1
                    for j in range(count):
                        t = j/(count-1)
                        q = _sample(source, t*length).copy()
                        q[0] += side*.0012*math.sin(phase+t*4.2)*math.sin(math.pi*t)
                        q[1] -= (.0005+.0014*((i*.754877666)%1))*math.sin(math.pi*t)
                        result[begin+j] = q
                        new_radius[begin+j] = radii[begin]*.59*max(.025, (1-t**1.6)**.7)
            if not np.all(np.isfinite(result)) or not np.all(np.isfinite(new_radius)):
                raise RuntimeError('Non-finite groom result')
            # Keep the same 20 mm threshold. Report actual points and the guide
            # stages when it rejects a strand, instead of hiding the cause with
            # relaxed limits or resampling a discontinuous curve.
            violations = []
            max_before = max_after = 0.0
            worst = None
            for strand, (begin, count) in enumerate(zip(starts, sizes)):
                old_steps = np.linalg.norm(np.diff(points[begin:begin+count], axis=0), axis=1)
                new_steps = np.linalg.norm(np.diff(result[begin:begin+count], axis=0), axis=1)
                max_before = max(max_before, float(old_steps.max()))
                segment = int(np.argmax(new_steps))
                if float(new_steps[segment]) > max_after:
                    max_after = float(new_steps[segment])
                    worst = {'strand':strand, 'segment':segment}
                for segment in np.flatnonzero(new_steps > .020):
                    segment = int(segment)
                    item = {
                        'object':obj.name, 'strand_index':strand,
                        'segment_index':segment, 'point_count':count,
                        'flat_point_indices':[begin+segment, begin+segment+1],
                        't':[segment/(count-1), (segment+1)/(count-1)],
                        'old_world_points':points[begin+segment:begin+segment+2].tolist(),
                        'new_world_points':result[begin+segment:begin+segment+2].tolist(),
                        'old_step_mm':float(old_steps[segment]*1000),
                        'new_step_mm':float(new_steps[segment]*1000),
                        'point_displacements_mm':(np.linalg.norm(
                            result[begin+segment:begin+segment+2]-points[begin+segment:begin+segment+2],
                            axis=1)*1000).tolist()}
                    if oi in (0, 1):
                        group = strand//cohort
                        trace = guide_trace[group]
                        item['cohort_index'] = group
                        item['guide_stages'] = {
                            key:[None if value[j] is None else
                                 (float(value[j]) if np.isscalar(value[j]) else np.asarray(value[j]).tolist())
                                 for j in (segment, segment+1)]
                            for key,value in trace.items()}
                    violations.append(item)
            summary = {'object':obj.name, 'strand_count':len(starts),
                       'max_original_step_mm':max_before*1000,
                       'max_new_step_mm':max_after*1000, 'worst':worst,
                       'rejected_segments':len(violations)}
            continuity_summary.append(summary)
            if violations:
                violations.sort(key=lambda x:x['new_step_mm'], reverse=True)
                diagnostic = {'reason':'strand continuity guard', 'threshold_mm':20.0,
                              'head_shift_z':shift, 'summary':summary,
                              'worst_segments':violations[:8],
                              'diagnostic_scope':'Eight longest rejected segments; all rejected segments counted',
                              'writes_performed':False}
                encoded = json.dumps(diagnostic, ensure_ascii=False, separators=(',',':'))
                print('R03_SOFTEN_DIAGNOSTIC '+encoded)
                raise RuntimeError('A discontinuous softened strand was rejected: '+encoded)
            prepared.append((obj, result, new_radius, starts, sizes, mw, points))

        if diagnostics_only:
            diagnostic = {'validation':'all existing strands passed the unchanged 20 mm continuity guard',
                          'head_shift_z':shift, 'objects':continuity_summary,
                          'radial_misses':radial_misses, 'writes_performed':False}
            print('R03_SOFTEN_DIAGNOSTIC '+json.dumps(diagnostic, ensure_ascii=False))
            return diagnostic

        # Optional fine layer uses only 308 existing scalp routes, maintaining
        # the filled crown and the same attachment instead of replacing the cap.
        fine_paths = []
        if add_secondary:
            _, points, _, starts, sizes, _, _ = prepared[0]
            for i in range(0, len(starts), 19):
                path = points[starts[i]:starts[i]+sizes[i]].copy()
                phase = i*.61803398875
                for j in range(len(path)):
                    t = j/(len(path)-1)
                    normal = np.array((Vector(path[j])-center).normalized())
                    path[j] += normal*(.0007+.0003*math.sin(phase+t*3))*math.sin(math.pi*t)
                fine_paths.append(path)

        # No geometry is modified until every candidate array passes validation.
        source_material = next((m for m in objects[0].data.materials if m), None)
        if source_material is None:
            raise RuntimeError('R03 strand material is missing')
        material = _new_material(source_material)
        report = []
        for obj, points, radius, starts, sizes, mw, original in prepared:
            backup = obj.data.copy()
            backup.name = obj.data.name+' pre-softening backup'
            backup.use_fake_user = True
            inv = np.linalg.inv(mw)
            local = points @ inv[:3,:3].T+inv[:3,3]
            obj.data.position_data.foreach_set('vector', local.ravel())
            obj.data.attributes['radius'].data.foreach_set('value', radius)
            obj.data.materials.clear()
            obj.data.materials.append(material)
            obj[MARKER] = True
            obj['R03_softening_backup_data'] = backup.name
            obj.data.update_tag()
            obj.update_tag()
            report.append({'object':obj.name, 'strands':len(starts),
                           'max_displacement_mm':float(np.linalg.norm(points-original,axis=1).max()*1000)})
        if fine_paths:
            parent_obj = objects[0]
            data = bpy.data.hair_curves.new(SECONDARY)
            data.add_curves([len(p) for p in fine_paths])
            inv = np.array(parent_obj.matrix_world.inverted())
            flat = np.concatenate(fine_paths)
            local = flat @ inv[:3,:3].T+inv[:3,3]
            data.position_data.foreach_set('vector', local.ravel())
            radius = data.attributes.new('radius','FLOAT','POINT')
            radius.data.foreach_set('value', [max(.0000015,.000021*(1-(j/(len(p)-1))**1.6)**.7)
                                               for p in fine_paths for j in range(len(p))])
            data.materials.append(material)
            obj = bpy.data.objects.new(SECONDARY, data)
            bpy.context.scene.collection.objects.link(obj)
            obj.parent = parent_obj.parent
            obj.parent_type = parent_obj.parent_type
            obj.parent_bone = parent_obj.parent_bone
            obj.matrix_parent_inverse = parent_obj.matrix_parent_inverse.copy()
            obj.matrix_basis = parent_obj.matrix_basis.copy()
            obj[MARKER] = True
            obj['R03_scope'] = 'Sparse secondary fibres copied from existing softened scalp routes'
        bpy.context.view_layer.update()
        output = {'modified':report, 'secondary_strands':len(fine_paths),
                  'head_shift_z':shift, 'radial_misses':radial_misses,
                  'continuity':continuity_summary,
                  'preserved':'twin tails, ties, root substrate, attachments and expressions',
                  'validation':'candidate continuity only; visible result still needs three-view review'}
        print(output)
        return output
    finally:
        if rig:
            rig.data.pose_position = pose
        bpy.context.view_layer.update()


def restore():
    """Reversibly restore retained pre-softening data; keep the rejected data too."""
    restored = []
    for name in NAMES:
        obj = bpy.data.objects.get(name)
        if not obj or not obj.get(MARKER):
            continue
        backup = bpy.data.hair_curves.get(obj.get('R03_softening_backup_data',''))
        if backup is None:
            raise RuntimeError('Retained pre-softening data is missing: '+name)
        obj.data.use_fake_user = True
        obj.data = backup
        obj.data.use_fake_user = False
        del obj[MARKER]
        del obj['R03_softening_backup_data']
        restored.append(name)
    secondary = bpy.data.objects.get(SECONDARY)
    if secondary:
        secondary.hide_render = True
        secondary.hide_set(True)
        secondary.name = SECONDARY+' restored-away'
    bpy.context.view_layer.update()
    return restored


if __name__ == '__main__':
    run()
