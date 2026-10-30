"""Pure-array local canthal closure design from an actual closed L3 surface.

This is not an installer. It proposes a bounded angular tissue displacement
at residual visible corner openings, preserving distance along rays to the
actual globe triangles. It never treats the globe as an ideal sphere.
"""
import numpy as np


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


def ray_radius(origins, directions, vertices, triangles, chunk_size=128):
    """First positive exact triangle intersection for each YZ section ray."""
    vertices = np.asarray(vertices, float); triangles = np.asarray(triangles, np.int32)
    faces = vertices[triangles]; edge1 = faces[:, 1]-faces[:, 0]; edge2 = faces[:, 2]-faces[:, 0]
    result = np.full(len(origins), np.nan)
    for first in range(0, len(origins), chunk_size):
        origin = origins[first:first+chunk_size, None, :]
        direction = directions[first:first+chunk_size, None, :]
        cross = np.cross(direction, edge2[None, :, :])
        det = np.sum(edge1[None, :, :]*cross, axis=2)
        valid = abs(det)>1e-15
        inverse = np.divide(1., det, out=np.zeros_like(det), where=valid)
        offset = origin-faces[None, :, 0, :]
        u = np.sum(offset*cross, axis=2)*inverse
        q = np.cross(offset, edge1[None, :, :])
        v = np.sum(direction*q, axis=2)*inverse
        distance = np.sum(edge2[None, :, :]*q, axis=2)*inverse
        hit = valid & (u>=-1e-9) & (v>=-1e-9) & (u+v<=1.+1e-9) & (distance>1e-8)
        nearest = np.min(np.where(hit, distance, np.inf), axis=1)
        nearest[~np.isfinite(nearest)] = np.nan
        result[first:first+len(nearest)] = nearest
    return result


def propose(observation, arrays, remaining_fraction=.10, lower_share=.20,
            upper_fade=.0025, lower_fade=.0025):
    """Return L3 deltas, actual-surface angle/radius fields and explicit metrics.

    This only addresses visible residual columns at the two ends. Central
    contact is untouched. A valid proposal still needs actual evaluation and
    contact/image review before an installer can be adopted.
    """
    if observation.get('blinkValues') != {'L': 1., 'R': 1.}:
        raise RuntimeError('Actual fully closed current geometry required')
    world = np.asarray(arrays['head_world'], float)
    delta = np.zeros_like(world); output = {}; report = {}
    for side in ('L', 'R'):
        eye = observation['eyes'][side]; center = np.asarray(eye['centerWorld'])
        diameter = np.mean(np.asarray(eye['diametersWorld'])[[0, 2]])
        side_angle = np.zeros(len(world)); side_radius = np.zeros(len(world))
        side_report = []
        for end in (-1., 1.):
            rows = [r for r in eye['contour'] if end*(r['x']-center[0])>diameter*.25
                    and all(not r[e].get('unboundedAtGlobeRim', True)
                            for e in ('lower', 'upper'))]
            if not rows:
                side_report.append({'end': end, 'visibleColumns': 0}); continue
            rows.sort(key=lambda row: row['x'])
            xs = np.array([r['x'] for r in rows])
            section = world-center; radius = np.linalg.norm(section[:, 1:], axis=1)
            angle = np.arctan2(section[:, 2], -section[:, 1])
            # Visible transition angles belong to the globe surface, while the
            # forward lid margin has a different radius. Classify upper/lower
            # tissue by the actual opening's world height, never by globe phi.
            low_z = np.interp(world[:, 0], xs, [r['lower']['worldZ'] for r in rows])
            high_z = np.interp(world[:, 0], xs, [r['upper']['worldZ'] for r in rows])
            z = world[:, 2]; midpoint = low_z+lower_share*(high_z-low_z)
            target_z = midpoint+(z-midpoint)*remaining_fraction
            above = z>high_z; below = z<low_z
            target_z[above] = z[above]-(1.-remaining_fraction)*(1.-lower_share)*(high_z[above]-low_z[above])*(1.-smooth((z[above]-high_z[above])/upper_fade))
            target_z[below] = z[below]+(1.-remaining_fraction)*lower_share*(high_z[below]-low_z[below])*(1.-smooth((low_z[below]-z[below])/lower_fade))
            target = np.arcsin(np.clip((target_z-center[2])/np.maximum(radius, 1e-10), -.999999, .999999))
            support = .0006; plateau_padding = .0004
            weight = smooth((world[:, 0]-(xs[0]-support-plateau_padding))/support)*smooth(((xs[-1]+support+plateau_padding)-world[:, 0])/support)
            weight *= 1.-smooth((radius-.014)/.006)
            weight *= (world[:, 0]*center[0]>0.) & (section[:, 1]<0.)
            shift = (target-angle)*weight
            ids = np.flatnonzero(abs(shift)>1e-8)
            if not len(ids): continue
            origins = np.broadcast_to(center, (len(ids), 3)).copy(); origins[:, 0] = world[ids, 0]
            old_directions = section[ids].copy(); old_directions[:, 0] = 0.
            old_directions /= radius[ids, None]
            new_angle = angle[ids]+shift[ids]
            new_directions = np.column_stack((np.zeros(len(ids)), -np.cos(new_angle), np.sin(new_angle)))
            globe = arrays['eye_'+side+'_world']; tri = arrays['eye_'+side+'_triangles']
            old_surface = ray_radius(origins, old_directions, globe, tri)
            proximity = 1.-smooth((radius[ids]-old_surface-.012)/.006)
            shift[ids] *= np.nan_to_num(proximity, nan=0.)
            new_angle = angle[ids]+shift[ids]
            new_directions = np.column_stack((np.zeros(len(ids)), -np.cos(new_angle), np.sin(new_angle)))
            new_surface = ray_radius(origins, new_directions, globe, tri)
            valid = np.isfinite(old_surface) & np.isfinite(new_surface) & (abs(shift[ids])>1e-8)
            rejected = ids[~valid].tolist(); ids = ids[valid]
            old_surface = old_surface[valid]; new_surface = new_surface[valid]
            # Preserve existing positive clearance and lift any pre-existing
            # negative radial gap to a small positive contact allowance.
            gap = radius[ids]-old_surface
            new_radius = new_surface+np.maximum(gap, .00002)
            desired_height = radius[ids]*np.sin(angle[ids]+shift[ids])
            final_angle = new_angle[valid].copy(); final_directions = new_directions[valid].copy()
            for _ in range(30):
                requested_angle = np.arcsin(np.clip(desired_height/np.maximum(new_radius, 1e-10), -.999999, .999999))
                final_angle = .75*final_angle+.25*requested_angle
                final_directions = np.column_stack((np.zeros(len(ids)), -np.cos(final_angle), np.sin(final_angle)))
                actual_radius = ray_radius(origins[valid], final_directions, globe, tri)
                if not np.isfinite(actual_radius).all():
                    raise RuntimeError('Actual globe section mapping left its measured domain')
                new_radius = actual_radius+np.maximum(gap, .00002)
            shift[ids] = final_angle-angle[ids]
            moved = origins[valid]+final_directions*new_radius[:, None]
            delta[ids] += moved-world[ids]
            side_angle[ids] += shift[ids]
            side_radius[ids] += new_radius-radius[ids]
            side_report.append({'end': end, 'visibleColumns': len(rows), 'xRangeMeters': [float(xs[0]), float(xs[-1])],
                                'changedVertices': len(ids), 'unmappedVertexIds': rejected,
                                'maximumDisplacementMeters': float(np.linalg.norm(moved-world[ids], axis=1).max()),
                                'actualRadialGapRangeMeters': [float(gap.min()), float(gap.max())],
                                'existingNegativeGapCount': int(np.sum(gap<-.00002)),
                                'maximumAngleRadians': float(abs(shift[ids]).max()),
                                'maximumActualGlobeRadiusChangeMeters': float(abs(new_radius-radius[ids]).max()),
                                'maximumClosureHeightResidualMeters': float(abs((moved[:, 2]-center[2])-desired_height).max())})
        output[side] = {'angle': side_angle, 'radiusChange': side_radius}
        report[side] = side_report
    return delta, output, report


def front_visibility(head_world, head_triangles, eye_world, eye_triangles, refine=False):
    """Exact orthographic triangle projection, same grid as actual BVH audit.

    Works on captured evaluated triangles without Blender. No backface cull.
    The caller must separately compare this numerical result with the source
    actual BVH report before relying on it for a new geometry proposal.
    """
    lo = eye_world.min(axis=0); hi = eye_world.max(axis=0)
    xs = np.linspace(lo[0]+.00015, hi[0]-.00015, 161)
    zs = np.linspace(lo[2]+.00010, hi[2]-.00010, 121)

    def depths(world, triangles):
        faces = world[triangles]
        xmin = faces[:, :, 0].min(axis=1); xmax = faces[:, :, 0].max(axis=1)
        answer = np.full((len(xs), len(zs)), np.inf); caches = []
        for j, x in enumerate(xs):
            p = faces[(xmin<=x+1e-12) & (xmax>=x-1e-12)]
            edge1 = p[:, 1]-p[:, 0]; edge2 = p[:, 2]-p[:, 0]
            determinant = edge1[:, 0]*edge2[:, 2]-edge1[:, 2]*edge2[:, 0]
            useful = abs(determinant)>1e-16
            p = p[useful]; edge1 = edge1[useful]; edge2 = edge2[useful]; determinant = determinant[useful]
            caches.append((p, edge1, edge2, determinant))
            dx = x-p[:, 0, 0]; dz = zs[:, None]-p[None, :, 0, 2]
            u = (dx[None, :]*edge2[None, :, 2]-dz*edge2[None, :, 0])/determinant[None, :]
            v = (edge1[None, :, 0]*dz-edge1[None, :, 2]*dx[None, :])/determinant[None, :]
            inside = (u>=-1e-9) & (v>=-1e-9) & (u+v<=1.+1e-9)
            y = p[None, :, 0, 1]+u*edge1[None, :, 1]+v*edge2[None, :, 1]
            if len(p): answer[j] = np.where(inside, y, np.inf).min(axis=1)
        return answer, caches

    skin, skin_cache = depths(head_world, head_triangles); globe, globe_cache = depths(eye_world, eye_triangles)
    visible = np.isfinite(globe) & (globe<skin-1e-7)
    report = {'visibleSampleCount': int(visible.sum()),
            'frontVisibleAreaSquareMeters': float(visible.sum()*(xs[1]-xs[0])*(zs[1]-zs[0])),
            'columns': [{'x': float(x), 'z': zs[mask].tolist()} for x, mask in zip(xs, visible) if mask.any()]}
    if refine:
        def depth_at(cache, x, z):
            p, edge1, edge2, determinant = cache
            dx = x-p[:, 0, 0]; dz = z-p[:, 0, 2]
            u = (dx*edge2[:, 2]-dz*edge2[:, 0])/determinant
            v = (edge1[:, 0]*dz-edge1[:, 2]*dx)/determinant
            inside = (u>=-1e-9) & (v>=-1e-9) & (u+v<=1.+1e-9)
            y = p[:, 0, 1]+u*edge1[:, 1]+v*edge2[:, 1]
            return float(np.min(np.where(inside, y, np.inf))) if len(y) else np.inf

        contours = []
        for index, (x, flags) in enumerate(zip(xs, visible)):
            ids = np.flatnonzero(flags)
            if not len(ids): continue
            row = {'x': float(x)}
            for label, inside, outside in [('lower', int(ids[0]), int(ids[0]-1)), ('upper', int(ids[-1]), int(ids[-1]+1))]:
                if outside<0 or outside>=len(zs):
                    row[label] = {'worldZ': float(zs[inside]), 'unboundedAtGlobeRim': True}; continue
                a = zs[inside]; b = zs[outside]
                for _ in range(16):
                    z = (a+b)*.5; eye_y = depth_at(globe_cache[index], x, z); skin_y = depth_at(skin_cache[index], x, z)
                    if np.isfinite(eye_y) and eye_y<skin_y-1e-7: a = z
                    else: b = z
                row[label] = {'worldZ': float((a+b)*.5), 'unboundedAtGlobeRim': False}
            contours.append(row)
        report['contour'] = contours
    return report
