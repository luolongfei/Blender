"""Read-only exact-triangle evidence for the observed eyelid inner lining.

Consumes the actual saved seven-state observation. No Blender is imported,
no field/mesh is modified, and no physical or visual acceptance is claimed.
The computed polygon sets are index evidence, not existing named face sets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def exact_nearest(points, vertices, triangles, chunk=128):
    """Euclidean distance to triangle interiors and all clipped segment edges."""
    faces = vertices[triangles]
    ab, ac = faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]
    normal = np.cross(ab, ac)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    aa = np.sum(ab*ab, axis=1)
    bb = np.sum(ab*ac, axis=1)
    cc = np.sum(ac*ac, axis=1)
    denominator = aa*cc-bb*bb
    answer = np.empty(len(points))
    locations = np.empty_like(points)
    face_ids = np.empty(len(points), np.int32)
    for start in range(0, len(points), chunk):
        point = points[start:start+chunk]
        offset = point[:, None, :]-faces[None, :, 0, :]
        plane = np.sum(offset*normal[None, :, :], axis=2)
        projection = point[:, None, :]-plane[:, :, None]*normal[None, :, :]
        projected = projection-faces[None, :, 0, :]
        d1 = np.sum(projected*ab[None, :, :], axis=2)
        d2 = np.sum(projected*ac[None, :, :], axis=2)
        u = (cc[None, :]*d1-bb[None, :]*d2)/denominator
        v = (aa[None, :]*d2-bb[None, :]*d1)/denominator
        inside = (u>=-1e-12) & (v>=-1e-12) & (u+v<=1.+1e-12)
        best_distance = np.where(inside, plane*plane, np.inf)
        best_location = projection.copy()
        for first, last in ((0, 1), (1, 2), (2, 0)):
            edge = faces[:, last]-faces[:, first]
            relative = point[:, None, :]-faces[None, :, first, :]
            amount = np.clip(np.sum(relative*edge[None, :, :], axis=2)/np.sum(edge*edge, axis=1), 0., 1.)
            location = faces[None, :, first, :]+amount[:, :, None]*edge[None, :, :]
            distance = np.sum((location-point[:, None, :])**2, axis=2)
            better = distance<best_distance
            best_distance[better] = distance[better]
            best_location[better] = location[better]
        nearest = best_distance.argmin(axis=1)
        rows = np.arange(len(point))
        answer[start:start+len(point)] = np.sqrt(best_distance[rows, nearest])
        locations[start:start+len(point)] = best_location[rows, nearest]
        face_ids[start:start+len(point)] = nearest
    return answer, locations, face_ids


def winding_numbers(points, vertices, triangles, chunk=128):
    """Independent solid-angle inside check on the closed actual triangle mesh."""
    faces = vertices[triangles]
    result = np.empty(len(points))
    for first in range(0, len(points), chunk):
        p = faces[None, :, :, :]-points[first:first+chunk, None, None, :]
        lengths = np.linalg.norm(p, axis=3)
        numerator = np.sum(p[:, :, 0]*np.cross(p[:, :, 1], p[:, :, 2]), axis=2)
        denominator = np.prod(lengths, axis=2)
        denominator += np.sum(p[:, :, 0]*p[:, :, 1], axis=2)*lengths[:, :, 2]
        denominator += np.sum(p[:, :, 1]*p[:, :, 2], axis=2)*lengths[:, :, 0]
        denominator += np.sum(p[:, :, 2]*p[:, :, 0], axis=2)*lengths[:, :, 1]
        result[first:first+len(p)] = np.sum(2.*np.arctan2(numerator, denominator), axis=1)/(4.*np.pi)
    return result


def globe_contract(vertices, triangles):
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {'vertices': len(vertices), 'triangles': len(triangles),
            'boundaryEdges': int((counts==1).sum()), 'nonmanifoldEdges': int((counts!=2).sum()),
            'minimumTriangleAreaSquareMeters': float(np.linalg.norm(np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]],
                                                                                 vertices[triangles[:, 2]]-vertices[triangles[:, 0]]), axis=1).min()/2.)}


def plot_sections(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    figure, axes = plt.subplots(2, 2, figsize=(11, 12), constrained_layout=True)
    for ax, item in zip(axes.ravel(), rows):
        center = item['center']
        for segments, colour, label in ((item['headSections'], '#3478ac', 'actual lid/skin'),
                                         (item['eyeSections'], '#222222', 'actual closed globe')):
            ax.add_collection(LineCollection((segments[:, :, 1:]-center[1:])*1000,
                                             colors=colour, linewidths=1.5, label=label))
        p = (item['point'][1:]-center[1:])*1000
        f = (item['front'][1:]-center[1:])*1000
        b = (item['back'][1:]-center[1:])*1000
        ax.plot([f[0], b[0]], [f[1], b[1]], color='#d68e00', linestyle='--', linewidth=1, label='front/back ray interval')
        ax.scatter(*p, c='#b40020', s=28, zorder=4, label='inside lining point')
        ax.scatter([f[0], b[0]], [f[1], b[1]], facecolors='none', edgecolors='#d68e00', s=25, zorder=4)
        ax.set_xlim(-18, 14)
        ax.set_ylim(-13, 15)
        ax.set_aspect('equal')
        ax.grid(alpha=.3)
        ax.set_xlabel('Y relative to actual eye centre (mm; front is left)')
        ax.set_ylabel('Z relative to actual eye centre (mm)')
        ax.set_title(item['label']+' | global vertex '+str(item['globalId'])+'\n'+
                     'YZ radial deficit %.3f mm; exact nearest distance %.3f mm' %
                     (item['radialDeficit']*1000, item['nearestDistance']*1000))
        ax.legend(fontsize=8, loc='upper right')
    figure.savefig(output, dpi=140)
    plt.close(figure)


def run(observation_relative, output_relative, plot=True):
    project = project_path().resolve()
    observation_file = (project/observation_relative).resolve()
    output = (project/output_relative).resolve()
    for path in (observation_file, output):
        path.relative_to((project/'work/reconstruction-r02').resolve())
    if output.suffix!='.json' or output.exists() or output.with_suffix('.npz').exists() or output.with_suffix('.png').exists():
        raise RuntimeError('Fresh independent evidence output required')
    observation = json.loads(observation_file.read_text(encoding='utf8'))
    array_file = observation_file.with_suffix('.npz')
    if sha(array_file)!=observation['arraysSha256']:
        raise RuntimeError('Actual source observation arrays changed')
    with np.load(array_file) as data:
        arrays = {key: data[key].copy() for key in data.files}
    ray_radius = api('adult_blink_corner_plan_r11')['ray_radius']
    surface = api('upper_lid_surface_study_r11')
    ids = arrays['head_originalVertexIds']
    payload = {}
    record = {'source': observation['source'], 'sourceSha256': observation['sourceSha256'],
              'observation': observation_relative, 'observationSha256': sha(observation_file),
              'observationArraysSha256': sha(array_file), 'helperSha256': sha(__file__),
              'states': [], 'globes': {}, 'sourceSaved': False, 'geometryModified': False,
              'scope': 'Independent actual-triangle front/back containment and solid-angle proof for existing front-side eyelid lining penetration at neutral and full closure.',
              'interpretation': 'YZ radial deficits and front-depth distances are directional metrics, not minimum Euclidean penetration distances; the exact nearest-triangle distance is separately recorded.',
              'indexSetMeaning': 'Computed original global evaluated point/polygon/triangle index sets, not existing Blender named face sets.',
              'limits': ['Observes the source REST geometry only; no dynamic correction or appearance acceptance.',
                         'Only previously detected front-half radial deficits greater than 20 micrometres are tested; this is not a full face/eye collision audit.']}
    plot_rows = []
    for side in ('L', 'R'):
        record['globes'][side] = globe_contract(arrays['eye_'+side+'_world'], arrays['eye_'+side+'_triangles'])
        if record['globes'][side]['nonmanifoldEdges']:
            raise RuntimeError('Solid-angle interpretation requires the actual closed globe mesh')
    for state_index in (0, 5):
        state = observation['states'][state_index]
        world = arrays[state['headWorldArray']]
        triangles = arrays[state['trianglesArray']]
        normals = surface['normals'](world, triangles)
        polygon_ids = arrays[state['trianglePolygonIdsArray']]
        triangle_ids = arrays[state['originalTriangleIdsArray']]
        for side in ('L', 'R'):
            eye = arrays['eye_'+side+'_world']
            eye_triangles = arrays['eye_'+side+'_triangles']
            center = np.asarray(observation['eyes'][side]['centerWorld'])
            relative = world-center
            radius = np.linalg.norm(relative[:, 1:], axis=1)
            candidates = ((world[:, 0]>eye[:, 0].min()+.00005) & (world[:, 0]<eye[:, 0].max()-.00005) &
                          (relative[:, 1]<0.) & (radius<.019) & (abs(relative[:, 2])<.012))
            indices = np.flatnonzero(candidates)
            origins = np.tile(center, (len(indices), 1))
            origins[:, 0] = world[indices, 0]
            direction = relative[indices].copy()
            direction[:, 0] = 0.
            direction /= radius[indices, None]
            section_radius = ray_radius(origins, direction, eye, eye_triangles)
            gap = radius[indices]-section_radius
            negative = np.isfinite(gap) & (gap<-.00002)
            indices, gap = indices[negative], gap[negative]
            points = world[indices]
            front_origins = points.copy()
            front_origins[:, 1] = center[1]-.04
            back_origins = points.copy()
            back_origins[:, 1] = center[1]+.04
            forward = np.zeros_like(points)
            forward[:, 1] = 1.
            front_distance = ray_radius(front_origins, forward, eye, eye_triangles)
            back_distance = ray_radius(back_origins, -forward, eye, eye_triangles)
            front = front_origins+forward*front_distance[:, None]
            back = back_origins-forward*back_distance[:, None]
            contained = np.isfinite(front_distance) & np.isfinite(back_distance) & (points[:, 1]>front[:, 1]) & (points[:, 1]<back[:, 1])
            winding = winding_numbers(points, eye, eye_triangles)
            winding_inside = abs(winding)>.999999
            nearest, nearest_points, nearest_triangles = exact_nearest(points, eye, eye_triangles)
            inward = center-points
            inward /= np.linalg.norm(inward, axis=1)[:, None]
            normal_inward = np.sum(normals[indices]*inward, axis=1)
            affected_triangles = np.isin(triangles, indices).any(axis=1)
            tag = 'state_'+str(state_index)+'_'+side
            payload.update({tag+'_localVertexIds': indices, tag+'_globalVertexIds': ids[indices],
                            tag+'_world': points, tag+'_normals': normals[indices], tag+'_radialGapMeters': gap,
                            tag+'_frontWorld': front, tag+'_backWorld': back, tag+'_solidAngleWinding': winding,
                            tag+'_nearestDistanceMeters': nearest, tag+'_nearestWorld': nearest_points,
                            tag+'_nearestGlobeTriangleIds': nearest_triangles,
                            tag+'_affectedOriginalPolygonIds': np.unique(polygon_ids[affected_triangles]),
                            tag+'_affectedOriginalTriangleIds': triangle_ids[affected_triangles],
                            tag+'_affectedGlobalTriangles': ids[triangles[affected_triangles]]})
            worst = int(gap.argmin())
            i = indices[worst]
            point = points[worst]
            head_sections = surface['sections'](world, triangles, point[0])
            eye_sections = surface['sections'](eye, eye_triangles, point[0])
            payload[tag+'_worstHeadSectionsWorld'] = head_sections
            payload[tag+'_worstGlobeSectionsWorld'] = eye_sections
            neutral_height = float(arrays['state_0_head_world'][i, 2]-center[2])
            row = {'stateIndex': state_index, 'blink': state['L'], 'side': side,
                   'negativeRadialPointCount': len(indices), 'frontBackInsideCount': int(contained.sum()),
                   'solidAngleInsideCount': int(winding_inside.sum()), 'allInFrontHemisphere': bool(np.all(points[:, 1]<center[1])),
                   'normalYRange': [float(normals[indices, 1].min()), float(normals[indices, 1].max())],
                   'normalTowardGlobeCentreDotRange': [float(normal_inward.min()), float(normal_inward.max())],
                   'maximumWindingUnitResidual': float(np.max(abs(abs(winding)-1.))),
                   'maximumYZRadialDeficitMeters': float(-gap.min()),
                   'maximumExactNearestDistanceMeters': float(nearest.max()),
                   'affectedOriginalPolygons': int(len(payload[tag+'_affectedOriginalPolygonIds'])),
                   'worstRadialPoint': {'globalVertexId': int(ids[i]), 'world': point.tolist(), 'relativeToEyeCentre': (point-center).tolist(),
                                        'frontGlobeHitWorld': front[worst].tolist(), 'backGlobeHitWorld': back[worst].tolist(),
                                        'insideFrontDepthMeters': float(point[1]-front[worst, 1]),
                                        'insideBackDepthMeters': float(back[worst, 1]-point[1]),
                                        'solidAngleWinding': float(winding[worst]),
                                        'minimumEuclideanDistanceMeters': float(nearest[worst]),
                                        'closestGlobePointWorld': nearest_points[worst].tolist(),
                                        'closestGlobeTriangleId': int(nearest_triangles[worst]),
                                        'neutralHeightAboveEyeCentreMeters': neutral_height,
                                        'normalWorld': normals[i].tolist()},
                   'arrayPrefix': tag}
            if not contained.all() or not winding_inside.all():
                raise RuntimeError('A directional radial deficit was not independently confirmed inside the actual globe')
            record['states'].append(row)
            plot_rows.append({'label': side+' blink '+str(state['L']), 'center': center, 'point': point,
                              'front': front[worst], 'back': back[worst], 'globalId': int(ids[i]),
                              'radialDeficit': float(-gap[worst]), 'nearestDistance': float(nearest[worst]),
                              'headSections': head_sections, 'eyeSections': eye_sections})
            print('R11_INNER_LINING_PROOF '+json.dumps({'state': state_index, 'side': side, 'points': len(indices),
                  'inside': int(contained.sum()), 'windingInside': int(winding_inside.sum()), 'maxRadialMm': -float(gap.min())*1000,
                  'worstActualNearestMm': float(nearest[worst])*1000}), flush=True)
    np.savez_compressed(output.with_suffix('.npz'), **payload)
    record.update(arrays=str(output.with_suffix('.npz').relative_to(project)), arraysSha256=sha(output.with_suffix('.npz')))
    if plot:
        plot_sections(plot_rows, output.with_suffix('.png'))
        record.update(sectionsImage=str(output.with_suffix('.png').relative_to(project)), sectionsImageSha256=sha(output.with_suffix('.png')))
    with output.open('x', encoding='utf8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    run(args.observation, args.output, not args.no_plot)
