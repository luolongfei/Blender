"""Read-only sections of the actual evaluated R11 upper eyelid observation.

No Blender access, deformation, key creation or acceptance claim. A visible
globe boundary is explicitly not treated as a certified inner wet/contact line.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sections(points, triangles, x):
    """Actual triangle/plane intersections; no smoothing or fitted eyelid."""
    p = np.asarray(points)[np.asarray(triangles)]
    p = p[(p[:, :, 0].min(1)<x) & (p[:, :, 0].max(1)>x)]
    output = []
    for triangle in p:
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = triangle[a], triangle[b]
            if (pa[0]-x)*(pb[0]-x)<0:
                hits.append(pa+(pb-pa)*(x-pa[0])/(pb[0]-pa[0]))
        if len(hits)==2:
            output.append(hits)
    return np.asarray(output, float).reshape((-1, 2, 3))


def normals(points, triangles):
    p = points[triangles]
    normal = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    result = np.zeros_like(points)
    for index in range(3):
        np.add.at(result, triangles[:, index], normal)
    return result/np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-15)


def run(observation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    project = project_path()
    observation = (project/observation).resolve()
    output = (project/output).resolve()
    observation.relative_to((project/'work/reconstruction-r02').resolve())
    output.relative_to((project/'work/reconstruction-r02').resolve())
    if output.exists() or output.with_suffix('.png').exists():
        raise RuntimeError('Fresh independent study output required')
    meta = json.loads(observation.read_text(encoding='utf-8'))
    array_path = observation.with_suffix('.npz')
    if sha(array_path)!=meta['arraysSha256']:
        raise RuntimeError('Actual observation array checksum mismatch')
    data = np.load(array_path)
    world = data['head_world']; triangles = data['head_triangles']
    normal = normals(world, triangles)
    result = dict(sourceObservation=str(observation.relative_to(project)),
        observationSha256=sha(observation), arraysSha256=sha(array_path),
        actualSource=meta['source'], actualSourceSha256=meta['sourceSha256'],
        helperSha256=sha(__file__), sourceSaved=False, deformationInstalled=False,
        appearanceAccepted=False, eyes={},
        limits=['These are actual evaluated triangle sections, not a fitted replacement head.',
                'Visible front boundary and inner physical contact/wet line are distinct.',
                'Sphere clearance is conservative for this non-spherical eye and not a signed anatomical contact distance.',
                'New geometry, optical materials, closure correctives and rendered appearance require their own current-source checks.'])
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for row, side in enumerate(('L', 'R')):
        eye = meta['eyes'][side]
        center = np.asarray(eye['centerWorld'])
        globe = data['eye_'+side+'_world']
        radius = np.linalg.norm(globe-center, axis=1).max()
        contour = [p for p in eye['contour'] if not p['upper']['unboundedAtGlobeRim'] and p['upper'].get('headOutside120Micrometers')]
        cx = np.asarray([p['x'] for p in contour])
        cz = np.asarray([p['upper']['worldZ'] for p in contour])
        cy = np.asarray([p['upper']['headOutside120Micrometers']['world'][1] for p in contour])
        dz = world[:, 2]-np.interp(world[:, 0], cx, cz)
        dy = world[:, 1]-np.interp(world[:, 0], cx, cy)
        clearance = np.linalg.norm(world-center, axis=1)-radius
        selection = (world[:, 0]>cx.min()+.002)&(world[:, 0]<cx.max()-.002)&(abs(dy)<.005)&(normal[:, 1]<-.2)
        bands = []
        for lo, hi in ((0., .00025), (.00025, .0008), (.0008, .0015), (.0015, .0025), (.0025, .004)):
            ids = np.flatnonzero(selection&(dz>=lo)&(dz<hi))
            bands.append(dict(heightAboveVisibleBoundaryMeters=[lo, hi], count=int(len(ids)),
                conservativeSphereClearanceMetersQuantiles=np.quantile(clearance[ids], [0, .1, .5, .9, 1]).tolist() if len(ids) else [],
                frontOffsetFromBoundaryMetersQuantiles=np.quantile(dy[ids], [0, .1, .5, .9, 1]).tolist() if len(ids) else []))
        slices = []
        for column, offset in enumerate((-.004, 0., .004)):
            x = center[0]+offset
            z = float(np.interp(x, cx, cz)); y = float(np.interp(x, cx, cy))
            head_segments = sections(world, triangles, x)
            eye_segments = sections(globe, data['eye_'+side+'_triangles'], x)
            midpoint = head_segments.mean(1)
            keep = (midpoint[:, 2]>z-.001)&(midpoint[:, 2]<z+.006)&(abs(midpoint[:, 1]-y)<.006)
            head_segments = head_segments[keep]
            ax = axes[row, column]
            origin = np.array([center[1], z])
            for segments, color, label in ((eye_segments, '#707070', 'actual globe'), (head_segments, '#1565c0', 'actual head')):
                ax.add_collection(LineCollection((segments[:, :, 1:3]-origin)*1000, colors=color, linewidths=1.5, label=label))
            ax.axhline(0, color='#ab47bc', linestyle='--', linewidth=.8, label='front-visible upper boundary')
            ax.scatter([(y-center[1])*1000], [.12], color='#c62828', s=16, zorder=5, label='actual +0.12mm ray hit')
            ax.set_xlim((y-center[1]-.003)*1000, (y-center[1]+.004)*1000)
            ax.set_ylim(-1, 6)
            ax.set_aspect('equal'); ax.grid(alpha=.2)
            ax.set_title(f'{side}: X offset {offset*1000:+.1f} mm')
            ax.set_xlabel('world Y relative to globe center (mm); front at left')
            ax.set_ylabel('height above visible upper boundary (mm)')
            if row==0 and column==0: ax.legend(fontsize=7)
            slices.append(dict(xWorld=float(x), visibleUpperZ=z,
                headOutside120MicrometersWorldY=y, headSegmentsWorld=head_segments.tolist(),
                globeSegmentsWorld=eye_segments.tolist()))
        result['eyes'][side] = dict(centerWorld=center.tolist(), conservativeBoundingRadiusMeters=float(radius), bands=bands, slices=slices)
    fig.suptitle('Actual R11 upper-lid sections | source 491b6c85 | read-only; no anatomy acceptance', fontsize=13)
    image = output.with_suffix('.png')
    fig.savefig(image, dpi=150)
    plt.close(fig)
    result['sectionPlot']=str(image.relative_to(project)); result['sectionPlotSha256']=sha(image)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), plot=str(image), helperSha256=result['helperSha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--observation', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    run(args.observation, args.output)
