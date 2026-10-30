"""Offline full-affine optical metric audit from the actual current eye export.

The iris sampling is reconstructed for this algebra/clearance test; production
prepare always reads actual iris vertices. Synthetic transforms are clearly
labelled stress cases, not evidence of the newer source's current matrix.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def front_hits(triangles, xz):
    a = triangles[:, 0][:, [0, 2]]; v = triangles[:, 1][:, [0, 2]] - a; u = triangles[:, 2][:, [0, 2]] - a
    det = v[:, 0] * u[:, 1] - u[:, 0] * v[:, 1]; safe = np.where(abs(det) > 1e-14, det, np.nan)
    result = []
    for start in range(0, len(xz), 128):
        d = xz[start:start + 128, None, :] - a[None, :, :]
        beta = (d[:, :, 0] * u[None, :, 1] - u[None, :, 0] * d[:, :, 1]) / safe
        gamma = (v[None, :, 0] * d[:, :, 1] - d[:, :, 0] * v[None, :, 1]) / safe
        alpha = 1 - beta - gamma
        valid = (alpha >= -1e-7) & (beta >= -1e-7) & (gamma >= -1e-7)
        y = alpha * triangles[None, :, 0, 1] + beta * triangles[None, :, 1, 1] + gamma * triangles[None, :, 2, 1]
        result.extend(np.where(valid, y, np.inf).min(1))
    return np.asarray(result)


def nearest_signed(triangles, points):
    a = triangles[:, 0]; b = triangles[:, 1]; c = triangles[:, 2]
    ab = b - a; ac = c - a; normal = np.cross(ab, ac); normal /= np.linalg.norm(normal, axis=1)[:, None]
    aa = np.einsum('ij,ij->i', ab, ab); cc = np.einsum('ij,ij->i', ac, ac); cross = np.einsum('ij,ij->i', ab, ac)
    det = aa * cc - cross * cross; lengths = []; signs = []
    for start in range(0, len(points), 64):
        p = points[start:start + 64]; v = p[:, None, :] - a[None, :, :]
        height = np.einsum('kij,ij->ki', v, normal); projected = v - height[:, :, None] * normal
        d20 = np.einsum('kij,ij->ki', projected, ab); d21 = np.einsum('kij,ij->ki', projected, ac)
        u = (cc * d20 - cross * d21) / det; w = (aa * d21 - cross * d20) / det
        inside = (u >= 0) & (w >= 0) & (u + w <= 1)
        best = np.where(inside, height * height, np.inf)
        best_delta = height[:, :, None] * normal
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x; q = p[:, None, :] - x
            t = np.clip(np.einsum('kij,ij->ki', q, edge) / np.einsum('ij,ij->i', edge, edge), 0, 1)
            delta = q - t[:, :, None] * edge
            distance = np.einsum('kij,kij->ki', delta, delta); better = distance < best
            best = np.where(better, distance, best); best_delta = np.where(better[:, :, None], delta, best_delta)
        ids = best.argmin(1); rows = np.arange(len(p)); lengths.extend(np.sqrt(best[rows, ids]))
        signs.extend(np.einsum('ij,ij->i', best_delta[rows, ids], normal[ids]))
    return np.asarray(lengths), np.asarray(signs)


def run():
    p = project_path(); folder = p / 'work/reconstruction-r02'
    helper = Path(__file__).with_name('adult_eye_depth_r11_v2.py'); api = runpy.run_path(str(helper))
    old = runpy.run_path(str(Path(__file__).with_name('adult_eye_depth_r11.py')))
    meta = json.loads((folder / 'adult-blink-current-shape-r11.json').read_text())
    arrays_path = folder / 'adult-blink-current-shape-r11.npz'
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != meta['arraysSha256']: raise RuntimeError('Actual observation changed')
    arrays = np.load(arrays_path); scale = meta['headMatrixWorld'][0][0]
    az, ax = np.deg2rad([23., -13.]); rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rotation = rz @ rx
    cases = [('observed-rest-uniform', np.eye(3), False), ('synthetic-rigid-rotation', rotation, False),
        ('synthetic-anisotropic', rotation @ np.diag([1.025, .985, 1.01]), False),
        ('synthetic-shear-separate-iris-matrix', rotation @ np.array([[1.01, .008, 0], [.018, .99, .012], [0, 0, 1.015]]), True)]
    rows = []
    for side in ('L', 'R'):
        world = arrays['eye_' + side + '_world']; center = (world.min(0) + world.max(0)) * .5
        globe = (world - center) / scale; triangles = arrays['eye_' + side + '_triangles']
        radius = .245 * ((np.ptp(world[:, 0]) + np.ptp(world[:, 2])) * .5) / scale
        angles = np.arange(96) * 2 * np.pi / 96
        xz = np.vstack([np.zeros((1, 2)), [[np.cos(a) * radius * r / 16, np.sin(a) * radius * r / 16] for r in range(1, 17) for a in angles]])
        front = front_hits(globe[triangles], xz)
        if not np.isfinite(front).all(): raise RuntimeError('Reconstructed source cap misses actual donor globe')
        iris = np.column_stack((xz[:, 0], front - .0000615 / scale, xz[:, 1]))
        for label, change, separate in cases:
            matrix = np.eye(4); matrix[:3, :3] = change * scale; matrix[:3, 3] = center
            im = matrix.copy()
            if separate: im[:3, 3] += np.array([.00000008, -.00000004, .00000006])
            result, plan = api['plan_iris_depth'](globe, iris, matrix, im)
            globe_world = api['_transform'](globe, matrix); iris_world = api['_transform'](result, im)
            frame = plan['opticalFrame']; q = np.asarray(frame['worldBasisColumns']); origin = np.asarray(frame['centerWorld'])
            gm = (globe_world - origin) @ q; ip = (iris_world - origin) @ q
            first = front_hits(gm[triangles], ip[:, [0, 2]]); gap = ip[:, 1] - first
            nearest, signed = nearest_signed(globe_world[triangles], iris_world)
            if not np.isfinite(gap).all() or gap.min() < .00025 or gap.max() > .0038 or nearest.min() < .0001 or signed.max() > -.000025:
                raise RuntimeError('Full-affine world clearance audit failed: ' + str((side, label, gap.min(), gap.max(), nearest.min(), signed.max())))
            row = dict(side=side, transformCase=label, plan=plan,
                worldRayGapMetersQuantiles=np.quantile(gap, [0, .5, 1]).tolist(),
                worldNearestMetersQuantiles=np.quantile(nearest, [0, .5, 1]).tolist(), maximumOutwardSignedMeters=float(signed.max()))
            if label == 'observed-rest-uniform':
                previous, _ = old['plan_iris_depth'](globe, iris, scale)
                row['priorUniformPlanCoordinateMaxDifference'] = float(np.max(np.abs(previous - result)))
                if row['priorUniformPlanCoordinateMaxDifference'] > 1e-7: raise RuntimeError('Uniform formula changed unexpectedly')
            rows.append(row)
    report = dict(helper='work/local-model/adult_eye_depth_r11_v2.py', helperSha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        sourceObservation=meta['source'], sourceObservationSha256=meta['sourceSha256'], arraysSha256=meta['arraysSha256'],
        verificationHelperSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=rows, productionBlenderExecuted=False, actualNewerB8MatrixObserved=False,
        limitations=['Reconstructed iris cap samples are offline evidence only; actual preparation reads the current iris geometry and full matrices.',
            'Synthetic affine cases test world-metric mathematics; they are not asserted to be the newer saved character transform.',
            'Actual Blender BVH, shader, saved/reopened source and rendered optical appearance remain required.'])
    output = folder / 'adult-eye-depth-v2-metric-offline.json'; output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), helperSha256=report['helperSha256'], cases=len(rows),
        smallestWorldNearestMeters=min(r['worldNearestMetersQuantiles'][0] for r in rows),
        largestTransverseErrorMeters=max(r['plan']['worldTransversePreservationErrorMeters'] for r in rows)), indent=2))
    return report


if __name__ == '__main__': run()
