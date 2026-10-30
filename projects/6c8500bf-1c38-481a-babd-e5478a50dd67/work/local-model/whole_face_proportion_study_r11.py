"""Offline current L3 shape measurements; artistic references are not calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[2]
D = P/'work/reconstruction-r02'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def plane_segments(world, triangles, axis, value):
    p = world[triangles]
    d = p[:, :, axis]-value
    p = p[(d.min(1) < 0) & (d.max(1) > 0)]
    d = p[:, :, axis]-value
    out = []
    for points, dd in zip(p, d):
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if dd[a]*dd[b] < 0:
                hits.append(points[a]+(points[b]-points[a])*(-dd[a])/(dd[b]-dd[a]))
        if len(hits) == 2:
            out.append(hits)
    return np.asarray(out, float)


def samples_at_z(segments, heights):
    result = []
    for z in heights:
        d = segments[:, :, 2]-z
        s = segments[(d[:, 0]*d[:, 1] <= 0) & (np.abs(d[:, 1]-d[:, 0]) > 1e-12)]
        if not len(s):
            continue
        t = (z-s[:, 0, 2])/(s[:, 1, 2]-s[:, 0, 2])
        hit = s[:, 0]+t[:, None]*(s[:, 1]-s[:, 0])
        result.append(hit[np.argmin(hit[:, 1])])
    return np.asarray(result)


def main():
    actual_path = D/'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz'
    capture_path = D/'head-neck-current-posture-r11-v2.npz'
    actual = np.load(actual_path)
    capture = np.load(capture_path)
    contract = json.loads((D/'head-neck-current-posture-r11-v2.json').read_text())
    eye_report = json.loads((D/'reference-adult-outer-eye-radial-r11-v2.outer-eye.json').read_text())
    w = actual['after_0.0_1.0World']
    triangles = actual['after_0.0_1.0Triangles']
    polygon_ids = actual['beforeWarm0.0TrianglePolygonIds']
    faces = actual['fullFaces']
    groups = capture['rest_head_faceGroups']
    assert np.array_equal(capture['rest_head_landmarkWorld'], w[capture['rest_head_landmarkIds']])
    assert np.array_equal(capture['rest_head_boundaryWorld'], w[capture['rest_head_boundaryVertexIds']])
    assert np.array_equal(actual['beforeWarm0.0Triangles'], triangles)
    assert contract['sourceSha256'] == '29e3e6b59a08487643d28642975a6ab1f92ca4af830f484af291f25434e08df8'
    assert contract['technicalAcceptance']
    external = np.isin(groups[polygon_ids], [1, 17, 22])
    surface_triangles = triangles[external]
    sagittal = plane_segments(w, surface_triangles, 0, .000001)
    profile = samples_at_z(sagittal, np.arange(1.2455, 1.4800001, .00025))
    eye = eye_report['currentNeutralContours']
    centers = np.asarray([eye[s]['centerWorld'] for s in ['L', 'R']])
    eye_height = float(centers[:, 2].mean())
    chin = w[3214]
    crown = w[np.argmax(w[:, 2])]
    report = {
        'sourceSha256': contract['sourceSha256'],
        'actualArraysSha256': sha(actual_path), 'currentCaptureArraysSha256': sha(capture_path),
        'capturedLandmarksAndBoundaryExact': True,
        'coordinateSpace': 'REST world meters; eyes retained from actual pre-radial contours. Radial key reset to zero before seal; seal does not affect eye region.',
        'limitations': ['No target dimensions are inferred from generated references.', 'Profile points are exact triangle-plane intersections followed by frontmost-Y selection; they are not automatically anatomical landmarks.', 'Ear bounds are true face-set surface bounds; frontal skin widths use an explicit Y cutoff and are not a claimed bizygomatic measurement.', 'The selected chin point is actual L3 vertex 3214 nearest existing cage seed 7972, not a labelled anatomical menton.'],
        'crown': {'vertexId': int(np.argmax(w[:, 2])), 'world': crown.tolist()},
        'chinAnchor': {'vertexId': 3214, 'world': chin.tolist()},
        'eyeCenters': centers.tolist(),
        'eyeCenterSpacingMm': float(np.linalg.norm(centers[0]-centers[1])*1000),
        'crownToChinVerticalMm': float((crown[2]-chin[2])*1000),
        'eyeCenterToChinVerticalMm': float((eye_height-chin[2])*1000),
        'eyeOpening': {}, 'ears': {}, 'skinWidthsByHeight': [], 'frontProfile': profile.tolist(),
    }
    for side in ['L', 'R']:
        report['eyeOpening'][side] = {'widthMm': eye[side]['frontVisibleWidthMeters']*1000, 'apertureMaximumMm': max(c['upper']['worldZ']-c['lower']['worldZ'] for c in eye[side]['contour'])*1000, 'globeDiametersMm': (np.asarray(eye[side]['diametersWorld'])*1000).tolist()}
    for group, side in [(4, 'positiveX'), (5, 'negativeX')]:
        ids = np.unique(faces[groups == group])
        ear = w[ids]
        report['ears'][side] = {'surfaceVertices': len(ids), 'bounds': [ear.min(0).tolist(), ear.max(0).tolist()], 'heightMm': float(np.ptp(ear[:, 2])*1000)}
    for z in [1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.40, 1.41, 1.42, eye_height, 1.44]:
        s = plane_segments(w, surface_triangles, 2, z).reshape(-1, 3)
        row = {'z': float(z), 'anteriorCutWidthsMm': {}}
        for ycut in [-.02, -.04, -.06]:
            points = s[s[:, 1] < ycut]
            row['anteriorCutWidthsMm'][str(ycut)] = float(np.ptp(points[:, 0])*1000) if len(points) else None
        report['skinWidthsByHeight'].append(row)
    nose = profile[(profile[:, 2] > 1.385) & (profile[:, 2] < 1.435)]
    tip = nose[np.argmin(nose[:, 1])]
    report['noseFrontmostInSpecifiedWindow'] = {'windowZ': [1.385, 1.435], 'point': tip.tolist()}
    below = profile[(profile[:, 2] >= 1.3845) & (profile[:, 2] <= 1.388)]
    subnasal = below[np.argmax(below[:, 1])]
    report['subnasalProfileTurnCandidate'] = {'windowZ': [1.3845, 1.388], 'point': subnasal.tolist(), 'verticalToChinMm': float((subnasal[2]-chin[2])*1000)}
    report['profileInterpretation'] = 'Use full sampled curve and actual renders to select a coordinated shape candidate; these values do not prescribe an ideal ethnicity or validate appearance.'
    path = D/'whole-face-proportion-study-r11.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf8')
    np.savez_compressed(path.with_suffix('.npz'), sagittalSegments=sagittal, frontProfile=profile)
    print(json.dumps({k: v for k, v in report.items() if k not in ['frontProfile', 'limitations', 'eyeOpening']}, indent=2))


if __name__ == '__main__':
    main()
