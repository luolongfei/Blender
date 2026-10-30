"""A single broad front-throat contour candidate, in actual REST coordinates.

No face-length/eye/nose/lip alteration. This is an artistic starting point,
not a formula for anatomy. Current surface direction defines the front mask.
"""
import json
from pathlib import Path
import runpy
import numpy as np

CONTROL = 'R11 chin throat contour'


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))


def section_check(before, after, triangles):
    study = runpy.run_path(str(Path(__file__).with_name('whole_face_proportion_study_r11.py')))
    heights = np.arange(1.280, 1.3350001, .0005)
    rows = []
    for x in (0., -.012, .012, -.022, .022):
        b = study['samples_at_z'](study['plane_segments'](before, triangles, 0, x+1e-6), heights)
        a = study['samples_at_z'](study['plane_segments'](after, triangles, 0, x+1e-6), heights)
        if b.shape != a.shape or len(b) != len(heights):
            raise RuntimeError('Complete corresponding actual throat sections required')
        db, da = np.diff(b[:, 1]), np.diff(a[:, 1])
        rows.append({'x': x, 'before': b.tolist(), 'after': a.tolist(),
            'priorReversedSteps': int((db > 0).sum()), 'afterReversedSteps': int((da > 0).sum()),
            'newReversedSteps': int(((db <= 0) & (da > 0)).sum()),
            'maximumForwardStepMeters': float(da.max())})
    return {'heightStepMeters': .0005, 'sections': rows,
        'newReversedSteps': sum(r['newReversedSteps'] for r in rows),
        'limitation': 'No added section turn is a useful shape guard, not proof that shading or appearance is acceptable.'}


def plan(world, faces, groups, triangles, amplitude=.004):
    w = np.asarray(world, float)
    if w.shape != (238889, 3) or np.asarray(faces).shape != (238672, 4):
        raise RuntimeError('Current complete fixed-L3 head required')
    if not .003 <= amplitude <= .004:
        raise ValueError('Reviewed artistic starting range is 3 to 4 mm')
    study = runpy.run_path(str(Path(__file__).with_name('whole_face_proportion_study_r11.py')))
    # Exact external front intersection; no whole-ring centroid/average or
    # generic cylinder is used as a substitute for the current front surface.
    surface_faces = faces[np.isin(groups, [1, 17, 22])]
    surface_tri = np.concatenate((surface_faces[:, [0, 1, 2]], surface_faces[:, [0, 2, 3]]))
    segments = study['plane_segments'](w, surface_tri, 0, .000001)
    profile = study['samples_at_z'](segments, np.arange(1.275, 1.3350001, .00025))
    if len(profile) < 235:
        raise RuntimeError('Current complete front-throat profile not found')
    z = w[:, 2]
    rise = smooth((z-1.280)/.032)
    fall = 1.-smooth((z-1.312)/.023)
    vertical = rise*fall
    # The actual +/-22 mm front sections are nearly straight. Taper well
    # before them: a broad full-strength plateau there creates a new waist.
    lateral = 1.-smooth((np.abs(w[:, 0])-.008)/.028)
    front_y = np.interp(z, profile[:, 2], profile[:, 1])
    # Full posterior shift over the actual forward 6 mm, a broad smooth fade
    # over the following 18 mm, then exact zero. Rear neck remains untouched.
    anterior = 1.-smooth((w[:, 1]-front_y-.006)/.018)
    weights = vertical*lateral*anterior
    weights[(z <= 1.280) | (z >= 1.335) | (np.abs(w[:, 0]) >= .036)] = 0.
    offset = np.zeros_like(w)
    offset[:, 1] = amplitude*weights
    active = offset[:, 1] != 0.
    oral_eyes_ears = np.unique(faces[np.isin(groups, [2, 3, 4, 5, 7, 8])])
    if np.any(offset[oral_eyes_ears]) or np.any(offset[3214]):
        raise RuntimeError('Field overlaps protected anatomy; fix domain rather than override motion')
    qapi = runpy.run_path(str(Path(__file__).with_name('upper_lid_roll_install_r11.py')))
    quality = qapi['quality'](w, w+offset, triangles, active)
    if quality['nonpositiveNormals'] or quality['minimumNormalAlignment'] < .85 or quality['areaRatioRange'][0] < .65 or quality['areaRatioRange'][1] > 1.5:
        raise RuntimeError('Broad contour field introduces excessive local strain: '+json.dumps(quality))
    sections = section_check(w, w+offset, triangles)
    if sections['newReversedSteps']:
        raise RuntimeError('Broad contour creates a new front-throat section waist')
    ids = np.flatnonzero(active).astype(np.int32)
    report = {'control': CONTROL, 'amplitudeMeters': amplitude, 'activeVertices': len(ids),
        'maximumWorldOffsetMeters': float(offset[:, 1].max()),
        'activeWorldBounds': [w[active].min(0).tolist(), w[active].max(0).tolist()],
        'supportZ': [1.280, 1.335], 'peakZ': 1.312, 'lateralCoreMeters': .008,
        'lateralOuterMeters': .036, 'frontMaskFullDepthMeters': .006, 'frontMaskFadeWidthMeters': .018,
        'pureQuality': quality, 'sectionCheck': sections,
        'visibleIntent': 'Recede one broad upper-front throat region to clarify the chin-underbody / throat turn; preserve chin front, lips, nose, eyes, ears and low neck seam.',
        'notAttempted': ['Shortening or tilting the whole head/neck', 'Changing eye or ear scale', 'Changing lip closure or default expression'],
        'appearanceAccepted': False, 'animationAcceptance': False}
    return offset, {'originalVertexIds': ids, 'worldOffsets': offset[ids], 'frontProfile': profile, 'weights': weights}, report


if __name__ == '__main__':
    p = Path(__file__).resolve().parents[2]
    d = p/'work/reconstruction-r02'
    a = np.load(d/'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz')
    g = np.load(d/'head-neck-current-posture-r11-v2.npz')['rest_head_faceGroups']
    offset, arrays, report = plan(a['after_0.0_1.0World'], a['fullFaces'], g, a['after_0.0_1.0Triangles'])
    np.savez_compressed(d/'chin-throat-contour-plan-r11.npz', **arrays)
    (d/'chin-throat-contour-plan-r11.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k != 'sectionCheck'}, indent=2))
