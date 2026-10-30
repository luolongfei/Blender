"""Offline Pillow contact sheet of actual R11 native leg-foot FK captures.

No Blender imports, launches, model edits or synthetic pose data. The actual
left and right arrays are projected independently. Four rows (each side's
lateral/medial views) share one metric scale and fixed cameras across all six
states. The software depth-sorted triangles are technical diagrams, not
Blender renders or continuous-contact/appearance acceptance.
"""
import argparse
import hashlib
import json
import math
import os
import runpy
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

NATIVE_HELPER = 'work/local-model/foot_leg_native_trial_r11.py'
NATIVE_HELPER_SHA = '45b06a7bf873fc6296c2145c17a676caf666554c224a0ed16ca0e46508dc6670'
CASES = ('REST', 'neutral_fk', 'ankle_dorsiflex_15', 'ankle_plantarflex_25', 'knee_flex_45', 'toe_flex_25')
TITLES = ('REST', 'Neutral FK', 'Ankle dorsiflex 15 deg', 'Ankle plantarflex 25 deg', 'Knee flex 45 deg', 'Toe flex 25 deg')
VIEWS = (('L', 'lateral', 1), ('L', 'medial', -1), ('R', 'lateral', -1), ('R', 'medial', 1))
CELL_W, CELL_H = 480, 520
HEADER_H, FOOTER_H = 100, 112
PAD_X, PLOT_TOP, PLOT_BOTTOM = 24, 52, 432


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def path_for(relative):
    path = (project_path() / relative.replace('\\', '/')).resolve()
    path.relative_to(project_path().resolve())
    return path


def dependencies():
    helper = path_for(NATIVE_HELPER)
    if sha(helper) != NATIVE_HELPER_SHA:
        raise RuntimeError('Frozen native capture helper changed')
    # This exact pinned helper defines functions only; it does not import or
    # execute Blender at module scope. Do not invoke its run()/surface().
    api = runpy.run_path(str(helper))
    if tuple(row[0] for row in api['CASE_SPECS']) != CASES:
        raise RuntimeError('Frozen six-state capture contract differs')
    for relative, digest in api['PINS'].items():
        if sha(path_for(relative)) != digest:
            raise RuntimeError('Frozen capture dependency changed: ' + relative)
    return api['PINS']


def read_probe(relative, expected_sha):
    pins = dependencies()
    path = path_for(relative)
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise RuntimeError('Explicit actual probe SHA-256 mismatch')
    report = json.loads(path.read_text(encoding='utf8'))
    if (report.get('mode') != 'actual-native-isolated-FK-trial'
            or report.get('helperSha256') != NATIVE_HELPER_SHA
            or report.get('dependencies') != pins
            or report.get('restorationExact') is not True
            or report.get('savedBlend') is not False
            or report.get('sourceGeometryEdited') is not False):
        raise RuntimeError('Expected restored actual native trial, not an offline candidate or combined REST-only probe')
    if not isinstance(report.get('sourceSha256'), str) or len(report['sourceSha256']) != 64:
        raise RuntimeError('Actual capture lacks explicit source identity')
    arrays_path = path_for(report['arrays'])
    if sha(arrays_path) != report['arraysSha256']:
        raise RuntimeError('Actual probe array SHA-256 mismatch')
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    records = report['cases']
    if tuple(row['name'] for row in records) != CASES:
        raise RuntimeError('All six unique actual finite states must be present in capture order')
    for side, word in (('L', 'left'), ('R', 'right')):
        tri, roi = arrays['triangles_' + side], arrays['roiTriangleIds_' + side]
        if tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu' or not len(tri):
            raise RuntimeError('Invalid actual triangle array')
        if (roi.ndim != 1 or roi.dtype.kind not in 'iu' or not len(roi)
                or roi.min() < 0 or roi.max() >= len(tri) or len(np.unique(roi)) != len(roi)):
            raise RuntimeError('Invalid fixed REST region of interest')
        count = None
        for record in records:
            row = record['sides'][side]
            expected_prefix = record['name'] + '_' + side
            if row['arrayPrefix'] != expected_prefix:
                raise RuntimeError('Actual pose prefix does not match its named state/side')
            points = arrays[expected_prefix + '_world']
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise RuntimeError('Invalid actual evaluated world positions')
            if count is None:
                count = len(points)
            if len(points) != count or tri.min() < 0 or tri.max() >= count:
                raise RuntimeError('Actual six-state native topology differs')
            if row['vertices'] != count or row['triangles'] != len(tri) or row['roiTriangles'] != len(roi):
                raise RuntimeError('Actual report counts disagree with arrays')
            pairs = arrays[expected_prefix + '_candidatePairs']
            if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
                raise RuntimeError('Invalid captured candidate-pair array')
            if len(pairs) and (pairs.min() < 0 or pairs.max() >= len(tri)):
                raise RuntimeError('Captured candidate pair refers outside actual triangles')
            if len(pairs) != row['localSurfaceCandidates']['nonAdjacentCandidatePairs']:
                raise RuntimeError('Actual candidate-pair count differs')
            for suffix in ('Foot', 'Toes'):
                bone = row['actualPoseBones'][word + suffix]
                for key in ('headWorld', 'tailWorld'):
                    values = np.asarray(bone[key], dtype=float)
                    if values.shape != (3,) or not np.isfinite(values).all():
                        raise RuntimeError('Actual pose joint endpoints missing')
        expected_roi = np.flatnonzero(np.min(arrays['REST_' + side + '_world'][tri, 2], axis=1) < .355)
        if not np.array_equal(roi, expected_roi):
            raise RuntimeError('Captured ROI differs from fixed native REST definition')
    return report, arrays


@lru_cache(maxsize=12)
def font(size):
    candidates = []
    if os.environ.get('WINDIR'):
        candidates += [str(Path(os.environ['WINDIR']) / 'Fonts' / 'arial.ttf'),
                       str(Path(os.environ['WINDIR']) / 'Fonts' / 'segoeui.ttf')]
    candidates.append('DejaVuSans.ttf')
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def framing(arrays):
    # One world Y/Z extent from both real sides and all finite states. Neither
    # per-pose recentering nor per-pose auto zoom is permitted.
    lower = np.array([math.inf, math.inf])
    upper = -lower
    for side in ('L', 'R'):
        used = np.unique(arrays['triangles_' + side][arrays['roiTriangleIds_' + side]])
        for case in CASES:
            yz = arrays[case + '_' + side + '_world'][used, 1:3]
            lower = np.minimum(lower, yz.min(axis=0))
            upper = np.maximum(upper, yz.max(axis=0))
    lower -= np.array([.020, .015])
    upper += np.array([.020, .015])
    span = upper - lower
    if not np.isfinite(span).all() or np.any(span <= 0):
        raise RuntimeError('Actual ROI framing is degenerate')
    scale = min((CELL_W - 2 * PAD_X) / span[0], (PLOT_BOTTOM - PLOT_TOP) / span[1])
    return {'centreWorldY': float((lower[0] + upper[0]) * .5),
            'centreWorldZ': float((lower[1] + upper[1]) * .5),
            'pixelsPerMetre': float(scale), 'worldYZBoundsWithMargin': [lower.tolist(), upper.tolist()],
            'orthographic': True, 'sameMetricScaleAll24Panels': True,
            'cameraDoesNotFollowPose': True, 'rightSideCreatedByMirroringLeft': False}


def project(points, camera_sign, frame):
    points = np.asarray(points, dtype=float)
    x = CELL_W * .5 + camera_sign * (points[:, 1] - frame['centreWorldY']) * frame['pixelsPerMetre']
    y = (PLOT_TOP + PLOT_BOTTOM) * .5 - (points[:, 2] - frame['centreWorldZ']) * frame['pixelsPerMetre']
    return np.column_stack((x, y))


def dashed(draw, start, end, fill, width=1, segment=8):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length == 0:
        return
    for offset in np.arange(0, length, segment * 2):
        a, b = start + (end - start) * (offset / length), start + (end - start) * (min(offset + segment, length) / length)
        draw.line([tuple(a), tuple(b)], fill=fill, width=width)


def draw_panel(report, arrays, record, side, view, camera_sign, frame):
    image = Image.new('RGB', (CELL_W, CELL_H), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CELL_W - 1, CELL_H - 1), outline=(210, 214, 221))
    index = CASES.index(record['name'])
    draw.text((16, 8), TITLES[index], fill=(24, 32, 45), font=font(20))
    draw.text((16, 31), side + ' ' + view + ' | camera ' + ('+X' if camera_sign == 1 else '-X'), fill=(75, 85, 102), font=font(15))
    row = record['sides'][side]
    points = arrays[row['arrayPrefix'] + '_world']
    tri = arrays['triangles_' + side][arrays['roiTriangleIds_' + side]]
    vertices = points[tri]
    xy = project(points, camera_sign, frame)
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    length = np.linalg.norm(normals, axis=1)
    valid = length > 1e-16
    normals[valid] /= length[valid, None]
    light = np.array([.55 * camera_sign, -.25 * camera_sign, .8])
    light /= np.linalg.norm(light)
    illumination = np.clip(.40 + .60 * np.maximum(normals @ light, 0), 0, 1)
    # Uniform diagnostic colour is not a material or new skin colour.
    base = np.array([221, 184, 170])
    colour = np.clip(base[None, :] * illumination[:, None], 0, 255).astype(np.uint8)
    # A painter's depth sort is explicitly approximate for crossing triangles.
    # Actual collision evidence remains the captured triangle data, not pixels.
    order = np.argsort(camera_sign * vertices[:, :, 0].mean(axis=1), kind='stable')
    for face in order:
        draw.polygon([tuple(v) for v in xy[tri[face]]], fill=tuple(int(c) for c in colour[face]))
    datum = float(report['sides'][side]['axes']['bareSoleDatumWorldZ'])
    ref_y = project(np.array([[0, frame['centreWorldY'], datum]]), camera_sign, frame)[0, 1]
    if PLOT_TOP <= ref_y <= PLOT_BOTTOM:
        dashed(draw, (12, ref_y), (CELL_W - 12, ref_y), (170, 177, 185))
    word = 'left' if side == 'L' else 'right'
    foot, toes = [row['actualPoseBones'][word + name] for name in ('Foot', 'Toes')]
    joint_world = np.array([foot['headWorld'], toes['headWorld'], toes['tailWorld']])
    joint_xy = project(joint_world, camera_sign, frame)
    draw.line([tuple(v) for v in joint_xy], fill=(28, 106, 142), width=2)
    for point, label in zip(joint_xy[:2], ('A', 'T')):
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(246, 251, 255), outline=(28, 106, 142), width=2)
        draw.text((x + 6, y - 8), label, fill=(20, 84, 116), font=font(14))
    bar = .05 * frame['pixelsPerMetre']
    draw.line((18, 450, 18 + bar, 450), fill=(54, 63, 75), width=3)
    draw.text((18, 455), '50 mm', fill=(54, 63, 75), font=font(13))
    count = row['localSurfaceCandidates']['nonAdjacentCandidatePairs']
    draw.text((148, 449), 'BVH pairs: ' + str(count) + ' (unconfirmed)', fill=(91, 77, 57), font=font(14))
    draw.text((148, 470), 'ROI degenerate triangles: ' + str(row['roiDegenerateTriangles']), fill=(75, 85, 102), font=font(14))
    arrow_left = 18 if camera_sign == 1 else 92
    arrow_right = 92 if camera_sign == 1 else 18
    draw.line((arrow_left, 499, arrow_right, 499), fill=(96, 105, 119), width=1)
    draw.line((arrow_right - 6 * camera_sign, 495, arrow_right, 499,
               arrow_right - 6 * camera_sign, 503), fill=(96, 105, 119), width=1)
    # The arrow points toward world +Y (posterior); text records the convention
    # instead of visually relabelling an actual side as its mirror.
    draw.text((106, 490), '+Y = posterior | +Z = up', fill=(96, 105, 119), font=font(13))
    return image


def run(probe_relative, probe_sha, output_relative):
    output = path_for(output_relative)
    sidecar = output.with_suffix('.json')
    if output.suffix.lower() != '.png' or output.exists() or sidecar.exists():
        raise FileExistsError('Choose a new PNG/JSON output pair; preserve previous review evidence')
    report, arrays = read_probe(probe_relative, probe_sha)
    frame = framing(arrays)
    image = Image.new('RGB', (CELL_W * 6, HEADER_H + CELL_H * 4 + FOOTER_H), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), 'ACTUAL NATIVE LEG / FOOT - SIX FINITE FK STATES', fill=(24, 32, 45), font=font(29))
    draw.text((20, 49), 'Fixed orthographic cameras and one metric scale | Software mesh diagram, NOT a Blender render', fill=(71, 81, 95), font=font(20))
    draw.text((20, 76), 'Source SHA256 ' + report['sourceSha256'] + ' | Probe ' + probe_sha[:16], fill=(103, 111, 125), font=font(14))
    for row_index, (side, view, sign) in enumerate(VIEWS):
        for column, record in enumerate(report['cases']):
            panel = draw_panel(report, arrays, record, side, view, sign, frame)
            image.paste(panel, (column * CELL_W, HEADER_H + row_index * CELL_H))
    footer = HEADER_H + CELL_H * 4 + 12
    lines = (
        'Only captured foot + bridge ROI is shown. A/T and blue segments are actual trial ankle/toe proxy joints, not certified anatomical centres.',
        'Grey dashed line = original bare-sole height reference, NOT a floor. Each side uses its own actual mesh; right-side data are never synthesized by reflection.',
        'Depth-sorted triangle illustration: no materials, texture, native Blender lighting, exact pixel occlusion, motion-between-frames, floor/shoe/sock or production-control acceptance.',
        'BVH counts are unconfirmed broad-phase candidates. Continuous triangle analysis and actual visual review remain separate. No model or source was modified.',
    )
    for index, line in enumerate(lines):
        draw.text((20, footer + index * 22), line, fill=(75, 85, 102), font=font(16))
    metadata = {
        'schemaVersion': 1, 'mode': 'offline-actual-native-six-state-technical-projection',
        'helperSha256': sha(__file__), 'probe': probe_relative, 'probeSha256': probe_sha,
        'probeArrays': report['arrays'], 'probeArraysSha256': report['arraysSha256'],
        'nativeCaptureHelperSha256': NATIVE_HELPER_SHA,
        'source': report['source'], 'sourceSha256': report['sourceSha256'],
        'image': output.relative_to(project_path()).as_posix(),
        'framing': frame, 'panelViews': [{'side': s, 'view': v, 'cameraWorldXSign': x} for s, v, x in VIEWS],
        'caseOrder': list(CASES), 'sourceProxyAxes': {s: report['sides'][s]['axes'] for s in ('L', 'R')},
        'roi': 'Captured fixed REST triangle IDs with any native vertex below world Z .355 m; same IDs reused in every pose.',
        'projection': 'Pillow depth-sorted actual triangles; approximate visibility at crossing triangles. No invented geometry or pose interpolation.',
        'notProven': ['Blender material/texture appearance', 'Exact render occlusion', 'Between-state motion or collision freedom',
                      'Certified anatomical axes', 'Floor, shoes or socks contact', 'Production IK/FK or old-action retargeting',
                      'Original source versus new native full-leg before/after appearance', 'Final appearance or delivery acceptance'],
        'executedBlender': False, 'modelModified': False, 'continuousContactAnalysisPerformed': False,
        'appearanceAccepted': False, 'deliveryReady': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text('Description', metadata['projection'])
    png_info.add_text('ActualProbeSHA256', probe_sha)
    png_info.add_text('ActualSourceSHA256', report['sourceSha256'])
    with output.open('xb') as stream:
        image.save(stream, format='PNG', pnginfo=png_info)
    metadata['imageSha256'] = sha(output)
    with sidecar.open('x', encoding='utf8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {'image': metadata['image'], 'imageSha256': metadata['imageSha256'],
            'metadata': sidecar.relative_to(project_path()).as_posix(), 'metadataSha256': sha(sidecar),
            'executedBlender': False, 'appearanceAccepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe')
    parser.add_argument('--probe-sha')
    parser.add_argument('--output')
    parser.add_argument('--check-dependencies', action='store_true')
    args = parser.parse_args()
    if args.check_dependencies:
        pins = dependencies()
        print(json.dumps({'helperSha256': sha(__file__), 'dependencyCheckPassed': True,
                          'nativeHelperSha256': NATIVE_HELPER_SHA, 'frozenInputCount': len(pins),
                          'actualProbeLoaded': False, 'imageGenerated': False, 'executedBlender': False}, indent=2))
        return
    if not args.probe or not args.probe_sha or not args.output:
        parser.error('--probe, --probe-sha and --output are required for actual-data plotting')
    print('R11_NATIVE_LEG_PLOT ' + json.dumps(run(args.probe, args.probe_sha, args.output)), flush=True)


if __name__ == '__main__':
    main()
