"""Offline two-bone cervical skin plan from a fresh actual R11 observation.

This writes a pinned proposal, not a .blend and not applied weights. Face,
chin and ears are protected by a smooth anatomical mask; the cervical axis
comes from the actual rig. Final Blender conversion must interpolate these
cage weights through the unchanged Multires/Mask and transport the actual
evaluated boundary weights onto the bridge. Old R01 weights are not substituted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
try:
    import numpy as np
except ImportError:
    sys.path.insert(0, str(project_path() / 'work/verification-r11/python-deps'))
    import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def path(relative):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\', '/').split('/'):
        raise ValueError('Project-relative path required')
    p = (project_path() / relative).resolve()
    p.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    return p


def smooth(a, b, x):
    t = np.clip((np.asarray(x) - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def propose_weights(world, neck_root, head_pivot):
    """An explicitly reviewable initial cervical mask, no geometry changes."""
    world = np.asarray(world, float)
    a, b = np.asarray(neck_root, float), np.asarray(head_pivot, float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if not .075 <= length <= .15 or axis[2] / length < .80:
        raise RuntimeError('Unexpected actual cervical axis; inspect before assigning weights')
    t = (world - a) @ axis / (length * length)
    # Head rigid above the actual upper cervical pivot; lower neck mainly neck.
    release = smooth(.08, .98, t)
    x, y, z = world.T
    anterior = 1. - smooth(b[1] - .090, b[1] - .050, y)
    chin_height = smooth(b[2] - .060, b[2] - .030, z)
    jaw_protect = anterior * chin_height
    lateral = smooth(.045, .068, np.abs(x - b[0]))
    ear_height = smooth(b[2] - .075, b[2] - .032, z)
    cranial_protect = np.maximum(jaw_protect, lateral * ear_height)
    head = 1. - (1. - release) * (1. - cranial_protect)
    head[z >= b[2]] = 1.
    neck = 1. - head
    result = np.column_stack((head, neck)).astype(np.float32)
    if not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise RuntimeError('Non-finite or out-of-range proposed weights')
    return result, {'actualNeckAxisLengthMeters': length,
        'releaseAxisFraction': [.08, .98], 'frontProtectionRelativeToHeadPivotY': [-.090, -.050],
        'chinProtectionRelativeToHeadPivotZ': [-.060, -.030],
        'sideProtectionAbsoluteXOffsetMeters': [.045, .068],
        'earProtectionRelativeToHeadPivotZ': [-.075, -.032],
        'method': 'Smooth two-bone cervical release with anterior jaw/side ear protection. Initial proposal only, requires actual intermediate/extreme pose images.'}


def protect_anatomical_mouth_and_eyes(result, edges):
    """Keep actual oral bag, lips, socket loops and two support rings rigid."""
    landmark_path = path('work/reconstruction-r02/female-face-landmarks-r11.json')
    landmark = json.loads(landmark_path.read_text(encoding='utf-8'))
    raw_path = path('work/reconstruction-r02/female-basemesh-actual-r11.npz')
    if sha(raw_path) != landmark['sourceArraysSha256'] or len(result) != landmark['originalVertexCount']:
        raise RuntimeError('Author anatomical index space changed')
    seeds = set()
    for entry in landmark['eyes'].values():
        seeds.update(entry['sourceVertexIndices'])
    mouth = landmark['mouth']
    for entry in mouth.values():
        if isinstance(entry, dict) and 'sourceVertexIndices' in entry:
            seeds.update(entry['sourceVertexIndices'])
    seeds.update(mouth['vermilionStripVertexIndices'])
    with np.load(raw_path, allow_pickle=False) as raw:
        loops = raw['object_00_corner_vertices']
        starts, sizes = raw['object_00_face_starts'], raw['object_00_face_lengths']
        for field in ('oralUpperFaceIndices', 'oralLowerFaceIndices', 'requiredPosteriorClosureFaceIndices'):
            for fi in mouth[field]:
                seeds.update(loops[starts[fi]:starts[fi] + sizes[fi]].tolist())
    edges = np.asarray(edges, np.int32)
    rings = [np.asarray(sorted(seeds), np.int32)]
    seen = set(seeds)
    for _ in range(4):
        touches = np.isin(edges[:, 0], rings[-1]) | np.isin(edges[:, 1], rings[-1])
        new = set(edges[touches].ravel().tolist()) - seen
        rings.append(np.asarray(sorted(new), np.int32))
        seen.update(new)
    changed = result.copy()
    for ring, strength in zip(rings, (1., 1., 1., 2. / 3., 1. / 3.)):
        changed[ring, 1] *= 1. - strength
        changed[ring, 0] = 1. - changed[ring, 1]
    protected = np.concatenate(rings[:3])
    if np.max(changed[protected, 1]) != 0:
        raise RuntimeError('Anatomical oral/eye support leaked neck weighting')
    return changed, {'landmarks': str(landmark_path.relative_to(project_path())),
        'landmarksSha256': sha(landmark_path), 'sourceRawArraysSha256': sha(raw_path),
        'seedVertexCount': len(rings[0]), 'supportRingCounts': [len(r) for r in rings[1:]],
        'fullyProtectedVertexIds': protected.tolist(), 'maximumProtectedNeckWeight': 0.,
        'actualChangedWeights': int(np.any(changed != result, axis=1).sum()),
        'reason': 'Spatial neck masks alone incorrectly weighted part of the internal oral bag. Actual author oral/lip/eye topology plus two rigid support rings preserves facial performance.'}


def harmonic_bridge_weights(vertices, edges, boundary_ids, boundary_weights,
                            tolerance=1e-10, max_iterations=6000):
    """Positive inverse-edge-length Dirichlet harmonic extension.

    Explicit boundary weights are immutable. This is intended for the ACTUAL
    new head-L3 top weights plus unchanged evaluated body-bottom weights.
    Works with the existing or expanded bridge without a hardcoded vertex count.
    Returns weights and solver diagnostics; it does not write Blender data.
    """
    v = np.asarray(vertices, float)
    e = np.asarray(edges, np.int32)
    ids = np.asarray(boundary_ids, np.int32)
    bw = np.asarray(boundary_weights, float)
    if len(np.unique(ids)) != len(ids) or bw.shape[0] != len(ids):
        raise ValueError('Distinct explicit boundary IDs required')
    if bw.min() < -1e-7 or np.max(abs(bw.sum(1) - 1.)) > 2e-5:
        raise ValueError('Normalized nonnegative boundary skin weights required')
    if e.ndim != 2 or e.shape[1] != 2 or np.any(e < 0) or np.any(e >= len(v)):
        raise ValueError('Actual mesh edges required')
    e = np.unique(np.sort(e, axis=1), axis=0)
    length = np.linalg.norm(v[e[:, 1]] - v[e[:, 0]], axis=1)
    if np.any(length < 1e-8):
        raise RuntimeError('Degenerate actual bridge edge')
    conductance = 1. / length
    boundary = np.zeros(len(v), bool)
    boundary[ids] = True
    interior = np.flatnonzero(~boundary)
    index = np.full(len(v), -1, np.int32)
    index[interior] = np.arange(len(interior))
    degree = np.bincount(e.ravel(), weights=np.repeat(conductance, 2), minlength=len(v))
    if np.any(degree[interior] <= 0):
        raise RuntimeError('Disconnected bridge interior')
    both = (~boundary[e[:, 0]]) & (~boundary[e[:, 1]])
    ie = index[e[both]]
    iw = conductance[both]
    fixed = np.zeros((len(v), bw.shape[1]))
    fixed[ids] = bw
    rhs = np.zeros((len(interior), bw.shape[1]))
    for left, right in ((0, 1), (1, 0)):
        selected = (~boundary[e[:, left]]) & boundary[e[:, right]]
        for c in range(bw.shape[1]):
            rhs[:, c] += np.bincount(index[e[selected, left]],
                weights=conductance[selected] * fixed[e[selected, right], c], minlength=len(interior))
    diagonal = degree[interior]
    def mv(x):
        return diagonal * x - np.bincount(ie.ravel(),
            weights=np.column_stack((iw * x[ie[:, 1]], iw * x[ie[:, 0]])).ravel(), minlength=len(interior))
    result = fixed.copy()
    diagnostics = []
    for c in range(bw.shape[1]):
        x = np.full(len(interior), float(bw[:, c].mean()))
        r = rhs[:, c] - mv(x)
        z = r / diagonal
        direction = z.copy()
        rz = float(r @ z)
        scale = max(float(np.linalg.norm(rhs[:, c])), 1.)
        iterations = 0
        for iterations in range(max_iterations):
            if np.linalg.norm(r) <= tolerance * scale:
                break
            q = mv(direction)
            denom = float(direction @ q)
            if denom <= 0:
                raise RuntimeError('Bridge Laplacian is not positive definite')
            alpha = rz / denom
            x += alpha * direction
            r -= alpha * q
            z = r / diagonal
            new_rz = float(r @ z)
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        residual = float(np.linalg.norm(rhs[:, c] - mv(x)) / scale)
        if residual > tolerance * 10:
            raise RuntimeError('Bridge interpolation did not converge: ' + str(residual))
        result[interior, c] = x
        diagnostics.append({'channel': c, 'iterations': iterations, 'relativeResidual': residual})
    if result.min() < -1e-6 or result.max() > 1.000001 or np.max(abs(result.sum(1) - 1)) > 2e-5:
        raise RuntimeError('Bridge harmonic weights violate convex normalized skin interpolation')
    result = np.clip(result, 0, 1)
    result[ids] = bw
    return result.astype(np.float32), diagnostics


def plan(observation_relative, output_relative):
    obs_path = path(observation_relative)
    out = path(output_relative)
    npz = out.with_suffix('.npz')
    if out.suffix != '.json' or out.exists() or npz.exists():
        raise RuntimeError('Use a fresh JSON/NPZ proposal')
    obs = json.loads(obs_path.read_text(encoding='utf-8'))
    source = path(obs['source'])
    arrays = path(obs['arrays'])
    if sha(source) != obs['sourceSha256'] or sha(arrays) != obs['arraysSha256']:
        raise RuntimeError('Observation no longer matches saved source/arrays')
    with np.load(arrays, allow_pickle=False) as z:
        world = z['head_cage_current_world']
        original = z['head_cage_weights']
        result, parameters = propose_weights(world, obs['cervicalRigWorld']['neckHead'], obs['cervicalRigWorld']['headHead'])
        result, anatomy = protect_anatomical_mouth_and_eyes(result, z['head_cage_edges'])
        groups = dict(obs['sourceContracts']['R11 Female head study']['groups'])
        masks = obs['headMaskModifiers']
        if len(masks) != 1 or masks[0]['group'] not in groups:
            raise RuntimeError('Explicit single current head Mask required')
        visible = original[:, groups[masks[0]['group']]] >= masks[0]['threshold']
        if masks[0]['invert']:
            visible = ~visible
        report = {'source': obs['source'], 'sourceSha256': obs['sourceSha256'],
                  'observation': observation_relative, 'observationSha256': sha(obs_path),
                  'observationArraysSha256': obs['arraysSha256'], 'helperSha256': sha(__file__),
                  'parameters': parameters, 'cageVertices': len(world), 'visibleMaskCageVertices': int(visible.sum()),
                  'anatomicalProtection': anatomy,
                  'visibleMixedHeadNeckVertices': int(((result[:, 0] > .001) & (result[:, 0] < .999) & visible).sum()),
                  'visibleNeckDominantVertices': int(((result[:, 1] > .5) & visible).sum()),
                  'weightGroups': ['head', 'neck'], 'weightsApplied': False, 'savedBlend': False,
                  'bridgeWeightsPending': 'Must evaluate proposed head weights through actual unchanged Multires/Mask in Blender; no nearest-old-head or raw-index substitute.',
                  'conversionContract': [
                      'Preserve the input Blend unchanged. Keep the existing head Mesh/Key datablock IDs to avoid breaking self-targeted shape-key drivers; add only explicitly new head/neck groups and modifier. Back up exact prior bridge skin weights and group mapping before modifying those weights.',
                      'Keep all original raw coordinates, relative shape keys, key values/drivers, UVs, material attributes and non-skin groups exact.',
                      'Create R11 Cranial attachment space with the old head bone-parent/inverse/basis and R11_original_head_space_source marker; all direct OBJECT children move to this carrier without changing their local matrices.',
                      'Convert head itself from BONE parent to OBJECT parent on existing rig while preserving exact REST world matrix; no bone rest or body action changes.',
                      'Append an Armature modifier after current Multires/Mask; preserve evaluated topology and use the existing rig with LBS initially.',
                      'Do not enable DQ on only one side of an unwelded seam; initial LBS matches retained body. Judge volume in actual poses and add explicit correction if needed.',
                      'Evaluate head L3 and sample its ACTUAL new skin weights at observed top boundary IDs. Match bridge top IDs exactly; use actual retained body weights at bottom.',
                      'Use positive harmonic interpolation on actual bridge edges for interior weights, preserving upper and lower endpoint weights exactly. No hardcoded total bridge vertices.',
                      'Preserve eye-pivot/gaze/donor/fibre object world matrices through the carrier across original and temporary head poses. Update gaze coordinate-space contracts explicitly before integration.',
                      'Face-fibre GN continues reading the same head object/evaluated topology; validate every strand and actual GN parity after conversion.',
                      'Check REST world parity, every old key coordinate, all source-body actions and raw body unchanged; inspect neutral/half/full neck turns before adopting.',
                      'Restore only if recorded candidate geometry, key IDs, skin weights and attachments remain unchanged; remove only added skin groups/modifier, restore recorded original bridge weights and parenting, never overwrite later edits.'
                  ],
                  'requiresCoordination': ['adult_gaze_control_r11.py must resolve carrier space instead of assuming the skinned head object is the rigid cranial transform.',
                      'New neck deformation invalidates collision envelopes and existing tail source-contract caches.'],
                  'appearanceAccepted': False}
    np.savez_compressed(npz, proposed_head_neck_weights=result)
    report.update(arrays=npz.relative_to(project_path()).as_posix(), arraysSha256=sha(npz))
    with out.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': output_relative, 'sourceSha256': report['sourceSha256'],
        'visibleMixedHeadNeckVertices': report['visibleMixedHeadNeckVertices'], 'arraysSha256': report['arraysSha256']}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan(a.observation, a.output)
