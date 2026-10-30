"""Offline source/topology diagnostics; never reads or writes a blend file.

Builds an exploratory rim-only correspondence from licensed donor landmarks
and previously recorded HMB topology. This is not a production transfer map.
"""
import collections
import hashlib
import heapq
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path()
A = P / 'assets/mpfb-facial-pilot-r11'
O = P / 'work/reconstruction-r02/mpfb-face-pilot-offline-r11'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_obj(p):
    vertices, faces, groups, group = [], [], [], ''
    for line in p.read_text().splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == 'v':
            vertices.append(list(map(float, s[1:4])))
        elif s[0] == 'g':
            group = s[1]
        elif s[0] == 'f':
            faces.append([int(q.split('/')[0]) - 1 for q in s[1:]])
            groups.append(group)
    return np.asarray(vertices), faces, groups


def mesh_graph(faces):
    adj, edge_faces = collections.defaultdict(set), collections.defaultdict(list)
    for i, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            adj[a].add(b)
            adj[b].add(a)
            edge_faces[tuple(sorted((a, b)))].append(i)
    return adj, edge_faces


def shortest(v, adj, start, end):
    queue, distance, previous = [(0., start)], {start: 0.}, {}
    while queue:
        value, here = heapq.heappop(queue)
        if here == end:
            break
        if value != distance[here]:
            continue
        for other in sorted(adj[here]):
            d = value + float(np.linalg.norm(v[other] - v[here]))
            if d < distance.get(other, float('inf')):
                distance[other], previous[other] = d, here
                heapq.heappush(queue, (d, other))
    result = [end]
    while result[-1] != start:
        result.append(previous[result[-1]])
    return result[::-1]


def opposite_loop(adj, edge_faces, faces, start, second):
    result = [start, second]
    for _ in range(1000):
        previous, current = result[-2:]
        excluded = {previous}
        for fi in edge_faces[tuple(sorted((previous, current)))]:
            f = faces[fi]
            j = f.index(current)
            excluded.update((f[j - 1], f[(j + 1) % len(f)]))
        choices = adj[current] - excluded
        if len(choices) != 1:
            raise ValueError('Ambiguous loop at %d' % current)
        nxt = choices.pop()
        if nxt == start:
            return result
        if nxt in result:
            raise ValueError('Self intersecting topology traversal')
        result.append(nxt)
    raise ValueError('Unbounded topology loop')


def split_ring(v, ring, start, end):
    i = ring.index(start)
    ordered = ring[i:] + ring[:i]
    j = ordered.index(end)
    two = [ordered[:j + 1], [ordered[0]] + list(reversed(ordered[j:]))]
    return {'upper': max(two, key=lambda ids: v[ids, 2].mean()),
            'lower': min(two, key=lambda ids: v[ids, 2].mean())}


def normalized_arc(points):
    d = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    if d[-1] < 1e-8 or np.any(np.diff(d) < 1e-12):
        raise ValueError('Collapsed ring segment')
    return d / d[-1]


def arc_map(donor, target, donor_ids, target_ids):
    d = normalized_arc(donor[donor_ids])
    t = normalized_arc(target[target_ids])
    rows = []
    for index, value in zip(target_ids, t):
        j = min(int(np.searchsorted(d, value, side='right')) - 1, len(d) - 2)
        j = max(0, j)
        weight = float((value - d[j]) / (d[j + 1] - d[j]))
        rows.append({'recipientVertex': int(index), 'donorEdge': [int(donor_ids[j]), int(donor_ids[j + 1])],
                     'weights': [1. - weight, weight], 'normalizedArc': float(value)})
    return rows


def font(size):
    for family in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(family, size)
        except OSError:
            pass
    return ImageFont.load_default()


def project(points, rect, xlim, ylim, axes=(0, 2)):
    x, y, w, h = rect
    scale = min(w / (xlim[1] - xlim[0]), h / (ylim[1] - ylim[0]))
    xx = x + w / 2 + (points[:, axes[0]] - sum(xlim) / 2) * scale
    yy = y + h / 2 - (points[:, axes[1]] - sum(ylim) / 2) * scale
    return np.stack([xx, yy], axis=1)


def poly(draw, pixels, color, width=3, close=False):
    coords = [tuple(map(float, q)) for q in pixels]
    if close:
        coords.append(coords[0])
    if len(coords) > 1:
        draw.line(coords, fill=color, width=width)


def draw_mesh(draw, vertices, faces, rect, xlim, ylim, front_max_y=None):
    px = project(vertices, rect, xlim, ylim)
    for f in faces:
        q = vertices[f]
        if q[:, 0].min() < xlim[0] or q[:, 0].max() > xlim[1] or q[:, 2].min() < ylim[0] or q[:, 2].max() > ylim[1]:
            continue
        if front_max_y is not None and q[:, 1].mean() > front_max_y:
            continue
        poly(draw, px[f], '#bcc4ca', 1, True)
    return px


def main():
    if O.exists():
        raise FileExistsError('Preserve earlier diagnostics')
    manifest = json.loads((A / 'source-manifest.json').read_text(encoding='utf8'))
    for row in manifest['files']:
        if sha(A / row['path']) != row['sha256']:
            raise ValueError('Downloaded asset changed')
    rig_source = json.loads((A / 'rig-landmark-source.json').read_text())
    if sha(A / 'rig.human.json') != rig_source['sha256']:
        raise ValueError('Rig anchor data changed')
    rig = json.loads((A / 'rig.human.json').read_text())
    raw, all_faces, groups = read_obj(A / 'base-hm08.obj')
    # Pinned MPFB TargetService uses x,-z,y; HumanService uses default scale .1.
    donor = raw[:, [0, 2, 1]] * np.array([.1, -.1, .1])
    body_faces = [f for f, g in zip(all_faces, groups) if g == 'body']
    body_ids = sorted(set(i for f in body_faces for i in f))
    body = np.zeros(len(raw), bool)
    body[body_ids] = True
    adj, ef = mesh_graph(body_faces)
    land_path = P / 'work/reconstruction-r02/female-face-landmarks-r11.json'
    land = json.loads(land_path.read_text())
    hmb_path = P / 'work/reconstruction-r02/female-basemesh-actual-r11.npz'
    if sha(hmb_path) != land['sourceArraysSha256']:
        raise ValueError('HMB topology provenance mismatch')
    with np.load(hmb_path, allow_pickle=False) as data:
        hmb = data['object_00_local'].astype(float)
        corners, starts, lengths = [data['object_00_' + s] for s in ('corner_vertices', 'face_starts', 'face_lengths')]
        hmb_faces = [corners[s:s + n].tolist() for s, n in zip(starts, lengths)]
    if len(hmb) != 10582 or len(raw) != 19158:
        raise ValueError('Unexpected canonical topology')
    donor_rings, donor_paths, recipient_paths = {}, {}, {}
    for side in ('L', 'R'):
        start = rig['bones']['lid.T.' + side]['head']['vertex_index']
        following = rig['bones']['lid.T.' + side]['tail']['vertex_index']
        short = shortest(donor, adj, start, following)
        ring = opposite_loop(adj, ef, body_faces, start, short[1])
        guide_ids = set()
        for key, bone in rig['bones'].items():
            if key.startswith(('lid.T.' + side, 'lid.B.' + side)):
                guide_ids.update(bone[end]['vertex_index'] for end in ('head', 'tail'))
        if not guide_ids.issubset(ring) or len(ring) != 28:
            raise ValueError('Rig guide does not fit one canonical eyelid ring')
        inner = min(ring, key=lambda i: abs(donor[i, 0]))
        outer = max(ring, key=lambda i: abs(donor[i, 0]))
        paths = split_ring(donor, ring, inner, outer)
        donor_rings['eyelid_' + side] = {'ids': ring, 'inner': inner, 'outer': outer, 'allRigLidGuidesOnRing': True}
        for vertical in ('upper', 'lower'):
            key = side + '_' + vertical + '_lid'
            donor_paths[key] = paths[vertical]
            recipient_paths[key] = land['eyes'][side]['ordered' + vertical.title() + 'InnerToOuter']
    # The lip rig uses MEAN anchors across multiple rows. Keep both candidates
    # explicit; no claim that the first row is a visually accepted contact seam.
    for label, a, b in (('mouth_inner_candidate', 509, 452), ('mouth_outer_reference', 571, 550)):
        guide = shortest(donor, adj, a, b)
        ring = opposite_loop(adj, ef, body_faces, a, guide[1])
        if len(ring) != 38:
            raise ValueError('Unexpected lip ring topology')
        left = min(ring, key=lambda i: donor[i, 0])
        right = max(ring, key=lambda i: donor[i, 0])
        donor_rings[label] = {'ids': ring, 'negativeXCorner': left, 'positiveXCorner': right,
                              'status': 'geometric candidate; MEAN rig guide alone does not prove contact rim'}
        if label == 'mouth_inner_candidate':
            paths = split_ring(donor, ring, left, right)
            for vertical in ('upper', 'lower'):
                key = vertical + '_lip'
                donor_paths[key] = paths[vertical]
                recipient_paths[key] = land['mouth']['openingRimCandidate']['ordered' + vertical.title() + 'LeftToRight']
    mappings = {k: arc_map(donor, hmb, donor_paths[k], recipient_paths[k]) for k in donor_paths}
    target_records, target_vectors = [], {}
    for target in sorted((A / 'targets').glob('*.target')):
        values = np.loadtxt(target)
        ids = values[:, 0].astype(int)
        if not np.all(values[:, 0] == ids) or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(raw) or not np.isfinite(values).all():
            raise ValueError('Invalid sparse target')
        d = values[:, [1, 3, 2]] * np.array([.1, -.1, .1])
        dense = np.zeros_like(donor)
        dense[ids] = d
        target_vectors[target.stem] = dense
        moving = np.linalg.norm(d, axis=1) > 1e-12
        side_energy = np.linalg.norm(d, axis=1)
        row = {'name': target.stem, 'sha256': sha(target), 'rows': len(ids), 'indexRange': [int(ids.min()), int(ids.max())],
               'duplicateIndices': 0, 'finite': True, 'indicesInCanonicalBase': True,
               'bodyRows': int(body[ids].sum()), 'helperRows': int((~body[ids]).sum()),
               'nonzeroRows': int(moving.sum()), 'maxDeltaMillimeters': float(np.linalg.norm(d, axis=1).max() * 1000),
               'maxBodyDeltaMillimeters': float(np.linalg.norm(d[body[ids]], axis=1).max() * 1000),
               'positiveXDisplacementSumFraction': float(side_energy[donor[ids, 0] > 1e-8].sum() / side_energy.sum()),
               'negativeXDisplacementSumFraction': float(side_energy[donor[ids, 0] < -1e-8].sum() / side_energy.sum()),
               'mappedRimMaxMillimeters': {k: float(max(np.linalg.norm(np.sum(dense[r['donorEdge']] * np.asarray(r['weights'])[:, None], axis=0)) for r in rows) * 1000) for k, rows in mappings.items()}}
        target_records.append(row)
    # All pixels are plots of known arrays, not an edited character image.
    O.mkdir()
    image = Image.new('RGB', (1900, 1300), '#f6f8fa')
    dr = ImageDraw.Draw(image)
    dr.text((45, 24), 'Facial transfer: source topology check only', fill='#162536', font=font(34))
    dr.text((45, 73), 'Canonical donor and historical HMB scaffold. No current character deformation or new head candidate.', fill='#485969', font=font(20))
    panels = [(donor, body_faces, (70, 175, 800, 950), (-.079, .079), (.607, .83), -.055,
               'MakeHuman hm08 / 19,158 vertices'),
              (hmb, hmb_faces, (1030, 175, 800, 950), (-.095, .095), (1.39, 1.67), -.025,
               'HMB source cage / 10,582 vertices')]
    colors = {'eyelid_L': '#1373be', 'eyelid_R': '#1373be', 'mouth_inner_candidate': '#e07a21', 'mouth_outer_reference': '#9671b9'}
    for n, (v, f, rect, xx, zz, front, title) in enumerate(panels):
        dr.text((rect[0], 125), title, fill='#162536', font=font(28))
        px = draw_mesh(dr, v, f, rect, xx, zz, front)
        if n == 0:
            rings = [(row['ids'], colors[k]) for k, row in donor_rings.items()]
        else:
            rings = [(land['eyes'][s]['sourceVertexIndices'], '#1373be') for s in ('L', 'R')]
            rings.append((land['mouth']['openingRimCandidate']['sourceVertexIndices'], '#e07a21'))
        for ids, color in rings:
            poly(dr, px[ids], color, 4, True)
        dr.text((rect[0], 1155), 'Blue: eyelid loop   Orange: lip-rim candidate', fill='#253849', font=font(21))
        dr.text((rect[0], 1190), 'Separate plot scales; identity transfer is not evaluated here.', fill='#536574', font=font(19))
    image.save(O / 'topology-comparison.png')
    overview = Image.new('RGB', (2100, 1420), '#f6f8fa')
    dd = ImageDraw.Draw(overview)
    dd.text((35, 20), 'Six CC0 source targets: body-only projected displacements', fill='#162536', font=font(30))
    dd.text((35, 65), 'Blue: source   Red: displaced at strength 1   Lines: real displacement, not a finished facial expression.', fill='#485969', font=font(20))
    for n, row in enumerate(target_records):
        x, y = 30 + (n % 3) * 690, 120 + (n // 3) * 640
        rect = (x + 20, y + 40, 620, 475)
        xx, zz = (-.077, .077), (.61, .805)
        pp = draw_mesh(dd, donor, body_faces, rect, xx, zz, -.065)
        delta = target_vectors[row['name']]
        moved = project(donor + delta, rect, xx, zz)
        length = np.linalg.norm(delta, axis=1)
        chosen = np.flatnonzero(body & (length > .0002))
        chosen = chosen[np.argsort(length[chosen])[-120:]]
        for i in chosen:
            dd.line([tuple(pp[i]), tuple(moved[i])], fill='#cc513d', width=2)
            dd.ellipse((pp[i, 0] - 2, pp[i, 1] - 2, pp[i, 0] + 2, pp[i, 1] + 2), fill='#177cb7')
        dd.text((x + 20, y), row['name'], fill='#162536', font=font(26))
        dd.text((x + 20, y + 530), 'body %d | helpers excluded %d | max %.2f mm' % (row['bodyRows'], row['helperRows'], row['maxBodyDeltaMillimeters']), fill='#4b5966', font=font(18))
        dd.text((x + 20, y + 565), 'Canonical base only; no HMB transfer applied.', fill='#687784', font=font(18))
    overview.save(O / 'six-target-source-check.png')
    for key, rows in mappings.items():
        for r in rows:
            if abs(sum(r['weights']) - 1.) > 1e-12 or min(r['weights']) < -1e-12 or max(r['weights']) > 1. + 1e-12:
                raise ValueError('Invalid arc weights')
    report = {
        'schemaVersion': 1, 'scope': 'Offline source and rim-only correspondence feasibility; no Blender process, current-head read, geometry mutation or shape-key installation.',
        'sourceManifest': 'assets/mpfb-facial-pilot-r11/source-manifest.json', 'sourceManifestSha256': sha(A / 'source-manifest.json'),
        'rigLandmarkManifest': 'assets/mpfb-facial-pilot-r11/rig-landmark-source.json', 'rigLandmarkSha256': rig_source['sha256'],
        'downloadedOriginalBytesIncludingRigMetadata': manifest['totalBytes'] + rig_source['bytes'],
        'canonicalBase': {'vertices': len(raw), 'allFaces': len(all_faces), 'bodyVertices': len(body_ids), 'bodyFaces': len(body_faces),
            'bodyBoundaryEdges': sum(len(v) == 1 for v in ef.values()), 'sha256': sha(A / 'base-hm08.obj'),
            'basisMatch': 'Official MPFB v2.0.18 hm08 canonical base, identical Git blob to official MakeHuman base; compatible vertex space for the advertised packs. No separate artist-personalized neutral basis was provided by the pack author.'},
        'coordinateTransform': {'matrixFromOriginalMakehumanToMeters': [[.1, 0, 0], [0, 0, -.1], [0, .1, 0]],
            'evidence': ['https://github.com/makehumancommunity/mpfb2/blob/d0a32e57a7f915cb2f2b95410e2117648c7bbb7e/src/mpfb/services/targetservice.py',
                         'https://github.com/makehumancommunity/mpfb2/blob/d0a32e57a7f915cb2f2b95410e2117648c7bbb7e/src/mpfb/services/humanservice.py'],
            'note': 'Target parser x,-z,y and HumanService create_human default scale=.1. No donor placement translation is applied to target deltas.'},
        'hmbTopology': {'sourceArrays': str(hmb_path.relative_to(P)).replace('\\', '/'), 'sha256': sha(hmb_path),
            'landmarks': str(land_path.relative_to(P)).replace('\\', '/'), 'vertices': len(hmb),
            'currentCharacterNeutralCapture': False, 'limitation': 'Original licensed HMB cage used only to establish index and arc feasibility; current natural default face includes further geometry layers and requires a fresh source-specific capture.'},
        'targets': target_records, 'donorRingCandidates': donor_rings,
        'correspondence': {'method': 'Separate upper/lower, left/right rim arcs; normalized 3D arc-length interpolation between donor edge endpoints, recipient vertex IDs preserved.',
            'donorPaths': donor_paths, 'recipientPaths': recipient_paths, 'rows': mappings,
            'normalization': 'Each row sums to one, nonnegative; endpoints exact; no upper/lower crossing because arc domains are distinct.',
            'productionReady': False,
            'remaining': ['Visually confirm the donor lip contact-rim candidate; Rigify lip guides use MEAN across rows.',
                'Extend only verified rim domains into skin triangles with restricted region labels and a bounded local deformation basis.',
                'Separate original targets body, helper, eyelid lining, oral lining and jaw-linked support; helper deltas are excluded from a face-only recipient.',
                'Recompute weights/metric alignment from the current frozen neutral, not this historical HMB source cage.',
                'Preserve saved static shape controls; new relative-to-Basis keys store Basis+delta, not neutral+delta.']},
        'newlyConfirmedRisks': ['Canonical donor body is closed; eye/lip rim cannot be recovered as mesh boundary edges.',
            'Left-name targets legitimately contain nonzero opposite-side support; use anatomical semantics rather than blindly hard-clipping x.',
            'Some source targets displace helper geometry and joints; full-file copy would move non-skin data.',
            'Canonical max-strength source displacement is not a safe or natural magnitude for the current custom character.'],
        'images': ['topology-comparison.png', 'six-target-source-check.png'],
        'blenderExecuted': False, 'characterModified': False, 'expressionsTransferred': False,
        'appearanceAccepted': False, 'helperSha256': sha(Path(__file__))
    }
    with (O / 'correspondence-feasibility.json').open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savez_compressed(O / 'source-target-deltas.npz', donor_coordinate_matrix=np.array([[.1, 0, 0], [0, 0, -.1], [0, .1, 0]]),
                        donor_body_mask=body, **target_vectors)
    print(json.dumps({'output': str(O.relative_to(P)), 'targets': len(target_records), 'rimDomains': len(mappings),
                      'mappedRimRows': sum(len(rows) for rows in mappings.values()), 'productionReady': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
