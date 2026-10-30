"""Offline numerical audit of the CURRENT R10 observation. No Blender import.

Uses an exact bounded-distance NumPy KD tree to evaluate the same spatial field.
Distances beyond 33 mm are capped, where the field's anchor weight is already 1.
This checks array continuity only; actual surface BVH checks remain in Blender.
"""
import json
import argparse
import runpy
from pathlib import Path
import numpy as np


class AnchorTree:
    def __init__(self, points, cap=.033):
        self.points = np.asarray(points, dtype=float)
        self.cap = float(cap)
        self.root = self._build(np.arange(len(points), dtype=np.int64))

    def _build(self, ids):
        point = self.points[ids]; low = point.min(0); high = point.max(0)
        if len(ids) <= 32:
            return (low, high, ids, None, None, None, None)
        axis = int(np.argmax(high-low)); order = np.argsort(point[:, axis], kind='stable')
        ids = ids[order]; half = len(ids)//2
        split = float((self.points[ids[half-1], axis]+self.points[ids[half], axis])*.5)
        return (low, high, None, axis, split, self._build(ids[:half]), self._build(ids[half:]))

    def distances(self, points):
        points = np.asarray(points, dtype=float)
        best = np.full(len(points), self.cap*self.cap)

        def visit(node, ids):
            if not len(ids):
                return
            low, high, leaf, axis, split, left, right = node
            q = points[ids]
            bound = np.maximum(np.maximum(low-q, q-high), 0.)
            ids = ids[np.einsum('ij,ij->i', bound, bound) <= best[ids]]
            if not len(ids):
                return
            if leaf is not None:
                anchor = self.points[leaf]
                for start in range(0, len(ids), 2048):
                    group = ids[start:start+2048]
                    delta = points[group, None, :]-anchor[None, :, :]
                    distance = np.einsum('ijk,ijk->ij', delta, delta).min(1)
                    best[group] = np.minimum(best[group], distance)
                return
            first_left = points[ids, axis] <= split
            a = ids[first_left]; b = ids[~first_left]
            visit(left, a); visit(right, b)
            visit(right, a); visit(left, b)

        visit(self.root, np.arange(len(points), dtype=np.int64))
        return np.sqrt(best)


def audit(observation_relative=None, output_relative=None):
    api = runpy.run_path(str(Path(__file__).with_name('crown_volume_candidate_r10.py')))
    path, record, arrays = api['_load_observation'](observation_relative or api['OBSERVATION'])
    # Independently check the query implementation against direct distances on
    # a deterministic subset before trusting it for the production-size arrays.
    _, anchors = api['_fixed_points'](record, arrays)
    tree = AnchorTree(anchors)
    rng = np.random.default_rng(103)
    queries = arrays['root_world'][rng.choice(len(arrays['root_world']), 128, replace=False)]
    actual = tree.distances(queries)
    direct = []
    for q in queries:
        direct.append(min(.033, float(np.linalg.norm(anchors-q, axis=1).min())))
    error = float(np.max(np.abs(actual-direct)))
    if error > 1e-12:
        raise RuntimeError('Offline distance query disagrees with direct current-array distances')
    local, world, report, _, _ = api['propose'](record, arrays, tree_factory=AnchorTree)
    if record.get('observationVersion', 1) >= 2:
        _, old_record, old_arrays = api['_load_observation'](record['baseObservation'])
        old_local, _, _, old_setup, _ = api['propose'](old_record, old_arrays, tree_factory=AnchorTree)
        same = {name: bool(np.array_equal(local[name], old_local[name])) for name in api['NAMES']}
        if not all(same.values()) or report['field'] != old_setup:
            raise RuntimeError('Additional transport changed the existing five-object target or original common field')
        report['originalFiveObjectProposalCoordinatesBitExact'] = same
        report['originalFieldSetupUnchanged'] = True
        report['additionalTransportObjects'] = record['additionalTransportObjects']
    report.update(sourceSha256=record['sourceSha256'], observationSha256=api['_sha'](path),
                  arraysSha256=record['arrays']['sha256'], helperSha256=api['_sha'](Path(api['__file__'])),
                  offlineOnly=True, blenderExecuted=False, sourceGeometryWritten=False,
                  exactBoundedDistanceProbeMaximumError=error,
                  pending='Actual head/root BVH checks, independent candidate save, matched front/quarter/side visual decision')
    output = api['_path'](output_relative or 'work/reconstruction-r02/crown-numerical-audit-r10.json', '.json')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print('R10_NUMPY_AUDIT '+str(output), flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation');parser.add_argument('--output')
    args=parser.parse_args();audit(args.observation,args.output)
