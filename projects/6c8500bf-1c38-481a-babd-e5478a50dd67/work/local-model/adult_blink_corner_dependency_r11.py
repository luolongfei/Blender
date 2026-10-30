"""Read-only current closed surface differences and actual canthal dependencies.

The old broad eye-side observation box is retained as diagnostic evidence.
Compatibility is separately measured on actual nonzero node inputs, incident
faces, two connected rings, and a 2 mm spatial halo around each corner.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def api(name): return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def compare_closed(closed, observation, arrays, fields, tolerance=2e-7):
    original_ids = arrays['head_originalVertexIds']; triangles = arrays['head_triangles']
    if not np.array_equal(closed['originalVertexIds'], original_ids) or not np.array_equal(closed['triangles'], triangles):
        raise RuntimeError('Captured final eye-region topology/order changed; coordinate-only dependency comparison forbidden')
    before = np.asarray(arrays['head_world'], float); current = np.asarray(closed['world'], float)
    delta = current-before; magnitude = np.linalg.norm(delta, axis=1)
    changed = np.max(abs(delta), axis=1)>tolerance
    nonzero = np.zeros(len(before), bool)
    per_side = {}
    for side in ('L', 'R'):
        active = (abs(fields[side]['angle'])>1e-10) | (abs(fields[side]['radiusChange'])>1e-10)
        per_side[side] = active; nonzero |= active
    one_ring = nonzero.copy(); one_ring[np.unique(triangles[nonzero[triangles].any(axis=1)])] = True
    two_ring = one_ring.copy(); two_ring[np.unique(triangles[one_ring[triangles].any(axis=1)])] = True
    spatial = np.zeros(len(before), bool); boxes = []
    for side in ('L', 'R'):
        center_x = observation['eyes'][side]['centerWorld'][0]
        for end in (-1, 1):
            active = per_side[side] & ((before[:, 0]-center_x)*end>0.)
            if not active.any(): continue
            lo = before[active].min(axis=0)-.002; hi = before[active].max(axis=0)+.002
            spatial |= np.all((before>=lo)&(before<=hi), axis=1)
            boxes.append({'side': side, 'end': end, 'minimumWorld': lo.tolist(), 'maximumWorld': hi.tolist()})
    spatial[np.unique(triangles[spatial[triangles].any(axis=1)])] = True
    dependency = two_ring | spatial
    masks = {'nonzeroGNInput': nonzero, 'incidentFaceVertices': one_ring,
             'twoConnectedRings': two_ring, 'spatialTwoMillimeterHalo': spatial, 'completeDependency': dependency}
    domains = {}
    for name, mask in masks.items():
        ids = np.flatnonzero(mask)
        domains[name] = {'vertexCount': len(ids), 'changedCount': int((changed&mask).sum()),
                         'maximumComponentDifferenceMeters': float(abs(delta[ids]).max()) if len(ids) else 0.,
                         'maximumWorldDifferenceMeters': float(magnitude[ids].max()) if len(ids) else 0.}
    changed_ids = np.flatnonzero(changed)
    ranked = changed_ids[np.argsort(magnitude[changed_ids])[-20:][::-1]]
    report = {'coordinateToleranceMeters': tolerance, 'capturedROIVertexCount': len(before),
              'changedVertexCount': len(changed_ids), 'maximumComponentDifferenceMeters': float(abs(delta).max()),
              'maximumWorldDifferenceMeters': float(magnitude.max()),
              'changedBoundsWorld': {'minimum': current[changed_ids].min(axis=0).tolist(), 'maximum': current[changed_ids].max(axis=0).tolist()} if len(changed_ids) else None,
              'largestDifferences': [{'originalVertexId': int(original_ids[i]), 'observedWorld': before[i].tolist(),
                                      'currentWorld': current[i].tolist(), 'deltaWorld': delta[i].tolist(),
                                      'meters': float(magnitude[i])} for i in ranked],
              'domains': domains, 'spatialHaloBoxes': boxes,
              'compatibleActualDependency': not bool((changed&dependency).any()),
              'globalTopologySubsetExact': True,
              'interpretation': 'Changed points outside the exact GN input, incident triangles, two rings and 2 mm spatial halo are recorded, not ignored. This is compatibility only, not final closure/contact acceptance.'}
    payload = {'originalVertexIds': original_ids, 'observedWorld': before, 'currentWorld': current,
               'deltaWorld': delta, 'changedOriginalVertexIds': original_ids[changed_ids],
               'triangles': triangles, **{name+'Mask': mask for name, mask in masks.items()}}
    return report, payload


def probe(source_relative, plan_relative, output_relative):
    import bpy
    base = api('adult_blink_corner_install_r11_v2')
    source = base['path'](source_relative, '.blend'); plan_file = base['path'](plan_relative, '.json')
    output = base['path'](output_relative, '.json'); npz_file = output.with_suffix('.npz')
    if output.exists() or npz_file.exists() or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved independent source/diagnostic paths required')
    plan = base['load_json'](plan_file); observation_file = base['path'](plan['observation'], '.json')
    observation = base['load_json'](observation_file); fields_file = base['path'](plan['fields'], '.npz')
    if base['sha'](observation_file)!=plan['observationSha256'] or base['sha'](observation_file.with_suffix('.npz'))!=observation['arraysSha256'] or base['sha'](fields_file)!=plan['fieldsSha256']:
        raise RuntimeError('Frozen plan inputs changed')
    with np.load(observation_file.with_suffix('.npz')) as data: arrays = {k: data[k].copy() for k in data.files}
    with np.load(fields_file) as data: fields = {s: {k: data[s+'_'+k].copy() for k in ('angle', 'radiusChange')} for s in ('L', 'R')}
    head = bpy.data.objects[base['HEAD']]; keys = head.data.shape_keys.key_blocks
    names = ['R11 blink L', 'R11 blink R']; previous = {name: float(keys[name].value) for name in names}
    source_sha = base['sha'](source); capture = api('adult_blink_recompute_r11')
    pose_api = api('production_uv_r08'); pose_before = pose_api['_pose_contract']()
    try:
        for name in names: keys[name].value = 1.
        bpy.context.view_layer.update()
        with api('face_contact_audit_r11')['rest_evaluation']():
            closed = capture['_snapshot'](head); report, payload = compare_closed(closed, observation, arrays, fields)
            report['headTransformMaximumDifference'] = float(np.max(abs(np.asarray(head.matrix_world)-np.asarray(observation['headMatrixWorld']))))
            report['globes'] = {}
            for side in ('L', 'R'):
                eye = capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+side])
                reference = arrays['eye_'+side+'_world']
                report['globes'][side] = {'shapeEqual': eye['world'].shape==reference.shape,
                    'maximumComponentDifferenceMeters': float(np.max(abs(eye['world']-reference))) if eye['world'].shape==reference.shape else None}
    finally:
        for name, value in previous.items(): keys[name].value = value
        bpy.context.view_layer.update()
    if source_sha!=base['sha'](source) or pose_api['_pose_contract']()!=pose_before: raise RuntimeError('Read-only probe changed source or pose')
    np.savez_compressed(npz_file, **payload)
    report.update(source=source_relative, sourceSha256=source_sha, observedSourceSha256=observation['sourceSha256'],
                  plan=plan_relative, planSha256=base['sha'](plan_file), helperSha256=base['sha'](__file__),
                  arraysSha256=base['sha'](npz_file), sourceSaved=False, controlsRestored=True)
    with output.open('x', encoding='utf8') as stream: json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R11_CANTHAL_DEPENDENCY_OBSERVED '+json.dumps({'output': output_relative, 'changedVertexCount': report['changedVertexCount'], 'domains': report['domains']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:]); probe(args.source, args.plan, args.output)
