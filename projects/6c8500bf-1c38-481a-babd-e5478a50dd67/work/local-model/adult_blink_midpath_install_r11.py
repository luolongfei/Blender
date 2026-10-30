"""Install the bounded arc and separate dynamic inner-lining candidate.

Independent saved, unbound REST source only. The current outer-crown and
canthal modifiers stay intact. Actual arc geometry is captured before the
inner layer so its fresh triangle tessellation is the contact reference.
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

HEAD = 'R11 Female head study'
MARKER = 'R11_middle_blink_and_inner_contact'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def path(relative, suffix):
    result = (project_path()/relative).resolve()
    result.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if result.suffix != suffix:
        raise ValueError('Unexpected artifact extension')
    return result


def read(file):
    return json.loads(Path(file).read_text(encoding='utf8'))


def prepare(source_relative, plan_relative, output_relative):
    import bpy
    source = path(source_relative, '.blend')
    output = path(output_relative, '.blend')
    plan_file = path(plan_relative, '.json')
    report_file = output.with_suffix('.midpath.json')
    attempt_file = output.with_suffix('.midpath-attempt.json')
    if any(p.exists() for p in (output, report_file, attempt_file)) or source == output:
        raise RuntimeError('Fresh independent candidate outputs required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Fresh saved independent background source required')
    if MARKER in bpy.context.scene or 'R11_blink_corner_surface_refinement' not in bpy.context.scene:
        raise RuntimeError('Install once after the actual current canthal candidate')
    head = bpy.data.objects[HEAD]
    if any(m.type == 'ARMATURE' for m in head.modifiers):
        raise RuntimeError('This installation requires the unbound REST source; cranial query mapping must precede later posed use')
    plan = read(plan_file)
    if not plan.get('offlineGeometryPassed'):
        raise RuntimeError('A frozen locally verified proposal is required')
    for filename, checksum in plan.get('helperHashes', {}).items():
        if sha(Path(__file__).with_name(filename)) != checksum:
            raise RuntimeError('Frozen candidate helper changed: '+filename)
    obs_file = path(plan['observation'], '.json')
    obs = read(obs_file)
    fields_file = path(plan['fields'], '.npz')
    if (sha(obs_file) != plan['observationSha256'] or
            sha(obs_file.with_suffix('.npz')) != obs['arraysSha256'] or
            sha(fields_file) != plan['fieldsSha256']):
        raise RuntimeError('Frozen observation or fields changed')
    with np.load(obs_file.with_suffix('.npz')) as data:
        observed = {k: data[k].copy() for k in data.files}
    with np.load(fields_file) as data:
        fields = {k: data[k].copy() for k in data.files}
    ids = observed['head_originalVertexIds']
    if not np.array_equal(ids, fields['originalVertexIds']):
        raise RuntimeError('Field evaluated indices do not match observation')
    arcs = {s: fields['arc_'+s] for s in ('L', 'R')}
    masks = {s: fields['innerMask_'+s].astype(bool) for s in ('L', 'R')}
    if any(np.any(arcs[s][[0, -1]]) for s in ('L', 'R')):
        raise RuntimeError('Arc endpoint fields must be identically zero')
    capture = api('adult_blink_midpath_observe_r11')
    binding = api('face_fibre_binding_r11')
    reference = api('adult_blink_midpath_contact_r11')
    arc_api = api('adult_blink_midpath_plan_r11')
    visibility = api('adult_blink_corner_plan_r11')['front_visibility']
    key_api = api('adult_jaw_contour_r11')
    pose_api = api('production_uv_r08')
    original_keys = key_api['_keys'](head.data)
    original_pose = pose_api['_pose_contract']()
    original_active = head.active_shape_key_index
    original_materials = [m.name if m else None for m in head.data.materials]
    original_modifiers = [m.name for m in head.modifiers]
    keys = head.data.shape_keys.key_blocks
    names = (*capture['BLINK'].values(), *capture['SMILES'])
    previous = {name: float(keys[name].value) for name in names}
    if any(abs(v)>1e-7 for v in previous.values()):
        raise RuntimeError('Start the installation with neutral public expressions')
    source_sha = sha(source)
    states = [dict(row) for row in obs['states']]
    states += [{'id': 'between_035', 'L': .35, 'R': .35, 'warm': 0.},
               {'id': 'between_065', 'L': .65, 'R': .65, 'warm': 0.}]
    dependency = masks['L'] | masks['R']
    for side in ('L', 'R'):
        dependency |= np.any(np.linalg.norm(arcs[side], axis=2)>1e-10, axis=0)
    triangles = observed['state_0_head_triangles']
    for _ in range(2):
        dependency[np.unique(triangles[dependency[triangles].any(1)])] = True
    modifier = group = cache = mesh = None
    success = False
    report = {'source': source_relative, 'sourceSha256': source_sha,
              'plan': plan_relative, 'planSha256': sha(plan_file),
              'helperSha256': sha(__file__), 'checks': [],
              'querySpace': 'unbound_rest_head_object', 'sourceSaved': False,
              'appearanceAccepted': False, 'allMotionAccepted': False,
              'layers': plan['layers'],
              'limits': ['Cranial/bone pose requires the explicit pre-Armature query mapping before any posed use.',
                         'Actual samples and local surface checks do not constitute all-expression or visible appearance acceptance.',
                         'Outer-wall allowance is a front-axis separation, not a certified minimum Euclidean wall thickness.']}

    def set_state(state):
        for name in capture['SMILES']:
            keys[name].value = state['warm'] if name == capture['WARM'] else 0.
        for side in ('L', 'R'):
            keys[capture['BLINK'][side]].value = state[side]
        bpy.context.view_layer.update()
        expected = {capture['BLINK'][s]: state[s] for s in ('L', 'R')}
        expected.update({name: state['warm'] if name == capture['WARM'] else 0. for name in capture['SMILES']})
        if any(abs(float(keys[name].value)-value)>1e-6 for name, value in expected.items()):
            raise RuntimeError('Animation overrides a requested candidate sample')

    def snapshot():
        result = capture['capture_mesh'](head, binding)
        evaluated = head.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        result['pointNormals'] = binding['_array'](evaluated.vertices, 'normal').astype(float)
        return result

    def write_attempt():
        attempt_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            baseline = {}
            for state in states:
                set_state(state)
                current = snapshot()
                baseline[state['id']] = current
                if current['counts']['points'] != obs['headCounts']['points']:
                    raise RuntimeError('Current full evaluated point count changed')
                if current['topologySha256'] != obs['headTopologySha256']:
                    raise RuntimeError('Current real polygon/loop/edge/index topology changed')
                if not np.array_equal(current['matrix'], np.asarray(obs['headMatrixWorld'])):
                    raise RuntimeError('Current REST head transform changed')
                if 'index' in state:
                    expected = observed[state['headWorldArray']]
                    delta = current['world'][ids]-expected
                    maximum = float(abs(delta[dependency]).max())
                    if maximum>2e-7:
                        diagnostic = output.with_suffix('.dependency-failed.npz')
                        np.savez_compressed(diagnostic, originalVertexIds=ids, dependencyMask=dependency,
                                            observedWorld=expected, currentWorld=current['world'][ids], deltaWorld=delta)
                        raise RuntimeError('Actual arc/inner dependency changed; renew observation: '+state['id'])
            base = baseline['blink_000']
            mapping = np.full(base['counts']['points'], -1, np.int32)
            mapping[ids] = np.arange(len(ids), dtype=np.int32)
            face_ids = observed['head_originalPolygonIds']
            eyes = {s: capture['capture_mesh'](bpy.data.objects['R11 Adult sclera '+s], binding) for s in ('L', 'R')}
            for side in ('L', 'R'):
                if not np.array_equal(eyes[side]['world'], observed['eye_'+side+'_world']):
                    raise RuntimeError('Current true globe changed: '+side)
            centers = {s: np.asarray(obs['eyes'][s]['centerWorld']) for s in ('L', 'R')}
            set_state(states[0])
            modifier, group, cache, mesh = api('adult_blink_midpath_nodes_r11')['build'](
                head, base['counts']['points'], ids, arcs, masks, centers, base['matrix'],
                clearance=plan['clearanceMeters'], minimum_wall=plan['minimumWallMeters'])
            strength = group.nodes['R11 inner contact strength'].outputs[0]
            for state in states:
                set_state(state)
                strength.default_value = 0.
                bpy.context.view_layer.update()
                actual_arc = snapshot()
                original = baseline[state['id']]
                if not capture['exact_topology'](original, actual_arc):
                    raise RuntimeError('Arc changed actual polygon connectivity')
                expected_arc = original['world'].copy()
                arc_delta = sum(arc_api['evaluate_delta'](state[s], arcs[s]) for s in ('L', 'R'))
                expected_arc[ids] += arc_delta
                arc_parity = float(np.linalg.norm(actual_arc['world']-expected_arc, axis=1).max())
                if arc_parity>2e-6:
                    raise RuntimeError('Actual arc field coordinate parity exceeded 2 micrometres')
                tri, _, _ = capture['roi_triangles'](actual_arc, face_ids, mapping)
                arc_world = actual_arc['world'][ids]
                inner_delta, inner_metrics = reference['correct'](
                    arc_world, tri, {s: eyes[s]['world'] for s in ('L', 'R')},
                    {s: eyes[s]['triangles'] for s in ('L', 'R')}, centers, masks,
                    clearance=plan['clearanceMeters'], minimum_wall=plan['minimumWallMeters'],
                    point_normals=actual_arc['pointNormals'][ids])
                strength.default_value = 1.
                bpy.context.view_layer.update()
                final = snapshot()
                if not capture['exact_topology'](original, final):
                    raise RuntimeError('Inner layer changed actual polygon connectivity')
                expected_final = actual_arc['world'].copy()
                expected_final[ids] += inner_delta
                parity = float(np.linalg.norm(final['world']-expected_final, axis=1).max())
                if parity>2e-6:
                    diagnostic = output.with_suffix('.contact-parity-failed.npz')
                    np.savez_compressed(diagnostic, originalVertexIds=ids, arcWorld=arc_world,
                                        arcTriangles=tri, arcPointNormals=actual_arc['pointNormals'][ids],
                                        actualWorld=final['world'][ids], expectedWorld=expected_final[ids])
                    raise RuntimeError('Actual dynamic contact parity exceeded 2 micrometres: '+str(parity))
                actual_tri, _, _ = capture['roi_triangles'](final, face_ids, mapping)
                actual_world = final['world'][ids]
                changed = np.linalg.norm(actual_world-original['world'][ids], axis=1)>1e-9
                crossings = reference['changed_intersections'](actual_world, actual_tri, changed)
                if crossings:
                    diagnostic = output.with_suffix('.crossing-failed.npz')
                    np.savez_compressed(diagnostic, originalVertexIds=ids, actualWorld=actual_world,
                                        actualTriangles=actual_tri, crossingTrianglePairs=np.asarray(sorted(crossings)))
                    raise RuntimeError('Candidate created actual local surface interior crossings')
                for side in ('L', 'R'):
                    for label in ('changedLocalVertexIds', 'remainingNegativeLocalVertexIds'):
                        inner_metrics[side][label] = [int(v) for v in inner_metrics[side][label]]
                    if inner_metrics[side]['remainingNegativeFrontDepthCount']:
                        raise RuntimeError('Inner points remain inside the actual globe after the bounded contact step')
                row = {'id': state['id'], 'controls': {s: state[s] for s in ('L', 'R')}, 'warm': state['warm'],
                       'actualArcParityMaxMeters': arc_parity, 'actualContactParityMaxMeters': parity,
                       'polygonConnectivityExact': True, 'actualInteriorCrossings': 0,
                       'innerContact': inner_metrics}
                if state['id'] in ('blink_000', 'blink_100'):
                    outer = original['pointNormals'][:, 1]<=.1
                    outer_delta = np.linalg.norm(final['world'][outer]-original['world'][outer], axis=1)
                    row['endpointOuterVertexMaximumDifferenceMeters'] = float(outer_delta.max())
                    if np.any(outer_delta):
                        raise RuntimeError('Endpoint outer surface changed')
                    row['endpointVisibleGlobes'] = {}
                    for side in ('L', 'R'):
                        before_tri, _, _ = capture['roi_triangles'](original, face_ids, mapping)
                        before = visibility(original['world'][ids], before_tri, eyes[side]['world'], eyes[side]['triangles'], refine=True)
                        after = visibility(actual_world, actual_tri, eyes[side]['world'], eyes[side]['triangles'], refine=True)
                        if before['visibleSampleCount'] != after['visibleSampleCount']:
                            raise RuntimeError('Endpoint actual visible eye aperture changed')
                        if state['id']=='blink_100' and after['visibleSampleCount']:
                            raise RuntimeError('Full closed eye regained visible globe samples')
                        if before['contour'] != after['contour']:
                            raise RuntimeError('Endpoint front-visible boundary changed')
                        row['endpointVisibleGlobes'][side] = {'visibleSampleCount': after['visibleSampleCount'],
                                                              'frontVisibleAreaSquareMeters': after['frontVisibleAreaSquareMeters'],
                                                              'contourExact': True}
                if state['id'] in ('blink_050', 'blink_100'):
                    row['actualFibreBinding'] = binding['validate'](neutral=False)
                report['checks'].append(row)
                write_attempt()
                print('R11_MIDPATH_ACTUAL '+json.dumps({'state': state['id'], 'arcParityMeters': arc_parity,
                      'innerParityMeters': parity, 'innerChanged': {s: inner_metrics[s]['changedPoints'] for s in ('L', 'R')}}), flush=True)
            for name, value in previous.items():
                keys[name].value = value
            bpy.context.view_layer.update()
        now_keys = key_api['_keys'](head.data)
        if set(now_keys) != set(original_keys):
            raise RuntimeError('Original shape keys changed')
        for name, old in original_keys.items():
            if not np.array_equal(old['coords'], now_keys[name]['coords']) or any(old[k]!=now_keys[name][k] for k in old if k!='coords'):
                raise RuntimeError('Original static/expression key changed: '+name)
        if [m.name if m else None for m in head.data.materials] != original_materials:
            raise RuntimeError('Original materials changed')
        if [m.name for m in head.modifiers if m!=modifier] != original_modifiers:
            raise RuntimeError('Original modifier order changed')
        if pose_api['_pose_contract']()!=original_pose or sha(source)!=source_sha:
            raise RuntimeError('Original rig pose or source changed')
        head.active_shape_key_index = original_active
        report.update(modifier=modifier.name, nodeGroup=group.name, lookupObject=cache.name,
                      sampledLocalGeometryPassed=True, fullClosurePreserved=True,
                      neutralVisibleAperturePreserved=True, existingKeysMaterialsModifiersPreserved=True,
                      headTurnCompatibilityVerified=False)
        bpy.context.scene[MARKER] = json.dumps(report)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
            raise RuntimeError('Candidate save failed')
        report.update(output=output_relative, outputSha256=sha(output), candidateSaved=True)
        with report_file.open('x', encoding='utf8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        success = True
        print('R11_MIDPATH_SAVED '+json.dumps({'output': output_relative, 'sha256': report['outputSha256']}), flush=True)
        return report
    except Exception as exc:
        report.update(error=str(exc), candidateSaved=False)
        write_attempt()
        raise
    finally:
        for name, value in previous.items():
            keys[name].value = value
        head.active_shape_key_index = original_active
        if not success:
            if modifier is not None:
                head.modifiers.remove(modifier)
            if cache is not None:
                bpy.data.objects.remove(cache, do_unlink=True)
            if mesh is not None and mesh.users==0:
                bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:
                bpy.data.node_groups.remove(group)
            if MARKER in bpy.context.scene:
                del bpy.context.scene[MARKER]
        bpy.context.view_layer.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.plan, args.output)
