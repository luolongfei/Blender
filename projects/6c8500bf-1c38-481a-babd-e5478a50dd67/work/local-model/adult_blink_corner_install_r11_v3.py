"""Install a frozen actual-L3 late canthal correction in an independent source.

Existing shape keys, materials and globe geometry remain intact. Only a
measured point-domain geometry-node field is added, inactive through blink .8.
This must be followed by an actual closed/partial image review.
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
MARKER = 'R11_blink_corner_surface_refinement'
DEPENDENCY_HELPER_SHA = 'eb8e03cb02f12d97d51c33f0989d003e7892a4099d677b6b33d9bec64d2f1695'


def api(name): return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def path(relative, suffix):
    value = (project_path()/relative).resolve()
    value.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if value.suffix != suffix: raise RuntimeError('Independent reconstruction artifact required')
    return value


def load_json(file): return json.loads(file.read_text(encoding='utf8'))


def prepare(source_relative, plan_relative, output_relative):
    import bpy
    source = path(source_relative, '.blend'); output = path(output_relative, '.blend')
    plan_path = path(plan_relative, '.json'); record_path = output.with_suffix('.blink-corner.json')
    if source==output or output.exists() or record_path.exists() or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved independent source/output required')
    if MARKER in bpy.context.scene or 'R11_blink_closure_refinement' not in bpy.context.scene:
        raise RuntimeError('Install once after the actual first closure refinement')
    dependency_path = output.with_suffix('.dependency.json'); dependency_arrays_path = output.with_suffix('.dependency.npz')
    if dependency_path.exists() or dependency_arrays_path.exists(): raise RuntimeError('Preserve earlier dependency diagnosis; use a fresh output name')
    if sha(Path(__file__).with_name('adult_blink_corner_dependency_r11.py'))!=DEPENDENCY_HELPER_SHA: raise RuntimeError('Frozen dependency helper changed')
    plan = load_json(plan_path)
    if not plan.get('offlineGeometryPassed'): raise RuntimeError('Frozen offline physical geometry plan required')
    for name, checksum in plan['helperHashes'].items():
        if sha(Path(__file__).with_name(name)) != checksum: raise RuntimeError('Frozen helper changed '+name)
    observation_path = path(plan['observation'], '.json'); observation = load_json(observation_path)
    arrays_path = observation_path.with_suffix('.npz'); fields_path = path(plan['fields'], '.npz')
    if sha(observation_path)!=plan['observationSha256'] or sha(arrays_path)!=observation['arraysSha256'] or sha(fields_path)!=plan['fieldsSha256']:
        raise RuntimeError('Frozen actual observation or geometry field changed')
    with np.load(arrays_path) as data: arrays = {k: data[k].copy() for k in data.files}
    with np.load(fields_path) as data: fields_data = {k: data[k].copy() for k in data.files}
    fields = {s: {k: fields_data[s+'_'+k] for k in ('angle', 'radiusChange')} for s in ('L', 'R')}
    head = bpy.data.objects[HEAD]; keys = head.data.shape_keys.key_blocks
    if any(abs(k.value)>1e-7 for k in keys if 'blink' in k.name.lower() or 'smile' in k.name.lower()):
        raise RuntimeError('Expression controls must start at neutral')
    key_api = api('adult_jaw_contour_r11'); prior_keys = key_api['_keys'](head.data)
    pose_api = api('production_uv_r08'); prior_pose = pose_api['_pose_contract']()
    capture = api('adult_blink_recompute_r11'); binding = api('face_fibre_binding_r11')
    source_sha = sha(source); original_active = head.active_shape_key_index
    modifier = group = cache = mesh = None; success = False; result = {}

    def set_values(value):
        for side in ('L', 'R'): keys['R11 blink '+side].value = value
        bpy.context.view_layer.update()

    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            baseline = {}
            def current_fibres():
                dg = bpy.context.evaluated_depsgraph_get(); result = {}
                for name in binding['OBJECTS']:
                    evaluated = bpy.data.objects[name].evaluated_get(dg)
                    result[name] = binding['_transform'](binding['_array'](evaluated.data.position_data, 'vector'), evaluated.matrix_world)
                return result
            set_values(1.); closed = capture['_snapshot'](head); baseline[1.] = closed
            if not np.array_equal(closed['originalVertexIds'], arrays['head_originalVertexIds']) or not np.array_equal(closed['triangles'], arrays['head_triangles']):
                raise RuntimeError('Observed final head topology changed')
            dependency, dependency_arrays = api('adult_blink_corner_dependency_r11')['compare_closed'](closed, observation, arrays, fields)
            np.savez_compressed(dependency_arrays_path, **dependency_arrays)
            dependency.update(source=source_relative, sourceSha256=source_sha,
                              observedSourceSha256=observation['sourceSha256'],
                              helperSha256=DEPENDENCY_HELPER_SHA,
                              arraysSha256=sha(dependency_arrays_path), sourceSaved=False,
                              scope='Actual comparison before any geometry-node installation; all old-box differences retained.')
            with dependency_path.open('x', encoding='utf8') as stream: json.dump(dependency, stream, ensure_ascii=False, indent=2)
            result['actualDependencyCompatibility'] = dependency
            result['dependencyDiagnostic'] = str(dependency_path.relative_to(project_path()))
            if not dependency['compatibleActualDependency']:
                raise RuntimeError('Actual canthal input/incident faces/two rings/2 mm halo changed; see '+str(dependency_path))
            if np.max(abs(np.asarray(head.matrix_world)-np.asarray(observation['headMatrixWorld'])))>2e-7:
                raise RuntimeError('Observed REST head transform changed')
            eyes = {s: capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+s]) for s in ('L', 'R')}
            for side, eye in eyes.items():
                if eye['world'].shape!=arrays['eye_'+side+'_world'].shape or np.max(abs(eye['world']-arrays['eye_'+side+'_world']))>2e-7:
                    raise RuntimeError('Observed actual globe envelope changed '+side)
            for value in (0., .8, .9):
                set_values(value); baseline[value] = capture['_snapshot'](head)
                if value==0.: neutral_fibres_before = current_fibres()
            modifier, group, cache, mesh = api('adult_blink_corner_nodes_r11')['build'](
                head, closed['fullCounts'][0], arrays['head_originalVertexIds'], fields,
                {s: observation['eyes'][s]['centerWorld'] for s in ('L', 'R')}, observation['headMatrixWorld'], onset=.8)
            samples = []
            contact_api = api('adult_blink_corner_plan_r11')
            for value in (.8, .9, 1.):
                set_values(value); snap = capture['_snapshot'](head)
                if snap['fullCounts']!=closed['fullCounts'] or not np.array_equal(snap['originalVertexIds'], closed['originalVertexIds']) or not np.array_equal(snap['triangles'], closed['triangles']):
                    raise RuntimeError('Canthal field changed final topology')
                if value==.8 and np.max(abs(snap['world']-baseline[value]['world']))>2e-6:
                    raise RuntimeError('Early/mid blink altered by late field')
                if value==1.:
                    expected = closed['world']+fields_data['deltaWorld']
                    error = float(np.linalg.norm(snap['world']-expected, axis=1).max())
                    if error>2e-6: raise RuntimeError('Actual GN field differs from frozen geometry plan '+str(error))
                    result['actualGNPlanParityMaxMeters'] = error
                measured = {s: capture['_measure'](snap, eyes[s]) for s in ('L', 'R')}
                contact = {}
                for side in ('L', 'R'):
                    selected = (abs(fields[side]['angle'])>1e-10) | (abs(fields[side]['radiusChange'])>1e-10)
                    center = np.asarray(observation['eyes'][side]['centerWorld'])
                    gaps = []
                    for surface in (baseline[value]['world'], snap['world']):
                        points = surface[selected]; origins = np.broadcast_to(center, points.shape).copy(); origins[:, 0] = points[:, 0]
                        radial = points-origins; radius = np.linalg.norm(radial, axis=1); direction = radial/radius[:, None]
                        globe_radius = contact_api['ray_radius'](origins, direction, arrays['eye_'+side+'_world'], arrays['eye_'+side+'_triangles'])
                        gaps.append(radius-globe_radius)
                    valid = np.isfinite(gaps[0]) & np.isfinite(gaps[1])
                    if not valid.any(): raise RuntimeError('No actual globe contact samples '+side)
                    before = gaps[0][valid]; after = gaps[1][valid]
                    worsened = np.maximum(0., np.minimum(before, 0.)-np.minimum(after, 0.))
                    contact[side] = {'samples': int(valid.sum()), 'unmappedSamples': int((~valid).sum()),
                                     'minimumOriginalRadialGapMeters': float(before.min()),
                                     'minimumNewRadialGapMeters': float(after.min()),
                                     'maximumAdditionalPenetrationMeters': float(worsened.max())}
                    if worsened.max()>.00002: raise RuntimeError('Late path adds actual globe penetration '+side+' '+str(worsened.max()))
                samples.append({'blink': value, 'actualGlobeRadialContact': contact, 'eyes': {s: {k: v for k, v in item.items() if k not in ('contour', 'disconnectedVisibleColumns')} for s, item in measured.items()}})
                print('R11_CANTHAL_ACTUAL '+json.dumps(samples[-1]), flush=True)
            result['fullClosedBinding'] = binding['validate'](neutral=False)
            set_values(0.)
            neutral_after = capture['_snapshot'](head)
            if neutral_after['fullCounts']!=baseline[0.]['fullCounts']:
                raise RuntimeError('Current neutral topology changed')
            neutral_error = float(np.linalg.norm(neutral_after['fullWorld']-baseline[0.]['fullWorld'], axis=1).max())
            if neutral_error>2e-6: raise RuntimeError('Current neutral head surface changed '+str(neutral_error))
            neutral_fibres_after = current_fibres(); fibre_errors = {}
            for name, before in neutral_fibres_before.items():
                after = neutral_fibres_after[name]
                if after.shape!=before.shape: raise RuntimeError('Current neutral fibre count changed '+name)
                error = float(np.linalg.norm(after-before, axis=1).max()); fibre_errors[name] = error
                if error>2e-6: raise RuntimeError('Current neutral fibre shape changed '+name+' '+str(error))
            result['currentNeutralSurfaceParityMaxMeters'] = neutral_error
            result['currentNeutralFibreParityMaxMeters'] = fibre_errors
            result['currentNeutralBinding'] = binding['validate'](neutral=False)
            result['neutralReference'] = 'Actual current saved 24.6 mm static shape before this installation, not the historical 22.4 mm original binding.'
            result['samples'] = samples
            result['fullClosureFrontVisibilityPassed'] = all(samples[-1]['eyes'][s]['visibleSampleCount']==0 for s in ('L', 'R'))
        head.active_shape_key_index = original_active
        after_keys = key_api['_keys'](head.data)
        for name, original in prior_keys.items():
            current = after_keys[name]
            if not np.array_equal(current['coords'], original['coords']) or any(current[k]!=original[k] for k in original if k!='coords'):
                raise RuntimeError('Existing expression/static key changed '+name)
        if pose_api['_pose_contract']()!=prior_pose or sha(source)!=source_sha:
            raise RuntimeError('Pose or source changed')
        result.update(source=source_relative, sourceSha256=source_sha, plan=plan_relative, planSha256=sha(plan_path),
                      helperSha256=sha(__file__), modifier=modifier.name, nodeGroup=group.name, lookupObject=cache.name,
                      preservedKeys=list(prior_keys), appearanceAccepted=False, actualContactVerified=False,
                      scope='Late local eye-corner geometry, inactive through blink .8; actual image and combined expressions still require review.')
        bpy.context.scene[MARKER] = json.dumps(result)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)): raise RuntimeError('Canthal candidate save failed')
        result.update(output=output_relative, outputSha256=sha(output), savedBlend=True)
        with record_path.open('x', encoding='utf8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)
        success = True
        print('R11_CANTHAL_SAVED '+json.dumps({'output': output_relative, 'sha256': result['outputSha256'], 'fullClosureFrontVisibilityPassed': result['fullClosureFrontVisibilityPassed']}), flush=True)
        return result
    finally:
        set_values(0.); head.active_shape_key_index = original_active
        if not success:
            if modifier is not None: head.modifiers.remove(modifier)
            if cache is not None: bpy.data.objects.remove(cache, do_unlink=True)
            if mesh is not None and mesh.users==0: bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0: bpy.data.node_groups.remove(group)
            if MARKER in bpy.context.scene: del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    prepare(args.source, args.plan, args.output)
