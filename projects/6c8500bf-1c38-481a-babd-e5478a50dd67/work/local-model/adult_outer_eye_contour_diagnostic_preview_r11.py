"""DIAGNOSTIC ONLY: one neutral image with two known outer-bridge crossings.

Loads an explicitly offline proposal, verifies its actual local dependency
geometry on the chosen current source, and adds a temporary position layer.
No blend, production file or expression key is saved or created.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

BASE = 'upper_lid_roll_review_r11.py'
BASE_SHA = 'a70b7dff532af7b5414c2d7c120d45033386e8fcb749cf66ddf881b5d7a15ff6'
HEAD = 'R11 Female head study'
OUTER = 'R11 refined outer eye corners'
FIXED = {'R11 blink L': 0., 'R11 blink R': 0., 'R11 gentle smile': 0.,
         'R11 natural soft smile': 0., 'R11 warm closed smile': 0.,
         'R11 refined upper lid roll': 1., 'R11 lower face contour v2': 1., OUTER: 0.}


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def build(head, local_delta):
    import bpy
    name = 'R11 continuous outer contour PREVIEW ONLY'
    modifier = group = obj = mesh = None
    try:
        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(len(local_delta))
        mesh.vertices.foreach_set('co', np.zeros(len(local_delta)*3, np.float32))
        mesh.attributes.new('contour_preview_delta', 'FLOAT_VECTOR', 'POINT').data.foreach_set('vector', local_delta.astype(np.float32).ravel())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.hide_render = True
        obj.hide_set(True)
        group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        nodes, links = group.nodes, group.links
        socket = api('adult_blink_midpath_nodes_r11')['_compatible_geometry_socket']()
        source, output = nodes.new('NodeGroupInput'), nodes.new('NodeGroupOutput')
        lookup = nodes.new('GeometryNodeObjectInfo')
        lookup.transform_space = 'RELATIVE'
        lookup.inputs['Object'].default_value = obj
        lookup.inputs['As Instance'].default_value = False
        named = nodes.new('GeometryNodeInputNamedAttribute')
        named.data_type = 'FLOAT_VECTOR'
        named.inputs['Name'].default_value = 'contour_preview_delta'
        index = nodes.new('GeometryNodeInputIndex')
        sample = nodes.new('GeometryNodeSampleIndex')
        sample.data_type, sample.domain, sample.clamp = 'FLOAT_VECTOR', 'POINT', False
        links.new(lookup.outputs['Geometry'], socket(sample.inputs))
        links.new(named.outputs['Attribute'], sample.inputs['Value'])
        links.new(index.outputs['Index'], sample.inputs['Index'])
        length = nodes.new('ShaderNodeVectorMath')
        length.operation = 'LENGTH'
        links.new(sample.outputs['Value'], length.inputs[0])
        selected = nodes.new('ShaderNodeMath')
        selected.operation = 'GREATER_THAN'
        selected.inputs[1].default_value = 0.
        links.new(length.outputs['Value'], selected.inputs[0])
        position = nodes.new('GeometryNodeSetPosition')
        links.new(socket(source.outputs), socket(position.inputs))
        links.new(selected.outputs[0], position.inputs['Selection'])
        links.new(sample.outputs['Value'], position.inputs['Offset'])
        links.new(socket(position.outputs), socket(output.inputs))
        modifier = head.modifiers.new(name, 'NODES')
        modifier.node_group = group
        return modifier, group, obj, mesh
    except Exception:
        if modifier is not None:
            head.modifiers.remove(modifier)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        if group is not None:
            bpy.data.node_groups.remove(group)
        raise


def main():
    import bpy
    if sha(Path(__file__).with_name(BASE)) != BASE_SHA:
        raise RuntimeError('Frozen evaluated-camera renderer changed')
    renderer = api('upper_lid_roll_review_r11');project = renderer['P']
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'plan', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    plan_path = (project/args.plan).resolve();plan_path.relative_to(project.resolve())
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if plan.get('view') not in ('eyes', 'front') or plan.get('hideGroom') is not False or len(plan['frames']) != 1 or plan['frames'][0]['keys'] != FIXED:
        raise RuntimeError('Use the one-frame neutral contour preview plan')
    output = (project/args.output).resolve();output.relative_to((project/'previews/reconstruction-r02').resolve())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError('Preserve earlier images and diagnostics')
    proposal_path = project/'work/reconstruction-r02/adult-outer-eye-contour-surface-preview-r11-v2.json'
    proposal = json.loads(proposal_path.read_text(encoding='utf8'))
    if sha(proposal_path) != '93446b87baf9de8b20bef0d73301b513c9a00c1a720ca834f5e8cb34985fa30f' or sha(proposal_path.with_suffix('.npz')) != '7a896907ef20fd8c165d4442627416a8afeccb5725b3530f341aa3f0b8f6aefe':
        raise RuntimeError('Explicitly authorized two-crossing diagnostic proposal changed')
    if proposal.get('failure') or proposal.get('sourceTessellationCrossings') != 2:
        raise RuntimeError('Only the specifically recorded two-pair diagnostic is authorized')
    data = dict(np.load(proposal_path.with_suffix('.npz')))
    if set(map(tuple, data['crossingPairs'].tolist())) != {(174822, 174832), (174822, 174835)}:
        raise RuntimeError('Known diagnostic pair identity changed')
    known_local = np.unique(data['triangles'][data['crossingPairs']].ravel())
    known_mask = np.zeros(len(data['originalVertexIds']), bool)
    known_mask[known_local] = True
    patch_local = np.unique(data['triangles'][known_mask[data['triangles']].any(1)])
    known_patch_global = data['originalVertexIds'][patch_local]
    evidence_path = (project/proposal['actualInput']).resolve();evidence_path.relative_to(project.resolve())
    if sha(evidence_path) != proposal['actualInputSha256']:
        raise RuntimeError('Actual geometric baseline evidence changed')
    evidence = dict(np.load(evidence_path))
    head = bpy.data.objects[HEAD];keys = head.data.shape_keys.key_blocks
    if any(name not in keys or float(keys[name].value) != value for name, value in FIXED.items()):
        raise RuntimeError('Saved neutral source with rejected outer layer disabled required')
    if float(head.get('R11 lower face continuity', -1.)) != 0. or any(m.type == 'ARMATURE' for m in head.modifiers):
        raise RuntimeError('Current unbound neutral eye-production source required')
    capture = api('adult_blink_recompute_r11');contact = api('adult_blink_midpath_contact_r11')
    shared = api('upper_lid_relief_install_r11');rest = api('face_contact_audit_r11')['rest_evaluation']
    layers = api('adult_neutral_eye_outerends_radial_install_r11_v2')['actual_layers']
    payload, state = {}, {}
    created = []
    diagnostics = {'previewOnly': True, 'diagnosticOnly': True, 'technicalAcceptance': False,
                   'knownDefect': 'Two recorded near-neighbour outer-bridge surface crossings; this image is solely for artistic direction. Do not adopt or deliver this geometry.',
                   'knownSourceTrianglePairs': data['crossingPairs'],
                   'knownGlobalVertexIds': data['originalVertexIds'][known_local],
                   'knownCrossingPatchGlobalVertexIds': known_patch_global,
                   'source': args.source, 'sourceSaved': False,
                   'appearanceAccepted': False, 'motionValidated': False, 'allPhysicalContactPassed': False,
                   'helperSha256': sha(__file__), 'proposalSha256': sha(proposal_path), 'frames': []}

    def persist():
        output.mkdir(parents=True, exist_ok=True)
        array_file = output/'actual-neutral-contour-preview.npz'
        np.savez_compressed(array_file, **payload)
        diagnostics['arraysSha256'] = sha(array_file)
        (output/'actual-neutral-contour-preview.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=serial), encoding='utf8')

    def prepare():
        with rest():
            before = capture['_snapshot'](head)
            ids = data['originalVertexIds'];delta = data['deltaWorld'];tri = data['triangles']
            active = np.linalg.norm(delta, axis=1) > 1e-10
            dependency = active.copy()
            for _ in range(2):
                dependency[np.unique(tri[dependency[tri].any(1)])] = True
            dep_ids = ids[dependency]
            if dep_ids.max() >= len(before['fullWorld']):
                raise RuntimeError('Current evaluated index range no longer matches local proposal dependencies')
            difference = np.linalg.norm(before['fullWorld'][dep_ids]-data['beforeWorld'][dependency], axis=1)
            if difference.max() > 2e-7:
                raise RuntimeError('Actual local eye dependency geometry changed; rebuild the neutral proposal')
            eyes = {side: capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+side]) for side in ('L', 'R')}
            for side, eye in eyes.items():
                if eye['world'].shape != evidence['eye_'+side+'_world'].shape or np.linalg.norm(eye['world']-evidence['eye_'+side+'_world'], axis=1).max() > 2e-7 or not np.array_equal(eye['triangles'], evidence['eye_'+side+'_triangles']):
                    raise RuntimeError('Actual current globe differs from this preview proposal')
            full = np.zeros_like(before['fullWorld'])
            full[ids] = delta
            local = full@np.linalg.inv(np.asarray(head.matrix_world, float)[:3, :3]).T
            created.extend(build(head, local))
            head.update_tag();bpy.context.view_layer.update()
            state.update(before=before, eyes=eyes, full=full, activeFull=np.linalg.norm(full, axis=1) > 0.,
                         topology=shared['polygon_contract'](head))
            payload.update(beforeFullWorld=before['fullWorld'], expectedDeltaFullWorld=full,
                           dependencyGlobalVertexIds=dep_ids, dependencyDifferenceMeters=difference)
            diagnostics['dependency'] = {'points': len(dep_ids), 'maximumDifferenceMeters': float(difference.max()),
                'scope': 'Only actual nonzero eye displacement plus two graph rings; unrelated lower-face successor geometry is permitted.'}
        persist()

    original_api = renderer['main'].__globals__['api']

    def preview_api(name):
        if name == 'final_detail_review_r09':
            prepare()
        result = original_api('face_fibre_binding_r11_v2' if name == 'face_fibre_binding_r11' else name)
        if name == 'face_fibre_binding_r11':
            validate = result['validate']

            def validate_preview(*positional, **keywords):
                with rest():
                    before = state['before'];after = capture['_snapshot'](head)
                    payload.update(afterFullWorld=after['fullWorld'], afterTriangles=after['triangles'],
                                   afterOriginalVertexIds=after['originalVertexIds'], afterOriginalTriangleIds=after['originalTriangleIds'])
                    if shared['polygon_contract'](head) != state['topology'] or not np.array_equal(before['originalVertexIds'], after['originalVertexIds']):
                        raise RuntimeError('Temporary neutral position layer changed actual polygon topology or ROI')
                    error = float(np.linalg.norm(after['fullWorld']-before['fullWorld']-state['full'], axis=1).max())
                    outside = ~state['activeFull']
                    if error > 2e-6 or not np.array_equal(before['fullWorld'][outside], after['fullWorld'][outside]):
                        raise RuntimeError('Actual native contour offset failed local parity or outside exactness')
                    row = {'nativeReferenceMaximumDifferenceMeters': error, 'outsideDisplacementSupportExact': True}
                    diagnostics['frames'].append(row);persist()
                    row['actualGlobeAndLayers'], row['pendingInnerGlobalVertexIds'] = layers(before, after, state['eyes'], contact)
                    changed = np.linalg.norm(after['world']-before['world'], axis=1) > 1e-10
                    cross = contact['changed_intersections'](after['world'], after['triangles'], changed)
                    payload['crossingPairs'] = np.asarray(sorted(cross), np.int32).reshape(-1, 2)
                    row['actualPostTessellationCrossings'] = len(cross);persist()
                    row['actualCrossingLocations'] = [{'actualTriangleRows': list(pair),
                        'globalVertexIds': after['originalVertexIds'][after['triangles'][list(pair)]],
                        'worldVertices': after['world'][after['triangles'][list(pair)]]} for pair in sorted(cross)]
                    outside_known = [pair for pair in cross if not np.isin(after['originalVertexIds'][after['triangles'][list(pair)]], known_patch_global).all()]
                    row['crossingsOutsideKnownDiagnosticPatch'] = len(outside_known)
                    row['technicalAcceptance'] = False
                    row['diagnosticOnly'] = True
                    persist()
                    if outside_known:
                        raise RuntimeError('Unexpected crossings outside the explicitly authorized diagnostic patch')
                    print('DIAGNOSTIC_ONLY_KNOWN_OUTER_BRIDGE_CROSSINGS '+json.dumps({'actualCount': len(cross), 'productionAccepted': False}), flush=True)
                return validate(*positional, **keywords)

            result['validate'] = validate_preview
        return result

    renderer['main'].__globals__['api'] = preview_api
    try:
        renderer['main']()
        report_path = output/'render-report.json'
        report = json.loads(report_path.read_text())
        report['continuousContourPreview'] = {'helper': Path(__file__).name, 'helperSha256': sha(__file__),
            'previewOnly': True, 'diagnosticOnly': True, 'technicalAcceptance': False, 'geometryReport': 'actual-neutral-contour-preview.json', 'temporaryLayerRemoved': False,
            'knownDefect': 'Two source outer-bridge surface crossings, with actual current pair locations in the geometry report.', 'scope': 'Diagnostic artistic-direction image only. No blend save, delivery, technical acceptance or motion acceptance.'}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    except Exception as error:
        diagnostics['failure'] = str(error);persist();raise
    finally:
        renderer['main'].__globals__['api'] = original_api
        if created:
            modifier, group, obj, mesh = created
            head.modifiers.remove(modifier)
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
            bpy.data.node_groups.remove(group)
            head.update_tag();bpy.context.view_layer.update()
        final_report_path = output/'render-report.json'
        if final_report_path.exists():
            final_report = json.loads(final_report_path.read_text(encoding='utf8'))
            if 'continuousContourPreview' in final_report:
                final_report['continuousContourPreview']['temporaryLayerRemoved'] = True
                final_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding='utf8')
    print('R11_CONTINUOUS_OUTER_CONTOUR_DIAGNOSTIC_DONE '+json.dumps({'output': args.output, 'sourceSaved': False}), flush=True)


if __name__ == '__main__':
    main()
