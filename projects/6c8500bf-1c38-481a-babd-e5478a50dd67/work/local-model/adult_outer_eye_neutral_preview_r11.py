"""Two neutral direction previews, in memory only; no blend is saved.

Frame 1 uses the rejected field at 0.25 for a limited comparison. Frame 2 uses
new surface-distance support and a smooth canthal angle profile. The original
lookup, public Key value and source file are preserved. Full motion validation
must wait for an actual visual decision on these explicitly marked previews.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

RENDERER = 'upper_lid_roll_review_r11.py'
RENDERER_SHA = 'a70b7dff532af7b5414c2d7c120d45033386e8fcb749cf66ddf881b5d7a15ff6'
SURFACE = 'adult_outer_eye_surface_support_preview_r11.py'
SURFACE_SHA = 'a53ca6cb8c7c403e58fbb3f50303d5ef90026d7afdb0d6fcc10d84da69036d93'
BINDING_SHA = '47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49'
INSTALLER_SHA = '1dc971cd8c6b4b8b395a3641f8cc4cdef1844299fa91fbc7142540db52b073dc'
HEAD = 'R11 Female head study'
OUTER = 'R11 refined outer eye corners'
MARKER = 'R11_refined_radial_outer_eye_candidate'
FIXED = {'R11 blink L': 0., 'R11 blink R': 0., 'R11 gentle smile': 0.,
         'R11 natural soft smile': 0., 'R11 warm closed smile': 0.,
         'R11 refined upper lid roll': 1., 'R11 lower face contour v2': 1.}
CREASE_EDGES = ((46479, 180975), (39725, 93132), (48743, 203050), (25070, 115256))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def crease_records(world, triangles, ids):
    result = []
    for pair in CREASE_EDGES:
        local = [np.flatnonzero(ids == vertex) for vertex in pair]
        if any(len(row) != 1 for row in local):
            raise RuntimeError('Current observed eye ROI lost a known diagnostic edge')
        a, b = (int(row[0]) for row in local)
        faces = np.flatnonzero(np.any(triangles == a, axis=1) & np.any(triangles == b, axis=1))
        row = {'globalVertexIds': pair, 'lengthMeters': float(np.linalg.norm(world[a]-world[b])),
               'adjacentActualTriangleRows': faces}
        if len(faces) == 2:
            points = world[triangles[faces]]
            normal = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
            row['adjacentNormalAngleDegrees'] = float(np.degrees(np.arccos(np.clip(normal[0]@normal[1], -1., 1.))))
        result.append(row)
    return result


def main():
    import bpy

    folder = Path(__file__).resolve().parent
    for file, expected in ((RENDERER, RENDERER_SHA), (SURFACE, SURFACE_SHA),
                           ('face_fibre_binding_r11_v2.py', BINDING_SHA),
                           ('adult_neutral_eye_outerends_radial_install_r11_v2.py', INSTALLER_SHA)):
        if sha(folder/file) != expected:
            raise RuntimeError('Reviewed preview dependency changed: '+file)
    renderer = api('upper_lid_roll_review_r11')
    project = renderer['P']
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'plan', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    plan_path = (project/args.plan).resolve()
    plan_path.relative_to(project.resolve())
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    if plan.get('view') != 'eyes' or plan.get('hideGroom') is not False or len(plan['frames']) != 2:
        raise RuntimeError('Use only the explicit two-neutral-eye preview plan')
    for row, strength in zip(plan['frames'], (.25, 1.)):
        if row['keys'] != {**FIXED, OUTER: strength}:
            raise RuntimeError('Only the planned neutral outer-eye candidates may change')
    head = bpy.data.objects[HEAD]
    keys = head.data.shape_keys.key_blocks
    if any(name not in keys or float(keys[name].value) != value for name, value in {**FIXED, OUTER: 0.}.items()):
        raise RuntimeError('Start from the saved outer-strength-zero reset or intact successor')
    if 'R11 lower face continuity' not in head or float(head['R11 lower face continuity']) != 0.:
        raise RuntimeError('Old lower-face object property must remain zero')
    if any(mod.type == 'ARMATURE' for mod in head.modifiers) or MARKER not in bpy.context.scene:
        raise RuntimeError('Current unbound radial-edit-layer source required')
    marker = json.loads(bpy.context.scene[MARKER])
    group = bpy.data.node_groups.get(marker.get('nodeGroup', ''))
    cache = bpy.data.objects.get(marker.get('lookupObject', ''))
    if group is None or cache is None:
        raise RuntimeError('Existing editable radial layer is missing')
    attributes = {side: 'R11_outer_eye_radial_channels_'+side for side in ('L', 'R')}
    binding = api('face_fibre_binding_r11_v2')
    originals = {side: binding['_array'](cache.data.attributes[name].data, 'vector').copy()
                 for side, name in attributes.items()}
    output = (project/args.output).resolve()
    output.relative_to((project/'previews/reconstruction-r02').resolve())
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError('Preserve prior preview evidence; use a fresh output directory')
    capture = api('adult_blink_recompute_r11')
    contact = api('adult_blink_midpath_contact_r11')
    shared = api('upper_lid_relief_install_r11')
    installation = api('adult_neutral_eye_outerends_radial_install_r11_v2')
    surface = api('adult_outer_eye_surface_support_preview_r11')
    radial = surface['base_api']()
    rest = api('face_contact_audit_r11')['rest_evaluation']
    payload, state = {}, {}
    diagnostics = {'source': args.source, 'previewOnly': True, 'sourceSaved': False,
                   'productionReady': False, 'appearanceAccepted': False, 'allMotionValidated': False,
                   'helperSha256': sha(__file__), 'surfaceHelperSha256': SURFACE_SHA,
                   'frames': [], 'scope': 'Fresh actual neutral geometry and matched camera previews only; no nine-state acceptance.'}

    def persist():
        output.mkdir(parents=True, exist_ok=True)
        array_path = output/'neutral-geometry-preview.npz'
        np.savez_compressed(array_path, **payload)
        diagnostics['arraysSha256'] = sha(array_path)
        (output/'neutral-geometry-preview.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=serial), encoding='utf8')

    def prepare():
        if state:
            return
        with rest():
            before = capture['_snapshot'](head)
            eyes = {side: capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+side]) for side in ('L', 'R')}
            contours = {side: capture['_measure'](before, eyes[side]) for side in eyes}
            fields, arrays, metadata = surface['plan'](before['world'], before['triangles'], before['originalVertexIds'], eyes, contours)
            for side in eyes:
                old, new = marker['plan']['sides'][side], metadata['sides'][side]
                for name in ('centerWorld', 'axisInsideXRange', 'outwardActualAxisExitX', 'actualGlobeOuterX'):
                    if not np.allclose(old[name], new[name], rtol=0., atol=2e-7):
                        raise RuntimeError('Existing native real-globe constants no longer match the actual source')
            dense = {side: np.zeros_like(originals[side]) for side in fields}
            for side in dense:
                if len(dense[side]) != len(before['fullWorld']):
                    raise RuntimeError('Actual current head and existing lookup sizes differ')
                dense[side][before['originalVertexIds']] = fields[side].astype(np.float32)
            state.update(before=before, eyes=eyes, fields=fields, metadata=metadata, dense=dense,
                         topology=shared['polygon_contract'](head))
            payload.update(arrays)
            payload['beforeFullWorld'] = before['fullWorld']
            for side, eye in eyes.items():
                payload['eye_'+side+'_world'] = eye['world']
                payload['eye_'+side+'_triangles'] = eye['triangles']
            diagnostics.update(plan=metadata, neutralCreaseEdges=crease_records(before['world'], before['triangles'], before['originalVertexIds']))
        persist()

    def set_lookup(values):
        for side, name in attributes.items():
            cache.data.attributes[name].data.foreach_set('vector', values[side].astype(np.float32).ravel())
        cache.data.update()
        cache.update_tag()
        head.update_tag()
        bpy.context.view_layer.update()

    frame_number = 0
    original_api = renderer['main'].__globals__['api']

    def preview_api(name):
        if name == 'final_detail_review_r09':
            prepare()  # Base renderer has already checked its untouched saved source.
        result = original_api('face_fibre_binding_r11_v2' if name == 'face_fibre_binding_r11' else name)
        if name == 'face_fibre_binding_r11':
            validate = result['validate']

            def validate_preview(*positional, **keywords):
                nonlocal frame_number
                frame_number += 1
                if frame_number not in (1, 2) or float(head['R11 lower face continuity']) != 0.:
                    raise RuntimeError('Unexpected preview frame or fixed head property')
                chosen = originals if frame_number == 1 else state['dense']
                set_lookup(chosen)
                with rest():
                    before = state['before']
                    after = capture['_snapshot'](head)
                    payload['frame_'+str(frame_number)+'_fullWorld'] = after['fullWorld']
                    payload['frame_'+str(frame_number)+'_triangles'] = after['triangles']
                    payload['frame_'+str(frame_number)+'_originalTriangleIds'] = after['originalTriangleIds']
                    if shared['polygon_contract'](head) != state['topology'] or not np.array_equal(before['originalVertexIds'], after['originalVertexIds']):
                        raise RuntimeError('Neutral preview changed actual polygon connectivity or ROI indexing')
                    fields = {side: chosen[side][before['originalVertexIds']] for side in chosen}
                    metadata = marker['plan'] if frame_number == 1 else state['metadata']
                    expected, physical = radial['apply'](before['world'], before['triangles'], fields, state['eyes'], metadata,
                                                        strength=float(keys[OUTER].value))
                    actual = after['world']-before['world']
                    error = float(np.linalg.norm(actual-expected, axis=1).max())
                    changed = np.linalg.norm(actual, axis=1) > 1e-10
                    payload['frame_'+str(frame_number)+'_expectedDelta'] = expected
                    payload['frame_'+str(frame_number)+'_actualDelta'] = actual
                    row = {'frame': frame_number, 'candidate': 'existing-field-strength-0.25' if frame_number == 1 else 'surface-distance-smoothed-profile',
                           'actualStrength': float(keys[OUTER].value), 'actualPureMaximumDifferenceMeters': error,
                           'physicalReference': physical, 'actualCreaseEdges': crease_records(after['world'], after['triangles'], after['originalVertexIds']),
                           'previewOnly': True, 'appearanceAccepted': False, 'motionValidated': False}
                    diagnostics['frames'].append(row)
                    persist()
                    if error > 2e-6 or any(value['rejectedCount'] for value in physical.values()):
                        raise RuntimeError('Neutral native preview differs from its actual-source reference')
                    row['actualGlobeAndLayers'], row['pendingInnerGlobalVertexIds'] = installation['actual_layers'](before, after, state['eyes'], contact)
                    crossings = contact['changed_intersections'](after['world'], after['triangles'], changed)
                    payload['frame_'+str(frame_number)+'_crossingPairs'] = np.asarray(sorted(crossings), np.int32).reshape(-1, 2)
                    row['actualChangedTriangleCrossings'] = len(crossings)
                    persist()
                    if crossings:
                        raise RuntimeError('Actual neutral preview introduces local head intersections')
                return validate(*positional, **keywords)

            result['validate'] = validate_preview
        return result

    renderer['main'].__globals__['api'] = preview_api
    try:
        renderer['main']()
        if frame_number != 2:
            raise RuntimeError('Both neutral previews must be measured and rendered')
        report_path = output/'render-report.json'
        report = json.loads(report_path.read_text(encoding='utf8'))
        report['neutralDirectionPreview'] = {'helper': Path(__file__).name, 'helperSha256': sha(__file__),
            'geometryReport': 'neutral-geometry-preview.json', 'previewOnly': True, 'sourceSaved': False,
            'fixedHeadProperties': {'R11 lower face continuity': 0.},
            'scope': 'Two in-memory neutral eye directions; existing editable layer will be restored, no production model is written.'}
        for row, measured in zip(report['frames'], diagnostics['frames']):
            row['candidate'] = measured['candidate']
            row['previewOnly'] = True
            frame_path = output/('frame-%04d.json'%row['frame'])
            frame_meta = json.loads(frame_path.read_text(encoding='utf8'))
            frame_meta.update(candidate=measured['candidate'], previewOnly=True)
            frame_path.write_text(json.dumps(frame_meta, ensure_ascii=False, indent=2), encoding='utf8')
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    except Exception as exc:
        diagnostics['failure'] = str(exc)
        persist()
        raise
    finally:
        renderer['main'].__globals__['api'] = original_api
        set_lookup(originals)
        keys[OUTER].value = 0.
        head.update_tag()
        bpy.context.view_layer.update()
    print('R11_OUTER_EYE_NEUTRAL_PREVIEW_DONE '+json.dumps({'output': args.output, 'frames': 2, 'sourceSaved': False}), flush=True)


if __name__ == '__main__':
    main()
