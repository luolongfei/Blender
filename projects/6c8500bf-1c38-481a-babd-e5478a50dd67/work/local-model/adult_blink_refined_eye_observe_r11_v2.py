"""Nine actual states after selecting the new neutral eye shape.

Read only: a single fresh capture supplies current topology, real globe meshes,
per-state head geometry and two warm-smile lash samples. No old face observation
is a geometry dependency. Public controls and pose are restored; no source save.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

BASE = 'adult_blink_midpath_observe_r11.py'
BASE_SHA = '49f56dfee0553da77bd168799dabd47bcd5e718ec7ca678deb0af3a52cb4ffeb'
ROLL = 'R11 refined upper lid roll'
OUTER = 'R11 refined outer eye corners'
MARKER = 'R11_refined_radial_outer_eye_candidate'
EXTRA_STATES = (
    {'id': 'blink_035', 'L': .35, 'R': .35, 'warm': 0.},
    {'id': 'warm_L020_R020', 'L': .2, 'R': .2, 'warm': 1.},
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe(source_relative, output_relative):
    import bpy

    base_file = Path(__file__).with_name(BASE)
    if sha(base_file) != BASE_SHA:
        raise RuntimeError('Frozen base observer changed')
    module = runpy.run_path(str(base_file))
    function = module['observe']
    runtime = function.__globals__
    head = bpy.data.objects[module['HEAD']]
    keys = head.data.shape_keys.key_blocks
    if any(m.type == 'ARMATURE' for m in head.modifiers):
        raise RuntimeError('This fresh shaping capture requires the unbound REST source')
    if any(name not in keys for name in (ROLL, OUTER)) or MARKER not in bpy.context.scene:
        raise RuntimeError('Selected installed roll and radial outer-eye source required')
    controls = {name: float(keys[name].value) for name in (ROLL, OUTER)}
    if any(abs(value-1.) > 1e-7 for value in controls.values()):
        raise RuntimeError('Capture the selected new neutral eye shape with both strengths one')
    marker = json.loads(bpy.context.scene[MARKER])
    if marker.get('controlKey') != OUTER or not marker.get('surfaceRefinementChecksPassed'):
        raise RuntimeError('Installed actual radial candidate record is incomplete')
    groups = [m.node_group for m in head.modifiers if m.type == 'NODES' and m.node_group and
              (m.node_group.name.startswith('R11 refined upper lid roll surface') or
               m.node_group.name.startswith('R11 refined outer eye corners radial'))]
    if len(groups) != 2 or marker.get('nodeGroup') not in [g.name for g in groups]:
        raise RuntimeError('Exactly one installed roll and radial group required')

    binding = module['api']('face_fibre_binding_r11')
    support = {}
    expected_attributes = {'R11_refined_upper_roll_weights_L_R': (0, 1),
                           'R11_outer_eye_radial_channels_L': (0,),
                           'R11_outer_eye_radial_channels_R': (0,)}
    for group in groups:
        for node in group.nodes:
            if node.bl_idname != 'GeometryNodeObjectInfo':
                continue
            obj = node.inputs['Object'].default_value
            if obj is None or obj.type != 'MESH':
                continue
            for name, channels in expected_attributes.items():
                if name not in obj.data.attributes:
                    continue
                if name in support:
                    raise RuntimeError('Ambiguous installed refinement lookup '+name)
                values = binding['_array'](obj.data.attributes[name].data, 'vector')
                ids = np.flatnonzero(np.any(values[:, channels] > 0., axis=1)).astype(np.int32)
                support[name] = {'object': obj.name, 'nodeGroup': group.name,
                                 'pointCount': len(values), 'attributeSha256': module['array_sha'](values),
                                 'activeGlobalVertexIds': ids.tolist()}
    if set(support) != set(expected_attributes):
        raise RuntimeError('Actual refined-eye lookup fields are missing')
    pending = np.asarray(marker.get('requiredInnerContactGlobalVertexIds', ()), np.int32)
    extra_ids = np.unique(np.concatenate([pending, *[
        np.asarray(row['activeGlobalVertexIds'], np.int32) for row in support.values()]]))
    old_states = runtime['STATES']
    original_capture = runtime['capture_mesh']
    original_roi = runtime['fixed_roi']
    # Preserve the base observer's proven endpoint indices (neutral=0, full=5).
    # Consumers must find states by id/value, never old positional slices.
    runtime['STATES'] = old_states + EXTRA_STATES
    snapshots = []

    def roi_with_current_support(neutral, eyes, old_extra_ids=()):
        if any(row['pointCount'] != len(neutral['world']) for row in support.values()):
            raise RuntimeError('Installed refinement lookup size differs from actual evaluated head')
        combined = np.unique(np.concatenate([np.asarray(old_extra_ids, np.int32), extra_ids]))
        return original_roi(neutral, eyes, combined)

    def capture_with_refinement(obj, capture_binding):
        result = original_capture(obj, capture_binding)
        if obj == head:
            actual = {name: float(keys[name].value) for name in controls}
            if actual != controls:
                raise RuntimeError('A static refinement strength changed during expression capture')
            scalars = {}
            graph = bpy.context.evaluated_depsgraph_get()
            for group in groups:
                evaluated = group.evaluated_get(graph)
                scalars[group.name] = {node.name: float(node.outputs[0].default_value)
                    for node in evaluated.nodes if node.bl_idname == 'ShaderNodeValue' and
                    'influence' in node.name.lower()}
            snapshots.append({'actualStrengthKeys': actual, 'evaluatedInfluenceNodes': scalars})
        return result

    runtime['fixed_roi'] = roi_with_current_support
    runtime['capture_mesh'] = capture_with_refinement
    try:
        report = function(source_relative, output_relative)
    finally:
        runtime['capture_mesh'] = original_capture
        runtime['fixed_roi'] = original_roi
        runtime['STATES'] = old_states
    if len(snapshots) != 9 or len(report['states']) != 9:
        raise RuntimeError('Nine fresh state captures required')
    for row, scalars in zip(report['states'], snapshots):
        row['refinedEyeControls'] = scalars
    states_by_id = {row['id']: row for row in report['states']}
    if len(states_by_id) != 9:
        raise RuntimeError('Actual state identifiers must be unique')
    for label, state_id in (('neutral', 'blink_000'), ('fullClosed', 'blink_100')):
        endpoint = report['endpoints'][label]
        if endpoint['stateIndex'] != states_by_id[state_id]['index']:
            raise RuntimeError('Actual endpoint index mapping is inconsistent')
    report['baseObserver'] = {'file': BASE, 'sha256': BASE_SHA}
    report['helperSha256'] = sha(__file__)
    report['roiSelection'] += ' Union includes current roll/radial positive support and declared pending inner points.'
    report['refinementObservation'] = {
        'strengthKeys': controls, 'nodeGroups': [group.name for group in groups],
        'installedLookupSupport': support,
        'radialPlan': marker['plan'],
        'requiredInnerContactGlobalVertexIds': pending.tolist(),
        'requiredInnerContactPolygonIds': marker.get('requiredInnerContactPolygonIds', []),
        'planningBlinkKnotOrder': [0., .2, .35, .5, .8, .9, 1.],
        'earlyWarmStateAdded': True,
        'stateLookupRule': 'Find by exact id or L/R/warm values; stored order preserves historical endpoint indices.',
        'scope': 'Fresh baseline for monotone early outer aperture, inner actual-globe clearance and outer-wall ray-hit dependencies.'}
    report['scope'] = 'Nine actual local states of the selected new eye shape; no old f501 geometry is accepted as current.'
    report['limits'].extend([
        'The capture records geometry only; old half-blink contour is not a preservation target.',
        'Front-axis or continued-radius gaps do not certify true closest-surface clearance.',
        'New arc and inner contact layers need actual changed-face and intermediate-state checks before motion acceptance.'])
    destination = module['project_file'](output_relative, '.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('R11_REFINED_EYE_OBSERVED '+json.dumps({'output': output_relative,
          'arraysSha256': report['arraysSha256'], 'states': len(report['states'])}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.output)
