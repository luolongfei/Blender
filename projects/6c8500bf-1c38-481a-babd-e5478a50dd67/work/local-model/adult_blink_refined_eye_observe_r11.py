"""Read-only eight-state observation after an actual refined-eye candidate.

Reuses the frozen mesh/topology observer without editing it. Adds one early
warm-smile state because the old warm L.9/R.5 state disables the new roll.
No source save, render, binding check, field edit or expression-key creation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


BASE = 'adult_blink_midpath_observe_r11.py'
BASE_SHA = '49f56dfee0553da77bd168799dabd47bcd5e718ec7ca678deb0af3a52cb4ffeb'
ROLL = 'R11 refined upper lid roll'
OUTER = 'R11 refined outer eye corners'


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
    if ROLL not in keys:
        raise RuntimeError('Observe an installed actual refined upper-lid candidate')
    controls = {name: float(keys[name].value) for name in (ROLL, OUTER) if name in keys}
    if abs(controls[ROLL]-1.) > 1e-7 or any(not 0. <= value <= 1. for value in controls.values()):
        raise RuntimeError('Observe the selected actual roll at strength one and bounded optional outer-eye strength')
    groups = [m.node_group for m in head.modifiers if m.type == 'NODES' and m.node_group and
              (m.node_group.name.startswith('R11 refined upper lid roll surface') or
               m.node_group.name.startswith('R11 refined outer eye corners'))]
    if not groups:
        raise RuntimeError('Refined-eye modifiers are absent')
    old_states = runtime['STATES']
    original_capture = runtime['capture_mesh']
    runtime['STATES'] = old_states + ({'id': 'warm_L020_R020', 'L': .2, 'R': .2, 'warm': 1.},)
    snapshots = []

    def capture_with_refinement(obj, binding):
        result = original_capture(obj, binding)
        if obj == head:
            actual = {name: float(keys[name].value) for name in controls}
            if actual != controls:
                raise RuntimeError('A new refinement strength changed during expression-only observation')
            evaluated_scalars = {}
            graph = bpy.context.evaluated_depsgraph_get()
            for group in groups:
                evaluated = group.evaluated_get(graph)
                evaluated_scalars[group.name] = {
                    node.name: float(node.outputs[0].default_value)
                    for node in evaluated.nodes if node.bl_idname == 'ShaderNodeValue' and
                    'influence' in node.name.lower()}
            snapshots.append({'actualStrengthKeys': actual,
                              'evaluatedInfluenceNodes': evaluated_scalars})
        return result

    runtime['capture_mesh'] = capture_with_refinement
    try:
        report = function(source_relative, output_relative)
    finally:
        runtime['capture_mesh'] = original_capture
        runtime['STATES'] = old_states
    if len(snapshots) != len(report['states']):
        raise RuntimeError('Refinement scalar observations do not match actual state captures')
    for row, scalars in zip(report['states'], snapshots):
        row['refinedEyeControls'] = scalars
    report['baseObserver'] = {'file': BASE, 'sha256': BASE_SHA}
    report['helperSha256'] = sha(__file__)
    report['refinementObservation'] = {
        'strengthKeys': controls, 'nodeGroups': [g.name for g in groups],
        'earlyWarmStateAdded': True,
        'scope': 'Actual new baseline for rebuilding arc, inner masks, outer-wall ray support and dependencies. Old neutral contour targets are not accepted as current geometry.'}
    report['scope'] = 'Eight actual local states after refined-eye installation; no previous neutral eye outline is assumed compatible.'
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
