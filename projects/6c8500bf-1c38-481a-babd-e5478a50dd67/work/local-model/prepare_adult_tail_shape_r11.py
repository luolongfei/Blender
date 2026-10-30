"""Save the independent shorter low-tail study after the donor restyle.

The original source and its native playback data remain preserved. This is a
static candidate: the old cache is deliberately retired, never declared valid
for the new rest geometry. Run through the configured Windows Blender only.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

P = project_path()
EXPECTED_HELPER = 'd04d94c528cc5a2a9792501f29be55acd81989fd68aacff12ea5302b4b872255'
TAILS = ('R02 Strands / R02 Twin tail L', 'R02 Strands / R02 Twin tail R')
DONORS = ('R11 Donor / long hair main', 'R11 Donor / long hair strands')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_contract(obj):
    """Raw protected geometry and key values, independent of pose evaluation."""
    data = obj.data
    h = hashlib.sha256()
    if obj.type == 'MESH':
        a = np.empty(len(data.vertices) * 3, np.float32)
        data.vertices.foreach_get('co', a)
    elif obj.type == 'CURVES':
        a = np.empty(len(data.position_data) * 3, np.float32)
        data.position_data.foreach_get('vector', a)
    else:
        raise RuntimeError('Unexpected protected object type: ' + obj.name)
    h.update(a.tobytes())
    keys = getattr(data, 'shape_keys', None)
    if keys:
        for key in keys.key_blocks:
            a = np.empty(len(key.data) * 3, np.float32)
            key.data.foreach_get('co', a)
            h.update(key.name.encode()); h.update(a.tobytes())
            h.update(repr(float(key.value)).encode())
    return {'geometryAndKeysSha256': h.hexdigest(), 'data': data.name,
            'materials': [m.name if m else None for m in data.materials],
            'parent': obj.parent.name if obj.parent else None,
            'parentType': obj.parent_type, 'parentBone': obj.parent_bone,
            'matrixLocal': [list(row) for row in obj.matrix_local]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source = (P / args.source).resolve(); output = (P / args.output).resolve()
    for path in (source, output):
        path.relative_to((P / 'work/reconstruction-r02').resolve())
    report_path = output.with_suffix('.tail-shape.json')
    if (not bpy.app.background or bpy.data.is_dirty
            or Path(bpy.data.filepath).resolve() != source
            or source == output or output.exists() or report_path.exists()):
        raise RuntimeError('Fresh independent source and unused output required')
    module = Path(__file__).with_name('tail_shape_adult_r11.py')
    if sha(module) != EXPECTED_HELPER:
        raise RuntimeError('Review changed tail shape helper before execution')
    for name in DONORS:
        obj = bpy.data.objects.get(name)
        if obj is None or not obj.get('R11_donor_restyle'):
            raise RuntimeError('Apply and review the donor-to-tail restyle first: ' + name)
    protected = [obj for obj in bpy.data.objects
                 if obj.type in {'MESH', 'CURVES'} and obj.name not in TAILS
                 and (obj.name in {'R11 Female head study', 'R01 Head'}
                      or obj.name.startswith('R11 Donor / ')
                      or 'Tie' in obj.name
                      or any(word in obj.name.lower() for word in ('blouse', 'skirt', 'collar')))]
    before = {obj.name: geometry_contract(obj) for obj in protected}
    rig = bpy.data.objects['Character_Rig']; pose = rig.data.pose_position
    scene = bpy.context.scene
    frame = scene.frame_current; subframe = scene.frame_subframe
    action = rig.animation_data.action if rig.animation_data else None
    source_sha = sha(source)
    try:
        rig.data.pose_position = 'REST'; bpy.context.view_layer.update()
        report = runpy.run_path(str(module))['apply_adult_tail_shape'](
            expected_source_relative=args.source)
    finally:
        rig.data.pose_position = pose; bpy.context.view_layer.update()
    after = {obj.name: geometry_contract(obj) for obj in protected}
    if before != after:
        raise RuntimeError('Protected geometry/keys/parent/material contract changed')
    current_action = rig.animation_data.action if rig.animation_data else None
    if current_action != action or (scene.frame_current, scene.frame_subframe) != (frame, subframe):
        raise RuntimeError('Body action or review frame changed')
    if any(bpy.data.objects[name].modifiers.get('R04 Native hair playback') for name in TAILS):
        raise RuntimeError('Obsolete tail playback is still installed')
    if sha(source) != source_sha:
        raise RuntimeError('Saved input was modified')
    report['protectedGeometryAndKeys'] = before
    report['bodyActionAndFrameUnchanged'] = True
    report['oldNativeTailPlaybackRemoved'] = True
    report['faceFibreBindingAfterTailShape'] = runpy.run_path(str(
        Path(__file__).with_name('face_fibre_binding_r11.py')))['validate'](neutral=False)
    scene['R11_adult_tail_shape'] = json.dumps(report, ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
        raise RuntimeError('Save failed')
    report.update(savedBlend=True, output=args.output, outputSha256=sha(output),
                  wrapperSha256=sha(__file__))
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print('R11_ADULT_TAIL_SHAPE_SAVED ' + json.dumps({
        'output': args.output, 'sha256': report['outputSha256']}), flush=True)


if __name__ == '__main__':
    main()
