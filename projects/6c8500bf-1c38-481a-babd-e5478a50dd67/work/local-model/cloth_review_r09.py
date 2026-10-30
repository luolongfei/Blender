"""Fresh-open four actual dynamic-cloth poses, never the old static skirt.

Background integrator invocation:
  --python cloth_review_r09.py --
    --source work/reconstruction-r02/body-cloth-native-r09.blend
No save, bake or source/log rewrite. New preview directory only. Before render,
verify captured action, points, shape action and downstream surface numerically.
"""
import argparse
import json
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import bpy
from mathutils import Vector

M = runpy.run_path(str(Path(__file__).with_name('cloth_merge_r09.py')))
C = M['C']
FRAMES = (1, 55, 151, 247)


@contextmanager
def _review_layers(rig, cloth):
    visibility, modifiers, drivers = [], [], []
    before_switch = rig.get('R04_native_hair_enabled')
    head = bpy.data.objects.get('R01 Head')
    keys = head.data.shape_keys if head else None
    ad = keys.animation_data if keys else None
    action, slot = (ad.action, ad.action_slot) if ad else (None, None)
    nla = [(track, track.mute) for track in ad.nla_tracks] if ad else []
    values = {key.name: float(key.value) for key in keys.key_blocks} if keys else {}
    record = {'visibleCloth': cloth.name, 'hiddenCloth': [], 'disabledHair': [],
              'frozenHeadShapeAction': action.name if action else None}
    try:
        if before_switch is not None:
            rig['R04_native_hair_enabled'] = 0
        if ad:
            ad.action = None
        for track, _ in nla:
            track.mute = True
        for obj in bpy.context.scene.objects:
            dynamic = any(m.type == 'CLOTH' for m in obj.modifiers)
            native = 'Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name
            if dynamic or native or obj.name in C['BASE']['ORIGINALS']:
                visibility.append((obj, obj.hide_viewport, obj.hide_render, obj.hide_get()))
                hidden = obj != cloth
                obj.hide_viewport = obj.hide_render = hidden
                obj.hide_set(hidden)
                if hidden:
                    record['hiddenCloth'].append(obj.name)
            for modifier in obj.modifiers:
                hair = modifier.type == 'NODES' and modifier.name.startswith('R04 Native hair playback')
                if modifier.type == 'CLOTH' or hair:
                    modifiers.append((modifier, modifier.show_viewport, modifier.show_render))
                    for prop in ('show_viewport', 'show_render'):
                        curve = obj.animation_data.drivers.find(modifier.path_from_id(prop)) if obj.animation_data else None
                        if curve:
                            drivers.append((curve, curve.mute))
                            curve.mute = True
                    modifier.show_viewport = modifier.show_render = False
                    if hair:
                        record['disabledHair'].append({'object': obj.name, 'modifier': modifier.name})
        if cloth.hide_viewport or cloth.hide_render or cloth.hide_get():
            raise RuntimeError('The new R09 native cloth must be visible')
        yield record
    finally:
        for modifier, viewport, render in modifiers:
            modifier.show_viewport, modifier.show_render = viewport, render
        for curve, mute in drivers:
            curve.mute = mute
        for obj, viewport, render, hidden in visibility:
            obj.hide_viewport, obj.hide_render = viewport, render
            obj.hide_set(hidden)
        if before_switch is not None:
            rig['R04_native_hair_enabled'] = before_switch
        if ad:
            ad.action = action
            if slot:
                ad.action_slot = slot
        for track, mute in nla:
            track.mute = mute
        if keys:
            for name, value in values.items():
                keys.key_blocks[name].value = value


def render(source, data_dir, out):
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Use a new empty preview directory; existing images and logs are retained')
    M['_check_native_source'](source, data_dir)
    C['_open'](source)
    source_sha = C['_sha'](source)
    rig = C['_rig']()
    cloth = bpy.data.objects[C['NATIVE']]
    scene = bpy.context.scene
    old_frame = (scene.frame_current, scene.frame_subframe)
    settings = [(scene, 'camera'), (scene.render, 'engine'), (scene.render, 'resolution_x'),
        (scene.render, 'resolution_y'), (scene.render, 'resolution_percentage'), (scene.render, 'filepath'),
        (scene.render, 'threads_mode'), (scene.render, 'threads'), (scene.render, 'use_persistent_data'),
        (scene.render.image_settings, 'file_format'), (scene.cycles, 'device'),
        (scene.cycles, 'samples'), (scene.cycles, 'use_denoising')]
    original = [(owner, name, getattr(owner, name)) for owner, name in settings]
    camera, data = None, None
    rows = []
    report = None
    try:
        with _review_layers(rig, cloth) as layers:
            # This only reads the existing capture reports and writes nothing.
            verification = M['verify_loaded'](data_dir, cloth)
            data = bpy.data.cameras.new('R09 temporary dynamic-cloth review camera')
            data.type = 'ORTHO'
            data.ortho_scale = 1.90
            data.dof.use_dof = False
            camera = bpy.data.objects.new('R09 temporary dynamic-cloth review camera', data)
            scene.collection.objects.link(camera)
            camera.location = (3., -4., 1.12)
            target = Vector((0., 0., .83))
            camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
            scene.camera = camera
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'CPU'
            scene.cycles.samples = 12
            scene.cycles.use_denoising = True
            scene.render.threads_mode = 'FIXED'
            scene.render.threads = 6
            scene.render.use_persistent_data = True
            scene.render.resolution_x, scene.render.resolution_y = 480, 640
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = 'PNG'
            out.mkdir(parents=True, exist_ok=True)
            for frame in FRAMES:
                C['_set_frame'](frame)
                if cloth.hide_render or any(m.type in {'ARMATURE', 'CLOTH'} for m in cloth.modifiers):
                    raise RuntimeError('Unexpected dynamic-cloth visibility or duplicate deformation')
                path = out / f'frame-{frame:04d}.png'
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                rows.append({'frame': frame, 'path': str(path)})
                print('R09_DYNAMIC_CLOTH_RENDER ' + str(frame), flush=True)
            report = {'sourceBlend': C['_relative'](source), 'sourceSha256': source_sha,
                'action': C['ACTION'], 'verification': verification, 'layers': layers,
                'frames': rows, 'resolution': [480, 640], 'samples': 12,
                'cameraPosition': list(camera.location), 'cameraTarget': list(target), 'orthoScale': 1.90,
                'lightsAndMaterialsUnchanged': True, 'savedBlend': False,
                'scope': 'Actual new R09 simulated cloth at four poses. Retained groom shown without old dynamic hair; head morph action frozen. No continuous-motion or full collision acceptance claimed.'}
    finally:
        for owner, name, value in original:
            setattr(owner, name, value)
        if camera is not None:
            bpy.data.objects.remove(camera, do_unlink=True)
        if data is not None and not data.users:
            bpy.data.cameras.remove(data)
        scene.frame_set(old_frame[0], subframe=old_frame[1])
        bpy.context.view_layer.update()
    if C['_sha'](source) != source_sha:
        raise RuntimeError('Source file changed externally during review')
    if report:
        report['sourceAndSceneSettingsPreserved'] = True
        C['_write'](out / 'render-report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='work/reconstruction-r02/body-cloth-native-r09.blend')
    parser.add_argument('--data-dir', default='work/reconstruction-r02/cloth-body-r09')
    parser.add_argument('--preview-dir', default='previews/reconstruction-r02/cloth-poses-r09')
    args = parser.parse_args(argv)
    if not bpy.app.background:
        raise RuntimeError('Explicit serial background Blender required')
    out = (C['project_path']() / args.preview_dir).resolve()
    out.relative_to((C['project_path']() / 'previews').resolve())
    return render(C['_path'](args.source), C['_path'](args.data_dir), out)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
