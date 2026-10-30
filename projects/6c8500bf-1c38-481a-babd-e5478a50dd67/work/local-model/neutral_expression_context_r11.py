"""Exact R11 expression roles for temporary neutral review, never all-key zero.

Roles come from the real 662054 warm-smile render report and authored helpers.
Static proportions keep their current frame-one values. Only public expression
controls are set to zero; internal drivers keep evaluating. Unknown nonzero
keys and internal residual expression values are explicit errors.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
STATIC = (
    'R11 lower jaw soft contour', 'R11 gentle upper lid coverage',
    'R11 continuous low neck contour', 'R11 adult eye proportion',
    'R11 lower face proportion', 'R11 natural middle neck proportion',
    'R11 soft upper vermilion contour', 'R11 softer nasal profile',
    'R11 rounded lower jaw taper')
CONTROLS = ('R11 gentle smile', 'R11 natural soft smile', 'R11 warm closed smile',
            'R11 blink L', 'R11 blink R', 'R11 jaw open')
INTERNAL = tuple(
    ['R11 blink path ' + side + ' ' + str(i).zfill(2) for side in ('L', 'R') for i in range(1, 16)] +
    ['R11 blink closure ' + side for side in ('L', 'R')] +
    ['R11 blink closure path ' + side + ' ' + str(i).zfill(2) for side in ('L', 'R') for i in range(1, 16)] +
    ['R11 smile lower lid ' + side for side in ('L', 'R')] +
    ['R11 warm smile lower lid ' + side for side in ('L', 'R')])
EVIDENCE = 'previews/reconstruction-r02/warm-smile-front-r11/render-report.json'
EVIDENCE_SHA = '73a285e10af3420a0d1b9f954a0439c9b091b5eb7d5826a4b5031bbfce8c7e7c'
TOLERANCE = 1e-7


def classify(values, basis='Basis'):
    if basis not in values or not all(np.isfinite(float(v)) for v in values.values()):
        raise RuntimeError('Explicit reference key and finite source values required')
    result = {name: {} for name in ('basis', 'static', 'controls', 'internal', 'unknownZero')}
    unknown = {}
    for name, value in values.items():
        value = float(value)
        kind = 'basis' if name == basis else 'static' if name in STATIC else 'controls' if name in CONTROLS else 'internal' if name in INTERNAL else 'unknownZero'
        if kind == 'unknownZero' and abs(value) > TOLERANCE:
            unknown[name] = value
        result[kind][name] = value
    if unknown:
        raise RuntimeError('Unknown active shape keys; do not guess expression/static roles: ' + json.dumps(unknown))
    return result


def assert_neutral(values, preserved, basis='Basis'):
    roles = classify(values, basis)
    residual = {n: v for group in ('controls', 'internal') for n, v in roles[group].items() if abs(v) > TOLERANCE}
    if residual:
        raise RuntimeError('Known expression controls/derived fields remain active: ' + json.dumps(residual))
    changed = {n: [value, values.get(n)] for n, value in preserved.items()
               if n not in values or abs(float(values[n]) - value) > TOLERANCE}
    if changed:
        raise RuntimeError('Preserved static/reference/unknown-zero shape values changed: ' + json.dumps(changed))
    return roles


def _values(keys):
    return {k.name: float(k.value) for k in keys.key_blocks}


def _structure_hash(keys):
    h = hashlib.sha256()
    for key in keys.key_blocks:
        h.update(json.dumps([key.name, key.relative_key.name, key.vertex_group, key.mute,
                             key.slider_min, key.slider_max, key.interpolation]).encode())
        a = np.empty(len(key.data) * 3, np.float32)
        key.data.foreach_get('co', a)
        h.update(a.tobytes())
    return h.hexdigest()


@contextlib.contextmanager
def neutral_session(bpy, report):
    scene = bpy.context.scene
    head = bpy.data.objects[HEAD]
    keys = head.data.shape_keys
    if keys is None or not keys.use_relative:
        raise RuntimeError('Existing relative R11 shape keys required')
    bpy.context.view_layer.update()
    entry_frame = (scene.frame_current, scene.frame_subframe)
    entry_values = _values(keys)
    basis = keys.reference_key.name
    ad = keys.animation_data
    action = ad.action if ad else None
    slot = getattr(ad, 'action_slot', None) if ad else None
    use_nla = ad.use_nla if ad else None
    tracks = [(t, t.mute) for t in ad.nla_tracks] if ad else []
    drivers = [(f, f.mute) for f in ad.drivers] if ad else []
    structures = _structure_hash(keys)
    controller_paths = {keys.key_blocks[n].path_from_id('value'): n for n in CONTROLS if n in keys.key_blocks}
    control_drivers = [(f, mute, controller_paths[f.data_path]) for f, mute in drivers if f.data_path in controller_paths]
    report.update({
        'roleEvidence': EVIDENCE, 'roleEvidenceSha256': EVIDENCE_SHA,
        'entryFrame': list(entry_frame), 'entryKeyValues': entry_values,
        'entryAction': action.name if action else None,
        'entryActionSlot': getattr(slot, 'identifier', None),
        'entryUseNla': use_nla,
        'entryNlaTracks': [{'name': t.name, 'mute': mute} for t, mute in tracks],
        'entryDrivers': [{'path': f.data_path, 'arrayIndex': f.array_index, 'mute': mute} for f, mute in drivers],
        'temporarilyMutedControllerDrivers': [name for _, _, name in control_drivers],
        'internalDriversDeliberatelyLeftUnchanged': True,
        'geometryKeyCoordinatesChanged': False, 'restored': False})
    try:
        scene.frame_set(1)
        bpy.context.view_layer.update()
        frame_one = _values(keys)
        roles = classify(frame_one, basis)
        preserved = {n: v for group in ('basis', 'static', 'unknownZero') for n, v in roles[group].items()}
        report.update(frameOneValues=frame_one, rolesAtFrameOne=roles, preservedValues=preserved)
        # Unbind only this Key datablock; never mute shared Action F-curves.
        if ad:
            ad.action = None
            ad.use_nla = False
        for curve, _, _ in control_drivers:
            curve.mute = True
        for name, value in preserved.items():
            keys.key_blocks[name].value = value
        for name in roles['controls']:
            keys.key_blocks[name].value = 0.
        keys.update_tag()
        bpy.context.view_layer.update()
        neutral = _values(keys)
        assert_neutral(neutral, preserved, basis)
        report['neutralKeyValues'] = neutral
        report['changedPublicControls'] = {n: [frame_one[n], neutral[n]] for n in roles['controls'] if frame_one[n] != neutral[n]}
        yield report
        assert_neutral(_values(keys), preserved, basis)
    finally:
        # All restoration steps are attempted even after a render/guard failure.
        errors = []
        def attempt(label, fn):
            try:
                fn()
            except Exception as exc:
                errors.append(label + ': ' + str(exc))
        if ad:
            attempt('restore action', lambda: setattr(ad, 'action', action))
            if action is not None and hasattr(ad, 'action_slot'):
                attempt('restore action slot', lambda: setattr(ad, 'action_slot', slot))
            attempt('restore NLA evaluation', lambda: setattr(ad, 'use_nla', use_nla))
        for curve, mute in drivers:
            attempt('restore driver mute ' + curve.data_path, lambda f=curve, m=mute: setattr(f, 'mute', m))
        attempt('restore frame', lambda: scene.frame_set(entry_frame[0], subframe=entry_frame[1]))
        for name, value in entry_values.items():
            attempt('restore key ' + name, lambda n=name, v=value: setattr(keys.key_blocks[n], 'value', v))
        attempt('tag restored keys', keys.update_tag)
        attempt('evaluate restored session', bpy.context.view_layer.update)
        after = _values(keys)
        report['restoredKeyValues'] = after
        if set(after) != set(entry_values) or any(abs(after.get(n, float('inf')) - v) > TOLERANCE for n, v in entry_values.items()):
            errors.append('Entry key values did not restore exactly within 1e-7')
        if ad and (ad.action != action or getattr(ad, 'action_slot', None) != slot or ad.use_nla != use_nla):
            errors.append('Action/slot/NLA did not restore')
        if any(t.mute != mute for t, mute in tracks) or any(f.mute != mute for f, mute in drivers):
            errors.append('NLA track or driver mute state changed')
        if (scene.frame_current, scene.frame_subframe) != entry_frame or _structure_hash(keys) != structures:
            errors.append('Original frame or key geometry/structure changed')
        report['restorationErrors'] = errors
        report['restored'] = not errors
        if errors:
            raise RuntimeError('Neutral session restoration failed: ' + '; '.join(errors))
