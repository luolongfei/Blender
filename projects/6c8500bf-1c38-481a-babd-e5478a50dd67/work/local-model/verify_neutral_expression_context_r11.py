"""Offline role/source and restoration-logic checks; not a Blender execution test."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import types

HERE = Path(__file__).resolve().parent
API = runpy.run_path(str(HERE / 'neutral_expression_context_r11.py'))
P = API['project_path']()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data(list):
    def foreach_get(self, attr, output):
        output[:] = [0., 0., 0.]


class Block:
    def __init__(self, name, value):
        self.name, self.value = name, value
        self.relative_key = self
        self.vertex_group, self.mute = '', False
        self.slider_min, self.slider_max, self.interpolation = 0., 1., 'KEY_LINEAR'
        self.data = Data([None])

    def path_from_id(self, field):
        return 'key_blocks[' + json.dumps(self.name) + '].' + field


class Blocks(dict):
    def __iter__(self):
        return iter(self.values())


class Fake:
    def __init__(self, source, *, unknown=False, frozen_internal=False, no_slot=False):
        values = dict(source)
        # Non-unit sculpt value proves neutrality preserves values, not merely names.
        values[API['STATIC'][0]] = .37
        if unknown:
            values['Unknown new active sculpt'] = .2
        self.blocks = Blocks({n: Block(n, v) for n, v in values.items()})
        self.action = types.SimpleNamespace(name='shared source Action')
        self.slot = None if no_slot else types.SimpleNamespace(identifier='source Key slot')
        self.control = types.SimpleNamespace(data_path=self.blocks['R11 blink L'].path_from_id('value'), array_index=0, mute=False)
        self.internal = types.SimpleNamespace(data_path=self.blocks['R11 blink closure L'].path_from_id('value'), array_index=0, mute=frozen_internal)
        self.ad = types.SimpleNamespace(action=self.action, action_slot=self.slot, use_nla=True,
             nla_tracks=[types.SimpleNamespace(name='source NLA', mute=False)], drivers=[self.control, self.internal])
        self.keys = types.SimpleNamespace(use_relative=True, key_blocks=self.blocks,
             reference_key=self.blocks['Basis'], animation_data=self.ad, update_tag=lambda: None)
        self.scene = types.SimpleNamespace(frame_current=8, frame_subframe=.25)
        self.scene.frame_set = self.frame_set
        self.bpy = types.SimpleNamespace(context=types.SimpleNamespace(scene=self.scene,
             view_layer=types.SimpleNamespace(update=self.update)),
             data=types.SimpleNamespace(objects={API['HEAD']: types.SimpleNamespace(data=types.SimpleNamespace(shape_keys=self.keys))}))
        if frozen_internal:
            self.blocks['R11 blink closure L'].value = .25
        self.update()

    def update(self):
        if self.ad.action is not None:
            self.blocks['R11 warm closed smile'].value = .4
        if self.ad.use_nla:
            self.blocks['R11 blink R'].value = .6
        if not self.control.mute:
            self.blocks['R11 blink L'].value = .8
        if not self.internal.mute:
            self.blocks['R11 blink closure L'].value = self.blocks['R11 blink L'].value
        self.blocks['R11 warm smile lower lid L'].value = self.blocks['R11 warm closed smile'].value * (1-self.blocks['R11 blink L'].value)

    def frame_set(self, frame, subframe=0.):
        self.scene.frame_current, self.scene.frame_subframe = frame, subframe
        self.update()


def check_restore(values, *, body_error=False, unknown=False, frozen_internal=False, no_slot=False):
    fake = Fake(values, unknown=unknown, frozen_internal=frozen_internal, no_slot=no_slot)
    entry = API['_values'](fake.keys)
    report = {}
    error = None
    reached = False
    try:
        with API['neutral_session'](fake.bpy, report):
            reached = True
            assert fake.blocks[API['STATIC'][0]].value == .37
            assert all(fake.blocks[n].value == 0 for n in API['CONTROLS'] if n in fake.blocks)
            assert fake.ad.action is None and not fake.ad.use_nla and fake.control.mute
            assert not fake.internal.mute
            if body_error:
                raise RuntimeError('injected render failure')
    except RuntimeError as exc:
        error = str(exc)
    assert report['restored'] and report['restorationErrors'] == []
    assert API['_values'](fake.keys) == entry
    assert fake.ad.action is fake.action and fake.ad.action_slot is fake.slot and fake.ad.use_nla
    assert not fake.control.mute and fake.internal.mute == frozen_internal
    assert (fake.scene.frame_current, fake.scene.frame_subframe) == (8, .25)
    if unknown:
        assert not reached and 'Unknown active' in error
    elif frozen_internal:
        assert not reached and 'derived fields remain active' in error
    elif body_error:
        assert reached and error == 'injected render failure'
    else:
        assert reached and error is None
    return {'passed': True, 'expectedFailure': error, 'entryRestored': report['restored']}


def main():
    context = HERE / 'neutral_expression_context_r11.py'
    review = HERE / 'adult_neutral_lighting_review_r11_v2.py'
    for path in (context, review, Path(__file__)):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    evidence = P / API['EVIDENCE']
    assert sha(evidence) == API['EVIDENCE_SHA']
    values = json.loads(evidence.read_text(encoding='utf-8'))['savedKeyValues']
    roles = API['classify'](values)
    assert {k: len(v) for k,v in roles.items()} == {'basis': 1, 'static': 9, 'controls': 5, 'internal': 66, 'unknownZero': 0}
    preserved = {n:v for category in ('basis', 'static', 'unknownZero') for n,v in roles[category].items()}
    API['assert_neutral'](values, preserved)
    tests = {
        'actual662SavedNeutralWithNineNonzeroSculpts': {'passed': True, 'roleCounts': {k:len(v) for k,v in roles.items()}},
        'actionNlaAndIncomingControlDriverSuccess': check_restore(values),
        'originalEmptyActionSlotRestores': check_restore(values, no_slot=True),
        'injectedRenderFailureRestoresEntry': check_restore(values, body_error=True),
        'unknownActiveKeyRejectsAndRestores': check_restore(values, unknown=True),
        'frozenInternalExpressionRejectsAndRestores': check_restore(values, frozen_internal=True)}
    report = {'scope': 'Offline Python syntax, actual saved role evidence, and mocked animation/restoration logic. Not Blender runtime or visual validation.',
        'helperSha256': sha(review), 'contextSha256': sha(context), 'verifierSha256': sha(__file__),
        'evidence': API['EVIDENCE'], 'evidenceSha256': sha(evidence), 'checks': tests, 'passed': True}
    output = P / 'work/reconstruction-r02/neutral-expression-offline-r11-v2.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(output), 'sha256': sha(output), **report}, ensure_ascii=False))


if __name__ == '__main__':
    main()
