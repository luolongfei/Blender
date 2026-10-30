"""Reversible dark-brown brow-gel material on the two actual bound R11 brows.

Only two Curves material slots and their explicitly allowed binding material
contracts change. Original fibre positions, density, radius, growth direction,
root maps, head skin, eyelids, lashes and lights stay exact. No skin stripe.
Use a fresh independent source; compare at identical lighting/camera/samples.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

MARKER = 'R11_brow_finish_candidate'
NAMES = ('R11 Adult natural brow L', 'R11 Adult natural brow R')
PARAMETERS = dict(darkLinear=(.0035, .0025, .0019, 1.), lightLinear=(.008, .0058, .0042, 1.),
    roughness=.64, ior=1.45, specularIORLevel=.25, transmission=0., coat=0.)


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _external():
    """All scene object transforms, identities and non-brow slots remain exact."""
    import bpy
    records = []
    for obj in bpy.context.scene.objects:
        row = dict(name=obj.name, type=obj.type, pointer=obj.as_pointer(),
            data=getattr(getattr(obj, 'data', None), 'as_pointer', lambda: None)(),
            world=[list(v) for v in obj.matrix_world],
            parent=obj.parent.name if obj.parent else None, parentType=obj.parent_type, parentBone=obj.parent_bone,
            hideRender=obj.hide_render, hideViewport=obj.hide_viewport,
            modifiers=[(m.name, m.type) for m in obj.modifiers])
        if obj.name not in NAMES: row['materials'] = [s.material.as_pointer() if s.material else None for s in obj.material_slots]
        records.append(row)
    return _api('face_fibre_binding_r11')['_hash'](records)


def _material():
    import bpy
    material = bpy.data.materials.new('R11 natural black brown brow gel')
    material.use_nodes = True
    nodes = material.node_tree.nodes; links = material.node_tree.links; nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial'); output.location = (540, 80)
    shader = nodes.new('ShaderNodeBsdfPrincipled'); shader.location = (240, 80)
    for name, value in [('Roughness', PARAMETERS['roughness']), ('IOR', PARAMETERS['ior']),
        ('Specular IOR Level', PARAMETERS['specularIORLevel']), ('Transmission Weight', 0.),
        ('Coat Weight', 0.), ('Metallic', 0.), ('Alpha', 1.)]:
        shader.inputs[name].default_value = value
    info = nodes.new('ShaderNodeHairInfo'); info.location = (-520, 80)
    ramp = nodes.new('ShaderNodeValToRGB'); ramp.location = (-220, 80)
    ramp.color_ramp.interpolation = 'EASE'
    ramp.color_ramp.elements[0].position = 0.; ramp.color_ramp.elements[0].color = PARAMETERS['darkLinear']
    ramp.color_ramp.elements[1].position = 1.; ramp.color_ramp.elements[1].color = PARAMETERS['lightLinear']
    links.new(info.outputs['Random'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], shader.inputs['Base Color'])
    links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    material.diffuse_color = tuple((a + b) * .5 for a, b in zip(PARAMETERS['darkLinear'], PARAMETERS['lightLinear']))
    material['R11_scope'] = 'Authored restrained opaque black-brown brow cosmetic, with subtle per-strand color variation; no measured cosmetic dataset.'
    return material


def apply():
    import bpy
    source = Path(bpy.data.filepath).resolve()
    source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or MARKER in bpy.context.scene:
        raise RuntimeError('Fresh independent background brow finish required; no stacking')
    api = _api('face_fibre_binding_r11'); scene = bpy.context.scene
    before = api['validate'](neutral=False)
    record = json.loads(scene[api['MARKER']]); unchanged_record = copy.deepcopy(record)
    head = bpy.data.objects[api['HEAD']]
    protected = api['_protected'](head); external = _external()
    pose = _api('production_uv_r08')['_pose_contract']()
    objects = [bpy.data.objects[n] for n in NAMES]
    if any(o.type != 'CURVES' or len(o.data.materials) != 1 or o.hide_render or o.data.materials[0] is None for o in objects):
        raise RuntimeError('Two visible, bound, single-material adult brows required')
    for obj in objects:
        if record['objects'][obj.name].get('currentMaterialContractSha256'):
            raise RuntimeError('Inspect/restore existing brow material study before another candidate')
    material_api = _api('runtime_face_refresh_r08')['_material_contract']
    originals = {obj.data.materials[0].name: dict(material=obj.data.materials[0], fakeUser=obj.data.materials[0].use_fake_user,
        contract=material_api(obj.data.materials[0])) for obj in objects}
    original_slots = {obj.name: obj.data.materials[0] for obj in objects}
    before_contracts = {obj.name: api['_curve_contract'](obj.data) for obj in objects}
    material = None; changes = []
    try:
        material = _material()
        for row in originals.values(): row['material'].use_fake_user = True
        for obj in objects:
            prior = before_contracts[obj.name]
            obj.data.materials[0] = material
            current = api['_curve_contract'](obj.data)
            geometry = dict(current); geometry['materials'] = prior['materials']
            if geometry != prior: raise RuntimeError('Brow fibre positions, radius, topology or attributes changed')
            change = dict(object=obj.name, oldMaterial=original_slots[obj.name].name, newMaterial=material.name,
                reason='Independent restrained opaque black-brown brow-gel material to improve existing fibre readability',
                helperSha256=_sha(__file__))
            item = record['objects'][obj.name]
            item['currentMaterialContractSha256'] = api['_hash'](current); item['materialChange'] = change
            changes.append(change)
        scene[api['MARKER']] = json.dumps(record, ensure_ascii=False)
        bpy.context.view_layer.update()
        after = api['validate'](neutral=False)
        if api['_protected'](head) != protected or _external() != external or _api('production_uv_r08')['_pose_contract']() != pose:
            raise RuntimeError('Brow material operation altered head/skin, scene, other slots or rig pose')
        for row in originals.values():
            if material_api(row['material']) != row['contract']:
                raise RuntimeError('Original brow material nodes changed')
        for name, item in unchanged_record['objects'].items():
            if name not in NAMES and record['objects'][name] != item:
                raise RuntimeError('Other fibre binding/material entry changed')
        report = dict(source=str(source.relative_to(project_path())), sourceSha256=_sha(source), helperSha256=_sha(__file__),
            changes=changes, parameters=PARAMETERS, bindingBefore=before, bindingAfter=after,
            originalBindingItems={name: unchanged_record['objects'][name] for name in NAMES},
            originalMaterialFakeUser={name: row['fakeUser'] for name, row in originals.items()},
            existingFibreCounts={obj.name: len(obj.data.curves) for obj in objects},
            material=material.name, originalFibreGeometryRadiusDensityAndBindingExact=True,
            headEyesLashesAndLightingUnchanged=True, appearanceAccepted=False, sourceSaved=False,
            scope='Only the material of the two existing natural brow Curves; no painted skin strip or geometric thickening.',
            limitations=['Opaque cosmetic approximation, not measured uncoated keratin optics.',
                'Existing strand counts remain exact; this is a material-only test, not proof of sufficient density.',
                'Compare actual same-light front and eyebrow macro; neutral brightness does not validate brow expressions.'])
        scene[MARKER] = json.dumps(report, ensure_ascii=False)
        return report
    except Exception:
        for obj in objects: obj.data.materials[0] = original_slots[obj.name]
        scene[api['MARKER']] = json.dumps(unchanged_record, ensure_ascii=False)
        for row in originals.values(): row['material'].use_fake_user = row['fakeUser']
        if material is not None and material.users == 0: bpy.data.materials.remove(material)
        bpy.context.view_layer.update()
        raise


def restore():
    """Restore only these brow slots/entries; retain later edits to other fibres."""
    import bpy
    api = _api('face_fibre_binding_r11'); scene = bpy.context.scene
    report = json.loads(scene[MARKER]); record = json.loads(scene[api['MARKER']])
    api['validate'](neutral=False)
    for change in report['changes']:
        obj = bpy.data.objects[change['object']]
        if len(obj.data.materials) != 1 or obj.data.materials[0].name != change['newMaterial']:
            raise RuntimeError('Brow finish no longer current; inspect later work before restore')
        if change['oldMaterial'] not in bpy.data.materials: raise RuntimeError('Retained original brow material missing')
    for change in report['changes']:
        bpy.data.objects[change['object']].data.materials[0] = bpy.data.materials[change['oldMaterial']]
        record['objects'][change['object']] = report['originalBindingItems'][change['object']]
    for name, fake_user in report['originalMaterialFakeUser'].items(): bpy.data.materials[name].use_fake_user = fake_user
    scene[api['MARKER']] = json.dumps(record, ensure_ascii=False)
    del scene[MARKER]
    bpy.context.view_layer.update()
    return api['validate'](neutral=False)


def prepare(source_relative, output_relative):
    import bpy
    root = (project_path() / 'work/reconstruction-r02').resolve()
    source = (project_path() / source_relative).resolve(); output = (project_path() / output_relative).resolve()
    for path in (source, output): path.relative_to(root)
    report_path = output.with_suffix('.brow-finish.json')
    if source.suffix != '.blend' or output.suffix != '.blend' or output.exists() or report_path.exists():
        raise RuntimeError('Preserve existing sources; fresh .blend and report output required')
    if bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Open the exact saved independent source')
    source_sha = _sha(source)
    report = apply()
    if _sha(source) != source_sha: raise RuntimeError('Original source changed')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True):
        raise RuntimeError('Independent brow candidate save failed')
    report.update(output=str(output.relative_to(project_path())), outputSha256=_sha(output), outputSaved=True)
    with report_path.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_BROW_FINISH_PREPARED ' + json.dumps(dict(output=str(output), sha256=report['outputSha256'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.output)
