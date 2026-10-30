"""V2 bounded synthetic lip microrelief on the retained refined natural skin.

apply()/restore() only change a copied skin material assignment. No UV,
attributes, geometry, keys, lights, poses or source file are edited. The nominal
micrometre setting bounds the NEW shader height field, not mesh displacement
or the total microstructure already present in the original normal chain.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
REST = 'R04 skin rest position'
MASK = 'R11 true vermilion rose support'
COLOR = 'R11 refined natural skin colour'
MARKER = 'R11_lip_microrelief_candidate'
PREFIX = 'R11 lip microrelief / '
IMAGE = 'assets/generated/lip-striation-height-r11-v2.png'
IMAGE_SHA = '7e7ef84d437de154784f3fdafeffb594fb751d3a2993db0c504bc9b7a5b4b9af'
MARGIN = .06


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _array(items, field, width=3, dtype=np.float32):
    result = np.empty((len(items), width), dtype=dtype)
    items.foreach_get(field, result.ravel())
    return result


def _ah(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode() + a.dtype.str.encode() + a.tobytes()).hexdigest()


def _path(relative):
    p = (project_path() / relative).resolve()
    p.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if p.suffix != '.blend':
        raise ValueError('Explicit independent reconstruction .blend required')
    return p


def field_plan(rest, mask, strip, outer, green, microns):
    """Pure-array preflight; image samples are linear data, never sRGB colour."""
    rest = np.asarray(rest, dtype=float)
    mask = np.asarray(mask, dtype=float).reshape(-1)
    strip = np.asarray(strip, dtype=np.int32)
    outer = np.asarray(outer, dtype=np.int32)
    green = np.asarray(green, dtype=float).reshape(-1)
    if rest.shape != (10582, 3) or mask.shape != (10582,):
        raise RuntimeError('Complete actual female rest/mask arrays required')
    if not np.isfinite(rest).all() or not np.isfinite(mask).all() or not np.isfinite(green).all():
        raise RuntimeError('Nonfinite rest, mask or image values')
    if not 0 <= float(microns) <= 15:
        raise ValueError('Bounded study requires 0..15 nominal micrometres')
    if len(strip) != 120 or len(set(strip)) != 120 or len(outer) != 40 or not set(outer).issubset(set(strip)):
        raise RuntimeError('Actual 120-point lip-strip identities required')
    allowed = np.zeros(len(mask), dtype=bool)
    allowed[strip] = True
    if np.any(mask[~allowed] != 0) or np.any(mask[outer] != 0) or mask.min() < 0 or mask.max() > 1:
        raise RuntimeError('Existing lip mask escaped true vermilion or outer fade')
    if not 50 <= int(np.count_nonzero(mask)) <= 80:
        raise RuntimeError('Expected sparse existing vermilion mask')
    lo = rest[strip][:, [0, 2]].min(0)
    hi = rest[strip][:, [0, 2]].max(0)
    span = hi - lo
    if not .020 < span[0] < .080 or not .002 < span[1] < .030:
        raise RuntimeError('Unexpected physical lip rest-space bounds')
    gmin, gmax, baseline = float(green.min()), float(green.max()), float(np.median(green))
    if not 0 <= gmin < baseline <= gmax <= 1 or not .02 < gmax - gmin < .35:
        raise RuntimeError('Unexpected non-colour image range')
    normalized = ((gmin - baseline) / (gmax - gmin), (gmax - baseline) / (gmax - gmin))
    return {
        'micronsNominalNewHeightField': float(microns),
        'distanceMeters': float(microns) * 1e-6,
        'restXZMinMeters': lo.tolist(), 'restXZMaxMeters': hi.tolist(),
        'restXZSpanMeters': span.tolist(), 'imageCoordinateRange': [MARGIN, 1 - MARGIN],
        'greenRange': [gmin, gmax], 'greenMedianBaseline': baseline,
        'normalizedHeightBoundsBeforeMask': list(normalized),
        'normalizedPeakToValleyUpperBound': 1.0,
        'maskRule': 'square(clamp(existing vermilion support,0,1)); never strengthens source mask',
        'maskNonzeroRows': int(np.count_nonzero(mask)),
        'restAttributeSha256': _ah(np.asarray(rest, np.float32)),
        'maskAttributeSha256': _ah(np.asarray(mask, np.float32)),
        'interpretation': 'Nominal NEW bump height-field bound only. No geometry displacement. Original procedural normal remains and total detail is not bounded by this number.'
    }


def _normal_link(shader):
    links = list(shader.inputs['Normal'].links)
    if len(links) > 1:
        raise RuntimeError('Ambiguous current skin normal input')
    if not links:
        return None
    link = links[0]
    return [link.from_node.name, list(link.from_node.outputs).index(link.from_socket),
            shader.name, list(shader.inputs).index(link.to_socket)]


def _normalized_tree(material, original_link, added_names, face):
    tree = copy.deepcopy(face['_tree_contract'](material.node_tree))
    owned = set(added_names)
    tree['nodes'] = [n for n in tree['nodes'] if n['name'] not in owned]
    tree['links'] = [tuple(l) for l in tree['links'] if l[0] not in owned and l[2] not in owned]
    if original_link is not None:
        tree['links'].append(tuple(original_link))
    tree['links'] = sorted(tree['links'])
    return _hash(tree)


def _wire(candidate, shader_name, image, plan):
    nt = candidate.node_tree
    shader = nt.nodes[shader_name]
    old_link = _normal_link(shader)
    old_normal = shader.inputs['Normal'].links[0].from_socket if old_link else None
    added = []

    def node(kind, name):
        n = nt.nodes.new(kind)
        n.name = PREFIX + name
        n.label = name
        added.append(n.name)
        return n

    rest = node('ShaderNodeAttribute', 'stable rest coordinates')
    rest.attribute_name = REST
    split = node('ShaderNodeSeparateXYZ', 'rest XZ')
    nt.links.new(rest.outputs['Vector'], split.inputs[0])
    combine = node('ShaderNodeCombineXYZ', 'single interior patch')
    for index, axis, output in ((0, 'X', 'X'), (1, 'Z', 'Y')):
        mapping = node('ShaderNodeMapRange', axis + ' bounded image coordinate')
        mapping.data_type = 'FLOAT'
        mapping.interpolation_type = 'LINEAR'
        mapping.clamp = True
        mapping.inputs['From Min'].default_value = plan['restXZMinMeters'][index]
        mapping.inputs['From Max'].default_value = plan['restXZMaxMeters'][index]
        mapping.inputs['To Min'].default_value = MARGIN
        mapping.inputs['To Max'].default_value = 1 - MARGIN
        nt.links.new(split.outputs[axis], mapping.inputs['Value'])
        nt.links.new(mapping.outputs['Result'], combine.inputs[output])
    texture = node('ShaderNodeTexImage', 'synthetic nonrepeating lip patch')
    texture.image = image
    texture.projection = 'FLAT'
    texture.extension = 'EXTEND'
    texture.interpolation = 'Linear'
    nt.links.new(combine.outputs['Vector'], texture.inputs['Vector'])
    channels = node('ShaderNodeSeparateColor', 'green data channel')
    channels.mode = 'RGB'
    nt.links.new(texture.outputs['Color'], channels.inputs['Color'])
    center = node('ShaderNodeMath', 'remove median plateau')
    center.operation = 'SUBTRACT'
    center.inputs[1].default_value = plan['greenMedianBaseline']
    nt.links.new(channels.outputs['Green'], center.inputs[0])
    normalize = node('ShaderNodeMath', 'unit peak-to-valley data')
    normalize.operation = 'DIVIDE'
    normalize.inputs[1].default_value = plan['greenRange'][1] - plan['greenRange'][0]
    nt.links.new(center.outputs[0], normalize.inputs[0])
    support = node('ShaderNodeAttribute', 'existing true vermilion support')
    support.attribute_name = MASK
    clamp = node('ShaderNodeClamp', 'bounded existing mask')
    clamp.clamp_type = 'MINMAX'
    clamp.inputs['Min'].default_value = 0
    clamp.inputs['Max'].default_value = 1
    nt.links.new(support.outputs['Fac'], clamp.inputs['Value'])
    square = node('ShaderNodeMath', 'conservative smooth outer fade')
    square.operation = 'MULTIPLY'
    nt.links.new(clamp.outputs['Result'], square.inputs[0])
    nt.links.new(clamp.outputs['Result'], square.inputs[1])
    masked = node('ShaderNodeMath', 'lip-only centered height')
    masked.operation = 'MULTIPLY'
    nt.links.new(normalize.outputs[0], masked.inputs[0])
    nt.links.new(square.outputs[0], masked.inputs[1])
    bump = node('ShaderNodeBump', 'bounded added normal')
    bump.inputs['Strength'].default_value = 1
    bump.inputs['Distance'].default_value = plan['distanceMeters']
    bump.invert = False
    nt.links.new(masked.outputs[0], bump.inputs['Height'])
    if old_normal is not None:
        nt.links.new(old_normal, bump.inputs['Normal'])
    nt.links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    return added, old_link


def apply(microns=5.0):
    import bpy
    source = Path(bpy.data.filepath).resolve()
    source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Independent background Object Mode candidate required')
    if MARKER in bpy.context.scene:
        raise RuntimeError('Restore/preserve the existing lip microrelief candidate first')
    head = bpy.data.objects[HEAD]
    mesh = head.data
    if head.library or mesh.library or mesh.users != 1 or len(mesh.vertices) != 10582:
        raise RuntimeError('Complete local single-user R11 female head required')
    skin = _api('natural_skin_adult_r11')
    donor, landmarks, provenance = skin['_donor']()
    if not np.array_equal(_array(mesh.loops, 'vertex_index', 1, np.int32).ravel(), donor['corner_vertices']):
        raise RuntimeError('Actual female lip topology identity changed')
    rest = mesh.attributes.get(REST)
    support = mesh.attributes.get(MASK)
    if not rest or (rest.domain, rest.data_type) != ('POINT', 'FLOAT_VECTOR'):
        raise RuntimeError('Actual stable rest attribute missing')
    if not support or (support.domain, support.data_type) != ('POINT', 'FLOAT'):
        raise RuntimeError('Actual rose-lip support attribute missing')
    candidates = [(i, s.material) for i, s in enumerate(head.material_slots)
                  if s.material and s.material.use_nodes and any(n.type == 'VERTEX_COLOR' and n.layer_name == COLOR for n in s.material.node_tree.nodes)]
    if len(candidates) != 1:
        raise RuntimeError('Expected one current refined-natural-skin slot')
    slot, original = candidates[0]
    shaders = [n for n in original.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
    if len(shaders) != 1 or any(n.name.startswith(PREFIX) for n in original.node_tree.nodes):
        raise RuntimeError('Explicit single unmodified skin Principled required')
    path = project_path() / IMAGE
    if _sha(path) != IMAGE_SHA:
        raise RuntimeError('Generated image differs from reviewed v2')
    face = _api('runtime_face_refresh_r08')
    eye = _api('eye_margin_refine_r11')
    bpy.context.view_layer.update()
    before = eye['_state'](head, slot)
    outside = eye['_external'](bpy, head, [], '')
    old_hash = face['_material_contract'](original)
    old_tree = _hash(face['_tree_contract'](original.node_tree))
    source_sha = _sha(source)
    source_was_dirty = bool(bpy.data.is_dirty)
    old_fake = original.use_fake_user
    image = candidate = None
    success = False
    try:
        # Never repurpose an existing shared image or alter its colour space.
        image = bpy.data.images.load(str(path), check_existing=False)
        image.name = 'R11 lip striation v2 data'
        image.colorspace_settings.name = 'Non-Color'
        image.alpha_mode = 'NONE'
        if tuple(image.size) != (1254, 1254) or image.channels not in (3, 4) or len(image.pixels) != 1254 * 1254 * 4:
            raise RuntimeError('Unexpected decoded v2 image dimensions/channels')
        rgba = np.empty(len(image.pixels), np.float32)
        image.pixels.foreach_get(rgba)
        rgba = rgba.reshape(-1, 4)
        plan = field_plan(_array(rest.data, 'vector'), _array(support.data, 'value', 1).ravel(),
                          landmarks['mouth']['vermilionStripVertexIndices'],
                          landmarks['mouth']['outerVermilionLoopCandidate']['sourceVertexIndices'],
                          rgba[:, 1], microns)
        image.pack()
        if not image.packed_file or hashlib.sha256(bytes(image.packed_file.data)).hexdigest() != IMAGE_SHA:
            raise RuntimeError('Packed candidate texture differs from reviewed source')
        candidate = original.copy()
        candidate.name = 'R11 restrained lip microrelief ' + format(float(microns), 'g') + ' um'
        added, old_link = _wire(candidate, shaders[0].name, image, plan)
        if _normalized_tree(candidate, old_link, added, face) != old_tree:
            raise RuntimeError('Original material graph changed beyond the added normal chain')
        original.use_fake_user = True
        head.material_slots[slot].material = candidate
        bpy.context.view_layer.update()
        if _hash(eye['_state'](head, slot)) != _hash(before):
            raise RuntimeError('Head geometry, maps, UV, keys, weights or modifiers changed')
        if eye['_external'](bpy, head, [], '') != outside or face['_material_contract'](original) != old_hash or _sha(source) != source_sha:
            raise RuntimeError('External scene, original material or saved source changed')
        report = {
            'source': source.relative_to(project_path()).as_posix(), 'sourceSha256': source_sha,
            'sourceMayContainEarlierInMemoryModules': source_was_dirty,
            'helperSha256': _sha(__file__), 'head': HEAD, 'slot': slot, 'provenance': provenance,
            'originalMaterial': original.name, 'originalMaterialSha256': old_hash,
            'originalFakeUser': old_fake, 'candidateMaterial': candidate.name,
            'candidateMaterialSha256': face['_material_contract'](candidate),
            'image': IMAGE, 'imageSha256': IMAGE_SHA, 'packedImage': image.name,
            'imageMode': 'Non-Color / Green channel / FLAT / EXTEND / Linear; no repeated border',
            'plan': plan, 'addedNodes': added, 'originalNormalLink': old_link,
            'originalMaterialGraphPreserved': True,
            'headGeometryMapsUVKeysWeightsModifiersExact': True,
            'externalObjectsActionsFrameExact': True,
            'sourceSaved': False, 'appearanceAccepted': False,
            'limits': ['Synthetic art texture, not a scan or measured physical displacement.',
                       'Existing skin normal remains; nominal microns concern only the added height field.',
                       'Source masks and UV unchanged. Not baked to delivery UV.',
                       'Requires same-camera original/5um diagnostic review first, then selected expression follow-up.']
        }
        candidate['R11_lip_microrelief_scope'] = 'Very light bounded local normal study; not appearance accepted.'
        bpy.context.scene[MARKER] = json.dumps(report)
        success = True
        return report
    finally:
        if not success:
            head.material_slots[slot].material = original
            original.use_fake_user = old_fake
            if MARKER in bpy.context.scene:
                del bpy.context.scene[MARKER]
            if candidate is not None:
                candidate.use_fake_user = False
                if candidate.users == 0:
                    bpy.data.materials.remove(candidate)
            if image is not None and image.users == 0:
                bpy.data.images.remove(image)
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:
        return False
    report = json.loads(bpy.context.scene[MARKER])
    head = bpy.data.objects[report['head']]
    original = bpy.data.materials[report['originalMaterial']]
    candidate = bpy.data.materials[report['candidateMaterial']]
    face = _api('runtime_face_refresh_r08')
    if head.material_slots[report['slot']].material != candidate:
        raise RuntimeError('Skin assignment changed after candidate; preserve it before restore')
    if face['_material_contract'](original) != report['originalMaterialSha256'] or face['_material_contract'](candidate) != report['candidateMaterialSha256']:
        raise RuntimeError('Material edited since candidate; refuse destructive restore')
    if _ah(_array(head.data.attributes[REST].data, 'vector')) != report['plan']['restAttributeSha256'] or _ah(_array(head.data.attributes[MASK].data, 'value', 1).ravel()) != report['plan']['maskAttributeSha256']:
        raise RuntimeError('Rest coordinates or lip support changed since candidate')
    head.material_slots[report['slot']].material = original
    original.use_fake_user = report['originalFakeUser']
    candidate.use_fake_user = True
    del bpy.context.scene[MARKER]
    bpy.context.view_layer.update()
    return {'restoredOriginalMaterial': original.name, 'retainedCandidateMaterial': candidate.name,
            'packedTextureRetained': report['packedImage'], 'sourceSaved': False}


def prepare(source_relative, output_relative, expected_source_sha256, microns=5.0):
    import bpy
    source, output = _path(source_relative), _path(output_relative)
    report_path = output.with_suffix('.lip-microrelief.json')
    if source == output or output.exists() or report_path.exists():
        raise RuntimeError('Preserve source/history; use unused independent output paths')
    if Path(bpy.data.filepath).resolve() != source or bpy.data.is_dirty or _sha(source) != expected_source_sha256:
        raise RuntimeError('Fresh explicit source and exact expected SHA required')
    report = apply(microns)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
        raise RuntimeError('Independent candidate save failed')
    report.update(sourceSaved=True, output=output.relative_to(project_path()).as_posix(), outputSha256=_sha(output))
    with report_path.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_LIP_MICRORELIEF_SAVED ' + json.dumps({'output': report['output'], 'sha256': report['outputSha256']}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected-source-sha256', required=True)
    p.add_argument('--microns', type=float, default=5.0)
    args = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source, args.output, args.expected_source_sha256, args.microns)
