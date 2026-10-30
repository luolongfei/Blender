"""Moderate reversible eye-colour candidate, strictly material-only.

run(source_relative=...) copies the existing shared iris and sclera materials.
It preserves the documented iris image and all iris reflection/coat settings,
adds restrained desaturation and a linear colour multiplier before Base Color,
and changes only sclera diffuse colour plus a very weak clear coat. No source
image write, eye/eyelid movement, geometry edit, save, render or light change.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SOURCE = 'work/reconstruction-r02/reference-forelock-flow-r09.blend'
IRIS = ('R01 Textured iris L', 'R01 Textured iris R')
SCLERA = ('R01 Sclera L', 'R01 Sclera R')
MARKER = 'R09_eye_tone_candidate'
BACKUP = 'R09_eye_tone_previous_materials'
SATURATION = .78
MULTIPLIER = (.54, .53, .51, 1.)
SCLERA_COLOR = (.44, .42, .38, 1.)
IRIS_TEXTURE_SHA = '6bddc86582fa1b4541376082816c4d734deabbbcbe38e65710415823f0566784'


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _value(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    return [float(v) for v in value]


def _material_state(material):
    rows = []
    for node in material.node_tree.nodes:
        row = {'name':node.name, 'type':node.type,
               'inputs':{s.name:_value(s.default_value) for s in node.inputs if hasattr(s, 'default_value')}}
        for attr in ('blend_type', 'operation', 'use_clamp', 'interpolation', 'extension', 'projection'):
            if hasattr(node, attr):
                row[attr] = _value(getattr(node, attr))
        if node.type == 'TEX_IMAGE':
            row['image'] = None if node.image is None else {
                'name':node.image.name, 'filepath':node.image.filepath,
                'colorspace':node.image.colorspace_settings.name, 'isDirty':node.image.is_dirty,
                'size':list(node.image.size), 'packed':bool(node.image.packed_file)}
        rows.append(row)
    links = [(l.from_node.name, l.from_socket.name, l.to_node.name, l.to_socket.name)
             for l in material.node_tree.links]
    return {'material':material.name, 'useNodes':material.use_nodes,
            'diffuseColor':list(material.diffuse_color), 'nodes':rows, 'links':sorted(links)}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _shader(material):
    if not material.use_nodes or material.node_tree is None:
        raise RuntimeError('Existing node material required: '+material.name)
    nodes = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
    if len(nodes) != 1:
        raise RuntimeError('Expected one existing Principled surface: '+material.name)
    return nodes[0]


def _expected(shader, name, expected):
    socket = shader.inputs[name]
    if socket.is_linked or np.max(np.abs(np.asarray(socket.default_value)-np.asarray(expected))) > 2e-5:
        raise RuntimeError('Actual saved eye optical input differs from reviewed setup: '+name)


def _geometry(bpy, util, uv):
    rows = util['_protected'](bpy, uv)
    # The material edit touches only these four mesh slots. Explicitly hash
    # their geometry/UVs and transforms, in addition to head keys/hair/rig.
    for name in IRIS+SCLERA:
        obj = bpy.data.objects[name]
        rows[name] = {'mesh':util['_data_hash'](obj.data), 'matrix':uv['_plain'](obj.matrix_world),
            'parent':uv['_plain'](obj.parent), 'parentType':obj.parent_type, 'parentBone':obj.parent_bone,
            'modifiers':[uv['_scalar_rna'](m) for m in obj.modifiers]}
    return rows


def run(source_relative=SOURCE, diagnostics_only=False):
    import bpy
    p = project_path(); source = (p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or Path(bpy.data.filepath).resolve() != source or bpy.data.is_dirty:
        raise RuntimeError('Exact fresh saved independent source in background Object Mode required')
    if bpy.context.scene.get(MARKER) or any(BACKUP in bpy.data.objects[n] for n in IRIS+SCLERA):
        raise RuntimeError('An eye tone candidate is already present; retain or restore before retrying')
    if not bpy.data.objects['R01 Head'].get('R09_lower_face_candidate'):
        raise RuntimeError('Use the current independent R09 production candidate')
    objects = [bpy.data.objects[n] for n in IRIS+SCLERA]
    for obj in objects:
        if obj.type != 'MESH' or obj.library or obj.data.library or obj.data.users != 1 or len(obj.material_slots) != 1:
            raise RuntimeError('Expected local single-user eye mesh with one material slot: '+obj.name)
    old_iris = objects[0].material_slots[0].material
    old_sclera = objects[2].material_slots[0].material
    if old_iris is None or old_sclera is None or objects[1].material_slots[0].material != old_iris or objects[3].material_slots[0].material != old_sclera:
        raise RuntimeError('Expected the current shared bilateral iris and sclera materials')
    ibs = _shader(old_iris); sbs = _shader(old_sclera)
    for name, value in [('Roughness', .14), ('Coat Weight', 1.), ('Coat Roughness', .03),
                        ('Coat IOR', 1.376), ('Specular IOR Level', .5)]:
        _expected(ibs, name, value)
    for name, value in [('Base Color', (.64, .59, .49, 1.)), ('Roughness', .19),
                        ('Coat Weight', 0.), ('Specular IOR Level', .5)]:
        _expected(sbs, name, value)
    links = list(ibs.inputs['Base Color'].links)
    if len(links) != 1 or links[0].from_node.type != 'TEX_IMAGE' or links[0].from_socket.name != 'Color':
        raise RuntimeError('Expected the documented iris texture directly feeding Base Color')
    image_node = links[0].from_node; image = image_node.image
    if image is None or image.colorspace_settings.name != 'sRGB':
        raise RuntimeError('Existing documented sRGB iris image is required')
    texture = p/'textures/procedural-r07/iris-brown-r07-v3.png'
    if not texture.is_file() or _sha(texture) != IRIS_TEXTURE_SHA or Path(image.filepath).name != texture.name:
        raise RuntimeError('Iris image identity differs from the documented unchanged R07 source')
    util = _api('posterior_groom_r09'); uv = _api('production_uv_r08')
    geometry_before = _geometry(bpy, util, uv); source_sha = _sha(source)
    originals = {m.name:{'material':m, 'state':_material_state(m), 'fakeUser':m.use_fake_user}
                 for m in (old_iris, old_sclera)}
    slots = {obj.name:[{'index':i, 'material':s.material.name, 'link':s.link}
                       for i, s in enumerate(obj.material_slots)] for obj in objects}
    copies = []; committed = False
    try:
        iris = old_iris.copy(); iris.name = 'R09 restrained deep brown iris candidate'; copies.append(iris)
        sclera = old_sclera.copy(); sclera.name = 'R09 soft warm sclera candidate'; copies.append(sclera)
        bs = _shader(iris); tree = iris.node_tree
        source_node = tree.nodes[image_node.name]
        if source_node.image != image:
            raise RuntimeError('Copy changed the shared original iris image')
        saturation = tree.nodes.new('ShaderNodeHueSaturation')
        saturation.name = 'R09 restrained iris saturation'
        saturation.label = 'Existing texture; modest saturation reduction'
        saturation.inputs['Hue'].default_value = .5
        saturation.inputs['Saturation'].default_value = SATURATION
        saturation.inputs['Value'].default_value = 1.
        saturation.inputs['Fac'].default_value = 1.
        multiply = tree.nodes.new('ShaderNodeMixRGB')
        multiply.name = 'R09 deep brown reflectance'; multiply.label = 'Linear reflectance only; highlights unchanged'
        multiply.blend_type = 'MULTIPLY'; multiply.use_clamp = False
        multiply.inputs[0].default_value = 1.; multiply.inputs[2].default_value = MULTIPLIER
        saturation.location = (source_node.location.x+220, source_node.location.y)
        multiply.location = (source_node.location.x+440, source_node.location.y)
        tree.links.new(source_node.outputs['Color'], saturation.inputs['Color'])
        tree.links.new(saturation.outputs['Color'], multiply.inputs[1])
        tree.links.new(multiply.outputs['Color'], bs.inputs['Base Color'])
        # No tint or attenuation of the existing specular/coat layer. The pale
        # reflection component may persist; that must be judged in the render.
        for name in ('Roughness', 'IOR', 'Specular IOR Level', 'Coat Weight', 'Coat Roughness', 'Coat IOR', 'Coat Tint'):
            if _value(bs.inputs[name].default_value) != _value(ibs.inputs[name].default_value):
                raise RuntimeError('Iris optical input changed: '+name)
        sb = _shader(sclera)
        sb.inputs['Base Color'].default_value = SCLERA_COLOR
        sb.inputs['Coat Weight'].default_value = .06
        sb.inputs['Coat Roughness'].default_value = .03
        sb.inputs['Coat IOR'].default_value = 1.376
        iris['R09_source_material'] = old_iris.name; sclera['R09_source_material'] = old_sclera.name
        for name, record in originals.items():
            if _material_state(record['material']) != record['state']:
                raise RuntimeError('Original material was modified: '+name)
        luminance = np.array([.2126, .7152, .0722])
        report = {'source':str(source.relative_to(p)), 'sourceSha256':source_sha,
            'texture':str(texture.relative_to(p)), 'textureSha256':IRIS_TEXTURE_SHA,
            'originalMaterials':{n:{'stateSha256':_hash(r['state']), 'fakeUser':r['fakeUser']} for n,r in originals.items()},
            'originalSlots':slots, 'candidateMaterials':[iris.name, sclera.name],
            'iris':{'saturation':SATURATION, 'linearMultiplier':MULTIPLIER,
                    'existingImageSharedUnchanged':True, 'allReflectionAndCoatInputsUnchanged':True},
            'sclera':{'oldLinearBaseColor':[.64,.59,.49,1.], 'newLinearBaseColor':SCLERA_COLOR,
                      'diffuseLuminanceRatio':float(np.dot(luminance,SCLERA_COLOR[:3])/np.dot(luminance,[.64,.59,.49])),
                      'roughnessUnchanged':.19, 'coatWeight':.06, 'coatRoughness':.03, 'coatIOR':1.376},
            'geometryEyesEyelidsRigUnchanged':True, 'imagesWritten':False,
            'savedBlend':False, 'writtenInMemory':not diagnostics_only, 'appearanceAccepted':False,
            'scope':'Only four eye material assignments. Existing iris texture, pupil/limbal detail, iris coat highlights and all eye geometry/eyelid contact preserved.',
            'requiredEvidence':'One same-light front closeup versus current source: darker brown iris with retained catchlights, sclera still readable, no gray/dull eye appearance.',
            'limit':'This diffuse-colour candidate does not remove the existing physical coat reflection or correct round eye shape.'}
        if diagnostics_only:
            return report
        for record in originals.values():
            record['material'].use_fake_user = True
        for obj in objects:
            obj[BACKUP] = json.dumps(slots[obj.name])
            obj.material_slots[0].material = iris if obj.name in IRIS else sclera
        bpy.context.scene[MARKER] = json.dumps(report)
        bpy.context.view_layer.update()
        if _geometry(bpy, util, uv) != geometry_before:
            raise RuntimeError('Geometry, transforms, UVs, rig or head keys changed during material assignment')
        if _sha(texture) != IRIS_TEXTURE_SHA or _sha(source) != source_sha:
            raise RuntimeError('Saved source or original texture changed')
        for name, record in originals.items():
            if _material_state(record['material']) != record['state']:
                raise RuntimeError('Original material graph changed after assignment: '+name)
        committed = True
        return report
    finally:
        if not committed:
            for obj in objects:
                for row in slots[obj.name]:
                    slot = obj.material_slots[row['index']]
                    slot.link = row['link']; slot.material = bpy.data.materials[row['material']]
                if BACKUP in obj:
                    del obj[BACKUP]
            if MARKER in bpy.context.scene:
                del bpy.context.scene[MARKER]
            for record in originals.values():
                record['material'].use_fake_user = record['fakeUser']
            for material in copies:
                material.use_fake_user = False
                if material.users == 0:
                    bpy.data.materials.remove(material)
            bpy.context.view_layer.update()


def restore():
    import bpy
    scene = bpy.context.scene
    if not scene.get(MARKER):
        return False
    record = json.loads(scene[MARKER])
    for name, old in record['originalMaterials'].items():
        material = bpy.data.materials.get(name)
        if material is None or _hash(_material_state(material)) != old['stateSha256']:
            raise RuntimeError('Retained original eye material missing or changed; refuse blind restore: '+name)
    for name in IRIS+SCLERA:
        obj = bpy.data.objects[name]
        if json.loads(obj[BACKUP]) != record['originalSlots'][name]:
            raise RuntimeError('Eye material restore metadata differs: '+name)
    for name in record['candidateMaterials']:
        if name in bpy.data.materials:
            bpy.data.materials[name].use_fake_user = True
    for name, slots in record['originalSlots'].items():
        obj = bpy.data.objects[name]
        for row in slots:
            slot = obj.material_slots[row['index']]
            slot.link = row['link']; slot.material = bpy.data.materials[row['material']]
        del obj[BACKUP]
    for name, old in record['originalMaterials'].items():
        bpy.data.materials[name].use_fake_user = old['fakeUser']
    del scene[MARKER]; bpy.context.view_layer.update()
    return {'restoredObjects':list(IRIS+SCLERA), 'retainedCandidateMaterials':record['candidateMaterials'], 'savedBlend':False}


if __name__ == '__main__':
    print('Loaded only. Explicit run(source_relative=...) creates a material-only candidate without saving/rendering.')
