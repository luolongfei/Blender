"""One reversible iris-only material candidate on the already softened eyes.

Preserve the current sclera, original image, optical coat, geometry and eyelids.
Restore saturation to 1, use the requested warm linear multiplier, and remove
only the iris base-lobe reflection. The existing sharp coat remains the sole
reflective lobe. Explicit run(source_relative=...), no source save or render.
Restore this layer before optionally restoring eye_tone_candidate_r09.
"""
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
IRIS=('R01 Textured iris L','R01 Textured iris R')
SCLERA=('R01 Sclera L','R01 Sclera R')
MARKER='R09_iris_single_reflection_candidate'
BACKUP='R09_iris_optics_previous_material'
MULTIPLIER=(.85,.72,.58,1.)


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def run(source_relative,diagnostics_only=False):
    import bpy
    p=project_path();source=(p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Exact fresh saved independent current appearance source required')
    if not bpy.context.scene.get('R09_eye_tone_candidate'):
        raise RuntimeError('Use the current softened-sclera/eye-tone source')
    if bpy.context.scene.get(MARKER) or any(BACKUP in bpy.data.objects[n] for n in IRIS):
        raise RuntimeError('Iris optical candidate already exists; retain or restore before retrying')
    helper=_api('eye_tone_candidate_r09');util=_api('posterior_groom_r09');uv=_api('production_uv_r08')
    objects=[bpy.data.objects[n] for n in IRIS]
    for obj in objects:
        if obj.type!='MESH' or obj.library or obj.data.library or len(obj.material_slots)!=1:
            raise RuntimeError('Expected existing local one-slot iris objects')
    old=objects[0].material_slots[0].material
    if old is None or objects[1].material_slots[0].material!=old:
        raise RuntimeError('Expected current shared bilateral iris material')
    bs=helper['_shader'](old)
    for name,value in [('Specular IOR Level',.5),('Roughness',.14),('Coat Weight',1.),
                       ('Coat Roughness',.03),('Coat IOR',1.376)]:helper['_expected'](bs,name,value)
    sat=old.node_tree.nodes.get('R09 restrained iris saturation')
    mult=old.node_tree.nodes.get('R09 deep brown reflectance')
    if sat is None or sat.type!='HUE_SAT' or mult is None or mult.type!='MIX_RGB' or mult.blend_type!='MULTIPLY':
        raise RuntimeError('Expected the actual R09 eye-tone nodes; do not add a second correction chain')
    if abs(sat.inputs['Saturation'].default_value-.78)>2e-5 or np.max(np.abs(np.asarray(mult.inputs[2].default_value)-[.54,.53,.51,1.]))>2e-5:
        raise RuntimeError('Current iris color controls differ from reviewed grey-eye candidate')
    base_links=list(bs.inputs['Base Color'].links)
    if len(base_links)!=1 or base_links[0].from_node!=mult:
        raise RuntimeError('Actual iris Base Color path differs from the reviewed correction chain')
    texture=p/'textures/procedural-r07/iris-brown-r07-v3.png'
    texture_sha=helper['_sha'](texture)
    if texture_sha!=helper['IRIS_TEXTURE_SHA']:raise RuntimeError('Documented iris source texture changed')
    geometry_before=helper['_geometry'](bpy,util,uv)
    old_state=helper['_material_state'](old);old_fake=old.use_fake_user
    sclera_before={n:{'material':bpy.data.objects[n].material_slots[0].material.name,
                     'state':helper['_material_state'](bpy.data.objects[n].material_slots[0].material)} for n in SCLERA}
    slots={obj.name:{'material':old.name,'link':obj.material_slots[0].link} for obj in objects}
    source_sha=helper['_sha'](source);candidate=None;committed=False
    try:
        candidate=old.copy();candidate.name='R09 warm iris single reflection candidate'
        node=candidate.node_tree.nodes['R09 restrained iris saturation'];node.inputs['Saturation'].default_value=1.
        node.label='Original iris saturation restored'
        node=candidate.node_tree.nodes['R09 deep brown reflectance'];node.inputs[2].default_value=MULTIPLIER
        node.label='Warm linear diffuse color; original pixels retained'
        shader=helper['_shader'](candidate);shader.inputs['Specular IOR Level'].default_value=0.
        for name in ('Roughness','IOR','Coat Weight','Coat Roughness','Coat IOR','Coat Tint','Coat Normal'):
            a=shader.inputs[name];b=bs.inputs[name]
            if hasattr(a,'default_value') and helper['_value'](a.default_value)!=helper['_value'](b.default_value):
                raise RuntimeError('Existing coat/roughness input changed: '+name)
        for node in old.node_tree.nodes:
            if node.type=='TEX_IMAGE' and candidate.node_tree.nodes[node.name].image!=node.image:
                raise RuntimeError('Original iris image was replaced')
        # Verify the exact small change set in the copied graph: no hidden
        # emission, opacity, roughness, normal, IOR or other material adjustment.
        candidate_state=helper['_material_state'](candidate)
        normalized=json.loads(json.dumps(candidate_state));normalized['material']=old_state['material']
        old_rows={row['name']:row for row in old_state['nodes']}
        for node in normalized['nodes']:
            if node['name']=='R09 restrained iris saturation':
                node['inputs']['Saturation']=old_rows[node['name']]['inputs']['Saturation']
            if node['name']=='R09 deep brown reflectance':
                socket_name=mult.inputs[2].name
                node['inputs'][socket_name]=old_rows[node['name']]['inputs'][socket_name]
            if node['type']=='BSDF_PRINCIPLED':
                node['inputs']['Specular IOR Level']=old_rows[node['name']]['inputs']['Specular IOR Level']
        # The JSON deep copy converts link tuples to lists. Compare the same
        # canonical serialization, retaining every material/socket/link field.
        if helper['_hash'](normalized)!=helper['_hash'](old_state):
            raise RuntimeError('Candidate changed more than the three requested iris inputs')
        report={'source':str(source.relative_to(p)),'sourceSha256':source_sha,
            'originalMaterial':old.name,'originalStateSha256':helper['_hash'](old_state),
            'originalFakeUser':old_fake,'originalSlots':slots,'candidateMaterial':candidate.name,
            'texture':str(texture.relative_to(p)),'textureSha256':texture_sha,
            'changes':{'saturation':[.78,1.],'linearMultiplier':[[.54,.53,.51,1.],MULTIPLIER],
                       'baseSpecularIORLevel':[.5,0.]},
            'preserved':{'irisRoughness':.14,'coatWeight':1.,'coatRoughness':.03,'coatIOR':1.376,
                         'scleraMaterialNames':{n:r['material'] for n,r in sclera_before.items()},
                         'originalTexturePixels':True,'eyeGeometryAndEyelids':True},
            'savedBlend':False,'writtenInMemory':not diagnostics_only,'appearanceAccepted':False,
            'requiredEvidence':'One same-light front crop: restored warm dark iris detail without broad grey base reflection; retain sharp coat catchlights and readable sclera.',
            'runtimeConsequence':'If adopted, read actual source material parameters; previous runtime optical-unchanged assumptions no longer apply.',
            'rollbackOrder':'Restore this optical layer before optionally restoring eye_tone_candidate_r09.'}
        if diagnostics_only:return report
        old.use_fake_user=True
        for obj in objects:
            obj[BACKUP]=json.dumps(slots[obj.name]);obj.material_slots[0].material=candidate
        bpy.context.scene[MARKER]=json.dumps(report);bpy.context.view_layer.update()
        if helper['_geometry'](bpy,util,uv)!=geometry_before:raise RuntimeError('Protected geometry/rig changed')
        for name,row in sclera_before.items():
            material=bpy.data.objects[name].material_slots[0].material
            if material.name!=row['material'] or helper['_material_state'](material)!=row['state']:
                raise RuntimeError('Accepted soft sclera changed: '+name)
        if helper['_material_state'](old)!=old_state or helper['_sha'](texture)!=texture_sha or helper['_sha'](source)!=source_sha:
            raise RuntimeError('Retained original iris/source/texture changed')
        committed=True;return report
    finally:
        if not committed:
            for obj in objects:
                obj.material_slots[0].link=slots[obj.name]['link'];obj.material_slots[0].material=old
                if BACKUP in obj:del obj[BACKUP]
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            old.use_fake_user=old_fake
            if candidate is not None:
                candidate.use_fake_user=False
                if candidate.users==0:bpy.data.materials.remove(candidate)
            bpy.context.view_layer.update()


def restore():
    import bpy
    scene=bpy.context.scene
    if not scene.get(MARKER):return False
    helper=_api('eye_tone_candidate_r09');report=json.loads(scene[MARKER])
    old=bpy.data.materials.get(report['originalMaterial'])
    if old is None or helper['_hash'](helper['_material_state'](old))!=report['originalStateSha256']:
        raise RuntimeError('Retained previous iris material missing/changed; refuse blind restore')
    for name,row in report['originalSlots'].items():
        obj=bpy.data.objects[name]
        if json.loads(obj[BACKUP])!=row:raise RuntimeError('Iris material restore metadata changed')
    candidate=bpy.data.materials.get(report['candidateMaterial'])
    if candidate is not None:candidate.use_fake_user=True
    for name,row in report['originalSlots'].items():
        obj=bpy.data.objects[name];obj.material_slots[0].link=row['link'];obj.material_slots[0].material=old
        del obj[BACKUP]
    old.use_fake_user=report['originalFakeUser'];del scene[MARKER];bpy.context.view_layer.update()
    return {'restored':old.name,'retainedCandidate':report['candidateMaterial'],'savedBlend':False}


if __name__=='__main__':print('Loaded only. Explicit run(source_relative=...) creates one reversible iris material candidate.')
