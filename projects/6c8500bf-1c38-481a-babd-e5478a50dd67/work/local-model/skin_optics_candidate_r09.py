"""One material-only optics candidate; no import-time Blender execution.

The original physical-skin material is retained under a backup name. A copy
keeps the canonical R04 physical skin name for downstream material-domain
recognition. Only Principled IOR and Diffuse Roughness change. All existing
region colour, roughness, rest-space micro-normal and SSS inputs remain exact.
Use run(source_relative=...) in an explicitly opened independent source, or
apply_and_save() for a NEW candidate file. No render or light modifications.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R01 Head'; MATERIAL='R04 physical skin'
MARKER='R09_skin_optics_candidate'
BACKUP_NAME='R09 skin optics original - R04 physical skin'
CHANGES={'IOR':(1.5,1.4),'Diffuse Roughness':(0.,.12)}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _path(relative,suffix):
    path=(project_path()/relative).resolve()
    path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix!=suffix:raise ValueError('Unexpected independent work artifact type')
    return path


def _shader(material):
    if not material.use_nodes or not material.node_tree:raise RuntimeError('Expected existing node-based skin')
    nodes=[node for node in material.node_tree.nodes if node.type=='BSDF_PRINCIPLED']
    if len(nodes)!=1:raise RuntimeError('Expected one existing skin Principled shader')
    return nodes[0]


def _constant(shader,name,expected):
    socket=shader.inputs[name]
    if socket.is_linked or np.max(np.abs(np.asarray(socket.default_value)-np.asarray(expected)))>2e-6:
        raise RuntimeError('Skin optical input differs from actual observed R09 setup: '+name)


def _from(socket,node_type):
    if len(socket.links)!=1 or socket.links[0].from_node.type!=node_type:
        raise RuntimeError('Existing skin input chain changed: '+socket.name)
    return socket.links[0].from_node


def _source_chain(head,shader):
    colour=_from(shader.inputs['Base Color'],'VERTEX_COLOR')
    if colour.layer_name!='R01_region_color':raise RuntimeError('Preserve the actual R01 region colour source')
    rough=_from(shader.inputs['Roughness'],'ATTRIBUTE')
    if rough.attribute_name!='R04 skin roughness':raise RuntimeError('Preserve anatomical roughness point map')
    bump=_from(shader.inputs['Normal'],'BUMP');noise=_from(bump.inputs['Height'],'TEX_NOISE')
    rest=_from(noise.inputs['Vector'],'ATTRIBUTE')
    if rest.attribute_name!='R04 skin rest position':raise RuntimeError('Preserve existing rest-space micro surface')
    for name,kind in (('R04 skin rest position','FLOAT_VECTOR'),('R04 skin roughness','FLOAT')):
        attr=head.data.attributes.get(name)
        if attr is None or attr.domain!='POINT' or attr.data_type!=kind:
            raise RuntimeError('Original source skin attribute unavailable: '+name)
    if head.data.color_attributes.get('R01_region_color') is None:raise RuntimeError('Original region colours unavailable')
    for name,value in [('Specular IOR Level',.5),('Subsurface Weight',.14),('Subsurface Scale',.0018),
                       ('Subsurface Radius',(1.,.4,.2)),('Coat Weight',0.),('Metallic',0.)]:
        _constant(shader,name,value)
    for name,(old,new) in CHANGES.items():_constant(shader,name,old)
    return {'baseColourAttribute':colour.layer_name,'roughnessAttribute':rough.attribute_name,
            'microNormal':{'bumpNode':bump.name,'noiseNode':noise.name,'positionAttribute':rest.attribute_name},
            'colourNodesAdded':False,'noiseNodesAdded':False,'subsurfaceUnchanged':True}


def _graph_except_inputs(material,shader_name,face_api):
    state=face_api['_tree_contract'](material.node_tree)
    rows=[row for row in state['nodes'] if row['name']==shader_name]
    if len(rows)!=1:raise RuntimeError('Skin shader identity changed')
    for socket in rows[0]['inputs']:
        if socket['name'] in CHANGES:socket['value']='EXPLICITLY_CHANGED_INPUT'
    return state


def _head_contract(head,index,uv):
    state=uv['_original_contract'](head,tuple(layer.name for layer in head.data.uv_layers))
    state.pop('sha256')
    state['materialSlots'][index]={'link':head.material_slots[index].link,'explicitSkinSlot':index}
    return state


def _other_material_slots(bpy,head,index):
    return {obj.name:[{'index':i,'link':slot.link,'material':slot.material.name if slot.material else None,
                      'pointer':slot.material.as_pointer() if slot.material else None}
                     for i,slot in enumerate(obj.material_slots) if not (obj==head and i==index)]
            for obj in bpy.context.scene.objects}


def run(source_relative,diagnostics_only=False):
    import bpy
    source=_path(source_relative,'.blend')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the explicit independent saved source in background Object Mode')
    head=bpy.data.objects[HEAD]
    if bpy.context.scene.get(MARKER) or bpy.data.materials.get(BACKUP_NAME):
        raise RuntimeError('Preserve or restore the previous skin optics candidate first')
    if head.library or head.data.library or head.data.users!=1:raise RuntimeError('Local single-user production head required')
    indices=[i for i,slot in enumerate(head.material_slots) if slot.material and slot.material.name==MATERIAL]
    if len(indices)!=1:raise RuntimeError('Expected one explicit production physical-skin domain')
    index=indices[0];slot=head.material_slots[index];original=slot.material
    if original.library:raise RuntimeError('Local original physical-skin material required')
    shared=[obj.name for obj in bpy.data.objects for s in obj.material_slots if s.material==original and obj!=head]
    if shared:raise RuntimeError('Original face skin also assigned to other objects: '+repr(shared))
    shader=_shader(original);chain=_source_chain(head,shader)
    face_api=_api('runtime_face_refresh_r08');uv=_api('production_uv_r08')
    groom=_api('posterior_groom_r09');perioral=_api('perioral_delta_r09')
    bpy.context.view_layer.update()
    head_before=_head_contract(head,index,uv)
    others_before=perioral['_other_scene_contract'](bpy,uv,groom)
    slots_before=_other_material_slots(bpy,head,index)
    graph_before=_graph_except_inputs(original,shader.name,face_api)
    original_hash=face_api['_material_contract'](original);original_fake=original.use_fake_user
    material_settings=uv['_scalar_rna'](original)
    for key in ('name','use_fake_user'):material_settings.pop(key,None)
    source_sha=_sha(source)
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
        'object':HEAD,'materialSlot':index,'canonicalMaterial':MATERIAL,
        'changes':{name:{'old':old,'new':new} for name,(old,new) in CHANGES.items()},
        'normalIncidenceFresnel':{'old':((1.5-1)/(1.5+1))**2,'new':((1.4-1)/(1.4+1))**2,
            'meaning':'Interface F0 only; not a prediction of rendered image brightness.'},
        'preservedSkinChain':chain,'originalMaterialSha256':original_hash,
        'colourVariationAdded':False,'reasonForRestrainedScope':'Evaluate reduced specular reflection under the same light before adding pigment/noise variation.',
        'savedBlend':False,'writtenInMemory':False,'appearanceAccepted':False,
        'requiredEvidence':'One same-light front closeup versus explicit source; inspect forehead/nose/cheek highlights, lip/eye colour and face-to-neck transition.',
        'limits':['Existing colour uniformity and face geometry are not repaired by optical constants.',
                  'SSS and roughness remain unchanged; the render may still show waxiness from shape or lighting.',
                  'Blender Principled diffuse roughness/SSS are not automatically reproduced by VRM MToon.']}
    if diagnostics_only:return report
    candidate=None;committed=False
    try:
        candidate=original.copy();candidate.name='R09 skin optics pending'
        bs=_shader(candidate)
        for name,(_,new) in CHANGES.items():bs.inputs[name].default_value=new
        if _graph_except_inputs(candidate,bs.name,face_api)!=graph_before:
            raise RuntimeError('Skin graph changed beyond the two explicit optical constants')
        candidate_settings=uv['_scalar_rna'](candidate)
        for key in ('name','use_fake_user'):candidate_settings.pop(key,None)
        if candidate_settings!=material_settings:raise RuntimeError('Material settings changed during copy')
        for name,(_,new) in CHANGES.items():_constant(bs,name,new)
        if face_api['_material_contract'](original)!=original_hash:raise RuntimeError('Original skin graph changed')
        original.name=BACKUP_NAME;original.use_fake_user=True
        candidate.name=MATERIAL
        if candidate.name!=MATERIAL or original.name!=BACKUP_NAME:raise RuntimeError('Canonical skin name could not be retained uniquely')
        slot.material=candidate;bpy.context.view_layer.update()
        if _head_contract(head,index,uv)!=head_before:raise RuntimeError('Head coordinates, keys, UV, attributes, weights or other material domains changed')
        if perioral['_other_scene_contract'](bpy,uv,groom)!=others_before:
            raise RuntimeError('Other scene geometry, native cache keys, actions or transforms changed')
        if _other_material_slots(bpy,head,index)!=slots_before:raise RuntimeError('A non-skin material assignment changed')
        if face_api['_material_contract'](original)!=original_hash or _sha(source)!=source_sha:
            raise RuntimeError('Retained original material or saved source changed')
        report.update(writtenInMemory=True,retainedOriginalMaterial=original.name,originalFakeUser=original_fake,
            candidateMaterialSha256=face_api['_material_contract'](candidate),
            geometryKeysUVAttributesWeightsAndCacheUnchanged=True,otherMaterialAssignmentsUnchanged=True,
            existingColourRoughnessMicroNormalAndSSSUnchanged=True,
            sourceMaterialNameRetainedForDownstreamRecognition=True)
        bpy.context.scene[MARKER]=json.dumps(report)
        committed=True;return report
    finally:
        if not committed:
            slot.material=original
            if candidate is not None:
                candidate.use_fake_user=False
                if candidate.users==0:bpy.data.materials.remove(candidate)
            original.name=MATERIAL;original.use_fake_user=original_fake
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


def apply_and_save(source_relative,output_relative='work/reconstruction-r02/reference-skin-optics-r09.blend',
                   report_relative='work/reconstruction-r02/skin-optics-r09.json'):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=_path(report_relative,'.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New candidate and report paths required')
    report=run(source_relative)
    if output.exists():raise RuntimeError('Candidate output appeared during preparation')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    record=json.loads(bpy.context.scene[MARKER]);face_api=_api('runtime_face_refresh_r08')
    original=bpy.data.materials.get(record['retainedOriginalMaterial']);candidate=bpy.data.materials.get(MATERIAL)
    if original is None or face_api['_material_contract'](original)!=record['originalMaterialSha256']:
        raise RuntimeError('Retained original skin material missing or changed')
    if candidate is None or face_api['_material_contract'](candidate)!=record['candidateMaterialSha256']:
        raise RuntimeError('Candidate skin was edited after this pass; preserve edits before restoring')
    slot=bpy.data.objects[HEAD].material_slots[record['materialSlot']]
    if slot.material!=candidate:raise RuntimeError('Skin assignment was changed after this pass')
    candidate.name='R09 skin optics restored candidate';candidate.use_fake_user=True
    original.name=MATERIAL;original.use_fake_user=record['originalFakeUser'];slot.material=original
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'restoredMaterial':original.name,'retainedCandidateMaterial':candidate.name,'savedBlend':False}


if __name__=='__main__':print('Loaded only. Explicit run/apply_and_save(source_relative=...) creates one independent optical candidate.')
