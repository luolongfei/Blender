"""Separate reversible R09 hair collider and highlight candidates; no auto-run.

These edits do not resolve/claim to resolve missing posterior scalp coverage.
No topology, UV, weights, morph keys, images, spring joint settings or rig rest
state is changed. The integrator saves a NEW runtime checkpoint after review.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
GROUPS=('Character body','Shoulders · ponytails','Back garment · ponytails')
SPRINGS=('RT06 tail L','RT06 tail R')
ROUGHNESS={'scalp':.78,'gather':.76,'fringe':.76,'wisps':.78,'tail':.76}
COLLIDER_BACKUP='RT09_tail_collider_backup';MATERIAL_BACKUP='RT09_hair_material_backup'


def _guard():
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Saved background runtime Object Mode required')
    path=Path(bpy.data.filepath).resolve();path.relative_to((project_path()/'work/runtime-r06').resolve())
    if not path.is_file():raise RuntimeError('Saved runtime checkpoint required')
    return bpy.data.objects['Character_Rig']


def _report(name,value):
    value.update(sourceRuntime=bpy.data.filepath,savedBlend=False,appearanceAccepted=False)
    root=project_path()/'work/runtime-r06'
    for index in range(1,1000):
        path=root/f'{name}-{index:03d}.json'
        try:
            with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
            return value
        except FileExistsError:continue
    raise RuntimeError('Too many preserved review reports')


def _refs(spring):
    # VRM add-on 4.10.2 exposes UUID plus this read-only resolved display name.
    # collider_group_name belongs to the older API and is absent here.
    return [{'uuid':ref.collider_group_uuid,'name':ref.collider_group_display_name} for ref in spring.collider_groups]


def _set_refs(spring,refs):
    if any(not value['uuid'] for value in refs):raise RuntimeError('An explicit valid 4.10.2 collider UUID is required')
    spring.collider_groups.clear()
    for value in refs:
        ref=spring.collider_groups.add()
        ref.collider_group_uuid=value['uuid']
        if ref.collider_group_uuid!=value['uuid']:raise RuntimeError('VRM add-on rejected unknown collider group UUID')


def _joints(spring):
    return [{'bone':j.node.bone_name,'stiffness':float(j.stiffness),'gravityPower':float(j.gravity_power),
        'gravityDir':list(j.gravity_dir),'dragForce':float(j.drag_force),'hitRadius':float(j.hit_radius)} for j in spring.joints]


def link_tail_colliders():
    rig=_guard()
    if rig.get(COLLIDER_BACKUP):raise RuntimeError('Collider candidate already applied')
    sb=rig.data.vrm_addon_extension.spring_bone1;groups={g.vrm_name:g for g in sb.collider_groups};springs={s.vrm_name:s for s in sb.springs}
    if any(name not in groups or not groups[name].colliders for name in GROUPS):raise RuntimeError('Existing body/shoulder/back collider groups are required')
    if any(name not in springs for name in SPRINGS):raise RuntimeError('Both runtime tail chains are required')
    old={name:_refs(springs[name]) for name in SPRINGS};joint_state={name:_joints(springs[name]) for name in SPRINGS}
    for name in SPRINGS:
        side=name[-1];expected=[f'RT06 tail_{side}_{i:02d}' for i in range(6)]+[f'RT06 tail_{side}_05_CharakitEnd']
        if [j['bone'] for j in joint_state[name]]!=expected:raise RuntimeError('Unexpected tail joint ordering: '+name)
    desired=[{'uuid':groups[name].uuid,'name':groups[name].vrm_name} for name in GROUPS]
    rig[COLLIDER_BACKUP]=json.dumps(old,ensure_ascii=False)
    try:
        for name in SPRINGS:_set_refs(springs[name],desired)
        bpy.context.view_layer.update()
        for name in SPRINGS:
            if _joints(springs[name])!=joint_state[name]:raise RuntimeError('Joint settings unexpectedly changed')
            if [r.collider_group_uuid for r in springs[name].collider_groups]!=[r['uuid'] for r in desired]:raise RuntimeError('Collider references did not persist')
    except Exception:
        for name in SPRINGS:_set_refs(springs[name],old[name])
        del rig[COLLIDER_BACKUP];raise
    return _report('r09-tail-collider-link',{'before':old,'after':{name:_refs(springs[name]) for name in SPRINGS},
        'groupNames':list(GROUPS),'jointsUnchanged':joint_state,
        'reason':'R08 tails referenced only Character body; existing shoulder/back collider groups were never associated.',
        'reviewRequired':'Repeat front/side/back physics ON/OFF and bounded head/body motion. Existing collider coverage/radii may still need measured adjustment.'})


def restore_tail_colliders():
    rig=_guard();data=rig.get(COLLIDER_BACKUP)
    if not data:return False
    springs={s.vrm_name:s for s in rig.data.vrm_addon_extension.spring_bone1.springs}
    for name,refs in json.loads(data).items():_set_refs(springs[name],refs)
    del rig[COLLIDER_BACKUP];bpy.context.view_layer.update();return True


def soften_highlights(include_root=True):
    rig=_guard()
    if rig.get(MATERIAL_BACKUP):raise RuntimeError('Highlight candidate already applied')
    materials={};users={}
    for obj in bpy.context.scene.objects:
        if obj.hide_render or obj.type!='MESH':continue
        for index,material in enumerate(obj.data.materials):
            if material and material.get('RT06_source_projected_atlas'):
                region=material.get('RT06_hair_region')
                if region not in ROUGHNESS:raise RuntimeError('Unreviewed projected hair region')
                if region in materials and materials[region]!=material:raise RuntimeError('Duplicate active region material')
                materials[region]=material;users.setdefault(material.name,[]).append((obj,index))
    if set(materials)!=set(ROUGHNESS):raise RuntimeError('All five current projected atlas materials are required')
    if include_root:
        root=bpy.data.objects['R06 attached scalp root substrate']
        if len(root.data.materials)!=1 or not root.data.materials[0].get('r06_baked_material'):raise RuntimeError('Expected baked root substrate')
        material=root.data.materials[0];materials['root']=material;users[material.name]=[(root,0)]
    backups=[];rows=[];copies=[]
    try:
        for region,material in materials.items():
            original_shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
            if original_shader.inputs['Specular IOR Level'].is_linked:raise RuntimeError('Linked specular needs explicit adaptation')
            if region!='root' and original_shader.inputs['Roughness'].is_linked:raise RuntimeError('Unexpected linked card roughness')
            copy=material.copy();copy.name=material.name+' / R09 soft highlight';copies.append(copy)
            shader=next(n for n in copy.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
            row={'region':region,'sourceMaterial':material.name,'candidateMaterial':copy.name,
                'sourceSpecularIORLevel':float(original_shader.inputs['Specular IOR Level'].default_value)}
            shader.inputs['Specular IOR Level'].default_value=.18 if region=='root' else .14
            row['candidateSpecularIORLevel']=float(shader.inputs['Specular IOR Level'].default_value)
            if region!='root':
                row['sourceRoughness']=float(original_shader.inputs['Roughness'].default_value)
                shader.inputs['Roughness'].default_value=ROUGHNESS[region];row['candidateRoughness']=ROUGHNESS[region]
            else:row['roughnessTexture']='Existing baked roughness link retained unchanged'
            for obj,index in users[material.name]:
                backups.append({'object':obj.name,'slot':index,'material':material.name});material.use_fake_user=True;obj.data.materials[index]=copy
            rows.append(row)
        rig[MATERIAL_BACKUP]=json.dumps(backups,ensure_ascii=False)
    except Exception:
        for row in backups:bpy.data.objects[row['object']].data.materials[row['slot']]=bpy.data.materials[row['material']]
        raise
    return _report('r09-hair-highlight',{'materials':rows,'imagesAlphaUVGeometryWeightsAndExpressions':'Unchanged',
        'limits':'Only a conservative highlight candidate. R08 has no exported hair anisotropy/strand normal map. These constants cannot fix exposed cap, faceting or missing posterior fibres.'})


def restore_highlights():
    rig=_guard();data=rig.get(MATERIAL_BACKUP)
    if not data:return False
    for row in json.loads(data):bpy.data.objects[row['object']].data.materials[row['slot']]=bpy.data.materials[row['material']]
    del rig[MATERIAL_BACKUP];return True


if __name__=='__main__':print('Loaded only. link_tail_colliders() first; soften_highlights() is a separate optional candidate. Save/review new checkpoint explicitly.')
