"""Complete explicit runtime bindings/selection without touching production files.

Only background copies under work/runtime-r06/. The cloth replacement transfers
the retained original skirt's authored weights through its exact sampled index
map; it does not pretend to export the Blender cloth simulation/cache. No save,
material conversion, export or visual certification happens in this module.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
try:
    import bpy
    from mathutils import Matrix
except ImportError:
    bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SKIRT='RT06 Skirt spring preview'

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Disposable background process required')
    path=Path(bpy.data.filepath).resolve();path.relative_to((project_path()/'work/runtime-r06').resolve())
    if not path.is_file() or bpy.context.mode!='OBJECT':raise RuntimeError('Saved Object-mode runtime copy required')
    return bpy.data.objects['Character_Rig']

def _helper(name):
    path=Path(__file__).with_name(name+'.py');spec=importlib.util.spec_from_file_location('rt06_'+name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def _select(obj):
    bpy.ops.object.select_all(action='DESELECT');obj.hide_set(False);obj.select_set(True);bpy.context.view_layer.objects.active=obj

def skirt_from_control():
    rig=_guard()
    if bpy.data.objects.get(SKIRT):raise RuntimeError('Skirt already adapted')
    control=bpy.data.objects['R02 Cloth Skirt control'];original=bpy.data.objects['Skirt_Sixteen_Pleats']
    if len(control.data.vertices)!=2176 or len(original.data.vertices)!=256*31:raise RuntimeError('Exact skirt sample correspondence changed')
    columns=[16*panel+i for panel in range(16) for i in (0,3,6,7,8,9,10,13)]
    rows=[0,1,2]+list(range(4,31,2));source_indices=[r*256+c for r in rows for c in columns]
    names={g.index:g.name for g in original.vertex_groups};weights=[]
    for index in source_indices:
        w={names[g.group]:float(g.weight) for g in original.data.vertices[index].groups
           if names[g.group] in rig.data.bones and rig.data.bones[names[g.group]].use_deform and g.weight>1e-8}
        if not w or abs(sum(w.values())-1)>1e-4:raise RuntimeError('Original skirt weights are missing/non-normalized')
        weights.append(w)
    obj=control.copy();obj.data=control.data.copy();obj.name=SKIRT;bpy.context.scene.collection.objects.link(obj)
    obj.parent=None;obj.matrix_world=control.matrix_world.copy();obj.hide_render=False;obj.hide_set(False)
    for modifier in list(obj.modifiers):
        if modifier.type in {'CLOTH','COLLISION','ARMATURE'}:obj.modifiers.remove(modifier)
        elif modifier.type=='SUBSURF':modifier.levels=modifier.render_levels=min(1,modifier.render_levels)
    obj.vertex_groups.clear();groups={n:obj.vertex_groups.new(name=n) for n in sorted({n for w in weights for n in w})}
    for i,w in enumerate(weights):
        for n,value in w.items():groups[n].add([i],value,'REPLACE')
    mod=obj.modifiers.new('RT06 Original skirt spring weights','ARMATURE');mod.object=rig;mod.use_deform_preserve_volume=False
    _select(obj)
    while list(obj.modifiers).index(mod)>0:bpy.ops.object.modifier_move_up(modifier=mod.name)
    obj.parent=rig;obj.matrix_parent_inverse=rig.matrix_world.inverted();obj['part']='lower_body';obj['deform_role']='skirt'
    obj['RT06_source_object']=control.name;obj['RT06_weight_source']=original.name;obj['RT06_not_cloth_simulation']=True
    control.hide_render=True;control.hide_set(True)
    # The production source may contain the independent walk simulation too.
    # Keep its mesh/cache in the copy, but render/export only the adapted skirt.
    for name in ('R07 Native Walk Cloth playback','R06 Walk cloth','Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam'):
        previous=bpy.data.objects.get(name)
        if previous:previous.hide_render=True;previous.hide_set(True)
    maximum_drop=max(sum(sorted(w.values(),reverse=True)[4:]) for w in weights)
    report={'mesh':obj.name,'vertices':len(weights),'sourceIndexMap':'17 rows x128 columns, same map as cloth_setup_r02.py',
            'deformGroups':list(groups),'maxWeightSumError':max(abs(sum(w.values())-1) for w in weights),
            'maxWeightDiscardedIfExportedWithFourInfluences':maximum_drop,
            'scope':'Rest control surface with retained authored spring weights; Blender cloth cache is not retained in VRM'}
    (project_path()/'work/runtime-r06/skirt-adaptation-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

def bind_bone_parent_meshes():
    """Give visible rigid attachments explicit armature weights for VRM."""
    rig=_guard();rows=[]
    for obj in list(bpy.context.scene.objects):
        if obj.type!='MESH' or obj.hide_render or obj.parent!=rig or obj.parent_type!='BONE':continue
        if obj.data.shape_keys:raise RuntimeError('Sample keyed facial attachments first: '+obj.name)
        if any(m.type=='ARMATURE' for m in obj.modifiers) or obj.vertex_groups:raise RuntimeError('Mixed parent/skin binding needs review: '+obj.name)
        bone=obj.parent_bone
        if bone not in rig.data.bones or not rig.data.bones[bone].use_deform:raise RuntimeError('Invalid deform parent')
        world=obj.matrix_world.copy();transform=rig.matrix_world.inverted()@world
        obj.data=obj.data.copy();obj.data.transform(transform);obj.data.update()
        obj.parent=rig;obj.parent_type='OBJECT';obj.parent_bone='';obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
        group=obj.vertex_groups.new(name=bone);group.add(list(range(len(obj.data.vertices))),1,'REPLACE')
        modifier=obj.modifiers.new('RT06 Rigid attachment','ARMATURE');modifier.object=rig;modifier.use_deform_preserve_volume=False
        _select(obj)
        while list(obj.modifiers).index(modifier)>0:bpy.ops.object.modifier_move_up(modifier=modifier.name)
        rows.append({'mesh':obj.name,'bone':bone,'vertices':len(obj.data.vertices)})
    return rows

def current_meshes():
    rig=_guard();meshes=[];unhandled=[]
    for obj in bpy.context.scene.objects:
        if obj.hide_render:continue
        associated=(obj.parent==rig or any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers))
        if not associated:continue
        if obj.type=='MESH':
            if any(m.type=='CLOTH' for m in obj.modifiers):raise RuntimeError('Unadapted visible cloth remains: '+obj.name)
            meshes.append(obj)
        elif obj.type in {'CURVE','CURVES','SURFACE','FONT','META'}:unhandled.append(obj.name)
    if unhandled:raise RuntimeError('Visible non-mesh character parts remain: '+repr(unhandled))
    required={'RT06 Face / R01 Head',SKIRT,'Hand L · continuous anatomy','Hand R · continuous anatomy',
              'Body_Complete_Shin_Foot_L','Body_Complete_Shin_Foot_R','Leg L · knee calf ankle','Leg R · knee calf ankle'}
    missing=required-{o.name for o in meshes}
    if missing:raise RuntimeError('Essential body/face parts missing from runtime selection: '+repr(sorted(missing)))
    if not any(o.name.startswith('RT06 Cards / ') for o in meshes):raise RuntimeError('Runtime hair cards missing')
    return rig,meshes

def configure_springs(collider_group='Character body',tail_collider_groups=('Shoulders · ponytails','Back garment · ponytails')):
    rig,meshes=current_meshes();api=_helper('vrm_helpers');api._enabled()
    sb=rig.data.vrm_addon_extension.spring_bone1
    if collider_group not in {g.vrm_name for g in sb.collider_groups}:raise RuntimeError('Named source body collider group missing')
    required_tail_groups=[collider_group,*tail_collider_groups]
    if len(set(required_tail_groups))!=len(required_tail_groups) or any(name not in {g.vrm_name for g in sb.collider_groups} for name in required_tail_groups):
        raise RuntimeError('Explicit existing body/shoulder/back collider groups required for runtime tails')
    old=json.loads(rig.get('charakit_spring_plan_v1','null'))
    if not old or old.get('version')!=1:raise RuntimeError('Retained source spring plan missing')
    names={o.name for o in meshes};targets=[]
    for old_target in old['targets']:
        target=dict(old_target)
        if target['mesh']=='Skirt_Sixteen_Pleats':target['mesh']=SKIRT
        if target['mesh'] in names:targets.append(target)
    hair=json.loads((project_path()/'work/runtime-r06/haircards-report.json').read_text())
    targets.extend(hair['springTargets']);chains=hair['springChainProposal']
    for chain in chains:chain['colliderGroups']=required_tail_groups
    covered={t['mesh'] for t in targets}
    for obj in meshes:
        if obj.name in covered:continue
        if obj.name.startswith(('R06 attached scalp','R02 Tie ')):
            targets.append({'mesh':obj.name,'source':'hair','mode':'fixed',
                            'reason':'Measured scalp substrate or fitted tie is rigidly attached to head; free tail is a separate spring target.'})
    if len({t['mesh'] for t in targets})!=len(targets):raise RuntimeError('Duplicate explicit spring target')
    required_chains={label for t in targets if t['mode']=='spring' for label in t['chains']}
    removed=[]
    for i in range(len(sb.springs)-1,-1,-1):
        if sb.springs[i].vrm_name not in required_chains:
            removed.append(sb.springs[i].vrm_name);sb.springs.remove(i)
    result=api.register_springs(rig,chains,targets)
    result.update(runtimeRemovedUnusedHistoricalChains=removed,colliderGroupInherited=collider_group,tailColliderGroups=required_tail_groups,
        colliderVisualRevalidationRequired=True,sourceBlendChangedOnDisk=False)
    (project_path()/'work/runtime-r06/spring-registration-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def inventory():
    rig,meshes=current_meshes();api=_helper('vrm_helpers');binding=api.audit_bindings(rig,meshes)
    result={'runtimeCopy':bpy.data.filepath,'rig':rig.name,'meshes':[o.name for o in meshes],'bindings':binding,
            'visibleNativeGeometryRemaining':False,'savedBlend':False,
            'scope':'Explicit mesh selection and structural skin inventory; no export, runtime motion or appearance acceptance'}
    (project_path()/'work/runtime-r06/assembly-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':print('Loaded only. skirt_from_control(), bind_bone_parent_meshes(), configure_springs(), inventory().')
