"""Sample the current character identity and facial drivers into runtime meshes.

Only an independently saved background file under work/runtime-r06/ is allowed.
The current identity keys remain active in the sampled Basis. Individual output
keys are actual evaluated results for explicit control states, including curves,
eye motion and the jaw-parented oral components. No source Blend is overwritten.
No material bake, save, export or automatic visual acceptance happens here.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix
except ImportError:
    bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PREFIX='RT06 Face / '
CUSTOM_EXPRESSIONS=('mbp','pucker','squintLeft','squintRight')
STATIC_IDENTITY_KEYS=('R09_lower_jaw_curvature',)
FACE_NAMES=['R01 Head']+[f'R01 {kind} {s}' for kind in ('Sclera','Textured iris') for s in ('L','R')]+[
 'R02 Soft brow -1','R02 Soft brow 1']+[f'R02 {kind} {s}' for kind in ('Upper lashes','Upper lid rim') for s in ('L','R')]+[
 'R02 Upper teeth','R02 Lower teeth','R02 Tongue','R03 Upper gingiva','R03 Lower gingiva']
CONTROLS=['R02_blink_L','R02_blink_R','R03_squint_L','R03_squint_R','R02_jaw_open','R02_mouth_wide',
          'R02_smile','R02_frown','R02_pucker','R02_lip_press','R02_brow_raise','R02_brow_furrow']
DEFAULT_STATES={
 'blink':{'R02_blink_L':1,'R02_blink_R':1},'blinkLeft':{'R02_blink_L':1},'blinkRight':{'R02_blink_R':1},
 'happy':{'R02_smile':1,'R02_brow_raise':.15},
 'angry':{'R02_smile':0,'R02_brow_furrow':.9,'R02_frown':.25,'R02_lip_press':.25},
 'sad':{'R02_smile':0,'R02_frown':.8,'R02_brow_raise':.15},
 'relaxed':{'R02_smile':.6},'surprised':{'R02_smile':0,'R02_brow_raise':1,'R02_jaw_open':.55},
 'aa':{'R02_jaw_open':.65,'R02_mouth_wide':.15},'ee':{'R02_jaw_open':.2,'R02_mouth_wide':.75},
 'ih':{'R02_jaw_open':.12,'R02_mouth_wide':.35},'oh':{'R02_jaw_open':.45,'R02_pucker':.60},
 'ou':{'R02_jaw_open':.2,'R02_pucker':.85},
 'mbp':{'R02_lip_press':.85},'pucker':{'R02_pucker':1},
 'squintLeft':{'R03_squint_L':1},'squintRight':{'R03_squint_R':1},
 'lookLeft':{'look_yaw':18},'lookRight':{'look_yaw':-18},
 'lookUp':{'look_pitch':-18},'lookDown':{'look_pitch':18}}

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Disposable background Blender required')
    p=project_path();file=Path(bpy.data.filepath).resolve();file.relative_to((p/'work/runtime-r06').resolve())
    if not file.is_file() or bpy.context.mode!='OBJECT':raise RuntimeError('Saved runtime copy in Object mode required')
    if any(o.name.startswith(PREFIX) for o in bpy.data.objects):raise RuntimeError('Facial snapshot already exists')
    return p,bpy.data.objects['Character_Rig'],bpy.data.objects['R01 Head']

def _coords(mesh,matrix):
    values=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',values);values=values.reshape(-1,3)
    m=np.array(matrix);return np.ascontiguousarray(values@m[:3,:3].T+m[:3,3],dtype=np.float32)

def _topology(mesh):
    h=hashlib.sha256()
    for p in mesh.polygons:h.update(np.asarray(tuple(p.vertices),dtype=np.int32).tobytes())
    return len(mesh.vertices),len(mesh.polygons),h.hexdigest()

def _clear_attachment_weights(obj):
    """Remove orphan evaluated deform data without a topology round trip.

    new_from_object() retains mesh deform indices and, in current Blender,
    may also retain group definitions. Materialize any missing definitions
    through the highest retained index, remove actual memberships explicitly,
    then remove the definitions. This applies only to the newly sampled rigid
    attachments, never the unified head/body or retained source objects.
    """
    original_groups=[g.name for g in obj.vertex_groups]
    members={}
    for vertex in obj.data.vertices:
        for item in vertex.groups:members.setdefault(int(item.group),[]).append(vertex.index)
    if any(index<0 for index in members):raise RuntimeError('Invalid evaluated deform index')
    for index in range(len(obj.vertex_groups),max(members,default=-1)+1):obj.vertex_groups.new(name='RT06 discarded attachment group '+str(index))
    for index,vertices in members.items():obj.vertex_groups[index].remove(vertices)
    if any(vertex.groups for vertex in obj.data.vertices):raise RuntimeError('Attachment deform data was not fully removed: '+obj.name)
    obj.vertex_groups.clear()
    return {'removedDeformIndices':sorted(members),'removedMemberships':sum(map(len,members.values())),
            'removedOriginalGroupDefinitions':original_groups}


def prepare(states=None,neutral_controls=None,max_subdivision=1):
    """Create face meshes and sampled target keys in memory; no save/export."""
    p,rig,head=_guard();states=dict(DEFAULT_STATES if states is None else states)
    if not 0<=max_subdivision<=1 or not 1<=len(states)<=24:raise ValueError('Bounded snapshot scope required')
    objects=[]
    for name in FACE_NAMES:
        obj=bpy.data.objects.get(name)
        if obj is None or obj.hide_render or obj.type not in {'MESH','CURVE'}:raise RuntimeError('Expected visible facial component: '+name)
        if any(m.type not in {'SUBSURF','ARMATURE','SOLIDIFY','BEVEL','WEIGHTED_NORMAL','NORMAL_EDIT'} for m in obj.modifiers):
            raise RuntimeError('Unreviewed facial modifier: '+name)
        objects.append(obj)
    ks=head.data.shape_keys.key_blocks
    if any(n not in ks for n in CONTROLS):raise RuntimeError('R05 facial controls are incomplete')
    identity={k.name:float(k.value) for k in ks if k.name not in CONTROLS and k.name!='Basis'}
    static_identity={name:identity[name] for name in STATIC_IDENTITY_KEYS if name in identity}
    for name in static_identity:
        key=ks[name]
        if key.relative_key.name!='Basis' or key.mute:
            raise RuntimeError('Static identity must remain an enabled Basis-relative source key: '+name)
    baseline={n:(.45 if n=='R02_smile' else 0.) for n in CONTROLS}
    if neutral_controls:baseline.update(neutral_controls)
    if set(baseline)-set(CONTROLS):raise ValueError('Unknown neutral control')
    known=set(CONTROLS)|{'look_yaw','look_pitch'}
    for name,values in states.items():
        if set(values)-known:raise ValueError('Unknown facial control in '+name)
        if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite facial control')
    # Work-copy rig is neutralized deliberately. The source on disk remains unchanged.
    if rig.animation_data:rig.animation_data.action=None
    for bone in rig.pose.bones:bone.matrix_basis.identity()
    master=rig.pose.bones.get('CTRL_R02_Master')
    if master:
        for key in ('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R'):
            if key in master:master[key]=0.
    rig.data.pose_position='POSE'
    subdivision_sampling=[]
    for obj in objects:
        if obj.data.shape_keys and obj.data.shape_keys.animation_data:
            obj.data.shape_keys.animation_data.action=None
            for track in obj.data.shape_keys.animation_data.nla_tracks:track.mute=True
        for modifier in obj.modifiers:
            if modifier.type=='ARMATURE':modifier.show_viewport=modifier.show_render=False
            elif modifier.type=='SUBSURF':
                original_viewport=int(modifier.levels);original_render=int(modifier.render_levels)
                sampled=min(max(original_viewport,original_render),max_subdivision)
                modifier.levels=modifier.render_levels=sampled
                subdivision_sampling.append({'object':obj.name,'modifier':modifier.name,
                    'originalViewportLevels':original_viewport,'originalRenderLevels':original_render,
                    'sampledLevels':sampled,'maximumAllowedLevels':max_subdivision,
                    'showViewport':bool(modifier.show_viewport),'showRender':bool(modifier.show_render)})
    rig.update_tag()
    def controls(values):
        for name,value in baseline.items():ks[name].value=values.get(name,value)
        rig['R03_look_yaw']=float(values.get('look_yaw',0));rig['R03_look_pitch']=float(values.get('look_pitch',0))
        rig.update_tag();head.update_tag();bpy.context.view_layer.update()
        for name,value in static_identity.items():
            if abs(float(ks[name].value)-value)>1e-7:
                raise RuntimeError('Static identity changed while sampling an expression: '+name)
    controls({});graph=bpy.context.evaluated_depsgraph_get();packets={}
    for obj in objects:
        evaluated=obj.evaluated_get(graph)
        mesh=bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=graph)
        mat=rig.matrix_world.inverted()@evaluated.matrix_world
        packet={'source':obj,'mesh':mesh,'basis':_coords(mesh,mat),'topology':_topology(mesh),'states':{}}
        mesh.vertices.foreach_set('co',packet['basis'].ravel());mesh.update();packets[obj.name]=packet
    try:
        for state,values in states.items():
            controls(values);graph=bpy.context.evaluated_depsgraph_get()
            for obj in objects:
                evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
                try:
                    packet=packets[obj.name]
                    if _topology(mesh)!=packet['topology']:raise RuntimeError('Expression changed evaluated topology: '+obj.name+'/'+state)
                    co=_coords(mesh,rig.matrix_world.inverted()@evaluated.matrix_world)
                    if not np.isfinite(co).all():raise RuntimeError('Nonfinite expression coordinates')
                    if float(np.linalg.norm(co-packet['basis'],axis=1).max())>1e-7:packet['states'][state]=co
                finally:evaluated.to_mesh_clear()
            print('RT06_FACE_SAMPLE',state,flush=True)
        controls({})
        collection=bpy.data.collections.new('RT06 Sampled facial meshes');bpy.context.scene.collection.children.link(collection)
        rows=[]
        for name,packet in packets.items():
            source=packet['source'];mesh=packet['mesh'];obj=bpy.data.objects.new(PREFIX+name,mesh);collection.objects.link(obj)
            obj.matrix_world=rig.matrix_world.copy();obj['part']='face' if source==head else 'face_detail';obj['facial_role']='runtime_sampled'
            obj['RT06_source_object']=name
            used=sorted({face.material_index for face in mesh.polygons});materials=list(mesh.materials)
            remap={old:i for i,old in enumerate(used)}
            if any(i>=len(materials) or materials[i] is None for i in used):raise RuntimeError('Used facial material slot is missing')
            old_indices=[face.material_index for face in mesh.polygons]
            mesh.materials.clear()
            for i in used:mesh.materials.append(materials[i])
            for face,old_index in zip(mesh.polygons,old_indices):face.material_index=remap[old_index]
            if source==head:
                for g in sorted(source.vertex_groups,key=lambda x:x.index):
                    if g.index<len(obj.vertex_groups):
                        if obj.vertex_groups[g.index].name!=g.name:raise RuntimeError('Evaluated head group-index mapping changed')
                    else:obj.vertex_groups.new(name=g.name)
                if len(obj.vertex_groups)!=len(source.vertex_groups):raise RuntimeError('Evaluated head has unexpected group definitions')
                if any(not v.groups for v in mesh.vertices):raise RuntimeError('Evaluated head lost weights')
                weight_transfer={'mode':'preserved evaluated unified head/body weights'}
            else:
                # Oral/eye parent transforms are captured into the morph targets.
                # Uniform head weighting prevents applying those rotations twice.
                weight_transfer=_clear_attachment_weights(obj)
                weight_transfer['mode']='sampled attachment; old deform data removed; uniform head skin'
                if _topology(mesh)!=packet['topology']:raise RuntimeError('Attachment weight cleanup changed topology: '+name)
                group=obj.vertex_groups.new(name='head');group.add(list(range(len(mesh.vertices))),1,'REPLACE')
                if any(len(v.groups)!=1 or v.groups[0].group!=group.index or abs(v.groups[0].weight-1)>1e-6 for v in mesh.vertices):
                    raise RuntimeError('Attachment head weights are not exactly normalized: '+name)
            modifier=obj.modifiers.new('RT06 facial skin','ARMATURE');modifier.object=rig;modifier.use_deform_preserve_volume=False
            basis=obj.shape_key_add(name='Basis',from_mix=False);basis.data.foreach_set('co',packet['basis'].ravel())
            for state,co in packet['states'].items():
                key=obj.shape_key_add(name='RT06_'+state,from_mix=False);key.data.foreach_set('co',co.ravel());key.value=0
            expected_keys={'Basis',*('RT06_'+state for state in packet['states'])}
            if {key.name for key in obj.data.shape_keys.key_blocks}!=expected_keys:
                raise RuntimeError('Identity or unrequested source keys leaked into runtime expression targets')
            source.hide_render=True;source.hide_set(True);source['RT06_replaced_by']=obj.name
            rows.append({'source':name,'mesh':obj.name,'vertices':len(mesh.vertices),'faces':len(mesh.polygons),
                         'keys':list(packet['states']),'basisSha256':hashlib.sha256(packet['basis'].tobytes()).hexdigest(),
                         'materialSlots':[m.name for m in mesh.materials],'sourceMaterialSlotRemap':remap,
                         'weightTransfer':weight_transfer})
        report={'runtimeCopy':bpy.data.filepath,'identityValuesCaptured':identity,'neutralControls':baseline,'states':states,'meshes':rows,
                'staticIdentityBakedIntoBasis':static_identity,
                'staticIdentityPolicy':'Preserve current source values for every sample. No original identity key is exported as an expression; only Basis plus explicit RT06 states are created.',
                'subdivisionSampling':subdivision_sampling,
                'sourceModifiedOnDisk':False,'savedBlend':False,'materialBakePerformed':False,
                'limits':['VRM morph combinations are additive; the full Blender driver graph is not exported',
                          'Mood presets are explicit templates sampled from existing controls and require appearance review',
                          'Gaze is sampled into head-weighted eye meshes; use expression LookAt, not simultaneous eye-bone LookAt',
                          'Snapshot geometry includes at most one level of subdivision; inspect close-up curvature',
                          'Old source objects/drivers remain hidden for reference and must be excluded from export selection']}
        (p/'work/runtime-r06/face-snapshot-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'objects':len(rows),'states':len(states),'savedBlend':False}));return report
    except Exception:
        # The disposable process can be discarded. Source file is not saved.
        controls({});raise

def configure_vrm(manifest_relative='work/runtime-r06/face-snapshot-report.json'):
    """Bind sampled presets and four explicit VRM1 custom expressions (4.10.2)."""
    if bpy is None or not bpy.app.background:raise RuntimeError('Background copy required')
    Path(bpy.data.filepath).resolve().relative_to((project_path()/'work/runtime-r06').resolve())
    report=json.loads((project_path()/manifest_relative).read_text(encoding='utf-8'));rig=bpy.data.objects['Character_Rig']
    expressions=rig.data.vrm_addon_extension.vrm1.expressions
    expressions.initial_automatic_expression_assignment=False;expressions.fill_missing_expression_names()
    presets=expressions.preset.name_to_expression_dict();mapping=dict(presets);bound={};custom_bound={};settings={}
    missing=set(CUSTOM_EXPRESSIONS)-set(report['states'])
    unknown=set(report['states'])-set(presets)-set(CUSTOM_EXPRESSIONS)
    if missing or unknown:raise RuntimeError('Required expression scope mismatch: missing custom='+repr(sorted(missing))+' unknown='+repr(sorted(unknown)))
    for name in CUSTOM_EXPRESSIONS:
        existing=[e for e in expressions.custom if e.custom_name==name]
        if len(existing)>1 or name in presets:raise RuntimeError('Custom expression name collision: '+name)
        e=existing[0] if existing else expressions.custom.add()
        if not existing:e.custom_name=name
        if e.custom_name!=name:raise RuntimeError('VRM addon changed required custom name: '+name)
        mapping[name]=e
    for name in report['states']:
        e=mapping[name];e.preview=0;e.morph_target_binds.clear();e.material_color_binds.clear();e.texture_transform_binds.clear();e.is_binary=False
        e.override_blink='none';e.override_mouth='blend' if name in {'happy','angry','sad','relaxed','surprised'} else 'none'
        e.override_look_at='blend' if name in {'blink','blinkLeft','blinkRight'} else 'none'
        if name in {'mbp','pucker'}:e.override_mouth='blend'
        if name in {'squintLeft','squintRight'}:e.override_look_at='blend'
        binds=[]
        for row in report['meshes']:
            if name not in row['keys']:continue
            obj=bpy.data.objects[row['mesh']];key='RT06_'+name
            if not obj.data.shape_keys.key_blocks.get(key):raise RuntimeError('Snapshot key missing')
            bind=e.morph_target_binds.add();bind.node.mesh_object_name=obj.name;bind.index=key;bind.weight=1
            binds.append({'mesh':obj.name,'key':key})
        if not binds:raise RuntimeError('Sampled expression has no actual morph bindings: '+name)
        (custom_bound if name in CUSTOM_EXPRESSIONS else bound)[name]=binds
        settings[name]={'isBinary':bool(e.is_binary),'overrideBlink':e.override_blink,
            'overrideMouth':e.override_mouth,'overrideLookAt':e.override_look_at}
    look=rig.data.vrm_addon_extension.vrm1.look_at
    look.type=look.TYPE_EXPRESSION.identifier
    for attr in ('range_map_horizontal_inner','range_map_horizontal_outer','range_map_vertical_down','range_map_vertical_up'):
        value=getattr(look,attr);value.input_max_value=18;value.output_scale=1
    result={'presets':bound,'custom':custom_bound,'settings':settings,'unboundAdditionalKeys':[],
            'customSemantics':'mbp/pucker blend down vowel presets; squintLeft/Right blend down gaze presets; existing preset override settings are unchanged.',
            'lookAt':'expression','blinkLookAtOverride':'blend',
            'scope':'Bindings only; actual VRM viewer must verify left/right gaze signs, partial/combined blinks, vowels and morph sums.'}
    (project_path()/'work/runtime-r06/face-vrm-bindings.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':print('Loaded only. prepare(), then material bake and configure_vrm() in disposable runtime copy.')
