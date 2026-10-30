"""Two explicit stages: source-only face payload, then reversible runtime swap.

Never launches Blender or renders. prepare_payload() saves a new disposable
source checkpoint and a 16-object library without rig dependencies. replace()
changes only the opened runtime copy in memory; the integrator saves a NEW
checkpoint after inspection. Existing hair/cloth atlases are not rebuilt.
"""
import hashlib,importlib.util,json,re,sys
from pathlib import Path
import numpy as np
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
RIG='Character_Rig';PREFIX='RT06 Face / '
TAIL_BONES={f'RT06 tail_{side}_{index:02d}' for side in ('L','R') for index in range(6)}
TAIL_TERMINALS={f'RT06 tail_{side}_05_CharakitEnd':f'RT06 tail_{side}_05' for side in ('L','R')}


def _api(name):
    spec=importlib.util.spec_from_file_location('refresh08_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def _path(relative,area='work/runtime-r06'):
    p=project_path();path=(p/relative).resolve();path.relative_to((p/area).resolve());return path


def _guard():
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Background Object Mode required')
    source=Path(bpy.data.filepath).resolve()
    if not source.is_file():raise RuntimeError('Saved independent source required')
    return source


def _rig_contract(rig,names=None):
    names=sorted(names if names is not None else rig.data.bones.keys())
    if any(name not in rig.data.bones for name in names):raise RuntimeError('Rest rig bone missing')
    rows=[]
    for name in names:
        b=rig.data.bones[name]
        rows.append({'name':name,'parent':b.parent.name if b.parent else None,'matrix':[list(row) for row in b.matrix_local],
            'head':list(b.head_local),'tail':list(b.tail_local),'deform':bool(b.use_deform),'connected':bool(b.use_connect),
            'inheritScale':b.inherit_scale,'inheritRotation':bool(b.use_inherit_rotation),'localLocation':bool(b.use_local_location)})
    value={'objectMatrix':[list(row) for row in rig.matrix_world],'bones':rows}
    return {'sha256':_hash(value),'data':value}


def _runtime_extra_bones(rig,source_names):
    extras=set(rig.data.bones.keys())-set(source_names)
    expected=TAIL_BONES|set(TAIL_TERMINALS)
    if extras!=expected:
        raise RuntimeError('Expected exactly twelve runtime tail bones and their two Charakit terminal endpoints; '
            'missing='+str(sorted(expected-extras))+' unexpected='+str(sorted(extras-expected)))
    terminals=[]
    # runtime_assembly.configure_springs -> vrm_helpers.register_springs ->
    # locked scripts/spring-setup.py appends one non-deforming _CharakitEnd.
    # Match that upstream validation instead of treating endpoints as deformers.
    for name,parent_name in TAIL_TERMINALS.items():
        bone=rig.data.bones[name]
        if bone.parent is None or bone.parent.name!=parent_name or bone.use_deform:
            raise RuntimeError('Invalid runtime terminal parent/deform state: '+name)
        terminals.append({'bone':name,'parent':bone.parent.name,'useDeform':bool(bone.use_deform),
            'useConnect':bool(bone.use_connect),'source':'tools/charakuru/1.1.0/scripts/spring-setup.py'})
    return extras,terminals


def _rig_differences(expected,actual):
    """Diagnostic only. No rounding/tolerance can authorize a mismatched rig."""
    left={row['name']:row for row in expected['data']['bones']};right={row['name']:row for row in actual['data']['bones']}
    rows=[]
    for name in sorted(set(left)&set(right)):
        before,after=left[name],right[name]
        if before==after:continue
        head_a,head_b=np.asarray(before['head']),np.asarray(after['head'])
        tail_a,tail_b=np.asarray(before['tail']),np.asarray(after['tail'])
        matrix_a,matrix_b=np.asarray(before['matrix']),np.asarray(after['matrix'])
        vector_a,vector_b=tail_a-head_a,tail_b-head_b
        length_a,length_b=float(np.linalg.norm(vector_a)),float(np.linalg.norm(vector_b))
        direction_a=vector_a/length_a if length_a else vector_a
        direction_b=vector_b/length_b if length_b else vector_b
        angle=float(np.degrees(np.arctan2(np.linalg.norm(np.cross(direction_a,direction_b)),np.dot(direction_a,direction_b)))) if length_a and length_b else None
        rows.append({'bone':name,'sourceHead':head_a.tolist(),'runtimeHead':head_b.tolist(),
            'headDeltaMeters':(head_b-head_a).tolist(),'headDeltaLengthMeters':float(np.linalg.norm(head_b-head_a)),
            'sourceTail':tail_a.tolist(),'runtimeTail':tail_b.tolist(),'tailDeltaMeters':(tail_b-tail_a).tolist(),
            'tailDeltaLengthMeters':float(np.linalg.norm(tail_b-tail_a)),
            'sourceDirection':direction_a.tolist(),'runtimeDirection':direction_b.tolist(),'directionAngleDegrees':angle,
            'sourceLengthMeters':length_a,'runtimeLengthMeters':length_b,'lengthDeltaMeters':length_b-length_a,
            'sourceMatrixColumnScale':np.linalg.norm(matrix_a[:3,:3],axis=0).tolist(),
            'runtimeMatrixColumnScale':np.linalg.norm(matrix_b[:3,:3],axis=0).tolist(),
            'maximumMatrixElementDelta':float(np.max(np.abs(matrix_b-matrix_a))),
            'matrixDelta':(matrix_b-matrix_a).tolist(),
            'structuralChanges':{key:{'source':before[key],'runtime':after[key]} for key in
                ('parent','deform','connected','inheritScale','inheritRotation','localLocation') if before[key]!=after[key]}})
    world_a=np.asarray(expected['data']['objectMatrix']);world_b=np.asarray(actual['data']['objectMatrix'])
    return {'sourceSha256':expected['sha256'],'runtimeSha256':actual['sha256'],
        'sourceObjectMatrix':world_a.tolist(),'runtimeObjectMatrix':world_b.tolist(),
        'objectMatrixDelta':(world_b-world_a).tolist(),'maximumObjectMatrixDelta':float(np.max(np.abs(world_b-world_a))),
        'missingBones':sorted(set(left)-set(right)),'extraBonesCompared':sorted(set(right)-set(left)),
        'changedBones':rows,'comparedBoneCount':len(left),'changedBoneCount':len(rows),
        'decision':'Rejected. Diagnostics do not waive the exact rest fingerprint check.'}


def _record_rig_mismatch(stage,expected,actual):
    report=_rig_differences(expected,actual);root=project_path()/'work/runtime-r06'
    for index in range(1,1000):
        path=root/f'{stage}-rig-rest-mismatch-{index:03d}.json'
        if path.exists():continue
        with path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        print('RT08_RIG_REST_MISMATCH',json.dumps({'report':str(path),'changedBones':report['changedBoneCount'],
            'maximumHeadDeltaMeters':max((row['headDeltaLengthMeters'] for row in report['changedBones']),default=0.),
            'maximumTailDeltaMeters':max((row['tailDeltaLengthMeters'] for row in report['changedBones']),default=0.),
            'maximumObjectMatrixDelta':report['maximumObjectMatrixDelta']}),flush=True)
        return path
    raise RuntimeError('Too many prior rig mismatch diagnostic files')


def _plain(value):
    if isinstance(value,(str,bool,int,float)) or value is None:return value
    if isinstance(value,set):return sorted(_plain(v) for v in value)
    try:return [_plain(v) for v in value]
    except TypeError:raise ValueError('Unhandled material parameter: '+str(type(value)))


def _tree_contract(tree,stack=()):
    if tree.as_pointer() in stack:raise RuntimeError('Recursive node group')
    rows=[]
    ignored={'rna_type','name','label','location','location_absolute','width','height','dimensions','select','parent','color',
             'use_custom_color','show_options','show_preview','show_texture','hide','warning_propagation','node_tree','image'}
    for node in sorted(tree.nodes,key=lambda n:n.name):
        props={}
        for prop in node.bl_rna.properties:
            if prop.identifier in ignored or prop.is_readonly or prop.type=='COLLECTION':continue
            value=getattr(node,prop.identifier)
            if prop.type=='POINTER':
                if value is not None:raise RuntimeError('Face payload material depends on external datablock: '+node.name+'/'+prop.identifier)
                continue
            props[prop.identifier]=_plain(value)
        row={'name':node.name,'type':node.bl_idname,'properties':props,
            'inputs':[{'name':s.name,'identifier':s.identifier,'value':_plain(s.default_value) if hasattr(s,'default_value') else None} for s in node.inputs],
            'outputs':[{'name':s.name,'identifier':s.identifier,'value':_plain(s.default_value) if hasattr(s,'default_value') else None} for s in node.outputs]}
        for block,fields in {
            'texture_mapping':('translation','rotation','scale','vector_type','use_min','use_max','min','max','mapping_x','mapping_y','mapping_z'),
            'image_user':('frame_start','frame_duration','frame_offset','use_cyclic','use_auto_refresh'),
            'color_mapping':('blend_type','blend_color','blend_factor','brightness','contrast','saturation','use_color_ramp')}.items():
            value=getattr(node,block,None)
            if value is not None:row[block]={field:_plain(getattr(value,field)) for field in fields if hasattr(value,field)}
        ramp=getattr(node,'color_ramp',None)
        if ramp is not None:row['colorRamp']={'interpolation':ramp.interpolation,'colorMode':ramp.color_mode,
            'hueInterpolation':ramp.hue_interpolation,'elements':[(float(e.position),list(e.color)) for e in ramp.elements]}
        if node.type=='TEX_IMAGE' and node.image:
            image=node.image
            if image.is_dirty:raise RuntimeError('Pack/save modified source image before preparing payload: '+image.name)
            if image.packed_file:pixel_sha=hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
            else:
                path=Path(bpy.path.abspath(image.filepath,library=image.library))
                if not path.is_file():raise RuntimeError('Material image must be packed or saved: '+image.name)
                pixel_sha=_sha(path)
            row['image']={'bytesSha256':pixel_sha,'size':list(image.size),'colorspace':image.colorspace_settings.name,'alphaMode':image.alpha_mode}
        if node.type=='GROUP' and node.node_tree:row['group']=_tree_contract(node.node_tree,stack+(tree.as_pointer(),))
        rows.append(row)
    links=sorted((l.from_node.name,list(l.from_node.outputs).index(l.from_socket),l.to_node.name,list(l.to_node.inputs).index(l.to_socket)) for l in tree.links)
    return {'nodes':rows,'links':links}


def _material_contract(material):
    if not material.use_nodes or material.node_tree is None:raise RuntimeError('Expected node material: '+material.name)
    return _hash({'tree':_tree_contract(material.node_tree),
        'settings':{name:_plain(getattr(material,name)) for name in ('use_backface_culling','surface_render_method','use_transparent_shadow','diffuse_color','metallic','roughness') if hasattr(material,name)}})


def _mesh_contract(obj):
    h=hashlib.sha256();mesh=obj.data
    def add(data,prop,width,dtype=np.float32):
        values=np.empty(len(data)*width,dtype=dtype);data.foreach_get(prop,values);h.update(values.tobytes())
    add(mesh.vertices,'co',3)
    for face in mesh.polygons:h.update(np.asarray([face.material_index,*face.vertices],np.int32).tobytes())
    for layer in mesh.uv_layers:h.update(layer.name.encode());add(layer.data,'uv',2)
    attribute_hash=hashlib.sha256();attribute_api=_api('runtime_torso_cull_r07')
    for attr in mesh.attributes:
        if attr.name in {'position','.edge_verts','.corner_vert','.corner_edge'} or attr.name.startswith(('.select','.vs.','.es.')):continue
        attribute_hash.update(json.dumps([attr.name,attr.domain,attr.data_type]).encode())
        try:values=attribute_api._attribute(attr)
        except (ValueError,TypeError,RuntimeError) as error:
            raise RuntimeError('Unable to preserve attribute on '+obj.name+': '+attr.name+'/'+attr.data_type) from error
        attribute_hash.update(values.tobytes())
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks:
            h.update(json.dumps([key.name,key.value,key.relative_key.name]).encode());add(key.data,'co',3)
    for group in obj.vertex_groups:h.update(json.dumps([group.index,group.name]).encode())
    for vertex in mesh.vertices:
        h.update(np.asarray([(g.group,g.weight) for g in vertex.groups],dtype=np.float64).tobytes())
    return {'vertices':len(mesh.vertices),'faces':len(mesh.polygons),'geometryMorphUVWeightSha256':h.hexdigest(),
            'colorCustomAttributeSha256':attribute_hash.hexdigest(),
            'materials':[m.name for m in mesh.materials],'matrixWorld':[list(row) for row in obj.matrix_world]}


def prepare_payload(source_relative='work/reconstruction-r02/reference-face-flow-r08.blend',stage='r08'):
    """Run with the exact approved high-quality candidate open; no runtime merge."""
    source=_guard();expected=_path(source_relative,'work/reconstruction-r02')
    if source!=expected:raise RuntimeError('Open the explicitly supplied production candidate')
    if not re.fullmatch(r'r0[89][a-z0-9-]*',stage):raise ValueError('Use an explicit r08 or r09 stage name')
    root=project_path()/'work/runtime-r06';root.mkdir(parents=True,exist_ok=True)
    copy=root/(stage+'-face-source.blend');payload=root/(stage+'-face-payload.blend');manifest=root/(stage+'-face-refresh.json')
    archive=root/(stage+'-face-previous-report.json')
    if any(path.exists() for path in (copy,payload,manifest,archive)):raise FileExistsError('Preserve existing face refresh checkpoints; choose another explicit stage name')
    generic=root/'face-snapshot-report.json';previous=generic.read_bytes() if generic.exists() else None
    if previous is not None:archive.write_bytes(previous)
    source_sha=_sha(source);rig=bpy.data.objects[RIG];rig_contract=_rig_contract(rig)
    if any(name in rig.data.bones for name in TAIL_BONES|set(TAIL_TERMINALS)):raise RuntimeError('Prepare from the production candidate, before runtime hair bone creation')
    if rig.animation_data and any(not track.mute for track in rig.animation_data.nla_tracks):raise RuntimeError('Source has active NLA; neutral face sampling requires explicit resolution')
    saved=bpy.ops.wm.save_as_mainfile(filepath=str(copy))
    if 'FINISHED' not in saved:raise RuntimeError('Failed to create independent face source copy')
    detached=[]
    try:
        face=_api('runtime_face_snapshot_r06');snapshot=face.prepare(max_subdivision=1)
        objects=[bpy.data.objects[row['mesh']] for row in snapshot['meshes']]
        if len(objects)!=16:raise RuntimeError('Explicit sixteen-object face scope changed')
        materials={m.name:m for obj in objects for m in obj.data.materials}
        material_rows={name:_material_contract(m) for name,m in materials.items()}
        meshes={obj.name:_mesh_contract(obj) for obj in objects}
        for obj in objects:
            if obj.parent or obj.constraints or obj.animation_data or (obj.data.shape_keys and obj.data.shape_keys.animation_data):
                raise RuntimeError('Sampled face must have no parent/constraints/animation dependency: '+obj.name)
            mods=list(obj.modifiers)
            if len(mods)!=1 or mods[0].type!='ARMATURE' or mods[0].object!=rig:raise RuntimeError('Unexpected sampled skin: '+obj.name)
            detached.append((mods[0],rig));mods[0].object=None
        bpy.data.libraries.write(str(payload),set(objects),path_remap='RELATIVE',fake_user=True,compress=True)
        with bpy.data.libraries.load(str(payload),link=False) as (available,requested):
            if set(available.objects)!={obj.name for obj in objects}:raise RuntimeError('Payload unexpectedly contains object dependencies, possibly another rig')
        if _sha(source)!=source_sha:raise RuntimeError('Production candidate changed on disk')
        result={'schema':1,'stage':stage,'source':{'file':source_relative,'sha256':source_sha},
            'sourceCopy':{'file':str(copy.relative_to(project_path())),'sha256':_sha(copy)},
            'payload':{'file':str(payload.relative_to(project_path())),'sha256':_sha(payload)},
            'rigRest':rig_contract,'meshes':meshes,'materials':material_rows,'snapshot':snapshot,
            'sourceEyeToneCandidate':json.loads(bpy.context.scene.get('R09_eye_tone_candidate','null')),
            'sourceIrisOpticsCandidate':json.loads(bpy.context.scene.get('R09_iris_single_reflection_candidate','null')),
            'rigIncludedInPayload':False,'preparedOnly':True,'appearanceAccepted':False}
        manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result
    finally:
        for modifier,original_rig in detached:modifier.object=original_rig
        if previous is not None:generic.write_bytes(previous)
        elif generic.exists():generic.unlink()


def replace(manifest_relative='work/runtime-r06/r08-face-refresh.json',runtime_relative='work/runtime-r06/r07-material-sock.blend',recull=True):
    """Swap in memory, retain hidden old faces, re-cull ONLY the new unified Head.

    Existing source materials are reused only after exact graph/image/settings
    fingerprint equality. Changed colliding materials stop for explicit handling.
    Export prepare() later rebuilds expression bindings; do not export old binds.
    """
    source=_guard();source.relative_to((project_path()/'work/runtime-r06').resolve())
    if source!=_path(runtime_relative):raise RuntimeError('Open the explicitly supplied existing runtime checkpoint')
    manifest=_path(manifest_relative);spec=json.loads(manifest.read_text(encoding='utf-8'));stage=spec['stage']
    payload=_path(spec['payload']['file'])
    if _sha(payload)!=spec['payload']['sha256']:raise RuntimeError('Face payload hash mismatch')
    rig=bpy.data.objects[RIG];names=[b['name'] for b in spec['rigRest']['data']['bones']]
    actual_rig=_rig_contract(rig,names)
    if actual_rig['sha256']!=spec['rigRest']['sha256']:
        diagnostic=_record_rig_mismatch(stage,spec['rigRest'],actual_rig)
        raise RuntimeError('Production rest-bone or rig object transform fingerprint differs; see '+str(diagnostic))
    extras,terminal_checks=_runtime_extra_bones(rig,names)
    # libraries.load replaces strings with datablocks in the assigned list.
    # Retain immutable names and hand Blender a separate mutable list below.
    expected=tuple(spec['meshes']);old=[bpy.data.objects[name] for name in expected]
    if len(old)!=16 or any(not obj.name.startswith(PREFIX) or obj.type!='MESH' for obj in old):raise RuntimeError('Expected sixteen current runtime faces')
    archive_prefix=stage.upper()+' retained / '
    if any(obj.name.startswith(archive_prefix) for obj in bpy.data.objects):raise RuntimeError('Refresh already applied')
    for name,sha in spec['materials'].items():
        present=bpy.data.materials.get(name)
        if present and _material_contract(present)!=sha:raise RuntimeError('Source material changed; explicit mapping required before merge: '+name)
    assembly=_api('runtime_assembly_r06');_,all_meshes=assembly.current_meshes();retained=[obj for obj in all_meshes if obj not in old]
    retained_state={obj.name:(obj.as_pointer(),obj.data.as_pointer(),tuple(m.as_pointer() for m in obj.data.materials),_mesh_contract(obj)) for obj in retained}
    rig_ids={obj.as_pointer() for obj in bpy.data.objects if obj.type=='ARMATURE'}
    for attempt in range(1,1000):
        suffix='' if attempt==1 else f'-attempt-{attempt:03d}'
        out=_path('work/runtime-r06/'+stage+'-face-merge'+suffix)
        try:out.mkdir(parents=True,exist_ok=False);break
        except FileExistsError:continue
    else:raise RuntimeError('Too many retained face merge attempt folders')
    merge_report_relative=str((out/'merge-report.json').relative_to(project_path()))
    report_names=('face-snapshot-report.json','material-stage-plan.json','r07-torso-cull-audit.json','r07-torso-cull-candidate.json')
    previous={name:(out.parent/name).read_bytes() if (out.parent/name).is_file() else None for name in report_names}
    for name,data in previous.items():
        if data is not None:(out/('before-'+name)).write_bytes(data)
    old_states=[(obj,obj.name,obj.hide_get(),obj.hide_render,obj.hide_viewport) for obj in old]
    imported=[];complete=False
    try:
        with bpy.data.libraries.load(str(payload),link=False) as (available,requested):requested.objects=list(expected)
        imported=list(requested.objects)
        if any(obj is None for obj in imported) or len(imported)!=16:raise RuntimeError('Face import incomplete')
        if {obj.as_pointer() for obj in bpy.data.objects if obj.type=='ARMATURE'}!=rig_ids:raise RuntimeError('Payload appended a duplicate rig')
        collection=bpy.data.collections.new(stage.upper()+' refreshed face');bpy.context.scene.collection.children.link(collection)
        for obj,name in zip(imported,expected):
            collection.objects.link(obj)
            # Library names gain numeric suffixes on collision. Compare payload
            # geometry independently, then map materials to verified originals.
            actual=_mesh_contract(obj);actual['materials']=spec['meshes'][name]['materials']
            if actual!=spec['meshes'][name]:raise RuntimeError('Imported mesh/morph/UV/weights changed: '+name)
            for index,material_name in enumerate(spec['meshes'][name]['materials']):
                incoming=obj.data.materials[index]
                if _material_contract(incoming)!=spec['materials'][material_name]:raise RuntimeError('Imported material changed: '+material_name)
                existing=bpy.data.materials.get(material_name)
                if existing:obj.data.materials[index]=existing
                else:incoming.name=material_name
            if obj.parent or len(obj.modifiers)!=1 or obj.modifiers[0].type!='ARMATURE' or obj.modifiers[0].object:
                raise RuntimeError('Payload retained an object/skin dependency')
            for vertex in obj.data.vertices:
                if any(obj.vertex_groups[g.group].name not in rig.data.bones for g in vertex.groups):raise RuntimeError('Skin weight references unavailable bone')
            matrix=obj.matrix_world.copy();obj.parent=rig;obj.matrix_parent_inverse=rig.matrix_world.inverted();obj.matrix_world=matrix
            obj.modifiers[0].object=rig;obj['RT08_refresh_source']=spec['source']['file'];obj['RT08_refresh_payload_sha256']=spec['payload']['sha256']
        for obj,name,_,_,_ in old_states:
            obj.name=archive_prefix+name[len(PREFIX):];obj.hide_render=True;obj.hide_set(True);obj.hide_viewport=True
        for obj,name in zip(imported,expected):obj.name=name;obj.hide_render=False;obj.hide_viewport=False;obj.hide_set(False)
        cull_report=None
        if recull:
            cull=_api('runtime_torso_cull_r07');cull.audit();cull_report=cull.apply_candidate()
        for obj in retained:
            actual=(obj.as_pointer(),obj.data.as_pointer(),tuple(m.as_pointer() for m in obj.data.materials),_mesh_contract(obj))
            if actual!=retained_state[obj.name]:raise RuntimeError('A retained non-face runtime object changed: '+obj.name)
        snapshot=spec['snapshot'];snapshot['runtimeCopy']=bpy.data.filepath;snapshot['refreshSource']=spec['source'];snapshot['refreshPayload']=spec['payload']
        (out.parent/'face-snapshot-report.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
        if previous['material-stage-plan.json'] is not None:
            plan=json.loads(previous['material-stage-plan.json']);head=bpy.data.objects[PREFIX+'R01 Head']
            slots=[i for i,m in enumerate(head.data.materials) if m.name=='R04 physical skin']
            if len(slots)!=1:raise RuntimeError('Refreshed skin domain missing')
            plan['skin']['slots']=slots;plan['source']=bpy.data.filepath;plan['faceRefreshSource']=spec['source']
            plan['faceRefreshEyeToneCandidate']=spec.get('sourceEyeToneCandidate')
            plan['faceRefreshIrisOpticsCandidate']=spec.get('sourceIrisOpticsCandidate')
            plan['explicitMeshes']=[obj.name for obj in assembly.current_meshes()[1]];plan['explicitMeshCount']=len(plan['explicitMeshes'])
            material_api=_api('runtime_material_stage_r06');plan['shared']['iris']=material_api._iris_stage([bpy.data.objects[n] for n in plan['explicitMeshes']],material_api._api())
            plan['refreshNotes']='Preserved completed non-face stages; only skin and current iris revalidated. Do not repeat full plan() after shared material baking.'
            (out.parent/'material-stage-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        result={'sourceRuntime':str(source),'sourceRuntimeSha256':_sha(source),'faceRefreshManifest':manifest_relative,
            'mergeAttempt':attempt,'mergeReport':merge_report_relative,
            'productionSource':spec['source'],'payload':spec['payload'],'rigRestFingerprintMatched':True,
            'sourceEyeToneCandidate':spec.get('sourceEyeToneCandidate'),
            'sourceIrisOpticsCandidate':spec.get('sourceIrisOpticsCandidate'),
            'runtimeExtraBones':sorted(extras),'validatedRuntimeTerminals':terminal_checks,
            'duplicateRigAdded':False,'newFaces':[obj.name for obj in imported],
            'retainedOldFaces':[obj.name for obj in old],
            'oldFaceStates':[{'archiveName':obj.name,'originalName':name,'hideSet':hidden,'hideRender':hide_render,'hideViewport':hide_viewport}
                for obj,name,hidden,hide_render,hide_viewport in old_states],
            'nonFaceObjectsUnchanged':len(retained),'torsoCull':cull_report,
            'expressionBindingsNeedRefresh':True,'nextSteps':['Save a NEW merged checkpoint','Review face/collar/wrist','Run skin bake using refreshed existing plan','Export prepare() must rebuild current morph bindings'],
            'savedBlend':False,'appearanceAccepted':False}
        (out/'merge-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        rig['RT08_face_refresh_manifest']=manifest_relative;rig['RT08_face_refresh_merge_report']=merge_report_relative
        complete=True;return result
    finally:
        if not complete:
            for obj in imported:
                if obj and obj.name in bpy.data.objects:bpy.data.objects.remove(obj,do_unlink=True)
            for obj,name,hidden,hide_render,hide_viewport in old_states:
                obj.name=name;obj.hide_viewport=hide_viewport;obj.hide_render=hide_render;obj.hide_set(hidden)
            for name,data in previous.items():
                target=out.parent/name
                if data is not None:target.write_bytes(data)
                elif target.exists():target.unlink()


def restore(manifest_relative='work/runtime-r06/r08-face-refresh.json',merge_report_relative=None):
    """Restore old sixteen faces/report aliases in memory; keep new faces hidden."""
    source=_guard();source.relative_to((project_path()/'work/runtime-r06').resolve())
    manifest=_path(manifest_relative);spec=json.loads(manifest.read_text(encoding='utf-8'));stage=spec['stage']
    rig=bpy.data.objects[RIG]
    if rig.get('RT08_face_refresh_manifest')!=manifest_relative:raise RuntimeError('This runtime does not have the stated active face refresh')
    recorded=rig.get('RT08_face_refresh_merge_report')
    if merge_report_relative is not None and recorded and merge_report_relative!=recorded:
        raise RuntimeError('Explicit rollback report differs from the successful merge recorded in this rig')
    report_relative=merge_report_relative or recorded
    if not report_relative:raise RuntimeError('Supply the actual successful merge report path; never infer it from failed attempt folders')
    report_path=_path(report_relative);folder=report_path.parent;report=json.loads(report_path.read_text(encoding='utf-8'))
    if report['faceRefreshManifest']!=manifest_relative:raise RuntimeError('Rollback report belongs to another face refresh')
    current=[bpy.data.objects[name] for name in report['newFaces']]
    previous=[bpy.data.objects[row['archiveName']] for row in report['oldFaceStates']]
    reverted_prefix=stage.upper()+' reverted / '
    if any(obj.name.startswith(reverted_prefix) for obj in bpy.data.objects):raise RuntimeError('A reverted face set already exists')
    for obj in current:
        obj.name=reverted_prefix+obj.name[len(PREFIX):];obj.hide_render=True;obj.hide_set(True);obj.hide_viewport=True
    for obj,row in zip(previous,report['oldFaceStates']):
        obj.name=row['originalName'];obj.hide_viewport=row['hideViewport'];obj.hide_render=row['hideRender'];obj.hide_set(row['hideSet'])
    for filename in ('face-snapshot-report.json','material-stage-plan.json','r07-torso-cull-audit.json','r07-torso-cull-candidate.json'):
        backup=folder/('before-'+filename);alias=folder.parent/filename
        if backup.is_file():alias.write_bytes(backup.read_bytes())
        elif alias.exists():alias.unlink()
    del rig['RT08_face_refresh_manifest']
    if 'RT08_face_refresh_merge_report' in rig:del rig['RT08_face_refresh_merge_report']
    result={'restoredFaces':[obj.name for obj in previous],'newFacesRetainedHidden':[obj.name for obj in current],
        'savedBlend':False,'expressionBindingsNeedRefresh':True}
    (folder/'restore-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');return result


if __name__=='__main__':print('Loaded only. prepare_payload(source_relative) on production candidate, then replace() on saved runtime copy. Never automatically renders or exports.')
