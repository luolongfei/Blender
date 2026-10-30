"""Explicit prepare/export stages for an R06 prototype PBR VRM 1.0 and GLB.

No work on import, no Blend saves, no MToon conversion, no production exports.
Run only in a saved disposable background copy under work/runtime-r06/.
prepare(source_lineage=[...], require_baked=False) is useful before baking.
After baking/review, prepare(..., require_baked=True), then export().
Source lineage paths are project-relative existing files chosen by integrator.

VRM Add-on 4.10.2 keeps ordinary Principled glTF materials unless MToon/MMD/
legacy-shader tags apply. We reject those tags and inspect actual outputs.
The kit's inspect_vrm() requires all-MToon, so it is deliberately not called.
"""
import hashlib,importlib.util,json,math,struct,sys,tempfile
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PREP='work/runtime-r06/export-preparation.json'
REQUIRED_CUSTOM_EXPRESSIONS={'mbp','pucker','squintLeft','squintRight'}
LOSSES=[
 'PBR preview keeps base color/alpha, metallic-roughness and tangent normal; no MToon conversion is performed.',
 'Source Cycles SSS, layered/hair scattering, cloth caches and full facial driver graphs are not encoded in these PBR atlases.',
 'Hair card atlases represent strand coverage (analytic or native-path projected, identified per material), not native strand-shader lighting.',
 'KHR_materials_clearcoat retains coat weight/roughness, but fixes coat IOR at 1.5; a different Blender Coat IOR is recorded as an unresolved shading difference, without silently altering the base-layer IOR.',
 'Four bone influences are exported for viewer compatibility; discarded source influence sums are reported.',
 'GLB contains rig and morph targets but no VRM expression/LookAt/spring semantics; use VRM for those semantics.',
 'JSON/texture checks do not establish appearance, spring stability, viewer compatibility or frame rate.'
]

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def _json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def _spring_numeric_contract(rig):
    """Installed 4.10.2 property values in their actual VRM export coordinates."""
    sb=rig.data.vrm_addon_extension.spring_bone1;colliders={}
    for c in sb.colliders:
        if c.extensions.vrmc_spring_bone_extended_collider.enabled:raise ValueError('Extended collider requires separate contract support')
        kind='capsule' if c.shape_type==c.SHAPE_TYPE_CAPSULE.identifier else 'sphere'
        shape=getattr(c.shape,kind);data={'offset':list(shape.offset),'radius':float(shape.radius)}
        if kind=='capsule':data['tail']=list(shape.tail)
        colliders[c.uuid]={'bone':c.node.bone_name,'shape':{kind:data}}
    groups={g.vrm_name:[colliders[r.collider_uuid] for r in g.colliders] for g in sb.collider_groups}
    springs={s.vrm_name:{'center':s.center.bone_name or None,'joints':[{'bone':j.node.bone_name,
        'hitRadius':float(j.hit_radius),'stiffness':float(j.stiffness),'gravityPower':float(j.gravity_power),
        'gravityDir':[float(j.gravity_dir[0]),float(j.gravity_dir[2]),float(-j.gravity_dir[1])],
        'dragForce':float(j.drag_force)} for j in s.joints]} for s in sb.springs}
    return {'groups':groups,'springs':springs}


def _inspect_spring_numeric(spring,contract,nodes):
    def same(actual,expected,label):
        if isinstance(expected,dict):
            if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Spring shape/fields changed: '+label)
            for key in expected:same(actual[key],expected[key],label+'/'+key)
        elif isinstance(expected,list):
            if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Spring list length changed: '+label)
            for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+'/'+str(i))
        elif isinstance(expected,(int,float)):
            if not isinstance(actual,(int,float)) or not math.isfinite(actual) or abs(actual-expected)>2e-6:raise ValueError('Spring numeric value changed: '+label)
        elif actual!=expected:raise ValueError('Spring bone/reference changed: '+label)
    actual_groups={}
    for group in spring.get('colliderGroups',[]):
        rows=[]
        for index in group.get('colliders',[]):
            c=_at(spring.get('colliders',[]),index,'collider');node=_at(nodes,c.get('node'),'collider bone')
            rows.append({'bone':node.get('name'),'shape':c.get('shape')})
        actual_groups[group.get('name')]=rows
    actual_springs={s.get('name'):{'center':_at(nodes,s['center'],'spring center').get('name') if 'center' in s else None,
        'joints':[{'bone':_at(nodes,j.get('node'),'spring joint').get('name'),'hitRadius':j.get('hitRadius',0),
            'stiffness':j.get('stiffness',1),'gravityPower':j.get('gravityPower',0),'gravityDir':j.get('gravityDir',[0,-1,0]),
            'dragForce':j.get('dragForce',.5)} for j in s.get('joints',[])]} for s in spring.get('springs',[])}
    same({'groups':actual_groups,'springs':actual_springs},contract,'VRMC_springBone')

def _module(name):
    s=importlib.util.spec_from_file_location('r06_delivery_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Saved disposable background copy required')
    root=(project_path()/'work/runtime-r06').resolve();source=Path(bpy.data.filepath).resolve()
    source.relative_to(root)
    if not source.is_file() or bpy.context.mode!='OBJECT':raise RuntimeError('Saved runtime copy in Object Mode required')
    return root,source

def _within(relative,root):
    path=(project_path()/relative).resolve();path.relative_to(root.resolve());return path

def _image_source(node):
    image=node.image
    if image is None or min(image.size)<1:raise ValueError('Missing image data: '+node.name)
    packed=image.packed_file
    external=Path(bpy.path.abspath(image.filepath)).resolve() if image.filepath else None
    if packed:
        raw=bytes(packed.data);digest=hashlib.sha256(raw).hexdigest();storage='packed'
    elif external and external.is_file():digest=_sha(external);storage=str(external)
    else:raise ValueError('Image is neither packed nor saved: '+image.name)
    return {'image':image.name,'size':list(image.size),'colorSpace':image.colorspace_settings.name,
            'sha256':digest,'storage':storage}

def _linked_image(socket,output):
    if len(socket.links)!=1:raise ValueError('Expected one direct image connection: '+socket.name)
    link=socket.links[0]
    if link.from_node.type!='TEX_IMAGE' or link.from_socket.name!=output:raise ValueError('Only direct baked image channels are supported: '+socket.name)
    return link.from_node

def _material_contract(material):
    ext=getattr(material,'vrm_addon_extension',None)
    if ext and ext.mtoon1.enabled:raise ValueError('MToon material would override PBR channels: '+material.name)
    if not material.use_nodes:raise ValueError('PBR node material required')
    output=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    if output is None or len(output.inputs['Surface'].links)!=1:raise ValueError('One direct surface required')
    shader=output.inputs['Surface'].links[0].from_node
    if shader.type!='BSDF_PRINCIPLED':raise ValueError('Bake the non-Principled source first: '+material.name)
    if output.inputs['Displacement'].is_linked:raise ValueError('Unbaked displacement: '+material.name)
    base_socket=shader.inputs['Base Color'];textures={};nodes=[];base=None;base_constant=None
    if base_socket.is_linked:
        base=_linked_image(base_socket,'Color');textures['baseColor']=_image_source(base);nodes.append(base)
    else:base_constant=list(base_socket.default_value)
    rough=shader.inputs['Roughness']
    if rough.is_linked:
        image=_linked_image(rough,'Color');textures['roughness']=_image_source(image);nodes.append(image);rough_constant=None
    else:rough_constant=float(rough.default_value)
    normal=shader.inputs['Normal'];normal_strength=None
    if normal.is_linked:
        n=normal.links[0].from_node
        if n.type!='NORMAL_MAP' or n.space!='TANGENT' or n.inputs['Strength'].is_linked:raise ValueError('Expected baked tangent NormalMap: '+material.name)
        image=_linked_image(n.inputs['Color'],'Color');textures['normal']=_image_source(image);nodes.append(image);normal_strength=float(n.inputs['Strength'].default_value)
    alpha=shader.inputs['Alpha'];transparent=alpha.is_linked or float(alpha.default_value)<.999
    if alpha.is_linked:
        alpha_node=_linked_image(alpha,'Alpha')
        if base is None or alpha_node.image!=base.image:raise ValueError('Alpha must be packed into base-color atlas: '+material.name)
    elif transparent:raise ValueError('Non-unit constant alpha needs explicit texture/factor handling: '+material.name)
    for name in ('Transmission Weight',):
        socket=shader.inputs.get(name)
        if socket and (socket.is_linked or abs(float(socket.default_value))>1e-7):raise ValueError('Unhandled physical input '+name+': '+material.name)
    constants={}
    for name in ('Metallic','Coat Weight','Coat Roughness','Coat IOR','IOR','Specular IOR Level','Subsurface Weight'):
        socket=shader.inputs.get(name)
        if socket and socket.is_linked:raise ValueError('Unhandled linked PBR input '+name+': '+material.name)
        constants[name]=float(socket.default_value) if socket else 0.
    if shader.inputs['Coat Normal'].is_linked:raise ValueError('Coat normal needs explicit mapping')
    uv_names=set()
    for node in nodes:
        links=node.inputs['Vector'].links
        if not links:uv_names.add('ACTIVE_RENDER');continue
        if len(links)!=1 or links[0].from_node.type!='UVMAP' or links[0].from_socket.name!='UV':raise ValueError('Explicit untransformed runtime UV map required')
        uv_names.add(links[0].from_node.uv_map or 'ACTIVE_RENDER')
    if len(uv_names)>1:raise ValueError('Mixed UV channels require explicit adaptation: '+material.name)
    return {'name':material.name,'textures':textures,'alpha':transparent,'roughnessConstant':rough_constant,
            'baseColorConstant':base_constant,'constants':constants,
            'sourceSSSNotPreserved':constants['Subsurface Weight']>0,
            'sourceCoatIORNotPreserved':constants['Coat Weight']>0 and abs(constants['Coat IOR']-1.5)>1e-6,
            'normalStrength':normal_strength,'uvMap':next(iter(uv_names),None),'shading':'glTF metallic-roughness PBR',
            'analyticHairPreview':bool(material.get('RT06_analytic_preview_material')),
            'protectedAlphaAtlas':bool(material.get('r06_preserve_alpha_atlas')),
            'sourceProjectedHairAtlas':bool(material.get('RT06_source_projected_atlas')),
            'hairRegion':material.get('RT06_hair_region'),
            'sourceRecord':material.get('source_record')}

def _mesh_contract(obj,rig):
    h=hashlib.sha256();mesh=obj.data
    xyz=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',xyz);h.update(xyz.tobytes())
    loops=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get('vertex_index',loops);h.update(loops.tobytes())
    keys=[];weights=[]
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks[1:]:
            key.data.foreach_get('co',xyz);h.update(key.name.encode());h.update(xyz.tobytes());keys.append(key.name);weights.append(float(key.value))
    deform={g.index for g in obj.vertex_groups if g.name in rig.data.bones and rig.data.bones[g.name].use_deform}
    discarded=max((sum(sorted((g.weight for g in v.groups if g.group in deform),reverse=True)[4:]) for v in mesh.vertices),default=0.)
    active_uv=next((u.name for u in mesh.uv_layers if u.active_render),None)
    return {'name':obj.name,'vertices':len(mesh.vertices),'faces':len(mesh.polygons),'geometryAndMorphSha256':h.hexdigest(),
            'shapeKeys':keys,'defaultMorphWeights':weights,'activeRenderUV':active_uv,'materials':[m.name for m in mesh.materials],
            'maxDiscardedWeightAfterFourInfluences':discarded,
            'sourceObject':obj.get('RT06_source_object',obj.name)}

def read_vrm_semantics(rig):
    """Read saved 4.10.2 bindings/LookAt without fixing them or writing reports."""
    face=_module('runtime_face_snapshot_r06');expressions=rig.data.vrm_addon_extension.vrm1.expressions
    presets=dict(expressions.preset.name_to_expression_dict());custom={}
    for item in expressions.custom:
        if item.custom_name in presets or item.custom_name in custom:raise ValueError('Duplicate expression name')
        custom[item.custom_name]=item
    mapping={**presets,**custom};required=set(face.DEFAULT_STATES)
    if required-set(mapping):raise ValueError('Saved expression is missing: '+repr(sorted(required-set(mapping))))
    bound={};custom_bound={};settings={};previews={}
    for name,e in mapping.items():
        previews[name]=float(e.preview)
        if abs(float(e.preview))>1e-8:raise ValueError('Saved expression preview must be neutral: '+name)
        if e.material_color_binds or e.texture_transform_binds:raise ValueError('Unexpected non-morph expression binding: '+name)
        if name not in required:
            if e.morph_target_binds:raise ValueError('Unexpected additional bound expression: '+name)
            continue
        rows=[]
        for bind in e.morph_target_binds:
            obj=bpy.data.objects.get(bind.node.mesh_object_name)
            if obj is None or obj.type!='MESH' or not obj.data.shape_keys or bind.index not in obj.data.shape_keys.key_blocks:
                raise ValueError('Saved expression points to a missing mesh/key: '+name)
            if abs(float(bind.weight)-1.)>1e-8:raise ValueError('Saved expression weight differs from sampled unit binding: '+name)
            rows.append({'mesh':obj.name,'key':bind.index})
        if not rows:raise ValueError('Saved required expression has no binding: '+name)
        (custom_bound if name in face.CUSTOM_EXPRESSIONS else bound)[name]=rows
        settings[name]={'isBinary':bool(e.is_binary),'overrideBlink':e.override_blink,
            'overrideMouth':e.override_mouth,'overrideLookAt':e.override_look_at}
    if set(custom)!=set(face.CUSTOM_EXPRESSIONS):raise ValueError('Saved custom expression scope differs')
    look=rig.data.vrm_addon_extension.vrm1.look_at
    if look.type!='expression' or look.enable_preview:raise ValueError('Expected expression LookAt with preview disabled')
    look_values={'type':look.type,'offsetFromHeadBone':list(look.offset_from_head_bone),'ranges':{}}
    for attr in ('range_map_horizontal_inner','range_map_horizontal_outer','range_map_vertical_down','range_map_vertical_up'):
        item=getattr(look,attr);look_values['ranges'][attr]={'inputMaxValue':float(item.input_max_value),'outputScale':float(item.output_scale)}
    human=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    human_values={str(name.value):item.node.bone_name for name,item in human.human_bone_name_to_human_bone().items()}
    return {'bindings':{'presets':bound,'custom':custom_bound,'settings':settings,'unboundAdditionalKeys':[]},
            'expressionPreviewValues':previews,'lookAt':look_values,'humanoid':human_values,
            'automaticExpressionAssignment':bool(expressions.initial_automatic_expression_assignment)}


def prepare(source_lineage,require_baked=True,write=True,configure_expressions=True):
    """Create explicit scene contract. Does not bake, export or save a Blend.

    Default explicitly configures expressions. Set configure_expressions=False
    and write=False after a fresh reopen for read-only internal export checks.
    """
    root,source=_guard();assembly=_module('runtime_assembly_r06')
    if configure_expressions:inventory=assembly.inventory()
    else:
        rig,objects=assembly.current_meshes()
        inventory={'rig':rig.name,'meshes':[o.name for o in objects],
            'bindings':_module('vrm_helpers').audit_bindings(rig,objects)}
    rig=bpy.data.objects[inventory['rig']];meshes=[bpy.data.objects[n] for n in inventory['meshes']]
    bad=[x for x in inventory['bindings'] if x['unweightedVertices'] or x['nonNormalizedVertices'] or x['armatureModifiers']!=1]
    if bad:raise ValueError('Runtime binding defects: '+json.dumps(bad))
    if configure_expressions:_module('vrm_helpers')._enabled()
    if rig.data.vrm_addon_extension.spec_version!='1.0':raise ValueError('Expected explicitly mapped VRM 1.0 rig')
    human=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    if configure_expressions:type(human).update_all_bone_name_candidates(bpy.context,rig.data.name,force=True)
    if list(human.error_messages()):raise ValueError('Humanoid mapping errors: '+str(list(human.error_messages())))
    bindings=(_module('runtime_face_snapshot_r06').configure_vrm() if configure_expressions else read_vrm_semantics(rig)['bindings'])
    if set(bindings.get('custom',{}))!=REQUIRED_CUSTOM_EXPRESSIONS or bindings['unboundAdditionalKeys']:
        raise ValueError('Four required custom expressions must all be explicitly bound')
    all_bindings={**bindings['presets'],**bindings['custom']}
    expression_kinds={name:kind for kind,key in (('preset','presets'),('custom','custom')) for name in bindings[key]}
    empty=[name for name,rows in all_bindings.items() if not rows]
    if empty:raise ValueError('Empty required expression bindings: '+repr(empty))
    unique={m.name:m for o in meshes for m in o.data.materials};materials=[];blockers=[]
    for obj in meshes:
        if obj.get('RT09_root_refresh_manifest') and (obj.get('RT09_root_bake_required',True) or
            obj.get('RT09_root_baked_geometry_source_sha256')!=obj.get('RT09_root_geometry_source_sha256')):
            blockers.append({'mesh':obj.name,'reason':'Refreshed PN root requires a new actual bake tied to its geometry source; old root normal/alpha maps are not accepted'})
    for name,m in unique.items():
        try:materials.append(_material_contract(m))
        except ValueError as exc:blockers.append({'material':name,'reason':str(exc)})
    projected=[m for m in materials if m['sourceProjectedHairAtlas']]
    analytic=[m for m in materials if m['analyticHairPreview']]
    regions={m['hairRegion'] for m in projected}
    if projected and (regions!={'scalp','gather','fringe','wisps','tail'} or len(projected)!=5):
        blockers.append({'reason':'Expected all five distinct native-path projected hair region materials','actualRegions':sorted(str(r) for r in regions)})
    if any(not m['protectedAlphaAtlas'] or not m['alpha'] for m in projected):
        blockers.append({'reason':'Projected hair region must retain protected base-color RGBA alpha connection'})
    hair_representation={'method':'native-path projected regional coverage' if projected else 'legacy analytic coverage',
        'regions':sorted(str(r) for r in regions),'regionMaterialCount':len(projected),'legacyAnalyticMaterialCount':len(analytic),
        'appearanceStatus':('Five-region projected atlas requires its own actual render and VRM alpha review; no automatic acceptance.' if projected else
            'Legacy shared analytic hair-card render failed likeness (sparse fringe, dark cap and gather loops); not deliverable.')}
    mesh_rows=[_mesh_contract(o,rig) for o in meshes]
    by_name={m['name']:m for m in materials}
    for row in mesh_rows:
        for name in row['materials']:
            if name not in by_name:continue
            required=by_name[name]['uvMap']
            if required is not None and (not row['activeRenderUV'] or required not in ('ACTIVE_RENDER',row['activeRenderUV'])):
                blockers.append({'mesh':row['name'],'material':name,'reason':'Active render UV differs from material UV; VRM addon strips inactive UV maps'})
    if require_baked and blockers:raise ValueError('Material adaptation incomplete: '+json.dumps(blockers,ensure_ascii=False))
    if not isinstance(source_lineage,list) or not source_lineage:raise ValueError('Provide explicit nonempty source lineage paths')
    lineage=[]
    for relative in source_lineage:
        path=_within(relative,project_path())
        if not path.is_file():raise FileNotFoundError(path)
        lineage.append({'file':str(path.relative_to(project_path())),'sha256':_sha(path),'bytes':path.stat().st_size})
    spring=rig.data.vrm_addon_extension.spring_bone1
    spring_names=[s.vrm_name for s in spring.springs]
    if len([n for n in spring_names if n.startswith('RT06')])<2:raise ValueError('Both current runtime tail springs are required')
    group_names={g.uuid:g.vrm_name for g in spring.collider_groups}
    spring_groups={}
    for item in spring.springs:
        refs=[group_names.get(ref.collider_group_uuid) for ref in item.collider_groups]
        if any(name is None for name in refs) or len(set(refs))!=len(refs):raise ValueError('Unresolved/duplicate spring collider groups: '+item.vrm_name)
        spring_groups[item.vrm_name]=refs
    required_tail_groups={'Character body','Shoulders · ponytails','Back garment · ponytails'}
    for name in ('RT06 tail L','RT06 tail R'):
        if not required_tail_groups.issubset(spring_groups.get(name,[])):raise ValueError('Runtime tail lacks existing shoulder/back collision groups: '+name)
    report={'schema':2,'status':'ready for prototype export' if not blockers else 'materials need adaptation',
        'savedRuntimeCopy':str(source.relative_to(project_path())),'savedRuntimeSha256':_sha(source),'sourceLineage':lineage,
        'rig':rig.name,'meshes':mesh_rows,'materials':materials,'materialBlockers':blockers,
        'expressionBindings':all_bindings,'expressionKinds':expression_kinds,'expressionSettings':bindings['settings'],
        'expressionBindingCount':sum(map(len,all_bindings.values())),
        'expressionPresetCount':len(bindings['presets']),'expressionCustomCount':len(bindings['custom']),
        'additionalUnboundKeys':bindings['unboundAdditionalKeys'],'springNames':spring_names,'springColliderGroups':spring_groups,
        'springNumericContract':_spring_numeric_contract(rig),'hairRepresentation':hair_representation,
        'blender':bpy.app.version_string,'toolchainLockSha256':_sha(ROOT/'toolchain.lock.json'),
        'scriptSha256':_sha(__file__),'losses':LOSSES,'productionFilesChanged':False,'blendSaved':False}
    fingerprint={k:report[k] for k in ('rig','meshes','materials','expressionBindings','expressionKinds','expressionSettings','springNames','springColliderGroups','springNumericContract')}
    report['sceneContractSha256']=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if write:_json(project_path()/PREP,report)
    return report

def read_container(path):
    """Pure parser; reject external images/buffers and malformed binary ranges."""
    raw=Path(path).read_bytes()
    if len(raw)<20 or struct.unpack_from('<III',raw)!=(0x46546C67,2,len(raw)):raise ValueError('Invalid GLB v2 container')
    offset=12;document=None;binary=None
    while offset<len(raw):
        if offset+8>len(raw):raise ValueError('Truncated chunk header')
        length,kind=struct.unpack_from('<II',raw,offset);start=offset+8
        if length%4 or start+length>len(raw):raise ValueError('Invalid chunk length/alignment')
        if kind==0x4E4F534A and document is None and offset==12:document=json.loads(raw[start:start+length])
        elif kind==0x004E4942 and binary is None:binary=raw[start:start+length]
        else:raise ValueError('Unexpected/duplicate container chunk')
        offset=start+length
    if not isinstance(document,dict) or document.get('asset',{}).get('version')!='2.0' or binary is None:raise ValueError('Missing glTF2/BIN data')
    buffers=document.get('buffers',[])
    if len(buffers)!=1 or 'uri' in buffers[0] or not 0<=len(binary)-buffers[0].get('byteLength',-1)<=3:raise ValueError('Nonembedded/inconsistent buffer')
    for view in document.get('bufferViews',[]):
        start=view.get('byteOffset',0);length=view['byteLength']
        if view.get('buffer')!=0 or start<0 or length<0 or start+length>buffers[0]['byteLength']:raise ValueError('Buffer view exceeds embedded data')
    widths={5120:1,5121:1,5122:2,5123:2,5125:4,5126:4};components={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT2':4,'MAT3':9,'MAT4':16}
    for accessor in document.get('accessors',[]):
        if accessor.get('sparse'):raise ValueError('Dense prototype accessors required; disable sparse morph export')
        width=widths[accessor['componentType']];size=components[accessor['type']];element=width*size
        if 'bufferView' not in accessor:
            if accessor.get('count',0)<1 or accessor.get('byteOffset',0)!=0:raise ValueError('Invalid implicit-zero accessor')
            continue
        view=_at(document.get('bufferViews',[]),accessor.get('bufferView'),'accessor bufferView')
        stride=view.get('byteStride',element);count=accessor['count'];offset=accessor.get('byteOffset',0)
        if count<1 or offset<0 or stride<element or offset+(count-1)*stride+element>view['byteLength']:raise ValueError('Accessor exceeds buffer view')
        if accessor['componentType']==5126:
            values=np.ndarray((count,size),dtype='<f4',buffer=binary,offset=view.get('byteOffset',0)+offset,strides=(stride,width))
            if not np.isfinite(values).all():raise ValueError('Nonfinite exported accessor values')
    return document,binary

def _at(values,index,label):
    if type(index)is not int or not 0<=index<len(values):raise ValueError('Invalid '+label+' reference: '+repr(index))
    return values[index]

def _image_bytes(document,binary,index):
    image=_at(document.get('images',[]),index,'image')
    if 'uri' in image or 'bufferView' not in image:raise ValueError('Image is not embedded')
    view=_at(document.get('bufferViews',[]),image['bufferView'],'image bufferView');start=view.get('byteOffset',0)
    raw=binary[start:start+view['byteLength']]
    if image.get('mimeType')!='image/png' or raw[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Lossless PNG required for preview channel checks')
    if len(raw)<24:raise ValueError('Truncated PNG')
    return raw,list(struct.unpack_from('>II',raw,16))

def _texture(document,binary,info):
    texture=_at(document.get('textures',[]),info.get('index'),'texture')
    raw,size=_image_bytes(document,binary,texture.get('source'))
    return {'imageIndex':texture['source'],'size':size,'sha256':hashlib.sha256(raw).hexdigest(),
            'texCoord':info.get('extensions',{}).get('KHR_texture_transform',{}).get('texCoord',info.get('texCoord',0))}

def _inspect_vrm_expressions(vrm,contract,nodes,shape_lookup):
    groups=vrm.get('expressions',{});kinds=contract.get('expressionKinds',{})
    required_custom={name for name,kind in kinds.items() if kind=='custom'}
    if required_custom!=REQUIRED_CUSTOM_EXPRESSIONS:raise ValueError('Contract lacks the four required custom expressions')
    if set(kinds)!=set(contract['expressionBindings']):raise ValueError('Expression kind coverage differs from binding contract')
    rows={}
    for name,required in contract['expressionBindings'].items():
        kind=kinds[name]
        if kind not in {'preset','custom'}:raise ValueError('Unknown expression kind: '+name)
        if name in groups.get('custom' if kind=='preset' else 'preset',{}):raise ValueError('Expression exported in wrong/duplicate group: '+name)
        expression=groups.get(kind,{}).get(name)
        if not isinstance(expression,dict):raise ValueError('Missing '+kind+' expression: '+name)
        actual_settings={key:expression.get(key,default) for key,default in
            (('isBinary',False),('overrideBlink','none'),('overrideMouth','none'),('overrideLookAt','none'))}
        if actual_settings!=contract['expressionSettings'][name]:raise ValueError('Expression override settings changed: '+name)
        if expression.get('materialColorBinds') or expression.get('textureTransformBinds'):raise ValueError('Unexpected material-changing expression binds: '+name)
        binds=expression.get('morphTargetBinds',[])
        if not binds or len(binds)!=len(required):raise ValueError('Empty/changed expression binding count: '+name)
        actual_binds=[]
        for bind in binds:
            node=_at(nodes,bind.get('node'),'expression node');keys=shape_lookup.get(bind['node'],[])
            key=_at(keys,bind.get('index'),'expression target');weight=float(bind.get('weight',0))
            if not math.isfinite(weight) or abs(weight-1)>1e-6:raise ValueError('Unexpected expression weight')
            actual_binds.append({'mesh':node['name'],'key':key})
        if sorted(actual_binds,key=lambda x:(x['mesh'],x['key']))!=sorted(required,key=lambda x:(x['mesh'],x['key'])):
            raise ValueError('Expression binding targets changed: '+name)
        rows[name]={'kind':kind,'binds':len(binds),'targets':actual_binds,'settings':actual_settings}
    return rows


def inspect(path,contract,vrm=False):
    """Read actual exported JSON; no Blender required for this structural check."""
    document,binary=read_container(path);nodes=document.get('nodes',[]);meshes=document.get('meshes',[])
    actual=[n.get('name') for n in nodes if 'mesh' in n];expected={m['name']:m for m in contract['meshes']}
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):raise ValueError('Explicit mesh coverage mismatch: '+repr({'missing':sorted(set(expected)-set(actual)),'unexpected':sorted(set(actual)-set(expected))}))
    if document.get('animations') or document.get('cameras') or 'KHR_lights_punctual' in document.get('extensions',{}):raise ValueError('Unexpected animation/camera/light in static prototype')
    counts={'meshes':len(actual),'primitives':0,'triangles':0,'vertices':0,'morphTargets':0,'skins':len(document.get('skins',[]))}
    used_materials=set();material_uv={};shape_lookup={};mesh_reports=[]
    for node_index,node in enumerate(nodes):
        if 'mesh' not in node:continue
        mesh=_at(meshes,node['mesh'],'mesh');row=expected[node['name']]
        skin=_at(document.get('skins',[]),node.get('skin'),'mesh skin')
        for joint in skin.get('joints',[]):_at(nodes,joint,'skin joint')
        if not skin.get('joints'):raise ValueError('Empty skin')
        names=mesh.get('extras',{}).get('targetNames',[]);shape_lookup[node_index]=names
        if names!=row['shapeKeys']:raise ValueError('Morph target names/order changed: '+node['name'])
        weights=node.get('weights',mesh.get('weights',[0.]*len(names)))
        if len(weights)!=len(names) or any(abs(a-b)>1e-5 for a,b in zip(weights,row['defaultMorphWeights'])):raise ValueError('Default morph weights changed: '+node['name'])
        counts['morphTargets']+=len(names)
        for primitive in mesh.get('primitives',[]):
            if primitive.get('mode',4)!=4:raise ValueError('Nontriangle primitive')
            attributes=primitive['attributes']
            for required in ('POSITION','NORMAL','JOINTS_0','WEIGHTS_0'):
                if required not in attributes:raise ValueError('Missing skinned mesh attribute '+required+': '+node['name'])
            position=_at(document['accessors'],attributes['POSITION'],'position');vertex_count=position['count']
            count=_at(document['accessors'],primitive['indices'],'indices')['count'] if 'indices' in primitive else vertex_count
            if count%3:raise ValueError('Bad triangle index count')
            targets=primitive.get('targets',[])
            if len(targets)!=len(names):raise ValueError('Morph target count differs across primitives')
            for target in targets:
                for index in target.values():
                    if _at(document['accessors'],index,'morph')['count']!=vertex_count:raise ValueError('Morph accessor count changed')
            index=primitive.get('material');_at(document.get('materials',[]),index,'material');used_materials.add(index)
            material_uv.setdefault(index,[]).append({int(k.split('_')[1]) for k in attributes if k.startswith('TEXCOORD_')})
            counts['primitives']+=1;counts['vertices']+=vertex_count;counts['triangles']+=count//3
        mesh_reports.append({'name':node['name'],'node':node_index,'morphTargets':len(names),'defaultMorphWeights':weights})
    material_expected={m['name']:m for m in contract['materials']};material_reports=[]
    if {document['materials'][i].get('name') for i in used_materials}!=set(material_expected):raise ValueError('Material coverage differs from explicit scene contract')
    for index in sorted(used_materials):
        m=document['materials'][index];req=material_expected[m['name']];pbr=m.get('pbrMetallicRoughness',{})
        if any(k in m.get('extensions',{}) for k in ('VRMC_materials_mtoon','KHR_materials_unlit')):raise ValueError('PBR material became MToon or unlit: '+m['name'])
        factor=req['baseColorConstant'] or [1,1,1,1]
        if any(abs(a-b)>1e-5 for a,b in zip(pbr.get('baseColorFactor',[1,1,1,1]),factor)):raise ValueError('Unexpected base color multiplier: '+m['name'])
        if abs(pbr.get('metallicFactor',1)-req['constants']['Metallic'])>1e-6:raise ValueError('Metallic factor changed: '+m['name'])
        # Installed Blender 5.2 glTF exporter extensions/specular.py multiplies
        # Principled Specular IOR Level by 2, with factor capped at 1. In
        # particular the R09 single-reflection iris MUST retain factor 0.
        specular=m.get('extensions',{}).get('KHR_materials_specular',{})
        expected_specular=min(1.,2.*req['constants']['Specular IOR Level'])
        if abs(specular.get('specularFactor',1.)-expected_specular)>1e-6:
            raise ValueError('Base-layer specular factor changed: '+m['name'])
        coat=m.get('extensions',{}).get('KHR_materials_clearcoat',{})
        if abs(coat.get('clearcoatFactor',0)-req['constants']['Coat Weight'])>1e-5:raise ValueError('Clearcoat factor changed: '+m['name'])
        if req['constants']['Coat Weight']>0 and abs(coat.get('clearcoatRoughnessFactor',0)-req['constants']['Coat Roughness'])>1e-5:raise ValueError('Clearcoat roughness changed')
        checks={}
        if 'baseColor' in req['textures']:checks['baseColor']=_texture(document,binary,pbr.get('baseColorTexture',{}))
        if 'roughness' in req['textures']:
            checks['roughness']=_texture(document,binary,pbr.get('metallicRoughnessTexture',{}))
            if abs(pbr.get('roughnessFactor',1)-1)>1e-6:raise ValueError('Unexpected roughness multiplier')
        elif abs(pbr.get('roughnessFactor',1)-req['roughnessConstant'])>1e-5:raise ValueError('Constant hair roughness changed')
        if 'normal' in req['textures']:
            checks['normal']=_texture(document,binary,m.get('normalTexture',{}))
            if abs(m['normalTexture'].get('scale',1)-req['normalStrength'])>1e-6:raise ValueError('Normal-map strength changed')
        if req['alpha'] and m.get('alphaMode','OPAQUE')!='BLEND':raise ValueError('Continuous alpha changed/lost: '+m['name'])
        for channel,info in checks.items():
            if info['size']!=req['textures'][channel]['size']:raise ValueError('Texture resolution changed')
            if any(info['texCoord'] not in uvs for uvs in material_uv[index]):raise ValueError('Missing UV set for exported texture')
        material_reports.append({'material':m['name'],'index':index,'alphaMode':m.get('alphaMode','OPAQUE'),'textures':checks,
            'specularFactor':specular.get('specularFactor',1.),'sourceSpecularIORLevel':req['constants']['Specular IOR Level'],
            'sourceCoatIOR':req['constants']['Coat IOR'],'coatIORNotPreserved':req['sourceCoatIORNotPreserved']})
    # Validate every embedded image, including an optional VRM metadata thumbnail.
    images=[]
    for i in range(len(document.get('images',[]))):
        raw,size=_image_bytes(document,binary,i);images.append({'index':i,'size':size,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    expression_report={};springs=[]
    if vrm:
        ext=document.get('extensions',{});v=ext.get('VRMC_vrm',{})
        if v.get('specVersion')!='1.0' or v.get('lookAt',{}).get('type')!='expression':raise ValueError('VRM1/expression LookAt missing')
        if not v.get('humanoid',{}).get('humanBones'):raise ValueError('VRM humanoid is empty')
        expression_report=_inspect_vrm_expressions(v,contract,nodes,shape_lookup)
        spring=ext.get('VRMC_springBone',{})
        if spring.get('specVersion')!='1.0':raise ValueError('VRM spring extension missing')
        springs=spring.get('springs',[])
        _inspect_spring_numeric(spring,contract['springNumericContract'],nodes)
        if {s.get('name') for s in springs}!=set(contract['springNames']):raise ValueError('Spring chain coverage changed')
        for s in springs:
            if len(s.get('joints',[]))<2:raise ValueError('Spring is missing terminal/chain joints')
            for joint in s['joints']:_at(nodes,joint.get('node'),'spring joint')
            for group in s.get('colliderGroups',[]):_at(spring.get('colliderGroups',[]),group,'spring collider group')
            actual_groups=[spring['colliderGroups'][index].get('name') for index in s.get('colliderGroups',[])]
            expected_groups=contract['springColliderGroups'][s['name']]
            if len(actual_groups)!=len(expected_groups) or set(actual_groups)!=set(expected_groups):raise ValueError('Exported spring collider-group association changed: '+s['name'])
    return {'file':str(path),'sha256':_sha(path),'bytes':Path(path).stat().st_size,'counts':counts,
            'meshes':mesh_reports,'materials':material_reports,'images':images,'expressions':expression_report,
            'expressionBindingCount':sum(r['binds'] for r in expression_report.values()),'springChains':len(springs),
            'extensionsUsed':document.get('extensionsUsed',[]),'checks':'Container, explicit coverage, PBR channel references, morph weights/bindings and spring references'},document

def _sample_pixels(image):
    size=list(image.size);pixels=np.empty(size[0]*size[1]*4,np.float32);image.pixels.foreach_get(pixels)
    if not np.isfinite(pixels).all():raise ValueError('Nonfinite image pixels')
    gridx=np.linspace(0,size[0]-1,min(64,size[0]),dtype=int);gridy=np.linspace(0,size[1]-1,min(64,size[1]),dtype=int)
    return pixels.reshape(size[1],size[0],4)[gridy[:,None],gridx[None,:]].copy()

def _restore_default_coat_roughness(path,contract):
    """Correct this installed glTF exporter's omitted Blender .03 default.

    KHR_materials_clearcoat defaults to zero, so omission would change the new
    documented iris. Patch only an absent field when the known source value is
    .03 and a coat already exists. The binary buffer stays byte-for-byte intact.
    """
    document,binary=read_container(path);source={m['name']:m for m in contract['materials']};changes=[]
    for material in document.get('materials',[]):
        req=source.get(material.get('name'))
        if not req or req['constants']['Coat Weight']<=0:continue
        coat=material.get('extensions',{}).get('KHR_materials_clearcoat')
        value=req['constants']['Coat Roughness']
        if coat and 'clearcoatRoughnessFactor' not in coat and abs(value-.03)<=1e-5:
            coat['clearcoatRoughnessFactor']=value
            changes.append({'material':material['name'],'field':'KHR_materials_clearcoat.clearcoatRoughnessFactor',
                'sourceValue':value,'reason':'Installed Blender exporter omits its .03 default; glTF omission means 0'})
    if changes:
        encoded=json.dumps(document,ensure_ascii=False,separators=(',',':')).encode('utf-8');encoded+=b' '*((-len(encoded))%4)
        container=struct.pack('<III',0x46546C67,2,28+len(encoded)+len(binary))
        container+=struct.pack('<II',len(encoded),0x4E4F534A)+encoded+struct.pack('<II',len(binary),0x004E4942)+binary
        Path(path).write_bytes(container)
    return changes

def _pixel_audit(path,contract,report,temporary):
    """Decode actual exported PNGs; sample RGBA, packed roughness G and normals."""
    document,binary=read_container(path);by_name={m['name']:m for m in contract['materials']};rows=[]
    for material in report['materials']:
        req=by_name[material['material']]
        for channel,info in material['textures'].items():
            raw,_=_image_bytes(document,binary,info['imageIndex']);png=temporary/('check-'+str(info['imageIndex'])+'.png');png.write_bytes(raw)
            exported=bpy.data.images.load(str(png),check_existing=False)
            try:
                exported.colorspace_settings.name='sRGB' if channel=='baseColor' else 'Non-Color'
                actual=_sample_pixels(exported);source=_sample_pixels(bpy.data.images[req['textures'][channel]['image']])
                if channel=='roughness':delta=np.abs(actual[:,:,1]-source[:,:,0]);tolerance=.015
                elif channel=='normal':delta=np.abs(actual[:,:,:3]-source[:,:,:3]);tolerance=.015
                else:
                    # RGB under exactly transparent pixels can be changed by PNG premultiplication.
                    mask=(source[:,:,3]>.05) if req['alpha'] else np.ones(source.shape[:2],bool)
                    rgb=np.abs(actual[:,:,:3]-source[:,:,:3])[mask].ravel();alpha=np.abs(actual[:,:,3]-source[:,:,3]).ravel() if req['alpha'] else np.array([])
                    delta=np.concatenate([rgb,alpha]);tolerance=.025
                error=float(delta.max(initial=0))
                if error>tolerance:raise ValueError('Exported pixel channel changed '+material['material']+'/'+channel+': '+str(error))
                rows.append({'material':material['material'],'channel':channel,'sampleGridMaximum':64,'maximumAbsoluteError':error,'tolerance':tolerance})
            finally:bpy.data.images.remove(exported)
    return {'method':'Decoded PNG grid samples; roughness uses glTF packed green channel; normal and alpha compared independently','samples':rows}

def export(output_relative='work/runtime-r06/prototype-export',overwrite=False):
    """Export temporary files, inspect both, then publish only to runtime folder."""
    root,source=_guard();preparation=json.loads((project_path()/PREP).read_text(encoding='utf-8'))
    current=prepare([r['file'] for r in preparation['sourceLineage']],require_baked=True,write=False)
    if current['sceneContractSha256']!=preparation['sceneContractSha256']:raise ValueError('Scene changed since prepare(); run explicit prepare again')
    if current['sourceLineage']!=preparation['sourceLineage']:raise ValueError('Source lineage files changed since prepare')
    dest=_within(output_relative,root);dest.mkdir(parents=True,exist_ok=True)
    outputs=['review-preview.vrm','review-preview.glb','review-preview.vrm.json','review-preview.glb.json','export-report.json']
    if not overwrite and any((dest/n).exists() for n in outputs):raise FileExistsError('Use a fresh prototype output directory')
    rig=bpy.data.objects[current['rig']];meshes=[bpy.data.objects[r['name']] for r in current['meshes']]
    selected=list(bpy.context.selected_objects);active=bpy.context.view_layer.objects.active
    hidden={o:o.hide_get() for o in [rig,*meshes]};meta=rig.data.vrm_addon_extension.vrm1.meta
    original_name=meta.vrm_name;authors=[a.value for a in meta.authors]
    try:
        meta.vrm_name='R06 independent PBR prototype review'
        if not authors:meta.authors.add().value='__CHARAKIT_REVIEW__'
        bpy.ops.object.select_all(action='DESELECT')
        for o in [rig,*meshes]:o.hide_set(False);o.select_set(True)
        bpy.context.view_layer.objects.active=rig
        with tempfile.TemporaryDirectory(prefix='.r06-preview-',dir=dest) as directory:
            temporary=Path(directory);glb=temporary/outputs[1];vrm=temporary/outputs[0]
            result=bpy.ops.export_scene.gltf(filepath=str(glb),export_format='GLB',use_selection=True,
                export_materials='EXPORT',export_image_format='AUTO',export_animations=False,export_current_frame=True,
                export_rest_position_armature=True,export_apply=False,export_morph=True,export_morph_normal=True,
                export_skins=True,export_all_influences=False,export_influence_nb=4,export_yup=True,
                export_cameras=False,export_lights=False,export_extras=True,export_try_sparse_sk=False)
            if 'FINISHED' not in result:raise RuntimeError('PBR GLB export did not finish')
            result=bpy.ops.export_scene.vrm(filepath=str(vrm),armature_object_name=rig.name,
                export_only_selections=True,export_invisibles=False,use_addon_preferences=False,
                enable_advanced_preferences=True,export_all_influences=False,export_lights=False,
                export_gltf_animations=False,export_try_sparse_sk=False)
            if 'FINISHED' not in result:raise RuntimeError('PBR VRM export did not finish')
            results={}
            for path,is_vrm in ((glb,False),(vrm,True)):
                corrections=_restore_default_coat_roughness(path,current)
                checked,document=inspect(path,current,vrm=is_vrm)
                checked['portableConstantCorrections']=corrections
                checked['pixelAudit']=_pixel_audit(path,current,checked,temporary)
                checked['file']=str((dest/path.name).relative_to(project_path()));results[path.suffix[1:]]=checked
                _json(temporary/(path.name+'.json'),document)
            after=prepare([r['file'] for r in current['sourceLineage']],require_baked=True,write=False)
            if after['sceneContractSha256']!=current['sceneContractSha256'] or _sha(source)!=current['savedRuntimeSha256']:
                raise RuntimeError('Export changed runtime scene contract or saved source')
            report={'stage':'prototype only','source':current,'outputs':results,'materialPolicy':'PBR preserved; no MToon conversion',
                'metadataPolicy':'Review name; existing authors retained or explicit technical placeholder. No ownership/license assertion.',
                'blendSaved':False,'productionFilesChanged':False,'viewerValidation':'pending; inspect actual VRM in target software',
                'appearanceStatus':current['hairRepresentation']['appearanceStatus'],
                'losses':LOSSES}
            _json(temporary/'export-report.json',report)
            for name in outputs:(temporary/name).replace(dest/name)
        return report
    finally:
        meta.vrm_name=original_name;meta.authors.clear()
        for value in authors:meta.authors.add().value=value
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects:obj.select_set(True)
        if active and active.name in bpy.context.view_layer.objects:bpy.context.view_layer.objects.active=active
        for obj,value in hidden.items():obj.hide_set(value)

if __name__=='__main__':print('Loaded only. Explicit prepare(source_lineage=[...], require_baked=...), then export().')
