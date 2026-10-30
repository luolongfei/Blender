"""Optional, reversible runtime-only torso cull candidate. Never auto-applied.

audit() reads current geometry and clothing bounds; it does NOT establish cloth
occlusion. apply_candidate() accepts only the sampled RT06 unified mesh in a
saved runtime copy and validates every retained morph, weight, UV/color/custom
attribute through original-index maps. Original mesh is retained for restore().
The current face snapshot helper is deliberately unchanged. Render the candidate
at rest and in collar/shoulder/wrist poses before deciding to keep it.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
OBJECT='RT06 Face / R01 Head'
REPORT='work/runtime-r06/r07-torso-cull-audit.json'
BACKUP='RT07_unculled_mesh'
CLOTHES=('Blouse_Tucked_Bodice','Blouse_PeterPan_Collar_L','Blouse_PeterPan_Collar_R',
 'Blouse_Puff_Sleeve_L','Blouse_Puff_Sleeve_R','Blouse_Cuff_L','Blouse_Cuff_R',
 'Hand L · continuous anatomy','Hand R · continuous anatomy')
PROTECTED={'R04 physical skin','R02 oral mucosa'}
BODY='Character / anatomical body skin'
MAPS={'POINT':'RT07_original_vertex','FACE':'RT07_original_face','CORNER':'RT07_original_corner'}
ATTRIBUTE_FIELDS={'FLOAT':('value',1),'INT':('value',1),'INT8':('value',1),'BOOLEAN':('value',1),
 'INT16_2D':('value',2),
 'FLOAT_VECTOR':('vector',3),'FLOAT2':('vector',2),'FLOAT_COLOR':('color',4),'BYTE_COLOR':('color',4)}
INTEGER_ATTRIBUTE_TYPES={'INT','INT8','INT16_2D'}

def _coordinates(mesh,key=None):
    data=key.data if key else mesh.vertices;a=np.empty(len(data)*3,np.float32);data.foreach_get('co',a);return a.reshape(-1,3)

def _neutral(obj):
    mesh=obj.data
    if mesh.shape_keys:
        keys=mesh.shape_keys.key_blocks;a=_coordinates(mesh,keys[0]).copy()
        for key in keys[1:]:
            if key.value:a+=key.value*(_coordinates(mesh,key)-_coordinates(mesh,key.relative_key))
    else:a=_coordinates(mesh)
    mat=np.array(bpy.data.objects['Character_Rig'].matrix_world.inverted()@obj.matrix_world)
    return a@mat[:3,:3].T+mat[:3,3]

def _fingerprint(mesh):
    h=hashlib.sha256();h.update(_coordinates(mesh).tobytes())
    for face in mesh.polygons:h.update(np.asarray([face.material_index,*face.vertices],np.int32).tobytes())
    if mesh.shape_keys:
        for key in mesh.shape_keys.key_blocks:h.update(key.name.encode());h.update(_coordinates(mesh,key).tobytes())
    return h.hexdigest()

def _candidate(mesh,a):
    materials=[m.name if m else None for m in mesh.materials]
    if not PROTECTED.issubset(materials) or BODY not in materials:raise ValueError('Reviewed material partition changed')
    initial=[]
    for p in mesh.polygons:
        material=materials[p.material_index]
        if material in PROTECTED:initial.append(p);continue
        if material!=BODY:raise ValueError('Unreviewed material in unified head/body')
        points=a[list(p.vertices)]
        # 20 mm below the proposed 1.24 cut; protect every physical neck polygon.
        # The .59 wrist bridge overlaps source hand bounds beginning at .5965.
        keep=((points[:,2]>=1.22)&(np.abs(points[:,0])<=.11))|(np.abs(points[:,0])>=.59)
        if keep.any():initial.append(p)
    vertices={v for p in initial for v in p.vertices}
    faces=[p for p in mesh.polygons if all(v in vertices for v in p.vertices)]
    return vertices,faces

def audit(object_name=OBJECT,write=True):
    if bpy is None:raise RuntimeError('Blender read access required')
    obj=bpy.data.objects[object_name];mesh=obj.data;a=_neutral(obj);keep,faces=_candidate(mesh,a)
    edge_count={}
    for p in faces:
        v=list(p.vertices)
        for x,y in zip(v,v[1:]+v[:1]):edge=tuple(sorted((x,y)));edge_count[edge]=edge_count.get(edge,0)+1
    boundary=sorted({v for edge,n in edge_count.items() if n==1 for v in edge})
    bounds=[]
    for name in CLOTHES:
        other=bpy.data.objects.get(name)
        if other is None:raise ValueError('Required clothing/hand reference missing: '+name)
        points=_neutral(other);bounds.append({'object':name,'min':points.min(0).tolist(),'max':points.max(0).tolist()})
    by_material=[]
    for i,material in enumerate(mesh.materials):
        original=[p for p in mesh.polygons if p.material_index==i];retained=[p for p in faces if p.material_index==i]
        by_material.append({'material':material.name,'originalFaces':len(original),'retainedFaces':len(retained),
                            'originalTriangles':sum(len(p.vertices)-2 for p in original),'retainedTriangles':sum(len(p.vertices)-2 for p in retained)})
    report={'source':bpy.data.filepath,'object':obj.name,'geometryAndMorphSha256':_fingerprint(mesh),
        'original':{'vertices':len(mesh.vertices),'faces':len(mesh.polygons),'triangles':sum(len(p.vertices)-2 for p in mesh.polygons)},
        'candidate':{'vertices':len(keep),'faces':len(faces),'triangles':sum(len(p.vertices)-2 for p in faces)},
        'materialPartition':by_material,'boundaryVertices':len(boundary),
        'boundaryBounds':{'min':a[boundary].min(0).tolist(),'max':a[boundary].max(0).tolist()} if boundary else None,
        'clothingNeutralBounds':bounds,'policy':{'allPhysicalFaceNeckAndOralFaces':'protected','bodyCollarMinZ':1.22,'bodyCollarAbsX':.11,'wristBridgeAbsXMin':.59},
        'coordinates':'Rig-local neutral shape mixture, without armature/cloth deformation',
        'occlusionVerified':False,'recommendation':'Do not auto-cull. Bounds support a candidate but cannot prove collar/sleeve coverage during motion.',
        'writesPerformed':False}
    if write:(project_path()/REPORT).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Saved disposable background runtime copy required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/runtime-r06').resolve())
    if not source.is_file() or bpy.context.mode!='OBJECT':raise RuntimeError('Saved runtime Object Mode required')
    obj=bpy.data.objects[OBJECT]
    if obj.get(BACKUP) or any(m.type!='ARMATURE' for m in obj.modifiers):raise RuntimeError('Fresh sampled face with only armature modifier required')
    if obj.data.uv_layers.get('R06 RuntimeUV'):raise RuntimeError('Cull candidate before runtime UV baking')
    return obj

def _attribute(attribute):
    if attribute.data_type not in ATTRIBUTE_FIELDS:raise ValueError('Attribute type requires explicit preservation support: '+attribute.name+'/'+attribute.data_type)
    prop,width=ATTRIBUTE_FIELDS[attribute.data_type]
    # Blender 5.2 custom_normal is CORNER INT16_2D: RNA exposes `value`
    # as a two-component INT array. Its foreach buffer uses RNA's int32,
    # retaining each signed 16-bit encoded normal component exactly.
    dtype=np.bool_ if attribute.data_type=='BOOLEAN' else np.int32 if attribute.data_type in INTEGER_ATTRIBUTE_TYPES else np.float32
    data=np.empty(len(attribute.data)*width,dtype)
    attribute.data.foreach_get(prop,data)
    if attribute.data_type=='INT16_2D' and (np.any(data < -32768) or np.any(data > 32767)):
        raise ValueError('INT16_2D component outside signed 16-bit range: '+attribute.name)
    return data.reshape(len(attribute.data),width)

def _attribute_equal(actual,expected,data_type):
    if data_type in INTEGER_ATTRIBUTE_TYPES or data_type=='BOOLEAN':return bool(np.array_equal(actual,expected))
    return bool(np.allclose(actual,expected,atol=2e-6,rtol=0))

def _layer_settings(mesh):
    colors=mesh.color_attributes
    return {'uvNames':[u.name for u in mesh.uv_layers],'uvActiveIndex':mesh.uv_layers.active_index,
            'uvRender':[u.name for u in mesh.uv_layers if u.active_render],
            'colorNames':[a.name for a in colors],
            'colorActiveIndex':getattr(colors,'active_color_index',None),
            'colorRenderIndex':getattr(colors,'render_color_index',None)}

def apply_candidate():
    """Create candidate only; retain unculled mesh and validate exact remapping."""
    obj=_guard();before=audit(write=False);record=json.loads((project_path()/REPORT).read_text(encoding='utf-8'))
    if record['geometryAndMorphSha256']!=before['geometryAndMorphSha256']:raise ValueError('Run audit() on the actual fresh sampled mesh first')
    original=obj.data;keep,planned_faces=_candidate(original,_neutral(obj));remove={i for i in range(len(original.vertices)) if i not in keep}
    expected_faces={p.index for p in planned_faces}
    expected_corners={li for p in planned_faces for li in p.loop_indices}
    protected_faces={p.index for p in original.polygons if original.materials[p.material_index].name in PROTECTED}
    if not protected_faces.issubset(expected_faces):raise RuntimeError('Candidate unexpectedly removes protected face/neck/oral polygons')
    layer_settings=_layer_settings(original);material_slots=[m.name if m else None for m in original.materials]
    skip={'position','.edge_verts','.corner_vert','.corner_edge'}
    attributes={a.name:(a.domain,a.data_type,_attribute(a)) for a in original.attributes if a.name not in skip and not a.name.startswith(('.select','.vs.','.es.'))}
    old_weights=[{g.group:float(g.weight) for g in v.groups} for v in original.vertices]
    old_groups=[g.name for g in obj.vertex_groups]
    selected=list(bpy.context.selected_objects);active=bpy.context.view_layer.objects.active
    active_key=obj.active_shape_key_index;only_key=obj.show_only_shape_key;select_mode=tuple(bpy.context.scene.tool_settings.mesh_select_mode)
    copied=original.copy();copied.name=original.name+' optional torso cull r07';obj.data=copied
    complete=False
    try:
        for domain,name in MAPS.items():
            if copied.attributes.get(name):raise ValueError('Existing map name collision')
            attr=copied.attributes.new(name=name,type='INT',domain=domain);attr.data.foreach_set('value',np.arange(len(attr.data),dtype=np.int32))
        bpy.ops.object.select_all(action='DESELECT');obj.hide_set(False);obj.select_set(True);bpy.context.view_layer.objects.active=obj
        obj.active_shape_key_index=0;obj.show_only_shape_key=True
        bpy.context.scene.tool_settings.mesh_select_mode=(True,False,False)
        bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='DESELECT')
        import bmesh
        bm=bmesh.from_edit_mesh(copied);bm.verts.ensure_lookup_table()
        original_ids=bm.verts.layers.int.get(MAPS['POINT'])
        if original_ids is None:raise RuntimeError('BMesh dropped the explicit original vertex index map')
        for vertex in bm.verts:vertex.select_set(vertex[original_ids] in remove)
        bmesh.update_edit_mesh(copied,loop_triangles=False,destructive=False)
        bpy.ops.mesh.delete(type='VERT');bpy.ops.object.mode_set(mode='OBJECT')
        maps={domain:np.asarray([d.value for d in copied.attributes[name].data],dtype=np.int64) for domain,name in MAPS.items()}
        point_map=maps['POINT']
        if set(point_map)!=keep or len(point_map)!=len(keep):raise RuntimeError('Vertex map differs from audited candidate')
        if set(maps['FACE'])!=expected_faces or len(maps['FACE'])!=len(expected_faces):raise RuntimeError('Face set differs from audited candidate')
        if set(maps['CORNER'])!=expected_corners or len(maps['CORNER'])!=len(expected_corners):raise RuntimeError('UV corner set differs from audited candidate')
        if not protected_faces.issubset(set(maps['FACE'])):raise RuntimeError('A physical face/neck/oral polygon was lost')
        if [m.name if m else None for m in copied.materials]!=material_slots:raise RuntimeError('Material slot order changed')
        if _layer_settings(copied)!=layer_settings:raise RuntimeError('UV/color active or render-layer settings changed')
        edge_lookup={tuple(sorted(e.vertices)):e.index for e in original.edges}
        maps['EDGE']=np.asarray([edge_lookup[tuple(sorted(int(point_map[v]) for v in e.vertices))] for e in copied.edges])
        if [g.name for g in obj.vertex_groups]!=old_groups:raise RuntimeError('Vertex group definitions changed')
        for vertex,old in zip(copied.vertices,point_map):
            if {g.group:float(g.weight) for g in vertex.groups}!=old_weights[old]:raise RuntimeError('Retained skin weights changed')
        if original.shape_keys:
            if copied.shape_keys is None or len(copied.shape_keys.key_blocks)!=len(original.shape_keys.key_blocks):raise RuntimeError('Morph set changed')
            for old,new in zip(original.shape_keys.key_blocks,copied.shape_keys.key_blocks):
                if old.name!=new.name or old.value!=new.value or old.relative_key.name!=new.relative_key.name:raise RuntimeError('Morph metadata changed')
                if not np.allclose(_coordinates(copied,new),_coordinates(original,old)[point_map],atol=2e-7,rtol=0):raise RuntimeError('Retained morph coordinates changed: '+old.name)
        rebuilt_attributes=[]
        for name,(domain,dtype,values) in attributes.items():
            actual=copied.attributes.get(name)
            expected=values[maps[domain]]
            if actual is None:
                # Blender's edit-mesh conversion can omit default-valued built-in
                # layers (observed sharp_face). Rebuild the exact original data
                # through the already verified domain map, then test it normally.
                actual=copied.attributes.new(name=name,type=dtype,domain=domain)
                if actual.name!=name:raise RuntimeError('Unable to restore exact attribute name: '+name)
                prop,width=ATTRIBUTE_FIELDS[dtype]
                actual.data.foreach_set(prop,np.ascontiguousarray(expected).reshape(-1))
                rebuilt_attributes.append({'name':name,'domain':domain,'type':dtype,'retainedElements':len(expected),
                    'allOriginalRetainedValuesZero':bool(np.all(expected==0)),
                    'method':'Recreated missing layer from original values using verified domain index map'})
            if actual.domain!=domain or actual.data_type!=dtype:raise RuntimeError('Attribute type/domain changed: '+name)
            if not _attribute_equal(_attribute(actual),expected,dtype):raise RuntimeError('Attribute values changed: '+name)
        for new,old_index in zip(copied.polygons,maps['FACE']):
            old=original.polygons[old_index]
            if new.material_index!=old.material_index or tuple(int(point_map[v]) for v in new.vertices)!=tuple(old.vertices):raise RuntimeError('Face/UV-corner correspondence changed')
        original.use_fake_user=True;obj[BACKUP]=original.name;obj['RT07_cull_candidate_unreviewed']=True
        report={'before':before,'retainedOriginalMesh':original.name,'after':{'vertices':len(copied.vertices),'faces':len(copied.polygons),'triangles':sum(len(p.vertices)-2 for p in copied.polygons)},
                'protectedFacesRetained':len(protected_faces),'materialSlotOrder':material_slots,'retainedLayerSettings':layer_settings,
                'rebuiltMissingAttributes':rebuilt_attributes,
                'retainedOriginalVertexMapSha256':hashlib.sha256(point_map.tobytes()).hexdigest(),
                'preservationChecks':['All sampled morph coordinates and metadata','all weights/group indices','UV/color/custom attributes in all supported domains','face and corner index correspondence'],
                'appearanceAndMotionReview':'pending; candidate only','blendSaved':False}
        (project_path()/'work/runtime-r06/r07-torso-cull-candidate.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        complete=True;return report
    finally:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        if not complete:obj.data=original
        obj.active_shape_key_index=active_key;obj.show_only_shape_key=only_key;bpy.context.scene.tool_settings.mesh_select_mode=select_mode
        bpy.ops.object.select_all(action='DESELECT')
        for old in selected:
            if old.name in bpy.context.view_layer.objects:old.select_set(True)
        if active and active.name in bpy.context.view_layer.objects:bpy.context.view_layer.objects.active=active

def restore():
    if bpy is None or not bpy.app.background:raise RuntimeError('Background runtime only')
    Path(bpy.data.filepath).resolve().relative_to((project_path()/'work/runtime-r06').resolve())
    obj=bpy.data.objects[OBJECT];name=obj.get(BACKUP)
    if not name:return False
    obj.data.use_fake_user=True;obj.data=bpy.data.meshes[name];del obj[BACKUP]
    if 'RT07_cull_candidate_unreviewed' in obj:del obj['RT07_cull_candidate_unreviewed']
    return True

if __name__=='__main__':print('Loaded only: audit(); optional apply_candidate(); restore(). No automatic cull or save.')
