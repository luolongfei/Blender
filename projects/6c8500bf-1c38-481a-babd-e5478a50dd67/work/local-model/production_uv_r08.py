"""Add one optional production UV layer through a disposable current-mix proxy.

Integrator, in a saved independent production work copy and Object Mode:
    run()             # memory only; no blend save, render, bake, or material edit
    run(write=False)  # complete proxy/probe rehearsal with no source UV write

The proxy is built from Basis + current unmuted relative-key deltas, not raw
mesh.vertices. It has no rig, parent, modifiers, animation, or shape keys.
Each of three explicit material domains owns its own 0..1 atlas; cross-material
overlap is intentional. Original UV layers and active/render/clone settings
remain unchanged. Failed probes never copy UVs into the production source.
"""
import hashlib
from functools import lru_cache
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

try:
    import bpy
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

OBJECT='R01 Head'
UV='R08 ProductionUV'
MAPS={'POINT':'R08_proxy_source_vertex','FACE':'R08_proxy_source_face','CORNER':'R08_proxy_source_corner'}
DOMAINS=(
    {'name':'physical-skin','material':'R04 physical skin','resolution':2048},
    {'name':'oral-mucosa','material':'R02 oral mucosa','resolution':1024},
    {'name':'anatomical-body','material':'Character / anatomical body skin','resolution':2048},
)


@lru_cache(maxsize=None)
def _api(name): return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _plain(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,set):return sorted(_plain(v) for v in value)
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if hasattr(value,'to_dict'):return _plain(value.to_dict())
    if hasattr(value,'to_list'):return _plain(value.to_list())
    if hasattr(value,'as_pointer') and hasattr(value,'name'):
        return {'id':value.name,'pointer':value.as_pointer()}
    return [_plain(v) for v in value]


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def _values(collection,prop,width=1,dtype=np.float32):
    a=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(prop,a)
    return a.reshape(len(collection),width)


def _array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def _scalar_rna(block):
    result={}
    for prop in block.bl_rna.properties:
        if prop.identifier=='rna_type' or prop.is_readonly or prop.type in {'POINTER','COLLECTION'}:continue
        result[prop.identifier]=_plain(getattr(block,prop.identifier))
    return result


def _custom(block):return {name:_plain(block[name]) for name in block.keys()}


def _action_state(action):
    if action is None:return None
    curves=[]
    for curve in _api('native_hair_playback_r04')['_fcurves'](action):
        curves.append({'path':curve.data_path,'index':curve.array_index,'mute':curve.mute,
            'extrapolation':curve.extrapolation,
            'keys':[_scalar_rna(key) for key in curve.keyframe_points],
            'samples':_array_hash(_values(curve.sampled_points,'co',2)),
            'modifiers':[_scalar_rna(m) for m in curve.modifiers]})
    return {'name':action.name,'pointer':action.as_pointer(),'curvesSha256':_hash(curves),
            'custom':_custom(action),'fakeUser':action.use_fake_user}


def _animation(block):
    animation=getattr(block,'animation_data',None)
    if animation is None:return None
    drivers=[]
    for curve in animation.drivers:
        driver=curve.driver
        drivers.append({'path':curve.data_path,'index':curve.array_index,'mute':curve.mute,
            'type':driver.type,'expression':driver.expression,'useSelf':driver.use_self,
            'variables':[{'name':v.name,'type':v.type,'targets':[{'id':_plain(t.id),
                'dataPath':t.data_path,'bone':t.bone_target,'transformType':t.transform_type,
                'transformSpace':t.transform_space} for t in v.targets]} for v in driver.variables]})
    nla=[{'settings':_scalar_rna(track),'strips':[{'settings':_scalar_rna(strip),
           'action':_action_state(getattr(strip,'action',None))} for strip in track.strips]}
         for track in animation.nla_tracks]
    return {'action':_action_state(animation.action),
            'slot':getattr(getattr(animation,'action_slot',None),'identifier',None),
            'drivers':drivers,'nla':nla}


def current_mix(obj):
    """Exact relative-key formula in object space; no raw-vertex substitution."""
    keys=obj.data.shape_keys
    if keys is None or not keys.use_relative or keys.reference_key is None:
        raise RuntimeError('Expected relative production shape keys with a Basis reference')
    if obj.show_only_shape_key:
        raise RuntimeError('Show Only Shape Key is active; current relative mix would not describe display')
    arrays={key.name:_values(key.data,'co',3).astype(np.float64) for key in keys.key_blocks}
    mix=arrays[keys.reference_key.name].copy();active=[]
    for key in keys.key_blocks:
        if key==keys.reference_key:continue
        if key.vertex_group:
            raise RuntimeError('Vertex-group-masked key needs an explicit evaluator: '+key.name)
        if not key.mute and key.value!=0:
            mix+=(arrays[key.name]-arrays[key.relative_key.name])*float(key.value)
            active.append({'key':key.name,'relativeTo':key.relative_key.name,'value':float(key.value)})
    if mix.shape!=(len(obj.data.vertices),3) or not np.isfinite(mix).all():
        raise RuntimeError('Invalid current shape combination')
    raw=_values(obj.data.vertices,'co',3).astype(float)
    return np.ascontiguousarray(mix,dtype=np.float32),{
        'formula':'Basis + sum(value * (key - relative_key)) for each unmuted relative key',
        'basis':keys.reference_key.name,'activeRelativeKeys':active,
        'rawVertexMaximumDifferenceMeters':float(np.abs(mix-raw).max()),
        'currentMixSha256':_array_hash(np.asarray(mix,dtype=np.float32)),
        'rigAndOtherModifiersApplied':False}


def _uv_settings(mesh):
    return {'activeIndex':mesh.uv_layers.active_index,
            'layers':[{'name':uv.name,'render':uv.active_render,
                       'clone':getattr(uv,'active_clone',None)} for uv in mesh.uv_layers],
            'activeAttributeIndex':getattr(mesh.attributes,'active_index',None),
            'activeColorIndex':getattr(mesh.color_attributes,'active_color_index',None),
            'renderColorIndex':getattr(mesh.color_attributes,'render_color_index',None)}


def _restore_uv_settings(mesh,state):
    new=mesh.uv_layers.get(UV)
    if new:
        new.active_render=False
        if hasattr(new,'active_clone'):new.active_clone=False
    mesh.uv_layers.active_index=state['activeIndex']
    for row in state['layers']:
        layer=mesh.uv_layers[row['name']]
        layer.active_render=row['render']
        if row['clone'] is not None:layer.active_clone=row['clone']
    for collection,attr,key in ((mesh.attributes,'active_index','activeAttributeIndex'),
        (mesh.color_attributes,'active_color_index','activeColorIndex'),
        (mesh.color_attributes,'render_color_index','renderColorIndex')):
        if state[key] is not None:setattr(collection,attr,state[key])


def _original_contract(obj,original_uv_names):
    mesh=obj.data;attribute_api=_api('runtime_torso_cull_r07')
    ignore={UV,'.vs.'+UV,'.es.'+UV,'.pn.'+UV}
    attributes=[]
    for attr in mesh.attributes:
        if attr.name in ignore or attr.name in {'position','.edge_verts','.corner_vert','.corner_edge'}:continue
        attributes.append({'name':attr.name,'domain':attr.domain,'type':attr.data_type,
                           'sha256':_array_hash(attribute_api['_attribute'](attr))})
    key_data=[]
    for key in mesh.shape_keys.key_blocks:
        key_data.append({'name':key.name,'coordinatesSha256':_array_hash(_values(key.data,'co',3)),
                         'settings':_scalar_rna(key),'relativeKey':key.relative_key.name})
    material_api=_api('runtime_face_refresh_r08')
    shape_state={'keySettings':_scalar_rna(mesh.shape_keys),'keys':key_data,
                 'custom':_custom(mesh.shape_keys),'animation':_animation(mesh.shape_keys)}
    state={'geometry':{
        'vertices':_array_hash(_values(mesh.vertices,'co',3)),
        'edges':_array_hash(_values(mesh.edges,'vertices',2,np.int32)),
        'loopVertices':_array_hash(_values(mesh.loops,'vertex_index',1,np.int32)),
        'loopEdges':_array_hash(_values(mesh.loops,'edge_index',1,np.int32)),
        'faceLoopStarts':_array_hash(_values(mesh.polygons,'loop_start',1,np.int32)),
        'faceLoopCounts':_array_hash(_values(mesh.polygons,'loop_total',1,np.int32)),
        'faceMaterials':_array_hash(_values(mesh.polygons,'material_index',1,np.int32)),
        'faceSmooth':_array_hash(_values(mesh.polygons,'use_smooth',1,np.bool_)),
        'edgeSeams':_array_hash(_values(mesh.edges,'use_seam',1,np.bool_))},
        'originalUVs':{name:_array_hash(_values(mesh.uv_layers[name].data,'uv',2)) for name in original_uv_names},
        'attributes':attributes,'shapeKeysSha256':_hash(shape_state),
        'weightsSha256':_hash({'groups':[{'name':g.name,'index':g.index,'locked':g.lock_weight} for g in obj.vertex_groups],
                              'weights':[[(g.group,float(g.weight)) for g in vertex.groups] for vertex in mesh.vertices]}),
        'materialSlots':[{'link':slot.link,'pointer':slot.material.as_pointer(),'name':slot.material.name,
                          'materialSha256':material_api['_material_contract'](slot.material)} for slot in obj.material_slots],
        'objectSettings':_scalar_rna(obj),'objectCustom':_custom(obj),'meshCustom':_custom(mesh),
        'objectAnimation':_animation(obj),'meshAnimation':_animation(mesh),
        'parent':_plain(obj.parent),'matrixWorld':_plain(obj.matrix_world),
        'matrixBasis':_plain(obj.matrix_basis),'parentInverse':_plain(obj.matrix_parent_inverse),
        'modifiers':[{'settings':_scalar_rna(m),'target':_plain(getattr(m,'object',None))} for m in obj.modifiers],
        'constraints':[{'settings':_scalar_rna(c),'target':_plain(getattr(c,'target',None))} for c in obj.constraints],
        'uvSettings':_uv_settings(mesh)}
    state['uvSettings']['layers']=[row for row in state['uvSettings']['layers'] if row['name'] in original_uv_names]
    state['sha256']=_hash(state)
    return state


def _pose_contract():
    rows=[]
    for rig in bpy.context.scene.objects:
        if rig.type!='ARMATURE':continue
        rows.append({'name':rig.name,'world':_plain(rig.matrix_world),'animation':_animation(rig),
            'bones':[{'name':bone.name,'basis':_plain(bone.matrix_basis),'matrix':_plain(bone.matrix),
                      'custom':_custom(bone)} for bone in rig.pose.bones]})
    return {'frame':bpy.context.scene.frame_current,'subframe':bpy.context.scene.frame_subframe,'sha256':_hash(rows)}


def _domains(obj):
    materials=[slot.material.name if slot.material else None for slot in obj.material_slots]
    if len(materials)!=3 or set(materials)!={d['material'] for d in DOMAINS}:
        raise RuntimeError('Expected the three explicit production material domains: '+repr(materials))
    domains=[]
    for definition in DOMAINS:
        slot=materials.index(definition['material'])
        faces=[p.index for p in obj.data.polygons if p.material_index==slot]
        if not faces:raise RuntimeError('Empty material domain: '+definition['name'])
        domains.append(dict(definition,slot=slot,faces=faces))
    if sum(len(d['faces']) for d in domains)!=len(obj.data.polygons):raise RuntimeError('Domains do not partition all faces')
    return domains


def _proxy(source,coordinates):
    mesh=bpy.data.meshes.new('R08 temporary production UV proxy mesh')
    proxy=None
    try:
        mesh.from_pydata(coordinates.tolist(),[list(e.vertices) for e in source.data.edges],
                         [list(p.vertices) for p in source.data.polygons]);mesh.update()
        proxy=bpy.data.objects.new('R08 temporary production UV proxy',mesh)
        bpy.context.scene.collection.objects.link(proxy)
        proxy.hide_render=True
        for slot in source.material_slots:mesh.materials.append(slot.material)
        mesh.polygons.foreach_set('material_index',_values(source.data.polygons,'material_index',1,np.int32).ravel())
        if len(mesh.loops)!=len(source.data.loops) or not np.array_equal(
            _values(mesh.loops,'vertex_index',1,np.int32),_values(source.data.loops,'vertex_index',1,np.int32)):
            raise RuntimeError('Fresh proxy did not preserve source face/corner order')
        for domain,count in (('POINT',len(mesh.vertices)),('FACE',len(mesh.polygons)),('CORNER',len(mesh.loops))):
            attr=mesh.attributes.new(MAPS[domain],'INT',domain);attr.data.foreach_set('value',np.arange(count,dtype=np.int32))
        layer=mesh.uv_layers.new(name=UV,do_init=False)
        mesh.uv_layers.active_index=mesh.uv_layers.find(UV);layer.active_render=True
        if proxy.parent or proxy.modifiers or proxy.data.shape_keys or proxy.animation_data:
            raise RuntimeError('UV proxy acquired an unintended evaluation dependency')
        return proxy
    except Exception:
        if proxy is not None:bpy.data.objects.remove(proxy,do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise


def _remap_proxy_uv(proxy,source):
    mesh=proxy.data
    ids={domain:_values(mesh.attributes[name].data,'value',1,np.int32).ravel() for domain,name in MAPS.items()}
    for domain,count in (('POINT',len(source.data.vertices)),('FACE',len(source.data.polygons)),('CORNER',len(source.data.loops))):
        if not np.array_equal(np.sort(ids[domain]),np.arange(count)):
            raise RuntimeError('Proxy original-index map is no longer bijective: '+domain)
    current_vertex=_values(mesh.loops,'vertex_index',1,np.int32).ravel()
    source_vertex=_values(source.data.loops,'vertex_index',1,np.int32).ravel()
    if not np.array_equal(ids['POINT'][current_vertex],source_vertex[ids['CORNER']]):
        raise RuntimeError('Proxy loop map no longer refers to the same source vertices')
    for face in mesh.polygons:
        original=source.data.polygons[int(ids['FACE'][face.index])]
        if list(ids['CORNER'][list(face.loop_indices)])!=list(original.loop_indices):
            raise RuntimeError('Proxy polygon loop order changed')
        if face.material_index!=original.material_index:raise RuntimeError('Proxy material domain changed')
    uv=_values(mesh.uv_layers[UV].data,'uv',2)
    result=np.empty_like(uv);result[ids['CORNER']]=uv
    return result,ids


def run(write=True,source_relative=None,angles=(45,30,20,12),probe_grids=(96,256)):
    if bpy is None or not bpy.app.background or bpy.context.mode!='OBJECT':
        raise RuntimeError('Use an independent background production copy in Object Mode')
    file=Path(bpy.data.filepath).resolve();file.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.data.filepath or not file.is_file():raise RuntimeError('Saved production work copy required')
    if source_relative and file!=(project_path()/source_relative).resolve():raise RuntimeError('Opened source differs from explicit request')
    if tuple(angles)!=(45,30,20,12) or tuple(probe_grids)!=(96,256):
        raise ValueError('Use the reviewed bounded angle sequence and probe grids')
    source=bpy.data.objects.get(OBJECT)
    if source is None or source.type!='MESH' or source.library or source.data.library or source.data.users!=1:
        raise RuntimeError('Expected local single-user R01 Head')
    if source.data.uv_layers.get(UV) or source.data.attributes.get(UV):raise RuntimeError('Production UV already exists; preserve it')
    if any(slot.material is None for slot in source.material_slots):raise RuntimeError('Empty source material slot')
    bpy.context.view_layer.update()
    mesh=source.data;domains=_domains(source);old_uv_names=tuple(layer.name for layer in mesh.uv_layers)
    if not old_uv_names:raise RuntimeError('Expected preserved original UV layers')
    uv_settings=_uv_settings(mesh);coordinates,mix=current_mix(source)
    before=_original_contract(source,old_uv_names);pose_before=_pose_contract();source_sha=_sha(file)
    selected=list(bpy.context.selected_objects);active=bpy.context.view_layer.objects.active
    tool=bpy.context.scene.tool_settings;select_mode=tuple(tool.mesh_select_mode);sync=tool.use_uv_select_sync
    proxy=None;new_layer=None;committed=False;attempts={d['name']:[] for d in domains}
    probe=_api('runtime_material_bake_r06')['_uv_probe']
    try:
        proxy=_proxy(source,coordinates)
        bpy.ops.object.select_all(action='DESELECT');proxy.select_set(True);bpy.context.view_layer.objects.active=proxy
        tool.mesh_select_mode=(False,False,True);tool.use_uv_select_sync=False
        import bmesh
        for domain in domains:
            selected_faces=set(domain['faces']);accepted=False
            for angle in angles:
                bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='DESELECT')
                bm=bmesh.from_edit_mesh(proxy.data);bm.faces.ensure_lookup_table();bm.faces.index_update();bm.select_mode={'FACE'}
                original_face=bm.faces.layers.int.get(MAPS['FACE'])
                if original_face is None:raise RuntimeError('Edit mesh lost original face IDs')
                for face in bm.faces:face.select_set(face[original_face] in selected_faces)
                bm.select_flush_mode();bmesh.update_edit_mesh(proxy.data,loop_triangles=False,destructive=False)
                actual_selected={face[original_face] for face in bm.faces if face.select}
                if actual_selected!=selected_faces:raise RuntimeError('UV operator selection escaped its material domain')
                bpy.ops.uv.smart_project(angle_limit=math.radians(angle),island_margin=12/domain['resolution'],
                    area_weight=0.,correct_aspect=False,scale_to_bounds=True)
                bpy.ops.object.mode_set(mode='OBJECT')
                _,ids=_remap_proxy_uv(proxy,source)
                face_ids=[i for i,value in enumerate(ids['FACE']) if int(value) in selected_faces]
                probes=[probe(proxy.data,proxy.data.uv_layers[UV],face_ids,grid=grid) for grid in probe_grids]
                accepted=all(p['coveredCellCenters']>0 and not any(p[field] for field in
                    ('outsideUnitSquareTriangles','degenerateUVTriangles','overlappingCellCenters')) for p in probes)
                attempt={'angleDegrees':angle,'accepted':accepted,'probes':probes};attempts[domain['name']].append(attempt)
                print('R08_PRODUCTION_UV_ATTEMPT '+json.dumps({'domain':domain['name'],**attempt}),flush=True)
                if accepted:break
            if not accepted:
                raise RuntimeError('Bounded UV attempts failed; source was not written: '+json.dumps(attempts))
        planned_uv,ids=_remap_proxy_uv(proxy,source)
        final_probes={}
        for domain in domains:
            domain_faces=set(domain['faces'])
            proxy_faces=[i for i,value in enumerate(ids['FACE']) if int(value) in domain_faces]
            final_probes[domain['name']]=[probe(proxy.data,proxy.data.uv_layers[UV],proxy_faces,grid=grid)
                                          for grid in probe_grids]
        if any(p['outsideUnitSquareTriangles'] or p['degenerateUVTriangles'] or p['overlappingCellCenters']
               for values in final_probes.values() for p in values):raise RuntimeError('Later domain unwrap altered an earlier accepted domain')
        proxy_mesh=proxy.data;bpy.data.objects.remove(proxy,do_unlink=True);proxy=None;bpy.data.meshes.remove(proxy_mesh)
        # Restore UI/tool state before hashing original persistent data.
        tool.mesh_select_mode=select_mode;tool.use_uv_select_sync=sync
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects:obj.select_set(True)
        bpy.context.view_layer.objects.active=active
        if _original_contract(source,old_uv_names)!=before or _pose_contract()!=pose_before:
            raise RuntimeError('Original source data or pose changed during proxy preparation; no UV copied')
        if write:
            new_layer=mesh.uv_layers.new(name=UV,do_init=False)
            new_layer.data.foreach_set('uv',np.ascontiguousarray(planned_uv,dtype=np.float32).ravel())
            _restore_uv_settings(mesh,uv_settings);mesh.update_tag()
            if not np.array_equal(_values(new_layer.data,'uv',2),planned_uv):raise RuntimeError('Copied source UV differs from accepted proxy loops')
        after=_original_contract(source,old_uv_names);pose_after=_pose_contract()
        if before!=after or pose_before!=pose_after:raise RuntimeError('Original-data hash changed; new layer will be rolled back')
        if _sha(file)!=source_sha:raise RuntimeError('Saved source file changed externally during UV preparation')
        report={'source':str(file.relative_to(project_path())),'sourceFileSha256':source_sha,
            'object':source.name,'newLayer':UV,'writtenInMemory':bool(write),'savedBlend':False,
            'currentMix':mix,'proxyHadNoRigOrShapeKeys':True,'loopMapWasBijective':True,
            'loopUVSha256':_array_hash(planned_uv),'boundedAngles':list(angles),'probeGrids':list(probe_grids),
            'domains':[{'name':d['name'],'material':d['material'],'slot':d['slot'],'faces':len(d['faces']),
                        'resolutionForMargin':d['resolution'],'attempts':attempts[d['name']],
                        'finalProbes':final_probes[d['name']]} for d in domains],
            'originalContractBefore':before,'originalContractAfter':after,'originalDataUnchanged':True,
            'poseBefore':pose_before,'poseAfter':pose_after,'originalUVSettingsPreserved':uv_settings,
            'limits':['Each material domain owns an independent 0..1 atlas; cross-material UV overlap is intentional.',
                      'Zero outside/degenerate/sample-overlap counts at grids 96 and 256; finite sampling can miss narrow overlaps and boundary contacts.',
                      'Current shader bindings and active/render/clone UV remain unchanged; explicitly select the new named UV for a future texture workflow.',
                      'Only UV data is added. This is not a geometry, material, deformation or visual-acceptance certificate.']}
        committed=True;return report
    finally:
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        if proxy is not None:
            proxy_mesh=proxy.data;bpy.data.objects.remove(proxy,do_unlink=True);bpy.data.meshes.remove(proxy_mesh)
        if new_layer is not None and not committed:
            layer=mesh.uv_layers.get(UV)
            if layer:mesh.uv_layers.remove(layer)
            _restore_uv_settings(mesh,uv_settings)
        tool.mesh_select_mode=select_mode;tool.use_uv_select_sync=sync
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects:obj.select_set(True)
        if active and active.name in bpy.context.view_layer.objects:bpy.context.view_layer.objects.active=active
        if not committed:
            print('R08_PRODUCTION_UV_ABORT '+json.dumps({'sourceLayerRetained':mesh.uv_layers.get(UV) is not None,
                'originalDataUnchanged':_original_contract(source,old_uv_names)==before,
                'poseUnchanged':_pose_contract()==pose_before,'attempts':attempts}),flush=True)


if __name__=='__main__':
    print('Loaded only. Call run(write=False) for proxy rehearsal or run() for in-memory UV addition; no auto-save.')
