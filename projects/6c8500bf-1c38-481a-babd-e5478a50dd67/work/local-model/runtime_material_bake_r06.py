"""Bounded CPU material-channel baking in an explicitly saved runtime copy.

No work runs on import. probe() reads mesh/material/UV facts. bake() writes PNGs
and changes only the opened disposable runtime copy's UV/material assignments;
it never saves a Blend. Only project/work/runtime-r06/ Blend files are accepted.
Real execution, images and exported VRM still require integrator verification.
"""
import hashlib,json,math,re,sys
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
UV='R06 RuntimeUV'
CHANNELS=('basecolor','roughness','normal','alpha')

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Use a disposable background Blender process')
    root=(project_path()/'work/runtime-r06').resolve();file=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or not file.is_file():raise RuntimeError('Saved runtime copy required')
    file.relative_to(root)
    if bpy.context.mode!='OBJECT':raise RuntimeError('Object Mode required')
    return file,root

def _objects(objects):
    result=[bpy.data.objects[o] if isinstance(o,str) else o for o in objects]
    if not result or len({o.name for o in result})!=len(result):raise ValueError('Explicit unique objects required')
    for o in result:
        if o.type!='MESH' or o.library or o.data.library or not o.data.polygons:raise ValueError('Local polygon meshes required: '+o.name)
        if o.data.users!=1:raise ValueError('Single-user runtime mesh required: '+o.name)
        if o.name not in bpy.context.view_layer.objects:raise ValueError('Object outside current view layer: '+o.name)
        if not o.data.materials or any(m is None for m in o.data.materials):raise ValueError('Empty material slot: '+o.name)
        if any(s.link!='DATA' for s in o.material_slots):raise ValueError('Data-linked material slots required: '+o.name)
    return result

def _root_mix(material,out):
    """Read only: accept precisely the R06 root feather's one-level shader mix."""
    mix=out.inputs['Surface'].links[0].from_node
    if mix.type!='MIX_SHADER':return None
    reason='Only the explicit R06 root feather Transparent/Principled mix is supported: '+material.name
    if any(len(mix.inputs[i].links)!=1 for i in (0,1,2)):raise ValueError(reason)
    factor=mix.inputs[0].links[0];transparent=mix.inputs[1].links[0].from_node;shader=mix.inputs[2].links[0].from_node
    if factor.from_node.type!='ATTRIBUTE' or factor.from_node.attribute_name!='R06_root_edge_alpha':raise ValueError(reason+' (factor source)')
    factor_socket=factor.from_node.outputs.get('Fac') or factor.from_node.outputs.get('Factor')
    if factor.from_socket!=factor_socket:raise ValueError(reason+' (factor socket '+factor.from_socket.name+'/'+factor.from_socket.identifier+')')
    if transparent.type!='BSDF_TRANSPARENT' or shader.type!='BSDF_PRINCIPLED':raise ValueError(reason)
    color=transparent.inputs['Color']
    if color.is_linked or any(abs(float(v)-1)>1e-6 for v in color.default_value):raise ValueError(reason+' (tinted transparency)')
    alpha=shader.inputs['Alpha']
    if not alpha.is_linked and not math.isfinite(float(alpha.default_value)):raise ValueError('Nonfinite source Alpha: '+material.name)
    return mix,shader

def _surface(material):
    if not material.use_nodes or material.node_tree is None:raise ValueError('Node material required: '+material.name)
    out=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    if out is None or len(out.inputs['Surface'].links)!=1:raise ValueError('One active surface output required: '+material.name)
    shader=out.inputs['Surface'].links[0].from_node
    root_mix=_root_mix(material,out)
    if root_mix:shader=root_mix[1]
    if shader.type!='BSDF_PRINCIPLED':
        raise ValueError('Explicit adaptation required for surface '+shader.type+': '+material.name)
    if out.inputs['Displacement'].is_linked:raise ValueError('Displacement needs explicit geometry/normal transfer: '+material.name)
    unsupported=[]
    for name in ('Transmission Weight',):
        socket=shader.inputs.get(name)
        if socket and (socket.is_linked or float(socket.default_value)>1e-6):unsupported.append(name)
    for name in ('Metallic','Coat Weight','Coat Roughness','Coat IOR'):
        socket=shader.inputs.get(name)
        if socket and socket.is_linked:unsupported.append(name+' linked')
    emit=shader.inputs.get('Emission Color');strength=shader.inputs.get('Emission Strength')
    if emit and strength and (emit.is_linked or any(float(v)>1e-6 for v in emit.default_value[:3])) and (strength.is_linked or float(strength.default_value)>1e-6):unsupported.append('Emission')
    # These channels are outside this skin/cloth pass; never silently discard.
    if unsupported:raise ValueError('Unhandled physical inputs '+str(unsupported)+': '+material.name)
    return out,shader

def _normalize_root_mix(material):
    """Call only on a fresh runtime material copy, never the retained source."""
    out,shader=_surface(material);root_mix=_root_mix(material,out)
    if root_mix is None:return False
    nt=material.node_tree;mix=root_mix[0];alpha=shader.inputs['Alpha']
    # Both shader opacities clamp to [0,1]. Preserve a linked/non-unit source
    # Principled Alpha as well: effective opacity = feather * original alpha.
    def clamped(socket,label):
        node=nt.nodes.new('ShaderNodeMath');node.operation='MULTIPLY';node.use_clamp=True;node.label=label;node.inputs[1].default_value=1
        if socket.is_linked:nt.links.new(socket.links[0].from_socket,node.inputs[0])
        else:node.inputs[0].default_value=float(socket.default_value)
        return node.outputs[0]
    feather=clamped(mix.inputs[0],'R06 root feather opacity')
    original_alpha=clamped(alpha,'R06 original Principled opacity')
    multiply=nt.nodes.new('ShaderNodeMath');multiply.operation='MULTIPLY';multiply.use_clamp=True;multiply.label='R06 feather x source opacity'
    nt.links.new(feather,multiply.inputs[0]);nt.links.new(original_alpha,multiply.inputs[1]);nt.links.new(multiply.outputs[0],alpha)
    nt.links.new(shader.outputs['BSDF'],out.inputs['Surface'])
    material['r06_root_mix_normalized']=True
    return True

def _source_opacity(materials):
    """Classify authored shader opacity, never image background/unused UV texels."""
    rows=[]
    for material in materials:
        out,shader=_surface(material);root_mix=_root_mix(material,out);socket=shader.inputs['Alpha']
        if root_mix:kind='root feather shader mix';may_be_transparent=True;value=None
        elif socket.is_linked:kind='linked alpha';may_be_transparent=True;value=None
        else:
            value=float(socket.default_value)
            if not math.isfinite(value):raise ValueError('Nonfinite source alpha: '+material.name)
            value=min(1.,max(0.,value));kind='constant alpha';may_be_transparent=value<1.
        rows.append({'material':material.name,'kind':kind,'constantClampedAlpha':value,'mayBeTransparent':may_be_transparent})
    return {'mayBeTransparent':any(row['mayBeTransparent'] for row in rows),'sources':rows,
            'decisionScope':'Authored shader alpha. Linked alpha and the supported root mix conservatively retain transparency; unused texture texels cannot change an opaque source.'}


def _raster_triangle(mask,points):
    """Union UV triangle pixel-center occupancy; excludes bake dilation margins."""
    height,width=mask.shape;points=np.asarray(points,dtype=np.float64)
    if points.shape!=(3,2) or not np.all(np.isfinite(points)):raise ValueError('Invalid UV triangle')
    lo=np.maximum(np.ceil(points.min(axis=0)*[width,height]-.5).astype(int),[0,0])
    hi=np.minimum(np.floor(points.max(axis=0)*[width,height]-.5).astype(int),[width-1,height-1])
    if np.any(lo>hi):return
    a,b,c=points;area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if abs(area)<1e-16:return
    x=(np.arange(lo[0],hi[0]+1)+.5)/width
    y=((np.arange(lo[1],hi[1]+1)+.5)/height)[:,None]
    e0=(b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])
    e1=(c[0]-b[0])*(y-b[1])-(c[1]-b[1])*(x-b[0])
    e2=(a[0]-c[0])*(y-c[1])-(a[1]-c[1])*(x-c[0])
    inside=(e0>=-1e-12)&(e1>=-1e-12)&(e2>=-1e-12) if area>0 else (e0<=1e-12)&(e1<=1e-12)&(e2<=1e-12)
    mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1]|=inside


def _uv_pixel_coverage(mesh,uv,faces,size):
    mesh.calc_loop_triangles();selected=set(faces);mask=np.zeros((size,size),dtype=np.bool_)
    for triangle in mesh.loop_triangles:
        if triangle.polygon_index in selected:_raster_triangle(mask,[uv.data[i].uv[:] for i in triangle.loops])
    return mask.reshape(-1)


def _uv_probe(mesh,uv,face_ids=None,grid=96):
    if uv is None:return {'present':False,'scope':'No UV layer'}
    mesh.calc_loop_triangles();wanted=set(face_ids) if face_ids is not None else None
    occupancy=np.zeros((grid,grid),np.uint16);outside=degenerate=triangles=0
    for tri in mesh.loop_triangles:
        if wanted is not None and tri.polygon_index not in wanted:continue
        q=np.array([uv.data[i].uv[:] for i in tri.loops],float);triangles+=1
        if not np.isfinite(q).all():raise ValueError('Nonfinite UV')
        outside+=int(bool(np.any(q < -1e-6) or np.any(q > 1+1e-6)))
        a,b,c=q;ab=b-a;ac=c-a;den=ab[0]*ac[1]-ab[1]*ac[0]
        if abs(den)<1e-12:degenerate+=1;continue
        low=np.maximum(0,np.floor(q.min(axis=0)*grid).astype(int));high=np.minimum(grid-1,np.floor(q.max(axis=0)*grid).astype(int))
        if np.any(high<low):continue
        xx,yy=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1));points=np.stack(((xx+.5)/grid,(yy+.5)/grid),axis=-1)-a
        u=(points[...,0]*ac[1]-points[...,1]*ac[0])/den
        v=(ab[0]*points[...,1]-ab[1]*points[...,0])/den
        inside=(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)
        region=occupancy[low[1]:high[1]+1,low[0]:high[0]+1];region+=inside.astype(np.uint16)
    covered=int((occupancy>0).sum());overlap=int((occupancy>1).sum())
    return {'present':True,'layer':uv.name,'triangles':triangles,'outsideUnitSquareTriangles':outside,
        'degenerateUVTriangles':degenerate,'grid':grid,'coveredCellCenters':covered,'overlappingCellCenters':overlap,
        'overlapFractionOfCoveredSamples':overlap/max(1,covered),
        'scope':'Finite cell-center coverage sample. It can miss small overlaps, narrow islands and boundary contacts; not a full UV proof.'}

def probe(objects):
    source,_=_guard();rows=[]
    for o in _objects(objects):
        mats=[]
        for m in o.data.materials:
            if m.get('r06_preserve_alpha_atlas'):
                mats.append({'name':m.name,'protectedAlphaAtlas':True});continue
            try:
                out,s=_surface(m);root_mix=_root_mix(m,out)
                inputs={n:{'linked':s.inputs[n].is_linked,'upstreamTypes':[l.from_node.type for l in s.inputs[n].links]} for n in ('Base Color','Roughness','Normal','Alpha')}
                if root_mix:
                    inputs['Alpha']={'linked':True,'upstreamTypes':['ATTRIBUTE']+[l.from_node.type for l in s.inputs['Alpha'].links],
                        'attribute':'R06_root_edge_alpha','effectiveFormula':'clamp(root feather Fac) * clamp(original Principled Alpha)',
                        'originalPrincipledAlpha':inputs['Alpha'],'conversionScope':'Fresh temporary runtime material copy only'}
                mats.append({'name':m.name,'inputs':inputs,'bakeableSurface':True,
                    'surfacePattern':'R06 transparent-principled root feather' if root_mix else 'direct Principled',
                    'transparentMixSupported':bool(root_mix),'sssLossInMToon':bool(s.inputs.get('Subsurface Weight') and s.inputs['Subsurface Weight'].default_value>0)})
            except ValueError as exc:mats.append({'name':m.name,'bakeableSurface':False,'reason':str(exc),
                'actualLinks':[{'from':l.from_node.name,'fromType':l.from_node.type,'socket':l.from_socket.name,
                    'socketIdentifier':l.from_socket.identifier,'attribute':getattr(l.from_node,'attribute_name',None),
                    'to':l.to_node.name,'toType':l.to_node.type,'input':l.to_socket.name,
                    'inputIndex':list(l.to_node.inputs).index(l.to_socket)} for l in m.node_tree.links]})
        rows.append({'object':o.name,'vertices':len(o.data.vertices),'polygons':len(o.data.polygons),'materials':mats,'activeUV':_uv_probe(o.data,o.data.uv_layers.active)})
    return {'source':str(source),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'objects':rows,'status':'inventory only'}

def _slug(value):return re.sub(r'[^A-Za-z0-9_-]+','-',value).strip('-')[:72] or 'material'

def _domains(obj,resolution,material_domains,preserve_unlisted=False):
    spec=(material_domains or {}).get(obj.name)
    if spec is None:
        size=resolution.get(obj.name,1024) if isinstance(resolution,dict) else resolution
        spec=[{'name':'surface','slots':list(range(len(obj.data.materials))),'resolution':size}]
    result=[];allslots=[]
    for d in spec:
        size=int(d.get('resolution',resolution.get(obj.name,1024) if isinstance(resolution,dict) else resolution))
        slots=list(d['slots'])
        if not 256<=size<=4096 or not slots or any(not isinstance(i,int) or i<0 or i>=len(obj.data.materials) for i in slots):raise ValueError('Invalid texture domain on '+obj.name)
        result.append({'name':_slug(d['name']),'slots':slots,'resolution':size,'faces':[p.index for p in obj.data.polygons if p.material_index in slots]});allslots+=slots
    if len(allslots)!=len(set(allslots)) or (not preserve_unlisted and sorted(allslots)!=list(range(len(obj.data.materials)))):raise ValueError('Domains must partition selected material slots exactly once: '+obj.name)
    if len({d['name'] for d in result})!=len(result):raise ValueError('Domain names must be unique')
    if any(not d['faces'] for d in result):raise ValueError('Every domain must own actual faces; remove empty runtime slots first')
    return result

def _select(obj):
    bpy.ops.object.select_all(action='DESELECT');obj.hide_set(False);obj.hide_render=False;obj.select_set(True);bpy.context.view_layer.objects.active=obj

def _prepare_uv(obj,domains,repack):
    mesh=obj.data;source=next((u for u in mesh.uv_layers if u.active_render),mesh.uv_layers.active)
    method='SMART_PROJECT' if repack is True else 'COPY' if repack is False else repack
    if method not in {'SMART_PROJECT','SMART_PROJECT_BOUNDED','COPY','SOURCE_PACK'}:raise ValueError('Explicit supported runtime UV method required')
    source_name=source.name if source else None
    source_probes={d['name']:_uv_probe(mesh,source,d['faces']) for d in domains}
    if method=='SOURCE_PACK' and (source is None or any(
        p['outsideUnitSquareTriangles'] or p['degenerateUVTriangles'] or p['overlappingCellCenters'] for p in source_probes.values())):
        raise ValueError('Source UV must already be valid before packing its existing islands: '+json.dumps(source_probes))
    layer=mesh.uv_layers.get(UV)
    if layer is not None:raise ValueError('Runtime UV already exists; use a new staged copy or explicitly inspect before rebaking')
    source_coordinates=None
    if source:
        source_coordinates=np.empty(len(mesh.loops)*2,np.float32);source.data.foreach_get('uv',source_coordinates)
    layer=mesh.uv_layers.new(name=UV)
    if source_coordinates is not None:layer.data.foreach_set('uv',source_coordinates)
    mesh.uv_layers.active_index=mesh.uv_layers.find(UV);layer.active_render=True
    attempts={d['name']:[] for d in domains}
    if method in {'SMART_PROJECT','SMART_PROJECT_BOUNDED','SOURCE_PACK'}:
        _select(obj)
        for d in domains:
            selected=set(d['faces'])
            angles=(45,30,20) if method=='SMART_PROJECT_BOUNDED' else (66,) if method=='SMART_PROJECT' else (None,)
            for angle in angles:
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='DESELECT')
                import bmesh
                bm=bmesh.from_edit_mesh(mesh);bm.faces.ensure_lookup_table();bm.faces.index_update()
                for face in bm.faces:face.select_set(face.index in selected)
                bm.select_flush_mode();bmesh.update_edit_mesh(mesh,loop_triangles=False,destructive=False)
                if angle is not None:
                    bpy.ops.uv.smart_project(angle_limit=math.radians(angle),island_margin=12/d['resolution'],area_weight=0.,correct_aspect=True,scale_to_bounds=True)
                else:
                    # Keep the authored island shape, seams and continuity.
                    bpy.ops.uv.select_all(action='SELECT')
                    bpy.ops.uv.pack_islands(rotate=True,margin=12/d['resolution'])
                bpy.ops.object.mode_set(mode='OBJECT')
                result=_uv_probe(mesh,mesh.uv_layers[UV],d['faces'])
                valid=not any(result[field] for field in ('outsideUnitSquareTriangles','degenerateUVTriangles','overlappingCellCenters'))
                attempts[d['name']].append({'angleDegrees':angle,'accepted':valid,'probe':result})
                print('RT06_UV_ATTEMPT',json.dumps({'object':obj.name,'domain':d['name'],**attempts[d['name']][-1]}),flush=True)
                if valid:break
        layer=mesh.uv_layers[UV]
    if source_coordinates is not None:
        unchanged=np.empty(len(mesh.loops)*2,np.float32);mesh.uv_layers[source_name].data.foreach_get('uv',unchanged)
        if not np.array_equal(source_coordinates,unchanged):raise RuntimeError('Original source UV changed during runtime preparation')
    probes=[dict(domain=d['name'],method=method,sourceLayer=source_name,sourceProbe=source_probes[d['name']],
        attempts=attempts[d['name']],originalSourceUVUnchanged=True,**_uv_probe(mesh,layer,d['faces'])) for d in domains]
    if method=='SMART_PROJECT_BOUNDED':
        record={'sourceRuntime':bpy.data.filepath,'object':obj.name,'method':method,'boundedAnglesDegrees':[45,30,20],
            'thresholds':'Zero outside, degenerate and overlapping sampled cell centers; unchanged strict interior test at grid 96.',
            'domains':probes,'limits':'Finite UV sampling is not an exhaustive geometric overlap proof. All attempts failed means seam-based unwrap needs a separately reviewed seam layout.'}
        for index in range(1,1000):
            report_path=project_path()/f'work/runtime-r06/uv-attempts-{_slug(obj.name)}-{index:03d}.json'
            if report_path.exists():continue
            with report_path.open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
            for probe in probes:probe['attemptReport']=str(report_path.relative_to(project_path()))
            break
        else:raise RuntimeError('Too many retained UV attempt reports')
    if any(x['outsideUnitSquareTriangles'] or x['degenerateUVTriangles'] or x['overlappingCellCenters'] for x in probes):raise ValueError('Runtime UV domain requires correction: '+json.dumps(probes))
    return layer,probes

def _freeze_uv(material,uv_name):
    # Only the outer tree is changed. Existing explicit named UVs remain intact.
    nt=material.node_tree;node=None
    def output():
        nonlocal node
        if not uv_name:raise ValueError('Implicit source texture UV missing: '+material.name)
        if node is None:node=nt.nodes.new('ShaderNodeUVMap');node.uv_map=uv_name
        return node.outputs['UV']
    for n in list(nt.nodes):
        if n.type=='GROUP' and n.node_tree:
            # Copy nested groups before fixing their implicit texture coordinates.
            copied=n.node_tree.copy();n.node_tree=copied
            wrapper=type('TreeOwner',(),{'node_tree':copied,'name':material.name})()
            _freeze_uv(wrapper,uv_name)
        elif n.type=='TEX_IMAGE' and n.image and not n.inputs['Vector'].is_linked:nt.links.new(output(),n.inputs['Vector'])
        elif n.type=='UVMAP' and not n.uv_map:n.uv_map=uv_name or ''
        elif n.type=='TEX_COORD':
            for link in list(n.outputs['UV'].links):destination=link.to_socket;nt.links.remove(link);nt.links.new(output(),destination)

def _channel(material,channel,image):
    nt=material.node_tree;out,shader=_surface(material);original=out.inputs['Surface'].links[0].from_socket
    target=nt.nodes.new('ShaderNodeTexImage');target.image=image;nt.nodes.active=target
    for node in nt.nodes:node.select=node==target
    emission=None
    if channel!='normal':
        socket=shader.inputs[{'basecolor':'Base Color','roughness':'Roughness','alpha':'Alpha'}[channel]]
        emission=nt.nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=1
        if socket.is_linked:nt.links.new(socket.links[0].from_socket,emission.inputs['Color'])
        else:
            value=socket.default_value
            emission.inputs['Color'].default_value=tuple(value) if hasattr(value,'__len__') else (value,value,value,1)
        nt.links.new(emission.outputs[0],out.inputs['Surface'])
    return out,original,target,emission

def _restore_channel(material,state):
    out,original,target,emission=state;nt=material.node_tree
    nt.links.new(original,out.inputs['Surface']);nt.nodes.remove(target)
    if emission:nt.nodes.remove(emission)

def _save_image(image,path):
    pixels=np.empty(image.size[0]*image.size[1]*4,np.float32);image.pixels.foreach_get(pixels)
    if not np.isfinite(pixels).all():raise RuntimeError('Nonfinite baked image: '+image.name)
    temp=path.with_suffix('.tmp.png');image.file_format='PNG';image.filepath_raw=str(temp);image.save()
    if not temp.is_file() or temp.stat().st_size<128:raise RuntimeError('Image save failed: '+str(path))
    temp.replace(path);image.filepath_raw=str(path);image.source='FILE';image.reload();image.pack()
    return {'file':str(path.relative_to(project_path())),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'size':list(image.size),'packed':bool(image.packed_file)}

def _portable(name,images,transparent,source_shader=None):
    m=bpy.data.materials.new(name);m.use_nodes=True;nt=m.node_tree;nt.nodes.clear()
    uv=nt.nodes.new('ShaderNodeUVMap');uv.uv_map=UV
    shader=nt.nodes.new('ShaderNodeBsdfPrincipled');out=nt.nodes.new('ShaderNodeOutputMaterial');nt.links.new(shader.outputs[0],out.inputs['Surface'])
    if source_shader:
        for prop in ('Metallic','Coat Weight','Coat Roughness','Coat IOR','IOR','Specular IOR Level'):
            if source_shader.inputs.get(prop) and shader.inputs.get(prop):shader.inputs[prop].default_value=source_shader.inputs[prop].default_value
    tex={}
    for key in ('basecolor','roughness','normal'):
        n=nt.nodes.new('ShaderNodeTexImage');n.image=images[key];n.extension='EXTEND';tex[key]=n;nt.links.new(uv.outputs['UV'],n.inputs['Vector'])
    nt.links.new(tex['basecolor'].outputs['Color'],shader.inputs['Base Color']);nt.links.new(tex['roughness'].outputs['Color'],shader.inputs['Roughness'])
    normal=nt.nodes.new('ShaderNodeNormalMap');normal.space='TANGENT';normal.uv_map=UV;normal.inputs['Strength'].default_value=1
    nt.links.new(tex['normal'].outputs['Color'],normal.inputs['Color']);nt.links.new(normal.outputs['Normal'],shader.inputs['Normal'])
    if transparent:nt.links.new(tex['basecolor'].outputs['Alpha'],shader.inputs['Alpha'])
    m['r06_baked_material']=True;m['r06_alpha_present']=transparent
    m['r06_mtoon_losses']='MToon does not retain the source SSS/hair scattering/physical roughness response. PBR channels retained separately.'
    return m

def bake(objects,resolution=1024,material_domains=None,repack_uv=True,output_relative='work/runtime-r06/textures',overwrite=False,preserve_unlisted=False):
    """Bake explicit opaque/alpha Principled meshes; never save the source Blend.

    material_domains={object:[{name:'head',slots:[0],resolution:2048}, ...]}.
    Domains partition material slots (or a subset with preserve_unlisted=True) and may reuse 0..1 UV space because each
    gets separate textures. Mark existing card materials r06_preserve_alpha_atlas
    to protect them; mixed protected/unprotected slots require explicit splitting.
    """
    source,root=_guard();objects=_objects(objects);dest=(project_path()/output_relative).resolve();dest.relative_to(root);dest.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene;view=bpy.context.view_layer;selected=list(bpy.context.selected_objects);active=view.objects.active
    engine=scene.render.engine;cycles={n:getattr(scene.cycles,n) for n in ('device','samples')}
    render={n:getattr(scene.render,n) for n in ('threads_mode','threads')}
    bake_props={n:getattr(scene.render.bake,n) for n in ('use_selected_to_active','use_clear','use_pass_direct','use_pass_indirect','use_pass_color','margin','margin_type','target','normal_space')}
    select_mode=tuple(scene.tool_settings.mesh_select_mode);sync=scene.tool_settings.use_uv_select_sync
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();rows=[]
    try:
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1;scene.render.threads_mode='FIXED';scene.render.threads=6
        scene.tool_settings.mesh_select_mode=(False,False,True);scene.tool_settings.use_uv_select_sync=False
        scene.render.bake.use_selected_to_active=False;scene.render.bake.use_clear=True;scene.render.bake.margin=8;scene.render.bake.margin_type='EXTEND';scene.render.bake.target='IMAGE_TEXTURES';scene.render.bake.normal_space='TANGENT'
        for obj in objects:
            protected=[bool(m.get('r06_preserve_alpha_atlas')) for m in obj.data.materials]
            if any(protected):
                if not all(protected):raise ValueError('Mixed protected atlas slots require explicit object/domain handling: '+obj.name)
                rows.append({'object':obj.name,'skipped':'Explicitly protected existing alpha atlas; no material conversion or preservation claim'});continue
            for m in obj.data.materials:_surface(m)
            domains=_domains(obj,resolution,material_domains,preserve_unlisted);original=list(obj.data.materials);uv_original=obj.data.uv_layers.active_index
            uv_render=[u.name for u in obj.data.uv_layers if u.active_render];source_uv=next((u.name for u in obj.data.uv_layers if u.active_render),obj.data.uv_layers.active.name if obj.data.uv_layers.active else None)
            hidden=obj.hide_get();hidden_render=obj.hide_render;states=[];copies=[];completed=False
            stem=_slug(obj.name);files={(d['name'],c):dest/f'{stem}-{d["name"]}-{c}.png' for d in domains for c in CHANNELS}
            if not overwrite and any(f.exists() for f in files.values()):raise FileExistsError('Texture output already exists; choose a fresh stage directory')
            try:
                for i,m in enumerate(original):
                    m.use_fake_user=True;copy=m.copy();copy.name='R06 bake source / '+obj.name+' / '+str(i);_freeze_uv(copy,source_uv);_normalize_root_mix(copy);copies.append(copy);obj.data.materials[i]=copy
                uv,uv_report=_prepare_uv(obj,domains,repack_uv);_select(obj);images={};domain_slots={i:d for d in domains for i in d['slots']}
                for c in CHANNELS:
                    for d in domains:
                        im=bpy.data.images.new(stem+' / '+d['name']+' / '+c,width=d['resolution'],height=d['resolution'],alpha=True,float_buffer=False)
                        im.generated_color=(0,0,0,0);im.colorspace_settings.name='sRGB' if c=='basecolor' else 'Non-Color';images[(d['name'],c)]=im
                    discard=None
                    if preserve_unlisted and len(domain_slots)<len(copies):
                        discard=bpy.data.images.new(stem+' / discarded unselected / '+c,width=32,height=32,alpha=True)
                        discard.colorspace_settings.name='sRGB' if c=='basecolor' else 'Non-Color'
                    states=[(m,_channel(m,c,images[(domain_slots[i]['name'],c)] if i in domain_slots else discard)) for i,m in enumerate(copies)]
                    try:
                        result=bpy.ops.object.bake(type='NORMAL' if c=='normal' else 'EMIT',uv_layer=UV,use_selected_to_active=False,use_clear=True,margin=8,margin_type='EXTEND')
                        if 'FINISHED' not in result:raise RuntimeError('Bake did not finish: '+obj.name+'/'+c)
                    finally:
                        for m,state in states:_restore_channel(m,state)
                        states=[]
                        if discard:bpy.data.images.remove(discard)
                reports=[]
                for d in domains:
                    ims={c:images[(d['name'],c)] for c in CHANNELS};size=d['resolution'];rgba=np.empty(size*size*4,np.float32);alpha=np.empty_like(rgba)
                    ims['basecolor'].pixels.foreach_get(rgba);ims['alpha'].pixels.foreach_get(alpha);rgba=rgba.reshape(-1,4);alpha=alpha.reshape(-1,4)
                    # Cycles may clear the image's own A channel to one even
                    # outside islands. It is neither UV coverage nor opacity.
                    coverage=_uv_pixel_coverage(obj.data,uv,d['faces'],size)
                    if coverage.sum()<16:raise RuntimeError('Too little UV pixel-center coverage: '+obj.name+'/'+d['name'])
                    opacity=_source_opacity([original[i] for i in d['slots']]);transparency=opacity['mayBeTransparent']
                    rgba[:,3]=np.clip(alpha[:,0],0,1) if transparency else 1.
                    ims['basecolor'].pixels.foreach_set(rgba.ravel());ims['basecolor'].update()
                    image_reports={c:_save_image(ims[c],files[(d['name'],c)]) for c in CHANNELS}
                    constants=[{n:float(_surface(original[i])[1].inputs[n].default_value) for n in ('Metallic','Coat Weight','Coat Roughness','Coat IOR','IOR','Specular IOR Level')} for i in d['slots']]
                    if any(v!=constants[0] for v in constants[1:]):raise ValueError('Domain combines differing metallic/coat/IOR constants; split it explicitly')
                    material=_portable('R06 '+obj.name+' / '+d['name'],ims,transparency,_surface(original[d['slots'][0]])[1])
                    for slot in d['slots']:obj.data.materials[slot]=material
                    material['r06_opacity_decision']=json.dumps(opacity)
                    reports.append({'name':d['name'],'slots':d['slots'],'resolution':size,'coveragePixels':int(coverage.sum()),
                        'coverageScope':'Union of selected-domain UV triangle pixel centers at bake resolution; excludes margin dilation and subpixel areas. Derived from UV geometry, not image alpha.',
                        'transparentBakedCoveredPixelCenters':int((alpha[coverage,0]<.999).sum()),'opacityDecision':opacity,
                        'images':image_reports,'material':material.name})
                for i,m in enumerate(original):
                    if i not in domain_slots:obj.data.materials[i]=m
                obj['r06_material_bake_source_sha256']=source_hash;obj['r06_material_originals']=json.dumps([m.name for m in original]);obj['r06_material_bake_domains']=json.dumps([{k:v for k,v in d.items() if k!='faces'} for d in domains])
                rows.append({'object':obj.name,'domains':reports,'uvProbe':uv_report});completed=True
            finally:
                for m,state in states:_restore_channel(m,state)
                if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
                if not completed:
                    for i,m in enumerate(original):obj.data.materials[i]=m
                    layer=obj.data.uv_layers.get(UV)
                    if layer:obj.data.uv_layers.remove(layer)
                    if obj.data.uv_layers:obj.data.uv_layers.active_index=max(0,min(uv_original,len(obj.data.uv_layers)-1))
                    for u in obj.data.uv_layers:u.active_render=u.name in uv_render
                obj.hide_set(hidden);obj.hide_render=hidden_render
        if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Saved source changed during bake')
    finally:
        scene.render.engine=engine
        for n,v in cycles.items():setattr(scene.cycles,n,v)
        for n,v in render.items():setattr(scene.render,n,v)
        for n,v in bake_props.items():setattr(scene.render.bake,n,v)
        scene.tool_settings.mesh_select_mode=select_mode;scene.tool_settings.use_uv_select_sync=sync
        bpy.ops.object.select_all(action='DESELECT')
        for o in selected:
            if o.name in view.objects:o.select_set(True)
        if active and active.name in view.objects:view.objects.active=active
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_hash,'blender':bpy.app.version_string,'objects':rows,
        'savedBlend':False,'status':'Baked runtime-copy channels; requires image/render/export review',
        'limits':['UV overlap uses finite samples, not exact full coverage','No MToon conversion or VRM export performed','Physical SSS, hair scattering and view-dependent highlights are not encoded as base color','Normal channel requires actual rendered comparison; nonempty data alone does not establish fidelity','Partial texture outputs may remain after a failed later domain; source Blend is never saved']}
    path=dest/'material-bake-report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report


def bake_shared_channel(objects,material_name,channel='normal',resolution=1024,output_relative='work/runtime-r06/shared-atlases'):
    """One actual shared atlas for single-material objects; serial CPU baking.

    For the observed constant-color blouse/skirt/sock surfaces use normal.
    For both iris meshes use basecolor to evaluate their actual Multiply node.
    Original per-object UV coordinates are kept for source texture evaluation;
    new runtime UV islands are packed together. Every other linked channel is
    rejected so changing the active UV cannot silently disturb another atlas.
    """
    source,root=_guard();objects=_objects(objects)
    if channel not in {'normal','basecolor'} or not 256<=resolution<=2048:raise ValueError('Bounded normal/basecolor shared atlas required')
    originals=[]
    for obj in objects:
        if len(obj.data.materials)!=1 or obj.data.materials[0].name!=material_name:raise ValueError('Explicit same single material required: '+obj.name)
        material=obj.data.materials[0];out,shader=_surface(material)
        if _root_mix(material,out):raise ValueError('Use full alpha-aware bake for root substrate')
        allowed={'Normal'} if channel=='normal' else {'Base Color'}
        for socket in shader.inputs:
            if socket.is_linked and socket.name not in allowed:raise ValueError('Another linked channel would need its source UV retained: '+material.name+'/'+socket.name)
        originals.append(material)
    dest=_within_output(output_relative,root);filename=dest/(_slug(material_name)+'-'+channel+'.png')
    if filename.exists():raise FileExistsError('Use a fresh shared-atlas output directory')
    scene=bpy.context.scene;selected=list(bpy.context.selected_objects);active=bpy.context.view_layer.objects.active
    engine=scene.render.engine;device=scene.cycles.device;samples=scene.cycles.samples;threads=(scene.render.threads_mode,scene.render.threads)
    mode=tuple(scene.tool_settings.mesh_select_mode);sync=scene.tool_settings.use_uv_select_sync
    states=[];copies=[];uv_originals=[];complete=False
    try:
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=1;scene.render.threads_mode='FIXED';scene.render.threads=6
        scene.tool_settings.mesh_select_mode=(False,False,True);scene.tool_settings.use_uv_select_sync=False
        for obj,material in zip(objects,originals):
            old=next((u for u in obj.data.uv_layers if u.active_render),obj.data.uv_layers.active)
            uv_originals.append(old.name if old else None)
            copy=material.copy();_freeze_uv(copy,old.name if old else None);obj.data.materials[0]=copy;copies.append(copy)
            domains=_domains(obj,resolution,None);_prepare_uv(obj,domains,True)
        # Pack all objects in one edit session; no join or topology conversion.
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:obj.hide_set(False);obj.select_set(True)
        bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT');bpy.ops.uv.pack_islands(rotate=True,margin=12/resolution)
        bpy.ops.object.mode_set(mode='OBJECT')
        image=bpy.data.images.new(material_name+' / shared '+channel,width=resolution,height=resolution,alpha=True)
        image.generated_color=(0,0,0,0);image.colorspace_settings.name='sRGB' if channel=='basecolor' else 'Non-Color'
        for obj,material in zip(objects,copies):
            _select(obj);state=_channel(material,channel,image);states.append((material,state))
            try:
                result=bpy.ops.object.bake(type='NORMAL' if channel=='normal' else 'EMIT',normal_space='TANGENT',
                    uv_layer=UV,use_selected_to_active=False,use_clear=(obj==objects[0]),margin=8,margin_type='EXTEND')
                if 'FINISHED' not in result:raise RuntimeError('Shared channel bake failed: '+obj.name)
            finally:_restore_channel(material,state);states=[]
        pixels=np.empty(resolution*resolution*4,np.float32);image.pixels.foreach_get(pixels)
        if not np.isfinite(pixels).all() or np.count_nonzero(pixels.reshape(-1,4)[:,3]>.5)<16:raise RuntimeError('Shared atlas has no finite covered pixels')
        image_report=_save_image(image,filename)
        shared=originals[0].copy();shared.name='RT06 shared '+material_name+' / '+channel
        nt=shared.node_tree;_,shader=_surface(shared);uv=nt.nodes.new('ShaderNodeUVMap');uv.uv_map=UV
        tex=nt.nodes.new('ShaderNodeTexImage');tex.image=image;nt.links.new(uv.outputs['UV'],tex.inputs['Vector'])
        if channel=='normal':
            normal=nt.nodes.new('ShaderNodeNormalMap');normal.space='TANGENT';normal.uv_map=UV
            nt.links.new(tex.outputs['Color'],normal.inputs['Color']);nt.links.new(normal.outputs['Normal'],shader.inputs['Normal'])
        else:nt.links.new(tex.outputs['Color'],shader.inputs['Base Color'])
        shared['r06_shared_channel_bake']=channel;shared['r06_source_material']=material_name
        for obj in objects:obj.data.materials[0]=shared
        report={'source':str(source.relative_to(project_path())),'sourceSha256':_sha_local(source),'objects':[o.name for o in objects],
            'sourceMaterial':material_name,'material':shared.name,'channel':channel,'image':image_report,
            'preserved':'Other constant PBR inputs including metallic, coat, IOR and source UV layers',
            'uvProbe':[_uv_probe(o.data,o.data.uv_layers[UV]) for o in objects],
            'scope':'One atlas shared across explicitly packed objects; actual rendered/material comparison still required',
            'blendSaved':False}
        _write_report=dest/(_slug(material_name)+'-'+channel+'-report.json');_write_report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        complete=True;return report
    finally:
        for material,state in states:_restore_channel(material,state)
        if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        if not complete:
            for i,obj in enumerate(objects):
                obj.data.materials[0]=originals[i]
                layer=obj.data.uv_layers.get(UV)
                if layer:obj.data.uv_layers.remove(layer)
                if i<len(uv_originals) and uv_originals[i]:
                    layer=obj.data.uv_layers.get(uv_originals[i])
                    if layer:obj.data.uv_layers.active_index=obj.data.uv_layers.find(layer.name);layer.active_render=True
        scene.render.engine=engine;scene.cycles.device=device;scene.cycles.samples=samples
        scene.render.threads_mode,scene.render.threads=threads;scene.tool_settings.mesh_select_mode=mode;scene.tool_settings.use_uv_select_sync=sync
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:
            if obj.name in bpy.context.view_layer.objects:obj.select_set(True)
        if active and active.name in bpy.context.view_layer.objects:bpy.context.view_layer.objects.active=active

def _within_output(relative,root):
    path=(project_path()/relative).resolve();path.relative_to(root);path.mkdir(parents=True,exist_ok=True);return path

def _sha_local(path):return hashlib.sha256(path.read_bytes()).hexdigest()
