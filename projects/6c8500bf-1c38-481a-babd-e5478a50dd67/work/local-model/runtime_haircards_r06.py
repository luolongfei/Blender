"""Build a bounded hair-card preview from the actual R06 authored groom.

Explicitly run in a disposable background copy under work/runtime-r06/. Does
not save a Blend, register VRM springs, export, or certify appearance. Original
CURVES remain in that copy, hidden and tagged with their replacement objects.
Source point data is never rewritten. V2 alpha maps rasterize actual native
strand paths/radii in each card's own UV projection, with separate regional
atlases. This is a geometry coverage bake, not a Cycles strand-shader bake.
"""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix,Vector
except ImportError:
    bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PREFIX='RT06 '
SPECS={
 'R06 continuous scalp flow':{'groups':[32]*192+[12]*48,'samples':28,'cards':2,'mode':'fixed','region':'scalp'},
 'R06 full gathered roots L':{'groups':[310]*8,'samples':30,'cards':3,'mode':'fixed','region':'gather'},
 'R06 full gathered roots R':{'groups':[310]*8,'samples':30,'cards':3,'mode':'fixed','region':'gather'},
 'R03 Forelocks from crown':{'groups':[480]*7,'samples':40,'cards':3,'mode':'fixed','region':'fringe'},
 'R03 curved cheek wisps':{'groups':[105]*6,'samples':44,'cards':2,'mode':'fixed','region':'wisps'},
 'R02 Strands / R02 Twin tail L':{'groups':[310]*8,'samples':35,'cards':3,'mode':'spring','side':'L','region':'tail'},
 'R02 Strands / R02 Twin tail R':{'groups':[310]*8,'samples':35,'cards':3,'mode':'spring','side':'R','region':'tail'},
}
ATLAS_REGIONS={
 'scalp':{'tile':(64,128),'color':(.0060,.0052,.0048),'roughness':.64},
 'gather':{'tile':(96,256),'color':(.0060,.0052,.0048),'roughness':.62},
 'fringe':{'tile':(192,512),'color':(.0060,.0052,.0048),'roughness':.64},
 'wisps':{'tile':(96,512),'color':(.0060,.0052,.0048),'roughness':.65},
 'tail':{'tile':(128,512),'color':(.0060,.0052,.0048),'roughness':.62},
}

def resolved_spec(obj):
    """Use explicit appended R09 cohorts; never infer strand ordering spatially.

    Earlier cohorts keep their exact index prefix. An unannotated object keeps
    the locked R06 grouping and still fails on an unexpected native count.
    """
    base=SPECS[obj.name];spec={**base,'groups':list(base['groups'])}
    raw=obj.get('RT09_runtime_group_counts')
    if raw is None:return spec
    if obj.name!='R06 continuous scalp flow':raise RuntimeError('R09 cohort metadata is only approved for continuous scalp flow')
    if not isinstance(raw,str):raise RuntimeError('R09 group metadata must be an explicit JSON string')
    groups=json.loads(raw)
    if not isinstance(groups,list) or not groups or any(type(n) is not int or n<1 or n>512 for n in groups):
        raise RuntimeError('Invalid R09 native cohort sizes')
    if groups[:len(base['groups'])]!=base['groups']:
        raise RuntimeError('R09 scalp cohorts must retain the original 240-group prefix')
    if len(groups)>2048 or sum(groups)!=len(obj.data.curves):
        raise RuntimeError('R09 metadata group total does not match native curve count')
    provenance=obj.get('RT09_runtime_group_provenance','')
    if not isinstance(provenance,str) or not provenance.strip():raise RuntimeError('R09 cohort provenance is required')
    spec['groups']=groups;spec['groupProvenance']=provenance
    return spec

def _unit(v):
    n=float(np.linalg.norm(v))
    if n<1e-9:raise RuntimeError('Degenerate card frame')
    return v/n

def _sample(path,t):
    u=np.asarray(t)*(len(path)-1);i=np.minimum(u.astype(int),len(path)-2);f=u-i
    return path[i]*(1-f[:,None])+path[i+1]*f[:,None]

def card_packet(paths,radii,spec):
    """Pure numeric kernel. All input coordinates are rig-local rest space."""
    if len(paths)!=sum(spec['groups']):raise RuntimeError('Unexpected strand group ordering/count')
    vertices=[];faces=[];uv=[];longitudinal=[];stats=[];tiles=[];offset=0
    t=np.linspace(0,1,spec['samples'])
    for group,count in enumerate(spec['groups']):
        strands=np.array([_sample(path,t) for path in paths[offset:offset+count]])
        mean=strands.mean(axis=0);tangent=np.gradient(mean,axis=0)
        tangent=np.array([_unit(v) for v in tangent]);deviation=strands-mean[None,:,:]
        axis=[]
        # Parallel transport avoids arbitrary PCA sign flips and winding flips.
        first=np.array([1.,0.,0.]);first-=tangent[0]*np.dot(first,tangent[0])
        if np.linalg.norm(first)<.05:first=np.cross(tangent[0],[0.,0.,1.])
        axis.append(_unit(first))
        for j in range(1,len(t)):
            v=axis[-1]-tangent[j]*np.dot(axis[-1],tangent[j])
            if np.linalg.norm(v)<.05:v=np.cross(tangent[j],[0.,0.,1.])
            axis.append(_unit(v))
        axis=np.array(axis);other=np.cross(tangent,axis)
        max_radius=max(float(np.max(r)) for r in radii[offset:offset+count])
        for card in range(spec['cards']):
            angle=math.pi*card/spec['cards'];across=axis*math.cos(angle)+other*math.sin(angle)
            projection=np.einsum('nsd,sd->ns',deviation,across)
            low=np.quantile(projection,.005,axis=0)-max_radius
            high=np.quantile(projection,.995,axis=0)+max_radius
            # Keep tiny roots and ends visible while respecting measured width.
            mid=(low+high)*.5;half=np.maximum((high-low)*.5,max_radius*1.1)
            start=len(vertices)
            for j in range(len(t)):
                vertices.extend([mean[j]+across[j]*(mid[j]-half[j]),mean[j]+across[j]*(mid[j]+half[j])])
                uv.extend([(0,float(t[j])),(1,float(t[j]))]);longitudinal.extend([float(t[j])]*2)
            for j in range(len(t)-1):
                i=start+j*2;faces.append((i,i+1,i+3,i+2))
            stats.append({'group':group,'card':card,'maxWidthMeters':float((half*2).max())})
            tiles.append({'group':group,'card':card,'first_vertex':start,
                          'ribbon':np.asarray(vertices[start:start+len(t)*2],dtype=float).reshape(-1,2,3),
                          'paths':paths[offset:offset+count],'radii':radii[offset:offset+count]})
        offset+=count
    array=np.asarray(vertices,dtype=np.float32)
    if not np.all(np.isfinite(array)):raise RuntimeError('Nonfinite card geometry')
    return {'vertices':array,'faces':faces,'uv':uv,'t':longitudinal,'stats':stats,
            'triangles':len(faces)*2,'projection_tiles':tiles,'region':spec.get('region','scalp')}

def _guard():
    if bpy is None or not bpy.app.background:raise RuntimeError('Use a disposable background Blender process')
    root=(project_path()/'work/runtime-r06').resolve();file=Path(bpy.data.filepath).resolve()
    file.relative_to(root)
    if not file.is_file():raise RuntimeError('Saved runtime copy required')
    if bpy.context.mode!='OBJECT':raise RuntimeError('Object mode required')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.type!='ARMATURE':raise RuntimeError('Character_Rig missing')
    if any(o.name.startswith(PREFIX+'Cards / ') for o in bpy.data.objects):raise RuntimeError('Cards already built')
    return rig

def _read(obj,rig,allow_hidden=False):
    if obj.type!='CURVES' or (obj.hide_render and not allow_hidden):raise RuntimeError('Expected native groom: '+obj.name)
    if any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Disable/restore native dynamic modifiers first: '+obj.name)
    data=np.empty(len(obj.data.position_data)*3,np.float32);obj.data.position_data.foreach_get('vector',data);data=data.reshape(-1,3)
    matrix=np.array(rig.matrix_world.inverted()@obj.matrix_world)
    data=data@matrix[:3,:3].T+matrix[:3,3]
    radius=np.empty(len(obj.data.position_data),np.float32);obj.data.attributes['radius'].data.foreach_get('value',radius)
    paths=[];radii=[]
    for curve in obj.data.curves:
        start=curve.first_point_index;end=start+curve.points_length
        if curve.points_length<4:raise RuntimeError('Short native strand')
        paths.append(data[start:end]);radii.append(radius[start:end])
    return paths,radii

def _opacity_material(folder):
    """Deterministic analytic strand mask; explicitly not a strand-shader bake."""
    folder.mkdir(parents=True,exist_ok=True);path=folder/'hair-preview-analytic.png'
    if path.exists():raise RuntimeError('Refuse overwriting an existing preview texture')
    width,height=512,1024;x=np.linspace(0,1,width)[None,:];y=np.linspace(0,1,height)[:,None]
    rng=np.random.default_rng(606104);density=np.zeros((height,width),np.float32)
    for strand in range(68):
        base=(strand+.5)/68+float(rng.uniform(-.004,.004));width_u=float(rng.uniform(.0013,.0038))
        wav=.0018*np.sin(y*float(rng.uniform(4,11))+float(rng.uniform(0,6.28)))
        tip=float(rng.uniform(.80,1.01));taper=np.clip((tip-y)/.12,0,1)
        density=np.maximum(density,np.exp(-((x-base-wav)/width_u)**2)*taper)
    edge=np.clip(x/.035,0,1)*np.clip((1-x)/.035,0,1)
    alpha=np.clip(density*edge,0,1)
    color=.78+.22*np.sin(x*math.tau*29+y*1.7)**2
    pixels=np.empty((height,width,4),np.float32)
    for c,value in enumerate((.065,.052,.043)):pixels[:,:,c]=value*color
    pixels[:,:,3]=alpha
    image=bpy.data.images.new(PREFIX+'Hair analytic opacity preview',width,height,alpha=True)
    image.pixels.foreach_set(pixels.ravel());image.filepath_raw=str(path);image.file_format='PNG';image.save();image.pack()
    mat=bpy.data.materials.new(PREFIX+'Hair card preview');mat.use_nodes=True
    nt=mat.node_tree;bs=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED')
    tex=nt.nodes.new('ShaderNodeTexImage');tex.image=image
    nt.links.new(tex.outputs['Color'],bs.inputs['Base Color']);nt.links.new(tex.outputs['Alpha'],bs.inputs['Alpha'])
    bs.inputs['Roughness'].default_value=.56
    if bs.inputs.get('Specular IOR Level'):bs.inputs['Specular IOR Level'].default_value=.30
    if hasattr(mat,'use_backface_culling'):mat.use_backface_culling=False
    if hasattr(mat,'surface_render_method'):
        enums={e.identifier for e in mat.bl_rna.properties['surface_render_method'].enum_items}
        if 'DITHERED' in enums:mat.surface_render_method='DITHERED'
    mat['RT06_analytic_preview_material']=True
    mat['r06_preserve_alpha_atlas']=True
    mat['RT06_material_limit']='Shared analytic opacity/color, not an exact bake of native strand anisotropy or per-fibre shading. Requires multi-angle visual review.'
    return mat,{'path':str(path.relative_to(project_path())),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':width,'height':height,'provenance':'New analytic mask; not scanned/baked strand texture'}

def projected_opacity(tile,width,height,supersample=4):
    """Pure numeric, per-card coverage raster of actual 3D fibres and radii.

    The ribbon was constructed by sampling each source strand at the same
    normalized point parameter (_sample in card_packet). Preserve that exact
    root-to-tip correspondence here: a nearest-spine search can jump to an
    unrelated segment where a short scalp path folds back near itself. Only
    the transverse coordinate is projected geometrically onto the actual
    ribbon. Every source point and radius participates; there is no monotone
    clamp, discarded reversal, or shared tip fade. Four horizontal subsamples
    preserve thin fibres. This is coverage, not shader baking.
    """
    ribbon=np.asarray(tile['ribbon'],dtype=float)
    if ribbon.ndim!=3 or ribbon.shape[1:]!=(2,3) or len(ribbon)<2 or not np.isfinite(ribbon).all():
        raise RuntimeError('Malformed/nonfinite source ribbon')
    spine=ribbon.mean(axis=1);across=ribbon[:,1]-ribbon[:,0]
    if len(tile['paths'])!=len(tile['radii']):
        raise RuntimeError('Source path/radius counts differ')
    high_width=width*supersample
    alpha=np.zeros((height,high_width),dtype=np.float32)
    rows_v=(np.arange(height)+.5)/height
    used=0;visible=0;outside_points=0;source_points=0;max_outside_u=0.
    for fibre,(path,radii) in enumerate(zip(tile['paths'],tile['radii'])):
        path=np.asarray(path,dtype=float);radii=np.asarray(radii,dtype=float)
        if path.ndim!=2 or path.shape[1:]!=(3,) or len(path)<2 or radii.shape!=(len(path),):
            raise RuntimeError(f'Malformed source fibre {fibre}: path {path.shape}, radii {radii.shape}')
        if not np.isfinite(path).all() or not np.isfinite(radii).all() or np.any(radii<0):
            raise RuntimeError(f'Nonfinite/negative source fibre data: {fibre}')
        # Point parameter, rather than arclength, intentionally matches the
        # existing card generator even when source segment lengths differ.
        v=np.linspace(0.,1.,len(path))
        center=_sample(spine,v);cross=_sample(across,v)
        cross_width=np.linalg.norm(cross,axis=1)
        if np.any(cross_width<1e-7):
            point=int(np.argmin(cross_width))
            raise RuntimeError(f'Degenerate ribbon cross-section for fibre {fibre}, point {point}: {cross_width[point]:.9g} m')
        normal=cross/cross_width[:,None]
        u=.5+((path-center)*normal).sum(axis=1)/cross_width
        radius_u=radii/cross_width
        outside=np.maximum.reduce((-u,u-1,np.zeros_like(u)))
        outside_points+=int(np.count_nonzero(outside));source_points+=len(path)
        max_outside_u=max(max_outside_u,float(outside.max(initial=0)))
        selected=np.flatnonzero((rows_v>=v[0]-.5/height)&(rows_v<=v[-1]+.5/height))
        if not len(selected):raise RuntimeError(f'No atlas rows for source fibre {fibre}')
        c=np.interp(rows_v[selected],v,u)*high_width-.5
        r=np.maximum(0,np.interp(rows_v[selected],v,radius_u)*high_width)
        vertical=(np.clip((rows_v[selected]-v[0])*height+.5,0,1)
                  *np.clip((v[-1]-rows_v[selected])*height+.5,0,1))
        # Bin wide tip rows separately so a tiny collapsed ribbon end does not
        # force a full-width allocation for every row of every native fibre.
        lower=-1.;has_coverage=False
        for upper in (2.,8.,32.,float('inf')):
            pick=np.flatnonzero((r>lower)&(r<=upper));lower=upper
            if not len(pick):continue
            extent=min(high_width,int(math.ceil(float(r[pick].max())))+1)
            offsets=np.arange(-extent,extent+1)
            columns=np.floor(c[pick]).astype(int)[:,None]+offsets[None,:]
            left=np.maximum(columns-.5,c[pick,None]-r[pick,None])
            right=np.minimum(columns+.5,c[pick,None]+r[pick,None])
            coverage=np.clip(right-left,0,1)*vertical[pick,None]
            valid=(columns>=0)&(columns<high_width)&(coverage>0)
            has_coverage=has_coverage or bool(valid.any())
            rr=np.broadcast_to(selected[pick,None],columns.shape)
            np.maximum.at(alpha,(rr[valid],columns[valid]),coverage[valid])
        used+=1;visible+=int(has_coverage)
    alpha=alpha.reshape(height,width,supersample).mean(axis=2)
    core=alpha[int(height*.15):int(height*.85),int(width*.2):int(width*.8)]
    return alpha,{'actual_fibres':len(tile['paths']),'rasterized_fibres':used,
                  'visible_fibres':visible,'source_points':source_points,
                  'transverse_outside_points':outside_points,'max_transverse_outside_uv':max_outside_u,
                  'core_mean_alpha':float(core.mean()),'overall_mean_alpha':float(alpha.mean()),
                  'last_10_percent_mean_alpha':float(alpha[int(height*.9):].mean()),
                  'longitudinal_mapping':'normalized source point parameter, identical to card construction',
                  'discarded_source_points':0}

def _fresh_atlas_folder(folder):
    folder.mkdir(parents=True,exist_ok=True)
    candidate=folder/'source-projection-v2';number=2
    while candidate.exists():
        candidate=folder/f'source-projection-v2-{number:02d}';number+=1
    candidate.mkdir();return candidate

def _projected_materials(folder,packets):
    """One actual-coverage tile per existing card; five separate regional maps."""
    folder=_fresh_atlas_folder(folder)
    regions={name:[] for name in ATLAS_REGIONS}
    for source,packet in packets.items():
        for tile in packet['projection_tiles']:
            regions[packet['region']].append((source,packet,tile))
    materials={};metadata=[]
    for region,items in regions.items():
        if not items:continue
        config=ATLAS_REGIONS[region];w,h=config['tile'];pad=3
        columns=int(math.ceil(math.sqrt(len(items)*(h+pad*2)/(w+pad*2))))
        rows=int(math.ceil(len(items)/columns))
        width=columns*(w+pad*2);height=rows*(h+pad*2)
        pixels=np.empty((height,width,4),dtype=np.float32)
        pixels[:,:,:3]=config['color'];pixels[:,:,3]=0
        tile_reports=[]
        for index,(source,packet,tile) in enumerate(items):
            try:
                alpha,coverage=projected_opacity(tile,w,h)
            except RuntimeError as error:
                raise RuntimeError(f'{source} group {tile["group"]} card {tile["card"]}: {error}') from error
            x=(index%columns)*(w+pad*2)+pad;y=(index//columns)*(h+pad*2)+pad
            pixels[y:y+h,x:x+w,3]=alpha
            # Very restrained colour variation; coverage, not brightness, forms
            # the fibres. This stays close to the .0045–.007 native root/hair base.
            stripe=.94+.06*np.sin(np.arange(w)[None,:]*.47+index*.71)**2
            pixels[y:y+h,x:x+w,:3]=np.asarray(config['color'])[None,None,:]*stripe[:,:,None]
            start=tile['first_vertex'];samples=len(tile['ribbon'])
            for j in range(samples):
                t=j/(samples-1)
                packet['uv'][start+j*2]=((x+.5)/width,(y+.5+t*(h-1))/height)
                packet['uv'][start+j*2+1]=((x+w-.5)/width,(y+.5+t*(h-1))/height)
            tile_reports.append({'source':source,'group':tile['group'],'card':tile['card'],**coverage})
        path=folder/f'hair-{region}-projected.png'
        image=bpy.data.images.new(PREFIX+'Hair projected '+region,width,height,alpha=True)
        image.pixels.foreach_set(pixels.ravel());image.filepath_raw=str(path)
        image.file_format='PNG';image.save();image.pack()
        material=bpy.data.materials.new(PREFIX+'Hair projected '+region);material.use_nodes=True
        tree=material.node_tree;bs=next(n for n in tree.nodes if n.type=='BSDF_PRINCIPLED')
        texture=tree.nodes.new('ShaderNodeTexImage');texture.image=image;texture.interpolation='Linear';texture.extension='EXTEND'
        tree.links.new(texture.outputs['Color'],bs.inputs['Base Color'])
        tree.links.new(texture.outputs['Alpha'],bs.inputs['Alpha'])
        bs.inputs['Roughness'].default_value=config['roughness']
        if bs.inputs.get('Specular IOR Level'):bs.inputs['Specular IOR Level'].default_value=.25
        if bs.inputs.get('Anisotropic IOR Level'):bs.inputs['Anisotropic IOR Level'].default_value=.35
        material.use_backface_culling=False
        if hasattr(material,'surface_render_method'):
            enums={e.identifier for e in material.bl_rna.properties['surface_render_method'].enum_items}
            if 'DITHERED' in enums:material.surface_render_method='DITHERED'
        material['r06_preserve_alpha_atlas']=True
        material['RT06_source_projected_atlas']=True
        material['RT06_hair_region']=region
        material['RT06_material_limit']='Per-card actual native path/radius coverage; not a native hair BSDF lighting bake. Multi-angle alpha sorting and spring review remain required.'
        materials[region]=material
        metadata.append({'region':region,'path':str(path.relative_to(project_path())),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'width':width,'height':height,'tileWidth':w,'tileHeight':h,
                         'tiles':len(items),'colorLinear':list(config['color']),
                         'coverage':tile_reports,
                         'provenance':'Rasterized actual native strand paths and radii in each card UV projection; no shared analytic tip fade'})
        del pixels
    return materials,{'version':2,'regions':metadata,'folder':str(folder.relative_to(project_path()))}

def _tail_bones(rig,tail_paths):
    bone_names={};points={}
    for side,paths in tail_paths.items():
        samples=np.array([_sample(path,np.linspace(0,1,35)) for path in paths]);mean=samples.mean(axis=0)
        points[side]=mean[[2,7,13,19,25,30,34]]
        bone_names[side]=[PREFIX+f'tail_{side}_{i:02d}' for i in range(6)]
        if any(n in rig.data.bones for n in bone_names[side]):raise RuntimeError('Tail bones already exist')
    rig.hide_set(False);bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    for side,names in bone_names.items():
        parent=rig.data.edit_bones['head']
        for i,name in enumerate(names):
            bone=rig.data.edit_bones.new(name);bone.head=points[side][i];bone.tail=points[side][i+1]
            bone.parent=parent;bone.use_connect=i>0;bone.use_deform=True;parent=bone
    bpy.ops.object.mode_set(mode='OBJECT')
    return bone_names

def _weights(obj,rig,packet,side,bone_names):
    head=obj.vertex_groups.new(name='head')
    if side is None:head.add(list(range(len(obj.data.vertices))),1,'REPLACE')
    else:
        names=bone_names[side];groups=[head]+[obj.vertex_groups.new(name=n) for n in names]
        nodes=np.array([2,7,13,19,25,30,34],dtype=float)/34
        for i,t in enumerate(packet['t']):
            if t<=2/34:head.add([i],1,'REPLACE');continue
            j=min(len(nodes)-1,int(np.searchsorted(nodes,t)-1))
            if j==len(nodes)-1:groups[j].add([i],1,'REPLACE')
            else:
                w=(t-nodes[j])/(nodes[j+1]-nodes[j]);groups[j].add([i],1-w,'REPLACE');groups[j+1].add([i],w,'REPLACE')
    mod=obj.modifiers.new(PREFIX+'Skin','ARMATURE');mod.object=rig;mod.use_deform_preserve_volume=False

def build(texture_relative='work/runtime-r06/textures'):
    """Create preview meshes/bones/materials in memory, no automatic save/export."""
    rig=_guard();p=project_path();folder=(p/texture_relative).resolve();folder.relative_to((p/'work/runtime-r06').resolve())
    scene=bpy.context.scene;old_pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    packets={};paths_by_tail={};source_hashes={}
    try:
        for name,spec in SPECS.items():
            obj=bpy.data.objects.get(name)
            if obj is None:raise RuntimeError('Current R06 native groom required: '+name)
            spec=resolved_spec(obj)
            paths,radii=_read(obj,rig);packets[name]=card_packet(paths,radii,spec)
            source_hashes[name]=hashlib.sha256(np.concatenate(paths).astype(np.float32).tobytes()).hexdigest()
            if spec.get('side'):paths_by_tail[spec['side']]=paths
        materials,texture=_projected_materials(folder,packets);bones=_tail_bones(rig,paths_by_tail)
        collection=bpy.data.collections.new(PREFIX+'Hair card preview');scene.collection.children.link(collection)
        rows=[];targets=[]
        for name,spec in SPECS.items():
            packet=packets[name];mesh=bpy.data.meshes.new(PREFIX+'Cards data / '+name)
            mesh.from_pydata(packet['vertices'].tolist(),[],packet['faces']);mesh.update()
            uv=mesh.uv_layers.new(name='RuntimeHairUV')
            for face in mesh.polygons:
                face.use_smooth=True
                for li in face.loop_indices:uv.data[li].uv=packet['uv'][mesh.loops[li].vertex_index]
            obj=bpy.data.objects.new(PREFIX+'Cards / '+name,mesh);collection.objects.link(obj)
            obj.matrix_world=rig.matrix_world.copy();mesh.materials.append(materials[spec['region']]);obj['part']='hair'
            obj['RT06_source_native']=name;obj['RT06_source_points_sha256']=source_hashes[name]
            _weights(obj,rig,packet,spec.get('side'),bones)
            original=bpy.data.objects[name];original['RT06_replaced_by']=obj.name;original.hide_render=True;original.hide_set(True)
            actual_spec=resolved_spec(original)
            row={'source':name,'mesh':obj.name,'triangles':packet['triangles'],'cards':len(packet['stats']),'sourcePointSha256':source_hashes[name],'mode':spec['mode'],
                 'groups':actual_spec['groups'],'groupProvenance':actual_spec.get('groupProvenance','Locked R06 authored cohort ordering')}
            rows.append(row)
            target={'mesh':obj.name,'source':'hair','mode':spec['mode']}
            if spec['mode']=='spring':target['chains']=[PREFIX+'tail '+spec['side']]
            else:target['reason']='Scalp-attached short/gathered part preserved as fixed for this explicit preview; native source has head attachment, no separate dynamics for this part.'
            targets.append(target)
        chains=[{'name':PREFIX+'tail '+side,'bones':names,'stiffness':.7,'gravityPower':.12,'gravityDir':[0,0,-1],'dragForce':.42,'hitRadius':.014,'colliderGroups':[]} for side,names in bones.items()]
        report={'sourceRuntimeCopy':bpy.data.filepath,'sourceUnchangedOnDisk':True,'meshes':rows,'triangles':sum(x['triangles'] for x in rows),'texture':texture,
                'springChainProposal':chains,'springTargets':targets,'springRegistrationPerformed':False,
                'scope':'Hair-card and bone construction only. Need collider selection, actual VRM spring registration, ON/OFF response, render and collision review.',
                'limitations':['Native strand anisotropy, fine flyaways, native PBD and Blender GN caches are not preserved','Per-card source-projected coverage is not a source-shader lighting bake; crossed cards still need multi-angle sorting review','Scalp, gathered roots, forelocks and wisps are fixed to head in this preview']}
        (p/'work/runtime-r06/haircards-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        rig['RT06_haircard_report']='work/runtime-r06/haircards-report.json';print(json.dumps({'meshes':len(rows),'triangles':report['triangles'],'savedBlend':False}));return report
    finally:
        rig.data.pose_position=old_pose;bpy.context.view_layer.update()

def upgrade_atlases(texture_relative='work/runtime-r06/textures'):
    """Replace existing prototype UV/materials using its exact current cards.

    No card positions, weights, bones, modifiers, source points, native flags or
    spring configuration changes. Old meshes/materials/images remain retained.
    Does not save, export, render, or convert the new PBR materials to MToon.
    """
    if bpy is None:raise RuntimeError('The saved runtime copy is required')
    p=project_path();file=Path(bpy.data.filepath).resolve()
    file.relative_to((p/'work/runtime-r06').resolve())
    if not file.is_file():raise RuntimeError('Saved runtime copy required')
    if bpy.context.mode!='OBJECT':raise RuntimeError('Object mode required')
    rig=bpy.data.objects['Character_Rig'];old_pose=rig.data.pose_position
    folder=(p/texture_relative).resolve();folder.relative_to((p/'work/runtime-r06').resolve())
    objects={}
    for name in SPECS:
        obj=bpy.data.objects.get(PREFIX+'Cards / '+name)
        if obj is None or obj.type!='MESH':raise RuntimeError('Existing unjoined prototype cards required: '+name)
        if obj.get('RT06_projection_v2_backup'):raise RuntimeError('Projected upgrade already applied; restore_atlases() before retrying')
        if obj.data.users!=1:raise RuntimeError('Single-user runtime card data required')
        objects[name]=obj
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    try:
        packets={};source_hashes={}
        for name,spec in SPECS.items():
            spec=resolved_spec(bpy.data.objects[name])
            paths,radii=_read(bpy.data.objects[name],rig,allow_hidden=True)
            packet=card_packet(paths,radii,spec)
            obj=objects[name]
            if len(obj.data.vertices)!=len(packet['vertices']):
                raise RuntimeError('Card topology changed; rebuild from the new runtime copy: '+name)
            co=np.empty(len(obj.data.vertices)*3,dtype=np.float32)
            obj.data.vertices.foreach_get('co',co);co=co.reshape(-1,3)
            matrix=np.array(rig.matrix_world.inverted()@obj.matrix_world)
            co=co@matrix[:3,:3].T+matrix[:3,3]
            # UV coverage is projected onto the actual existing mesh. The
            # reconstructed numeric packet only supplies validated group order.
            for tile in packet['projection_tiles']:
                start=tile['first_vertex'];length=len(tile['ribbon'])*2
                tile['ribbon']=co[start:start+length].reshape(-1,2,3)
            packets[name]=packet
            source_hashes[name]=hashlib.sha256(np.concatenate(paths).astype(np.float32).tobytes()).hexdigest()
        materials,texture=_projected_materials(folder,packets)
        report={'sourceRuntimeCopy':str(file),'texture':texture,'meshes':[],
                'geometryWeightsBonesModified':False,'nativeSourceModified':False,
                'savedBlend':False,'materialPipeline':'New region PBR sources; apply the separate MToon pipeline afterwards if required',
                'scope':'Actual path/radius projected alpha; no shared analytic tip fade. Requires actual front/quarter/back render and VRM alpha review.'}
        for name,obj in objects.items():
            original=obj.data;original.use_fake_user=True
            mesh=original.copy();mesh.name=original.name+' projected-alpha-v2'
            uv=mesh.uv_layers.get('RuntimeHairUV') or mesh.uv_layers.new(name='RuntimeHairUV')
            for face in mesh.polygons:
                for li in face.loop_indices:uv.data[li].uv=packets[name]['uv'][mesh.loops[li].vertex_index]
            old_materials=[m.name for m in original.materials if m]
            for material in original.materials:
                if material:material.use_fake_user=True
            mesh.materials.clear();mesh.materials.append(materials[SPECS[name]['region']])
            for face in mesh.polygons:face.material_index=0
            obj['RT06_projection_v2_backup']=original.name
            obj['RT06_projection_v2_prior_source_hash']=obj.get('RT06_source_points_sha256','')
            obj['RT06_source_projected_atlas']=True;obj.data=mesh
            obj['RT06_source_points_sha256']=source_hashes[name]
            mesh.update_tag();obj.update_tag()
            report['meshes'].append({'source':name,'mesh':obj.name,'region':SPECS[name]['region'],
                                     'previousMaterials':old_materials,'retainedMesh':original.name,
                                     'sourcePointSha256':source_hashes[name]})
        path=p/texture['folder']/'upgrade-report.json'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        rig['RT06_haircard_projection_report']=str(path.relative_to(p))
        bpy.context.view_layer.update();print(json.dumps({'updated':len(objects),'report':str(path),'savedBlend':False}));return report
    finally:
        rig.data.pose_position=old_pose;bpy.context.view_layer.update()

def restore_atlases():
    """Restore retained prototype meshes/material slots; keep v2 assets too."""
    restored=[]
    for name in SPECS:
        obj=bpy.data.objects.get(PREFIX+'Cards / '+name)
        if not obj or not obj.get('RT06_projection_v2_backup'):continue
        old=bpy.data.meshes.get(obj['RT06_projection_v2_backup'])
        if old is None:raise RuntimeError('Retained prototype mesh missing: '+name)
        obj.data.use_fake_user=True;obj.data=old;old.use_fake_user=False
        del obj['RT06_projection_v2_backup']
        if 'RT06_projection_v2_prior_source_hash' in obj:
            obj['RT06_source_points_sha256']=obj['RT06_projection_v2_prior_source_hash']
            del obj['RT06_projection_v2_prior_source_hash']
        if 'RT06_source_projected_atlas' in obj:del obj['RT06_source_projected_atlas']
        restored.append(obj.name)
    bpy.context.view_layer.update();return restored

if __name__=='__main__':
    print('Loaded only. Call build() in a fresh saved runtime copy, or upgrade_atlases() for existing RT06 cards. Neither saves or renders.')
