"""True Cycles tangent-normal bake on E4's actual crown mesh.

Only a copied cap material and one new packed Non-Color texture are installed.
Original geometry, UVs, colour map, roughness and every other object stay exact.
"""
import hashlib,struct,math,json,sys
from pathlib import Path


def hashes(obj):
    h=hashlib.sha256();u=hashlib.sha256()
    for v in obj.data.vertices:h.update(struct.pack('<3d',*v.co))
    for p in obj.data.polygons:
        h.update(struct.pack('<I?',len(p.vertices),p.use_smooth))
        for k in p.vertices:h.update(struct.pack('<I',k))
    for row in obj.matrix_world:
        for x in row:h.update(struct.pack('<d',x))
    for layer in obj.data.uv_layers:
        u.update(layer.name.encode())
        for loop in layer.data:u.update(struct.pack('<2d',*loop.uv))
    return {'geometry':h.hexdigest(),'uv':u.hexdigest()}


def _height_shader(mat,distance=.0014):
    n,l=mat.node_tree.nodes,mat.node_tree.links;n.clear()
    uv=n.new('ShaderNodeUVMap');uv.uv_map='HairUV'
    sep=n.new('ShaderNodeSeparateXYZ');l.new(uv.outputs['UV'],sep.inputs[0])
    def calc(op,a,b=None):
        nd=n.new('ShaderNodeMath');nd.operation=op
        for i,val in enumerate((a,b)):
            if val is None:continue
            if isinstance(val,(int,float)):nd.inputs[i].default_value=val
            else:l.new(val,nd.inputs[i])
        return nd.outputs[0]
    def clamp(v):return calc('MINIMUM',calc('MAXIMUM',v,0),1)
    def smooth(v):
        v=clamp(v);return calc('MULTIPLY',calc('MULTIPLY',v,v),calc('SUBTRACT',3,calc('MULTIPLY',2,v)))
    u,v=sep.outputs['X'],sep.outputs['Y']
    q=calc('MINIMUM',u,calc('SUBTRACT',1,u))
    denominator=calc('SUBTRACT',1,calc('MULTIPLY',.32,calc('POWER',v,1.7)))
    flow=calc('ADD',.33,calc('DIVIDE',calc('SUBTRACT',q,.33),denominator))
    broad=calc('COSINE',calc('MULTIPLY',2*math.pi*14,flow))
    fine_phase=calc('ADD',calc('MULTIPLY',2*math.pi*60,flow),calc('MULTIPLY',.3,calc('SINE',calc('MULTIPLY',2*math.pi,v))))
    fine=calc('COSINE',fine_phase)
    # Fade at the UV pole and shell lip, where tangents become singular.
    fade=calc('MULTIPLY',smooth(calc('DIVIDE',calc('SUBTRACT',v,.025),.18)),smooth(calc('DIVIDE',calc('SUBTRACT',.995,v),.055)))
    height=calc('ADD',.5,calc('MULTIPLY',fade,calc('ADD',calc('MULTIPLY',.42,broad),calc('MULTIPLY',.045,fine))))
    bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=1
    bump.inputs['Distance'].default_value=distance;l.new(height,bump.inputs['Height'])
    bs=n.new('ShaderNodeBsdfPrincipled');bs.inputs['Base Color'].default_value=(.03,.028,.025,1)
    bs.inputs['Metallic'].default_value=0;bs.inputs['Roughness'].default_value=.56
    l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    out=n.new('ShaderNodeOutputMaterial');l.new(bs.outputs['BSDF'],out.inputs['Surface'])


def _enum_set(owner,name,value):
    allowed=[v.identifier for v in owner.bl_rna.properties[name].enum_items]
    if value not in allowed:raise RuntimeError(f'{name} does not support {value}: {allowed}')
    setattr(owner,name,value)


def bake_actual_cap(cap,path,spec,size=2048):
    import bpy
    from array import array
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if len(cap.data.materials)!=1 or 'HairUV' not in cap.data.uv_layers:
        raise RuntimeError('Unexpected cap material/UV structure')
    if cap.modifiers:raise RuntimeError('Cap modifiers require evaluated bake parity, not assumed')
    rows=len(spec['cap']['surface']);around=len(spec['cap']['surface'][0]);outer_count=(rows-1)*around+around
    if len(cap.data.polygons)!=2*(rows-1)*around+3*around:
        raise RuntimeError('Cannot identify actual outer/inner cap face domains')
    original=hashes(cap);scene=bpy.data.scenes.new('TEMP_ActualCrownNormalBake')
    old_scene=bpy.context.window.scene if bpy.context.window else None
    obj=cap.copy();obj.data=cap.data.copy();obj.name='TEMP_ActualCrownMesh';obj.animation_data_clear()
    scene.collection.objects.link(obj);obj.matrix_world=cap.matrix_world.copy();obj.hide_viewport=False;obj.hide_render=False
    obj.data.uv_layers.active=obj.data.uv_layers['HairUV'];obj.data.uv_layers['HairUV'].active_render=True
    outer=bpy.data.materials.new('TEMP_CrownFlowBump');outer.use_nodes=True;_height_shader(outer)
    inner=bpy.data.materials.new('TEMP_CrownInnerBakeIgnored');inner.use_nodes=True
    obj.data.materials.clear();obj.data.materials.append(outer);obj.data.materials.append(inner)
    for p in obj.data.polygons:p.material_index=0 if p.index<outer_count else 1
    im=bpy.data.images.new('Hair_Crown_Flow_Tangent_Normal_V5',width=size,height=size,alpha=False)
    im.colorspace_settings.name='Non-Color';im.generated_color=(.5,.5,1,1)
    ignored=bpy.data.images.new('TEMP_IgnoreInnerOverlappingUV',width=8,height=8,alpha=False)
    ignored.colorspace_settings.name='Non-Color'
    for mat,target in ((outer,im),(inner,ignored)):
        node=mat.node_tree.nodes.new('ShaderNodeTexImage');node.image=target;mat.node_tree.nodes.active=node
    try:
        scene.render.engine='CYCLES';scene.cycles.samples=1
        scene.render.bake.use_clear=True;scene.render.bake.use_selected_to_active=False
        scene.render.bake.margin=8
        _enum_set(scene.render.bake,'normal_space','TANGENT')
        for channel,axis in [('normal_r','POS_X'),('normal_g','POS_Y'),('normal_b','POS_Z')]:_enum_set(scene.render.bake,channel,axis)
        scene.view_layers[0].objects.active=obj;obj.select_set(True,view_layer=scene.view_layers[0])
        if bpy.context.window:bpy.context.window.scene=scene
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0],object=obj,active_object=obj,selected_objects=[obj],selected_editable_objects=[obj]):
            bpy.ops.object.bake(type='NORMAL')
        path.parent.mkdir(parents=True,exist_ok=True);im.filepath_raw=str(path);im.file_format='PNG';im.save();im.pack()
        # Numeric proof that this is a perturbed normal bake, not base colour.
        pixels=array('f',[0])*(size*size*4);im.pixels.foreach_get(pixels)
        angles=[];mins=[1,1,1];maxs=[0,0,0]
        for y in range(int(size*.08),int(size*.95),23):
            for x in range(16,size-16,23):
                rgb=pixels[(y*size+x)*4:(y*size+x)*4+3]
                if rgb[2]<.5:continue
                normal=[2*v-1 for v in rgb];length=math.sqrt(sum(v*v for v in normal))
                if length<.1:continue
                angles.append(math.degrees(math.acos(max(-1,min(1,normal[2]/length)))))
                mins=[min(a,b) for a,b in zip(mins,rgb)];maxs=[max(a,b) for a,b in zip(maxs,rgb)]
        if not angles or max(angles)<.5:raise RuntimeError('Normal bake is effectively flat; no hair-flow detail produced')
        angles.sort()
        if hashes(cap)!=original:raise RuntimeError('Source cap changed during bake')
        return {'texture':str(path),'textureSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'imageSize':[size,size],
                'bake':'Cycles NORMAL / TANGENT on actual cap copy with Bump height source',
                'heightDistance':.0014,'normalChannels':['+X','+Y','+Z'],'rgbMin':mins,'rgbMax':maxs,
                'sampleCount':len(angles),'medianBakedTiltDegrees':angles[len(angles)//2],
                'p95BakedTiltDegrees':angles[int(len(angles)*.95)],'maxBakedTiltDegrees':max(angles),
                'sourceCapHashes':original,'capGeometryAndUVExact':True,
                'innerShellUVOverlapsExcludedViaSeparateBakeTarget':True,'outerFacesBaked':outer_count}
    finally:
        if old_scene and bpy.context.window:bpy.context.window.scene=old_scene
        mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True);bpy.data.meshes.remove(mesh)
        bpy.data.materials.remove(outer);bpy.data.materials.remove(inner);bpy.data.images.remove(ignored)
        bpy.data.scenes.remove(scene)


def apply_crown_normal(cap,normal_path,expected_hashes=None,strength=.85):
    import bpy
    before=hashes(cap)
    if expected_hashes and before!=expected_hashes:raise RuntimeError('Normal map was baked for a different cap geometry/UV')
    if len(cap.data.materials)!=1:raise RuntimeError('Expected exactly one cap material')
    old=cap.data.materials[0];mat=old.copy();mat.name='Rebuilt_Hair_Crown_BakedNormal_V5'
    bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    if bs.inputs['Normal'].is_linked:raise RuntimeError('Unexpected existing normal network; preserve and review it')
    im=bpy.data.images.load(str(normal_path),check_existing=True);im.colorspace_settings.name='Non-Color';im.pack()
    uv=mat.node_tree.nodes.new('ShaderNodeUVMap');uv.uv_map='HairUV'
    tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=im;tex.interpolation='Linear'
    normal=mat.node_tree.nodes.new('ShaderNodeNormalMap');normal.space='TANGENT';normal.uv_map='HairUV';normal.inputs['Strength'].default_value=strength
    mat.node_tree.links.new(uv.outputs['UV'],tex.inputs['Vector']);mat.node_tree.links.new(tex.outputs['Color'],normal.inputs['Color']);mat.node_tree.links.new(normal.outputs['Normal'],bs.inputs['Normal'])
    cap.data.materials[0]=mat
    mat['detail_scope']='Actual tangent-space Cycles normal bake from stylised directional Bump. No added geometry, no basecolor-as-normal.'
    if hashes(cap)!=before:raise RuntimeError('Material-only change touched cap geometry/UV')
    return {'oldMaterial':old.name,'newMaterial':mat.name,'normalStrength':strength,'normalImage':im.name,
            'imagePacked':bool(im.packed_file),'imageColorSpace':im.colorspace_settings.name,'capGeometryUVExact':True,
            'baseColorNetworkCopied':True,'roughness':bs.inputs['Roughness'].default_value}


if __name__=='__main__':
    import bpy
    from mathutils import Vector
    sys.path.insert(0,str(Path(__file__).resolve().parent));from common import OUT,neutral_review
    source=OUT/'stages/E4-final-refined.blend';out=OUT/'agents/texture/E5-hair-normal-candidate.blend'
    if out.exists():raise FileExistsError(out)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest();bpy.ops.wm.open_mainfile(filepath=str(source))
    cap=bpy.data.objects['Hair_Crown_SourceEnvelope'];protected={o.name:hashes(o) for o in bpy.context.scene.objects if o.type=='MESH'}
    spec=json.loads((OUT/'reports/hair-spec-v4.json').read_text())
    path=OUT/'textures/hair-crown-flow-normal-v5-2048.png'
    report=bake_actual_cap(cap,path,spec)
    report['install']=apply_crown_normal(cap,path,report['sourceCapHashes'])
    if protected!={o.name:hashes(o) for o in bpy.context.scene.objects if o.type=='MESH'}:raise RuntimeError('Geometry or UV changed in material-only task')
    scene=neutral_review();scene.render.resolution_x=600;scene.render.resolution_y=600
    cam=scene.camera;cam.location=(1.5,4,.86);target=Vector((0,.025,.79));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=.59
    out.parent.mkdir(parents=True,exist_ok=True);bpy.ops.wm.save_as_mainfile(filepath=str(out))
    screenshot=OUT/'checks/hair-normal-v5/back-quarter.png';screenshot.parent.mkdir(parents=True,exist_ok=True)
    scene.render.filepath=str(screenshot);bpy.ops.render.render(write_still=True)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Root source modified')
    report.update(source=str(source),sourceSha256=source_hash,sourceUnchanged=True,candidate=str(out),candidateSha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  screenshot=str(screenshot),allMeshGeometryUVExact=True,status='Awaiting actual visual inspection',
                  function='apply_crown_normal(cap, texture_path, report[sourceCapHashes], strength=.85)',
                  scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'reports/hair-normal-delta-v5.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
