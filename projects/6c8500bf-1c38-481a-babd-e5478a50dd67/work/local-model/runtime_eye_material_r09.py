"""Audit/bake the CURRENT documented iris colour graph, without retinting it.

The final source may use the direct R07 image or the explicit R09 HSV/Multiply
candidate. The latter is actually evaluated into one shared iris base-color
atlas. All existing optical constants and both sclera assignments stay exact.
No work on import; no Blender launch, save or render preview.
"""
import hashlib,json,math
from pathlib import Path
ORIGINAL='R07 documented brown iris';LEGACY='R02 deep charcoal brown iris'
CANDIDATE='R09 restrained deep brown iris candidate'
DOCUMENTED_IMAGE_SHA='6bddc86582fa1b4541376082816c4d734deabbbcbe38e65710415823f0566784'


def _plain(value):
    if isinstance(value,(bool,str,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise RuntimeError('Nonfinite eye shader input')
        return value
    return [_plain(float(v)) for v in value]


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _image(node,api):
    image=node.image
    if image is None or image.colorspace_settings.name!='sRGB' or min(image.size)<1 or image.is_dirty:
        raise RuntimeError('Saved/packed finite sRGB iris image required')
    if image.packed_file:sha=hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
    else:
        path=Path(api.bpy.path.abspath(image.filepath,library=image.library))
        if not path.is_file():raise RuntimeError('Iris image file is missing')
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return {'name':image.name,'size':list(image.size),'colorspace':image.colorspace_settings.name,
        'alphaMode':image.alpha_mode,'bytesSha256':sha}


def _color_graph(socket,api,seen=()):
    if not socket.is_linked:return {'constant':_plain(socket.default_value)}
    if len(socket.links)!=1:raise RuntimeError('Ambiguous iris colour link')
    link=socket.links[0];node=link.from_node
    if node.as_pointer() in seen:raise RuntimeError('Cyclic iris colour graph')
    seen=(*seen,node.as_pointer());row={'type':node.type,'output':link.from_socket.name}
    if node.type=='TEX_IMAGE':
        if link.from_socket.name!='Color':raise RuntimeError('Iris base colour must use image Color')
        vector=node.inputs['Vector']
        if vector.is_linked and (len(vector.links)!=1 or vector.links[0].from_node.type!='UVMAP'):
            raise RuntimeError('Unreviewed iris image coordinate transform')
        row.update(image=_image(node,api),interpolation=node.interpolation,extension=node.extension,
            sourceUVMap=vector.links[0].from_node.uv_map if vector.is_linked else 'ACTIVE_SOURCE_UV')
    elif node.type=='HUE_SAT':
        fields=('Hue','Saturation','Value','Fac')
        if any(node.inputs[n].is_linked for n in fields):raise RuntimeError('Animated/procedural iris HSV controls need explicit adaptation')
        row.update(parameters={n:_plain(node.inputs[n].default_value) for n in fields},
                   color=_color_graph(node.inputs['Color'],api,seen))
    elif node.type=='MIX_RGB':
        if node.blend_type!='MULTIPLY' or node.inputs[0].is_linked:raise RuntimeError('Unreviewed iris MixRGB operation')
        row.update(operation=node.blend_type,clamp=bool(node.use_clamp),factor=_plain(node.inputs[0].default_value),
            color1=_color_graph(node.inputs[1],api,seen),color2=_color_graph(node.inputs[2],api,seen))
    else:raise RuntimeError('Unreviewed current iris colour node: '+node.type)
    return row


def _constants(shader):
    result={}
    for socket in shader.inputs:
        if socket.name=='Base Color':continue
        if socket.is_linked:raise RuntimeError('Iris requires another channel bake: '+socket.name)
        if hasattr(socket,'default_value'):result[socket.identifier]=_plain(socket.default_value)
    return result


def _sclera(objects,api):
    lookup={o.name:o for o in objects};rows=[]
    for side in ('L','R'):
        name='RT06 Face / R01 Sclera '+side
        if name not in lookup or len(lookup[name].data.materials)!=1:raise RuntimeError('Both current sclera materials required')
        material=lookup[name].data.materials[0];_,shader=api._surface(material)
        if shader.inputs['Base Color'].is_linked:raise RuntimeError('Unexpected linked sclera colour')
        rows.append({'object':name,'material':material.name,'sourceMaterial':material.get('R09_source_material'),
            'baseColor':_plain(shader.inputs['Base Color'].default_value),'constants':_constants(shader)})
    return rows


def stage(objects,api):
    names=['RT06 Face / R01 Textured iris '+side for side in ('L','R')];lookup={o.name:o for o in objects}
    if any(n not in lookup or len(lookup[n].data.materials)!=1 for n in names):raise RuntimeError('Both current single-material iris meshes required')
    material=lookup[names[0]].data.materials[0]
    if material is None or lookup[names[1]].data.materials[0]!=material:raise RuntimeError('Current bilateral iris materials differ')
    _,shader=api._surface(material);graph=_color_graph(shader.inputs['Base Color'],api)
    constants=_constants(shader);sclera=_sclera(objects,api)
    row={'material':material.name,'objects':names,'channel':'basecolor','resolution':512,
        'currentColorGraph':graph,'currentColorGraphSha256':_hash(graph),'preservedOpticalConstants':constants,
        'scleraPreserved':sclera,'sourceMaterial':material.get('R09_source_material'),
        'preserves':'Current source colour evaluation and all PBR constants, including coat weight/roughness/IOR/tint; no reset to old R07 source image.'}
    direct=graph.get('type')=='TEX_IMAGE'
    traced_bake=False
    if material.get('RT09_iris_bake_source_color_chain'):
        stored=json.loads(material['RT09_iris_bake_source_color_chain'])
        traced_bake=_hash(stored)==material.get('RT09_iris_bake_source_graph_sha256') and material.get('R09_source_material')==ORIGINAL
        if not traced_bake:raise RuntimeError('Stored current iris colour-bake provenance differs')
    if material.name==ORIGINAL and direct:
        row.update(action='preserve',reason='Current documented R07 image connects directly; no colour operations to bake.',
            image=graph['image']['name'],imageSize=graph['image']['size'],packedImageSha256=graph['image']['bytesSha256'])
    elif material.get('r06_shared_channel_bake')=='basecolor' and (material.get('r06_source_material') in (ORIGINAL,LEGACY,CANDIDATE) or traced_bake) and direct:
        row.update(action='preserve',reason='Current source colour chain was already evaluated into this shared base-color atlas.',
            bakedSourceColorGraphSha256=material.get('RT09_iris_bake_source_graph_sha256'))
    elif material.name==LEGACY:
        row.update(action='bake',reason='Legacy iris colour operations require actual base-color evaluation.')
    elif material.get('R09_source_material')==ORIGINAL:
        if graph.get('type')!='MIX_RGB' or graph['color1'].get('type')!='HUE_SAT' or 'constant' not in graph['color2']:
            raise RuntimeError('R09 iris no longer has the documented image -> HSV -> Multiply chain')
        image=graph['color1']['color']
        if image.get('type')!='TEX_IMAGE' or image['image']['bytesSha256']!=DOCUMENTED_IMAGE_SHA:
            raise RuntimeError('R09 iris colour chain input differs from the documented unchanged source image')
        row.update(action='bake',reason='Evaluate CURRENT R09 HSV and linear Multiply into shared iris colour pixels; preserve current coat and sclera.',
            explicitColorChanges={'hsv':graph['color1']['parameters'],'multiply':graph['color2']['constant'],
                'multiplyFactor':graph['factor'],'clamp':graph['clamp'],'sourceImage':image['image']})
    else:raise RuntimeError('Unreviewed current iris material/colour chain: '+material.name)
    return row


def bake_current(row,objects,api,output_relative):
    current=stage(objects,api)
    if current!=row or row['action']!='bake':raise RuntimeError('Current eye stage changed before baking')
    originals=[o.data.materials[0] for o in objects if o.name in row['objects']]
    if len(originals)!=2 or originals[0]!=originals[1]:raise RuntimeError('Bilateral iris source changed')
    result=api.bake_shared_channel(row['objects'],row['material'],channel='basecolor',resolution=row['resolution'],output_relative=output_relative)
    _,source_shader=api._surface(originals[0])
    if _color_graph(source_shader.inputs['Base Color'],api)!=row['currentColorGraph'] or _constants(source_shader)!=row['preservedOpticalConstants']:
        raise RuntimeError('Original iris colour/optics changed during shared bake')
    lookup={o.name:o for o in objects};materials=[lookup[n].data.materials[0] for n in row['objects']]
    if materials[0]!=materials[1]:raise RuntimeError('Baked iris atlas is not shared')
    _,shader=api._surface(materials[0])
    if _constants(shader)!=row['preservedOpticalConstants'] or _sclera(objects,api)!=row['scleraPreserved']:
        raise RuntimeError('Iris optical constants or sclera changed during colour bake')
    baked_graph=_color_graph(shader.inputs['Base Color'],api)
    if baked_graph.get('type')!='TEX_IMAGE':raise RuntimeError('Baked current iris colour is not a direct portable image')
    materials[0]['RT09_iris_bake_source_graph_sha256']=row['currentColorGraphSha256']
    materials[0]['RT09_iris_bake_source_color_chain']=json.dumps(row['currentColorGraph'])
    result.update(currentSourceEyeStage=row,bakedColorGraph=baked_graph,sourceImageAndShaderUnchanged=True,
        allOpticalConstantsExact=True,scleraAssignmentsAndConstantsExact=True,revertedToUnadjustedR07Image=False,
        limits='Actual shared base-color bake still requires matched viewer images; coat IOR mapping remains the documented glTF extension limitation.')
    path=api.project_path()/output_relative/'iris-current-source-color-report.json'
    with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result
