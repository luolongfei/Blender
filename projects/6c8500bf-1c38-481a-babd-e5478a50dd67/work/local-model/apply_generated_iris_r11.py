"""Single reversible generated dark-brown iris material study.

Offline inspect_texture() reads original pixels; it never resizes, edits or
rewrites the image. Blender apply() copies the current two R11 iris materials,
replaces the one documented R07 image, and uses a neutral (1,1,1) multiplier.
Geometry, IrisUV, all optical parameters and source image/material stay exact.
This is a generated albedo candidate, not a captured/scan asset or approval.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

TEXTURE='textures/generated-r11/iris-dark-brown-r11.png'
TEXTURE_SHA='8aa48663c77fec6c668c4bc67c0673776fde6d3ae9c4a3fed81a64b776ae3e0e'
OLD_SHA='6bddc86582fa1b4541376082816c4d734deabbbcbe38e65710415823f0566784'
INSPECTION='work/reconstruction-r02/iris-dark-brown-inspection-r11.json'
OBJECTS=('R11 Adult iris L','R11 Adult iris R')
TEX_NODE='R11 documented R07 iris'
MULT_NODE='R11 warm linear reflectance'
MARKER='R11_generated_dark_iris_candidate'
MULTIPLIER=(1.,1.,1.,1.)


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _hash(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _array(items,field,width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype=dtype);items.foreach_get(field,a.ravel());return a
def _path(relative,suffix='.blend'):
    P=project_path().resolve();p=(P/relative).resolve();p.relative_to(P/'work/reconstruction-r02')
    if p.suffix!=suffix:raise ValueError('Explicit reconstruction artifact required')
    return p


def inspect_texture():
    """Pure offline pixel measurements. Returns data; never alters the PNG."""
    from PIL import Image
    path=project_path()/TEXTURE;meta_path=path.with_suffix('.json')
    if _sha(path)!=TEXTURE_SHA:raise RuntimeError('Generated image bytes changed; inspect the intended new asset explicitly')
    with Image.open(path) as image:
        if image.mode not in {'RGB','RGBA'}:raise RuntimeError('Inspect unexpected image colour channels')
        pixels=np.asarray(image).copy();mode=image.mode;size=image.size
    h,w=pixels.shape[:2]
    if h!=w:raise RuntimeError('Current unmodified circular IrisUV requires a square image')
    yy,xx=np.indices((h,w));center=np.array([(w-1)*.5,(h-1)*.5]);radius=np.hypot(xx-center[0],yy-center[1])
    pupil_fits=[]
    for threshold in (14,16,18,20):
        mask=(pixels[:,:,:3].max(2)<threshold)&(radius<w*.22)
        seen=np.zeros((h,w),bool);seed=(h//2,w//2)
        if not mask[seed]:raise RuntimeError('Near-black central pupil seed missing')
        q=collections.deque([seed]);seen[seed]=True
        while q:
            y,x=q.popleft()
            for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
                ny,nx=y+dy,x+dx
                if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;q.append((ny,nx))
        boundary=seen&(~np.roll(seen,1,0)|~np.roll(seen,-1,0)|~np.roll(seen,1,1)|~np.roll(seen,-1,1))
        y,x=np.where(boundary)
        if len(x)<300:raise RuntimeError('Insufficient measured central pupil perimeter')
        c=np.linalg.lstsq(np.column_stack((2*x,2*y,np.ones(len(x)))),x*x+y*y,rcond=None)[0]
        r=float(np.sqrt(c[2]+c[0]**2+c[1]**2));error=np.abs(np.hypot(x-c[0],y-c[1])-r)
        pupil_fits.append({'maxRGBThreshold':threshold,'centerPixels':c[:2].tolist(),'radiusPixels':r,'radiusUV':r/w,
            'radialResidual95Pixels':float(np.quantile(error,.95)),
            'centerOffsetPixels':(c[:2]-center).tolist(),'pupilDiameterFractionOfMappedIris':r/w/.477})
    # Integer pixel reads measure the approximate outer circular transition.
    # This is analysis, not a resampled output image; dark irregular fibres make
    # this less exact than the pupil fit, so report quantiles, not a perfect rim.
    angles=np.linspace(0,2*np.pi,720,endpoint=False);r=np.arange(w*.474,w*.499,1.)
    xs=np.rint(center[0]+np.cos(angles[:,None])*r).astype(int);ys=np.rint(center[1]+np.sin(angles[:,None])*r).astype(int)
    rows=pixels[ys,xs,:3].astype(float);gradient=np.linalg.norm(rows[:,2:]-rows[:,:-2],axis=2)
    peaks=r[gradient.argmax(1)+1]/w
    bands=[]
    for lo,hi in ((0,.12),(.16,.22),(.25,.40),(.45,.477),(.49,.55)):
        values=pixels[(radius/w>=lo)&(radius/w<hi),:3]
        bands.append({'radiusUV':[lo,hi],'medianRGB8':np.median(values,axis=0).tolist(),'p95RGB8':np.quantile(values,.95,axis=0).tolist()})
    meta=json.loads(meta_path.read_text(encoding='utf8'))
    return {'texture':TEXTURE,'sha256':TEXTURE_SHA,'width':w,'height':h,'mode':mode,'imageEdited':False,
        'generationMetadata':str(meta_path.relative_to(project_path())),'generationMetadataSha256':_sha(meta_path),
        'requestedModel':meta.get('requestedModel'),'actualModel':meta.get('actualModel'),
        'requested2048PixelsActuallyDelivered':bool(w==2048 and h==2048),
        'pupilFits':pupil_fits,'approximateOuterGradientRadiusUVQuantiles':np.quantile(peaks,[.1,.5,.9]).tolist(),
        'radialColourMeasurements':bands,'existingUV':{'irisRadius':.477,'nominalPupilRadius':.145,'targetPupilIrisRatio':.304},
        'interpretation':['Actual file is 1254 square, not the 2048 requested in the generation prompt.',
            'Measured pupil is slightly larger and a few pixels off-centre. This is acceptable for a bounded visual candidate, not an exact analytic pupil or dynamic dilation asset.',
            'The existing .477 UV radius lies slightly inside the generated outer circle, within its dark peripheral band. Preserve UV and inspect the rendered limbal edge.',
            'No image resampling, crop, repaint, colour correction or highlight removal was performed.',
            'The generated base colour has visible local fibre contrast; absence of a painted catchlight is a visual observation, not proof of calibrated flat albedo.'],
        'appearanceAccepted':False,'requiredReview':'Same-light eyes macro and full face with real coat reflection; judge dark-brown appearance, limbal edge, pupil centring and whether radial detail remains too conspicuous.'}


def write_inspection():
    report=inspect_texture();path=_path(INSPECTION,'.json')
    if path.exists():raise RuntimeError('Retain existing inspection instead of overwriting')
    with path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    return report


def _image_sha(image):
    import bpy
    if image.is_dirty:raise RuntimeError('Original iris image has unsaved pixel edits')
    if image.packed_file:return hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
    return _sha(Path(bpy.path.abspath(image.filepath,library=image.library)))


def _nodes(material):
    nodes=material.node_tree.nodes if material.use_nodes else []
    images=[n for n in nodes if n.type=='TEX_IMAGE' and n.image]
    if len(images)!=1 or images[0].name!=TEX_NODE or _image_sha(images[0].image)!=OLD_SHA:
        raise RuntimeError('Only the unique documented original R07 iris image may be replaced')
    tex=images[0];mult=nodes.get(MULT_NODE)
    if not mult or mult.type!='MIX_RGB' or mult.blend_type!='MULTIPLY' or mult.inputs[2].is_linked:
        raise RuntimeError('Expected original explicit iris reflectance multiplier')
    uv=tex.inputs['Vector'].links
    if len(uv)!=1 or uv[0].from_node.type!='UVMAP' or uv[0].from_node.uv_map!='IrisUV':raise RuntimeError('Actual iris uses a different UV mapping')
    if len(mult.inputs[1].links)!=1 or mult.inputs[1].links[0].from_node!=tex:raise RuntimeError('Original image-to-multiplier graph differs')
    return tex,mult


def _normalized_material(material,face):
    tree=copy.deepcopy(face['_tree_contract'](material.node_tree))
    for node in tree['nodes']:
        if node['name']==TEX_NODE:node['image']='EXPLICIT_REPLACED_IMAGE'
        if node['name']==MULT_NODE:node['inputs'][2]['value']='EXPLICIT_NEUTRAL_MULTIPLIER'
    return _hash({'tree':tree,'settings':{k:face['_plain'](getattr(material,k)) for k in ('use_backface_culling','surface_render_method','use_transparent_shadow','diffuse_color','metallic','roughness') if hasattr(material,k)}})


def _mesh_state(obj,face):
    state=face['_mesh_contract'](obj);state.pop('materials')
    state['modifiers']=[(m.name,m.type) for m in obj.modifiers]
    return _hash(state)


def _external(bpy):
    rows=[]
    for o in bpy.context.scene.objects:
        rows.append({'name':o.name,'object':o.as_pointer(),'data':getattr(getattr(o,'data',None),'as_pointer',lambda:None)(),
            'parent':o.parent.as_pointer() if o.parent else None,'world':[list(r) for r in o.matrix_world],
            'materials':[] if o.name in OBJECTS else [s.material.as_pointer() if s.material else None for s in o.material_slots],
            'pose':[(b.name,[list(r) for r in b.matrix_basis]) for b in o.pose.bones] if o.type=='ARMATURE' else None})
    return _hash({'objects':rows,'actions':[(a.name,a.as_pointer()) for a in bpy.data.actions],
        'frame':[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe]})


def apply():
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    if MARKER in bpy.context.scene:raise RuntimeError('Retain or restore the earlier generated-iris candidate')
    texture=project_path()/TEXTURE;inspection_path=_path(INSPECTION,'.json')
    inspection=json.loads(inspection_path.read_text(encoding='utf8'))
    if _sha(texture)!=TEXTURE_SHA or inspection['sha256']!=TEXTURE_SHA:raise RuntimeError('Current generated image differs from its actual offline inspection')
    meta_path=texture.with_suffix('.json')
    if _sha(meta_path)!=inspection['generationMetadataSha256']:raise RuntimeError('Generation provenance changed since inspection')
    face=_api('runtime_face_refresh_r08');objects=[bpy.data.objects[n] for n in OBJECTS]
    bpy.context.view_layer.update();source_sha=_sha(source);external=_external(bpy)
    head=bpy.data.objects['R11 Female head study'];head_before=_api('face_fibre_binding_r11')['_protected'](head)
    mesh_before={o.name:_mesh_state(o,face) for o in objects};originals={};assignments=[];materials=[];image=None;success=False
    for obj in objects:
        if obj.type!='MESH' or obj.library or obj.data.library or obj.data.users!=1 or len(obj.material_slots)!=1:raise RuntimeError('Explicit independent adult iris mesh required')
        uv=obj.data.uv_layers.get('IrisUV')
        if not uv:raise RuntimeError('Original IrisUV missing')
        coords=_array(uv.data,'uv',2);r=np.linalg.norm(coords-.5,axis=1)
        if abs(float(r.max())-.477)>2e-6:raise RuntimeError('Actual IrisUV radius differs from the inspected texture mapping')
        old=obj.material_slots[0].material;tex,mult=_nodes(old)
        originals.setdefault(old.as_pointer(),{'material':old,'fake':old.use_fake_user,'sha256':face['_material_contract'](old),
            'normalized':_normalized_material(old,face),'oldImage':tex.image.name,'oldMultiplier':list(mult.inputs[2].default_value)})
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
        'texture':TEXTURE,'textureSha256':TEXTURE_SHA,'inspection':INSPECTION,'inspectionSha256':_sha(inspection_path),
        'generationMetadata':str(meta_path.relative_to(project_path())),'generationMetadataSha256':_sha(meta_path),
        'neutralLinearMultiplier':list(MULTIPLIER),'assignments':[],'appearanceAccepted':False,'sourceSaved':False,
        'scope':'Current R11 iris material copies; documented image and reflectance multiplier only. Original image pixels, all geometry/UV, skin, eyebrows/lashes, sclera, coat/roughness/IOR/specular and lights are retained.',
        'requiredReview':inspection['requiredReview'],'limits':['Synthetic base-colour candidate, not a captured or scanned human iris. Requested image2.5 model cannot be verified because the built-in image tool did not expose its actual model.',
            'Existing single-coat optical approximation is unchanged. This image swap does not establish gaze, pupil dilation, blink or photoreal appearance.']}
    try:
        image=bpy.data.images.load(str(texture),check_existing=False);image.name='R11 generated dark brown iris - original pixels'
        image.colorspace_settings.name='sRGB';image.pack()
        if not image.packed_file or hashlib.sha256(bytes(image.packed_file.data)).hexdigest()!=TEXTURE_SHA:raise RuntimeError('Packed generated image bytes differ from the original PNG')
        if list(image.size)!=[inspection['width'],inspection['height']]:raise RuntimeError('Blender image dimensions differ from offline evidence')
        copies={}
        for pointer,item in originals.items():
            old=item['material'];new=old.copy();new.name='R11 generated deep brown iris';materials.append(new)
            tex=new.node_tree.nodes[TEX_NODE];mult=new.node_tree.nodes[MULT_NODE]
            tex.image=image;tex.label='Generated R11 dark brown iris; original pixels, neutral reflectance'
            mult.inputs[2].default_value=MULTIPLIER
            if _normalized_material(new,face)!=item['normalized']:raise RuntimeError('Iris shader changed beyond the image and neutral multiplier')
            new['R11_original_optics_note']=str(old.get('R11_optics',''))
            new['R11_optics']='Original single-coat optics unchanged; generated original PNG times neutral RGB (1,1,1).'
            new['R11_texture_sha256']=TEXTURE_SHA;new['source_record']=str(meta_path.relative_to(project_path()))
            new['R11_original_material']=old.name;old.use_fake_user=True;copies[pointer]=new
        for obj in objects:
            old=obj.material_slots[0].material;item=originals[old.as_pointer()];new=copies[old.as_pointer()]
            assignments.append((obj,old,item['fake']));obj.material_slots[0].material=new
            report['assignments'].append({'object':obj.name,'slot':0,'originalMaterial':old.name,'originalFakeUser':item['fake'],
                'originalMaterialSha256':item['sha256'],'candidateMaterial':new.name,'candidateMaterialSha256':face['_material_contract'](new),
                'originalImage':item['oldImage'],'originalMultiplier':item['oldMultiplier']})
        bpy.context.view_layer.update()
        if any(_mesh_state(obj,face)!=mesh_before[obj.name] for obj in objects):raise RuntimeError('Actual iris geometry/UV/attributes/weights/transform changed')
        if _api('face_fibre_binding_r11')['_protected'](head)!=head_before or _external(bpy)!=external:raise RuntimeError('Head or external object/material/action/pose identity changed')
        if any(face['_material_contract'](x['material'])!=x['sha256'] for x in originals.values()) or _sha(source)!=source_sha or _sha(texture)!=TEXTURE_SHA:raise RuntimeError('Original material, source file or generated PNG changed')
        report.update(image=image.name,imagePacked=True,actualSize=list(image.size),actualColorspace=image.colorspace_settings.name,
            exactOriginalGeometryAndUV=True,opticalGraphUnchanged=True,
            validationScope='Exact target iris geometry/UV/attribute/weight contracts, head editable state and normalized shader graphs. Other objects/actions retain identity, assignments and pose; full simulation caches are not re-read.')
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            for obj,old,fake in reversed(assignments):obj.material_slots[0].material=old
            for item in originals.values():item['material'].use_fake_user=item['fake']
            for material in materials:
                if material.users==0:bpy.data.materials.remove(material)
            if image and image.users==0:bpy.data.images.remove(image)
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);face=_api('runtime_face_refresh_r08')
    for row in report['assignments']:
        obj=bpy.data.objects[row['object']];old=bpy.data.materials[row['originalMaterial']];new=bpy.data.materials[row['candidateMaterial']]
        if obj.material_slots[0].material!=new or face['_material_contract'](old)!=row['originalMaterialSha256'] or face['_material_contract'](new)!=row['candidateMaterialSha256']:raise RuntimeError('Materials changed after the candidate; retain those edits before restore')
    for row in report['assignments']:
        old=bpy.data.materials[row['originalMaterial']];new=bpy.data.materials[row['candidateMaterial']]
        bpy.data.objects[row['object']].material_slots[0].material=old;old.use_fake_user=row['originalFakeUser'];new.use_fake_user=True
    bpy.data.images[report['image']].use_fake_user=True;del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'originalIrisMaterialsRestored':True,'generatedCandidateAndPackedImageRetained':True,'sourceSaved':False}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.iris.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Preserve history; use a new output')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit source required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),candidateSaved=True)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_GENERATED_IRIS_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source');p.add_argument('--output');p.add_argument('--inspect-only',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if a.inspect_only:
        r=write_inspection();print(json.dumps({'inspection':INSPECTION,'size':[r['width'],r['height']],'sha256':r['sha256']}))
    elif not a.source or not a.output:p.error('--source and --output are required for Blender prepare')
    else:prepare(a.source,a.output)
