"""One restrained rose-lip surface candidate on the actual new female topology.

The current actual front/quarter shows existing lip volume, but very weak
vermilion/skin separation. This pass therefore changes NO geometry or keys.
It copies current colour/roughness maps and material, preserving the eye-margin
shader layer, nose/philtrum, oral contact, all optical constants and microdetail.
Original material/maps remain available. No black seam, expanded lipstick mask,
new noise or photographic/ethnic colour claim is introduced.
"""
import argparse
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

HEAD='R11 Female head study'
SOURCE_COLOR='R11 natural adult skin colour'
SOURCE_ROUGH='R11 natural adult skin roughness'
COLOR='R11 restrained rose lip colour'
ROUGH='R11 restrained rose lip roughness'
SUPPORT='R11 true vermilion rose support'
ATTRS=(COLOR,ROUGH,SUPPORT)
MATERIAL='R11 natural skin with restrained rose lips'
MARKER='R11_rose_lip_surface_candidate'
PARAMETERS={'relativeLinearRGB':[.99,.84,1.06],
    'upperColourStrength':1.,'lowerColourStrength':.85,
    'openingControlSupport':.55,'outerVermilionSupport':0.,
    'lateralFadeFraction':.22,'upperRoughnessReduction':.008,'lowerRoughnessReduction':.024}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _array(items,field,width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype=dtype);items.foreach_get(field,a.ravel());return a
def _ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()
def _path(relative):
    base=project_path().resolve();p=(base/relative).resolve();p.relative_to(base/'work/reconstruction-r02')
    if p.suffix!='.blend':raise ValueError('Independent reconstruction Blend required')
    return p


def build_maps(donor,landmarks,current_colour,current_roughness):
    """Pure arrays, original female vertex identities, current source pigments."""
    old=np.asarray(current_colour,np.float32);oldrough=np.asarray(current_roughness,np.float32).reshape(-1)
    points=np.asarray(donor['local'],float);mouth=landmarks['mouth']
    if old.shape!=(10582,4) or oldrough.shape!=(10582,):raise RuntimeError('Actual full new-head point maps required')
    strip=np.asarray(mouth['vermilionStripVertexIndices'],np.int32)
    opening=np.asarray(mouth['openingRimCandidate']['sourceVertexIndices'],np.int32)
    middle=np.asarray(mouth['vermilionMidLoop']['sourceVertexIndices'],np.int32)
    outer=np.asarray(mouth['outerVermilionLoopCandidate']['sourceVertexIndices'],np.int32)
    if len(strip)!=120 or len(set(strip))!=120 or any(len(x)!=40 for x in (opening,middle,outer)):
        raise RuntimeError('Actual two-strip vermilion topology differs')
    qa=_api('face_contact_audit_r11');upper=np.zeros(len(points),bool);lower=np.zeros(len(points),bool)
    for ring in (opening,middle,outer):
        left=int(ring[np.argmin(points[ring,0])]);right=int(ring[np.argmax(points[ring,0])])
        u,l=qa['_split'](list(map(int,ring)),points,left,right);upper[u]=True;lower[l]=True
    center=float(np.mean(points[opening,0]));half=float(np.max(np.abs(points[opening,0]-center)))
    fraction=np.clip((1.-abs(points[:,0]-center)/half)/PARAMETERS['lateralFadeFraction'],0.,1.)
    lateral= fraction*fraction*(3.-2.*fraction)
    support=np.zeros(len(points));support[middle]=1.;support[opening]=PARAMETERS['openingControlSupport']
    support[outer]=0.;support*=lateral
    strength=support*np.where(upper,PARAMETERS['upperColourStrength'],PARAMETERS['lowerColourStrength'])
    active=strength>1e-8;allowed=np.zeros(len(points),bool);allowed[strip]=True
    if np.any(active&(~allowed)) or not 50<int(active.sum())<=80:raise RuntimeError('Rose colour escaped the measured lip strips')
    result=old.copy();gain=np.asarray(PARAMETERS['relativeLinearRGB'],float)-1.
    result[active,:3]=(old[active,:3]*(1.+strength[active,None]*gain)).astype(np.float32)
    rough=oldrough.copy();reduction=np.where(upper,PARAMETERS['upperRoughnessReduction'],PARAMETERS['lowerRoughnessReduction'])
    rough[active]=(oldrough[active]-support[active]*reduction[active]).astype(np.float32)
    if not np.array_equal(result[~active],old[~active]) or not np.array_equal(rough[~active],oldrough[~active]) or not np.array_equal(result[:,3],old[:,3]):
        raise RuntimeError('Non-lip map rows or alpha changed')
    if result.min()<0 or result.max()>1 or rough.min()<.30 or rough.max()>.75:raise RuntimeError('Lip candidate exceeded bounded source-map range')
    delta=result[:,:3]-old[:,:3]
    if abs(delta).max()>.055:raise RuntimeError('Unexpectedly large lip pigment change')
    plan={'parameters':PARAMETERS,'changedCageRows':int(active.sum()),'anatomicalSupportRows':len(strip),
        'upperRows':int((active&upper).sum()),'lowerRows':int((active&lower).sum()),
        'sharedCornerRows':int((active&upper&lower).sum()),
        'sourceActiveMedianLinearRGB':np.median(old[active,:3],axis=0).tolist(),
        'candidateActiveMedianLinearRGB':np.median(result[active,:3],axis=0).tolist(),
        'maximumAbsoluteLinearRGBChange':np.max(abs(delta),axis=0).tolist(),
        'sourceActiveRoughnessRange':[float(oldrough[active].min()),float(oldrough[active].max())],
        'candidateActiveRoughnessRange':[float(rough[active].min()),float(rough[active].max())],
        'outerRingRowsExact':bool(np.array_equal(result[outer],old[outer]) and np.array_equal(rough[outer],oldrough[outer])),
        'scope':'True opening/mid/outer vermilion topology only. The opening control ring is not treated as the physical wet-line or drawn black; its tint is weaker than the lip body.',
        'geometryChangeMeters':0.,'colourInterpretation':'Authored restrained rose colour variation, not a calibrated cosmetic or sampled lighting-dependent photo colour.'}
    return {COLOR:result,ROUGH:rough,SUPPORT:strength.astype(np.float32)},plan


def _nodes(material):
    if not material.use_nodes:raise RuntimeError('Current node skin required')
    colors=[n for n in material.node_tree.nodes if n.type=='VERTEX_COLOR' and n.layer_name==SOURCE_COLOR]
    rough=[n for n in material.node_tree.nodes if n.type=='ATTRIBUTE' and n.attribute_name==SOURCE_ROUGH]
    if len(colors)!=1 or len(rough)!=1:raise RuntimeError('Expected current natural skin map inputs beneath the existing eye-margin layer')
    return colors[0].name,rough[0].name


def _normalized(material,node_names,face):
    tree=copy.deepcopy(face['_tree_contract'](material.node_tree))
    for row in tree['nodes']:
        if row['name']==node_names[0]:row['properties']['layer_name']='EXPLICIT_LIP_COLOUR_COPY'
        if row['name']==node_names[1]:row['properties']['attribute_name']='EXPLICIT_LIP_ROUGHNESS_COPY'
    return _hash({'tree':tree,'settings':{k:face['_plain'](getattr(material,k)) for k in ('use_backface_culling','surface_render_method','use_transparent_shadow','diffuse_color','metallic','roughness') if hasattr(material,k)}})


def _protected(head,slot):
    state=_api('eye_margin_refine_r11')['_state'](head,slot)
    state['attributes']=[row for row in state['attributes'] if row[0] not in ATTRS]
    return state


def apply():
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    head=bpy.data.objects[HEAD];mesh=head.data
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582:raise RuntimeError('Actual complete local new female head required')
    if MARKER in bpy.context.scene or any(mesh.attributes.get(n) for n in ATTRS):raise RuntimeError('Retain/restore the earlier lip surface candidate first')
    skin=_api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']()
    if not np.array_equal(_array(mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):raise RuntimeError('Female anatomical vertex identities changed')
    source_color=mesh.color_attributes.get(SOURCE_COLOR);source_rough=mesh.attributes.get(SOURCE_ROUGH)
    if not source_color or source_color.domain!='POINT' or not source_rough or source_rough.domain!='POINT':raise RuntimeError('Current original natural skin point maps missing')
    candidates=[]
    for index,slot in enumerate(head.material_slots):
        if slot.material and slot.material.use_nodes and any(n.type=='VERTEX_COLOR' and n.layer_name==SOURCE_COLOR for n in slot.material.node_tree.nodes):candidates.append(index)
    if len(candidates)!=1:raise RuntimeError('Expected one actual natural-skin material slot')
    slot=candidates[0];original=head.material_slots[slot].material;node_names=_nodes(original)
    maps,plan=build_maps(donor,landmarks,_array(source_color.data,'color',4),_array(source_rough.data,'value',1))
    face=_api('runtime_face_refresh_r08');uv=_api('production_uv_r08');eye=_api('eye_margin_refine_r11');binding=_api('face_fibre_binding_r11')
    bpy.context.view_layer.update();settings=uv['_uv_settings'](mesh);before=_protected(head,slot);external=eye['_external'](bpy,head,[], '')
    evaluated=binding['_head_mesh'](head);original_hash=face['_material_contract'](original);graph_hash=_normalized(original,node_names,face);source_sha=_sha(source)
    original_fake=original.use_fake_user;created=[];candidate=None;success=False
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'head':HEAD,'materialSlot':slot,
        'helperSha256':_sha(__file__),'sourceMayContainEarlierInMemoryModules':bool(bpy.data.is_dirty),
        'provenance':provenance,'plan':plan,'originalMaterial':original.name,
        'originalMaterialSha256':original_hash,'originalFakeUser':original_fake,'originalColourSettings':settings,
        'appearanceAccepted':False,'sourceSaved':False,'scope':'Copied current skin material and derived true-vermilion colour/roughness maps only; existing eye-margin shader and complete face keys are retained.',
        'decisionEvidence':{'views':'previews/reconstruction-r02/adult-face-contour-r11/structure-front.png and structure-quarter.png',
            'observedSourceSha256':'504503e475b1ccdf05a8f30008baacefa5f22fc8f7108ceb5ed74b05de6dcb6d',
            'reason':'Actual lip protrusion is visible, especially lower lip; weak colour separation makes the dark mouth seam dominate. No geometric filling is justified in this single trial.'},
        'requiredReview':'Same-light front/quarter plus mouth closeup at smile 0, 0.5 and 1. Reject lipstick-like border, dark painted seam, reduced apparent lip bulk or plastic shine. Current smile is not accepted as natural performance.'}
    try:
        for name,values in maps.items():
            attribute=mesh.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT') if name==COLOR else mesh.attributes.new(name,'FLOAT','POINT')
            created.append(name);attribute.data.foreach_set('color' if name==COLOR else 'value',values.ravel())
        uv['_restore_uv_settings'](mesh,settings)
        candidate=original.copy();candidate.name=MATERIAL
        candidate.node_tree.nodes[node_names[0]].layer_name=COLOR;candidate.node_tree.nodes[node_names[1]].attribute_name=ROUGH
        if _normalized(candidate,node_names,face)!=graph_hash:raise RuntimeError('Current optical/eye-margin graph changed beyond two copied map references')
        candidate['R11_lip_source_material']=original.name;candidate['R11_lip_scope']='Restrained true-vermilion rose colour and roughness; geometry and smile unchanged, pending visual review.'
        original.use_fake_user=True;head.material_slots[slot].material=candidate;bpy.context.view_layer.update()
        after=_protected(head,slot)
        if _hash(before)!=_hash(after):raise RuntimeError('Original face source contract changed: '+json.dumps(skin['_differences'](before,after)))
        actual=binding['_head_mesh'](head)
        if actual['topologySha256']!=evaluated['topologySha256'] or not np.array_equal(actual['co'],evaluated['co']) or not np.array_equal(actual['matrix'],evaluated['matrix']):raise RuntimeError('Actual evaluated face geometry changed')
        if eye['_external'](bpy,head,[], '')!=external or face['_material_contract'](original)!=original_hash or _sha(source)!=source_sha:raise RuntimeError('External scene, original material or source file changed')
        report.update(candidateMaterial=candidate.name,candidateMaterialSha256=face['_material_contract'](candidate),
            candidateAttributeHashes={name:_ah(value) for name,value in maps.items()},
            actualEvaluatedGeometryDifferenceMeters=0.,actualHeadTopologySha256=actual['topologySha256'],
            originalKeysUVWeightsMapsAndContactGeometryExact=True,existingEyeMarginAndOpticsExact=True)
        if binding['MARKER'] in bpy.context.scene:report['actualFibreBindingValidation']=binding['validate'](neutral=False)
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            head.material_slots[slot].material=original;original.use_fake_user=original_fake
            for name in reversed(created):
                if mesh.attributes.get(name):mesh.attributes.remove(mesh.attributes[name])
            uv['_restore_uv_settings'](mesh,settings)
            if candidate and candidate.users==0:bpy.data.materials.remove(candidate)
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];mesh=head.data;face=_api('runtime_face_refresh_r08')
    old=bpy.data.materials[report['originalMaterial']];new=bpy.data.materials[report['candidateMaterial']]
    if head.material_slots[report['materialSlot']].material!=new or face['_material_contract'](old)!=report['originalMaterialSha256'] or face['_material_contract'](new)!=report['candidateMaterialSha256']:raise RuntimeError('Skin material changed since candidate; retain edits before restore')
    for name,digest in report['candidateAttributeHashes'].items():
        attr=mesh.attributes.get(name)
        if not attr:raise RuntimeError('Candidate map missing: '+name)
        values=_array(attr.data,'color',4) if name==COLOR else _array(attr.data,'value',1).ravel()
        if _ah(values)!=digest:raise RuntimeError('Lip map edited; retain edits before restore')
    head.material_slots[report['materialSlot']].material=old;old.use_fake_user=report['originalFakeUser']
    for name in reversed(ATTRS):mesh.attributes.remove(mesh.attributes[name])
    _api('production_uv_r08')['_restore_uv_settings'](mesh,report['originalColourSettings']);new.use_fake_user=True
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'originalSkinMaterialRestored':True,'candidateMaterialRetained':True,'sourceSaved':False}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.lip-surface.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Preserve history; use a new output')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit source required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),candidateSaved=True)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_LIP_SURFACE_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output)
