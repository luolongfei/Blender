"""One reversible region-colour/roughness candidate after the actual lighting A/B.

No geometry, UV, face keys, rest micro-normal, SSS, exposure or lights change.
The original material/maps remain. The new maps are confined above the donor
chin floor, so the existing lower-neck/bridge boundary is unchanged exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
SOURCE_COLOR = 'R11 restrained rose lip colour'
SOURCE_ROUGH = 'R11 restrained rose lip roughness'
SOURCE_LIP = 'R11 true vermilion rose support'
COLOR = 'R11 refined natural skin colour'
ROUGH = 'R11 refined natural skin roughness'
SUPPORT = 'R11 refined skin face support'
ATTRS = (COLOR, ROUGH, SUPPORT)
MATERIAL = 'R11 refined natural skin and soft rose lips'
MARKER = 'R11_natural_skin_refinement_candidate'
PARAMETERS = {
    'faceRelativeRGB': [.995, .988, .975],
    'cheekRelativeDeltaRGB': [.060, -.035, -.030],
    'noseRelativeDeltaRGB': [.025, -.014, -.010],
    'earRelativeDeltaRGB': [.035, -.018, -.012],
    'foreheadRelativeDeltaRGB': [0., .006, -.016],
    'templeRelativeDeltaRGB': [-.018, -.006, .015],
    'underEyeRelativeDeltaRGB': [-.010, -.008, .008],
    'lipRelativeRGB': [.98, .78, .96],
    'cheekRoughnessDelta': .018, 'noseRoughnessDelta': -.035,
    'foreheadRoughnessDelta': -.018, 'templeRoughnessDelta': .008,
    'lipRoughnessDelta': .025,
    'mesoRelativeColourAmplitude': .008, 'mesoRoughnessAmplitude': .006,
    'chinUpwardFadeMeters': .020}
EXPECTED_OPTICS = {'IOR': 1.4, 'Diffuse Roughness': .12, 'Subsurface Weight': .14,
    'Subsurface Scale': .0018, 'Subsurface Radius': (1., .4, .2),
    'Subsurface IOR': 1.4, 'Subsurface Anisotropy': 0., 'Specular IOR Level': .5,
    'Coat Weight': 0., 'Transmission Weight': 0., 'Metallic': 0.}


def _api(name): return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()
def _hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def _array(items, field, width=3, dtype=np.float32):
    a = np.empty((len(items), width), dtype); items.foreach_get(field, a.ravel()); return a
def _ah(a):
    a = np.ascontiguousarray(a); return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()
def _smooth(x):
    x = np.clip(x, 0., 1.); return x*x*(3.-2.*x)
def _path(relative, suffix='.blend'):
    p = (project_path()/relative).resolve(); p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix != suffix: raise ValueError('Independent reconstruction artifact required')
    return p


def build_maps(donor, landmarks, source_colour, source_rough, lip_support, skin, neck_ring):
    """Pure point-map calculation on verified donor vertex identities."""
    natural = _api('natural_skin_adult_r11')
    points = np.asarray(donor['local'], float); x, y, z = points.T
    old = np.asarray(source_colour, np.float32); oldrough = np.asarray(source_rough, np.float32).reshape(-1)
    lip = np.asarray(lip_support, np.float32).reshape(-1); skin = np.asarray(skin, bool)
    if old.shape != (10582, 4) or any(a.shape != (10582,) for a in (oldrough, lip, skin)):
        raise RuntimeError('Full current 10582-point maps required')
    if not all(np.isfinite(a).all() for a in (old, oldrough, lip)) or lip.min() < 0 or lip.max() > 1:
        raise RuntimeError('Finite current maps and bounded actual vermilion support required')
    groups = natural['_vertices_of_groups']
    facial = groups(donor, {17, 22}); ear = groups(donor, {4, 5})
    external = skin & (facial | ear | groups(donor, {1}))
    graph = natural['_graph'](points, donor['edges'], external)
    eye_centers = np.asarray([landmarks['eyes'][s]['boundsEllipsoidCenter'] for s in ('L','R')])
    eye = eye_centers.mean(0); ipd = abs(eye_centers[0,0]-eye_centers[1,0])
    mouth = landmarks['mouth']; opening = np.asarray(mouth['openingRimCandidate']['sourceVertexIndices'])
    strip=np.asarray(mouth['vermilionStripVertexIndices'],np.int32)
    allowed_lip=np.isin(np.arange(len(points)),strip)
    if len(strip)!=120 or np.any((lip>1e-8)&~allowed_lip) or not 50<int(np.count_nonzero(lip>1e-8))<=80:
        raise RuntimeError('Actual vermilion support escaped the two-strip lip topology')
    mouth_center = points[opening].mean(0); height = eye[2]-mouth_center[2]
    if not .055 < ipd < .075 or not .045 < height < .085: raise RuntimeError('Actual anatomical scale changed')
    front = facial & (y < eye[1]+.025)
    def seed(tx, tz, eligible):
        ids = np.flatnonzero(eligible)
        if not len(ids): raise RuntimeError('Empty anatomical domain')
        return int(ids[np.argmin(((x[ids]-tx)/(.2*ipd))**2+((z[ids]-tz)/(.2*ipd))**2)])
    cheeks = [seed(sign*.57*ipd, eye[2]-.42*height, front & (z<eye[2]) & (z>mouth_center[2]+.2*height)) for sign in (-1,1)]
    nasal = np.flatnonzero(front & (abs(x)<.24*ipd) & (z>mouth_center[2]+.30*height) & (z<eye[2]-.15*height))
    if not len(nasal): raise RuntimeError('No nose-tip support')
    nose_seed = int(nasal[np.argmin(y[nasal])])
    forehead_seed = seed(0., eye[2]+.70*height, front & (z>eye[2]+.30*height))
    temples = [seed(sign*.73*ipd, eye[2]+.13*height, front & (z>eye[2]-.2*height) & (abs(x)>.45*ipd)) for sign in (-1,1)]
    distance = natural['_distance']
    def field(seeds, sigma, limit=.08): return np.exp(-(distance(graph, seeds, limit)/sigma)**2)*front
    cheek = field(cheeks, .36*ipd); nose = field([nose_seed], .20*ipd, .05)
    forehead = field([forehead_seed], .36*ipd); temple = field(temples, .20*ipd, .05)
    lower = [i for s in ('L','R') for i in landmarks['eyes'][s]['orderedLowerInnerToOuter']]
    under = field(lower, .12*ipd, .035)*_smooth((eye[2]-z)/(.15*height))
    ears = natural['_diffuse'](ear.astype(float), graph, 3)*external
    chin_floor = float(points[groups(donor, {22}), 2].min())
    coverage = _smooth((z-chin_floor)/PARAMETERS['chinUpwardFadeMeters'])*external
    ring = np.asarray(neck_ring, np.int32)
    if len(ring)!=54 or len(np.unique(ring))!=54 or ring.min()<0 or ring.max()>=len(points):
        raise RuntimeError('Verified original 54-point neck cage boundary required')
    protected = set(map(int,ring)); edges=np.asarray(donor['edges'],np.int32)
    for _ in range(2):
        touched=np.any(np.isin(edges,list(protected)),axis=1)
        protected.update(map(int,edges[touched].ravel()))
    protect_ids=np.asarray(sorted(protected),np.int32)
    coverage *= _smooth(distance(graph, list(protected), .10)/.018)
    coverage[protect_ids]=0.
    # Closed boundary below the chin remains bit-identical; no neck-map guesses.
    nonlip = coverage*(1.-lip)
    rgb = old[:,:3].astype(float).copy()
    rgb *= 1.+nonlip[:,None]*(np.asarray(PARAMETERS['faceRelativeRGB'])-1.)
    fields = {'cheek':cheek, 'nose':nose, 'ear':ears, 'forehead':forehead, 'temple':temple, 'underEye':under}
    for name, f in fields.items(): rgb *= 1.+(f*nonlip)[:,None]*np.asarray(PARAMETERS[name+'RelativeDeltaRGB'])
    # Bounded 18-41 mm variation on the surface map; never pore/noise amplification.
    meso = (.6*np.sin(2*np.pi*(x/.027+(z-eye[2])/.041)+.4)*np.cos(2*np.pi*(z-eye[2])/.033)
            +.4*np.sin(2*np.pi*(x/.018-(z-eye[2])/.029)+1.2))
    meso *= nonlip*front
    rgb *= 1.+(PARAMETERS['mesoRelativeColourAmplitude']*meso)[:,None]
    rgb *= 1.+lip[:,None]*(np.asarray(PARAMETERS['lipRelativeRGB'])-1.)
    colour = old.copy(); colour[:,:3] = rgb.astype(np.float32)
    rough = oldrough.astype(float).copy()
    rough += nonlip*(PARAMETERS['cheekRoughnessDelta']*cheek+PARAMETERS['noseRoughnessDelta']*nose+
        PARAMETERS['foreheadRoughnessDelta']*forehead+PARAMETERS['templeRoughnessDelta']*temple)
    rough += PARAMETERS['mesoRoughnessAmplitude']*meso+PARAMETERS['lipRoughnessDelta']*lip
    rough = rough.astype(np.float32)
    changed = (coverage>0)|(lip>0)
    if not np.array_equal(colour[~changed], old[~changed]) or not np.array_equal(rough[~changed], oldrough[~changed]):
        raise RuntimeError('Unchanged lower neck or non-skin rows changed')
    if not np.array_equal(colour[~skin], old[~skin]) or not np.array_equal(rough[~skin], oldrough[~skin]):
        raise RuntimeError('Non-skin map changed')
    if np.any(lip[~skin]) or np.any(lip[z<=chin_floor]): raise RuntimeError('Actual lip support extends outside facial domain')
    if not np.array_equal(colour[:,3], old[:,3]): raise RuntimeError('Alpha changed')
    if colour[:,:3].min()<0 or colour[:,:3].max()>1 or rough.min()<.30 or rough.max()>.65 or np.max(abs(colour[:,:3]-old[:,:3]))>.09:
        raise RuntimeError('Candidate map exceeded bounded colour/roughness range')
    return {COLOR:colour, ROUGH:rough, SUPPORT:coverage.astype(np.float32)}, {
        'parameters':PARAMETERS, 'seeds':{'cheeks':cheeks,'nose':nose_seed,'forehead':forehead_seed,'temples':temples},
        'donorChinFloorZ':chin_floor,'newMapZeroAtOrBelowDonorZ':chin_floor,
        'protectedNeckTwoRingCount':len(protected),'protectedNeckRowsExact':bool(np.array_equal(colour[protect_ids],old[protect_ids]) and np.array_equal(rough[protect_ids],oldrough[protect_ids])),
        'noNewNeckColourChange':True,'existingFaceNeckBaseGradientRetained':True,
        'sourceRGBQuantiles':np.percentile(old[coverage>.5,:3],[5,50,95],axis=0).tolist(),
        'candidateRGBQuantiles':np.percentile(colour[coverage>.5,:3],[5,50,95],axis=0).tolist(),
        'sourceRoughnessRange':[float(oldrough.min()),float(oldrough.max())],
        'candidateRoughnessRange':[float(rough.min()),float(rough.max())],
        'lipActiveCount':int(np.count_nonzero(lip)), 'changedRows':int(np.count_nonzero(np.any(colour!=old,axis=1))),
        'maximumAbsoluteColourChange':np.max(abs(colour-old),axis=0).tolist(),
        'interpretation':'Authored spatial colour/roughness candidate, not a measured ethnicity palette or spectral cosmetics model. No global redness, freckles, painted shadow, new normal texture or geometric wrinkle.'}


def _nodes_and_optics(material):
    if not material.use_nodes: raise RuntimeError('Current node material required')
    nt = material.node_tree
    colours = [n for n in nt.nodes if n.type=='VERTEX_COLOR' and n.layer_name==SOURCE_COLOR]
    roughs = [n for n in nt.nodes if n.type=='ATTRIBUTE' and n.attribute_name==SOURCE_ROUGH]
    shaders = [n for n in nt.nodes if n.type=='BSDF_PRINCIPLED']
    if len(colours)!=1 or len(roughs)!=1 or len(shaders)!=1: raise RuntimeError('Expected actual current skin map sources and single Principled')
    bs = shaders[0]
    def ancestors(socket):
        result=set(); stack=[l.from_node for l in socket.links]
        while stack:
            n=stack.pop()
            if n.name in result: continue
            result.add(n.name); stack.extend(l.from_node for s in n.inputs for l in s.links)
        return result
    if colours[0].name not in ancestors(bs.inputs['Base Color']) or roughs[0].name not in ancestors(bs.inputs['Roughness']):
        raise RuntimeError('Current map nodes are not in active Principled input chains')
    actual = {}
    for name, expected in EXPECTED_OPTICS.items():
        s=bs.inputs[name]; value=np.asarray(s.default_value)
        if s.is_linked or np.max(abs(value-np.asarray(expected)))>2e-6: raise RuntimeError('Actual optical input changed; inspect before candidate: '+name)
        actual[name]=value.tolist()
    normal=bs.inputs['Normal']
    if len(normal.links)!=1 or normal.links[0].from_node.type!='BUMP': raise RuntimeError('Current active normal chain changed')
    bump=normal.links[0].from_node
    for name,value in (('Strength',.12),('Distance',.00009)):
        if bump.inputs[name].is_linked or abs(bump.inputs[name].default_value-value)>2e-7: raise RuntimeError('Current micro-normal changed: '+name)
    h=bump.inputs['Height']
    if len(h.links)!=1 or h.links[0].from_node.type!='TEX_NOISE': raise RuntimeError('Current bump height chain changed')
    noise=h.links[0].from_node; vector=noise.inputs['Vector']
    if noise.inputs['Scale'].is_linked or abs(noise.inputs['Scale'].default_value-5000)>1e-5 or len(vector.links)!=1 or getattr(vector.links[0].from_node,'attribute_name',None)!='R04 skin rest position':
        raise RuntimeError('Current rest-space microdetail changed')
    actual.update(subsurfaceMethod=bs.subsurface_method, distribution=bs.distribution,
        normalChain={'bump':bump.name,'strength':float(bump.inputs['Strength'].default_value),'distance':float(bump.inputs['Distance'].default_value),'noiseScale':float(noise.inputs['Scale'].default_value)},
        retainedExact=True, radiusTimesScaleMeters=(np.asarray(actual['Subsurface Radius'])*actual['Subsurface Scale']).tolist())
    return (colours[0].name,roughs[0].name),actual


def _protected(head, slot):
    state=_api('eye_margin_refine_r11')['_state'](head,slot)
    state['attributes']=[row for row in state['attributes'] if row[0] not in ATTRS]
    return state


def _boundary_compare(before, after):
    if not np.array_equal(before['ring']['ids'],after['ring']['ids']) or not np.array_equal(before['ring']['points'],after['ring']['points']):
        raise RuntimeError('Actual evaluated head-neck boundary geometry changed')
    mapping={SOURCE_COLOR:COLOR,SOURCE_ROUGH:ROUGH}; records={}
    for name, item in before['samples'].items():
        destination=mapping.get(name,name); other=after['samples'].get(destination)
        if other is None or item['absent']!=other['absent']:
            raise RuntimeError('Actual boundary input availability changed: '+name)
        if not item['absent']:
            if item['type']!=other['type'] or not np.array_equal(item['values'],other['values']):
                raise RuntimeError('Actual subdivided head-neck input changed: '+name)
            records[name]={'candidateAttribute':destination,'sha256':_ah(item['values']),'maximumDifference':0.}
    return {'evaluatedBoundaryCount':len(before['ring']['ids']),'positionAndIdsExact':True,
        'allShaderAttributeSamplesExact':True,'inputs':records}


def apply():
    import bpy
    source=Path(bpy.data.filepath).resolve(); source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT': raise RuntimeError('Independent background Object Mode candidate required')
    head=bpy.data.objects[HEAD]; mesh=head.data
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582: raise RuntimeError('Actual complete local female head required')
    if MARKER in bpy.context.scene or any(mesh.attributes.get(n) for n in ATTRS) or bpy.data.materials.get(MATERIAL): raise RuntimeError('Preserve earlier refinement candidate')
    natural=_api('natural_skin_adult_r11'); donor,landmarks,provenance=natural['_donor']()
    if not np.array_equal(_array(mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']): raise RuntimeError('Actual donor vertex identities changed')
    arrays=[]
    for name,kind,width,field in ((SOURCE_COLOR,'FLOAT_COLOR',4,'color'),(SOURCE_ROUGH,'FLOAT',1,'value'),(SOURCE_LIP,'FLOAT',1,'value')):
        attr=mesh.attributes.get(name)
        if not attr or attr.domain!='POINT' or attr.data_type!=kind: raise RuntimeError('Actual source map missing: '+name)
        arrays.append(_array(attr.data,field,width))
    slots=[i for i,s in enumerate(head.material_slots) if s.material and s.material.use_nodes and any(n.type=='VERTEX_COLOR' and n.layer_name==SOURCE_COLOR for n in s.material.node_tree.nodes)]
    if len(slots)!=1: raise RuntimeError('One current skin material slot required')
    slot=slots[0]; original=head.material_slots[slot].material
    node_names,optics=_nodes_and_optics(original)
    skin=np.zeros(len(mesh.vertices),bool)
    for poly in mesh.polygons:
        if poly.material_index==slot: skin[list(poly.vertices)]=True
    selection_path=project_path()/'work/reconstruction-r02/female-head-selection-r11.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if selection['source_npz_sha256']!=provenance['sourceArraysSha256']: raise RuntimeError('Neck boundary source arrays changed')
    maps,plan=build_maps(donor,landmarks,*arrays,skin,selection['neck_ring_old_indices'])
    face=_api('runtime_face_refresh_r08'); uv=_api('production_uv_r08'); eye=_api('eye_margin_refine_r11'); lip=_api('lip_surface_finish_r11')
    bpy.context.view_layer.update(); settings=uv['_uv_settings'](mesh); before=_hash(_protected(head,slot)); external=eye['_external'](bpy,head,[],'')
    original_hash=face['_material_contract'](original); graph=lip['_normalized'](original,node_names,face); old_fake=original.use_fake_user; source_sha=_sha(source)
    boundary_api=_api('adult_neck_surface_continuity_r11'); boundary_before=boundary_api['_source_surface'](head)
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
        'sourceMayContainEarlierInMemoryModules':bool(bpy.data.is_dirty),'head':HEAD,'materialSlot':slot,'originalMaterial':original.name,
        'originalMaterialSha256':original_hash,'originalFakeUser':old_fake,'originalColourSettings':settings,'provenance':provenance,
        'actualOpticsBefore':optics,'plan':plan,'neckSelectionSha256':_sha(selection_path),'sourceSaved':False,'appearanceAccepted':False,
        'evidence':{'lightingAB':'previews/reconstruction-r02/neutral-lighting-ab-r11-v2-front/lighting-comparison.json',
            'observedSourceSha256':'41ce34d8a6f52e59b29ea40c80e33758359fbffb9ca7c217c1c29be6ee8122e8',
            'finding':'Actual original light flattens form, small-key neutral overdarkens hair; both show weak lip distinction and uniform pale/clay-like skin. Colour/roughness trial does not repair remaining shape/eye/hair issues.'},
        'scope':'Copied material and derived colour/roughness point maps only; same source optics, SSS, active normal chain, eye-margin layers, UVs, geometry, keys, rigs and lights.',
        'requiredReview':'Same balanced lighting source/candidate front and close-up, then smile and face-neck transition. No visual acceptance without images.'}
    made=[]; candidate=None; success=False
    try:
        for name,values in maps.items():
            attr=mesh.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT') if name==COLOR else mesh.attributes.new(name=name,type='FLOAT',domain='POINT')
            made.append(name); attr.data.foreach_set('color' if name==COLOR else 'value',values.ravel())
        uv['_restore_uv_settings'](mesh,settings); candidate=original.copy(); candidate.name=MATERIAL
        candidate.node_tree.nodes[node_names[0]].layer_name=COLOR; candidate.node_tree.nodes[node_names[1]].attribute_name=ROUGH
        if lip['_normalized'](candidate,node_names,face)!=graph: raise RuntimeError('Existing graph changed beyond two map references')
        original.use_fake_user=True; head.material_slots[slot].material=candidate; bpy.context.view_layer.update()
        boundary_guard=_boundary_compare(boundary_before,boundary_api['_source_surface'](head))
        if _hash(_protected(head,slot))!=before or eye['_external'](bpy,head,[],'')!=external: raise RuntimeError('Geometry/keys/UV/source maps or other objects changed')
        if face['_material_contract'](original)!=original_hash or _sha(source)!=source_sha: raise RuntimeError('Original material or saved source changed')
        report.update(candidateMaterial=candidate.name,candidateMaterialSha256=face['_material_contract'](candidate),
            candidateAttributeHashes={name:_ah(value) for name,value in maps.items()},originalSourceAndOtherObjectsExact=True,
            actualOpticsRetainedExact=True,headGeometryKeysAndExistingMapsExact=True,actualHeadNeckBoundary=boundary_guard)
        bpy.context.scene[MARKER]=json.dumps(report); success=True; return report
    finally:
        if not success:
            head.material_slots[slot].material=original; original.use_fake_user=old_fake
            for name in reversed(made):
                if mesh.attributes.get(name): mesh.attributes.remove(mesh.attributes[name])
            uv['_restore_uv_settings'](mesh,settings)
            if candidate is not None and candidate.users==0: bpy.data.materials.remove(candidate)
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene: return False
    report=json.loads(bpy.context.scene[MARKER]); head=bpy.data.objects[report['head']]; mesh=head.data; face=_api('runtime_face_refresh_r08')
    old=bpy.data.materials[report['originalMaterial']]; new=bpy.data.materials[report['candidateMaterial']]
    if head.material_slots[report['materialSlot']].material!=new or face['_material_contract'](old)!=report['originalMaterialSha256'] or face['_material_contract'](new)!=report['candidateMaterialSha256']: raise RuntimeError('Material edited after candidate; preserve edits')
    for name,digest in report['candidateAttributeHashes'].items():
        attr=mesh.attributes.get(name)
        if attr is None: raise RuntimeError('Candidate map missing')
        a=_array(attr.data,'color',4) if name==COLOR else _array(attr.data,'value',1).ravel()
        if _ah(a)!=digest: raise RuntimeError('Candidate map edited; preserve edits')
    head.material_slots[report['materialSlot']].material=old; old.use_fake_user=report['originalFakeUser']
    for name in reversed(ATTRS): mesh.attributes.remove(mesh.attributes[name])
    _api('production_uv_r08')['_restore_uv_settings'](mesh,report['originalColourSettings'])
    new.use_fake_user=True; del bpy.context.scene[MARKER]; bpy.context.view_layer.update()
    return {'restoredMaterial':old.name,'retainedCandidateMaterial':new.name,'savedBlend':False}


def prepare(source_relative, output_relative, expected_sha):
    import bpy
    source=_path(source_relative); output=_path(output_relative); record=output.with_suffix('.skin-refinement.json')
    if source==output or output.exists() or record.exists(): raise RuntimeError('New independent output paths required')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty or _sha(source)!=expected_sha: raise RuntimeError('Fresh exact SHA-verified source required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)): raise RuntimeError('Candidate save failed')
    if _sha(source)!=expected_sha: raise RuntimeError('Input source changed')
    report.update(candidateSaved=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with record.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_SKIN_REFINEMENT_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--source',required=True); parser.add_argument('--output',required=True); parser.add_argument('--expected-source-sha256',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare(args.source,args.output,args.expected_source_sha256)
