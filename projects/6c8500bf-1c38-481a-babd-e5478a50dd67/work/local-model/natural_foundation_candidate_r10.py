"""Current-source skin observation and one reversible light-foundation study.

No import-time Blender activity. Observation may read the saved master; any
candidate must use an explicit independent work copy and a new output file.
No race-derived skin swatch, lighting/exposure change, geometry edit or render.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R01 Head';MATERIAL='R04 physical skin'
OBSERVATION='work/reconstruction-r02/skin-observation-r10.json'
MARKER='R10_natural_foundation_candidate'
COLOR='R10 natural foundation colour'
ROUGH='R10 natural foundation roughness'
MASK='R10 light foundation coverage'
CANDIDATE_MATERIAL='R10 natural warm light foundation candidate'
NEW_ATTRIBUTES=(COLOR,ROUGH,MASK)
PARAMETERS={'foundationLinearGain':.035,'powderRoughnessIncrement':.012,
    'cheekRelativePigment':(.035,-.008,-.015),'noseRelativePigment':(.018,-.004,-.008),
    'lipExistingColourResidualGain':.08}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _path(relative,suffix,allow_master=False):
    p=project_path().resolve();path=(p/relative).resolve()
    if not (allow_master and path==p/'work/character.blend'):
        path.relative_to(p/'work/reconstruction-r02')
    if path.suffix!=suffix:raise ValueError('Unexpected file type')
    return path


def _array(collection,field,width=3,dtype=np.float32):
    result=np.empty(len(collection)*width,dtype);collection.foreach_get(field,result)
    return result.reshape(-1,width)


def _array_hash(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _guard(source_relative,allow_master=False):
    import bpy
    source=_path(source_relative,'.blend',allow_master)
    if (not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.data.is_dirty
            or Path(bpy.data.filepath).resolve()!=source):
        raise RuntimeError('Open the exact saved source in a fresh background process')
    head=bpy.data.objects[HEAD]
    slots=[i for i,s in enumerate(head.material_slots) if s.material and s.material.name==MATERIAL]
    if len(slots)!=1:raise RuntimeError('Expected the explicit current R04 physical-skin material domain')
    if head.library or head.data.library or head.data.users!=1:raise RuntimeError('Expected local single-user head data')
    return source,head,slots[0],head.material_slots[slots[0]].material


def _shader(material):
    nodes=[n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED']
    if len(nodes)!=1:raise RuntimeError('One actual Principled skin shader required')
    return nodes[0]


def _input_record(shader):
    face=_api('runtime_face_refresh_r08');rows={}
    for socket in shader.inputs:
        rows[socket.name]={'value':face['_plain'](socket.default_value) if hasattr(socket,'default_value') else None,
            'links':[{'node':l.from_node.name,'type':l.from_node.type,'socket':l.from_socket.name,
                      'attribute':getattr(l.from_node,'attribute_name',None),
                      'colourLayer':getattr(l.from_node,'layer_name',None)} for l in socket.links]}
    return rows


def observe(source_relative,output_relative=OBSERVATION):
    """Export current colour/roughness maps and actual material inputs only."""
    import bpy
    source,head,index,material=_guard(source_relative,allow_master=True)
    output=_path(output_relative,'.json');payload=output.with_suffix('.npz')
    if output.exists() or payload.exists():raise RuntimeError('Use new observation names; retain prior evidence')
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08');groom=_api('posterior_groom_r09')
    mesh=head.data;colour=mesh.color_attributes.get('R01_region_color')
    rough=mesh.attributes.get('R04 skin roughness');rest=mesh.attributes.get('R04 skin rest position')
    if colour is None or colour.domain!='POINT' or colour.data_type!='FLOAT_COLOR':
        raise RuntimeError('Expected the actual point-domain FLOAT_COLOR region source; do not infer another mapping')
    if rough is None or rough.domain!='POINT' or rough.data_type!='FLOAT' or rest is None or rest.domain!='POINT':
        raise RuntimeError('Current anatomical roughness/rest-space inputs need explicit inspection')
    source_sha=_sha(source);material_hash=face['_material_contract'](material)
    geometry_hash=groom['_data_hash'](mesh)
    bpy.context.view_layer.update()
    pose_before=uv['_pose_contract']();rig=bpy.data.objects['Character_Rig']
    pose=rig.data.pose_position;frame=bpy.context.scene.frame_current;subframe=bpy.context.scene.frame_subframe
    arrays={};report={}
    try:
        bpy.context.scene.frame_set(1);rig.data.pose_position='REST';bpy.context.view_layer.update()
        mixed,mix_record=uv['current_mix'](head)
        material_mask=np.zeros(len(mesh.vertices),bool)
        for polygon in mesh.polygons:
            if polygon.material_index==index:material_mask[list(polygon.vertices)]=True
        arrays.update(coordinates_local=mixed,coordinates_world=groom['_world'](mixed,head.matrix_world),
            basis=_array(mesh.shape_keys.reference_key.data,'co'),
            source_colour=_array(colour.data,'color',4),source_roughness=_array(rough.data,'value',1).ravel(),
            original_micro_rest=_array(rest.data,'vector'),skin_material_vertices=material_mask,
            polygon_materials=_array(mesh.polygons,'material_index',1,np.int32).ravel(),
            loops=_array(mesh.loops,'vertex_index',1,np.int32).ravel(),
            face_starts=_array(mesh.polygons,'loop_start',1,np.int32).ravel(),
            face_sizes=_array(mesh.polygons,'loop_total',1,np.int32).ravel(),
            head_matrix=np.asarray(head.matrix_world,dtype=float))
        eye_centers=[]
        for side in ('L','R'):
            obj=bpy.data.objects['R01 Sclera '+side]
            points=groom['_world'](_array(obj.data.vertices,'co'),obj.matrix_world)
            eye_centers.append((points.min(0)+points.max(0))*.5)
        arrays['eye_centers_world']=np.asarray(eye_centers)
        usage=[]
        for obj in bpy.context.scene.objects:
            if obj.hide_render:continue
            for slot_index,slot in enumerate(obj.material_slots):
                mat=slot.material
                if mat is None:continue
                if mat==material or 'skin' in mat.name.lower() or obj.name.startswith('Hand '):
                    shaders=[n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'] if mat.use_nodes else []
                    usage.append({'object':obj.name,'slot':slot_index,'material':mat.name,
                        'sameHeadMaterial':mat==material,
                        'principled':[_input_record(s) for s in shaders]})
        selected=arrays['source_colour'][material_mask,:3]
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'helperSha256AtObservation':_sha(Path(__file__)),'head':HEAD,'material':material.name,'materialSlot':index,
            'materialSha256':material_hash,'materialGraph':face['_tree_contract'](material.node_tree),
            'principledInputs':_input_record(_shader(material)),
            'materialUsage':usage,'colourAttribute':{'name':colour.name,'domain':colour.domain,'type':colour.data_type},
            'roughnessAttribute':{'name':rough.name,'domain':rough.domain,'type':rough.data_type},
            'sourceColourRange':[selected.min(0).tolist(),selected.max(0).tolist()],
            'sourceColourQuantiles':np.quantile(selected,[.05,.5,.95],axis=0).tolist(),
            'sourceRoughnessRange':[float(arrays['source_roughness'][material_mask].min()),float(arrays['source_roughness'][material_mask].max())],
            'skinMaterialVertexCount':int(material_mask.sum()),'headCurrentMix':mix_record,
            'geometryDataSha256':geometry_hash,
            'sourceAttributeHashes':{name:_array_hash(arrays[name]) for name in ('source_colour','source_roughness','original_micro_rest')},
            'colourSettings':uv['_uv_settings'](mesh),
            'colourManagement':{'view':bpy.context.scene.view_settings.view_transform,'look':bpy.context.scene.view_settings.look,
                'exposure':bpy.context.scene.view_settings.exposure,'gamma':bpy.context.scene.view_settings.gamma},
            'lights':[{'name':o.name,'energy':o.data.energy,'colour':list(o.data.color),
                'matrixWorld':[list(r) for r in o.matrix_world]} for o in bpy.context.scene.objects if o.type=='LIGHT'],
            'savedBlend':False,'geometryEdited':False,
            'scope':'Actual current physical-skin colour/roughness mapping, frame-one relative mix in temporary REST. No race-derived target colour, tone change, render or master write.'}
    finally:
        rig.data.pose_position=pose;bpy.context.scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before or groom['_data_hash'](mesh)!=geometry_hash:
        raise RuntimeError('Observation did not restore the original pose/data')
    if face['_material_contract'](material)!=material_hash or _sha(source)!=source_sha:
        raise RuntimeError('Material or saved source changed during observation')
    with payload.open('xb') as handle:np.savez_compressed(handle,**arrays)
    report['arrays']={'path':str(payload.relative_to(project_path())),'sha256':_sha(payload)}
    with output.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R10_SKIN_OBSERVED '+json.dumps({'report':str(output),'sourceSha256':source_sha,'arraysSha256':report['arrays']['sha256']}),flush=True)
    return report


def _smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def build_maps(arrays):
    """Pure NumPy; masks use this observation's geometry and existing lip pigment.

    This is an art-direction candidate, not a calibrated spectral skin model.
    Gains are relative to the observed warm baseline, not an ethnic swatch.
    """
    world=np.asarray(arrays['coordinates_world'],float)
    old=np.asarray(arrays['source_colour'],np.float32)
    old_rough=np.asarray(arrays['source_roughness'],np.float32)
    skin=np.asarray(arrays['skin_material_vertices'],bool)
    eyes=np.asarray(arrays['eye_centers_world'],float)
    eye=eyes.mean(0);ipd=float(np.linalg.norm(eyes[0]-eyes[1]))
    if not .04<ipd<.10:raise RuntimeError('Unexpected actual eye spacing; inspect the source instead of scaling guessed masks')
    base=np.median(old[skin,:3],axis=0).astype(float)
    # The current source has explicit darker warm vermilion vertices. Retain
    # their full continuous residual; never repaint a geometric rectangle.
    deficit=np.maximum(base[1]-old[:,1],0.)
    lip_samples=skin&(deficit>.015)&(abs(world[:,0]-eye[0])<.45*ipd)&(world[:,2]<eye[2])
    if not 30<=int(lip_samples.sum())<=250:
        raise RuntimeError('Observed lip-pigment support differs; require new anatomical inspection')
    mouth=np.median(world[lip_samples],axis=0);height=float(eye[2]-mouth[2])
    if not .035<height<.075:raise RuntimeError('Observed eye-to-pigment spacing differs')
    lip_protect=_smooth(deficit/.012)
    dx=world[:,0]-eye[0];z=world[:,2];y=world[:,1]
    # Soft front-to-side and lower-neck fades. Rear scalp, shoulders and hands
    # retain the source exactly. There is no hard foundation edge at the jaw.
    front=1.-_smooth((y-(eye[1]+.12*ipd))/(.72*ipd))
    side=1.-_smooth((abs(dx)-.98*ipd)/(.35*ipd))
    neck=_smooth((z-(mouth[2]-1.72*height))/(1.04*height))
    lid_protect=np.zeros(len(world))
    for center in eyes:
        r2=((world[:,0]-center[0])/(.32*ipd))**2+((z-center[2])/(.18*ipd))**2
        lid_protect=np.maximum(lid_protect,np.exp(-r2*r2))
    coverage=skin.astype(float)*front*side*neck*(1.-lip_protect)*(1.-.7*lid_protect)
    cheek=np.exp(-((abs(dx)-.62*ipd)/(.27*ipd))**2-((z-(mouth[2]+.56*height))/(.35*height))**2)
    cheek*=skin*front*(1.-lip_protect)*(1.-lid_protect)
    # The nose location is measured from actual frontmost central skin points,
    # bounded between the observed lips and eyes. It is not an old R04 z value.
    nasal=skin&(abs(dx)<.18*ipd)&(z>mouth[2]+.22*height)&(z<eye[2]-.15*height)
    ids=np.flatnonzero(nasal)
    if len(ids)<5:raise RuntimeError('No measured central nose support')
    ids=ids[np.argsort(world[ids,1],kind='stable')[:5]];nose=np.median(world[ids],axis=0)
    nose_mask=np.exp(-((world[:,0]-nose[0])/(.13*ipd))**2-((z-nose[2])/(.22*height))**2)
    nose_mask*=skin*front*(1.-lip_protect)
    result=old.copy();rgb=old[:,:3].astype(float)
    rgb*=1.+PARAMETERS['foundationLinearGain']*coverage[:,None]
    rgb+=base[None,:]*(cheek[:,None]*np.asarray(PARAMETERS['cheekRelativePigment'])+
                       nose_mask[:,None]*np.asarray(PARAMETERS['noseRelativePigment']))
    lip_zone=skin&(abs(dx)<.48*ipd)&(abs(z-mouth[2])<.32*height)
    lip_residual=(old[:,:3]-base)*PARAMETERS['lipExistingColourResidualGain']*lip_protect[:,None]*lip_zone[:,None]
    rgb+=lip_residual
    result[:,:3]=rgb.astype(np.float32)
    rough=(old_rough.astype(float)+PARAMETERS['powderRoughnessIncrement']*coverage).astype(np.float32)
    coverage=coverage.astype(np.float32)
    if np.any(result<0) or np.any(result>1):raise RuntimeError('Candidate would clip pigment values')
    delta=result[:,:3]-old[:,:3]
    if float(np.abs(delta).max())>.040001:raise RuntimeError('Unexpected large pigmentation change')
    if not np.array_equal(result[~skin],old[~skin]) or not np.array_equal(rough[~skin],old_rough[~skin]):
        raise RuntimeError('A non-skin material domain changed')
    if not np.array_equal(result[:,3],old[:,3]):raise RuntimeError('Skin alpha must remain exact')
    report={'parameters':PARAMETERS,'observedBaselineLinearRGB':base.tolist(),
        'eyeCentersWorld':eyes.tolist(),'interEyeDistanceMeters':ipd,
        'measuredMouthPigmentCenterWorld':mouth.tolist(),'measuredNoseFrontWorld':nose.tolist(),
        'lipSupportVertexCount':int(lip_samples.sum()),'skinVertexCount':int(skin.sum()),
        'exactUniformSourceVertexCount':int(np.all(old[skin,:3]==base,axis=1).sum()),
        'coverageOverOnePercentVertexCount':int((coverage>.01).sum()),
        'maximumAbsoluteRGBChange':float(np.abs(delta).max()),
        'rgbDeltaMinimum':delta[skin].min(0).tolist(),'rgbDeltaMaximum':delta[skin].max(0).tolist(),
        'maximumRoughnessIncrease':float((rough-old_rough).max()),
        'regionCoordinates':'Current frame-one REST combination, without changing geometry; masks follow existing point indices through all expressions.',
        'lipTreatment':'Protect original warm lip colours from foundation; increase only the existing colour residual by at most 8%, without expanding the vermilion footprint.',
        'outsideSkinAndAlphaUnchanged':True,
        'limits':['Face proportions, hard cheek planes, eyes and hair volume remain unchanged.',
                  'Relative pigment and roughness adjustment is not a measured ethnic colour target or a spectral skin calibration.',
                  'Same-light front, quarter and side renders remain required; numerical bounds do not approve appearance.']}
    return {COLOR:result,ROUGH:rough,MASK:coverage},report


def _load_observation(relative):
    path=_path(relative,'.json');record=json.loads(path.read_text(encoding='utf-8'))
    payload=_path(record['arrays']['path'],'.npz')
    if _sha(payload)!=record['arrays']['sha256']:raise RuntimeError('Observed skin arrays changed')
    with np.load(payload) as data:arrays={name:data[name] for name in data.files}
    return record,arrays,_sha(path)


def _head_contract(head,index,uv):
    state=uv['_original_contract'](head,tuple(x.name for x in head.data.uv_layers));state.pop('sha256')
    state['attributes']=[row for row in state['attributes'] if row['name'] not in NEW_ATTRIBUTES]
    state['materialSlots'][index]={'link':head.material_slots[index].link,'explicitSkinSlot':index}
    return _hash(state)


def _source_nodes(material):
    shader=_shader(material);nodes=[]
    for socket,node_type,property_name,expected in (
        ('Base Color','VERTEX_COLOR','layer_name','R01_region_color'),
        ('Roughness','ATTRIBUTE','attribute_name','R04 skin roughness')):
        links=shader.inputs[socket].links
        if len(links)!=1 or links[0].from_node.type!=node_type:
            raise RuntimeError('Current skin input chain needs explicit inspection: '+socket)
        node=links[0].from_node
        if getattr(node,property_name)!=expected:raise RuntimeError('Unexpected original skin map: '+socket)
        nodes.append((node.name,property_name,expected))
    return nodes


def _normalized_graph(material,face,nodes):
    graph=face['_tree_contract'](material.node_tree)
    for node_name,property_name,old_name in nodes:
        row=next(r for r in graph['nodes'] if r['name']==node_name)
        row['properties'][property_name]=old_name
    return _hash(graph)


def _slot_contract(bpy,head,index):
    return {o.name:[(i,s.link,s.material.as_pointer() if s.material else None)
        for i,s in enumerate(o.material_slots) if not(o==head and i==index)] for o in bpy.context.scene.objects}


def _current_observation_match(head,material,observation,arrays,uv,face,groom):
    import bpy
    if groom['_data_hash'](head.data)!=observation['geometryDataSha256']:
        raise RuntimeError('Actual head mesh/attributes differ from the current-source observation')
    if face['_material_contract'](material)!=observation['materialSha256']:
        raise RuntimeError('Actual skin shader differs from the observed source')
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position
    frame=bpy.context.scene.frame_current;subframe=bpy.context.scene.frame_subframe
    before=uv['_pose_contract']()
    try:
        bpy.context.scene.frame_set(1);rig.data.pose_position='REST';bpy.context.view_layer.update()
        mix,record=uv['current_mix'](head)
        world=groom['_world'](mix,head.matrix_world)
        if record['currentMixSha256']!=observation['headCurrentMix']['currentMixSha256']:
            raise RuntimeError('Current relative-key combination differs from the observed source')
        if np.max(np.abs(world-arrays['coordinates_world']))>2e-7:
            raise RuntimeError('Current head attachment differs from the observed source')
    finally:
        rig.data.pose_position=pose;bpy.context.scene.frame_set(frame,subframe=subframe);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=before:raise RuntimeError('Source matching did not restore original motion/pose')
    return {'actualMeshAttributesMaterialCurrentMixAndAttachmentMatch':True,
            'sourcePathOrByteSHAEqualityRequired':False,
            'reason':'The observed master and an explicit independent refined copy may have different .blend bytes. Actual mesh, map, shader, mixed-coordinate and attachment fingerprints must match.'}


def run(source_relative,observation_relative=OBSERVATION,diagnostics_only=False):
    import bpy
    source,head,index,original=_guard(source_relative)
    if bpy.context.scene.get(MARKER) or bpy.data.materials.get(CANDIDATE_MATERIAL):
        raise RuntimeError('Preserve or restore the previous foundation candidate first')
    if any(head.data.attributes.get(name) for name in NEW_ATTRIBUTES):raise RuntimeError('Candidate attribute name already exists')
    if original.library:raise RuntimeError('Local original material required')
    observation,arrays,observation_sha=_load_observation(observation_relative)
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08');groom=_api('posterior_groom_r09')
    perioral=_api('perioral_delta_r09');bpy.context.view_layer.update()
    source_match=_current_observation_match(head,original,observation,arrays,uv,face,groom)
    maps,plan=build_maps(arrays);nodes=_source_nodes(original)
    original_hash=face['_material_contract'](original);graph_hash=_normalized_graph(original,face,nodes)
    source_sha=_sha(source)
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
        'observation':observation_relative,'observationSha256':observation_sha,
        'observedSourceSha256':observation['sourceSha256'],'observedArraysSha256':observation['arrays']['sha256'],
        'sourceContentMatch':source_match,'helperSha256':_sha(Path(__file__)),
        'object':HEAD,'materialSlot':index,'originalMaterial':original.name,
        'originalMaterialSha256':original_hash,'candidateMaterial':CANDIDATE_MATERIAL,
        'plan':plan,'appearanceAccepted':False,'savedBlend':False,'writtenInMemory':False,
        'materialScope':'Derived colour/roughness maps only. Existing IOR, specular, subsurface, rest-space noise, bump, alpha and all lighting remain exact.',
        'requiredEvidence':'Same-light current-source and candidate front, quarter and side, with no exposure/DOF changes; inspect face-to-neck transition and natural lips/cheeks.',
        'runtimeRefreshRequired':'New explicit R10 material and attributes must be rebaked/adapted before any runtime equivalence claim.'}
    if diagnostics_only:return report
    mesh=head.data;settings=uv['_uv_settings'](mesh)
    head_before=_head_contract(head,index,uv);others_before=perioral['_other_scene_contract'](bpy,uv,groom)
    slots_before=_slot_contract(bpy,head,index);old_fake=original.use_fake_user
    material_settings=uv['_scalar_rna'](original)
    for key in ('name','use_fake_user'):material_settings.pop(key,None)
    candidate=None;created=[];committed=False
    try:
        for name,array in maps.items():
            if name==COLOR:
                attribute=mesh.color_attributes.new(name=name,type='FLOAT_COLOR',domain='POINT');field='color'
            else:
                attribute=mesh.attributes.new(name=name,type='FLOAT',domain='POINT');field='value'
            created.append(name);attribute.data.foreach_set(field,np.asarray(array,np.float32).ravel())
        uv['_restore_uv_settings'](mesh,settings)
        candidate=original.copy();candidate.name=CANDIDATE_MATERIAL
        if candidate.name!=CANDIDATE_MATERIAL:raise RuntimeError('Candidate name was not unique')
        candidate.node_tree.nodes[nodes[0][0]].layer_name=COLOR
        candidate.node_tree.nodes[nodes[1][0]].attribute_name=ROUGH
        candidate['R10_source_material']=original.name
        candidate['R10_source_material_sha256']=original_hash
        candidate['R10_observation_sha256']=observation_sha
        candidate['R10_candidate_status']='Unaccepted; actual same-light review required'
        if _normalized_graph(candidate,face,nodes)!=graph_hash:
            raise RuntimeError('Skin graph changed beyond the two explicit attribute references')
        changed_settings=uv['_scalar_rna'](candidate)
        for key in ('name','use_fake_user'):changed_settings.pop(key,None)
        if _hash(changed_settings)!=_hash(material_settings):raise RuntimeError('Material render settings changed during copy')
        original.use_fake_user=True;head.material_slots[index].material=candidate;bpy.context.view_layer.update()
        if _head_contract(head,index,uv)!=head_before:raise RuntimeError('Original geometry/keys/attributes/UV/weights/attachment changed')
        if perioral['_other_scene_contract'](bpy,uv,groom)!=others_before:
            raise RuntimeError('Other geometry, native motion cache, action or transform changed')
        if _slot_contract(bpy,head,index)!=slots_before:raise RuntimeError('Other material assignment changed')
        if face['_material_contract'](original)!=original_hash or _sha(source)!=source_sha:
            raise RuntimeError('Original material or source file changed')
        hashes={name:_array_hash(_array(mesh.attributes[name].data,'color',4) if name==COLOR
            else _array(mesh.attributes[name].data,'value',1).ravel()) for name in NEW_ATTRIBUTES}
        report.update(writtenInMemory=True,originalFakeUser=old_fake,originalColourSettings=settings,
            candidateMaterialSha256=face['_material_contract'](candidate),candidateAttributeHashes=hashes,
            originalGeometryKeysUVAttributesRigAndCachesUnchanged=True,originalOtherMaterialAssignmentsUnchanged=True,
            originalSkinGraphAndMapsRetained=True,headOriginalContractSha256=head_before)
        bpy.context.scene[MARKER]=json.dumps(report)
        committed=True;return report
    finally:
        if not committed:
            head.material_slots[index].material=original
            for name in reversed(created):
                attribute=mesh.attributes.get(name)
                if attribute:mesh.attributes.remove(attribute)
            uv['_restore_uv_settings'](mesh,settings);original.use_fake_user=old_fake
            if candidate is not None:
                candidate.use_fake_user=False
                if candidate.users==0:bpy.data.materials.remove(candidate)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            bpy.context.view_layer.update()


def apply_and_save(source_relative,output_relative,observation_relative=OBSERVATION):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New independent candidate/report paths required')
    report=run(source_relative,observation_relative)
    if output.exists():raise RuntimeError('Output appeared during preparation')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2)
    print('R10_FOUNDATION_SAVED '+json.dumps({'sourceSha256':report['sourceSha256'],'output':str(output),'outputSha256':report['outputSha256']}),flush=True)
    return report


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[HEAD];mesh=head.data
    uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    original=bpy.data.materials.get(report['originalMaterial']);candidate=bpy.data.materials.get(report['candidateMaterial'])
    if original is None or face['_material_contract'](original)!=report['originalMaterialSha256']:
        raise RuntimeError('Original physical-skin material missing or edited')
    if candidate is None or face['_material_contract'](candidate)!=report['candidateMaterialSha256']:
        raise RuntimeError('Candidate material was edited; preserve those edits before rollback')
    index=report['materialSlot']
    if head.material_slots[index].material!=candidate:raise RuntimeError('Candidate skin assignment changed')
    for name,expected in report['candidateAttributeHashes'].items():
        attr=mesh.attributes.get(name)
        if attr is None:raise RuntimeError('Candidate attribute missing: '+name)
        data=_array(attr.data,'color',4) if name==COLOR else _array(attr.data,'value',1).ravel()
        if _array_hash(data)!=expected:raise RuntimeError('Candidate attribute was edited: '+name)
    if _head_contract(head,index,uv)!=report['headOriginalContractSha256']:
        raise RuntimeError('Other head state changed after this pass; inspect before rollback')
    head.material_slots[index].material=original
    for name in reversed(NEW_ATTRIBUTES):mesh.attributes.remove(mesh.attributes[name])
    uv['_restore_uv_settings'](mesh,report['originalColourSettings'])
    original.use_fake_user=report['originalFakeUser'];candidate.use_fake_user=True
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    if _head_contract(head,index,uv)!=report['headOriginalContractSha256']:
        raise RuntimeError('Original head data contract was not restored')
    return {'restoredMaterial':original.name,'originalMapsAndHeadStateRestored':True,
            'retainedCandidateMaterial':candidate.name,'savedBlend':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('observe','prepare','audit'),required=True)
    parser.add_argument('--source');parser.add_argument('--observation',default=OBSERVATION)
    parser.add_argument('--output')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if args.mode=='observe':
        if not args.source:parser.error('--source required')
        observe(args.source,args.observation)
    elif args.mode=='prepare':
        if not args.source or not args.output:parser.error('--source and --output required')
        apply_and_save(args.source,args.output,args.observation)
    else:
        _,arrays,_=_load_observation(args.observation)
        print(json.dumps(build_maps(arrays)[1],ensure_ascii=False,indent=2))
