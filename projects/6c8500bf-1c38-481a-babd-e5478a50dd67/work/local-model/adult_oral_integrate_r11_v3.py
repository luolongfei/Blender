"""Create fitted CC0 teeth, gingiva and tongue on the current native adult jaw.

Adds editable derived meshes and retains the exact imported donor hidden.
Preserves donor UVs when present, source face/vertex IDs and the old oral
assets. The head's continuous original oral bag remains; donor bag157 is not
duplicated. Production native jaw controls all lower parts, with a softer
posterior attachment for gingiva and tongue. No main-source overwrite.
V3 treats the old fit only as a candidate placement. Current evaluated L3
lip-domain fingerprints and continuous dental/lip contact are checked through
early jaw opening and warm-smile combinations before any candidate is saved.
Permitted tooth-root/gingival embedding is outside the dental/lip pair scope.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HEAD='R11 Female head study';JAW='R11 jaw open';MARKER='R11_complete_oral_asset_fit'
SEAL_CONTROL='R11 relaxed lip seal';SEAL_GROUP='R11 relaxed lip seal surface'
WARM='R11 warm closed smile'
HELPER_PINS={
    'adult_jaw_open_r11.py':'6bc0abd3d836a8016a42ae327b6b857eb6c7967792eb75e0672362d7defddcf2',
    'adult_jaw_open_r11_v2.py':'120a1f56098ed10b29cbe4645acf44706bb6c4bbe0de4777566f49232a4d30e3',
    'relaxed_lip_seal_nodes_r11_v2.py':'80f7c6c20df7685dd3a48c0f8b1b4e20cad7a090ab3e701ed914ccab18a4aa49',
    'adult_oral_contact_r11.py':'84eeb7305d28239a73944a11c432b99b7babc5348b692c5c743650af6c968065',
    'hand_wrist_surface_contact_r11.py':'c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e',
    'face_fibre_binding_r11_v2.py':'47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49',
    'upper_lash_corrective_validation_r11.py':'36a8713e93a962504b6f1546fb466afd70e3a7af394c449b40176b32999e3c0a',
}


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a
def transform(v,m):
    m=np.asarray(m,float);return np.asarray(v)@m[:3,:3].T+m[:3,3]


def material(name,kind):
    import bpy
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;l=m.node_tree.links
    bs=next(n for n in n if n.type=='BSDF_PRINCIPLED')
    if kind=='teeth':colour=(.67,.62,.52,1.);rough=.26;sss=.045;radius=(1.,.65,.35);scale=.00045
    elif kind=='tongue':colour=(.28,.065,.074,1.);rough=.36;sss=.18;radius=(1.,.35,.20);scale=.0010
    else:colour=(.24,.054,.063,1.);rough=.31;sss=.16;radius=(1.,.4,.25);scale=.0010
    bs.inputs['Base Color'].default_value=colour;bs.inputs['Roughness'].default_value=rough
    bs.inputs['Subsurface Weight'].default_value=sss;bs.inputs['Subsurface Radius'].default_value=radius;bs.inputs['Subsurface Scale'].default_value=scale
    bs.inputs['IOR'].default_value=1.52 if kind=='teeth' else 1.38;bs.inputs['Specular IOR Level'].default_value=.4
    bs.inputs['Coat Weight'].default_value=.06;bs.inputs['Coat Roughness'].default_value=.12
    if kind=='tongue':
        tex=n.new('ShaderNodeTexCoord');noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=1400.;noise.inputs['Detail'].default_value=2.
        l.new(tex.outputs['Object'],noise.inputs['Vector']);bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00006
        l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    m['R11_scope']='Physical oral material candidate; natural enamel / moist mucosa, not appearance acceptance.'
    return m


def create_part(name,row,arrays,donor,collection,parent,head,jaw_report,materials):
    import bpy
    from mathutils import Matrix
    ids=arrays[name+'_source_ids'];fi=arrays[name+'_source_face_ids'];world=arrays[name+'_world'];local=transform(world,np.linalg.inv(np.asarray(parent.matrix_world,float)))
    loops=arrays[name+'_loops'];starts=arrays[name+'_face_starts'];sizes=arrays[name+'_face_sizes'];faces=[loops[s:s+c].tolist() for s,c in zip(starts,sizes)]
    mesh=bpy.data.meshes.new(row['objectName']+' mesh');mesh.from_pydata(local.tolist(),[],faces);mesh.update()
    obj=bpy.data.objects.new(row['objectName'],mesh);collection.objects.link(obj);obj.parent=parent;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    for poly in mesh.polygons:poly.use_smooth=True
    a=mesh.attributes.new('R11 oral donor vertex ID','INT','POINT');a.data.foreach_set('value',ids)
    a=mesh.attributes.new('R11 oral donor face ID','INT','FACE');a.data.foreach_set('value',fi)
    a=mesh.attributes.new('.sculpt_face_set','INT','FACE');a.data.foreach_set('value',arrays[name+'_face_sets'])
    source_loop_ids=np.concatenate([np.arange(donor.data.polygons[int(i)].loop_start,donor.data.polygons[int(i)].loop_start+donor.data.polygons[int(i)].loop_total) for i in fi])
    for original in donor.data.uv_layers:
        uv=mesh.uv_layers.new(name=original.name);raw=array(original.data,'uv',2);uv.data.foreach_set('uv',raw[source_loop_ids].ravel())
    original_crease=donor.data.attributes.get('crease_edge')
    if original_crease is not None:
        values=array(original_crease.data,'value',1).ravel();source_edges={tuple(sorted(e.vertices)):float(values[e.index]) for e in donor.data.edges}
        new=mesh.attributes.new('crease_edge','FLOAT','EDGE');new.data.foreach_set('value',np.asarray([source_edges.get(tuple(sorted((int(ids[e.vertices[0]]),int(ids[e.vertices[1]])))),0.) for e in mesh.edges],np.float32))
    kind='teeth' if name.endswith('teeth') else ('tongue' if name=='tongue' else 'gingiva');mesh.materials.append(materials[kind])
    sub=obj.modifiers.new('R11 oral fine surface','SUBSURF');sub.subdivision_type='CATMULL_CLARK';sub.levels=2;sub.render_levels=2
    if name in ('lower_gingiva','tongue'):
        # The dental margin/tongue tip follows the mandible fully. Posterior
        # mucosa blends toward its pharyngeal attachment instead of being rigid.
        t=np.clip((world[:,1]+.045)/.015,0,1);t=t*t*(3-2*t);weights=(1.-.7*t).astype(np.float32)
        attribute=mesh.attributes.new('R11 mandibular rotation weight','FLOAT','POINT');attribute.data.foreach_set('value',weights)
        tree=api('adult_jaw_open_r11')['make_nodes'](head,jaw_report['estimatedPivotLocal'],jaw_report['axisLocal'],jaw_report['translationLocal'])
        mod=obj.modifiers.new('R11 oral mandibular arc','NODES');mod.node_group=tree
        obj['R11_posterior_jaw_weight_minimum']=float(weights.min())
    obj['R11_oral_donor_part']=name;obj['R11_source_face_sets']=json.dumps(row['sourceFaceSets']);obj['R11_editable_derived_mesh']=True
    return obj


def actual_clearance(head,objects,strength):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    head.data.shape_keys.key_blocks[JAW].value=strength;bpy.context.view_layer.update();qa=api('face_contact_audit_r11');h=qa['_evaluated_mesh'](head)
    bvh=BVHTree.FromPolygons([Vector(p) for p in h['world']],h['faces'],all_triangles=False);rows={}
    for name,obj in objects.items():
        data=qa['_evaluated_mesh'](obj);world=data['world'];sample_ids=np.arange(len(world)) if strength==0 else np.unique(np.linspace(0,len(world)-1,min(3000,len(world))).astype(np.int32))
        signed=[]
        for i in sample_ids:
            co=Vector(world[i]);hit,normal,index,distance=bvh.find_nearest(co)
            if hit is None:raise RuntimeError('Actual oral/head nearest query failed')
            signed.append(float((co-hit).dot(normal.normalized())))
        s=np.asarray(signed);rows[name]={'evaluatedVertices':len(world),'sampledVertices':len(sample_ids),
            'signedHeadDistanceMetersP0P01P50P99P100':np.percentile(s,[0,1,50,99,100]).tolist(),
            'pointsInsideHeadBeyondHalfMillimeter':int((s<-.0005).sum()),'worldBounds':[world.min(0).tolist(),world.max(0).tolist()]}
    return {'jawStrength':strength,'objects':rows,'scope':'Actual evaluated Subsurf2 vertices versus actual current L3 head BVH. Neutral uses all oral vertices; open states use at most3000 per part. Not full edge-face collision or visual acceptance.'}


def attachment_spaces(head,jaw_report):
    """Resolve the live parent relation after a possible head-bind migration.

    A jaw installed before the head binder retains a historical cranialParent
    string in its scene report. The binder migrates the actual jaw carrier to
    the cranial Empty; the historical string is therefore never authoritative.
    """
    import bpy
    jaw=bpy.data.objects.get(jaw_report['attachmentCarrier'])
    if jaw is None or jaw.type!='EMPTY' or jaw.get('R11_mandibular_attachment_source')!=HEAD or jaw.get('R11_jaw_control_key')!=JAW:
        raise RuntimeError('Actual native mandibular carrier identity differs')
    if jaw.parent_type!='OBJECT' or jaw.parent is None:
        raise RuntimeError('Native mandibular carrier requires its actual cranial object parent')
    cranial=jaw.parent
    if cranial==head:
        if head.parent_type!='BONE' or head.parent_bone!='head' or head.parent is None or head.parent.type!='ARMATURE' or any(m.type=='ARMATURE' for m in head.modifiers):
            raise RuntimeError('Skinned head cannot be the oral cranial attachment space')
    else:
        tagged=[o for o in bpy.data.objects if o.get('R11_original_head_space_source')==HEAD]
        if len(tagged)!=1 or tagged[0]!=cranial or cranial.type!='EMPTY' or cranial.parent_type!='BONE' or cranial.parent_bone!='head' or cranial.parent is None or cranial.parent.type!='ARMATURE' or cranial.parent!=head.parent:
            raise RuntimeError('Actual mandibular parent is not the unique original head-bone space')
    return jaw,cranial,{'recordedCranialParent':jaw_report['cranialParent'],'actualCranialParent':cranial.name,
        'actualMandibularParent':jaw.parent.name,'resolvedFromActualParent':True,
        'historicalCranialParentWasStale':jaw_report['cranialParent']!=cranial.name,
        'scope':'Resolve current parent graph; retain the original jaw scene report as historical evidence.'}


def rest_attachment_check(head,jaw,cranial,jaw_report):
    # Jaw GN uses original head-local coordinates. Both the cranial meshes and
    # rigid mandibular carrier must still use that exact coordinate frame.
    matrix_error=float(np.max(np.abs(np.asarray(head.matrix_world,float)-np.asarray(cranial.matrix_world,float))))
    if matrix_error>2e-6:raise RuntimeError('Neutral head and actual cranial coordinate spaces differ')
    expected=np.eye(4);expected[:3,3]=jaw_report['estimatedPivotLocal']
    basis_error=float(np.max(np.abs(np.asarray(jaw.matrix_basis,float)-expected)))
    inverse_error=float(np.max(np.abs(np.asarray(jaw.matrix_parent_inverse,float)-np.eye(4))))
    if basis_error>2e-6 or inverse_error>2e-6:raise RuntimeError('Neutral jaw carrier no longer matches its recorded native pivot')
    return {'neutralHeadCranialMatrixMaximumError':matrix_error,
        'neutralMandibularBasisMaximumError':basis_error,'mandibularParentInverseMaximumError':inverse_error}


def current_seal_contract(head,jaw_report):
    """Read the actual connected layer and its nonzero evaluated-index domain."""
    found=[m for m in head.modifiers if m.type=='NODES' and m.node_group and
           (m.node_group.name==SEAL_GROUP or m.node_group.name.startswith(SEAL_GROUP+'.'))]
    if not found:
        if SEAL_CONTROL in head.data.shape_keys.key_blocks:raise RuntimeError('Seal Key has no actual surface layer')
        return None,None
    if len(found)!=1 or found[0].node_group.name!=SEAL_GROUP or not found[0].show_viewport or not found[0].show_render:
        raise RuntimeError('One exact active seal layer required')
    connection=jaw_report.get('sealConnection');group=found[0].node_group
    if not connection or not connection.get('connectionInstalledBeforeActualChecks') or connection.get('postSaveReconnectAllowed') is not False:
        raise RuntimeError('Install seal-aware jaw transaction before testing dental placement')
    if connection['helperSha256']!=HELPER_PINS['relaxed_lip_seal_nodes_r11_v2.py'] or group.get('R11_jaw_reconnect_required',True):
        raise RuntimeError('Seal connection is pending or uses an unreviewed helper')
    if connection['group']!=group.name or connection['zeroSealAtJaw']!=float(group['R11_jaw_zero_seal_at']):
        raise RuntimeError('Saved and actual seal connection differ')
    node=group.nodes.get(connection['node'])
    if node is None or node.bl_idname!='ShaderNodeValue':raise RuntimeError('Recorded seal scalar node disappeared')
    path=node.outputs[0].path_from_id('default_value')
    curves=[f for f in group.animation_data.drivers if f.data_path==path] if group.animation_data else []
    if len(curves)!=1 or curves[0].mute or curves[0].array_index!=0:raise RuntimeError('Unexpected current seal driver')
    driver=curves[0].driver
    if driver.type!='SCRIPTED' or driver.expression!=connection['expression'] or len(driver.variables)!=2:
        raise RuntimeError('Current seal driver changed after jaw acceptance')
    if sorted(v.name for v in driver.variables)!=['c','j']:raise RuntimeError('Current seal variables changed')
    for v in driver.variables:
        if v.type!='SINGLE_PROP' or len(v.targets)!=1:raise RuntimeError('Unexpected seal driver target type')
        t=v.targets[0];key=SEAL_CONTROL if v.name=='c' else JAW
        if t.id_type!='KEY' or t.id!=head.data.shape_keys or t.data_path!=head.data.shape_keys.key_blocks[key].path_from_id('value'):
            raise RuntimeError('Current seal driver no longer reads the recorded Key')
    jaw_index=next(i for i,m in enumerate(head.modifiers) if m.name=='R11 anatomical jaw rotation')
    if list(head.modifiers).index(found[0])>=jaw_index:raise RuntimeError('Seal must precede actual mandibular rotation')
    attribute='R11_relaxed_lip_seal_local_offset'
    caches=[]
    for n in group.nodes:
        if n.bl_idname=='GeometryNodeObjectInfo':
            obj=n.inputs['Object'].default_value
            if obj is not None and obj.type=='MESH' and attribute in obj.data.attributes:caches.append(obj)
    if len(caches)!=1:raise RuntimeError('Expected one actual seal index-lookup mesh')
    cache=caches[0];active=np.zeros(len(cache.data.vertices),bool);hashes={}
    for name in (attribute,attribute+'_upper'):
        a=cache.data.attributes.get(name)
        if a is None or a.domain!='POINT' or a.data_type!='FLOAT_VECTOR':raise RuntimeError('Seal offset field is missing')
        values=array(a.data,'vector')
        if not np.isfinite(values).all():raise RuntimeError('Nonfinite seal offset field')
        active|=np.any(values!=0.,axis=1);hashes[name]=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    if not active.any():raise RuntimeError('Seal contains no actual local field')
    return {**connection,'lookupObject':cache.name,'lookupPointCount':len(active),'actualOffsetHashes':hashes,
            'nonzeroActualVertexCount':int(active.sum()),'actualKeyValue':float(head.data.shape_keys.key_blocks[SEAL_CONTROL].value)},np.flatnonzero(active)


def inspection_states(connection):
    values=[0.,.25,.5,1.]
    if connection:
        t=connection['zeroSealAtJaw']
        values.extend([t*q for q in (.25,.5,.75)]+[t-.001,t,t+.001])
    states=[(float(j),0.) for j in sorted(set(values))]
    early=connection['zeroSealAtJaw'] if connection else .12
    states.extend((j,.5) for j in (0.,early*.5,early))
    states.extend((j,1.) for j in (0.,early*.25,early*.5,early*.75,early,.5,1.))
    return states


def prepare(source_relative,output_relative,plan_relative):
    import bpy
    source=(P/source_relative).resolve();output=(P/output_relative).resolve();rp=output.with_suffix('.oral-fit.json');planpath=(P/plan_relative).resolve()
    for path in (source,output,rp,planpath):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent source required')
    if output.exists() or rp.exists() or MARKER in bpy.context.scene:raise RuntimeError('Retain prior oral candidates')
    for name,expected in HELPER_PINS.items():
        if sha(Path(__file__).with_name(name))!=expected:raise RuntimeError('Review and freeze changed dental dependency '+name)
    plan=json.loads(planpath.read_text(encoding='utf8'));ap=P/plan['arrays']
    if sha(ap)!=plan['arraysSha256']:raise RuntimeError('Oral fit plan arrays changed')
    with np.load(ap) as n:arrays={k:n[k].copy() for k in n.files}
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if JAW not in keys or 'R11_native_mandibular_motion' not in bpy.context.scene:raise RuntimeError('Install the actual native adult jaw before dental fitting')
    if WARM not in keys:raise RuntimeError('Current warm-smile control required for dental combination checks')
    if any(abs(k.value)>1e-7 for k in keys if k.name.startswith(('R11 blink','R11 natural soft smile','R11 warm closed smile',JAW))):raise RuntimeError('Neutral face required for dental fitting')
    jaw_report=json.loads(bpy.context.scene['R11_native_mandibular_motion']);jaw_parent,cranial,attachment_resolution=attachment_spaces(head,jaw_report)
    seal,required_lip_vertices=current_seal_contract(head,jaw_report)
    if seal and jaw_report.get('helperSha256')!=HELPER_PINS['adult_jaw_open_r11_v2.py']:
        raise RuntimeError('The current seal-aware jaw report must match the frozen actual successor')
    donor_path=P/plan['donor']
    if sha(donor_path)!=plan['donorSha256']:raise RuntimeError('Exact installed CC0 donor required')
    original_head=api('adult_head_bind_observe_r11')['data_contract'](head);uv=api('production_uv_r08');pose=uv['_pose_contract']();before_file=sha(source)
    pools=('objects','collections','meshes','materials','node_groups','images','shape_keys');old={name:{v.as_pointer() for v in getattr(bpy.data,name)} for name in pools}
    saved=False;report={'source':source_relative,'sourceSha256':before_file,'plan':plan_relative,
                       'planSha256':sha(planpath),'helperSha256':sha(__file__),'helperPins':HELPER_PINS,'savedBlend':False}
    key_values={k.name:float(k.value) for k in keys};active_key=head.active_shape_key_index
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            attachment_resolution.update(rest_attachment_check(head,jaw_parent,cranial,jaw_report))
            mix,_=uv['current_mix'](head);current=transform(mix,head.matrix_world);skin=api('natural_skin_adult_r11');topology,land,_=skin['_donor']()
            ids=np.flatnonzero(skin['_vertices_of_groups'](topology,{7,8}));error=float(np.linalg.norm(current[ids]-arrays['source_observed_cage_world'][ids],axis=1).max())
            if error>2e-6:raise RuntimeError('Actual neutral oral cage changed; refit before appending dental data: '+str(error))
            contact=api('adult_oral_contact_r11');jaw_api=api('adult_jaw_open_r11_v2')
            domain=contact['build_domain'](head,required_head_vertices=required_lip_vertices)
            report.update(actualNeutralOralCageErrorMeters=error,currentLipDomain=domain,actualSealContract=seal)
            with bpy.data.libraries.load(str(donor_path),link=False) as (src,dst):dst.objects=['Jaw - Realistic']
            donor=dst.objects[0]
            if donor is None or donor.type!='MESH' or not np.array_equal(array(donor.data.vertices),arrays['donor_reference_local']):raise RuntimeError('Actual donor differs from observed original mesh')
            backup=bpy.data.collections.new('R11 original oral donor retained');bpy.context.scene.collection.children.link(backup);backup.objects.link(donor)
            donor.name='R11 Original CC0 Jaw donor (hidden)';donor.hide_render=True;donor.hide_viewport=True;donor.hide_set(True);donor.data.use_fake_user=True
            donor['R11_source_path']=plan['donor'];donor['R11_source_sha256']=plan['donorSha256'];donor['R11_license']='CC0'
            col=bpy.data.collections.new('R11 Adult complete oral anatomy');bpy.context.scene.collection.children.link(col)
            mats={k:material('R11 natural '+k+' material',k) for k in ('teeth','gingiva','tongue')}
            objects={}
            for name,row in plan['parts'].items():
                if row['objectName'] in bpy.data.objects:raise RuntimeError('Retain existing derived oral object '+row['objectName'])
                parent=jaw_parent if name=='lower_teeth' else cranial
                objects[name]=create_part(name,row,arrays,donor,col,parent,head,jaw_report,mats)
            checks=[];continuous_failures=[];states=inspection_states(seal);report['actualClearanceChecks']=checks
            for strength,warm in states:
                keys[JAW].value=strength;keys[WARM].value=warm;head.update_tag();bpy.context.view_layer.update()
                report['evaluatingState']={'requestedJaw':strength,'actualJaw':float(keys[JAW].value),
                                           'requestedWarmSmile':warm,'actualWarmSmile':float(keys[WARM].value)}
                row=actual_clearance(head,objects,strength);row['warmSmile']=warm
                row['actualSealStrength']=jaw_api['seal_strength_check'](head,seal,strength)
                row['currentSurfaceContact']=contact['observe'](head,objects,domain)
                checks.append(row)
                if not row['currentSurfaceContact']['continuousLipContactPassed']:
                    continuous_failures.append({'jaw':strength,'warmSmile':warm,'failures':row['currentSurfaceContact']['criticalFailures']})
                    break
            keys[JAW].value=0.;keys[WARM].value=0.;bpy.context.view_layer.update()
            neutral=checks[0]['objects'];critical={name:r for name,r in neutral.items() if r['signedHeadDistanceMetersP0P01P50P99P100'][0]<(-.0003 if name.endswith('teeth') else -.0005)}
            report={'source':source_relative,'sourceSha256':before_file,'plan':plan_relative,'planSha256':sha(planpath),'planArraysSha256':plan['arraysSha256'],
                'donor':plan['donor'],'donorSha256':plan['donorSha256'],'license':'CC0','originalDonorObject':donor.name,
                'parts':plan['parts'],'actualNeutralOralCageErrorMeters':error,'actualClearanceChecks':checks,
                'cageGuardMeaning':'Original oral control coordinates only; never evidence of current evaluated inner-lip or dental compatibility.',
                'candidatePlacementRetainedFromPriorPlan':True,'currentLipDomain':domain,'actualSealContract':seal,
                'requestedJawWarmStates':[{'jaw':j,'warmSmile':w} for j,w in states],
                'continuousDentalLipFailures':continuous_failures,'completedStateCount':len(checks),
                'currentContinuousDentalLipPassed':not continuous_failures and len(checks)==len(states),
                'helperPins':HELPER_PINS,
                'attachmentSpaceResolution':attachment_resolution,
                'newObjects':[o.name for o in objects.values()],'originalHeadBagRetained':True,'oldOralObjectsRetainedHidden':True,
                'subdivisionLevels':2,'uvPolicy':'Exact original donor corner UVs transferred per retained source face when UV layers exist.',
                'jawBehavior':'Lower teeth on mandibular carrier; lower gingiva and tongue use same native rotation with posterior mucosal weighting. Upper teeth/gingiva follow cranial carrier.',
                'helperSha256':sha(__file__),'appearanceAccepted':False,'fullOralCollisionPassed':False,
                'criticalNeutralClearanceFailures':critical,
                'limits':['Requires genuine neutral/open/smile mouth images before adoption.',
                          'Continuous tests cover the fixed anterior lip domain against both actual tooth meshes; permitted tooth-root/gingiva embedding is not tested as an illegal pair.',
                          'The old whole-head signed vertex diagnostics remain limited measurements, not global oral collision certification.',
                          'Tongue, deep mucosa, tooth-to-tooth occlusion, lip-to-lip self-contact during jaw motion, speech phonemes and continuous time between sampled states require separate checks.',
                          'Do not automatically apply the old upper-oral-clearance key after this seal; that would invalidate these actual lip contacts.']}
            if critical or continuous_failures:
                report['savedBlend']=False
                with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
                raise RuntimeError('Actual current dental/lip geometry requires refitting; diagnostic report saved')
            report['fibreBinding']=api('face_fibre_binding_r11_v2')['validate'](neutral=False)
            head.active_shape_key_index=active_key
        if api('adult_head_bind_observe_r11')['data_contract'](head)!=original_head or uv['_pose_contract']()!=pose or sha(source)!=before_file:raise RuntimeError('Dental fitting changed original head, pose or source')
        bpy.context.scene[MARKER]=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Oral candidate save failed')
        report.update(output=output_relative,outputSha256=sha(output),savedBlend=True)
        with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        saved=True;print('R11_ORAL_FIT_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256']}),flush=True)
    except Exception as exc:
        report.update(error=str(exc),savedBlend=output.exists(),appearanceAccepted=False)
        if not rp.exists():
            with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        raise
    finally:
        for name,value in key_values.items():keys[name].value=value
        head.active_shape_key_index=active_key
        if not saved:
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            fresh=[v for name in pools for v in getattr(bpy.data,name) if v.as_pointer() not in old[name]]
            if fresh:bpy.data.batch_remove(ids=fresh)
        bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(a.source,a.output,a.plan)
