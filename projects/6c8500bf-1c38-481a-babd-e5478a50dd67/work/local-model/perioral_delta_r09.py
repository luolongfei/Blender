"""One bounded topology-constrained repair of existing mouth deformation fields.

plan() is offline NumPy analysis of actual exported R09 arrays. It exactly maps
the retained prejoin lip loops through the documented R05 affine proportion
change and validates their edges in the current joined head. No stale indices.

run(source_relative=...) changes only four existing expression key coordinate
arrays in memory, keeping the Key ID, actions/drivers, Basis, all static keys,
lip targets, oral cavity, eye/nose protection, UV/materials and weights.
apply_and_save() saves only a new candidate. restore() uses embedded originals.
The numerical field criterion does not establish visible expression quality.
"""
import base64
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

BASE=project_path()/'work/reconstruction-r02'
OBJECT='R01 Head'
KEYS=('R02_mouth_wide','R02_pucker','R02_jaw_open','R02_smile')
CAPS={'R02_mouth_wide':.0015,'R02_pucker':.0015,'R02_jaw_open':.0012,'R02_smile':.0010}
MARKER='R09_perioral_delta_flow';BACKUP='R09_perioral_delta_backup'
ITERATIONS=24;LAMBDA=6.


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def _hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)


def _coords(collection):
    a=np.empty((len(collection),3),np.float32);collection.foreach_get('co',a.ravel());return a


def _path(relative,suffix):
    path=(project_path()/relative).resolve();path.relative_to(BASE.resolve())
    if path.suffix!=suffix:raise ValueError('Unexpected work artifact type')
    return path


def _mean(values,edges,weights,degree):
    result=np.zeros_like(values,dtype=float)
    np.add.at(result,edges[:,0],values[edges[:,1]]*weights[:,None])
    np.add.at(result,edges[:,1],values[edges[:,0]]*weights[:,None])
    return result/np.maximum(degree[:,None],1e-20)


def _metric(delta,edges,weights,degree,region,coordinates):
    lap=delta-_mean(delta,edges,weights,degree)
    selected=region[edges[:,0]]&region[edges[:,1]]
    lengths=np.linalg.norm(coordinates[edges[:,0]]-coordinates[edges[:,1]],axis=1)
    gradient=np.linalg.norm(delta[edges[:,0]]-delta[edges[:,1]],axis=1)/np.maximum(lengths,1e-8)
    magnitude=np.linalg.norm(lap[region],axis=1)
    return {'vertices':int(region.sum()),'edges':int(selected.sum()),
            'laplacianRmsMeters':float(np.sqrt(np.mean(magnitude*magnitude))),
            'laplacianP95Meters':float(np.quantile(magnitude,.95)),
            'edgeGradientP95':float(np.quantile(gradient[selected],.95)),
            'maximumEdgeGradient':float(gradient[selected].max())}


def _pin_map(observed,topology,landmarks,reference,arrays):
    original=np.asarray(topology['prejoinBasis'],float);matrix=np.asarray(topology['prejoinMatrixWorld'],float)
    original=original@matrix[:3,:3].T+matrix[:3,3]
    center=np.array([0.,.003,1.423])
    influence=_smooth((original[:,2]-1.245)/.065)*_smooth((.14-np.abs(original[:,0]))/.04)
    transformed=original+(original-center)*.10*influence[:,None]
    transformed[:,2]-=.015*influence
    reference_co=np.asarray(reference['keys']['Basis']['co'],float);reference_ids=np.asarray(reference['ids'],np.int32)
    loop_names=('contact_rim_cycle','outer_vermilion_cycle','posterior_oral_entrance_cycle')
    needed=np.asarray(sorted({index for name in loop_names for index in landmarks[name]}),np.int32)
    distances=np.linalg.norm(transformed[needed,None]-reference_co[None],axis=2)
    nearest=np.argmin(distances,axis=1);errors=distances[np.arange(len(needed)),nearest]
    if errors.max()>2e-6 or len(np.unique(nearest))!=len(needed):
        raise RuntimeError('Exact retained prejoin -> current head lip mapping not established')
    lookup={int(old):int(reference_ids[new]) for old,new in zip(needed,nearest)}
    edges={tuple(sorted(pair)) for pair in arrays['head_edges'].tolist()}
    loops={name:[lookup[i] for i in landmarks[name]] for name in loop_names}
    for name,loop in loops.items():
        if any(tuple(sorted((a,b))) not in edges for a,b in zip(loop,loop[1:]+loop[:1])):
            raise RuntimeError('Mapped lip loop does not follow actual current edges: '+name)
    return loops,{name:lookup[i] for name,i in landmarks['landmarks'].items()},float(errors.max())


def plan():
    observation=BASE/'surface-observation-r09.json';payload=observation.with_suffix('.npz')
    topology_path=BASE/'oral-topology-observation-r09.json';reference_path=BASE/'face-contour-input-r06.json';landmark_path=BASE/'lip-landmarks.json'
    observed=json.loads(observation.read_text(encoding='utf-8'));topology=json.loads(topology_path.read_text(encoding='utf-8'))
    reference=json.loads(reference_path.read_text(encoding='utf-8'));landmarks=json.loads(landmark_path.read_text(encoding='utf-8'))
    if observed['sourceSha256']!=topology['sourceSha256']:raise RuntimeError('Numeric and topology observations use different sources')
    with np.load(payload,allow_pickle=False) as file:arrays={name:file[name].copy() for name in file.files if name.startswith('key_') or name=='head_edges'}
    rows=observed['headKeys'];keys={row['name']:arrays['key_'+str(row['index'])] for row in rows};basis=keys['Basis']
    if any(row['relative']!='Basis' for row in rows):raise RuntimeError('This bounded repair expects the measured all-Basis key graph')
    if np.max(np.abs(np.asarray(observed['headMatrix'])-np.eye(4)))>1e-6:raise RuntimeError('Measured head coordinates must use the verified unit world transform')
    neutral=basis.astype(float).copy()
    for row in rows[1:]:neutral+=(keys[row['name']].astype(float)-basis)*row['value']
    loops,lm,map_error=_pin_map(observed,topology,landmarks,reference,arrays)
    count=len(neutral);edges=arrays['head_edges'];skin=np.zeros(count,bool);oral=np.zeros(count,bool)
    for face,material_index in zip(topology['currentHeadPolygons'],topology['currentHeadMaterialIndices']):
        material=topology['currentMaterialNames'][material_index]
        if material=='R04 physical skin':skin[face]=True
        if material=='R02 oral mucosa':oral[face]=True
    lip_ids=np.asarray(sorted({v for loop in loops.values() for v in loop}),np.int32)
    mouth_center=(neutral[lm['upper_midline']]+neutral[lm['lower_midline']])*.5
    x,y,z=neutral.T
    # Smooth outer window, distinct from the old super-Gaussian rectangular
    # expression support. Only exterior physical skin can be edited.
    ellipse=(x/.070)**2+((z-mouth_center[2])/.045)**2
    weight=_smooth(1-ellipse)*_smooth((-y-.035)/.030)*skin
    pinned=oral.copy();pinned[lip_ids]=True
    nose=(np.abs(x)<.027)&(z>=mouth_center[2]+.021)&(y<-.030)
    ocular=np.zeros(count,bool)
    for row in rows:
        if any(token in row['name'].lower() for token in ('blink','squint','gaze','globe_contact','aperture')):
            ocular|=np.linalg.norm(keys[row['name']].astype(float)-basis,axis=1)>1e-7
    pinned|=nose|ocular;weight[pinned]=0.
    # A two-edge transition away from the preserved lip/cavity targets avoids
    # an abrupt join between the fixed detailed lip surface and repaired skin.
    distance=np.full(count,99,np.int32);distance[lip_ids]=0;distance[oral]=0
    for step in (1,2):
        a=edges[:,0];b=edges[:,1];candidate=np.unique(np.concatenate((b[distance[a]==step-1],a[distance[b]==step-1])))
        distance[candidate[distance[candidate]>step]]=step
    weight[distance==1]*=.35;weight[distance==2]*=.75
    editable=(weight>1e-5)&(~pinned);weight[~editable]=0.
    if not 100<int(editable.sum())<1800:raise RuntimeError('Unexpected perioral edit scope')
    lengths=np.linalg.norm(neutral[edges[:,0]]-neutral[edges[:,1]],axis=1)
    # Bound tiny-edge influence around the dense vermilion; those details are
    # pinned rather than allowed to dominate an unconstrained whole-face solve.
    edge_weights=1/np.maximum(lengths,.0015)
    degree=np.zeros(count);np.add.at(degree,edges[:,0],edge_weights);np.add.at(degree,edges[:,1],edge_weights)
    targets={};metrics={}
    for name in KEYS:
        original=keys[name].astype(float)-basis;current=original.copy()
        for _ in range(ITERATIONS):
            mean=_mean(current,edges,edge_weights,degree)
            proposal=(original+LAMBDA*mean)/(1+LAMBDA)
            current=original+(proposal-original)*weight[:,None]
        change=current-original;amount=np.linalg.norm(change,axis=1);u=amount/CAPS[name]
        scale=np.ones_like(u);nonzero=u>1e-12;scale[nonzero]=np.tanh(u[nonzero])/u[nonzero]
        corrected=original+change*scale[:,None]
        target=keys[name].copy();target[editable]=(basis[editable]+corrected[editable]).astype(np.float32)
        if not np.array_equal(target[pinned],keys[name][pinned]):raise RuntimeError('Protected expression target changed')
        targets[name]=target
        after=target.astype(float)-basis;actual=target.astype(float)-keys[name]
        before_metric=_metric(original,edges,edge_weights,degree,editable,neutral)
        after_metric=_metric(after,edges,edge_weights,degree,editable,neutral)
        if after_metric['laplacianRmsMeters']>before_metric['laplacianRmsMeters']*1.01:
            raise RuntimeError('Perioral field roughness increased: '+name)
        metrics[name]={'before':before_metric,'after':after_metric,
            'maximumKeyCoordinateCorrectionMeters':float(np.linalg.norm(actual,axis=1).max()),
            'fullStrengthCapMeters':CAPS[name],'changedVertices':int((np.linalg.norm(actual,axis=1)>1e-8).sum()),
            'lipTargetsBitExact':bool(np.array_equal(target[lip_ids],keys[name][lip_ids])),
            'oralCavityBitExact':bool(np.array_equal(target[oral],keys[name][oral])),
            'noseAndOcularBitExact':bool(np.array_equal(target[nose|ocular],keys[name][nose|ocular])),
            'outOfRegionBitExact':bool(np.array_equal(target[~editable],keys[name][~editable]))}
    contributions={}
    for row in rows[1:]:
        if row['value'] or row['name'] in KEYS:
            factor=float(row['value']) if row['name'] not in KEYS else 1.
            contributions[row['name']]={'factor':factor,'kind':'expression at full strength' if row['name'] in KEYS else 'active static/design contribution',
                **_metric((keys[row['name']].astype(float)-basis)*factor,edges,edge_weights,degree,editable,neutral)}
    review_combinations={
        'ee63':{'R02_jaw_open':.2,'R02_smile':.45,'R02_mouth_wide':.75},
        'ou98':{'R02_jaw_open':.2,'R02_smile':.45,'R02_pucker':.85},
        'blink-smile':{'R02_smile':1.},
        'neutral-smile045':{'R02_smile':.45}}
    combined={}
    for label,values in review_combinations.items():
        delta=sum((targets[name].astype(float)-keys[name])*value for name,value in values.items())
        combined[label]={'values':values,'maximumCoordinateCorrectionMeters':float(np.linalg.norm(delta,axis=1).max()),
                         'lipTargetsBitExact':bool(np.count_nonzero(delta[lip_ids])==0)}
    report={'observationSource':observed['source'],'observationSourceSha256':observed['sourceSha256'],
        'inputFiles':{path.name:_sha(path) for path in (observation,payload,topology_path,reference_path,landmark_path)},
        'edgeArraySha256':_hash(edges),
        'polygonAndMaterialSha256':hashlib.sha256(json.dumps([topology['currentHeadPolygons'],
            topology['currentHeadMaterialIndices'],topology['currentMaterialNames']],separators=(',',':')).encode()).hexdigest(),
        'headVertices':count,'targetKeys':list(KEYS),'iterations':ITERATIONS,'lambda':LAMBDA,
        'editableVertices':int(editable.sum()),'protectedLipVertices':len(lip_ids),'protectedOralVertices':int(oral.sum()),
        'mouthCenter':mouth_center.tolist(),'mappedLandmarks':lm,'mappedLoops':loops,
        'maximumPrejoinMapErrorMeters':map_error,'loopEdgesValidated':True,'keyMetrics':metrics,
        'reviewCombinationCorrections':combined,
        'staticAndExpressionFieldContributions':contributions,
        'method':'Anchored topology diffusion of existing expression DELTAS only, with exact lip/oral/nose/eye targets and smooth outer/inner influence windows.',
        'staticGeometryKeysChanged':False,'basisChanged':False,'expressionValuesOrActionCurvesChanged':False,
        'limits':['Existing static cheek/chin geometry contributes to the silhouette and is not altered in this pass.',
                  'The old fourth/sixth-power mouth support has steep peripheral gradients. This pass redistributes the surrounding skin response, preserving exact lip targets and expression strength there.',
                  'Laplacian/edge metrics quantify field continuity, not appearance or anatomical correctness.',
                  'Actual same-view EE63, OU98, blink-smile and neutral renders are required before adoption.']}
    return report,targets,keys,editable,pinned


def _contract(head,uv):
    # Keep the original Key datablock/driver identities. Geometry elsewhere,
    # scalar key settings, UV, materials, weights and rig remain separately held.
    state=uv['_original_contract'](head,tuple(layer.name for layer in head.data.uv_layers))
    state.pop('shapeKeysSha256');state.pop('sha256')
    # Actual head-only probe isolated a 6.81 micrometre Y dimension change
    # from the active smile delta. This is an evaluated bounding-box result,
    # not a transform input. Keep location/rotation/scale and all explicit
    # world/basis/parent matrices; report the derived dimensions separately.
    state['objectSettings'].pop('dimensions',None)
    key_settings={k.name:{'settings':uv['_scalar_rna'](k),'relative':k.relative_key.name} for k in head.data.shape_keys.key_blocks}
    return {'mesh':state,'keySettings':key_settings,'keyBlockSettings':uv['_scalar_rna'](head.data.shape_keys),
            'keyBlockCustom':uv['_custom'](head.data.shape_keys),'animation':uv['_animation'](head.data.shape_keys),
            'keyIdPointer':head.data.shape_keys.as_pointer()}


def _contract_diff(before,after,path='',limit=60):
    """Bounded exact field diagnostics; no relaxed comparison/ignored fields."""
    rows=[]
    def visit(a,b,p):
        if len(rows)>=limit:return
        if type(a)!=type(b):rows.append({'path':p,'before':a,'after':b});return
        if isinstance(a,dict):
            for name in sorted(set(a)|set(b)):
                q=p+'.'+str(name) if p else str(name)
                if name not in a or name not in b:
                    rows.append({'path':q,'before':a.get(name,'<missing>'),'after':b.get(name,'<missing>')})
                else:visit(a[name],b[name],q)
                if len(rows)>=limit:break
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):rows.append({'path':p+'.length','before':len(a),'after':len(b)})
            for i,(x,y) in enumerate(zip(a,b)):
                visit(x,y,p+'['+str(i)+']')
                if len(rows)>=limit:break
        elif a!=b:rows.append({'path':p,'before':a,'after':b})
    visit(before,after,path)
    return rows


def diagnose_head_contract(source_relative,report_relative='work/reconstruction-r02/perioral-contract-diagnostic-r09.json'):
    """Only exercise four head arrays, then ALWAYS roll back; never save blend.

    Skips all other-object/cache key snapshots. This diagnoses a contract
    failure and is not an alternative validation or production apply path.
    """
    import bpy
    source=_path(source_relative,'.blend');report_path=_path(report_relative,'.json')
    if report_path.exists():raise RuntimeError('Preserve the existing diagnostic; choose a new report path')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open explicit source in background Object Mode')
    head=bpy.data.objects[OBJECT]
    if MARKER in head or BACKUP in head:raise RuntimeError('Do not diagnose over an applied candidate')
    report,targets,observed,_,_=plan();blocks=head.data.shape_keys.key_blocks
    if [key.name for key in blocks]!=list(observed):raise RuntimeError('Observed head key order changed')
    originals={key.name:_coords(key.data) for key in blocks}
    if any(not np.array_equal(originals[name],co) for name,co in observed.items()):
        raise RuntimeError('Current head arrays differ from observed input')
    uv=_api('production_uv_r08');bpy.context.view_layer.update()
    before=_contract(head,uv);source_sha=_sha(source);written=False
    result={'source':source_relative,'sourceSha256':source_sha,'savedBlend':False,
            'scope':'Four temporary expression key coordinate writes, head-only contract, unconditional rollback; no whole-scene/cache snapshot.'}
    try:
        written=True
        for name,target in targets.items():blocks[name].data.foreach_set('co',target.ravel())
        head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
        after=_contract(head,uv);difference=_contract_diff(before,after)
        result.update(contractEqual=before==after,differences=difference,
            currentExpressionValues={name:blocks[name].value for name in KEYS},
            candidateArraysMatch=all(np.array_equal(_coords(blocks[name].data),targets[name]) for name in KEYS))
        print('R09_PERIORAL_HEAD_CONTRACT_DIFF '+json.dumps(difference,ensure_ascii=False),flush=True)
    finally:
        if written:
            for name in KEYS:blocks[name].data.foreach_set('co',originals[name].ravel())
            head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
        restored=_contract(head,uv)
        result['rollbackCoordinatesBitExact']=all(np.array_equal(_coords(blocks[name].data),originals[name]) for name in KEYS)
        result['rollbackContractEqual']=restored==before
        result['rollbackContractDifferences']=_contract_diff(before,restored)
        result['savedSourceSha256Unchanged']=_sha(source)==source_sha
        with report_path.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        if not result['rollbackCoordinatesBitExact'] or not result['rollbackContractEqual']:
            raise RuntimeError('Head-only diagnostic rollback did not match; inspect diagnostic report, do not save')
    return result


def _modifier_custom(modifier,uv):
    # Some Blender modifiers expose bpy_struct.keys() but do not support
    # IDProperties. Only that unsupported-container case is empty; retain
    # actual Geometry Nodes inputs and propagate property/conversion errors.
    try:names=tuple(modifier.keys())
    except TypeError:return {}
    return {name:uv['_plain'](modifier[name]) for name in names}


def _other_scene_contract(bpy,uv,groom):
    """Include the new native hair/cloth carriers without evaluating motion.

    Stream one key at a time so a 301-frame cloth carrier is not duplicated in
    memory. All scene geometry except the explicitly edited head is held; this
    also includes the scalp which the earlier posterior append guard excluded.
    """
    rows={}
    for obj in bpy.context.scene.objects:
        if obj.name==OBJECT:continue
        row={'type':obj.type,'settings':uv['_scalar_rna'](obj),'custom':uv['_custom'](obj),
             'world':uv['_plain'](obj.matrix_world),'parent':uv['_plain'](obj.parent),
             'animation':uv['_animation'](obj),
             'modifiers':[{'settings':uv['_scalar_rna'](m),'custom':_modifier_custom(m,uv),
                           'nodeGroup':uv['_plain'](getattr(m,'node_group',None)),
                           'target':uv['_plain'](getattr(m,'object',None))} for m in obj.modifiers]}
        data=getattr(obj,'data',None)
        if data is not None:
            row.update(dataName=data.name,dataPointer=data.as_pointer(),dataCustom=uv['_custom'](data),
                       dataAnimation=uv['_animation'](data))
            if obj.type in {'MESH','CURVES'}:row['geometrySha256']=groom['_data_hash'](data)
            keys=getattr(data,'shape_keys',None)
            if keys:
                row['keys']={'pointer':keys.as_pointer(),'animation':uv['_animation'](keys),
                    'custom':uv['_custom'](keys),'settings':uv['_scalar_rna'](keys),
                    'blocks':[{'name':key.name,'hash':_hash(_coords(key.data)),
                               'settings':uv['_scalar_rna'](key),'relative':key.relative_key.name}
                              for key in keys.key_blocks]}
        rows[obj.name]=row
    rows['rigPose']=uv['_pose_contract']()
    return rows


def run(source_relative,diagnostics_only=False):
    import bpy
    source=_path(source_relative,'.blend')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the explicit independent current production source in background Object Mode')
    head=bpy.data.objects[OBJECT]
    if MARKER in head or BACKUP in head:raise RuntimeError('Preserve/restore the existing mouth-field candidate')
    if head.library or head.data.library or head.data.users!=1:raise RuntimeError('Local single-user production head required')
    report,targets,observed,editable,pinned=plan();blocks=head.data.shape_keys.key_blocks
    if [k.name for k in blocks]!=list(observed):raise RuntimeError('Observed head key order changed')
    if not head.data.shape_keys.use_relative or any(k.relative_key.name!='Basis' or k.vertex_group for k in blocks):
        raise RuntimeError('Observed unmasked all-Basis relative key graph changed')
    actual_edges=np.empty((len(head.data.edges),2),np.int32);head.data.edges.foreach_get('vertices',actual_edges.ravel())
    actual_topology=[[list(face.vertices) for face in head.data.polygons],
                     [face.material_index for face in head.data.polygons],
                     [slot.material.name if slot.material else None for slot in head.material_slots]]
    actual_topology_hash=hashlib.sha256(json.dumps(actual_topology,separators=(',',':')).encode()).hexdigest()
    if _hash(actual_edges)!=report['edgeArraySha256'] or actual_topology_hash!=report['polygonAndMaterialSha256']:
        raise RuntimeError('Actual topology/material domains differ from the exact observed lip mapping')
    originals={key.name:_coords(key.data) for key in blocks}
    if any(not np.array_equal(originals[name],co) for name,co in observed.items()):raise RuntimeError('Head coordinates differ from the observed R09 input; do not reuse stale mouth masks')
    uv=_api('production_uv_r08');groom=_api('posterior_groom_r09');bpy.context.view_layer.update()
    dimensions_before=list(head.dimensions)
    contract=_contract(head,uv);protected=_other_scene_contract(bpy,uv,groom)
    source_sha=_sha(source);backup=None;committed=False;written=False
    try:
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,
                      writtenInMemory=not diagnostics_only,savedBlend=False,appearanceAccepted=False)
        if diagnostics_only:return report
        stream=io.BytesIO();np.savez_compressed(stream,**{name:originals[name] for name in KEYS})
        backup=bpy.data.texts.new('R09 perioral original key coordinates');backup.use_fake_user=True
        backup.write(json.dumps({'sourceSha256':source_sha,'arrays':base64.b64encode(stream.getvalue()).decode(),
            'originalHashes':{name:_hash(originals[name]) for name in KEYS},
            'candidateHashes':{name:_hash(targets[name]) for name in KEYS}}))
        written=True
        for name,target in targets.items():blocks[name].data.foreach_set('co',target.ravel())
        head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
        dimensions_after=list(head.dimensions)
        report['derivedHeadBoundingDimensions']={'before':dimensions_before,'after':dimensions_after,
            'differenceMeters':[b-a for a,b in zip(dimensions_before,dimensions_after)],
            'excludedFromInvariant':'Only objectSettings.dimensions, the evaluated bounding-box result.',
            'transformCoordinatesKeysUVAndMatricesRemainProtected':True}
        head_after=_contract(head,uv)
        if head_after!=contract:
            difference=_contract_diff(contract,head_after)
            print('R09_PERIORAL_HEAD_CONTRACT_DIFF '+json.dumps(difference,ensure_ascii=False),flush=True)
            raise RuntimeError('Non-coordinate source contract changed: '+json.dumps(difference,ensure_ascii=False))
        after=_other_scene_contract(bpy,uv,groom)
        if after!=protected:raise RuntimeError('Hair, root, rig or other protected data changed')
        for name,old in originals.items():
            current=_coords(blocks[name].data)
            if not np.array_equal(current,targets.get(name,old)):raise RuntimeError('Unexpected key coordinates after assignment: '+name)
        if _sha(source)!=source_sha:raise RuntimeError('Saved source changed externally')
        head[MARKER]=json.dumps({'keys':list(KEYS),'observationSourceSha256':report['observationSourceSha256']})
        head[BACKUP]=backup.name
        report.update(backupText=backup.name,originalDataExceptSelectedKeyCoordinatesUnchanged=True,
                      lipTargetsAndUnselectedKeysBitExact=True,headKeyIdAndDriversPreserved=True,
                      otherSceneGeometryAndNativeCacheKeysUnchanged=True,
                      runtimeConsequence='Refresh face morph snapshot/expressions from this new source if visually adopted; material/UV are unchanged.')
        committed=True;return report
    finally:
        if not committed:
            if written:
                for name in KEYS:blocks[name].data.foreach_set('co',originals[name].ravel())
            for prop in (MARKER,BACKUP):
                if prop in head:del head[prop]
            if backup:bpy.data.texts.remove(backup)
            if written:head.data.update_tag();head.update_tag();bpy.context.view_layer.update()


def apply_and_save(source_relative,output_relative='work/reconstruction-r02/reference-perioral-flow-r09.blend',
                   report_relative='work/reconstruction-r02/perioral-delta-r09.json'):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=_path(report_relative,'.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New candidate and report paths required')
    report=run(source_relative)
    if output.exists():raise RuntimeError('Output appeared during preparation')
    bpy.ops.wm.save_as_mainfile(filepath=str(output));report['savedBlend']=True
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


def restore():
    import bpy
    head=bpy.data.objects[OBJECT]
    if BACKUP not in head:return False
    record=json.loads(bpy.data.texts[head[BACKUP]].as_string())
    with np.load(io.BytesIO(base64.b64decode(record['arrays'])),allow_pickle=False) as file:
        arrays={name:file[name].copy() for name in file.files}
    for name,co in arrays.items():
        if _hash(co)!=record['originalHashes'][name]:raise RuntimeError('Embedded expression rollback changed')
        if _hash(_coords(head.data.shape_keys.key_blocks[name].data))!=record['candidateHashes'][name]:
            raise RuntimeError('Expression was edited after this candidate; preserve those edits before restoring')
    for name,co in arrays.items():head.data.shape_keys.key_blocks[name].data.foreach_set('co',co.ravel())
    for prop in (MARKER,BACKUP):del head[prop]
    head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
    return {'restoredKeys':list(arrays),'savedBlend':False}


if __name__=='__main__':print('Loaded only. plan() is offline; run/apply_and_save require explicit independent Blender invocation.')
