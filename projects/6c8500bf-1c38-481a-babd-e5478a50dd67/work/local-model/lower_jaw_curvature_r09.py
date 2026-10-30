"""One bounded, volume-constrained static lower-face curvature candidate.

plan() uses actual exported topology and the exact adopted perioral arrays.
It smooths 3-D positions inside one lower-face window with fixed lip/cavity
targets, then constrains the signed volume of affected triangles. It does not
scale or delete any previous sculpt and does not use a hand-built ellipsoid.
run(source_relative=...) adds one reversible Basis-relative shape key only.
No Blender execution, rendering or file saving occurs on import.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
BASE=project_path()/'work/reconstruction-r02'
HEAD='R01 Head';KEY='R09_lower_jaw_curvature';MARKER='R09_lower_jaw_curvature_candidate'
ITERATIONS=16;FLOW=.45;CAP=.003


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _smooth(value):
    value=np.clip(value,0,1);return value*value*(3-2*value)


def _path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to(BASE.resolve())
    if p.suffix!=suffix:raise ValueError('Unexpected independent work artifact type')
    return p


def _volume(co,triangles):
    a,b,c=(co[triangles[:,i]] for i in range(3))
    return float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum()/6)


def _normals(co,triangles):
    a,b,c=(co[triangles[:,i]] for i in range(3))
    cross=np.cross(b-a,c-a);acc=np.zeros_like(co);areas=np.zeros(len(co))
    for i in range(3):
        np.add.at(acc,triangles[:,i],cross)
        np.add.at(areas,triangles[:,i],np.linalg.norm(cross,axis=1)/6)
    lengths=np.linalg.norm(acc,axis=1)
    return acc/np.maximum(lengths[:,None],1e-15),areas


def _correct_volume(original,candidate,triangles,direction):
    """One scalar normal correction, bounded Newton solve, fixed patch boundary."""
    wanted=_volume(original,triangles);co=candidate.copy();total=0.
    for _ in range(6):
        error=_volume(co,triangles)-wanted
        if abs(error)<1e-13:break
        step=1e-5
        slope=(_volume(co+direction*step,triangles)-_volume(co-direction*step,triangles))/(2*step)
        if abs(slope)<1e-10:raise RuntimeError('Lower-face volume correction has degenerate support')
        correction=-error/slope
        if abs(total+correction)>.002:raise RuntimeError('Lower-face volume restoration would exceed 2 mm')
        co+=correction*direction;total+=correction
    return co,total


def plan():
    mouth=_api('perioral_delta_r09');mouth_report,mouth_targets,observed,_,protected=mouth['plan']()
    current={name:mouth_targets.get(name,co) for name,co in observed.items()}
    metadata=json.loads((BASE/'surface-observation-r09.json').read_text(encoding='utf-8'))
    topology=json.loads((BASE/'oral-topology-observation-r09.json').read_text(encoding='utf-8'))
    with np.load(BASE/'surface-observation-r09.npz',allow_pickle=False) as file:
        triangles=file['head_triangles'].copy();edges=file['head_edges'].copy()
    basis=current['Basis'];neutral=basis.astype(float).copy()
    for row in metadata['headKeys'][1:]:neutral+=(current[row['name']].astype(float)-basis)*row['value']
    lip=np.asarray(sorted({v for loop in mouth_report['mappedLoops'].values() for v in loop}),np.int32)
    skin=np.zeros(len(neutral),bool);oral=np.zeros(len(neutral),bool)
    for face,index in zip(topology['currentHeadPolygons'],topology['currentHeadMaterialIndices']):
        name=topology['currentMaterialNames'][index]
        if name=='R04 physical skin':skin[face]=True
        if name=='R02 oral mucosa':oral[face]=True
    x,y,z=neutral.T;mouth_z=mouth_report['mouthCenter'][2]
    ellipse=(x/.070)**2+((z-(mouth_z-.019))/.041)**2
    weight=_smooth(1-ellipse)*_smooth((-y-.035)/.035)
    weight*=1-_smooth((z-(mouth_z-.003))/.012)
    weight*=_smooth((z-(mouth_z-.055))/.012)*skin
    pinned=protected|oral; pinned[lip]=True;weight[pinned]=0
    distance=np.full(len(neutral),99,np.int32);distance[lip]=0;distance[oral]=0
    for step in (1,2,3):
        a,b=edges.T;ids=np.unique(np.concatenate((b[distance[a]==step-1],a[distance[b]==step-1])))
        distance[ids[distance[ids]>step]]=step
    # The real triangulation includes a very narrow corner triangle adjoining
    # the outer lip ring. Preserve its complete first ring as a contact guard,
    # rather than accepting an inversion during larger static curvature flow.
    pinned|=distance<=1;weight[pinned]=0
    weight[distance==2]*=.4;weight[distance==3]*=.8
    editable=weight>1e-5;weight[~editable]=0
    if not 80<int(editable.sum())<900:raise RuntimeError('Unexpected local lower-face scope')
    lengths=np.linalg.norm(neutral[edges[:,0]]-neutral[edges[:,1]],axis=1)
    ew=1/np.maximum(lengths,.0015);degree=np.zeros(len(neutral))
    np.add.at(degree,edges[:,0],ew);np.add.at(degree,edges[:,1],ew)
    co=neutral.copy()
    for _ in range(ITERATIONS):
        co+=FLOW*weight[:,None]*(mouth['_mean'](co,edges,ew,degree)-co)
    normals,areas=_normals(neutral,triangles)
    touched=np.any(editable[triangles],axis=1);patch=triangles[touched]
    direction=normals*weight[:,None]
    co,restore_a=_correct_volume(neutral,co,patch,direction)
    maximum=float(np.linalg.norm(co-neutral,axis=1).max());scale=min(1.,CAP/max(maximum,1e-20))
    co=neutral+(co-neutral)*scale
    co,restore_b=_correct_volume(neutral,co,patch,direction)
    delta=np.asarray(co-neutral,np.float32);delta[~editable]=0
    target=np.asarray(basis+delta,np.float32);delta_actual=target.astype(float)-basis
    final=neutral+delta_actual
    if np.linalg.norm(delta_actual,axis=1).max()>CAP+2e-5:raise RuntimeError('Bounded curvature correction exceeds 3 mm')
    if not np.array_equal(target[pinned],basis[pinned]) or not np.array_equal(target[~editable],basis[~editable]):
        raise RuntimeError('Protected lower-face boundary changed')
    volume_error=_volume(final,patch)-_volume(neutral,patch)
    if abs(volume_error)>2e-10:raise RuntimeError('Float32 lower-face volume restoration escaped tolerance')
    # Actual face normals must retain orientation; quantify stretch without
    # calling its reduction an appearance verdict.
    def cross(p):return np.cross(p[patch[:,1]]-p[patch[:,0]],p[patch[:,2]]-p[patch[:,0]])
    before_cross=cross(neutral);after_cross=cross(final)
    orientation=np.einsum('ij,ij->i',before_cross,after_cross)
    if np.any(orientation<=0):
        bad=np.flatnonzero(orientation<=0)
        raise RuntimeError('Candidate inverted triangles: '+json.dumps([
            {'ids':patch[i].tolist(),'before':neutral[patch[i]].tolist(),'delta':delta_actual[patch[i]].tolist()}
            for i in bad[:4]]))
    expression_checks={}
    for label,values in {'ee63':{'R02_mouth_wide':.75,'R02_jaw_open':.2},
                         'ou98':{'R02_pucker':.85,'R02_jaw_open':.2},
                         'smile1':{'R02_smile':.55}}.items():
        posed=neutral.copy()
        for name,value in values.items():posed+=(current[name].astype(float)-basis)*value
        old_cross=cross(posed);new_cross=cross(posed+delta_actual)
        inverted=np.einsum('ij,ij->i',old_cross,new_cross)<=0
        if np.any(inverted):raise RuntimeError('Static curvature candidate inverted a local triangle in '+label)
        expression_checks[label]={'localTriangleInversions':0,'trianglesChecked':len(patch),
            'minimumTriangleAreaRatio':float(np.min(np.linalg.norm(new_cross,axis=1)/np.maximum(np.linalg.norm(old_cross,axis=1),1e-15))),
            'scope':'Local unrigged actual shape-key combination; not render/occlusion approval.'}
    signed=np.einsum('ij,ij->i',delta_actual,normals)
    before_lap=mouth['_mean'](neutral,edges,ew,degree)-neutral
    after_lap=mouth['_mean'](final,edges,ew,degree)-final
    report={'observationSourceSha256':metadata['sourceSha256'],
        'requiredInput':'Observed static R09 arrays plus exact adopted perioral four-key targets',
        'key':KEY,'keyValue':1.,'relativeKey':'Basis','vertices':len(neutral),
        'editableVertices':int(editable.sum()),'changedVertices':int((np.linalg.norm(delta_actual,axis=1)>1e-8).sum()),
        'protectedLipVertices':len(lip),'lipAndOralAndNoseEyeCoordinatesBitExact':True,
        'originalShapeKeysChanged':False,'wholeFaceSculptAttenuation':False,
        'method':'16 bounded topology-curvature flow steps on actual 3-D positions in one lower-face window, exact lip/cavity pins, then local signed-volume restoration along actual mesh normals.',
        'maximumCorrectionMeters':float(np.linalg.norm(delta_actual,axis=1).max()),
        'maximumAbsoluteXYZMeters':np.max(np.abs(delta_actual),axis=0).tolist(),
        'normalDisplacementRangeMeters':[float(signed[editable].min()),float(signed[editable].max())],
        'localSurfaceAreaMeters2':float(areas[editable].sum()),
        'signedPatchVolumeChangeMeters3':volume_error,
        'normalRestorationMeters':[restore_a,restore_b],'globalCapScale':scale,
        'weightedMeanNormalChangeMeters':float((signed*areas).sum()/max(areas[editable].sum(),1e-20)),
        'localLaplacianRmsMeters':{'before':float(np.sqrt(np.mean(np.sum(before_lap[editable]**2,axis=1)))),
                                'after':float(np.sqrt(np.mean(np.sum(after_lap[editable]**2,axis=1))))},
        'triangleInversions':0,'candidateKeySha256':_hash(target),
        'expressionLocalChecks':expression_checks,'protectedLipAndOralFirstRingVertices':int((distance<=1).sum()),
        'requiredInputKeyHashes':{name:_hash(co) for name,co in current.items()},
        'topology':{name:mouth_report[name] for name in ('edgeArraySha256','polygonAndMaterialSha256')},
        'limits':['Volume means signed volume change of the affected real triangles with fixed boundary; not an anatomical measurement.',
                  'Curvature/volume metrics do not establish attractive or reference-matching appearance.',
                  'One matched front and quarter neutral render plus EE/OU should assess lower-lip platform, corner folds and preservation of the soft oval silhouette.',
                  'This additive static key preserves existing expression deltas; extreme jaw combinations remain a separate visual check.'],
        'appearanceAccepted':False,'savedBlend':False}
    return report,target,current,editable,pinned


def _head_contract(head,uv,original_names):
    state=uv['_original_contract'](head,tuple(layer.name for layer in head.data.uv_layers))
    state.pop('shapeKeysSha256');state.pop('sha256')
    state['objectSettings'].pop('dimensions',None)
    keys=head.data.shape_keys
    return {'head':state,'keyIdPointer':keys.as_pointer(),'keySettings':uv['_scalar_rna'](keys),
        'keyCustom':uv['_custom'](keys),'animation':uv['_animation'](keys),
        'existingKeys':{name:{'settings':uv['_scalar_rna'](keys.key_blocks[name]),
                             'relative':keys.key_blocks[name].relative_key.name}
                        for name in original_names}}


def run(source_relative,diagnostics_only=False):
    import bpy
    source=_path(source_relative,'.blend')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open explicit independent source in background Object Mode')
    head=bpy.data.objects[HEAD];blocks=head.data.shape_keys.key_blocks
    if KEY in blocks or MARKER in head:raise RuntimeError('Preserve or restore existing lower-jaw candidate')
    if head.library or head.data.library or head.data.users!=1:raise RuntimeError('Local single-user head required')
    report,target,expected,editable,pinned=plan();mouth=_api('perioral_delta_r09')
    if [key.name for key in blocks]!=list(expected):raise RuntimeError('Existing shape-key order changed')
    if not head.data.shape_keys.use_relative or any(key.relative_key.name!='Basis' or key.vertex_group for key in blocks):
        raise RuntimeError('Expected original unmasked Basis-relative production key graph')
    originals={key.name:mouth['_coords'](key.data) for key in blocks}
    if any(not np.array_equal(originals[name],co) for name,co in expected.items()):
        raise RuntimeError('Current head differs from exact adopted perioral input; do not reuse stale curvature field')
    actual_edges=np.empty((len(head.data.edges),2),np.int32);head.data.edges.foreach_get('vertices',actual_edges.ravel())
    actual_topology=[[list(face.vertices) for face in head.data.polygons],[face.material_index for face in head.data.polygons],
                     [slot.material.name if slot.material else None for slot in head.material_slots]]
    topology_sha=hashlib.sha256(json.dumps(actual_topology,separators=(',',':')).encode()).hexdigest()
    if _hash(actual_edges)!=report['topology']['edgeArraySha256'] or topology_sha!=report['topology']['polygonAndMaterialSha256']:
        raise RuntimeError('Actual geometry/material domains differ from observed topology')
    uv=_api('production_uv_r08');groom=_api('posterior_groom_r09');bpy.context.view_layer.update()
    names=list(originals);active_index=head.active_shape_key_index
    contract=_head_contract(head,uv,names);others=mouth['_other_scene_contract'](bpy,uv,groom)
    source_sha=_sha(source);dimensions_before=list(head.dimensions)
    report.update(source=source_relative,sourceSha256=source_sha,writtenInMemory=False)
    if diagnostics_only:return report
    key=None;committed=False
    try:
        key=head.shape_key_add(name=KEY,from_mix=False)
        key.relative_key=blocks['Basis'];key.data.foreach_set('co',target.ravel());key.value=1.
        head.active_shape_key_index=active_index
        head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
        after=_head_contract(head,uv,names)
        if after!=contract:
            diff=mouth['_contract_diff'](contract,after)
            print('R09_LOWER_JAW_CONTRACT_DIFF '+json.dumps(diff,ensure_ascii=False),flush=True)
            raise RuntimeError('Original source contract changed: '+json.dumps(diff,ensure_ascii=False))
        for name,co in originals.items():
            if not np.array_equal(mouth['_coords'](blocks[name].data),co):raise RuntimeError('Original key changed: '+name)
        if not np.array_equal(mouth['_coords'](key.data),target):raise RuntimeError('New static key coordinates differ from planned target')
        if mouth['_other_scene_contract'](bpy,uv,groom)!=others:raise RuntimeError('Other geometry, rig, actions or native caches changed')
        if _sha(source)!=source_sha:raise RuntimeError('Saved source changed during preparation')
        dimensions_after=list(head.dimensions)
        report.update(writtenInMemory=True,originalKeysDriversUVMaterialsAndCachesUnchanged=True,
            derivedHeadBoundingDimensions={'before':dimensions_before,'after':dimensions_after,
                                          'differenceMeters':[b-a for a,b in zip(dimensions_before,dimensions_after)]},
            reversible='Set R09_lower_jaw_curvature to 0 for comparison, or call restore() to remove only this new key.',
            runtimeConsequence='If adopted, refresh the face snapshot from this static-key-enabled source; existing materials and UV are unchanged.')
        head[MARKER]=json.dumps({'key':KEY,'candidateKeySha256':report['candidateKeySha256'],
            'sourceSha256':source_sha,'originalKeyNames':names,'originalKeyHashes':report['requiredInputKeyHashes']})
        committed=True;return report
    finally:
        if not committed:
            if key is not None and KEY in head.data.shape_keys.key_blocks:head.shape_key_remove(head.data.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active_index
            if MARKER in head:del head[MARKER]
            head.data.update_tag();head.update_tag();bpy.context.view_layer.update()


def apply_and_save(source_relative,output_relative='work/reconstruction-r02/reference-lower-jaw-curvature-r09.blend',
                   report_relative='work/reconstruction-r02/lower-jaw-curvature-r09.json'):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=_path(report_relative,'.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('New independent candidate/report paths required')
    report=run(source_relative)
    if output.exists():raise RuntimeError('Output appeared during preparation')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


def restore():
    import bpy
    head=bpy.data.objects[HEAD]
    if MARKER not in head:return False
    record=json.loads(head[MARKER]);blocks=head.data.shape_keys.key_blocks;mouth=_api('perioral_delta_r09')
    if KEY not in blocks or _hash(mouth['_coords'](blocks[KEY].data))!=record['candidateKeySha256']:
        raise RuntimeError('Static candidate was edited; preserve edits before restoring')
    if [key.name for key in blocks if key.name!=KEY]!=record['originalKeyNames']:
        raise RuntimeError('Other shape keys added or removed since this candidate')
    for name,expected in record['originalKeyHashes'].items():
        if _hash(mouth['_coords'](blocks[name].data))!=expected:raise RuntimeError('Original key changed after candidate: '+name)
    head.shape_key_remove(blocks[KEY]);del head[MARKER]
    head.data.update_tag();head.update_tag();bpy.context.view_layer.update()
    return {'removedKey':KEY,'savedBlend':False}


if __name__=='__main__':print('Loaded only. plan() is offline; run/apply_and_save require explicit independent source.')
