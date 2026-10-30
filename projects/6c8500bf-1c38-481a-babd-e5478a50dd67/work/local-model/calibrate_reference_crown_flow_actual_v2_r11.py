"""Recalibrate the frozen crown-flow plan from one actual post-tie observation.

Pure NumPy/exact triangle BVH only. No Blender import, launch, model edits or
render. The actual report and every source array hash are verified first.
Outputs a single actual-source plan JSON/NPZ pair for the production wrapper.
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
P=project_path();HERE=Path(__file__).resolve().parent
PURE_SHA='9068dc7642b5fe523d365bf10583213f46730e7f4be1a68a2d4f9ddff9396e2d'
SURFACE_SHA='bc8ba9d59fff02a95396a20ac78621dfd0491f098fc77ea80914d3e67d7a8d85'
OBSERVER_SHA='479a76f0587898d1a4ab476110360d281b1919fd8739b18651ecaa4a02f90e0e'
SEMANTIC_SHA='2efc236d84d5e32b67c9e20082805535aae96bae31cfe69fca10bb1c75f30683'
TAKEOFF_EXCEPTIONS={10330,10527,10540,10541,10542}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def array_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()

def scoped(relative):
    path=(P/str(relative).replace('\\','/')).resolve();path.relative_to((P/'work/reconstruction-r02').resolve());return path

def need(condition,message):
    if not condition:raise RuntimeError(message)

def load_semantic_evidence(gather_path,gather_record,tie_record):
    """Resolve only hash-linked recorded provenance; never infer a path."""
    chain=[]
    if tie_record is not None:
        evidence=tie_record.get('verifiedFaceCurtainEvidence')
        need(isinstance(evidence,dict),'Successful tie report lacks verifiedFaceCurtainEvidence')
        relative=evidence.get('path');digest=evidence.get('sha256')
        origin='successfulTieReport.verifiedFaceCurtainEvidence'
    else:
        current_path=gather_path;current_record=gather_record;seen=set()
        while not current_record.get('semanticReview'):
            need(str(current_path) not in seen,'Cycle in gather base-plan semantic provenance')
            seen.add(str(current_path))
            relative_base=current_record.get('baseGatherPlan');base_sha=current_record.get('baseGatherPlanSha256')
            need(isinstance(relative_base,str) and bool(relative_base) and isinstance(base_sha,str) and bool(base_sha),
                 'Gather plan has neither semanticReview nor complete baseGatherPlan/baseGatherPlanSha256 provenance: '+str(current_path.relative_to(P)))
            base_path=scoped(relative_base)
            need(base_path.is_file(),'Recorded base gather plan does not exist: '+str(base_path.relative_to(P)))
            need(sha(base_path)==base_sha,'Base gather plan semantic provenance hash mismatch: '+str(base_path.relative_to(P)))
            chain.append({'path':str(base_path.relative_to(P)),'sha256':base_sha})
            current_path=base_path;current_record=json.loads(base_path.read_text(encoding='utf8'))
        relative=current_record.get('semanticReview');digest=current_record.get('semanticReviewSha256')
        origin='gatherPlan.semanticReview' if not chain else 'baseGatherPlan.semanticReview'
    need(isinstance(relative,str) and bool(relative),'Recorded face-curtain semantic evidence path is missing')
    need(digest==SEMANTIC_SHA,'Recorded face-curtain semantic evidence hash is not the frozen identity')
    semantic_path=scoped(relative)
    need(semantic_path.is_file(),'Recorded face-curtain semantic evidence does not exist: '+str(semantic_path.relative_to(P)))
    need(sha(semantic_path)==SEMANTIC_SHA,'Frozen face-curtain semantics changed')
    semantic=json.loads(semantic_path.read_text(encoding='utf8'))
    if tie_record is not None and 'curveIds' in evidence:
        need(evidence['curveIds']==semantic.get('suspectDenseCurveIds'),'Successful tie semantic curve IDs differ from the frozen evidence')
    return semantic,{'path':str(semantic_path.relative_to(P)),'sha256':SEMANTIC_SHA,'origin':origin,'baseGatherPlanChain':chain}

def load_actual(report_path):
    report=json.loads(report_path.read_text(encoding='utf8'))
    need(report.get('status')=='collected','Actual observer has not completed successfully')
    need(report.get('helperSha256')==OBSERVER_SHA,'Expected one frozen complete post-tie observer')
    need(report.get('savedBlend') is False and report.get('rendered') is False,'Observation must be read-only')
    need(report.get('observationPose')=='REST','Current observed geometry must be in REST')
    need(not report.get('readOnlyRestoredPoseChanges'),'Observer changed a protected raw contract')
    for key in ('sourceUnchanged','sourcePoseRestored','sourceFrameAndActionUnchanged'):
        need(report.get(key) is True,'Actual observation contract failed: '+key)
    source=scoped(report['source']);need(sha(source)==report['sourceSha256'],'Actual observed source file has changed')
    payload=scoped(report['arrays']);need(sha(payload)==report['arraysSha256'],'Actual observation payload hash mismatch')
    values={}
    with np.load(payload,allow_pickle=False) as data:
        need(set(data.files)==set(report['arrayManifest']),'Actual array manifest keys differ')
        for key in data.files:
            value=data[key];meta=report['arrayManifest'][key]
            need(list(value.shape)==meta['shape'] and str(value.dtype)==meta['dtype'] and array_sha(value)==meta['sha256'],'Actual array manifest mismatch: '+key)
            values[key]=value
    required=['head_world','head_triangles','growth_actual_density','target_carrier_local','target_carrier_world','target_carrier_matrix']
    for prefix in ('post-tie','final'):
        required.extend(prefix+'_'+suffix for suffix in ('world','starts','lengths','group','radius'))
    for prefix in ('native_post_tie','native_final'):
        required.extend(prefix+'_'+suffix for suffix in ('world','starts','lengths','group','original_curve_ids'))
    required.extend('face_wisps_raw_'+suffix for suffix in ('world','local','starts','lengths','radius','matrix'))
    for side in ('positive','negative'):required.append('tail_'+side+'_first3_world')
    need(all(key in values for key in required),'One or more required actual observation fields are absent')
    return report,payload,values

def native_contract(a,prefix,base):
    ids=a[prefix+'_original_curve_ids'];starts=a[prefix+'_starts'];lengths=a[prefix+'_lengths'];world=a[prefix+'_world']
    need(ids.ndim==1 and len(ids)==len(np.unique(ids)) and np.all(np.diff(ids)>0),'Native original curve IDs must be ordered and unique: '+prefix)
    need(np.all(lengths==128) and np.array_equal(starts,np.arange(len(ids),dtype=np.int32)*128),'Actual native crown must be 128 samples per curve')
    need(np.array_equal(a[prefix+'_group'],a[base+'_group'][ids]),'Native curve groups do not match actual control identities')
    delta=np.linalg.norm(world[starts]-a[base+'_world'][a[base+'_starts'][ids]],axis=1)
    need(float(delta.max())<=2e-6,'Actual native roots do not match the corresponding stage')
    return {'curves':len(ids),'samplesPerCurve':128,'rootDifferenceMaxMeters':float(delta.max())}

def endpoint_protection(a,gather,semantic):
    s=a['post-tie_starts'];n=a['post-tie_lengths'];g=a['post-tie_group'];q=a['post-tie_world']
    for suffix in ('starts','lengths','group'):
        need(np.array_equal(a['post-tie_'+suffix],gather['source_'+suffix]),'Frozen connector map no longer matches actual ordered topology: '+suffix)
    delta=np.linalg.norm(q-gather['target_world'],axis=1)
    need(float(delta.max())<=2e-6,'Actual saved post-tie stage differs from its frozen v3 target; explicit new connector map required')
    excluded=gather['preserved_face_curtain_curve_ids']
    need(np.array_equal(excluded,np.asarray(semantic['suspectDenseCurveIds'],np.int32)) and len(excluded)==46,'Exact verified 46-curve semantic identity mismatch')
    ids=gather['gather_curve_ids'];last=gather['preserved_upper_last_point']
    need(len(ids)==len(np.unique(ids)) and len(ids)==len(last),'Frozen gather connector map is malformed')
    need(not np.intersect1d(ids,excluded).size and np.all(np.isin(g[ids]%256,(40,41))),'Connector mapping selects a protected/non-gather curve')
    protect=n-3;protect[ids]=np.minimum(protect[ids],last)
    need(np.all(protect>=0) and np.all(protect<=n-3),'Invalid preserved-upper connector boundary')
    zero=set(map(int,np.flatnonzero(protect==0)))
    need(zero.issubset(TAKEOFF_EXCEPTIONS),'Unexpected complete-curve connector protection exception')
    need(np.array_equal(a['post-tie_group'],a['final_group']),'Actual final curve order/roles differ from post-tie stage')
    need(len(a['face_wisps_raw_starts'])==158,'Expected 158 current independent face-wisp curves')
    need(a['target_carrier_world'].shape==q.shape,'Target-carrier point identity count differs')
    carrier_delta=float(np.linalg.norm(a['target_carrier_world']-q,axis=1).max())
    need(carrier_delta<=2e-6,'Current carrier and post-tie point indices no longer correspond')
    expected=np.zeros(len(n),bool);expected[excluded]=True;expected=np.repeat(expected,n)
    need(np.array_equal(a.get('post-tie_verified_temple_curtain'),expected),'Actual POINT semantic mask does not preserve the verified 46 curves')
    return protect,excluded,{'actualPostTieToFrozenTargetMaximumMeters':float(delta.max()),'carrierCorrespondenceMaximumMeters':carrier_delta,'verifiedFaceCurtains':46,'currentIndependentFaceWisps':158}

def summarize_native(a,ids,recovery,pure):
    result={};rows={}
    for prefix in ('native_post_tie','native_final'):
        lookup={int(i):j for j,i in enumerate(a[prefix+'_original_curve_ids'])}
        need(all(int(i) in lookup for i in ids),'Actual native crown ROI does not cover the entire proposed edit')
        first=[];turn=[]
        for i,end in zip(ids,recovery):
            j=lookup[int(i)];s=int(a[prefix+'_starts'][j]);n=int(a[prefix+'_lengths'][j]);q=a[prefix+'_world'][s:s+n].astype(float)
            ar=pure['arc'](q);first.append(pure['initial'](q));joint=pure['turns'](q);turn.extend(joint[ar[1:-1]<=end])
        rows[prefix+'_selected_initial_direction']=np.asarray(first)
        result[prefix]={'selectedCurves':len(ids),'actualSamplesPerCurve':128,'withinRecoveryNativeSampleTurnDegreesP50P95P100':np.percentile(turn,[50,95,100]).tolist(),
                        'meaning':'Actual baseline native path, before proposed edit; does not certify the proposed path.'}
    return result,rows

def validate_estimated_paths(a,output,pure,surface):
    """Current observed final roots + new cleared controls; explicit estimate."""
    source=output['source_post_tie_world'];target=output['target_world'];starts=output['starts'];lengths=output['lengths']
    fs=a['final_starts'];fn=a['final_lengths'];final=a['final_world'];radius=a['final_radius'];signed=[];points=[];curve_ids=[];offsets=[];kinds=[];turn_before=[];turn_after=[];copied=[]
    for ix,i in enumerate(output['selected_curve_ids']):
        if ix%200==0:print('CROWN_ACTUAL_PATH_ESTIMATE '+str(ix)+'/'+str(len(output['selected_curve_ids'])),flush=True)
        s=int(starts[i]);n=int(lengths[i]);f=int(fs[i]);need(n==fn[i],'Selected curve has downstream topology change; explicit path mapping required')
        before=source[s:s+n].astype(float);candidate=target[s:s+n].astype(float);changed=np.any(candidate!=before,axis=1)
        # Existing head contact is downstream. For every unmodified control,
        # use its current actually observed final coordinate. New controls
        # preserve >=0.4mm exact triangle clearance in the pure planner.
        estimated=candidate.copy();estimated[~changed]=final[f:f+n][~changed];copied.append(int((~changed).sum()))
        segment=np.flatnonzero(changed[:-1]|changed[1:]);joint=np.flatnonzero(changed[:-2]|changed[1:-1]|changed[2:]);controls=np.flatnonzero(changed)
        turn_before.extend(pure['turns'](final[f:f+n].astype(float))[joint]);turn_after.extend(pure['turns'](estimated)[joint])
        native,owners=pure['catmull_samples'](estimated,segment)
        bound=float(radius[f:f+n].max())
        for kind,q,owner in ((0,estimated[controls],controls),(1,(estimated[segment]+estimated[segment+1])*.5,segment),(2,native,owners)):
            for point,j in zip(q,owner):
                hit,no,_,_=surface.find_nearest(point);signed.append(float(np.dot(point-hit,no))-bound);points.append(point);curve_ids.append(int(i));offsets.append(int(j));kinds.append(kind)
    signed=np.asarray(signed);output.update(estimated_surface_signed=signed,estimated_surface_world=np.asarray(points),estimated_surface_curve_id=np.asarray(curve_ids,np.int32),estimated_surface_control_or_segment_offset=np.asarray(offsets,np.int32),estimated_surface_kind=np.asarray(kinds,np.int8))
    worst=np.argsort(signed)[:24];error=[] if np.min(signed)>=-.0005 else ['Current-head offline proposed path estimate penetrates beyond -0.5 mm']
    return {'samples':len(signed),'actualRadiusIncluded':True,'surfaceClearanceMetersP0P1P50P100':np.percentile(signed,[0,1,50,100]).tolist(),
            'belowMinusHalfMillimeter':int((signed<-.0005).sum()),'errors':error,
            'beforeChangedJointTurnDegreesP50P95P100':np.percentile(turn_before,[50,95,100]).tolist(),
            'estimatedChangedJointTurnDegreesP50P95P100':np.percentile(turn_after,[50,95,100]).tolist(),
            'worst':[{'curve':curve_ids[k],'pointOrSegmentOffset':offsets[k],'kind':kinds[k],'surfaceClearanceMeters':float(signed[k])} for k in worst],
            'method':'Changed controls, changed segment midpoints and 5 uniform Catmull-Rom interior estimates per changed segment against complete actual current head triangles. Unmodified controls use actual current final-stage coordinates, so preserved pre-contact roots are interpreted through their observed downstream contact.',
            'nativeModifiedPathExecuted':False,'completeActualNativeAndRenderStillRequired':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observation',required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--tie-report');group.add_argument('--gather-plan')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    observation_path=scoped(args.observation);out=scoped(args.output).with_suffix('.json');npz=out.with_suffix('.npz')
    need(not out.exists() and not npz.exists(),'New actual-source plan outputs required')
    need(sha(HERE/'reference_crown_flow_plan_r11.py')==PURE_SHA,'Frozen pure plan changed')
    need(sha(HERE/'collar_continuous_offline_r11.py')==SURFACE_SHA,'Exact triangle nearest oracle changed')
    observation,payload,a=load_actual(observation_path)
    tie_record=None;tie_path=None
    if args.tie_report:
        tie_path=scoped(args.tie_report);tie_record=json.loads(tie_path.read_text(encoding='utf8'))
        need(tie_record.get('savedBlend') is True and tie_record.get('technicalAcceptance') is True and not tie_record.get('errors'),'Only an actually saved successful tie report may select the gather plan')
        need(tie_record.get('outputSha256')==observation['sourceSha256'],'Successful tie source is not the actually observed source')
        gather_path=scoped(tie_record['gatherPlan']);need(sha(gather_path)==tie_record['gatherPlanSha256'],'Successful tie gather proposal hash mismatch')
    else:gather_path=scoped(args.gather_plan)
    record={'status':'actual-source-proposal-failed','observation':str(observation_path.relative_to(P)),
            'observationSha256':sha(observation_path),'observationArraysSha256':sha(payload),
            'source':observation['source'],'sourceSha256':observation['sourceSha256'],
            'gatherPlan':str(gather_path.relative_to(P)),'gatherPlanSha256':sha(gather_path),
            'helperSha256':sha(__file__),'pureHelperSha256':PURE_SHA,'exactTriangleOracleSha256':SURFACE_SHA,
            'savedBlend':False,'actualModifiedGeometryNodesExecuted':False,'appearanceAccepted':False,'errors':[]}
    output={}
    try:
        gather_record=json.loads(gather_path.read_text(encoding='utf8'))
        need(gather_record.get('status')=='proposed' and not gather_record.get('errors'),'A reviewed error-free actual gather proposal is required')
        gather_arrays=scoped(gather_record['arrays']);need(sha(gather_arrays)==gather_record['arraysSha256'],'Actual gather payload hash mismatch')
        if tie_record is not None:
            need(tie_record['gatherArraysSha256']==gather_record['arraysSha256'],'Successful tie report points to a different gather payload')
            record['successfulTieReport']=str(tie_path.relative_to(P));record['successfulTieReportSha256']=sha(tie_path)
        with np.load(gather_arrays,allow_pickle=False) as f:gather={key:f[key] for key in ('source_starts','source_lengths','source_group','target_world','gather_curve_ids','preserved_upper_last_point','preserved_face_curtain_curve_ids')}
        semantic,record['verifiedFaceCurtainEvidence']=load_semantic_evidence(gather_path,gather_record,tie_record)
        protect,excluded,record['correspondence']=endpoint_protection(a,gather,semantic)
        record['actualNativeBaseline']={prefix:native_contract(a,prefix,base) for prefix,base in (('native_post_tie','post-tie'),('native_final','final'))}
        pure=runpy.run_path(str(HERE/'reference_crown_flow_plan_r11.py'));Surface=runpy.run_path(str(HERE/'collar_continuous_offline_r11.py'))['Surface']
        surface=Surface(a['head_world'],a['head_triangles'])
        guards={key:array_sha(value) for key,value in a.items() if key.startswith(('face_wisps_raw_','tail_positive_','tail_negative_','target_carrier_','main_raw_','growth_'))}
        print('CROWN_ACTUAL_RECALIBRATION_START',flush=True)
        # The new tie proposal may declare five lower takeoffs protected from
        # p0 onward. Frozen pure planner accepts boundaries >=1; its p0 is
        # already exact. Pass >=1 and explicitly restore the stronger complete
        # curve protection after confirming those curves were never selected.
        _,plan_report,output=pure['plan'](a['post-tie_world'],a['post-tie_starts'],a['post-tie_lengths'],a['post-tie_group'],excluded,surface,connector_first_protected=np.maximum(protect,1))
        fully_protected=np.flatnonzero(protect==0)
        need(not np.intersect1d(fully_protected,output['selected_curve_ids']).size,'Crown plan selected a fully protected corrected takeoff')
        output['protected_from_point_offset']=protect.copy()
        output['connector_protected_point_mask']=np.concatenate([np.arange(count)>=boundary for count,boundary in zip(output['lengths'],protect)])
        output['fully_protected_corrected_takeoff_curve_ids']=fully_protected.astype(np.int32)
        protected_points=output['connector_protected_point_mask']
        need(np.array_equal(output['source_post_tie_world'][protected_points],output['target_world'][protected_points]),'Corrected takeoff/connector protection changed')
        record['fullyProtectedCorrectedTakeoffs']=fully_protected.tolist()
        need(plan_report['changedCurves']>0,'Actual plan selected no crown curves')
        record['plan']=plan_report
        print('CROWN_ACTUAL_PLAN_READY '+json.dumps({'curves':plan_report['changedCurves'],'points':plan_report['changedPoints']}),flush=True)
        record['nativeBaselineDirections'],extra=summarize_native(a,output['selected_curve_ids'],output['recovery_arc_per_selected_curve'],pure);output.update(extra)
        record['pathEstimate']=validate_estimated_paths(a,output,pure,surface)
        record['errors'].extend(record['pathEstimate']['errors'])
        record['observedUnchangedOtherAssets']={'faceWisps':158,'scope':'Offline inputs remain exact; future scene invariance must be checked again by production wrapper.','arrayHashes':guards}
        need(all(array_sha(a[key])==digest for key,digest in guards.items()),'Planner mutated an actual observed independent asset array')
        for suffix in ('world','starts','lengths','radius','matrix'):output['face_wisps_guard_'+suffix]=a['face_wisps_raw_'+suffix].copy()
        for side in ('positive','negative'):output['tail_'+side+'_guard_first3_world']=a['tail_'+side+'_first3_world'].copy()
        output['source_head_world']=a['head_world'].copy();output['source_head_triangles']=a['head_triangles'].copy()
        record['actualGrowthDensity']={'minimum':float(a['growth_actual_density'].min()),'maximum':float(a['growth_actual_density'].max()),'arraySha256':array_sha(a['growth_actual_density'])}
        record['allCurrentHeadTrianglesUsed']=len(a['head_triangles'])
        record['installation']={'after':'R11 reference gather tie transport','before':'R11 full head clearance','strength':1.0,
                                'offsetSpace':'World vectors; installer converts using donor inverse linear 3x3 only, no translation.','carrierPolicy':'Add POINT vector+bool attributes on copied existing carrier data. Keep all existing positions/attributes exact and old data retained.',
                                'rootAndConnectorPolicy':'SetPosition Selection false at every protected point; do not rely on a zero vector to preserve exact positions.',
                                'restore':'Strength=0 or bypass restores current post-tie geometry. Intermediate strengths require actual native validation.'}
        record['limitations']=['This proposal is calibrated from an actual saved source, but its modified paths are still offline estimates.',
                               'Actual modified native output, current-surface collisions, all preservation contracts and same-camera images remain mandatory.',
                               'The unchanged 158 independent wisps and tail pins are recorded as input guards, not falsely claimed to have been re-evaluated after a Blender edit.']
        if not record['errors']:record['status']='actual-source-proposal'
    except Exception as exc:
        record['errors'].append(str(exc));record['error']=str(exc)
    if output:
        with npz.open('xb') as f:np.savez_compressed(f,**output)
        record['arrays']=str(npz.relative_to(P));record['arraysSha256']=sha(npz)
    with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    print('CROWN_ACTUAL_PROPOSAL '+json.dumps({'status':record['status'],'report':str(out.relative_to(P)),'errors':record['errors']}),flush=True)
    if record['errors']:raise RuntimeError('; '.join(record['errors']))

if __name__=='__main__':main()
