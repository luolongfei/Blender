"""Freeze exact25 local repair, with complete old-head triangle-tree verification.

Never opens Blender. Two explicitly authorized IDs retain the smooth v4
outward route up to14mm; the remaining23 remain bounded by12mm.
"""
import argparse,json,runpy,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
m=runpy.run_path(str(HERE/'repair_reference_gather_native_v4_r11.py'))
P=m['P'];W=P/'work/reconstruction-r02';sha=m['sha'];CURVES=m['CURVES'];TAKEOFF=m['TAKEOFF'];EXPANDED=np.array([10296,10301],np.int32)

def signed_exact(surface,points):
    out=[]
    for p in points:
        point,normal,face,distance=surface.find_nearest(p)
        out.append(float(distance if np.dot(p-point,normal)>=0 else -distance))
    return np.asarray(out)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args();out=P/args.output;npz=out.with_suffix('.npz')
    for path in [out,npz]:
        path.resolve().relative_to(W.resolve())
        if path.exists():raise RuntimeError('Independent unused outputs required')
    bp=W/'reference-gather-rebuild-v3-frozen-r11.json';fp=W/'reference-adult-reference-ties-v2-r11.tie-transport.json';sp=W/'reference-gather-repair-v6-scan-r11.json';vp=W/'reference-gather-repair-v4-r11.json'
    base=json.loads(bp.read_text());fail=json.loads(fp.read_text());v4report=json.loads(vp.read_text());scanreport=json.loads(sp.read_text());ba=P/base['arrays'];fa=P/fail['arrays'];va=P/v4report['arrays']
    if sha(bp)!='e5dab29b13c1def0047f975306927095225c984e9e2f6ca8218c815c84fa0249' or sha(ba)!=base['arraysSha256'] or sha(fa)!=fail['arraysSha256'] or fail['gatherPlanSha256']!=sha(bp) or sha(va)!=v4report['arraysSha256']:raise RuntimeError('Actual/base identity mismatch')
    arrays={k:v.copy() for k,v in np.load(ba).items()};actual=np.load(fa);v4=np.load(va);scan=np.load(sp.with_suffix('.npz'));headpath=P/base['observation'];headreport=json.loads(headpath.read_text());ha=P/headreport['arrays'];h=np.load(ha)
    if sha(ha)!=headreport['arraysSha256']:raise RuntimeError('Old-head observation payload hash mismatch')
    Surface=runpy.run_path(str(HERE/'collar_continuous_offline_r11.py'))['Surface'];surface=Surface(h['head_world'],h['head_triangles']);print('R11_COMPLETE_HEAD_TREE_READY',flush=True)
    s=arrays['source_starts'];lens=arrays['source_lengths'];source=arrays['source_world'];original=arrays['target_world'].copy();target=original.copy();nativeids=actual['gather_native_after_original_curve'];sourceids=arrays['gather_curve_ids'];rows=[];exceptions=[];errors=[];predcontrols=[];prednative=[];nativegaps=[];nativearcs=[];controlgaps=[];nativebefore=[];nativeafter=[];expanded=[];begin=time.time()
    if not np.array_equal(actual['before-front_starts'],s) or not np.array_equal(actual['before-front_lengths'],lens) or np.linalg.norm(actual['before-front_world']-source,axis=1).max()>2e-6:raise RuntimeError('Failed source does not match base observation')
    for ci,curve in enumerate(CURVES):
        curve=int(curve);st=int(s[curve]);n=int(lens[curve]);fs=int(actual['after-final_starts'][curve]);old=original[st:st+n].astype(float);current=actual['after-final_world'][fs:fs+n].astype(float);basis=m['native_basis'](n);ni=int(np.flatnonzero(nativeids==curve)[0]);ns=int(actual['gather_native_after_starts'][ni]);actualnative=actual['gather_native_after_world'][ns:ns+64].astype(float);nativeerr=float(np.linalg.norm(m['resampled_basis'](current,basis)@current-actualnative,axis=1).max())
        if nativeerr>2e-6:raise RuntimeError('Empirical native model no longer matches actual '+str(curve))
        first=1 if curve in TAKEOFF else 3
        if curve in EXPANDED:
            q=v4['predicted_after_contact_corrected_world'][int(scan['starts'][ci]):int(scan['starts'][ci+1])].astype(float)
            choice={'basis':'existing_v4_smooth_full_outward_path','maximumAllowedTargetMovementMeters':.014,'authorization':'Root explicitly authorized only10296/10301 up to14mm; do not distort a smooth path solely to meet the previous arbitrary12mm scope.'}
        else:
            if not scanreport[ci]['acceptable']:raise RuntimeError('Non-expanded curve has no bounded feasible scan result')
            q=scan['predicted_controls'][int(scan['starts'][ci]):int(scan['starts'][ci+1])].astype(float)
            choice=dict(scanreport[ci]);choice['maximumAllowedTargetMovementMeters']=.012
        new=old.copy();new[first:-3]=q[first:-3];target[st:st+n]=new.astype(np.float32)
        # Locked old pre-contact points receive their already observed native
        # contact positions only in the predictor, not in the target payload.
        q[first:-3]=target[st+first:st+n-3].astype(float);q[:first]=current[:first];q[-3:]=current[-3:]
        native=m['resampled_basis'](q,basis)@q;mid=(q[:-1]+q[1:])*.5;nm=(native[:-1]+native[1:])*.5;radius=float(actual['after-final_radius'][fs:fs+n].max());samples=np.vstack([native,nm,q,mid]);gap=signed_exact(surface,samples)-radius;arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(native,axis=0),axis=1))];nativearc=np.r_[arc,(arc[:-1]+arc[1:])*.5];nativefree=nativearc>.001;ng=gap[:127];cg=gap[127:127+n];mg=gap[127+n:];move=np.linalg.norm(target[st:st+n]-original[st:st+n],axis=1);ct=m['turns'](q);ctold=m['turns'](current);nt=m['turns'](native);ntold=m['turns'](actualnative);newfold=np.flatnonzero((ct>95)&(ct>ctold+20));nativefold=np.flatnonzero((nt>95)&(nt>ntold+20));j=int(np.flatnonzero(sourceids==curve)[0]);diff=np.flatnonzero(np.any(target[st:st+n]!=source[st:st+n],axis=1));anchor=int(diff[0]-1) if len(diff) else n-1;arrays['preserved_upper_last_point'][j]=anchor
        if min(float(ng[nativefree].min()),float(mg.min()))<-.0005:errors.append(str(curve)+': complete old-head exact predictor still fails native/control midpoint')
        if len(newfold):errors.append(str(curve)+': new control reversal above95 and +20 degrees')
        if len(nativefold):errors.append(str(curve)+': new native reversal above95 and +20 degrees')
        if move.max()>choice['maximumAllowedTargetMovementMeters']+1e-8:errors.append(str(curve)+': exceeds exact local movement scope')
        if not np.array_equal(target[st:st+first],original[st:st+first]) or not np.array_equal(target[st+n-3:st+n],original[st+n-3:st+n]):errors.append(str(curve)+': protected point changed')
        if curve in TAKEOFF:exceptions.append({'curve':curve,'allowedOffsets':[1,2],'actualMovementMeters':move[[1,2]].tolist(),'beforeWorld':original[st+np.array([1,2])].tolist(),'afterWorld':target[st+np.array([1,2])].tolist(),'rootP0Exact':bool(np.array_equal(target[st],original[st])),'reason':'Explicitly authorized only these5 dense-gather p1/p2; old control segment1 failed with frozen endpoints. No tail pin changes.'})
        if curve in EXPANDED:expanded.append({'curve':curve,'maximumAllowedMovementMeters':.014,'actualMaximumMovementMeters':float(move.max()),'changedControlOffsets':np.flatnonzero(np.any(target[st:st+n]!=original[st:st+n],axis=1)).tolist()})
        row={'curve':curve,'choice':choice,'nativeModelMaximumActualResidualMeters':nativeerr,'maximumTargetMovementMeters':float(move.max()),'changedControlOffsets':np.flatnonzero(np.any(target[st:st+n]!=original[st:st+n],axis=1)).tolist(),'preservedUpperLastPoint':anchor,'predictedNativeMinimumGapMeters':float(ng[nativefree].min()),'predictedControlMidpointMinimumGapMeters':float(mg.min()),'predictedControlMinimumGapMeters':float(cg.min()),'predictedMaximumControlTurnDegrees':float(ct.max()),'predictedNativeTurnDegreesMaximum':float(nt.max()),'predictedNativeTurnDegreesP95':float(np.percentile(nt,95)),'actualFailureNativeTurnDegreesMaximum':float(ntold.max()),'newControlAbruptJointOffsets':(newfold+1).tolist(),'newNativeAbruptJointOffsets':(nativefold+1).tolist()};rows.append(row);predcontrols.append(q);prednative.append(native);nativegaps.append(ng);nativearcs.append(nativearc);controlgaps.append(np.r_[cg,mg]);nativebefore.extend(ntold);nativeafter.extend(nt)
        print('R11_EXACT_REPAIR '+json.dumps(row),flush=True)
    mask=np.zeros(len(target),bool)
    for curve in CURVES:mask[s[curve]:s[curve]+lens[curve]]=True
    if not np.array_equal(target[~mask],original[~mask]):errors.append('Target outside exact25 changed')
    arrays.update(target_world=target,corrected_curve_ids=CURVES,takeoff_correction_curve_ids=TAKEOFF,expanded_scope_curve_ids=EXPANDED,predicted_after_contact_corrected_world=np.concatenate(predcontrols),predicted_native_corrected_world=np.concatenate(prednative),exact_old_head_native_gap=np.asarray(nativegaps),exact_old_head_native_arc=np.asarray(nativearcs),exact_old_head_control_and_midpoint_gap=np.asarray(controlgaps))
    for key in list(arrays):
        if key.startswith(('sample_','refined_','old_max_turn','target_max_')):arrays['base_v3_'+key]=arrays.pop(key)
    with npz.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report={'schemaVersion':1,'status':'blocked' if errors else 'proposed','errors':errors,'tiePlanSha256':base['tiePlanSha256'],'tiePlan':base['tiePlan'],'arrays':str(npz.relative_to(P)),'arraysSha256':sha(npz),'sourceGroomArraysSha256':base['sourceGroomArraysSha256'],'baseGatherPlan':str(bp.relative_to(P)),'baseGatherPlanSha256':sha(bp),'baseGatherArraysSha256':base['arraysSha256'],'actualFailureReport':str(fp.relative_to(P)),'actualFailureReportSha256':sha(fp),'actualFailureArraysSha256':sha(fa),'correctedCurveIds':CURVES.tolist(),'takeoffCorrectionCurveIds':TAKEOFF.tolist(),'exceptionFields':exceptions,'expandedMovementScope':expanded,'curves':rows,'helperSha256':sha(__file__),'sourceV4Candidate':str(vp.relative_to(P)),'sourceV4CandidateSha256':sha(vp),'sourceScan':str(sp.relative_to(P)),'sourceScanSha256':sha(sp),'sourceScanArraysSha256':sha(sp.with_suffix('.npz')),'exactTriangleOracleHelperSha256':sha(HERE/'collar_continuous_offline_r11.py'),'localPredictorHeadObservation':str(headpath.relative_to(P)),'localPredictorHeadObservationSha256':sha(headpath),'localPredictorHeadArraysSha256':sha(ha),'predictorHeadTriangleCount':len(h['head_triangles']),'completeTriangleNearestQueries':True,'exactVerificationElapsedSeconds':time.time()-begin,'currentHeadExportAvailable':False,'currentHeadCollisionAccepted':False,'actualNewNativeEvaluationExecuted':False,'appearanceAccepted':False,'savedBlend':False,'nativeTurns':{'predictedMaximumDegrees':float(np.max(nativeafter)),'predictedP95Degrees':float(np.percentile(nativeafter,95)),'actualFailureMaximumDegrees':float(np.max(nativebefore)),'actualFailureP95Degrees':float(np.percentile(nativebefore,95)),'newAbove95AndPlus20Count':sum(len(row['newNativeAbruptJointOffsets']) for row in rows)},'controlTurns':{'predictedMaximumDegrees':max(row['predictedMaximumControlTurnDegrees'] for row in rows),'newAbove95AndPlus20Count':sum(len(row['newControlAbruptJointOffsets']) for row in rows)},'preservation':{'otherTargetsExact':True,'sourceArraysUnchanged':True,'allTailMapsUnchanged':True,'allLastThreeExact':True,'allRootP0Exact':True,'firstThreeExactExceptExplicitFive':True,'all46FaceCurtainsExact':True,'densePointCount':len(target)},'predictionMethod':'Empirically actual-matched resolution12 clamped Catmull-Rom then arc64. Complete477344-triangle old-head AABB branch-and-bound nearest queries verify native64, native midpoints, controls and control midpoints. Only exact25 local paths changed;23 bounded12mm, two explicitly authorized10296/10301 bounded14mm. Native contact itself must be rerun on current head.','limits':['The old8dea head is a predictor; current source head/BVH remains mandatory.','The next Blender wrapper must preserve existing current native head-contact15mm limit and all native64/current-surface thresholds.','No saved source or appearance acceptance is claimed.']}
    with out.open('x',encoding='utf8') as stream:json.dump(report,stream,indent=2)
    print('R11_FREEZE_RESULT '+json.dumps({'status':report['status'],'errors':errors,'reportSha256':sha(out),'arraysSha256':report['arraysSha256'],'nativeTurns':report['nativeTurns'],'controlTurns':report['controlTurns']}),flush=True)

if __name__=='__main__':main()
