"""Pure correspondence and junction checks for the measured gather rebuild."""
import numpy as np


def _turns(q):
    d=np.diff(q,axis=0);n=np.linalg.norm(d,axis=1)
    v=d/np.maximum(n[:,None],1e-15)
    return np.degrees(np.arccos(np.clip(np.sum(v[:-1]*v[1:],axis=1),-1.,1.)))


def compare(arrays,targets):
    errors=[]
    a=np.asarray(arrays['before-final_world']);b=np.asarray(arrays['after-final_world'])
    starts=arrays['before-final_starts'];lengths=arrays['before-final_lengths'];groups=arrays['before-final_group']
    for key in ('starts','lengths','group'):
        if not np.array_equal(arrays['before-final_'+key],arrays['after-final_'+key]):errors.append('Final dense topology/ordered identities changed: '+key)
        if not np.array_equal(arrays['before-front_'+key],arrays['after-front_'+key]):errors.append('Original upstream branch topology/identities changed: '+key)
        if not np.array_equal(arrays['after-front_'+key],arrays['after-transport_'+key]):errors.append('Reconnection stage topology/identities changed: '+key)
    if errors or a.shape!=b.shape:return {'errors':errors+(['Final correspondence shape changed'] if a.shape!=b.shape else [])}
    ids=np.asarray(targets['gather_curve_ids'],np.int32)
    curtain=np.asarray(targets['preserved_face_curtain_curve_ids'],np.int32)
    tail_ids=np.asarray(targets['tail_curve_index'],np.int32)
    if ids.ndim!=1 or tail_ids.shape!=ids.shape or len(np.unique(ids))!=len(ids):raise RuntimeError('Explicit gather/tail mapping has invalid dimensions')
    if np.any(ids<0) or np.any(ids>=len(groups)) or not np.all(np.isin(groups[ids]%256,(40,41))):raise RuntimeError('Explicit gathered-curve identities are invalid')
    if np.any(tail_ids<0) or np.any(tail_ids>=2480):raise RuntimeError('Actual tail-root map is out of bounds')
    if np.intersect1d(ids,curtain).size:raise RuntimeError('Verified face curtain was selected for tail reconnection')
    selected=np.zeros(len(groups),bool);selected[ids]=True;mask=np.repeat(selected,lengths)
    unchanged=np.array_equal(a[~mask],b[~mask])
    radius_unchanged=np.array_equal(arrays['before-final_radius'][~mask],arrays['after-final_radius'][~mask])
    if not unchanged:errors.append('An unselected final curve position changed, including protected temple curtains')
    if not radius_unchanged:errors.append('An unselected final curve radius changed')
    if not np.array_equal(arrays['before-front_world'],arrays['after-front_world']):errors.append('Existing original/front style was modified')
    stage_starts=arrays['after-front_starts'];stage_lengths=arrays['after-front_lengths']
    stage_selected=np.repeat(selected,stage_lengths);stage_before=arrays['after-front_world'];stage_after=arrays['after-transport_world']
    if not np.array_equal(stage_before[~stage_selected],stage_after[~stage_selected]):errors.append('Native measured stage changed an unselected point')
    target=np.asarray(arrays['gather_target_effective_world'])
    stage_error=np.linalg.norm(stage_after-target,axis=1)
    if float(stage_error.max())>2e-6:errors.append('Actual native sampled positions differ from measured target by over 2 micrometres')
    roots=np.concatenate([np.arange(starts[i],starts[i]+3) for i in ids])
    pre_roots=np.concatenate([np.arange(stage_starts[i],stage_starts[i]+3) for i in ids])
    roots_exact=np.array_equal(a[roots],b[roots])
    if not roots_exact:errors.append('Actual final first-three root/takeoff points changed')
    if not np.array_equal(stage_before[pre_roots],stage_after[pre_roots]):errors.append('Native position sampling changed a protected root/takeoff')
    rows=[];folds=[];all_old_turn=[];all_new_turn=[]
    for side,role in (('positive',40),('negative',41)):
        selected_rows=np.flatnonzero(groups[ids]%256==role);tail=arrays['tail_'+side+'_after_world'].reshape(2480,35,3)
        old_tail=arrays['tail_'+side+'_world'].reshape(2480,35,3)
        distances=[];old_distances=[];angles=[];last3_residual=[]
        for row in selected_rows:
            curve=int(ids[row]);tail_curve=int(tail_ids[row]);s=int(starts[curve]);n=int(lengths[curve]);end=s+n-1
            current=b[s:s+n];previous=a[s:s+n]
            distances.append(float(np.linalg.norm(current[-1]-tail[tail_curve,0])))
            old_distances.append(float(np.linalg.norm(previous[-1]-old_tail[tail_curve,0])))
            incoming=current[-1]-current[-2];outgoing=tail[tail_curve,1]-tail[tail_curve,0]
            den=np.linalg.norm(incoming)*np.linalg.norm(outgoing)
            angles.append(float(np.degrees(np.arccos(np.clip(np.dot(incoming,outgoing)/max(den,1e-15),-1.,1.)))))
            ps=int(stage_starts[curve]);pn=int(stage_lengths[curve]);desired=target[ps+pn-3:ps+pn]
            last3_residual.append(float(np.linalg.norm(current[-3:]-desired,axis=1).max()))
            oldturn=_turns(previous);newturn=_turns(current)
            all_old_turn.append(float(oldturn.max()));all_new_turn.append(float(newturn.max()))
            invalid=np.flatnonzero((newturn>95.)&(newturn>oldturn+20.))
            if len(invalid):folds.append({'curve':curve,'jointOffsets':(invalid+1).tolist(),'newDegrees':newturn[invalid].tolist(),'beforeDegrees':oldturn[invalid].tolist()})
        if max(distances,default=0.)>3e-6:errors.append(side+' gathered terminals do not meet their actual moved tail roots')
        if max(angles,default=0.)>3.:errors.append(side+' gathered terminal tangents differ from actual tail takeoff by over 3 degrees')
        if max(last3_residual,default=0.)>3e-6:errors.append(side+' current head contact moved a new terminal connection beyond 3 micrometres')
        rows.append({'side':side,'gatherCurves':len(selected_rows),
                     'beforeMappedTailRootDistanceMetersP0P50P95P100':np.percentile(old_distances,[0,50,95,100]).tolist(),
                     'afterMappedTailRootDistanceMetersP0P50P95P100':np.percentile(distances,[0,50,95,100]).tolist(),
                     'terminalToTailTangentDegreesP0P50P95P100':np.percentile(angles,[0,50,95,100]).tolist(),
                     'maximumFinalLastThreeTargetResidualMeters':max(last3_residual,default=0.)})
    if folds:errors.append(str(len(folds))+' reconstructed curves introduce a >95-degree turn increased by >20 degrees')
    contact=np.linalg.norm(arrays['after-contact_world']-stage_after,axis=1)
    if float(contact.max())>.015:errors.append('New head-contact correction exceeds existing 15 mm gate')
    radii=arrays['after-final_radius']
    if not np.isfinite(radii).all() or float(radii.min())<0. or float(radii.max())>.00008:errors.append('Actual final donor radii invalid')
    expected_tag=np.zeros(len(groups),bool);expected_tag[curtain]=True
    expected_points=np.repeat(expected_tag,lengths)
    actual_tag=arrays.get('after-final_verified_temple_curtain')
    if actual_tag is None or not np.array_equal(actual_tag,expected_points):errors.append('Explicit verified temple-curtain semantic attribute missing or incorrect')
    return {'errors':errors,'actualCurves':len(groups),'reconnectedCurves':len(ids),'protectedCurves':int((~selected).sum()),
            'protectedFaceCurtainCurves':curtain.tolist(),'allUnselectedPositionsExact':bool(unchanged),
            'allUnselectedRadiiExact':bool(radius_unchanged),'firstThreeFinalRootPointsExact':bool(roots_exact),
            'maximumActualTargetPositionResidualMeters':float(stage_error.max()),'newTailRootJunctions':rows,
            'controlTurnMaximumPerCurveDegreesBeforeP50P95P99Max':np.percentile(all_old_turn,[50,95,99,100]).tolist(),
            'controlTurnMaximumPerCurveDegreesAfterP50P95P99Max':np.percentile(all_new_turn,[50,95,99,100]).tolist(),
            'introducedFoldExamples':folds[:64],'introducedFoldCount':len(folds),
            'headCorrectionAgainstRebuiltStageMetersP50P95P99Max':np.percentile(contact,[50,95,99,100]).tolist(),
            'semantics':'Verified guide-13 temple-curtain curves are entirely protected despite their historical role-41 label. Only explicitly identified true gathers reconnect to mapped actual tail roots.'}
