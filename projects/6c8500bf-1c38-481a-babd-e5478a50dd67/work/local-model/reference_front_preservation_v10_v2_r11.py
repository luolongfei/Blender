"""Validate the actual two-guide interpolation influence instead of a box guess."""
import numpy as np
SIDE_ROOT_BOUNDS=np.array([[-.080,-.020,1.435],[-.058,.055,1.490]])
RESIDUAL_LIMIT=1e-6
WEIGHT_EPSILON=5e-5

def interpolation_influence(arrays):
    a=arrays['before-interpolate_world'];b=arrays['interpolate_world'];sa=arrays['before-interpolate_starts'];sb=arrays['interpolate_starts'];na=arrays['before-interpolate_lengths'];nb=arrays['interpolate_lengths'];ga=arrays['before-interpolate_group'];gb=arrays['interpolate_group'];errors=[]
    result={'errors':errors,'basisRawGuides':[88,94],'maximumFitResidualAllowedMeters':RESIDUAL_LIMIT,'weightRoundoffTolerance':WEIGHT_EPSILON}
    eligible=np.zeros(len(sa),bool);weights=np.zeros((len(sa),2),np.float64);residual=np.zeros(len(sa),np.float64);copies=[]
    if not np.array_equal(sa,sb) or not np.array_equal(na,nb) or not np.array_equal(ga,gb):
        errors.append('Two-guide edit changed ordered interpolation topology/groups');return result,eligible
    if not np.array_equal(a[sa],b[sb]):errors.append('Two-guide edit moved distributed interpolation roots')
    basis=np.asarray(arrays['actual_raw_gather_displacement_basis'],np.float64)
    if basis.shape!=(2,20,3):errors.append('Missing actual two-guide displacement basis');return result,eligible
    raw_before=np.asarray(arrays['actual_raw_gather_before'],np.float64);raw_after=np.asarray(arrays['actual_raw_gather_after'],np.float64)
    if raw_before.shape!=(2,20,3) or raw_after.shape!=(2,20,3):errors.append('Missing actual raw guide paths for direct source-copy verification');return result,eligible
    def resample(path,count):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        u=np.linspace(0.,arc[-1],count)
        return np.column_stack([np.interp(u,arc,path[:,k]) for k in range(3)])
    matrix=basis.reshape(2,-1).T;changed=[];bad=[]
    for i,(s,n) in enumerate(zip(sa,na)):
        delta=b[s:s+n].astype(float)-a[s:s+n].astype(float)
        if np.array_equal(a[s:s+n],b[s:s+n]):continue
        changed.append(int(i))
        if ga[i]%256!=41 or n!=20:
            bad.append(int(i));continue
        # The author interpolation joins its two changed source guides after
        # an explicit uniform-distance resample. Those two direct copies have
        # reparameterized lower samples; verify their complete before/after
        # paths against the actual raw data instead of a delta blend.
        direct=np.flatnonzero(np.linalg.norm(raw_before[:,0]-a[s],axis=1)<RESIDUAL_LIMIT)
        if len(direct)==1:
            k=int(direct[0]);pre_error=float(np.linalg.norm(resample(raw_before[k],int(n))-a[s:s+n],axis=1).max());post_error=float(np.linalg.norm(resample(raw_after[k],int(n))-b[s:s+n],axis=1).max())
            maximum=max(pre_error,post_error);residual[i]=maximum
            if maximum<=RESIDUAL_LIMIT:
                eligible[i]=True;weights[i,k]=1.;copies.append({'interpolationCurve':int(i),'rawGuide':(88,94)[k],'beforeResidualMeters':pre_error,'afterResidualMeters':post_error});continue
            bad.append(int(i));continue
        coefficients=np.linalg.lstsq(matrix,delta.ravel(),rcond=None)[0]
        error=delta-(matrix@coefficients).reshape(20,3)
        maximum=float(np.linalg.norm(error,axis=1).max());weights[i]=coefficients;residual[i]=maximum
        valid=(maximum<=RESIDUAL_LIMIT and coefficients.min()>=-WEIGHT_EPSILON and coefficients.sum()<=1.+WEIGHT_EPSILON)
        if valid:eligible[i]=True
        else:bad.append(int(i))
    if bad:errors.append(str(len(bad))+' interpolation curves cannot be explained by a nonnegative blend of only the two measured guide edits')
    arrays['gather_interpolation_affected']=eligible;arrays['gather_interpolation_weights']=weights;arrays['gather_interpolation_fit_residual']=residual
    result.update(changedInterpolationCurves=len(changed),verifiedInfluenceCurves=int(eligible.sum()),unexplainedCurveIndices=bad,
        maximumActualFitResidualMeters=float(residual.max()),verifiedDirectResampledGuideCopies=copies,allOtherInterpolationPositionsExact=not bad,
        affectedWeightRange=[weights[eligible].min(0).tolist(),weights[eligible].max(0).tolist()] if eligible.any() else [],
        scope='Current input and output interpolation captured in the same REST scene; generated curves must equal nonnegative blends of actual guide88/94 displacements; the two direct source copies must match complete uniform-distance resampling of the actual before/after guides, all within 1 micrometre.')
    return result,eligible

def compare(arrays,before='before-final',after='final-taper'):
    influence,eligible=interpolation_influence(arrays);errors=list(influence['errors'])
    a=arrays[before+'_world'];b=arrays[after+'_world'];sa=arrays[before+'_starts'];sb=arrays[after+'_starts'];na=arrays[before+'_lengths'];nb=arrays[after+'_lengths'];ga=arrays[before+'_group'];gb=arrays[after+'_group'];ra=arrays[before+'_radius'];rb=arrays[after+'_radius']
    ka=[p.tobytes() for p in a[sa]];kb=[p.tobytes() for p in b[sb]]
    result={'errors':errors,'beforeCurves':len(sa),'afterCurves':len(sb),'actualTwoGuideInterpolationInfluence':influence}
    if len(ka)!=len(set(ka)) or len(kb)!=len(set(kb)) or set(ka)!=set(kb):
        errors.append('Actual dense unique root set changed; exact roots/correspondence required');return result
    roots=arrays['interpolate_world'][arrays['interpolate_starts']];root_lookup={p.tobytes():i for i,p in enumerate(roots)}
    styled_roots=arrays['styled-before-contact_world'][arrays['styled-before-contact_starts']]
    if len(root_lookup)!=len(roots) or any(p.tobytes() not in root_lookup for p in styled_roots):
        errors.append('Cannot match each pre-contact dense curve to an exact unique interpolation root');return result
    interpolation_map=np.asarray([root_lookup[p.tobytes()] for p in styled_roots],np.int32)
    lookup={k:i for i,k in enumerate(kb)};mapping=np.asarray([lookup[k] for k in ka],np.int32)
    side=eligible[interpolation_map[mapping]];central=np.isin(ga%256,(18,19));selected=central|side
    changed=[];bad_radius=[];bad_groups=[];bad_gather_ends=[];changed_side=[];max_delta=0.
    for i,j in enumerate(mapping):
        old=a[sa[i]:sa[i]+na[i]];new=b[sb[j]:sb[j]+nb[j]]
        if ga[i]!=gb[j]:bad_groups.append(int(i))
        if ga[i]%256 in (40,41) and not np.array_equal(old[-1],new[-1]):bad_gather_ends.append(int(i))
        if side[i] and not np.array_equal(old,new):changed_side.append(int(i))
        if selected[i]:continue
        if not np.array_equal(old,new):
            changed.append(int(i))
            if old.shape==new.shape:max_delta=max(max_delta,float(np.linalg.norm(new-old,axis=1).max()))
        if not np.array_equal(ra[sa[i]:sa[i]+na[i]],rb[sb[j]:sb[j]+nb[j]]):bad_radius.append(int(i))
    if changed:errors.append(str(len(changed))+' dense curves outside the two central locks and verified two-guide influence changed')
    if bad_radius:errors.append(str(len(bad_radius))+' non-influenced dense curve radii changed')
    if bad_groups:errors.append('Dense interpolation role/native-island identity changed')
    if bad_gather_ends:errors.append(str(len(bad_gather_ends))+' dense gather endpoints changed')
    arrays['before_to_after_curve_index']=mapping;arrays['final_to_interpolation_curve_index']=interpolation_map
    old_side_points=np.repeat(ga%256==41,na);new_side_points=np.repeat(gb%256==41,nb)
    old_spike=old_side_points&(a[:,0]<-.074)&(a[:,2]>1.470)&(a[:,2]<1.495)
    new_spike=new_side_points&(b[:,0]<-.074)&(b[:,2]>1.470)&(b[:,2]<1.495)
    old_box=np.all(a[sa]>=SIDE_ROOT_BOUNDS[0],axis=1)&np.all(a[sa]<=SIDE_ROOT_BOUNDS[1],axis=1)
    result.update(allActualRootsPreservedExact=True,allGroupsAndIslandsPreservedExact=not bad_groups,allGatherEndpointsPreservedExact=not bad_gather_ends,
        centralCurves=int(central.sum()),verifiedSideInfluenceCurves=int(side.sum()),actuallyChangedSideCurves=len(changed_side),protectedCurves=int((~selected).sum()),
        verifiedInfluencedCurvesOutsideEarlierBox=int((side&~old_box).sum()),earlierBoxOnlyDiagnostic=SIDE_ROOT_BOUNDS.tolist(),
        protectedPositionsAndCountsPreservedExact=not changed,protectedRadiiPreservedExact=not bad_radius,
        changedProtectedCurveIndices=changed,changedProtectedRadiusCurveIndices=bad_radius,wrongGroupCurveIndices=bad_groups,changedGatherEndpointCurveIndices=bad_gather_ends,
        observedLeftUpwardOutlineBand={'worldXBelow':-.074,'worldZBetween':[1.470,1.495],'beforePoints':int(old_spike.sum()),'afterPoints':int(new_spike.sum()),'appearanceAccepted':False},
        protectedPointMovementMaximumMeters=max_delta,scope='Only central18/19 and mathematically verified actual interpolation influence of the two measured left guides may change. Every root/group/gather endpoint and every non-influenced dense position/radius remains exact.')
    return result

def own_lock_stage(arrays):
    a=arrays['styled-before-contact_world'];b=arrays['reference-front-detail_world'];sa=arrays['styled-before-contact_starts'];sb=arrays['reference-front-detail_starts'];na=arrays['styled-before-contact_lengths'];nb=arrays['reference-front-detail_lengths'];ga=arrays['styled-before-contact_group'];gb=arrays['reference-front-detail_group'];errors=[]
    if not np.array_equal(sa,sb) or not np.array_equal(na,nb) or not np.array_equal(ga,gb):return {'errors':['Own-lock end shaping changed topology or groups']}
    selected=np.isin(ga%256,(18,19));point_selected=np.repeat(selected,na)
    protected_exact=np.array_equal(a[~point_selected],b[~point_selected]);roots_exact=np.array_equal(a[sa],b[sb]);delta=np.linalg.norm(b-a,axis=1)
    if not protected_exact:errors.append('Own-lock end shaping changed another lock')
    if not roots_exact:errors.append('Own-lock end shaping changed a root')
    if float(delta.max())>.025:errors.append('Own-lock end shaping exceeds the reviewed 25 mm local displacement envelope')
    return {'errors':errors,'selectedCurves':int(selected.sum()),'allOtherPositionsExact':protected_exact,'allRootsExact':roots_exact,'maximumDisplacementMeters':float(delta.max()),'noCrossLockMixing':True}
