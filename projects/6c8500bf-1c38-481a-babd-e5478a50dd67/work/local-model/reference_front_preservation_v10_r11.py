"""Exact dense preservation outside two central locks and the measured side patch."""
import numpy as np
SIDE_ROOT_BOUNDS=np.array([[-.080,-.020,1.435],[-.058,.055,1.490]])

def compare(arrays,before='before-final',after='final-taper'):
    a=arrays[before+'_world'];b=arrays[after+'_world'];sa=arrays[before+'_starts'];sb=arrays[after+'_starts'];na=arrays[before+'_lengths'];nb=arrays[after+'_lengths'];ga=arrays[before+'_group'];gb=arrays[after+'_group'];ra=arrays[before+'_radius'];rb=arrays[after+'_radius'];errors=[]
    ka=[p.tobytes() for p in a[sa]];kb=[p.tobytes() for p in b[sb]]
    result={'errors':errors,'beforeCurves':len(sa),'afterCurves':len(sb),'beforeUniqueRoots':len(set(ka)),'afterUniqueRoots':len(set(kb))}
    if len(ka)!=len(set(ka)) or len(kb)!=len(set(kb)) or set(ka)!=set(kb):
        errors.append('Actual dense unique root set changed; exact roots/correspondence required');return result
    lookup={k:i for i,k in enumerate(kb)};mapping=np.asarray([lookup[k] for k in ka],np.int32);roots=a[sa]
    side=(ga%256==41)&np.all(roots>=SIDE_ROOT_BOUNDS[0],axis=1)&np.all(roots<=SIDE_ROOT_BOUNDS[1],axis=1)
    central=np.isin(ga%256,(18,19));selected=central|side;changed=[];bad_radius=[];bad_groups=[];bad_gather_ends=[];changed_side=[];max_delta=0.
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
    if changed:errors.append(str(len(changed))+' dense curves outside the reviewed central/side patch changed')
    if bad_radius:errors.append(str(len(bad_radius))+' non-target dense curve radii changed')
    if bad_groups:errors.append('Dense interpolation role/native-island identity changed')
    if bad_gather_ends:errors.append(str(len(bad_gather_ends))+' dense gather endpoints changed; current tie connections must stay exact')
    arrays['before_to_after_curve_index']=mapping
    old_side_points=np.repeat(ga%256==41,na);new_side_points=np.repeat(gb%256==41,nb)
    old_spike=old_side_points&(a[:,0]<-.074)&(a[:,2]>1.470)&(a[:,2]<1.495)
    new_spike=new_side_points&(b[:,0]<-.074)&(b[:,2]>1.470)&(b[:,2]<1.495)
    result.update(allActualRootsPreservedExact=True,allGroupsAndIslandsPreservedExact=not bad_groups,allGatherEndpointsPreservedExact=not bad_gather_ends,
        centralCurves=int(central.sum()),allowedSidePatchCurves=int(side.sum()),actuallyChangedSidePatchCurves=len(changed_side),protectedCurves=int((~selected).sum()),
        sideRootWorldBounds=SIDE_ROOT_BOUNDS.tolist(),protectedPositionsAndCountsPreservedExact=not changed,protectedRadiiPreservedExact=not bad_radius,
        changedProtectedCurveIndices=changed,changedProtectedRadiusCurveIndices=bad_radius,wrongGroupCurveIndices=bad_groups,changedGatherEndpointCurveIndices=bad_gather_ends,
        observedLeftUpwardOutlineBand={'worldXBelow':-.074,'worldZBetween':[1.470,1.495],'beforePoints':int(old_spike.sum()),'afterPoints':int(new_spike.sum()),'appearanceAccepted':False},
        protectedPointMovementMaximumMeters=max_delta,scope='Only roles 18/19 and left gather roots in the fixed measured low-side patch may change; every other actual curve and every gather endpoint remains exact.')
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
