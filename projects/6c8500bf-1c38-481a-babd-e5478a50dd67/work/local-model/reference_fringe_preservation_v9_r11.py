"""Compare actual full dense curves by exact unique root, without Blender."""
import numpy as np
TARGET_ROLES=(10,13,14,15)

def compare(arrays,before='before-final',after='final-taper'):
    a=arrays[before+'_world'];b=arrays[after+'_world'];sa=arrays[before+'_starts'];sb=arrays[after+'_starts'];na=arrays[before+'_lengths'];nb=arrays[after+'_lengths']
    ga=arrays[before+'_group'];gb=arrays[after+'_group'];ra=arrays[before+'_radius'];rb=arrays[after+'_radius'];errors=[]
    ka=[row.tobytes() for row in a[sa]];kb=[row.tobytes() for row in b[sb]]
    report={'errors':errors,'beforeCurves':len(sa),'afterCurves':len(sb),'beforeUniqueRoots':len(set(ka)),'afterUniqueRoots':len(set(kb))}
    if len(ka)!=len(set(ka)) or len(kb)!=len(set(kb)):
        errors.append('Actual dense roots are not unique; exact correspondence needs separate review');return report
    if set(ka)!=set(kb):
        errors.append('Actual dense root set changed; every root must remain bit-exact')
        report.update(missingOldRoots=len(set(ka)-set(kb)),unexpectedNewRoots=len(set(kb)-set(ka)));return report
    lookup={key:i for i,key in enumerate(kb)};mapping=np.asarray([lookup[key] for key in ka],np.int32)
    protected=[];changed=[];bad_roles=[];bad_radii=[];target_count=0;max_delta=0.
    for i,j in enumerate(mapping):
        old_role=int(ga[i]%256);new_role=int(gb[j]%256)
        if ga[i]//256!=gb[j]//256:bad_roles.append(int(i))
        if old_role in TARGET_ROLES:
            target_count+=1
            if (old_role==13 and new_role not in (18,19)) or (old_role!=13 and new_role!=old_role):bad_roles.append(int(i))
            continue
        protected.append(int(i))
        if ga[i]!=gb[j]:bad_roles.append(int(i))
        old=a[sa[i]:sa[i]+na[i]];new=b[sb[j]:sb[j]+nb[j]]
        if not np.array_equal(old,new):
            changed.append(int(i))
            if old.shape==new.shape:max_delta=max(max_delta,float(np.linalg.norm(old-new,axis=1).max()))
        if not np.array_equal(ra[sa[i]:sa[i]+na[i]],rb[sb[j]:sb[j]+nb[j]]):bad_radii.append(int(i))
    if changed:errors.append(str(len(changed))+' non-target dense curves changed positions or point counts')
    if bad_roles:errors.append(str(len(set(bad_roles)))+' dense curves changed protected role or native island')
    if bad_radii:errors.append(str(len(bad_radii))+' non-target dense curves changed radius')
    arrays['before_to_after_curve_index']=mapping
    report.update(allActualRootsPreservedExact=True,curveOrderPreservedExact=bool(np.array_equal(mapping,np.arange(len(mapping)))),
        targetCurves=target_count,protectedCurves=len(protected),protectedPositionsAndCountsPreservedExact=not changed,
        protectedRadiiPreservedExact=not bad_radii,protectedRoleAndIslandPreservedExact=not bad_roles,
        changedProtectedCurveIndices=changed,wrongRoleOrIslandCurveIndices=sorted(set(bad_roles)),changedProtectedRadiusCurveIndices=bad_radii,
        protectedPointMovementMaximumMeters=max_delta,scope='Every actual dense curve in the render branch, matched by an exact unique world-space root; only raw roles 10,13,14,15 may change shape.')
    return report
