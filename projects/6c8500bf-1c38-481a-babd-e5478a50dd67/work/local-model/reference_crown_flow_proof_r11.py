"""Exact scoped crown-flow comparison plus actual current surface checks.

No import-time Blender work. compare() is pure NumPy; surfaces() is called by
the serial Blender installer after final native sampling. Passing these checks
does not mean the original-reference appearance has been accepted.
"""
import numpy as np


def _maximum(value):
    return float(np.linalg.norm(value,axis=1).max()) if len(value) else 0.


def _turns(q):
    d=np.diff(q,axis=0);unit=d/np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-14)
    return np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1.,1.)))


def compare(a,plan):
    errors=[];ids=np.asarray(plan['selected_curve_ids'],np.int32)
    starts=np.asarray(plan['starts'],np.int32);lengths=np.asarray(plan['lengths'],np.int32)
    chosen=np.zeros(len(starts),bool);chosen[ids]=True;points=np.asarray(plan['target_point_mask'],bool)
    curve_points=np.repeat(chosen,lengths);connector=np.asarray(plan['connector_protected_point_mask'],bool)
    q=a['before-gather_world'];new=a['after-flow_world'];expected=q.astype(float)+np.asarray(plan['offset_world'],float)
    topology={}
    for label in ('after-gather','after-flow'):
        for key in ('starts','lengths','group'):
            topology[label+'_'+key]=bool(np.array_equal(a[label+'_'+key],a['before-gather_'+key]))
    if not all(topology.values()):raise RuntimeError('Crown field changed pre-contact curve correspondence')
    for key in ('world','starts','lengths','group','radius'):
        if not np.array_equal(a['after-gather_'+key],a['before-gather_'+key]):errors.append('Incoming measured tie stage changed: '+key)
    exact={
        'allUnselectedPointPositions':bool(np.array_equal(new[~points],q[~points])),
        'allUnselectedCurves':bool(np.array_equal(new[~curve_points],q[~curve_points])),
        'allRoots':bool(np.array_equal(new[starts],q[starts])),
        'allTieConnectorPoints':bool(np.array_equal(new[connector],q[connector])),
    }
    end=np.concatenate([np.arange(s+n-3,s+n) for s,n in zip(starts,lengths)])
    exact['allLastThreePoints']=bool(np.array_equal(new[end],q[end]))
    target_error=_maximum(new-expected)
    if target_error>2e-6:errors.append('Stored crown field differs from actual-source plan by more than 2 micrometres')
    if not all(exact.values()):errors.append('Explicit crown point protection failed')
    if not np.array_equal(a['after-flow_radius'],a['before-gather_radius']):errors.append('Crown Set Position changed incoming radii')
    for key in ('starts','lengths','group'):
        if not np.array_equal(a['after-final_'+key],a['before-final_'+key]):raise RuntimeError('Final crown output changed curve correspondence: '+key)
    fs=a['after-final_starts'];fn=a['after-final_lengths'];fq=a['after-final_world'];old=a['before-final_world']
    if np.any(fn[ids]!=lengths[ids]):raise RuntimeError('Selected crown curve changes point count downstream; explicit remapping required')
    final_selection=np.repeat(chosen,fn)
    final_exact={
        'allUnselectedCurvePositions':bool(np.array_equal(fq[~final_selection],old[~final_selection])),
        'allUnselectedCurveRadii':bool(np.array_equal(a['after-final_radius'][~final_selection],a['before-final_radius'][~final_selection])),
        'allRoots':bool(np.array_equal(fq[fs],old[fs])),
    }
    fend=np.concatenate([np.arange(s+n-3,s+n) for s,n in zip(fs,fn)])
    final_exact['allLastThreePointPositions']=bool(np.array_equal(fq[fend],old[fend]))
    # Map by curve/within-curve index, never reuse pre-trim flattened offsets.
    protected=[]
    for i in ids:
        within=np.flatnonzero(connector[starts[i]:starts[i]+lengths[i]])
        protected.extend((fs[i]+within).tolist())
    protected=np.asarray(protected,np.int64)
    final_exact['selectedTieConnectorPositions']=bool(np.array_equal(fq[protected],old[protected]))
    excluded=np.asarray(plan['excluded_curve_ids'],np.int32)
    if len(excluded)!=46 or np.any(chosen[excluded]):errors.append('All 46 verified temple-curtain curves must remain excluded')
    excluded_points=np.concatenate([np.arange(fs[i],fs[i]+fn[i]) for i in excluded])
    final_exact['all46TempleCurtainPositions']=bool(np.array_equal(fq[excluded_points],old[excluded_points]))
    final_exact['all46TempleCurtainRadii']=bool(np.array_equal(a['after-final_radius'][excluded_points],a['before-final_radius'][excluded_points]))
    if not all(final_exact.values()):errors.append('Final dense protected geometry/radius changed outside the crown patch')
    radius=np.asarray(a['after-final_radius'],float)
    if not np.isfinite(radius).all() or radius.min()<0 or radius.max()>.000080:errors.append('Actual final world radius exceeds the retained donor range')
    joint_before=[];joint_after=[];bad=[];max_push=0.
    for i in ids:
        s,n=int(starts[i]),int(lengths[i]);f=int(fs[i])
        before=_turns(old[f:f+n]);after=_turns(fq[f:f+n])
        introduced=np.flatnonzero((after>95.)&(after>before+20.))
        if len(introduced):bad.extend({'curve':int(i),'joint':int(j+1),'beforeDegrees':float(before[j]),'afterDegrees':float(after[j])} for j in introduced)
        joint_before.extend(before.tolist());joint_after.extend(after.tolist())
        max_push=max(max_push,_maximum(fq[f:f+n]-new[s:s+n]))
    if bad:errors.append('Crown field introduced sharp final path foldbacks')
    if max_push>.015:errors.append('Actual head-contact correction exceeds 15 mm')
    semantic=a.get('after-flow_selected_curve_tag')
    tag_ok=semantic is not None and np.array_equal(semantic,chosen)
    if not tag_ok:errors.append('Actual CURVE selection tag differs from explicit measured IDs')
    return {'errors':errors,'selectedCurves':len(ids),'selectedControlPoints':int(points.sum()),'topologyExact':topology,
            'preContactExactProtection':exact,'finalExactProtection':final_exact,'maximumTargetResidualMeters':target_error,
            'actualCurveTagExact':tag_ok,'maximumSelectedHeadContactCorrectionMeters':max_push,
            'selectedFinalJointDegreesBeforeP50P95P99Max':np.percentile(joint_before,[50,95,99,100]).tolist(),
            'selectedFinalJointDegreesAfterP50P95P99Max':np.percentile(joint_after,[50,95,99,100]).tolist(),
            'introducedSharpFoldbacks':bad,'finalRadiusMetersP0P50P95Max':np.percentile(radius,[0,50,95,100]).tolist(),
            'appearanceAccepted':False}


def surfaces(a,ids,observer,proof):
    """Actual selected control/mid/native paths versus current head and eyes."""
    import bpy
    from mathutils.bvhtree import BVHTree
    errors=[];temp={}
    head=observer['collect_mesh'](bpy.data.objects['R11 Female head study'],'crown_current_head',temp,require_actual_evaluation=True)
    q=temp['crown_current_head_world'];tri=temp['crown_current_head_triangles']
    surface={'name':'R11 Female head study','prefix':'head','lo':q.min(0),'hi':q.max(0),'tree':BVHTree.FromPolygons(q.tolist(),tri.tolist(),all_triangles=True)}
    bundle=proof['_bundle'](a,'after-final',world_radius=True)
    control,mid,segments=proof['_samples'](bundle,np.asarray(ids,np.int32))
    rows={
        'selected_control_points':proof['_family']('crown_selected_control_points',control,None,[surface],a,errors,-.0005),
        'selected_control_midpoints':proof['_family']('crown_selected_control_midpoints',mid,segments,[surface],a,errors,-.0005),
    }
    native=proof['_bundle'](a,'native-final',world_radius=True)
    if not np.array_equal(a['native-final_original_curve_ids'],ids):raise RuntimeError('Native final crown paths do not match the selected actual curves')
    pp,mm,ss=proof['_samples'](native,np.arange(len(native['s'])))
    rows['selected_native_points']=proof['_family']('crown_selected_native_points',pp,None,[surface],a,errors,-.0005)
    rows['selected_native_midpoints']=proof['_family']('crown_selected_native_midpoints',mm,ss,[surface],a,errors,-.0005)
    eyes=[]
    for side in ('L','R'):
        obj=bpy.data.objects['R11 Adult sclera '+side]
        eye_arrays={};row=observer['collect_mesh'](obj,'eye',eye_arrays,require_actual_evaluation=True)
        ev=eye_arrays['eye_world'];center=(ev.min(0)+ev.max(0))*.5;radius=float((ev.max(0)-ev.min(0)).max())*.5
        checks=[]
        for name,paths in (('control',segments),('native',ss)):
            aa=paths['a'];dd=paths['b']-aa;t=np.clip(np.sum((center-aa)*dd,axis=1)/np.maximum(np.sum(dd*dd,axis=1),1e-15),0.,1.)
            gap=np.linalg.norm(aa+t[:,None]*dd-center,axis=1)-radius-paths['radius']
            count=int(np.count_nonzero(gap<.001));minimum=float(gap.min())
            checks.append({'path':name,'segments':len(gap),'minimumSurfaceGapMeters':minimum,'segmentsBelowOneMillimeter':count})
            if count:errors.append('Selected '+name+' crown path enters current one-millimetre eye envelope '+side)
        eyes.append({'object':obj.name,'currentEvaluatedMesh':row,'envelopeCenterWorld':center.tolist(),'envelopeRadiusMeters':radius,'checks':checks})
    return {'errors':errors,'currentHead':head,'families':rows,'currentEyeEnvelopes':eyes,
            'rootAttachmentArcMeters':.001,'freePathSurfaceLimitMeters':-.0005,
            'scope':'Every selected actual control/native point and segment midpoint; root arc up to 1 mm reported separately. Native count 128. Current head/eye geometry, bounded real ray probes. Unchanged tails/connectors are verified exact by other checks.',
            'limitations':['Nearest triangle normal signed distance is not a watertight containment proof.','Static geometry and native sampling do not certify appearance or animation.']}
