"""Complete actual v10 stages, preserving all strict v8 dense checks."""
import hashlib,runpy
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
HERE=Path(__file__).resolve().parent
BASE=HERE/'prepare_reference_fringe_selection_v8_r11.py'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!='497b618b9108ba3d705eb9251375f2aa8668005e0b97c8f24b8c691280a3c9b0':raise RuntimeError('Frozen actual dense proof changed')
V8=runpy.run_path(str(BASE));GROUP=V8['GROUP'];integer_attribute=V8['integer_attribute'];dense_length_proof=V8['dense_length_proof'];actual_wisp_proof=V8['actual_wisp_proof']

def actual_dense_proof(arrays):
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    probe=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))
    obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    original=modifier.node_group;memo={};replacements=[];rows=[];clump_checks=[];errors=[]
    stages=(('interpolate','Group.005'),('after-style-bypass','R11 v5 after small clump bypass'),
            ('styled-before-contact','Join Geometry.001'),('reference-front-detail','R11 reference local front detail'),('head-contact','R11 full head clearance'),('reference-eye-length','R11 reference fringe native selection trim'),('final-taper','R11 strand tip taper'))
    try:
        tree=review['copy_render_branch'](original,memo,replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        for label,name in stages:
            node=tree.nodes[name];geometry=next(s for s in node.outputs if s.type=='GEOMETRY')
            for link in list(output.links):tree.links.remove(link)
            tree.links.new(geometry,output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
            rows.append(probe['capture'](obj,label,arrays));groups=integer_attribute(obj,GROUP);arrays[label+'_group']=groups
            if label in ('interpolate','after-style-bypass') and np.any(groups==0):errors.append(label+' has ungrouped curves')
    finally:
        modifier.node_group=original;obj.update_tag();bpy.context.view_layer.update()
    baseline=arrays['interpolate_world'];bypassed=arrays['after-style-bypass_world']
    exact=np.array_equal(baseline,bypassed) and np.array_equal(arrays['interpolate_starts'],arrays['after-style-bypass_starts']) and np.array_equal(arrays['interpolate_lengths'],arrays['after-style-bypass_lengths'])
    group_exact=np.array_equal(arrays['interpolate_group'],arrays['after-style-bypass_group'])
    constraint={'allDensePointsAndTopologyUnchangedAcrossBypassedStyling':bool(exact),'allCurveGroupsUnchangedAcrossBypassedStyling':bool(group_exact)}
    if not exact or not group_exact:errors.append('Legacy styling bypass unexpectedly changed dense positions/topology/groups')
    a=arrays['reference-front-detail_world'];b=arrays['head-contact_world'];c=arrays['final-taper_world']
    trimmed=arrays['reference-eye-length_world']
    if a.shape!=b.shape or not np.array_equal(trimmed,c):errors.append('Head contact changed dense topology or final profile changed trimmed positions')
    local_proof=runpy.run_path(str(HERE/'reference_front_preservation_v10_r11.py'))['own_lock_stage'](arrays)
    errors.extend(local_proof['errors'])
    length_proof=dense_length_proof(arrays)
    errors.extend(length_proof['errors'])
    delta=np.linalg.norm(b-a,axis=1);largest=float(delta.max())
    if largest>.015:errors.append('Dense head-contact correction exceeds 15 mm: '+str(largest))
    radii=arrays['final-taper_radius']
    if float(radii.max())>.000080 or float(radii.min())<0.:errors.append('Final evaluated radius outside candidate range')
    # Independent BVH signed clearance for every final dense point, including
    # the secondary branch. This does not infer geometry safety from node setup.
    donor=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));_,skin,head_summary=donor['_head_trees'](bpy.data.objects['R11 Female head study'])
    signed=np.empty(len(c),np.float32)
    for i,point in enumerate(c):
        hit,normal,index,distance=skin.find_nearest(Vector(point))
        if hit is None:raise RuntimeError('Actual final dense head-clearance lookup failed')
        signed[i]=float((Vector(point)-hit).dot(normal.normalized()))
    arrays['final_signed_head_clearance']=signed
    count=int(np.count_nonzero(signed<-.0005))
    if count:errors.append(str(count)+' final dense points remain more than 0.5 mm inside actual skin')
    # The independently modelled eyeballs are not part of the head BVH.
    # Check every real dense segment, excluding gaps between separate curves.
    starts=arrays['final-taper_starts'];lengths=arrays['final-taper_lengths'];valid=np.ones(len(c)-1,bool);ends=starts+lengths-1;valid[ends[ends<len(c)-1]]=False
    aa=c[:-1][valid];dd=(c[1:]-c[:-1])[valid];den=np.maximum(np.sum(dd*dd,axis=1),1e-15);eye_checks=[]
    for side in ('L','R'):
        eye=bpy.data.objects['R11 Adult sclera '+side];ev=donor['_world'](donor['_coords'](eye),eye.matrix_world);center=(ev.min(0)+ev.max(0))*.5;radius=float((ev.max(0)-ev.min(0)).max())*.5
        u=np.clip(np.sum((center-aa)*dd,axis=1)/den,0.,1.);gap=np.linalg.norm(aa+u[:,None]*dd-center,axis=1)-radius;minimum=float(gap.min())
        eye_checks.append({'eye':eye.name,'segments':len(gap),'minimumEnvelopeGapMeters':minimum,'segmentsBelowOneMillimeter':int(np.count_nonzero(gap<.001))})
        if minimum<.001:errors.append('Final dense fringe enters one-millimetre eye envelope '+side+' '+str(minimum))
    return {'stages':rows,'renderBranchReplacements':replacements,'legacyClumpPassesActive':False,
            'actualBypassChecks':constraint,'headContactDisplacementMetersP50P95P99Max':np.percentile(delta,[50,95,99,100]).tolist(),
            'headContactRootDisplacementMaxMeters':float(delta[arrays['head-contact_starts']].max()),
            'allFinalPointsSignedHeadClearance':{'actualPoints':len(c),'minimumMeters':float(signed.min()),'pointsBelowMinusHalfMillimeter':count,'headSurface':head_summary},
            'allFinalSegmentsEyeEnvelope':eye_checks,
            'actualOwnLockEndShaping':local_proof,'actualFringeLengthCorrection':length_proof,'profilePreservesDensePositionsExact':bool(np.array_equal(trimmed,c)),'rendered':False,'errors':errors,
            'scope':'Actual full dense render-branch positions, interpolation groups, exact styling bypasses, radii and independent final skin clearance; appearance and motion still require actual images/tests.'}
