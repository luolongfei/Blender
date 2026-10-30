"""Validate the measured reference fringe layer length and save an independent v7 groom."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
EXPECTED_HELPER='967e94622da587d4eadba6fe5c5f5bc485cad372d416c980a6c7f19433bc2423'
GROUP='R11 groom interpolation group'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def integer_attribute(obj,name):
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());attr=evaluated.data.attributes.get(name)
    if attr is None or attr.data_type!='INT' or attr.domain!='CURVE':raise RuntimeError('Missing evaluated curve attribute '+name)
    value=np.empty(len(attr.data),np.int32);attr.data.foreach_get('value',value);return value


def actual_dense_proof(arrays):
    review=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    probe=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))
    obj=bpy.data.objects['R11 Donor / long hair main'];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    original=modifier.node_group;memo={};replacements=[];rows=[];clump_checks=[];errors=[]
    stages=(('interpolate','Group.005'),('after-style-bypass','R11 v5 after small clump bypass'),
            ('styled-before-contact','Join Geometry.001'),('head-contact','R11 full head clearance'),('reference-eye-length','R11 reference fringe eye length'),('final-taper','R11 strand tip taper'))
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
    a=arrays['styled-before-contact_world'];b=arrays['head-contact_world'];c=arrays['final-taper_world']
    trimmed=arrays['reference-eye-length_world']
    if a.shape!=b.shape or not np.array_equal(trimmed,c):errors.append('Head contact changed dense topology or final profile changed trimmed positions')
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
            'actualFringeLengthCorrection':length_proof,'profilePreservesDensePositionsExact':bool(np.array_equal(trimmed,c)),'rendered':False,'errors':errors,
            'scope':'Actual full dense render-branch positions, interpolation groups, exact styling bypasses, radii and independent final skin clearance; appearance and motion still require actual images/tests.'}


def dense_length_proof(arrays):
    a=arrays['head-contact_world'];b=arrays['reference-eye-length_world'];starts=arrays['head-contact_starts'];lengths=arrays['head-contact_lengths'];groups=arrays['head-contact_group'];errors=[]
    topology=a.shape==b.shape and np.array_equal(starts,arrays['reference-eye-length_starts']) and np.array_equal(lengths,arrays['reference-eye-length_lengths'])
    if not topology:return {'errors':['Native reference trim unexpectedly changed dense point topology']}
    active=(groups%256==12)&(a[starts,1]>-.012);point_active=np.repeat(active,lengths);movement=np.linalg.norm(b-a,axis=1)
    untouched=float(movement[~point_active].max());root_move=float(movement[starts].max())
    if untouched>0.000001:errors.append('Native length correction moved another lock or an unselected front-root curve '+str(untouched))
    if root_move>0.000001:errors.append('Native length correction moved a dense root '+str(root_move))
    arc_deltas=[]
    for i,(start,length) in enumerate(zip(starts,lengths)):
        if active[i]:arc_deltas.append(float(np.linalg.norm(np.diff(a[start:start+length],axis=0),axis=1).sum()-np.linalg.norm(np.diff(b[start:start+length],axis=0),axis=1).sum()))
    if min(arc_deltas)<-.0005 or max(arc_deltas)>.012:errors.append('Measured fringe trim is outside the reviewed local 0-12 mm arc-length window')
    return {'errors':errors,'activeLayerCurves':int(active.sum()),'allCurves':len(starts),'unselectedPointMovementMaxMeters':untouched,'allRootMovementMaxMeters':root_move,'arcLengthRemovedMetersP0P50P100':np.percentile(arc_deltas,[0,50,100]).tolist(),'denseGroupsPreservedExact':bool(np.array_equal(groups,arrays['reference-eye-length_group']))}


def actual_wisp_proof(arrays):
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));_,skin,_=fit['_head_trees'](bpy.data.objects['R11 Female head study'])
    obj=bpy.data.objects['R11 Donor / reference face wisps'];row=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture'](obj,'reference-face-wisps',arrays)
    signed=[]
    for p in arrays['reference-face-wisps_world']:
        hit,n,_,distance=skin.find_nearest(Vector(p));signed.append(float((Vector(p)-hit).dot(n)))
    if min(signed)<-.0005:raise RuntimeError('Actual fine face wisps penetrate current head')
    return {'actualGeometry':row,'minimumHeadClearanceMeters':min(signed),'allPointsChecked':len(signed)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve()
    report_path=out.with_suffix('.groom-fringe-length.json');array_path=out.with_suffix('.groom-fringe-length.npz')
    for path in (source,out,report_path,array_path):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or any(p.exists() for p in (out,report_path,array_path)):raise RuntimeError('Fresh independent source and unused outputs required')
    module=HERE/'reference_fringe_length_v7_r11.py'
    if sha(module)!=EXPECTED_HELPER:raise RuntimeError('Review changed v7 helper')
    contract=runpy.run_path(str(HERE/'prepare_adult_tail_shape_r11.py'))['geometry_contract']
    names=('R11 Female head study','R01 Head','R11 Donor / long hair strands',
           'R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
    before={name:contract(bpy.data.objects[name]) for name in names}
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));scalp=bpy.data.objects['R11 Donor / long hair growth mesh'];scalp_before=hashlib.sha256(fit['_coords'](scalp).tobytes()).hexdigest()
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;arrays={};report={'source':a.source,'sourceSha256':sha(source),'savedBlend':False}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        print('R11_REFERENCE_V7_REBUILDING_FRONT',flush=True)
        report.update(runpy.run_path(str(module))['apply'](expected_source_relative=a.source))
        print('R11_REFERENCE_V7_MEASURING_ACTUAL_DENSE_OUTPUT',flush=True)
        proof=actual_dense_proof(arrays);report['actualDenseOutput']=proof
        report['fineWispsActualProof']=actual_wisp_proof(arrays)
        print('R11_REFERENCE_V7_ACTUAL_DENSE_CHECKED '+json.dumps({'curves':proof['stages'][-1]['curves'],'errors':proof['errors']}),flush=True)
        if proof['errors']:raise RuntimeError('; '.join(proof['errors']))
    except Exception as exc:
        report['error']=str(exc)
        if arrays:
            with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
            report['actualDenseArrays']=str(array_path.relative_to(P));report['actualDenseArraysSha256']=sha(array_path)
        with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    if before!={name:contract(bpy.data.objects[name]) for name in names}:raise RuntimeError('Protected head/secondary/growth/tail contracts changed')
    if hashlib.sha256(fit['_coords'](scalp).tobytes()).hexdigest()!=scalp_before:raise RuntimeError('Density update changed scalp vertex positions')
    report['scalpCoordinatesPreservedExact']=True
    report['protectedGeometryContracts']=before
    report['faceFibreBinding']=runpy.run_path(str(HERE/'face_fibre_binding_r11.py'))['validate'](neutral=False)
    bpy.context.scene['R11_reference_front_groom_v7']=json.dumps(report,ensure_ascii=False)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Candidate save failed')
    with array_path.open('xb') as f:np.savez_compressed(f,**arrays)
    report.update(savedBlend=True,output=a.output,outputSha256=sha(out),wrapperSha256=sha(__file__),
                  actualDenseArrays=str(array_path.relative_to(P)),actualDenseArraysSha256=sha(array_path))
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_REFERENCE_FRONT_GROOM_V7_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)

if __name__=='__main__':main()
