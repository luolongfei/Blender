"""Read-only exact witness mapping from the actual failed tie-v2 arrays."""
import hashlib,json,runpy
from pathlib import Path
import numpy as np

def main():
    root=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());p=root/'projects/6c8500bf-1c38-481a-babd-e5478a50dd67';w=p/'work/reconstruction-r02';out=w/'reference-tie-v2-native-failure-analysis-r11.json'
    if out.exists():raise RuntimeError('New evidence output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    report_path=w/'reference-adult-reference-ties-v2-r11.tie-transport.json';record=json.loads(report_path.read_text());data_path=w/'reference-adult-reference-ties-v2-r11.tie-transport.npz'
    if sha(data_path)!=record['arraysSha256']:raise RuntimeError('Actual failed payload hash mismatch')
    a=np.load(data_path);v=np.load(w/'reference-gather-rebuild-v3-r11.npz');head=np.load(w/'reference-tie-alignment-r11-v2.npz')
    starts=a['after-final_starts'];lengths=a['after-final_lengths'];after=a['after-final_world'].astype(float);before=a['before-final_world'].astype(float);transport=a['after-transport_world'].astype(float);contact=a['after-contact_world'].astype(float);stage_starts=a['after-transport_starts'];stage_lengths=a['after-transport_lengths'];groups=a['after-final_group'];mapping=a['tieproof_native_to_main_curve'];native=a['gather_native_after_world'].astype(float)
    boundary=dict(zip(v['gather_curve_ids'].tolist(),v['preserved_upper_last_point'].tolist()))
    Surface=runpy.run_path(str(Path(__file__).with_name('collar_continuous_offline_r11.py')))['Surface'];surface=Surface(head['head_world'],head['head_triangles'])
    allrows=[];families=[]
    for label in ('gather_control_midpoints','gather_native_points','gather_native_midpoints'):
        prefix='tieproof_'+label;tested=a[prefix+'_against_head_sample_index'];signed=a[prefix+'_against_head_signed_surface'];signed_center=a[prefix+'_against_head_signed_center'];arc=a[prefix+'_arc'];curve=a[prefix+'_curve'];within=a[prefix+'_within_curve'];source=a[prefix+'_source_point_or_vertex']
        bad=np.flatnonzero((signed<-.0005)&(arc[tested]>.001));rows=[]
        for k in bad:
            j=int(tested[k]);c=int(curve[j]);is_native='native' in label;c=int(mapping[c]) if is_native else c;offset=int(within[j]);index=int(source[j]);s=int(starts[c]);n=int(lengths[c]);q=after[s:s+n]
            point=native[index] if label.endswith('native_points') else (native[index]+native[index+1])*.5 if is_native else (after[index]+after[index+1])*.5
            d=np.diff(q,axis=0);t=np.clip(np.sum((point-q[:-1])*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-20),0.,1.);near=q[:-1]+t[:,None]*d;dist=np.linalg.norm(near-point,axis=1);segment=int(dist.argmin())
            if not is_native:segment=offset
            control_ids=np.arange(max(0,segment-1),min(n,segment+3));ps=int(stage_starts[c]);pn=int(stage_lengths[c]);contact_delta=contact[ps:ps+pn]-transport[ps:ps+pn]
            oldhit,oldnormal,_,_=surface.find_nearest(point);oldestimate=float(np.dot(point-oldhit,oldnormal))
            # Fit the actual native witness to a uniform Catmull interval to
            # identify the local influence stencil. This is labelled an
            # estimate and its residual is retained, not assumed exact.
            fit=[]
            for seg in range(max(0,segment-1),min(n-1,segment+2)):
                pa,pb,pc,pd=q[max(seg-1,0)],q[seg],q[seg+1],q[min(seg+2,n-1)];tt=np.linspace(0.,1.,201)[:,None]
                xyz=.5*(2*pb+(-pa+pc)*tt+(2*pa-5*pb+4*pc-pd)*tt**2+(-pa+3*pb-3*pc+pd)*tt**3);dd=np.linalg.norm(xyz-point,axis=1);ix=int(dd.argmin());fit.append((float(dd[ix]),seg,float(tt[ix,0])))
            residual,fitseg,fitt=min(fit)
            old=before[s:s+n];pa,pb,pc,pd=old[max(fitseg-1,0)],old[fitseg],old[fitseg+1],old[min(fitseg+2,n-1)];tt=fitt
            oldpoint=.5*(2*pb+(-pa+pc)*tt+(2*pa-5*pb+4*pc-pd)*tt**2+(-pa+3*pb-3*pc+pd)*tt**3)
            h,no,_,_=surface.find_nearest(oldpoint);old_path_signed=float(np.dot(oldpoint-h,no))-(float(signed_center[k])-float(signed[k]))
            row={'family':label,'mainCurve':c,'role':int(groups[c]%256),'sampleIndex':j,'nativeWithinOffset':offset if is_native else None,
                 'controlSegment':segment,'nearestControlSegmentFraction':float(t[segment]),'distanceToControlPolylineMeters':float(dist[segment]),
                 'world':point.tolist(),'actualSignedSurfaceMeters':float(signed[k]),'actualSignedCenterMeters':float(signed_center[k]),'arcMeters':float(arc[j]),
                 'preservedUpperLastPoint':boundary[c],'insideProtectedFirstThreeSegment':segment<2,
                 'segmentRelationToOldUpperBoundary':'inside-preserved-upper' if segment<boundary[c] else 'at-or-after-connector-start',
                 'influenceControlOffsets':control_ids.tolist(),'actualContactDisplacementMeters':np.linalg.norm(contact_delta[control_ids],axis=1).tolist(),
                 'beforeToAfterControlDisplacementMeters':np.linalg.norm(q[control_ids]-old[control_ids],axis=1).tolist(),
                 'catmullEstimatedSegment':fitseg,'catmullEstimatedT':fitt,'catmullToActualWitnessResidualMeters':residual,
                 'oldObservedHeadAtSameWitnessSignedCenterMeters':oldestimate,'oldHeadMinusCurrentRecordedSignedCenterMeters':oldestimate-float(signed_center[k]),
                 'beforeFinalCatmullAtMatchedParameterOnOldHeadSignedSurfaceMeters':old_path_signed}
            rows.append(row);allrows.append(row)
        families.append({'family':label,'samples':len(rows),'curves':sorted(set(r['mainCurve'] for r in rows)),'worstSignedSurfaceMeters':min(r['actualSignedSurfaceMeters'] for r in rows)})
    curve_rows=[]
    for c in sorted(set(r['mainCurve'] for r in allrows)):
        rows=[r for r in allrows if r['mainCurve']==c];s=int(starts[c]);n=int(lengths[c]);ps=int(stage_starts[c]);pn=int(stage_lengths[c]);q=after[s:s+n];old=before[s:s+n]
        curve_rows.append({'curve':c,'role':int(groups[c]%256),'pointCount':n,'preservedUpperLastPoint':boundary[c],
                           'controlMidSegments':sorted(set(r['controlSegment'] for r in rows if r['family']=='gather_control_midpoints')),
                           'nativeNearestControlSegments':sorted(set(r['controlSegment'] for r in rows if 'native' in r['family'])),
                           'nativeEstimatedCatmullSegments':sorted(set(r['catmullEstimatedSegment'] for r in rows if 'native' in r['family'])),
                           'worstSignedSurfaceMeters':min(r['actualSignedSurfaceMeters'] for r in rows),'firstThreeFinalControlsExact':bool(np.array_equal(old[:3],q[:3])),
                           'lastThreeNewConnectorControlsWorld':q[-3:].tolist(),'rootWorld':q[0].tolist(),
                           'beforeFinalWorld':old.tolist(),'afterFinalWorld':q.tolist(),'afterTransportWorld':transport[ps:ps+pn].tolist(),
                           'actualHeadContactDeltaWorld':(contact[ps:ps+pn]-transport[ps:ps+pn]).tolist(),
                           'maximumEstimatedCatmullFitResidualMeters':max(r['catmullToActualWitnessResidualMeters'] for r in rows)})
    actual_head=next(x for x in record['currentSurfaces']['currentCollisionSurfaces'] if x['prefix']=='current_head')
    result={'status':'actual-failed-native-witness-analysis','source':record['source'],'sourceSha256':record['sourceSha256'],
            'failedReportSha256':sha(report_path),'failedArraysSha256':sha(data_path),'actualCurrentHeadWorldSha256':actual_head['worldPositionsSha256'],
            'currentHeadTrianglesIncludedInFailurePayload':False,'oldHeadPredictor':'reference-tie-alignment-r11-v2.npz',
            'oldHeadPredictorSha256':sha(w/'reference-tie-alignment-r11-v2.npz'),'families':families,'curveIds':[x['curve'] for x in curve_rows],
            'curves':curve_rows,'witnesses':allrows,
            'oldHeadWitnessSignedAgreementMetersP0P50P100':np.percentile([r['oldHeadMinusCurrentRecordedSignedCenterMeters'] for r in allrows],[0,50,100]).tolist(),
            'crossings':[x for x in record['currentSurfaces']['families']['gather_control_midpoints']['segmentRayEvidence'] if x['surface']=='R11 Female head study'],
            'findings':['All recorded failing control/native sample families map to the same 25 role-40 curves.',
                        'Actual head contact corrected controls but control chords and native interpolation still enter the surface.',
                        'Control first-three exact does not freeze the native first-two segments: a later Catmull control can change them.',
                        'For Catmull segment p1 to p2, coefficient of p3 is -0.5*t^2*(1-t); an outward p3 displacement can move this protected-start interval inward.'],
            'recommendation':'Use the actual failed segment stencil for a compact smooth correction, preserve p0/p1/p2 and last3 exactly, and verify the complete native path. For protected-start intrusion consider restoring the old safe next control and beginning reconnection later; never push every neighbouring control outward with the same sign.',
            'limits':['Current actual failure signs and coordinates are authoritative. The payload did not retain the current head triangles.',
                      'Old v10e head is used only as a labelled predictor; old-path signed comparisons are not new actual current-head acceptance.',
                      'Catmull parameter matching is an offline estimate with residual explicitly reported; actual native resampling must decide the repair.']}
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'report':str(out),'curves':[{'id':x['curve'],'boundary':x['preservedUpperLastPoint'],'control':x['controlMidSegments'],'native':x['nativeNearestControlSegments'],'worstMm':x['worstSignedSurfaceMeters']*1000} for x in curve_rows],'oldHeadAgreement':result['oldHeadWitnessSignedAgreementMetersP0P50P100']}))

if __name__=='__main__':main()
