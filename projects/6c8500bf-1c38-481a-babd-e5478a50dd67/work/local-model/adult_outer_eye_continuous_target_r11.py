"""Pure observed-jet quintic outer-eye contour targets; never changes a mesh.

All coordinates are metres and Z is world Z. Start jets describe the measured
contour at a 0.6 mm observation scale; the raw triangulated outline is not a
globally C2 curve. The polynomial exactly matches those value/slope/curvature
jets and preserves the original supplied samples before the start.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial import polynomial as poly


def _real_roots(c):
    c=np.asarray(c,float)
    while len(c)>1 and abs(c[-1])<max(np.max(abs(c)),1e-30)*1e-13:c=c[:-1]
    if len(c)<2:return np.empty(0)
    roots=poly.polyroots(c/max(np.max(abs(c)),1e-30))
    return np.asarray(sorted(float(r.real) for r in roots if abs(r.imag)<1e-7 and 1e-8<r.real<1-1e-8))


def _observed_jet(x,z,at,center_z):
    use=abs(x-at)<=.00030001
    if use.sum()<7 or x[use].min()>at-.0002 or x[use].max()<at+.0002:
        raise ValueError('At least seven actual contour samples spanning the 0.6 mm start-jet neighbourhood are required')
    xx=(x[use]-at)*1000;zz=(z[use]-center_z)*1000
    q=np.polyfit(xx,zz,2)
    return dict(valueWorldZ=float(np.interp(at,x,z)),slope=float(q[1]),curvaturePerMeter=float(2*q[0]*1000),
                observationSpanMeters=.0006,fitRmsMicrometers=float(np.sqrt(np.mean((np.polyval(q,xx)-zz)**2))*1000),
                samples=int(use.sum()),method='Value from actual sample interpolation; first and second derivatives from a local quadratic over 0.6 mm')


def _quintic(start,end,length):
    c=np.zeros(6);c[:3]=[start['valueWorldZ'],start['slope']*length,start['curvaturePerMeter']*length*length*.5]
    rhs=[end['valueWorldZ']-sum(c[:3]),end['slope']*length-c[1]-2*c[2],end['curvaturePerMeter']*length*length-2*c[2]]
    c[3:]=np.linalg.solve(np.array([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),rhs)
    return c


def _analyse(c,length,expected_direction):
    d=poly.polyder(c)/length;dd=poly.polyder(d)/length;ddd=poly.polyder(dd)/length
    curvature_stationary=poly.polysub(poly.polymul(ddd,poly.polyadd([1.],poly.polymul(d,d))),
                                    3*poly.polymul(d,poly.polymul(dd,dd)))
    slope_t=np.r_[0.,1.,_real_roots(dd)]
    curvature_t=np.r_[0.,1.,_real_roots(curvature_stationary)]
    second_t=np.r_[0.,1.,_real_roots(ddd)]
    slope=poly.polyval(slope_t,d)
    curvature=poly.polyval(curvature_t,dd)/(1+poly.polyval(curvature_t,d)**2)**1.5
    second=poly.polyval(second_t,dd)
    sign_ok=bool(np.all(expected_direction*slope>=-1e-10) and np.all(expected_direction*second>=-1e-7))
    roots=_real_roots(dd)
    return dict(slopeRange=[float(slope.min()),float(slope.max())],
                signedCurvatureRangePerMm=[float(curvature.min()/1000),float(curvature.max()/1000)],
                curvatureExtremaNormalizedX=curvature_t.tolist(),
                secondDerivativeRangePerMeter=[float(second.min()),float(second.max())],
                interiorCurvatureRoots=roots.tolist(),singleDirectionCurvature=bool(np.all(expected_direction*second>=-1e-7)),
                monotoneEdge=bool(np.all(expected_direction*slope>=-1e-10)),noMacroscopicSInflection=sign_ok)


def target_profile(outward_meters,original_upper_z,original_lower_z,eye_center_z,
                   end_outward_meters,end_center_z,start_outward_meters=.0038):
    """Return upper/lower targets on the supplied increasing outward grid.

    The endpoint is a measured neutral head-bridge support, not the globe rim.
    A 0.15 mm terminal gap prevents collapsing distinct traced margin layers.
    End slopes are reflected about each secant: 2*secant-startSlope. This
    avoids the forced steep-then-flat transition of a secant-valued end slope.
    """
    x=np.asarray(outward_meters,float);upper=np.asarray(original_upper_z,float);lower=np.asarray(original_lower_z,float)
    if x.ndim!=1 or upper.shape!=x.shape or lower.shape!=x.shape or len(x)<20 or not np.all(np.diff(x)>0):
        raise ValueError('Aligned increasing one-dimensional actual contour samples required')
    if not all(np.isfinite(v).all() for v in (x,upper,lower)) or np.any(upper<=lower):
        raise ValueError('Finite ordered actual margin samples required')
    start=float(start_outward_meters);end=float(end_outward_meters);length=end-start
    if not np.isfinite([eye_center_z,end_center_z,start,end]).all() or start<.003 or length<.006 or end>x.max()+.001:
        raise ValueError('Preserve at least the central 3 mm and use the measured nearby outer bridge with at least 6 mm transition')
    report=dict(startOutwardMeters=start,endOutwardMeters=end,transitionLengthMeters=length,
                eyeCenterWorldZ=float(eye_center_z),endCenterWorldZ=float(end_center_z),endpointGapMeters=.00015,
                sourceOriginalOutwardMeters=x.tolist(),sourceOriginalUpperWorldZ=upper.tolist(),sourceOriginalLowerWorldZ=lower.tolist(),
                startJetDefinition='C2 relative to the measured 0.6 mm start jet. Original triangulated samples before start remain exactly unchanged; no claim that the raw source polyline itself is globally C2.',
                endpointPolicy='Neutral front-halfspace head-bridge neighbourhood, not a clipped globe-visible endpoint',
                meshModified=False,physicalContactPassed=False,appearanceAccepted=False,productionReady=False,edges={})
    for name,z,sign in [('upper',upper,-1),('lower',lower,1)]:
        initial=_observed_jet(x,z,start,eye_center_z)
        final_z=float(end_center_z+(.000075 if name=='upper' else -.000075))
        secant=(final_z-initial['valueWorldZ'])/length
        final=dict(valueWorldZ=final_z,slope=float(2*secant-initial['slope']),curvaturePerMeter=0.)
        c=_quintic(initial,final,length);analysis=_analyse(c,length,sign)
        if not analysis['noMacroscopicSInflection']:
            raise ValueError('The observed start jet and chosen endpoint do not admit the selected single-direction quintic: '+name)
        report['edges'][name]=dict(coefficientsWorldZAscending=c.tolist(),normalizedCoordinate='t=(outward-start)/(end-start)',
            startJet=initial,endJet=final,secantSlope=float(secant),analysis=analysis,
            endResidual=dict(valueMeters=float(poly.polyval(1.,c)-final_z),slope=float(poly.polyval(1.,poly.polyder(c))/length-final['slope']),
                             curvaturePerMeter=float(poly.polyval(1.,poly.polyder(c,2))/length**2)))
    cu=np.array(report['edges']['upper']['coefficientsWorldZAscending']);cl=np.array(report['edges']['lower']['coefficientsWorldZAscending'])
    gap=cu-cl;t=np.r_[0.,1.,_real_roots(poly.polyder(gap))];values=poly.polyval(t,gap)
    if values.min()<.00015-1e-10:raise ValueError('Target margins cross or fall below their terminal separation')
    closing_rate=report['edges']['lower']['endJet']['slope']-report['edges']['upper']['endJet']['slope']
    extension=min(.000025,(.00015-.0001)/closing_rate)
    report.update(minimumTargetGapMeters=float(values.min()),shortExtensionMeters=float(extension),extensionLimitMeters=end+extension,
        shortExtensionPolicy='Linear continuation of the matching end tangent and zero curvature for at most 25 micrometres, with at least 0.10 mm gap. Beyond the explicit limit, the independent-margin evaluator rejects extrapolation; the existing connected skin bridge needs its own transport support.')
    out_upper=evaluate_target(x,report,'upper');out_lower=evaluate_target(x,report,'lower')
    if not np.array_equal(out_upper[x<start],upper[x<start]) or not np.array_equal(out_lower[x<start],lower[x<start]):
        raise RuntimeError('Central supplied samples changed')
    report['centralOriginalSamplesExact']=True
    return out_upper,out_lower,report


def evaluate_target(outward_meters,report,edge='upper'):
    """Evaluate original samples / quintic / bounded tangent continuation.

    Coordinates beyond extensionLimitMeters are rejected, not silently
    continued into crossing upper/lower curves in the outer skin bridge.
    """
    if edge not in ('upper','lower'):raise ValueError('upper or lower required')
    x=np.asarray(outward_meters,float)
    if not np.isfinite(x).all() or np.any(x>report['extensionLimitMeters']+1e-12):
        raise ValueError('Query outside the explicitly bounded independent-margin target domain')
    initial=np.asarray(report['sourceOriginalUpperWorldZ' if edge=='upper' else 'sourceOriginalLowerWorldZ'])
    out=np.interp(x,np.asarray(report['sourceOriginalOutwardMeters']),initial)
    start,end=report['startOutwardMeters'],report['endOutwardMeters'];row=report['edges'][edge]
    t=(x-start)/(end-start);curve=poly.polyval(np.clip(t,0,1),np.asarray(row['coefficientsWorldZAscending']))
    out=np.where(x>=start,curve,out)
    tangent=row['endJet']['valueWorldZ']+row['endJet']['slope']*(x-end)
    return np.where(x>end,tangent,out)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def run(contour_npz,metadata_json,bridge_json,output_json):
    source,meta_file,bridge_file,out=map(Path,(contour_npz,metadata_json,bridge_json,output_json))
    if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('Preserve previous target proposal')
    data=np.load(source);metadata=json.loads(meta_file.read_text());bridge=json.loads(bridge_file.read_text());payload={}
    result=dict(basis='Actual BEFORE / outer strength zero; rejected preview is not the neutral basis',
                sourceContourSha256=_sha(source),bridgeEvidenceSha256=_sha(bridge_file),helperSha256=_sha(__file__),
                contourProposalOnly=True,meshModified=False,physicalContactPassed=False,appearanceAccepted=False,sides={})
    for side in 'LR':
        md=metadata['plan']['sides'][side];sign=md['outwardSign'];center=np.array(md['centerWorld'])
        rows=data[side+'_before_contour'];x=sign*(rows[:,0]-center[0]);assert np.all(np.diff(x)>0)
        endpoint_abs=np.mean(bridge['supportRecommendation']['frontHalfspaceBridgeAbsoluteXMmBracket'])*.001
        end_out=endpoint_abs-abs(center[0]);end_z=center[2]+bridge['lastOpenBefore'][side]['opening']['centerOffsetMm']*.001
        upper,lower,r=target_profile(x,rows[:,2],rows[:,1],center[2],end_out,end_z)
        r['outwardSign']=sign;r['eyeCenterWorldX']=float(center[0]);r['endWorldX']=float(sign*endpoint_abs)
        result['sides'][side]=r
        order=np.argsort(rows[:,0]);payload[side+'_worldX']=rows[order,0]
        payload[side+'_sourceUpper']=rows[order,2];payload[side+'_sourceLower']=rows[order,1]
        payload[side+'_targetUpper']=upper[order];payload[side+'_targetLower']=lower[order]
        grid=np.linspace(.0038,r['extensionLimitMeters'],1001)
        payload[side+'_denseOutward']=grid;payload[side+'_denseTargetUpper']=evaluate_target(grid,r,'upper');payload[side+'_denseTargetLower']=evaluate_target(grid,r,'lower')
        print('TARGET '+side+' '+json.dumps({k:r[k] for k in ('startOutwardMeters','endOutwardMeters','transitionLengthMeters','minimumTargetGapMeters','shortExtensionMeters')}),flush=True)
        for name in ('upper','lower'):print(name,json.dumps(r['edges'][name]['analysis']),flush=True)
    np.savez_compressed(out.with_suffix('.npz'),**payload);result['arraysSha256']=_sha(out.with_suffix('.npz'))
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=_json),encoding='utf8')
    print('TARGET_DONE '+_sha(out),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('contours','metadata','bridge','output'):p.add_argument('--'+arg,required=True)
    a=p.parse_args();run(a.contours,a.metadata,a.bridge,a.output)
