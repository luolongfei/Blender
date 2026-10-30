"""Independent actual fixed-X radial transport for an open-eye corner study.

The failed front-projection prototypes remain untouched. The shared inner and
outer sheet follows a monotone angular field in actual (globe gap, angle)
coordinates. Discrete triangles are validated independently of this mapping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)
def fade(b):return 0. if b>=.5 else 1.-float(smooth(max(b,0.)/.5))


def _rays(points,theta,eye,center):
    origins=np.asarray(points,float).copy();origins[:,1:]=center[1:]
    directions=np.column_stack([np.zeros(len(theta)),-np.cos(theta),np.sin(theta)])
    return api('adult_blink_corner_plan_r11')['ray_radius'](origins,directions,eye['world'],eye['triangles'])


def _axis(eye):
    center=(eye['world'].min(0)+eye['world'].max(0))*.5
    rays=api('adult_blink_corner_plan_r11')['ray_radius']
    radius=rays(np.array([center,center]),np.array([[-1.,0.,0.],[1.,0.,0.]]),eye['world'],eye['triangles'])
    if not np.isfinite(radius).all() or np.any(radius<.005):raise RuntimeError('Actual centerline exit is unavailable')
    return center,np.array([center[0]-radius[0],center[0]+radius[1]])


def query_coordinates(points,metadata,side):
    """C1 query continuation; actual skin X remains untouched."""
    q=np.asarray(points,float).copy();details=metadata['sides'][side];sign=details['outwardSign']
    safe=sign*details['outwardActualAxisExitX']-metadata['querySafeInsetMeters']
    span=metadata['queryClampTransitionMeters'];start=safe-span*.5;u=sign*q[:,0]
    t=np.clip((u-start)/span,0.,1.);mapped=start+span*(t-t**3+.5*t**4)
    mapped=np.where(u<=start,u,np.where(u>=start+span,safe,mapped));q[:,0]=sign*mapped
    return q


def plan(neutral_world,triangles,original_ids,eyes,contours,compression=.70,width=.003):
    w=np.asarray(neutral_world,float);t=np.asarray(triangles,np.int32);ids=np.asarray(original_ids,np.int32)
    if w.shape!=(len(ids),3) or not 0<compression<=.75:raise ValueError('Aligned actual surface and ratio >= .25 required')
    cls=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex'];head=cls(w,t)
    fields={};arrays={'originalVertexIds':ids,'neutralWorld':w,'triangles':t}
    report={'compression':compression,'widthMeters':width,'gapCoreMeters':.006,'gapFadeMeters':.006,
            'angularPaddingRadians':.12,'angularFadeRadians':.45,'blinkFadeEnd':.5,
            'axisFadeMeters':.0005,'axisInsetMeters':.00002,'minimumRadiusMeters':1e-7,
            'querySafeInsetMeters':.0008,'queryClampTransitionMeters':.0008,
            'negativeGapFadeStartMeters':-.003,'negativeGapFadeEndMeters':-.002,
            'bridgeFullExtensionMeters':.0005,'bridgeFadeEndMeters':.002,
            'fieldChannels':['xEnvelope','thetaMid','angularCoreHalf'],'sides':{},'appearanceAccepted':False,'installed':False}
    for side in ('L','R'):
        eye=eyes[side];center,axis=_axis(eye);sign=1. if center[0]>0 else -1.
        rows=contours[side]['contour'];x=np.asarray([q['x'] for q in rows]);lo=np.asarray([q['lower']['worldZ'] for q in rows]);hi=np.asarray([q['upper']['worldZ'] for q in rows])
        globe=cls(eye['world'],eye['triangles']);angles=[];usable=[]
        for label,z,direction in [('lower',lo,-1.),('upper',hi,1.)]:
            samples=np.column_stack([x,np.zeros(len(x)),z+direction*.00002]);hy,ht,_=head.front(samples);gy,_,_=globe.front(samples)
            valid=np.isfinite(hy)&np.isfinite(gy)&(hy<=gy+2e-6)&(hy<center[1])
            # Actual covered skin rays, not the globe-surface Y substituted
            # for a head edge. Missing extreme samples are explicitly tagged.
            theta=np.arctan2(samples[:,2]-center[2],center[1]-hy)
            angles.append(theta);usable.append(valid)
            arrays['edge_'+side+'_'+label+'_headWorld']=np.column_stack([x,hy,samples[:,2]])
            arrays['edge_'+side+'_'+label+'_headTriangleIds']=ht
            arrays['edge_'+side+'_'+label+'_coveredMask']=valid
        valid=usable[0]&usable[1]
        if valid.sum()<50:raise RuntimeError('Insufficient actual covered head edges for '+side)
        lower=angles[0][valid];upper=angles[1][valid]
        if np.any(upper<=lower):raise RuntimeError('Actual front skin angular edges are reversed')
        mid=(lower+upper)*.5;half=(upper-lower)*.5+.12
        endpoint=x.max() if sign>0 else x.min();exit_x=axis[1] if sign>0 else axis[0]
        inward=sign*(endpoint-w[:,0]);exit_distance=sign*(exit_x-w[:,0])
        envelope=(1.-smooth(np.maximum(inward,0.)/width))*smooth((inward+.002)/.0015)
        envelope[(inward>=width)|(inward<=-.002)]=0.
        field=np.column_stack([envelope,np.interp(w[:,0],x[valid],mid),np.interp(w[:,0],x[valid],half)])
        fields[side]=field;arrays['field_'+side]=field
        report['sides'][side]={'centerWorld':center.tolist(),'axisInsideXRange':axis.tolist(),'outwardSign':sign,
            'endpointX':float(endpoint),'actualCoveredEdgeSamples':int(valid.sum()),'missingExtremeEdgeSamples':int((~valid).sum()),
            'edgeInterpolationXRange':[float(x[valid].min()),float(x[valid].max())],
            'outwardActualAxisExitX':float(exit_x),'positiveXEnvelopePoints':int((envelope>0).sum())}
    if np.any((fields['L'][:,0]>0)&(fields['R'][:,0]>0)):raise RuntimeError('Independent radial side envelopes overlap')
    return fields,arrays,report


def apply(world,triangles,fields,eyes,metadata,blink=None,strength=1.):
    w=np.asarray(world,float);blink=blink or {'L':0.,'R':0.};delta=np.zeros_like(w);reports={}
    if not 0<=strength<=1:raise ValueError('Bounded strength required')
    normal=api('adult_blink_midpath_contact_r11')['normals'](w,triangles)
    for side in ('L','R'):
        factor=fade(float(blink[side]))*strength;f=np.asarray(fields[side],float);center,axis=_axis(eyes[side])
        if f.shape!=w.shape or not np.isfinite(f).all():raise ValueError('Finite radial channel fields required')
        if factor==0:reports[side]={'fade':0.,'changedPoints':0,'rejectedCount':0};continue
        theta=np.arctan2(w[:,2]-center[2],center[1]-w[:,1]);rho=np.linalg.norm(w[:,1:]-center[None,1:],axis=1)
        angle=theta-f[:,1];angular=1.-smooth((abs(angle)-f[:,2])/metadata['angularFadeRadians'])
        # This radius bound is only an early-out; all moved points use actual
        # triangle hits and actual centerline support below.
        extent=np.linalg.norm(eyes[side]['world'][:,1:]-center[None,1:],axis=1).max()+metadata['gapCoreMeters']+metadata['gapFadeMeters']
        candidates=np.flatnonzero((f[:,0]>0)&(angular>0)&(rho<extent)&(rho>metadata['minimumRadiusMeters']))
        query=query_coordinates(w[candidates],metadata,side)
        old_r=_rays(query,theta[candidates],eyes[side],center)
        gap=rho[candidates]-old_r
        radial=1.-smooth((gap-metadata['gapCoreMeters'])/metadata['gapFadeMeters'])
        radial*=smooth((gap-metadata['negativeGapFadeStartMeters'])/(metadata['negativeGapFadeEndMeters']-metadata['negativeGapFadeStartMeters']))
        amount=metadata['compression']*f[candidates,0]*angular[candidates]*radial*factor
        target_theta=theta[candidates]-amount*angle[candidates]
        new_r=_rays(query,target_theta,eyes[side],center);target_rho=new_r+gap
        interior=(query[:,0]>axis[0]+metadata['axisInsetMeters'])&(query[:,0]<axis[1]-metadata['axisInsetMeters'])
        valid=np.isfinite(old_r)&np.isfinite(new_r)&(old_r>0)&(new_r>0)&interior&(target_rho>metadata['minimumRadiusMeters'])
        active=valid&(amount>1e-12);ids=candidates[active]
        proposed=np.column_stack([w[ids,0],center[1]-target_rho[active]*np.cos(target_theta[active]),center[2]+target_rho[active]*np.sin(target_theta[active])])
        delta[ids]=proposed-w[ids]
        after_gap=target_rho[active]-new_r[active]
        outer=normal[ids,1]<-.1
        new_negative=outer&(gap[active]>=-2e-8)&(after_gap<-2e-8)
        # In the continued domain this is a reference gap, not local contact.
        # Check the actual unchanged full globe independently for outer skin.
        closest=api('upper_lid_relief_surface_r11')['closest_surface']
        outer_ids=ids[outer]
        before_actual=closest(w[outer_ids],eyes[side]['world'],eyes[side]['triangles'])
        after_actual=closest(w[outer_ids]+delta[outer_ids],eyes[side]['world'],eyes[side]['triangles'])
        new_actual_negative=(before_actual['signed']>=-2e-8)&(after_actual['signed']<-2e-8)
        reports[side]={'fade':factor,'changedPoints':len(ids),'rejectedCount':int((~valid&(amount>1e-12)).sum()),
            'rejectedLocalIds':candidates[~valid&(amount>1e-12)],'changedLocalIds':ids,
            'maximumMeters':float(np.linalg.norm(delta[ids],axis=1).max()) if len(ids) else 0.,
            'maximumRadialGapChangeMeters':float(abs(after_gap-gap[active]).max()) if len(ids) else 0.,
            'minimumNewRadiusMeters':float(target_rho[active].min()) if len(ids) else None,
            'referenceContinuationChangedPoints':int((abs(query[active,0]-w[ids,0])>1e-12).sum()),
            'localActualGapPreservationClaim':False,
            'newActualOuterGlobeInsideCount':int(new_actual_negative.sum()),
            'newActualOuterGlobeInsideLocalIds':outer_ids[new_actual_negative],
            'minimumActualOuterSignedAfterMeters':float(after_actual['signed'].min()) if len(outer_ids) else None,
            'originalNegativeInnerCount':int(((normal[ids,1]>.1)&(gap[active]<0)).sum()),
            'newOuterRadialNegativeCount':int(new_negative.sum())}
        if new_negative.any():raise RuntimeError('Radial field changed the preserved reference gap')
    return delta,reports


def _json(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)


def run(observation_relative,contours_relative,output_relative,compression=.70):
    p=project_path();of,cf,out=[(p/q).resolve() for q in (observation_relative,contours_relative,output_relative)]
    for file in (of,cf,out):file.relative_to((p/'work/reconstruction-r02').resolve())
    if out.exists() or out.with_suffix('.npz').exists():raise RuntimeError('Fresh independent radial proposal paths required')
    obs=json.loads(of.read_text(encoding='utf8'))
    if sha(of.with_suffix('.npz'))!=obs['arraysSha256']:raise RuntimeError('Actual observation checksum mismatch')
    with np.load(of.with_suffix('.npz')) as a:data={k:a[k].copy() for k in a.files}
    contours=json.loads(cf.read_text(encoding='utf8'));eyes={s:{'world':data['eye_'+s+'_world'],'triangles':data['eye_'+s+'_triangles']} for s in ('L','R')}
    fields,arrays,report=plan(data['state_0_head_world'],data['state_0_head_triangles'],data['head_originalVertexIds'],eyes,contours['blink_000'],compression)
    roll=api('upper_lid_roll_surface_r11');weights,_,_=roll['plan'](data['state_0_head_world'],data['state_0_head_triangles'],data['head_originalVertexIds'],eyes,contours['blink_000'])
    contact=api('adult_blink_midpath_contact_r11');slice_api=api('adult_neutral_eye_outerends_r11')['aperture_slices']
    report.update(source=obs['source'],sourceSha256=obs['sourceSha256'],observation=observation_relative,observationArraysSha256=obs['arraysSha256'],
        helperSha256=sha(__file__),rollHelperSha256=sha(Path(__file__).with_name('upper_lid_roll_surface_r11.py')),states=[],
        limits=['Current captured triangles are the offline reference; a native evaluated mesh may legally retessellate quads.',
                'Gap refers to the smoothly continued actual reference slice where query X differs from skin X; actual globe outer-skin checks are separate.',
                'The prescribed early fade requires a newly derived midpath to avoid reopening in early blink.',
                'Positive continuous angular Jacobian is not used as a substitute for discrete triangle checks.'])
    for n in (0,1):
        state=obs['states'][n];w=data[state['headWorldArray']];tri=data[state['trianglesArray']];blink={s:state[s] for s in ('L','R')}
        rd,rm=roll['apply'](w,tri,weights,eyes,blink=blink);d,metrics=apply(w+rd,tri,fields,eyes,report,blink=blink);result=w+rd+d
        changed=np.linalg.norm(result-w,axis=1)>1e-9;faceids=np.flatnonzero(changed[tri].any(1));q0=w[tri[faceids]];q1=result[tri[faceids]]
        v0=np.cross(q0[:,1]-q0[:,0],q0[:,2]-q0[:,0]);v1=np.cross(q1[:,1]-q1[:,0],q1[:,2]-q1[:,0]);a0=np.linalg.norm(v0,axis=1);a1=np.linalg.norm(v1,axis=1);good=a0>1e-14
        alignment=np.sum(v0[good]*v1[good],axis=1)/(a0[good]*a1[good]);ratio=a1[good]/a0[good]
        print('RADIAL_CHECK '+state['id']+' changed '+str(changed.sum()),flush=True)
        crossings=contact['changed_intersections'](result,tri,changed)
        row={'stateIndex':n,'id':state['id'],'radial':metrics,'roll':rm,'actualChangedInteriorCrossings':len(crossings),'minimumNormalAlignment':float(alignment.min()),
            'normalAlignmentQuantiles':np.quantile(alignment,[0,.01,.05,.5,1]),'areaRatioQuantiles':np.quantile(ratio,[0,.01,.05,.5,.95,.99,1]),'aperture':{}}
        for s in ('L','R'):
            args=(eyes[s],report['sides'][s]['endpointX'],report['sides'][s]['outwardSign'])
            distances=(0.,.00025,.0005,.001,.0015,.002,.003,.004)
            row['aperture'][s]={'before':slice_api(w,tri,*args,distances=distances),'after':slice_api(result,tri,*args,distances=distances)}
        arrays['state_'+str(n)+'_resultWorld']=result;arrays['state_'+str(n)+'_radialDeltaWorld']=d;arrays['state_'+str(n)+'_rollDeltaWorld']=rd
        arrays['state_'+str(n)+'_crossingTrianglePairs']=np.asarray(sorted(crossings),np.int32).reshape(-1,2)
        arrays['state_'+str(n)+'_changedTriangleRows']=faceids;arrays['state_'+str(n)+'_triangleAreaRatios']=ratio;arrays['state_'+str(n)+'_normalAlignments']=alignment
        report['states'].append(row)
        print('RADIAL_RESULT '+json.dumps({'state':state['id'],'crossings':len(crossings),'minAlignment':float(alignment.min()),'areaRange':[float(ratio.min()),float(ratio.max())],'rejected':{s:metrics[s]['rejectedCount'] for s in ('L','R')},'newActualOuterInside':{s:metrics[s]['newActualOuterGlobeInsideCount'] for s in ('L','R')},'gap1mm':{s:row['aperture'][s]['after'][3]['gapMeters']*1000 for s in ('L','R')}}),flush=True)
    report['lateStatesExactlyZero']={}
    for n in (2,3,4,5,6):
        state=obs['states'][n];d,_=apply(data[state['headWorldArray']],data[state['trianglesArray']],fields,eyes,report,blink={s:state[s] for s in ('L','R')})
        if np.any(d):raise RuntimeError('Radial field is not exact zero after early fade')
        report['lateStatesExactlyZero'][state['id']]=True
    report['offlineGeometryPassed']=all(r['actualChangedInteriorCrossings']==0 and r['minimumNormalAlignment']>0 and r['areaRatioQuantiles'][0]>.25 and r['areaRatioQuantiles'][-1]<4 and all(r['radial'][s]['rejectedCount']==0 and r['radial'][s]['newActualOuterGlobeInsideCount']==0 for s in ('L','R')) for r in report['states'])
    np.savez_compressed(out.with_suffix('.npz'),**arrays);report['arraysSha256']=sha(out.with_suffix('.npz'));out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=_json),encoding='utf8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observation',required=True);parser.add_argument('--contours',required=True);parser.add_argument('--output',required=True);parser.add_argument('--compression',type=float,default=.70)
    a=parser.parse_args();run(a.observation,a.contours,a.output,a.compression)
