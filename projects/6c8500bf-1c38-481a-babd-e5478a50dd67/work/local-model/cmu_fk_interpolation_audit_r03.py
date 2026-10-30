"""Finite offline FK interpolation diagnostic. Pure NumPy; never imports bpy.

Models linear location/scale and normalized, sign-continuous quaternion channel
interpolation through a normal parent/rest hierarchy. This is not Blender's
dependency graph: constraints, drivers, alternate inheritance and arbitrary
rotation modes are outside this model. Compare against a real Blender audit
before relying on it for a specific exported target.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def interpolate_fk_plan(plan,target,sample_frames):
    """Approximate keyed FK between planned poses, not world-matrix blending."""
    frames=np.asarray(plan['target_frames'],dtype=float)
    samples=np.asarray(sample_frames,dtype=float)
    if len(frames)<2 or len(samples)<2 or not np.all(np.isfinite(samples)):
        raise ValueError('At least two finite source and requested frames required')
    if np.any(np.diff(frames)<=0) or np.any(np.diff(samples)<=0):
        raise ValueError('Frames must be strictly increasing')
    if samples[0]<frames[0] or samples[-1]>frames[-1]:
        raise ValueError('No extrapolation beyond the planned clip')
    parser=runpy.run_path(str(Path(__file__).with_name('cmu_asf_amc_r03.py')))
    names=plan['names'];index={n:i for i,n in enumerate(names)}
    world=np.asarray(plan['matrices'],dtype=float)
    rest={n:np.asarray(b['matrix'],dtype=float) for n,b in target['bones'].items()}
    bases=[];parent_links=[]
    for i,name in enumerate(names):
        parent=target['bones'][name]['parent'];parent_index=index.get(parent)
        if parent_index is not None and parent_index>=i:
            raise ValueError('Plan names must be in hierarchy order')
        # Non-transferred ancestors are treated as constant rest transforms.
        parent_rest=rest[parent] if parent else np.eye(4)
        parent_pose=world[:,parent_index] if parent_index is not None else parent_rest
        link=np.linalg.inv(parent_rest)@rest[name]
        basis=np.linalg.inv(link)@np.linalg.inv(parent_pose)@world[:,i]
        bases.append(basis);parent_links.append((parent_index,parent_rest,link))
    bases=np.stack(bases,axis=1)
    scale=np.linalg.norm(bases[:,:,:3,:3],axis=2)
    if np.any(scale<1e-8): raise ValueError('Degenerate local scale')
    rotations=bases[:,:,:3,:3]/scale[:,:,None,:]
    if np.max(np.abs(np.linalg.det(rotations)-1))>1e-4:
        raise ValueError('Reflections or significant local shear are unsupported')
    quaternion=parser['_matrix_quaternions'](rotations)
    for f in range(1,len(frames)):
        quaternion[f]*=np.where(np.sum(quaternion[f-1]*quaternion[f],axis=-1)<0,-1,1)[:,None]
    hi=np.clip(np.searchsorted(frames,samples,side='right'),1,len(frames)-1);lo=hi-1
    t=(samples-frames[lo])/(frames[hi]-frames[lo])
    q=quaternion[lo]*(1-t[:,None,None])+quaternion[hi]*t[:,None,None]
    r=parser['_quaternion_matrices'](q)
    s=scale[lo]*(1-t[:,None,None])+scale[hi]*t[:,None,None]
    location=bases[lo,:,:3,3]*(1-t[:,None,None])+bases[hi,:,:3,3]*t[:,None,None]
    local=np.tile(np.eye(4),(len(samples),len(names),1,1))
    local[:,:,:3,:3]=r*s[:,:,None,:];local[:,:,:3,3]=location
    result=np.empty_like(local)
    for i,(parent_index,parent_rest,link) in enumerate(parent_links):
        parent_pose=result[:,parent_index] if parent_index is not None else parent_rest
        result[:,i]=parent_pose@link@local[:,i]
    output=dict(plan,matrices=result,target_frames=samples,times=np.interp(samples,frames,plan['times']))
    output['contact_weights']=np.stack([np.interp(samples,frames,plan['contact_weights'][:,i]) for i in range(2)],axis=1)
    output['offlineInterpolationScope']='Normalized linear local quaternion channels, linear location/scale, normal hierarchy; finite samples only'
    return output


def diagnose_v2(output_relative=None):
    """Compare this finite approximation with the recorded real quarter audit."""
    target_path=project_path()/'work/reconstruction-r02/cmu-target-r04-actual.json'
    actual_path=project_path()/'work/reconstruction-r02/cmu-v2-foot-audit-r05-quarter.json'
    adapter_path=Path(__file__).with_name('cmu_retarget_r03_v2.py')
    target=json.loads(target_path.read_text(encoding='utf-8-sig'))
    actual=json.loads(actual_path.read_text(encoding='utf-8-sig'))
    adapter=runpy.run_path(str(adapter_path))
    parser=runpy.run_path(str(Path(__file__).with_name('cmu_asf_amc_r03.py')))
    clip=parser['sample_clip'](parser['load_clip'](),30,include_endpoint=True)
    plan=adapter['build_plan'](clip,target)
    rows=actual['actions'][0]['samples'];samples=[row['frame'] for row in rows]
    interpolated=interpolate_fk_plan(plan,target,samples)
    audit=adapter['audit_plan'](interpolated,target)
    key_roundtrip=interpolate_fk_plan(plan,target,plan['target_frames'])
    comparisons={}
    for long,short in [('left','L'),('right','R')]:
        predicted=audit['sides'][long]
        heights=[predicted['minimumHeightByTargetFrame'][str(float(row['frame']))] for row in rows]
        actual_heights=[row['feet'][short]['minimumClearanceMeters'] for row in rows]
        comparisons[long]={'maximumHeightDifferenceFromBlenderMeters':float(np.max(np.abs(np.array(heights)-actual_heights))),
            'offlineMinimumMeters':predicted['minimumCageHeightMeters'],
            'actualMinimumMeters':actual['actions'][0]['feet'][short]['minimumClearanceMeters'],
            'offlineContactPeakMetersPerSecond':predicted['maximumNearGroundHorizontalSpeedMetersPerSecond'],
            'actualContactPeakMetersPerSecond':actual['actions'][0]['feet'][short]['maxContactPointHorizontalSpeed']}
    report={'scope':interpolated['offlineInterpolationScope'],
        'adapterSha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        'targetSha256':hashlib.sha256(target_path.read_bytes()).hexdigest(),
        'actualAuditSha256':hashlib.sha256(actual_path.read_bytes()).hexdigest(),
        'keyMatrixRoundtripMaxError':float(np.max(np.abs(key_roundtrip['matrices']-plan['matrices']))),
        'sampleCount':len(samples),'comparison':comparisons,'offlineAudit':audit,
        'limitations':['This approximation is checked against this recorded action only.',
            'Dense keys reduce between-key curvature error; they do not establish absence of error at arbitrary subframes.',
            'At 120 Hz, contact detection smoothing/window/minimum-duration parameters must retain their original time spans.',
            'Real Blender constraints/drivers/inheritance/rotation modes can invalidate the approximation.',
            'Shoe probes remain finite source-cage points and omit rendered bevel surfaces.'],
        'blenderExecuted':False,'accepted':False}
    if output_relative is not None:
        destination=(project_path()/output_relative).resolve();destination.relative_to((project_path()/'work').resolve())
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print('Loaded only: interpolate_fk_plan() or diagnose_v2() explicitly perform a finite offline diagnostic.')
