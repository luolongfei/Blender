"""A bounded right-only wrist correction from all three actual v3 pose minima.

Preserve the accepted left wrist byte-for-byte and the prior total wrist plan.
Only a compact, smooth inward field around actual right-side near contacts is
added. No global margin increase, subdivision/weight change or relaxed gate.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PARENT='work/reconstruction-r02/hand-integrated-wrist-fit-plan-r11-v2.json'
PARENT_SHA='88f306549c2f1f78327cfe363ac87772f3b04ed102228538304d5ad8fe7e8c56'
PARENT_ARRAYS_SHA='e920cc2039c80eed7ddee94a3141fb6ad0f8e1b43c4943e359782d14e6d3ea41'
WRIST_SHA='715b366dc374f2755574b8ef640d6076d64711cc9e856e8d7a4a68475de26759'
MERGER_SHA='c09486f3c9502fa5b574d8656d8c55c72d9a0431a2bf1f1cff9fb9be1f495100'
PROBE_SHA='1eab2f40740f36772de19601b1ba88209531c9c4e0d1dec97c8bd758efff39da'


def _path(relative):
    p=(project_path()/relative).resolve();p.relative_to(project_path().resolve());return p


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def build(observation_relative,output_relative,amplitude=.0003,radius=.014):
    if not .0001<=amplitude<=.0003 or not .010<=radius<=.018:raise ValueError('Keep bounded local micro-correction')
    output=_path(output_relative);archive=output.with_suffix('.npz')
    if output.exists() or archive.exists():raise FileExistsError('Preserve prior target corrections')
    if _sha(_path(PARENT))!=PARENT_SHA:raise RuntimeError('Frozen parent dense-target plan changed')
    parent=json.loads(_path(PARENT).read_text(encoding='utf-8'))
    if _sha(_path(parent['arrays']))!=PARENT_ARRAYS_SHA:raise RuntimeError('Frozen parent payload changed')
    obs=json.loads(_path(observation_relative).read_text(encoding='utf-8'))
    if obs['frozenMergerSha256']!=MERGER_SHA or obs['helperSha256']!=PROBE_SHA:
        raise RuntimeError('Actual observation was not made by the frozen v3 merger/probe pair')
    if obs['package']!=parent['package'] or obs['packageSha256']!=parent['packageSha256'] or _sha(_path(obs['package']))!=parent['packageSha256']:
        raise RuntimeError('Actual observation uses a different refined-hand package')
    if _sha(_path(obs['source']))!=obs['sourceSha256']:raise RuntimeError('Actual probed target changed on disk')
    if [row['pose'] for row in obs['poses']]!=['relaxed','wrist-flex','wrist-extend']:
        raise RuntimeError('All three actual v3 wrist poses are required exactly once')
    observed_body=obs['bodyBeforeIntegration']
    if observed_body['modifiers']!=parent['targetBodyModifiers'] or observed_body['objectAnimation']!=parent['targetBodyObjectAnimation']:
        raise RuntimeError('Dense body modifier/animation state differs from parent')
    if observed_body['deformGroups']!=parent['targetBodyGroups'] or observed_body['rawVertices']!=parent['targetBodyRawVertices']:
        raise RuntimeError('Dense body cage/group contract differs from parent')
    if obs['installedTargetWristPlanSha256']!=PARENT_SHA or obs['installedTargetWristArraysSha256']!=PARENT_ARRAYS_SHA:
        raise RuntimeError('Actual v3 observation is not the parent correction')
    if not obs['rollbackVerified'] or obs['savedBlend'] or _sha(_path(obs['arrays']))!=obs['arraysSha256']:
        raise RuntimeError('Exact actual source-preserving probe required')
    if any(row['sides']['L']['failedRequiredGate'] for row in obs['poses']):
        raise RuntimeError('Left no longer passes all three poses; this right-only scope is inapplicable')
    with np.load(_path(parent['arrays'])) as original:arrays={name:original[name].copy() for name in original.files}
    points=[];directions=[];weights=[];evidence=[]
    with np.load(_path(obs['arrays'])) as measured:
        if not np.array_equal(measured['raw_R_ids'],arrays['R_ids']):raise RuntimeError('Actual right cage domain differs')
        measured_right_mix=measured['raw_R_mix'].astype(np.float64).copy()
        for pose in obs['poses']:
            prefix=pose['pose'].replace('-','_')+'_R';record=pose['sides']['R']
            distance=measured[prefix+'_distance'];bad=np.flatnonzero(distance<.00022)
            if record['outsideByThreeRayMajority'] or len(bad)>20:raise RuntimeError('Right contact is broader than the measured tiny-clearance correction')
            body_rest=measured[prefix+'_body_unposed'][measured[prefix+'_body_core']]
            hand_rest=measured[prefix+'_handRest_positions'];hand_actual=measured[prefix+'_hand_positions']
            hand_triangles=measured[prefix+'_hand_triangles']
            if len(hand_rest)!=len(hand_actual):raise RuntimeError('Actual hand vertex topology changes across pose')
            for index in bad:
                triangle=hand_triangles[measured[prefix+'_closestTriangle'][index]]
                tri=hand_rest[triangle].astype(np.float64);outward=np.cross(tri[1]-tri[0],tri[2]-tri[0])
                outward[0]=0;length=np.linalg.norm(outward)
                if length<1e-10:raise RuntimeError('Near-contact normal has no stable transverse component')
                inward=-outward/length;weight=max(.0003-float(distance[index]),1e-6)
                points.append(body_rest[index]);directions.append(inward);weights.append(weight)
                evidence.append({'pose':pose['pose'],'coreIndex':int(index),'bodyEvaluatedIndex':record['coreOriginalVertexIds'][int(index)],
                    'restPosition':body_rest[index].tolist(),'currentDistanceMeters':float(distance[index]),
                    'belowRequired150Microns':bool(distance[index]<.00015),'inwardRestYZ':inward.tolist()})
    if not points:raise RuntimeError('No actual right near-contact needs this correction')
    points=np.asarray(points,np.float64);weights=np.asarray(weights);directions=np.asarray(directions)
    center=np.average(points,axis=0,weights=weights);direction=np.average(directions,axis=0,weights=weights)
    strength=float(np.linalg.norm(direction))
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(direction)) or strength<.75:
        raise RuntimeError('Actual near-contact normals do not form a stable inward direction')
    direction/=strength
    if np.min(directions@direction)<.5:raise RuntimeError('Average correction is not inward at every measured near-contact triangle')
    if np.max(np.linalg.norm(points-center,axis=1))>.008 or direction[0]!=0:
        raise RuntimeError('Actual deficient points are not one bounded transverse wrist patch')
    old_delta=arrays['R_delta'].astype(np.float64);baseline=arrays['R_baselineMix'].astype(np.float64);current=baseline+old_delta
    if np.max(np.abs(current-measured_right_mix))>2e-7:raise RuntimeError('Actual v3 right cage mix differs from the parent plan')
    radius_normalized=np.linalg.norm(current-center,axis=1)/radius
    t=np.clip(1-radius_normalized,0,1);falloff=t**4*(4*radius_normalized+1)
    lower=np.asarray((-.631,-.06,1.175));upper=np.asarray((-.590,.06,1.295))
    if np.any(center-radius<=lower) or np.any(center+radius>=upper):
        raise RuntimeError('Compact support intersects a hard wrist-domain boundary; use a smaller local radius')
    allowed=(current[:,0]<-.590)&(current[:,0]>-.631)&(np.abs(current[:,1])<.06)&(np.abs(current[:,2]-1.235)<.06)
    if np.any(falloff[~allowed]!=0):raise RuntimeError('Compact field was clipped before its C2 zero boundary')
    falloff[~allowed]=0
    if falloff.max()<.1:raise RuntimeError('No nearby raw cage support for the actual minimum')
    falloff/=falloff.max();extra=amplitude*falloff[:,None]*direction
    affected=np.flatnonzero(np.linalg.norm(extra,axis=1)>1e-10)
    if not 3<=len(affected)<=80:raise RuntimeError('Right micro-correction support is not local: '+str(len(affected)))
    new_delta=old_delta+extra;new_mix=baseline+new_delta
    helper=HERE/'hand_wrist_clearance_r11_v2.py'
    if _sha(helper)!=WRIST_SHA:raise RuntimeError('Frozen raw triangle guard changed')
    wrist=runpy.run_path(str(helper));metrics=wrist['_triangle_metrics'](baseline,new_mix,arrays['R_triangles'])
    if np.linalg.norm(new_delta,axis=1).max()>.0045 or metrics['flippedTriangles'] or metrics['edgeLengthRatioRange'][0]<.4:
        raise RuntimeError('Right-only micro-correction fails unchanged raw geometry guards: '+json.dumps(metrics))
    arrays['R_delta']=new_delta.astype(np.float32)
    changed_ids=[];deltas=[]
    for side in ('L','R'):
        changed=np.flatnonzero(np.linalg.norm(arrays[side+'_delta'],axis=1)>1e-10)
        changed_ids.extend(arrays[side+'_ids'][changed].tolist());deltas.extend(arrays[side+'_delta'][changed].tolist())
    arrays['changedIds']=np.asarray(changed_ids,np.int32);arrays['delta']=np.asarray(deltas,np.float32)
    original_left_sha=hashlib.sha256(arrays['L_delta'].tobytes()).hexdigest()
    with np.load(_path(parent['arrays'])) as original:
        if any(not np.array_equal(arrays[k],original[k]) for k in arrays if k.startswith('L_')):
            raise RuntimeError('Left correction changed')
    arrays['rightLocalAffectedIds']=arrays['R_ids'][affected];arrays['rightLocalExtra']=extra[affected].astype(np.float32)
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as stream:np.savez_compressed(stream,**arrays)
    result=dict(parent)
    result.update({'observation':observation_relative,'observationSha256':_sha(_path(observation_relative)),
        'observedArraysSha256':obs['arraysSha256'],'observedTargetSource':obs['source'],'observedTargetSourceSha256':obs['sourceSha256'],
        'helperSha256':_sha(__file__),'arrays':str(archive.relative_to(project_path())),'arraysSha256':_sha(archive),
        'parentPlan':PARENT,'parentPlanSha256':PARENT_SHA,'parentArraysSha256':PARENT_ARRAYS_SHA,'totalChangedRawPoints':len(changed_ids),
        'inheritedGeometryBaselineObservation':{'report':parent['observation'],'reportSha256':parent['observationSha256'],
            'arraysSha256':parent['observedArraysSha256'],'source':parent['observedTargetSource'],'sourceSha256':parent['observedTargetSourceSha256'],
            'scope':'Unchanged raw Basis, original pre-v3 integratedMixBefore and weights used by the merger identity guard.'},
        'rightLocalObservation':{'report':observation_relative,'reportSha256':_sha(_path(observation_relative)),
            'arraysSha256':obs['arraysSha256'],'source':obs['source'],'sourceSha256':obs['sourceSha256'],
            'scope':'Actual v3 evaluated contact points and raw mix used only to derive the additional right-side field.'},
        'method':'Preserved measured 1.7 mm base envelope plus a single compact C2 field on the right, oriented by actual near-contact hand triangle normals in all three poses.',
        'rightLocalRefinement':{'actualNearContactSamples':evidence,'centerRest':center.tolist(),'inwardDirectionRestYZ':direction.tolist(),
            'supportRadiusMeters':radius,'maximumAdditionalRawDisplacementMeters':float(np.linalg.norm(extra,axis=1).max()),
            'affectedRightRawPoints':len(affected),'affectedRightRawVertexIds':arrays['R_ids'][affected].tolist(),
            'leftPlanArraysExact':True,'leftDeltaSha256':original_left_sha,'longitudinalCoordinatesUnchanged':True},
        'appearanceAccepted':False,'actualPosesPassed':False})
    result['sides']['R']=dict(result['sides']['R']);result['sides']['R']['triangleDiagnostics']=metrics
    right_changed=np.flatnonzero(np.linalg.norm(new_delta,axis=1)>1e-10)
    result['sides']['R']['changedRawPoints']=len(right_changed)
    result['sides']['R']['absoluteXRange']=[float(np.abs(baseline[right_changed,0]).min()),float(np.abs(baseline[right_changed,0]).max())]
    result['sides']['R']['maximumAdditionalCorrectionMeters']=float(np.linalg.norm(new_mix-arrays['R_integratedMixBefore'],axis=1).max())
    result['sides']['R']['maximumTotalWristKeyMeters']=float(np.linalg.norm(new_delta,axis=1).max())
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--amplitude',type=float,default=.0003);parser.add_argument('--radius',type=float,default=.014)
    a=parser.parse_args();print(json.dumps(build(a.observation,a.output,a.amplitude,a.radius),ensure_ascii=False))


if __name__=='__main__':main()
