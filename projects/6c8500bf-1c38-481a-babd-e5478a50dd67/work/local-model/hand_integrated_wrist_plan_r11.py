"""Plan a local wrist-key revision from the actual integrated dense hand probe.

Offline only. Uses the observed hand after its real Subsurf modifier with its
Armature temporarily disabled. Does not alter weights, rest bones, body
subdivision, hand/nail topology or any head/neck coordinate.
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
WRIST_SHA='715b366dc374f2755574b8ef640d6076d64711cc9e856e8d7a4a68475de26759'


def _path(relative):
    value=(project_path()/relative).resolve();value.relative_to(project_path().resolve());return value


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _limit(segments,point,margin):
    points=segments.reshape(-1,2);center=(points.min(0)+points.max(0))*.5
    direction=point-center;radius=float(np.linalg.norm(direction))
    if radius<1e-8:return point.copy(),radius,radius
    direction/=radius;edge=segments[:,1]-segments[:,0];q=segments[:,0]-center
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    denominator=cross(direction,edge);stable=np.abs(denominator)>1e-12
    t=np.full(len(edge),np.nan);u=np.full(len(edge),np.nan)
    t[stable]=cross(q[stable],edge[stable])/denominator[stable]
    u[stable]=cross(q[stable],direction)/denominator[stable]
    hits=t[stable&(u>=-1e-7)&(u<=1+1e-7)]
    positive=hits[hits>1e-7];negative=hits[hits<-1e-7]
    if not len(positive) or not len(negative):raise RuntimeError('Actual section center is not bracketed')
    limit=float(positive.min());target=min(radius,max(.003,limit-margin))
    return center+direction*target,radius,limit


def build(observation_relative,output_relative,margin=.0015,fade_start=.590,fade_end=.600):
    if not .001<=margin<=.0025 or not .585<=fade_start<fade_end<=.602:
        raise ValueError('Keep bounded measured wrist margin and fade')
    output=_path(output_relative);archive=output.with_suffix('.npz')
    if output.exists() or archive.exists():raise FileExistsError('Preserve existing plans')
    probe=json.loads(_path(observation_relative).read_text(encoding='utf-8'))
    if not probe['rollbackVerified'] or probe['savedBlend'] or _sha(_path(probe['arrays']))!=probe['arraysSha256']:
        raise RuntimeError('Exact successful actual rollback observation required')
    package=json.loads(_path(probe['package']).read_text(encoding='utf-8'))
    if _sha(_path(probe['package']))!=probe['packageSha256'] or _sha(_path(package['arrays']))!=package['arraysSha256']:
        raise RuntimeError('Captured hand/wrist source package changed')
    helper=HERE/'hand_wrist_clearance_r11_v2.py'
    if _sha(helper)!=WRIST_SHA:raise RuntimeError('Frozen wrist math changed')
    wrist=runpy.run_path(str(helper));arrays={};sides={};all_ids=[];all_delta=[]
    with np.load(_path(probe['arrays'])) as observed,np.load(_path(package['arrays'])) as original:
        old_ids=original['wrist_ids'];old_delta=original['wrist_delta']
        for side in ('L','R'):
            ids=observed['raw_'+side+'_ids'];base=observed['raw_'+side+'_base'].astype(np.float64)
            old_mix=observed['raw_'+side+'_mix'].astype(np.float64);baseline=old_mix.copy()
            id_lookup={int(value):i for i,value in enumerate(ids)}
            for index,delta in zip(old_ids,old_delta):
                if int(index) in id_lookup:baseline[id_lookup[int(index)]]-=delta
            hand=observed['relaxed_'+side+'_handRest_positions'].astype(np.float64).copy()
            triangles=observed['relaxed_'+side+'_handRest_triangles']
            hand[:,0]=np.abs(hand[:,0]);normalized=baseline.copy();normalized[:,0]=np.abs(normalized[:,0])
            delta=np.zeros_like(baseline);samples=[]
            for i,point in enumerate(normalized):
                if not fade_start<point[0]<.631:continue
                section=wrist['_section_segments'](hand,triangles,float(point[0]))
                target,radius,limit=_limit(section,point[1:],margin)
                delta[i,1:]=(target-point[1:])*wrist['_smooth'](fade_start,fade_end,point[0])
                samples.append((i,point[0],radius,limit))
            proposed=baseline+delta;changed=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-10)
            raw_triangles=observed['raw_body_triangles'];valid=np.all(np.isin(raw_triangles,ids),axis=1)
            local_triangles=np.asarray([[id_lookup[int(v)] for v in tri] for tri in raw_triangles[valid]],np.int32)
            metrics=wrist['_triangle_metrics'](baseline,proposed,local_triangles)
            maximum=float(np.linalg.norm(delta,axis=1).max())
            if maximum>.0045 or metrics['flippedTriangles'] or metrics['edgeLengthRatioRange'][0]<.4:
                raise RuntimeError('Proposed local key exceeds unchanged geometry guards: '+json.dumps({'side':side,'maximum':maximum,**metrics}))
            arrays[side+'_ids']=ids;arrays[side+'_base']=base.astype(np.float32)
            arrays[side+'_baselineMix']=baseline.astype(np.float32);arrays[side+'_integratedMixBefore']=old_mix.astype(np.float32)
            arrays[side+'_delta']=delta.astype(np.float32);arrays[side+'_weights']=observed['raw_'+side+'_weights']
            arrays[side+'_triangles']=local_triangles;arrays[side+'_sections']=np.asarray(samples)
            all_ids.extend(ids[changed].tolist());all_delta.extend(delta[changed].tolist())
            sides[side]={'changedRawPoints':len(changed),'maximumTotalWristKeyMeters':maximum,
                'maximumAdditionalCorrectionMeters':float(np.linalg.norm(proposed-old_mix,axis=1).max()),
                'absoluteXRange':[float(normalized[changed,0].min()),float(normalized[changed,0].max())],
                'triangleDiagnostics':metrics,'longitudinalCoordinateUnchanged':bool(np.all(delta[:,0]==0)),
                'actualEvaluatedHandVertices':len(hand)}
    if len(set(all_ids))!=len(all_ids):raise RuntimeError('Left/right wrist plan domains overlap')
    arrays['changedIds']=np.asarray(all_ids,np.int32);arrays['delta']=np.asarray(all_delta,np.float32)
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open('xb') as stream:np.savez_compressed(stream,**arrays)
    result={'schemaVersion':1,'observation':observation_relative,'observationSha256':_sha(_path(observation_relative)),
        'observedArraysSha256':probe['arraysSha256'],'observedTargetSource':probe['source'],'observedTargetSourceSha256':probe['sourceSha256'],
        'package':probe['package'],'packageSha256':probe['packageSha256'],'helperSha256':_sha(__file__),'wristMathSha256':WRIST_SHA,
        'arrays':str(archive.relative_to(project_path())),'arraysSha256':_sha(archive),
        'targetBodyModifiers':probe['bodyBeforeIntegration']['modifiers'],'targetBodyObjectAnimation':probe['bodyBeforeIntegration']['objectAnimation'],
        'targetBodyRawVertices':probe['bodyBeforeIntegration']['rawVertices'],'targetBodyGroups':probe['bodyBeforeIntegration']['deformGroups'],
        'targetBodySubdivisionAndWeightsMustRemainUnchanged':True,'marginMeters':margin,'fadeAbsoluteX':[fade_start,fade_end],
        'sides':sides,'totalChangedRawPoints':len(all_ids),'appearanceAccepted':False,'actualPosesPassed':False,
        'method':'Real target hand Subsurf surface in unposed coordinates; measured triangle/plane sections, bounded radial inward movement of old body wrist only.',
        'requiredActualGate':{'minimumDistanceMeters':.00015,'outsideByThreeRayMajority':0,'poses':['relaxed','wrist-flex','wrist-extend']},
        'limits':['Numeric plan only: the actual integration must pass unchanged dense-body pose checks before saving.',
                  'Internal closed overlap remains, not a welded wrist; rendered wrist and cuff reviews remain required.']}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--margin',type=float,default=.0015);parser.add_argument('--fade-start',type=float,default=.590);parser.add_argument('--fade-end',type=float,default=.600)
    a=parser.parse_args();print(json.dumps(build(a.observation,a.output,a.margin,a.fade_start,a.fade_end),ensure_ascii=False))


if __name__=='__main__':main()
