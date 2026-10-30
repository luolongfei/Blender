"""Bounded offline continuous triangle checks of the new calf bridge only."""
import hashlib,json,runpy,sys
from pathlib import Path
from itertools import product,combinations
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def check():
    p=project_path();s=p/'work/reconstruction-r02';plan=s/'foot-leg-unified-candidate-r11-v2.json'
    if sha(plan)!='8977e539f122dc0cdf501b3a8f3fef2f72acab6ef94337aadc8229046a178a11':raise RuntimeError('Candidate changed')
    j=json.loads(plan.read_text());array=p/j['arrays']
    if sha(array)!=j['arraysSha256']:raise RuntimeError('Candidate array changed')
    helper=Path(__file__).with_name('hand_wrist_surface_contact_r11.py')
    if sha(helper)!='c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e':raise RuntimeError('Frozen continuous triangle helper changed')
    api=runpy.run_path(str(helper));a=np.load(array);results={}
    for side in ('L','R'):
        pre='legFoot_'+side;v=a[pre+'_positions'];loops=a[pre+'_loops'];starts=a[pre+'_polygonStarts'];sizes=a[pre+'_polygonSizes'];regions=a[pre+'_faceRegion'];triangles=[];triangle_region=[]
        for pi,(start,size) in enumerate(zip(starts,sizes)):
            f=loops[start:start+size]
            for k in range(1,size-1):triangles.append((f[0],f[k],f[k+1]));triangle_region.append(regions[pi])
        triangles=np.asarray(triangles,np.int32);triangle_region=np.asarray(triangle_region);points=v[triangles]
        bridge=np.flatnonzero(triangle_region==1);lo=points[bridge].min((0,1))-.000001;hi=points[bridge].max((0,1))+.000001
        tl=points.min(1);th=points.max(1);local=np.flatnonzero(((th>=lo)&(tl<=hi)).all(1));cells={};cell=.005
        for i in local:
            aa=np.floor((tl[i]-.000001)/cell).astype(int);bb=np.floor((th[i]+.000001)/cell).astype(int)
            for index in product(*(range(a,b+1) for a,b in zip(aa,bb))):cells.setdefault(index,[]).append(int(i))
        pairs=set()
        for ids in cells.values():
            for i,k in combinations(ids,2):
                if triangle_region[i]!=1 and triangle_region[k]!=1:continue
                if np.any(tl[i]>th[k]+1e-6) or np.any(tl[k]>th[i]+1e-6):continue
                if set(triangles[i])&set(triangles[k]):continue
                pairs.add(tuple(sorted((i,k))))
        ordered=np.asarray(sorted(pairs),np.int32).reshape(-1,2);intersections=[];contacts=[];minimum=None
        for start in range(0,len(ordered),4096):
            batch=ordered[start:start+4096];metrics=api['triangle_pair_metrics'](points[batch[:,0]],points[batch[:,1]])
            intersections.extend(batch[metrics['intersects']].tolist());contacts.extend(batch[metrics['contactWithinTolerance']].tolist())
            if len(batch):minimum=float(metrics['distance'].min()) if minimum is None else min(minimum,float(metrics['distance'].min()))
        areas=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]),axis=1)*.5
        results[side]={'bridgeTriangles':len(bridge),'localTriangles':len(local),'nonAdjacentBroadPhasePairs':len(ordered),
            'continuousIntersectionPairs':len(intersections),'contactPairsWithin1Micrometre':len(contacts),'minimumCandidatePairDistanceMeters':minimum,
            'bridgeMinimumTriangleAreaM2':float(areas[bridge].min()),'bridgeDegenerateTriangles':int((areas[bridge]<1e-14).sum()),
            'intersectionTriangleIds':intersections[:100],'contactTriangleIds':contacts[:100],
            'scope':'Actual raw candidate bridge versus itself and incident donor/retained-leg surfaces; shared-vertex pairs excluded as topological neighbours. Native subdivision and deformed poses are separate pending checks.'}
    out=s/'foot-leg-unified-candidate-r11-v2-geometry-check.json'
    if out.exists():raise FileExistsError('Preserve prior evidence')
    record={'candidateSha256':sha(plan),'candidateArraysSha256':j['arraysSha256'],'helperSha256':sha(__file__),
        'continuousHelperSha256':sha(helper),'sides':results,'executedBlender':False,'sourceGeometryEdited':False,'appearanceAccepted':False,
        'bridgeRawGeometryCheckPassed':all(r['continuousIntersectionPairs']==0 and r['contactPairsWithin1Micrometre']==0 and r['bridgeDegenerateTriangles']==0 for r in results.values())}
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8');print(json.dumps({'report':str(out),'sha256':sha(out),**record}))

if __name__=='__main__':check()
