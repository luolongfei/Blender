"""Offline exact triangle checks of explicit frozen native leg/shoe/sock captures."""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def path_for(relative):
    path=(project_path()/relative.replace('\\','/')).resolve();path.relative_to(project_path().resolve());return path

def jobs(record,a):
    if 'pairSets' in record:
        for r in record['pairSets']:
            first,second=r['firstArrayPrefix'],r['secondArrayPrefix']
            yield r['label'],a[first+'_world'],a[first+'_triangles'],a[second+'_world'],a[second+'_triangles'],a[r['pairsArray']],{'first':first,'second':second,'bvhEpsilonMeters':r['epsilonMeters']}
    elif 'cases' in record:
        for case in record['cases']:
            for side,r in case['sides'].items():
                pre=r['arrayPrefix'];v=a[pre+'_world'];t=a['triangles_'+side]
                yield case['name']+'/'+side,v,t,v,t,a[pre+'_candidatePairs'],{'case':case['name'],'side':side,'restRoiRule':r['roiRule'],'roiDegenerateTriangles':r['roiDegenerateTriangles'],'bvhEpsilonMeters':r['localSurfaceCandidates']['epsilonMeters']}
    elif 'shoes' in record and 'feet' in record:
        for row in record['shoes']:
            pre=row['arrayPrefix'];side=row['side'];foot='foot_'+side
            yield 'skin / '+row['name'],a[foot+'_world'],a[foot+'_triangles'],a[pre+'_world'],a[pre+'_triangles'],a[pre+'_footPairs'],{'first':foot,'second':pre,'bvhEpsilonMeters':0,'scope':'Original shoe-only probe uses zero BVH expansion; positive submicrometre near misses outside those bounds are not covered.'}
    else:raise RuntimeError('Unsupported native capture schema')

def run(probe,probe_sha,output):
    source=path_for(probe);out=path_for(output);outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve prior evidence')
    if len(probe_sha)!=64 or sha(source)!=probe_sha:raise RuntimeError('Explicit actual capture SHA mismatch')
    j=json.loads(source.read_text());array=path_for(j['arrays'])
    if not j['restorationExact'] or sha(array)!=j['arraysSha256']:raise RuntimeError('Actual capture incomplete or changed')
    helper=Path(__file__).with_name('hand_wrist_surface_contact_r11.py')
    if sha(helper)!='c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e':raise RuntimeError('Frozen continuous geometry helper changed')
    api=runpy.run_path(str(helper));a=np.load(array,allow_pickle=False);rows=[];packed={}
    for name,fv,ft,sv,st,pairs,scope in jobs(j,a):
        pairs=np.asarray(pairs,np.int32).reshape(-1,2);distances=np.full(len(pairs),np.nan);intersects=np.zeros(len(pairs),bool);contact=intersects.copy();degenerate=[]
        for start in range(0,len(pairs),4096):
            batch=pairs[start:start+4096];f=fv[ft[batch[:,0]]];s=sv[st[batch[:,1]]]
            fa=np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1);sa=np.linalg.norm(np.cross(s[:,1]-s[:,0],s[:,2]-s[:,0]),axis=1);valid=(fa>=1e-16)&(sa>=1e-16)
            degenerate.extend((start+np.flatnonzero(~valid)).tolist())
            if valid.any():
                metrics=api['triangle_pair_metrics'](f[valid],s[valid],tolerance=1e-6);ids=start+np.flatnonzero(valid);distances[ids]=metrics['distance'];intersects[ids]=metrics['intersects'];contact[ids]=metrics['contactWithinTolerance']
        pre='check_'+str(len(rows));packed[pre+'_pairs']=pairs;packed[pre+'_distance']=distances;packed[pre+'_intersects']=intersects;packed[pre+'_contactWithin1Micrometre']=contact;packed[pre+'_degeneratePairIndices']=np.asarray(degenerate,np.int32)
        hits=[]
        for index in np.flatnonzero(intersects)[:128]:
            x,y=pairs[index];f=fv[ft[x]];s=sv[st[y]];hits.append({'firstTriangle':int(x),'secondTriangle':int(y),'overlapAabbMeters':[np.maximum(f.min(0),s.min(0)).tolist(),np.minimum(f.max(0),s.max(0)).tolist()]})
        finite=np.isfinite(distances)
        rows.append({'label':name,'arrayPrefix':pre,'scope':scope,'broadPhasePairs':len(pairs),'continuousIntersectionPairs':int(intersects.sum()),'contactPairsWithin1Micrometre':int(contact.sum()),'degenerateCandidatePairs':len(degenerate),'minimumCandidatePairDistanceMeters':float(distances[finite].min()) if finite.any() else None,'intersectionDiagnostics':hits,'diagnosticsTruncated':int(intersects.sum())>128})
    np.savez_compressed(outa,**packed);r={'schemaVersion':1,'probe':probe,'probeSha256':probe_sha,'probeArraysSha256':j['arraysSha256'],'source':j['source'],'sourceSha256':j['sourceSha256'],'helperSha256':sha(__file__),'continuousHelperSha256':sha(helper),'checks':rows,'intersectionPairs':sum(x['continuousIntersectionPairs'] for x in rows),'contactPairsWithin1Micrometre':sum(x['contactPairsWithin1Micrometre'] for x in rows),'degenerateCandidatePairs':sum(x['degenerateCandidatePairs'] for x in rows),'arrays':str(outa.relative_to(project_path())),'arraysSha256':sha(outa),'executedBlender':False,'appearanceAccepted':False,'deliveryReady':False,'scope':'Exact closed triangle SAT plus point/face and every edge/edge feature distance on the actual capture BVH pairs. Numerical tolerance is 1 micrometre, not a required anatomical gap. Does not prove positive room away from the broad phase, intervening motion or final appearance.'}
    out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':output,'sha256':sha(out),'intersectionPairs':r['intersectionPairs'],'contactPairs':r['contactPairsWithin1Micrometre'],'degenerateCandidatePairs':r['degenerateCandidatePairs'],'nonzero':[{k:x[k] for k in ('label','continuousIntersectionPairs','contactPairsWithin1Micrometre','degenerateCandidatePairs')} for x in rows if x['contactPairsWithin1Micrometre'] or x['degenerateCandidatePairs']]}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe',required=True);parser.add_argument('--probe-sha',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();run(a.probe,a.probe_sha,a.output)
