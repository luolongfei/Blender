"""Vertical native inner-roof sampling; a fitting diagnostic, not full clearance."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def roof(points,triangles):
    """Return minimum downward-facing roof intersection above z=.04 at each XY."""
    q=np.asarray(triangles);n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);unit=n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)
    ids=np.flatnonzero((unit[:,2]<-.30)&(q[:,:,2].max(1)>.040));q=q[ids];a=q[:,0];u=q[:,1]-a;v=q[:,2]-a;den=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    output=np.full(len(points),np.nan);face=np.full(len(points),-1,np.int32)
    for start in range(0,len(points),128):
        p=points[start:start+128];d=p[:,None,:2]-a[None,:,:2]
        s=(d[:,:,0]*v[None,:,1]-d[:,:,1]*v[None,:,0])/den[None,:]
        t=(u[None,:,0]*d[:,:,1]-u[None,:,1]*d[:,:,0])/den[None,:]
        z=a[None,:,2]+s*u[None,:,2]+t*v[None,:,2]
        valid=(s>=-1e-9)&(t>=-1e-9)&(s+t<=1+1e-9)&(z>.04)
        z[~valid]=np.inf;k=z.argmin(1);hits=z[np.arange(len(p)),k];found=np.isfinite(hits)
        output[start:start+len(p)][found]=hits[found];face[start:start+len(p)][found]=ids[k[found]]
    return output,face

def measure():
    p=project_path();s=p/'work/reconstruction-r02';source=s/'footwear-225-native-probe-r11.json'
    if sha(source)!='0afca9ada441483c1f946fe8e0d5b0a9a3c61f67bfdca0a410984a25581357c0':raise RuntimeError('Actual probe changed')
    j=json.loads(source.read_text());a=np.load(p/j['arrays']);out=s/'footwear-225-toe-roof-measure-r11.json';outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve measured evidence')
    if sha(p/j['arrays'])!=j['arraysSha256']:raise RuntimeError('Actual arrays changed')
    results={};arrays={}
    for side in ('L','R'):
        fv=a['foot_'+side+'_world'];ft=a['foot_'+side+'_triangles'];q=fv[ft];norm=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None];c=q.mean(1)
        if side=='R':norm*=-1  # actual donor object has a negative-determinant mirror transform
        chosen=(norm[:,2]>.3)&(c[:,1]<-.09)&(c[:,2]>.04)
        samples=np.unique(np.concatenate((q[chosen].reshape(-1,3),c[chosen])),axis=0)
        pre=next(r['arrayPrefix'] for r in j['shoes'] if r['name']=='Shoe_Leather_Upper_'+side)
        z,face=roof(samples,a[pre+'_world'][a[pre+'_triangles']]);gap=z-samples[:,2];finite=np.isfinite(gap);near=finite&(gap<.002)
        arrays[side+'_sampleWorld']=samples;arrays[side+'_innerRoofZ']=z;arrays[side+'_roofTriangleId']=face
        results[side]={'samples':len(samples),'roofHits':int(finite.sum()),'minimumVerticalGapMeters':float(gap[finite].min()),'p05VerticalGapMeters':float(np.percentile(gap[finite],5)),'below2mmSamples':int(near.sum()),'below2mmSampleBounds':[samples[near].min(0).tolist(),samples[near].max(0).tolist()] if near.any() else None,
            'worstSamples':[{'point':samples[i].tolist(),'verticalGapMeters':float(gap[i]),'roofTriangleId':int(face[i])} for i in np.flatnonzero(finite)[np.argsort(gap[finite])[:12]]]}
    np.savez_compressed(outa,**arrays);result={'schemaVersion':1,'sourceProbeSha256':sha(source),'helperSha256':sha(__file__),'sides':results,'arrays':str(outa.relative_to(p)),'arraysSha256':sha(outa),'scope':'Actual native downward-facing shoe inner roof vertical samples over upward-facing forefoot triangle vertices and centroids. Directional gap, not certified global normal-distance clearance. 2 mm is exploratory sock-room threshold, not anatomical standard.'}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'report':str(out),'sha256':sha(out),'sides':results}))

if __name__=='__main__':measure()
