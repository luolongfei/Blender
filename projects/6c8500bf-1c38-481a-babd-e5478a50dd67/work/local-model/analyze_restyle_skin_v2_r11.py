"""Offline nearest-triangle preflight of every actual restyled guide."""
import hashlib,json,itertools,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();W=P/'work/reconstruction-r02'


class NearSurface:
    def __init__(self,v,f):
        self.v=v;self.f=f;self.tri=v[f];self.cell=.004;self.grid={};self.incident=[[] for _ in v]
        for i,key in enumerate(np.floor(v/self.cell).astype(int)):self.grid.setdefault(tuple(key),[]).append(i)
        for i,tri in enumerate(f):
            for index in tri:self.incident[index].append(i)
        self.offsets=np.array(list(itertools.product(range(-3,4),repeat=3)))
    def __call__(self,p):
        key=np.floor(p/self.cell).astype(int);ids=[]
        for off in self.offsets:ids.extend(self.grid.get(tuple(key+off),()))
        if len(ids)<24:
            d=np.sum((self.v-p)**2,axis=1);ids=np.argpartition(d,23)[:24]
        else:
            ids=np.asarray(ids);d=np.sum((self.v[ids]-p)**2,axis=1);ids=ids[np.argpartition(d,23)[:24]]
        faces=np.array(sorted(set(i for vertex in ids for i in self.incident[vertex])),np.int32)
        tri=self.tri[faces];a=tri[:,0];ab=tri[:,1]-a;ac=tri[:,2]-a;ap=p-a
        d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
        d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac);den=d00*d11-d01*d01
        u=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=np.abs(den)>1e-20)
        v=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=np.abs(den)>1e-20)
        q=a+u[:,None]*ab+v[:,None]*ac
        valid=(u>=0)&(v>=0)&(u+v<=1)&(np.abs(den)>1e-20)
        d=np.sum((q-p)**2,axis=1);d[~valid]=np.inf
        candidates=[q];distances=[d]
        for ia,ib in ((0,1),(1,2),(2,0)):
            start=tri[:,ia];edge=tri[:,ib]-start
            t=np.clip(np.einsum('ij,ij->i',p-start,edge)/np.maximum(np.einsum('ij,ij->i',edge,edge),1e-20),0,1)
            q=start+t[:,None]*edge;candidates.append(q);distances.append(np.sum((q-p)**2,axis=1))
        distances=np.stack(distances);k,i=np.unravel_index(distances.argmin(),distances.shape);q=candidates[k][i]
        n=np.cross(ab[i],ac[i]);n/=max(np.linalg.norm(n),1e-20)
        return q,n,float(np.dot(p-q,n))


def main():
    original=np.load(W/'fitted-groom-actual-r11.npz');new=np.load(W/'donor-restyle-v2-offline-r11.npz')
    shape=json.loads((W/'donor-restyle-v2-offline-r11.json').read_text(encoding='utf8'))
    if shape['failedGuides']:raise RuntimeError('Resolve geometric layout failures first')
    surface=NearSurface(original['head_world'],original['head_triangles'])
    old=runpy.run_path(str(Path(__file__).with_name('donor_groom_restyle_r11.py')))
    rows=[];failures=[];saved={}
    kinds={(r['group'],r['guide']):r['kind'] for r in shape['guides']}
    for group in range(2):
        q=new[f'guide_{group}_world'].copy();starts=original[f'guide_{group}_starts'];lengths=original[f'guide_{group}_lengths']
        for k,(start,length) in enumerate(zip(starts,lengths)):
            start=int(start);length=int(length);path=q[start:start+length];kind=kinds[group,k];maxpush=0.;count=0
            for j,p in enumerate(path):
                if j==0 or (kind=='gather' and j>=len(path)-2):continue
                hit,n,signed=surface(p)
                if signed<.00016:
                    push=.00020-signed
                    if push>.012:failures.append({'group':group,'guide':k,'kind':kind,'point':j,'skinPushMeters':push,'world':p.tolist()})
                    path[j]=p+n*push;maxpush=max(maxpush,push);count+=1
            gap=min(old['_segment_eye_gap'](path,original[f'eye_{side}_center'],float(original[f'eye_{side}_radius'])) for side in ('L','R'))
            maximum=float(np.linalg.norm(np.diff(path,axis=0),axis=1).max())
            if gap<.001 or maximum>.035:failures.append({'group':group,'guide':k,'kind':kind,'eyeGapMeters':gap,'maxSegmentMeters':maximum})
            rows.append({'group':group,'guide':k,'kind':kind,'correctedPoints':count,'maximumPushMeters':maxpush,'eyeGapMeters':gap,'maximumSegmentMeters':maximum})
        saved[f'guide_{group}_world']=q
        print('GROUP_COMPLETED '+str(group),flush=True)
    report={'scope':'Offline preflight using actual head triangles and 24-nearest-vertex incident triangles; Blender BVH remains the authoritative application check',
            'sourceSha256':shape['sourceSha256'],'sourceArraysSha256':shape['observedArraysSha256'],
            'layoutHelperSha256':shape['helperSha256'],'guides':rows,'failures':failures,'actualBlenderExecution':False}
    (W/'donor-restyle-v2-skin-preflight-r11.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(W/'donor-restyle-v2-skin-preflight-r11.npz',**saved)
    print(json.dumps({'guides':len(rows),'failures':failures,'maximumPush':max(r['maximumPushMeters'] for r in rows)},indent=2),flush=True)
if __name__=='__main__':main()
