"""Exact exported-triangle front-ray preview; does not start Blender."""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();W=P/'work/reconstruction-r02';HERE=Path(__file__).resolve().parent


class FrontSurface:
    def __init__(self,vertices,triangles):
        self.triangles=np.asarray(vertices)[triangles]
        self.cell=.003;self.grid={}
        xz=self.triangles[:,:,[0,2]]
        self.a=xz[:,0];self.b=xz[:,1]-self.a;self.c=xz[:,2]-self.a
        self.den=self.b[:,0]*self.c[:,1]-self.b[:,1]*self.c[:,0]
        lo=np.floor(xz.min(1)/self.cell).astype(int);hi=np.floor(xz.max(1)/self.cell).astype(int)
        for i in np.flatnonzero(np.abs(self.den)>1e-14):
            for x in range(lo[i,0],hi[i,0]+1):
                for z in range(lo[i,1],hi[i,1]+1):self.grid.setdefault((x,z),[]).append(int(i))
    def __call__(self,x,z):
        ids=self.grid.get((int(np.floor(x/self.cell)),int(np.floor(z/self.cell))))
        if not ids:return None,None
        ids=np.asarray(ids);d=np.array([x,z])-self.a[ids];den=self.den[ids]
        u=(d[:,0]*self.c[ids,1]-d[:,1]*self.c[ids,0])/den
        v=(self.b[ids,0]*d[:,1]-self.b[ids,1]*d[:,0])/den
        valid=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1.+1e-7)
        if not valid.any():return None,None
        ids=ids[valid];u=u[valid];v=v[valid]
        tri=self.triangles[ids];points=tri[:,0]+u[:,None]*(tri[:,1]-tri[:,0])+v[:,None]*(tri[:,2]-tri[:,0])
        i=int(points[:,1].argmin());normal=np.cross(tri[i,1]-tri[i,0],tri[i,2]-tri[i,0]);normal/=np.linalg.norm(normal)
        if normal[1]>0:normal=-normal
        return points[i],normal


def main():
    report=json.loads((W/'fitted-groom-actual-r11.json').read_text(encoding='utf8'))
    arrays=W/'fitted-groom-actual-r11.npz'
    if hashlib.sha256(arrays.read_bytes()).hexdigest()!=report['arraysSha256']:raise RuntimeError('Observed arrays changed')
    z=np.load(arrays,allow_pickle=False);original=np.load(W/'hair-donor-guides-r10.npz',allow_pickle=False)
    surface=FrontSurface(z['head_world'],z['head_triangles'])
    forelock=runpy.run_path(str(HERE/'donor_forelock_surface_r11.py'))
    old=runpy.run_path(str(HERE/'donor_groom_restyle_r11.py'))
    pins=[(z[f'tail_{i}_pins'],z[f'tail_{i}_directions']) for i in range(2)]
    rows=[];failures=[];new_arrays={}
    for group in range(2):
        q=z[f'guide_{group}_world'];starts=z[f'guide_{group}_starts'];lengths=z[f'guide_{group}_lengths'];out=q.copy()
        for k,(start,length) in enumerate(zip(starts,lengths)):
            start=int(start);length=int(length);path=q[start:start+length];root=original[f'guide_{group}_local'][start]
            front=float(old['_smooth']((-root[1]-.86)/.38));width=float(1.-old['_smooth']((abs(root[0])-.46)/.23));weight=front*width
            try:
                if group==1 and k in old['VISIBLE_ARC_SECONDARY']:
                    result=path.copy();row={'kind':'author-crown-layer'}
                elif weight>.12 and root[2]>15.70:
                    result,row=forelock['design'](path,root,k+group*1000,weight,length,surface)
                else:result,row=old['_gather'](path,k+group*1000,pins,length)
                gap=min(old['_segment_eye_gap'](result,z[f'eye_{side}_center'],float(z[f'eye_{side}_radius'])) for side in ('L','R'))
                maximum=float(np.linalg.norm(np.diff(result,axis=0),axis=1).max())
                if gap<.001:raise RuntimeError('Actual eye-envelope gap '+str(gap))
                if maximum>.035:raise RuntimeError('Actual maximum segment '+str(maximum))
                row.update(group=group,guide=k,eyeEnvelopeGapMeters=gap,maximumSegmentMeters=maximum)
                rows.append(row);out[start:start+length]=result
            except Exception as exc:
                failures.append({'group':group,'guide':k,'sourceFrontWeight':weight,'root':path[0].tolist(),'error':str(exc)})
        new_arrays[f'guide_{group}_world']=out
    output={'source':report['source'],'sourceSha256':report['sourceSha256'],'observedArraysSha256':report['arraysSha256'],
            'helperSha256':hashlib.sha256((HERE/'donor_forelock_surface_r11.py').read_bytes()).hexdigest(),
            'scope':'Offline actual head-triangle front projection and exact segment/eye tests; no Blender evaluation, skin-nearest correction or dense GN acceptance',
            'successfulGuides':len(rows),'failedGuides':len(failures),'failures':failures,'guides':rows,'actualBlenderExecution':False}
    (W/'donor-restyle-v2-offline-r11.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    np.savez_compressed(W/'donor-restyle-v2-offline-r11.npz',**new_arrays)
    print(json.dumps({'successful':len(rows),'failed':len(failures),'failures':failures},indent=2),flush=True)
if __name__=='__main__':main()
