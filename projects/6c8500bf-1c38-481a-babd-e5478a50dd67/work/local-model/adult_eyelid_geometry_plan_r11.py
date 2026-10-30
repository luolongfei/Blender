"""Read-only offline visible-aperture audit of the actual R11 head and eyes.

No Blender, render, mesh editing or assumption that the socket ring is a lash
line. Uses front-to-back triangle intersections on the saved actual head and
the unchanged donor eye geometry used by the successful eye builder.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


class FrontRay:
    def __init__(self,points,triangles,bounds,cell=.002):
        self.points=np.asarray(points,float);self.triangles=np.asarray(triangles,np.int32);self.cell=cell
        p=self.points[self.triangles];lo=p[:,:,[0,2]].min(1);hi=p[:,:,[0,2]].max(1)
        keep=np.all(hi>=bounds[0],axis=1)&np.all(lo<=bounds[1],axis=1)
        self.indices=np.flatnonzero(keep);self.p=p[keep];lo=np.floor(lo[keep]/cell).astype(int);hi=np.floor(hi[keep]/cell).astype(int)
        self.buckets={}
        for i,(a,b) in enumerate(zip(lo,hi)):
            for x in range(a[0],b[0]+1):
                for z in range(a[1],b[1]+1):self.buckets.setdefault((x,z),[]).append(i)

    def ray(self,x,z):
        ids=self.buckets.get((int(np.floor(x/self.cell)),int(np.floor(z/self.cell))))
        if not ids:return None
        p=self.p[ids];a=p[:,0][:,[0,2]];b=p[:,1][:,[0,2]];c=p[:,2][:,[0,2]]
        e=b-a;f=c-a;q=np.array([x,z])-a
        det=e[:,0]*f[:,1]-e[:,1]*f[:,0];valid=abs(det)>1e-18;inv=np.zeros(len(det));inv[valid]=1/det[valid]
        u=(q[:,0]*f[:,1]-q[:,1]*f[:,0])*inv;v=(e[:,0]*q[:,1]-e[:,1]*q[:,0])*inv
        valid&=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
        if not valid.any():return None
        y=p[:,0,1]*(1-u-v)+p[:,1,1]*u+p[:,2,1]*v;y[~valid]=np.inf;k=int(np.argmin(y))
        normal=np.cross(p[k,1]-p[k,0],p[k,2]-p[k,0]);normal/=np.linalg.norm(normal)
        return {'point':[float(x),float(y[k]),float(z)],'normal':normal.tolist(),'triangle':int(self.indices[ids[k]]),
                'barycentric':[float(1-u[k]-v[k]),float(u[k]),float(v[k])]}


def main():
    P=project_path();obs_path=P/'work/reconstruction-r02/adult-neck-observation-r11.json';obs=json.loads(obs_path.read_text())
    npz=P/obs['arrays']['path']
    if hashlib.sha256(npz.read_bytes()).hexdigest()!=obs['arrays']['sha256']:raise RuntimeError('Actual head observation changed')
    with np.load(npz) as f:
        head=f['rest_00_world'];tri=f['rest_00_triangles'];cage=f['rest_00_raw_world']
    T=np.asarray(obs['objects']['R11 Female head study']['matrixWorld'])
    eye_api=runpy.run_path(str(Path(__file__).with_name('eye_adult_fit_r11.py')));_,eyes=eye_api['_source'](P)
    report={'headObservation':str(obs_path.relative_to(P)),'headSourceSha256':obs['sourceSha256'],'headArraySha256':obs['arrays']['sha256'],
        'savedBlend':False,'geometryEdited':False,'rendered':False,'eyes':{},
        'scope':'Actual front-ray visible globe contour against the neutral R11 evaluated head. Eye arrays are exactly those used by the successful R11 eye module; later materials do not change these triangles.',
        'limits':['Front visibility is not a complete inner/outer lid contact diagnosis.',
            'Nearest raw vertices are diagnostic neighbors, not a closed wet-line or a blink mask.',
            'Any proposed upper-lid change must transport actual lashes and pass actual neutral/closed-eye checks.']}
    for side,data in eyes.items():
        v=data['co']@T[:3,:3].T+T[:3,3];faces=data['faces'];et=np.asarray([(f[0],f[k],f[k+1]) for f in faces for k in range(1,len(f)-1)],np.int32)
        lo=v.min(0);hi=v.max(0);center=(lo+hi)*.5;bounds=np.array([lo[[0,2]]-.004,hi[[0,2]]+.004])
        skin=FrontRay(head,tri,bounds);globe=FrontRay(v,et,bounds)
        def visible(x,z):
            g=globe.ray(x,z)
            if g is None:return False
            h=skin.ray(x,z);return h is None or g['point'][1]<h['point'][1]-1e-7
        contour=[]
        for x in np.linspace(lo[0]+.00035,hi[0]-.00035,151):
            zs=np.linspace(lo[2]+.0001,hi[2]-.0001,121);flags=np.array([visible(x,z) for z in zs]);ids=np.flatnonzero(flags)
            if len(ids)<3 or ids[0]==0 or ids[-1]==len(zs)-1:continue
            if np.any(np.diff(ids)!=1):raise RuntimeError('Noncontiguous actual front-visible aperture')
            ends=[]
            for inside,outside in ((int(ids[0]),int(ids[0]-1)),(int(ids[-1]),int(ids[-1]+1))):
                a=zs[inside];b=zs[outside]
                for _ in range(12):
                    m=(a+b)*.5
                    if visible(x,m):a=m
                    else:b=m
                z=(a+b)*.5;g=globe.ray(x,z);h=skin.ray(x,z)
                neighbor_z=z+(-.00015 if inside==ids[0] else .00015)
                external=skin.ray(x,neighbor_z)
                q=np.array((external or h or g)['point']);distance=np.linalg.norm(cage-q,axis=1);near=np.argsort(distance)[:8]
                ends.append({'z':float(z),'skin':h,'globe':g,'skinOutside150Micrometers':external,
                    'nearestRawDiagnosticVertices':near.tolist(),'nearestRawDiagnosticDistances':distance[near].tolist()})
            contour.append({'x':float(x),'lower':ends[0],'upper':ends[1]})
        xs=np.array([q['x'] for q in contour]);lower=np.array([q['lower']['z'] for q in contour]);upper=np.array([q['upper']['z'] for q in contour])
        iris_radius=float(((hi[0]-lo[0])+(hi[2]-lo[2]))*.5*.245)
        uz=float(np.interp(center[0],xs,upper));lz=float(np.interp(center[0],xs,lower))
        report['eyes'][side]={'centerWorld':center.tolist(),'diametersWorld':(hi-lo).tolist(),'irisRadiusWorld':iris_radius,
            'visibleApertureWidth':float(xs[-1]-xs[0]),'centralVisibleApertureHeight':uz-lz,
            'centralUpperIrisCoverage':float(center[2]+iris_radius-uz),'centralLowerIrisCoverage':float(lz-(center[2]-iris_radius)),
            'maximumVisibleApertureHeight':float((upper-lower).max()),'contour':contour}
    output=P/'work/reconstruction-r02/adult-eyelid-visible-audit-r11.json'
    if output.exists():raise RuntimeError('Retain prior eye audit')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({s:{k:v for k,v in d.items() if k!='contour'} for s,d in report['eyes'].items()},indent=2))


if __name__=='__main__':main()
