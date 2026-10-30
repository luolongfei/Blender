"""Offline actual-domain audit, NumPy only; never loads or launches Blender."""
import hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2]
W=P/'work/reconstruction-r02'
z=np.load(W/'adult-crown-domain-r11.npz');v=z['root_world'];tri=z['root_triangles'];center=z['center_world']
edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
edge,n=np.unique(edges,axis=0,return_counts=True);boundary=edge[n==1];q=v[boundary]-center
angles=np.linspace(-math.pi,math.pi,8193);limits=[]
for phi in angles:
    h=q[:,:,0]*math.sin(phi)-q[:,:,1]*math.cos(phi)
    denominator=h[:,1]-h[:,0]
    fraction=np.divide(-h[:,0],denominator,out=np.full(len(h),-1.),where=np.abs(denominator)>1e-15)
    cut=q[:,0]+fraction[:,None]*(q[:,1]-q[:,0])
    mask=(fraction>=0)&(fraction<=1)&((cut[:,0]*math.cos(phi)+cut[:,1]*math.sin(phi))>0)
    theta=np.arccos(cut[mask,2]/np.linalg.norm(cut[mask],axis=1))
    if not len(theta):raise RuntimeError('Boundary has no polar intersection')
    limits.append(theta.min())
limits=np.asarray(limits)
# A 2 mm XY index gives exact triangle/top-ray intersections for root samples.
points=v[tri];low=np.floor(points[:,:,:2].min(axis=1)/.002).astype(int);high=np.floor(points[:,:,:2].max(axis=1)/.002).astype(int)
grid={}
for i,(a,b) in enumerate(zip(low,high)):
    for x in range(a[0],b[0]+1):
        for y in range(a[1],b[1]+1):grid.setdefault((x,y),[]).append(i)

def top_root(x,y):
    ids=grid.get((math.floor(x/.002),math.floor(y/.002)))
    if not ids:return None
    p=points[ids];a,b,c=p[:,0],p[:,1],p[:,2]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    aa=np.divide((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]),den,out=np.full(len(den),-100.),where=np.abs(den)>1e-15)
    bb=np.divide((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]),den,out=np.full(len(den),-100.),where=np.abs(den)>1e-15)
    cc=1.-aa-bb;mask=(aa>=-1e-9)&(bb>=-1e-9)&(cc>=-1e-9)
    if not mask.any():return None
    height=aa*a[:,2]+bb*b[:,2]+cc*c[:,2]
    return np.array([x,y,height[mask].max()])

def audit(zrange):
    rng=np.random.default_rng(202610291103);minimum=100.;misses=[];checked=0;worst=None
    u=np.linspace(0.,1.,43)
    for side in (1,-1):
        for i in range(2900):
            root=None
            for attempt in range(40):
                near=rng.random()<.62
                x=.0013+side*(rng.uniform(.0012,.020) if near else rng.uniform(.014,.056))
                y=rng.uniform(-.046,.029) if near else rng.uniform(-.070,.020)
                root=top_root(x,y)
                if root is not None and root[2]>1.478:break
                root=None
            if root is None:raise RuntimeError('Root sample failed')
            end=np.array([side*rng.uniform(.048,.065),rng.uniform(.026,.052),rng.uniform(*zrange)])
            middle=np.array([side*(.057+.010*rng.random()),y*.35+.006,1.493+.007*rng.random()])
            dirs=np.array([root,middle,end])-center;dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
            d=(1.-u[:,None])**2*dirs[0]+2.*u[:,None]*(1.-u[:,None])*dirs[1]+u[:,None]**2*dirs[2]
            d/=np.linalg.norm(d,axis=1,keepdims=True)
            theta=np.arccos(d[:,2]);phi=np.arctan2(d[:,1],d[:,0]);limit=np.interp(phi,angles,limits)
            margin=limit-theta-.003 # conservative angular allowance for lookup interpolation
            j=int(margin.argmin());row={'side':side,'strand':i,'point':j,'direction':d[j].tolist(),'conservativeMarginRadians':float(margin[j])}
            if margin[j]<minimum:minimum=float(margin[j]);worst=row
            if margin[j]<0:misses.append(row)
            checked+=len(d)
            # Preserve generator random consumption after successful mapping.
            rng.uniform(.0008,.0048);rng.uniform(0.,.00035);rng.uniform(0.,math.tau);rng.uniform(.000026,.000043)
    return {'endpointZRange':list(zrange),'checkedRays':checked,'outsideConservativeBoundary':len(misses),
            'minimumConservativeAngularMargin':minimum,'worst':worst,'firstMisses':misses[:12]}

report={'actualArraysSha256':hashlib.sha256((W/'adult-crown-domain-r11.npz').read_bytes()).hexdigest(),
        'method':'Exact actual root triangle XY intersections for sampled roots; actual 806 boundary edges intersected with 8193 polar planes, conservative 0.003 rad lookup allowance. Not a substitute for Blender BVH/visual review.',
        'old':audit((1.421,1.443)),'new':audit((1.430,1.449)),
        'geometryEdited':False,'blenderStarted':False,'appearanceAccepted':False}
(W/'adult-crown-domain-fix-r11.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,indent=2))
