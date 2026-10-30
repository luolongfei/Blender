"""Offline actual forehead triangle ray check. NumPy only, no Blender launch."""
import hashlib,json,math,runpy
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2];W=P/'work/reconstruction-r02'
m=runpy.run_path(str(Path(__file__).with_name('adult_groom_refine_r11.py')))
a=np.load(W/'female-multires-actual-r11.npz');s=1.53119993/1.6414071321487427
v=a['eval_00_body_space']*s+np.array([0.,.030,0.]);corners=a['eval_00_corner_vertices'];starts=a['eval_00_face_starts'];lengths=a['eval_00_face_lengths']
assert np.all(lengths==4)
faces=corners.reshape(-1,4);take=(v[faces,2].min(axis=1)>1.300);faces=faces[take]
head_tri=np.concatenate([faces[:,[0,1,2]],faces[:,[0,2,3]]]);head_points=v[head_tri]
z=np.load(W/'adult-crown-domain-r11.npz');root_points=z['root_world'][z['root_triangles']]

class RayMesh:
    def __init__(self,points,axes,depth_axis,front_min):
        self.p=points;self.axes=axes;self.depth=depth_axis;self.front_min=front_min
        xy=points[:,:,axes];low=np.floor(xy.min(axis=1)/.002).astype(int);high=np.floor(xy.max(axis=1)/.002).astype(int)
        self.grid={}
        for i,(a,b) in enumerate(zip(low,high)):
            for x in range(a[0],b[0]+1):
                for y in range(a[1],b[1]+1):self.grid.setdefault((x,y),[]).append(i)
    def ray(self,x,y):
        ids=self.grid.get((math.floor(x/.002),math.floor(y/.002)))
        if not ids:return None
        p=self.p[ids];q=p[:,:,self.axes];a,b,c=q[:,0],q[:,1],q[:,2]
        den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        aa=np.divide((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]),den,out=np.full(len(den),-100.),where=np.abs(den)>1e-15)
        bb=np.divide((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]),den,out=np.full(len(den),-100.),where=np.abs(den)>1e-15)
        cc=1.-aa-bb;mask=(aa>=-1e-8)&(bb>=-1e-8)&(cc>=-1e-8)
        if not mask.any():return None
        depth=aa*p[:,0,self.depth]+bb*p[:,1,self.depth]+cc*p[:,2,self.depth]
        valid=np.flatnonzero(mask);j=valid[np.argmin(depth[mask]) if self.front_min else np.argmax(depth[mask])]
        point=aa[j]*p[j,0]+bb[j]*p[j,1]+cc[j]*p[j,2]
        n=np.cross(p[j,1]-p[j,0],p[j,2]-p[j,0]);n/=np.linalg.norm(n)
        return point,n

root=RayMesh(root_points,[0,1],2,False);skin=RayMesh(head_points,[0,2],1,True)
rows=[]
for spec in m['fringe_guide_plans']():
    original=np.asarray(spec['controls'],float);controls=original.copy();hit=root.ray(*controls[0,:2]);controls[0]=hit[0]+hit[1]*.00011
    oldguide=m['_catmull'](controls,76 if spec['kind']=='cheek' else 61);actual=[]
    for j in range(1,len(controls)):
        if spec['kind']=='cheek' and j>=4:continue
        hit=skin.ray(controls[j,0],controls[j,2])
        if hit is None:
            actual.append({'control':j,'rayMiss':True});continue
        p,n=hit
        if n[1]>0:n=-n
        lift=([0.,.0035,.0042,.0030,.0022,.0018][j] if spec['kind']=='fringe' else [0.,.0036,.0040,.0030][j])
        controls[j]=p+n*lift;actual.append({'control':j,'surface':p.tolist(),'normal':n.tolist(),'lift':lift,'adjustmentMeters':float(np.linalg.norm(controls[j]-original[j]))})
    guide=m['_catmull'](controls,len(oldguide));stats={}
    for label,path in [('old',oldguide),('new',guide)]:
        penetrations=[]
        for j,p in enumerate(path):
            hit=skin.ray(p[0],p[2])
            if hit is None:continue
            point,n=hit
            if n[1]>0:n=-n
            signed=float(np.dot(p-point,n))
            if signed<0:penetrations.append({'point':j,'signedNormalPlaneMeters':signed,'position':p.tolist()})
        stats[label]={'insideAnteriorTangentPlaneCount':len(penetrations),'worst':min(penetrations,key=lambda x:x['signedNormalPlaneMeters']) if penetrations else None}
    rows.append({'guide':spec['name'],'actualControls':actual,'stats':stats})
report={'helperSha256':hashlib.sha256(Path(m['__file__']).read_bytes()).hexdigest(),'actualHeadArraysSha256':hashlib.sha256((W/'female-multires-actual-r11.npz').read_bytes()).hexdigest(),
        'method':'Actual L3 head triangles in verified adult transform; exact forward Y rays in XZ grid. Signed tangent-plane distances are a conservative design diagnostic, not nearest-BVH clearance or final rendered acceptance.',
        'knownDifference':'Current source chin has <=1.6mm shaping below these forehead controls; lower free cheek endpoints still need actual current-head BVH.',
        'guides':rows,'blenderStarted':False,'geometryEdited':False}
(W/'adult-front-guide-surface-audit-r11.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
for row in rows:print(row['guide'],json.dumps(row['stats']), 'control ray misses', [x for x in row['actualControls'] if x.get('rayMiss')])
