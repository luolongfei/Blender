"""Reference-led continuous crown/front paths; pure geometry, no Blender IO."""
import math
import numpy as np
CENTER=np.array([0.,-.005,1.430])
EDGES=np.array([-.082,-.055,-.037,-.022,.008,.023,.040,.065,.082])
TIP_X=np.array([-.065,-.048,-.032,.001,.018,.046,.064,.073])
TIP_Z=np.array([1.427,1.444,1.437,1.440,1.451,1.451,1.434,1.417])

def smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)

def normalize(x):return x/max(float(np.linalg.norm(x)),1e-12)

def resample(p,n):
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    return np.column_stack([np.interp(np.linspace(0.,arc[-1],n),arc,p[:,k]) for k in range(3)])

def role(root):
    # The former front-only band left crown-front guides combing backwards.
    crown_front=root[1]<.014 and root[2]>1.478 and abs(root[0])<.077
    temple=root[1]<-.015 and root[2]>1.447 and abs(root[0])>.054
    if crown_front or temple:
        k=int(np.clip(np.searchsorted(EDGES,root[0],side='right')-1,0,7));return 10+k
    return 0

def front_path(root,index,count,project_ray,project_front):
    """project_ray(direction) returns the real skin hit and outward normal."""
    kind=role(root)
    if kind==0:raise RuntimeError('Not a reference front guide')
    k=kind-10;mid=(EDGES[k]+EDGES[k+1])*.5;half=(EDGES[k+1]-EDGES[k])*.5
    lateral=float(np.clip((root[0]-mid)/half,-1.,1.))
    # Unequal main locks, finer right splits, tapered/feathered tips. Outer
    # guide tips end earlier, avoiding a common horizontal cutting line.
    end_x=TIP_X[k]+lateral*.0021
    end_z=TIP_Z[k]+.010*abs(lateral)**1.35+.0015*math.sin(index*2.39996)
    if root[2]<1.478:end_z=min(end_z,root[2]-.026)
    target,target_normal=project_front(float(end_x),float(end_z))
    if target is None:raise RuntimeError('Reference fringe tip missed actual front surface '+str(index))
    a=normalize(root-CENTER);b=normalize(np.asarray(target)-CENTER)
    angle=math.acos(float(np.clip(np.dot(a,b),-.999999,1.)))
    out=[];max_gap=0.
    for t in np.linspace(0.,1.,max(96,count*5)):
        # Slow start spreads the takeoff across the scalp instead of creating
        # an abrupt vertical curtain from a narrow strip of frontal roots.
        u=float(t)
        direction=normalize((math.sin((1.-u)*angle)*a+math.sin(u*angle)*b)/max(math.sin(angle),1e-10))
        hit,normal=project_ray(direction)
        if hit is None:raise RuntimeError('Real scalp radial projection missed guide '+str(index))
        gap=.00030+.0062*math.sin(math.pi*u)**1.15+.0060*float(smooth((u-.58)/.42))
        point=np.asarray(hit)+np.asarray(normal)*gap
        # The measured root belongs to the same ray; only the submillimetre
        # offset is blended in, without keeping the old backwards tangent.
        if t==0.:point=root.copy()
        out.append(point);max_gap=max(max_gap,gap)
    result=resample(np.asarray(out),count);result[0]=root
    return result,{'kind':'reference-front-lock','role':kind,'lockIndex':k,'tip':result[-1].tolist(),'maximumDesignedSkinGapMeters':max_gap}

def wisp_center(side,kind,project_ray):
    desired=np.array([side*(.066+.002*kind),-.065,1.479-.006*kind]);hit,normal=project_ray(normalize(desired-CENTER));root=np.asarray(hit)+np.asarray(normal)*.00035
    endpoint=np.array([side*(.049+.006*kind),-.049-.005*kind,(1.315,1.346,1.374)[kind]])
    c1=root+np.array([side*.008,-.010,-.040]);c2=np.array([side*(.075+.001*kind),-.075,endpoint[2]+.022])
    t=np.linspace(0.,1.,100)[:,None];p=(1-t)**3*root+3*(1-t)**2*t*c1+3*(1-t)*t*t*c2+t**3*endpoint
    return resample(p,28)


class OfflineRadialSurface:
    """Exact exported-triangle ray tests with six small directional grids."""
    def __init__(self,vertices,triangles):
        self.tri=np.asarray(vertices)[triangles];relative=self.tri-CENTER
        self.grids={};self.cell=.035;self.e1=self.tri[:,1]-self.tri[:,0];self.e2=self.tri[:,2]-self.tri[:,0]
        self.s=CENTER-self.tri[:,0];self.q=np.cross(self.s,self.e1)
        self.normals=np.cross(self.e1,self.e2);self.normals/=np.maximum(np.linalg.norm(self.normals,axis=1,keepdims=True),1e-12)
        for axis in range(3):
            other=[k for k in range(3) if k!=axis]
            for sign in (-1,1):
                den=relative[:,:,axis]*sign;ids=np.flatnonzero(np.min(den,axis=1)>.001);uv=relative[ids][:,:,other]/den[ids,:,None]
                lo=uv.min(1);hi=uv.max(1);valid=(lo[:,0]<1.05)&(lo[:,1]<1.05)&(hi[:,0]>-1.05)&(hi[:,1]>-1.05)
                ids=ids[valid];lo=np.floor(np.clip(lo[valid],-1.05,1.05)/self.cell).astype(int);hi=np.floor(np.clip(hi[valid],-1.05,1.05)/self.cell).astype(int);grid={}
                for i,l,h in zip(ids,lo,hi):
                    for u in range(l[0],h[0]+1):
                        for v in range(l[1],h[1]+1):grid.setdefault((u,v),[]).append(int(i))
                self.grids[(axis,sign)]=grid
    def __call__(self,direction):
        d=normalize(direction);axis=int(np.argmax(abs(d)));sign=1 if d[axis]>0 else -1;other=[k for k in range(3) if k!=axis]
        uv=d[other]/abs(d[axis]);ids=self.grids[(axis,sign)].get(tuple(np.floor(uv/self.cell).astype(int)),[])
        if not ids:return None,None
        ids=np.asarray(ids);h=np.cross(np.broadcast_to(d,self.e2[ids].shape),self.e2[ids]);det=np.sum(self.e1[ids]*h,axis=1);safe=np.abs(det)>1e-12
        inv=np.zeros_like(det);inv[safe]=1./det[safe];u=np.sum(self.s[ids]*h,axis=1)*inv;v=np.sum(self.q[ids]*d,axis=1)*inv;t=np.sum(self.e2[ids]*self.q[ids],axis=1)*inv
        valid=safe&(u>=-1e-7)&(v>=-1e-7)&(u+v<=1.+1e-7)&(t>0.)
        if not valid.any():return None,None
        # Outermost head shell, equivalent to casting inward from free air.
        # The human mesh includes internal eye/mouth surfaces that are not a
        # valid first hit for an outward ray starting inside the cranium.
        ix=np.flatnonzero(valid)[np.argmax(t[valid])];normal=self.normals[ids[ix]].copy()
        if np.dot(normal,d)<0:normal=-normal
        return CENTER+d*t[ix],normal
