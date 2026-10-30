"""Surface-directed front guides; pure NumPy layout with explicit ray queries.

This replaces rotation of long folded donor paths around a short root-tip
vector. Original roots and local take-off direction are retained, while the
front length is authored on the real head surface instead of its chord.
"""
import math
import numpy as np


def smooth(t):
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)


def resample(points,count):
    p=np.asarray(points,dtype=float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if arc[-1]<1e-8:raise RuntimeError('Degenerate front path')
    return np.column_stack([np.interp(np.linspace(0.,arc[-1],count),arc,p[:,j]) for j in range(3)])


def design(path,source_root,index,weight,count,project_front):
    """project_front(x,z) returns actual front surface (point, normal)."""
    path=np.asarray(path,dtype=float);root=path[0].copy()
    x=float(np.clip(root[0]+.0030*math.sin(index*1.917+source_root[1]*3.),-.060,.060))
    outer=float(smooth((abs(x)-.034)/.026))
    fringe=weight>=.55 or abs(float(root[0]))<.040
    if fringe:
        # Classification confidence is not a length factor. A fringe may not
        # end below the eye merely because its front weight is only 0.6.
        z=1.449-.007*outer+.0024*math.sin(index*2.39996)
    else:
        x=math.copysign(.061+.004*math.sin(index*1.1),x if abs(x)>.005 else source_root[0]+.001)
        z=1.355+.062*float(smooth((weight-.12)/.43))
    endpoint,normal=project_front(x,z)
    if endpoint is None:
        if fringe:raise RuntimeError('Front target missed actual skin')
        endpoint=np.array([x,-.075,z])
    else:
        endpoint=np.asarray(endpoint,dtype=float)+np.asarray(normal,dtype=float)*(.0020+.0006*math.sin(index*1.7)**2)
    # A compact cubic in projected face coordinates preserves the first local
    # root direction, without importing the author's later loops and reversals.
    tangent=path[1]-root;tangent/=max(np.linalg.norm(tangent),1e-12)
    distance=float(np.linalg.norm(endpoint-root))
    handle=min(.012,distance*.20)
    c1=root+tangent*handle
    c1[2]=min(c1[2],root[2]+.0015)
    c1[0]=root[0]+np.clip(c1[0]-root[0],-.006,.006)
    c2=endpoint.copy();c2[2]+=distance*.27
    c2[0]=endpoint[0]+(root[0]-endpoint[0])*.18
    if not fringe:
        # Fit the depth handle to one real mid-forehead/temple surface point.
        # A root-to-low-tip depth chord otherwise cuts through the convex face.
        mid=.125*root+.375*c1+.375*c2+.125*endpoint
        anchor,anchor_normal=project_front(float(mid[0]),float(mid[2]))
        if anchor is not None:
            depth=float(anchor[1]+anchor_normal[1]*.0045)
            c2[1]=min(c2[1],(depth-.125*root[1]-.375*c1[1]-.125*endpoint[1])/.375)
    u=np.linspace(0.,1.,max(80,count*4));q=((1-u)**3)[:,None]*root+3*((1-u)**2*u)[:,None]*c1+3*((1-u)*u*u)[:,None]*c2+(u**3)[:,None]*endpoint
    built=[root.copy()];maximum=0.;rays=0
    for j in range(1,len(q)):
        p,n=project_front(float(q[j,0]),float(q[j,2]))
        if p is None:
            # The complete forward ray misses the head: this XZ column is
            # outside its outline. Keep the short crown/side free-air arc;
            # the common actual-skin and eye checks still apply afterwards.
            built.append(q[j]);continue
        p=np.asarray(p,dtype=float);n=np.asarray(n,dtype=float)
        gap=.00030+.0045*math.sin(math.pi*u[j])**1.15+.0017*float(smooth((u[j]-.70)/.30))
        surface=p+n*gap
        # At the exact root, preserve the original surface offset. A smooth
        # 12% transition avoids a spur between an author root and a front ray.
        blend=float(smooth(u[j]/.12))
        # Side curtains hang in front of or alongside the cheek. A front ray
        # that finds a recessed cheek/ear must not pull an already external
        # strand backwards to it, introducing a 3-5 cm lateral kink.
        target=q[j] if (not fringe and q[j,1]<=surface[1]) else surface
        result=q[j]*(1.-blend)+target*blend
        maximum=max(maximum,float(np.linalg.norm(target-q[j])))
        built.append(result);rays+=1
    if maximum>.025:
        raise RuntimeError('Surface-directed front path exceeds original 25 mm layout correction limit '+str({'guide':index,'meters':maximum}))
    result=resample(np.asarray(built),count);result[0]=root;result[-1]=endpoint
    return result,{'kind':'fringe' if fringe else 'temple-curtain','sourceFrontWeight':float(weight),
        'construction':'Actual front-surface cubic with original local root tangent; no root-tip rotation or stretch of folded author path',
        'actualFrontSurfaceSamples':rays,'maximumStyleSurfaceDistanceMeters':maximum,
        'targetTip':endpoint.tolist(),'rootTakeoffHandleMeters':handle}
