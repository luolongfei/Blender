"""Offline geometric audit of a neck annulus; not a visual acceptance test."""
import numpy as np
from itertools import product


def triangle_intersections(vertices,triangles,cell=.004):
    """Report strict proper triangle intersections after spatial-grid culling.

    Shared-edge pairs are excluded. Shared-vertex pairs remain testable.
    Coplanar positive-area overlaps are reported separately from crossings.
    """
    p=np.asarray(vertices)[triangles]
    lo=np.floor(p.min(1)/cell).astype(int);hi=np.floor(p.max(1)/cell).astype(int)
    buckets={}
    for i,(low,high) in enumerate(zip(lo,hi)):
        for key in product(*(range(low[d],high[d]+1) for d in range(3))):
            buckets.setdefault(key,[]).append(i)
    pairs=set()
    for group in buckets.values():
        for k,i in enumerate(group):
            for j in group[k+1:]:pairs.add((i,j))
    pairs=np.asarray(sorted(pairs),np.int32)
    shared=np.sum(triangles[pairs[:,0],:,None]==triangles[pairs[:,1],None,:],axis=(1,2))
    adjacent=shared>=2
    pairs=pairs[~adjacent]
    aa=p[pairs[:,0]];bb=p[pairs[:,1]]
    overlap=np.all(aa.min(1)<=bb.max(1)+1e-12,axis=1)&np.all(bb.min(1)<=aa.max(1)+1e-12,axis=1)
    pairs=pairs[overlap];aa=aa[overlap];bb=bb[overlap]
    def segment_test(seg,target):
        origin=seg[:,0];direction=seg[:,1]-origin
        edge1=target[:,1]-target[:,0];edge2=target[:,2]-target[:,0]
        h=np.cross(direction,edge2);det=np.einsum('ij,ij->i',edge1,h)
        valid=np.abs(det)>1e-18;inv=np.zeros(len(det));inv[valid]=1/det[valid]
        s=origin-target[:,0];u=inv*np.einsum('ij,ij->i',s,h)
        q=np.cross(s,edge1);v=inv*np.einsum('ij,ij->i',direction,q);t=inv*np.einsum('ij,ij->i',edge2,q)
        eps=1e-7
        return valid&(u>eps)&(v>eps)&(u+v<1-eps)&(t>eps)&(t<1-eps)
    hit=np.zeros(len(pairs),bool)
    for a,b in ((aa,bb),(bb,aa)):
        for k in range(3):hit|=segment_test(a[:,[k,(k+1)%3]],b)
    na=np.cross(aa[:,1]-aa[:,0],aa[:,2]-aa[:,0]);nb=np.cross(bb[:,1]-bb[:,0],bb[:,2]-bb[:,0])
    na/=np.linalg.norm(na,axis=1)[:,None];nb/=np.linalg.norm(nb,axis=1)[:,None]
    coplanar=(np.abs(np.einsum('ij,ij->i',na,nb))>1-1e-9)&(np.abs(np.einsum('ij,ij->i',na,bb[:,0]-aa[:,0]))<1e-10)
    overlaps=[]
    def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
    for index in np.flatnonzero(coplanar):
        dims=[k for k in range(3) if k!=int(np.argmax(np.abs(na[index])))];subject=list(aa[index][:,dims]);clip=bb[index][:,dims]
        orientation=np.sign(cross2(clip[1]-clip[0],clip[2]-clip[0]))
        for k in range(3):
            edge=clip[(k+1)%3]-clip[k];origin=clip[k];output=[]
            if not subject:break
            prev=subject[-1];dp=cross2(edge,prev-origin)*orientation
            for current in subject:
                dc=cross2(edge,current-origin)*orientation
                if (dc>=0)!=(dp>=0):output.append(prev+(current-prev)*dp/(dp-dc))
                if dc>=0:output.append(current)
                prev=current;dp=dc
            subject=output
        if len(subject)>=3:
            poly=np.asarray(subject);area=abs(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1)))*.5
            if area>1e-12:overlaps.append(pairs[index])
    return {'strictCrossingPairs':pairs[hit], 'coplanarOverlapPairs':np.asarray(overlaps,np.int32).reshape(-1,2), 'nonAdjacentBoundingBoxPairs':int(len(pairs)),
        'method':'Spatial grid, six strict segment/triangle tests and coplanar polygon clipping; shared edges excluded, shared vertices still tested. Pure edge contacts are not called positive-area intersections.'}
