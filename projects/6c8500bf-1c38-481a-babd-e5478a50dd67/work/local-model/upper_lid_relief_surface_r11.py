"""Pure actual-surface fields for a limited outer upper-lid crown reduction.

The first vertical-band draft left a small bend. This version instead measures
space distance to the observed visible edge and the actual triangulated globe,
then smooths the bounded field along the real evaluated surface. It never
identifies a coarse socket ring or a front-visible ray as a certified wet line.
"""
import numpy as np


def smooth(value):
    t=np.clip(value,0.,1.)
    return t*t*t*(t*(t*6.-15.)+10.)


def normals(world,triangles):
    q=np.asarray(world)[np.asarray(triangles)]
    n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);out=np.zeros_like(world,dtype=float)
    for j in range(3):np.add.at(out,triangles[:,j],n)
    return out/np.maximum(np.linalg.norm(out,axis=1)[:,None],1e-15)


def closest_surface(points,world,triangles,batch_size=96):
    """Exact point-to-triangle nearest distance and face-normal signed side.

    A convex actual globe has consistent outside signs. Projection to triangle
    interiors and all three finite edges are considered, not vertex distances
    or a fitted sphere. The signed quantity is used only for side protection.
    """
    points=np.asarray(points,float);tri=np.asarray(world,float)[np.asarray(triangles,np.int32)]
    a=tri[:,0];u=tri[:,1]-a;v=tri[:,2]-a
    normal=np.cross(u,v);area=np.linalg.norm(normal,axis=1)
    if np.any(area<1e-14):raise RuntimeError('Degenerate actual globe triangle')
    normal/=area[:,None];uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
    den=uu*vv-uv*uv
    distances=[];signed=[];closest=[];face_ids=[]
    for start in range(0,len(points),batch_size):
        p=points[start:start+batch_size];relative=p[:,None,:]-a[None,:,:]
        plane=np.einsum('nfi,fi->nf',relative,normal)
        pu=np.einsum('nfi,fi->nf',relative,u);pv=np.einsum('nfi,fi->nf',relative,v)
        s=(vv*pu-uv*pv)/den;t=(uu*pv-uv*pu)/den
        inside=(s>=0)&(t>=0)&(s+t<=1)
        best=p[:,None,:]-plane[:,:,None]*normal[None,:,:]
        d2=plane*plane;d2[~inside]=np.inf
        for x,y in ((0,1),(1,2),(2,0)):
            origin=tri[:,x];edge=tri[:,y]-origin
            factor=np.clip(np.einsum('nfi,fi->nf',p[:,None,:]-origin[None,:,:],edge)/np.einsum('fi,fi->f',edge,edge),0,1)
            q=origin[None,:,:]+factor[:,:,None]*edge[None,:,:]
            e=np.sum((p[:,None,:]-q)**2,axis=2);take=e<d2
            best[take]=q[take];d2[take]=e[take]
        indices=np.argmin(d2,axis=1);rows=np.arange(len(p));q=best[rows,indices]
        distances.append(np.sqrt(d2[rows,indices]));signed.append(np.einsum('ni,ni->n',p-q,normal[indices]));closest.append(q);face_ids.append(indices)
    if not len(points):return {'distance':np.empty(0),'signed':np.empty(0),'point':np.empty((0,3)),'triangle':np.empty(0,np.int32)}
    return {'distance':np.concatenate(distances),'signed':np.concatenate(signed),
        'point':np.concatenate(closest),'triangle':np.concatenate(face_ids)}


def _edge_distance(points,line):
    result=np.empty(len(points),float)
    a=line[:-1];edge=line[1:]-line[:-1];length2=np.einsum('ij,ij->i',edge,edge)
    for start in range(0,len(points),192):
        p=points[start:start+192]
        t=np.clip(np.einsum('nfi,fi->nf',p[:,None,:]-a[None,:,:],edge)/np.maximum(length2,1e-20),0,1)
        q=a[None,:,:]+t[:,:,None]*edge[None,:,:]
        result[start:start+len(p)]=np.sqrt(np.sum((p[:,None,:]-q)**2,axis=2).min(1))
    return result


def plan(head_world,head_triangles,eyes,eye_arrays,region_protected=None,maximum=.00030):
    world=np.asarray(head_world,float);tri=np.asarray(head_triangles,np.int32)
    if not .0001<=maximum<=.0004:raise ValueError('Bounded explicit outer-crown reduction required')
    n=normals(world,tri);protected=np.zeros(len(world),bool) if region_protected is None else np.asarray(region_protected,bool)
    edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    ends=edges.ravel();others=edges[:,::-1].ravel();degree=np.maximum(np.bincount(ends,minlength=len(world)),1)
    weights={};details={};total=np.zeros_like(world);arrays={}
    for side in ('L','R'):
        contour=[p for p in eyes[side]['contour'] if not p['upper']['unboundedAtGlobeRim'] and p['upper'].get('headOutside120Micrometers')]
        if len(contour)<40:raise RuntimeError('Actual neutral upper boundary is incomplete: '+side)
        xs=np.asarray([p['x'] for p in contour]);zs=np.asarray([p['upper']['worldZ'] for p in contour])
        near=np.asarray([p['upper']['headOutside120Micrometers']['world'] for p in contour])
        edge=np.asarray([(p['upper'].get('headAtVisibilityTransition') or p['upper']['headOutside120Micrometers'])['world'] for p in contour])
        height=world[:,2]-np.interp(world[:,0],xs,zs)
        front=world[:,1]-np.interp(world[:,0],xs,near[:,1])
        candidates=(world[:,0]>xs.min()+.0025)&(world[:,0]<xs.max()-.0025)&(height>.00015)&(height<.0048)&(abs(front)<.005)&(n[:,1]<-.15)&~protected
        ids=np.flatnonzero(candidates)
        globe=eye_arrays[side]
        closest=closest_surface(world[ids],globe['world'],globe['triangles'])
        margin=closest['distance'];positive=closest['signed']>0
        edge_distance=_edge_distance(world[ids],edge)
        local=smooth((height[ids]-.00015)/.0008)*(1-smooth((height[ids]-.0022)/.0026))
        local*=smooth((world[ids,0]-xs.min()-.0025)/.003)*smooth((xs.max()-.0025-world[ids,0])/.003)
        local*=smooth((margin-.0007)/.0005)*positive
        local*=smooth((edge_distance-.0008)/.0006)
        local*=smooth((-n[ids,1]-.15)/.6)
        local*=1-smooth((abs(front[ids])-.003)/.002)
        w=np.zeros(len(world));w[ids]=local;envelope=w.copy()
        # Smooth the real-surface field, never the existing head coordinates.
        # The original C2 envelope caps the average near each protection edge.
        for _ in range(10):
            average=np.bincount(ends,weights=w[others],minlength=len(world))/degree
            w=np.minimum(.5*w+.5*average,envelope*1.15)
        if np.any(w[protected]):raise RuntimeError('Protected actual region entered relief')
        changed=w>1e-8
        if not 80<int(changed.sum())<3000:raise RuntimeError('Unexpected measured crown domain '+side+': '+str(changed.sum()))
        # The native modifier additionally limits against actual current globe
        # distance at every expression, not just this neutral plan.
        amount=maximum*w
        full_margin=np.zeros(len(world));full_margin[ids]=margin
        amount=np.minimum(amount,np.maximum(full_margin-.0007,0))
        delta=-n*amount[:,None]
        if np.any(np.linalg.norm(total[changed],axis=1)>0):raise RuntimeError('Left/right crown fields overlap')
        total+=delta;weights[side]=w.astype(np.float32)
        after=closest_surface((world+delta)[changed],globe['world'],globe['triangles'])
        if after['distance'].min()<.0007-2e-8 or after['signed'].min()<=0:raise RuntimeError('Neutral relief consumes actual globe protection')
        map_local=np.full(len(world),-1,np.int32);map_local[ids]=np.arange(len(ids))
        details[side]={'affectedEvaluatedVertices':int(changed.sum()),'actualMaximumNeutralDisplacementMeters':float(amount.max()),
            'minimumActualGlobeDistanceBeforeMeters':float(margin[map_local[np.flatnonzero(changed)]].min()),
            'minimumActualGlobeDistanceAfterMeters':float(after['distance'].min()),
            'minimumDistanceToObservedBoundaryMeters':float(edge_distance[map_local[np.flatnonzero(changed)]].min()),
            'visibleBoundaryProtectedSpaceDistanceMeters':.0008,'actualGlobeProtectedDistanceMeters':.0007,
            'canthalXProtectionMeters':.0025,'diffusionIterations':10,
            'scope':'Front-facing external crown only; actual socket/inner-face regions supplied by the installer remain zero. Observed front boundary is a protection proxy, not a claimed anatomical wet-line tracing.'}
        arrays['weight_'+side]=w.astype(np.float32)
    changed=np.linalg.norm(total,axis=1)>1e-12;active_tri=tri[changed[tri].any(1)]
    p=world[active_tri];q=(world+total)[active_tri]
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1);valid=a0>1e-14
    align=np.sum(n0[valid]*n1[valid],axis=1)/(a0[valid]*a1[valid]);ratio=a1[valid]/a0[valid]
    if align.min()<=0 or ratio.min()<.5 or ratio.max()>1.6:raise RuntimeError('Actual-surface relief draft folds/strains')
    arrays.update(neutralDeltaWorld=total.astype(np.float32),neutralNormalWorld=n.astype(np.float32),regionProtected=protected)
    return weights,arrays,{'eyes':details,'requestedMaximumMeters':maximum,'affectedEvaluatedVertices':int(changed.sum()),
        'minimumNeutralTriangleNormalAlignment':float(align.min()),'neutralTriangleAreaRatioRange':[float(ratio.min()),float(ratio.max())],
        'fieldSource':'Current actual neutral head and exact triangulated globes. No old whole-ROI geometry lock.',
        'appearanceAccepted':False,'physicalWetLineCertified':False}
