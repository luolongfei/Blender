"""Pure actual-surface upper-lid roll reduction for the open-eye candidate.

The existing neutral relief is already part of the observed input. This layer
works on the anterior outer roll rather than shrinking the eye or moving its
globe. Front-visible rim is a measured proxy, not an anatomical wet-line claim.
Outer endpoint convergence is a separate jointly reviewed field.
"""
from pathlib import Path
import runpy
import numpy as np

MAXIMUM=.00065
GLOBE_GAP=.00045
FADE_END=.5
MINIMUM_WALL=.00020
RAY_EPSILON=.000001


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def fade(blink):
    if blink>=FADE_END:return 0.
    return 1.-float(smooth(blink/FADE_END))


def inner_wall(points,world,triangles):
    """First actual +Y exit after each outer point; same ray as native GN."""
    index=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex'](world,triangles)
    distance=np.zeros(len(points));valid=np.zeros(len(points),bool)
    for row,point in enumerate(points):
        cell=tuple(np.floor((point[[0,2]]-index.origin)/index.cell_size).astype(np.int32))
        ids=np.asarray(index.buckets.get(cell,[]),np.int32)
        if not len(ids):continue
        q=point-index.faces[ids,0];det=index.determinant[ids]
        u=(q[:,0]*index.e2[ids,2]-q[:,2]*index.e2[ids,0])/det
        v=(index.e1[ids,0]*q[:,2]-index.e1[ids,2]*q[:,0])/det
        inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1.+1e-8)
        ids=ids[inside];y=index.faces[ids,0,1]+u[inside]*index.e1[ids,1]+v[inside]*index.e2[ids,1]
        ahead=(y>point[1]+RAY_EPSILON)&(y<point[1]+.020)
        ids=ids[ahead];y=y[ahead]
        if not len(ids):continue
        first=int(np.argmin(y))
        if index.normal_y[ids[first]]<=.1:continue
        distance[row]=y[first]-point[1];valid[row]=True
    return distance,valid


def apply(world,triangles,weights,eyes,blink=None,strength=1.):
    surface=api('upper_lid_relief_surface_r11');world=np.asarray(world,float)
    normal=surface['normals'](world,triangles);total=np.zeros_like(world);reports={}
    blink=blink or {'L':0.,'R':0.}
    if not 0<=strength<=1:raise ValueError('Explicit bounded roll strength required')
    for side in ('L','R'):
        factor=fade(float(blink[side]))*strength
        if factor==0:
            reports[side]={'changedPoints':0,'fade':factor,'maximumMeters':0.};continue
        selected=(np.asarray(weights[side])>0)&(normal[:,1]<-.10)
        ids=np.flatnonzero(selected);eye=eyes[side]
        closest=surface['closest_surface'](world[ids],eye['world'],eye['triangles'])
        requested=MAXIMUM*np.asarray(weights[side])[ids]*factor
        allowed=np.maximum(closest['distance']-GLOBE_GAP,0.)*(closest['signed']>0)
        wall,valid_wall=inner_wall(world[ids],world,triangles)
        amount=np.minimum(np.minimum(requested,allowed),np.maximum(wall-MINIMUM_WALL,0.)*valid_wall)
        total[ids,1]+=amount
        moved=amount>1e-10
        after=surface['closest_surface'](world[ids[moved]]+np.column_stack([np.zeros(moved.sum()),amount[moved],np.zeros(moved.sum())]),eye['world'],eye['triangles'])
        reports[side]={'changedPoints':int(moved.sum()),'fade':factor,'maximumMeters':float(amount.max()) if len(amount) else 0.,
            'minimumActualGlobeDistanceAfterMeters':float(after['distance'].min()) if moved.any() else None,
            'minimumSameRayInnerWallAfterMeters':float((wall-amount)[moved].min()) if moved.any() else None,
            'noValidInnerWallPoints':int((~valid_wall).sum()),
            'maximumRequestedMeters':float(requested.max()) if len(requested) else 0.}
        if moved.any() and (after['distance'].min()<GLOBE_GAP-2e-8 or after['signed'].min()<=0):raise RuntimeError('Outer roll consumed actual globe allowance')
    return total,reports


def plan(neutral_world,triangles,original_ids,eyes,contours):
    surface=api('upper_lid_relief_surface_r11');w=np.asarray(neutral_world,float);t=np.asarray(triangles,np.int32)
    normals=surface['normals'](w,t);weights={};details={}
    edges=np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
    ends=edges.ravel();other=edges[:,::-1].ravel();degree=np.maximum(np.bincount(ends,minlength=len(w)),1)
    for side in ('L','R'):
        rows=contours[side]['contour'] if isinstance(contours[side],dict) else contours[side]
        rows=[p for p in rows if not p['upper'].get('unboundedAtGlobeRim',False)]
        x=np.asarray([p['x'] for p in rows]);z=np.asarray([p['upper']['worldZ'] for p in rows])
        if len(x)<50 or np.any(np.diff(x)<=0):raise RuntimeError('Actual ordered neutral visible upper contour required')
        h=w[:,2]-np.interp(w[:,0],x,z)
        center=(eyes[side]['world'].min(0)+eyes[side]['world'].max(0))*.5
        envelope=smooth((h-.00012)/.00115)*(1.-smooth((h-.00135)/.00265))
        envelope*=smooth((w[:,0]-x.min()-.0018)/.002)
        envelope*=smooth((x.max()-.0018-w[:,0])/.002)
        envelope*=smooth((-normals[:,1]-.10)/.50)
        envelope*=1.-smooth((w[:,1]-(center[1]-.006))/.003)
        envelope[(h<=.00012)|(h>=.004)|(w[:,1]>=center[1]-.003)|(normals[:,1]>=-.1)]=0.
        values=envelope.copy()
        for _ in range(10):
            avg=np.bincount(ends,weights=values[other],minlength=len(w))/degree
            values=np.minimum(.65*values+.35*avg,envelope)
        values[values<1e-8]=0.;weights[side]=values.astype(np.float32)
        details[side]={'weightedActualPoints':int((values>0).sum()),'upperRimProxyProtectedVerticalMeters':.00012,
            'lateralEndpointExcludedMeters':.0018,'rollBandAboveUpperVisibleBoundaryMeters':[.00012,.004],
            'scope':'Anterior outer surface only; its current normal is gated again at execution. Inner lining and actual endpoint convergence are handled separately.'}
    if np.any((weights['L']>0)&(weights['R']>0)):raise RuntimeError('Upper-roll side supports overlap')
    delta,actual=apply(w,t,weights,eyes)
    active=np.linalg.norm(delta,axis=1)>1e-10
    faces=t[active[t].any(1)];a=w[faces];b=(w+delta)[faces]
    n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1);good=l0>1e-14
    cosine=np.sum(n0[good]*n1[good],axis=1)/(l0[good]*l1[good]);ratio=l1[good]/l0[good]
    quality={'minimumNormalAlignment':float(cosine.min()),'triangleAreaRatioRange':[float(ratio.min()),float(ratio.max())],'nonpositiveNormals':int((cosine<=0).sum())}
    if quality['nonpositiveNormals'] or cosine.min()<.5 or ratio.min()<.45 or ratio.max()>1.8:raise RuntimeError('Outer-roll draft folds/strains: '+str(quality))
    arrays={'originalVertexIds':np.asarray(original_ids,np.int32),'neutralWorld':w,'triangles':t,
        'weight_L':weights['L'],'weight_R':weights['R'],'neutralDeltaWorld':delta.astype(np.float32)}
    return weights,arrays,{'maximumRequestedMeters':MAXIMUM,'minimumGlobeDistanceMeters':GLOBE_GAP,'minimumInnerWallMeters':MINIMUM_WALL,'blinkFadeEnd':FADE_END,
        'sides':details,'neutralChecks':actual,'neutralQuality':quality,'appearanceAccepted':False,
        'limits':['The front-visible rim proxy is not a certified anatomical wet-line trace.',
            'This draft must be evaluated together with outer-end convergence and the current inner contact layer.',
            'Neutral/early blink are intentionally changed; obsolete midpath contour targets must be rebuilt from the new actual observation.']}
