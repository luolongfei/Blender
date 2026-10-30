"""Actual continuous old-wrist core / hand surface contact inspection.

The core is explicitly clipped in unposed coordinates; the normal proximal
arm beyond the wrist is outside this local containment scope. Clipping creates
pieces of existing surface triangles, never artificial cap faces. One micron
is solely a floating-point contact tolerance, not an anatomical clearance.
"""
import numpy as np

TOLERANCE=1e-6
LEGACY_BUFFER=.00015


def _point_segment_squared(p,a,b):
    direction=b-a;length=np.einsum('ij,ij->i',direction,direction)
    t=np.clip(np.einsum('ij,ij->i',p-a,direction)/np.maximum(length,1e-30),0,1)
    delta=p-(a+t[:,None]*direction);return np.einsum('ij,ij->i',delta,delta)


def _point_triangle_squared(p,tri):
    a,b,c=tri[:,0],tri[:,1],tri[:,2];u=b-a;v=c-a;w=p-a
    uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
    wu=np.einsum('ij,ij->i',w,u);wv=np.einsum('ij,ij->i',w,v);den=uu*vv-uv*uv
    s=(wu*vv-wv*uv)/np.maximum(den,1e-30);t=(wv*uu-wu*uv)/np.maximum(den,1e-30)
    normal=np.cross(u,v);length2=np.einsum('ij,ij->i',normal,normal)
    plane=np.einsum('ij,ij->i',w,normal)**2/np.maximum(length2,1e-30)
    edge=np.minimum.reduce([_point_segment_squared(p,a,b),_point_segment_squared(p,b,c),_point_segment_squared(p,c,a)])
    inside=(s>=0)&(t>=0)&(s+t<=1)&(den>1e-30)
    return np.where(inside,np.minimum(edge,plane),edge)


def _segment_segment_squared(a,b,c,d):
    u=b-a;v=d-c;w=a-c
    aa=np.einsum('ij,ij->i',u,u);bb=np.einsum('ij,ij->i',u,v);cc=np.einsum('ij,ij->i',v,v)
    dd=np.einsum('ij,ij->i',u,w);ee=np.einsum('ij,ij->i',v,w);den=aa*cc-bb*bb
    s=(bb*ee-cc*dd)/np.maximum(den,1e-30);t=(aa*ee-bb*dd)/np.maximum(den,1e-30)
    delta=w+s[:,None]*u-t[:,None]*v;interior=np.einsum('ij,ij->i',delta,delta)
    valid=(den>1e-30)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    boundary=np.minimum.reduce([_point_segment_squared(a,c,d),_point_segment_squared(b,c,d),
                                _point_segment_squared(c,a,b),_point_segment_squared(d,a,b)])
    return np.where(valid,np.minimum(interior,boundary),boundary)


def triangle_pair_metrics(a,b,tolerance=TOLERANCE):
    """Closed triangle intersection plus exact feature-distance candidates.

    SAT includes both face normals, 9 edge cross-products, and in-plane edge
    normals so coplanar disjoint/overlapping triangles are also distinguished.
    All points/edges participate; triangle-centroid sampling is not used.
    """
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(3,3):raise ValueError('Paired triangles required')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):raise RuntimeError('Non-finite contact geometry')
    ea=np.roll(a,-1,axis=1)-a;eb=np.roll(b,-1,axis=1)-b
    na=np.cross(ea[:,0],-ea[:,2]);nb=np.cross(eb[:,0],-eb[:,2])
    if np.any(np.linalg.norm(na,axis=1)<1e-16) or np.any(np.linalg.norm(nb,axis=1)<1e-16):
        raise RuntimeError('Degenerate actual triangle prevents continuous surface certification')
    axes=np.concatenate((na[:,None],nb[:,None],np.cross(ea[:,:,None],eb[:,None,:]).reshape(-1,9,3),
                         np.cross(na[:,None],ea),np.cross(nb[:,None],eb)),axis=1)
    norms=np.linalg.norm(axes,axis=2);valid=norms>1e-30;unit=axes/np.maximum(norms[:,:,None],1e-30)
    pa=np.einsum('nvc,nac->nav',a,unit);pb=np.einsum('nvc,nac->nav',b,unit)
    gap=np.maximum(pb.min(2)-pa.max(2),pa.min(2)-pb.max(2));gap[~valid]=-np.inf
    separated=gap.max(1)>1e-12
    d2=np.full(len(a),np.inf)
    for i in range(3):
        d2=np.minimum(d2,_point_triangle_squared(a[:,i],b));d2=np.minimum(d2,_point_triangle_squared(b[:,i],a))
        for j in range(3):d2=np.minimum(d2,_segment_segment_squared(a[:,i],a[:,(i+1)%3],b[:,j],b[:,(j+1)%3]))
    d2[~separated]=0
    return {'intersects':~separated,'distance':np.sqrt(np.maximum(d2,0)),
            'contactWithinTolerance':d2<=tolerance*tolerance}


def clipped_core(unposed,actual,side):
    sign=1 if side=='L' else -1;rest=unposed['vertices'];world=actual['vertices'];triangles=actual['triangles']
    transformed=rest.copy();transformed[:,0]*=sign
    lower=np.asarray((.602,-.06,1.175));upper=np.asarray((.631,.06,1.295))
    positions=transformed[triangles];mask=np.all(positions.max(1)>=lower,axis=1)&np.all(positions.min(1)<=upper,axis=1)
    output=[];original=[];dropped=0;dropped_area=0.
    for tri_index in np.flatnonzero(mask):
        ids=triangles[tri_index];polygon=[(transformed[v].copy(),world[v].copy()) for v in ids]
        for axis in range(3):
            for boundary,inward in ((lower[axis],1),(upper[axis],-1)):
                if not polygon:break
                clipped=[]
                for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                    da=inward*(a[0][axis]-boundary);db=inward*(b[0][axis]-boundary)
                    if da>=0:clipped.append(a)
                    if (da>=0)!=(db>=0):
                        t=da/(da-db);clipped.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
                polygon=clipped
        for i in range(1,len(polygon)-1):
            tri=np.asarray((polygon[0][1],polygon[i][1],polygon[i+1][1]));area=.5*np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
            if area==0:dropped+=1;continue
            output.append(tri);original.append(int(tri_index))
    if not 100<len(output)<10000:raise RuntimeError('Unexpected clipped continuous wrist-core surface count: '+str(len(output)))
    return np.asarray(output),np.asarray(original,np.int32),{'triangles':len(output),'exactZeroAreaClippingTrianglesSkipped':dropped,
        'skippedSliverAreaSquareMeters':float(dropped_area),'restDomain':{'absoluteX':[.602,.631],'y':[-.06,.06],'z':[1.175,1.295]},
        'boundaryMeaning':'Only existing surface pieces inside this unposed box. No artificial clipping caps; normal proximal arm extension is outside the tested core.'}


def continuous_contact(core,hand,original_triangles,tolerance=TOLERANCE):
    from mathutils.bvhtree import BVHTree
    vertices=core.reshape(-1,3);indices=np.arange(len(vertices),dtype=np.int32).reshape(-1,3)
    a=BVHTree.FromPolygons(vertices.tolist(),indices.tolist(),all_triangles=True,epsilon=tolerance)
    b=BVHTree.FromPolygons(hand['vertices'].tolist(),hand['triangles'].tolist(),all_triangles=True,epsilon=tolerance)
    pairs=a.overlap(b)
    if len(pairs)>100000:raise RuntimeError('Local contact broad phase exceeded 100000-pair bound')
    if not pairs:return {'complete':True,'broadPhasePairs':0,'surfaceIntersectionPairs':0,'surfaceContactPairsWithinNumericalTolerance':0,
        'numericalToleranceMeters':tolerance,'closestBroadPhaseSurfaceDistanceMeters':None,'zeroContactsAtTolerance':True}
    pairs=np.asarray(pairs,np.int32);metrics=triangle_pair_metrics(core[pairs[:,0]],hand['vertices'][hand['triangles'][pairs[:,1]]],tolerance)
    hits=np.flatnonzero(metrics['contactWithinTolerance']);intersections=np.flatnonzero(metrics['intersects'])
    return {'complete':True,'broadPhasePairs':len(pairs),'surfaceIntersectionPairs':len(intersections),
        'surfaceContactPairsWithinNumericalTolerance':len(hits),'numericalToleranceMeters':tolerance,
        'closestBroadPhaseSurfaceDistanceMeters':float(metrics['distance'].min()),'zeroContactsAtTolerance':not len(hits),
        'contactDiagnosticsTruncated':len(hits)>128,
        'contactPairsForDiagnosis':[{'originalBodyTriangle':int(original_triangles[pairs[i,0]]),'handTriangle':int(pairs[i,1]),
            'distanceMeters':float(metrics['distance'][i]),'clippedBodyTriangleWorld':core[pairs[i,0]].tolist(),
            'handTriangleWorld':hand['vertices'][hand['triangles'][pairs[i,1]]].tolist()} for i in hits[:128]],
        'method':'Expanded BVH broad phase, full closed-triangle SAT including coplanar axes, point/triangle and every edge/edge distance.'}


def assess_sides(bpy,body,adapter,wrist):
    arms=[m for m in body.modifiers if m.type=='ARMATURE']
    if len(arms)!=1 or not arms[0].show_viewport:raise RuntimeError('Expected one active unchanged body Armature')
    try:
        arms[0].show_viewport=False;bpy.context.view_layer.update();unposed=wrist['_body_surface'](bpy,body)
    finally:
        arms[0].show_viewport=True;bpy.context.view_layer.update()
    actual=wrist['_body_surface'](bpy,body);topology=wrist['_same_body_topology'](unposed,actual);rows=[]
    for side in ('L','R'):
        sign=1 if side=='L' else -1;p=unposed['vertices']
        core=(p[:,0]*sign>=.602)&(p[:,0]*sign<=.631)&(np.abs(p[:,1])<.06)&(np.abs(p[:,2]-1.235)<.06)
        if not 50<int(core.sum())<3000:raise RuntimeError('Unexpected original wrist core sampling count')
        hand=adapter['_surface'](bpy,bpy.data.objects['Hand '+side+' · continuous anatomy'])
        row=wrist['_inside_samples'](actual['vertices'][core],hand)
        triangles,original,domain=clipped_core(unposed,actual,side);contacts=continuous_contact(triangles,hand,original)
        for pair in contacts.get('contactPairsForDiagnosis',[]):
            ids=actual['triangles'][pair['originalBodyTriangle']]
            pair['bodyEvaluatedVertexIds']=ids.tolist();pair['bodyOriginalTriangleUnposed']=unposed['vertices'][ids].tolist()
            pair['bodyOriginalTriangleWorld']=actual['vertices'][ids].tolist()
        row.update({'side':side,'actualBodyTopologyCheck':topology,'actualBodyVertexIndices':np.flatnonzero(core).tolist(),
            'continuousSurfaceDomain':domain,'continuousSurfaceContact':contacts,
            'legacySuggestedBufferMeters':LEGACY_BUFFER,'legacySuggestedBufferSatisfied':row['minimumSurfaceDistanceMeters']>=LEGACY_BUFFER,
            'legacySuggestedBufferPolicy':'Previously self-selected engineering buffer, with no anatomical or target-engine requirement. Historical diagnostic only; not a physical failure criterion.',
            'numericalTolerancePolicy':'1 micrometre accounts for float32 mesh coordinates near metre scale and geometric arithmetic. It is not a prescribed skin/wrist gap.',
            'physicalCoreGatePassed':not row['outsideByThreeRayMajority'] and contacts['zeroContactsAtTolerance'],
            'scope':'Only old wrist core inside the new closed hand, in three finite poses. No proximal arm containment, globally welded wrist, all-pose or visual acceptance claim.'})
        rows.append(row)
    return rows
