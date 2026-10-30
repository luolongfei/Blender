"""Small current-head surface audit for five explicitly excepted gather starts.

Call audit(arrays, head_prefix='actual-current-head') inside the production
wrapper. No scene edits, render, Blender launch or save. Input coordinates and
native radii are actual world metres. Baseline before curves are measured only;
no unavailable before-native paths are synthesized.
"""
import hashlib
from pathlib import Path
import numpy as np

CURVE_IDS=(10330,10527,10540,10541,10542)
LIMIT=-.0005
ROOT_ARC=.001

def _put(arrays,key,value):
    if key in arrays:raise RuntimeError('Takeoff audit output already exists: '+key)
    arrays[key]=np.ascontiguousarray(value)

def _radii(arrays,prefix,starts,lengths,*,native=False):
    key=prefix+'_radius_world' if prefix+'_radius_world' in arrays else prefix+'_radius'
    if key not in arrays:raise RuntimeError('Actual world radius is required: '+prefix)
    values=np.asarray(arrays[key],float)
    if len(values)==len(lengths):values=np.repeat(values,lengths)
    if values.shape!=(int(lengths.sum()),) or not np.isfinite(values).all() or np.any(values<0):raise RuntimeError('Invalid actual radius: '+prefix)
    # Main capture converts radius to world metres. Native capture must say so
    # explicitly, rather than silently interpreting a raw object-local field.
    if native and not key.endswith('_radius_world'):raise RuntimeError('Native takeoff audit requires explicit radius_world')
    return values

def _bundle(arrays,prefix,*,native=False):
    q=np.asarray(arrays[prefix+'_world'],float);s=np.asarray(arrays[prefix+'_starts'],np.int64);n=np.asarray(arrays[prefix+'_lengths'],np.int64)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or s.ndim!=1 or s.shape!=n.shape or not len(s) or np.any(n<2):raise RuntimeError('Invalid actual curve data: '+prefix)
    if not np.array_equal(s,np.r_[0,np.cumsum(n[:-1])]) or int(n.sum())!=len(q):raise RuntimeError('Noncontiguous actual curves: '+prefix)
    return q,s,n,_radii(arrays,prefix,s,n,native=native)

def _arc(q):return np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]

def _query(tree,point,radius):
    from mathutils import Vector
    hit,normal,face,distance=tree.find_nearest(Vector(point))
    if hit is None or normal.length_squared<1e-20:raise RuntimeError('Current-head nearest surface unavailable')
    normal=normal.normalized();signed=float((Vector(point)-hit).dot(normal))
    return signed,signed-float(radius),int(face),tuple(normal)

def audit(arrays,*,head_prefix='actual-current-head'):
    from mathutils.bvhtree import BVHTree
    head=np.asarray(arrays[head_prefix+'_world'],float);tri=np.asarray(arrays[head_prefix+'_triangles'],np.int32)
    if head.ndim!=2 or head.shape[1]!=3 or tri.ndim!=2 or tri.shape[1]!=3 or not np.isfinite(head).all() or not len(tri):raise RuntimeError('Actual current full head triangles required')
    if np.any(tri<0) or np.any(tri>=len(head)):raise RuntimeError('Current head triangle indices out of range')
    tree=BVHTree.FromPolygons(head.tolist(),tri.tolist(),all_triangles=True)
    before=_bundle(arrays,'before-final');after=_bundle(arrays,'after-final');native=_bundle(arrays,'gather_native_after',native=True)
    if any(i>=len(before[1]) or i>=len(after[1]) for i in CURVE_IDS):raise RuntimeError('Fixed main curve identity missing')
    for i in CURVE_IDS:
        if before[2][i]!=after[2][i]:raise RuntimeError('Excepted takeoff control topology changed')
    for prefix in ('before-final','after-final'):
        if prefix+'_group' in arrays and not np.all(np.asarray(arrays[prefix+'_group'])[list(CURVE_IDS)]%256==40):raise RuntimeError('Fixed takeoff identities no longer belong to the measured positive gather')
    map_keys=[key for key in ('gather_native_after_original_curve','gather_native_after_original_curve_ids','tieproof_native_to_main_curve') if key in arrays]
    if not map_keys:raise RuntimeError('Actual native-to-main original curve mapping required')
    mapping=np.asarray(arrays[map_keys[0]],np.int64)
    if mapping.shape!=native[1].shape or len(np.unique(mapping))!=len(mapping):raise RuntimeError('Native original curve map invalid')
    for key in map_keys[1:]:
        if not np.array_equal(mapping,arrays[key]):raise RuntimeError('Conflicting actual native curve mappings')
    reverse={int(value):i for i,value in enumerate(mapping)}
    if any(i not in reverse for i in CURVE_IDS):raise RuntimeError('Native sample omitted an excepted takeoff curve')
    result={'helperSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'curveIds':list(CURVE_IDS),'headPrefix':head_prefix,
            'headVertices':len(head),'headTriangles':len(tri),'hairFreeSurfaceLimitMeters':LIMIT,'rootAttachmentArcMeters':ROOT_ARC,
            'signedConvention':'Nearest actual current triangle normal signed centre distance minus actual world radius.',
            'beforeNativeObserved':False,'beforeNativeReason':'No actual pre-edit native curve samples are available in this input; none are fabricated.',
            'afterNativeObserved':True,'afterNativeSamplesPerCurve':64,'curves':[],'errors':[],
            'nativeRadiusSource':'Supplied gather_native_after_radius_world; retains actual-world values or a conservative per-curve maximum bound without synthesizing a thinner radius.',
            'scope':'Five main gather curves only. Their p0 remains protected; the explicit edit exception concerns p1/p2 only. This audit does not authorize tail-pin changes.'}
    records={};root_bad=[]
    def scan(label,owner,points,radii,arcs,offsets,kind,gate):
        rows=[]
        for point,radius,ar,offset in zip(points,radii,arcs,offsets):
            signed,surface,face,normal=_query(tree,point,radius);free=float(ar)>ROOT_ARC+1e-12;bad=surface<LIMIT
            row={'curve':int(owner),'pointOrSegment':int(offset),'kind':kind,'world':np.asarray(point).tolist(),'arcMeters':float(ar),
                 'radiusWorldMeters':float(radius),'signedCenterMeters':signed,'signedSurfaceMeters':surface,'headTriangle':face,
                 'freeArc':bool(free),'belowUnchangedLimit':bool(bad),'attachmentOnly':not free}
            rows.append(row);records.setdefault(label,[]).append(row)
            if gate and free and bad:result['errors'].append(label+' curve '+str(owner)+' '+kind+' '+str(offset)+': '+format(surface,'.9f')+' m below -0.0005 m')
            if gate and not free and bad:root_bad.append(row)
        free_values=[r['signedSurfaceMeters'] for r in rows if r['freeArc']]
        return {'samples':len(rows),'freeSamples':sum(r['freeArc'] for r in rows),'freeBelowLimit':sum(r['freeArc'] and r['belowUnchangedLimit'] for r in rows),
                'attachmentBelowLimit':sum(r['attachmentOnly'] and r['belowUnchangedLimit'] for r in rows),
                'minimumFreeSurfaceMeters':min(free_values) if free_values else None,'rows':rows}
    for owner in CURVE_IDS:
        row={'curve':owner}
        for label,bundle in (('before',before),('after',after)):
            q,s,n,r=bundle;start=int(s[owner]);count=int(n[owner])
            if count<4:raise RuntimeError('Takeoff needs actual p0 to p3')
            path=q[start:start+count];rad=r[start:start+count];ar=_arc(path)
            row[label+'Controls']=scan(label+'_controls',owner,path[:4],rad[:4],ar[:4],np.arange(4),'control',label=='after')
            row[label+'ChordMidpoints']=scan(label+'_chord_midpoints',owner,(path[:3]+path[1:4])*.5,(rad[:3]+rad[1:4])*.5,(ar[:3]+ar[1:4])*.5,np.arange(3),'chord_midpoint',label=='after')
        bs=int(before[1][owner]);as_=int(after[1][owner]);row['p0BeforeAfterExact']=bool(np.array_equal(before[0][bs],after[0][as_]))
        row['p1p2MovementWorldMeters']=(after[0][as_+1:as_+3]-before[0][bs+1:bs+3]).tolist()
        ix=reverse[owner];q,s,n,r=native;start=int(s[ix]);count=int(n[ix])
        if count!=64:raise RuntimeError('Actual native takeoff curve must have 64 samples')
        path=q[start:start+count];rad=r[start:start+count];ar=_arc(path)
        root_delta=float(np.linalg.norm(path[0]-after[0][as_]));row['nativeRootToActualFinalRootMeters']=root_delta
        if root_delta>2e-6:raise RuntimeError('Actual native root correspondence mismatch')
        row['afterNativePoints']=scan('after_native_points',owner,path,rad,ar,np.arange(count),'native_point',True)
        row['afterNativeMidpoints']=scan('after_native_midpoints',owner,(path[:-1]+path[1:])*.5,(rad[:-1]+rad[1:])*.5,(ar[:-1]+ar[1:])*.5,np.arange(count-1),'native_midpoint',True)
        result['curves'].append(row)
    for label,rows in records.items():
        prefix='takeoff_audit_'+label
        _put(arrays,prefix+'_world',np.asarray([r['world'] for r in rows],np.float64))
        for field,dtype in (('curve',np.int32),('pointOrSegment',np.int32),('arcMeters',np.float64),('radiusWorldMeters',np.float64),('signedCenterMeters',np.float64),('signedSurfaceMeters',np.float64),('freeArc',np.bool_)):
            _put(arrays,prefix+'_'+field,np.asarray([r[field] for r in rows],dtype))
    result['afterAttachmentSamplesBelowLimit']=root_bad
    result['attachmentPolicy']='Arc <=1 mm is listed separately and does not generate the free-segment error list. Any attachment intrusion remains explicit evidence requiring review, never a declaration that penetration is physically allowed.'
    result['allAfterFreeSamplesPassed']=not result['errors']
    result['allAfterSampleDistancesPassed']=not result['errors'] and not root_bad
    result['actualCurrentSourceGeometryOnly']=True
    result['savedBlend']=False;result['rendered']=False
    return result
