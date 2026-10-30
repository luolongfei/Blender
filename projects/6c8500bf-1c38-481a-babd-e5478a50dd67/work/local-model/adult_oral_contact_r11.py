"""Current evaluated dental / perioral contact, without controls or file edits.

Build the domain once at the final neutral jaw/warm state, then reuse its
actual polygon IDs in every state. Measured bands on both sides of the real
inner/outer lip boundaries (7/17 and 8/22) are inspected; author groups 17/22
also extend into surrounding face and are not treated as pure lip labels. Posterior
oral bag, dental-root/gingiva embedding, and tooth/tooth occlusion are outside
this local test. Tooth mesh vertices never share IDs with head vertices.

Only the caller may decide to save. No old cage or source hash bypasses the
actual evaluated L3/Subsurf2 geometry checks here.
"""
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path
import runpy

import numpy as np

TOLERANCE=1e-6
INNER_DEPTH=.012
OUTER_DEPTH=.020
METRICS_HELPER='hand_wrist_surface_contact_r11.py'
METRICS_SHA='c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e'
TEETH=('upper_teeth','lower_teeth')
HEAD_GROUPS=(7,8,17,22)


def _array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def _metrics():
    path=Path(__file__).with_name(METRICS_HELPER)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=METRICS_SHA:
        raise RuntimeError('Review changed continuous triangle contact helper')
    return runpy.run_path(str(path))['triangle_pair_metrics']


def _read(items,field,width=1,dtype=np.float64):
    a=np.empty((len(items),width),dtype)
    items.foreach_get(field,a.ravel())
    return a if width!=1 else a.ravel()


def capture_evaluated(obj,depsgraph=None):
    """Copy actual native loop triangulation and face provenance; no mutation."""
    import bpy
    if obj.type!='MESH':raise RuntimeError('Actual evaluated mesh object required')
    graph=depsgraph or bpy.context.evaluated_depsgraph_get()
    evaluated=obj.evaluated_get(graph)
    mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    if mesh is None:raise RuntimeError('Actual evaluated mesh is unavailable')
    try:
        local=_read(mesh.vertices,'co',3)
        matrix=np.asarray(evaluated.matrix_world,np.float64)
        world=local@matrix[:3,:3].T+matrix[:3,3]
        if not np.isfinite(world).all():raise RuntimeError('Nonfinite actual evaluated surface')
        loops=_read(mesh.loops,'vertex_index',dtype=np.int32)
        starts=_read(mesh.polygons,'loop_start',dtype=np.int32)
        sizes=_read(mesh.polygons,'loop_total',dtype=np.int32)
        mesh.calc_loop_triangles()
        triangles=_read(mesh.loop_triangles,'vertices',3,np.int32)
        triangle_polygons=_read(mesh.loop_triangles,'polygon_index',dtype=np.int32)
        attribute=mesh.attributes.get('.sculpt_face_set')
        groups=None
        if attribute is not None:
            if attribute.domain!='FACE' or attribute.data_type!='INT':
                raise RuntimeError('Author face-set attribute changed domain/type')
            groups=_read(attribute.data,'value',dtype=np.int32)
        row={'world':world,'loops':loops,'faceStarts':starts,'faceSizes':sizes,
             'triangles':triangles,'trianglePolygonIds':triangle_polygons,'groups':groups,
             'objectName':obj.name,'originalMeshName':obj.data.name,'matrixWorld':matrix.tolist(),
             'modifierStack':[{'name':m.name,'type':m.type,'viewport':bool(m.show_viewport),'render':bool(m.show_render),
                **({'levels':int(m.levels),'renderLevels':int(m.render_levels)} if m.type in ('SUBSURF','MULTIRES') else {})}
                for m in obj.modifiers]}
        _validate_mesh(row)
        return row
    finally:evaluated.to_mesh_clear()


def _validate_mesh(mesh):
    world=np.asarray(mesh['world']);loops=np.asarray(mesh['loops'])
    starts=np.asarray(mesh['faceStarts']);sizes=np.asarray(mesh['faceSizes'])
    triangles=np.asarray(mesh['triangles']);polygon_ids=np.asarray(mesh['trianglePolygonIds'])
    if world.ndim!=2 or world.shape[1]!=3 or not len(world) or not np.isfinite(world).all():
        raise ValueError('Finite N x 3 actual world surface required')
    if starts.shape!=sizes.shape or starts.ndim!=1 or not len(starts) or np.any(sizes<3):
        raise ValueError('Actual original polygon boundaries required')
    if not np.array_equal(starts,np.r_[0,np.cumsum(sizes)[:-1]]) or int(sizes.sum())!=len(loops):
        raise ValueError('Actual polygon loops are not complete and ordered')
    if np.any(loops<0) or np.any(loops>=len(world)):
        raise ValueError('Actual polygon vertex index is invalid')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or polygon_ids.shape!=(len(triangles),):
        raise ValueError('Actual native triangles and polygon mapping required')
    if np.any(triangles<0) or np.any(triangles>=len(world)) or np.any(polygon_ids<0) or np.any(polygon_ids>=len(starts)):
        raise ValueError('Actual triangle mapping is invalid')
    # A fan replacement or stale triangulation cannot silently change domain IDs.
    if not np.array_equal(np.bincount(polygon_ids,minlength=len(starts)),sizes-2):
        raise ValueError('Native triangulation does not cover each supplied polygon')
    groups=mesh.get('groups')
    if groups is not None and np.asarray(groups).shape!=(len(starts),):
        raise ValueError('Face-set count does not match actual polygons')


def _topology(mesh):
    return {'points':len(mesh['world']),'polygons':len(mesh['faceStarts']),
            'loops':_array_sha(mesh['loops']),'faceStarts':_array_sha(mesh['faceStarts']),
            'faceSizes':_array_sha(mesh['faceSizes']),
            'groups':_array_sha(mesh['groups']) if mesh.get('groups') is not None else None}


def _provenance(mesh):
    return {'object':mesh['objectName'],'originalMesh':mesh['originalMeshName'],
            'evaluatedVertices':len(mesh['world']),'evaluatedPolygons':len(mesh['faceStarts']),
            'actualLoopTriangles':len(mesh['triangles']),'worldCoordinatesSha256':_array_sha(mesh['world']),
            'topology':_topology(mesh),'nativeTrianglesSha256':_array_sha(mesh['triangles']),
            'nativeTrianglePolygonIdsSha256':_array_sha(mesh['trianglePolygonIds']),
            'matrixWorld':mesh['matrixWorld'],'modifierStack':mesh['modifierStack']}


def _faces(mesh,ids):
    return [mesh['loops'][int(mesh['faceStarts'][i]):int(mesh['faceStarts'][i]+mesh['faceSizes'][i])] for i in ids]


def _group_edges(mesh,group):
    rows=np.flatnonzero(mesh['groups']==group)
    if not len(rows):raise RuntimeError('Required actual mouth group is absent: '+str(group))
    faces=_faces(mesh,rows)
    edges=np.concatenate([np.column_stack((f,np.roll(f,-1))) for f in faces])
    return rows,edges


def _boundary_and_distance(mesh,inner,outer,depth):
    inner_faces,inside_edges=_group_edges(mesh,inner)
    _,outside_edges=_group_edges(mesh,outer)
    a={tuple(x) for x in np.sort(inside_edges,axis=1)}
    b={tuple(x) for x in np.sort(outside_edges,axis=1)}
    seam=np.asarray(sorted(a&b),np.int32)
    if seam.size==0:raise RuntimeError('No real inner/outer shared lip boundary for '+str(inner))
    seam=seam.reshape(-1,2);seeds=np.unique(seam)
    unique=np.asarray(sorted(a),np.int32)
    distance=np.linalg.norm(mesh['world'][unique[:,0]]-mesh['world'][unique[:,1]],axis=1)
    if not np.isfinite(distance).all() or np.any(distance<=1e-12):
        raise RuntimeError('Degenerate lip edges prevent a measured anatomical domain')
    adjacency={}
    for (u,v),weight in zip(unique,distance):
        adjacency.setdefault(int(u),[]).append((int(v),float(weight)))
        adjacency.setdefault(int(v),[]).append((int(u),float(weight)))
    measured={int(i):0. for i in seeds};queue=[(0.,int(i)) for i in seeds];heapq.heapify(queue)
    while queue:
        d,u=heapq.heappop(queue)
        if d!=measured[u] or d>depth:continue
        for v,w in adjacency[u]:
            candidate=d+w
            if candidate<=depth and candidate<measured.get(v,float('inf')):
                measured[v]=candidate;heapq.heappush(queue,(candidate,v))
    selected=[]
    for face_id,face in zip(inner_faces,_faces(mesh,inner_faces)):
        if any(int(i) in measured for i in face):selected.append(int(face_id))
    return selected,seeds,{'innerGroup':inner,'outerGroup':outer,'sharedBoundaryEdges':len(seam),
                         'sharedBoundaryVertices':len(seeds),'withinDepthVertices':len(measured),
                         'selectedInnerPolygons':len(selected),'totalInnerPolygons':len(inner_faces),
                         'maximumReachedDistanceMeters':max(measured.values())}


def domain_from_mesh(mesh,inner_depth=INNER_DEPTH,required_head_vertices=None,outer_depth=OUTER_DEPTH):
    """Pure-array fixed neutral polygon domain, using current evaluated lengths."""
    _validate_mesh(mesh)
    if mesh.get('groups') is None:raise RuntimeError('Actual head face-set layer is required')
    if any(not np.isfinite(value) or not .008<=value<=.020 for value in (inner_depth,outer_depth)):
        raise ValueError('Inspect explicit anterior lip bands between 8 and 20 mm')
    groups=np.asarray(mesh['groups']);selected=set()
    rows=[];boundaries=[]
    for inner,outer in ((7,17),(8,22)):
        face_ids,seeds,record=_boundary_and_distance(mesh,inner,outer,inner_depth)
        record['side']='inner';selected.update(face_ids);boundaries.extend(map(int,seeds));rows.append(record)
        face_ids,seeds,record=_boundary_and_distance(mesh,outer,inner,outer_depth)
        record['side']='outer';selected.update(face_ids);boundaries.extend(map(int,seeds));rows.append(record)
    selected=np.asarray(sorted(selected),np.int32)
    if not len(selected):raise RuntimeError('Empty actual perioral domain')
    vertices=np.unique(np.concatenate(_faces(mesh,selected)))
    orientation={}
    for face in _faces(mesh,selected):
        for u,v in zip(face,np.roll(face,-1)):
            pair=tuple(sorted((int(u),int(v))));orientation.setdefault(pair,[]).append(1 if u<v else -1)
    bad=[edge for edge,directions in orientation.items() if len(directions)>2 or (len(directions)==2 and sum(directions)!=0)]
    if bad:raise RuntimeError('Selected lip surface has inconsistent/nonmanifold oriented edges: '+str(bad[:8]))
    required=[];required_faces=np.empty(0,np.int32)
    if required_head_vertices is not None:
        required=np.asarray(required_head_vertices,np.int32).ravel()
        if np.any(required<0) or np.any(required>=len(mesh['world'])):raise ValueError('Invalid required actual head vertex IDs')
        mouth_ids=np.unique(np.concatenate(_faces(mesh,np.flatnonzero(np.isin(groups,HEAD_GROUPS)))))
        outside_groups=np.setdiff1d(required,mouth_ids)
        if len(outside_groups):
            raise RuntimeError('Required seal-active vertices are outside actual 7/8/17/22 mouth groups: '+str(outside_groups[:32].tolist()))
        required=np.unique(required)
        missing=np.setdiff1d(required,vertices)
        if len(missing):raise RuntimeError('Actual seal-affected mouth vertices escape the fixed oral check domain: '+str(missing[:32].tolist()))
        required_mask=np.zeros(len(mesh['world']),bool);required_mask[required]=True
        incident=np.add.reduceat(required_mask[mesh['loops']].astype(np.int32),mesh['faceStarts'])>0
        required_faces=np.flatnonzero(incident&np.isin(groups,HEAD_GROUPS)).astype(np.int32)
        missing_faces=np.setdiff1d(required_faces,selected)
        if len(missing_faces):
            raise RuntimeError('Actual mouth polygons incident to seal-active vertices escape the fixed domain: '+str(missing_faces[:32].tolist()))
        required=required.tolist()
    result={'schema':1,'sourceHeadTopology':_topology(mesh),'polygonIds':selected.tolist(),'vertexIds':vertices.tolist(),
            'boundaryVertexIds':sorted(set(boundaries)),'innerDepthMeters':float(inner_depth),'outerDepthMeters':float(outer_depth),'bands':rows,
            'selectedPolygonCountsByGroup':{str(g):int(np.count_nonzero(groups[selected]==g)) for g in HEAD_GROUPS},
            'neutralDomainWorldSha256':_array_sha(mesh['world'][vertices]),'requiredSealMouthVertices':required,
            'requiredIncidentMouthPolygons':len(required_faces),'requiredIncidentMouthPolygonIdsSha256':_array_sha(required_faces),
            'scope':'Actual 7/17 and 8/22 mouth-rim shared edges seed separate geodesic bands on both inner and outer surfaces. Polygons touching a reached vertex are included whole, with no clipping caps. Author 17/22 surrounding face outside the band is not called lip. Fixed polygon IDs reused across states.',
            'excluded':'Posterior oral bag beyond this front-lip band; dental-root/gingiva embedding; tooth/tooth occlusion; soft oral parts against each other.'}
    result['domainSha256']=_json_sha(result)
    return result


def build_domain(head,inner_depth=INNER_DEPTH,required_head_vertices=None,outer_depth=OUTER_DEPTH):
    """Call once with jaw=warm=0 after the final saved seal/jaw setup."""
    keys=head.data.shape_keys.key_blocks
    for name in ('R11 jaw open','R11 warm closed smile','R11 natural soft smile'):
        if name in keys and abs(float(keys[name].value))>1e-8:
            raise RuntimeError('Build the fixed oral domain from a neutral actual surface')
    return domain_from_mesh(capture_evaluated(head),inner_depth,required_head_vertices,outer_depth)


def _check_domain(mesh,domain):
    copy={k:v for k,v in domain.items() if k!='domainSha256'}
    if _json_sha(copy)!=domain['domainSha256'] or _topology(mesh)!=domain['sourceHeadTopology']:
        raise RuntimeError('Actual head polygon topology or recorded oral domain changed')
    ids=np.asarray(domain['polygonIds'],np.int32)
    if np.any(ids<0) or np.any(ids>=len(mesh['faceStarts'])) or not np.isin(mesh['groups'][ids],HEAD_GROUPS).all():
        raise RuntimeError('Invalid actual perioral polygon selection')
    selected=np.flatnonzero(np.isin(mesh['trianglePolygonIds'],ids))
    if not len(selected):raise RuntimeError('No native triangles in recorded oral domain')
    return selected


def _tree(mesh,triangles,tolerance=0.):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons(mesh['world'].tolist(),triangles.tolist(),all_triangles=True,epsilon=tolerance)


def _coplanar_area(a,b,normal):
    """Exact planar triangle clipping area for the declared coplanar pair."""
    axis=int(np.argmax(np.abs(normal)));keep=[i for i in range(3) if i!=axis]
    aa=(a-a[0])[:,keep];bb=(b-a[0])[:,keep]
    cross=lambda u,v:float(u[0]*v[1]-u[1]*v[0])
    orientation=1. if cross(bb[1]-bb[0],bb[2]-bb[0])>0 else -1.
    polygon=[v.copy() for v in aa]
    for c,d in zip(bb,np.roll(bb,-1,axis=0)):
        clipped=[]
        for x,y in zip(polygon,polygon[1:]+polygon[:1]):
            dx=orientation*cross(d-c,x-c);dy=orientation*cross(d-c,y-c)
            if dx>=0:clipped.append(x)
            if (dx>=0)!=(dy>=0):clipped.append(x+(y-x)*(dx/(dx-dy)))
        polygon=clipped
        if not polygon:return 0.
    if len(polygon)<3:return 0.
    p=np.asarray(polygon);area=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))*.5
    return float(area/max(abs(float(normal[axis])),1e-30))


def classify_contact(a,b,metrics,tolerance=TOLERANCE):
    """Separate proper crossings, normal touch and ambiguous area contact.

    A normal near-contact is not required to gain an arbitrary positive gap.
    Positive-area coplanar overlap remains ambiguous without tissue-side proof.
    """
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);na/=np.linalg.norm(na,axis=1)[:,None]
    nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);nb/=np.linalg.norm(nb,axis=1)[:,None]
    da=np.einsum('nvc,nc->nv',a-b[:,0,None,:],nb)
    db=np.einsum('nvc,nc->nv',b-a[:,0,None,:],na)
    closed=np.asarray(metrics['intersects'],bool)
    straddle=np.minimum.reduce((-da.min(1),da.max(1),-db.min(1),db.max(1)))
    proper=closed&(straddle>tolerance)
    coplanar=closed&(np.linalg.norm(np.cross(na,nb),axis=1)<1e-8)&(np.abs(da).max(1)<=tolerance)&(np.abs(db).max(1)<=tolerance)
    areas=np.zeros(len(a),float)
    for i in np.flatnonzero(coplanar):areas[i]=_coplanar_area(a[i],b[i],na[i])
    ambiguous=coplanar&(areas>tolerance*tolerance)
    # A tooth can pass through an internal edge shared by two head triangles:
    # neither head triangle then straddles the tooth plane, although the tooth
    # clearly spans both sides of the skin plane. Boundary closest-point
    # witnesses are intentionally not used as a signed containment proof.
    # Keep that case unresolved, never silently call it ordinary light touch.
    tooth_straddle=np.minimum(-db.min(1),db.max(1))
    boundary_ambiguous=closed&(tooth_straddle>tolerance)&~proper&~coplanar
    return {'properCrossing':proper,'ambiguousCoplanarArea':ambiguous,'coplanarAreaSquareMeters':areas,
            'ambiguousBoundaryCrossing':boundary_ambiguous,'toothAcrossHeadPlaneDepthMeters':tooth_straddle,
            'normalStraddleDepthMeters':straddle,
            'numericalCrossingCandidate':closed&(straddle>1e-12)&~proper&~ambiguous&~boundary_ambiguous,
            'normalTouchOrNearContact':metrics['contactWithinTolerance']&~proper&~ambiguous&~boundary_ambiguous}


def _continuous_pair(head,lip_rows,lip_tree,dental,tolerance):
    tooth_tree=_tree(dental,dental['triangles'],tolerance)
    pairs=lip_tree.overlap(tooth_tree)
    if len(pairs)>200000:raise RuntimeError('Local tooth/lip broad phase exceeds 200000-pair bound; inspect geometry')
    intersections=[];contacts=[];proper=[];ambiguous=[];boundary=[];touches=[];numerical=[];minimum=None;critical_sample=[];touch_sample=[]
    if pairs:
        pairs=np.asarray(pairs,np.int32)
        for start in range(0,len(pairs),4096):
            batch=pairs[start:start+4096]
            a=head['world'][head['triangles'][lip_rows[batch[:,0]]]]
            b=dental['world'][dental['triangles'][batch[:,1]]]
            metrics=_metrics()(a,b,tolerance)
            kind=classify_contact(a,b,metrics,tolerance)
            closest=float(metrics['distance'].min());minimum=closest if minimum is None else min(minimum,closest)
            intersections.extend((start+np.flatnonzero(metrics['intersects'])).tolist())
            hit=np.flatnonzero(metrics['contactWithinTolerance']);contacts.extend((start+hit).tolist())
            proper.extend((start+np.flatnonzero(kind['properCrossing'])).tolist())
            ambiguous.extend((start+np.flatnonzero(kind['ambiguousCoplanarArea'])).tolist())
            boundary.extend((start+np.flatnonzero(kind['ambiguousBoundaryCrossing'])).tolist())
            touches.extend((start+np.flatnonzero(kind['normalTouchOrNearContact'])).tolist())
            numerical.extend((start+np.flatnonzero(kind['numericalCrossingCandidate'])).tolist())
            for i in hit:
                is_critical=bool(kind['properCrossing'][i] or kind['ambiguousCoplanarArea'][i] or kind['ambiguousBoundaryCrossing'][i])
                target=critical_sample if is_critical else touch_sample
                if len(target)>=(128 if is_critical else 16):continue
                hi=int(lip_rows[batch[i,0]]);di=int(batch[i,1]);polygon=int(head['trianglePolygonIds'][hi])
                target.append({'headTriangleRow':hi,'headPolygonId':polygon,'headFaceSet':int(head['groups'][polygon]),
                               'dentalTriangleRow':di,'dentalPolygonId':int(dental['trianglePolygonIds'][di]),
                               'surfaceDistanceMeters':float(metrics['distance'][i]),'closedTriangleIntersection':bool(metrics['intersects'][i]),
                               'properCrossing':bool(kind['properCrossing'][i]),'ambiguousCoplanarArea':bool(kind['ambiguousCoplanarArea'][i]),
                               'ambiguousBoundaryCrossing':bool(kind['ambiguousBoundaryCrossing'][i]),
                               'toothAcrossHeadPlaneDepthMeters':float(kind['toothAcrossHeadPlaneDepthMeters'][i]),
                               'coplanarAreaSquareMeters':float(kind['coplanarAreaSquareMeters'][i]),
                               'normalStraddleDepthMeters':float(kind['normalStraddleDepthMeters'][i]),
                               'numericalCrossingCandidate':bool(kind['numericalCrossingCandidate'][i]),
                               'headTriangleWorld':a[i].tolist(),'dentalTriangleWorld':b[i].tolist()})
    return {'broadPhasePairs':len(pairs),'closedTriangleIntersectionPairs':len(intersections),
            'contactPairsWithinNumericalTolerance':len(contacts),'numericalToleranceMeters':tolerance,
            'properCrossingPairs':len(proper),'ambiguousCoplanarAreaPairs':len(ambiguous),'normalTouchOrNearContactPairs':len(touches),
            'ambiguousBoundaryCrossingPairs':len(boundary),
            'numericalCrossingCandidatePairs':len(numerical),
            'closestBroadPhaseSurfaceDistanceMeters':minimum,'diagnosticPairs':critical_sample+touch_sample,
            'criticalDiagnosticsTruncated':len(proper)+len(ambiguous)+len(boundary)>len(critical_sample),
            'touchDiagnosticsTruncated':len(touches)>len(touch_sample),
            'passed':not proper and not ambiguous and not boundary,'needsReview':bool(ambiguous or boundary),
            'touchPolicy':'Ordinary boundary touch, positive separation within 1 micrometre, and sub-tolerance crossing candidates are diagnosed without demanding artificial clearance. Proper crossings require both faces to straddle beyond 1 micrometre. Coplanar area above tolerance squared and edge-aligned tooth spanning of the head plane beyond tolerance are ambiguous and prevent automatic adoption.',
            'method':'Expanded native BVH broad phase; complete paired-triangle SAT with coplanar axes plus vertex/triangle and all edge/edge feature distances. No cross-object shared-index exclusion.',
            'minimumDistanceScope':'Minimum among broad-phase candidates only, not a global all-pair separation claim.'}


def _barycentric(point,triangle):
    u=triangle[1]-triangle[0];v=triangle[2]-triangle[0];w=point-triangle[0]
    uu=float(u@u);uv=float(u@v);vv=float(v@v);den=uu*vv-uv*uv
    if den<=1e-30:return None
    s=float((w@u)*vv-(w@v)*uv)/den;t=float((w@v)*uu-(w@u)*uv)/den
    return np.asarray((1-s-t,s,t))


def _local_penetration(head,domain,full_head_tree,dental,tolerance):
    """Local signed witnesses only when the closest FULL-head face is a lip.

    Interior closest-point witnesses avoid an open-ROI boundary half-space
    assumption. This is a local inward-surface diagnostic, not whole-head
    winding containment. Continuous triangle tests are independently required.
    """
    from mathutils import Vector
    ids=np.asarray(domain['vertexIds'],np.int32);points=head['world'][ids]
    lower=points.min(0)-.004;upper=points.max(0)+.004
    candidates=np.flatnonzero(np.all(dental['world']>=lower,axis=1)&np.all(dental['world']<=upper,axis=1))
    face_set=set(domain['polygonIds']);negative=[];boundary=0;tested=0;minimum=None;witnesses=[]
    for i in candidates:
        p=dental['world'][i];nearest,normal,row,distance=full_head_tree.find_nearest(Vector(p))
        if nearest is None:raise RuntimeError('Actual closest full-head query failed')
        polygon=int(head['trianglePolygonIds'][row])
        if polygon not in face_set:continue
        bary=_barycentric(np.asarray(nearest,float),head['world'][head['triangles'][row]])
        if bary is None:raise RuntimeError('Degenerate actual lip triangle in closest-point check')
        if float(bary.min())<=1e-6:
            boundary+=1;continue
        tested+=1;signed=float((Vector(p)-nearest).dot(normal.normalized()))
        minimum=signed if minimum is None else min(minimum,signed)
        if signed < -tolerance:
            negative.append(int(i))
            if len(witnesses)<128:
                witnesses.append({'dentalVertexId':int(i),'pointWorld':p.tolist(),'headTriangleRow':int(row),
                                  'headPolygonId':polygon,'headFaceSet':int(head['groups'][polygon]),
                                  'closestWorld':list(nearest),'signedDistanceMeters':signed,'barycentric':bary.tolist()})
    return {'candidateVertices':len(candidates),'closestInteriorLipWitnesses':tested,'closestBoundaryWitnessesSkipped':boundary,
            'inwardLipWitnessCount':len(negative),'minimumSignedInteriorLipDistanceMeters':minimum,
            'inwardWitnesses':witnesses,'diagnosticsTruncated':len(negative)>len(witnesses),'passed':not negative,
            'scope':'Actual dental vertices within the actual lip bounds plus 4 mm. Hard inward witnesses require the closest complete-head native triangle to belong to the fixed lip domain and its closest point to lie inside that triangle. No global tissue-containment claim.'}


def observe(head,objects,domain,tolerance=TOLERANCE):
    """Inspect current evaluated state only; JSON-safe report, caller rejects."""
    import bpy
    if tolerance!=TOLERANCE:raise ValueError('The reviewed 1-micron numerical tolerance is fixed')
    if any(name not in objects for name in TEETH):raise RuntimeError('Both actual dental arches are required')
    multi=[m for m in head.modifiers if m.type=='MULTIRES']
    if len(multi)!=1 or multi[0].levels!=3 or multi[0].render_levels!=3 or not multi[0].show_viewport or not multi[0].show_render:
        raise RuntimeError('Actual head Multires3 must match inspected and rendered surfaces')
    _metrics()  # Reject changed dependency before evaluating large geometry.
    graph=bpy.context.evaluated_depsgraph_get();h=capture_evaluated(head,graph);lip_rows=_check_domain(h,domain)
    lip_tree=_tree(h,h['triangles'][lip_rows],tolerance);full_tree=_tree(h,h['triangles'])
    rows={};provenance={};failures=[]
    for name in TEETH:
        obj=objects[name];subs=[m for m in obj.modifiers if m.type=='SUBSURF']
        if len(subs)!=1 or subs[0].levels!=2 or subs[0].render_levels!=2 or not subs[0].show_viewport or not subs[0].show_render:
            raise RuntimeError('Actual dental Subsurf2 must match inspected and rendered surfaces: '+name)
        d=capture_evaluated(obj,graph);surface=_continuous_pair(h,lip_rows,lip_tree,d,tolerance)
        inward=_local_penetration(h,domain,full_tree,d,tolerance)
        rows[name]={'continuousSurface':surface,'localInwardWitnesses':inward,'passed':surface['passed'] and inward['passed']}
        provenance[name]=_provenance(d)
        if surface['properCrossingPairs']:failures.append({'object':name,'reason':'Proper continuous tooth/lip surface crossing',
            'properCrossingPairs':surface['properCrossingPairs']})
        if surface['ambiguousCoplanarAreaPairs']:failures.append({'object':name,'reason':'Positive-area coplanar contact is ambiguous; requires review before adoption',
            'ambiguousCoplanarAreaPairs':surface['ambiguousCoplanarAreaPairs'],'needsReview':True})
        if surface['ambiguousBoundaryCrossingPairs']:failures.append({'object':name,'reason':'Tooth spans the head plane beyond tolerance at an edge-aligned intersection; cannot certify ordinary light touch',
            'ambiguousBoundaryCrossingPairs':surface['ambiguousBoundaryCrossingPairs'],'needsReview':True})
        if not inward['passed']:failures.append({'object':name,'reason':'Actual dental vertex lies on the inward side of a closest interior front-lip face',
            'witnesses':inward['inwardLipWitnessCount'],'minimumSignedDistanceMeters':inward['minimumSignedInteriorLipDistanceMeters']})
    ids=np.asarray(domain['vertexIds'],np.int32);face_ids=np.asarray(domain['polygonIds'],np.int32)
    keys=head.data.shape_keys.key_blocks
    controls={name:float(keys[name].value) for name in ('R11 jaw open','R11 warm closed smile','R11 natural soft smile','R11 relaxed lip seal') if name in keys}
    seals=[m.node_group for m in head.modifiers if m.type=='NODES' and m.node_group and m.node_group.name=='R11 relaxed lip seal surface']
    effective_seal=None
    if seals:
        if len(seals)!=1:raise RuntimeError('Ambiguous current relaxed-seal layer')
        effective_seal=float(seals[0].nodes['R11 relaxed lip seal normalized control'].outputs[0].default_value)
    return {'schema':1,'blendFile':bpy.data.filepath,'frame':[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe],
            'controls':controls,'effectiveSealStrength':effective_seal,'continuousLipContactPassed':not failures,
            'criticalFailures':failures,'objects':rows,'geometryProvenance':{
                'head':_provenance(h),'domainSha256':domain['domainSha256'],'currentDomainWorldSha256':_array_sha(h['world'][ids]),
                'domainPolygonIdsSha256':_array_sha(face_ids),'currentDomainFaceSetsSha256':_array_sha(h['groups'][face_ids]),
                'actualCurrentDomainTrianglesSha256':_array_sha(h['triangles'][lip_rows]),
                'actualCurrentDomainTrianglePolygonIdsSha256':_array_sha(h['trianglePolygonIds'][lip_rows]),
                'actualDentalMeshes':provenance},'contactHelperSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'triangleMetricsHelperSha256':METRICS_SHA,'numericalToleranceMeters':tolerance,
            'scope':domain['scope'],'excluded':domain['excluded'],
            'limits':['Finite current state only; no interpolation or appearance acceptance.',
                      '1 micrometre is a numerical ambiguity boundary, not an anatomical lip-tooth gap prescription.',
                      'No posterior cavity, tooth-tooth occlusion, gingiva/tongue, or whole-body collision certificate.']}
