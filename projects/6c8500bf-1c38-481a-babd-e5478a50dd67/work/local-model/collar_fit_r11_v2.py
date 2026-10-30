"""Independent neckline and collar clearance candidate, preserving original data.

Offline geometry analysis is available without Blender. Preparation is explicit,
serial and restricted to a new work copy. Never changes anatomy or skirt caches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

COLLARS = ('Blouse_PeterPan_Collar_L', 'Blouse_PeterPan_Collar_R')
SEAMS = ('Collar_Edge_Stitch_L', 'Collar_Edge_Stitch_R')
BODICE = 'Blouse_Tucked_Bodice'
KEY = 'R11 tailored neckline clearance'
MARKER = 'R11_tailored_neckline_candidate'
SKIN = ('R11 Female head study', 'R11 Continuous neck bridge')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(x*(x*6-15)+10)


def nearest_triangle(point, points, triangles):
    """Exact closest-point search over the supplied finite triangle surface."""
    t = points[triangles]
    box_distance = np.maximum(np.maximum(t.min(1)-point, point-t.max(1)), 0.)
    box_d2 = (box_distance*box_distance).sum(1)
    # A nearest vertex supplies an upper bound, hence this broad phase cannot
    # exclude the true closest triangle within the supplied surface.
    upper = np.min(((points-point)**2).sum(1)) + 1e-12
    ids = np.flatnonzero(box_d2 <= upper)
    t = t[ids]
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = (normal*normal).sum(1)
    good = n2 > 1e-24
    if not good.any():
        raise RuntimeError('Degenerate candidate surface')
    t, ids, a, ab, ac, normal, n2 = (x[good] for x in (t, ids, a, ab, ac, normal, n2))
    signed_plane = ((point-a)*normal).sum(1)/n2
    projected = point-signed_plane[:, None]*normal
    ap = projected-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    d20, d21 = (ap*ab).sum(1), (ap*ac).sum(1)
    denominator = d00*d11-d01*d01
    u = (d11*d20-d01*d21)/denominator
    v = (d00*d21-d01*d20)/denominator
    inside = (u >= 0) & (v >= 0) & (u+v <= 1)
    candidates = [projected]
    scores = [np.where(inside, ((projected-point)**2).sum(1), np.inf)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        direction = t[:, j]-t[:, i]
        fraction = np.clip(((point-t[:, i])*direction).sum(1)
                           / np.maximum((direction*direction).sum(1), 1e-24), 0, 1)
        q = t[:, i]+fraction[:, None]*direction
        candidates.append(q)
        scores.append(((q-point)**2).sum(1))
    scores = np.stack(scores)
    kind, index = np.unravel_index(np.argmin(scores), scores.shape)
    n = normal[index]/np.sqrt(n2[index])
    q = candidates[kind][index]
    return q, n, int(ids[index]), float(np.sqrt(scores[kind, index]))


def _mid_bodice_from_observation(data, row):
    """Recover an approximate REST midsurface from observed Solidify pairs.

    Used for offline diagnosis only. Blender preparation reads the actual raw
    mesh; it never writes these averaged observations back into the garment.
    """
    prefix = row['prefix']
    all_count = row['evaluatedVerticesTotal']
    if all_count % 2:
        raise RuntimeError('Expected paired Solidify layers')
    count = all_count//2
    ids = data[prefix+'_vertex_ids']
    world = data[prefix+'_world'].astype(float)
    triangles = data[prefix+'_triangles']
    original_triangles = ids[triangles]
    index = {int(v): i for i, v in enumerate(ids)}
    paired = [i for i in ids if i < count and i+count in index]
    remap = {int(v): i for i, v in enumerate(paired)}
    points = np.array([(world[index[int(i)]]+world[index[int(i+count)]])*.5 for i in paired])
    ts = np.array([[remap[int(i)] for i in tri] for tri in original_triangles
                   if all(int(i) in remap for i in tri)], np.int32)
    if not len(ts):
        raise RuntimeError('No original-shell triangles recovered')
    return points, ts


def analyze(observation, output):
    record = json.loads(observation.read_text(encoding='utf-8'))
    payload = observation.with_suffix('.npz')
    if _sha(payload) != record['payloadSha256']:
        raise RuntimeError('Observation payload changed')
    rest = next(s for s in record['states'] if s['state'] == 'rest')
    bodice = next(o for o in rest['objects'] if o['object'] == BODICE)
    results = []
    with np.load(payload, allow_pickle=False) as data:
        points, triangles = _mid_bodice_from_observation(data, bodice)
        for i, name in enumerate(COLLARS):
            cage = data[f'raw_collar_{i}_local'].astype(float)
            matrix = data[f'raw_collar_{i}_matrix_world_rest']
            cage = cage@matrix[:3, :3].T+matrix[:3, 3]
            distances = []
            for p in cage:
                q, n, face, d = nearest_triangle(p, points, triangles)
                distances.append(float(np.dot(p-q, n)))
            distances = np.array(distances)
            results.append({'object': name, 'rawCagePoints': len(cage),
                'signedBodiceMidSurfaceDistanceQuantilesMeters': np.quantile(distances, (0, .01, .5, .99, 1)).tolist(),
                'cagePointsInsideBodiceByMoreThanHalfMillimeter': np.flatnonzero(distances < -.0005).tolist(),
                'cagePointsBelowCombinedThicknessPlusOneMillimeter': np.flatnonzero(distances < .0025).tolist(),
                'interpretation': 'Both original collar surfaces are smooth. Broad image patches may be bodice/collar overlap; actual trial renders are still required.'})
    result = {'sourceBlend': record['sourceBlend'], 'sourceSha256': record['sourceSha256'],
        'observation': observation.relative_to(project_path()).as_posix(),
        'observationSha256': _sha(observation), 'helperSha256': _sha(Path(__file__)), 'collars': results,
        'method': 'Offline closest-point analysis of actual raw collar cages against average paired solidify bodice observations.',
        'savedBlend': False, 'limitations': ['Averaged paired surfaces are only an offline approximation.',
            'No current-source fix or appearance acceptance is claimed.']}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(output), 'collars': results}), flush=True)
    return result


def _array(data, field='co', width=3, dtype=np.float32):
    values = np.empty((len(data), width), dtype)
    data.foreach_get(field, values.ravel())
    return values


def _world(obj, local):
    matrix = np.asarray(obj.matrix_world, float)
    return local@matrix[:3, :3].T+matrix[:3, 3]


def _local(obj, world):
    matrix = np.asarray(obj.matrix_world.inverted(), float)
    return world@matrix[:3, :3].T+matrix[:3, 3]


def _cage_world(obj):
    keys = obj.data.shape_keys
    if keys:
        # This first garment candidate intentionally refuses unrelated shape
        # mixes instead of baking or overwriting their existing deformation.
        if any(abs(k.value) > 1e-8 for k in keys.key_blocks[1:]):
            raise RuntimeError('Inspect existing active garment shape keys: '+obj.name)
        return _world(obj, _array(keys.reference_key.data).astype(float))
    return _world(obj, _array(obj.data.vertices).astype(float))


def _triangles(mesh):
    mesh.calc_loop_triangles()
    return _array(mesh.loop_triangles, 'vertices', 3, np.int32)


def _bvh(points, triangles):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)


def _skin_trees():
    import bpy
    result = []
    graph = bpy.context.evaluated_depsgraph_get()
    for name in SKIN:
        obj = bpy.data.objects[name]
        ev = obj.evaluated_get(graph)
        mesh = ev.to_mesh()
        try:
            points = _world(ev, _array(mesh.vertices).astype(float))
            triangles = _triangles(mesh)
            tp = points[triangles]
            keep = (tp.max(1)[:, 2] >= 1.23) & (tp.min(1)[:, 2] <= 1.31)
            result.append((name, _bvh(points, triangles[keep])))
        finally:
            ev.to_mesh_clear()
    return result


def _radial_clear(point, skin, clearance=.003):
    """Enlarge only the actual neck opening; no lower torso displacement.

    Rays start outside the observed neck volume, at the point's own Z, and use
    the first real skin intersection along its radial direction. This avoids
    interpreting the open cut's signed nearest distance as a closed solid.
    """
    from mathutils import Vector
    p = np.asarray(point, float)
    if not 1.23 < p[2] < 1.31:
        return p.copy()
    center = np.array((0., .008, p[2]))
    direction = p-center
    direction[2] = 0
    radius = np.linalg.norm(direction)
    if radius < 1e-8 or radius >= .14:
        return p.copy()
    direction /= radius
    outer = center+direction*.22
    skin_radius = None
    for name, tree in skin:
        hit, normal, face, distance = tree.ray_cast(Vector(outer), Vector(-direction), .23)
        if face is not None:
            r = float(np.dot(np.asarray(hit)-center, direction))
            if 0. < r < .13:
                skin_radius = max(skin_radius or 0., r)
    if skin_radius is None:
        return p.copy()
    required = skin_radius+clearance-radius
    if required <= 0:
        return p.copy()
    # Actual neck intersections are above the seam; taper only in regions where
    # the garment already lies outside the neck volume, never across a contact.
    return p+direction*required


def _bspline_basis(x, count, degree=3):
    """Clamped cubic basis in the ORIGINAL nonuniform collar UV parameter."""
    x=np.asarray(x,float);knots=np.r_[np.zeros(degree+1),np.linspace(0,1,count-degree+1)[1:-1],np.ones(degree+1)]
    basis=np.array([((x>=knots[i])&(x<knots[i+1])).astype(float) for i in range(len(knots)-1)]).T
    for order in range(1,degree+1):
        result=np.zeros((len(x),len(knots)-order-1))
        for i in range(result.shape[1]):
            if knots[i+order]>knots[i]:result[:,i]+=(x-knots[i])/(knots[i+order]-knots[i])*basis[:,i]
            if knots[i+order+1]>knots[i+1]:result[:,i]+=(knots[i+order+1]-x)/(knots[i+order+1]-knots[i+1])*basis[:,i+1]
        basis=result
    basis[x==1]=0.;basis[x==1,-1]=1.
    if np.max(np.abs(basis.sum(1)-1))>1e-12:raise RuntimeError('Invalid cubic collar partition')
    return basis


def _collar_basis():
    rows=np.array((0,.015,.055,.14,.27,.42,.58,.73,.86,.945,.985,1.))
    bu=_bspline_basis(np.arange(49)/48.,9);bv=_bspline_basis(rows,5)
    basis=np.einsum('ri,cj->rcij',bv,bu).reshape(588,45)
    penalty=[]
    for r in range(5):
        for c in range(7):
            row=np.zeros((5,9));row[r,c:c+3]=[1,-2,1];penalty.append(row.ravel())
    for r in range(3):
        for c in range(9):
            row=np.zeros((5,9));row[r:r+3,c]=[1,-2,1];penalty.append(row.ravel())
    regularizer=np.asarray(penalty);system=basis.T@basis+.025*(regularizer.T@regularizer)+np.eye(45)*1e-7
    projector=np.linalg.solve(system,basis.T)
    return rows,basis,projector,bu


def _normal_cosines(before,after,triangles):
    p=before[triangles];q=after[triangles]
    a=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);b=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return np.sum(a*b,axis=1)/np.maximum(denom,1e-30)


def _sewn_edge_targets(inner,before,after,triangles):
    """Exact correspondence on the actual closed raw upper-blouse boundary."""
    edges=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True);edges=edges[counts==1]
    p=before[edges];keep=(p[:,:,2].min(1)>1.25)&(np.abs(p[:,:,0]).max(1)<.12)&(np.linalg.norm(p[:,:,:2],axis=2).max(1)<.09)
    edges=edges[keep];ids,degree=np.unique(edges,return_counts=True)
    if len(ids)!=192 or not np.all(degree==2):raise RuntimeError('Inspect changed actual closed bodice neckline boundary')
    center=np.array((0.,.008));a=before[edges[:,0],:2]-center;b=before[edges[:,1],:2]-center;e=b-a
    result=[];mapping=[]
    for point in inner:
        direction=point[:2]-center;direction/=np.linalg.norm(direction)
        den=direction[0]*e[:,1]-direction[1]*e[:,0];valid=np.abs(den)>1e-12
        radius=(a[:,0]*e[:,1]-a[:,1]*e[:,0])/np.where(valid,den,1.)
        fraction=(a[:,0]*direction[1]-a[:,1]*direction[0])/np.where(valid,den,1.)
        valid&=(radius>0)&(fraction>=-1e-9)&(fraction<=1+1e-9)
        choices=np.flatnonzero(valid)
        if not len(choices):raise RuntimeError('No continuous original neckline radial correspondence')
        j=int(choices[np.argmin(np.abs(radius[choices]-np.linalg.norm(point[:2]-center)))])
        f=float(np.clip(fraction[j],0,1));result.append(after[edges[j,0]]*(1-f)+after[edges[j,1]]*f)
        mapping.append({'bodyEdge':edges[j].tolist(),'fraction':f})
    return np.asarray(result),mapping


def _project_collar(points, base, triangles, bodice, skin, sewn_edge=None):
    """Fit ONE continuous displacement surface, not 588 independent targets."""
    from mathutils import Vector
    if len(base)!=588:raise RuntimeError('Expected original 12-by-49 collar parameterization')
    rows,basis,projector,bu=_collar_basis();v=np.repeat(rows,49)
    targets=.0027+.0012*np.sin(np.pi*v)
    if sewn_edge is None:raise RuntimeError('Actual fixed bodice seam correspondence is required')
    sewn,mapping=sewn_edge
    # The entire inner row is sewn to the actual displaced blouse neckline.
    # No gravity or nearest-surface correction is allowed to move that row.
    d0=sewn-base[:49];opening=np.linalg.norm(d0,axis=1)
    field=np.tile(d0,(12,1))*(1-.5*v[:,None]);p=base+field;anchor=p.copy();history=[]
    # All free corrections vanish exactly at the seam. The original closely
    # spaced support rows and rounded silhouette remain in the same topology.
    free_basis=basis*_smooth(v/.14)[:,None]
    projector=np.linalg.solve(free_basis.T@free_basis+np.eye(45)*.02,free_basis.T)
    basis=free_basis
    if _normal_cosines(base,p,triangles).min()<=.02:raise RuntimeError('Sewn opening transport folds original collar')
    initial=p.copy()
    active=v>=.14
    attached=v<.14
    targets[attached]*=_smooth(v[attached]/.14)
    def constraints(current):
        correction=np.zeros_like(current);signed=[];skin_error=[]
        for i,point in enumerate(current):
            nearest,normal,face,distance=bodice.find_nearest(Vector(point))
            if face is None or distance>.065:raise RuntimeError('No actual blouse support within bounded collar fit')
            n=np.asarray(normal,float);distance=float(np.dot(point-np.asarray(nearest),n));signed.append(distance)
            clearance=_radial_clear(point,skin,.003)-point;skin_error.append(np.linalg.norm(clearance))
            # Collar support is unilateral: a resting flap may lift away from
            # the blouse. Pulling every point back to a prescribed distance
            # rolls its rounded rear edge onto the vertical neck opening.
            correction[i]=n*max(0.,targets[i]-distance)*2.5+clearance*1.7
            # A bounded gravity-direction adjustment softens free-flap height.
            # The inner seam and intentional collar lift are not projected.
            hit,hit_normal,hit_face,hit_distance=bodice.ray_cast(Vector(point+np.array((0.,0.,.065))),Vector((0.,0.,-1.)),.15)
            if hit_face is not None and hit_normal[2]>.12 and v[i]>.14:
                rest_z=float(hit[2])+targets[i]/max(float(hit_normal[2]),.2)
                # A collar is sewn at its inner edge, not glued onto the body.
                # Limit the free-edge settling to four millimetres from its
                # coherently transported shape; keep its deliberate soft roll.
                rest_z=max(rest_z,initial[i,2]-.004)
                drop=max(0.,point[2]-rest_z)*float(_smooth((v[i]-.14)/.6))
                correction[i,2]-=drop*.45
        return correction,np.asarray(signed),np.asarray(skin_error)
    for iteration in range(28):
        correction,distances,skin_error=constraints(p)
        residual=basis@(projector@(correction+.035*(anchor-p)))
        fraction=1.
        for backtrack in range(12):
            trial=p+residual*fraction
            if _normal_cosines(base,trial,triangles).min()>.025 and np.linalg.norm(trial-base,axis=1).max()<=.035:break
            fraction*=.5
        else:raise RuntimeError('Continuous collar field cannot meet orientation/extent guard: '+json.dumps({'iteration':iteration,'history':history[-3:],'minimumCurrentCosine':float(_normal_cosines(base,p,triangles).min()),'minimumTrialCosine':float(_normal_cosines(base,trial,triangles).min()),'maxTrialOffset':float(np.linalg.norm(trial-base,axis=1).max()),'worstTriangle':int(np.argmin(_normal_cosines(base,trial,triangles)))}))
        p=trial
        history.append({'iteration':iteration+1,'stepFraction':fraction,'minimumSupportDistanceMeters':float(distances.min()),'minimumFreeFlapSupportDistanceMeters':float(distances[active].min()),
            'maximumSkinCorrectionMeters':float(skin_error.max()),'maximumFieldStepMeters':float(np.linalg.norm(residual*fraction,axis=1).max())})
        if iteration>=5 and distances[active].min()>=.0015 and skin_error[active].max()<.0001 and skin_error[attached].max()<.0005 and np.linalg.norm(residual*fraction,axis=1).max()<.00005:break
    _,distances,skin_error=constraints(p)
    # Actual body-edge straight chords have 2.75 mm minimum skin clearance,
    # though their original vertices each targeted 3 mm. At the fixed sewn
    # band require 2.5 mm clearance; do not move it off the real body seam to
    # chase a 3 mm radial target. The free flap retains its 2.9 mm guard.
    if distances[active].min()<.0015 or skin_error[active].max()>.0001 or skin_error[attached].max()>.0005:
        raise RuntimeError('Continuous collar fit still lacks actual clearance: '+json.dumps({'minimumFreeFlapSupportMeters':float(distances[active].min()),'maximumSkinCorrectionMeters':float(skin_error.max()),'history':history[-3:]}))
    return p,{'supportDistanceQuantilesMeters':np.quantile(distances,(0,.01,.5,.99,1)).tolist(),
        'maximumSkinCorrectionMeters':float(skin_error.max()),'targetSeparationMeters':[float(targets.min()),float(targets.max())],
        'sewnEdgeDisplacementRangeMeters':[float(opening.min()),float(opening.max())],
        'sewnBoundaryMaximumErrorMeters':float(np.linalg.norm(p[:49]-sewn,axis=1).max()),'sewnBodyEdgeCorrespondence':mapping,
        'freeFlapSupportDistanceQuantilesMeters':np.quantile(distances[active],(0,.01,.5,.99,1)).tolist(),
        'iterations':iteration+1,'history':history,'controlLattice':[5,9],'originalNonuniformRowParameters':rows.tolist(),
        'freeFlapGravityAdjustmentLimitMeters':.004,'sewnBandMinimumSkinClearanceMeters':.0025,'freeFlapMinimumSkinClearanceMeters':.0029,
        'method':'Continuous displacement field anchored to the actual 192-point body neckline. Exact inner-row sewing correspondence, cubic free-flap correction, unilateral body contact and bounded four-millimetre free-edge settling. Original rounded silhouette and topology retained.'}


def _transport(points, old_collar, new_collar, triangles):
    from mathutils import Vector
    tree = _bvh(old_collar, triangles)
    result = []
    for point in points:
        nearest, normal, face, distance = tree.find_nearest(Vector(point))
        if face is None or distance > .006:
            raise RuntimeError('Original collar stitching is not near its actual flap')
        ids = triangles[face]
        a, b, c = old_collar[ids]
        ab, ac, ap = b-a, c-a, np.asarray(nearest)-a
        d00, d01, d11 = np.dot(ab, ab), np.dot(ab, ac), np.dot(ac, ac)
        denominator = d00*d11-d01*d01
        if denominator < 1e-20:
            raise RuntimeError('Degenerate stitching support triangle')
        u = (d11*np.dot(ap, ab)-d01*np.dot(ap, ac))/denominator
        v = (d00*np.dot(ap, ac)-d01*np.dot(ap, ab))/denominator
        weights = np.array((1-u-v, u, v))
        result.append(point+weights@(new_collar[ids]-old_collar[ids]))
    return np.asarray(result)


def _shape_guard(before, after, triangles, name):
    delta = np.linalg.norm(after-before, axis=1)
    # Preserve the original fitting scope and orientation thresholds.
    limit=.035
    if not np.isfinite(after).all() or delta.max() > limit:
        raise RuntimeError('Unexpected local garment displacement: '+name)
    old = np.cross(before[triangles[:, 1]]-before[triangles[:, 0]], before[triangles[:, 2]]-before[triangles[:, 0]])
    new = np.cross(after[triangles[:, 1]]-after[triangles[:, 0]], after[triangles[:, 2]]-after[triangles[:, 0]])
    lengths = np.linalg.norm(old, axis=1)*np.linalg.norm(new, axis=1)
    good = lengths > 1e-20
    cosine = np.sum(old[good]*new[good], axis=1)/lengths[good]
    if (cosine <= .02).any():
        raise RuntimeError('Garment face orientation folded relative to original: '+name)
    return {'maxDisplacementMeters': float(delta.max()), 'movedPoints': int((delta > 1e-8).sum()),
            'minimumOriginalNewNormalDot': float(cosine.min()), 'topologyChanged': False}


def _add_key(obj, after):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name='Basis', from_mix=False)
    if obj.data.shape_keys.key_blocks.get(KEY):
        raise RuntimeError('Do not compound the garment fitting candidate')
    key = obj.shape_key_add(name=KEY, from_mix=False)
    local = np.asarray(_local(obj, after), np.float32)
    key.data.foreach_set('co', local.ravel())
    key.relative_key = obj.data.shape_keys.reference_key
    key.value = 1.
    key.slider_min, key.slider_max = 0., 1.
    obj.data.update()
    return key


def _stable_contract(obj):
    """Original mesh/keys/materials/UVs/weights. New candidate key is excluded."""
    observer = _api('garment_neck_observe_r11')
    row = observer['_contract']([obj])[obj.name]
    row['keys'] = {k: v for k, v in row['keys'].items() if k not in {KEY, 'Basis'}}
    for modifier in row['modifiers']:
        modifier.pop('is_active', None)
    mesh = obj.data
    row['uv'] = {layer.name: observer['_digest_array'](_array(layer.data, 'uv', 2)) for layer in mesh.uv_layers}
    row['groups'] = [(g.name, g.index, g.lock_weight) for g in obj.vertex_groups]
    row['weights'] = [[(g.group, g.weight) for g in vertex.groups] for vertex in mesh.vertices]
    row['polygonsSmooth'] = observer['_digest_array'](_array(mesh.polygons, 'use_smooth', 1, np.bool_))
    return row


def prepare(source, output):
    import bpy
    if (not bpy.app.background or Path(bpy.data.filepath).resolve() != source
            or source == output or output.exists() or output.with_suffix('.collar-fit.json').exists()):
        raise RuntimeError('Fresh explicit source and unused independent output required')
    if bpy.context.scene.get(MARKER):
        raise RuntimeError('Restore the original garment shapes before refitting')
    if len(bpy.data.objects['R11 Continuous neck bridge'].data.vertices) != 2592:
        raise RuntimeError('Expected the actually observed upper-neck seam combination')
    source_sha = _sha(source)
    targets = [bpy.data.objects[name] for name in (BODICE,)+COLLARS+SEAMS]
    attachments = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'
                   and obj.name.startswith(('Blouse_Button_', 'Blouse_Placket_Edge_', 'Button_'))
                   and obj not in targets]
    targets += attachments
    if any(obj.type != 'MESH' or obj.data.users != 1 for obj in targets):
        raise RuntimeError('All garment targets must be observed single-user mesh objects')
    before = {obj.name: _stable_contract(obj) for obj in targets}
    existing_key_counts = {obj.name: len(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else 0 for obj in targets}
    protected = [obj for obj in bpy.data.objects if obj.type in {'MESH', 'CURVES'} and obj not in targets
                 and (obj.name in SKIN or obj.name == 'R01 Head' or 'Native Body Cloth' in obj.name
                      or obj.name.startswith(('R11 Donor /', 'R02 Strands /')))]
    geometry_contract = _api('prepare_adult_tail_shape_r11')['geometry_contract']
    protected_before = {obj.name: geometry_contract(obj) for obj in protected}
    pose = _api('production_uv_r08')['_pose_contract']()
    rig = bpy.data.objects['Character_Rig']
    old_pose = rig.data.pose_position
    added = []
    report = None
    try:
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        skin = _skin_trees()
        source_points = {obj.name: _cage_world(obj) for obj in targets}
        triangles = {obj.name: _triangles(obj.data) for obj in targets}
        new_points = {}
        for obj in [bpy.data.objects[BODICE]]+attachments:
            points = source_points[obj.name]
            new_points[obj.name] = np.array([_radial_clear(p, skin) for p in points])
        body_tree = _bvh(new_points[BODICE], triangles[BODICE])
        rows = []
        for name, seam_name in zip(COLLARS, SEAMS):
            original = source_points[name]
            radial = np.array([_radial_clear(p, skin) for p in original])
            sewn=_sewn_edge_targets(original[:49],source_points[BODICE],new_points[BODICE],triangles[BODICE])
            changed, details = _project_collar(radial, original, triangles[name], body_tree, skin,sewn)
            new_points[name] = changed
            new_points[seam_name] = _transport(source_points[seam_name], original, changed, triangles[name])
            rows.append({'object': name, **details})
        for obj in targets:
            checks = _shape_guard(source_points[obj.name], new_points[obj.name], triangles[obj.name], obj.name)
            # Unchanged lower buttons are left completely untouched.
            if checks['movedPoints']:
                _add_key(obj, new_points[obj.name])
                added.append(obj)
            rows.append({'object': obj.name, **checks})
        bpy.context.view_layer.update()
        report = {'sourceBlend': source.relative_to(project_path()).as_posix(), 'sourceSha256': source_sha,
            'helperSha256': _sha(Path(__file__)), 'key': KEY, 'defaultValue': 1., 'objects': rows,
            'method': 'Actual skin radial bodice clearance; collar inner row sewn to the displaced 192-point neckline with explicit original body-edge correspondence. Continuous free-flap field preserves rounded contour and permits lift. Edge stitches follow original triangle correspondence.',
            'originalGarmentDataPreserved': True, 'bodyAnatomyChanged': False,
            'oldPhysicsCachesChanged': False,
            'invalidatedChecks': ['Neck/collar/bodice intersection and shading review',
                'Collar and upper-blouse contact during head/body poses', 'Current hair against collar/shoulder motion',
                'Runtime garment geometry and material/normal bake derived from old garment',
                'Old R09 clip appearance cannot certify this new upper garment'],
            'limitations': ['Authored static fitting, not a cloth simulation.',
                'Actual evaluated front/quarter/side images and fresh edge-intersection observation required.',
                'No full motion, no skirt rebake or final appearance acceptance.']}
    except Exception:
        for obj in reversed(added):
            keys = obj.data.shape_keys
            if keys and keys.key_blocks.get(KEY):
                obj.shape_key_remove(keys.key_blocks[KEY])
            if not existing_key_counts[obj.name] and obj.data.shape_keys:
                obj.shape_key_remove(obj.data.shape_keys.reference_key)
        raise
    finally:
        rig.data.pose_position = old_pose
        bpy.context.view_layer.update()
    if {obj.name: _stable_contract(obj) for obj in targets} != before:
        raise RuntimeError('Original garment topology/geometry/keys/materials/weights/UVs changed')
    if {obj.name: geometry_contract(obj) for obj in protected} != protected_before:
        raise RuntimeError('Protected anatomy, hair or skirt cache changed')
    if _api('production_uv_r08')['_pose_contract']() != pose or _sha(source) != source_sha:
        raise RuntimeError('Pose/action or saved input changed')
    report['sourceAndOriginalContractsPreserved'] = True
    report['modifiedObjects'] = [obj.name for obj in added]
    bpy.context.scene[MARKER] = json.dumps(report, ensure_ascii=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):
        raise RuntimeError('Could not save independent garment candidate')
    report.update(output=output.relative_to(project_path()).as_posix(), outputSha256=_sha(output), savedBlend=True)
    with output.with_suffix('.collar-fit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R11_COLLAR_FIT_SAVED '+json.dumps({'output': str(output), 'sha256': report['outputSha256']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('analyze', 'prepare'), required=True)
    parser.add_argument('--observation', default='work/reconstruction-r02/garment-neck-current-r11.json')
    parser.add_argument('--source')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    observation = (project_path()/args.observation).resolve()
    output = (project_path()/args.output).resolve()
    output.relative_to((project_path()/'work').resolve())
    if args.mode == 'analyze':
        analyze(observation, output)
    else:
        if not args.source:
            raise RuntimeError('Explicit --source required')
        source = (project_path()/args.source).resolve()
        source.relative_to((project_path()/'work/reconstruction-r02').resolve())
        output.relative_to((project_path()/'work/reconstruction-r02').resolve())
        prepare(source, output)
