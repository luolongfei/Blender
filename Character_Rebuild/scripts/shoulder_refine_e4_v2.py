"""Frozen E4 delta: depth correction plus bounded local 3D ridge fairing."""
import numpy as np
import bpy
import bmesh
from shoulder_refine_e4 import (_coords,_uv_hash,_topology,_tree,_front_y,_ss,
                               apply_shoulder_refinement)


def apply_shoulder_refinement_v2(body):
    source=_coords(body.data)
    report={'depth_pass':apply_shoulder_refinement(body)}
    mesh=body.data;before=_coords(mesh);uv_before=_uv_hash(mesh);topology_before=_topology(mesh)
    before_tree=_tree(mesh)
    x,y,z=before.T
    weights=_ss(.045,.075,np.abs(x))*(1-_ss(.25,.29,np.abs(x)))
    weights*=_ss(.525,.550,z)*(1-_ss(.608,.632,z))*(1-_ss(.014,.055,y))
    # Lock the actual upper silhouette, measured in narrow X bins. Only internal
    # shoulder/under-collar support may move in X/Z; the neck boundary is also fixed.
    bins=np.floor(x/.008).astype(np.int32)
    maxima={}
    for i in np.flatnonzero(z>.50):maxima[bins[i]]=max(maxima.get(bins[i],-1e9),z[i])
    silhouette=np.array([z[i]>maxima.get(bins[i],1e9)-.010 for i in range(len(before))])
    weights[silhouette]=0
    bm=bmesh.new();bm.from_mesh(mesh)
    boundary={v.index for e in bm.edges if e.is_boundary for v in e.verts};bm.free()
    if boundary:weights[list(boundary)]=0
    edge_array=np.empty(len(mesh.edges)*2,dtype=np.int32)
    mesh.edges.foreach_get('vertices',edge_array);edge_array=edge_array.reshape(-1,2)
    active=weights>1e-8
    edges=edge_array[active[edge_array[:,0]]|active[edge_array[:,1]]]
    a,b=edges.T
    length=np.linalg.norm(before[a]-before[b],axis=1)
    c=1/np.sqrt(np.maximum(length,.002))
    degree=np.bincount(a,c,minlength=len(before))+np.bincount(b,c,minlength=len(before))
    def average(v):
        result=v.copy()
        for k in range(3):
            total=np.bincount(a,c*v[b,k],minlength=len(v))+np.bincount(b,c*v[a,k],minlength=len(v))
            result[:,k]=np.divide(total,degree,out=v[:,k].copy(),where=degree>0)
        return result
    current=before.copy()
    # Two-pass fairing reduces thin stair-like noise without continuously shrinking
    # the broad source blouse folds. Displacement is explicitly bounded locally.
    for _ in range(24):
        current+=.45*weights[:,None]*(average(current)-current)
        current-=.48*weights[:,None]*(average(current)-current)
        d=current-before;distance=np.linalg.norm(d,axis=1)
        limit=.007*weights
        ratio=np.minimum(1,np.divide(limit,distance,out=np.ones_like(distance),where=distance>1e-15))
        current=before+d*ratio[:,None]
    if not np.array_equal(current[silhouette],before[silhouette]):raise RuntimeError('Shoulder silhouette moved')
    mesh.vertices.foreach_set('co',current.ravel());mesh.update()
    after_tree=_tree(mesh)
    leaf_reports=[]
    for name in ('Rebuilt_FrontCollar_L','Rebuilt_FrontCollar_R'):
        leaf=bpy.data.objects.get(name)
        if leaf is None:continue
        old=_coords(leaf.data);new=old.copy();S=len(old)//2
        if len(old)!=1274:raise RuntimeError('Unexpected clean collar topology')
        delta=np.zeros(S)
        for i,p in enumerate(old[:S]):
            u=(i//13)/48
            if u>.86:continue
            vals=[]
            for dx,dz in ((0,0),(-.002,0),(.002,0),(0,-.002),(0,.002)):
                oy=_front_y(before_tree,p[0]+dx,p[2]+dz);ny=_front_y(after_tree,p[0]+dx,p[2]+dz)
                if oy is not None and ny is not None:vals.append(ny-oy)
            if vals:delta[i]=float(np.median(vals))*(1-float(_ss(.76,.86,u)))
        grid=delta.reshape(49,13)
        for _ in range(3):
            g=grid.copy();g[1:-1]=.6*grid[1:-1]+.2*grid[:-2]+.2*grid[2:];grid=g
        delta=grid.ravel()
        for i,p in enumerate(old[:S]):
            if (i//13)/48>.78:continue
            ny=_front_y(after_tree,p[0],p[2])
            if ny is not None:delta[i]=min(delta[i],ny-.0035-p[1])
        delta=np.clip(delta,-.009,.009)
        new[:S,1]+=delta;new[S:,1]+=delta
        luv=_uv_hash(leaf.data);top=_topology(leaf.data)
        leaf.data.vertices.foreach_set('co',new.ravel());leaf.data.update()
        assert _uv_hash(leaf.data)==luv and _topology(leaf.data)==top
        leaf_reports.append({'object':name,'maximum_additional_depth_shift':float(np.max(np.abs(delta))),
                             'uv_exact':True,'topology_exact':True,'x_z_exact':bool(np.array_equal(old[:,[0,2]],new[:,[0,2]]))})
    actual=_coords(mesh);delta=actual-before;moved=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-10)
    mesh.calc_loop_triangles();ids=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int32)
    def areas(v):
        q=v[ids];return np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)*.5
    before_area=areas(before);after_area=areas(actual)
    if np.count_nonzero(after_area<1e-12)>np.count_nonzero(before_area<1e-12):raise RuntimeError('New degenerate triangles')
    if _topology(mesh)!=topology_before or _uv_hash(mesh)!=uv_before:raise RuntimeError('Topology/UV changed')
    if not np.isfinite(actual).all():raise RuntimeError('Nonfinite geometry')
    body['e4_shoulder_refinement_v2_applied']=True
    report['ridge_pass']={'iterations':24,'positive_step':.45,'negative_step':-.48,
        'maximum_allowed_local_xyz_shift':.007,'moved_vertices':int(len(moved)),
        'max_actual_xyz_shift':float(np.max(np.linalg.norm(delta,axis=1))),
        'mean_xyz_shift':float(np.mean(np.linalg.norm(delta[moved],axis=1))),
        'upper_silhouette_vertices_fixed':int(np.count_nonzero(silhouette & (np.abs(x)<.31) & (z>.50))),
        'boundary_vertices_fixed':len(boundary),'uv_exact':True,'topology_exact':True,
        'degenerate_triangles_before':int(np.count_nonzero(before_area<1e-12)),
        'degenerate_triangles_after':int(np.count_nonzero(after_area<1e-12)),
        'collar_additional_refit':leaf_reports}
    report['overall_delta']={'moved_vertices':int(np.count_nonzero(np.linalg.norm(actual-source,axis=1)>1e-10)),
        'maximum_displacement':float(np.max(np.linalg.norm(actual-source,axis=1))),
        'geometry_scope':'Front shoulder and old-collar support only; no body remesh, no topology/UV change, original broad folds and shoulder outline retained.',
        'limitation':'No claim of global continuous collision testing or customer appearance acceptance.'}
    return report
