"""Local E4 shoulder fairing; preserves topology, UVs and front silhouette.

Only the depth coordinate of the existing shoulder/old-collar support is edited.
The new collar leaves receive a matching depth-only refit; body/leaf X and Z
coordinates are byte-preserved. No file saving or rendering in apply().
"""
import hashlib
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _ss(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)


def _coords(mesh):
    a=np.empty(len(mesh.vertices)*3,dtype=np.float64)
    mesh.vertices.foreach_get('co',a)
    return a.reshape(-1,3)


def _uv_hash(mesh):
    h=hashlib.sha256()
    for layer in mesh.uv_layers:
        a=np.empty(len(layer.data)*2,dtype=np.float32)
        layer.data.foreach_get('uv',a);h.update(layer.name.encode());h.update(a.tobytes())
    return h.hexdigest()


def _topology(mesh):
    bm=bmesh.new();bm.from_mesh(mesh)
    result={'vertices':len(bm.verts),'edges':len(bm.edges),'faces':len(bm.faces),
            'boundary_edges':sum(e.is_boundary for e in bm.edges),
            'nonmanifold_edges':sum(not e.is_manifold and not e.is_boundary for e in bm.edges)}
    bm.free()
    h=hashlib.sha256()
    for p in mesh.polygons:h.update(np.array(p.vertices,dtype=np.int32).tobytes())
    result['connectivity_sha256']=h.hexdigest()
    return result


def _tree(mesh):
    bm=bmesh.new();bm.from_mesh(mesh)
    t=BVHTree.FromBMesh(bm);bm.free();return t


def _front_y(tree,x,z):
    p,n,index,d=tree.ray_cast(Vector((x,-.8,z)),Vector((0,1,0)))
    return p.y if p is not None and n.y<.35 else None


def apply_shoulder_refinement(body, iterations=70, step=.52, max_depth_shift=.016):
    if body.type!='MESH':raise TypeError('Expected E4 retained body mesh')
    if body.get('e4_shoulder_refinement_applied'):
        raise RuntimeError('Shoulder refinement already applied; reopen E4 before rerunning')
    if any(abs(body.matrix_world[i][j]-(1 if i==j else 0))>1e-7 for i in range(4) for j in range(4)):
        raise RuntimeError('Expected untransformed measured E4 body')
    mesh=body.data
    before=_coords(mesh); uv_before=_uv_hash(mesh); topology_before=_topology(mesh)
    before_tree=_tree(mesh)
    x,y,z=before.T
    weights=(1-_ss(.235,.305,np.abs(x)))*_ss(.515,.550,z)*(1-_ss(.624,.646,z))*(1-_ss(.012,.060,y))
    # Existing deliberate neck opening is fixed exactly, so no new gaps are opened.
    bm=bmesh.new();bm.from_mesh(mesh)
    fixed={v.index for e in bm.edges if e.is_boundary for v in e.verts}
    bm.free()
    if fixed:weights[list(fixed)]=0
    edge_array=np.empty(len(mesh.edges)*2,dtype=np.int32)
    mesh.edges.foreach_get('vertices',edge_array);edge_array=edge_array.reshape(-1,2)
    active=weights>1e-8
    keep=active[edge_array[:,0]]|active[edge_array[:,1]]
    ed=edge_array[keep];a,b=ed[:,0],ed[:,1]
    lengths=np.linalg.norm(before[a]-before[b],axis=1)
    conductance=1/np.sqrt(np.maximum(lengths,.0015))
    degree=np.bincount(a,conductance,minlength=len(before))+np.bincount(b,conductance,minlength=len(before))
    def average(values):
        s=np.bincount(a,conductance*values[b],minlength=len(before))+np.bincount(b,conductance*values[a],minlength=len(before))
        return np.divide(s,degree,out=values.copy(),where=degree>0)
    yy=y.copy();initial_curvature=average(yy)-yy
    for _ in range(iterations):
        yy+=step*weights*(average(yy)-yy)
        yy=np.clip(yy,y-max_depth_shift*weights,y+max_depth_shift*weights)
    after=before.copy();after[:,1]=yy
    moved=np.flatnonzero(np.abs(yy-y)>1e-10)
    mesh.vertices.foreach_set('co',after.ravel());mesh.update()
    after_tree=_tree(mesh)
    # Follow the changed support with the fitted leaves, using paired upper/lower
    # vertex indices to preserve actual 1.5mm thickness and their rounded design.
    collar_reports=[]
    for name in ('Rebuilt_FrontCollar_L','Rebuilt_FrontCollar_R'):
        leaf=bpy.data.objects.get(name)
        if leaf is None:continue
        old=_coords(leaf.data);new=old.copy();S=len(old)//2
        if len(old)!=1274:
            raise RuntimeError('Unexpected front collar topology; body fairing remains only in memory, do not save')
        delta=np.zeros(S)
        for i,p in enumerate(old[:S]):
            u=(i//13)/48
            if u>.86:continue
            samples=[]
            for dx,dz in ((0,0),(-.002,0),(.002,0),(0,-.002),(0,.002)):
                oy=_front_y(before_tree,p[0]+dx,p[2]+dz);ny=_front_y(after_tree,p[0]+dx,p[2]+dz)
                if oy is not None and ny is not None:samples.append(ny-oy)
            if samples:delta[i]=float(np.median(samples))*(1-float(_ss(.76,.86,u)))
        # Smooth only the refit displacement; leave original leaf form untouched.
        grid=delta.reshape(49,13)
        for _ in range(4):
            g=grid.copy();g[1:-1]=.6*grid[1:-1]+.2*grid[:-2]+.2*grid[2:];grid=g
        delta=grid.ravel()
        for i,p in enumerate(old[:S]):
            if (i//13)/48>.78:continue
            ny=_front_y(after_tree,p[0],p[2])
            if ny is not None:delta[i]=min(delta[i],ny-.0035-p[1])
        delta=np.clip(delta,-.018,.018)
        new[:S,1]+=delta;new[S:,1]+=delta
        luv=_uv_hash(leaf.data);ltop=_topology(leaf.data)
        leaf.data.vertices.foreach_set('co',new.ravel());leaf.data.update()
        assert _uv_hash(leaf.data)==luv and _topology(leaf.data)==ltop
        collar_reports.append({'object':name,'moved_vertices':int(np.count_nonzero(np.abs(delta)>1e-10)*2),
                               'max_depth_shift':float(np.max(np.abs(delta))),
                               'x_z_exact':bool(np.array_equal(old[:,[0,2]],new[:,[0,2]])),
                               'uv_exact':True,'topology_exact':True})
    actual=_coords(mesh)
    after_topology=_topology(mesh);uv_after=_uv_hash(mesh)
    mesh.calc_loop_triangles()
    indices=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int32)
    def area_values(v):
        p=v[indices];return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    old_area=area_values(before);new_area=area_values(actual)
    if not np.isfinite(actual).all():raise RuntimeError('Nonfinite fairing geometry')
    if after_topology!=topology_before or uv_after!=uv_before:raise RuntimeError('Topology/UV preservation failed')
    if np.count_nonzero(new_area<1e-12)>np.count_nonzero(old_area<1e-12):raise RuntimeError('Fairing introduced degenerate triangles')
    if not np.array_equal(actual[:,[0,2]],before[:,[0,2]]):raise RuntimeError('Front silhouette coordinates changed')
    body['e4_shoulder_refinement_applied']=True
    body['e4_shoulder_refinement_scope']='Local Y-depth fairing of front shoulders and old collar support; X/Z, topology, UV and neck boundary unchanged; fitted front leaves followed in depth only.'
    delta=actual-before
    final_curvature=average(actual[:,1])-actual[:,1]
    region=weights>.1
    return {'object':body.name,'iterations':iterations,'step':step,'maximum_allowed_depth_shift':max_depth_shift,
            'moved_vertices':int(len(moved)),
            'changed_bounds':{'min':actual[moved].min(axis=0).tolist(),'max':actual[moved].max(axis=0).tolist()},
            'maximum_actual_depth_shift':float(np.max(np.abs(delta[:,1]))),
            'mean_moved_depth_shift':float(np.mean(np.abs(delta[moved,1]))),
            'x_z_exact':True,'uv_exact':uv_after==uv_before,'uv_sha256':uv_after,
            'topology_before':topology_before,'topology_after':after_topology,
            'degenerate_triangles_before':int(np.count_nonzero(old_area<1e-12)),
            'degenerate_triangles_after':int(np.count_nonzero(new_area<1e-12)),
            'local_y_curvature_rms_before':float(np.sqrt(np.mean(initial_curvature[region]**2))),
            'local_y_curvature_rms_after':float(np.sqrt(np.mean(final_curvature[region]**2))),
            'collar_refit':collar_reports,
            'limitations':'Topology/finite/area/UV checks cover this delta. This is not a global self-intersection or customer appearance acceptance claim.'}
