"""Minimal measured E7 repairs: rear-yoke seam and hidden right collar underside.

One underside quad is explicitly triangulated with identical face-corner UVs;
all collar vertex positions remain exact. No saving or large smoothing. Each
repair is checked with the same full-object BVH + closed-triangle SAT used by
the final GLB verifier.
"""
import bpy
import bmesh
import numpy as np
from mathutils import Vector
import geometry_checks as gc
from shoulder_refine_e4 import _coords,_uv_hash,_topology


def _face_uv_signature(mesh):
    return {tuple(sorted(p.vertices)):{
        'material':p.material_index,'smooth':p.use_smooth,
        'layers':{layer.name:{int(mesh.loops[li].vertex_index):tuple(layer.data[li].uv)
                              for li in p.loop_indices} for layer in mesh.uv_layers}}
        for p in mesh.polygons}


def _bad_uv_count(mesh):
    mesh.calc_loop_triangles();result={}
    for layer in mesh.uv_layers:
        count=0
        for tri in mesh.loop_triangles:
            a,b,c=[layer.data[i].uv for i in tri.loops]
            if abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))*.5<=1e-14:count+=1
        result[layer.name]=count
    return result


def sat_object(obj):
    mesh=obj.data;mesh.calc_loop_triangles()
    points=[tuple(obj.matrix_world@v.co) for v in mesh.vertices]
    triangles=[tuple(t.vertices) for t in mesh.loop_triangles]
    polygons=[t.polygon_index for t in mesh.loop_triangles]
    return gc._self_contacts(points,triangles,polygons,enabled=True,
                             max_triangles=200000,max_candidates=3000000,
                             max_samples=30,tolerance=1e-8)


def repair_clothing_contacts_e7(body=None,collar=None):
    body=body or bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
    collar=collar or bpy.data.objects['Rebuilt_FrontCollar_R']
    if body.get('e7_local_contact_repair_applied') or collar.get('e7_local_contact_repair_applied'):
        raise RuntimeError('E7 local contact repair already applied')
    original={o.name:_coords(o.data) for o in (body,collar)}
    original_meshes={o.name:o.data for o in (body,collar)}
    uv={o.name:_uv_hash(o.data) for o in (body,collar)}
    topo={o.name:_topology(o.data) for o in (body,collar)}
    before={o.name:sat_object(o) for o in (body,collar)}
    if before[body.name].get('contacts')!=1 or before[collar.name].get('contacts')!=4:
        raise RuntimeError('Actual E7 contacts differ from measured 1+4 set; do not apply stale delta')
    expected=Vector((-.1688709706068039,.11592613160610199,.5500567555427551))
    if (body.data.vertices[62754].co-expected).length>2e-6:
        raise RuntimeError('Rear seam source vertex no longer matches measured E7 position')
    # Work transactionally on copied mesh datablocks; source file is never saved.
    body.data=body.data.copy();collar.data=collar.data.copy()
    # The yoke boundary vertex is 23.7 microunits across the neighboring retained
    # blouse triangle plane. Move only that vertex to a 75-microunit clearance.
    a,b,c=[body.data.vertices[i].co.copy() for i in (62805,62752,62750)]
    normal=(b-a).cross(c-a).normalized()
    vertex=body.data.vertices[62754]
    signed=normal.dot(vertex.co-a)
    vertex.co-=normal*(signed+.000075)
    body.data.update()
    # Four reported pairs are the same top/underside quad. The implicit diagonals
    # were opposite on this slightly nonplanar thin shell. Explicitly split ONLY
    # the hidden underside quad along the matching top diagonal. No collar point
    # moves, no thickness changes, and every original face-corner UV is retained.
    mesh=collar.data
    if len(mesh.vertices)!=1274:raise RuntimeError('Unexpected front collar topology')
    old_signatures=_face_uv_signature(mesh);old_bad_uv=_bad_uv_count(mesh)
    bm=bmesh.new();bm.from_mesh(mesh);bm.faces.ensure_lookup_table();bm.verts.ensure_lookup_table()
    face=bm.faces[639]
    if set(v.index for v in face.verts)!={705,706,718,719}:
        bm.free();raise RuntimeError('Hidden underside source quad no longer matches measured E7')
    uv_layers={name:bm.loops.layers.uv[name] for name in bm.loops.layers.uv.keys()}
    uv_values={name:{l.vert.index:l[layer].uv.copy() for l in face.loops} for name,layer in uv_layers.items()}
    material=face.material_index;smooth=face.smooth
    bm.faces.remove(face)
    for ids in [(705,719,718),(705,706,719)]:
        new=bm.faces.new([bm.verts[i] for i in ids]);new.material_index=material;new.smooth=smooth
        for loop,index in zip(new.loops,ids):
            for name,layer in uv_layers.items():loop[layer].uv=uv_values[name][index]
    bm.to_mesh(mesh);bm.free();mesh.update()
    new_signatures=_face_uv_signature(mesh)
    removed_key=(705,706,718,719)
    for key,value in old_signatures.items():
        if key!=removed_key:assert new_signatures[key]==value,'An unaffected collar face/UV changed'
    for key in [(705,718,719),(705,706,719)]:
        for layer,values in new_signatures[key]['layers'].items():
            assert all(values[i]==old_signatures[removed_key]['layers'][layer][i] for i in key)
    assert np.array_equal(_coords(mesh),original[collar.name]),'Collar vertex position changed'
    assert all(v==0 for v in _bad_uv_count(mesh).values()),'New degenerate UV triangle'
    after={o.name:sat_object(o) for o in (body,collar)}
    if any(r.get('status')!='completed' or r.get('contacts')!=0 for r in after.values()):
        # This callable leaves no failed geometry on the integrator's objects.
        for o in (body,collar):o.data=original_meshes[o.name]
        raise RuntimeError('Measured contact repair did not pass full-object SAT; restored original geometry: '+str(after))
    report={'beforeSAT':before,'afterSAT':after,'objects':{}}
    for o in (body,collar):
        actual=_coords(o.data);delta=actual-original[o.name]
        moved=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-12)
        if o==body:assert _uv_hash(o.data)==uv[o.name] and _topology(o.data)==topo[o.name]
        assert np.isfinite(actual).all()
        o['e7_local_contact_repair_applied']=True
        report['objects'][o.name]={'movedVertices':moved.tolist(),
            'changes':[{'vertex':int(i),'before':original[o.name][i].tolist(),'after':actual[i].tolist(),
                        'distance':float(np.linalg.norm(delta[i]))} for i in moved],
            'maximumDisplacement':float(np.max(np.linalg.norm(delta,axis=1))),
            'uvPerCornerPreserved':True,'uvBytesExact':o==body,'uvSha256Before':uv[o.name],
            'uvSha256After':_uv_hash(o.data),'topologyExact':o==body,
            'topologyBefore':topo[o.name],'topologyAfter':_topology(o.data),
            'degenerateUvTrianglesAfter':_bad_uv_count(o.data)}
    report['bodyReason']='One rear yoke boundary point crossed an adjacent retained blouse face; repaired by measured 0.00009872-unit inward displacement.'
    report['collarReason']='Right front collar top/underside implicit tessellation used opposite diagonals. One hidden underside quad replaced by two explicit triangles using the matching diagonal 705-719. All 1274 collar vertex positions and all per-corner UV coordinates unchanged; visible top faces unchanged.'
    report['scope']='Same-object full BVH/SAT checks at 1e-8; excludes same-polygon/shared-index/shared-position pairs, as in final verifier. Inter-object clearance is a separate scope.'
    return report
