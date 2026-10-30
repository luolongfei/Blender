"""Repair the measured, integrated D-stage back yoke and neck opening.

The fitted plan is computed from the actual D mesh capture. It replaces the
failed open rear shoulder area with a quad annulus welded to its measured edge;
it does not merely put a floating cover in front of the hole. The original front
Peter Pan collar outside the measured cut is retained. No saves/renders here.
"""
import json
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


def _cloth_material():
    material = bpy.data.materials.get('Rebuilt_blouse')
    if material is not None:
        return material
    import materials
    return materials.principled_material('Rebuilt_blouse', (.940, .890, .805), .88)


def _boundary_report(bm):
    remaining = set(e for e in bm.edges if e.is_boundary)
    result = []
    while remaining:
        e = remaining.pop()
        queue, edges = [e], {e}
        while queue:
            current = queue.pop()
            for vertex in current.verts:
                for other in vertex.link_edges:
                    if other in remaining:
                        remaining.remove(other)
                        edges.add(other)
                        queue.append(other)
        vertices = {v for e in edges for v in e.verts}
        result.append({'edges':len(edges), 'vertices':len(vertices),
                       'closedLoop':all(sum(e in edges for e in v.link_edges)==2 for v in vertices),
                       'min':[min(v.co[k] for v in vertices) for k in range(3)],
                       'max':[max(v.co[k] for v in vertices) for k in range(3)]})
    return result


def _make_collar(obj, data, material):
    mesh = bpy.data.meshes.new('Rebuilt_BackCollar_EditableQuads')
    mesh.from_pydata(data['vertices'], [], data['faces'])
    mesh.update()
    uv = mesh.uv_layers.new(name='BackCollar_UV')
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        for li, vi in zip(polygon.loop_indices, polygon.vertices):
            uv.data[li].uv = data['uv'][vi]
    mesh.materials.append(material)
    bm = bmesh.new(); bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh); bm.free(); mesh.update()
    collar = bpy.data.objects.new('Rebuilt_BackCollar_Band', mesh)
    obj.users_collection[0].objects.link(collar)
    collar['reconstruction_note'] = 'New closed-thickness rear collar band, fitted to the actual reconstructed back yoke. Original front round collar remains on the source-retained body.'
    collar['inferred_detail'] = 'Rear collar thickness and hidden construction inferred conservatively from the supplied rear illustration.'
    return collar


def apply_integrated_back_yoke(obj):
    """Modify the exact D-stage retained-body object; return a serializable report."""
    if obj.type != 'MESH' or not obj.get('rebuild_clothing_cleanup_applied'):
        raise RuntimeError('Expected the integrated D-stage cleaned original mesh')
    if obj.get('rebuild_back_yoke_v2_applied'):
        raise RuntimeError('Back yoke v2 already applied; do not apply twice')
    if any(abs(obj.matrix_world[i][j]-(1 if i==j else 0))>1e-7 for i in range(4) for j in range(4)):
        raise RuntimeError('D-stage body transform changed; rebuild the measured plan')
    path = Path(__file__).resolve().parents[1]/'reports'/'back-yoke-plan.json'
    plan = json.loads(path.read_text(encoding='utf-8'))
    if bpy.data.objects.get('Rebuilt_BackCollar_Band'):
        raise RuntimeError('Existing rebuilt rear collar found; preserve previous stage')
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    if max(plan['boundaryVertexIndices'])>=len(bm.verts) or max(plan['deleteFaces'])>=len(bm.faces):
        bm.free(); raise RuntimeError('D-stage mesh index range no longer matches measured plan')
    boundary = [bm.verts[i] for i in plan['boundaryVertexIndices']]
    max_error = max((v.co-Vector(p)).length for v,p in zip(boundary,plan['boundaryOriginal']))
    if max_error>2e-6:
        bm.free(); raise RuntimeError(f'D-stage boundary moved by {max_error}; refusing stale repair plan')
    before_boundaries = _boundary_report(bm)
    remove = [bm.faces[i] for i in plan['deleteFaces']]
    bmesh.ops.delete(bm, geom=remove, context='FACES')
    if not all(v.is_valid for v in boundary):
        bm.free(); raise RuntimeError('Measured boundary was removed unexpectedly; original mesh untouched')
    for vertex,position in zip(boundary,plan['boundarySmoothed']):
        vertex.co = position
    material = _cloth_material()
    if material.name not in obj.data.materials:
        obj.data.materials.append(material)
    slot = list(obj.data.materials).index(material)
    data = plan['patch']
    vertices = boundary + [bm.verts.new(p) for p in data['vertices'][len(boundary):]]
    uv = bm.loops.layers.uv.active or bm.loops.layers.uv.new('UVMap')
    new_faces = []
    for indices in data['faces']:
        face = bm.faces.new([vertices[i] for i in indices])
        face.material_index = slot
        face.smooth = True
        for loop,index in zip(face.loops,indices):
            loop[uv].uv = data['uv'][index]
        new_faces.append(face)
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm,geom=loose,context='VERTS')
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    after_boundaries = _boundary_report(bm)
    unexpected = []
    for item in after_boundaries:
        xmin,ymin,zmin = item['min']; xmax,ymax,zmax = item['max']
        wrist = abs(abs(xmin)-.7525)<1e-5 and abs(xmax-xmin)<1e-5
        neck = (xmin>=-.05501 and xmax<=.05501 and ymin>=-.02801 and ymax<=.08801
                and zmin>=.61499 and zmax<=.62101 and item['closedLoop'])
        if not wrist and not neck:
            unexpected.append(item)
    if unexpected:
        bm.free()
        raise RuntimeError('Back-yoke repair would leave an unexpected opening; original mesh not committed: '+json.dumps(unexpected))
    bm.to_mesh(obj.data); bm.free(); obj.data.update()
    collar = _make_collar(obj,plan['collar'],material)
    obj['rebuild_back_yoke_v2_applied'] = True
    obj['back_yoke_uv_note'] = 'New yoke faces have a solid material and separate logical UV domain in UVMap. Repack/bake with explicit original-UV lookup before merging atlas materials.'
    obj['back_yoke_geometry_note'] = 'Measured rear shoulder opening replaced by welded quad annulus; correct neck and wrist openings remain. Rear old button/collar pillar removed.'
    return {'source':'D-integrated-geometry.blend','object':obj.name,
            'created_objects':[collar.name],'removed_faces':len(plan['deleteFaces']),
            'new_yoke_quads':len(data['faces']),'new_collar_quads':len(plan['collar']['faces']),
            'old_boundary_vertices':len(boundary),
            'max_boundary_displacement':plan['maxBoundaryDisplacement'],
            'boundary_measurement_error':max_error,'before_boundaries':before_boundaries,
            'after_boundaries':after_boundaries,'unexpected_openings':unexpected,
            'front_collar':'Original front collar faces outside y>-.035 cut retained',
            'neck_opening':{'xRadius':.055,'yCentre':.030,'yRadius':.058,'zRange':[.615,.621]},
            'material_note':'New inferred cloth uses clean simple PBR; bake separately or use explicit material-aware body atlas repack',
            'appearance_review':'Requires actual neutral back/side/quarter renders after this operation; no appearance acceptance claimed'}
