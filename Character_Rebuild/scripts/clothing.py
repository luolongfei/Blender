"""Conservative edits to the original Hunyuan mesh, called by the root integrator.

No file saves, renders, export, remeshing or reference modification happen here.
Coordinates are the original metre-free GLB dimensions, after Blender axis conversion.
Original face-loop UV coordinates survive welding and bisect operations. New local
shoulder repair faces intentionally use a separate plain blouse material until bake.
"""
import json
import math
import hashlib
from pathlib import Path

import bpy
import bmesh
from mathutils import Matrix, Vector, kdtree


def _bounds(vertices):
    vv = list(vertices)
    if not vv:
        return None
    return {"min": [min(v.co[k] for v in vv) for k in range(3)],
            "max": [max(v.co[k] for v in vv) for k in range(3)]}


def _boundary_components(bm):
    remaining = set(e for e in bm.edges if e.is_boundary)
    components = []
    while remaining:
        seed = remaining.pop()
        edges, queue = {seed}, [seed]
        while queue:
            edge = queue.pop()
            for vertex in edge.verts:
                for candidate in vertex.link_edges:
                    if candidate in remaining:
                        remaining.remove(candidate)
                        edges.add(candidate)
                        queue.append(candidate)
        vertices = {v for e in edges for v in e.verts}
        degree = {v: sum(e in edges for e in v.link_edges) for v in vertices}
        components.append({"edges": edges, "vertices": vertices,
                           "closed": all(d == 2 for d in degree.values()),
                           "bounds": _bounds(vertices)})
    return components


def _describe_boundary(component, head_cut, wrist_cut):
    vertices = component["vertices"]
    # Exact bisect edges are expected openings for replacement head/wrists.
    at_head = sum(abs(v.co.z-head_cut) < 2e-5 for v in vertices)
    at_wrist = sum(abs(abs(v.co.x)-wrist_cut) < 2e-5 for v in vertices)
    return {"vertices": len(vertices), "edges": len(component["edges"]),
            "closedLoop": component["closed"], "bounds": component["bounds"],
            "verticesOnHeadCut": at_head, "verticesOnWristCut": at_wrist,
            "containsExpectedPartOpening": at_head > 0 or at_wrist > 0}


def _source_sampler():
    p = Path(__file__).resolve().parents[1] / 'reports' / 'shoulder-source-colors.json'
    data = json.loads(p.read_text(encoding='utf-8'))
    if data['sourceSha256'] != '105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2':
        raise RuntimeError('Shoulder colour samples do not match the inspected source')
    points = data['points']
    kd = kdtree.KDTree(len(points))
    for i, row in enumerate(points):
        kd.insert(row[:3], i)
    kd.balance()

    def sample(co):
        nearest, index, distance = kd.find(co)
        if distance > .012:
            return None
        return points[index][3:6]
    return sample


def _linear(c):
    return c / 12.92 if c <= .04045 else ((c+.055)/1.055)**2.4


def _repair_material(obj):
    name = 'Rebuild_Blouse_LocalShoulderRepair'
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    shader = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    rgb = (.921, .878, .816)
    shader.inputs['Base Color'].default_value = (*map(_linear, rgb), 1)
    shader.inputs['Metallic'].default_value = 0
    shader.inputs['Roughness'].default_value = .90
    material.diffuse_color = (*rgb, 1)
    material['reconstruction_note'] = 'Inferred local blouse shoulder surface replacing fused generated hair; retexture/bake before final export.'
    if material.name not in obj.data.materials:
        obj.data.materials.append(material)
    return list(obj.data.materials).index(material)


def cleanup_original(obj, *, head_cut=.632, wrist_cut=.7525,
                     cut_head=True, cut_hands=True, remove_rear_tail=True,
                     fill_local_tail_holes=True, remove_old_neck_skin=True,
                     weld_distance=1e-6):
    """Return (obj, report) after conservative original body/clothing preparation.

    Call once on an independent source copy. Head and wrist openings are not capped.
    Any tail hole that merges into the neck opening is left explicit in the report,
    so the integrator can build a collar/back yoke instead of silently filling a neck.
    """
    if obj.type != 'MESH':
        raise TypeError('cleanup_original expects the imported source MESH')
    if obj.get('rebuild_clothing_cleanup_applied'):
        raise RuntimeError('Cleanup already applied; reopen the stage-A source rather than cutting twice')
    if obj.data.shape_keys:
        raise RuntimeError('Unexpected shape keys on original source; refusing destructive topology edits')
    source_path = Path(__file__).resolve().parents[2] / 'character-hq-textured.glb'
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != '105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2':
        raise RuntimeError('Original GLB changed; regenerate diagnostic colour samples before cleanup')
    if len(obj.data.polygons) != 200000:
        raise RuntimeError('Expected the unmodified 200,000-triangle input mesh; reopen stage A')
    report = {'source': 'character-hq-textured.glb',
              'sourceSha256': '105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2',
              'coordinateSystem': 'Original Blender X,-glTF-Z,glTF-Y; no scale change',
              'headCut': head_cut, 'wristCut': wrist_cut,
              'originalVertices': len(obj.data.vertices), 'originalFaces': len(obj.data.polygons),
              'positionWeldDistance': weld_distance, 'preservedUVLayers': [u.name for u in obj.data.uv_layers],
              'globalRemesh': False, 'globalSmoothing': False, 'warnings': []}
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # Work in the already inspected coordinates regardless of importer object rotation.
    transform = obj.matrix_world.copy()
    bmesh.ops.transform(bm, matrix=transform, verts=list(bm.verts))
    initial_bounds = _bounds(bm.verts)
    expected_min = (-.9908958077, -.2336365879, -.9990763664)
    expected_max = (.9878259897, .2476680130, .9946864247)
    if any(abs(initial_bounds['min'][k]-expected_min[k]) > 1e-4 or
           abs(initial_bounds['max'][k]-expected_max[k]) > 1e-4 for k in range(3)):
        bm.free()
        raise RuntimeError('Source world transform differs from measured input; do not use untransformed cut coordinates')
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld_distance)
    report['weldedVertices'] = len(bm.verts)
    if cut_head:
        bmesh.ops.bisect_plane(bm, geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                              plane_co=(0, 0, head_cut), plane_no=(0, 0, 1),
                              dist=1e-7, clear_outer=True, clear_inner=False)
    if cut_hands:
        for side in (-1, 1):
            bmesh.ops.bisect_plane(bm, geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
                                  plane_co=(side*wrist_cut, 0, 0), plane_no=(side, 0, 0),
                                  dist=1e-7, clear_outer=True, clear_inner=False)
    report['facesAfterPartCuts'] = len(bm.faces)
    uv_layer = bm.loops.layers.uv.active
    selected = set()
    sample = _source_sampler()
    if remove_old_neck_skin:
        old_neck = set()
        for face in bm.faces:
            c = face.calc_center_median()
            if not (.597 < c.z < head_cut+.0001 and abs(c.x) < .068 and -.065 < c.y < .097):
                continue
            rgb = sample(c)
            if rgb is not None and rgb[0] > 150 and rgb[0]-rgb[1] > 12 and rgb[1]-rgb[2] > 20:
                old_neck.add(face)
        # Remove only skin texels in the new neck volume, not the original cream
        # collar. A narrow connected skin boundary is an intended neck opening.
        report['removedOldNeckSkinFaces'] = len(old_neck)
        report['removedOldNeckSkinBounds'] = _bounds({v for f in old_neck for v in f.verts})
        if old_neck:
            bmesh.ops.delete(bm, geom=list(old_neck), context='FACES')
    if remove_rear_tail:
        for face in bm.faces:
            centre = face.calc_center_median()
            x, y, z = centre
            if not (.548 < z < head_cut+.0001 and .040 < abs(x) < .205 and y > .025):
                continue
            rgb = sample(centre)
            luminance = None if rgb is None else .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            # Core hair projects behind the back blouse (Y>.080), including white
            # texture contamination. Near the garment, require confirmed dark hair.
            protruding_core = (y > .080 and .055 < abs(x) < .19 and z > .558)
            dark_hair = luminance is not None and luminance < 135 and z > .554
            if protruding_core or dark_hair:
                selected.add(face)
        # One-ring growth is restricted to the same shoulder-hair spatial zone.
        # This removes skinny leftover black triangles along the confirmed boundary.
        grown = set(selected)
        for face in selected:
            for edge in face.edges:
                for adjacent in edge.link_faces:
                    c = adjacent.calc_center_median()
                    if .550<c.z<head_cut+.0001 and .043<abs(c.x)<.200 and c.y>.030:
                        rgb = sample(c)
                        luma = 255 if rgb is None else .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                        if luma < 170 or c.y > .080:
                            grown.add(adjacent)
        selected = grown
        report['removedRearTailFaces'] = len(selected)
        report['removedRearTailBounds'] = _bounds({v for f in selected for v in f.verts})
        if selected:
            bmesh.ops.delete(bm, geom=list(selected), context='FACES')
        # Remove only now-disconnected geometry, not additional surface islands.
        loose = [v for v in bm.verts if not v.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context='VERTS')
    report['tailBoundaryBeforeFill'] = []
    repaired_faces = []
    pending_boundaries = []
    for component in _boundary_components(bm):
        desc = _describe_boundary(component, head_cut, wrist_cut)
        box = component['bounds']
        mid = [(box['min'][k]+box['max'][k])/2 for k in range(3)]
        expected_neck_opening = (box['min'][2] > .585 and
                                 box['min'][0] > -.09 and box['max'][0] < .09)
        is_tail_region = (box['min'][2] > .53 and box['max'][2] < .64 and
                          box['max'][1] > .05 and max(abs(box['min'][0]),abs(box['max'][0])) < .225
                          and not expected_neck_opening)
        if not is_tail_region:
            continue
        report['tailBoundaryBeforeFill'].append(desc)
        if (fill_local_tail_holes and component['closed'] and
                not desc['containsExpectedPartOpening'] and
                box['min'][1] > .015 and
                box['max'][0]-box['min'][0] < .19 and len(component['edges']) < 600):
            # A closed bounded local garment defect can be repaired without altering
            # retained vertices. The new face is triangulated rather than exported as
            # an arbitrary large ngon. No expected neck/wrist opening is filled.
            result = bmesh.ops.holes_fill(bm, edges=list(component['edges']), sides=0)
            new_faces = [f for f in result.get('faces', []) if f.is_valid]
            if new_faces:
                index = _repair_material(obj)
                for face in new_faces:
                    face.material_index = index
                    face.smooth = True
                    if uv_layer:
                        for loop in face.loops:
                            loop[uv_layer].uv = (.5, .5)
                tri_result = bmesh.ops.triangulate(bm, faces=new_faces)
                repaired_faces.extend(tri_result.get('faces', []))
        else:
            pending_boundaries.append(desc)
    if pending_boundaries:
        report['warnings'].append('Rear hair removal has a boundary shared with the neck cut or a complex non-loop. It is deliberately not capped; a local back-yoke/collar bridge must be reviewed and rebuilt before delivery.')
    report['unresolvedTailBoundaries'] = pending_boundaries
    report['newLocalBlouseTriangles'] = len(repaired_faces)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    report['finalBoundaries'] = [_describe_boundary(c, head_cut, wrist_cut) for c in _boundary_components(bm)]
    report['remainingVertices'] = len(bm.verts)
    report['remainingFaces'] = len(bm.faces)
    report['bodyClothingGeometryReadyForReview'] = not pending_boundaries
    bm.to_mesh(obj.data)
    bm.free()
    obj.matrix_world = Matrix.Identity(4)
    obj.data.update()
    obj.name = 'Rebuild_Body_Clothing_SourceRetained'
    obj['rebuild_clothing_cleanup_applied'] = True
    obj['reconstruction_note'] = 'Original garment geometry retained; UV seams welded geometrically with loop UVs retained; head/wrist cuts and local rear generated-hair removal. See cleanup report for unresolved boundaries.'
    return obj, report
