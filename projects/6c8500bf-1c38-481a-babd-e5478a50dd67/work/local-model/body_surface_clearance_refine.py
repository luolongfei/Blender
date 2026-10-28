"""Remove obsolete internal garment walls around the completed anatomy.

The body remains complete and unmodified. Three overlapping blouse envelopes
are sewn at their true Boolean intersection; sleeves and bodice retain their
names, materials, visible surfaces and interpolated UV/skin data. Socks gain
an actual foot envelope inside the shoes instead of cutting across the instep.
"""
from pathlib import Path
import bpy
import bmesh
import hashlib
import json
import math
import runpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

VERSION = "anatomical-garment-interiors-02"
KEY = "body_surface_clearance_version"


def _activate(obj):
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _snapshot(obj):
    previous = [(m, m.show_viewport) for m in obj.modifiers]
    try:
        for modifier in obj.modifiers:
            if modifier.type in {"ARMATURE", "SOLIDIFY"}:
                modifier.show_viewport = False
        bpy.context.view_layer.update()
        deps = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(
            obj.evaluated_get(deps), preserve_all_data_layers=True, depsgraph=deps)
    finally:
        for modifier, show in previous:
            modifier.show_viewport = show
        bpy.context.view_layer.update()
    mesh.calc_loop_triangles()
    points = [v.co.copy() for v in mesh.vertices]
    triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    loops = [tuple(t.loops) for t in mesh.loop_triangles]
    groups = {g.index: g.name for g in obj.vertex_groups}
    weights = [{groups[g.group]: g.weight for g in v.groups if g.weight > 1e-8}
               for v in mesh.vertices]
    layers = {layer.name: [tuple(value.uv) for value in layer.data]
              for layer in mesh.uv_layers}
    return {
        "object": obj, "mesh": mesh, "points": points, "triangles": triangles,
        "loops": loops, "weights": weights, "uv": layers,
        "tree": BVHTree.FromPolygons(points, triangles, all_triangles=True),
        "activeUV": next((u.name for u in mesh.uv_layers if u.active_render), None),
    }


def _bary(point, a, b, c):
    u, v, w = b-a, c-a, point-a
    d00, d01, d11 = u.dot(u), u.dot(v), v.dot(v)
    denominator = d00*d11-d01*d01
    if abs(denominator) < 1e-20:
        return (1., 0., 0.)
    y = (d11*w.dot(u)-d01*w.dot(v))/denominator
    z = (d00*w.dot(v)-d01*w.dot(u))/denominator
    weights = [max(0., 1-y-z), max(0., y), max(0., z)]
    total = sum(weights)
    return [w/total for w in weights]


def _sample(source, point):
    hit, _, index, distance = source["tree"].find_nearest(point)
    ids = source["triangles"][index]
    bary = _bary(hit, *(source["points"][i] for i in ids))
    weights = {}
    for vertex, amount in zip(ids, bary):
        for name, value in source["weights"][vertex].items():
            weights[name] = weights.get(name, 0.) + amount*value
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Missing interpolated garment skin weights.")
    weights = {name: value/total for name, value in weights.items() if value > 1e-8}
    uvs = {}
    for name, values in source["uv"].items():
        corners = [values[i] for i in source["loops"][index]]
        us = [c[0] for c in corners]
        # UV seams already carry loop-specific coordinates in the source.
        uvs[name] = tuple(sum(c[d]*t for c, t in zip(corners, bary)) for d in (0, 1))
    return weights, uvs, distance


def _closed_mesh(snapshot, material, cap_material, *, foot=False):
    mesh = snapshot["mesh"].copy()
    mesh.materials.clear()
    for mat in material:
        mesh.materials.append(mat)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for face in bm.faces:
        face.material_index = snapshot["slot"]
    if foot:
        # Offset real skin by 1.2 mm to form the fabric envelope. Above the
        # instep it joins the original visible sock tube.
        bm.normal_update()
        for vertex in bm.verts:
            vertex.co += vertex.normal * .0012
        result = bmesh.ops.bisect_plane(
            bm, geom=[*bm.verts, *bm.edges, *bm.faces],
            plane_co=(0, 0, .110), plane_no=(0, 0, 1),
            clear_outer=True, clear_inner=False, dist=1e-7)
    boundary = [edge for edge in bm.edges if edge.is_boundary]
    caps = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)["faces"]
    for face in caps:
        face.material_index = cap_material
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bad = sum(not edge.is_manifold for edge in bm.edges)
    if bad:
        bm.free()
        raise ValueError("Garment envelope is not closed: " + snapshot["object"].name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh


def _union(sources, collection, *, foot_index=None):
    markers = [bpy.data.materials.new("QA envelope " + str(i))
               for i in range(len(sources)+1)]
    cap_slot = len(sources)
    objects = []
    try:
        for i, source in enumerate(sources):
            source["slot"] = i
            mesh = _closed_mesh(source, markers, cap_slot, foot=i == foot_index)
            obj = bpy.data.objects.new("QA garment envelope " + str(i), mesh)
            collection.objects.link(obj)
            objects.append(obj)
        base = objects[0]
        _activate(base)
        for operand in objects[1:]:
            modifier = base.modifiers.new("Actual garment envelope union", "BOOLEAN")
            modifier.operation = "UNION"
            modifier.solver = "EXACT"
            modifier.object = operand
            modifier.material_mode = "INDEX"
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        base.data.update()
        result = base.data.copy()
        # The union's only retained artificial faces would close actual
        # neck, hem and wrist openings. Omit these caps when creating cloth.
        return result, cap_slot
    finally:
        for obj in objects:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
        for marker in markers:
            if marker.users == 0:
                bpy.data.materials.remove(marker)


def _plan_union(sources, collection, *, foot_index=None, single_output=False):
    union, cap_slot = _union(sources, collection, foot_index=foot_index)
    try:
        points = [v.co.copy() for v in union.vertices]
        origin_sets = [set() for _ in points]
        for face in union.polygons:
            if face.material_index == cap_slot:
                continue
            for index in face.vertices:
                origin_sets[index].add(face.material_index)
        weights = []
        for point, origins in zip(points, origin_sets):
            if not origins:
                weights.append({})
                continue
            combined = {}
            for origin in origins:
                skin = _sample(sources[origin], point)[0]
                for name, amount in skin.items():
                    combined[name] = combined.get(name, 0.) + amount/len(origins)
            # Equal boundary weights across objects keep the sewn edge
            # coincident during shoulder and elbow deformation.
            combined = dict(sorted(combined.items(), key=lambda p:p[1], reverse=True)[:4])
            total = sum(combined.values())
            weights.append({name:value/total for name,value in combined.items()})
        outputs = []
        for output_index, source in enumerate(sources[:1] if single_output else sources):
            selected = [p for p in union.polygons
                        if p.material_index != cap_slot and
                        (single_output or p.material_index == output_index)]
            used = sorted({index for p in selected for index in p.vertices})
            mapping = {old:new for new,old in enumerate(used)}
            mesh = bpy.data.meshes.new(source["object"].name + " · anatomical garment passage")
            mesh.from_pydata([points[i] for i in used], [],
                             [tuple(mapping[i] for i in p.vertices) for p in selected])
            for mat in source["object"].data.materials:
                mesh.materials.append(mat)
            for polygon in mesh.polygons:
                polygon.use_smooth = True
            for name in source["uv"]:
                layer = mesh.uv_layers.new(name=name)
                for polygon, original in zip(mesh.polygons, selected):
                    for corner, loop in enumerate(polygon.loop_indices):
                        point = mesh.vertices[mesh.loops[loop].vertex_index].co
                        retained = union.uv_layers.get(name)
                        if retained and (not single_output or original.material_index == 0):
                            # Boolean interpolation retains the actual corner
                            # side of an existing UV seam; nearest-point UV
                            # lookup alone would choose either seam island.
                            layer.data[loop].uv = retained.data[
                                original.loop_indices[corner]].uv
                            continue
                        # A sock's new foot uses the same cloth map, sampled
                        # from the lower tube; visible tube UVs stay unchanged.
                        uv_source = sources[output_index if not single_output else 0]
                        uv = _sample(uv_source, point)[1][name]
                        layer.data[loop].uv = uv
                layer.active_render = name == source["activeUV"]
            mesh.update()
            outputs.append((source["object"], mesh, [weights[i] for i in used]))
        return outputs, {
            "inputObjects": [s["object"].name for s in sources],
            "unionVertices": len(points),
            "sewnSharedVertices": sum(len(origins) > 1 for origins in origin_sets),
            "surfaceFaces": sum(p.material_index != cap_slot for p in union.polygons),
            "artificialCapsDiscarded": sum(p.material_index == cap_slot for p in union.polygons),
        }
    finally:
        marker_materials = list(union.materials)
        bpy.data.meshes.remove(union)
        for material in marker_materials:
            if material.name.startswith("QA envelope ") and material.users == 0:
                bpy.data.materials.remove(material)


def _fingerprint(obj):
    h = hashlib.sha256()
    for vertex in obj.data.vertices:
        h.update(repr(tuple(vertex.co)).encode())
        h.update(repr([(g.group, g.weight) for g in vertex.groups]).encode())
    for polygon in obj.data.polygons:
        h.update(repr(tuple(polygon.vertices)).encode())
    return h.hexdigest()

def _ease_inner_arm_weights(plans):
    def smooth(a,b,value):
        t=max(0.,min(1.,(value-a)/(b-a)))
        return t*t*(3-2*t)
    changed=0
    for obj,mesh,weights in plans:
        if not obj.name.startswith('Blouse_'):
            continue
        for vertex,skin in zip(mesh.vertices,weights):
            x,y,z=vertex.co;ax=abs(x)
            if not (.13<ax<.23 and 1.16<z<1.245):
                continue
            name=('left' if x>=0 else 'right')+'UpperArm'
            previous=skin.get(name,0.)
            height=1-smooth(1.205,1.245,z)
            requested=max(previous,smooth(.135,.215,ax))
            target=previous+(requested-previous)*height
            if target-previous<1e-7:
                continue
            remainder=max(1e-12,1-previous)
            for bone in list(skin):
                if bone!=name:skin[bone]*=(1-target)/remainder
            skin[name]=target
            changed+=1
    return {'vertices':changed,'region':'inferior sleeve-root/armhole',
            'method':'smooth chest-to-upper-arm transition over 13.5–21.5 cm from center; shared seam positions use the same field'}


def apply(collection=None, rig=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    rig = rig or bpy.data.objects["Character_Rig"]
    bodice = bpy.data.objects["Blouse_Tucked_Bodice"]
    if bodice.get(KEY) == VERSION:
        return json.loads(bodice["body_surface_clearance_report"])
    if bodice.get(KEY):
        raise ValueError("Different garment clearance version already applied.")
    body = [obj for obj in collection.all_objects if obj.name.startswith("Body_Complete_")]
    before_body = {obj.name:_fingerprint(obj) for obj in body}
    old_selected = list(bpy.context.selected_objects)
    old_active = bpy.context.view_layer.objects.active
    snapshots = []
    plans = []
    report = {"version": VERSION, "bodyGeometryAndWeightsUnchanged": True, "operations": []}
    try:
        blouse = [_snapshot(bpy.data.objects[name]) for name in (
            "Blouse_Tucked_Bodice", "Blouse_Puff_Sleeve_L", "Blouse_Puff_Sleeve_R")]
        snapshots.extend(blouse)
        output, operation = _plan_union(blouse, collection)
        plans.extend(output)
        report["operations"].append(operation)
        for side in ("L", "R"):
            sock = _snapshot(bpy.data.objects["Sock_Ribbed_" + side])
            foot = _snapshot(bpy.data.objects["Body_Complete_Shin_Foot_" + side])
            snapshots.extend((sock, foot))
            output, operation = _plan_union(
                [sock, foot], collection, foot_index=1, single_output=True)
            plans.extend(output)
            report["operations"].append(operation)
        trunk = _snapshot(bpy.data.objects["Body_Complete_Trunk_Arms_Pelvis"])
        snapshots.append(trunk)
        anatomy = [trunk] + [s for s in snapshots
                            if s["object"].name.startswith("Body_Complete_Shin_Foot_")]
        helper_dir = Path(__file__).resolve().parent
        report["innerArmWeightRefinement"] = _ease_inner_arm_weights(plans)
        report["localFabricAllowance"] = runpy.run_path(
            str(helper_dir / "body_garment_clearance_fit.py"))["fit"](
                plans, anatomy, helper_dir / "body_clearance_motion_probes.json")
        report["objects"] = []
        for obj, mesh, weights in plans:
            old = obj.data
            old.use_fake_user = True
            obj["garment_clearance_original_mesh"] = old.name
            old_count = len(old.polygons)
            obj.data = mesh
            names = sorted({name for skin in weights for name in skin})
            obj.vertex_groups.clear()
            groups = {name:obj.vertex_groups.new(name=name) for name in names}
            for index, skin in enumerate(weights):
                for name, amount in skin.items():
                    groups[name].add([index], amount, "REPLACE")
            for modifier in list(obj.modifiers):
                if modifier.type == "SUBSURF":
                    obj.modifiers.remove(modifier)
            for modifier in obj.modifiers:
                if modifier.type == "SOLIDIFY":
                    modifier.use_even_offset = False
                    # Skin the already thickened fabric, exactly as the
                    # exported runtime does. Pose-dependent miter expansion
                    # at tiny Boolean triangles is not physical cloth.
                    source_index = list(obj.modifiers).index(modifier)
                    arm_index = next((i for i,m in enumerate(obj.modifiers)
                                      if m.type == "ARMATURE"), source_index)
                    if source_index > arm_index:
                        obj.modifiers.move(source_index,arm_index)
            obj[KEY] = VERSION
            report["objects"].append({
                "name": obj.name, "oldBaseFaces": old_count, "newFaces": len(mesh.polygons),
                "vertices": len(mesh.vertices), "uvLayers": list(mesh.uv_layers.keys()),
                "maxWeightError": max(abs(sum(s.values())-1) for s in weights),
                "bakedExistingSubdivision": True,
                "originalMeshRetained": old.name,
                "boundedThicknessBeforeArmature": True,
            })
        if before_body != {obj.name:_fingerprint(obj) for obj in body}:
            raise ValueError("Completed anatomy changed unexpectedly.")
        report["anatomySha256"] = before_body
        report["scope"] = (
            "Blouse internal overlap walls removed by exact outer-envelope union; "
            "shared shoulder seam weights synchronized. Socks form a real foot "
            "cover 1.2mm outside completed skin inside the shoe. Original anatomy, "
            "head, rig, animation and physics untouched.")
        bodice["body_surface_clearance_report"] = json.dumps(report)
        bpy.context.view_layer.update()
        return report
    finally:
        for snapshot in snapshots:
            mesh = snapshot["mesh"]
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in old_selected:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if old_active and old_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = old_active
