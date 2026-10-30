"""One bounded, key-preserving neck/torso topology join in an R03 work copy.

run() constructs and checks a NEW mesh before hiding/renaming either source.
Cuts are planned once in Basis/world coordinates. The exact source vertex or
edge interpolation map is then reused for every head shape key, UV, colour and
weight. No evaluated subdivision is applied, no Boolean/remesh is used, and no
file is saved. Unsupported animation/topology conditions stop with a guard.
"""
import bpy
import bmesh
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from mathutils import Matrix, Vector

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = "R01 Head"
BODY = "Body_Complete_Trunk_Arms_Pelvis"
ARCHIVE = "R03 prejoin Head preserved"
NEW = "R03 Head body unified pending"


def _smooth(a, b, x):
    t = max(0., min(1., (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def _guard():
    p = project_path()
    if not bpy.data.filepath:
        raise RuntimeError("Save an independent R03 work copy first")
    file = Path(bpy.data.filepath).resolve()
    if file == (p / "work/character.blend").resolve():
        raise RuntimeError("Never join the formal master directly")
    try:
        file.relative_to((p / "work").resolve())
    except ValueError:
        raise RuntimeError("Current copy must be inside project/work")
    if bpy.context.mode != "OBJECT":
        raise RuntimeError("Object Mode required")
    if bpy.data.objects.get(ARCHIVE) or bpy.data.objects.get(NEW):
        raise RuntimeError("A prior join or staged result exists; preserve and inspect it")
    head, body = (bpy.data.objects.get(n) for n in (HEAD, BODY))
    rig = bpy.data.objects.get("Character_Rig")
    if head is None or body is None or rig is None:
        raise RuntimeError("Expected head, completed trunk and humanoid rig")
    if rig.data.pose_position != "REST":
        raise RuntimeError("Explicitly set Character_Rig to REST before this join")
    if len(head.data.vertices) != 3171 or len(body.data.vertices) != 24001:
        raise RuntimeError("Mesh counts differ from the inspected R03 neck audit")
    if body.data.shape_keys:
        raise RuntimeError("Body has shape keys not covered by the audited join")
    if not head.data.shape_keys or not head.data.shape_keys.use_relative:
        raise RuntimeError("Expected relative head shape keys")
    if head.library or body.library or head.data.library or body.data.library:
        raise RuntimeError("Do not modify linked assets")
    for obj in (head, body):
        if obj.data.users != 1:
            raise RuntimeError("Source mesh data must be single-user: " + obj.name)
        if any(m.type not in {"ARMATURE", "SUBSURF"} for m in obj.modifiers):
            raise RuntimeError("Unreviewed modifier stack on " + obj.name)
        if any(m.type == "ARMATURE" and m.object != rig for m in obj.modifiers):
            raise RuntimeError("Different armature dependencies require manual integration")
        if any(a.domain not in {"POINT", "CORNER"} for a in obj.data.color_attributes):
            raise RuntimeError("Unsupported colour attribute domain")
    anim = head.data.shape_keys.animation_data
    if anim and len(anim.nla_tracks):
        raise RuntimeError("Shape-key NLA tracks require an explicit migration")
    if any(o.parent == head for o in bpy.data.objects):
        raise RuntimeError("Head-object children require an explicit parent migration")
    # Bound geometry modifiers store original vertex correspondence; an ID
    # retarget alone would not be a valid rebinding after a topology change.
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type in {"SURFACE_DEFORM", "MESH_DEFORM", "SHRINKWRAP"}:
                if getattr(mod, "target", None) == head or getattr(mod, "object", None) == head:
                    raise RuntimeError("Dependent deformation must be rebound: " + obj.name)
    return p, file, head, body, rig


def _clip(obj, plane, keep_above, kind):
    keys = obj.data.shape_keys
    source = keys.reference_key.data if keys else obj.data.vertices
    points = [obj.matrix_world @ v.co for v in source]
    vertices, maps, lookup, faces, corner_maps, material_ids, smooth = [], [], {}, [], [], [], []

    def add(token, weights):
        if token not in lookup:
            lookup[token] = len(vertices)
            vertices.append(sum((points[i] * w for i, w in weights), Vector()))
            maps.append((kind, weights))
        return lookup[token]

    def inside(index):
        return points[index].z >= plane if keep_above else points[index].z <= plane

    if min(abs(point.z - plane) for point in points) < 2e-7:
        raise RuntimeError("Cut passes too close to an existing vertex; offset the plane slightly")
    for polygon in obj.data.polygons:
        original = [(obj.data.loops[li].vertex_index, li) for li in polygon.loop_indices]
        output = []
        for previous, current in zip(original[-1:] + original[:-1], original):
            a, la = previous
            b, lb = current
            ina, inb = inside(a), inside(b)
            if ina != inb:
                t = (plane - points[a].z) / (points[b].z - points[a].z)
                token = ("edge", min(a, b), max(a, b))
                index = add(token, [(a, 1 - t), (b, t)])
                output.append((index, [(la, 1 - t), (lb, t)]))
            if inb:
                output.append((add(("vertex", b), [(b, 1.)]), [(lb, 1.)]))
        if not output:
            continue
        if len(output) < 3 or len({x[0] for x in output}) != len(output):
            raise RuntimeError("Degenerate clipped polygon on " + obj.name)
        faces.append(tuple(x[0] for x in output))
        corner_maps.append([(kind, x[1]) for x in output])
        material_ids.append(polygon.material_index)
        smooth.append(polygon.use_smooth)
    return {"points": vertices, "maps": maps, "faces": faces,
            "corners": corner_maps, "materials": material_ids, "smooth": smooth}


def _single_boundary(packet, plane):
    counts = defaultdict(int)
    neighbors = defaultdict(set)
    for face in packet["faces"]:
        for a, b in zip(face, face[1:] + face[:1]):
            counts[tuple(sorted((a, b)))] += 1
    for (a, b), count in counts.items():
        if count > 2:
            raise RuntimeError("Input has a non-manifold edge")
        if count == 1:
            neighbors[a].add(b)
            neighbors[b].add(a)
    if not neighbors or any(len(n) != 2 for n in neighbors.values()):
        raise RuntimeError("Expected a simple closed neck boundary")
    if any(abs(packet["points"][i].z - plane) > 1e-5 for i in neighbors):
        raise RuntimeError("Other boundaries survived the cut, including the old neck rim")
    pending = set(neighbors)
    stack = [pending.pop()]
    while stack:
        for n in neighbors[stack.pop()]:
            if n in pending:
                pending.remove(n)
                stack.append(n)
    if pending:
        raise RuntimeError("Cut created multiple loops; inspect before joining")
    center = sum((packet["points"][i] for i in neighbors), Vector()) / len(neighbors)
    angles = {i: math.atan2(packet["points"][i].y - center.y,
                            packet["points"][i].x - center.x) % math.tau for i in neighbors}
    ordered = sorted(neighbors, key=angles.get)
    if any(b not in neighbors[a] for a, b in zip(ordered, ordered[1:] + ordered[:1])):
        raise RuntimeError("Neck contour is not star-shaped; angular bridging would twist")
    return ordered, [angles[i] for i in ordered]


def _bridge(bottom, upper, bottom_angles, upper_angles):
    """Zipper two outward convex loops, keeping every boundary vertex shared."""
    n, m = len(bottom), len(upper)
    i = j = 0
    result = []
    while i < n or j < m:
        # The first interval includes wraparound at the end. Both sequences
        # start immediately after angle zero, preserving cyclic order.
        next_a = bottom_angles[(i + 1) % n] + (math.tau if i + 1 >= n else 0) if i < n else math.inf
        next_b = upper_angles[(j + 1) % m] + (math.tau if j + 1 >= m else 0) if j < m else math.inf
        a, b = bottom[i % n], upper[j % m]
        if i < n and j < m and abs(next_a - next_b) < .018:
            result.append((a, bottom[(i + 1) % n], upper[(j + 1) % m], b))
            i += 1; j += 1
        elif next_a < next_b:
            result.append((a, bottom[(i + 1) % n], b))
            i += 1
        else:
            result.append((a, upper[(j + 1) % m], b))
            j += 1
    return result


def _copy_rna(source, destination, exclude=()):
    for prop in source.bl_rna.properties:
        name = prop.identifier
        if name == "rna_type" or name in exclude or prop.is_readonly or prop.type in {"COLLECTION", "POINTER"}:
            continue
        try:
            setattr(destination, name, getattr(source, name))
        except (AttributeError, TypeError, ValueError):
            pass


def _copy_drivers(old, new):
    if not old.animation_data:
        return
    new.animation_data_create()
    if old.animation_data.action:
        new.animation_data.action = old.animation_data.action
        if hasattr(old.animation_data, "action_slot"):
            new.animation_data.action_slot = old.animation_data.action_slot
    for fc in old.animation_data.drivers:
        if len(fc.keyframe_points) or len(fc.sampled_points):
            raise RuntimeError("Nonstandard sampled driver curve requires explicit migration")
        try:
            target = new.driver_add(fc.data_path, fc.array_index)
        except TypeError:
            # Scalar key values still report array_index=0 on their FCurve.
            target = new.driver_add(fc.data_path)
        _copy_rna(fc, target, {"data_path", "array_index"})
        _copy_rna(fc.driver, target.driver)
        while target.driver.variables:
            target.driver.variables.remove(target.driver.variables[0])
        for variable in fc.driver.variables:
            copied = target.driver.variables.new()
            copied.name, copied.type = variable.name, variable.type
            for src, dst in zip(variable.targets, copied.targets):
                _copy_rna(src, dst)
                if hasattr(src, "id_type"):
                    dst.id_type = src.id_type
                dst.id = new if src.id == old else src.id
                dst.data_path = src.data_path
        # Generator modifiers can alter driver output; copy their settings.
        for modifier in list(target.modifiers):
            target.modifiers.remove(modifier)
        for modifier in fc.modifiers:
            copied = target.modifiers.new(modifier.type)
            _copy_rna(modifier, copied)


def _driver_ids():
    found = {}
    for attr in ("objects", "shape_keys", "meshes", "curves", "armatures", "materials",
                 "node_groups", "scenes", "worlds", "cameras", "lights"):
        for block in getattr(bpy.data, attr):
            found[block.as_pointer()] = block
            tree = getattr(block, "node_tree", None)
            if tree:
                found[tree.as_pointer()] = tree
    return list(found.values())


def _topology(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bad = sum(not e.is_manifold for e in bm.edges)
    winding = sum(not e.is_contiguous for e in bm.edges if e.is_manifold)
    left = set(bm.verts)
    components = 0
    while left:
        components += 1
        stack = [left.pop()]
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other in left:
                    left.remove(other)
                    stack.append(other)
    bm.free()
    return {"nonmanifold_edges": bad, "components": components,
            "inconsistent_winding_edges": winding}


def run(body_cut=1.272, head_cut=1.282, render_subdivision=1):
    project, file, head, body, rig = _guard()
    if not (1.268 < body_cut < 1.281 and body_cut + .004 < head_cut < 1.298):
        raise ValueError("Choose separated local neck cuts within the audited band")
    if render_subdivision not in (0, 1, 2):
        raise ValueError("Bounded subdivision 0, 1 or 2 only")
    old_key = head.data.shape_keys
    # Validate driver migration before constructing or retargeting anything.
    for fc in old_key.animation_data.drivers if old_key.animation_data else []:
        if len(fc.keyframe_points) or len(fc.sampled_points):
            raise RuntimeError("Nonstandard key driver curve")
    hp = _clip(head, head_cut, True, "head")
    bp = _clip(body, body_cut, False, "body")
    head_loop, head_angles = _single_boundary(hp, head_cut)
    body_loop, body_angles = _single_boundary(bp, body_cut)
    offset = len(hp["points"])
    points = hp["points"] + bp["points"]
    maps = hp["maps"] + bp["maps"]
    faces = hp["faces"] + [tuple(i + offset for i in f) for f in bp["faces"]]
    corners = hp["corners"] + bp["corners"]
    bridge = _bridge([i + offset for i in body_loop], head_loop, body_angles, head_angles)
    faces.extend(bridge)
    corners.extend([None] * len(bridge))
    if len(points) > 32000:
        raise RuntimeError("Unexpected base-mesh growth")
    mesh = bpy.data.meshes.new("R03 unified head trunk topology")
    mesh.from_pydata(points, [], faces)
    mesh.update()
    test = _topology(mesh)
    if test != {"nonmanifold_edges": 0, "components": 1, "inconsistent_winding_edges": 0}:
        bpy.data.meshes.remove(mesh)
        raise RuntimeError("Join is not a single manifold surface: " + repr(test))
    obj = head.copy()
    obj.name = NEW
    obj.data = mesh
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = None
    obj.matrix_world = Matrix.Identity(4)
    obj.modifiers.clear()
    obj.vertex_groups.clear()
    obj.hide_render = True
    obj.hide_set(True)
    # All construction below affects only NEW. On failure it remains available
    # for inspection and neither original source is hidden or renamed.
    source_objects = {"head": head, "body": body}
    for material in head.data.materials:
        mesh.materials.append(material)
    mat_offset = len(mesh.materials)
    for material in body.data.materials:
        mesh.materials.append(material)
    material_ids = hp["materials"] + [i + mat_offset for i in bp["materials"]] + [0] * len(bridge)
    for face, material, shading in zip(mesh.polygons, material_ids,
                                       hp["smooth"] + bp["smooth"] + [True] * len(bridge)):
        face.material_index = material
        face.use_smooth = shading
    # Union all UV layer names. Missing layers inherit that source's active UV
    # so materials using the default UV remain valid on the appended body.
    uv_names = list(dict.fromkeys([layer.name for o in (head, body) for layer in o.data.uv_layers]))
    for name in uv_names:
        layer = mesh.uv_layers.new(name=name)
        for polygon, cm in zip(mesh.polygons, corners):
            for li, vertex_index in zip(polygon.loop_indices, polygon.vertices):
                if cm is None:
                    p = points[vertex_index]
                    layer.data[li].uv = ((math.atan2(p.y - .006, p.x) % math.tau) / math.tau,
                                         (p.z - body_cut) / (head_cut - body_cut))
                else:
                    kind, source_loops = cm[li - polygon.loop_start]
                    source_mesh = source_objects[kind].data
                    uv_source = source_mesh.uv_layers.get(name) or source_mesh.uv_layers.active
                    if uv_source:
                        layer.data[li].uv = sum((uv_source.data[i].uv * w for i, w in source_loops), Vector((0., 0.)))
    if head.data.uv_layers.active:
        mesh.uv_layers.active = mesh.uv_layers[head.data.uv_layers.active.name]
        mesh.uv_layers.active.active_render = True
    colour_names = list(dict.fromkeys([a.name for o in (head, body) for a in o.data.color_attributes]))
    for name in colour_names:
        examples = [o.data.color_attributes.get(name) for o in (head, body)]
        example = next(a for a in examples if a)
        if any(a and (a.domain != example.domain or a.data_type != example.data_type) for a in examples):
            raise RuntimeError("Colour layer domain/type conflict: " + name)
        dest = mesh.color_attributes.new(name=name, type=example.data_type, domain=example.domain)
        if example.domain == "POINT":
            fallback = tuple(sum(example.data[i].color[c] for i in range(len(example.data))) / len(example.data) for c in range(4))
            for output, (kind, weights) in zip(dest.data, maps):
                source = source_objects[kind].data.color_attributes.get(name)
                output.color = tuple(sum(source.data[i].color[c] * w for i, w in weights) for c in range(4)) if source else fallback
        else:
            for poly, cm in zip(mesh.polygons, corners):
                for li in poly.loop_indices:
                    if cm is None:
                        dest.data[li].color = (.48, .29, .205, 1.)
                    else:
                        kind, weights = cm[li - poly.loop_start]
                        source = source_objects[kind].data.color_attributes.get(name)
                        dest.data[li].color = tuple(sum(source.data[i].color[c] * w for i, w in weights) for c in range(4)) if source else (.48, .29, .205, 1.)
    # Preserve all relative-key coordinates by the one immutable index map.
    key_by_name = {}
    for old in old_key.key_blocks:
        new = obj.shape_key_add(name=old.name, from_mix=False)
        for output, (kind, weights) in zip(new.data, maps):
            source = old.data if kind == "head" else body.data.vertices
            transform = source_objects[kind].matrix_world
            output.co = sum(((transform @ source[i].co) * w for i, w in weights), Vector())
        for prop in ("value", "slider_min", "slider_max", "mute", "interpolation", "vertex_group"):
            setattr(new, prop, getattr(old, prop))
        key_by_name[old.name] = new
    for old in old_key.key_blocks:
        key_by_name[old.name].relative_key = key_by_name[old.relative_key.name]
    for name in old_key.keys():
        value = old_key[name]
        obj.data.shape_keys[name] = value.to_dict() if hasattr(value, "to_dict") else value
    # Transfer every vertex group by name; interpolate only newly cut vertices.
    group_names = list(dict.fromkeys([g.name for o in (head, body) for g in o.vertex_groups] + ["upperChest", "neck"]))
    groups = {name: obj.vertex_groups.new(name=name) for name in group_names}
    deform = {bone.name for bone in rig.data.bones if bone.use_deform}
    rewritten = 0
    for idx, (kind, weights) in enumerate(maps):
        source_obj = source_objects[kind]
        values = defaultdict(float)
        for i, factor in weights:
            for membership in source_obj.data.vertices[i].groups:
                values[source_obj.vertex_groups[membership.group].name] += membership.weight * factor
        z = points[idx].z
        # Match upperChest/neck in the junction only. Fade back to retained
        # weights above 1.307; face/jaw groups farther up stay bit-for-bit.
        band = _smooth(body_cut - .012, body_cut, z) * (1 - _smooth(head_cut + .006, head_cut + .025, z))
        band *= (1-_smooth(.065,.085,abs(points[idx].x))) * (1-_smooth(.065,.085,abs(points[idx].y-.006)))
        if band > 0:
            t = _smooth(body_cut - .005, head_cut + .030, z)
            target = {"upperChest": 1 - t, "neck": t}
            for name in deform:
                values[name] = values.get(name, 0.) * (1 - band) + target.get(name, 0.) * band
            rewritten += 1
        total = sum(value for name, value in values.items() if name in deform)
        if total < 1e-7:
            raise RuntimeError("Unweighted appended vertex")
        for name, value in values.items():
            if value > 1e-8:
                groups[name].add([idx], value / total if name in deform else value, "REPLACE")
    sub = obj.modifiers.new("R03 unified preview subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = render_subdivision
    armature = obj.modifiers.new("R03 single humanoid deformation", "ARMATURE")
    armature.object = rig
    # The retained torso and sleeves were authored for linear skinning.
    # Applying dual quaternions to the appended arms caused visible elbow and
    # upper-arm bulges through those sleeves in the actual R04 pose review.
    armature.use_deform_preserve_volume = False
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    obj.matrix_world = Matrix.Identity(4)
    _copy_drivers(old_key, obj.data.shape_keys)
    report = {"source": str(file.relative_to(project)), "body_cut_m": body_cut,
              "head_cut_m": head_cut, "vertices": len(mesh.vertices),
              "faces": len(mesh.polygons), "body_loop": len(body_loop),
              "head_loop": len(head_loop), "bridge_faces": len(bridge),
              "topology": test, "shape_keys": list(key_by_name),
              "junction_weight_vertices": rewritten, "uv_layers": uv_names,
              "colour_layers": colour_names, "drivers_retargeted": [],
              "limitations": ["Static topology passed; visual seam and expressions need review",
                              "Body remains its original triangulated construction outside the neck",
                              "Full source action retained, no new motion certification",
                              "Preview subdivision differs from the old head-only stack"]}
    # Publish only after construction succeeded. External references to the old
    # Key datablock (eyelids, brows, jaw) move to the matching new Key datablock.
    for block in _driver_ids():
        anim = block.animation_data
        if not anim:
            continue
        for curve in anim.drivers:
            for variable in curve.driver.variables:
                for target in variable.targets:
                    if target.id == old_key:
                        target.id = obj.data.shape_keys
                        report["drivers_retargeted"].append([block.name, curve.data_path, variable.name])
                    elif target.id == head:
                        target.id = obj
                        report["drivers_retargeted"].append([block.name, curve.data_path, variable.name])
    head.name = ARCHIVE
    head.hide_render = True
    head.hide_set(True)
    body.hide_render = True
    body.hide_set(True)
    obj.name = HEAD
    obj.hide_render = False
    obj.hide_set(False)
    obj["r03_neck_join_report"] = json.dumps(report)
    bpy.context.view_layer.update()
    path = file.parent / "neck-join-r03-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return report
