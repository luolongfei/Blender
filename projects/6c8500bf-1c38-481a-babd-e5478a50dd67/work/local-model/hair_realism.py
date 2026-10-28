"""Reversible human-hair refinement of the existing bound twin ponytails.

No live operation or save is implicit in apply(). Existing hair topology, UVMap,
bone weights and expression data stay intact. Three small detail meshes inherit
the existing head / ponytail chains. All detail is actual exportable mesh.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import sys
import bpy
import bmesh
from mathutils import Vector

P = Path(__file__).resolve().parents[2]
VERSION = "natural-fine-layered-hair-06"
BACKUP = "HairRealismOriginal"
UV = "HairRealismUV"
MATERIAL = "Character / natural charcoal fibres"
CENTER = Vector((0, .004, 1.450))
NEW_NAMES = ("Hair_Realism_ScalpFibres",
             "Hair_Realism_Ponytail_L", "Hair_Realism_Ponytail_R")


def smooth(a, b, value):
    t = max(0, min(1, (value-a)/(b-a)))
    return t*t*(3-2*t)


def targets(collection):
    return [obj for obj in collection.all_objects
            if obj.type == "MESH" and obj.name.startswith("Hair_")
            and obj.name not in NEW_NAMES and not obj.name.startswith("Hair_Tie_")]


def params(obj):
    layer = obj.data.uv_layers["UVMap"]
    result = {}
    for loop in obj.data.loops:
        result.setdefault(loop.vertex_index, tuple(layer.data[loop.index].uv))
    return result


def rings(parameters):
    rows = {}
    for index, (u, t) in parameters.items():
        rows.setdefault(round(t, 6), []).append(index)
    return [(t, sorted(indices, key=lambda i: parameters[i][0]))
            for t, indices in sorted(rows.items())]


def checkpoint(obj):
    attr = obj.data.attributes.get(BACKUP)
    if attr is None:
        attr = obj.data.attributes.new(BACKUP, "FLOAT_VECTOR", "POINT")
        for vertex in obj.data.vertices:
            attr.data[vertex.index].vector = vertex.co
        obj["hair_realism_materials"] = json.dumps(
            [material.name if material else None for material in obj.data.materials])
        obj["hair_realism_render_uv"] = next(
            (layer.name for layer in obj.data.uv_layers if layer.active_render), "UVMap")
    if len(attr.data) != len(obj.data.vertices):
        raise RuntimeError("Hair topology changed after realism checkpoint: "+obj.name)
    for name in json.loads(obj["hair_realism_materials"]):
        if name:
            bpy.data.materials[name].use_fake_user = True
    return [point.vector.copy() for point in attr.data]


def envelope(co):
    delta = co-CENTER
    ry = .116 if delta.y >= 0 else .108
    radius = math.sqrt((delta.x/.121)**2+(delta.y/ry)**2+(delta.z/.145)**2)
    point = CENTER+delta/max(radius, .001)
    normal = Vector((point.x/.121**2, (point.y-.004)/ry**2,
                     (point.z-1.450)/.145**2)).normalized()
    return point, normal


def cap_point(theta, phi):
    ry = .116 if math.cos(phi) < 0 else .108
    return Vector((.121*math.sin(theta)*math.sin(phi),
                   .004-ry*math.sin(theta)*math.cos(phi),
                   1.450+.145*math.cos(theta)))


def gather_lower_cap(point, phi, t):
    direction = 1 if phi >= 0 else -1
    proximity = math.exp(-((abs(phi)-2.04)/.55)**2)
    amount = .90*proximity*t**5
    tie = Vector((direction*.105, .046, 1.384))
    return point.lerp(tie, amount)


def coordinates(obj, original, parameters):
    coords = [point.copy() for point in original]
    name = obj.name
    if name == "Hair_ScalpFoundation":
        # A continuous shell replaces intersecting upper plates. Fine strands
        # supply directional surface structure without a segmented scalp.
        for index, point in enumerate(coords):
            surface, normal = envelope(point)
            delta = surface-CENTER
            phi = math.atan2(delta.x/.121, -delta.y/(
                .116 if delta.y >= 0 else .108))
            theta = math.acos(max(-1, min(1, delta.z/.145)))
            t = parameters[index][1]
            # The supporting shell's concealed front rim used to cut across
            # the exposed fringe. Retract just that rim behind rooted strands.
            theta -= .20*max(0, math.cos(phi))**4*t**4
            # The rear rim was a low, rigid diagonal bob edge. A higher nape
            # follows the skull and leaves room for gathered locks to travel
            # toward the ties, rather than ending as a hanging cap.
            theta -= .24*max(0, -math.cos(phi))**1.2*t**5
            coords[index] = gather_lower_cap(cap_point(theta, phi), phi, t)
        return coords
    if name.startswith(("Hair_Crown_", "Hair_CrownFlyaway_")):
        return [CENTER+(point-CENTER)*.68 for point in coords]
    if name.startswith(("Hair_BackGather_", "Hair_NapeWisp_")):
        # The backing cap itself gathers continuously to the ties. Retaining
        # old detached lower leaves would expose gaps after the nape is raised.
        return [CENTER+(point-CENTER)*.68 for point in coords]
    if name.startswith("Hair_Fringe_") and "_Flow" not in name:
        number = int(name.split("_")[2])
        for t, indices in rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            for index in indices:
                point = coords[index]
                u, _ = parameters[index]
                # Airier lower fringe and more brow clearance for human eyes.
                point.x = center.x+(point.x-center.x)*(
                    1-.35*smooth(.46, .94, t))
                point.z += (.006+.003*(number % 3)/2)*smooth(.53, 1, t)
                point.x += .0013*math.sin(number*1.8)*smooth(.72, 1, t)
                q = max(.001, 1-(point.x/.121)**2-((point.z-1.450)/.145)**2)
                surface_y = .004-.108*math.sqrt(q)
                # Absolute height makes neighbouring roots share one smooth
                # surface. Only the lower free fringe emerges from the shell.
                emergence = 1-smooth(1.518, 1.568, point.z)
                offset = -.0014+.0040*emergence
                offset += .00028*number*emergence
                # The closed strip is thin, with no inflated root cap.
                point.y = surface_y-offset-.000085*math.sin(math.tau*u)
        return coords
    if "_Flow" in name:
        # Old thick accent strands are retained but moved beneath their locks.
        for t, indices in rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            for index in indices:
                coords[index] = center+(coords[index]-center)*.12
                if coords[index].z > 1.44:
                    coords[index] = CENTER+(coords[index]-CENTER)*.68
                if obj.get("hair_region") == "ponytail":
                    coords[index].y -= .010*(1-smooth(1.265, 1.335, coords[index].z))
        return coords
    if name.startswith(("Hair_Ponytail_", "Hair_TailCore_", "Hair_TailEdge_")):
        seed = sum(ord(char) for char in name)
        for t, indices in rings(parameters):
            center = sum((coords[i] for i in indices), Vector())/len(indices)
            width_factor = 1-.07*smooth(.30, .95, t)
            for index in indices:
                radial = coords[index]-center
                # Thin the broad leaves; retain the volume-bearing foundation.
                if "Ponytail_" in name:
                    radial *= width_factor
                    radial.y *= .77
                elif "TailCore_" in name:
                    radial *= .97
                    radial.y *= .86
                else:
                    radial *= .86
                coords[index] = center+radial
                tip = smooth(.75, 1, t)
                coords[index].z += .0028*math.sin(seed)*tip
                coords[index].x += .0016*math.cos(seed*.7)*tip
                # Extra shoulder clearance, independent of the separately
                # validated collider refinement. Root/tie positions stay put.
                coords[index].y -= .010*(1-smooth(1.265, 1.335, coords[index].z))
        return coords
    return coords


def material():
    existing = bpy.data.materials.get(MATERIAL)
    if existing and existing.get("hair_realism_version") == VERSION:
        return existing
    source = bpy.data.materials.get("Character / coherent layered hair")
    if source is None:
        raise RuntimeError("Apply the coherent layered hair checkpoint first.")
    if existing:
        existing.name += " previous"
    result = source.copy()
    result.name = MATERIAL
    nodes = result.node_tree.nodes
    shader = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value = .62
    shader.inputs["Specular IOR Level"].default_value = .14
    anisotropy = shader.inputs.get("Anisotropic IOR Level")
    if anisotropy:
        anisotropy.default_value = .28
    for node in nodes:
        if node.type in {"UVMAP", "NORMAL_MAP"}:
            node.uv_map = UV
        if node.type == "NORMAL_MAP":
            node.inputs["Strength"].default_value = .24
    result["hair_realism_version"] = VERSION
    return result


def apply_uv(obj, parameters):
    mesh = obj.data
    layer = mesh.uv_layers.get(UV) or mesh.uv_layers.new(name=UV)
    layer.active_render = True
    is_head = obj.get("hair_region") != "ponytail"
    for polygon in mesh.polygons:
        values = []
        for li in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[li].vertex_index]
            if is_head:
                delta = vertex.co-CENTER
                phi = math.atan2(delta.x, -delta.y)
                theta = math.acos(max(-1, min(1, delta.z/.145)))
                values.append(((.5+phi/math.tau) % 1, theta/2.2))
            else:
                u, t = parameters[vertex.index]
                values.append((u, t))
        # Avoid a UV wrap interpolation across the centre-back polygon.
        if is_head and max(u for u, v in values)-min(u for u, v in values) > .5:
            values = [(u+1 if u < .5 else u, v) for u, v in values]
        for li, value in zip(polygon.loop_indices, values):
            layer.data[li].uv = value


class Fibres:
    def __init__(self):
        self.vertices, self.faces, self.uvs = [], [], []

    def add(self, points, radius, phase=0):
        if len(points) < 4:
            return
        offset = len(self.vertices)
        sides = 3
        previous = None
        for i, point in enumerate(points):
            t = i/(len(points)-1)
            tangent = (points[min(i+1, len(points)-1)]-
                       points[max(0, i-1)]).normalized()
            normal = Vector((0, -1, .1))
            normal -= tangent*normal.dot(tangent)
            if normal.length < .01:
                normal = Vector((1, 0, 0))
                normal -= tangent*normal.dot(tangent)
            normal.normalize()
            width = tangent.cross(normal).normalized()
            if previous and width.dot(previous) < 0:
                width, normal = -width, -normal
            previous = width
            fade = max(.025, math.sin(math.pi*t)**.6)
            rad = radius*fade
            for j in range(sides):
                angle = math.tau*j/sides
                self.vertices.append(tuple(point+rad*(
                    width*math.cos(angle)+normal*math.sin(angle))))
                self.uvs.append(((j/sides+phase) % 1, t))
        for i in range(len(points)-1):
            for j in range(sides):
                a = offset+i*sides+j
                b = offset+i*sides+(j+1) % sides
                self.faces.append((a, b, b+sides, a+sides))
        self.faces.append(tuple(offset+j for j in reversed(range(sides))))
        last = offset+(len(points)-1)*sides
        self.faces.append(tuple(last+j for j in range(sides)))


def scalp_fibres(objects):
    randomizer = random.Random(271028)
    result = Fibres()
    for i in range(150):
        phi0 = -math.pi+math.tau*(i+randomizer.uniform(-.28, .55))/150
        front = max(0, math.cos(phi0))
        end = 1.245+.87*((1-math.cos(phi0))/2)**.64-.40*front**4
        start = randomizer.uniform(.06, .27)
        points = []
        for j in range(24):
            t = j/23
            theta = start+(end-start)*t
            phi = phi0+.07*math.sin(math.pi*t)*math.sin(phi0+1.1)
            if front < .2:
                target = math.copysign(1.95, phi0)
                phi += (target-phi0)*.20*smooth(.40, 1, t)
            theta -= .24*max(0, -math.cos(phi))**1.2*t**5
            point = cap_point(theta, phi)
            point = gather_lower_cap(point, phi, t)
            _, normal = envelope(point)
            lift = .00010+.00016*math.sin(math.pi*t)
            if i % 17 == 0:
                lift += .0014*math.sin(math.pi*t)**2
            point += normal*lift
            points.append(point)
        result.add(points, randomizer.uniform(.000065, .00013), randomizer.random())
    # Thin terminal subdivisions follow the actual fringe/temple surfaces.
    for obj in objects:
        if not obj.name.startswith(
                ("Hair_Fringe_", "Hair_Temple_")) or "_Flow" in obj.name:
            continue
        add_surface_fibres(result, obj, 5, randomizer, head=True)
    return result


def add_surface_fibres(result, obj, count, randomizer, head=False):
    parameters = params(obj)
    rows = rings(parameters)
    if len({len(indices) for _, indices in rows}) != 1:
        return
    size = len(rows[0][1])
    if size < 8:
        return
    coords = [vertex.co.copy() for vertex in obj.data.vertices]
    for strand in range(count):
        # Traverse the whole closed ribbon; hidden back fibres are still useful
        # when the ponytail turns in its spring chain.
        angle = (strand+.27+randomizer.uniform(-.12, .12))/count
        start = 0.16+randomizer.uniform(0, .13) if head else .035+randomizer.uniform(0, .09)
        stop = .96-randomizer.uniform(0, .08)
        points = []
        for t, indices in rows[::2]:
            if t < start or t > stop:
                continue
            floating = angle*size+.10*math.sin(t*math.tau+strand)
            left = math.floor(floating) % size
            frac = floating-math.floor(floating)
            point = coords[indices[left]].lerp(coords[indices[(left+1) % size]], frac)
            center = sum((coords[index] for index in indices), Vector())/size
            outward = (point-center).normalized()
            lift = .00012
            if strand % 5 == 2 and not head:
                lift += .0008*math.sin(math.pi*(t-start)/(stop-start))**2
            points.append(point+outward*lift)
        if len(points) > 3:
            result.add(points, randomizer.uniform(.000075, .00014),
                       randomizer.random())


def tail_fibres(objects, side):
    randomizer = random.Random(713 if side == "L" else 971)
    result = Fibres()
    for obj in objects:
        if obj.get("hair_region") != "ponytail" or obj.get("side") != side:
            continue
        if "_Flow" in obj.name:
            continue
        number = 28 if "TailCore" in obj.name else 13 if "Ponytail_" in obj.name else 6
        add_surface_fibres(result, obj, number, randomizer)
    return result


def weights(co, side):
    if side is None:
        return {"head": 1.0}
    z = co[2]
    if z > 1.355:
        amount = 1-smooth(1.355, 1.382, z)
        return {"head": 1-amount, "hair_"+side+"_1": amount}
    if z > 1.29:
        amount = smooth(1.29, 1.34, z)
        return {"hair_"+side+"_2": 1-amount, "hair_"+side+"_1": amount}
    amount = smooth(1.23, 1.29, z)
    return {"hair_"+side+"_3": 1-amount, "hair_"+side+"_2": amount}


def create_mesh(name, data, collection, rig, mat, side=None):
    mesh = bpy.data.meshes.new(name+" mesh")
    mesh.from_pydata(data.vertices, [], data.faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    for layer_name in ("UVMap", UV):
        layer = mesh.uv_layers.new(name=layer_name)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
            for li in polygon.loop_indices:
                layer.data[li].uv = data.uvs[mesh.loops[li].vertex_index]
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(mat)
    obj.parent = rig
    obj["part"] = "hair"
    obj["hair_region"] = "ponytail" if side else "scalp"
    obj["rig_attachment"] = "hair_"+side if side else "head"
    obj["hair_realism_generated"] = VERSION
    if side:
        obj["side"] = side
    groups = {}
    for vertex in mesh.vertices:
        vertex_weights = {name: value for name, value in weights(vertex.co, side).items()
                          if value > 1e-8}
        total = sum(vertex_weights.values())
        for bone, value in vertex_weights.items():
            if bone not in rig.data.bones:
                raise RuntimeError("Missing existing spring bone: "+bone)
            group = groups.get(bone)
            if group is None:
                group = groups[bone] = obj.vertex_groups.new(name=bone)
            group.add([vertex.index], value/total, "REPLACE")
    modifier = obj.modifiers.new("Existing hair chain deformation", "ARMATURE")
    modifier.object = rig
    modifier.use_deform_preserve_volume = True
    return obj


def remove_generated():
    for name in NEW_NAMES:
        obj = bpy.data.objects.get(name)
        if obj:
            if not obj.get("hair_realism_generated"):
                raise RuntimeError("Unrelated object occupies generated hair name: "+name)
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if not mesh.users:
                bpy.data.meshes.remove(mesh)


def apply(collection=None, rig=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    rig = rig or bpy.data.objects["Character_Rig"]
    objects = targets(collection)
    if not any(obj.name == "Hair_ScalpFoundation" for obj in objects):
        raise RuntimeError("Original scalp is missing.")
    for obj in objects:
        if obj.data.shape_keys or not obj.vertex_groups:
            raise RuntimeError("Unexpected hair topology/rig state: "+obj.name)
    mat = material()
    remove_generated()
    for obj in objects:
        original = checkpoint(obj)
        parameters = params(obj)
        for vertex, point in zip(obj.data.vertices, coordinates(obj, original, parameters)):
            vertex.co = point
        apply_uv(obj, parameters)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.data.update()
        obj["hair_realism_version"] = VERSION
    created = [create_mesh(NEW_NAMES[0], scalp_fibres(objects), collection, rig, mat)]
    for side, name in zip(("L", "R"), NEW_NAMES[1:]):
        created.append(create_mesh(name, tail_fibres(objects, side), collection, rig, mat, side))
    return {
        "version": VERSION, "changedMeshes": [obj.name for obj in objects],
        "newMeshes": [obj.name for obj in created],
        "newVertices": sum(len(obj.data.vertices) for obj in created),
        "newTriangles": sum(sum(len(poly.vertices)-2 for poly in obj.data.polygons)
                            for obj in created),
        "lowerTailClearanceShiftMeters": .010,
        "springTargetAdditions": [
            {"mesh": "Hair_Realism_Ponytail_"+side, "source": "hair", "mode": "spring",
             "chains": ["ponytail_"+side]} for side in ("L", "R")],
        "fixedTargetAddition": {"mesh": NEW_NAMES[0], "source": "hair", "mode": "fixed",
                                "reason": "Short scalp fibres follow the head."},
    }


def restore(collection=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    remove_generated()
    for obj in targets(collection):
        attr = obj.data.attributes.get(BACKUP)
        if attr is None:
            continue
        if len(attr.data) != len(obj.data.vertices):
            raise RuntimeError("Cannot restore changed hair topology: "+obj.name)
        for vertex in obj.data.vertices:
            vertex.co = attr.data[vertex.index].vector
        obj.data.materials.clear()
        for name in json.loads(obj["hair_realism_materials"]):
            if name:
                obj.data.materials.append(bpy.data.materials[name])
        layer = obj.data.uv_layers.get(UV)
        if layer:
            obj.data.uv_layers.remove(layer)
        original_uv = obj.data.uv_layers.get(obj.get("hair_realism_render_uv", "UVMap"))
        if original_uv:
            original_uv.active_render = True
        obj.data.update()
        if "hair_realism_version" in obj:
            del obj["hair_realism_version"]


def prepare_vrm_uv(meshes):
    """Flatten the authored secondary UV into UVMap in an export copy only.

    The project's MToon bridge copies image/scale/offset but not the named UV
    source. Flattening removes that ambiguity without changing editable UVMap.
    """
    if not bpy.app.background:
        raise RuntimeError("UV flattening is restricted to a disposable export process.")
    changed = []
    for obj in meshes:
        if obj.type != "MESH" or obj.get("part") != "hair":
            continue
        source = obj.data.uv_layers.get(UV)
        if source is None:
            continue
        target = obj.data.uv_layers.get("UVMap")
        if target is None:
            raise RuntimeError("Export hair is missing its primary UVMap: "+obj.name)
        for destination, point in zip(target.data, source.data):
            destination.uv = point.uv
        target.active_render = True
        changed.append(obj.name)
    return changed


def qa():
    if not bpy.app.background:
        raise RuntimeError("QA must run in an independent background copy.")
    out = P/"work/qa-hair-realism"
    out.mkdir(parents=True, exist_ok=True)
    source = Path(bpy.data.filepath)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 20
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 5
    scene.render.resolution_x, scene.render.resolution_y = 640, 744
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    camera = bpy.data.objects.new("Natural hair review camera",
                                  bpy.data.cameras.new("Natural hair review camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = .41
    target = Vector((0, .002, 1.405))
    views = {"front": (0, -3, 1.405), "front45": (2.5, -2.5, 1.405),
             "profile": (3, .002, 1.405)}
    if "--profile-only" in sys.argv:
        views = {"profile": views["profile"]}
    camera.location = (0, -3, 1.405)
    camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
    collection = bpy.data.collections["CHARACTER · Original geometry"]
    original_objects = targets(collection)

    def invariant():
        return hashlib.sha256(repr([
            (obj.name, [tuple(poly.vertices) for poly in obj.data.polygons],
             [tuple(item.uv) for item in obj.data.uv_layers["UVMap"].data],
             [group.name for group in obj.vertex_groups],
             [[(group.group, group.weight) for group in vertex.groups]
              for vertex in obj.data.vertices])
            for obj in original_objects]).encode()).hexdigest()

    def nonhair():
        return hashlib.sha256(repr([
            (obj.name, [tuple(vertex.co) for vertex in obj.data.vertices],
             [(key.name, key.value, [tuple(point.co) for point in key.data])
              for key in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else [])
            for obj in collection.all_objects if obj.type == "MESH"
            and obj.get("part") != "hair"]).encode()).hexdigest()

    def old_state():
        return [
            (obj.name, [tuple(vertex.co) for vertex in obj.data.vertices],
             [mat.name if mat else None for mat in obj.data.materials],
             [layer.name for layer in obj.data.uv_layers])
            for obj in original_objects]

    def full_coordinates():
        return hashlib.sha256(repr([
            (obj.name, [tuple(vertex.co) for vertex in obj.data.vertices])
            for obj in sorted(collection.all_objects, key=lambda item: item.name)
            if obj.type == "MESH" and obj.get("part") == "hair"]).encode()).hexdigest()

    invariant_before, nonhair_before, before = invariant(), nonhair(), old_state()
    report = apply()
    first = full_coordinates()
    apply()
    assert first == full_coordinates(), "Hair realism must be idempotent"
    assert invariant() == invariant_before
    assert nonhair() == nonhair_before
    maximum_error = max(
        abs(sum(group.weight for group in vertex.groups)-1)
        for name in NEW_NAMES for vertex in bpy.data.objects[name].data.vertices)
    assert maximum_error < 1e-6, "New hair weights must be normalized"
    restore()
    assert before == old_state(), "Restore must preserve original geometry/materials/UVs"
    apply()
    report["validation"] = {
        "idempotent": True, "restoreExact": True,
        "originalTopologyUvAndWeightsUnchanged": True,
        "allNonhairGeometryAndExpressionsUnchanged": True,
        "newWeightMaximumNormalizationError": maximum_error,
    }
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"hair-realism-review.blend"))
    states = ("after",) if "--after-only" in sys.argv else ("before", "after")
    for state in states:
        restore() if state == "before" else apply()
        for name, position in views.items():
            camera.location = position
            camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(out/f"{state}-{name}.png")
            bpy.ops.render.render(write_still=True)
            print("HAIR_REALISM_RENDER", state, name, flush=True)
    if "--profile-only" in sys.argv:
        material = bpy.data.materials.new("Natural nape review clay")
        material.use_nodes = True
        shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        shader.inputs["Base Color"].default_value = (.24, .21, .18, 1)
        shader.inputs["Roughness"].default_value = .72
        shader.inputs["Specular IOR Level"].default_value = .12
        scene.view_layers[0].material_override = material
        scene.render.filepath = str(out/"after-profile-clay.png")
        bpy.ops.render.render(write_still=True)
        scene.view_layers[0].material_override = None
    report["sourceSha256"] = source_sha
    report["sourceUnchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    report["renderedViewsThisVersion"] = list(views)
    report["clayViewsThisVersion"] = ["profile"] if "--profile-only" in sys.argv else []
    (out/"report.json").write_text(json.dumps(report, indent=2))
    print("HAIR_REALISM_COMPLETE", json.dumps(report), flush=True)


def clay_review():
    out = P/"work/qa-hair-realism"
    candidate = out/"hair-realism-review.blend"
    if not bpy.app.background or Path(bpy.data.filepath) != candidate:
        raise RuntimeError("Clay inspection only operates on the isolated review copy.")
    scene = bpy.context.scene
    material = bpy.data.materials.new("Natural hair review clay")
    material.use_nodes = True
    shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (.24, .21, .18, 1)
    shader.inputs["Roughness"].default_value = .72
    shader.inputs["Specular IOR Level"].default_value = .12
    scene.view_layers[0].material_override = material
    target = Vector((0, .002, 1.405))
    for name, position in (("front45", (2.5, -2.5, 1.405)),
                           ("profile", (3, .002, 1.405))):
        scene.camera.location = position
        scene.camera.rotation_euler = (
            target-scene.camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(out/f"after-{name}-clay.png")
        bpy.ops.render.render(write_still=True)
    scene.view_layers[0].material_override = None
    scene.camera.location = (0, -3, 1.405)
    scene.camera.rotation_euler = (
        target-scene.camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str(candidate))


if __name__ == "__main__":
    if "--qa-hair-realism" in sys.argv:
        qa()
    elif "--clay-hair-realism" in sys.argv:
        clay_review()
