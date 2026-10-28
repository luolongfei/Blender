"""A reversible finishing pass on the already reshaped head and hair.

This module does not replay hair_realism or head_proportion_refine. The geometry
checkpoint is the loaded, integrated hairstyle. Existing bones, face, expressions
and ties are never changed. New strands are opaque, weighted polygon geometry.
The optional background review writes only to work/qa-hair-refinement.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import random
import struct
import sys

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import closest_point_on_tri


HERE = Path(__file__).resolve().parent
P = HERE.parents[1]
VERSION = "fine-gathered-mobile-hair-07"
BACKUP = "HairRefinementOriginal"
BIND = "HairRefinementBindPosition"
MAT = "Character / soft gathered charcoal"
GENERATED = (
    "Hair_Refinement_ScalpLayers", "Hair_Refinement_Ponytail_L",
    "Hair_Refinement_Ponytail_R",
)


def helper():
    spec = importlib.util.spec_from_file_location("hair_base_helpers", HERE/"hair_realism.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


H = helper()
_TAIL_SOURCE_CENTERS = {}


def smooth(a, b, value):
    t = max(0.0, min(1.0, (value-a)/(b-a)))
    return t*t*(3.0-2.0*t)


def objects(collection):
    return [o for o in collection.all_objects if o.type == "MESH"
            and o.name.startswith("Hair_") and not o.name.startswith("Hair_Tie_")
            and o.name not in GENERATED]


def coordinate_hash(points):
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("<3f", *point))
    return digest.hexdigest()


def checkpoint(obj):
    attr = obj.data.attributes.get(BACKUP)
    if attr is None:
        attr = obj.data.attributes.new(BACKUP, "FLOAT_VECTOR", "POINT")
        for vertex in obj.data.vertices:
            attr.data[vertex.index].vector = vertex.co
        obj["hair_refinement_materials"] = json.dumps(
            [m.name if m else None for m in obj.data.materials])
        obj["hair_refinement_prior_version"] = obj.get("hair_refinement_version", "")
    if len(attr.data) != len(obj.data.vertices):
        raise RuntimeError("Hair topology changed after refinement checkpoint: "+obj.name)
    previous = obj.get("hair_refinement_output_sha")
    current = coordinate_hash(v.co for v in obj.data.vertices)
    original = coordinate_hash(point.vector for point in attr.data)
    if previous and current not in {previous, original}:
        raise RuntimeError("Hair was reshaped after this checkpoint; do not replay "
                           "the earlier patch over a later head warp: "+obj.name)
    for name in json.loads(obj["hair_refinement_materials"]):
        if name and name in bpy.data.materials:
            bpy.data.materials[name].use_fake_user = True
    if obj.get("hair_region") == "ponytail" and BIND not in obj.data.attributes:
        bind = obj.data.attributes.new(BIND, "FLOAT_VECTOR", "POINT")
        for index, point in enumerate(attr.data):
            bind.data[index].vector = point.vector
    return [point.vector.copy() for point in attr.data]


def make_bvh(obj):
    # All source meshes use the rig-local coordinate frame.
    return BVHTree.FromPolygons([tuple(v.co) for v in obj.data.vertices],
                               [tuple(p.vertices) for p in obj.data.polygons],
                               all_triangles=False)


def radial_normal(co):
    value = Vector((co.x/.112**2, (co.y-.003)/.098**2,
                    (co.z-1.442)/.136**2))
    return value.normalized() if value.length else Vector((0, 0, 1))


def soften_crown(point):
    point = point.copy()
    top = smooth(1.541, 1.576, point.z)
    point.z -= .0020*top
    return point


def cap_coordinates(obj, original, head_bvh):
    parameters = H.params(obj)
    result = []
    for index, source in enumerate(original):
        co = soften_crown(source)
        u, t = parameters[index]
        phi = math.atan2(source.x, -(source.y-.004))
        back = smooth(1.72, 2.72, abs(phi))
        amount = back*t**4
        # The previous posterior warp left the low cap flared out. Follow the
        # real cranium down to the nape and bury the terminal supporting rim.
        front = max(0, math.cos(phi))**4
        front_amount = front*t**4
        z = co.z-.0440*amount-.0270*front_amount
        direction = Vector((math.sin(phi), -math.cos(phi), 0))
        hit, normal, _, _ = head_bvh.ray_cast(
            Vector((0, .003, z)), direction, .25)
        if hit is not None:
            target = hit+direction*.0030
            target.z = z
            co = co.lerp(target, max(amount, front_amount))
            co.z = z
        else:
            co.z = z
            co.y -= .0370*amount
        # Fine geometry will form the visible hairline, not an exposed slab.
        co -= radial_normal(co)*(.0038*back*smooth(.935, 1, t))
        # Continuous backing below front roots prevents pale triangular holes.
        co.z += .0012*math.sin(phi*17+.4)*back*smooth(.86, 1, t)
        co.y -= .0016*front*smooth(1.505, 1.550, co.z)
        result.append(co)
    return result


def tail_shift(co, side):
    """Replace the old chest-crossing sweep with a posterior hanging line.

    This is a displacement field of the original coordinates, shared by the
    mesh and the spring rest bones. The endpoint at the tie is held exactly.
    Existing surface offsets, individual locks and fibres remain geometry.
    """
    if side not in _TAIL_SOURCE_CENTERS:
        core = bpy.data.objects["Hair_TailCore_"+side]
        attr = core.data.attributes.get(BACKUP)
        points = ([d.vector.copy() for d in attr.data] if attr else
                  [v.co.copy() for v in core.data.vertices])
        source = [c for _, c in ring_centers(core, points)]
        top = source[0]
        sign = 1 if side == "L" else -1
        target = []
        for c in source:
            q = max(0, min(1, (top.z-c.z)/(top.z-1.190)))
            tx = sign*(abs(top.x)+.035*smooth(.02, .43, q)
                       -.006*smooth(.65, 1, q))
            ty = top.y+.063*smooth(.02, .50, q)-.010*smooth(.68, 1, q)
            emergence = smooth(0, .10, q)
            target.append(Vector((
                c.x+(tx-c.x)*emergence, c.y+(ty-c.y)*emergence,
                c.z-.040*smooth(.22, 1, q))))
        def tangents(path):
            return [(path[min(i+1, len(path)-1)]-path[max(0, i-1)]).normalized()
                    for i in range(len(path))]
        _TAIL_SOURCE_CENTERS[side] = (
            source, target, tangents(source), tangents(target))
    source, target, source_tangents, target_tangents = _TAIL_SOURCE_CENTERS[side]
    if co.z >= source[0].z+.0001:
        return co.copy()
    closest = None
    for i, (a, b) in enumerate(zip(source, source[1:])):
        direction = b-a
        t = max(0, min(1, (co-a).dot(direction)/direction.length_squared))
        center = a.lerp(b, t)
        distance = (co-center).length_squared
        if closest is None or distance < closest[0]:
            closest = (distance, i, t, center)
    _, i, t, center = closest
    old_tangent = source_tangents[i].lerp(source_tangents[i+1], t).normalized()
    new_tangent = target_tangents[i].lerp(target_tangents[i+1], t).normalized()
    rotation = old_tangent.rotation_difference(new_tangent)
    # Transport the section instead of shearing its Y coordinates as a
    # function of Z; a diagonal cross-section must not inflate into a pouch.
    return target[i].lerp(target[i+1], t)+rotation @ (co-center)


def ring_centers(obj, coords):
    result = []
    for t, indices in H.rings(H.params(obj)):
        result.append((t, sum((coords[i] for i in indices), Vector())/len(indices)))
    return result


def center_at_z(centers, z):
    for (_, a), (_, b) in zip(centers, centers[1:]):
        if a.z >= z >= b.z:
            return a.lerp(b, (a.z-z)/max(1e-8, a.z-b.z))
    return min((c for _, c in centers), key=lambda c: abs(c.z-z)).copy()


def garment_front_trees(collection, rig):
    """Exact evaluated surfaces in T rest and the saved relaxed pose."""
    garments = [o for o in collection.all_objects if o.type == "MESH" and
                o.name.startswith(("Blouse_Tucked_Bodice", "Blouse_Puff_Sleeve_"))]
    if len(garments) != 3:
        raise RuntimeError("Expected one bodice and the two approved puff sleeves.")
    position = rig.data.pose_position
    result = []
    try:
        for pose in ("REST", "POSE"):
            rig.data.pose_position = pose
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            vertices, polygons = [], []
            for obj in garments:
                evaluated = obj.evaluated_get(dg)
                mesh = evaluated.to_mesh()
                transform = rig.matrix_world.inverted() @ evaluated.matrix_world
                offset = len(vertices)
                vertices.extend(tuple(transform @ v.co) for v in mesh.vertices)
                polygons.extend(tuple(offset+i for i in p.vertices) for p in mesh.polygons)
                evaluated.to_mesh_clear()
            result.append(BVHTree.FromPolygons(vertices, polygons, all_triangles=False))
    finally:
        rig.data.pose_position = position
        bpy.context.view_layer.update()
    return result


def measured_clearance(current, modified, body_trees):
    """Move whole neighbouring sections together, retaining strand thickness.

    A two-pose garment posterior envelope supplies the required offset. The field
    is smoothed along the ponytail, so the result is a curve with air clearance
    instead of individually flattened vertices conforming to a sleeve.
    """
    low, high, step = 1.110, 1.370, .002
    count = int(round((high-low)/step))+1
    report = {}
    for side in ("L", "R"):
        group = [o for o in current if o.get("hair_region") == "ponytail"
                 and o.get("side") == side]
        required = [0.0]*count
        for obj in group:
            for co in modified[obj.name]:
                if not low <= co.z <= 1.338:
                    continue
                rear = []
                for tree in body_trees:
                    hit, _, _, _ = tree.ray_cast(
                        Vector((co.x, .45, co.z)), Vector((0, -1, 0)), .9)
                    if hit is not None:
                        rear.append(hit.y)
                if not rear:
                    continue
                distance = max(0, max(rear)+.0060-co.y)
                index = max(0, min(count-1, round((co.z-low)/step)))
                required[index] = max(required[index], distance)
        # A Lipschitz upper envelope bridges sparse samples. Small convolution
        # removes angular kinks; its 2 mm allowance retains the target margin.
        envelope = [max(0, max(value-.38*abs(i-j)*step
                               for j, value in enumerate(required)))
                    for i in range(count)]
        field = []
        for i in range(count):
            value = sum(envelope[max(0, min(count-1, i+d))]*weight
                        for d, weight in ((-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1)))/9
            value += .0020*smooth(0, .001, value)
            field.append(value)
        for obj in group:
            for co in modified[obj.name]:
                if not low <= co.z <= high:
                    continue
                f = (co.z-low)/step
                i = max(0, min(count-2, int(f)))
                delta = field[i]*(1-(f-i))+field[i+1]*(f-i)
                co.y += delta
        report[side] = {
            "additionalMaximumPosteriorShiftMeters": max(field),
            "sampledTargetClearanceMeters": .006,
            "affectedVerticalRange": [low, high],
            "fieldLowZ": low, "fieldStep": step, "fieldPosteriorOffsets": field,
        }
    return report


def motion_center(co, side):
    """Map an original rest-chain node through the exact ponytail rest field.

    Intended for hair_chain_refine, which keeps a separate original arc map for
    the HairRefinementBindPosition attributes. Humanoid bones are not involved.
    """
    point = tail_shift(Vector(co), side)
    core = bpy.data.objects["Hair_TailCore_"+side]
    if "hair_refinement_motion_field" not in core:
        raise RuntimeError("Apply the hair geometry patch before aligning its hair chain.")
    field = json.loads(core["hair_refinement_motion_field"])
    low, step = field["fieldLowZ"], field["fieldStep"]
    offsets = field["fieldPosteriorOffsets"]
    f = (point.z-low)/step
    if 0 <= f <= len(offsets)-1:
        i = min(len(offsets)-2, int(f))
        point.y += offsets[i]*(1-(f-i))+offsets[i+1]*(f-i)
    return point


def material():
    existing = bpy.data.materials.get(MAT)
    if existing and existing.get("hair_refinement_version") == VERSION:
        return existing
    source = bpy.data.materials.get("Character / natural charcoal fibres")
    if source is None:
        raise RuntimeError("The integrated natural-charcoal material is required.")
    if existing is not None and existing.get("hair_refinement_version"):
        result = existing
    else:
        result = source.copy()
        result.name = MAT
    shader = next(n for n in result.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Roughness"].default_value = .69
    shader.inputs["Specular IOR Level"].default_value = .16
    if shader.inputs.get("Anisotropic IOR Level"):
        shader.inputs["Anisotropic IOR Level"].default_value = .22
    for node in result.node_tree.nodes:
        if node.type == "NORMAL_MAP":
            node.inputs["Strength"].default_value = .12
    result["hair_refinement_version"] = VERSION
    return result


class Surface:
    """Periodic bilinear sampler on the existing closed cross-section strips."""
    def __init__(self, obj, coords):
        self.rows = H.rings(H.params(obj))
        self.coords = coords
        self.n = len(self.rows[0][1])

    def point(self, u, t):
        t = max(0, min(1, t))
        row = max(0, min(len(self.rows)-2, int(t*(len(self.rows)-1))))
        ta, ia = self.rows[row]
        tb, ib = self.rows[row+1]
        f = max(0, min(1, (t-ta)/max(1e-8, tb-ta)))
        s = (u % 1)*self.n
        i = int(s) % self.n
        j = (i+1) % self.n
        a = self.coords[ia[i]].lerp(self.coords[ia[j]], s-int(s))
        b = self.coords[ib[i]].lerp(self.coords[ib[j]], s-int(s))
        return a.lerp(b, f)

    def center(self, t):
        return sum((self.point(i/self.n, t) for i in range(self.n)), Vector())/self.n


def map_near_surface(point, obj, original, modified, tree, maximum=.004):
    hit, normal, face, distance = tree.find_nearest(point)
    if hit is None or distance > maximum:
        return None
    indices = list(obj.data.polygons[face].vertices)
    if len(indices) > 4:
        return None
    best = None
    for k in range(1, len(indices)-1):
        ids = (indices[0], indices[k], indices[k+1])
        a, b, c = (original[i] for i in ids)
        q = closest_point_on_tri(point, a, b, c)
        if best is None or (q-point).length_squared < best[0]:
            ab, ac, aq = b-a, c-a, q-a
            denom = ab.dot(ab)*ac.dot(ac)-ab.dot(ac)**2
            if abs(denom) < 1e-18:
                continue
            v = (ac.dot(ac)*aq.dot(ab)-ab.dot(ac)*aq.dot(ac))/denom
            w = (ab.dot(ab)*aq.dot(ac)-ab.dot(ac)*aq.dot(ab))/denom
            weights = (1-v-w, v, w)
            mapped = sum((modified[i]*weight for i, weight in zip(ids, weights)), Vector())
            new_normal = ((modified[ids[1]]-modified[ids[0]]).cross(
                modified[ids[2]]-modified[ids[0]])).normalized()
            if new_normal.dot(radial_normal(mapped)) < 0:
                new_normal = -new_normal
            offset = max(-.00015, min(.0018, (point-q).dot(normal)))
            mapped += new_normal*(offset+.00028)
            best = ((q-point).length_squared, mapped)
    return best[1] if best else None


class Strands:
    def __init__(self):
        self.vertices, self.faces, self.uvs, self.binds = [], [], [], []

    def add(self, path, radius, phase, binds=None, normals=None, flatten=1,
            sides=3, root_embedded=False):
        if len(path) < 4:
            return
        offset = len(self.vertices)
        binds = binds or path
        previous = None
        for i, point in enumerate(path):
            t = i/(len(path)-1)
            tangent = (path[min(i+1, len(path)-1)]-
                       path[max(0, i-1)]).normalized()
            normal = normals[i].copy() if normals else radial_normal(point)
            normal -= tangent*normal.dot(tangent)
            if normal.length < .01:
                normal = Vector((0, -1, 0))
                normal -= tangent*normal.dot(tangent)
            normal.normalize()
            width = tangent.cross(normal).normalized()
            if previous is not None and width.dot(previous) < 0:
                width, normal = -width, -normal
            previous = width
            taper = max(.015, smooth(0, .13, t)*(1-smooth(.60, 1, t)))
            if not root_embedded:
                taper = max(.015, math.sin(math.pi*t)**.48)
            for k in range(sides):
                angle = math.tau*k/sides
                displacement = radius*taper*(width*math.cos(angle)+
                                            normal*math.sin(angle)*flatten)
                self.vertices.append(tuple(point+displacement))
                self.binds.append(tuple(binds[i]+displacement))
                self.uvs.append(((phase+k/sides*.13) % 1, t))
        for i in range(len(path)-1):
            for k in range(sides):
                a = offset+i*sides+k
                b = offset+i*sides+(k+1) % sides
                self.faces.append((a, b, b+sides, a+sides))
        self.faces.append(tuple(offset+i for i in reversed(range(sides))))
        end = offset+(len(path)-1)*sides
        self.faces.append(tuple(end+i for i in range(sides)))


def create_mesh(name, data, collection, rig, mat, side=None):
    mesh = bpy.data.meshes.new(name+" mesh")
    mesh.from_pydata(data.vertices, [], data.faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    for uvname in ("UVMap", "HairRealismUV"):
        layer = mesh.uv_layers.new(name=uvname)
        for polygon in mesh.polygons:
            polygon.use_smooth = True
            for li in polygon.loop_indices:
                layer.data[li].uv = data.uvs[mesh.loops[li].vertex_index]
        layer.active_render = uvname == "HairRealismUV"
    bind = mesh.attributes.new(BIND, "FLOAT_VECTOR", "POINT")
    for i, co in enumerate(data.binds):
        bind.data[i].vector = co
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(mat)
    obj.parent = rig
    obj["part"] = "hair"
    obj["hair_region"] = "ponytail" if side else "scalp"
    obj["rig_attachment"] = "hair_"+side if side else "head"
    obj["hair_refinement_generated"] = VERSION
    if side:
        obj["side"] = side
    groups = {}
    for i, co in enumerate(data.binds):
        weights = H.weights(co, side)
        total = sum(weights.values())
        for bone, weight in weights.items():
            if weight < 1e-8:
                continue
            if bone not in rig.data.bones:
                raise RuntimeError("Missing existing hair bone "+bone)
            group = groups.get(bone)
            if group is None:
                group = groups[bone] = obj.vertex_groups.new(name=bone)
            group.add([i], weight/total, "REPLACE")
    mod = obj.modifiers.new("Existing hair spring chain", "ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = True
    return obj


def remove_generated():
    for name in GENERATED:
        obj = bpy.data.objects.get(name)
        if obj:
            if not obj.get("hair_refinement_generated"):
                raise RuntimeError("Generated name occupied by unrelated object: "+name)
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if not mesh.users:
                bpy.data.meshes.remove(mesh)


def scalp_layers(cap, current, original, modified):
    randomizer = random.Random(481828)
    result = Strands()
    surface = Surface(cap, modified[cap.name])
    # Soft diagonal gather sweeps cross the posterior cap toward the low ties.
    for side in (-1, 1):
        for k in range(22):
            start_u = .13+(k+.35+randomizer.uniform(-.2, .2))*.365/22
            target_u = .318+randomizer.uniform(-.016, .016)
            start = randomizer.uniform(.08, .33)
            end = randomizer.uniform(.88, .995)
            path, normals = [], []
            for j in range(20):
                q = j/19
                t = start+(end-start)*q
                u = start_u+(target_u-start_u)*.72*smooth(.20, 1, q)
                u += .0024*math.sin(math.pi*q)*math.sin(k*1.63)
                u = u if side == 1 else 1-u
                point = surface.point(u, t)
                normal = radial_normal(point)
                # Roots and tapered ends sit back into the continuous shell.
                point += normal*(.00034+.00016*math.sin(math.pi*q)**.8)
                path.append(point)
                normals.append(normal)
            result.add(path, randomizer.uniform(.00065, .00115),
                       randomizer.random(), normals=normals,
                       flatten=.055, sides=4, root_embedded=True)
    # The concealed lower support ends in staggered nape hairs rather than a
    # uniform horizontal blade. These are visible closed geometry.
    for k in range(43):
        u = .275+(k+randomizer.uniform(-.3, .3))*.45/42
        path, normals = [], []
        for j in range(11):
            q = j/10
            t = .76+.24*q
            point = surface.point(u+.002*math.sin(q*math.pi+k), t)
            normal = radial_normal(point)
            point += normal*(.0006+.0030*smooth(.65, 1, q))
            point.z -= randomizer.uniform(.0004, .0025)*q**4
            path.append(point)
            normals.append(normal)
        result.add(path, randomizer.uniform(.00025, .00065),
                   randomizer.random(), normals=normals, flatten=.30,
                   sides=3, root_embedded=True)
    # Small terminal subdivisions of the fringe break wide triangular leaves.
    for obj in current:
        if not obj.name.startswith("Hair_Fringe_") or "_Flow" in obj.name:
            continue
        sf = Surface(obj, modified[obj.name])
        for k in range(3):
            u = .50+(k-1)*.17
            path, normals = [], []
            end = .92+randomizer.uniform(-.035, .065)
            for j in range(16):
                q = j/15
                t = .28+(end-.28)*q
                point = sf.point(u+.025*math.sin(q*math.pi+k), t)
                point.y -= .00030+.00045*math.sin(math.pi*q)
                path.append(point)
                normals.append(Vector((0, -1, .1)))
            result.add(path, randomizer.uniform(.0011, .0018),
                       randomizer.random(), normals=normals, flatten=.12,
                       sides=4, root_embedded=True)
    # Isolated fine fibres are deliberately sparse and varied.
    for k in range(60):
        u = randomizer.random()
        start = randomizer.uniform(.08, .30)
        end = randomizer.uniform(.65, .97)
        path, normals = [], []
        for j in range(12):
            q = j/11
            point = surface.point(u+.0038*math.sin(math.pi*q+k*.71),
                                  start+(end-start)*q)
            normal = radial_normal(point)
            lift = .00010+randomizer.uniform(.00012, .00026)
            if k % 11 == 0:
                lift += .0013*math.sin(math.pi*q)**2
            path.append(point+normal*lift)
            normals.append(normal)
        result.add(path, randomizer.uniform(.000055, .00009),
                   randomizer.random(), normals=normals, sides=3)
    return result


def tail_layers(current, original, modified, side):
    randomizer = random.Random(530 if side == "L" else 617)
    result = Strands()
    for obj in current:
        if obj.get("hair_region") != "ponytail" or obj.get("side") != side:
            continue
        if "_Flow" in obj.name or obj.name.startswith("Hair_Realism_"):
            continue
        sf = Surface(obj, modified[obj.name])
        before = Surface(obj, original[obj.name])
        count = 14 if "TailCore" in obj.name else 3 if "Ponytail_" in obj.name else 1
        for k in range(count):
            u = (k+.31+randomizer.uniform(-.15, .15))/count
            start, end = randomizer.uniform(.035, .12), randomizer.uniform(.86, .998)
            path, binds, normals = [], [], []
            for j in range(16):
                q = j/15
                t = start+(end-start)*q
                v = u+.015*math.sin(q*math.tau+u*8)
                point = sf.point(v, t)
                normal = (point-sf.center(t)).normalized()
                lift = .00035+.00075*math.sin(math.pi*q)**2
                path.append(point+normal*lift)
                binds.append(before.point(v, t)+normal*lift)
                normals.append(normal)
            result.add(path, randomizer.uniform(.0006, .00135),
                       randomizer.random(), binds, normals, flatten=.23,
                       sides=4, root_embedded=True)
        for k in range(max(2, count)):
            u = randomizer.random()
            start, end = randomizer.uniform(.03, .20), randomizer.uniform(.80, .99)
            path, binds, normals = [], [], []
            for j in range(15):
                q = j/14
                t = start+(end-start)*q
                v = u+.019*math.sin(q*math.tau+u*17)
                point = sf.point(v, t)
                normal = (point-sf.center(t)).normalized()
                lift = .00015+.0015*math.sin(math.pi*q)**2*(k % 4 == 0)
                path.append(point+normal*lift)
                binds.append(before.point(v, t)+normal*lift)
                normals.append(normal)
            result.add(path, randomizer.uniform(.00005, .000095),
                       randomizer.random(), binds, normals, sides=3)
    return result


def apply(collection=None, rig=None):
    _TAIL_SOURCE_CENTERS.clear()
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    rig = rig or bpy.data.objects["Character_Rig"]
    current = objects(collection)
    cap = bpy.data.objects["Hair_ScalpFoundation"]
    if cap.get("head_profile_version") != "balanced-cranial-profile-02":
        raise RuntimeError("This patch requires the integrated head-profile-02 geometry.")
    if any(o.data.shape_keys for o in current):
        raise RuntimeError("Unexpected shape-key hair; cannot safely checkpoint it.")
    original = {o.name: checkpoint(o) for o in current}
    remove_generated()
    modified = {name: [co.copy() for co in coords] for name, coords in original.items()}
    head = next(o for o in collection.all_objects if o.type == "MESH"
                and o.get("facial_role") == "continuous_skin_lids_lips")
    head_bvh = make_bvh(head)
    modified[cap.name] = cap_coordinates(cap, original[cap.name], head_bvh)
    cap_bvh = BVHTree.FromPolygons(
        [tuple(co) for co in modified[cap.name]],
        [tuple(p.vertices) for p in cap.data.polygons], all_triangles=False)
    old_cap_bvh = BVHTree.FromPolygons(
        [tuple(co) for co in original[cap.name]],
        [tuple(p.vertices) for p in cap.data.polygons], all_triangles=False)
    centers = {side: ring_centers(
        bpy.data.objects["Hair_TailCore_"+side], original["Hair_TailCore_"+side])
        for side in ("L", "R")}
    for obj in current:
        name = obj.name
        coords = modified[name]
        if name == cap.name:
            continue
        if obj.get("hair_region") == "ponytail":
            side = obj.get("side")
            if side not in centers:
                raise RuntimeError("Ponytail side tag is missing: "+name)
            if name.startswith("Hair_Realism_"):
                for i, source in enumerate(original[name]):
                    center = center_at_z(centers[side], source.z)
                    radial = source-center
                    fade = 1-smooth(1.337, 1.378, source.z)
                    radial.x *= 1-.31*fade
                    radial.y *= 1-.35*fade
                    coords[i] = tail_shift(center+radial, side)
            else:
                for t, indices in H.rings(H.params(obj)):
                    center = sum((original[name][i] for i in indices), Vector())/len(indices)
                    fade = smooth(.08, .30, t)
                    for i in indices:
                        radial = original[name][i]-center
                        if "TailCore" in name:
                            radial.x *= 1-.36*fade
                            radial.y *= 1-.40*fade
                        elif "Ponytail_" in name:
                            radial.x *= 1-.36*fade
                            radial.y *= 1-.58*fade
                        coords[i] = tail_shift(center+radial, side)
        elif name.startswith("Hair_Fringe_") and "_Flow" not in name:
            for t, indices in H.rings(H.params(obj)):
                center = sum((original[name][i] for i in indices), Vector())/len(indices)
                for i in indices:
                    point = soften_crown(original[name][i])
                    point.x = center.x+(point.x-center.x)*(1-.43*smooth(.38, .93, t))
                    point.z += .0020*smooth(.80, 1, t)
                    if point.z > 1.505:
                        hit, _, _, _ = cap_bvh.ray_cast(
                            Vector((point.x, -.30, point.z)), Vector((0, 1, 0)), .6)
                        if hit is not None:
                            emergence = 1-smooth(1.515, 1.550, point.z)
                            # Embed the tiny root loop; merge gradually into
                            # an almost flush, thin cross section.
                            desired = hit.y+.00045-.0012*emergence
                            point.y = point.y*(1-smooth(1.505, 1.535, point.z))+\
                                desired*smooth(1.505, 1.535, point.z)
                    coords[i] = point
        elif name == "Hair_Realism_ScalpFibres":
            for i, source in enumerate(original[name]):
                mapped = map_near_surface(source, cap, original[cap.name],
                                          modified[cap.name], old_cap_bvh)
                if mapped is not None:
                    coords[i] = mapped
                    continue
                # Free fringe fibres retain their local curved path.
                point = soften_crown(source)
                coords[i] = point
        elif name.startswith("Hair_Temple_") and "_Flow" not in name:
            for t, indices in H.rings(H.params(obj)):
                center = sum((original[name][i] for i in indices), Vector())/len(indices)
                for i in indices:
                    point = soften_crown(original[name][i])
                    point.x = center.x+(point.x-center.x)*(1-.18*smooth(.3, .9, t))
                    coords[i] = point
    clearance = measured_clearance(current, modified, garment_front_trees(collection, rig))
    for side in ("L", "R"):
        bpy.data.objects["Hair_TailCore_"+side]["hair_refinement_motion_field"] = json.dumps(
            clearance[side])
    mat = material()
    for obj in current:
        for vertex, co in zip(obj.data.vertices, modified[obj.name]):
            vertex.co = co
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj["hair_refinement_version"] = VERSION
        obj["hair_refinement_output_sha"] = coordinate_hash(
            vertex.co for vertex in obj.data.vertices)
        obj.data.update()
    new = [create_mesh(GENERATED[0], scalp_layers(cap, current, original, modified),
                       collection, rig, mat)]
    for name, side in zip(GENERATED[1:], ("L", "R")):
        new.append(create_mesh(name, tail_layers(current, original, modified, side),
                               collection, rig, mat, side))
    bpy.context.view_layer.update()
    return {
        "version": VERSION, "changedMeshes": [o.name for o in current],
        "newMeshes": [o.name for o in new],
        "newVertices": sum(len(o.data.vertices) for o in new),
        "newTriangles": sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in new),
        "existingBonesAndFaceUnchanged": True, "tieGeometryUnchanged": True,
        "bindingPositionAttribute": BIND,
        "maximumTipLoweringMeters": .040,
        "tailRestRoute": "posterior shoulder with near-vertical lower fall",
        "garmentClearance": clearance,
        "springTargetAdditions": [
            {"mesh": name, "source": "hair", "mode": "spring",
             "chains": ["ponytail_"+side]}
            for name, side in zip(GENERATED[1:], ("L", "R"))],
        "fixedTargetAddition": {
            "mesh": GENERATED[0], "source": "hair", "mode": "fixed",
            "reason": "Short scalp and nape layers are fixed to the head."},
    }


def restore(collection=None):
    collection = collection or bpy.data.collections["CHARACTER · Original geometry"]
    remove_generated()
    for obj in objects(collection):
        attr = obj.data.attributes.get(BACKUP)
        if attr is None:
            continue
        for vertex, point in zip(obj.data.vertices, attr.data):
            vertex.co = point.vector
        obj.data.materials.clear()
        for name in json.loads(obj["hair_refinement_materials"]):
            if name:
                obj.data.materials.append(bpy.data.materials[name])
        if "hair_refinement_version" in obj:
            del obj["hair_refinement_version"]
        if "hair_refinement_motion_field" in obj:
            del obj["hair_refinement_motion_field"]
        obj.data.update()
    bpy.context.view_layer.update()


def static_surface_check(collection, rig):
    trees = garment_front_trees(collection, rig)
    state = rig.data.pose_position
    report = {}
    try:
        for mode, tree in zip(("REST", "POSE"), trees):
            rig.data.pose_position = mode
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            vertices, polygons = [], []
            for obj in collection.all_objects:
                if obj.type != "MESH" or obj.get("hair_region") != "ponytail":
                    continue
                evaluated = obj.evaluated_get(dg)
                mesh = evaluated.to_mesh()
                matrix = rig.matrix_world.inverted() @ evaluated.matrix_world
                offset = len(vertices)
                vertices.extend(tuple(matrix @ v.co) for v in mesh.vertices)
                polygons.extend(tuple(offset+i for i in p.vertices) for p in mesh.polygons)
                evaluated.to_mesh_clear()
            hair_tree = BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
            pairs = tree.overlap(hair_tree)
            report[mode] = {"intersectingPolygonPairs": len(pairs),
                            "examples": pairs[:6],
                            "hairVertices": len(vertices)}
    finally:
        rig.data.pose_position = state
        bpy.context.view_layer.update()
    return report


def nonhair_signature(collection):
    digest = hashlib.sha256()
    for obj in sorted(collection.all_objects, key=lambda o: o.name):
        if obj.type != "MESH" or obj.name.startswith("Hair_"):
            continue
        digest.update(obj.name.encode())
        digest.update(coordinate_hash(v.co for v in obj.data.vertices).encode())
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                digest.update(key.name.encode())
                digest.update(coordinate_hash(p.co for p in key.data).encode())
                digest.update(repr(key.value).encode())
    return digest.hexdigest()


def background_review():
    if not bpy.app.background:
        raise RuntimeError("Review must run in a disposable background process.")
    source = Path(bpy.data.filepath).resolve()
    out = P/"work/qa-hair-refinement"
    if source != out/"source-snapshot.blend":
        raise RuntimeError("Review opens only work/qa-hair-refinement/source-snapshot.blend")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    collection = bpy.data.collections["CHARACTER · Original geometry"]
    nonhair_before = nonhair_signature(collection)
    baseline = {o.name: [tuple(v.co) for v in o.data.vertices]
                for o in objects(collection)}
    report = apply()
    first = {o.name: [tuple(v.co) for v in o.data.vertices]
             for o in objects(collection)}
    apply()
    assert all(first[o.name] == [tuple(v.co) for v in o.data.vertices]
               for o in objects(collection)), "Patch must be idempotent"
    restore()
    assert all(baseline[o.name] == [tuple(v.co) for v in o.data.vertices]
               for o in objects(collection)), "Restore must be exact"
    report = apply()
    report.update(source=str(source), sourceSha256=source_sha,
                  idempotent=True, restoreExact=True)
    assert nonhair_signature(collection) == nonhair_before
    report["allNonhairGeometryAndExpressionKeysUnchanged"] = True
    report["staticSurfaceCheck"] = static_surface_check(
        collection, bpy.data.objects["Character_Rig"])
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"hair-refinement-review.blend"))
    (out/"report.json").write_text(json.dumps(report, indent=2))
    if "--render" in sys.argv:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 12
        scene.cycles.use_denoising = True
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
        scene.render.resolution_x, scene.render.resolution_y = 560, 640
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        data = bpy.data.cameras.new("Hair refinement review camera")
        camera = bpy.data.objects.new("Hair refinement review camera", data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        data.type = "ORTHO"
        data.ortho_scale = .56
        target = Vector((0, .025, 1.35))
        for name, position in (
            ("portrait", (.5, -3, 1.36)), ("profile", (3, 0, 1.36)),
            ("back", (0, 3, 1.36)),
        ):
            camera.location = position
            camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(out/(name+".png"))
            bpy.ops.render.render(write_still=True)
            print("HAIR_REFINEMENT_RENDER", name, flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    print("HAIR_REFINEMENT_READY", json.dumps(report), flush=True)


if __name__ == "__main__":
    background_review()
