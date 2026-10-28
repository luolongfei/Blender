"""Original low twin-tail hairstyle, modeled from the approved character sheets.

Coordinates are metres, Z up, face towards -Y. No external model is used.
Public entry point: build_hair(collection, materials) -> list[bpy.types.Object].
All closed hair locks have explicit UVs and are separate editable mesh objects.
"""
import math
import bpy
import bmesh
from mathutils import Vector

TAU = math.tau


def _mesh(name, verts, faces, uv, material, collection, side=None):
    data = bpy.data.meshes.new(name + "_Mesh")
    data.from_pydata(verts, [], faces)
    data.update()
    bm = bmesh.new()
    bm.from_mesh(data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(data)
    bm.free()
    layer = data.uv_layers.new(name="UVMap")
    for poly in data.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            layer.data[li].uv = uv[data.loops[li].vertex_index]
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    if material:
        data.materials.append(material)
    obj["part"] = "hair"
    obj["hair_region"] = "scalp"
    obj["source"] = "Original parametric modeling from approved four-view artwork"
    if side:
        obj["side"] = side
    return obj


def _catmull(points, steps):
    points = [Vector(v) for v in points]
    samples = []
    for i in range(steps):
        t = i / (steps - 1)
        f = t * (len(points) - 1)
        k = min(int(f), len(points) - 2)
        u = f - k
        p0 = points[max(0, k - 1)]
        p1, p2 = points[k], points[k + 1]
        p3 = points[min(len(points) - 1, k + 2)]
        samples.append(0.5 * (
            2 * p1 + (-p0 + p2) * u
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
            + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u))
    return samples


def _frames(points, normal):
    frames = []
    for i, p in enumerate(points):
        tangent = points[min(i + 1, len(points) - 1)] - points[max(0, i - 1)]
        tangent.normalize()
        outward = Vector(normal(p, i / (len(points) - 1)) if callable(normal) else normal)
        outward -= tangent * outward.dot(tangent)
        if outward.length < 0.01:
            outward = Vector((0, -1, 0))
            outward -= tangent * outward.dot(tangent)
        outward.normalize()
        width_axis = tangent.cross(outward).normalized()
        frames.append((width_axis, outward))
    return frames


def _stroke(name, points, radius, material, collection, side=None):
    """Thin mesh line used sparingly to suggest grouped strand direction."""
    if len(points) < 3:
        return None
    frames = _frames(points, (0, -1, 0))
    verts, faces, uv = [], [], []
    sides = 5
    for i, p in enumerate(points):
        t = i / (len(points) - 1)
        w, n = frames[i]
        r = radius * max(0.06, math.sin(math.pi * t) ** 0.4)
        for j in range(sides):
            a = TAU * j / sides
            verts.append(tuple(p + r * (math.cos(a) * w + math.sin(a) * n)))
            uv.append((j / sides, t))
    for i in range(len(points) - 1):
        for j in range(sides):
            k = (j + 1) % sides
            faces.append((i * sides + j, i * sides + k,
                          (i + 1) * sides + k, (i + 1) * sides + j))
    faces += [tuple(reversed(range(sides))),
              tuple((len(points) - 1) * sides + j for j in range(sides))]
    return _mesh(name, verts, faces, uv, material, collection, side)


def _lock(name, controls, width, thickness, normal, material, collection,
          side=None, region="scalp", strands=0, stroke_mat=None,
          root_ratio=0.30, peak_at=0.40, steps=29, section=12,
          root_embed=0.0, tip_length=0.30):
    """Closed, flattened, convex hair ribbon with a gently rolled cross-section."""
    pts = _catmull(controls, steps)
    frames = _frames(pts, normal)
    if root_embed:
        # A rooted lock grows out through the cap, rather than placing a visible
        # squared end-cap on top of it. Keep the shallow emergence gradual.
        pts = [
            p - frames[i][1] * root_embed
            * max(0.0, 1 - (i / (steps - 1)) / 0.34) ** 2
            for i, p in enumerate(pts)
        ]
        frames = _frames(pts, normal)
    widths, depths = [], []
    verts, faces, uv = [], [], []
    for i, p in enumerate(pts):
        t = i / (steps - 1)
        # Broad upper-middle section, then continuously narrowing soft tip.
        bell = max(0, math.sin(
            math.pi * t ** (math.log(0.5) / math.log(peak_at)))) ** 0.6
        profile = max(0.008, root_ratio * (1 - t) ** 2 + (1 - root_ratio) * bell)
        # A long terminal taper makes a hair point, not a rounded teardrop.
        if t > 1 - tip_length:
            profile *= max(0.008, (1 - t) / tip_length) ** 0.75
        profile = max(0.0025, profile)
        half_width = width * profile
        depth = thickness * max(0.025, profile ** 0.7)
        widths.append(half_width)
        depths.append(depth)
        w, n = frames[i]
        for j in range(section):
            a = TAU * j / section
            # Flatten underside, gently convex visible surface.
            sn = math.sin(a)
            depth_scale = 1.0 if sn > 0 else 0.66
            ridge = 0.08 * math.cos(3 * a + 0.2 * t) * max(0, sn)
            offset = w * (half_width * math.cos(a))
            offset += n * (depth * (sn * depth_scale + ridge))
            verts.append(tuple(p + offset))
            uv.append((j / section, t))
    for i in range(steps - 1):
        for j in range(section):
            j2 = (j + 1) % section
            faces.append((i * section + j, i * section + j2,
                          (i + 1) * section + j2, (i + 1) * section + j))
    faces.extend((tuple(reversed(range(section))),
                  tuple((steps - 1) * section + j for j in range(section))))
    obj = _mesh(name, verts, faces, uv, material, collection, side)
    obj["hair_region"] = region
    objects = [obj]
    # Multiple sub-strands are only accents; leave ample uninterrupted areas.
    for strand in range(strands):
        across = (-0.35, 0.24, 0.57)[strand % 3]
        path = []
        start = max(2, round(steps * (0.16 + strand * 0.06)))
        stop = min(steps - 3, round(steps * (0.88 - strand * 0.07)))
        for i in range(start, stop):
            w, n = frames[i]
            top = math.sqrt(max(0, 1 - across * across))
            path.append(pts[i] + w * (widths[i] * across)
                        + n * (depths[i] * (top + 0.10) + 0.00010))
        if len(path) >= 3:
            line = _stroke(name + "_Flow%02d" % strand, path, 0.00023,
                           stroke_mat or material, collection, side)
            line["hair_region"] = region
            objects.append(line)
    return objects


def _ellipsoid_point(theta, phi, inflation=0):
    # Phi zero at forehead, pi at back. Crown has a tiny off-centre part.
    rx, ry, rz = 0.121 + inflation, 0.108 + inflation, 0.145 + inflation
    return Vector((rx * math.sin(theta) * math.sin(phi),
                   -ry * math.sin(theta) * math.cos(phi) + 0.004,
                   1.450 + rz * math.cos(theta)))


def _scalp_normal(p, _=0):
    return Vector((p.x / (0.121 ** 2),
                   (p.y - 0.004) / (0.108 ** 2),
                   (p.z - 1.450) / (0.145 ** 2))).normalized()


def _cap(collection, mats):
    verts, faces, uv = [], [], []
    around, rows = 96, 32
    # Lower edge stays above brows at front, passes over ears, cups the nape.
    for i in range(rows + 1):
        f = i / rows
        for j in range(around):
            phi = TAU * j / around
            backness = (1 - math.cos(phi)) / 2
            end = 1.245 + 0.87 * (backness ** 0.64)
            theta = 0.004 + f * (end - 0.004)
            p = _ellipsoid_point(theta, phi)
            verts.append(tuple(p))
            uv.append((j / around, f))
    for i in range(rows):
        for j in range(around):
            k = (j + 1) % around
            faces.append((i * around + j, i * around + k,
                          (i + 1) * around + k, (i + 1) * around + j))
    faces.append(tuple(reversed(range(around))))
    obj = _mesh("Hair_ScalpFoundation", verts, faces, uv, mats["hair"], collection)
    # Solid backing prevents visible open edges above ear/forehead.
    solid = obj.modifiers.new("Scalp edge thickness", "SOLIDIFY")
    solid.thickness = 0.0012
    solid.offset = -1
    return obj


def _tie(name, center, axis, collection, material, side):
    axis = Vector(axis).normalized()
    w = axis.cross(Vector((0, 1, 0))).normalized()
    n = axis.cross(w).normalized()
    verts, uv, faces = [], [], []
    steps, section = 44, 8
    center = Vector(center)
    for i in range(steps):
        a = TAU * i / steps
        radial = w * math.cos(a) + n * math.sin(a)
        c = center + w * (0.019 * math.cos(a)) + n * (0.0155 * math.sin(a))
        for j in range(section):
            b = TAU * j / section
            verts.append(tuple(c + 0.0024 * (radial * math.cos(b) + axis * math.sin(b))))
            uv.append((i / steps, j / section))
    for i in range(steps):
        for j in range(section):
            faces.append((i * section + j, ((i + 1) % steps) * section + j,
                          ((i + 1) % steps) * section + (j + 1) % section,
                          i * section + (j + 1) % section))
    obj = _mesh(name, verts, faces, uv, material, collection, side)
    obj["hair_region"] = "tie"
    return obj


def build_hair(collection, mats):
    objects = [_cap(collection, mats)]
    hair, highlight, shadow = mats["hair"], mats["hair_highlight"], mats["hair_shadow"]

    # Crown layers flow away from the part toward the temples and nape.
    for sign, side in ((-1, "R"), (1, "L")):
        for i in range(8):
            phi_end = sign * (0.55 + i * 0.342)
            end_theta = 1.39 + 0.68 * (i / 7) ** 0.7
            controls = []
            for j in range(6):
                t = j / 5
                root_theta = 0.19 + 0.045 * (i % 3)
                theta = root_theta + (end_theta - root_theta) * t
                phi = phi_end + sign * 0.32 * (1 - t) * math.sin(math.pi * t)
                p = _ellipsoid_point(theta, phi, 0.0027)
                p.x += 0.0035 * (1 - t) ** 2
                # Back layers gather subtly toward low tied sections.
                if i > 3 and t > 0.60:
                    blend = ((t - 0.60) / 0.40) ** 2 * 0.35
                    target = Vector((sign * 0.101, 0.054, 1.385))
                    p = p.lerp(target, blend)
                controls.append(tuple(p))
            objects += _lock("Hair_Crown_%s_%02d" % (side, i), controls,
                             width=0.0180 + 0.0038 * math.sin(i * 0.71) ** 2,
                             thickness=0.00125, normal=_scalp_normal,
                             material=hair, collection=collection, side=side,
                             strands=1 if i in (2, 6) else 0, stroke_mat=highlight,
                             root_ratio=0.025, peak_at=0.52, steps=38,
                             root_embed=0.0045, tip_length=0.24)

    # Individually designed, gently swept tapered fringe. The brows remain exposed.
    fringes = [
        (-0.091, 1.450, -0.042, 0.0160),
        (-0.067, 1.482, -0.013, 0.0190),
        (-0.044, 1.469,  0.002, 0.0180),
        (-0.020, 1.458,  0.014, 0.0160),
        ( 0.003, 1.450,  0.024, 0.0150),
        ( 0.025, 1.462,  0.030, 0.0150),
        ( 0.048, 1.478,  0.043, 0.0170),
        ( 0.070, 1.465,  0.055, 0.0180),
        ( 0.091, 1.447,  0.067, 0.0140),
    ]
    for i, (tipx, tipz, rootx, width) in enumerate(fringes):
        side = "L" if tipx > 0 else "R"
        bias = 0.013 if tipx > 0.015 else -0.009
        y_tip = -0.101 + 0.018 * (abs(tipx) / 0.095) ** 2
        controls = [
            (rootx * 0.56 + 0.009, -0.033, 1.583 - abs(rootx) * 0.24),
            (rootx - bias * 0.22, -0.073, 1.557 - abs(rootx) * 0.08),
            (tipx - bias * 1.4, -0.105, 1.515),
            (tipx - bias * 0.8, y_tip - 0.009, tipz + 0.020),
            (tipx, y_tip, tipz),
        ]
        objects += _lock("Hair_Fringe_%02d" % i, controls, width, 0.00165,
                         (0, -1, 0), hair, collection, side,
                         region="fringe", strands=1 if i in (2, 7) else 0,
                         stroke_mat=highlight, root_ratio=0.025, peak_at=0.45,
                         steps=42, root_embed=0.0060, tip_length=0.38)

    # Soft temple sweeps give depth around the ear without hiding the cheek.
    for sign, side in ((-1, "R"), (1, "L")):
        for j in range(3):
            controls = [
                (sign * (0.063 + j * 0.011), -0.048, 1.565 - j * 0.007),
                (sign * (0.105 + j * 0.004), -0.068, 1.511),
                (sign * (0.109 + j * 0.003), -0.064, 1.457),
                (sign * (0.103 + j * 0.003), -0.070, 1.409 - j * 0.010),
                (sign * (0.089 + j * 0.004), -0.078, 1.367 + j * 0.004),
            ]
            objects += _lock("Hair_Temple_%s_%02d" % (side, j), controls,
                             0.0105 - j * 0.0012, 0.0022,
                             (sign * 0.65, -0.76, 0), hair, collection, side,
                             region="side_lock", strands=1 if j == 0 else 0,
                             stroke_mat=highlight, root_ratio=0.04, peak_at=0.33,
                             root_embed=0.004, tip_length=0.35)

    # Two low ponytail groups: a softly folded foundation with overlapping leaves.
    for sign, side in ((-1, "R"), (1, "L")):
        core_controls = [
            (sign * 0.105, 0.046, 1.383),
            (sign * 0.131, 0.050, 1.338),
            (sign * 0.147, 0.008, 1.291),
            (sign * 0.160, -0.075, 1.244),
            (sign * 0.148, -0.100, 1.214),
            (sign * 0.131, -0.108, 1.191),
        ]
        objects += _lock("Hair_TailCore_" + side, core_controls,
                         0.035, 0.024, (0, -1, 0), hair, collection,
                         side, region="ponytail", root_ratio=0.32,
                         peak_at=0.48, steps=44, section=16, tip_length=0.35)
        for j in range(5):
            a = TAU * j / 5
            # Front/back overlap makes each tail a volume instead of a flat fan.
            dx = 0.018 * math.cos(a)
            dy = 0.015 * math.sin(a)
            end_z = (1.198, 1.226, 1.184, 1.213, 1.194)[j]
            end_x = (0.123, 0.150, 0.129, 0.141, 0.115)[j]
            controls = [
                (sign * (0.105 + dx * 0.35), 0.045 + dy * 0.4, 1.383),
                (sign * (0.132 + dx), 0.050 + dy, 1.337),
                (sign * (0.148 + dx * 1.12), 0.008 + dy * 1.4, 1.286),
                (sign * (0.160 + dx * 0.65), -0.078 + dy, 1.256),
                (sign * (0.150 + dx * 0.6), -0.101 + dy * 0.6, end_z + 0.027),
                (sign * end_x, -0.108 + dy * 0.7, end_z),
            ]
            normal = (sign * 0.50 * math.cos(a), math.sin(a), 0.13)
            objects += _lock("Hair_Ponytail_%s_%02d" % (side, j), controls,
                             0.024 + 0.002 * (j % 3), 0.0042, normal,
                             hair, collection, side, region="ponytail",
                             strands=1 if j in (0, 3, 4) else 0,
                             stroke_mat=highlight, root_ratio=0.32,
                             peak_at=0.40, steps=44, tip_length=0.36)
        # Three slender flowing perimeter locks soften the bundled silhouette.
        for j in range(2):
            controls = [
                (sign * 0.111, 0.041 + 0.010 * j, 1.378),
                (sign * (0.156 + 0.006 * j), 0.030 + 0.010 * j, 1.321),
                (sign * (0.180 + 0.004 * j), -0.018 + 0.008 * j, 1.273),
                (sign * (0.177 + 0.002 * j), -0.081 + 0.006 * j, 1.229),
                (sign * (0.149 - 0.004 * j), -0.103 + 0.004 * j, 1.215 - 0.007 * j),
            ]
            objects += _lock("Hair_TailEdge_%s_%02d" % (side, j), controls,
                             0.0062, 0.0034, (sign * 0.7, -0.7, 0),
                             hair, collection, side, region="ponytail",
                             strands=0, root_ratio=0.26, steps=32)
        objects.append(_tie("Hair_Tie_" + side, (sign * 0.105, 0.046, 1.382),
                            (sign * 0.30, 0.15, -1), collection,
                            mats["hair_tie"], side))

    # Tiny flyaway arcs across the crown, kept restrained and close to the cap.
    for sign, side in ((-1, "R"), (1, "L")):
        path = _catmull([
            (0.003, -0.005, 1.596),
            (sign * 0.031, 0.006, 1.601),
            (sign * 0.067, 0.015, 1.582),
            (sign * 0.088, 0.011, 1.559),
        ], 24)
        line = _stroke("Hair_CrownFlyaway_" + side, path, 0.00065,
                       hair, collection, side)
        line["hair_region"] = "scalp"
        objects.append(line)

    for obj in objects:
        obj["rig_attachment"] = (
            "hair_" + obj.get("side", "C")
            if obj.get("hair_region") == "ponytail" else "head")
    return objects
