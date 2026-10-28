"""Reversible lower-face proportion pass before facial topology authoring.

The original neutral geometry is kept in a point-vector attribute. The pass
rounds and shortens the chin, softens the lower jaw and protects the established
eye/mouth coordinates. It adds no expressions, materials, or facial components.
"""
import math
from pathlib import Path
import bpy
from mathutils import Vector

VERSION = "rounded-lower-face-03"
BACKUP = "FaceHarmonyOriginal"
SECTORS = 112

# Original ring height -> new half-width. The rapid but smooth radius increase
# just above the bottom creates a rounded chin rather than a pointed V.
WIDTH = [
    (1.31800, .0025), (1.32054, .0145), (1.32308, .0220),
    (1.32562, .0275), (1.32816, .0320), (1.33070, .0355),
    (1.33578, .0420), (1.34086, .0470), (1.35102, .0595),
    (1.36118, .0695), (1.37134, .0790), (1.38150, .0865),
    (1.39166, .0910), (1.40182, .0950), (1.41500, .0990),
]


def smoothstep(a, b, value):
    value = max(0.0, min(1.0, (value-a)/(b-a)))
    return value*value*(3-2*value)


def pchip(table, z):
    """Monotone Hermite interpolation; avoids overshooting the chin silhouette."""
    if z <= table[0][0]:
        return table[0][1]
    if z >= table[-1][0]:
        return table[-1][1]
    slopes = [(b[1]-a[1])/(b[0]-a[0]) for a, b in zip(table, table[1:])]
    tangents = [slopes[0]]
    for i in range(1, len(table)-1):
        left, right = slopes[i-1], slopes[i]
        if left*right <= 0:
            tangents.append(0.0)
        else:
            h0, h1 = table[i][0]-table[i-1][0], table[i+1][0]-table[i][0]
            w1, w2 = 2*h1+h0, h1+2*h0
            tangents.append((w1+w2)/(w1/left+w2/right))
    tangents.append(slopes[-1])
    for i, (a, b) in enumerate(zip(table, table[1:])):
        if a[0] <= z <= b[0]:
            h = b[0]-a[0]
            t = (z-a[0])/h
            return ((2*t**3-3*t*t+1)*a[1]
                    +(t**3-2*t*t+t)*h*tangents[i]
                    +(-2*t**3+3*t*t)*b[1]
                    +(t**3-t*t)*h*tangents[i+1])


def _face(face=None):
    if face is not None:
        return face
    return next(obj for obj in bpy.data.objects
                if obj.type == "MESH" and obj.name.startswith("Face ·"))


def _baseline(face):
    attr = face.data.attributes.get(BACKUP)
    if attr is None:
        attr = face.data.attributes.new(BACKUP, "FLOAT_VECTOR", "POINT")
        for vertex in face.data.vertices:
            attr.data[vertex.index].vector = vertex.co
    if len(attr.data) != len(face.data.vertices):
        raise RuntimeError("Cannot apply harmony after the face topology has changed.")
    return [item.vector.copy() for item in attr.data]


def _oral_protection(co, sector):
    cosine = math.cos(math.tau*sector/SECTORS)
    return ((1-smoothstep(.032, .055, abs(co.x)))
            *smoothstep(1.348, 1.355, co.z)
            *(1-smoothstep(1.378, 1.390, co.z))
            *smoothstep(.15, .45, cosine))


def _fair_lower_transition(face, original):
    """Remove small longwise curvature ripples around the protected oral field.

    This smooths the broad jaw transition over approximately four millimetres.
    The mouth core, eye region and lowest rounded chin rows remain fixed.
    """
    coords = [v.co.copy() for v in face.data.vertices]
    for iteration in range(12):
        updated = [co.copy() for co in coords]
        for row in range(1, 100):
            for sector in range(SECTORS):
                index = row*SECTORS+sector
                source = original[index]
                fade = (smoothstep(1.323, 1.333, source.z)
                        *(1-smoothstep(1.375, 1.397, source.z)))
                strength = .30*fade*(1-_oral_protection(source, sector))
                if strength <= 0:
                    continue
                longwise = (coords[index-SECTORS]+coords[index+SECTORS])*.5
                updated[index] = coords[index].lerp(longwise, strength)
        coords = updated
    for vertex, co in zip(face.data.vertices, coords):
        vertex.co = co


def apply(face=None):
    face = _face(face)
    if face.data.shape_keys:
        raise RuntimeError("Apply harmony before expression shape-key authoring.")
    if len(face.data.vertices) != 11312:
        raise RuntimeError("Harmony expects the original 101 × 112 head mesh.")
    original = _baseline(face)
    maximum = Vector((0, 0, 0))
    protected_count = 0
    for row in range(101):
        ring = original[row*SECTORS:(row+1)*SECTORS]
        z = ring[0].z
        if max(abs(co.z-z) for co in ring) > 1e-5:
            raise RuntimeError("Unexpected ring order: apply before facial retopology.")
        radius = max(abs(co.x) for co in ring)
        if z >= 1.415:
            for sector, co in enumerate(ring):
                face.data.vertices[row*SECTORS+sector].co = co
                protected_count += 1
            continue
        # Blend the lower-jaw target into the exact current upper cheek radius.
        new_radius = pchip(WIDTH, z)
        blend = 1-smoothstep(1.385, 1.415, z)
        radius_delta = (new_radius-radius)*blend
        # Shorten the lower third without moving the oral landmarks.
        raise_z = .0045*(1-smoothstep(1.318, 1.364, z))
        for sector, co in enumerate(ring):
            x, y = co.x, co.y
            cosine = math.cos(math.tau*sector/SECTORS)
            oral = _oral_protection(co, sector)
            movable = 1-oral
            dx = radius_delta*(x/radius)*movable if radius > 1e-7 else 0.0
            dz = raise_z*movable
            # A restrained posterior change softens the projecting chin; it
            # does not undo the previous maxillary and lip support correction.
            dy = (.0015*math.exp(-((z-1.339)/.014)**2)
                  *max(0, cosine)**2*movable)
            new = Vector((x+dx, y+dy, z+dz))
            face.data.vertices[row*SECTORS+sector].co = new
            for axis in range(3):
                maximum[axis] = max(maximum[axis], abs(new[axis]-co[axis]))
            if oral >= 1-1e-8:
                protected_count += 1
    _fair_lower_transition(face, original)
    maximum = Vector((0, 0, 0))
    for vertex, co in zip(face.data.vertices, original):
        for axis in range(3):
            maximum[axis] = max(maximum[axis], abs(vertex.co[axis]-co[axis]))
    face.data.update()
    face["face_harmony_version"] = VERSION
    return {
        "version": VERSION, "face": face.name,
        "max_axis_change_mm": [round(value*1000, 4) for value in maximum],
        "protected_vertices": protected_count,
        "chin_min_z": min(v.co.z for v in face.data.vertices),
        "preserved": ["UVs", "weights", "topology", "materials",
                      "mouth core coordinates", "all vertices at Z >= 1.415"],
    }


def restore(face=None):
    face = _face(face)
    attr = face.data.attributes.get(BACKUP)
    if attr is None:
        return False
    if len(attr.data) != len(face.data.vertices):
        raise RuntimeError("Cannot restore harmony after topology replacement.")
    for vertex in face.data.vertices:
        vertex.co = attr.data[vertex.index].vector
    face.data.update()
    if "face_harmony_version" in face:
        del face["face_harmony_version"]
    return True


def render_qa():
    """Run in a disposable background process; never saves the loaded master."""
    import hashlib
    import json
    project = Path(__file__).resolve().parents[2]
    out = project / "work/qa-face-harmony"
    out.mkdir(parents=True, exist_ok=True)
    face = _face()
    original = [tuple(vertex.co) for vertex in face.data.vertices]

    def invariant():
        data = {
            "uv": [[tuple(item.uv) for item in layer.data]
                   for layer in face.data.uv_layers],
            "weights": [[(g.group, g.weight) for g in vertex.groups]
                        for vertex in face.data.vertices],
            "polygons": [tuple(p.vertices) for p in face.data.polygons],
            "materials": [m.name for m in face.data.materials],
        }
        return hashlib.sha256(repr(data).encode()).hexdigest()

    before = invariant()
    report = apply(face)
    coords = [tuple(vertex.co) for vertex in face.data.vertices]
    apply(face)
    assert coords == [tuple(v.co) for v in face.data.vertices]
    assert invariant() == before
    assert all(tuple(v.co) == original[v.index]
               for v in face.data.vertices if original[v.index][2] >= 1.415)
    assert all(tuple(v.co) == original[v.index] for v in face.data.vertices
               if _oral_protection(Vector(original[v.index]),
                                   v.index % SECTORS) >= 1-1e-8)
    restore(face)
    assert original == [tuple(v.co) for v in face.data.vertices]
    apply(face)
    report["validation"] = {"idempotent": True, "restore_exact": True,
                            "uv_weights_topology_materials_preserved": True,
                            "upper_face_unchanged": True, "mouth_core_unchanged": True}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 20
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    camera_data = bpy.data.cameras.new("Harmony QA · orthographic")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = .342
    camera = bpy.data.objects.new("Harmony QA · orthographic", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    target = Vector((0, -.006, 1.441))
    views = {"front": Vector((0, -3, 1.441)),
             "front45": Vector((2.5, -2.506, 1.441)),
             "profile": Vector((3, -.006, 1.441))}
    visibility = {obj.name: obj.hide_render for obj in scene.objects}
    clay = bpy.data.materials.new("Harmony QA · clay")
    clay.use_nodes = True
    shader = next(n for n in clay.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (.48, .43, .38, 1)
    shader.inputs["Roughness"].default_value = .72
    shader.inputs["Specular IOR Level"].default_value = .12
    camera.location = views["front"]
    camera.rotation_euler = (target-camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "face-harmony-neutral.blend"))
    for style in ["textured", "clay", "before-textured"]:
        restore(face) if style.startswith("before") else apply(face)
        for obj in scene.objects:
            obj.hide_render = visibility[obj.name]
            if style == "clay" and obj.type == "MESH" and not obj.name.startswith(("Face ·", "Ear", "Neck")):
                obj.hide_render = True
        scene.view_layers[0].material_override = clay if style == "clay" else None
        for name, position in views.items():
            if style.startswith("before") and name != "front":
                continue
            camera.location = position
            camera.rotation_euler = (target-position).to_track_quat("-Z", "Y").to_euler()
            scene.render.filepath = str(out / f"{name}-{style}.png")
            bpy.ops.render.render(write_still=True)
            print("HARMONY_QA_RENDER", name, style, flush=True)
    print("HARMONY_QA_COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    import sys
    if "--harmony-qa" in sys.argv:
        render_qa()
