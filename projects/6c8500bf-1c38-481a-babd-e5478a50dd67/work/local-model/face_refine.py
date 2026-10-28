"""Reversible profile-aware neutral head refinement on the original bound mesh.

Only the anterior Y field is rebuilt. X, Z, UVs, vertex order, skin bindings,
shape-key policy and material assignments are retained. No expression or eye
tracking data are added. apply() is deterministic from its first saved baseline.
"""
import math
import bpy
from mathutils import Vector

VERSION = "continuous-neutral-profile-02"
ORIGINAL = "FaceRefineOriginal"
SECTORS = 112

# z, broad facial support surface at the centre, final centre-line silhouette.
# The broad support carries maxilla, lips and chin. The separate centre line
# gives the nasal bridge a controlled width rather than an isolated nose spike.
PROFILE = [
    (1.318, -.0640, -.0640), (1.325, -.0810, -.0810),
    (1.335, -.0895, -.0900), (1.345, -.0915, -.0920),
    (1.352, -.0925, -.0910), (1.359, -.0928, -.0965),
    (1.365, -.0930, -.0975), (1.370, -.0932, -.0985),
    (1.375, -.0930, -.0960), (1.380, -.0925, -.0960),
    (1.385, -.0915, -.1025), (1.394, -.0910, -.1080),
    (1.403, -.0910, -.1040), (1.413, -.0905, -.0970),
    (1.423, -.0890, -.0920), (1.434, -.0870, -.0885),
    (1.445, -.0875, -.0890), (1.455, -.0880, -.0895),
    (1.470, -.0865, -.0870), (1.485, -.0830, -.0832),
    (1.500, -.0780, -.0780), (1.530, -.0630, -.0630),
    (1.572,  .0030,  .0030),
]


def _smoothstep(a, b, value):
    value = max(0, min(1, (value-a)/(b-a)))
    return value*value*(3-2*value)


def _profile(z, column):
    """Shape-preserving cubic Hermite curve; extrema never overshoot."""
    if z <= PROFILE[0][0]:
        return PROFILE[0][column]
    if z >= PROFILE[-1][0]:
        return PROFILE[-1][column]
    slopes = [(b[column]-a[column])/(b[0]-a[0])
              for a, b in zip(PROFILE, PROFILE[1:])]
    tangents = [slopes[0]]
    for index in range(1, len(PROFILE)-1):
        left, right = slopes[index-1], slopes[index]
        if left*right <= 0:
            tangents.append(0.0)
        else:
            h0 = PROFILE[index][0]-PROFILE[index-1][0]
            h1 = PROFILE[index+1][0]-PROFILE[index][0]
            w1, w2 = 2*h1+h0, h1+2*h0
            tangents.append((w1+w2)/(w1/left+w2/right))
    tangents.append(slopes[-1])
    for index, (a, b) in enumerate(zip(PROFILE, PROFILE[1:])):
        if a[0] <= z <= b[0]:
            h = b[0]-a[0]
            t = (z-a[0])/h
            return ((2*t**3-3*t*t+1)*a[column]
                    +(t**3-2*t*t+t)*h*tangents[index]
                    +(-2*t**3+3*t*t)*b[column]
                    +(t**3-t*t)*h*tangents[index+1])


def _backup(obj):
    attr = obj.data.attributes.get(ORIGINAL)
    if attr is None:
        attr = obj.data.attributes.new(ORIGINAL, "FLOAT_VECTOR", "POINT")
        for vertex in obj.data.vertices:
            attr.data[vertex.index].vector = vertex.co
    if len(attr.data) != len(obj.data.vertices):
        raise RuntimeError("Face topology changed after refinement baseline.")
    return [item.vector.copy() for item in attr.data]


def _face_object(collection):
    if collection is None:
        collection = bpy.data.collections["CHARACTER · Original geometry"]
    return next(obj for obj in collection.all_objects
                if obj.type == "MESH" and obj.name.startswith("Face ·"))


def apply(collection=None, rig=None):
    face = _face_object(collection)
    if len(face.data.vertices) != 11312:
        raise RuntimeError("Expected original 101 × 112 face ring topology.")
    original = _backup(face)
    for row in range(101):
        ring = original[row*SECTORS:(row+1)*SECTORS]
        z = ring[0].z
        if max(abs(co.z-z) for co in ring) > 1e-5:
            raise RuntimeError("Face rings no longer have the expected ordering.")
        side_y = (ring[SECTORS//4].y + ring[3*SECTORS//4].y)/2
        broad, centre = _profile(z, 1), _profile(z, 2)
        # Fade forehead shaping underneath hair; preserve the entire crown.
        z_blend = 1-_smoothstep(1.490, 1.525, z)
        for sector, co in enumerate(ring):
            index = row*SECTORS+sector
            cosine = math.cos(math.tau*sector/SECTORS)
            if cosine <= 0 or z_blend <= 0:
                face.data.vertices[index].co = co
                continue
            x = co.x
            y = side_y-(side_y-broad)*cosine**.57
            # Broad nose root/bridge transitions into a rounded tip. At the
            # mouth the same field supports the lip, rather than pulling a
            # narrow central cylinder out of an otherwise receding upper jaw.
            nose_band = math.exp(-((z-1.405)/.032)**4)
            width = .030-.010*nose_band
            y += (centre-broad)*math.exp(-(x/width)**2)*cosine
            # Mild paired malar/infraorbital support, zero at the centre line.
            lateral = 1-math.exp(-(x/.023)**2)
            cheek = math.exp(-((abs(x)-.046)/.029)**2)
            y -= lateral*cheek*cosine**2*(
                .0030*math.exp(-((z-1.395)/.023)**2)
                +.0015*math.exp(-((z-1.419)/.012)**2))
            # The ala joins the rounded tip to the maxilla without a hole or
            # sharp nostril trench; source artwork remains the neutral paint.
            y -= .0015*math.exp(-((abs(x)-.014)/.0075)**2
                                -((z-1.387)/.007)**2)*cosine**2
            # Subtle continuous eyelid relief, not a recessed eyeball cavity.
            eye_x = (abs(x)-.0445)/.023
            if abs(eye_x) < 1.20:
                eye_mask = math.exp(-(eye_x/1.04)**8)
                arch = max(0, 1-eye_x*eye_x)
                upper = 1.430+.0125*arch
                lower = 1.429-.0065*arch
                y -= eye_mask*cosine**2*(
                    .00120*math.exp(-((z-upper)/.0035)**2)
                    +.00065*math.exp(-((z-lower)/.0038)**2))
            # Preserve the original lateral scalp/temple junction exactly.
            side_blend = _smoothstep(.025, .22, cosine)
            blend = z_blend*side_blend
            face.data.vertices[index].co = Vector((x, co.y+(y-co.y)*blend, z))
    face.data.update()
    face["face_refine_version"] = VERSION
    return {"version": VERSION, "mesh": face.name,
            "changed": "continuous anterior Y field",
            "preserved": ["X", "Z", "UVs", "weights", "topology",
                          "materials", "neutral gaze"]}


def restore(collection=None):
    face = _face_object(collection)
    attr = face.data.attributes.get(ORIGINAL)
    if attr is None:
        return {"restored": False, "reason": "No saved baseline"}
    if len(attr.data) != len(face.data.vertices):
        raise RuntimeError("Cannot restore changed face topology.")
    for vertex in face.data.vertices:
        vertex.co = attr.data[vertex.index].vector
    face.data.update()
    if "face_refine_version" in face:
        del face["face_refine_version"]
    return {"restored": True, "mesh": face.name}
