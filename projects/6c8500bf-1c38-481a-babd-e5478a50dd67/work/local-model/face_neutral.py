"""Apply the user-requested neutral, no-blush facial texture.

The approved source and prior baked image remain unchanged. The replacement
projection was edited with image_gen and is baked through the existing UVs.
"""
from pathlib import Path
import importlib.util
import json
import bpy

ROOT = Path(__file__).resolve().parents[2]


def skin_projection_mask(face):
    """Keep painted features on the face, excluding painted side-hair remnants."""
    # The approved frontal image places the outer eyes inside 73 mm of the
    # midline. Beneath the eyes the painted hair comes further toward the cheeks.
    # These boundaries were chosen from the actual neutral projection and model.
    profile = [
        (1.318, .018), (1.340, .036), (1.360, .052),
        (1.380, .055), (1.400, .058), (1.410, .065),
        (1.422, .077), (1.450, .081), (1.475, .080),
        (1.572, .080),
    ]
    attr = face.data.attributes["FaceProjection"]
    backup = face.data.attributes.get("FaceProjectionBeforeNeutral")
    if backup is None:
        backup = face.data.attributes.new("FaceProjectionBeforeNeutral", "FLOAT", "POINT")
        for original, target in zip(attr.data, backup.data):
            target.value = original.value
    def smooth(a, b, value):
        t = min(1.0, max(0.0, (value-a)/(b-a)))
        return t*t*(3-2*t)
    changed = 0
    for vertex, original, target in zip(face.data.vertices, backup.data, attr.data):
        z = vertex.co.z
        width = profile[-1][1]
        for a, b in zip(profile, profile[1:]):
            if z <= b[0]:
                t = smooth(a[0], b[0], z)
                width = a[1]*(1-t) + b[1]*t
                break
        taper = 1 - smooth(width-.006, width+.007, abs(vertex.co.x))
        # The drawing's chin outline belongs to the silhouette, not the skin.
        # Fade below the mouth so that the 3D jaw has no painted dark notch.
        jaw_gate = smooth(1.328, 1.347, z)
        value = original.value * taper * jaw_gate
        changed += abs(value-original.value) > 1e-6
        target.value = value
    return changed


def apply():
    face = bpy.data.objects["Face · sculpted planes & original illustration"]
    if face.get("neutral_face_texture_version") == 3:
        return {"alreadyApplied": True, "texture": face["face_bake_texture"]}
    spec = importlib.util.spec_from_file_location("neutral_bake", Path(__file__).with_name("bake_face.py"))
    bake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bake)
    bake.restore_face_projection(face)
    source = face.data.materials[0].copy()
    source.name = "Character / neutral face projection · no blush"
    image = bpy.data.images.load(str(ROOT / "textures/face-neutral-projection.png"), check_existing=True)
    image.name = "Character · neutral projection · no blush"
    image.pack()
    targets = [node for node in source.node_tree.nodes
               if node.type == "TEX_IMAGE" and node.image]
    if len(targets) != 1:
        raise ValueError("Expected the retained single facial projection image.")
    targets[0].image = image
    face.data.materials[0] = source
    masked_vertices = skin_projection_mask(face)
    result = bake.bake_face(face, str(ROOT / "textures/face-neutral-basecolor.png"), 2048)
    result["sideHairProjectionMaskedVertices"] = masked_vertices
    face["neutral_face_texture_version"] = 3
    face["painted_features"] = "Fixed neutral gaze and natural skin, no default blush"
    (ROOT / "work/local-model/face-neutral-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
