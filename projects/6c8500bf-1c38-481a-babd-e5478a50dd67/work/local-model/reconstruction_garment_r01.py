"""R01 static skirt silhouette proposal, applied only to a working copy.

This is authored rest-shape art direction, NOT cloth simulation or a collision
pass. Original mesh topology, UVs, Basis, weights, modifiers and existing keys
are retained. The same continuous field is used for skirt, hem stitching and
back seam, with zero movement at the waist. Set the new key to 0 for comparison.
No scene change on import; call run(). This module never saves a .blend file.
"""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

KEY = "R01_static_drape_proposal"
TAG = "reconstruction_garment_r01"
VERSION = 1
TARGETS = (
    "Skirt_Sixteen_Pleats",
    "Skirt_Turned_Hem_Stitch",
    "Skirt_Center_Back_Seam",
)


def _clamp(value):
    return max(0.0, min(1.0, value))


def _smooth(value):
    t = _clamp(value)
    return t * t * (3.0 - 2.0 * t)


def _proposal(point):
    """Metres, +Z up; original construction waist .994 and hem .615."""
    x, y, z = point
    t = _clamp((.994 - z) / .379)
    # Preserve tucked waist and hip clearance, then lower the unsupported hem.
    release = _smooth((t - .12) / .88)
    # Reduce excess lateral flare 17%, depth only 6% to keep leg clearance.
    # Existing knife-pleat folds survive; no periodic fake wrinkles are added.
    x *= 1.0 - .17 * release
    y *= 1.0 - .06 * release
    # Small offset describes uneven cloth settling, not a leaning character.
    x += .0015 * release * release
    y += .0008 * release
    # Broad 2 mm hem variation breaks exact axial symmetry without rippling.
    theta = math.atan2(y / .145, x / .237)
    z -= .011 * release
    z += (.0014 * math.sin(theta + .35)
          + .0006 * math.sin(3.0 * theta - .7)) * release * release
    return Vector((x, y, z))


def _basis(obj):
    keys = obj.data.shape_keys
    return keys.reference_key.data if keys else obj.data.vertices


def _bounds(points):
    return [[min(p[i] for p in points), max(p[i] for p in points)]
            for i in range(3)]


def run(strength=1.0):
    """Create/refresh one reversible proposal; rerunning cannot compound it."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    file = Path(bpy.data.filepath)
    if not bpy.data.filepath or (file.name == "character.blend"
                                 and file.parent.name == "work"):
        raise RuntimeError("Save a named reconstruction work copy first")
    objects = []
    for name in TARGETS:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError("Expected existing mesh: " + name)
        if obj.data.shape_keys and not obj.data.shape_keys.use_relative:
            raise RuntimeError("Absolute shape keys require manual review: " + name)
        if obj.data.users != 1:
            raise RuntimeError("Shared mesh needs an explicit single-user copy: " + name)
        if obj.get(TAG) not in (None, VERSION):
            raise RuntimeError("A different garment revision is already installed")
        objects.append(obj)
    # Reject a shifted/scaled or otherwise unfamiliar skirt before any writes.
    main = objects[0]
    points = [main.matrix_world @ p.co for p in _basis(main)]
    bounds = _bounds(points)
    if not (.59 < bounds[2][0] < .64 and .98 < bounds[2][1] < 1.02
            and .42 < bounds[0][1] - bounds[0][0] < .51):
        raise RuntimeError("Skirt basis differs from the reviewed R01 geometry")
    rows = []
    for obj in objects:
        if obj.data.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        keys = obj.data.shape_keys
        basis = keys.reference_key
        key = keys.key_blocks.get(KEY)
        if key is None:
            key = obj.shape_key_add(name=KEY, from_mix=False)
        key.relative_key = basis
        key.slider_min, key.slider_max = 0.0, 1.0
        world = obj.matrix_world.copy()
        inverse = world.inverted()
        before, after, displacements = [], [], []
        for source, destination in zip(basis.data, key.data):
            start = world @ source.co
            end = _proposal(start)
            destination.co = inverse @ end
            before.append(start)
            after.append(end)
            displacements.append((end - start).length)
        key.value = strength
        key.mute = False
        obj[TAG] = VERSION
        obj["r01_drape_method"] = "Authored static rest-shape; not cloth simulation"
        rows.append({
            "object": obj.name,
            "vertices": len(basis.data),
            "basis_bounds_m": _bounds(before),
            "proposal_bounds_m": _bounds(after),
            "max_displacement_m": max(displacements),
            "shape_key": KEY,
            "value": strength,
        })
    bpy.context.view_layer.update()
    report = {
        "revision": VERSION,
        "method": "Reversible authored static silhouette shape key",
        "objects": rows,
        "preserved": ["Basis", "all existing keys", "topology", "UVs",
                      "materials", "weights", "modifiers", "waist position"],
        "not_modified": "Blouse, sleeves, waistband, body, rig and animation",
        "limitations": [
            "Not a simulated drape or cached physics result",
            "Visual front/side/back review and leg clearance still required",
            "Existing skirt bone motion needs renewed testing before integration",
            "Blouse already had exact armhole union/clearance edits; untouched pending visual inspection",
        ],
    }
    bpy.context.scene["r01_garment_report"] = json.dumps(report)
    print(json.dumps(report, ensure_ascii=False))
    return report
