"""Local character VRM helpers for the installed VRM Add-on 4.10.2.

These helpers never create geometry, infer copyright ownership, replace the
canonical .charakit, or pretend local models came from Tripo.

Run inside Blender after the local model and explicit humanoid rig exist.
MToon conversion changes material node trees: render/save any separate PBR
look first, or pass duplicated export materials.
"""
import importlib.util
import math
from pathlib import Path
import re
import uuid

import bpy
from mathutils import Vector


_context_file = next(
    parent / "pipeline/context.py" for parent in Path(__file__).resolve().parents
    if (parent / "pipeline/context.py").is_file()
)
_context_spec = importlib.util.spec_from_file_location("blender_workspace_context", _context_file)
_context = importlib.util.module_from_spec(_context_spec)
_context_spec.loader.exec_module(_context)
ROOT = _context.tool_root()


def _module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enabled():
    api = _module("mtoon-export.py", "local_character_mtoon")
    api.enable_vrm()
    return api


def _snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def configure_vrm(armature, meshes, bone_map=None, *, convert_materials=False):
    """Map explicit VRM humanoid names to Blender bone names.

bone_map uses VRM keys such as "leftUpperArm", or their snake_case variants.
When omitted, only exact canonical/snake_case name matches are accepted.
No location-based or semantic guesses are made.
"""
    api = _enabled()
    if armature.type != "ARMATURE":
        raise TypeError("The supplied armature must be an ARMATURE object.")
    meshes = list(meshes)
    if any(o.type != "MESH" for o in meshes):
        raise TypeError("Pass actual mesh objects; convert curves before VRM export.")
    ext = armature.data.vrm_addon_extension
    ext.spec_version = "1.0"
    human = ext.vrm1.humanoid.human_bones
    human.initial_automatic_bone_assignment = False
    assigned = {}
    for key, target in human.human_bone_name_to_human_bone().items():
        vrm_name = key.value if hasattr(key, "value") else str(key)
        if bone_map is None:
            name = next((n for n in (vrm_name, _snake(vrm_name))
                         if n in armature.data.bones), None)
        else:
            name = bone_map.get(vrm_name, bone_map.get(_snake(vrm_name)))
        if name is None:
            continue
        if name not in armature.data.bones:
            raise ValueError(f"{vrm_name}: named Blender bone does not exist: {name}")
        target.node.bone_name = name
        assigned[vrm_name] = name
    type(human).update_all_bone_name_candidates(
        bpy.context, armature.data.name, force=True
    )
    bpy.context.view_layer.update()
    errors = list(human.error_messages())
    material_report = api.apply_mtoon(meshes) if convert_materials else None
    return {
        "assigned": assigned,
        "humanoidErrors": errors,
        "materials": material_report,
        "authorMetadataChanged": False,
    }


def add_colliders(armature, specs, group_name):
    """Create explicit sphere/capsule colliders and one referenced collider group.

Each spec: name, bone, center (rig-local XYZ), radius; optional tail for capsule.
Centers are transformed into the exact assigned bone's local coordinate frame.
Names must be unique. Repeated calls update existing named collider objects.
"""
    _enabled()
    if any(abs(v - 1) > 1e-6 for v in armature.scale):
        raise ValueError("Apply armature object scale before authoring colliders.")
    sb = armature.data.vrm_addon_extension.spring_bone1
    refs = []
    for spec in specs:
        name = spec["name"]
        bone_name = spec["bone"]
        bone = armature.data.bones.get(bone_name)
        if bone is None:
            raise ValueError(f"Collider {name} references missing bone {bone_name}")
        radius = float(spec["radius"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Collider radius must be finite and positive.")
        collider = next((c for c in sb.colliders
                         if c.bpy_object and c.bpy_object.name == name), None)
        if collider is None:
            collider = sb.colliders.add()
            collider.uuid = uuid.uuid4().hex
        collider.node.bone_name = bone_name
        capsule = "tail" in spec
        collider.shape_type = "Capsule" if capsule else "Sphere"
        collider.reset_bpy_object(bpy.context, armature)
        collider.bpy_object.name = name
        collider.bpy_object.hide_render = True
        shape = collider.shape.capsule if capsule else collider.shape.sphere
        shape.offset = bone.matrix_local.inverted() @ Vector(spec["center"])
        shape.radius = radius
        if capsule:
            shape.tail = bone.matrix_local.inverted() @ Vector(spec["tail"])
        refs.append(collider.uuid)
    group = next((g for g in sb.collider_groups if g.vrm_name == group_name), None)
    if group is None:
        group = sb.collider_groups.add()
        group.uuid = uuid.uuid4().hex
        group.vrm_name = group_name
    group.colliders.clear()
    for collider_uuid in refs:
        group.colliders.add().collider_uuid = collider_uuid
    bpy.context.view_layer.update()
    return {"group": group_name, "colliders": len(refs)}


def register_springs(armature, chains, targets):
    """Use the kit's audited terminal-joint/plan implementation.

chains entries contain name, bones, optional center, includesTerminal,
stiffness, gravityPower, gravityDir, dragForce, hitRadius, colliderGroups.
targets entries contain mesh, source and mode; mode=spring needs chains,
mode=fixed needs an explicit visual/structural reason.
Mesh weights and spring bones must already have been authored.
"""
    api = _module("spring-setup.py", "local_character_springs")
    return api.setup({
        "armature": armature.name,
        "chains": chains,
        "targets": targets,
    })


def audit_bindings(armature, meshes):
    """Report actual weight/material/UV structure without scoring appearance."""
    report = []
    for obj in meshes:
        deform_groups = {
            g.index for g in obj.vertex_groups
            if g.name in armature.data.bones and armature.data.bones[g.name].use_deform
        }
        sums = [
            sum(g.weight for g in v.groups if g.group in deform_groups)
            for v in obj.data.vertices
        ]
        modifiers = [m for m in obj.modifiers
                     if m.type == "ARMATURE" and m.object == armature]
        report.append({
            "mesh": obj.name,
            "vertices": len(obj.data.vertices),
            "faces": len(obj.data.polygons),
            "armatureModifiers": len(modifiers),
            "unweightedVertices": sum(s < 1e-6 for s in sums),
            "nonNormalizedVertices": sum(abs(s - 1) > 1e-4 for s in sums),
            "uvLayers": len(obj.data.uv_layers),
            "materialSlots": len(obj.material_slots),
        })
    return report


def export_review(armature, meshes, destination, *, convert_materials=True):
    """Export an explicitly marked preview VRM, restoring temporary metadata.

Mirrors scripts/export-review.py: "__CHARAKIT_REVIEW__" is a technical
placeholder, not an asserted author. This file is for review, not a licensed
final distribution. No Blend is saved here.
"""
    api = _enabled()
    meshes = list(meshes)
    human = armature.data.vrm_addon_extension.vrm1.humanoid.human_bones
    type(human).update_all_bone_name_candidates(
        bpy.context, armature.data.name, force=True
    )
    errors = list(human.error_messages())
    if errors:
        raise ValueError("Invalid humanoid mapping: " + "; ".join(errors))
    if convert_materials:
        api.apply_mtoon(meshes)
    meta = armature.data.vrm_addon_extension.vrm1.meta
    original_name = meta.vrm_name
    original_authors = [a.value for a in meta.authors]
    selection = list(bpy.context.selected_objects)
    active = bpy.context.view_layer.objects.active
    hidden = {obj: obj.hide_get() for obj in [armature, *meshes]}
    try:
        if not meta.vrm_name:
            meta.vrm_name = "Local character — motion review"
        if not meta.authors:
            meta.authors.add().value = "__CHARAKIT_REVIEW__"
        bpy.ops.object.select_all(action="DESELECT")
        for obj in [armature, *meshes]:
            obj.hide_set(False)
            obj.select_set(True)
        bpy.context.view_layer.objects.active = armature
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = bpy.ops.export_scene.vrm(
            filepath=str(destination),
            armature_object_name=armature.name,
            export_only_selections=True,
            export_invisibles=False,
            use_addon_preferences=False,
        )
        if "FINISHED" not in result:
            raise RuntimeError(f"VRM preview export did not finish: {result}")
        return api.inspect_vrm(destination)
    finally:
        meta.vrm_name = original_name
        meta.authors.clear()
        for author in original_authors:
            meta.authors.add().value = author
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selection:
            if obj.name in bpy.context.view_layer.objects:
                obj.select_set(True)
        bpy.context.view_layer.objects.active = active
        for obj, was_hidden in hidden.items():
            obj.hide_set(was_hidden)
