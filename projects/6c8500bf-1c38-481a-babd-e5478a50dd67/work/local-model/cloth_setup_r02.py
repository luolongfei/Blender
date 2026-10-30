"""Reproducible skirt cloth experiment on a saved working copy; never the master.

Usage in Blender Python (import through runpy or importlib):
    report = run()                         # 61-frame frozen-body settling test
    preview(True)                          # show simulation, preserve originals
    bpy.ops.wm.save_as_mainfile(filepath=...independent_work_copy...)
    bake()                                # blocking; invoke separately, one job only

After a successful static review, make another copy and call
run(follow_rig=True, frame_start=1, frame_end=121) for an existing action that
starts in a suitable standing pose. No artificial wind or sine animation is
created. This does not claim a validated fabric, motion cache or collision pass.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "toolchain.lock.json").is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

PREFIX = "R02 Cloth "
CONTROL = PREFIX + "Skirt control"
COLLECTION = PREFIX + "Validation"
COLLIDERS = PREFIX + "Body colliders"
PIN = "R02 waist pins"
ORIGINALS = ("Skirt_Sixteen_Pleats", "Skirt_Turned_Hem_Stitch",
             "Skirt_Center_Back_Seam")
BODY = ("Body_Complete_Trunk_Arms_Pelvis", "Leg L · knee calf ankle",
        "Leg R · knee calf ankle", "Body_Complete_Shin_Foot_L",
        "Body_Complete_Shin_Foot_R")


def _enum(owner, prop, value):
    valid = {item.identifier for item in owner.bl_rna.properties[prop].enum_items}
    if value not in valid:
        raise RuntimeError(f"{prop}: {value} unavailable; choices={sorted(valid)}")
    setattr(owner, prop, value)


def _guard():
    project = project_path()
    if not bpy.data.filepath:
        raise RuntimeError("Save an independent working copy first")
    file = Path(bpy.data.filepath).resolve()
    if file == (project / "work/character.blend").resolve():
        raise RuntimeError("Cloth experiment must not modify the working master")
    try:
        file.relative_to((project / "work").resolve())
    except ValueError:
        raise RuntimeError("Work copy must be inside this project's work directory")
    if bpy.context.mode != "OBJECT":
        raise RuntimeError("Return to Object Mode before cloth setup")
    return project, file


def _collection(name, parent):
    data = bpy.data.collections.new(name)
    parent.children.link(data)
    return data


def _morph_vertices(source):
    """Evaluate shape keys only. Existing skirt bone animation is not inherited."""
    temp = source.copy()
    temp.data = source.data.copy()
    temp.name = PREFIX + "Temporary morph evaluation"
    bpy.context.scene.collection.objects.link(temp)
    temp.parent = None
    temp.matrix_world = source.matrix_world.copy()
    temp.modifiers.clear()
    temp.hide_set(False)
    try:
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = temp.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            return [temp.matrix_world @ v.co for v in mesh.vertices]
        finally:
            evaluated.to_mesh_clear()
    finally:
        mesh_data = temp.data
        bpy.data.objects.remove(temp, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)


def _hip_transform(rig):
    bone = rig.data.bones.get("hips")
    pose = rig.pose.bones.get("hips")
    if bone is None or pose is None:
        raise RuntimeError("The retained hips bone is required")
    if rig.data.pose_position == "REST":
        return Matrix.Identity(4)
    return (rig.matrix_world @ pose.matrix @ bone.matrix_local.inverted()
            @ rig.matrix_world.inverted())


def _skirt_material():
    # The render/export skirt may use a shared clothing atlas. Its UV islands
    # are not the circumference/height coordinates on this reduced control
    # surface, so copying that material silently produces black cloth.
    material = bpy.data.materials.get("Character / skirt")
    if material is None:
        raise RuntimeError("Dedicated green 'Character / skirt' material missing; inspect materials instead of reusing the clothing atlas")
    return material


def _copy_collider(source, collection, follow_rig):
    if follow_rig:
        obj = source.copy()
        obj.data = source.data.copy()
        # Keep the authored rig deformation, discard inherited simulations.
        for modifier in list(obj.modifiers):
            if modifier.type in {"CLOTH", "COLLISION", "PARTICLE_SYSTEM"}:
                obj.modifiers.remove(modifier)
            elif modifier.type == "SUBSURF":
                modifier.show_viewport = False
                modifier.show_render = False
    else:
        graph = bpy.context.evaluated_depsgraph_get()
        evaluated = source.evaluated_get(graph)
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=graph)
        obj = bpy.data.objects.new(PREFIX + source.name, mesh)
        obj.matrix_world = source.matrix_world.copy()
    obj.name = PREFIX + source.name
    collection.objects.link(obj)
    obj.hide_render = True
    obj.hide_viewport = False
    obj.hide_set(False)
    _enum(obj, "display_type", "WIRE")
    obj.modifiers.new(PREFIX + "Collision", "COLLISION")
    obj.collision.thickness_outer = .002
    obj.collision.thickness_inner = .001
    obj.collision.cloth_friction = 5.0
    obj.collision.damping = .1
    obj.collision.use_culling = False
    obj["r02_collision_source"] = source.name
    return obj


def run(frame_start=1, frame_end=61, follow_rig=False):
    """Set up only. No saving, simulation, baking, action edits or master writes."""
    project, file = _guard()
    if bpy.data.collections.get(COLLECTION) or bpy.data.objects.get(CONTROL):
        raise RuntimeError("R02 cloth setup exists; preserve it or use another work copy")
    if not isinstance(frame_start, int) or not isinstance(frame_end, int):
        raise ValueError("Frame bounds must be integers")
    if not 1 <= frame_end - frame_start <= 360:
        raise ValueError("Use a bounded pass of 2–361 frames; 61-frame CPU trial was measured first")
    scene = bpy.context.scene
    if abs(scene.unit_settings.scale_length - 1.0) > 1e-6:
        raise RuntimeError("The verified character uses metres at unit scale 1")
    if abs(scene.gravity.z + 9.81) > .02 or abs(scene.gravity.x) > .001 or abs(scene.gravity.y) > .001:
        raise RuntimeError("Restore standard downward gravity before this experiment")
    if not scene.use_gravity:
        raise RuntimeError("Enable gravity before this experiment")
    source = bpy.data.objects.get(ORIGINALS[0])
    rig = bpy.data.objects.get("Character_Rig")
    if source is None or source.type != "MESH" or rig is None:
        raise RuntimeError("Expected retained skirt mesh and Character_Rig")
    missing = [name for name in BODY if bpy.data.objects.get(name) is None]
    if missing:
        raise RuntimeError("Missing body colliders: " + repr(missing))
    # Original garment construction is 256 columns x 31 rows. Sampling is
    # allowed only when that topology and row order are still present.
    if len(source.data.vertices) != 256 * 31:
        raise RuntimeError("Skirt topology changed; inspect and redesign the control sampling")
    material = _skirt_material()
    scene.frame_set(frame_start)
    points = _morph_vertices(source)
    for row in range(31):
        ring = points[row * 256:(row + 1) * 256]
        if max(v.z for v in ring) - min(v.z for v in ring) > .016:
            raise RuntimeError("Skirt vertex row order differs from the inspected construction")
    # Every original fold corner (column mod 16 = 7,9,10) survives. Broad flat
    # panels need fewer samples, unlike a uniform decimation of knife pleats.
    columns = [16 * panel + i for panel in range(16)
               for i in (0, 3, 6, 7, 8, 9, 10, 13)]
    rows = [0, 1, 2] + list(range(4, 31, 2))
    vertices = [points[row * 256 + col] for row in rows for col in columns]
    if not follow_rig:
        hip = _hip_transform(rig)
        vertices = [hip @ p for p in vertices]
    count = len(columns)
    faces = []
    for j in range(len(rows) - 1):
        for i in range(count):
            nxt = (i + 1) % count
            # Match the outward winding of the original garment.
            faces.append((j * count + i, (j + 1) * count + i,
                          (j + 1) * count + nxt, j * count + nxt))
    work = _collection(COLLECTION, scene.collection)
    colliders = _collection(COLLIDERS, work)
    mesh = bpy.data.meshes.new(CONTROL)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    control = bpy.data.objects.new(CONTROL, mesh)
    work.objects.link(control)
    mesh.materials.append(material)
    for face in mesh.polygons:
        face.use_smooth = True
    # UVs preserve the original circumference/height parameterization.
    uv = mesh.uv_layers.new(name="UVMap")
    for face in mesh.polygons:
        values = [(columns[mesh.loops[li].vertex_index % count] / 256,
                   rows[mesh.loops[li].vertex_index // count] / 30)
                  for li in face.loop_indices]
        wrap = max(v[0] for v in values) - min(v[0] for v in values) > .5
        for li, (u, v) in zip(face.loop_indices, values):
            uv.data[li].uv = (u + (1 if wrap and u < .5 else 0), v)
    pin = control.vertex_groups.new(name=PIN)
    for row_index, original_row in enumerate(rows):
        # Full waistband attachment, then a short soft transition. Most cloth
        # is free and receives no old skirt spring-bone weights.
        weight = {0: 1.0, 1: .75, 2: .2}.get(original_row, 0.0)
        if weight:
            pin.add(list(range(row_index * count, (row_index + 1) * count)), weight, "REPLACE")
    if follow_rig:
        control.vertex_groups.new(name="hips").add(list(range(len(vertices))), 1.0, "REPLACE")
        armature = control.modifiers.new(PREFIX + "Waist target", "ARMATURE")
        armature.object = rig
        armature.use_deform_preserve_volume = False
    collider_objects = [_copy_collider(bpy.data.objects[name], colliders, follow_rig)
                        for name in BODY]
    cloth = control.modifiers.new(PREFIX + "Cotton cloth", "CLOTH")
    settings = cloth.settings
    _enum(settings, "bending_model", "ANGULAR")
    # Blender's installed cotton.py preset, with quality raised from 5 to 8.
    # These are solver controls, not lab-measured SI material calibration.
    params = {"quality": 8, "mass": .3, "tension_stiffness": 15.0,
              "compression_stiffness": 15.0, "shear_stiffness": 15.0,
              "bending_stiffness": .5, "tension_damping": 5.0,
              "compression_damping": 5.0, "shear_damping": 5.0,
              "bending_damping": .5, "air_damping": 1.0,
              "time_scale": 1.0, "pin_stiffness": 1.0,
              "use_dynamic_mesh": False, "use_sewing_springs": False,
              "use_internal_springs": False, "use_pressure": False}
    for name, value in params.items():
        setattr(settings, name, value)
    settings.vertex_group_mass = PIN
    collision = cloth.collision_settings
    collision.collection = colliders
    collision.use_collision = True
    collision.collision_quality = 4
    collision.distance_min = .001
    collision.use_self_collision = True
    collision.self_distance_min = .001
    collision.self_friction = 5.0
    collision.friction = 5.0
    # Thickness is downstream of physics; it cannot inflate the simulated mesh.
    smooth = control.modifiers.new(PREFIX + "Render subdivision", "SUBSURF")
    smooth.levels = 0
    smooth.render_levels = 1
    shell = control.modifiers.new(PREFIX + "Cotton thickness", "SOLIDIFY")
    shell.thickness = .0012
    shell.offset = 0.0
    cache = cloth.point_cache
    cache.frame_start, cache.frame_end, cache.frame_step = frame_start, frame_end, 1
    cache.use_disk_cache = True
    cache.use_external = False
    cache.name = "R02_Skirt_Cotton"
    control.hide_render = True
    _enum(control, "display_type", "WIRE")
    report = {
        "method": "Blender Cloth, cotton preset baseline, bounded verification copy",
        "source": str(file.relative_to(project)),
        "source_saved_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "blender": bpy.app.version_string,
        "control_vertices": len(vertices), "control_quads": len(faces),
        "sample_rows": rows, "sample_columns": columns,
        "frame_start": frame_start, "frame_end": frame_end,
        "fps": scene.render.fps / scene.render.fps_base,
        "gravity": list(scene.gravity),
        "rig_action": rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None,
        "rig_pose_position": rig.data.pose_position,
        "control_geometry_sha256": hashlib.sha256(json.dumps(
            [list(v) for v in vertices], separators=(",", ":")).encode()).hexdigest(),
        "follow_rig": follow_rig, "parameters": params,
        "material": material.name,
        "colliders": [{"source": name, "object": obj.name,
                       "vertices": len(obj.data.vertices)}
                      for name, obj in zip(BODY, collider_objects)],
        "original_visibility": {name: {"render": bpy.data.objects[name].hide_render,
                                       "viewport": bpy.data.objects[name].hide_get()}
                                for name in ORIGINALS},
        "cache": "Internal disk point cache beside the saved work copy; save before bake",
        "status": "Configured only; unbaked and visually unverified",
        "limitations": ["Body has overlapping components; inspect hip/thigh collision seams",
                        "Control surface is a simulation preview, not full-resolution final tailoring",
                        "No sleeve simulation, no automatic seam transfer, no VRM cloth export",
                        "Animated mode needs an action with lead-in and renewed contact review"],
    }
    control["r02_cloth_report"] = json.dumps(report)
    print(json.dumps(report, ensure_ascii=False))
    return report


def preview(enabled=True):
    """Switch only this work copy's skirt visibility, reversibly."""
    _guard()
    control = bpy.data.objects[CONTROL]
    report = json.loads(control["r02_cloth_report"])
    for name, flags in report["original_visibility"].items():
        obj = bpy.data.objects[name]
        obj.hide_render = True if enabled else flags["render"]
        obj.hide_set(True if enabled else flags["viewport"])
    control.hide_render = not enabled
    _enum(control, "display_type", "TEXTURED" if enabled else "WIRE")


def bake():
    """Blocking, explicit bake of this cloth only. Save the copy before calling."""
    project, file = _guard()
    if bpy.data.is_dirty:
        raise RuntimeError("Save the independent work copy before baking its disk cache")
    control = bpy.data.objects[CONTROL]
    cloth = next(m for m in control.modifiers if m.type == "CLOTH")
    cache = cloth.point_cache
    if cache.is_baked:
        raise RuntimeError("A cache already exists; keep it or explicitly free it before rebaking")
    scene = bpy.context.scene
    scene.frame_set(cache.frame_start)
    with bpy.context.temp_override(scene=scene, object=control,
                                   active_object=control, point_cache=cache):
        result = bpy.ops.ptcache.bake(bake=True)
    report = json.loads(control["r02_cloth_report"])
    report["bake_result"] = sorted(result)
    report["cache_is_baked"] = bool(cache.is_baked)
    report["cache_is_outdated"] = bool(cache.is_outdated)
    report["cache_info"] = cache.info
    report["status"] = "Baked; visual and collision acceptance pending" if cache.is_baked else "Bake incomplete"
    # Record actual on-disk cache files rather than assuming the custom cache
    # name controls Blender's blendcache_<blend-stem> directory convention.
    cache_directory = file.parent / ("blendcache_" + file.stem)
    cache_files = list(cache_directory.glob("*"))
    report["expected_disk_cache_directory"] = str(cache_directory.relative_to(project))
    report["disk_cache_files_observed"] = [str(p.relative_to(project)) for p in cache_files if p.is_file()]
    out = file.parent / "cloth-r02-bake-report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    control["r02_cloth_report"] = json.dumps(report)
    print(json.dumps({"report": str(out), "baked": cache.is_baked,
                      "outdated": cache.is_outdated, "files": len(cache_files)}))
    return report
