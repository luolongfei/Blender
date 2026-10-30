"""Conservative, fully keyed E8 preview actions for an identity Blender armature.

Entry points:
    create_preview_actions(rig, meta=None, fps=30) -> JSON-safe manifest
    apply_preview_pose(rig, action_id, seconds, meta=None) -> pose diagnostics

The module never saves a project, changes mesh data, or runs an exporter.  All
rotations below are defined in model axes (X lateral, -Y forward, Z up), then
converted through each actual rest matrix.  No bone-roll assumption is needed.
"""

import math

from mathutils import Quaternion, Vector


ACTION_SPECS = (
    {"id": "idle", "clip": "idle", "label": "自然站立", "duration": 4.0, "loop": True},
    {"id": "wave", "clip": "wave", "label": "轻轻招手", "duration": 4.8, "loop": False},
    {"id": "look", "clip": "look", "label": "左右看看", "duration": 6.0, "loop": True},
    {"id": "walk", "clip": "walk", "label": "原地轻踏", "duration": 2.8, "loop": True},
    {"id": "tpose", "clip": "Tpose", "label": "还原 T 姿势", "duration": 1.0, "loop": False},
)

REQUIRED_BONES = (
    "Root", "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
)


def _q(axis, degrees):
    return Quaternion(Vector(axis), math.radians(degrees))


def _identity():
    return Quaternion((1.0, 0.0, 0.0, 0.0))


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _context(rig, meta=None):
    if rig.type != "ARMATURE":
        raise TypeError("Preview actions require a Blender armature object")
    missing = set(REQUIRED_BONES) - set(rig.pose.bones.keys())
    if missing:
        raise ValueError("Missing preview bones: " + ", ".join(sorted(missing)))
    # This rig is intentionally built in source model coordinates.  An object
    # transform would need explicit adaptation rather than silently wrong axes.
    ident_error = max(abs(rig.matrix_world[r][c] - float(r == c))
                      for r in range(4) for c in range(4))
    if ident_error > 1e-5:
        raise ValueError("Preview armature must have an identity object transform")

    def depth(b):
        n = 0
        while b.parent:
            n += 1
            b = b.parent
        return n

    bones = sorted(rig.data.bones, key=depth)
    return {
        "rig": rig,
        "bones": bones,
        "orientations": {b.name: b.matrix_local.to_quaternion().normalized() for b in bones},
        "meta": dict(meta or {}),
    }


def _walking_leg(ctx, side, phase, hips_delta):
    """Solve a short sagittal step with a fixed stance ankle and flat foot.

    There is no root travel.  For the first half-cycle the ankle is fixed at its
    rest position; during the other half it moves up/forward and returns with
    zero velocity at touchdown.  Foot rotation cancels the thigh/calf rotation.
    The source rig's small lateral knee/ankle offsets are preserved.
    """
    rig = ctx["rig"]
    thigh = rig.data.bones[side + "UpperLeg"]
    calf = rig.data.bones[side + "LowerLeg"]
    foot = rig.data.bones[side + "Foot"]
    hip = thigh.head_local.copy() + hips_delta
    target = foot.head_local.copy()
    swing_u = max(0.0, min(1.0, (phase - 0.5) * 2.0))
    lift_shape = math.sin(math.pi * swing_u) ** 2 if phase >= 0.5 else 0.0
    # Model height is approximately 2 units.  Deliberately small clearances
    # keep this preview within the static skirt's deformation envelope.
    target.y -= 0.010 * lift_shape
    target.z += 0.010 * lift_shape

    upper_rest = thigh.tail_local - thigh.head_local
    lower_rest = calf.tail_local - calf.head_local
    l1 = math.hypot(upper_rest.y, upper_rest.z)
    l2 = math.hypot(lower_rest.y, lower_rest.z)
    dy, dz = target.y - hip.y, target.z - hip.z
    distance = math.hypot(dy, dz)
    if distance >= l1 + l2 or distance <= abs(l1 - l2):
        raise ValueError("Unreachable ankle target for " + side)
    along = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance)
    height = math.sqrt(max(0.0, l1 * l1 - along * along))
    # (dz, -dy) selects the knee pole toward the model front, negative Y.
    knee_y = hip.y + dy * along / distance + dz * height / distance
    knee_z = hip.z + dz * along / distance - dy * height / distance
    desired_upper = math.atan2(knee_y - hip.y, -(knee_z - hip.z))
    desired_lower = math.atan2(target.y - knee_y, -(target.z - knee_z))
    upper_angle = desired_upper - math.atan2(upper_rest.y, -upper_rest.z)
    lower_angle = (desired_lower - math.atan2(lower_rest.y, -lower_rest.z)
                   - upper_angle)
    angles = tuple(math.degrees(a) for a in (upper_angle, lower_angle))
    return {
        side + "UpperLeg": _q((1, 0, 0), angles[0]),
        side + "LowerLeg": _q((1, 0, 0), angles[1]),
        side + "Foot": _q((1, 0, 0), -sum(angles)),
        side + "Toes": _identity(),
    }, {
        "ankleTarget": list(target), "stance": phase < 0.5,
        "lift": 0.010 * lift_shape,
        "upperLegDegrees": angles[0], "kneeDegrees": angles[1],
    }


def _pose(ctx, action_id, seconds):
    normalized_id = action_id.lower()
    spec = next((s for s in ACTION_SPECS if s["id"] == normalized_id), None)
    if spec is None:
        raise ValueError("Unknown preview action: " + str(action_id))
    duration = spec["duration"]
    t = max(0.0, min(float(seconds), duration))
    u = t / duration
    rotations = {b.name: _identity() for b in ctx["bones"]}
    locations = {b.name: Vector((0.0, 0.0, 0.0)) for b in ctx["bones"]}
    diagnostics = {"action": normalized_id, "seconds": t, "legs": {}}

    if normalized_id != "tpose":
        # The base pose is repeated in every clip, including both arms.  Local
        # Euler axes are NOT interchangeable across these differently oriented
        # bones; the model-axis deltas are converted below through rest matrices.
        rotations["LeftUpperArm"] = _q((0, 1, 0), 65.0)
        rotations["RightUpperArm"] = _q((0, 1, 0), -65.0)
        rotations["LeftLowerArm"] = _q((0, 0, 1), -6.0)
        rotations["RightLowerArm"] = _q((0, 0, 1), 6.0)
        rotations["LeftHand"] = _q((0, 0, 1), 2.0)
        rotations["RightHand"] = _q((0, 0, 1), -2.0)

        breath = 0.5 - 0.5 * math.cos(2.0 * math.pi * u)
        rotations["Chest"] = _q((1, 0, 0), 0.28 * breath)
        rotations["Head"] = _q((1, 0, 0), -0.16 * breath)
        locations["Spine"].z = 0.0007 * breath

    if normalized_id == "wave":
        envelope = _smoothstep(t / 0.85) * _smoothstep((duration - t) / 0.85)
        rotations["RightUpperArm"] = _q((0, 1, 0), -65.0 + 30.0 * envelope)
        # Raise the forearm by a real elbow flexion; distribute pronation along
        # the forearm instead of twisting only the wrist.  The modest twist is
        # a conservative inferred hand pose, not a recovered reference gesture.
        neutral = rotations["RightLowerArm"]
        raised = _q((0, 1, 0), 115.0) @ _q((1, 0, 0), -60.0)
        rotations["RightLowerArm"] = neutral.slerp(raised, envelope)
        rotations["Head"] = rotations["Head"] @ _q((0, 0, 1), -1.5 * envelope)

        # Wave about the true model Y axis after inheriting the raised forearm.
        # Convert this extra world-axis delta into parent-deformed coordinates.
        parent_deformation = _identity()
        for name in ("Root", "Hips", "Spine", "Chest", "RightShoulder",
                     "RightUpperArm", "RightLowerArm"):
            parent_deformation = parent_deformation @ rotations[name]
        oscillation = 7.0 * envelope * math.sin(2.0 * math.pi * 1.4 * (t - 0.85))
        extra = _q((0, 1, 0), oscillation)
        rotations["RightHand"] = (parent_deformation.inverted() @ extra
                                   @ parent_deformation @ rotations["RightHand"])

    elif normalized_id == "look":
        # Total yaw stays under ten degrees because the two ponytails are rigid
        # parts of Head in this preview; no collision/spring system is claimed.
        ease = math.sin(math.pi * u) ** 2
        turn = math.sin(2.0 * math.pi * u) * ease
        rotations["Neck"] = _q((0, 0, 1), 2.0 * turn)
        rotations["Head"] = (_q((0, 0, 1), 8.0 * turn)
                              @ _q((1, 0, 0), 1.0 * math.sin(4 * math.pi * u) * ease))

    elif normalized_id == "walk":
        # A tiny consistent pelvis drop gives the initially near-straight legs
        # room to bend.  Analytic ankle targets compensate this drop, so planted
        # feet stay on their original plane instead of following a pelvis bob.
        hips_delta = Vector((0.0, 0.0, -0.0015))
        locations["Hips"] = hips_delta
        for side, offset in (("Left", 0.0), ("Right", 0.5)):
            phase = (u + offset) % 1.0
            leg_rotations, leg_info = _walking_leg(ctx, side, phase, hips_delta)
            rotations.update(leg_rotations)
            diagnostics["legs"][side] = leg_info
        swing = 1.8 * math.sin(2.0 * math.pi * u)
        rotations["LeftUpperArm"] = _q((0, 1, 0), 65.0) @ _q((0, 0, 1), swing)
        rotations["RightUpperArm"] = _q((0, 1, 0), -65.0) @ _q((0, 0, 1), swing)

    rig = ctx["rig"]
    for bone in ctx["bones"]:
        pb = rig.pose.bones[bone.name]
        orientation = ctx["orientations"][bone.name]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (orientation.inverted() @ rotations[bone.name]
                                  @ orientation).normalized()
        pb.location = orientation.inverted() @ locations[bone.name]
        pb.scale = (1.0, 1.0, 1.0)
    rig.update_tag()
    return diagnostics


def apply_preview_pose(rig, action_id, seconds=0.0, meta=None):
    """Set a complete pose without editing any animation or scene frame.

    Call view_layer.update() before inspecting evaluated mesh/pose matrices.
    If an action is active, do not change the scene frame afterward unless you
    intend its existing animation to replace this temporary inspection pose.
    """
    return _pose(_context(rig, meta), action_id, seconds)


def _curves(action):
    """Support both older actions and Blender 4.4+/5.x layered actions."""
    seen = set()
    try:
        legacy = action.fcurves
    except AttributeError:
        legacy = ()
    for fc in legacy:
        pointer = fc.as_pointer()
        if pointer not in seen:
            seen.add(pointer)
            yield fc
    for layer in getattr(action, "layers", ()):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", ()):
                for fc in bag.fcurves:
                    pointer = fc.as_pointer()
                    if pointer not in seen:
                        seen.add(pointer)
                        yield fc


def create_preview_actions(rig, meta=None, fps=30):
    """Create five exportable actions and return the viewer's action manifest.

    Existing same-name actions are rejected to avoid silent '.001' suffixes in
    the web contract.  Invoke on the clean rigged-rest copy.  Sampling at 30 fps
    and linear interpolation prevent Bezier overshoot around foot touchdown.
    All bones have complete quaternion/location/scale tracks in every clip.
    """
    import bpy

    fps = int(fps)
    if fps < 24:
        raise ValueError("Use at least 24 fps for preview motion sampling")
    occupied = [s["clip"] for s in ACTION_SPECS if bpy.data.actions.get(s["clip"])]
    if occupied:
        raise ValueError("Existing preview actions must be handled by integrator: "
                         + ", ".join(occupied))
    ctx = _context(rig, meta)
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0
    rig.animation_data_create()
    created = []
    extrema = {"upperLegAbsDegrees": 0.0, "kneeDegrees": [float("inf"), -float("inf")],
               "maxFootLift": 0.0}
    for spec in ACTION_SPECS:
        action = bpy.data.actions.new(spec["clip"])
        action.use_fake_user = True
        rig.animation_data.action = action
        total_frames = round(spec["duration"] * fps)
        previous_quaternions = {}
        for sample in range(total_frames + 1):
            frame = 1 + sample
            info = _pose(ctx, spec["id"], sample / fps)
            for leg in info["legs"].values():
                extrema["upperLegAbsDegrees"] = max(extrema["upperLegAbsDegrees"],
                                                     abs(leg["upperLegDegrees"]))
                extrema["kneeDegrees"][0] = min(extrema["kneeDegrees"][0], leg["kneeDegrees"])
                extrema["kneeDegrees"][1] = max(extrema["kneeDegrees"][1], leg["kneeDegrees"])
                extrema["maxFootLift"] = max(extrema["maxFootLift"], leg["lift"])
            for bone in ctx["bones"]:
                pb = rig.pose.bones[bone.name]
                q = pb.rotation_quaternion.copy()
                previous = previous_quaternions.get(bone.name)
                if previous is not None and q.dot(previous) < 0.0:
                    q.negate()
                    pb.rotation_quaternion = q
                previous_quaternions[bone.name] = q
                for data_path in ("rotation_quaternion", "location", "scale"):
                    pb.keyframe_insert(data_path=data_path, frame=frame, group=bone.name)
        for curve in _curves(action):
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"
        action["preview_id"] = spec["id"]
        action["preview_loop"] = spec["loop"]
        action["preview_source"] = "E8 static rebuild / conservative preview rig"
        row = dict(spec)
        row.update({"fps": fps, "firstFrame": 1, "lastFrame": total_frames + 1,
                    "keyedBones": len(ctx["bones"]), "completePoseTracks": True})
        created.append(row)

    # Leave the saved source and default export pose unambiguously at bind T.
    rig.animation_data.action = bpy.data.actions["Tpose"]
    scene.frame_start = 1
    scene.frame_end = max(row["lastFrame"] for row in created)
    scene.frame_set(1)
    _pose(ctx, "tpose", 0.0)
    bpy.context.view_layer.update()
    return {
        "actions": created,
        "defaultAction": "idle",
        "resetAction": "tpose",
        "fps": fps,
        "coordinateSystem": "X lateral / -Y forward / Z up",
        "motionLimits": extrema,
        "notes": [
            "Every action keys every bone rotation, translation, and scale.",
            "Walk is conservative alternating stepping in place with analytic ankle targets; no root travel.",
            "Head and twin ponytails move together; no simulated hair or cloth collision.",
            "Face is static: no expression morph targets or inferred facial animation.",
            "Wave forearm pronation is a conservative inferred preview gesture.",
        ],
    }
