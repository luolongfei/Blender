"""Reproducible deformation review; restore the artist's pose after each render."""
from pathlib import Path
import importlib.util
import math
import bpy
from mathutils import Quaternion, Vector

HERE = Path(__file__).resolve().parent


def helper(name):
    spec = importlib.util.spec_from_file_location("review_" + name, HERE/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotate_world(rig, name, axis, degrees):
    bone = rig.pose.bones[name]
    basis = bone.bone.matrix_local.to_quaternion()
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = basis.inverted() @ Quaternion(Vector(axis), math.radians(degrees)) @ basis


def render_pose(style, resolution=1200):
    rig = bpy.data.objects["Character_Rig"]
    rb = helper("rig_builder")
    cb = helper("character_builder")
    transforms = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
    try:
        rb.pose(rig, "relaxed")
        if style == "arms":
            for sign, side in ((1, "left"), (-1, "right")):
                rotate_world(rig, side+"UpperArm", (0, 1, 0), sign*45)
                rotate_world(rig, side+"LowerArm", (0, 0, 1), -sign*70)
                for finger in ("Index", "Middle", "Ring", "Little"):
                    for segment, angle in (("Proximal", 25), ("Intermediate", 35), ("Distal", 20)):
                        rotate_world(rig, side+finger+segment, (0, 1, 0), sign*angle)
        elif style == "stride":
            rotate_world(rig, "leftUpperLeg", (1, 0, 0), -16)
            rotate_world(rig, "rightUpperLeg", (1, 0, 0), 16)
            rotate_world(rig, "leftLowerLeg", (1, 0, 0), 25)
            rotate_world(rig, "rightLowerLeg", (1, 0, 0), 8)
            rotate_world(rig, "leftLowerArm", (1, 0, 0), 10)
            rotate_world(rig, "rightLowerArm", (1, 0, 0), -10)
        elif style == "turn":
            rotate_world(rig, "chest", (0, 0, 1), 12)
            rotate_world(rig, "head", (0, 0, 1), -25)
            rotate_world(rig, "neck", (1, 0, 0), 6)
        else:
            raise ValueError("Unknown review pose")
        bpy.context.view_layer.update()
        return cb.render("three-quarter", "pose-"+style+".png", resolution, 24)
    finally:
        for name, basis in transforms.items():
            rig.pose.bones[name].matrix_basis = basis
        bpy.context.view_layer.update()
