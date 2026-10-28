"""Save a curated, full-detail client Blend from a disposable background copy.

Run after the final relaxed-pose work/character.blend has been saved:
  Blender --background --disable-autoexec work/character.blend \
    --python-exit-code 1 --python work/local-model/client_source.py

Only the confirmed unrelated Blender startup scene is removed. Character
geometry, subdivision, PBR materials, pose, rig, springs, studio and hidden
references are retained. This does not export or finalize VRM permissions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

import bpy


PROJECT = Path(__file__).resolve().parents[2]
PRIMARY_SCENE = "双马尾绿裙 · Local Atelier"
CHARACTER_COLLECTION = "CHARACTER · Original geometry"
REFERENCE_COLLECTION = "REFERENCES · approved untouched inputs"
RIG_NAME = "Character_Rig"
CAMERA_NAME = "Camera · three-quarter"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_values(matrix):
    return [float(value) for row in matrix for value in row]


def preservation_state(scene, collection, rig):
    """In-memory invariant checks exclude the intentional presentation edits."""
    return {
        "posePosition": rig.data.pose_position,
        "rigTransform": matrix_values(rig.matrix_world),
        "pose": {bone.name: matrix_values(bone.matrix_basis)
                 for bone in rig.pose.bones},
        "springPlan": rig.get("charakit_spring_plan_v1"),
        "springChains": [{
            "name": spring.vrm_name, "center": spring.center.bone_name,
            "joints": [{
                "bone": joint.node.bone_name,
                "stiffness": joint.stiffness,
                "gravity": joint.gravity_power,
                "direction": list(joint.gravity_dir),
                "drag": joint.drag_force,
                "radius": joint.hit_radius,
            } for joint in spring.joints],
        } for spring in rig.data.vrm_addon_extension.spring_bone1.springs],
        "meshes": {obj.name: {
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
            "transform": matrix_values(obj.matrix_world),
            "materials": [mat.name if mat else None for mat in obj.data.materials],
            "modifiers": [{
                "name": modifier.name, "type": modifier.type,
                "viewport": modifier.show_viewport,
                "render": modifier.show_render,
                "levels": getattr(modifier, "levels", None),
                "renderLevels": getattr(modifier, "render_levels", None),
            } for modifier in obj.modifiers],
        } for obj in collection.all_objects if obj.type == "MESH"},
        "referenceVisibility": {
            obj.name: {
                "hideViewport": obj.hide_viewport,
                "hideRender": obj.hide_render,
                "hideInViewLayer": obj.hide_get(),
            } for obj in bpy.data.collections[REFERENCE_COLLECTION].all_objects
        } if REFERENCE_COLLECTION in bpy.data.collections else {},
        "referenceCollectionVisibility": {
            "viewport": bpy.data.collections[REFERENCE_COLLECTION].hide_viewport,
            "render": bpy.data.collections[REFERENCE_COLLECTION].hide_render,
        } if REFERENCE_COLLECTION in bpy.data.collections else {},
    }


def remove_confirmed_startup_scene(primary):
    """Never delete another authored scene or an object shared with it."""
    scene = bpy.data.scenes.get("Scene")
    if scene is None or scene == primary:
        return {"removedScene": None, "removedObjects": [], "removedCollections": []}
    expected = {"Cube": "MESH", "Camera": "CAMERA", "Light": "LIGHT"}
    if {obj.name: obj.type for obj in scene.objects} != expected:
        return {
            "removedScene": None, "removedObjects": [], "removedCollections": [],
            "preservedScene": scene.name,
            "reason": "Does not exactly match the confirmed Blender startup scene",
        }
    cube = scene.objects["Cube"]
    if len(cube.data.vertices) != 8 or len(cube.data.polygons) != 6:
        return {
            "removedScene": None, "removedObjects": [], "removedCollections": [],
            "preservedScene": scene.name,
            "reason": "Startup-named mesh has authored geometry",
        }
    other_scene_objects = {
        obj for other in bpy.data.scenes if other != scene for obj in other.objects
    }
    objects = [obj for obj in scene.objects if obj not in other_scene_objects]
    collections = list(scene.collection.children_recursive)
    removed_scene = scene.name
    bpy.data.scenes.remove(scene)
    removed_objects = []
    for obj in objects:
        if obj.users_scene:
            continue
        removed_objects.append(obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
    kept_collections = {
        collection for other in bpy.data.scenes
        for collection in other.collection.children_recursive
    }
    removed_collections = []
    for collection in reversed(collections):
        if collection not in kept_collections and not collection.all_objects:
            removed_collections.append(collection.name)
            bpy.data.collections.remove(collection)
    return {
        "removedScene": removed_scene, "removedObjects": removed_objects,
        "removedCollections": removed_collections,
    }


def pack_images():
    report = []
    for image in bpy.data.images:
        if image.type in {"RENDER_RESULT", "COMPOSITING"}:
            continue
        if image.source not in {"FILE", "TILED", "GENERATED"}:
            if image.users:
                raise ValueError("Client source uses an unsupported external image: "
                                 + image.name + " (" + image.source + ")")
            continue
        if not image.packed_file and not image.packed_files:
            try:
                image.pack()
            except RuntimeError as exc:
                raise ValueError("Could not pack texture/reference: " + image.name) from exc
        packed = bool(image.packed_file or image.packed_files)
        if not packed:
            raise ValueError("Texture/reference remains external: " + image.name)
        report.append({
            "name": image.name, "size": list(image.size),
            "packed": packed, "colorSpace": image.colorspace_settings.name,
        })
    return report


def set_presentation(scene, rig, camera):
    scene.camera = camera
    for window in bpy.context.window_manager.windows:
        window.scene = scene
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = None
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.camera = camera
                space.overlay.show_overlays = False
                space.shading.type = "MATERIAL"
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
                if space.region_3d:
                    space.region_3d.view_perspective = "CAMERA"
                    space.region_3d.view_camera_zoom = 0
                    space.region_3d.view_camera_offset = (0, 0)
            elif area.type == "OUTLINER":
                space = area.spaces.active
                space.display_mode = "VIEW_LAYER"
                for name in ("show_restrict_column_viewport",
                             "show_restrict_column_render",
                             "show_restrict_column_hide"):
                    if hasattr(space, name):
                        setattr(space, name, True)
    bpy.context.view_layer.update()


def run(destination=None):
    if not bpy.app.background:
        raise RuntimeError("Run client_source.py in a separate background Blender process")
    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise ValueError("Open the saved master work/character.blend first")
    destination = (Path(destination).resolve() if destination
                   else PROJECT / "exports" / "avatar.blend")
    if destination == source:
        raise ValueError("The client file must not overwrite the working master")
    if destination.suffix.lower() != ".blend":
        raise ValueError("The client source destination must end in .blend")
    scene = bpy.data.scenes.get(PRIMARY_SCENE)
    collection = bpy.data.collections.get(CHARACTER_COLLECTION)
    rig = bpy.data.objects.get(RIG_NAME)
    camera = bpy.data.objects.get(CAMERA_NAME)
    if not scene or not collection or not rig or rig.type != "ARMATURE":
        raise ValueError("The original character scene, collection or rig is missing")
    if not camera or camera.type != "CAMERA" or camera.name not in scene.objects:
        raise ValueError("The authored three-quarter studio camera is missing")
    if not bpy.context.window:
        raise RuntimeError("A background Blender window context is required")
    bpy.context.window.scene = scene
    bpy.context.view_layer.update()
    before = preservation_state(scene, collection, rig)
    source_hash = sha256(source)
    cleanup = remove_confirmed_startup_scene(scene)
    images = pack_images()
    set_presentation(scene, rig, camera)
    if preservation_state(scene, collection, rig) != before:
        raise RuntimeError("Client preparation changed protected character data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(".avatar-client-" + uuid.uuid4().hex + ".blend")
    report_path = destination.with_name("avatar-source-report.json")
    try:
        if sha256(source) != source_hash:
            raise RuntimeError("The source changed during preparation; reopen the latest master")
        result = bpy.ops.wm.save_as_mainfile(
            filepath=str(stage), copy=True, relative_remap=True,
            check_existing=False,
        )
        if "FINISHED" not in result or not stage.is_file():
            raise RuntimeError("Could not save the curated client Blend")
        if sha256(source) != source_hash:
            raise RuntimeError("The source changed during saving; reopen the latest master")
        report = {
            "sourceBlend": str(source), "sourceSha256": source_hash,
            "sourceUnchanged": True, "outputBlend": str(destination),
            "outputSha256": sha256(stage), "outputBytes": stage.stat().st_size,
            "primaryScene": scene.name,
            "activeCamera": scene.camera.name,
            "remainingScenes": [item.name for item in bpy.data.scenes],
            "rootCollections": [item.name for item in scene.collection.children],
            "characterMeshes": len(before["meshes"]),
            "bones": len(rig.data.bones),
            "springChains": len(before["springChains"]),
            "colliders": len(rig.data.vrm_addon_extension.spring_bone1.colliders),
            "posePreserved": True, "highDetailModifiersPreserved": True,
            "pbrMaterialsPreserved": True, "referenceVisibilityPreserved": True,
            "allImageAssetsPacked": True, "images": images,
            "cleanup": cleanup, "vrmMetadataFinalized": False,
        }
        stage.replace(destination)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print("CLIENT_SOURCE_REPORT", json.dumps(report, ensure_ascii=False))
        return report
    finally:
        if stage.exists():
            stage.unlink()
        backup = Path(str(stage) + "1")
        if backup.exists():
            backup.unlink()


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    options = parser.parse_args(arguments)
    run(options.output)
