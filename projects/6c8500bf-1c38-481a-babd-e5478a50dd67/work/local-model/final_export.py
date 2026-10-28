"""Export a PBR posed GLB and a rest-pose MToon VRM from a background copy.

Invoke only after saving the final work/character.blend:
  Blender --background --disable-autoexec work/character.blend \
    --python-exit-code 1 --python work/local-model/final_export.py

The saved master Blend is never overwritten. This is a preview package:
real author and redistribution permissions remain unassigned.

Expression review from an independently saved modeling copy:
  Blender --background --disable-autoexec --python-exit-code 1 \
    --python work/local-model/final_export.py -- \
    --source-blend work/expression-authoring/current.blend --expressions
Defaults to exports/expression-review/ and refuses the main exports/ directory.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile

import bpy
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
from mathutils import Matrix, Quaternion, Vector


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
COLLECTION = "CHARACTER · Original geometry"
RIG = "Character_Rig"


def helper(name):
    spec = importlib.util.spec_from_file_location("delivery_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def glb_document(path):
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2 or total != len(data):
        raise ValueError(f"Invalid GLB/VRM container: {path}")
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if kind == 0x4E4F534A:
            return json.loads(data[offset + 8:offset + 8 + length])
        offset += 8 + length
    raise ValueError("Missing glTF JSON chunk.")


def glb_stats(document):
    triangles = 0
    vertices = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                raise ValueError("Unexpected non-triangle exported primitive.")
            position = document["accessors"][primitive["attributes"]["POSITION"]]
            vertices += position["count"]
            count = (document["accessors"][primitive["indices"]]["count"]
                     if "indices" in primitive else position["count"])
            if count % 3:
                raise ValueError("A triangle primitive has a nonmultiple-of-three index count.")
            triangles += count // 3
    return {
        "triangles": triangles, "exportedVertices": vertices,
        "meshes": len(document.get("meshes", [])),
        "materials": len(document.get("materials", [])),
        "images": len(document.get("images", [])),
        "skins": len(document.get("skins", [])),
    }


def assert_mesh_coverage(document, meshes, label):
    """Catch silently omitted detail objects or accidental studio geometry."""
    expected = {obj.name for obj in meshes}
    actual = {node.get("name", "") for node in document.get("nodes", [])
              if "mesh" in node}
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{label} mesh coverage mismatch: " + json.dumps({
            "missing": missing, "unexpected": unexpected,
        }, ensure_ascii=False))


def assert_normal_textures(document, requirements, label):
    materials = {material.get("name"): material
                 for material in document.get("materials", [])}
    checked = []
    for name, source in requirements.items():
        texture = materials.get(name, {}).get("normalTexture")
        if not texture or abs(texture.get("scale", 1) - source["strength"]) > 1e-6:
            raise ValueError(f"{label}: normal texture or strength changed: {name}")
        checked.append({"material": name, **source})
    return checked


def node_world_matrices(document):
    nodes = document.get("nodes", [])
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            if child in parents:
                raise ValueError("An exported node has more than one parent.")
            parents[child] = index
    cache = {}

    def world(index):
        if index in cache:
            return cache[index]
        node = nodes[index]
        if "matrix" in node:
            local = Matrix([node["matrix"][row::4] for row in range(4)])
        else:
            rotation = node.get("rotation", [0, 0, 0, 1])
            local = Matrix.LocRotScale(
                Vector(node.get("translation", [0, 0, 0])),
                Quaternion((rotation[3], *rotation[:3])),
                Vector(node.get("scale", [1, 1, 1])),
            )
        cache[index] = world(parents[index]) @ local if index in parents else local
        return cache[index]

    return {node.get("name", str(i)): world(i) for i, node in enumerate(nodes)}


def posed_joint_check(document, expected):
    matrices = node_world_matrices(document)
    checks = []
    for name, coordinates in expected.items():
        if name not in matrices:
            raise ValueError(f"Posed GLB is missing joint {name}.")
        actual = matrices[name].translation
        current = Vector(coordinates["current"])
        rest = Vector(coordinates["rest"])
        error = (actual - current).length
        if error > 2e-4:
            raise ValueError(f"{name}: exported pose position differs by {error:.6f} m.")
        checks.append({
            "joint": name,
            "position": list(actual),
            "currentPoseErrorMeters": error,
            "distanceFromRestMeters": (actual - rest).length,
        })
    if max(c["distanceFromRestMeters"] for c in checks) < .02:
        raise ValueError("The source is in a rest pose; save the intended relaxed pose first.")
    return checks


def avatar_only_scene(rig, meshes):
    """Remove studio objects only in this disposable background process."""
    scene = bpy.data.scenes.new("Avatar Runtime · VRM 1.0")
    avatar = bpy.data.collections[COLLECTION]
    scene.collection.children.link(avatar)
    collider_collection = bpy.data.collections.new("Avatar physics colliders")
    scene.collection.children.link(collider_collection)
    collider_objects = set()
    for collider in rig.data.vrm_addon_extension.spring_bone1.colliders:
        if collider.bpy_object:
            collider_objects.add(collider.bpy_object)
            collider_objects.update(collider.bpy_object.children_recursive)
    for obj in collider_objects:
        if obj.name not in collider_collection.objects:
            collider_collection.objects.link(obj)
        obj.hide_render = True
    allowed = {rig, *meshes, *collider_objects}
    if not bpy.context.window:
        raise RuntimeError("A Blender context window is required to select the runtime scene.")
    bpy.context.window.scene = scene
    for other in list(bpy.data.scenes):
        if other != scene:
            bpy.data.scenes.remove(other)
    for obj in list(bpy.data.objects):
        if obj not in allowed:
            bpy.data.objects.remove(obj, do_unlink=True)
    kept_collections = {avatar, collider_collection, *avatar.children_recursive}
    for collection in list(bpy.data.collections):
        if collection not in kept_collections:
            bpy.data.collections.remove(collection)
    scene.render.engine = "BLENDER_EEVEE"
    scene.view_settings.view_transform = "Standard"
    bpy.context.view_layer.update()
    return scene


def run(output_dir=None, *, source_blend=None, expressions=False,
        keep_runtime_subdivision=False, runtime_optimization=True):
    if not bpy.app.background:
        raise RuntimeError(
            "Run this exporter in a separate background Blender process. "
            "It creates an avatar-only runtime copy and must not alter the live PBR scene."
        )
    if source_blend:
        requested_source = Path(source_blend).resolve()
        if not requested_source.is_file():
            raise ValueError("Requested source Blend does not exist: " + str(requested_source))
        if Path(bpy.data.filepath).resolve() != requested_source:
            bpy.ops.wm.open_mainfile(filepath=str(requested_source))
    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise ValueError("Open the saved master character Blend before running export.")
    source_hash = digest(source)
    output_dir = (Path(output_dir).resolve() if output_dir else
                  PROJECT / "exports" / "expression-review" if expressions else
                  PROJECT / "exports")
    if expressions and output_dir == (PROJECT / "exports").resolve():
        raise ValueError("Expression review must use a separate output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rig = bpy.data.objects.get(RIG)
    collection = bpy.data.collections.get(COLLECTION)
    if not rig or rig.type != "ARMATURE" or not collection:
        raise ValueError("The saved original character rig/collection is missing.")
    nonmesh_surfaces = [
        obj.name for obj in collection.all_objects
        if obj.type in {"CURVE", "SURFACE", "META", "FONT"} and not obj.hide_render
    ]
    if nonmesh_surfaces:
        raise ValueError(
            "Convert and bind the remaining character detail surfaces before export: "
            + ", ".join(nonmesh_surfaces)
        )
    meshes = [obj for obj in collection.all_objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("The avatar collection contains no meshes.")
    optimizer = helper("runtime_optimize")
    textured_meshes = {obj.name for obj in meshes if optimizer.requires_uv(obj)}
    missing_render_uv = [obj.name for obj in meshes
                         if obj.name in textured_meshes
                         and not any(uv.active_render for uv in obj.data.uv_layers)]
    missing_materials = [obj.name for obj in meshes
                         if not obj.data.materials or any(m is None for m in obj.data.materials)]
    if missing_render_uv or missing_materials:
        raise ValueError("Incomplete render/export attributes: " + json.dumps({
            "missingActiveRenderUV": missing_render_uv,
            "missingMaterials": missing_materials,
        }, ensure_ascii=False))
    api = helper("vrm_helpers")
    api._enabled()
    expression_api = helper("expression_export")
    expression_manifest = (expression_api.configure(rig, meshes) if expressions else {
        "source": expression_api.inventory(meshes), "presets": {},
    })
    human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    type(human).update_all_bone_name_candidates(bpy.context, rig.data.name, force=True)
    errors = list(human.error_messages())
    if errors:
        raise ValueError("Invalid humanoid: " + "; ".join(errors))
    bindings = api.audit_bindings(rig, meshes)
    bad = [row for row in bindings if row["unweightedVertices"]
           or row["nonNormalizedVertices"] or row["armatureModifiers"] != 1
           or (row["mesh"] in textured_meshes and row["uvLayers"] < 1)]
    if bad:
        raise ValueError("Avatar bindings/UV need correction: " + json.dumps(bad))
    spring_api = api._module("spring-setup.py", "delivery_spring_audit")
    spring_audit = spring_api.audit(rig, meshes)
    plan = json.loads(rig["charakit_spring_plan_v1"])
    if len(plan["targets"]) > 128:
        raise ValueError("The existing JavaScript physics validator supports at most 128 targets.")
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in [rig, *meshes]:
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.context.view_layer.update()
    expected = {}
    for name in ("leftHand", "rightHand", "leftLowerArm", "rightLowerArm"):
        current = rig.matrix_world @ rig.pose.bones[name].head
        rest = rig.matrix_world @ rig.data.bones[name].head_local
        expected[name] = {
            "current": [current.x, current.z, -current.y],
            "rest": [rest.x, rest.z, -rest.y],
        }
    source_materials = {m for obj in meshes for m in obj.data.materials}
    normal_requirements = {}
    for material in source_materials:
        wrapper = PrincipledBSDFWrapper(material)
        texture = wrapper.normalmap_texture
        if texture and texture.image:
            normal_requirements[material.name] = {
                "image": texture.image.name,
                "strength": wrapper.normalmap_strength,
            }
    source_mesh_count = len(meshes)
    with tempfile.TemporaryDirectory(prefix=".avatar-export-", dir=output_dir) as stage_dir:
        stage = Path(stage_dir)
        posed_glb = stage / "avatar-preview.glb"
        has_shape_keys = bool(expression_manifest["source"])
        with (expression_api.evaluated_pbr_meshes(meshes)
              if has_shape_keys else nullcontext()):
            operation = bpy.ops.export_scene.gltf(
                filepath=str(posed_glb),
                export_format="GLB",
                use_selection=True,
                export_materials="EXPORT",
                export_animations=False,
                export_current_frame=True,
                export_rest_position_armature=False,
                export_apply=not has_shape_keys,
                export_morph=True,
                export_morph_normal=True,
                export_skins=True,
                export_all_influences=False,
                export_influence_nb=4,
                export_yup=True,
                export_cameras=False,
                export_lights=False,
            )
        if "FINISHED" not in operation:
            raise RuntimeError(f"PBR GLB export did not finish: {operation}")
        pbr_document = glb_document(posed_glb)
        assert_mesh_coverage(pbr_document, meshes, "PBR GLB")
        assert_normal_textures(pbr_document, normal_requirements, "PBR GLB")
        pose_checks = posed_joint_check(pbr_document, expected)
        pbr_expressions = expression_api.audit(posed_glb, expression_manifest)
        pbr_expressions["file"] = str(output_dir / posed_glb.name)
        if expression_api.inventory(meshes) != expression_manifest["source"]:
            raise ValueError("PBR export changed the original expression key coordinates or state")
        pbr_stats = glb_stats(pbr_document)
        if pbr_document.get("cameras") or "KHR_lights_punctual" in pbr_document.get("extensions", {}):
            raise ValueError("Studio camera/lights unexpectedly entered the posed GLB.")
        # Runtime output uses the rest pose and original detailed base meshes.
        rig.data.pose_position = "POSE"
        for pose_bone in rig.pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        lowered = []
        if not keep_runtime_subdivision:
            for obj in meshes:
                # Animated facial surfaces retain authored subdivision for
                # lid/lip curvature; runtime cost is reported after export.
                if obj.data.shape_keys or obj.get("facial_role"):
                    continue
                for modifier in obj.modifiers:
                    if modifier.type == "SUBSURF" and (modifier.levels or modifier.render_levels):
                        lowered.append({"mesh": obj.name, "modifier": modifier.name,
                                        "levels": modifier.levels,
                                        "renderLevels": modifier.render_levels})
                        modifier.levels = modifier.render_levels = 0
        # Convert while the original active UV layer names still exist. Runtime
        # grouping standardizes those layers only after MToon retains textures.
        hair_uv_preparation = helper("hair_realism").prepare_vrm_uv(meshes)
        api._enabled().apply_mtoon(meshes)
        material_finish = helper("material_finish").apply_runtime(meshes)
        optimization = None
        if runtime_optimization:
            meshes, plan, optimization = optimizer.optimize(
                rig, meshes, plan
            )
            runtime_bindings = api.audit_bindings(rig, meshes)
            runtime_textured_meshes = {
                obj.name for obj in meshes if optimizer.requires_uv(obj)
            }
            invalid = [row for row in runtime_bindings
                       if row["unweightedVertices"] or row["nonNormalizedVertices"]
                       or row["armatureModifiers"] != 1
                       or (row["mesh"] in runtime_textured_meshes and row["uvLayers"] != 1)]
            if invalid:
                raise ValueError("Runtime optimization damaged bindings/UV: "
                                 + json.dumps(invalid))
        avatar_only_scene(rig, meshes)
        bpy.context.view_layer.update()
        rest_errors = []
        for bone in rig.pose.bones:
            delta = max(abs(bone.matrix[i][j] - bone.bone.matrix_local[i][j])
                        for i in range(4) for j in range(4))
            if delta > 1e-5:
                rest_errors.append({"bone": bone.name, "matrixDelta": delta})
        if rest_errors:
            raise ValueError("Could not reset the runtime rig to rest pose: " + json.dumps(rest_errors))
        preview_vrm = stage / "avatar-preview.vrm"
        vrm_inspection = api.export_review(
            rig, meshes, preview_vrm, convert_materials=False
        )
        runtime_document = glb_document(preview_vrm)
        runtime_expressions = expression_api.audit(
            preview_vrm, expression_manifest, require_vrm=expressions
        )
        runtime_expressions["file"] = str(output_dir / preview_vrm.name)
        assert_mesh_coverage(runtime_document, meshes, "Runtime VRM")
        normal_checks = assert_normal_textures(
            runtime_document, normal_requirements, "Runtime VRM"
        )
        runtime_stats = glb_stats(runtime_document)
        runtime_plan = spring_api.audit(rig, meshes)
        runtime_blend = stage / "avatar-runtime.blend"
        saved = bpy.ops.wm.save_as_mainfile(filepath=str(runtime_blend), copy=True)
        if "FINISHED" not in saved:
            raise RuntimeError("The MToon runtime Blend could not be saved.")
        if digest(source) != source_hash:
            raise RuntimeError("The source Blend changed during export; regenerate from its latest save.")
        report = {
            "sourceBlend": str(source),
            "sourceBlendSha256": source_hash,
            "previewOnly": True,
            "authorMetadataFinalized": False,
            "appearanceReviewedByUser": False,
            "expressionReview": expressions,
            "expressionManifest": expression_manifest,
            "expressionChecks": {"pbr": pbr_expressions, "vrm": runtime_expressions},
            "baseMeshes": source_mesh_count,
            "runtimeMeshes": len(meshes),
            "totalArmatureBones": len(rig.data.bones),
            "humanoidBones": sum(
                bool(entry.node.bone_name)
                for entry in human.human_bone_name_to_human_bone().values()
            ),
            "sourceMaterials": len(source_materials),
            "unweightedVertices": sum(row["unweightedVertices"] for row in bindings),
            "nonNormalizedVertices": sum(row["nonNormalizedVertices"] for row in bindings),
            "posedPbrGlb": {**pbr_stats, "poseChecks": pose_checks},
            "runtimeVrm": {**runtime_stats, **vrm_inspection},
            "runtimeSubdivisionLowered": lowered,
            "runtimeTriangleBudget": 300000,
            "runtimeExceedsTriangleBudget": runtime_stats["triangles"] > 300000,
            "decimationApplied": bool(optimization and any(
                row["accepted"] for row in optimization["meshes"]
            )),
            "runtimeOptimization": optimization,
            "runtimeMaterialFinish": material_finish,
            "runtimeHairUvPrepared": hair_uv_preparation,
            "normalTextureChecks": normal_checks,
            "springChains": spring_audit["chains"],
            "springTargets": len(plan["targets"]),
            "runtimeSpringAuditChains": runtime_plan["chains"],
            "physicsMotionTest": "Run scripts/verify-springs.mjs separately; not claimed by this export.",
            "files": {},
        }
        for file in (posed_glb, preview_vrm, runtime_blend):
            report["files"][file.name] = {
                "sha256": digest(file), "bytes": file.stat().st_size,
            }
        for file in (posed_glb, preview_vrm, runtime_blend):
            file.replace(output_dir / file.name)
        (output_dir / "avatar-spring-plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "avatar-export-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("AVATAR_EXPORT_REPORT", json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--source-blend",
                        help="Open this saved source in the background export process.")
    parser.add_argument("--expressions", action="store_true",
                        help="Bind authored facial keys and stage a separate expression review.")
    parser.add_argument("--keep-runtime-subdivision", action="store_true")
    parser.add_argument("--no-runtime-optimization", action="store_true")
    opts = parser.parse_args(args)
    run(opts.output_dir, source_blend=opts.source_blend, expressions=opts.expressions,
        keep_runtime_subdivision=opts.keep_runtime_subdivision,
        runtime_optimization=not opts.no_runtime_optimization)
