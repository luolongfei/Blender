"""Apply individually reviewed corrections to the frozen integrated source.

Run from the live Blender MCP only after all candidate helper hashes are fixed.
Candidate helpers are passed explicitly; older integration deformations must
never be replayed over the integrated source.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import shutil
import bpy

P = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXPECTED = "ebf64d62920b0f2fff3471c7bafb45394f2787cc0fe5511a1a7991721864f6f3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location("finish_" + name, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def finish(helpers, *, entries=None, dependencies=None,
           verified_live_model=None, save=False):
    """helpers maps module names to reviewed SHA256 in their application order."""
    source = P / "work/character.blend"
    if Path(bpy.data.filepath).resolve() != source or sha(source) != EXPECTED:
        raise ValueError("The authoritative integrated source has changed.")
    if bpy.data.is_dirty and not verified_live_model:
        raise ValueError("Unsaved live edits must be reconciled before integration.")
    checkpoint = P / "work/qa-comprehensive/first-combination/character.blend"
    if not checkpoint.is_file() or sha(checkpoint) != EXPECTED:
        raise ValueError("The retained integrated source checkpoint is required.")
    for name, expected in helpers.items():
        if sha(HERE / (name + ".py")) != expected:
            raise ValueError("Reviewed helper changed: " + name)
    for name, expected in (dependencies or {}).items():
        if sha(HERE / name) != expected:
            raise ValueError("Reviewed helper dependency changed: " + name)
    if verified_live_model:
        if "live_source_fingerprint.py" not in (dependencies or {}):
            raise ValueError("Live-state reconciliation requires a frozen fingerprint helper.")
        live = load("live_source_fingerprint").fingerprint()
        if live["modelSha256"] != verified_live_model:
            (P / "work/checkpoints/pre-finish-live-state/realtime-fingerprint.json").write_text(
                json.dumps(live, ensure_ascii=False, indent=2) + "\n")
            raise ValueError("The live character changed since source reconciliation.")
    col = bpy.data.collections["CHARACTER · Original geometry"]
    rig = bpy.data.objects["Character_Rig"]
    face = load("face_expressions")
    face.pose({})
    before_physics = load("hair_chain_refine").sync_actual_spring_plan(rig)
    steps = {}
    for name in helpers:
        entry = (entries or {}).get(name, "apply")
        steps[name] = getattr(load(name), entry)(col, rig)
    face.pose({})
    after_physics = load("hair_chain_refine").sync_actual_spring_plan(rig)
    if after_physics != before_physics:
        raise ValueError("Post-integration corrections changed verified spring settings.")
    bpy.context.view_layer.update()
    meshes = [obj for obj in col.all_objects if obj.type == "MESH"]
    issues = []
    for obj in meshes:
        for vertex in obj.data.vertices:
            if not all(math.isfinite(value) for value in vertex.co):
                issues.append((obj.name, vertex.index, "nonfinite"))
                break
            if abs(sum(group.weight for group in vertex.groups) - 1) > 1e-4:
                issues.append((obj.name, vertex.index, "weights"))
                break
        if obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                if key.name != "Basis" and key.value != 0:
                    issues.append((obj.name, key.name, "nonneutral"))
                if any(not all(math.isfinite(value) for value in point.co)
                       for point in key.data):
                    issues.append((obj.name, key.name, "nonfinite morph"))
    if issues:
        raise ValueError("Final source validation failed: " + repr(issues[:20]))
    report = {
        "sourceBeforeSha256": EXPECTED,
        "helpers": helpers,
        "entrypoints": entries or {},
        "helperDependencies": dependencies or {},
        "reconciledLiveModelSha256": verified_live_model,
        "steps": steps,
        "meshCount": len(meshes),
        "boneCount": len(rig.data.bones),
        "allCoordinatesFinite": True,
        "allWeightsNormalized": True,
        "allExpressionsNeutral": True,
        "verifiedSpringSettingsPreserved": True,
        "appearanceAcceptance": "pending",
    }
    if save:
        for image in bpy.data.images:
            if image.source == "FILE" and image.has_data and not image.packed_file:
                image.pack()
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        report["sourceAfterSha256"] = sha(source)
        for relative in ("work/realism-authoring/current.blend",
                         "work/expression-authoring/current.blend"):
            shutil.copy2(source, P / relative)
        (P / "work/comprehensive-finish-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        integration_path = P / "work/comprehensive-integration-report.json"
        integration = json.loads(integration_path.read_text())
        integration["sourceBeforeFinishSha256"] = EXPECTED
        integration["sourceAfterSha256"] = report["sourceAfterSha256"]
        integration["finish"] = report
        integration["meshCount"] = len(meshes)
        integration_path.write_text(json.dumps(integration, ensure_ascii=False, indent=2) + "\n")
    return report
