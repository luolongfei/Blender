"""Integrate the reviewed, project-local refinement modules in dependency order.

The only live writer is the main agent. This helper refuses another base file
and retains the previous source in the named checkpoint before saving.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import shutil
import bpy

P=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
EXPECTED_SOURCE="7b08fc54bb873e5c03af24cdebaf8a1798423069c0357d2c168341b2df3b316c"
MODULES=("body_completion","neckline_clearance","ear_cartilage_refine","hair_refinement",
         "hair_chain_refine","motion_collision_refine","anatomy_material_refine","cranial_fairing",
         "hair_collision_radius_refine","face_expressions","facial_microstructure","dental_refine")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name):
    spec=importlib.util.spec_from_file_location("integrate_"+name,HERE/(name+".py"))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def integrate(save=False):
    source=P/"work/character.blend"
    if Path(bpy.data.filepath).resolve()!=source:
        raise ValueError("Open the current project work/character.blend before live integration.")
    if digest(source)!=EXPECTED_SOURCE:
        raise ValueError("The source changed; reconcile before applying project-wide modules.")
    missing=[name for name in MODULES if not (HERE/(name+".py")).is_file()]
    if missing:
        raise ValueError("Required authored modules are not ready: "+", ".join(missing))
    imports={name:module(name) for name in MODULES}
    checkpoint=P/"work/checkpoints/comprehensive-refinement-start/character.blend"
    if not checkpoint.is_file() or digest(checkpoint)!=EXPECTED_SOURCE:
        raise ValueError("Verified retained source checkpoint is required.")
    scene=bpy.context.scene
    col=bpy.data.collections["CHARACTER · Original geometry"]
    rig=bpy.data.objects["Character_Rig"]
    imports["face_expressions"].pose({})
    report={"sourceBeforeSha256":EXPECTED_SOURCE,
            "modules":{name:digest(HERE/(name+".py")) for name in MODULES},
            "steps":{},"defaultBlush":False,"appearanceAcceptance":"pending"}
    steps=report["steps"]
    steps["body"]=imports["body_completion"].apply(col,rig)
    steps["neckline"]=imports["neckline_clearance"].apply(col,rig)
    steps["ears"]=imports["ear_cartilage_refine"].apply(col,rig)
    steps["hair"]=imports["hair_refinement"].apply(col,rig)
    plan=json.loads(rig["charakit_spring_plan_v1"])
    known={target["mesh"] for target in plan["targets"]}
    for target in [*steps["hair"]["springTargetAdditions"],
                   steps["hair"]["fixedTargetAddition"]]:
        if target["mesh"] not in known:
            plan["targets"].append(target);known.add(target["mesh"])
    rig["charakit_spring_plan_v1"]=json.dumps(plan,ensure_ascii=False)
    steps["hairChain"]=imports["hair_chain_refine"].apply(
        rig=rig,centerline_transform=imports["hair_refinement"].motion_center,
        dynamics="graduated")
    steps["backHairCollisions"]=imports["motion_collision_refine"].apply_back_coverage(rig)
    steps["hairSurfaceRadius"]=imports["hair_collision_radius_refine"].apply(rig)
    steps["faceAndPalette"]=imports["anatomy_material_refine"].apply(bake=True)
    steps["cranialFairing"]=imports["cranial_fairing"].apply(col,rig)
    steps["actualSpringSettings"]=imports["hair_chain_refine"].sync_actual_spring_plan(rig)
    rig["comprehensive_head_field_applied"]=True
    imports["face_expressions"].pose({})
    bpy.context.view_layer.update()
    meshes=[obj for obj in col.all_objects if obj.type=="MESH"]
    bad=[]
    for obj in meshes:
        for vertex in obj.data.vertices:
            if not all(math.isfinite(x) for x in vertex.co):
                bad.append((obj.name,"nonfinite geometry"));break
            total=sum(g.weight for g in vertex.groups)
            if abs(total-1)>1e-4:
                bad.append((obj.name,"unnormalized weights",vertex.index,total));break
        if obj.data.shape_keys:
            if any(key.value for key in obj.data.shape_keys.key_blocks if key.name!="Basis"):
                bad.append((obj.name,"nonneutral key"))
            for key in obj.data.shape_keys.key_blocks:
                if any(not all(math.isfinite(x) for x in v.co) for v in key.data):
                    bad.append((obj.name,key.name,"nonfinite shape"))
    if bad:
        raise ValueError("Integration validation failed: "+repr(bad[:12]))
    report["meshCount"]=len(meshes)
    report["boneCount"]=len(rig.data.bones)
    report["morphMeshCount"]=sum(bool(obj.data.shape_keys) for obj in meshes)
    report["allCoordinatesFinite"]=True
    report["skinWeightsNormalized"]=True
    report["allExpressionsNeutral"]=True
    scene["comprehensive_refinement_modules"]=json.dumps(report["modules"])
    if save:
        for image in bpy.data.images:
            if image.source=="FILE" and image.has_data and not image.packed_file:
                image.pack()
        bpy.ops.wm.save_as_mainfile(filepath=str(source))
        report["sourceAfterSha256"]=digest(source)
        for path in (P/"work/realism-authoring/current.blend",
                     P/"work/expression-authoring/current.blend"):
            shutil.copy2(source,path)
        (P/"work/comprehensive-integration-report.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


if __name__=="__main__":
    print(json.dumps(integrate(save=False),ensure_ascii=False))
