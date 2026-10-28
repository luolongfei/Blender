"""Package the integrated facial/hair/motion revision with exact-source checks."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import zipfile

P = Path(__file__).resolve().parents[2]
E = P / "exports"
R = E / "expression-review"


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def read(file):
    return json.loads(file.read_text())


def main():
    export = read(R / "avatar-export-report.json")
    source = read(E / "avatar-source-report.json")
    renders = read(P / "previews/refined-current/render-report.json")
    master = P / "work/character.blend"
    assert sha(master) == export["sourceBlendSha256"] == source["sourceSha256"]
    assert sha(master) == renders["sourceSha256"]
    visual_assessment = read(E / "quality-assessment.json")
    assert visual_assessment["sourceBlendSha256"] == sha(master)
    assert source["allImageAssetsPacked"] and source["sourceUnchanged"]
    assert sha(E / "avatar.blend") == source["outputSha256"]
    for name, expected in export["files"].items():
        assert sha(R / name) == expected["sha256"], name
    for item in renders["files"]:
        assert sha(P / item["file"]) == item["sha256"]
    retained = P / "work/checkpoints/comprehensive-refinement-start/previous-review"
    old_render_report = read(retained / "previews/refined-current/render-report.json")
    old_profile = next(item for item in old_render_report["files"]
                       if item["file"].endswith("/profile.png"))
    assert sha(retained / old_profile["file"]) == old_profile["sha256"]
    before_image = P / "previews/refined-current/before-profile.png"
    shutil.copy2(retained / old_profile["file"], before_image)
    comparison = {
        "before": {
            "file": str(before_image.relative_to(P)),
            "sourceSha256": old_render_report["sourceSha256"],
            "imageSha256": sha(before_image),
        },
        "after": {
            "file": "previews/refined-current/profile.png",
            "sourceSha256": renders["sourceSha256"],
            "imageSha256": sha(P / "previews/refined-current/profile.png"),
        },
        "unalteredActualBlenderRenders": True,
        "note": "Both are actual profile renders; framing and lighting may differ.",
    }
    comparison_file = P / "previews/refined-current/comparison-provenance.json"
    comparison_file.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    vrm_sha = sha(R / "avatar-preview.vrm")
    required_reports = (
        "avatar-expression-runtime-report.json",
        "avatar-physics-report.json",
        "avatar-natural-motion-report.json",
        "avatar-body-completion-report.json",
    )
    reports = {}
    for name in required_reports:
        report = read(R / name)
        # Validators use vrmSha256 or inputSha256, depending on their scope.
        hashes = [report.get(k) for k in ("vrmSha256", "inputSha256", "sourceSha256")]
        if vrm_sha not in hashes:
            raise ValueError("Validation is not tied to the current VRM: " + name)
        reports[name] = report
    registration = read(P / "work/expression-authoring/project-preview-registration.json")
    assert registration["modelSha256"] == vrm_sha
    assert sha(P / "project.charakit") == registration["projectAfter"]
    limits = [
        "按用户要求的真人感商业终稿标准，当前仍未达标，不能作为已经验收的最终成品。",
        "面部近景仍有人偶感：眼睑、鼻唇细节与虹膜精细程度不一致，皮肤层次偏均匀。",
        "头发主轮廓仍有成片和成束感；已有细丝几何和可动骨链，不等于逐根真人毛发外观或逐根物理仿真。",
        "手部关节、甲床与衣料受力褶皱仍较简化，近景真实感需要继续雕刻和材质制作。",
        "造型和表情的审美验收仍由用户／客户确认。",
        "默认无腮红；当前提供表情和口型，尚未制作眼球追踪／LookAt。",
        "运行版用于桌面审阅；目标软件、设备及性能预算尚未指定。",
        "已测试的步行和招手不代表任意大幅动作都没有穿插。",
        "随包自然动作按当前鞋底校准；原套件标准步行仅作对照，尚有鞋底高度偏差，不能视为本角色已适配的交付动作。",
        "自然动作切换检查中有一帧双脚鞋底最低点约8.06毫米高于地面，不能称全部过渡始终精确贴地。",
        "衣内身体使用隐藏重叠连接，包含躯干、双臂、下胫和足部；不是一体焊接的裸体解剖模型。",
        "脚趾随当前刚性鞋一起运动；独立脚趾动作需要适配鞋的变形。",
        "正式模型名、作者、使用及再分发条款尚未确认；VRM仍为审阅元数据。",
    ]
    if export["runtimeExceedsTriangleBudget"]:
        limits.append(
            f"当前运行版有 {export['runtimeVrm']['triangles']:,} 个三角形，"
            f"高于项目脚本 {export['runtimeTriangleBudget']:,} 的目标；"
            "尚未完成目标设备的帧率和内存验收。"
        )
    manifest = {
        "projectId": P.name, "status": "comprehensive-refinement-review-candidate",
        "route": "local-original-Blender-MCP",
        "sourceBlendSha256": sha(master), "vrmSha256": vrm_sha,
        "files": {
            "editableMaster": "exports/avatar.blend",
            "previewVrm": "exports/expression-review/avatar-preview.vrm",
            "runtimeSource": "exports/expression-review/avatar-runtime.blend",
            "pbrModel": "exports/expression-review/avatar-preview.glb",
        },
        "projectWorkingMaster": "work/character.blend",
        "defaultBlush": False, "appearanceReviewedByUser": False,
        "photorealisticClientDeliveryReady": False,
        "qualityAssessment": "exports/quality-assessment.json",
        "authorMetadataFinalized": False, "canonicalCharakitUpdated": True,
        "productionStagesUnchanged": registration["productionStagesUnchanged"],
        "runtimeTriangles": export["runtimeVrm"]["triangles"],
        "runtimeMeshes": export["runtimeVrm"]["meshes"],
        "runtimeTriangleTarget": export["runtimeTriangleBudget"],
        "runtimeExceedsTriangleTarget": export["runtimeExceedsTriangleBudget"],
        "expressionPresets": list(export["expressionManifest"]["presets"]),
        "limits": limits,
        "previews": renders["files"],
        "validationReports": list(required_reports),
    }
    (E / "delivery-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    text = """双马尾绿裙角色 · 头部、完整衣内身体、发丝与动作精修审阅

此版在当前项目通过Blender MCP完成。原始素材与此前版本保存在项目checkpoint中。
当前文件用于实际模型审阅，客户审美验收和正式授权信息尚待确认。

文件
exports/avatar.blend：完整可编辑源文件，包含真实面部表情键、PBR材质、骨骼和物理。
exports/expression-review/avatar-preview.vrm：当前VRM 1.0审阅模型，包含MToon、表情、Humanoid和SpringBone。
exports/expression-review/avatar-runtime.blend：与当前VRM对应的运行版源文件。
exports/expression-review/avatar-preview.glb：完整细节PBR模型。
exports/motions/：角色动作文件（如有）。
previews/refined-current/：当前整合模型的真实Blender渲染。
审阅图册.html：离线静态图册；视频如已录制会一并显示。
checksums.sha256：包内文件校验。

本轮修改
修正头面宽度、后脑曲线和颅顶；保持连续眼睑、口周和多种表情。
统一中性肤色、米白衣料、低饱和绿裙及棕色皮鞋；保留衣料与头发法线。
新增鼻翼下的真实凹腔、随表情移动的细眉毛、独立牙冠与连续耳廓软骨表面。
默认无腮红，支持眨眼、单眼、情绪及五种口型，连续强度可调。
补齐衣服内的躯干、骨盆、双臂及袜鞋内小腿、踝和足部，并绑定现有骨架。
调整刘海发根、后脑发帽边缘和细发束；发丝包含真实几何及可导出的法线。
马尾使用更密的碰撞骨链，发骨中心跟随发束形状；实际摆动以当前VRM检查结果为准。
肩根引入锁骨过渡，平滑肘膝权重；动作加入接地、重心、肩臂和手指配合。

预览
原项目previews/motion-viewer.html用于查看动作、物理和原步行对照。
原项目previews/expression-viewer.html用于检查表情强度及侧面。
这两个交互页面需原项目本地服务；本包图册和视频可离线查看。

当前边界
""" + "\n".join("- " + item for item in limits) + "\n"
    (E / "交付说明.txt").write_text(text)
    files = [
        "exports/avatar.blend", "exports/avatar-source-report.json",
        "exports/delivery-manifest.json", "exports/交付说明.txt",
        "exports/quality-assessment.json", "exports/质量评估.txt",
        *(str((R / name).relative_to(P)) for name in (
            "avatar-preview.vrm", "avatar-preview.glb", "avatar-runtime.blend",
            "avatar-export-report.json", "avatar-spring-plan.json", *required_reports)),
        *(item["file"] for item in renders["files"]),
        "previews/refined-current/render-report.json",
        "previews/refined-current/before-profile.png",
        "previews/refined-current/comparison-provenance.json",
    ]
    files += [str(file.relative_to(P)) for file in (E / "motions").glob("*")
              if file.is_file() and file.suffix in {".vrma", ".json", ".glb", ".txt"}]
    videos = [str(file.relative_to(P)) for file in (P / "previews").glob("*")
              if file.is_file() and file.name in {
                  "character-motion-review.mp4", "character-motion-review.webm",
                  "character-expression-review.mp4",
              }]
    video_report_path=P / "previews" / "character-motion-review.json"
    if videos:
        video_report=read(video_report_path)
        assert video_report["vrmSha256"]==vrm_sha
        for file in videos:
            assert sha(P / file)==video_report["files"][Path(file).name]["sha256"]
        files.append(str(video_report_path.relative_to(P)))
    files += videos
    for name in ("avatar-hair-walk-report.json","avatar-hair-greet-report.json",
                 "avatar-hair-static-report.json","avatar-hair-standard-report.json",
                 "avatar-hair-dynamics-report.json","avatar-body-clothing-static-report.json",
                 "avatar-body-clothing-walk-report.json","avatar-body-clothing-greet-report.json",
                 "avatar-runtime-review-summary.json",
                 "avatar-skirt-motion-report.json","avatar-browser-review.json",
                 "avatar-body-completion-report.json"):
        file=R/name
        if file.is_file():
            extra=read(file)
            if vrm_sha in [extra.get(k) for k in ("vrmSha256","inputSha256","sourceSha256")]:
                files.append(str(file.relative_to(P)))
    for name in ("face-natural-basecolor.png", "face-natural-normal.png",
                 "skin-neutral-anatomical-basecolor.png",
                 "skin-neutral-anatomical-normal.png",
                 "iris-brown-realism.png",
                 "hair-charcoal-basecolor.png", "hair-fine-normal.png",
                 "clothing-basecolor.png", "clothing-normal.png"):
        file = P / "textures" / name
        if file.is_file():
            files.append(str(file.relative_to(P)))
    titles = {
        "hero": "整体", "front": "正面", "back": "背面", "portrait": "面部与头发",
        "profile": "侧面轮廓", "happy": "笑容", "blink": "闭眼", "hand": "手部与袖口",
        "aa":"张口与牙冠","head-front":"头部正面","head-left":"角色右侧头部",
        "head-left45":"角色右前斜侧","head-right45":"角色左前斜侧","head-back":"后脑",
        "head-back-left":"角色右后斜侧","head-back-right":"角色左后斜侧",
        "sleeve-inside":"衣内手臂检查（临时隐藏左袖，T姿势）",
        "torso-inside":"衣内躯干检查（临时隐藏上衣，T姿势）",
        "full-left":"角色右侧全身","full-right":"角色左侧全身","full-back45":"全身后斜侧",
        "blink-half":"半闭眼（强度50%）","aa-half":"张口中间形（强度50%）",
    }
    figures = "\n".join(
        '<figure><img src="' + html.escape(item["file"]) + '" loading="lazy">'
        "<figcaption>" + titles[Path(item["file"]).stem] + "</figcaption></figure>"
        for item in renders["files"])
    comparison_html = (
        '<figure><img src="previews/refined-current/before-profile.png">'
        '<figcaption>本轮调整前 · 实际侧面渲染</figcaption></figure>'
        '<figure><img src="previews/refined-current/profile.png">'
        '<figcaption>本轮调整后 · 实际侧面渲染（取景与光照可能不同）</figcaption></figure>'
    )
    video_html = "\n".join(
        '<video controls preload="metadata" src="' + html.escape(file) + '"></video>'
        for file in videos if file.endswith(".mp4"))
    gallery = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>角色整合审阅</title><style>
body{{margin:0;background:#f2f0eb;color:#252b25;font:16px/1.7 system-ui,sans-serif}}
header,main,footer{{max-width:1080px;margin:auto;padding:28px 22px}}
main{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
figure{{margin:0;background:white}}img,video{{display:block;width:100%}}
figcaption{{padding:12px 16px}}video{{grid-column:1/-1}}h1{{margin:0;font-size:30px}}
@media(max-width:680px){{main{{grid-template-columns:1fr}}}}</style>
<header><h1>双马尾绿裙角色 · 整合审阅</h1>
<p>头部、表情、发丝、衣内身体与动作精修。以下为当前Blender模型的实际渲染，尚待外观验收。</p>
<p>按真人感商业终稿标准，当前仍未达标。主要差距是面部的人偶感、发片与发束感，以及手部和衣料的简化细节。</p>
<a href="exports/交付说明.txt">文件说明与验收边界</a></header>
<main>{comparison_html}{video_html}{figures}</main>
<footer>图像为PBR源文件渲染，VRM使用MToon材质。最终外观以目标应用和客户验收为准。</footer>
</html>"""
    gallery_path = P / "previews" / "refined-current" / "review.html"
    # This local file has project-relative asset paths adjusted from the ZIP root.
    gallery_path.write_text(gallery.replace('src="previews/', 'src="../')
                            .replace('href="exports/', 'href="../../exports/'))
    archive_path = E / "character-client-review.zip"
    stage = archive_path.with_suffix(".zip.part")
    entries = []
    with zipfile.ZipFile(stage, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in dict.fromkeys(files):
            file = P / name
            archive.write(file, name)
            entries.append(f"{sha(file)}  {name}")
        data = gallery.encode()
        archive.writestr("审阅图册.html", data)
        entries.append(f"{hashlib.sha256(data).hexdigest()}  审阅图册.html")
        archive.writestr("checksums.sha256", "\n".join(entries) + "\n")
    with zipfile.ZipFile(stage) as archive:
        assert archive.testzip() is None
        for line in archive.read("checksums.sha256").decode().splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    stage.replace(archive_path)
    print(json.dumps({"package": str(archive_path), "sha256": sha(archive_path),
                      "bytes": archive_path.stat().st_size, "files": len(entries) + 1},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
