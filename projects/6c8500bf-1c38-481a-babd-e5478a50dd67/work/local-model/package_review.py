"""Build a local, self-contained customer review package from checked outputs."""
from pathlib import Path
import hashlib
import html
import json
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
EXPORTS = PROJECT / "exports"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    export = json.loads((EXPORTS / "avatar-export-report.json").read_text())
    source = json.loads((EXPORTS / "avatar-source-report.json").read_text())
    physics = json.loads((EXPORTS / "avatar-physics-report.json").read_text())
    master = PROJECT / "work/character.blend"
    assert digest(master) == export["sourceBlendSha256"] == source["sourceSha256"]
    assert source["sourceUnchanged"] and source["allImageAssetsPacked"]
    for name, expected in export["files"].items():
        assert digest(EXPORTS / name) == expected["sha256"], name
    assert digest(EXPORTS / "avatar.blend") == source["outputSha256"]
    images = ["hero", "portrait", "front", "side", "back", "hand", "shoes"]
    titles = ["整体 · 三分之四", "面部与发束", "正面", "侧面", "背面", "手部与袖口", "鞋袜"]
    preview_files = [f"previews/{name}-refined.png" for name in images]
    preview_files += [
        "previews/face-side-refined.png",
        "previews/face-three-quarter-refined.png",
        "previews/face-front-refined.png",
        "previews/face-left-refined.png",
        "previews/face-side-clay-before.png",
        "previews/face-side-clay-after.png",
        "previews/face-three-quarter-clay-before.png",
        "previews/face-three-quarter-clay-after.png",
    ]
    titles += [
        "面部精修 · 侧面", "面部精修 · 四十五度", "面部精修 · 正面",
        "面部精修 · 另一侧", "侧面灰模 · 修改前", "侧面灰模 · 修改后",
        "四十五度灰模 · 修改前", "四十五度灰模 · 修改后",
    ]
    limits = [
        "固定中性表情；未制作眨眼、口型、表情控制或 LookAt。",
        "真实模型名称、作者与使用／再分发条款尚未确定；VRM 含内部审阅占位元数据。",
        "运行版面向桌面审阅，未按客户的具体软件、设备、面数和性能预算验收。",
        "头发近景仍保留分束边界；造型与参考图的还原程度需由用户／客户确认。",
        "物理响应和标准步行检查不代表所有动作均无穿插；大幅动作需按目标用途补测。",
    ]
    motion_report_path = EXPORTS / "avatar-walk-report.json"
    motion_report = json.loads(motion_report_path.read_text()) if motion_report_path.exists() else None
    manifest = {
        "projectId": PROJECT.name,
        "route": "local-original-Blender-MCP",
        "status": "client-review-candidate",
        "files": {
            "editableMaster": "exports/avatar.blend",
            "workingMaster": "work/character.blend",
            "interactivePbrModel": "exports/avatar-preview.glb",
            "previewVrm": "exports/avatar-preview.vrm",
            "vrmSource": "exports/avatar-runtime.blend",
        },
        "sourceBlendSha256": digest(master),
        "previews": preview_files,
        "viewerUrl": f"http://127.0.0.1:8767/projects/{PROJECT.name}/previews/vrm-viewer.html",
        "validation": {
            "sourceMeshes": export["baseMeshes"],
            "runtimeMeshes": export["runtimeVrm"]["meshes"],
            "runtimeTriangles": export["runtimeVrm"]["triangles"],
            "pbrTriangles": export["posedPbrGlb"]["triangles"],
            "bonesTotalIncludingSpringsAndTerminals": export["humanoidBones"],
            "unweightedVertices": export["unweightedVertices"],
            "nonNormalizedVertices": export["nonNormalizedVertices"],
            "embeddedRuntimeImages": export["runtimeVrm"]["images"],
            "mtoonMaterials": export["runtimeVrm"]["mtoon"],
            "springChains": physics["chains"],
            "springJoints": physics["joints"],
            "colliders": physics["colliders"],
            "physicsFrames": physics["frames"],
            "texturesPacked": source["allImageAssetsPacked"],
            "standardWalk": motion_report,
        },
        "limits": limits,
        "canonicalCharakitUpdated": False,
        "authorMetadataFinalized": False,
        "appearanceReviewedByUser": False,
        "defaultBlush": False,
        "faceGeometryVersion": "continuous-neutral-profile-02",
        "faceTextureVersion": 3,
    }
    (EXPORTS / "delivery-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    readme = f"""双马尾绿裙角色 · 客户审阅候选包

本版在当前项目内通过 Blender MCP 制作。用途为造型和技术审阅，尚未由客户验收。

文件
exports/avatar.blend：完整可编辑源工程，保留高精度修改器、PBR材质、骨骼、物理和灯光／镜头。
exports/avatar-preview.glb：放松站姿的高精度PBR模型，包含绑定，不附带步行动画。
exports/avatar-preview.vrm：优化运行版VRM 1.0，T姿势，MToon材质、Humanoid和头发／裙摆SpringBone。
exports/avatar-runtime.blend：与VRM对应的运行版源工程。
textures/：实际使用的4张贴图；模型内部也已嵌入或打包。
previews/：本轮真实Blender渲染，含整体、面部、前／侧／后、手部及鞋袜近景。
打开“审阅图册.html”可离线浏览这些渲染。图册不运行3D或播放动作。
checksums.sha256：包内文件的SHA-256校验值。

已完成的细化
连续掌指网格与指甲，面颈连接和正常肤色，刘海／发根接缝与哑光发色，
衣物4K颜色和织物法线，领口、袖褶、纽扣、裙褶、袜纹与鞋带／鞋底细节。
本轮优先重建面部的额头—鼻梁／鼻翼—眼下／颧颊—口鼻—下颌连续曲面，
修正侧面口鼻区后缩；默认贴图已去腮红，清理了侧脸和下巴的图像投影残影。
保留了侧面和四十五度的修改前后灰模图，可在没有贴图干扰时查看实际几何。
运行版保留面部和手部拓扑，仅对通过表面偏差检查的其它网格减面，
按材质和物理行为合并对象；完整来源对应关系记录在导出报告内。

技术记录
源模型 {export["baseMeshes"]} 个网格，PBR GLB {export["posedPbrGlb"]["triangles"]:,} 三角面；
运行版 {export["runtimeVrm"]["meshes"]} 个网格、{export["runtimeVrm"]["triangles"]:,} 三角面，
{export["runtimeVrm"]["mtoon"]} 个MToon材质、{export["runtimeVrm"]["images"]} 张内嵌图片。
骨架共{export["humanoidBones"]}根（包含物理骨与末端），漏绑和未归一化顶点均为0。
{physics["chains"]}组物理链、{physics["joints"]}个物理关节、{physics["colliders"]}个碰撞体
通过{physics["frames"]}帧实际VRM物理ON/OFF响应检查。
几何、导出、源文件保全、物理检查详见同目录JSON报告。

客户最终交付前需确定
1. 实际使用软件／平台、设备、文件格式、性能预算及动作范围。
2. 是否需要眨眼、口型或表情；本轮保持固定中性表情。
3. 正式模型名、作者和使用／再分发条款。
4. 对造型、还原程度、材质和动作的视觉验收。

本版边界
""" + "\n".join("- " + item for item in limits) + """

原项目附带previews/vrm-viewer.html，可查看标准步行、暂停、T姿势与物理开关。
该预览依赖原项目的本地服务，未随本包分发第三方播放器和步行动作。
本包不包含供权重转移使用的素体资产，不附带Tripo服务或外部账号依赖。
正本.charakit和已确认参考图保留在原项目中；没有将Tripo流程标记为完成。
"""
    (EXPORTS / "交付说明.txt").write_text(readme)
    figures = "\n".join(
        f'<figure><img src="{html.escape(file)}" alt="{title}" loading="lazy">'
        f"<figcaption>{title}</figcaption></figure>"
        for title, file in zip(titles, preview_files)
    )
    gallery = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双马尾绿裙角色 · 审阅图册</title>
<style>
body{{margin:0;background:#f3f1ec;color:#252923;font:16px/1.7 system-ui,sans-serif}}
header,main,footer{{max-width:1100px;margin:auto;padding:36px 24px}}
header{{border-bottom:1px solid #d9dbd2}}h1{{font-size:34px;margin:0 0 12px}}
.tag{{font-size:13px;letter-spacing:.15em;color:#63715a}}p{{max-width:760px}}
main{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
figure{{margin:0;background:#fff;border:1px solid #d9dbd2;border-radius:10px;overflow:hidden}}
figure:first-child{{grid-column:1/-1}}img{{width:100%;display:block}}
figcaption{{padding:14px 18px}}footer{{font-size:14px;color:#596052}}
@media(max-width:700px){{main{{grid-template-columns:1fr}}h1{{font-size:27px}}}}
</style><header><div class="tag">CHARACTER REVIEW · BLENDER</div>
<h1>双马尾绿裙角色</h1><p>本轮细化后的真实模型渲染。此包供客户审阅，正式用途、表情需求和模型授权信息待确定。</p>
<a href="exports/交付说明.txt">交付说明</a></header><main>{figures}</main>
<footer>图像对应高精度PBR源文件。VRM运行版使用MToon，显示效果随目标程序的灯光与渲染设置变化。</footer></html>"""
    filenames = [
        "exports/avatar.blend", "exports/avatar-runtime.blend",
        "exports/avatar-preview.glb", "exports/avatar-preview.vrm",
        "exports/avatar-export-report.json", "exports/avatar-source-report.json",
        "exports/avatar-geometry-report.json", "exports/avatar-physics-report.json",
        "exports/avatar-spring-plan.json", "exports/delivery-manifest.json",
        "exports/交付说明.txt",
        "textures/face-neutral-basecolor.png", "textures/hair-charcoal-basecolor.png",
        "textures/clothing-basecolor.png", "textures/clothing-normal.png",
        *preview_files,
    ]
    if motion_report_path.exists():
        filenames.append("exports/avatar-walk-report.json")
    if (EXPORTS / "avatar-clearance-report.json").exists():
        filenames.append("exports/avatar-clearance-report.json")
    package = EXPORTS / "character-client-review.zip"
    checksums = []
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for name in filenames:
            path = PROJECT / name
            archive.write(path, name)
            checksums.append(f"{digest(path)}  {name}")
        gallery_bytes = gallery.encode()
        archive.writestr("审阅图册.html", gallery_bytes)
        checksums.append(f"{hashlib.sha256(gallery_bytes).hexdigest()}  审阅图册.html")
        archive.writestr("checksums.sha256", "\n".join(checksums) + "\n")
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for line in archive.read("checksums.sha256").decode().splitlines():
            expected, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    print(json.dumps({"package": str(package), "bytes": package.stat().st_size,
                      "files": len(filenames) + 2, "sha256": digest(package)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
