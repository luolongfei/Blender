"""Package already-saved work after an explicit user stop; never edit a Blend."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path()
DEST = P / 'exports/stopped-handoff-2026-10-29'
ZIP = P / 'exports/stopped-handoff-2026-10-29.zip'
EXPECTED = '8fd87255aa7576eeaa486cad396773350b608b27b5e57914332165afa67eb2d9'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

state_path = P / 'project-state.json'
before_state = state_path.read_bytes()
state = json.loads(before_state.decode('utf-8-sig'))
source_row = state['reconstructionR11']['latestIndependentCandidate']
source = P / source_row['path']
if source_row['sha256'] != EXPECTED or sha(source) != EXPECTED:
    raise RuntimeError('Saved source differs from the explicitly selected stopped handoff')
if ZIP.exists() or (DEST / 'current-editable.blend').exists():
    raise RuntimeError('Preserve any prior completed handoff')
audit_path = DEST / 'dependency-audit.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
if audit['sourceSha256'] != EXPECTED or audit['sourceSaved']:
    raise RuntimeError('Dependency audit does not match source')

records = []
def copy_file(src, relative, meaning):
    src = Path(src)
    dest = DEST / relative
    if dest.exists():
        raise RuntimeError('Refuse to overwrite handoff file: ' + str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    original_sha = sha(src)
    if sha(dest) != original_sha:
        raise RuntimeError('Copied file checksum mismatch')
    records.append({'path': dest.relative_to(DEST).as_posix(), 'source': str(src),
                    'sha256': original_sha, 'bytes': dest.stat().st_size, 'meaning': meaning})

copy_file(source, 'current-editable.blend', 'Latest successfully saved editable R11 source; byte-exact copy; not accepted final quality')
cache = source.parent / 'blendcache_reference-cloth-body-r04'
if cache.is_dir():
    for item in sorted(cache.rglob('*')):
        if item.is_file():
            copy_file(item, Path(cache.name) / item.relative_to(cache), 'Preserved historical cloth cache; not validated for the current reconstruction')

selected = [
 ('references/user-original-portrait-2026-10-28.png', 'references/original-design.png', 'User original design'),
 ('references/generated/natural-default-expression-reference-r11.png', 'references/generated-natural-expression.png', 'Generated art reference, not a model render'),
 ('references/generated/natural-default-expression-reference-r11.json', 'references/generated-natural-expression.json', 'Generated reference provenance'),
 ('references/generated/natural-face-profile-reference-r11.png', 'references/generated-face-profile.png', 'Generated art reference, not a model render'),
 ('references/generated/natural-face-profile-reference-r11.json', 'references/generated-face-profile.json', 'Generated reference provenance'),
 ('previews/reconstruction-r02/reference-ties-v3-r11/full-groom-front.png', 'previews/saved-0df7-before-band-fit-front.png', 'Actual prior 0df7 source render; the later four-band fit is not shown'),
 ('previews/reconstruction-r02/reference-ties-v3-r11/full-groom-quarter.png', 'previews/saved-0df7-before-band-fit-quarter.png', 'Actual prior 0df7 source render; the later four-band fit is not shown'),
 ('previews/reconstruction-r02/relaxed-mouth-corners-preview-r11/front-1.png', 'unmerged-previews/relaxed-mouth-corners.png', 'Unmerged in-memory mouth candidate; absent from current-editable.blend'),
 ('previews/reconstruction-r02/upper-lash-neutral-preview-r11/after.png', 'unmerged-previews/upper-lashes.png', 'Unmerged in-memory lash candidate; absent from current-editable.blend'),
 ('previews/reconstruction-r02/outer-eye-continuous-contour-diagnostic-front-r11/frame-0001.png', 'unmerged-previews/outer-eye-diagnostic.png', 'Completed diagnostic before stop; not reviewed or adopted; absent from current-editable.blend'),
 ('work/reconstruction-r02/reference-adult-fitted-bands-r11.band-fit.json', 'reports/saved-band-fit.json', 'Report for the delivered saved source'),
 ('work/reconstruction-r02/reference-current-band-native-proof-r11.json', 'reports/band-native-proof.json', 'Static sampled band-fit proof; not dynamic acceptance'),
 ('work/reconstruction-r02/asset-provenance-verification-r11.json', 'sources/asset-provenance-verification.json', 'Existing asset provenance and license findings, including limitations'),
 ('assets/blender-hair-styles-bystedt-2023/source-record.json', 'sources/hair-source-record.json', 'Existing hair asset source and attribution record'),
 ('work/reconstruction-r02/rig-frame-one-channel-mechanism-r11.json', 'reports/frame-one-channel-note.json', 'Known evaluation-state discrepancy and observed mechanism'),
]
for relative, dest, meaning in selected:
    copy_file(P / relative, dest, meaning)
copy_file(ROOT / 'toolchain.lock.json', 'toolchain.lock.json', 'Pinned toolchain')

for entry in audit['externalPaths']:
    path = Path(entry['path'])
    try:
        if path.is_dir():
            entry['pathKind'] = 'directory'
            entry['exists'] = True
            entry['bytes'] = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        elif path.is_file():
            entry['pathKind'] = 'file'
        else:
            entry['pathKind'] = 'unresolved'
    except OSError:
        entry['pathKind'] = 'unresolved'
audit['packagingNote'] = 'All 17 image datablocks are packed and no linked libraries remain. blend_paths also includes historical asset-origin references and brush locators; their presence is not evidence of a linked model dependency. The existing 301-file cloth cache is copied beside the Blend but is not current dynamic validation.'
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
state['reconstructionR11']['status'] = 'paused-by-user; dissatisfied-with-appearance; existing-work-handoff-only'
state['reconstructionR11']['stoppedHandoff'] = {
    'recordedAt': now, 'reason': 'User requested immediate stop and delivery of existing work due to dissatisfaction.',
    'source': source_row, 'directory': 'exports/stopped-handoff-2026-10-29',
    'archive': 'exports/stopped-handoff-2026-10-29.zip', 'masterUpdated': False,
    'formalExportsUpdated': False, 'appearanceAccepted': False, 'productionContinued': False,
    'notMerged': ['relaxed mouth corners', 'new upper lashes', 'continuous outer-eye diagnostic',
                  'crown flow and 14 ear-side tips', 'nasal refinement', 'new leg/foot/sock/shoe candidates']}
if state_path.read_bytes() != before_state:
    raise RuntimeError('Concurrent project-state update')
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
copy_file(state_path, 'project-state-at-stop.json', 'Exact project state at user-requested stop')

readme = '''停止制作后的现状交付
2026-10-29

已按要求停止制作。用户对当前效果不满意；本包是现状交接，不是已经达到照片级/AAA要求或获得外观验收的成品。

打开 current-editable.blend，即最后成功保存的 R11 可编辑副本。建议使用 Blender 5.2.3 LTS；原项目使用 VRM Add-on 4.10.2。该文件与原始 8fd87255 版本逐字节一致，没有为打包修改模型、材质、姿态或表情。17张图像素材全部内嵌，无链接库。旧衣物缓存已按相对目录一并复制，但它不能证明新角色的衣物或动作通过验证。

已保存内容：当前头脸与轻妆肤色、原设计黑色低双马尾、扎位与四个发圈贴合、新手部及此前已经保留的局部眼口修改。

未合入内容：最新放松嘴角、新上睫、连续外眼角诊断、头顶毛流与耳侧末梢修正、鼻部方向、新腿足鞋袜。unmerged-previews 中的图片只是内存试验，打开模型不会自动出现这些修改。所有对应脚本、数组和历史副本仍保留在原项目。

previews 中两张图来自前一保存版本0df7，尚未显示最后四个发圈的贴合变化；本次停止后没有另做渲染。references 中明确标为 generated 的图是生成参考，不是3D成果，工具实际模型版本未确认。

未完成事项：面部自然度与精致度、头发自然垂坠、眼口组合和视线、新头颈绑定、口腔/说话口型、身体与鞋袜适配、衣物及头发物理、完整动作、目标软件和实时性能验证。不要把旧动作、缓存或旧VRM的检查结果当作本版通过。

本包没有提供与8fd87255对应的新VRM。原 exports/avatar.blend 及既有VRM/GLB属于旧阶段，保留在原工程，未覆盖，也没有冒充为本包对应输出。

sources 保留既有资产来源与许可核对记录，包括尚未确定的边界；toolchain.lock.json 是版本记录。原工程和全部待合并副本均保留；本地打包与校验不等于已完成异机备份。

sha256-manifest.json 为包内文件校验。原工程位置见 project-state-at-stop.json 和 source-location.txt。
'''
(DEST / '交付说明.txt').write_text(readme, encoding='utf-8-sig')
(DEST / 'source-location.txt').write_text('Workspace: ' + str(ROOT) + '\nProject: ' + str(P) + '\nSaved source: ' + str(source) + '\nSHA256: ' + EXPECTED + '\n', encoding='utf-8')
(DEST / 'copy-provenance.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
manifest = [{'path': f.relative_to(DEST).as_posix(), 'bytes': f.stat().st_size, 'sha256': sha(f)}
            for f in sorted(DEST.rglob('*')) if f.is_file()]
(DEST / 'sha256-manifest.json').write_text(json.dumps({'createdAt': now, 'sourceSha256': EXPECTED, 'files': manifest}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with zipfile.ZipFile(ZIP, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
    for f in sorted(DEST.rglob('*')):
        if f.is_file():
            archive.write(f, f.relative_to(DEST).as_posix())
with zipfile.ZipFile(ZIP) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Archive CRC check failed')
    with archive.open('current-editable.blend') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != EXPECTED:
            raise RuntimeError('Archived editable source mismatch')
if sha(source) != EXPECTED:
    raise RuntimeError('Original saved source changed during packaging')
receipt = {'archive': str(ZIP), 'bytes': ZIP.stat().st_size, 'sha256': sha(ZIP),
           'files': len(manifest) + 1, 'archiveCrcVerified': True,
           'archivedBlendSha256Verified': EXPECTED, 'productionContinued': False,
           'appearanceAccepted': False, 'remoteBackupClaimed': False}
ZIP.with_suffix('.verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False), flush=True)
