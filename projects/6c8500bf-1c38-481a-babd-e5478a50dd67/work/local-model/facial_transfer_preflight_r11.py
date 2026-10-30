"""Offline, standard-library-only preparation for six facial transfer units.

Reads recorded observations and NPZ headers. Does not import Blender, download
assets, change geometry, install packages, or certify current blend contents.
Writes one explicitly new JSON report below work/reconstruction-r02.
"""
import argparse
import ast
import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

P = project_path()
R = 'work/reconstruction-r02/'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def path(relative):
    p = (P / relative).resolve()
    p.relative_to(P.resolve())
    return p


def read(relative):
    return json.loads(path(relative).read_text(encoding='utf-8-sig'))


def npz_headers(relative):
    """Read shape/dtype metadata only; never unpickle or allocate the arrays."""
    p = path(relative)
    if not p.exists():
        return {'path': relative, 'exists': False}
    arrays = {}
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            if not info.filename.endswith('.npy'):
                continue
            with z.open(info) as f:
                if f.read(6) != b'\x93NUMPY':
                    raise ValueError('Invalid NPY magic')
                version = tuple(f.read(2))
                if version not in ((1, 0), (2, 0), (3, 0)):
                    raise ValueError('Unsupported NPY version')
                size = 2 if version[0] == 1 else 4
                length = struct.unpack('<H' if size == 2 else '<I', f.read(size))[0]
                if length > 1048576:
                    raise ValueError('Unbounded NPY header')
                header = ast.literal_eval(f.read(length).decode('utf-8' if version[0] == 3 else 'latin1'))
                arrays[info.filename[:-4]] = {'shape': list(header['shape']), 'dtype': header['descr']}
    return {'path': relative, 'exists': True, 'bytes': p.stat().st_size, 'sha256': sha(p), 'arrays': arrays}


def prepare():
    state = read('project-state.json')['reconstructionR11']
    latest = state['latestIndependentCandidate']
    rest_path = 'previews/reconstruction-r02/natural-rest-overall-r11/render-report.json'
    rest = read(rest_path)
    land_path = R + 'female-face-landmarks-r11.json'
    land = read(land_path)
    seal_path = R + 'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.json'
    seal = read(seal_path)
    keys = rest['savedKeyValues']
    local_targets = sorted(str(p.relative_to(P)).replace('\\', '/') for p in (P / 'assets').rglob('*.target'))
    local_correspondence = sorted(str(p.relative_to(P)).replace('\\', '/') for p in (P / 'assets').rglob('*.mhclo'))
    arrays = [npz_headers(R + name) for name in (
        'female-basemesh-actual-r11.npz',
        'adult-head-bind-observation-r11.npz',
        'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz')]
    units = [
        {'name': 'browInnerUp', 'purpose': '双侧内眉上提，验证眉区跨度和中线连续性', 'reuse': '眉毛已有三角面附着流程；无已证实同义形态键', 'regions': ['brow_L', 'brow_R', 'glabella'], 'firstStrengths': [0, 0.25, 0.5]},
        {'name': 'browDownLeft', 'purpose': '角色解剖左眉下压，验证单侧与上眼睑分区', 'reuse': '仅复用附着验证；不能把眉毛整物体平移当皮肤形态', 'regions': ['brow_L', 'glabella_L'], 'firstStrengths': [0, 0.25, 0.5]},
        {'name': 'eyeBlinkLeft', 'purpose': '与现有左眼原生眨眼比较，不覆盖原键', 'reuse': 'R11 blink L、R11 blink closure L 及其路径键作为现有基准；路径键由控制器驱动，不手动相加', 'regions': ['upper_lid_L', 'lower_lid_L', 'canthi_L', 'inner_lid_L'], 'firstStrengths': [0, 0.25, 0.5, 0.75, 1]},
        {'name': 'eyeSquintLeft', 'purpose': '收紧左眼睑/上颊，区别于完全闭眼和微笑下眼睑', 'reuse': 'R11 smile lower lid L、R11 warm smile lower lid L 可做局部参考，不能直接认定为 ARKit squint', 'regions': ['lower_lid_L', 'upper_cheek_L', 'outer_eye_L'], 'firstStrengths': [0, 0.25, 0.5]},
        {'name': 'mouthSmileLeft', 'purpose': '左侧唇角上提，检查双唇接触与中线无串扰', 'reuse': 'R11 natural soft smile / R11 warm closed smile 仅作对照；不得与 donor 单元叠加且不可直接将整脸微笑硬切半边', 'regions': ['commissure_L', 'upper_vermilion_L', 'lower_vermilion_L', 'cheek_L'], 'firstStrengths': [0, 0.25, 0.5]},
        {'name': 'mouthFunnel', 'purpose': '双唇漏斗形收拢，检验口唇环的跨拓扑对应', 'reuse': '复用40点唇缘环、120点唇红带分区和当前自然闭口基准；没有已证实同义键', 'regions': ['upper_vermilion', 'lower_vermilion', 'upper_inner_lip', 'lower_inner_lip', 'commissures'], 'firstStrengths': [0, 0.15, 0.3], 'restriction': '首轮下颌固定；不得声称发音或张嘴组合通过'}
    ]
    for u in units:
        u['donorRelativePath'] = 'assets/faceunits01/targets/faceunits/' + u['name'] + '.target'
        u['candidateKey'] = 'R11 pilot MPFB ' + u['name']
        u['defaultValue'] = 0
    controls = ['R11 blink L', 'R11 blink R', 'R11 blink closure L', 'R11 blink closure R',
                'R11 gentle smile', 'R11 natural soft smile', 'R11 warm closed smile',
                'R11 smile lower lid L', 'R11 smile lower lid R',
                'R11 warm smile lower lid L', 'R11 warm smile lower lid R', 'R11 jaw open']
    anatomy = {
        side: {'innerCanthus': row['innerCanthusCandidate'], 'outerCanthus': row['outerCanthusCandidate'],
               'upperIds': row['orderedUpperInnerToOuter'], 'lowerIds': row['orderedLowerInnerToOuter']}
        for side, row in land['eyes'].items()
    }
    return {
        'schemaVersion': 1,
        'recordedDate': '2026-10-29',
        'scope': '离线准备完成；没有执行表情迁移，没有启动 Blender，没有新增头、下载资产或修改主源。',
        'nonBlocking': '自然默认脸修形继续；此试验在中性脸冻结后使用独立副本，当前缺项不作为阻塞修形的条件。',
        'latestCandidateRecordedByProjectState': latest,
        'currentBlendIndependentlyRead': False,
        'neutralEvidence': {'path': rest_path, 'sha256': sha(path(rest_path)), 'source': rest['source'], 'sourceSha256': rest['sourceSha256'],
            'sameAsLatestRecordedCandidate': rest['sourceSha256'] == latest['sha256'],
            'warning': '这是已保存29e3自然表情的观察快照；不能冒充当前后继模型的完整数组。正式试验先对当前源重新捕获控制器、cage和实际L3曲面。',
            'recordedKeyValues': keys,
            'preserveDefaultRule': '新试验键全部0；已有造型键保持保存值，尤其 relaxed lip seal=1、refined outer eye corners=0；不把所有键归零，不强设warm smile。'},
        'localInventory': {'assetTargetFiles': local_targets, 'assetMhcloFiles': local_correspondence,
            'makehumanSkinResearchIsNotExpressionSource': 'assets/makehuman-skin-research-r11 现有内容为材质/贴图，不能代替 MakeHuman 中性顶点序或面部 targets。',
            'topology': {'path': land_path, 'sha256': sha(path(land_path)), 'vertexCount': land['originalVertexCount'],
                'indexSpace': land['indexSpace'], 'eyeRegions': anatomy,
                'lipRimIds': land['mouth']['openingRimCandidate']['sourceVertexIndices'],
                'upperLipRimIds': land['mouth']['openingRimCandidate']['orderedUpperLeftToRight'],
                'lowerLipRimIds': land['mouth']['openingRimCandidate']['orderedLowerLeftToRight'],
                'lipStripVertexCount': len(land['mouth']['vermilionStripVertexIndices']),
                'limitations': '完整女性原顶点序10582点，非旧R01头；post-Multires Mask保持原索引。不要将旧 head-topology.json 或雕刻后L3点序直接用于cage映射。'},
            'arrayHeaderInventory': arrays},
        'sixUnits': units,
        'existingControls': [{'name': n, 'presentInNeutralSnapshot': n in keys, 'savedValue': keys.get(n)} for n in controls],
        'jawStatus': {'sealReport': seal_path, 'reportSha256': sha(path(seal_path)), 'jawAttenuation': seal['jawAttenuation'],
            'conclusion': '自然闭口报告明确 jawKeyPresent=false；已有 jaw 脚本/旧审阅计划不证明最新候选已含 jaw。首轮排除 jawOpen，后续另接 jaw 衰减并测0/.03/.06/.09/.12/1和口腔接触。'},
        'availableLocalBuildingBlocks': [
            {'path': 'work/local-model/face_fibre_binding_r11_v2.py', 'reuse': '纯NumPy triangle_frames/barycentric 思路与现有眉睫表面附着验证', 'boundary': '现有映射是眉睫到同一头的L3面，不是 MakeHuman 到 HMB 的跨头对应；不可直接复用其索引。'},
            {'path': 'work/local-model/relaxed_lip_seal_surface_r11.py', 'reuse': '纯NumPy唇部截面与接触分析，帮助检查默认闭口', 'boundary': '这不是通用表情转移器，也不代表已有口角交叉已解决。'},
            {'path': 'work/local-model/adult_expression_review_r11_v3.py', 'reuse': '明确控制值、同机位多态真实渲染、源哈希检查', 'boundary': '将来由整合者串行运行；本准备不启动该脚本。'},
            {'path': 'work/local-model/facial_transfer_preflight_r11.py', 'reuse': '本次新增可离线重跑的清单/NPZ头检查入口', 'boundary': '仅生成准备报告，不生成或注入任何表情形态。'}
        ],
        'requiredCorrespondence': {
            'donor': '固定MakeHuman中性base网格的版本、顶点序、面、坐标单位和六个targets的SHA256。targets只有顶点编号+三维差值，本地目前缺少该中性参考与六个文件。',
            'recipient': '当前保存源的Basis、已保存静态造型混合cage、实际L3中性曲面、modifier顺序和全部默认值。形态层与Geometry Nodes层分别捕获，不能只读Basis推断自然脸。',
            'landmarks': '两侧内外眼角、上下眼睑路径、上下唇缘/唇红环、唇角、鼻翼/鼻底/鼻梁、眉头/眉峰/眉尾、中线及下巴。现有HMB眼口索引可用，MakeHuman对应和眉鼻标志待建立。左右按角色解剖定义核对，不能按图像左右猜。',
            'regionLabels': ['brow_L', 'brow_R', 'upper_lid_L', 'lower_lid_L', 'inner_lid_L', 'canthi_L', 'upper_vermilion', 'lower_vermilion', 'upper_inner_lip', 'lower_inner_lip', 'commissures', 'cheeks', 'protected_nose', 'protected_neck'],
            'mappingRows': ['recipient_vertex_id', 'donor_triangle_vertex_ids[3]', 'barycentric_weights[3]', 'region_id', 'donor_to_recipient_delta_basis[3,3]', 'support_weight', 'distance', 'normal_alignment'],
            'method': '先相似变换和标志约束拟合只用于建立对应；在同解剖分区内取三角面和重心坐标，传输 donor 的单元差值而非替换当前中性位置。向量变换只含线性旋转/尺度，不含平移；局部尺度需受约束并报告，不能随意拉伸。',
            'noNearestAcrossContact': '上/下唇、口内/外皮、眼睑/眼球单独分区；禁止无标签全局最近点投射。嘴角共界显式混合，不能把双唇焊成一圈。',
            'storage': '计划为独立 correspondence.npz + manifest.json，保存两端拓扑/中性SHA、单位、分区与未匹配顶点清单。当前没有该映射，报告不生成假权重。',
            'identityEquation': 'candidate_position = current_neutral_position + transported_expression_delta；强度0时增量必须精确0。额外变形应落在固定10582点cage，而不是替换中性模型；实际L3/GN结果须另检查。',
            'shapeKeyStorage': '上述是可见结果方程，不是新key的绝对坐标。若新key relative_key=Basis，其key.data应为Basis坐标+delta，而非current_neutral+delta；后者会将已启用静态造型重复相加。保存姿态保持原默认值，新增pilot键为0。',
            'geometryLayerRisk': '当前闭口等为零位移shape key控制L3字段，部分字段在固定几何空间定义；cage新形态可能改变其有效域。必须观察modifier顺序和组合，不能仅凭cage差值通过。'
        },
        'license': {'assetLicense': 'CC0-1.0：仅所核官方 faceunits01 / extra-targets 数据',
            'codeLicense': '如直接复制MPFB代码则GPLv3；本清单入口为本地原创标准库逻辑，未复制MPFB代码。',
            'sourceCommit': '7eaba3453134385bb5ea9811ef0b33b85b4b556d',
            'urls': ['https://static.makehumancommunity.org/assets/assetpacks/index.html#functional-asset-packs',
                     'https://github.com/makehumancommunity/extra-targets/blob/7eaba3453134385bb5ea9811ef0b33b85b4b556d/LICENSE',
                     'https://github.com/makehumancommunity/mpfb2/blob/master/LICENSE.md'],
            'sourceLockNotDownloadedArtifactProof': True},
        'readiness': {'offlinePreflightRunnableNow': True, 'offlineTransferRunnableNow': False,
            'blockersForTransferOnly': ['六个CC0 target文件与对应MakeHuman中性顶点序尚不在本地', '当前自然脸同源cage/L3数组尚需整合者在冻结节点捕获', '两种拓扑之间的分区三角对应不存在'],
            'noNeed': ['新头候选', '全套MPFB贴图/服装或完整场景包', 'Tripo', '并发重渲染', '为本次清单安装插件'],
            'lightweightFutureFetch': '将来只取固定提交下六个target及确认为同一顶点序的小型中性base参考，并逐项留许可证/哈希；本轮不下载。',
            'nextExecutableEntry': '用config/local.win32.json的pythonExecutable运行 work/local-model/facial_transfer_preflight_r11.py --output work/reconstruction-r02/<新的报告名>.json；路径由pipeline.context解析。'},
        'minimalValidation': {
            'neutral': '默认新键全0，当前源中性几何、已有键值、材质/头发/骨架不变；cage逐点零增量，实际L3中性同源一致。',
            'pilot': '先分区映射审阅，再逐单元低强度；blink额外查全闭合。嘴唇保留既有交叉缺陷记录，不把旧缺陷归零。',
            'combinations': ['browInnerUp=0.25 + eyeBlinkLeft=0.5', 'eyeSquintLeft=0.25 + mouthSmileLeft=0.25', 'mouthFunnel=0.15 + mouthSmileLeft=0.15（仅检查冲突，不作为自然表情）'],
            'samplingOrder': '先离线有限性/零基准/分区越界检查，再由整合者串行真实曲面检查和少量同光同机位图。未通过中间值不增加单元数。',
            'abort': '若上/下唇或睑/眼混绑、静态自然脸变动、眉睫脱离或要求整体换头，停止本试验，保留原模型；把未通过原因交回整合者。'
        }
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = path(args.output)
    out.relative_to(path(R))
    if out.suffix != '.json' or out.exists():
        raise ValueError('Use a new JSON path under work/reconstruction-r02')
    result = prepare()
    result['helperSha256'] = sha(Path(__file__))
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'report': str(out.relative_to(P)).replace('\\', '/'), 'sha256': sha(out),
                      'units': len(result['sixUnits']), 'readiness': result['readiness']['offlineTransferRunnableNow']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
