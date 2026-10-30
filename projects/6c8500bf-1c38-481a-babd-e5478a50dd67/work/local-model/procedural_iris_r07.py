"""Generate an original deterministic iris base-color texture, without Blender.

No source image pixels are read. Only the existing iris UV disk layout is kept.
Run from the workspace root using the Python executable in local configuration.
The output is a material color map: reflections remain the cornea shader's job.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

SEED = 2026102807
IRIS_RADIUS = 0.477
PUPIL_RADIUS = 0.145
TAU = math.tau


def smooth(a, b, value):
    t = max(0.0, min(1.0, (value - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


class PolarNoise:
    """Periodic angular value noise; no discontinuity across the U seam."""
    def __init__(self, angular_cells, radial_cells, rng):
        self.w = angular_cells
        self.h = radial_cells
        self.grid = [[rng.random() for _ in range(self.w)]
                     for _ in range(self.h + 1)]

    def __call__(self, angle, radius):
        x = (angle % 1.0) * self.w
        y = min(self.h - 0.00001, max(0.0, radius * self.h))
        xi, yi = int(x), int(y)
        fx, fy = x - xi, y - yi
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        row0, row1 = self.grid[yi], self.grid[yi + 1]
        nxt = (xi + 1) % self.w
        a = row0[xi] * (1.0 - fx) + row0[nxt] * fx
        b = row1[xi] * (1.0 - fx) + row1[nxt] * fx
        return a * (1.0 - fy) + b * fy


def chunk(kind, data):
    return (struct.pack('>I', len(data)) + kind + data
            + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))


def png_bytes(size, pixels):
    scanlines = b''.join(b'\x00' + pixels[y * size * 3:(y + 1) * size * 3]
                         for y in range(size))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'sRGB', b'\x00')
            + chunk(b'IDAT', zlib.compress(scanlines, 7))
            + chunk(b'IEND', b''))


def generate(size):
    rng = random.Random(SEED)
    broad = PolarNoise(19, 7, rng)
    drift = PolarNoise(49, 12, rng)
    strand = PolarNoise(239, 23, rng)
    fine = PolarNoise(919, 139, rng)
    tissue = PolarNoise(367, 307, rng)
    branching = PolarNoise(557, 73, rng)
    crypt_seed = [(rng.uniform(.39, .76), rng.uniform(.014, .046),
                   rng.uniform(.15, .28), rng.uniform(.40, .72))
                  for _ in range(83)]
    result = bytearray(size * size * 3)
    pupil = PUPIL_RADIUS / IRIS_RADIUS
    for py in range(size):
        vy = .5 - (py + .5) / size
        for px in range(size):
            vx = (px + .5) / size - .5
            radius = math.hypot(vx, vy) / IRIS_RADIUS
            offset = (py * size + px) * 3
            if radius > 1.02:
                color = (15, 11, 9)
            elif radius < pupil - .006:
                color = (2, 2, 2)
            else:
                angle = (math.atan2(vy, vx) / TAU) % 1.0
                b = broad(angle, radius)
                d = drift(angle, radius)
                warp = angle + .006 * (d - .5) + .002 * math.sin(radius * 24 + angle * TAU * 9)
                s = strand(warp, radius)
                branch = branching(warp, radius)
                f = fine(warp + .0015 * (s - .5) + .0014 * (branch - .5), radius)
                t = tissue(warp, radius)
                # Curved radial fibre bundles with fine bifurcation and local breaks.
                veins = smooth(.46, .84, .62 * f + .38 * branch)
                detail = .30 * s + .42 * f + .28 * t
                strength = .50 + .43 * detail + .13 * b + .10 * veins
                # Irregular collarette and soft, elongated crypt openings.
                collar = .468 + .026 * math.sin(angle * TAU * 17) + .035 * (b - .5)
                rim = math.exp(-((radius - collar) / .018) ** 2)
                slot = (angle * 83 + .23 * math.sin(radius * 37)) % 83
                idx = int(slot)
                cr, crh, caw, darkness = crypt_seed[idx]
                ca = (slot - idx - .5) / caw
                rr = (radius - cr) / crh
                crypt = math.exp(-(ca * ca + rr * rr) * 1.9)
                strength *= 1.0 - darkness * crypt
                strength += .06 * rim * (.35 + .65 * branch)
                # Moderate concentric contraction furrows in the outer iris.
                furrow = (.5 + .5 * math.cos(radius * 103 + 3.6 * b + 1.7 * d)) ** 8
                strength *= 1.0 - .025 * furrow * smooth(.70, .86, radius) * b
                # Dark brown, varied pigment; no painted lights or catchlights.
                golden = (1.0 - smooth(.50, .74, radius)) * .12 + .12 * (b - .5)
                rgb = [94 * strength * (1 + golden),
                       65 * strength * (1 + .50 * golden),
                       43 * strength * (1 - .25 * golden)]
                limbus = smooth(.925, .996, radius)
                rgb = [v * (1 - .76 * limbus) for v in rgb]
                pupil_blend = smooth(pupil - .004, pupil + .011 + .005 * (s - .5), radius)
                outer_blend = smooth(.995, 1.012, radius)
                color = tuple(round(max(0, min(255,
                    ((v * pupil_blend + 2 * (1 - pupil_blend)) * (1 - outer_blend)
                     + bg * outer_blend)))) for v, bg in zip(rgb, (15, 11, 9)))
            result[offset:offset + 3] = bytes(color)
    # Irregular pigment fibres are independent curved strokes rather than a
    # repeated angular lattice. Their opacity varies with radial tissue bands.
    rng = random.Random(SEED + 600)
    iris_px = size * IRIS_RADIUS
    for strand_index in range(2700):
        theta = rng.random() * TAU
        start = rng.uniform(pupil + .012, .68)
        end = rng.uniform(max(.71, start + .15), .990)
        phase = rng.random() * TAU
        phase2 = rng.random() * TAU
        phase3 = rng.random() * TAU
        amplitude = rng.uniform(.001, .007)
        frequency = rng.uniform(19, 43)
        width = rng.uniform(.42, .82) * size / 1536
        light = strand_index % 4 != 0
        tone = rng.uniform(.82, 1.24)
        pigment = tuple(c * tone for c in ((139, 99, 64) if light else (32, 20, 13)))
        alpha = rng.uniform(.20, .62)
        steps = math.ceil((end - start) * iris_px * 1.35)
        for step in range(steps + 1):
            fraction = step / steps
            r = start + (end - start) * fraction
            turn = theta + amplitude * math.sin(frequency * r + phase)
            turn += .0025 * math.sin(r * 103 + phase2) + .005 * math.sin(r * 14 + phase3)
            x = size * (.5 + IRIS_RADIUS * r * math.cos(turn)) - .5
            y = size * (.5 - IRIS_RADIUS * r * math.sin(turn)) - .5
            fade = smooth(0, .10, fraction) * (1 - smooth(.86, 1, fraction))
            fade *= (1 - .88 * smooth(.925, .997, r))
            fade *= .56 + .44 * math.sin(r * 81 + phase2) ** 2
            ix, iy = int(x), int(y)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    xx, yy = ix + ox, iy + oy
                    if not (0 <= xx < size and 0 <= yy < size):
                        continue
                    distance = math.hypot(xx - x, yy - y)
                    cover = max(0, min(1, width + .55 - distance)) * alpha * fade
                    if cover <= 0:
                        continue
                    at = (yy * size + xx) * 3
                    for ch in range(3):
                        result[at + ch] = round(result[at + ch] * (1 - cover) + pigment[ch] * cover)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=1536)
    parser.add_argument('--project')
    args = parser.parse_args()
    if not 256 <= args.size <= 4096:
        parser.error('size must be 256..4096')
    project = project_path(args.project)
    folder = project / 'textures/procedural-r07'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'iris-brown-r07-v3.png'
    payload = png_bytes(args.size, generate(args.size))
    digest = hashlib.sha256(payload).hexdigest()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise FileExistsError('Existing output differs. Preserve it and choose a new revision.')
    target.write_bytes(payload)
    script = Path(__file__).resolve()
    record = {
        'schemaVersion': 1, 'revision': 'r07-v3', 'createdDate': '2026-10-28',
        'origin': '当前工作区确定性程序生成；不读取、复制或转换任何外部图像像素。',
        'author': '当前工作区制作，未指定个人署名',
        'license': '未另行指定公开许可证；不宣称 CC0。',
        'externalImageInputs': [], 'seed': SEED,
        'generator': script.relative_to(project).as_posix(),
        'generatorSha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'requires': 'Python 3 标准库与 pipeline.context；不调用 Blender、imagegen 或网络。',
        'image': {'path': target.relative_to(project).as_posix(), 'sha256': digest,
                  'bytes': len(payload), 'width': args.size, 'height': args.size,
                  'channels': 'RGB 8-bit', 'colorSpace': 'sRGB', 'purpose': 'iris base color'},
        'uv': {'center': [.5, .5], 'irisRadius': IRIS_RADIUS,
               'pupilRadius': PUPIL_RADIUS, 'diskRadius': .5,
               'layoutEvidence': 'work/local-model/reconstruction_groom_r01.py::add_textured_iris',
               'note': '保持现有 R01 Textured iris 圆盘 UV；旧图仅用于布局观察，没有取样旧像素。'},
        'features': ['黑色瞳孔', '棕色多尺度弯曲径向纤维', '不规则虹膜领圈与隐窝',
                     '柔和外周收缩纹', '深色角膜缘环', '无烘入反光/高光'],
        'integrationStatus': '生成完成，待整合者实际加载与渲染；未替换模型或旧图片。',
        'integrationNotes': [
            '作为 sRGB Base Color 输入，保持现有 UV。不要作为 Non-Color 或法线图加载。',
            '检查旧深棕色节点的乘色，避免新图被重复过度压暗。',
            '角膜透明、高光、瞳孔深度仍由几何与材质完成，不能由这张颜色图代替。',
            '左右眼可使用同图但会共享虹膜细节；本轮没有宣称照片扫描或 AAA 验收。'],
        'legacyTexture': {'path': 'textures/iris-brown-realism.png', 'status': '保留，不覆盖',
                          'origin': 'unknown：全 local-model 与 assets 追溯仅找到加载、继承，未找到授权来源。'}
    }
    (folder / 'iris-source-r07.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'image': str(target), 'sha256': digest, 'bytes': len(payload),
                      'sourceRecord': str(folder / 'iris-source-r07.json')}))


if __name__ == '__main__':
    main()
