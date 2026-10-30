/**
 * Read-only GLB 2.0 delivery audit. No Blender, external image reads or dependencies.
 * From the workspace root:
 * node Character_Rebuild/scripts/validate_glb.mjs Character_Rebuild/character-rebuilt-textured.glb --report Character_Rebuild/reports/final-glb-validation.json
 * Exit 0: these structural checks passed; exit 1: failed; exit 2: CLI/read failure.
 * Passing is not visual acceptance, a topology/collision test, or a Blender reimport.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {workspacePath, workspaceRoot} from '../../pipeline/context.mjs';

const COMPONENT = {
  5120: [1, 'readInt8', -128, 127], 5121: [1, 'readUInt8', 0, 255],
  5122: [2, 'readInt16LE', -32768, 32767], 5123: [2, 'readUInt16LE', 0, 65535],
  5125: [4, 'readUInt32LE', 0, 4294967295], 5126: [4, 'readFloatLE', null, null],
};
const WIDTH = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16};
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const finite = values => Array.isArray(values) && values.every(Number.isFinite);
const emptyBounds = () => ({min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity]});
function extend(bounds, v) { for (let k = 0; k < 3; ++k) { bounds.min[k] = Math.min(bounds.min[k], v[k]); bounds.max[k] = Math.max(bounds.max[k], v[k]); } }
function finishBounds(bounds) { return bounds.min.every(Number.isFinite) ? {...bounds, size: bounds.max.map((n, k) => n - bounds.min[k])} : null; }

function imageSize(b) {
  if (b.length >= 24 && b.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])))
    return {format: 'png', width: b.readUInt32BE(16), height: b.readUInt32BE(20)};
  if (b.length >= 4 && b.readUInt16BE(0) === 0xffd8) {
    let p = 2;
    while (p + 3 < b.length) {
      if (b[p++] !== 0xff) continue;
      while (b[p] === 0xff) ++p;
      const marker = b[p++];
      if (marker === 0xd9 || marker === 0xda) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (p + 2 > b.length) break;
      const length = b.readUInt16BE(p);
      if (length < 2 || p + length > b.length) break;
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker) && length >= 7)
        return {format: 'jpeg', width: b.readUInt16BE(p + 5), height: b.readUInt16BE(p + 3)};
      p += length;
    }
  }
  if (b.length >= 30 && b.toString('ascii', 0, 4) === 'RIFF' && b.toString('ascii', 8, 12) === 'WEBP') {
    const kind = b.toString('ascii', 12, 16);
    if (kind === 'VP8X') return {format: 'webp', width: 1 + b.readUIntLE(24, 3), height: 1 + b.readUIntLE(27, 3)};
    if (kind === 'VP8 ' && b[23] === 0x9d && b[24] === 0x01 && b[25] === 0x2a)
      return {format: 'webp', width: b.readUInt16LE(26) & 0x3fff, height: b.readUInt16LE(28) & 0x3fff};
    if (kind === 'VP8L' && b[20] === 0x2f) {
      const v = b.readUInt32LE(21); return {format: 'webp', width: (v & 0x3fff) + 1, height: ((v >>> 14) & 0x3fff) + 1};
    }
  }
  return null;
}

export function validateGlbBuffer(bytes, inputLabel = '<buffer>') {
  const errors = [], warnings = [];
  const report = {
    schemaVersion: 1, input: inputLabel, bytes: bytes.length, sha256: hash(bytes),
    checkedAt: new Date().toISOString(), errors, warnings,
    scope: 'GLB structure, finite numeric attributes, declared PBR and embedded image references. Not a full Khronos validator.',
    visualAcceptance: false, blenderReimportVerified: false,
    unchecked: ['Visual likeness and texture appearance', 'Five-finger geometry and hair silhouette', 'Self-intersections and manifold topology', 'Physical scale and orientation against references', 'Compressed geometry decoding', 'Image pixel decoding and texture color-space appearance'],
  };
  const fail = message => { throw new Error(message); };
  const must = (ok, message) => { if (!ok) fail(message); };
  try {
    must(bytes.length >= 20, 'File is too short for GLB.');
    must(bytes.readUInt32LE(0) === 0x46546c67, 'GLB magic is invalid.');
    must(bytes.readUInt32LE(4) === 2, 'Only GLB version 2 is supported.');
    must(bytes.readUInt32LE(8) === bytes.length, 'GLB header length differs from actual file length.');
    const chunks = [];
    for (let p = 12; p < bytes.length;) {
      must(p + 8 <= bytes.length, 'Truncated chunk header.');
      const length = bytes.readUInt32LE(p), type = bytes.readUInt32LE(p + 4);
      must(length % 4 === 0, 'Chunk length is not 4-byte aligned.');
      must(p + 8 + length <= bytes.length, 'Chunk exceeds file bounds.');
      chunks.push({type, data: bytes.subarray(p + 8, p + 8 + length)}); p += 8 + length;
    }
    must(chunks[0]?.type === 0x4e4f534a, 'First GLB chunk must be JSON.');
    must(chunks.filter(c => c.type === 0x4e4f534a).length === 1, 'Expected one JSON chunk.');
    must(chunks.filter(c => c.type === 0x004e4942).length <= 1, 'Multiple binary chunks are unsupported.');
    const doc = JSON.parse(chunks[0].data.toString('utf8').trim());
    const bin = chunks.find(c => c.type === 0x004e4942)?.data ?? Buffer.alloc(0);
    must(doc.asset?.version === '2.0', 'glTF asset.version is not 2.0.');
    report.generator = doc.asset.generator ?? null;
    report.extensionsUsed = doc.extensionsUsed ?? [];
    report.extensionsRequired = doc.extensionsRequired ?? [];
    for (const ext of ['KHR_draco_mesh_compression', 'EXT_meshopt_compression'])
      must(!report.extensionsUsed.includes(ext), `${ext} requires decoding; this audit cannot establish attribute validity.`);
    const buffers = doc.buffers ?? [];
    must(buffers.length === 1 && !buffers[0].uri, 'Delivery must contain one internal BIN buffer and no external/data URI buffer.');
    must(Number.isSafeInteger(buffers[0].byteLength) && buffers[0].byteLength <= bin.length && bin.length - buffers[0].byteLength <= 3, 'Declared BIN length/padding is invalid.');
    const views = doc.bufferViews ?? [], accessors = doc.accessors ?? [];
    const viewBytes = i => {
      must(Number.isSafeInteger(i) && views[i], `Missing bufferView ${i}.`);
      const v = views[i], start = v.byteOffset ?? 0;
      must(v.buffer === 0 && Number.isSafeInteger(start) && start >= 0 && Number.isSafeInteger(v.byteLength) && v.byteLength >= 0 && start + v.byteLength <= buffers[0].byteLength, `bufferView ${i} is out of bounds.`);
      return bin.subarray(start, start + v.byteLength);
    };
    views.forEach((v, i) => viewBytes(i));
    const cache = new Map();
    const readAccessor = i => {
      if (cache.has(i)) return cache.get(i);
      const a = accessors[i]; must(a && COMPONENT[a.componentType] && WIDTH[a.type], `Unsupported/missing accessor ${i}.`);
      must(Number.isSafeInteger(a.count) && a.count >= 0, `Invalid accessor ${i} count.`);
      const [size, method, lo, hi] = COMPONENT[a.componentType], width = WIDTH[a.type];
      const rows = a.type.startsWith('MAT') ? Math.sqrt(width) : 1;
      const columnBytes = rows > 1 ? Math.ceil(rows * size / 4) * 4 : width * size;
      const elementBytes = rows > 1 ? columnBytes * rows : width * size;
      const offsets = Array.from({length: width}, (_, k) => rows > 1 ? Math.floor(k / rows) * columnBytes + (k % rows) * size : k * size);
      const out = new Float64Array(a.count * width);
      const scalar = (b, p) => { const n = b[method](p); return a.normalized && hi !== null ? Math.max(lo < 0 ? -1 : 0, n / hi) : n; };
      if (a.bufferView !== undefined) {
        const b = viewBytes(a.bufferView), stride = views[a.bufferView].byteStride ?? elementBytes, start = a.byteOffset ?? 0;
        must(Number.isSafeInteger(start) && start >= 0 && Number.isSafeInteger(stride) && stride >= elementBytes && stride % size === 0, `Invalid accessor ${i} stride/offset.`);
        must(a.count === 0 || start + (a.count - 1) * stride + elementBytes <= b.length, `Accessor ${i} exceeds its bufferView.`);
        for (let j = 0; j < a.count; ++j) for (let k = 0; k < width; ++k) out[j * width + k] = scalar(b, start + j * stride + offsets[k]);
      } else must(a.sparse, `Accessor ${i} has no bufferView or sparse values.`);
      if (a.sparse) {
        const s = a.sparse, ic = COMPONENT[s.indices?.componentType];
        must(Number.isSafeInteger(s.count) && s.count >= 0 && s.count <= a.count && [5121, 5123, 5125].includes(s.indices?.componentType), `Invalid sparse accessor ${i}.`);
        const ib = viewBytes(s.indices.bufferView), vb = viewBytes(s.values.bufferView), io = s.indices.byteOffset ?? 0, vo = s.values.byteOffset ?? 0;
        must(io >= 0 && vo >= 0 && io + s.count * ic[0] <= ib.length && vo + s.count * elementBytes <= vb.length, `Sparse accessor ${i} exceeds buffer bounds.`);
        let previous = -1;
        for (let j = 0; j < s.count; ++j) {
          const index = ib[ic[1]](io + j * ic[0]); must(index > previous && index < a.count, `Sparse accessor ${i} indices are invalid.`); previous = index;
          for (let k = 0; k < width; ++k) out[index * width + k] = scalar(vb, vo + j * elementBytes + offsets[k]);
        }
      }
      for (const n of out) must(Number.isFinite(n), `Accessor ${i} contains non-finite values.`);
      cache.set(i, out); return out;
    };
    accessors.forEach((a, i) => readAccessor(i));
    report.accessorCount = accessors.length;
    report.allAccessorNumericValuesFinite = true;
    report.images = (doc.images ?? []).map((im, i) => {
      must(im.bufferView !== undefined && !im.uri, `Image ${i} is not embedded as a bufferView.`);
      const b = viewBytes(im.bufferView), size = imageSize(b);
      must(size && size.width > 0 && size.height > 0, `Image ${i} has unsupported or invalid image dimensions.`);
      must(im.mimeType === `image/${size.format}`, `Image ${i} MIME type does not match encoded image.`);
      return {index: i, name: im.name ?? null, bufferView: im.bufferView, mimeType: im.mimeType, bytes: b.length, sha256: hash(b), ...size};
    });
    report.textures = (doc.textures ?? []).map((t, i) => {
      const source = t.source ?? t.extensions?.EXT_texture_webp?.source ?? t.extensions?.KHR_texture_basisu?.source;
      must(Number.isSafeInteger(source) && report.images[source], `Texture ${i} refers to missing image ${source}.`);
      return {index: i, source, name: t.name ?? null};
    });
    function textureInfo(t, label) {
      if (!t) return null;
      must(report.textures[t.index], `${label} refers to missing texture.`);
      return {texture: t.index, image: report.textures[t.index].source, texCoord: t.extensions?.KHR_texture_transform?.texCoord ?? t.texCoord ?? 0};
    }
    report.materials = (doc.materials ?? []).map((m, i) => {
      const p = m.pbrMetallicRoughness ?? {};
      const baseColor = p.baseColorFactor ?? [1, 1, 1, 1], metallic = p.metallicFactor ?? 1, roughness = p.roughnessFactor ?? 1;
      must(finite(baseColor) && baseColor.length === 4 && baseColor.every(n => n >= 0 && n <= 1), `Material ${i} base color factor is invalid.`);
      must(Number.isFinite(metallic) && metallic >= 0 && metallic <= 1 && Number.isFinite(roughness) && roughness >= 0 && roughness <= 1, `Material ${i} metallic/roughness factors are invalid.`);
      const textures = Object.fromEntries(Object.entries({baseColor: p.baseColorTexture, metallicRoughness: p.metallicRoughnessTexture, normal: m.normalTexture, occlusion: m.occlusionTexture, emissive: m.emissiveTexture}).filter(([, t]) => t).map(([name, t]) => [name, textureInfo(t, `Material ${i} ${name}`)]));
      return {index: i, name: m.name ?? null, baseColor, metallic, roughness, textures, alphaMode: m.alphaMode ?? 'OPAQUE', doubleSided: m.doubleSided ?? false, extensions: Object.keys(m.extensions ?? {})};
    });
    report.meshes = (doc.meshes ?? []).map((m, mi) => {
      const bounds = emptyBounds();
      const primitives = (m.primitives ?? []).map((p, pi) => {
        const label = `Mesh ${mi} primitive ${pi}`, attrs = p.attributes ?? {}, a = accessors[attrs.POSITION];
        must(a?.type === 'VEC3', `${label} has no VEC3 POSITION.`);
        const pos = readAccessor(attrs.POSITION), norm = accessors[attrs.NORMAL];
        must(norm?.type === 'VEC3' && norm.count === a.count, `${label} has missing/incompatible NORMAL.`);
        const ns = readAccessor(attrs.NORMAL); let zeroNormals = 0, nonUnitNormals = 0;
        for (let k = 0; k < pos.length; k += 3) { extend(bounds, pos.subarray(k, k + 3)); const n = Math.hypot(ns[k], ns[k + 1], ns[k + 2]); if (n < 1e-7) ++zeroNormals; else if (Math.abs(n - 1) > 0.01) ++nonUnitNormals; }
        must(!zeroNormals, `${label} has ${zeroNormals} zero-length normals.`);
        if (nonUnitNormals) warnings.push(`${label} has ${nonUnitNormals} normals more than 1% from unit length.`);
        for (const [semantic, ai] of Object.entries(attrs)) must(accessors[ai]?.count === a.count, `${label} attribute ${semantic} vertex count differs.`);
        const uv = accessors[attrs.TEXCOORD_0]; must(uv?.type === 'VEC2' && uv.count === a.count, `${label} has missing/incompatible TEXCOORD_0.`);
        if (p.material !== undefined) must(report.materials[p.material], `${label} has missing material.`);
        for (const t of Object.values(report.materials[p.material]?.textures ?? {})) must(attrs[`TEXCOORD_${t.texCoord}`] !== undefined, `${label} lacks UV set ${t.texCoord} used by its material.`);
        const mode = p.mode ?? 4; must([4, 5, 6].includes(mode), `${label} is not a triangle mesh.`);
        let indices = null;
        if (p.indices !== undefined) { const ia = accessors[p.indices]; must(ia?.type === 'SCALAR' && [5121, 5123, 5125].includes(ia.componentType) && !ia.normalized, `${label} indices have invalid type.`); indices = readAccessor(p.indices); for (const n of indices) must(Number.isInteger(n) && n >= 0 && n < a.count, `${label} index is outside vertex range.`); }
        const count = indices?.length ?? a.count;
        must(mode !== 4 || count % 3 === 0, `${label} triangle index count is not divisible by 3.`);
        const triangles = mode === 4 ? count / 3 : Math.max(0, count - 2);
        for (const target of p.targets ?? []) for (const [semantic, ai] of Object.entries(target)) must(accessors[ai]?.count === a.count, `${label} morph ${semantic} count differs.`);
        return {index: pi, vertices: a.count, triangles, material: p.material ?? null, mode, attributes: Object.keys(attrs), morphTargets: p.targets?.length ?? 0, zeroNormals, nonUnitNormals};
      });
      return {index: mi, name: m.name ?? null, bounds: finishBounds(bounds), triangles: primitives.reduce((sum, p) => sum + p.triangles, 0), primitives};
    });
    report.nodes = (doc.nodes ?? []).map((n, i) => {
      for (const [field, length] of [['matrix', 16], ['translation', 3], ['rotation', 4], ['scale', 3]]) if (n[field]) must(finite(n[field]) && n[field].length === length, `Node ${i} has invalid ${field}.`);
      if (n.mesh !== undefined) must(report.meshes[n.mesh], `Node ${i} references missing mesh.`);
      if (n.skin !== undefined) must(doc.skins?.[n.skin], `Node ${i} references missing skin.`);
      for (const c of n.children ?? []) must(doc.nodes[c] && c !== i, `Node ${i} has invalid child ${c}.`);
      return {index: i, name: n.name ?? null, mesh: n.mesh ?? null, skin: n.skin ?? null, camera: n.camera ?? null, children: n.children ?? [], matrix: n.matrix ?? null, translation: n.translation ?? [0, 0, 0], rotation: n.rotation ?? [0, 0, 0, 1], scale: n.scale ?? [1, 1, 1], extensions: Object.keys(n.extensions ?? {})};
    });
    report.skins = (doc.skins ?? []).map((s, i) => { must(s.joints?.every(j => doc.nodes?.[j]), `Skin ${i} has invalid joints.`); return {index: i, name: s.name ?? null, joints: s.joints.length, inverseBindMatrices: s.inverseBindMatrices ?? null}; });
    report.animations = (doc.animations ?? []).map((a, i) => ({index: i, name: a.name ?? null, channels: a.channels?.length ?? 0, samplers: a.samplers?.length ?? 0}));
    report.summary = {nodes: report.nodes.length, meshes: report.meshes.length, trianglesPerMeshDefinitions: report.meshes.reduce((s, m) => s + m.triangles, 0), trianglesAcrossMeshNodes: report.nodes.reduce((s, n) => s + (report.meshes[n.mesh]?.triangles ?? 0), 0), materials: report.materials.length, images: report.images.length, textures: report.textures.length, skins: report.skins.length, animations: report.animations.length, containsSkeleton: report.skins.length > 0, allImagesEmbedded: true, cameraNodes: report.nodes.filter(n => n.camera !== null).length, lightNodes: report.nodes.filter(n => n.extensions.includes('KHR_lights_punctual')).length};
    if (report.summary.cameraNodes || report.summary.lightNodes) errors.push('Delivery contains camera/light nodes; exclude helpers from final export.');
    if (!report.meshes.length || !report.nodes.some(n => n.mesh !== null)) errors.push('No mesh geometry is referenced by nodes.');
    if (!report.images.length) warnings.push('No embedded images. Constant-color PBR may be valid, but this does not establish textured delivery.');
    report.defaultScene = doc.scene ?? null;
    report.scenes = (doc.scenes ?? []).map((s, i) => ({index: i, name: s.name ?? null, roots: s.nodes ?? []}));
  } catch (error) { errors.push(error.message); }
  report.passed = errors.length === 0;
  return report;
}

function main(argv) {
  if (!argv.length || argv.includes('--help')) {
    console.log('Usage: node Character_Rebuild/scripts/validate_glb.mjs <workspace-relative.glb> [--report <workspace-relative.json>]');
    return;
  }
  const inputArg = argv[0], flag = argv.indexOf('--report');
  if (argv.some((a, i) => i > 0 && i !== flag && i !== flag + 1)) throw new Error('Unknown arguments. Use --help.');
  if (flag >= 0 && !argv[flag + 1]) throw new Error('--report requires a path.');
  const input = workspacePath(inputArg);
  const report = validateGlbBuffer(fs.readFileSync(input), path.relative(workspaceRoot, input).split(path.sep).join('/'));
  if (flag >= 0) { const output = workspacePath(argv[flag + 1]); if (path.extname(output).toLowerCase() !== '.json') throw new Error('Report path must end in .json.'); fs.mkdirSync(path.dirname(output), {recursive: true}); fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n'); }
  console.log(JSON.stringify({input: report.input, sha256: report.sha256, passed: report.passed, summary: report.summary ?? null, errors: report.errors, warnings: report.warnings, report: flag >= 0 ? argv[flag + 1] : null}, null, 2));
  process.exitCode = report.passed ? 0 : 1;
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(process.argv.slice(2)); } catch (error) { console.error(error.message); process.exitCode = 2; }
}
