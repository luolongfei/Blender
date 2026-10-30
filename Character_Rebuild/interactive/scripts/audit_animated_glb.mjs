/**
 * Read-only GLB skin/animation-data audit, without Blender or external packages.
 * node Character_Rebuild/interactive/scripts/audit_animated_glb.mjs
 * Optional: <workspace-relative.glb> --report <workspace-relative.json>
 * Exit 0 = declared data checks passed, 1 = audit failed, 2 = CLI/read failure.
 * This does not certify visual deformation, collisions, or browser playback.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {workspacePath} from '../../../pipeline/context.mjs';

const COMPONENT = {
  5120: [1, 'readInt8', 127], 5121: [1, 'readUInt8', 255],
  5122: [2, 'readInt16LE', 32767], 5123: [2, 'readUInt16LE', 65535],
  5125: [4, 'readUInt32LE', 4294967295], 5126: [4, 'readFloatLE', null],
};
const WIDTH = {SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16};
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const norm = v => Math.hypot(...v);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const finite = a => a.every(Number.isFinite);

export function auditAnimatedGlb(bytes, inputLabel = '<buffer>', expectedClips = ['idle', 'wave', 'look', 'walk', 'Tpose']) {
  const errors = [], warnings = [];
  const report = {
    schemaVersion: 1, input: inputLabel, bytes: bytes.length,
    sha256: hash(bytes), checkedAt: new Date().toISOString(), errors, warnings,
    scope: 'Embedded GLB skin weights, inverse bind matrices and actual joint-target animation channel data.',
    modelModified: false, blenderLaunched: false,
    unchecked: [
      'Visual deformation quality and correct anatomical weighting',
      'Intersections, ground contact, balance, foot sliding, physical plausibility',
      'Browser playback, renderer compatibility and texture appearance',
      'Inverse-bind/rest-pose congruence and animation retargeting accuracy',
      'Skeleton scale and orientation against the artwork',
    ],
    skins: [], skinnedPrimitives: [], clips: [],
  };
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  try {
    check(Buffer.isBuffer(bytes) && bytes.length >= 20, 'Input is not a complete GLB buffer.');
    check(bytes.readUInt32LE(0) === 0x46546c67 && bytes.readUInt32LE(4) === 2, 'Expected GLB 2.0.');
    check(bytes.readUInt32LE(8) === bytes.length, 'GLB byteLength differs from actual length.');
    const chunks = [];
    for (let offset = 12; offset < bytes.length;) {
      check(offset + 8 <= bytes.length, 'Truncated GLB chunk header.');
      const length = bytes.readUInt32LE(offset), type = bytes.readUInt32LE(offset + 4);
      check(length % 4 === 0 && offset + 8 + length <= bytes.length, 'Invalid GLB chunk bounds/alignment.');
      chunks.push({type, data: bytes.subarray(offset + 8, offset + 8 + length)});
      offset += 8 + length;
    }
    check(chunks[0]?.type === 0x4e4f534a, 'First GLB chunk must be JSON.');
    check(chunks.filter(c => c.type === 0x4e4f534a).length === 1, 'Expected exactly one JSON chunk.');
    check(chunks.filter(c => c.type === 0x004e4942).length === 1, 'Expected exactly one BIN chunk.');
    const doc = JSON.parse(chunks[0].data.toString('utf8').trim());
    const bin = chunks.find(c => c.type === 0x004e4942).data;
    check(doc.asset?.version === '2.0', 'glTF asset.version must be 2.0.');
    check(doc.buffers?.length === 1 && !doc.buffers[0].uri, 'External or multiple buffers are unsupported.');
    check(Number.isSafeInteger(doc.buffers[0].byteLength) && doc.buffers[0].byteLength <= bin.length && bin.length - doc.buffers[0].byteLength <= 3, 'Invalid declared BIN length.');
    for (const ext of ['KHR_draco_mesh_compression', 'EXT_meshopt_compression'])
      check(!(doc.extensionsUsed ?? []).includes(ext), `${ext} requires decoding, which this audit does not implement.`);
    const nodes = doc.nodes ?? [], meshes = doc.meshes ?? [], skins = doc.skins ?? [];
    const animations = doc.animations ?? [], accessors = doc.accessors ?? [], views = doc.bufferViews ?? [];
    const nodeExists = i => Number.isSafeInteger(i) && i >= 0 && i < nodes.length;
    const accessorCache = new Map();
    function view(i) {
      check(Number.isSafeInteger(i) && views[i], `Missing bufferView ${i}.`);
      const v = views[i], offset = v.byteOffset ?? 0;
      check(v.buffer === 0 && Number.isSafeInteger(offset) && offset >= 0 && Number.isSafeInteger(v.byteLength) && v.byteLength >= 0 && offset + v.byteLength <= doc.buffers[0].byteLength, `bufferView ${i} is out of bounds.`);
      return {descriptor: v, data: bin.subarray(offset, offset + v.byteLength)};
    }
    function readAccessor(i) {
      if (accessorCache.has(i)) return accessorCache.get(i);
      check(Number.isSafeInteger(i) && accessors[i], `Missing accessor ${i}.`);
      const a = accessors[i], component = COMPONENT[a.componentType], width = WIDTH[a.type];
      check(component && width, `Unsupported accessor ${i} type/componentType.`);
      check(Number.isSafeInteger(a.count) && a.count >= 0, `Invalid accessor ${i} count.`);
      const [size, method, denominator] = component, itemBytes = size * width;
      const normalize = x => a.normalized && denominator ? Math.max(-1, x / denominator) : x;
      const values = Array.from({length: a.count}, () => Array(width).fill(0));
      if (a.bufferView !== undefined) {
        const {descriptor, data} = view(a.bufferView), offset = a.byteOffset ?? 0;
        const stride = descriptor.byteStride ?? itemBytes;
        check(Number.isSafeInteger(offset) && offset >= 0 && offset % size === 0 && Number.isSafeInteger(stride) && stride >= itemBytes && stride % size === 0, `Accessor ${i} has invalid offset/stride.`);
        check(a.count === 0 || offset + (a.count - 1) * stride + itemBytes <= data.length, `Accessor ${i} exceeds bufferView.`);
        for (let row = 0; row < a.count; ++row)
          for (let col = 0; col < width; ++col) values[row][col] = normalize(data[method](offset + row * stride + col * size));
      } else check(a.sparse, `Accessor ${i} has neither bufferView nor sparse data.`);
      if (a.sparse) {
        const sparse = a.sparse, ic = COMPONENT[sparse.indices?.componentType];
        check(Number.isSafeInteger(sparse.count) && sparse.count > 0 && sparse.count <= a.count && [5121, 5123, 5125].includes(sparse.indices?.componentType), `Invalid accessor ${i} sparse count/index type.`);
        const iv = view(sparse.indices.bufferView), vv = view(sparse.values.bufferView);
        const io = sparse.indices.byteOffset ?? 0, vo = sparse.values.byteOffset ?? 0;
        check(Number.isSafeInteger(io) && io >= 0 && Number.isSafeInteger(vo) && vo >= 0 && io + sparse.count * ic[0] <= iv.data.length && vo + sparse.count * itemBytes <= vv.data.length, `Sparse accessor ${i} exceeds bufferView.`);
        let previous = -1;
        for (let row = 0; row < sparse.count; ++row) {
          const target = iv.data[ic[1]](io + row * ic[0]);
          check(target > previous && target < a.count, `Sparse accessor ${i} indices must increase and remain in range.`);
          for (let col = 0; col < width; ++col) values[target][col] = normalize(vv.data[method](vo + row * itemBytes + col * size));
          previous = target;
        }
      }
      check(values.every(finite), `Accessor ${i} contains non-finite values.`);
      const result = {descriptor: a, values}; accessorCache.set(i, result); return result;
    }

    const allJoints = new Set();
    check(skins.length > 0, 'No skins: this asset does not contain an actual skinned character.');
    for (const [index, skin] of skins.entries()) {
      const entry = {index, name: skin.name ?? `skin_${index}`, jointCount: skin.joints?.length ?? 0};
      report.skins.push(entry);
      try {
        check(Array.isArray(skin.joints) && skin.joints.length > 0, `Skin ${index} has no joints.`);
        check(skin.joints.every(nodeExists), `Skin ${index} references an invalid joint node.`);
        check(new Set(skin.joints).size === skin.joints.length, `Skin ${index} repeats joint node indices.`);
        if (skin.skeleton !== undefined) check(nodeExists(skin.skeleton), `Skin ${index} has an invalid skeleton node.`);
        skin.joints.forEach(j => allJoints.add(j));
        entry.joints = skin.joints.map(j => ({node: j, name: nodes[j].name ?? `node_${j}`}));
        if (skin.inverseBindMatrices === undefined) {
          entry.inverseBindMatrices = {source: 'implicit_identity', finite: true, count: skin.joints.length};
          warnings.push(`Skin ${index} omits inverseBindMatrices; glTF identity defaults apply.`);
        } else {
          const {descriptor: a, values} = readAccessor(skin.inverseBindMatrices);
          check(a.type === 'MAT4' && a.componentType === 5126 && !a.normalized && a.count === skin.joints.length, `Skin ${index} inverseBindMatrices must be FLOAT MAT4 with one matrix per joint.`);
          entry.inverseBindMatrices = {source: 'accessor', accessor: skin.inverseBindMatrices, finite: true, count: values.length};
        }
        entry.valid = true;
      } catch (error) { entry.valid = false; errors.push(error.message); }
    }

    let skinVertices = 0, normalizedVertices = 0, nonzeroWeightedVertices = 0;
    const skinnedMeshNodes = [];
    for (const [nodeIndex, node] of nodes.entries()) {
      if (node.skin === undefined) continue;
      skinnedMeshNodes.push(nodeIndex);
      if (!Number.isSafeInteger(node.skin) || !skins[node.skin] || !Number.isSafeInteger(node.mesh) || !meshes[node.mesh]) {
        errors.push(`Skinned node ${nodeIndex} has invalid mesh or skin.`); continue;
      }
      const jointCount = skins[node.skin].joints?.length ?? 0;
      for (const [primitiveIndex, primitive] of (meshes[node.mesh].primitives ?? []).entries()) {
        const entry = {node: nodeIndex, name: node.name ?? `node_${nodeIndex}`, mesh: node.mesh, primitive: primitiveIndex, skin: node.skin};
        report.skinnedPrimitives.push(entry);
        try {
          const attrs = primitive.attributes ?? {}, pos = readAccessor(attrs.POSITION);
          const sets = Object.keys(attrs).filter(k => /^JOINTS_\d+$/.test(k)).map(k => Number(k.slice(7))).sort((a, b) => a - b);
          const weightSets = Object.keys(attrs).filter(k => /^WEIGHTS_\d+$/.test(k)).map(k => Number(k.slice(8))).sort((a, b) => a - b);
          check(sets.length > 0 && sets[0] === 0 && JSON.stringify(sets) === JSON.stringify(weightSets), `Node ${nodeIndex} primitive ${primitiveIndex} JOINTS/WEIGHTS sets are missing/unpaired.`);
          check(sets.every((n, i) => n === i), `Node ${nodeIndex} primitive ${primitiveIndex} skin attribute sets are not consecutive.`);
          const data = sets.map(set => {
            const joints = readAccessor(attrs[`JOINTS_${set}`]), weights = readAccessor(attrs[`WEIGHTS_${set}`]);
            check(joints.descriptor.type === 'VEC4' && [5121, 5123].includes(joints.descriptor.componentType) && !joints.descriptor.normalized, `JOINTS_${set} must be non-normalized unsigned VEC4.`);
            check(weights.descriptor.type === 'VEC4' && [5121, 5123, 5126].includes(weights.descriptor.componentType), `WEIGHTS_${set} must be FLOAT or normalized unsigned VEC4.`);
            check(weights.descriptor.componentType === 5126 ? !weights.descriptor.normalized : weights.descriptor.normalized === true, `WEIGHTS_${set} has invalid normalization metadata.`);
            check(joints.values.length === pos.values.length && weights.values.length === pos.values.length, `Skin attribute vertex count differs from POSITION at node ${nodeIndex}.`);
            return {set, joints, weights};
          });
          const quantizationTolerance = data.reduce((t, d) => t + (d.weights.descriptor.componentType === 5126 ? 0 : 2 / COMPONENT[d.weights.descriptor.componentType][2]), 0);
          const tolerance = Math.max(1e-4, quantizationTolerance + 1e-7);
          let badJointIndices = 0, negativeWeights = 0, weightsAboveOne = 0, zeroWeightVertices = 0, unnormalizedVertices = 0;
          let minSum = Infinity, maxSum = -Infinity, maxNormalizationError = 0, maxInfluences = 0;
          const usedJoints = new Set();
          for (let vertex = 0; vertex < pos.values.length; ++vertex) {
            let sum = 0, influences = 0;
            for (const {joints, weights} of data) for (let k = 0; k < 4; ++k) {
              const joint = joints.values[vertex][k], weight = weights.values[vertex][k];
              if (!Number.isInteger(joint) || joint < 0 || joint >= jointCount) ++badJointIndices;
              if (weight < 0) ++negativeWeights;
              if (weight > 1 + tolerance) ++weightsAboveOne;
              if (weight > 0) { ++influences; usedJoints.add(joint); }
              sum += weight;
            }
            minSum = Math.min(minSum, sum); maxSum = Math.max(maxSum, sum);
            maxNormalizationError = Math.max(maxNormalizationError, Math.abs(sum - 1));
            maxInfluences = Math.max(maxInfluences, influences);
            if (sum <= 1e-12) ++zeroWeightVertices;
            if (Math.abs(sum - 1) > tolerance) ++unnormalizedVertices;
          }
          entry.vertices = pos.values.length; entry.attributeSets = sets;
          entry.jointIndexErrors = badJointIndices; entry.negativeWeights = negativeWeights;
          entry.weightsAboveOne = weightsAboveOne; entry.zeroWeightVertices = zeroWeightVertices;
          entry.unnormalizedVertices = unnormalizedVertices; entry.normalizationTolerance = tolerance;
          entry.maxNormalizationError = maxNormalizationError;
          entry.weightSumRange = pos.values.length ? [minSum, maxSum] : null;
          entry.maxNonzeroInfluencesPerVertex = maxInfluences;
          entry.usedJointIndices = [...usedJoints].sort((a, b) => a - b);
          skinVertices += pos.values.length; normalizedVertices += pos.values.length - unnormalizedVertices;
          nonzeroWeightedVertices += pos.values.length - zeroWeightVertices;
          check(!badJointIndices && !negativeWeights && !weightsAboveOne && !zeroWeightVertices && !unnormalizedVertices, `Node ${nodeIndex} primitive ${primitiveIndex} has invalid indices or weights (see counts).`);
          entry.valid = true;
        } catch (error) { entry.valid = false; errors.push(error.message); }
      }
    }
    check(skinnedMeshNodes.length > 0 && report.skinnedPrimitives.length > 0 && skinVertices > 0, 'No actual mesh vertices are bound to the skins.');

    const clipNames = animations.map((a, i) => a.name ?? `animation_${i}`);
    report.clipContract = {expected: expectedClips, actual: clipNames,
      allowedStaticRestClips: ['Tpose'],
      missing: expectedClips.filter(name => !clipNames.some(actual => actual.toLowerCase() === name.toLowerCase()))};
    if (report.clipContract.missing.length) errors.push(`Missing contracted clips: ${report.clipContract.missing.join(', ')}.`);
    if (new Set(clipNames.map(n => n.toLowerCase())).size !== clipNames.length) errors.push('Animation names are not unique ignoring case.');
    if (animations.length !== expectedClips.length) warnings.push(`Expected ${expectedClips.length} named clips; actual animation count is ${animations.length}.`);
    for (const [clipIndex, animation] of animations.entries()) {
      const allowStaticRest = clipNames[clipIndex].toLowerCase() === 'tpose';
      const clip = {index: clipIndex, name: clipNames[clipIndex],
        role: allowStaticRest ? 'static_rest_pose' : 'motion',
        requiresNonStaticJointMotion: !allowStaticRest,
        channels: [], errors: [], startTime: null, endTime: null, duration: null};
      report.clips.push(clip);
      let first = Infinity, last = -Infinity;
      const targetKeys = new Set(), jointNodes = new Set(), changedJointNodes = new Set();
      for (const [channelIndex, channel] of (animation.channels ?? []).entries()) {
        const nodeIndex = channel.target?.node, targetPath = channel.target?.path;
        const entry = {index: channelIndex, node: nodeIndex ?? null, nodeName: nodeExists(nodeIndex) ? nodes[nodeIndex].name ?? `node_${nodeIndex}` : null, path: targetPath, targetsSkinJoint: allJoints.has(nodeIndex)};
        clip.channels.push(entry);
        try {
          check(nodeExists(nodeIndex), `Clip ${clip.name} channel ${channelIndex} has invalid target node.`);
          check(['translation', 'rotation', 'scale', 'weights'].includes(targetPath), `Unsupported animation target path ${targetPath}.`);
          const targetKey = `${nodeIndex}:${targetPath}`;
          check(!targetKeys.has(targetKey), `Clip ${clip.name} has duplicate channel target ${targetKey}.`); targetKeys.add(targetKey);
          const sampler = animation.samplers?.[channel.sampler];
          check(Number.isSafeInteger(channel.sampler) && sampler, `Clip ${clip.name} channel ${channelIndex} has invalid sampler.`);
          const input = readAccessor(sampler.input), output = readAccessor(sampler.output);
          check(input.descriptor.type === 'SCALAR' && input.descriptor.componentType === 5126 && !input.descriptor.normalized, 'Animation times must be FLOAT SCALAR.');
          const times = input.values.map(row => row[0]);
          check(times.length > 0 && times[0] >= 0 && times.every((t, i) => i === 0 || t > times[i - 1]), `Clip ${clip.name} channel ${channelIndex} times must be nonnegative and strictly increasing.`);
          const interpolation = sampler.interpolation ?? 'LINEAR';
          check(['LINEAR', 'STEP', 'CUBICSPLINE'].includes(interpolation), `Unsupported animation interpolation ${interpolation}.`);
          check(output.descriptor.componentType === 5126 && !output.descriptor.normalized, 'Animation output must be non-normalized FLOAT.');
          const multiplier = interpolation === 'CUBICSPLINE' ? 3 : 1;
          let values;
          if (targetPath === 'weights') {
            const primitive = meshes[nodes[nodeIndex].mesh]?.primitives?.[0];
            const targetCount = primitive?.targets?.length ?? nodes[nodeIndex].weights?.length ?? meshes[nodes[nodeIndex].mesh]?.weights?.length ?? 0;
            check(targetCount > 0 && output.descriptor.type === 'SCALAR' && output.values.length === times.length * multiplier * targetCount, 'Morph-weight animation output count/type is invalid.');
            values = times.map((_, i) => output.values.slice((i * multiplier + (multiplier === 3 ? 1 : 0)) * targetCount, (i * multiplier + (multiplier === 3 ? 1 : 0) + 1) * targetCount).map(row => row[0]));
          } else {
            check(output.descriptor.type === (targetPath === 'rotation' ? 'VEC4' : 'VEC3') && output.values.length === times.length * multiplier, `Clip ${clip.name} channel ${channelIndex} output type/count is invalid.`);
            values = times.map((_, i) => output.values[i * multiplier + (multiplier === 3 ? 1 : 0)]);
          }
          first = Math.min(first, times[0]); last = Math.max(last, times.at(-1));
          entry.keys = times.length; entry.interpolation = interpolation; entry.timeRange = [times[0], times.at(-1)];
          entry.timesStrictlyIncreasing = true; entry.finite = true;
          if (targetPath === 'rotation') {
            const norms = values.map(norm), maxError = Math.max(...norms.map(n => Math.abs(n - 1)));
            entry.quaternionNormRange = [Math.min(...norms), Math.max(...norms)]; entry.maxQuaternionNormError = maxError;
            check(norms.every(n => n > 1e-12) && maxError <= 1e-3, `Clip ${clip.name} channel ${channelIndex} has invalid/non-unit quaternion samples.`);
            const base = values[0].map(x => x / norms[0]);
            entry.maxAngleFromFirstRadians = Math.max(...values.map((q, i) => 2 * Math.acos(Math.min(1, Math.abs(dot(base, q.map(x => x / norms[i])))))));
            entry.nonStatic = entry.maxAngleFromFirstRadians > 1e-5;
          } else {
            entry.maxDistanceFromFirst = Math.max(...values.map(v => norm(v.map((x, i) => x - values[0][i]))));
            entry.nonStatic = entry.maxDistanceFromFirst > 1e-7;
          }
          if (entry.targetsSkinJoint && targetPath !== 'weights') {
            jointNodes.add(nodeIndex); if (entry.nonStatic) changedJointNodes.add(nodeIndex);
          }
          entry.valid = true;
        } catch (error) { entry.valid = false; clip.errors.push(error.message); }
      }
      clip.startTime = Number.isFinite(first) ? first : null; clip.endTime = Number.isFinite(last) ? last : null;
      clip.duration = clip.startTime === null ? null : last - first;
      clip.channelsCount = clip.channels.length;
      clip.jointChannels = clip.channels.filter(c => c.targetsSkinJoint && c.path !== 'weights').length;
      clip.nonStaticJointChannels = clip.channels.filter(c => c.targetsSkinJoint && c.path !== 'weights' && c.nonStatic).length;
      clip.animatedJointNodes = [...jointNodes]; clip.changedJointNodes = [...changedJointNodes];
      if (allowStaticRest) {
        // One finite key at t=0 is a legal rest pose; valid multi-key times still
        // have to increase strictly. No other clip receives this exception.
        if (!(clip.duration >= 0)) clip.errors.push(`Rest clip ${clip.name} has no valid time range.`);
        if (!clip.jointChannels) clip.errors.push(`Rest clip ${clip.name} has no actual skin-joint channels.`);
      } else {
        if (!(clip.duration > 0)) clip.errors.push(`Clip ${clip.name} has no positive duration.`);
        if (!clip.nonStaticJointChannels) clip.errors.push(`Clip ${clip.name} has no genuinely changing skin-joint channels.`);
      }
      clip.valid = clip.errors.length === 0;
      errors.push(...clip.errors);
    }
    report.summary = {
      skins: skins.length, uniqueJointNodes: allJoints.size,
      skinnedMeshNodes: skinnedMeshNodes.length, skinnedPrimitives: report.skinnedPrimitives.length,
      skinnedVertices: skinVertices, normalizedWeightVertices: normalizedVertices, nonzeroWeightedVertices,
      clips: report.clips.length,
      clipSummary: report.clips.map(c => ({name: c.name, role: c.role, duration: c.duration, jointChannels: c.jointChannels, nonStaticJointChannels: c.nonStaticJointChannels, changedJoints: c.changedJointNodes.length, valid: c.valid})),
      allInverseBindMatricesFinite: report.skins.every(s => s.valid && s.inverseBindMatrices?.finite),
      allWeightsValidAndNormalized: report.skinnedPrimitives.length > 0 && report.skinnedPrimitives.every(p => p.valid),
      allClipsMeetMotionOrRestContract: report.clips.length > 0 && report.clips.every(c => c.valid),
      allActionClipsContainActualJointMotion: report.clips.filter(c => c.requiresNonStaticJointMotion).length > 0
        && report.clips.filter(c => c.requiresNonStaticJointMotion).every(c => c.valid && c.nonStaticJointChannels > 0),
    };
  } catch (error) { errors.push(error.message); }
  report.passed = errors.length === 0;
  return report;
}

function main() {
  const args = process.argv.slice(2);
  let input = 'Character_Rebuild/interactive/animated-preview.glb';
  let output = 'Character_Rebuild/interactive/reports/animation-data-audit.json';
  if (args[0] && !args[0].startsWith('--')) input = args.shift();
  while (args.length) {
    const flag = args.shift();
    if (flag === '--report' && args[0]) output = args.shift();
    else throw new Error(`Unknown/incomplete argument ${flag}.`);
  }
  const inputPath = workspacePath(input), outputPath = workspacePath(output);
  if (inputPath === outputPath) throw new Error('Report path must differ from input asset.');
  const bytes = fs.readFileSync(inputPath), report = auditAnimatedGlb(bytes, input);
  if (hash(fs.readFileSync(inputPath)) !== report.sha256) throw new Error('Input changed during audit; report was not written.');
  fs.mkdirSync(path.dirname(outputPath), {recursive: true});
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({passed: report.passed, sha256: report.sha256, summary: report.summary, errors: report.errors, warnings: report.warnings, report: output}, null, 2));
  process.exitCode = report.passed ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 2; }
}
