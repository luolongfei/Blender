// Read-only raw-bind LBS matrices for a finite set of actual animation frames.
import fs from 'node:fs/promises';
import path from 'node:path';
import {THREE, loadRuntime, sha} from './runtime_cpu.mjs';
const {NaturalMotion} = await import('../../previews/natural-motion.mjs');
const args = process.argv.slice(2);
const arg = (n, d) => args.includes(n) ? args[args.indexOf(n) + 1] : d;
const input = path.resolve(arg('--input', ''));
const output = path.resolve(arg('--output', ''));
const mode = arg('--mode', 'walk');
const requestedTimes = arg('--times', Array.from({length: 20}, (_, i) => (i + 1) / 5).join(','))
  .split(',').map(Number);
if (requestedTimes.some(t => !Number.isFinite(t) || t < 0)) throw Error('Invalid times');
const requestedFrames = new Set(requestedTimes.map(t => Math.round(t * 60)));
const runtime = await loadRuntime(input);
try {
  const {avatar, json} = runtime;
  avatar.model.updateMatrixWorld(true);
  const inverseBinds = new Map(), samples = [];
  avatar.model.traverse(mesh => {
    if (!mesh.isSkinnedMesh) return;
    mesh.skeleton.update();
    const factor = mesh.bindMatrix.clone().multiply(mesh.matrixWorld.clone().invert());
    if (factor.elements.some((v, i) => Math.abs(v - (i % 5 === 0 ? 1 : 0)) > 1e-6))
      throw Error('Unsupported nonidentity mesh bind factor');
    mesh.skeleton.bones.forEach((bone, i) => {
      const inverse = mesh.skeleton.boneInverses[i], previous = inverseBinds.get(bone.name);
      if (previous && inverse.elements.some((v, k) => Math.abs(v - previous.elements[k]) > 1e-6))
        throw Error('Inconsistent inverse bind for ' + bone.name);
      inverseBinds.set(bone.name, inverse.clone());
    });
    if (mesh.geometry.morphAttributes.position?.length) return;
    for (let index = 0; index < mesh.geometry.attributes.position.count; index += 257) {
      const raw = new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position, index)
        .applyMatrix4(mesh.matrixWorld);
      const si = mesh.geometry.attributes.skinIndex, sw = mesh.geometry.attributes.skinWeight;
      samples.push({mesh, index, raw, weights: Array.from({length: 4}, (_, c) => ({
        bone: mesh.skeleton.bones[si.getComponent(index, c)].name,
        weight: sw.getComponent(index, c),
      })).filter(w => w.weight > 1e-8)});
    }
  });
  const bones = [];
  for (const index of [...new Set(json.skins.flatMap(s => s.joints))]) {
    const node = await avatar.resources.parser.getDependency('node', index);
    bones.push({name: json.nodes[index].name, index, node, rest: node.matrixWorld.clone()});
  }
  const C = new THREE.Matrix4().set(1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,0,0,1);
  const Ci = C.clone().invert();
  const poses = [];
  const capture = frame => {
    avatar.model.updateMatrixWorld(true);
    const rows = bones.map(({name, index, node, rest}) => {
      const deformation = node.matrixWorld.clone().multiply(inverseBinds.get(name));
      return {
        name, nodeIndex: index,
        restToPosedRuntime: deformation.toArray(),
        restToPosedBlender: C.clone().multiply(deformation).multiply(Ci).toArray(),
        posedWorldRuntime: node.matrixWorld.toArray(),
        loadedRestToPosedRuntime: node.matrixWorld.clone().multiply(rest.clone().invert()).toArray(),
      };
    });
    const byName = new Map(rows.map(r => [r.name, new THREE.Matrix4().fromArray(r.restToPosedRuntime)]));
    let error = 0;
    for (const s of samples) {
      const replay = new THREE.Vector3();
      for (const w of s.weights)
        replay.addScaledVector(s.raw.clone().applyMatrix4(byName.get(w.bone)), w.weight);
      s.mesh.skeleton.update();
      const actual = s.mesh.getVertexPosition(s.index, new THREE.Vector3()).applyMatrix4(s.mesh.matrixWorld);
      error = Math.max(error, replay.distanceTo(actual));
    }
    if (error > 1e-5) throw Error('Raw-bind LBS replay mismatch: ' + error);
    poses.push({mode, frame, time: frame / 60, bones: rows,
      sampledLbsReplayVertices: samples.length, maxRawBindLbsReplayErrorMeters: error});
  };
  const motion = new NaturalMotion(avatar.vrm);
  motion.mode = mode; motion.modeStart = 0; motion.originZ = 0; motion.transition = null;
  avatar.setPhysics(true);
  motion.apply(0); avatar.vrm.update(0); avatar.springManager?.reset();
  if (requestedFrames.has(0)) capture(0);
  for (let frame = 1; frame <= Math.max(...requestedFrames); frame++) {
    motion.apply(frame / 60); avatar.vrm.update(1 / 60);
    if (requestedFrames.has(frame)) capture(frame);
  }
  const report = {
    version: 2, input, vrmSha256: runtime.inputSha, mode, timestep: 1 / 60,
    matrixStorage: 'column-major 16 numeric values, column vectors',
    runtimeToBlender: C.toArray(),
    restBones: bones.map(({name, index, rest}) => ({
      name, nodeIndex: index, loadedRestWorldRuntime: rest.toArray(),
      inverseBindRuntime: inverseBinds.get(name).toArray(),
    })),
    poses,
    sourceUnchanged: sha(await fs.readFile(input)) === runtime.inputSha,
    scope: 'Finite actual animation snapshots. restToPosed maps raw source-bind POSITION coordinates using posedBoneWorld × inverseBind. Blender maps C × D × inverse(C). These are linear skinning matrices, not loaded-rest-relative transforms or dual-quaternion skinning.',
  };
  if (!report.sourceUnchanged) throw Error('Input changed during inspection');
  await fs.writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({output, vrmSha256: runtime.inputSha, mode, poses: poses.length,
    maximumReplayError: Math.max(...poses.map(p => p.maxRawBindLbsReplayErrorMeters))}));
} finally {
  runtime.avatar.dispose();
}
