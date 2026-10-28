import {toolPath} from '../../../../pipeline/context.mjs';
// CPU validation of the exact exported VRM's real morph target bindings.
// Images/materials are omitted in memory; geometry, skins and expressions stay intact.
import {register} from 'node:module';
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const args = process.argv.slice(2);
const option = (name) => args.includes(name) ? args[args.indexOf(name) + 1] : null;
if (!option('--input') || !option('--output')) {
  throw Error('Usage: node qa_expressions.mjs --input exact.vrm --output report.json');
}
const inputPath = path.resolve(option('--input'));
const outputPath = path.resolve(option('--output'));
const threeURL = pathToFileURL(toolPath('public/vendor/three.module.js')).href;
register('data:text/javascript,' + encodeURIComponent(
  `export async function resolve(s,c,next){return s==='three'?{url:${JSON.stringify(threeURL)},shortCircuit:true}:next(s,c);}`
), import.meta.url);
const THREE = await import('three');
const {PreviewAvatar} = await import(pathToFileURL(toolPath('public/preview-avatar.mjs')));
const sha = data => createHash('sha256').update(data).digest('hex');
const input = await fs.readFile(inputPath);
const inputSha = sha(input);
const jsonLength = input.readUInt32LE(12);
const json = JSON.parse(input.subarray(20, 20 + jsonLength));
const bin = input.subarray(28 + jsonLength);
json.materials = (json.materials || []).map(m => ({
  name: m.name, pbrMetallicRoughness: {baseColorFactor: [1, 1, 1, 1]}
}));
delete json.images;
delete json.textures;
for (const key of ['extensionsUsed', 'extensionsRequired']) {
  json[key] = (json[key] || []).filter(name =>
    !['VRMC_materials_mtoon', 'KHR_texture_transform', 'KHR_materials_unlit'].includes(name));
}
const encoded = Buffer.from(JSON.stringify(json));
const padded = Math.ceil(encoded.length / 4) * 4;
const bytes = Buffer.alloc(28 + padded + bin.length, 32);
bytes.writeUInt32LE(0x46546c67, 0);
bytes.writeUInt32LE(2, 4);
bytes.writeUInt32LE(bytes.length, 8);
bytes.writeUInt32LE(padded, 12);
bytes.writeUInt32LE(0x4e4f534a, 16);
encoded.copy(bytes, 20);
bytes.writeUInt32LE(bin.length, 20 + padded);
bytes.writeUInt32LE(0x004e4942, 24 + padded);
bin.copy(bytes, 28 + padded);

const avatar = new PreviewAvatar({scene: new THREE.Scene()});
avatar.setActive(true);
const names = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised',
  'blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh'];
try {
  await avatar.loadVRM(new File([bytes], 'expression-audit.vrm'));
  avatar.setPhysics(false);
  const manager = avatar.vrm.expressionManager;
  if (!manager) throw Error('Missing VRM expression manager');
  for (const name of names) if (!manager.getExpression(name)) {
    throw Error('Missing runtime expression ' + name);
  }
  const meshes = [];
  avatar.model.traverse(obj => {
    if (obj.isMesh && obj.geometry.morphAttributes.position?.length) meshes.push(obj);
  });
  if (!meshes.length) throw Error('No runtime morph meshes');
  const vector = new THREE.Vector3();
  function capture(values = {}) {
    manager.resetValues();
    for (const [name, value] of Object.entries(values)) manager.setValue(name, value);
    avatar.vrm.update(0);
    avatar.model.updateMatrixWorld(true);
    const points = [];
    for (const mesh of meshes) {
      mesh.skeleton?.update();
      for (let i = 0; i < mesh.geometry.attributes.position.count; i++) {
        mesh.getVertexPosition(i, vector).applyMatrix4(mesh.matrixWorld);
        if (![vector.x, vector.y, vector.z].every(Number.isFinite)) {
          throw Error('Nonfinite runtime expression vertex: ' + mesh.name);
        }
        points.push(vector.x, vector.y, vector.z);
      }
    }
    return Float64Array.from(points);
  }
  const basis = capture();
  const results = {};
  for (const name of names) {
    const half = capture({[name]: .5});
    const full = capture({[name]: 1});
    let maximum = 0, changed = 0, halfwayError = 0;
    for (let i = 0; i < basis.length; i += 3) {
      const displacement = Math.hypot(full[i] - basis[i], full[i+1] - basis[i+1],
        full[i+2] - basis[i+2]);
      maximum = Math.max(maximum, displacement);
      if (displacement > 1e-7) changed++;
      halfwayError = Math.max(halfwayError, Math.hypot(
        half[i] - (basis[i] + full[i]) / 2,
        half[i+1] - (basis[i+1] + full[i+1]) / 2,
        half[i+2] - (basis[i+2] + full[i+2]) / 2));
    }
    if (name !== 'neutral' && maximum < 1e-6) throw Error(name + ': no actual deformation');
    if (name === 'neutral' && maximum > 1e-7) throw Error('Neutral changes geometry');
    if (halfwayError > 5e-6) throw Error(name + ': unexpected nonlinear half-weight response');
    results[name] = {changedVertices: changed, maxDisplacementMeters: maximum,
      midpointLinearityErrorMeters: halfwayError, finite: true};
  }
  const combinations = [
    {happy: .7, aa: .5, blink: .5},
    {angry: .6, oh: .5, blinkLeft: .5},
    {sad: .6, ih: .5, blinkRight: .5},
  ];
  for (const values of combinations) capture(values);
  const reset = capture();
  let resetError = 0;
  for (let i = 0; i < reset.length; i++) resetError = Math.max(resetError, Math.abs(reset[i] - basis[i]));
  if (resetError > 1e-7) throw Error('Morph geometry did not reset to neutral');
  if (sha(await fs.readFile(inputPath)) !== inputSha) throw Error('Input VRM changed during audit');
  const report = {
    version: 1, vrmPath: inputPath, vrmSha256: inputSha, sourceUnchanged: true,
    morphMeshes: meshes.length, testedVerticesPerState: basis.length / 3,
    strengths: [0, .5, 1], presets: results,
    finiteCombinationChecks: combinations, neutralResetErrorMeters: resetError,
    scope: 'Real three-vrm expressions, morphs and skinning in CPU evaluation. '
      + 'Appearance, eyelid/lip contact, surface intersections, aesthetic quality, '
      + 'materials, lighting and animated combinations require visual review.',
  };
  await fs.mkdir(path.dirname(outputPath), {recursive: true});
  await fs.writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
} finally {
  avatar.dispose();
}
