// Aggregate actual full-body/clothing tests, preserving input identity and failures.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';

const args = process.argv.slice(2);
const arg = (key, fallback) =>
  args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const required = key => {
  const value = arg(key);
  if (!value) throw Error(`Required argument: ${key}`);
  return path.resolve(value);
};
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const input = required('--input');
const output = required('--output');
const vrmSha256 = hash(await fs.readFile(input));
const sources = {};
const cases = {};

for (const mode of ['static', 'walk', 'greet']) {
  const file = required(`--${mode}`);
  const bytes = await fs.readFile(file);
  const report = JSON.parse(bytes);
  if (report.vrmSha256 !== vrmSha256 || report.sourceUnchanged !== true) {
    throw Error(`Input identity/source integrity mismatch: ${file}`);
  }
  if (report.mode !== mode || report.motionUnchanged !== true) {
    throw Error(`Wrong motion or modified animation: ${file}`);
  }
  if (!report.completeAnatomyNodes || !report.groups?.skin?.some(g => g.complete)) {
    throw Error(`Completed anatomy was not inspected: ${file}`);
  }
  if (!report.perFrame?.length || report.perFrame.length !== report.frames) {
    throw Error(`Missing actual frame evidence: ${file}`);
  }
  const worst = report.perFrame.reduce((a, b) =>
    b.intersectionPairs > a.intersectionPairs ? b : a);
  sources[mode] = {path: file, sha256: hash(bytes)};
  cases[mode] = {
    frames: report.frames,
    timestep: report.timestep,
    sampleEveryFrames: report.sampleEveryFrames,
    sampleTimeRange: [
      Math.min(...report.perFrame.map(f => f.time)),
      Math.max(...report.perFrame.map(f => f.time)),
    ],
    physics: report.physics,
    motionPath: report.motionPath,
    motionSha256: report.motionSha256,
    framesWithIntersection: report.framesWithIntersection,
    framesWithCompletedAnatomyIntersection: report.framesWithCompletedAnatomyIntersection,
    maximumIntersectionPairs: report.maximumIntersectionPairs,
    passed: report.framesWithIntersection === 0 &&
      report.maximumIntersectionPairs === 0,
    worstFrame: {
      time: worst.time, pose: worst.pose,
      intersectionPairs: worst.intersectionPairs,
      pairs: worst.pairs, pairBounds: worst.pairBounds,
      examplesByPair: worst.examplesByPair,
    },
    groups: report.groups,
  };
}
const sourceUnchanged = hash(await fs.readFile(input)) === vrmSha256;
if (!sourceUnchanged) throw Error('VRM changed while aggregating checks');
const failedCases = Object.entries(cases).filter(([, c]) => !c.passed).map(([m]) => m);
const result = {
  version: 1,
  vrmPath: input,
  vrmSha256,
  sourceUnchanged,
  threshold: {
    maximumAllowedDetectedEdgeTriangleIntersections: 0,
    spatialExclusions: [],
    automaticSeamExemptions: [],
  },
  passed: failedCases.length === 0,
  failedCases,
  framesInspected: Object.values(cases).reduce((sum, c) => sum + c.frames, 0),
  sources,
  cases,
  scope: 'Actual exported skinned completed anatomy plus existing legs, hands and neck against all exported upper/lower garment meshes, including skirts, socks and shoes. Bidirectional edge/triangle surface tests on the exact VRM, without moving or hiding geometry.',
  limitations: [
    'These finite sampled motions do not establish collision freedom for arbitrary poses or frames between samples.',
    'Counts are intersecting edge/triangle pairs, not penetration depth or number of visible skin patches.',
    'Coplanar overlaps and exact tangency are not robustly detected.',
    'Anatomy coverage follows source completed_anatomy tags and exported geometry; this is not a medical anatomy validation or a claim that the body is one welded manifold.',
    'Bodies correctly enclosed by garments are not classified as intersections.',
  ],
};
await fs.mkdir(path.dirname(output), {recursive: true});
await fs.writeFile(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({output, vrmSha256, passed: result.passed, failedCases,
  framesInspected: result.framesInspected}));
