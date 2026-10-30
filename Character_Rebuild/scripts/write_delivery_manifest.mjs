/** Build evidence inventory only. This never turns successful export into artistic acceptance.
 * node Character_Rebuild/scripts/write_delivery_manifest.mjs
 * Safe before final files exist: absent/stale evidence remains explicitly pending.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {workspacePath, toolchain} from '../../pipeline/context.mjs';

const base = 'Character_Rebuild';
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
function record(relative) {
  const file = workspacePath(relative);
  if (!fs.existsSync(file)) return {path: relative, exists: false, bytes: null, sha256: null};
  const bytes = fs.readFileSync(file);
  return {path: relative, exists: true, bytes: bytes.length, sha256: sha(bytes)};
}
function jsonEvidence(relative) {
  const file = record(relative);
  if (!file.exists) return {file, data: null, parseError: null};
  try { return {file, data: JSON.parse(fs.readFileSync(workspacePath(relative), 'utf8')), parseError: null}; }
  catch (e) { return {file, data: null, parseError: e.message}; }
}
function allFiles(relative) {
  const full = workspacePath(relative);
  if (!fs.existsSync(full)) return [];
  return fs.readdirSync(full, {withFileTypes: true}).flatMap(item => {
    const child = relative + '/' + item.name;
    return item.isDirectory() ? allFiles(child) : item.isFile() ? [record(child)] : [];
  });
}

const input = record('character-hq-textured.glb');
const editable = record(base + '/character-rebuilt.blend');
const glb = record(base + '/character-rebuilt-textured.glb');
const source = jsonEvidence(base + '/reports/final-source-geometry.json');
const binary = jsonEvidence(base + '/reports/final-glb-validation.json');
const reimport = jsonEvidence(base + '/reports/clean-reimport.json');
const visual = jsonEvidence(base + '/reports/final-visual-review.json');
const stage = source.data?.sourceStage ? record(source.data.sourceStage) : null;
const inputUnchanged = input.exists && input.sha256 === '105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2';
const sourceFresh = Boolean(editable.exists && glb.exists && source.data?.blendSha256 === editable.sha256 && source.data?.glbSha256 === glb.sha256 && stage?.sha256 === source.data?.sourceStageSha256);
const binaryFresh = Boolean(glb.exists && binary.data?.sha256 === glb.sha256 && binary.data?.passed === true);
const reimportFresh = Boolean(glb.exists && editable.exists && reimport.data?.glbSha256 === glb.sha256 && reimport.data?.blendSha256 === editable.sha256 && reimport.data?.structuralReimportPassed === true && reimport.data?.glbUnchangedDuringReimport === true);
const visualFresh = Boolean(glb.exists && visual.data?.glbSha256 === glb.sha256 && visual.data?.reviewCompleted === true);
const images = [...allFiles(base + '/checks/final'), ...allFiles(base + '/checks/reimport')].filter(f => /\.(png|jpg|jpeg)$/i.test(f.path));
const pending = [];
if (!inputUnchanged) pending.push('Original input hash does not match the recorded untouched input.');
if (!editable.exists || !glb.exists) pending.push('Final .blend and/or .glb file is absent.');
if (!sourceFresh) pending.push('Source export report is absent, stale, or not matched to final files and selected stage.');
if (!binaryFresh) pending.push('Matching passing GLB binary validation is absent.');
if (!reimportFresh) pending.push('Matching passing clean Blender reimport evidence is absent.');
if (!reimport.data?.reviewImagesRendered) pending.push('Clean-reimport review images have not been recorded as rendered.');
// Do not automatically certify likeness or five fingers from object names/report counts.
if (!visualFresh) pending.push('Matching agent visual review of actual final reimport views is not recorded.');

const manifest = {
  schemaVersion: 1,
  createdAt: new Date().toISOString(),
  status: sourceFresh && binaryFresh && reimportFresh && inputUnchanged && visualFresh ? 'technical-and-visual-inspection-recorded; artistic-limitations-remain' : 'pending-final-delivery-validation',
  completionClaim: false,
  appearanceAcceptedByUser: false,
  scope: 'Static anime T-pose character rebuild. File inventory and evidence correspondence only.',
  files: {input, editable, glb, sourceStage: stage},
  checks: {inputUnchanged, sourceExportReportMatches: sourceFresh, binaryValidationMatchesAndPasses: binaryFresh, cleanReimportMatchesAndPasses: reimportFresh, agentVisualReviewMatches: visualFresh},
  toolVersions: {locked: toolchain.environment, actualBlender: reimport.data?.blenderVersion ?? source.data?.blenderVersion ?? null},
  statistics: binaryFresh ? { ...binary.data.summary, imageDimensions: binary.data.images.map(i => ({name: i.name, width: i.width, height: i.height, format: i.format, sha256: i.sha256})), coordinateConvention: 'glTF Y up; source/reimport world-bounds correspondence reported separately' } : null,
  evidence: {source: source.file, binary: binary.file, reimport: reimport.file, visual: visual.file},
  parseErrors: [source, binary, reimport].filter(e => e.parseError).map(e => ({path: e.file.path, error: e.parseError})),
  reviewImages: images,
  evidenceLimits: ['A successful structural audit does not prove likeness, clean silhouettes, no intersections, correct fingers or texture appearance.', 'Source and exported model can have UV/material seam vertex count differences; triangle counts are compared independently.', 'Original hand details not visible in the reference are inferred. The final completion note must describe those inferred features.', 'No skeleton/animation production is required this round; report actual exported skins/animations above.', 'No off-machine backup or user visual acceptance is implied.'],
  pending,
};
fs.mkdirSync(workspacePath(base + '/reports'), {recursive: true});
const destination = base + '/reports/delivery-manifest.json';
fs.writeFileSync(workspacePath(destination), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify({manifest: destination, status: manifest.status, structuralChecks: manifest.checks, completionClaim: false}, null, 2));
