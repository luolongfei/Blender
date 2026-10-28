#!/usr/bin/env node
import fs from 'node:fs/promises';
import { constants } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const codeExtensions = new Set(['.py', '.mjs', '.js', '.html', '.css', '.md', '.txt', '.toml', '.yaml', '.yml']);
const excludedDirectories = new Set(['.git', 'node_modules', '__pycache__']);
const metadataNames = new Set(['.asset-code-paths.json', 'project-state.json']);

function projectId(value) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value) || value === '..') {
    throw new Error('Invalid project ID; use one directory name without slashes.');
  }
  return value;
}

function relativePath(value) {
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0')
      || /^[A-Za-z]:/.test(value) || path.posix.isAbsolute(value)
      || value.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error(`Unsafe relative path: ${JSON.stringify(value)}`);
  }
  return value;
}

function sameStat(a, b) {
  return a.dev === b.dev && a.ino === b.ino && a.size === b.size
    && a.mtimeNs === b.mtimeNs && a.ctimeNs === b.ctimeNs;
}

async function maybeStat(filename) {
  try { return await fs.lstat(filename, { bigint: true }); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

async function regularStat(filename) {
  const stat = await fs.lstat(filename, { bigint: true });
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`Not a regular file: ${filename}`);
  return stat;
}

async function stableHash(filename) {
  const before = await regularStat(filename);
  const handle = await fs.open(filename, 'r');
  try {
    const opened = await handle.stat({ bigint: true });
    if (!sameStat(before, opened)) throw new Error(`File changed while opening: ${filename}`);
    const hash = createHash('sha256');
    for await (const chunk of handle.createReadStream({ autoClose: false })) hash.update(chunk);
    const after = await handle.stat({ bigint: true });
    const current = await regularStat(filename);
    if (!sameStat(before, after) || !sameStat(before, current)) {
      throw new Error(`File changed during hashing: ${filename}`);
    }
    return { bytes: Number(before.size), sha256: hash.digest('hex'), stat: before };
  } finally { await handle.close(); }
}

async function safeDirectory(base, relative, create = false) {
  let current = base;
  for (const part of relative.split('/').filter(Boolean)) {
    current = path.join(current, part);
    let stat = await maybeStat(current);
    if (!stat && create) {
      await fs.mkdir(current).catch(error => { if (error.code !== 'EEXIST') throw error; });
      stat = await maybeStat(current);
    }
    if (!stat || stat.isSymbolicLink() || !stat.isDirectory()) {
      throw new Error(`Missing or unsafe directory: ${current}`);
    }
  }
  return current;
}

async function context(options, create = false) {
  const id = projectId(options.project);
  const root = await fs.realpath(path.resolve(options.root ?? defaultRoot));
  const directory = await safeDirectory(root, `projects/${id}`, create);
  return { id, root, directory, manifest: path.join(root, 'asset-manifests', `${id}.json`) };
}

async function codePaths(directory) {
  const filename = path.join(directory, '.asset-code-paths.json');
  if (!(await maybeStat(filename))) return null;
  await regularStat(filename);
  const data = JSON.parse(await fs.readFile(filename, 'utf8'));
  const paths = Array.isArray(data) ? data : data.paths;
  if (!Array.isArray(paths)) throw new Error(`${filename} must contain an array or {"paths":[...]}.`);
  return new Set(paths.map(relativePath));
}

async function assertCodePaths(directory, expected) {
  const current = await codePaths(directory);
  if ((current === null) !== (expected === null) || (current && (current.size !== expected.size
      || [...current].some(value => !expected.has(value))))) {
    throw new Error('Asset code paths changed during the operation. Pause writers and retry.');
  }
}

function omitted(relative, codes) {
  const parts = relative.split('/');
  const name = parts.at(-1);
  if (parts.some(part => excludedDirectories.has(part))) return true;
  if (name === '.DS_Store' || name.endsWith('.pyc') || name.startsWith('.asset-copy-')) return true;
  if (name === '.env' || name.startsWith('.env.') || /^(auth|credentials)\.json$/i.test(name)
      || /^(id_rsa|id_ed25519)$/.test(name) || /\.(pem|key)$/i.test(name)) return true;
  if (metadataNames.has(relative)) return true;
  return codes ? codes.has(relative) : codeExtensions.has(path.extname(name).toLowerCase());
}

async function listAssets(directory, codes) {
  const rows = [];
  async function visit(relative = '') {
    const entries = await fs.readdir(path.join(directory, relative), { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name, 'en'));
    for (const entry of entries) {
      const rel = relative ? `${relative}/${entry.name}` : entry.name;
      if (omitted(rel, codes)) continue;
      relativePath(rel);
      const filename = path.join(directory, ...rel.split('/'));
      const stat = await fs.lstat(filename, { bigint: true });
      if (stat.isSymbolicLink()) throw new Error(`Symbolic links are not supported: ${rel}`);
      if (stat.isDirectory()) await visit(rel);
      else if (stat.isFile()) rows.push({ path: rel, stat });
      else throw new Error(`Unsupported file type: ${rel}`);
    }
  }
  await visit();
  return rows;
}

async function assertSnapshot(directory, rows, codes) {
  const current = await listAssets(directory, codes);
  const before = new Map(rows.map(row => [row.path, row.stat]));
  if (current.length !== rows.length || current.some(row => !before.has(row.path)
      || !sameStat(before.get(row.path), row.stat))) {
    throw new Error('Asset files changed during the operation. Save and pause the producer, then retry.');
  }
}

async function saveManifest(ctx, manifest) {
  await safeDirectory(ctx.root, 'asset-manifests', true);
  const existing = await maybeStat(ctx.manifest);
  if (existing && (!existing.isFile() || existing.isSymbolicLink())) {
    throw new Error(`Unsafe manifest destination: ${ctx.manifest}`);
  }
  const temp = `${ctx.manifest}.${randomUUID()}.tmp`;
  try {
    await fs.writeFile(temp, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
    await fs.rename(temp, ctx.manifest);
  } finally { await fs.rm(temp, { force: true }); }
}

export async function inventoryProject(options) {
  const ctx = await context(options);
  const codes = await codePaths(ctx.directory);
  const rows = await listAssets(ctx.directory, codes);
  const files = [];
  for (const row of rows) {
    const value = await stableHash(path.join(ctx.directory, ...row.path.split('/')));
    if (!sameStat(row.stat, value.stat)) throw new Error(`File changed during inventory: ${row.path}`);
    files.push({ path: row.path, bytes: value.bytes, sha256: value.sha256 });
  }
  await assertSnapshot(ctx.directory, rows, codes);
  await assertCodePaths(ctx.directory, codes);
  const manifest = {
    version: 1, projectId: ctx.id, assetRoot: `projects/${ctx.id}`,
    snapshotAt: new Date().toISOString(),
    totalFiles: files.length, totalBytes: files.reduce((sum, file) => sum + file.bytes, 0), files,
  };
  await saveManifest(ctx, manifest);
  return { status: 'ok', command: 'inventory', projectId: ctx.id,
    manifest: `asset-manifests/${ctx.id}.json`, totalFiles: manifest.totalFiles, totalBytes: manifest.totalBytes };
}

function validateManifest(manifest, ctx) {
  if (manifest.version !== 1 || manifest.projectId !== ctx.id
      || manifest.assetRoot !== `projects/${ctx.id}` || !Array.isArray(manifest.files)) {
    throw new Error('Manifest version, project ID, asset root or file list is invalid.');
  }
  const seen = new Set();
  for (const entry of manifest.files) {
    relativePath(entry.path);
    if (seen.has(entry.path) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0
        || !/^[a-f0-9]{64}$/.test(entry.sha256)) throw new Error(`Invalid manifest entry: ${entry.path}`);
    seen.add(entry.path);
  }
  if (manifest.totalFiles !== manifest.files.length
      || manifest.totalBytes !== manifest.files.reduce((sum, entry) => sum + entry.bytes, 0)) {
    throw new Error('Manifest file count or byte total is inconsistent.');
  }
}

export async function verifyProject(options) {
  const ctx = await context(options);
  await safeDirectory(ctx.root, 'asset-manifests');
  await regularStat(ctx.manifest);
  const manifest = JSON.parse(await fs.readFile(ctx.manifest, 'utf8'));
  validateManifest(manifest, ctx);
  const failures = [];
  const verified = [];
  for (const entry of manifest.files) {
    try {
      await safeDirectory(ctx.directory, path.posix.dirname(entry.path) === '.' ? '' : path.posix.dirname(entry.path));
      const actual = await stableHash(path.join(ctx.directory, ...entry.path.split('/')));
      if (actual.bytes !== entry.bytes || actual.sha256 !== entry.sha256) {
        failures.push({ path: entry.path, reason: 'changed' });
      } else verified.push({ path: entry.path, stat: actual.stat });
    } catch (error) {
      failures.push({ path: entry.path, reason: error.code === 'ENOENT' ? 'missing' : error.message });
    }
  }
  const codes = await codePaths(ctx.directory);
  const rows = await listAssets(ctx.directory, codes);
  const expected = new Set(manifest.files.map(entry => entry.path));
  const extras = rows.filter(row => !expected.has(row.path)).length;
  for (const entry of verified) {
    try {
      const parent = path.posix.dirname(entry.path);
      await safeDirectory(ctx.directory, parent === '.' ? '' : parent);
      const current = await regularStat(path.join(ctx.directory, ...entry.path.split('/')));
      if (!sameStat(entry.stat, current)) failures.push({ path: entry.path, reason: 'changed during verification' });
    } catch (error) {
      failures.push({ path: entry.path, reason: error.code === 'ENOENT' ? 'missing' : error.message });
    }
  }
  return { status: failures.length ? 'failed' : 'ok', command: 'verify', projectId: ctx.id,
    checkedFiles: manifest.files.length, checkedBytes: manifest.files.reduce((sum, row) => sum + row.bytes, 0),
    extraFiles: extras, failures };
}

export async function copyProject(options) {
  if (!options.source) throw new Error('copy requires --source <project directory>.');
  const ctx = await context(options, true);
  const sourceInput = path.resolve(options.source);
  const sourceStat = await fs.lstat(sourceInput);
  if (!sourceStat.isDirectory() || sourceStat.isSymbolicLink()) throw new Error('Source must be a real directory.');
  const source = await fs.realpath(sourceInput);
  const relation = path.relative(source, ctx.directory);
  const reverse = path.relative(ctx.directory, source);
  if (!relation || (!relation.startsWith(`..${path.sep}`) && relation !== '..' && !path.isAbsolute(relation))
      || (!reverse.startsWith(`..${path.sep}`) && reverse !== '..' && !path.isAbsolute(reverse))) {
    throw new Error('Source and destination must be separate, non-nested directories.');
  }
  const codes = await codePaths(ctx.directory);
  if (!codes && await codePaths(source)) {
    throw new Error('Source has .asset-code-paths.json but destination does not. Restore Git-managed project metadata before copying assets.');
  }
  const rows = await listAssets(source, codes);
  const result = { status: 'ok', command: 'copy', projectId: ctx.id, copied: 0, skipped: 0, copiedBytes: 0, failures: [] };
  for (const row of rows) {
    const from = path.join(source, ...row.path.split('/'));
    const to = path.join(ctx.directory, ...row.path.split('/'));
    let temp;
    try {
      const value = await stableHash(from);
      if (!sameStat(row.stat, value.stat)) throw new Error('Source changed since copy started.');
      const parent = path.posix.dirname(row.path);
      await safeDirectory(ctx.directory, parent === '.' ? '' : parent, true);
      const old = await maybeStat(to);
      if (old) {
        const existing = await stableHash(to);
        if (existing.sha256 === value.sha256 && existing.bytes === value.bytes) {
          result.skipped++; continue;
        }
        if (!options.refresh) throw new Error('Destination differs; preserved. Use --refresh only to replace it deliberately.');
      }
      temp = path.join(path.dirname(to), `.asset-copy-${randomUUID()}`);
      await fs.copyFile(from, temp, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE);
      const copied = await stableHash(temp);
      if (copied.sha256 !== value.sha256 || copied.bytes !== value.bytes
          || !sameStat(value.stat, await regularStat(from))) throw new Error('Source changed during copying.');
      const now = await maybeStat(to);
      if (old) {
        if (!now || !sameStat(old, now)) throw new Error('Destination changed during copying; preserved.');
        await fs.rename(temp, to);
        temp = undefined;
      } else {
        if (now) throw new Error('Destination appeared during copying; preserved.');
        await fs.copyFile(temp, to, constants.COPYFILE_EXCL | constants.COPYFILE_FICLONE);
      }
      result.copied++; result.copiedBytes += value.bytes;
    } catch (error) {
      result.failures.push({ path: row.path, reason: error.message });
    } finally {
      if (temp) await fs.rm(temp, { force: true });
    }
  }
  try {
    await assertSnapshot(source, rows, codes);
    await assertCodePaths(ctx.directory, codes);
  }
  catch (error) { result.failures.push({ path: '.', reason: error.message }); }
  if (result.failures.length) result.status = 'failed';
  return result;
}

function parseArgs(args) {
  const [command, ...rest] = args;
  const options = {};
  for (let i = 0; i < rest.length; i++) {
    const key = rest[i];
    if (key === '--refresh') options.refresh = true;
    else if (['--project', '--root', '--source'].includes(key)) {
      const value = rest[++i];
      if (!value || value.startsWith('--')) throw new Error(`Missing value for ${key}.`);
      options[key.slice(2)] = value;
    } else throw new Error(`Unknown argument: ${key}`);
  }
  return { command, options };
}

async function main() {
  try {
    if (!process.argv[2] || ['--help', '-h', 'help'].includes(process.argv[2])) {
      console.log('Usage: node scripts/assets.mjs <inventory|verify|copy> --project <id> [--root <workspace>]\n'
        + 'copy also requires --source <project directory>; --refresh permits replacement of different assets.\n'
        + 'Pause source writers before copying or inventory. No command deletes project files.\n'
        + 'An explicit .asset-code-paths.json identifies Git-managed files; all other regular files are assets.');
      return;
    }
    const { command, options } = parseArgs(process.argv.slice(2));
    const action = { inventory: inventoryProject, verify: verifyProject, copy: copyProject }[command];
    if (!action) throw new Error(`Unknown command: ${command}`);
    const result = await action(options);
    console.log(JSON.stringify(result, null, 2));
    if (result.status !== 'ok') process.exitCode = 2;
  } catch (error) {
    console.error(JSON.stringify({ status: 'failed', error: error.message }, null, 2));
    process.exitCode = 2;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
