import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { inventoryProject, verifyProject, copyProject } from '../scripts/assets.mjs';

async function fixture(t) {
  const base = await fs.mkdtemp(path.join(os.tmpdir(), 'blender-assets-'));
  t.after(() => fs.rm(base, { recursive: true, force: true }));
  const root = path.join(base, 'workspace');
  const directory = path.join(root, 'projects', 'character');
  const source = path.join(base, 'original');
  await fs.mkdir(directory, { recursive: true });
  await fs.mkdir(source);
  return { base, root, directory, source, options: { root, project: 'character' } };
}

async function write(base, name, content) {
  const filename = path.join(base, ...name.split('/'));
  await fs.mkdir(path.dirname(filename), { recursive: true });
  await fs.writeFile(filename, content);
}

test('explicit code inventory preserves untracked checkpoint source and detects corrupt or missing assets', async t => {
  const f = await fixture(t);
  await write(f.directory, '.asset-code-paths.json', JSON.stringify({ paths: ['work/local-model/build.py', 'previews/viewer.mjs'] }));
  await write(f.directory, 'work/local-model/build.py', 'source');
  await write(f.directory, 'previews/viewer.mjs', 'source');
  await write(f.directory, 'work/checkpoint/build.py', 'historical source');
  await write(f.directory, 'work/character.blend', 'blend-one');
  await write(f.directory, 'textures/skin.png', 'png');
  await write(f.directory, 'report.json', '{}');
  await write(f.directory, 'project-state.json', '{}');
  await write(f.directory, '.env', 'DO_NOT_COPY=secret');
  await write(f.directory, '.git/config', 'not an asset');
  await write(f.directory, '__pycache__/build.pyc', 'cache');
  const inventory = await inventoryProject(f.options);
  assert.equal(inventory.totalFiles, 4);
  const manifest = JSON.parse(await fs.readFile(path.join(f.root, inventory.manifest), 'utf8'));
  assert.equal(manifest.assetRoot, 'projects/character');
  assert.ok(manifest.files.some(file => file.path === 'work/checkpoint/build.py'));
  assert.equal(JSON.stringify(manifest).includes(f.root), false);
  assert.equal((await verifyProject(f.options)).status, 'ok');
  await write(f.directory, 'extra.bin', 'unlisted');
  await write(f.directory, 'work/character.blend', 'blend-two');
  await fs.unlink(path.join(f.directory, 'textures/skin.png'));
  const result = await verifyProject(f.options);
  assert.equal(result.status, 'failed');
  assert.equal(result.extraFiles, 1);
  assert.deepEqual(result.failures.map(file => file.path).sort(), ['textures/skin.png', 'work/character.blend']);
});

test('copy keeps changed destinations unless refresh is explicit and never removes extras', async t => {
  const f = await fixture(t);
  await write(f.directory, '.asset-code-paths.json', JSON.stringify(['code.py']));
  await write(f.directory, 'code.py', 'new adapter');
  await write(f.source, 'code.py', 'old adapter');
  await write(f.source, 'source-checkpoint.py', 'preserve historical code');
  await write(f.source, 'work/character.blend', 'new model');
  await write(f.source, 'textures/skin.png', 'identical');
  await write(f.directory, 'work/character.blend', 'old model');
  await write(f.directory, 'textures/skin.png', 'identical');
  await write(f.directory, 'keep.bin', 'keep');
  const result = await copyProject({ ...f.options, source: f.source });
  assert.equal(result.status, 'failed');
  assert.equal(result.skipped, 1);
  assert.equal(result.copied, 1);
  assert.match(result.failures[0].reason, /Destination differs/);
  assert.equal(await fs.readFile(path.join(f.directory, 'work/character.blend'), 'utf8'), 'old model');
  const refreshed = await copyProject({ ...f.options, source: f.source, refresh: true });
  assert.equal(refreshed.status, 'ok');
  assert.equal(await fs.readFile(path.join(f.directory, 'work/character.blend'), 'utf8'), 'new model');
  assert.equal(await fs.readFile(path.join(f.directory, 'code.py'), 'utf8'), 'new adapter');
  assert.equal(await fs.readFile(path.join(f.directory, 'keep.bin'), 'utf8'), 'keep');
  assert.equal((await fs.readdir(path.join(f.directory, 'work'))).some(name => name.startsWith('.asset-copy-')), false);
});

test('copy requires destination code metadata and preserves source code omitted by the new code list', async t => {
  const f = await fixture(t);
  await write(f.source, '.asset-code-paths.json', JSON.stringify(['code.py', 'old-code.py']));
  await write(f.source, 'code.py', 'old adapter');
  await write(f.source, 'old-code.py', 'historical source');
  await write(f.source, 'model.blend', 'model');
  await write(f.directory, 'code.py', 'new adapter');
  await assert.rejects(copyProject({ ...f.options, source: f.source, refresh: true }), /Restore Git-managed project metadata/);
  assert.equal(await fs.readFile(path.join(f.directory, 'code.py'), 'utf8'), 'new adapter');
  await assert.rejects(fs.stat(path.join(f.directory, 'model.blend')), { code: 'ENOENT' });
  await write(f.directory, '.asset-code-paths.json', JSON.stringify(['code.py']));
  const copied = await copyProject({ ...f.options, source: f.source, refresh: true });
  assert.equal(copied.status, 'ok');
  assert.equal(copied.copied, 2);
  assert.equal(await fs.readFile(path.join(f.directory, 'code.py'), 'utf8'), 'new adapter');
  assert.equal(await fs.readFile(path.join(f.directory, 'old-code.py'), 'utf8'), 'historical source');
  assert.equal((await inventoryProject(f.options)).totalFiles, 2);
  assert.equal((await verifyProject(f.options)).status, 'ok');
});

test('copy refuses a changing source before replacing an existing destination', async t => {
  const f = await fixture(t);
  await write(f.source, 'model.blend', 'new model');
  await write(f.directory, 'model.blend', 'saved model');
  const sourceFile = await fs.realpath(path.join(f.source, 'model.blend'));
  const originalCopyFile = fs.copyFile;
  t.mock.method(fs, 'copyFile', async (from, to, ...args) => {
    await originalCopyFile(from, to, ...args);
    if (from === sourceFile) await write(f.source, 'model.blend', 'source changed');
  });
  const result = await copyProject({ ...f.options, source: f.source, refresh: true });
  assert.equal(result.status, 'failed');
  assert.equal(result.copied, 0);
  assert.match(result.failures[0].reason, /Source changed during copying/);
  assert.equal(await fs.readFile(path.join(f.directory, 'model.blend'), 'utf8'), 'saved model');
  assert.deepEqual(await fs.readdir(f.directory), ['model.blend']);
});

test('verify detects an earlier file changing while a later file is hashed', async t => {
  const f = await fixture(t);
  await write(f.directory, 'a.blend', 'model-a');
  await write(f.directory, 'b.blend', 'model-b');
  await inventoryProject(f.options);
  const originalOpen = fs.open;
  const laterFile = await fs.realpath(path.join(f.directory, 'b.blend'));
  let mutated = false;
  t.mock.method(fs, 'open', async (filename, ...args) => {
    if (filename === laterFile && !mutated) {
      mutated = true;
      await write(f.directory, 'a.blend', 'changed-a');
    }
    return originalOpen(filename, ...args);
  });
  const result = await verifyProject(f.options);
  assert.equal(mutated, true);
  assert.equal(result.status, 'failed');
  assert.deepEqual(result.failures, [{ path: 'a.blend', reason: 'changed during verification' }]);
});

test('verify rejects a manifest with a truncated file list', async t => {
  const f = await fixture(t);
  await write(f.directory, 'model.blend', 'model');
  const inventory = await inventoryProject(f.options);
  const filename = path.join(f.root, inventory.manifest);
  const manifest = JSON.parse(await fs.readFile(filename, 'utf8'));
  manifest.files = [];
  await fs.writeFile(filename, JSON.stringify(manifest));
  await assert.rejects(verifyProject(f.options), /file count or byte total/);
});

test('rejects path traversal from project ID, code list and manifest before reading outside assets', async t => {
  const f = await fixture(t);
  await assert.rejects(inventoryProject({ ...f.options, project: '../outside' }), /Invalid project ID/);
  await write(f.directory, '.asset-code-paths.json', JSON.stringify(['../outside']));
  await assert.rejects(inventoryProject(f.options), /Unsafe relative path/);
  await write(f.directory, '.asset-code-paths.json', '[]');
  await write(f.directory, 'model.blend', 'model');
  await inventoryProject(f.options);
  const filename = path.join(f.root, 'asset-manifests/character.json');
  const manifest = JSON.parse(await fs.readFile(filename, 'utf8'));
  manifest.files[0].path = '../outside';
  await fs.writeFile(filename, JSON.stringify(manifest));
  await assert.rejects(verifyProject(f.options), /Unsafe relative path/);
});

test('refuses symlinks, nested copy locations and writes through linked project parents', async t => {
  const f = await fixture(t);
  await write(f.directory, '.asset-code-paths.json', '[]');
  await write(f.source, 'source.blend', 'source');
  try {
    await fs.symlink(path.join(f.source, 'source.blend'), path.join(f.directory, 'linked.blend'));
  } catch (error) {
    if (error.code === 'EPERM') return t.skip('Symlinks require platform privileges.');
    throw error;
  }
  await assert.rejects(inventoryProject(f.options), /Symbolic links/);
  await fs.unlink(path.join(f.directory, 'linked.blend'));
  await assert.rejects(copyProject({ ...f.options, source: f.directory }), /non-nested/);
  await fs.symlink(f.source, path.join(f.directory, 'linked-dir'), 'junction');
  await assert.rejects(inventoryProject(f.options), /Symbolic links/);
});

test('default mode excludes source files while still inventorying binary assets and reports', async t => {
  const f = await fixture(t);
  await write(f.directory, 'build.py', 'code');
  await write(f.directory, 'README.md', 'documentation');
  await write(f.directory, 'character.blend', 'model');
  await write(f.directory, 'report.json', '{}');
  assert.equal((await inventoryProject(f.options)).totalFiles, 2);
});
