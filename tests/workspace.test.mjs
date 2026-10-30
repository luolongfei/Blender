import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {once} from 'node:events';
import {inside, createPreviewServer, validatePythonFiles} from '../scripts/workspace.mjs';

test('paths stay inside their assigned root', () => {
  const root = path.resolve(os.tmpdir(), 'test-workspace');
  assert.equal(inside(root, 'projects/a'), path.join(root, 'projects/a'));
  assert.throws(() => inside(root, '../outside'), /escapes/);
});

test('preview serves locked runtime and project resources, including binary ranges', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'preview-test-'));
  t.after(() => fs.rm(root, {recursive: true, force: true}));
  const tool = path.join(root, 'tool');
  await fs.mkdir(path.join(tool, 'public'), {recursive: true});
  await fs.mkdir(path.join(root, 'projects', 'a'), {recursive: true});
  await fs.writeFile(path.join(tool, 'public/runtime.mjs'), 'export const test = 1;');
  await fs.writeFile(path.join(root, 'projects/a/avatar.vrm'), '0123456789');
  await fs.writeFile(path.join(root, 'projects/a/secret.blend'), 'private');
  const server = createPreviewServer({workspaceRoot: root, toolRoot: tool});
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => server.close(resolve)));
  const base = `http://127.0.0.1:${server.address().port}`;
  const runtime = await fetch(base + '/public/runtime.mjs');
  assert.equal(runtime.status, 200);
  assert.match(await runtime.text(), /export const/);
  const range = await fetch(base + '/projects/a/avatar.vrm', {headers: {Range: 'bytes=2-5'}});
  assert.equal(range.status, 206);
  assert.equal(await range.text(), '2345');
  assert.equal((await fetch(base + '/projects/a/secret.blend')).status, 403);
  assert.equal((await fetch(base + '/tool/public/runtime.mjs')).status, 404);
  assert.equal((await fetch(base + '/public/missing.mjs')).status, 404);
});


test('Python validation handles lists beyond Windows argument limits and reports syntax errors', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'syntax-test-'));
  t.after(() => fs.rm(inside(os.tmpdir(), path.relative(os.tmpdir(), directory)), {recursive: true, force: true}));
  const valid = path.join(directory, '中文 source.py');
  const invalid = path.join(directory, 'invalid.py');
  await fs.writeFile(valid, 'value = 1\n');
  await fs.writeFile(invalid, 'def broken(:\n');
  const files = Array(2000).fill(valid);
  assert.ok(files.join(' ').length > 32767);
  await validatePythonFiles(files);
  await assert.rejects(validatePythonFiles([...files, invalid]), /SyntaxError/);
});
