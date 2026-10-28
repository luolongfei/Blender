import fs from 'node:fs/promises';
import {createReadStream, createWriteStream} from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {spawn, spawnSync} from 'node:child_process';
import {createServer} from 'node:http';
import {Readable} from 'node:stream';
import {pipeline} from 'node:stream/promises';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const exists = async p => fs.access(p).then(() => true, () => false);
const json = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const hash = async p => {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(p)) h.update(chunk);
  return h.digest('hex');
};
export function inside(base, relative) {
  const resolved = path.resolve(base, relative);
  if (resolved !== base && !resolved.startsWith(base + path.sep)) {
    throw new Error(`Path escapes its root: ${relative}`);
  }
  return resolved;
}
async function settings() {
  const lock = await json(path.join(root, 'toolchain.lock.json'));
  const workspace = await json(path.join(root, 'config/workspace.json'));
  const local = await json(path.join(root, 'config/local.json')).catch(e => {
    if (e.code === 'ENOENT') return {};
    throw e;
  });
  return {lock, workspace, local, tool: inside(root, lock.charakuru.path)};
}
function option(args, key, fallback) {
  const i = args.indexOf(key);
  if (i < 0) return fallback;
  if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`Missing value for ${key}`);
  return args[i + 1];
}
async function verifyTool(tool, lock) {
  const manifest = await json(path.join(tool, 'RELEASE-MANIFEST.json'));
  if (manifest.version !== lock.charakuru.version) throw Error('Tool version differs from lock');
  for (const file of manifest.files) {
    if (await hash(inside(tool, file.path)) !== file.sha256) {
      throw Error(`Tool checksum mismatch: ${file.path}`);
    }
  }
  for (const file of ['server.mjs', 'scripts/mtoon-export.py', 'scripts/spring-setup.py',
    'public/preview-avatar.mjs', 'assets/donor/weighted-body.blend']) {
    if (!await exists(path.join(tool, file))) throw Error(`Incomplete release: ${file}`);
  }
  return manifest.files.length;
}
function python(local = {}) {
  const candidates = [local.pythonExecutable, process.env.PYTHON, 'python3', 'python'].filter(Boolean);
  for (const executable of candidates) {
    const r = spawnSync(executable, ['-c', 'import sys; assert sys.version_info >= (3,8)'],
      {stdio: 'ignore'});
    if (r.status === 0) return executable;
  }
  throw Error('Python 3.8+ is required for ZIP installation and Python syntax checks. Set config/local.json pythonExecutable.');
}
async function install(args, s) {
  if (await exists(s.tool)) {
    console.log(JSON.stringify({installed: true, verifiedFiles: await verifyTool(s.tool, s.lock)}));
    return;
  }
  const downloads = inside(root, path.join(s.workspace.stateRoot, 'downloads'));
  await fs.mkdir(downloads, {recursive: true});
  let archive = option(args, '--archive');
  if (archive) archive = path.resolve(archive);
  else {
    archive = path.join(downloads, `charakuru-${s.lock.charakuru.version}.zip`);
    if (!await exists(archive)) {
      const response = await fetch(s.lock.charakuru.archive.url);
      if (!response.ok || !response.body) throw Error(`Download failed: HTTP ${response.status}`);
      await pipeline(Readable.fromWeb(response.body), createWriteStream(archive + '.partial', {flags: 'wx'}));
      await fs.rename(archive + '.partial', archive);
    }
  }
  if (await hash(archive) !== s.lock.charakuru.archive.sha256) throw Error('Release ZIP checksum mismatch');
  const staging = await fs.mkdtemp(path.join(downloads, 'extract-'));
  const code = `
import pathlib, sys, zipfile, stat
archive, destination, expected = sys.argv[1:]
base = pathlib.Path(destination).resolve()
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        rel = pathlib.PurePosixPath(item.filename)
        if not rel.parts or rel.parts[0] != expected or rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe archive path: ' + item.filename)
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError('Symlink archive entry: ' + item.filename)
        target = base.joinpath(*rel.parts)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as src, target.open('xb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)
`;
  const result = spawnSync(python(s.local),
    ['-c', code, archive, staging, s.lock.charakuru.archive.rootDirectory], {encoding: 'utf8'});
  if (result.status !== 0) throw Error(result.stderr || 'ZIP extraction failed');
  const extracted = path.join(staging, s.lock.charakuru.archive.rootDirectory);
  const verifiedFiles = await verifyTool(extracted, s.lock);
  await fs.mkdir(path.dirname(s.tool), {recursive: true});
  await fs.rename(extracted, s.tool);
  await fs.rmdir(staging);
  console.log(JSON.stringify({installed: true, version: s.lock.charakuru.version, verifiedFiles}));
}
async function walk(dir) {
  const rows = [];
  if (!await exists(dir)) return rows;
  for (const entry of await fs.readdir(dir, {withFileTypes: true})) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory() && entry.name !== '__pycache__') rows.push(...await walk(p));
    else if (entry.isFile()) rows.push(p);
  }
  return rows;
}
async function validate(s) {
  const all = [...await walk(path.join(root, 'scripts')), ...await walk(path.join(root, 'pipeline'))];
  const projectRoot = inside(root, s.workspace.projectsRoot);
  for (const p of await fs.readdir(projectRoot, {withFileTypes: true})) {
    if (!p.isDirectory()) continue;
    const role = path.join(projectRoot, p.name);
    const list = await json(path.join(role, '.asset-code-paths.json')).catch(e => {
      if (e.code === 'ENOENT') return {paths: []};
      throw e;
    });
    all.push(...list.paths.map(f => inside(role, f)));
  }
  let js = 0, py = 0;
  for (const p of new Set(all)) {
    if (['.mjs', '.js'].includes(path.extname(p))) {
      const result = spawnSync(process.execPath, ['--check', p], {encoding: 'utf8'});
      if (result.status !== 0) throw Error(result.stderr);
      js++;
    }
  }
  const pythonFiles = [...new Set(all)].filter(p => p.endsWith('.py'));
  if (pythonFiles.length) {
    const code = 'import ast,sys,pathlib; [ast.parse(pathlib.Path(p).read_text(encoding="utf-8"), filename=p) for p in sys.argv[1:]]';
    const result = spawnSync(python(s.local), ['-c', code, ...pythonFiles], {encoding: 'utf8'});
    if (result.status !== 0) throw Error(result.stderr);
    py = pythonFiles.length;
  }
  console.log(JSON.stringify({valid: true, javascriptFiles: js, pythonFiles: py,
    scope: 'Syntax only; this does not execute Blender or certify model appearance.'}));
}
export function createPreviewServer({workspaceRoot, toolRoot}) {
  const types = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript',
    '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
    '.glb': 'model/gltf-binary', '.vrm': 'model/gltf-binary',
    '.vrma': 'model/gltf-binary', '.ico': 'image/x-icon'};
  return createServer(async (req, res) => {
    try {
      if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
      const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      const prefix = pathname.startsWith('/public/') ? '/public/' :
        pathname.startsWith('/projects/') ? '/projects/' : null;
      if (!prefix) { res.writeHead(404); res.end(); return; }
      const base = prefix === '/public/' ? path.join(toolRoot, 'public') : path.join(workspaceRoot, 'projects');
      const file = inside(base, pathname.slice(prefix.length));
      const actual = await fs.realpath(file);
      const realBase = await fs.realpath(base);
      inside(realBase, path.relative(realBase, actual));
      const type = types[path.extname(file).toLowerCase()];
      if (!type) { res.writeHead(403); res.end(); return; }
      const stat = await fs.stat(file);
      if (!stat.isFile()) { res.writeHead(404); res.end(); return; }
      const headers = {'Content-Type': type, 'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff'};
      let start = 0, end = stat.size - 1, status = 200;
      if (req.headers.range) {
        const m = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range);
        if (!m) { res.writeHead(416); res.end(); return; }
        start = Number(m[1]); end = m[2] ? Number(m[2]) : end;
        if (start > end || end >= stat.size) { res.writeHead(416); res.end(); return; }
        status = 206; headers['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
      }
      headers['Content-Length'] = stat.size ? end - start + 1 : 0;
      res.writeHead(status, headers);
      if (req.method === 'HEAD' || !stat.size) { res.end(); return; }
      await pipeline(createReadStream(actual, {start, end}), res);
    } catch (e) {
      if (!res.headersSent) res.writeHead(e.code === 'ENOENT' ? 404 : 403);
      res.end();
    }
  });
}
async function main() {
  const [command = 'doctor', ...args] = process.argv.slice(2);
  if (Number(process.versions.node.split('.')[0]) < 22) throw Error('Node.js 22+ required');
  const s = await settings();
  if (command === 'install-tool') return install(args, s);
  if (command === 'validate-source') return validate(s);
  if (command === 'doctor') {
    const checks = {node: process.versions.node, platform: process.platform, tool: s.lock.charakuru.version,
      toolVerified: false, project: s.workspace.defaultProject};
    try { checks.verifiedToolFiles = await verifyTool(s.tool, s.lock); checks.toolVerified = true; }
    catch (e) { checks.toolError = e.message; }
    const role = inside(root, path.join(s.workspace.projectsRoot, s.workspace.defaultProject));
    checks.workingBlendPresent = await exists(path.join(role, 'work/character.blend'));
    checks.assetManifestPresent = await exists(path.join(root, 'asset-manifests', s.workspace.defaultProject + '.json'));
    checks.blenderConnection = 'Not tested. Restore Blender, VRM Add-on and MCP separately.';
    console.log(JSON.stringify(checks, null, 2));
    if (!checks.toolVerified || !checks.workingBlendPresent || !checks.assetManifestPresent) process.exitCode = 1;
    return;
  }
  await verifyTool(s.tool, s.lock);
  if (command === 'start') {
    const state = inside(root, s.workspace.stateRoot);
    await fs.mkdir(state, {recursive: true});
    const stateFile = path.join(state, 'charakuru-session.json');
    const projectFile = inside(root, path.join(s.workspace.projectsRoot,
      s.workspace.defaultProject, 'project.charakit'));
    if (!await exists(stateFile) && await exists(projectFile)) {
      const {unpackProject} = await import(pathToFileURL(path.join(s.tool, 'public/project.mjs')));
      const bytes = await fs.readFile(projectFile);
      const {project} = await unpackProject(bytes);
      if (project.id !== s.workspace.defaultProject) throw Error('Saved project ID differs from workspace config');
      await fs.writeFile(stateFile, JSON.stringify({version: 0, request: null,
        active: {id: project.id, name: project.name, path: projectFile,
          hash: createHash('sha256').update(bytes).digest('hex')}}, null, 2) + '\n', {flag: 'wx'});
    }
    const env = {...process.env, PORT: option(args, '--port', String(s.workspace.appPort)),
      CHARAKIT_PROJECTS_ROOT: inside(root, s.workspace.projectsRoot),
      CHARAKIT_SESSION_FILE: stateFile};
    if (s.local.blenderExecutable) env.BLENDER = s.local.blenderExecutable;
    const externalConfig = s.local.charakuruConfig ?
      path.resolve(root, s.local.charakuruConfig) : path.join(root, 'config/charakuru.local.json');
    if (await exists(externalConfig)) env.CHARAKIT_CONFIG = externalConfig;
    else delete env.CHARAKIT_CONFIG;
    const child = spawn(process.execPath, [path.join(s.tool, 'server.mjs')], {cwd: s.tool, env, stdio: 'inherit'});
    child.on('error', e => { console.error(e.message); process.exitCode = 1; });
    child.on('exit', code => { process.exitCode = code ?? 1; });
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
    return;
  }
  if (command === 'preview') {
    const server = createPreviewServer({workspaceRoot: root, toolRoot: s.tool});
    const port = Number(option(args, '--port', String(s.workspace.previewPort)));
    server.listen(port, '127.0.0.1', () => console.log(
      `Preview: http://127.0.0.1:${server.address().port}/projects/${s.workspace.defaultProject}/previews/motion-viewer.html`));
    return;
  }
  throw Error(`Unknown command: ${command}`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(e => { console.error(e.message); process.exitCode = 1; });
}
