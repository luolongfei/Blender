import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export function findWorkspace(start = fileURLToPath(import.meta.url)) {
  let current = path.resolve(start);
  if (fs.existsSync(current) && fs.statSync(current).isFile()) current = path.dirname(current);
  for (;;) {
    if (fs.existsSync(path.join(current, 'toolchain.lock.json')) &&
        fs.existsSync(path.join(current, 'config', 'workspace.json'))) return current;
    const parent = path.dirname(current);
    if (parent === current) throw new Error(`Cannot locate Blender workspace from ${start}`);
    current = parent;
  }
}

function within(root, relative) {
  if (typeof relative !== 'string' || !relative || path.posix.isAbsolute(relative) ||
      path.win32.isAbsolute(relative) || relative.split(/[\\/]/).includes('..')) {
    throw new Error(`Expected a workspace-relative path, got ${JSON.stringify(relative)}`);
  }
  const resolved = path.resolve(root, relative);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new Error(`Path leaves the workspace: ${relative}`);
  }
  return resolved;
}

export const workspaceRoot = findWorkspace();
export const workspaceConfig = JSON.parse(
  fs.readFileSync(path.join(workspaceRoot, 'config', 'workspace.json'), 'utf8'));
export const toolchain = JSON.parse(
  fs.readFileSync(path.join(workspaceRoot, 'toolchain.lock.json'), 'utf8'));
export const toolRoot = within(workspaceRoot, toolchain.charakuru.path);
export const workspacePath = (...parts) => within(workspaceRoot, parts.join('/'));
export const toolPath = (...parts) => within(toolRoot, parts.join('/'));

export function projectPath(id = workspaceConfig.defaultProject) {
  if (typeof id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(id)) {
    throw new Error('Project ID must be a single directory name.');
  }
  return workspacePath(workspaceConfig.projectsRoot || 'projects', id);
}
