"""Resolve portable workspace and pinned Charakuru paths without importing Blender."""
import json
from pathlib import Path, PureWindowsPath


def find_workspace(start=None):
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if ((directory / "toolchain.lock.json").is_file()
                and (directory / "config/workspace.json").is_file()):
            return directory
    raise RuntimeError(f"Cannot locate Blender workspace from {current}")


def _within(root, relative):
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute()
            or ".." in relative.replace("\\", "/").split("/")):
        raise ValueError(f"Expected a workspace-relative path, got {relative!r}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"Path leaves the workspace: {relative}")
    return result


def workspace_root():
    return find_workspace()


def workspace_path(*parts):
    return _within(workspace_root(), "/".join(parts))


def tool_root():
    root = workspace_root()
    lock = json.loads((root / "toolchain.lock.json").read_text(encoding="utf-8"))
    return _within(root, lock["charakuru"]["path"])


def project_path(project_id=None):
    root = workspace_root()
    config = json.loads((root / "config/workspace.json").read_text(encoding="utf-8"))
    project_id = config["defaultProject"] if project_id is None else project_id
    if (not isinstance(project_id, str) or not project_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in project_id)):
        raise ValueError("Project ID must be a single directory name.")
    return _within(root, f"{config.get('projectsRoot', 'projects')}/{project_id}")
