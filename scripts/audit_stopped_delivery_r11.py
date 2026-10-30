"""Read saved-Blend dependency paths for a user-requested stopped handoff.

No scene evaluation, rendering, channel edits, asset generation or Blend save.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
source = Path(bpy.data.filepath)
def file_status(name):
    try:
        item = Path(name)
        exists = item.is_file()
        return {'exists': exists, 'bytes': item.stat().st_size if exists else None}
    except OSError as exc:
        return {'exists': None, 'bytes': None, 'pathError': str(exc)}

images = []
for item in bpy.data.images:
    if item.type in {'RENDER_RESULT', 'COMPOSITING'}:
        continue
    packed = bool(item.packed_file) or bool(len(item.packed_files))
    raw = item.filepath_raw
    resolved = bpy.path.abspath(raw, library=item.library) if raw else ''
    images.append({'name': item.name, 'source': item.source, 'path': raw,
                   'resolvedPath': resolved, 'packed': packed,
                   'fileStatus': file_status(resolved) if resolved and not packed else None})
paths = sorted(set(bpy.utils.blend_paths(absolute=True, packed=False, local=False)))
external = [{'path': name, **file_status(name)}
            for name in paths if name]
caches = []
for obj in bpy.data.objects:
    for mod in obj.modifiers:
        cache = getattr(mod, 'point_cache', None)
        if cache is not None:
            caches.append({'object': obj.name, 'modifier': mod.name, 'type': mod.type,
                           'baked': cache.is_baked, 'diskCache': cache.use_disk_cache,
                           'path': cache.filepath})
with source.open('rb') as stream:
    source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
report = {'source': str(source), 'sourceSha256': source_sha,
          'blenderVersion': bpy.app.version_string, 'sourceSaved': False,
          'sceneEvaluated': False, 'images': images, 'externalPaths': external,
          'libraries': [{'path': lib.filepath, 'linkedData':
              {category: [item.name for item in getattr(bpy.data, category) if item.library == lib]
               for category in ('objects', 'meshes', 'materials', 'node_groups', 'images', 'worlds', 'brushes')}}
              for lib in bpy.data.libraries], 'simulationCaches': caches,
          'objects': len(bpy.data.objects), 'materials': len(bpy.data.materials),
          'packedImages': sum(x['packed'] for x in images), 'imageCount': len(images)}
out = Path(a.output)
if out.exists():
    raise RuntimeError('Preserve prior audit')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('STOPPED_DELIVERY_DEPENDENCIES ' + json.dumps({k: report[k] for k in
      ['sourceSha256', 'packedImages', 'imageCount', 'objects', 'sourceSaved']}), flush=True)
