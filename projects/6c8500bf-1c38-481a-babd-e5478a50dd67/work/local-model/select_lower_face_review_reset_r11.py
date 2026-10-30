"""Keep the rejected continuity layer editable but select its reviewed zero state."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'source-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    project = project_path().resolve()
    source, output = [(project / value).resolve() for value in (args.source, args.output)]
    for path in (source, output):
        path.relative_to((project / 'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('A fresh background source is required')
    if digest(source) != args.source_sha or output.exists() or output.with_suffix('.json').exists():
        raise RuntimeError('Source hash mismatch or prior output exists')
    head = bpy.data.objects['R11 Female head study']
    control = 'R11 lower face continuity'
    previous = float(head[control])
    if previous != 1.0:
        raise RuntimeError('Expected the actually reviewed strength-one candidate')
    head[control] = 0.0
    head.update_tag()
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    if digest(source) != args.source_sha:
        raise RuntimeError('Source changed on disk')
    report = {
        'source': args.source, 'sourceSha256': args.source_sha,
        'output': args.output, 'outputSha256': digest(output),
        'control': control, 'previousValue': previous, 'selectedValue': 0.0,
        'reason': 'Actual identical-camera quarter A/B showed a new thin lip-root ridge at strength one; reject that appearance while preserving the editable layer.',
        'review': 'previews/reconstruction-r02/lower-face-continuity-quarter-r11/render-report.json',
        'appearanceAccepted': False, 'masterUpdated': False, 'formalDelivery': False,
    }
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('R11_LOWER_FACE_REVIEW_RESET ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
