"""Read-only current-source dependency/credit audit, run by the sole integrator.

Uses the established source audit. Does not evaluate modifiers, change frames,
save/pack a blend, export, simulate, or determine license/appearance acceptance.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def metadata(block):
    data = getattr(block, 'asset_data', None)
    return None if data is None else {key: getattr(data, key, None)
                                     for key in ('author', 'description', 'copyright', 'license')}


def probe(source_relative, expected_sha256, output_relative, allow_master=False):
    import bpy
    project = project_path().resolve()
    source = (project / source_relative).resolve()
    output = (project / output_relative).resolve()
    source.relative_to(project)
    output.relative_to(project)
    if output.suffix != '.json' or output.exists():
        raise RuntimeError('Use a new in-project JSON audit output')
    if Path(bpy.data.filepath).resolve() != source or bpy.data.is_dirty:
        raise RuntimeError('Audit requires exact saved source with no unsaved changes')
    if sha(source) != expected_sha256:
        raise RuntimeError('Opened source hash differs from requested source')
    helper = Path(__file__).with_name('production_source_audit_r05.py')
    spec = importlib.util.spec_from_file_location('r11_original_source_audit', helper)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.report(expected_source_relative=source_relative, allow_master=allow_master)
    result['r11Supplement'] = {
        'recordedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'helperSha256': sha(Path(__file__)), 'baseAuditSha256': sha(helper),
        'originalTextNotices': [{'name': t.name, 'sha256': hashlib.sha256(t.as_string().encode('utf-8')).hexdigest(),
                                'content': t.as_string()}
                               for t in bpy.data.texts
                               if t.name in {'Hair demo file info', 'License', 'README'}
                               or ('credit' in t.name.lower() or 'license' in t.name.lower())],
        'nodeGroupMetadata': [{'name': g.name, 'users': g.users,
                               'library': g.library.filepath if g.library else None,
                               'assetMetadata': metadata(g),
                               'attributionLabels': [{'name': n.name, 'label': n.label}
                                                     for n in g.nodes
                                                     if any(s in (n.name + ' ' + n.label).lower()
                                                            for s in ('bystedt', 'copyright', 'license', 'author', 'credit'))]}
                              for g in bpy.data.node_groups],
        'objectAssetMetadata': [{'name': o.name, 'assetMetadata': metadata(o)}
                                for o in bpy.data.objects if getattr(o, 'asset_data', None)],
        'reproductionMarkers': [{'owner': 'Scene/' + scene.name, 'key': key,
                                 'value': str(scene[key]) if len(str(scene[key])) <= 32768 else None,
                                 'characters': len(str(scene[key])),
                                 'sha256': hashlib.sha256(str(scene[key]).encode('utf-8')).hexdigest()}
                                for scene in bpy.data.scenes for key in scene.keys()
                                if any(tag in key.lower() for tag in ('cache', 'native', 'hair', 'cloth', 'r11'))],
        'limits': ['Asset metadata and notices are recorded as found; missing metadata is not permission to invent a license.',
                   'Scene custom-property markers describe saved intent, not successful cache replay or current collision validity.',
                   'This inventory supplements production_source_audit_r05; its explicitly stated coverage limitations remain.',
                   'No generated-model selection proof, legal clearance, actual visual approval or delivery completion is inferred.']}
    if bpy.data.is_dirty or sha(source) != expected_sha256:
        raise RuntimeError('Source changed during read-only audit')
    result['writtenReport'] = output.relative_to(project).as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('R11_DEPENDENCIES_OBSERVED', result['source']['sha256'], result['summary'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-master', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    probe(args.source, args.expected_sha256, args.output, args.allow_master)
