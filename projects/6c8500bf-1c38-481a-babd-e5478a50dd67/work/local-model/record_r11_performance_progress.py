"""Record actually saved expression candidates and observed failures, not acceptance."""
import datetime, hashlib, json, sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path
P = project_path()
STAGE = 'work/reconstruction-r02/'

def item(rel):
    path = P/rel
    return {'path': rel, 'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}

source = item(STAGE+'reference-adult-natural-smile-r11.blend')
assert source['sha256'] == 'b0c7089ca09022bd59ae6b1068b68f78f46629ad87722f60ef54df5aa0858447'
blink = item(STAGE+'reference-adult-blink-r11.blend')
assert blink['sha256'] == '28250e7948afa3f8a9fc7e29caef8034a1a97f4e17b926b984e42cf5307fe1dc'
review = 'previews/reconstruction-r02/blink-closure-diagnostic-r11/'
image = item(review+'frame-0001.png')
frame = json.loads((P/(review+'frame-0001.json')).read_text())
assert frame['sourceSha256'] == source['sha256'] and frame['imageSha256'] == image['sha256']
master = item('work/character.blend')
assert master['sha256'] == '3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166'
record = {
    'recordedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'latestIndependentCandidate': source, 'blinkCandidate': blink,
    'reports': [item(STAGE+x) for x in ('reference-adult-blink-r11.blink.json',
        'reference-adult-natural-smile-r11.natural-smile.json',
        'fibre-parity-reference-change-r11.json')],
    'actualClosedEyeReview': {'image': image, 'report': item(review+'render-report.json'),
        'decision': 'REJECTED closure: clear sclera gaps at both corners, exaggerated U-shaped upper lid and thick lid contour. The center closes. Corrective geometry is prepared but not established by this report.'},
    'actualFindings': {
        'fibreBinding': 'Stepwise float32 reference using actual GN input passes unchanged 2-micron tolerance; older double-reference discrepancy retained in reports. This validates transport, not lid contact.',
        'smile': 'Saved at zero with measured cheek/lower-lid movement. Full smile appearance pending; local lip-section crossing still requires signed-depth diagnosis.',
        'groomV4': 'Rejected before save: big/small clumps select 488/145 cross-group guides and move roots about 8.97 mm. Role partition reduced head-contact maximum correction from 39.17 to 9.75 mm; no final point below -0.5 mm skin clearance. V5 is being executed separately.',
        'collar': 'First candidate rejected before save because left collar orientation folded. Independent continuous-displacement repair in progress.',
        'wrist': 'First candidate stopped before save because triangulation changed during pose comparison. V2 checks copied original polygon/loop topology separately from triangulation; not yet executed.'},
    'master': master, 'masterChanged': False, 'formalExportsChanged': False,
    'appearanceAccepted': False, 'photorealisticReady': False,
    'limits': ['No new motion or runtime acceptance.', 'No frozen or verified off-machine backup.',
        'Generated references and lip-striation resource are not production render evidence.']}
output = P/(STAGE+'performance-progress-r11.json')
with output.open('x', encoding='utf8') as stream:
    json.dump(record, stream, ensure_ascii=False, indent=2)
path = P/'project-state.json'
original = path.read_bytes()
state = json.loads(original.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate'] = source
state['reconstructionR11']['performanceProgress'] = record
state['reconstructionR11']['status'] = 'blink-smile-installed; closure-and-groom-rejected-under-repair; delivery-incomplete'
state['recordedAt'] = record['recordedAt']
if path.read_bytes() != original:
    raise RuntimeError('Concurrent project-state update')
path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
registry = P/'.asset-code-paths.json'
data = json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths'] = sorted(set(data['paths']) | {f.relative_to(P).as_posix()
    for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print('R11_PERFORMANCE_RECORDED', source['sha256'])
