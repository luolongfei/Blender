"""Record current saved work and the explicit original-design constraint."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path(); stage='work/reconstruction-r02/'
def item(rel):
    path=P/rel
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':rel,'bytes':path.stat().st_size,'sha256':digest}
source=item(stage+'reference-adult-neck-surface-r11.blend')
assert source['sha256']=='43eef27aa61adaf8af5ba00faf73f9bdcc30d8ea5409d891ce5769dca4ce8ef6'
groom=item(stage+'reference-adult-unified-hair-r11.blend')
assert groom['sha256']=='b8b858b59ada63947489cf64c3e970c819582c24ff8f06f1deee7382e70e3eb1'
view='previews/reconstruction-r02/adult-full-groom-v5-r11/full-groom-front'
render=json.loads((P/(view+'.json')).read_text(encoding='utf8'))
png=item(view+'.png')
assert render['sourceSha256']==groom['sha256'] and render['imageSha256']==png['sha256']
direction={'identityReference':'references/user-original-portrait-2026-10-28.png',
    'priority':'Original user anime design is authoritative; generated realistic views are supporting references only.',
    'hair':'Keep black low twin tails, original behind-ear ties, layered separated fringe and fine face-framing strands. Repair grooming without redesigning the hairstyle.',
    'face':'Requested adult Japanese / East Asian woman appearance; follow the specific original gentle character, not a formula for ethnicity.',
    'skin':'Natural Asian complexion, slightly fair with restrained makeup.',
    'approvalRequiredForRoutineWork':False}
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'latestIndependentCandidate':source,'hairCandidate':groom,
    'actualHairReview':{'image':png,'report':item(view+'.json'),
        'decision':'REJECTED: exposed anterior crown, blunt brush-like fringe, thick side curtains and hard tail bends. Fewer tangles and unified tail/crown colour are improvements, not hairstyle acceptance.'},
    'closureCandidate':item(stage+'reference-adult-blink-closure-r11.blend'),
    'closureResidualAreaSquareMillimeters':{'L':5.188038144828768,'R':5.311566447745657},
    'closureAccepted':False,'neckSurface':'Actual material continuity candidate saved; original head/body and bridge geometry preserved. Image review pending.',
    'failedCandidates':{'eyeDepthV1':'Rejected before mutation because current eye transform is not uniform scale; full-matrix v2 prepared.',
        'roundedJawV1':'Rejected before save: protected region max 0.252210 mm, eye/ear max 0.002415 mm; exact domain diagnostic required.'},
    'artDirection':direction,'masterChanged':False,'formalExportsChanged':False,
    'appearanceAccepted':False,'photorealisticReady':False}
out=P/(stage+'groom-closure-progress-r11.json')
with out.open('x',encoding='utf8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
path=P/'project-state.json'; original=path.read_bytes(); state=json.loads(original.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=source
state['reconstructionR11']['groomClosureProgress']=record
state['reconstructionR11']['artDirectionLatest']=direction
state['reconstructionR11']['status']='original-design-guided-repair; groom-v5-and-full-closure-not-accepted; production-incomplete'
state['recordedAt']=record['recordedAt']
if path.read_bytes()!=original:raise RuntimeError('Concurrent project state modification')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json'; data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_GROOM_CLOSURE_RECORDED',source['sha256'])
