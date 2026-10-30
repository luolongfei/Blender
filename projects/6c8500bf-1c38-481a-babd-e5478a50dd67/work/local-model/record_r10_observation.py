"""Record current-source detail evidence and the next bounded geometry work."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(relative):
    path=P/relative
    return {'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)}
state_path=P/'project-state.json';state_sha=sha(state_path)
state=json.loads(state_path.read_text(encoding='utf-8-sig'))
master=identity('work/character.blend')
if master['sha256']!='3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166':
    raise RuntimeError('This bounded observation concerns the actual adopted R09 master')
cases=[]
for case in ('hand-left','hand-right','eye-mouth','face-side'):
    relative='previews/reconstruction-r02/master-details-r09/'+case+'-report.json'
    report=json.loads((P/relative).read_text())
    # The reporter includes the exact source path/hash in its source record.
    candidates=[report.get('sourceSha256'),report.get('sha256')]
    if isinstance(report.get('source'),dict):candidates.append(report['source'].get('sha256'))
    if master['sha256'] not in candidates:raise RuntimeError('Detail source mismatch: '+case)
    cases.append({'case':case,'report':identity(relative),'image':identity('previews/reconstruction-r02/master-details-r09/'+case+'.png')})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baselineMaster':master,
    'candidateBaseline':identity('work/reconstruction-r02/reference-production-refined-r09.blend'),
    'currentSourceDetails':cases,'crownObservation':identity('work/reconstruction-r02/crown-observation-r10.json'),
    'sourceChangedByObservation':False,'newR10GeometrySaved':False,
    'visibleFindings':['Current left/right hand images retain overly flat dorsal volumes and simple cylindrical digits; nails look like thin patches.',
                       'Current crown is narrow/low relative to the original portrait after eye-line alignment; whole-crown coherent volume should be tested.',
                       'Actual profile exposes an abrupt dark nape shell edge. Crown-volume change alone cannot establish a natural nape.',
                       'Iris broad grey reflection is reduced, but skin/face and hair remain visibly stylized CG.'],
    'next':['Observe and assess installed CC0 Human Base Meshes realistic hand asset before choosing shape transfer/replacement.',
            'Prepare one reversible coherent root/scalp/fringe/gather crown candidate from the exact actual observed arrays; preserve tied-tail anchors.',
            'Actually render adopted geometry and key poses before final runtime bakes and continuous delivery videos.'],
    'clientAppearanceAccepted':False,'photorealisticDeliveryReady':False}
out=P/'work/reconstruction-r02/current-detail-decision-r10.json'
with out.open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
state['reconstructionR10']=dict(record,decisionFile=str(out.relative_to(P)))
state['recordedAt']=record['recordedAt']
if sha(state_path)!=state_sha:raise RuntimeError('State changed concurrently')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('R10_DETAIL_DECISION',master['sha256'],flush=True)
