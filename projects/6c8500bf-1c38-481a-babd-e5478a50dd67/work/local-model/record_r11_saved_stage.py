"""Register an explicitly selected, hash-verified working candidate (not acceptance)."""
import argparse, datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()

def item(relative):
    path=(P/relative).resolve(); path.relative_to(P.resolve())
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(P).as_posix(),'bytes':path.stat().st_size,'sha256':digest}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',required=True); parser.add_argument('--sha',required=True)
parser.add_argument('--label',required=True); parser.add_argument('--note',required=True)
parser.add_argument('--report',action='append',default=[])
a=parser.parse_args(); source=item(a.source)
if source['sha256']!=a.sha.lower():raise RuntimeError('Explicit source hash mismatch')
if Path(source['path']).parent.as_posix()!='work/reconstruction-r02' or 'preview' in Path(source['path']).name.lower():
    raise RuntimeError('Only a non-preview independent reconstruction candidate may become latest')
entry={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':source,
    'reports':[item(p) for p in a.report],'note':a.note,'appearanceAccepted':False,
    'formalDelivery':False,'masterUpdated':False}
state_path=P/'project-state.json'; before=state_path.read_bytes(); state=json.loads(before.decode('utf-8-sig'))
r=state['reconstructionR11']; ledger=r.setdefault('savedStageLedger',{})
if a.label in ledger:raise RuntimeError('Stage label already recorded; preserve prior entry')
ledger[a.label]=entry; r['latestIndependentCandidate']=source
r['status']='active-original-design-reconstruction; appearance-and-delivery-incomplete'
state['recordedAt']=entry['recordedAt']
if state_path.read_bytes()!=before:raise RuntimeError('Concurrent state update')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json'; data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'recordedStage':a.label,'source':source},ensure_ascii=False))
