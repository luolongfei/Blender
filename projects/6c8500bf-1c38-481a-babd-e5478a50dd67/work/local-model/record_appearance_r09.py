"""Record the integrator's actual image decisions, without claiming final likeness acceptance."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def record(relative):
    path=P/relative
    if not path.is_file():raise FileNotFoundError(path)
    return {'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)}

def run():
    candidate=record('work/reconstruction-r02/reference-production-refined-r09.blend')
    rendered=json.loads((P/'previews/reconstruction-r02/appearance-refined-r09/render-report.json').read_text())
    if rendered['sha256']!=candidate['sha256']:raise RuntimeError('New source does not match the actual neutral images')
    if {row['view'] for row in rendered['views']}!={'face-front','face-quarter'}:raise RuntimeError('Both actual neutral views required')
    expression=[]
    for case in ('aa42','ee63','ou98'):
        file='previews/reconstruction-r02/lower-jaw-keyframes-r09/'+case+'-report.json'
        report=json.loads((P/file).read_text())
        if report['sourceSha256']!=record('work/reconstruction-r02/reference-lower-jaw-curvature-r09.blend')['sha256']:
            raise RuntimeError('Expression evidence changed source')
        expression.append({'report':record(file),'image':record('previews/reconstruction-r02/lower-jaw-keyframes-r09/'+case+'.png')})
    decision={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'candidate':candidate,'decisionBy':'integrating agent','adoptedAsWorkingImprovement':True,
        'clientAppearanceAccepted':False,'photorealisticDeliveryReady':False,
        'adopted':['Local perioral target deltas, preserving the mapped lip-ring targets',
                   'Connected dental arch orientation/contact adjustment',
                   '380-vertex reversible local lower-jaw curvature key, with original keys retained',
                   'Skin optical adjustment and softer sclera',
                   'Warm iris with one retained sharp reflection layer; broad grey reflection visibly reduced',
                   'Previously inspected posterior PN root/scalp transport and bounded small-flow forelocks'],
        'rejected':[{'layer':'R09 elliptical forelock volume','source':record('work/reconstruction-r02/reference-forelock-ellipse-r09.blend'),
                     'evidence':'previews/reconstruction-r02/appearance-ellipse-r09/render-report.json',
                     'reason':'Same-light quarter view exposes dark root seams and broad smooth ribbon surfaces. Restored the exact retained small-flow native curves in the new combination.'}],
        'neutralEvidence':record('previews/reconstruction-r02/appearance-refined-r09/render-report.json'),
        'neutralImages':[record('previews/reconstruction-r02/appearance-refined-r09/'+name+'.png') for name in ('face-front','face-quarter')],
        'jawAndDentalExpressionEvidence':expression,
        'expressionScope':'These actual AA/EE/OU images use the same new jaw/dental geometry but precede only the final iris optical change and rejected-fringe restoration. They do not prove the final continuous performance.',
        'remaining':['Forelocks and tail masses remain visibly stylized; near-view hair is not established as photoreal.',
                     'Skin and facial structure remain stylized CG; the adopted jaw improvement does not resolve all square perioral transitions.',
                     'Fresh full-source integration, continuous body/face/walk, current runtime refresh and final lights/detail review remain required.']}
    out=P/'work/reconstruction-r02/appearance-visible-decision-r09.json'
    with out.open('x',encoding='utf-8') as stream:json.dump(decision,stream,ensure_ascii=False,indent=2)
    state_path=P/'project-state.json';state_sha=sha(state_path)
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    if state.get('masterIntegrationR09'):raise RuntimeError('Do not replace integrated source state with candidate-only state')
    state['reconstructionR02']['currentSavedCandidate']=candidate
    state['reconstructionR02']['integratedIntoMaster']=False
    state['reconstructionR02']['latestGeometryPass']['appearanceDecision']='work/reconstruction-r02/appearance-visible-decision-r09.json'
    state['reconstructionR09'].update(currentSavedCandidate=candidate,appearanceDecision='work/reconstruction-r02/appearance-visible-decision-r09.json')
    state['recordedAt']=decision['recordedAt']
    if sha(state_path)!=state_sha:raise RuntimeError('State changed concurrently')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
    data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
    registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('R09_APPEARANCE_DECISION',json.dumps(candidate),flush=True)

if __name__=='__main__':run()
