"""Record explicit R07 provenance without promoting a candidate to master."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
def run(source_relative):
    p=project_path();source=p/source_relative
    stage='r08' if 'r08' in source.name else 'r07'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    state_path=p/'project-state.json';state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    if state.get('masterIntegrationR09'):
        raise RuntimeError('R09 master integration exists; this historical checkpoint must not reset its provenance. Use master_integrate_r09.')
    data=state['reconstructionR02']
    data['currentSavedCandidate']={'path':source_relative,'bytes':source.stat().st_size,'sha256':digest}
    data['integratedIntoMaster']=False
    data['masterVerifiedUnchanged']=hashlib.sha256((p/'work/character.blend').read_bytes()).hexdigest()==state['workingMaster']['sha256']
    data['latestGeometryPass']={
        'stage':'R07 face/iris and head-neck integration; visual refinement remains active',
        'changes':['Basis-aware eyebrow contact repair and local cheek continuity shape',
                   'Documented procedural iris v3 with wet clearcoat approximation',
                   'Calibrated neck/head rest pivot, revised neck weight transition and posterior neck contour',
                   'R06 continuous scalp/root groom and R05 original-reference proportions retained'],
        'knownLimits':['Still visibly stylized CG; photoreal/AAA appearance has not been established',
                       'Eyebrow fibre pattern and cheek-mouth transition remain under visual review']}
    proof_path=p/'work/reconstruction-r02'/('combined-walk-reopen-'+stage+'.json')
    proof=json.loads(proof_path.read_text(encoding='utf-8')) if proof_path.exists() else {}
    match=proof.get('sha256')==digest
    data['checks']['currentCombinedWalk']={
        'report':proof_path.relative_to(p).as_posix(),'matchesCurrentSavedCandidate':match,
        'clothMaxPayloadErrorMeters':proof.get('cloth',{}).get('payload_max_error_m'),
        'hairMaxPointErrorMeters':proof.get('hair',{}).get('maximumErrorMeters'),
        'visualReview':'Four actual frames 1/15/35/61 rendered; no large garment breakthrough seen at this view. Continuous motion and additional views pending.',
        'scope':'80-frame CMU-derived walk with authored blink; embedded cloth payload and guide dynamics. No external cache needed for these two playback objects.'}
    data['checks']['priorEvidence']='Other R03/R04/R05 and CMU foot records remain historical and apply only to their explicit source/geometry/action.'
    if stage=='r08':
        data['latestGeometryPass']['stage']='R08 local cheek-mouth transition and eyebrow flow refinement, adopted after two real closeup renders'
        data['latestGeometryPass']['changes']+=['R08 cheek-side depth continuity: 100 vertices >0.1 mm; exact ocular/lip bands remain fixed',
            'R08 rank-distributed eyebrow fibres and tapered flow, actual common adjustment strength 0.8']
        data['latestGeometryPass']['visualEvidence']='previews/reconstruction-r02/face-flow-r08/render-report.json'
        data['latestGeometryPass']['knownLimits']=['Still visibly stylized CG; photoreal/AAA appearance has not been established',
            'Current-source combined facial performance and continuous-motion review still in progress']
    data['evidence']['combinedWalkR07']='work/reconstruction-r02/combined-walk-r07.json'
    data['evidence']['combinedWalkReopenR07']=proof_path.relative_to(p).as_posix()
    data['evidence']['currentCombinedWalkSparse']='previews/reconstruction-r02/combined-walk-sparse-'+stage+'/render-report.json'
    data['remaining']=['Current-source final face and expression refinement/review',
        'Continuous integrated walk and additional body-performance review',
        'Runtime PBR bake, actual export/viewer appearance/expression/physics verification',
        'Final master integration, exact source-export correspondence and delivery package']
    state['recordedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    registry=p/'.asset-code-paths.json';registered=json.loads(registry.read_text(encoding='utf-8-sig'))
    registered['paths']=sorted(set(registered['paths'])|{f.relative_to(p).as_posix() for f in (p/'work/local-model').glob('*') if f.suffix in {'.py','.js','.mjs'}})
    registry.write_text(json.dumps(registered,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data['currentSavedCandidate']))
if __name__=='__main__':run(sys.argv[1])
