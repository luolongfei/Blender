"""Record actual saved R09 work and unresolved visual issues, not acceptance."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def record(relative):
    file=project_path()/relative
    if not file.is_file():return {'path':relative,'exists':False}
    return {'path':relative,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}

def run(candidate='work/reconstruction-r02/reference-lower-face-candidate-r09.blend'):
    p=project_path();state_path=p/'project-state.json'
    state=json.loads(state_path.read_text(encoding='utf-8-sig'))
    if state.get('masterIntegrationR09'):
        raise RuntimeError('R09 master integration exists. This historical candidate checkpoint must not reset master provenance/integrated flags; use master_integrate_r09 instead.')
    saved=record(candidate)
    if not saved.get('sha256'):raise RuntimeError('Explicit saved candidate required')
    data=state['reconstructionR02'];data['currentSavedCandidate']=saved;data['integratedIntoMaster']=False
    data['masterVerifiedUnchanged']=record('work/character.blend')['sha256']==state['workingMaster']['sha256']
    data['latestGeometryPass']={'stage':'R09 local likeness refinement in progress',
        'changes':['R08 optional production UV, preserving original shader and UV state',
                   'R09 lower-face proportion candidate: 11 percent weighted vertical compression; protected ocular region unchanged',
                   'All head relative keys and oral attachments mapped consistently, with driven jaw compensation',
                   'Posterior scalp coverage added and first bounded forelock flow pass saved independently',
                   'Measured posterior root curvature and scalp/gather strands transported together; real rear views adopted',
                   'Larger forelock-volume candidate rejected after visible sheet-like result; retained as history'],
        'neutralVisualEvidence':'previews/reconstruction-r02/face-proportion-r09/render-report.json',
        'knownLimits':['Still visibly stylized CG, not established as AAA/photoreal',
                       'Posterior root curvature adopted after actual back/rear-quarter views; final full character review remains',
                       'First bounded forelock change had only a small visible effect; volume redesign is in progress',
                       'Actual R09 EE/OU renders still expose square perioral and chin deformation; correction is pending']}
    data['remaining']=['Complete R09 forelock and mouth/eye real-view refinement',
        'Render the matching R09 body/hair/cloth continuous motion after final visible source changes',
        'Refresh runtime face/scalp/fringe and calibrate actual hair/sleeve collision',
        'Integrate master, reopen all required outputs, prepare verified delivery package']
    state['reconstructionR09']={'currentSavedCandidate':saved,
        'bodyAction':record('work/reconstruction-r02/body-action-r09.blend'),
        'bodyActionReview':'previews/reconstruction-r02/body-poses-r09/render-report.json',
        'bodyCombined':record('work/reconstruction-r02/reference-body-combined-r09.blend'),
        'bodyCombinedReopen':'work/reconstruction-r02/reference-body-combined-r09.blend.reopen.json',
        'bodyCombinedVisualEvidence':'previews/reconstruction-r02/body-combined-sparse-r09/render-report.json',
        'bodyCombinedStatus':'Matching 301-frame native hair/cloth integrated and fresh reopened. Five actual rendered poses completed; squat/bend/reach inspected without the prior large garment penetrations. Continuous video remains pending.',
        'rootSurface':record('work/reconstruction-r02/reference-root-flow-r09.blend'),
        'groomVisualDecision':'work/reconstruction-r02/groom-visible-decision-r09.json',
        'clothNative':record('work/reconstruction-r02/body-cloth-native-r09.blend'),
        'clothEvidenceDirectory':'work/reconstruction-r02/cloth-body-r09',
        'clothStatus':'301 recorded simulation frames saved with native playback; fresh-file finite parity and four actual poses completed. Squat/bend cloth visually covers the previous leg penetration. Final matched hair/cloth combination remains pending.',
        'clothVisualEvidence':'previews/reconstruction-r02/cloth-poses-r09/render-report.json',
        'clothMergePayload':record('work/reconstruction-r02/body-cloth-payload-r09.blend'),
        'lowerFaceEvidence':'work/reconstruction-r02/reference-lower-face-candidate-r09.json',
        'r08WalkVideo':'previews/reconstruction-r02/combined-walk-continuous-r08/walk-preview-r08.mp4',
        'r08FaceSparse':'previews/reconstruction-r02/face-performance-sparse-r08/render-report.json',
        'runtimeSource':record('work/runtime-r06/r09-tail-colliders.blend'),
        'runtimePreview':record('work/runtime-r06/preview-tail-colliders-r09/review-preview.vrm'),
        'runtimeStatus':'Saved Blender runtime is still prior collider-link candidate. A separately identified JSON-parameter VRM test materially reduces shoulder clipping in the actual viewer; its tested settings remain to be integrated into the editable runtime.',
        'runtimeParameterCandidate':record('work/runtime-r06/preview-tail-envelope-test-r09/test-preview.vrm'),
        'runtimeParameterComparison':'work/runtime-r06/r09-tail-profile-comparison.json',
        'runtimePriorVisualEvidence':'work/runtime-r06/runtime-viewer-observation-r08.json',
        'masterIntegrated':False,'newFormalExports':False,'offMachineCopyVerified':False}
    if 'priorDeliveryLimitations' not in state:state['priorDeliveryLimitations']=state.get('knownLimitations',[])
    state['knownLimitations']=['Current R09 source is still a working candidate; the fixed master and formal export names still contain the older delivery.',
        'Current physical renders remain visibly stylized; AAA/photoreal appearance and user/client acceptance have not been established.',
        'R09 forelock stiffness and box-like mouth morphs still need source refinement and actual combined review; a larger sheet-like forelock candidate was rejected.',
        'R09 body action has matching hair and cloth with fresh reopen and five pose renders; continuous motion video is still pending.',
        'Runtime target device and performance budget are not specified; no frame-rate guarantee.',
        'Author metadata and redistribution license are not invented; review VRM metadata remains explicit placeholder.']
    state['acceptance'].update(appearanceReviewedByUser=False,photorealisticClientDeliveryReady=False,
        technicalChecks='See exact-source reports. Recent R09 updates are not covered by prior migration or R08 visual checks.')
    state['recordedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    registry=p/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
    data['paths']=sorted(set(data['paths'])|{f.relative_to(p).as_posix() for f in (p/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
    registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidate':saved,'masterUnchanged':state['reconstructionR02']['masterVerifiedUnchanged']}))

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'work/reconstruction-r02/reference-lower-face-candidate-r09.blend')
