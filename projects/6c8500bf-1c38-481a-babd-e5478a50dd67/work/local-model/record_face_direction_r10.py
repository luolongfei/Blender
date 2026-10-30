"""Record the user's explicit rejection and changed production priority."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();state_path=P/'project-state.json';before=state_path.read_bytes()
state=json.loads(before.decode('utf-8-sig'))
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'userFeedback':'用户明确拒绝当前面部外观，认为与预期完全不同；要求精致、自然的日本女性面部。',
    'faceAppearanceAccepted':False,'priority':'Full adult feminine face structure reconstruction before further skin polish, continuous delivery rendering or runtime release.',
    'designDirection':'Natural adult female anatomy and refined soft facial forms informed by the requested Japanese woman appearance; preserve original character identity cues without forcing enlarged round eyes or flattened nose. Do not imply one universal ethnic face geometry.',
    'preserveRequirements':['Original black low twin tails and clothing design intent','Physically credible motion and details','Natural warm skin with subtle light makeup; slightly fair appearance permitted'],
    'parallelism':{'userRequestedSubagents':10,'availableConcurrentSubagents':3,
        'streams':['Face geometry reconstruction','Reusable mature female face asset research','Independent visible face review and proportion checks'],
        'integrator':'Only root writes the working master and runs heavy Blender tasks serially.'},
    'currentMaster':{'path':'work/character.blend','sha256':hashlib.sha256((P/'work/character.blend').read_bytes()).hexdigest()},
    'notDeliveryReady':True,
    'retainedIndependentWork':{'handCandidate':'work/reconstruction-r02/reference-asset-hands-r10-v2.blend',
        'handSha256':'7f45d5a151c8b5efb61afda46d482d97dc60ed5774a988847d82cd6447dd9b4b',
        'handStatus':'Actually saved and both dorsal stills rendered; no finger-flex, appearance adoption or master integration yet.',
        'tailShapeHelper':'work/local-model/tail_shape_candidate_r10.py',
        'tailStatus':'Offline continuity calculation only; not executed in Blender or adopted. Face takes priority.',
        'skinStatus':'Actual current material observation completed; light-foundation helper remains unapplied until new face direction is established.'}}
out=P/'work/reconstruction-r02/user-face-direction-r10.json'
with out.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
state['currentFaceDirection']=record;state['recordedAt']=record['recordedAt']
state.setdefault('reconstructionR10',{})['userFaceRejection']=record
if state_path.read_bytes()!=before:raise RuntimeError('Concurrent state change')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('RECORDED_CURRENT_USER_FACE_REJECTION')
