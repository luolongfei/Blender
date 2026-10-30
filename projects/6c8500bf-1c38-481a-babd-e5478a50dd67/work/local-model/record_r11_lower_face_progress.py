"""Record the actual shortened-face source and its three rendered views."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def item(rel):
    p=P/rel
    return {'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
source=item('work/reconstruction-r02/reference-adult-lower-face-r11.blend')
assert source['sha256']=='9fc66ef37f9fbda70cb5260dfc81344b43da7ff6888e7a31372c425cf484a379'
views=[]
for case in ('structure-front','structure-quarter','structure-side'):
    base='previews/reconstruction-r02/adult-proportions-r11/'+case
    report=json.loads((P/(base+'.json')).read_text());img=item(base+'.png')
    assert report['sourceSha256']==source['sha256'] and report['imageSha256']==img['sha256']
    views.append({'case':case,'sourceSha256':source['sha256'],'image':img,'report':item(base+'.json')})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latestIndependentCandidate':source,
    'actualViews':views,'actualStage':item('work/reconstruction-r02/reference-adult-lower-face-r11.lower-face.json'),
    'visibleDecision':{'retain':'Lower-face shortening visibly reduces excessive chin length; retain for further reconstruction.',
        'pending':'Nose remains projected/angular, upper lip and mouth corners insufficiently soft, brows/lashes pale, eye expression stiff. Neck width/seams and reference hairstyle remain unfinished.',
        'appearanceAccepted':False,'photorealisticReady':False},
    'generatedExpressionReference':[item('references/generated/realistic-character-expression-board-r11.'+ext) for ext in ('png','json')],
    'generatedReferenceScope':'Concept/reference only; not actual Blender or physical contact evidence. Actual generation model not exposed.',
    'master':item('work/character.blend'),'masterChanged':False,'formalExportsChanged':False,
    'pendingExecution':'Narrow-neck, upper-neck bridge, upper lip and donor restyle v2 are being executed separately; this record does not claim they succeeded.'}
out=P/'work/reconstruction-r02/lower-face-progress-r11.json'
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
path=P/'project-state.json';old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=source
state['reconstructionR11']['lowerFaceProgress']=record
state['recordedAt']=record['recordedAt']
if path.read_bytes()!=old:raise RuntimeError('Concurrent project-state update')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_LOWER_FACE_PROGRESS_RECORDED',source['sha256'])
