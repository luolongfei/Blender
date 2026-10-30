"""Record actual new-face artifacts and visible failures, never acceptance."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def entry(relative):
    p=P/relative
    return {'path':relative,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
state_path=P/'project-state.json';old=state_path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
head=entry('work/reconstruction-r02/reference-adult-female-head-r11.blend')
brows=entry('work/reconstruction-r02/reference-adult-female-brows-r11.blend')
if head['sha256']!='080f5d5d6b214deace3148923028f7ffac1a12ce748d953008e2e2be79ec3e48' or brows['sha256']!='575c5ff9732963229b54a644360a8e70170575193d5e329abb4fc93783489b14':raise RuntimeError('Saved stage changed')
images=[]
for mode in ('structure','groom'):
    for view in ('front','quarter','side'):
        case=mode+'-'+view;relative='previews/reconstruction-r02/adult-female-head-r11/'+case
        report=json.loads((P/(relative+'.json')).read_text())
        if report['sourceSha256']!=head['sha256']:raise RuntimeError('View belongs to another source')
        image=entry(relative+'.png')
        if image['sha256']!=report['imageSha256']:raise RuntimeError('Rendered image changed')
        images.append({'case':case,'image':image,'report':entry(relative+'.json')})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'adult-female-base-retained-for-reconstruction; integrated-appearance-failed',
    'headCandidate':head,'latestIndependentCandidate':brows,
    'masterChanged':False,'currentMaster':entry('work/character.blend'),
    'userObjectiveCopy':entry('references/user-goal-objective-20261028-update.md'),
    'actualSixViews':images,
    'independentVisibleReview':entry('work/reconstruction-r02/adult-female-head-visible-review-r11.json'),
    'actualNeckObservation':entry('work/reconstruction-r02/adult-neck-observation-r11.json'),
    'workingObjects':{'head':'R11 Female head study','eyes':['R11 Adult sclera L','R11 Adult sclera R'],'collection':'R11 Adult female face candidate'},
    'visibleDecision':{'faceBase':'Retain adult eye/nose/lip volume for further shaping and expression work.',
        'neckBodyFit':'FAILED: exposed upper-chest/shoulder stub and jagged posterior cut edge.',
        'groomFit':'FAILED: new scalp penetrates old crown, temples/nape have floating shell gaps, fringe crosses an eye.',
        'brows':'Actually added to a new independent copy; neutral stills are being produced.',
        'appearanceAccepted':False,'photorealisticReady':False},
    'next':['Join the actual new neck boundary to the retained body/collar region.','Refit root layer, scalp and fringe to the new head without inflating an opaque cap.','Review the new natural brows and apply a restrained natural light-makeup skin candidate.','Rebuild/check face performance, then revalidate motion, physics and runtime adaptation.'],
    'limitations':['Current head is a static shape candidate with rigid head-parent motion, not a new verified face rig.','Legacy scripts hardcoded to R01 Head cannot be treated as new-head adaptation.','Old hair/cloth caches remain as existing context; new-head collision compatibility is not proven.','No new final exports or off-machine backup.']}
out=P/'work/reconstruction-r02/face-rebuild-progress-r11.json'
with out.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
state['reconstructionR11']=record;state['recordedAt']=record['recordedAt']
if state_path.read_bytes()!=old:raise RuntimeError('Concurrent state edit')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('R11_PROGRESS_RECORDED',brows['sha256'])
