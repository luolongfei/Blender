"""Record actual R11 work copies and the rejected crown/fringe appearance."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def entry(s):
    p=P/s
    return {'path':s,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
stage='work/reconstruction-r02/'
candidate=entry(stage+'reference-adult-female-lashes-r11.blend')
if candidate['sha256']!='9be0d305a1fa57a949df84f5b8db4087e8238d6feab05000118934239ddd7ab4':raise RuntimeError('Actual new source changed')
skin=entry(stage+'reference-adult-female-skin-r11.blend');images=[]
for case in ('groom-front','groom-quarter'):
    base='previews/reconstruction-r02/adult-female-skin-r11/'+case
    report=json.loads((P/(base+'.json')).read_text());item=entry(base+'.png')
    if report['sourceSha256']!=skin['sha256'] or report['imageSha256']!=item['sha256']:raise RuntimeError('Actual image correspondence changed')
    images.append({'case':case,'image':item,'report':entry(base+'.json')})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'recordedAt':now,'latestIndependentCandidate':candidate,'skinCandidate':skin,
        'groomCandidate':entry(stage+'reference-adult-female-groom-r11.blend'),'actualLookViews':images,
        'actualFaceContactAudit':entry(stage+'face-contact-actual-r11.json'),
        'actualAnatomicalNeckObservation':entry(stage+'adult-neck-anatomical-observation-r11.json'),
        'visibleDecision':{'scalpContact':'Improved visible scalp fit; retained as a geometric starting point.',
            'crownAndFringe':'REJECTED: short cap-like fringe, coarse parallel strips and lateral floating turn; active re-combing task.',
            'skin':'Restrained warmer base and lip variation saved; does not establish photorealism or specific ethnic appearance.',
            'neck':'Visible shoulder stub and incomplete cervical connection remain; bridge task active.',
            'lashes':'Saved after local-normal root-growth fix; neutral macro and face renders pending.',
            'appearanceAccepted':False},
        'masterChanged':False,'currentMaster':entry('work/character.blend'),'formalExportsChanged':False,
        'pending':['Continuous neck/body fit and neck weights','New crown/fringe style','Face shape and expressive refinement','Brow/lash surface binding with blink and gaze','Oral rig and speech','Integrated motion/physics and runtime refresh'],
        'limits':['Working copy includes rejected fringe pending replacement.','Old tail/cloth caches are retained context, not new-head collision evidence.','No new off-machine backup or frozen full-project archive.']}
out=P/(stage+'face-look-progress-r11.json')
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
state_path=P/'project-state.json';old=state_path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=candidate;state['reconstructionR11']['lookProgress']=record;state['recordedAt']=now
if state_path.read_bytes()!=old:raise RuntimeError('Concurrent state edit')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_LOOK_PROGRESS_RECORDED',candidate['sha256'])
