"""Record actual retained lip/mascara/low-neck candidates and failed look issues."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def item(path):
    p=P/path;return {'path':path,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
stage='work/reconstruction-r02/';source=item(stage+'reference-adult-low-neck-r11.blend')
if source['sha256']!='3a6baaa6271917cdd0c4c2b75c374651262ddde62d7ec7cb88e849fb31bb9d7f':raise RuntimeError('Intended source changed')
images=[]
for folder,case in [('adult-low-neck-r11','structure-front'),('adult-low-neck-r11','structure-side'),('adult-makeup-r11','mouth'),('adult-makeup-r11','structure-front'),('adult-makeup-r11','eyes')]:
    base='previews/reconstruction-r02/'+folder+'/'+case;report=json.loads((P/(base+'.json')).read_text());image=item(base+'.png')
    expected=source['sha256'] if folder=='adult-low-neck-r11' or case=='mouth' else 'be47284ec4b41518c3dfcd972f1206f767c2de8bb94d9c550ecc89349a3ac01a'
    if report['sourceSha256']!=expected or report['imageSha256']!=image['sha256']:raise RuntimeError('Actual image/source correspondence changed')
    images.append({'case':case,'sourceSha256':expected,'image':image,'report':item(base+'.json')})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'latestIndependentCandidate':source,'actualLookViews':images,
    'stages':[item(stage+n) for n in ('reference-adult-rose-lips-r11.lip-surface.json','reference-adult-mascara-r11.mascara.json','reference-adult-low-neck-r11.json')],
    'visibleDecision':{'neck':'Low side stubs and posterior angular step reduced, but overall thickness and posterior lower seam remain visibly unfinished.',
        'lips':'Restrained pink lip body is visible, but upper-lip bow/edge and volumetric detail are insufficient; no speech/contact acceptance.',
        'lashes':'Opaque mascara material correctly assigned/evaluated, but actual same-light image remains pale. Red-emission temporary diagnostic identifies current fibres; optical/denoising study pending.',
        'eyes':'Current adult scale looks too small relative to character reference; measured coherent globe/lid size candidate in progress.',
        'groom':'Native donor scalp domain mapped, then source-buried secondary guide tails triggered strict skin correction guard. No donor candidate saved yet.',
        'appearanceAccepted':False},
    'masterChanged':False,'master':item('work/character.blend'),'formalExportsChanged':False,
    'limitations':['Retained old groom and old dynamics are context only.','No newly accepted facial performance, continuous body/hair/cloth video, final VRM or backup freeze.']}
out=P/(stage+'makeup-neck-progress-r11.json')
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
path=P/'project-state.json';old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=source;state['reconstructionR11']['makeupNeckProgress']=record;state['recordedAt']=record['recordedAt']
if path.read_bytes()!=old:raise RuntimeError('Concurrent state write')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_MAKEUP_NECK_PROGRESS_RECORDED',source['sha256'])
