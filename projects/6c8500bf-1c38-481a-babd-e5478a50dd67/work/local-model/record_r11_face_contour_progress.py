"""Record only actual saved/rendered R11 face work; never promote the master."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def entry(s):
    p=P/s;h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':s,'bytes':p.stat().st_size,'sha256':h.hexdigest()}
prefix='work/reconstruction-r02/'
candidate=entry(prefix+'reference-adult-dark-iris-r11.blend')
if candidate['sha256']!='504503e475b1ccdf05a8f30008baacefa5f22fc8f7108ceb5ed74b05de6dcb6d':raise RuntimeError('Actual intended source changed')
images=[]
for case in ('structure-front','structure-quarter','eyes'):
    base='previews/reconstruction-r02/adult-face-contour-r11/'+case
    report=json.loads((P/(base+'.json')).read_text());item=entry(base+'.png')
    if report['sourceSha256']!=candidate['sha256'] or report['imageSha256']!=item['sha256']:raise RuntimeError('Actual image/source mismatch')
    images.append({'case':case,'image':item,'report':entry(base+'.json')})
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
stage_reports=[entry(prefix+name) for name in ('reference-adult-eye-surface-r11.eye-surface.json',
    'reference-adult-face-fibres-bound-r11.fibre-binding.json','reference-adult-jaw-contour-r11.json',
    'reference-adult-upper-lid-r11.json','reference-adult-dark-iris-r11.iris.json')]
record={'recordedAt':now,'latestIndependentCandidate':candidate,'actualLookViews':images,'actualStages':stage_reports,
    'generatedResources':[entry('references/generated/realistic-character-three-views-r11.png'),
        entry('references/generated/realistic-character-three-views-r11.json'),
        entry('textures/generated-r11/iris-dark-brown-r11.png'),entry('textures/generated-r11/iris-dark-brown-r11.json')],
    'visibleDecision':{'iris':'New generated dark brown iris is visibly darker/more suitable; original pixels packed. Not a calibrated scan or full eye optics.',
        'face':'Adult anatomy retained as foundation; target softness, eye proportions, expression and likeness remain unfinished.',
        'lashes':'Current bound v3 lashes render too pale; independent opaque mascara candidate pending actual image.',
        'lips':'Existing lip volume reads as skin plus dark mouth slit; restrained real lip-body material candidate pending.',
        'neck':'Current bridge retains lateral low-neck bulges and posterior angle; first contour attempt failed before save because upperChest group was absent on bridge.',
        'groom':'277676 refined crown rejected: two tall bright crown masses, needle fringe, exposed cap; retained separately. Current face uses older bb678d groom context.',
        'appearanceAccepted':False},
    'donorHairPreview':{'image':entry('previews/reconstruction-r02/hair-donor-native-r11/donor-quarter.png'),
        'report':entry('previews/reconstruction-r02/hair-donor-native-r11/donor-quarter.json'),
        'decision':'Native author guide/GN system looks more organically layered; independent target fit in progress, not yet character hairstyle.',
        'author':'Daniel Bystedt','license':'CC BY-SA; version not established by available official evidence'},
    'masterChanged':False,'currentMaster':entry('work/character.blend'),'formalExportsChanged':False,
    'limits':['Generated three-view is design direction, never actual Blender evidence or calibrated reconstruction photographs.',
        'Requested image2.5 could not be selected/verified by built-in image tool; actual model not exposed.',
        'Old tail/cloth caches do not prove new head/groom collision quality. New integrated motions/physics/VRM remain unfinished.',
        'No new off-machine backup or frozen full-project archive.']}
if record['currentMaster']['sha256']!='3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166':raise RuntimeError('Unexpected master change')
out=P/(prefix+'face-contour-progress-r11.json')
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
state_path=P/'project-state.json';old=state_path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=candidate;state['reconstructionR11']['contourProgress']=record;state['recordedAt']=now
if state_path.read_bytes()!=old:raise RuntimeError('Concurrent state write')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_CONTOUR_PROGRESS_RECORDED',candidate['sha256'])
