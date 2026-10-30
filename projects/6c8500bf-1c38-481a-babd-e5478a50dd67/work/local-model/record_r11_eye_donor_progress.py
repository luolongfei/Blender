"""Record verified donor fit/current-size eye observation, without source promotion."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path(); stage='work/reconstruction-r02/'
def item(path):
    p=P/path
    return {'path':path,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
source=item(stage+'reference-adult-eye-proportion-r11.blend')
assert source['sha256']=='c2afb5d4499ff44475b57f09853c1d64d2d22b39bdab71b97f42e801ef605553'
donor=item(stage+'reference-adult-donor-groom-r11.blend')
assert donor['sha256']=='e10888af4c641d3c32338850dfd467b4e830199ac6ad23b6de5ba9874df98997'
views=[]
for case in ('groom-front','groom-quarter'):
    base='previews/reconstruction-r02/adult-donor-fit-r11/'+case
    report=json.loads((P/(base+'.json')).read_text())
    img=item(base+'.png')
    assert report['sourceSha256']==donor['sha256'] and report['imageSha256']==img['sha256']
    views.append({'image':img,'report':item(base+'.json'),'sourceSha256':donor['sha256']})
observations=[]
for name in ('adult-blink-current-size-r11','adult-neck-precision-r11'):
    doc=json.loads((P/(stage+name+'.json')).read_text())
    arr=item(stage+name+'.npz')
    assert doc['sourceSha256']==source['sha256'] and doc['arraysSha256']==arr['sha256']
    observations.append({'report':item(stage+name+'.json'),'arrays':arr,'sourceSha256':source['sha256']})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'latestIndependentCandidate':source,'donorFitSource':donor,'actualDonorViews':views,
    'currentGeometryObservations':observations,
    'stages':[item(stage+n) for n in ('reference-adult-donor-groom-r11.donor-fit.json','reference-adult-eye-proportion-r11.json')],
    'lashOpticsDiagnostic':item('previews/reconstruction-r02/lash-reflection-r11/report.json'),
    'visibleDecision':{'donor':'Retain genuine layered author groom as foundation; currently loose short hair, not reference bangs or twin-tail styling. Float ties and dark temple block remain.',
        'eyes':'24.6 mm eyes and coherent orbital deformation actually saved and observed. New proportion beauty images pending. Small outer-corner deep-socket diagnostic remains, not declared seamless.',
        'lowerFace':'Previous retry failed protected-surface displacement validation, no shortened-face candidate saved at time of this record. Diagnostic retry running.',
        'lashes':'128-sample single-eye diagnostics show black fibres at roughness .5; .85 produced no material visible benefit. Retain .5. Full-face readability and comb-like style still unfinished.',
        'neck':'Actual current/rest seam probe completed. Width, posterior seam and material continuity still under repair.',
        'appearanceAccepted':False,'photorealisticReady':False},
    'masterChanged':False,'formalExportsChanged':False,'master':item('work/character.blend'),
    'limitations':['No installed/verified new blink or gaze yet.','No new accepted continuous motion, physics, runtime output, or off-machine backup.','Generated design references are not Blender evidence; actual image model is not exposed.']}
out=P/(stage+'eye-donor-progress-r11.json')
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
path=P/'project-state.json';old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
state['reconstructionR11']['latestIndependentCandidate']=source
state['reconstructionR11']['eyeDonorProgress']=record
state['recordedAt']=record['recordedAt']
if path.read_bytes()!=old:raise RuntimeError('Concurrent project state write')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_EYE_DONOR_PROGRESS_RECORDED',source['sha256'])
