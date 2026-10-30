"""Record actual assembled geometry and the explicitly failed groom appearance."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();stage='work/reconstruction-r02/'
def item(rel):
    p=P/rel
    return {'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
source=item(stage+'reference-adult-upper-neck-seam-r11.blend')
assert source['sha256']=='491b6c85c9c5261e619bba14a5b622bd09dc237d52eb2474d26509f0eb5ba707'
face=item(stage+'reference-adult-brow-finish-r11.blend')
assert face['sha256']=='d4e5facee0721371ddfdf8762489d07ffd6b94120fca6058e753e4ea44d8f5c0'
groom=item(stage+'reference-adult-c-tails-r11.blend')
assert groom['sha256']=='c55bcc610afaf65b8068b5ccadb7b7351b50fe79d89c7a73e4b87fdce48bd25e'
views=[]
for case in ('groom-front','groom-quarter','groom-side'):
    base='previews/reconstruction-r02/adult-style-combined-r11/'+case
    report=json.loads((P/(base+'.json')).read_text());img=item(base+'.png')
    assert report['sourceSha256']==groom['sha256'] and report['imageSha256']==img['sha256']
    views.append({'image':img,'report':item(base+'.json'),'sourceSha256':groom['sha256']})
record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'latestIndependentCandidate':source,'lastFaceCandidateBeforeRejectedGroom':face,
    'groomAppearanceRejectedCandidate':groom,'actualGroomViews':views,
    'actualStages':[item(stage+x) for x in (
        'reference-adult-neck-proportion-r11.neck-continuity.json',
        'reference-adult-upper-lip-r11.upper-lip.json',
        'reference-adult-soft-nose-r11.nose-contour.json',
        'reference-adult-brow-finish-r11.brow-finish.json',
        'reference-adult-donor-restyle-v2-r11.donor-restyle.json',
        'reference-adult-c-tails-r11.tail-shape.json',
        'reference-adult-upper-neck-seam-r11.upper-neck-seam.json')],
    'actualGroomDiagnostic':[item(stage+'donor-gn-stages-r11.'+ext) for ext in ('json','npz')],
    'visibleDecision':{'groom':'REJECTED: forehead scalp exposure, tangled/upright fringe, radiating needles near ties, oversized posterior volume and dense temple blocks.',
        'face':'Shorter lower face retained for continued work; upper lip/nose/brow changes actually saved and seen in combined views, but natural expression and photoreal appearance remain unfinished.',
        'neck':'Narrowed geometry saved. New 2592-vertex upper seam bridge saved after explicit sharp/custom-normal attribute handling; no actual image of this latest bridge yet. Lower seam and neck skinning remain pending.',
        'tails':'Shorter curved tails saved; entire tips still need full-frame review. Old native motion deliberately retired; new dynamics unbaked.',
        'appearanceAccepted':False,'photorealisticReady':False},
    'blink':'First install refused outdated c2 eye-region observation after nasal sculpting; no blink candidate saved. New current-shape observation/installation is running separately.',
    'master':item('work/character.blend'),'masterChanged':False,'formalExportsChanged':False,
    'limits':['Latest source contains rejected groom for active repair, not an adopted delivery.',
        'Old motion/runtime reports do not validate the new head, groom, bridge or tails.',
        'Interactive Blender is currently running a separate 9fc lower-face preview; root has not overwritten its session.',
        'No new off-machine backup or full-project freeze.']}
assert record['master']['sha256']=='3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166'
out=P/(stage+'static-assembly-progress-r11.json')
with out.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
path=P/'project-state.json';old=path.read_bytes();state=json.loads(old.decode('utf-8-sig'))
r=state['reconstructionR11'];r['latestIndependentCandidate']=source;r['staticAssemblyProgress']=record
r['status']='static-reconstruction-continued; groom-v2-appearance-rejected; new-performance-in-progress'
r['next']=['Repair actual donor GN curl/clump/profile behavior and posterior volume, then render full hairstyle.',
    'Recompute and actually review blink/smile from latest static face, then bind new head/neck and attachments.',
    'Finish hands/body/garment refinements and rebake matching motion, physics and runtime delivery.']
state['recordedAt']=record['recordedAt']
if path.read_bytes()!=old:raise RuntimeError('Concurrent state update')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=P/'.asset-code-paths.json';data=json.loads(registry.read_text(encoding='utf-8-sig'))
data['paths']=sorted(set(data['paths'])|{f.relative_to(P).as_posix() for f in (P/'work/local-model').iterdir() if f.suffix in {'.py','.js','.mjs','.html'}})
registry.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('R11_STATIC_ASSEMBLY_RECORDED',source['sha256'])
