"""Record the actual rejected R10 crown and the user's current art direction."""
import datetime, hashlib, json, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
def identity(relative):
    p=P/relative
    return {'path':relative,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    state_path=P/'project-state.json';old=state_path.read_bytes()
    state=json.loads(old.decode('utf-8-sig'))
    master=identity('work/character.blend')
    if master['sha256']!='3d0e397a6ba0b8742054718d01c4eed5b4ff0daa4a603fac43aa020b87707166':
        raise RuntimeError('This decision concerns the retained R09 master only')
    candidate=identity('work/reconstruction-r02/reference-crown-volume-r10.blend')
    if candidate['sha256']!='3fe0565d295e74593f61f8d9eb727db2d4664dd4369652690ab68037a3491cce':
        raise RuntimeError('Crown candidate changed after visual inspection')
    record={'recordedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'baselineMaster':master,'rejectedCandidate':candidate,'decision':'reject-entire-crown-volume-candidate',
        'images':[identity('previews/reconstruction-r02/crown-volume-r10/'+case+'.png')
                  for case in ('face-front','face-quarter','face-side')],
        'reason':['Front and quarter show a hat-like side ledge and bent fringe ribbons.',
                  'The attached opaque root substrate was expanded into the outer hair volume.',
                  'Surface clearance and continuity checks do not establish convincing hair appearance.'],
        'next':'Retain the attached R09 PN root substrate; form outer volume through layered, free-arching fibre paths. Inspect a mature official Curves asset before adopting a new groom.',
        'retainedCandidate':identity('work/reconstruction-r02/reference-production-refined-r09.blend'),
        'clientAppearanceAccepted':False,'photorealisticDeliveryReady':False,
        'userDirection':{
            'physicalBehaviour':'Actions and details must obey physical behaviour; inspect gravity, inertia, contact, joint volume and settling with actual motion.',
            'skinAppearance':'Natural Asian/Japanese adult woman complexion with light makeup, optionally slightly fairer; preserve warm undertone, subtle blood colour and skin detail. Do not infer a fixed ethnic colour swatch.',
            'approval':'User delegates routine and artistic decisions; do not wait for confirmations.',
            'parallelWork':'Use all available subagent slots for independent edits; sole integrator controls master and serial heavy Blender jobs on current hardware.'}}
    out=P/'work/reconstruction-r02/crown-visible-decision-r10.json'
    with out.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    state.setdefault('reconstructionR10',{})['crownDecision']=record
    state['reconstructionR10']['currentNext']=record['next']
    state['currentUserDirection']=record['userDirection'];state['recordedAt']=record['recordedAt']
    if state_path.read_bytes()!=old:raise RuntimeError('Concurrent state modification')
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RECORDED_R10_CROWN_REJECTION',candidate['sha256'])
if __name__=='__main__':main()
