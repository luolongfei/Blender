"""Parameter-only spring test VRM. Not the final Blend-corresponding export."""
import copy,hashlib,json,struct,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
EXPECTED='a525dd967002848abd874db60297e115ef22a972bc502a01b2703592efa47bb6'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def run():
    p=project_path();source=p/'work/runtime-r06/preview-tail-colliders-r09/review-preview.vrm'
    original=source.read_bytes();source_sha=hashlib.sha256(original).hexdigest()
    if source_sha!=EXPECTED:raise RuntimeError('Unexpected source preview identity')
    magic,version,total,json_size,json_kind=struct.unpack('<IIIII',original[:20])
    if (magic,version,total,json_kind)!=(0x46546c67,2,len(original),0x4e4f534a):raise RuntimeError('Expected actual GLB2 container')
    before=json.loads(original[20:20+json_size]);doc=copy.deepcopy(before)
    binary_chunk=original[20+json_size:]
    config_path=p/'work/runtime-r06/r09-tail-fit-measured-candidate.json';config=json.loads(config_path.read_text())
    audit=json.loads((p/'work/runtime-r06/r09-tail-motion-fit-measured60.json').read_text())
    if audit['inputSha']!=source_sha or audit['overrides']!=config:raise RuntimeError('Unmeasured parameter config')
    sb=doc['extensions']['VRMC_springBone'];new_indices=[]
    for row in config['sleeveCapsules']:
        bone_node=doc['extensions']['VRMC_vrm']['humanoid']['humanBones'][row['bone']]['node']
        new_indices.append(len(sb['colliders']));sb['colliders'].append({'node':bone_node,'shape':{'capsule':{k:row[k] for k in ('offset','tail','radius')}}})
    group_index=len(sb['colliderGroups']);sb['colliderGroups'].append({'name':'R09 measured puff sleeves','colliders':new_indices})
    for side in ('L','R'):
        spring=next(s for s in sb['springs'] if s['name']=='RT06 tail '+side)
        if len(spring['joints'])!=7:raise RuntimeError('Six deform joints and terminal expected')
        for i,value in enumerate(config['hitRadiiBySide'][side]):spring['joints'][i]['hitRadius']=value
        spring['colliderGroups'].append(group_index)
    unmodified_before=copy.deepcopy(before);unmodified_after=copy.deepcopy(doc)
    del unmodified_before['extensions']['VRMC_springBone'];del unmodified_after['extensions']['VRMC_springBone']
    if unmodified_before!=unmodified_after:raise RuntimeError('Non-spring JSON content changed')
    raw=json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode();raw+=b' '*((-len(raw))%4)
    output=struct.pack('<IIIII',magic,2,20+len(raw)+len(binary_chunk),len(raw),json_kind)+raw+binary_chunk
    folder=p/'work/runtime-r06/preview-tail-envelope-test-r09';folder.mkdir(exist_ok=False)
    target=folder/'test-preview.vrm';target.write_bytes(output)
    saved=target.read_bytes();n=struct.unpack('<I',saved[12:16])[0]
    if saved[20+n:]!=binary_chunk:raise RuntimeError('BIN chunk changed')
    loaded=json.loads(saved[20:20+n]);check=copy.deepcopy(loaded);del check['extensions']['VRMC_springBone']
    if check!=unmodified_before:raise RuntimeError('Saved non-spring JSON content changed')
    report={'kind':'temporary spring-parameter test preview; NOT final Blend-corresponding export',
        'source':str(source.relative_to(p)),'sourceSha256':source_sha,'output':str(target.relative_to(p)),
        'outputSha256':hashlib.sha256(saved).hexdigest(),'config':config,'configSha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'onlyChangedExtension':'VRMC_springBone','binChunkByteIdentical':True,'binChunkSha256':hashlib.sha256(binary_chunk).hexdigest(),
        'allOtherJsonContentIdentical':True,'allOtherJsonSha256':digest(unmodified_before),
        'meshContentSha256':digest(before['meshes']),'materialContentSha256':digest(before['materials']),
        'expressionContentSha256':digest(before['extensions']['VRMC_vrm']['expressions']),
        'remainingIssue':'CPU sampling retained a few lower-tail/bodice crossings; visual acceptance pending.',
        'adoption':'Apply runtime_tail_envelope_r09.apply() to a new Blend copy and re-export with full contract checks before any final delivery.'}
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_sha:raise RuntimeError('Original VRM changed')
    (folder/'parameter-test-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));return report

if __name__=='__main__':run()
