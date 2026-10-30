"""Prepare shared shoe toe-roof lift from measured native inner-shell evidence."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def smoother(t):
    t=np.clip(t,0,1);return t*t*t*(t*(t*6-15)+10)

def lift_weight(points):
    q=np.asarray(points)
    return (1-smoother((q[:,1]+.130)/.040))*smoother((q[:,2]-.028)/.018)

def main():
    p=project_path();s=p/'work/reconstruction-r02';v4=s/'footwear-225-shared-map-r11-v4.json';measure=s/'footwear-225-toe-roof-measure-r11.json';capture=s/'foot-proportion-capture-r11.json'
    expected=[(v4,'217b9b4b4689b7521482779d382f35d205a69a9170a7f1aeeae2375364fa9fb5'),(measure,'2fd3e7ca6d5d5b2cea872184d26e2a236d9e4e4911d4d763b73e4febcf1fe22d'),(capture,'c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f')]
    for path,digest in expected:
        if sha(path)!=digest:raise RuntimeError('Frozen actual evidence changed')
    old=json.loads(v4.read_text());m=json.loads(measure.read_text());c=json.loads(capture.read_text())
    for row in (old,m):
        if sha(p/row['arrays'])!=row['arraysSha256']:raise RuntimeError('Frozen evidence array changed')
    a=np.load(p/old['arrays']);ma=np.load(p/m['arrays']);packed={k:a[k] for k in a.files};src={r['name']:r for r in c['objects']}
    needed=[];target_room=.0018;native_reassessment_allowance=.00025
    for side in ('L','R'):
        samples=ma[side+'_sampleWorld'];roof=ma[side+'_innerRoofZ'];q=samples.copy();q[:,2]=roof;weight=lift_weight(q);gap=roof-samples[:,2]
        select=np.isfinite(gap)&(gap<target_room+native_reassessment_allowance)
        if np.any(weight[select]<1e-5):raise RuntimeError('Measured deficiency lies outside proposed smooth lift support')
        needed.extend(((target_room+native_reassessment_allowance-gap[select])/weight[select]).tolist())
    amplitude=float(np.ceil(max(needed)/.0001)*.0001)
    if amplitude>.004:raise RuntimeError('Shoe roof needs a new fitting decision rather than this bounded local lift')
    rows=[];sides={}
    for row in old['objects']:
        pre=row['arrayPrefix'];matrix=np.asarray(src[row['name']]['matrixWorld']);v=a[pre+'_targetRaw'];world=v@matrix[:3,:3].T+matrix[:3,3];weight=lift_weight(world);target=world.copy();target[:,2]+=amplitude*weight
        inv=np.linalg.inv(matrix);packed[pre+'_targetRaw']=target@inv[:3,:3].T+inv[:3,3]
        floor=world[:,2]<=.028
        if not np.array_equal(target[floor],world[floor]):raise RuntimeError('Sole/heel lower construction changed')
        info=dict(row);info.update({'maximumAdditionalToeRoofLiftMeters':float((target[:,2]-world[:,2]).max()),'additionalLiftVertices':int((weight>0).sum()),'verticesAtOrBelow28mmExact':int(floor.sum())});rows.append(info)
    for side in ('L','R'):
        samples=ma[side+'_sampleWorld'];roof=ma[side+'_innerRoofZ'];q=samples.copy();q[:,2]=roof;predicted=roof+amplitude*lift_weight(q)-samples[:,2]
        sides[side]={'measuredV4MinimumVerticalGapMeters':m['sides'][side]['minimumVerticalGapMeters'],'transportedNativeRoofPredictedMinimumGapMeters':float(np.nanmin(predicted)),'predictionIsNotNativeV5Evaluation':True}
    out=s/'footwear-225-shared-map-r11-v5.json';outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve candidate')
    packed['toeRoof_liftAmplitudeMeters']=np.array(amplitude);np.savez_compressed(outa,**packed)
    report=dict(old);report.update({'mode':'offline-unapplied-shared-225mm-shoe-map-v5-native-contact-correction','helperSha256':sha(__file__),'priorPlanSha256':sha(v4),'actualInnerRoofMeasureSha256':sha(measure),'objects':rows,'sides':sides,'arrays':str(outa.relative_to(p)),'arraysSha256':sha(outa),
        'toeRoofLift':{'amplitudeMeters':amplitude,'plateauYAtOrBeforeMeters':-.130,'smoothFadeToZeroYMeters':-.090,'soleFixedAtOrBelowZMeters':.028,'smoothFullLiftAtZMeters':.046,'measuredTargetDirectionalRoomMeters':target_room,'nativeReassessmentAllowanceMeters':native_reassessment_allowance,'meaning':'Measured actual native inner shell penetration plus provisional sock-layer room. These fitting allowances are not body or footwear standards.'},
        'mapping':'V4 common XY last map plus one C2 quintic forefoot Z lift applied identically to all 74 components. Sole and heel at/below 28 mm remain exact, forefoot upper and decorations move together. Foot skin is unchanged. Z derivative is nonnegative everywhere, so this added map cannot reverse local volume.',
        'unverified':['Actual Blender V5 subdivision and solidify response, continuous skin/sock/inner-shell intersections and positive room.','Sock candidate and actual new unified leg-foot native surface.','Actual shoe style under fixed-camera review, rig and motion.']})
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(outa),'amplitudeMeters':amplitude,'sides':sides}))

if __name__=='__main__':main()
