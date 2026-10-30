"""Continuous narrow phase on the frozen actual native 225 mm shoe probe."""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=project_path();s=p/'work/reconstruction-r02';src=s/'footwear-225-native-probe-r11.json'
    if sha(src)!='0afca9ada441483c1f946fe8e0d5b0a9a3c61f67bfdca0a410984a25581357c0':raise RuntimeError('Actual native probe changed')
    j=json.loads(src.read_text());npz=p/j['arrays']
    if sha(npz)!='f6648bcff4ac348edd33b6f47728b9254f023a5708aa94ac643b3114f2e85597':raise RuntimeError('Native arrays changed')
    helper=Path(__file__).with_name('hand_wrist_surface_contact_r11.py')
    if sha(helper)!='c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e':raise RuntimeError('Frozen triangle helper changed')
    api=runpy.run_path(str(helper));a=np.load(npz);rows=[]
    for row in j['shoes']:
        side=row['side'];pre=row['arrayPrefix'];fv=a['foot_'+side+'_world'];ft=a['foot_'+side+'_triangles'];v=a[pre+'_world'];tri=a[pre+'_triangles'];pairs=a[pre+'_footPairs'].reshape(-1,2)
        result={'name':row['name'],'arrayPrefix':pre,'side':side,'broadPhasePairs':len(pairs),'confirmedIntersections':0,'contactWithin1Micrometre':0,'intersections':[]}
        if len(pairs):
            fp=fv[ft[pairs[:,0]]];sp=v[tri[pairs[:,1]]]
            metrics=api['triangle_pair_metrics'](fp,sp,tolerance=1e-6)
            result['confirmedIntersections']=int(metrics['intersects'].sum());result['contactWithin1Micrometre']=int(metrics['contactWithinTolerance'].sum())
            result['minimumCandidateDistanceMeters']=float(metrics['distance'].min())
            for i in np.flatnonzero(metrics['intersects']):
                low=np.maximum(fp[i].min(0),sp[i].min(0));high=np.minimum(fp[i].max(0),sp[i].max(0));n=np.cross(sp[i,1]-sp[i,0],sp[i,2]-sp[i,0]);n=n/np.linalg.norm(n)
                result['intersections'].append({'footTriangleId':int(pairs[i,0]),'shoeTriangleId':int(pairs[i,1]),'overlapAabbMeters':[low.tolist(),high.tolist()],'shoeFaceUnitNormal':n.tolist()})
        rows.append(result)
    out=s/'footwear-225-native-contact-analysis-r11.json'
    if out.exists():raise FileExistsError('Preserve actual evidence')
    report={'schemaVersion':1,'probeSha256':sha(src),'probeArraysSha256':sha(npz),'source':j['source'],'sourceSha256':j['sourceSha256'],'helperSha256':sha(__file__),'continuousHelperSha256':sha(helper),'objects':rows,'intersectionPairs':sum(r['confirmedIntersections'] for r in rows),'contactPairsWithin1Micrometre':sum(r['contactWithin1Micrometre'] for r in rows),'technicalAcceptance':False,'appearanceAccepted':False,'executedBlender':False,'scope':'Exact triangle narrow phase for actual native shoe versus actual native donor foot BVH overlap candidates. Does not measure all positive shell clearance, sock thickness, poses or image appearance.'}
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out),'sha256':sha(out),'intersectionPairs':report['intersectionPairs'],'contactPairs':report['contactPairsWithin1Micrometre'],'components':[{k:r[k] for k in ['name','broadPhasePairs','confirmedIntersections','contactWithin1Micrometre']} for r in rows if r['broadPhasePairs']]}))

if __name__=='__main__':main()
