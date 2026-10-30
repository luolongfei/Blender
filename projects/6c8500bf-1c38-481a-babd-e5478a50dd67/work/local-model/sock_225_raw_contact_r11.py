"""Raw middle-surface versus candidate skin exact contact triage, not thickness acceptance."""
import argparse,hashlib,json,runpy,sys
from itertools import product
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def triangles(l,s,n):
    return np.asarray([(l[a],l[a+k],l[a+k+1]) for a,z in zip(s,n) for k in range(1,z-1)],np.int32)

def main(relative):
    p=project_path();path=p/relative;j=json.loads(path.read_text());a=np.load(p/j['arrays']);s=p/'work/reconstruction-r02';c=json.loads((s/'foot-proportion-capture-r11.json').read_text());b=np.load(p/c['arrays']);l=json.loads((s/'foot-leg-unified-candidate-r11-v2.json').read_text());d=np.load(p/l['arrays'])
    for r in (j,c,l):
        if sha(p/r['arrays'])!=r['arraysSha256']:raise RuntimeError('Frozen input changed')
    helper=Path(__file__).with_name('hand_wrist_surface_contact_r11.py')
    if sha(helper)!='c29c10380e05d70e9d26969f4e0b3b44603c0a378c35fc2c461b3f9a8d553a6e':raise RuntimeError('Continuous helper changed')
    api=runpy.run_path(str(helper));reports=[];cell=.004
    for side in ('L','R'):
        pre='legFoot_'+side;v=d[pre+'_positions'];t=triangles(d[pre+'_loops'],d[pre+'_polygonStarts'],d[pre+'_polygonSizes']);pts=v[t];tl=pts.min(1);th=pts.max(1);cells={}
        for i in np.flatnonzero(tl[:,2]<.30):
            aa=np.floor((tl[i]-1e-6)/cell).astype(int);bb=np.floor((th[i]+1e-6)/cell).astype(int)
            for key in product(*(range(x,y+1) for x,y in zip(aa,bb))):cells.setdefault(key,[]).append(int(i))
        for row in j['objects']:
            if row['side']!=side:continue
            pre=row['arrayPrefix'];v=a[pre+'_targetWorld'];st=triangles(b[pre+'_raw_loops'],b[pre+'_raw_polygonStarts'],b[pre+'_raw_polygonSizes']);sp=v[st];sl=sp.min(1);sh=sp.max(1);pairs=[]
            for i in range(len(st)):
                aa=np.floor((sl[i]-1e-6)/cell).astype(int);bb=np.floor((sh[i]+1e-6)/cell).astype(int);ids=set()
                for key in product(*(range(x,y+1) for x,y in zip(aa,bb))):ids.update(cells.get(key,()))
                if ids:
                    ids=np.asarray(sorted(ids));ids=ids[((th[ids]>=sl[i]-1e-6)&(tl[ids]<=sh[i]+1e-6)).all(1)];pairs.extend((i,int(k)) for k in ids)
            pairs=np.asarray(pairs,np.int32).reshape(-1,2);hits=[];contacts=[]
            for k in range(0,len(pairs),4096):
                pair=pairs[k:k+4096];r=api['triangle_pair_metrics'](sp[pair[:,0]],pts[pair[:,1]])
                hits.extend(pair[r['intersects']].tolist());contacts.extend(pair[r['contactWithinTolerance']].tolist())
            reports.append({'name':row['name'],'broadPhasePairs':len(pairs),'continuousIntersections':len(hits),'contactWithin1Micrometre':len(contacts),'intersectionPairs':hits[:256],'intersectionAabbMeters':None if not hits else [np.minimum(sp[np.asarray(hits)[:,0]].min((0,1)),pts[np.asarray(hits)[:,1]].min((0,1))).tolist(),np.maximum(sp[np.asarray(hits)[:,0]].max((0,1)),pts[np.asarray(hits)[:,1]].max((0,1))).tolist()]})
            print(row['name'],len(pairs),len(hits),flush=True)
    out=path.with_name(path.stem+'-raw-contact.json')
    if out.exists():raise FileExistsError('Preserve prior raw check')
    out.write_text(json.dumps({'candidateSha256':sha(path),'helperSha256':sha(__file__),'objects':reports,'scope':'Continuous raw sock middle-surface versus unified candidate raw skin only; actual thickness and native subdivision are excluded and remain required.','technicalAcceptance':False,'appearanceAccepted':False},indent=2)+'\n')
    print(json.dumps({'report':str(out),'sha256':sha(out),'intersections':sum(r['continuousIntersections'] for r in reports)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True);a=parser.parse_args();main(a.candidate)
