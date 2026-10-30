"""Bound the v3 plantar ray fit to the actual foot surface, never upper-leg hits."""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=project_path();s=p/'work/reconstruction-r02';src=s/'sock_225_section_fit_r11_v3.json'
    if sha(src)!='e0edc343e0c3ae839862f0cf503ce21d5186ff4996d9168ec6e01e9fb23bbedd':raise RuntimeError('Frozen v3 changed')
    j=json.loads(src.read_text());a=np.load(p/j['arrays']);base=np.load(s/'sock_225_section_fit_r11_v2.npz');cj=json.loads((s/'foot-proportion-capture-r11.json').read_text());c=np.load(p/cj['arrays']);lj=json.loads((s/'foot-leg-unified-candidate-r11-v2.json').read_text());d=np.load(p/lj['arrays'])
    for r in (j,cj,lj):
        if sha(p/r['arrays'])!=r['arraysSha256']:raise RuntimeError('Frozen arrays changed')
    api=runpy.run_path(str(Path(__file__).with_name('sock_225_section_fit_r11_v3.py')));packed={k:a[k] for k in a.files};by={r['name']:r for r in cj['objects']}
    for row in j['objects']:
        pre=row['arrayPrefix'];m=np.asarray(by[row['name']]['matrixWorld']);v=a[pre+'_sourceRaw']@m[:3,:3].T+m[:3,3];q=a[pre+'_targetWorld'].copy()
        if row['name'].startswith('Sock_Ribbed_'):
            q[:,2]=base[pre+'_targetWorld'][:,2];tri=api['triangles'](c[pre+'_raw_loops'],c[pre+'_raw_polygonStarts'],c[pre+'_raw_polygonSizes']);n=api['normals'](v,tri);ids=np.flatnonzero((v[:,2]<.070)&(n[:,2]<-.08));pl='legFoot_'+row['side'];tv=d[pl+'_positions'];tt=api['triangles'](d[pl+'_loops'],d[pl+'_polygonStarts'],d[pl+'_polygonSizes']);tt=tt[tv[tt,2].max(1)<.105];bottom=api['lower_raycast'](q[ids],tv,tt);valid=np.isfinite(bottom)&(bottom<.090);ids=ids[valid];bottom=bottom[valid];w=api['smooth']((-n[ids,2]-.08)/.42);q[ids,2]=(1-w)*q[ids,2]+w*(bottom-.0009)
            if q[:,2].max()>.289 or q[:,2].min()<.02:raise RuntimeError('Bounded foot fit invalid')
            row['soleRayFitVertices']=len(ids);row['soleRayGeometryScope']='Only raw anatomical foot triangles entirely below Z .105 m; no leg or cap hit can relocate plantar cloth upward.'
        inv=np.linalg.inv(m);packed[pre+'_targetWorld']=q;packed[pre+'_targetRaw']=q@inv[:3,:3].T+inv[:3,3];row['candidateBoundsMeters']=[q.min(0).tolist(),q.max(0).tolist()];row['maximumDisplacementMeters']=float(np.linalg.norm(q-v,axis=1).max());row['jacobianScope']='Common XY field only; explicit normal-weighted plantar surface correction requires separate exact mesh checks.'
    out=s/'sock_225_section_fit_r11_v4.json';outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve prior evidence')
    np.savez_compressed(outa,**packed);j.update({'helperSha256':sha(__file__),'parentCandidateSha256':sha(src),'arrays':str(outa.relative_to(p)),'arraysSha256':sha(outa),'rejectedPriorCandidate':'v1 cap jump, v2 visible skin penetration, v3 unrestricted upward ray catches: all rejected, preserved, never installed.'});out.write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(outa)}))
if __name__=='__main__':main()
