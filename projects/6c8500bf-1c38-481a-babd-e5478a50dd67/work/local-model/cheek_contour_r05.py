"""Reduce the accumulated cheek bulge without moving eyelid or mouth rims."""
import bpy,numpy as np,json,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run():
    h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
    if ks.get('R05_cheek_contour') or Path(bpy.data.filepath).name=='character.blend':raise RuntimeError('Use new work copy')
    arrays=[]
    for key in ks:
        a=np.empty(len(key.data)*3,np.float32);key.data.foreach_get('co',a);arrays.append(a.reshape(-1,3))
    co=arrays[0].copy()
    for key,a in zip(list(ks)[1:],arrays[1:]):co+=(a-arrays[0])*key.value
    x,y,z=co.T
    front=np.clip((-y-.025)/.035,0,1)
    weight=np.exp(-((np.abs(x)-.047)/.022)**4-((z-1.372)/.024)**4)*front
    protect=np.clip((1.397-z)/.011,0,1);weight*=protect*protect*(3-2*protect)
    mouth=1-np.exp(-(x/.031)**6-((z-1.354)/.012)**4);weight*=mouth
    edges=np.array([tuple(e.vertices) for e in h.data.edges]);a,b=edges.T
    counts=np.bincount(np.r_[a,b],minlength=len(co)).astype(float);work=co.astype(float).copy()
    for _ in range(10):
        totals=np.zeros_like(work);np.add.at(totals,a,work[b]);np.add.at(totals,b,work[a])
        average=totals/np.maximum(counts[:,None],1)
        work+=(average-work)*(.23*weight[:,None])
    delta=work-co
    # Prior soft-tissue passes added overlapping anterior fullness. Remove part
    # of this local projection so the cheek remains a continuous broad plane.
    delta[:,1]+=.0024*weight
    distance=np.linalg.norm(delta,axis=1);delta*=np.minimum(1,.003/np.maximum(distance,1e-10))[:,None]
    key=h.shape_key_add(name='R05_cheek_contour',from_mix=False)
    key.data.foreach_set('co',np.ascontiguousarray(arrays[0]+delta,dtype=np.float32).ravel());key.value=1
    report={'affectedOver01mm':int((np.linalg.norm(delta,axis=1)>.0001).sum()),'maxDisplacementMeters':float(np.linalg.norm(delta,axis=1).max()),'eyelidsProtectedAboveZ':1.397,'newKey':key.name,'visualCheckPending':True}
    (project_path()/'work/reconstruction-r02/cheek-contour-r05.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    bpy.context.view_layer.update();print(json.dumps(report));return report
