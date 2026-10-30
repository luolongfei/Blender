"""Bounded, offline search starting from the existing outward v4 candidate."""
import json,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
m=runpy.run_path(str(HERE/'repair_reference_gather_native_v4_r11.py'))
P=m['P']; W=P/'work/reconstruction-r02'
b=np.load(W/'reference-gather-rebuild-v3-r11.npz'); f=np.load(W/'reference-adult-reference-ties-v2-r11.tie-transport.npz'); v=np.load(W/'reference-gather-repair-v4-r11.npz'); h=np.load(W/'reference-tie-alignment-r11-v2.npz')
head=m['HeadPredictor'](h['head_world'],h['head_triangles']); curves=m['CURVES']; sizes=b['source_lengths'][curves]; starts=np.r_[0,np.cumsum(sizes)]; basis=m['native_basis'](20)

def evaluate(curve,q,current,old,radius):
    native=m['resampled_basis'](q,basis)@q; mid=(q[:-1]+q[1:])*.5; nm=(native[:-1]+native[1:])*.5
    points=np.vstack([native,nm,mid,q]); arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(native,axis=0),axis=1))]; keep=np.r_[arc,(arc[:-1]+arc[1:])*.5,np.ones(len(mid)),np.ones(len(q))]>.001
    gaps=head.signed(points)-radius; gaps=gaps[keep]; turn=m['turns'](q); before=m['turns'](current); fold=int(np.sum((turn>95)&(turn>before+20)))
    first=1 if curve in m['TAKEOFF'] else 3; target=old.copy(); target[first:-3]=q[first:-3]; movement=float(np.linalg.norm(target-old,axis=1).max())
    return float(gaps.min()), movement,fold

rows=[]; result=[]
for ci,curve in enumerate(curves):
    n=int(sizes[ci]); fs=int(f['after-final_starts'][curve]); st=int(b['source_starts'][curve]); current=f['after-final_world'][fs:fs+n].astype(float); old=b['target_world'][st:st+n].astype(float); radius=float(f['after-final_radius'][fs:fs+n].max()); ref=v['predicted_after_contact_corrected_world'][starts[ci]:starts[ci+1]].astype(float); delta=ref-current
    candidates=[]
    # Cap each control's true displacement to the base proposal, rather than
    # shrinking all control offsets just because one point was overshot.
    for factor in [1.,.975,.95,.925,.9,.85,.8,.7,.6,.5,.4,.3]:
      for cap in [.0118,.0113,.0108]:
        q=current+factor*delta; first=1 if curve in m['TAKEOFF'] else 3; change=q[first:-3]-old[first:-3]; length=np.linalg.norm(change,axis=1); q[first:-3]=old[first:-3]+change*np.minimum(1,cap/np.maximum(length,1e-15))[:,None]
        gap,move,fold=evaluate(curve,q,current,old,radius); candidates.append((gap,move,fold,factor,cap,q))
    acceptable=[x for x in candidates if x[0]>=-.00035 and x[1]<=.012 and x[2]==0]
    if acceptable: best=min(acceptable,key=lambda x:x[1])
    else: best=max(candidates,key=lambda x:x[0]-.01*x[2]-max(0,x[1]-.012))
    row={'curve':int(curve),'gap':best[0],'movement':best[1],'fold':best[2],'factor':best[3],'cap':best[4],'acceptable':bool(acceptable)}; rows.append(row);result.append(best[5]);print(json.dumps(row),flush=True)
np.savez_compressed(W/'reference-gather-repair-v6-scan-r11.npz',curves=curves,predicted_controls=np.concatenate(result),starts=starts)
(W/'reference-gather-repair-v6-scan-r11.json').write_text(json.dumps(rows,indent=2))
