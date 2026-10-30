"""Only two bounded ear-side connector paths, old observed triangles offline."""
import json,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;m=runpy.run_path(str(HERE/'repair_reference_gather_native_v4_r11.py'));P=m['P'];W=P/'work/reconstruction-r02'
b=np.load(W/'reference-gather-rebuild-v3-r11.npz');f=np.load(W/'reference-adult-reference-ties-v2-r11.tie-transport.npz');v=np.load(W/'reference-gather-repair-v4-r11.npz');h=np.load(W/'reference-tie-alignment-r11-v2.npz');head=m['HeadPredictor'](h['head_world'],h['head_triangles']);basis=m['native_basis'](20)
out=[];rows=[]
for curve in [10296,10301]:
    ci=list(m['CURVES']).index(curve);fs=int(f['after-final_starts'][curve]);st=int(b['source_starts'][curve]);current=f['after-final_world'][fs:fs+20].astype(float);old=b['target_world'][st:st+20].astype(float);ref=v['predicted_after_contact_corrected_world'][ci*20:(ci+1)*20].astype(float);radius=float(f['after-final_radius'][fs:fs+20].max());best=None;valid=[]
    shape=np.exp(-.5*((np.arange(20)-9)/1.7)**2);shape[:3]=0;shape[-3:]=0
    for ay in [-.005,-.003,-.001,0.,.001,.003,.005]:
      for az in [-.005,-.003,-.001,0.,.001,.003,.005]:
        q=ref.copy();q[:,1]+=shape*ay;q[:,2]+=shape*az;d=q[3:-3]-old[3:-3];q[3:-3]=old[3:-3]+d*np.minimum(1,.01185/np.maximum(np.linalg.norm(d,axis=1),1e-15))[:,None]
        native=m['resampled_basis'](q,basis)@q;mid=(q[:-1]+q[1:])*.5;nm=(native[:-1]+native[1:])*.5;points=np.vstack([native,nm,mid,q]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(native,axis=0),axis=1))];keep=np.r_[arc,(arc[:-1]+arc[1:])*.5,np.ones(len(mid)),np.ones(len(q))]>.001;gaps=head.signed(points)-radius;gap=float(gaps[keep].min());turn=m['turns'](q);fold=int(np.sum((turn>95)&(turn>m['turns'](current)+20)));move=float(np.linalg.norm(q[3:-3]-old[3:-3],axis=1).max());row={'curve':curve,'yAmplitude':ay,'zAmplitude':az,'minimumGapMeters':gap,'maxMovementMeters':move,'folds':fold}
        score=gap-.01*fold
        if best is None or score>best[0]:best=(score,q,row)
        if gap>-.00025 and fold==0:valid.append((abs(ay)+abs(az),-gap,q,row))
    if valid:
        _,_,q,row=min(valid,key=lambda r:(r[0],r[1]))
    else:_,q,row=best
    row['acceptable']=bool(valid);rows.append(row);out.append(q);print(json.dumps(row),flush=True)
np.savez_compressed(W/'reference-gather-repair-v6-two-r11.npz',curves=np.array([10296,10301],np.int32),predicted_controls=np.concatenate(out))
(W/'reference-gather-repair-v6-two-r11.json').write_text(json.dumps(rows,indent=2))
