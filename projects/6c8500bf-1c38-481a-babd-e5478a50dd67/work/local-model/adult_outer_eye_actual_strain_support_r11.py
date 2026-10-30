"""Bounded transition-field explanation and actual-section figure; no Blender."""
import json,hashlib,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parent.parent;W=P/'work/reconstruction-r02'
OUT=W/'adult-outer-eye-actual-strain-support-r11'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def run():
    if OUT.with_suffix('.json').exists():raise RuntimeError('Preserve previous diagnostic')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection,PolyCollection
    j=json.loads((W/'adult-outer-eye-actual-strain-r11.json').read_text())
    a=np.load(W/'adult-outer-eye-actual-strain-r11.npz');raw=np.load(W/'reference-adult-outer-eye-radial-r11-v2.outer-eye.npz')
    meta=json.loads((W/'reference-adult-outer-eye-radial-r11-v2.outer-eye.json').read_text())['plan']
    ids=a['originalVertexIds'];old=a['beforeWorld'];new=a['afterWorld'];tri=a['actualTriangles'];delta=new-old
    sections=runpy.run_path(str(HERE/'upper_lid_surface_study_r11.py'))['sections']
    fig,axes=plt.subplots(2,2,figsize=(12,12),constrained_layout=True)
    report={'measurementReportSha256':sha(W/'adult-outer-eye-actual-strain-r11.json'),
            'actualCandidateSha256':j['actualSavedCandidateSha256'],'helperSha256':sha(__file__),
            'fieldValuesAreActualStoredPlan':True,'gapWeightMethod':'Effective radial weight inferred from actual theta change divided by compression, xEnvelope and angularWeight. Used only for stable nonzero denominators; no ideal eye or old eye mesh substitution.',
            'findings':[
                'The upper fold is in the near-full angular and radial core, not only its angular fade boundary. Full-strength rotation moves the outer crown down approximately 3.4 mm.',
                'The maximum upper fold neighbours a rapid theta-mid/core variation and the endpoint plateau. These parameters affect all supported depths, not only the visible aperture edge.',
                'The lower crease occurs roughly 12 mm below the eye centre, where angular and positive-gap fades still carry substantial displacement. The intended aperture refinement therefore transports visible cheek skin.',
                'Broad vertical/depth support combined with narrow X envelopes produces severe surface shear; widening X alone would still move non-target cheek tissue.',
                'These are measured shape defects, despite zero recorded intersections. Lower strength is an appearance experiment, not proof that the support is appropriate.'],
            'edgeNeighbourhoods':{},'sectionPlanes':{},'sourceModified':False}
    save={}
    for col,s in enumerate('LR'):
        md=meta['sides'][s];center=np.array(md['centerWorld']);sign=md['outwardSign'];f=raw['field_'+s]
        theta=np.arctan2(old[:,2]-center[2],center[1]-old[:,1]);angle=theta-f[:,1]
        q=np.clip((np.abs(angle)-f[:,2])/meta['angularFadeRadians'],0,1);wa=1-q*q*(3-2*q)
        aftertheta=np.arctan2(new[:,2]-center[2],center[1]-new[:,1])
        amount=np.divide(theta-aftertheta,angle,out=np.zeros(len(ids)),where=np.abs(angle)>1e-8)
        wg=np.divide(amount,meta['compression']*f[:,0]*wa,out=np.full(len(ids),np.nan),where=f[:,0]*wa>1e-6)
        for label in ('upperCrown','lowerCheek'):
            name=s+'_'+label;edge=j['regions'][name]['largestNewDihedrals'][0]
            rows=edge['actualRoiTriangleRows'];vv=np.unique(tri[rows])
            records=[]
            for v in vv:
                records.append(dict(globalVertexId=int(ids[v]),positionBeforeRelativeEyeMm=(old[v]-center)*1000,
                    actualDeltaMm=delta[v]*1000,xEnvelope=float(f[v,0]),theta=float(theta[v]),thetaMid=float(f[v,1]),
                    angularCoreHalf=float(f[v,2]),angularWeight=float(wa[v]),effectiveGapWeight=float(wg[v])))
            report['edgeNeighbourhoods'][name]={'sharedEdge':edge,'vertices':records}
        mask=a[s+'_upperCrown_mask']|a[s+'_lowerCheek_mask']|a[s+'_cornerBand_mask']
        t=tri[mask];q0=old[t];q1=new[t]
        def front(p):return np.stack([sign*(p[...,0]-md['endpointX'])*1000,(p[...,2]-center[2])*1000],axis=-1)
        pc=PolyCollection(front(q1),array=a['principalStretches'][mask,0],cmap='viridis',clim=(1,4),edgecolors='none')
        axes[0,col].add_collection(pc)
        ee=q0[:,[0,1,2,0],:];segments=np.stack([ee[:,:-1],ee[:,1:]],axis=2).reshape(-1,2,3)
        axes[0,col].add_collection(LineCollection(front(segments),colors='grey',linewidths=.18,alpha=.45))
        axes[0,col].set(xlim=(-4,4),ylim=(-15,8),aspect='equal',xlabel='Outward from neutral outer endpoint (mm)',ylabel='Above eye centre (mm)',title=s+': actual deformed outer surface; grey = prior edges')
        fig.colorbar(pc,ax=axes[0,col],label='Maximum principal stretch',shrink=.7)
        x=j['regions'][s+'_upperCrown']['largestNewDihedrals'][0]['midpointBeforeWorld'][0]
        report['sectionPlanes'][s]={'actualWorldX':x,'selection':'X through largest new upper-crown dihedral'}
        for label,p,color in [('before',old,'#808080'),('after',new,'#df611f')]:
            seg=sections(p,tri,x);save[s+'_'+label+'_actualYZSegmentsWorld']=seg
            yz=(seg[:,:,1:]-center[1:])*1000
            axes[1,col].add_collection(LineCollection(yz,colors=color,linewidths=1.15,label=label))
        axes[1,col].set(xlim=(-13,1),ylim=(-15,8),aspect='equal',xlabel='Y relative to eye centre (mm; front is left)',ylabel='Above eye centre (mm)',title=s+': actual triangle section X = %.3f mm'%(x*1000))
        axes[1,col].legend(loc='upper left');axes[1,col].grid(alpha=.2)
    fig.suptitle('Actual neutral outer-eye defect diagnosis — strength 1, not accepted',fontsize=15)
    fig.savefig(OUT.with_suffix('.png'),dpi=170);plt.close(fig)
    np.savez_compressed(OUT.with_suffix('.npz'),**save)
    report['sectionArraysSha256']=sha(OUT.with_suffix('.npz'));report['figureSha256']=sha(OUT.with_suffix('.png'))
    OUT.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=js),encoding='utf8')
    print(json.dumps({'jsonSha256':sha(OUT.with_suffix('.json')),'arraysSha256':report['sectionArraysSha256'],'figureSha256':report['figureSha256'],'path':str(OUT.with_suffix('.png'))}),flush=True)
if __name__=='__main__':run()
