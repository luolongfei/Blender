"""Diagram of the standalone contour design, with no model edits."""
from pathlib import Path
import hashlib,json,runpy
import numpy as np
from numpy.polynomial import polynomial as poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;P=HERE.parent.parent
stem=P/'work/reconstruction-r02/adult-outer-eye-continuous-target-r11'
j=json.loads(stem.with_suffix('.json').read_text());api=runpy.run_path(str(HERE/'adult_outer_eye_continuous_target_r11.py'))
fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
for col,s in enumerate('LR'):
    r=j['sides'][s];x0=np.array(r['sourceOriginalOutwardMeters']);center=r['eyeCenterWorldZ']
    x=np.linspace(0,r['endOutwardMeters'],1001);t=np.linspace(0,1,1001)
    xx=r['startOutwardMeters']+t*r['transitionLengthMeters']
    for edge,color in [('upper','#c34b24'),('lower','#2765b0')]:
        source=np.array(r['sourceOriginalUpperWorldZ' if edge=='upper' else 'sourceOriginalLowerWorldZ'])
        target=api['evaluate_target'](x,r,edge)
        axes[0,col].plot(x0*1000,(source-center)*1000,'--',color=color,alpha=.6,lw=1.3,label='actual neutral '+edge)
        axes[0,col].plot(x*1000,(target-center)*1000,color=color,lw=2,label='target '+edge)
        c=np.array(r['edges'][edge]['coefficientsWorldZAscending']);length=r['transitionLengthMeters']
        slope=poly.polyval(t,poly.polyder(c))/length
        curvature=poly.polyval(t,poly.polyder(c,2))/length**2/(1+slope*slope)**1.5/1000
        axes[1,col].plot(xx*1000,slope,color=color,lw=1.7,label=edge)
        axes[2,col].plot(xx*1000,curvature,color=color,lw=1.7,label=edge)
        axes[0,col].plot([r['endOutwardMeters']*1000],[(r['edges'][edge]['endJet']['valueWorldZ']-center)*1000],'o',color=color,ms=4)
    for row in range(3):
        axes[row,col].axvspan(0,3.8,color='#d9e6ef',alpha=.55)
        axes[row,col].axvline(3.8,color='#7c7c7c',ls=':',lw=1)
        axes[row,col].set(xlim=(0,13),xlabel='Outward from eye centre (mm)')
        axes[row,col].grid(alpha=.2)
    axes[0,col].set(ylim=(-5.5,5.5),ylabel='Z from eye centre (mm)',title=s+': actual neutral and proposed continuous target')
    axes[0,col].legend(fontsize=8,ncol=2,loc='lower left')
    axes[1,col].set(ylim=(-1.2,1.2),ylabel='Slope dz/dx',title='Single-direction upper/lower slopes')
    axes[2,col].set(ylim=(-.16,.16),ylabel='Signed curvature (1/mm)',title='No interior sign reversal; end curvature = 0')
fig.suptitle('Continuous contour proposal only — no mesh or contact acceptance',fontsize=15)
out=stem.with_suffix('.png');assert not out.exists();fig.savefig(out,dpi=160);plt.close(fig)
print(hashlib.sha256(out.read_bytes()).hexdigest())
