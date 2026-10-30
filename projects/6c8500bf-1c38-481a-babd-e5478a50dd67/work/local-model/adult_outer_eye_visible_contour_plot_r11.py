"""Plot saved actual contours; no model or reference image is edited."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent.parent.parent
W=P/'work/reconstruction-r02';stem=W/'adult-outer-eye-visible-contour-r11'
a=np.load(stem.with_suffix('.npz'))
meta=json.loads((P/'previews/reconstruction-r02/outer-eye-neutral-direction-preview-r11/neutral-geometry-preview.json').read_text())
bridge=json.loads((W/'adult-outer-eye-head-bridge-support-r11.json').read_text())
fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
for col,s in enumerate('LR'):
    md=meta['plan']['sides'][s];c=np.array(md['centerWorld']);sign=md['outwardSign']
    for label,style in [('before','--'),('preview','-')]:
        rows=a[s+'_'+label+'_contour'];x=sign*(rows[:,0]-c[0])*1000
        for edge,index,color in [('upper',2,'#c34b24'),('lower',1,'#2765b0')]:
            axes[0,col].plot(x,(rows[:,index]-c[2])*1000,style,color=color,lw=1.6,label=label+' '+edge)
            derivatives=a[s+'_'+label+'_'+edge+'Derivatives']
            axes[1,col].plot(x,derivatives[:,0],style,color=color,lw=1.3)
            axes[2,col].plot(x,derivatives[:,2],style,color=color,lw=1.3)
    for k in range(3):
        axes[k,col].axvspan(0,3,color='#d5e5ef',alpha=.6)
        axes[k,col].axvline(sign*(md['endpointX']-c[0])*1000-3.8,color='#6d6d6d',ls=':',lw=1.2)
        axes[k,col].set(xlim=(2,13),xlabel='Outward from eye centre (mm)')
        axes[k,col].grid(alpha=.2)
    for row in bridge['results'][s]['before']:
        if row['opening']:
            x=abs(row['xMm'])-abs(c[0])*1000
            axes[0,col].plot([x,x],[row['opening']['lowerOffsetMm'],row['opening']['upperOffsetMm']],color='#3b9963',lw=2)
    axes[0,col].set(ylim=(-5,5),ylabel='Z from eye centre (mm)',title=s+': true visible boundary; green = neutral head opening')
    axes[0,col].legend(fontsize=8,ncol=2,loc='upper right')
    axes[1,col].set(ylim=(-3,3),ylabel='Visible-edge slope dz/dx',title='Dashed: neutral / solid: rejected preview')
    axes[2,col].set(ylim=(-2,2),ylabel='Signed curvature (1/mm)',title='Measured over 0.6 mm; dotted line = old support start')
fig.suptitle('Actual outer-eye S-step evidence — neutral source retained',fontsize=15)
out=stem.with_suffix('.png');assert not out.exists();fig.savefig(out,dpi=160);plt.close(fig)
print(hashlib.sha256(out.read_bytes()).hexdigest())
