"""Scientific before/after plots of actual mesh-derived nasal direction study."""
import json
from pathlib import Path
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

P=Path(__file__).resolve().parents[2];D=P/'work/reconstruction-r02'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def profile(ax,w,tri,x,color,label):
    study=api('whole_face_proportion_study_r11')
    segments=study['plane_segments'](w,tri,0,x+1e-6)
    p=study['samples_at_z'](segments,np.arange(1.3825,1.4110001,.0001))
    pair=np.stack([p[:-1,[1,2]],p[1:,[1,2]]],axis=1)*1000
    pair=pair[np.linalg.norm(pair[:,1]-pair[:,0],axis=1)<2.]
    ax.add_collection(LineCollection(pair,colors=color,linewidths=1.7,label=label))
    ax.autoscale();ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    ax.set_xlabel('Anterior / posterior Y (mm)');ax.set_ylabel('Height Z (mm)')


def main():
    a=np.load(D/'nasal-rim-direction-study-r11.npz');w=a['beforeWorld'];q=a['afterWorld'];tri=a['triangles']
    contour=api('nasal_apparent_contour_r11')['measure']
    before,rb=contour(w,tri);after,ra=contour(q,tri)
    np.savez_compressed(D/'nasal-rim-direction-contours-r11.npz',**{'before_'+k:v for k,v in before.items()},**{'after_'+k:v for k,v in after.items()})
    (D/'nasal-rim-direction-contours-r11.json').write_text(json.dumps({'before':rb,'after':ra,'limits':'New partial apparent contour recomputed after deformation, not old contour points simply shifted. Not a complete anatomical rim or a production acceptance.'},indent=2),encoding='utf8')
    source=np.load(D/'nose-base-current-surface-study-r11.npz');xs=source['x']*1000;zs=source['z']*1000
    ids=source['originalTriangleIds'];amount=np.linalg.norm(a['offsets'][tri[ids]],axis=2).mean(2)*1000
    fig,axs=plt.subplots(2,2,figsize=(12.2,10),layout='constrained')
    ax=axs[0,0]
    image=ax.pcolormesh(xs,zs,amount,cmap='Blues',shading='nearest',vmin=0,vmax=1.4)
    ax.contour(xs,zs,source['frontY']*1000,levels=np.arange(-124,-101,2),colors='#999999',linewidths=.5,alpha=.7)
    for ww,c,colour,label in [(w,before,'#222222','Current visible rim'),(q,after,'#e76818','Proposed visible rim')]:
        edges=c['edges'][c['visibleRimWithDepthJump']]
        line=ww[edges][:,:,[0,2]]*1000
        ax.add_collection(LineCollection(line,colors=colour,linewidths=1.8,label=label))
    ax.set(xlim=(-18,18),ylim=(1383,1408),xlabel='Width X (mm)',ylabel='Height Z (mm)',title='Front: actual visible rim + affected surface')
    ax.set_aspect('equal');ax.legend(fontsize=8,loc='upper right');fig.colorbar(image,ax=ax,label='Theoretical displacement (mm)',shrink=.8)
    for ax,x,title in [(axs[0,1],0.,'True center profile: tip projection retained'),(axs[1,0],.0075,'Nostril meridian: X = +7.5 mm'),(axs[1,1],-.0075,'Nostril meridian: X = -7.5 mm')]:
        profile(ax,w,tri,x,'#222222','Current');profile(ax,q,tri,x,'#e76818','Proposed')
        ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('R11 nasal upper-rim direction study — actual mesh, not a rendered appearance verdict\n10° local angular transport + 0.35 mm basal/philtral smoothing; current29e3 only',fontsize=12)
    fig.savefig(D/'nasal-rim-direction-front-profile-r11.png',dpi=150)
    plt.close(fig)
    print(json.dumps({'before':rb,'after':ra},indent=2))


if __name__=='__main__':main()
