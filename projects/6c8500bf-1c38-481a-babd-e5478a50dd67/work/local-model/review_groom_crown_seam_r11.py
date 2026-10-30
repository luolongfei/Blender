"""Read-only current native root/initial-flow diagnostic; no Blender or edits."""
import json, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
P=ROOT/'projects/6c8500bf-1c38-481a-babd-e5478a50dd67'
W=P/'work/reconstruction-r02'
FILES={'dense':'reference-adult-front-detail-v10e-r11.groom-front-detail.npz',
       'v6':'reference-adult-front-groom-v6-r11.groom-reference.npz',
       'growth':'front-groom-actual-v6-r11.npz',
       'raw':'reference-tie-alignment-r11-v2.npz'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=W/'reference-crown-seam-flow-review-r11'
    if out.with_suffix('.json').exists() or out.with_suffix('.png').exists():
        raise RuntimeError('New diagnostic output required')
    a=np.load(W/FILES['dense']);b=np.load(W/FILES['v6']);m=np.load(W/FILES['growth']);raw=np.load(W/FILES['raw'])
    q=a['final-taper_world'].astype(float);s=a['final-taper_starts'];n=a['final-taper_lengths'];g=a['final-taper_group']%256;r=q[s]
    delta5=np.zeros_like(r)
    for i,(start,count) in enumerate(zip(s,n)):
        p=q[start:start+count];arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        delta5[i]=np.array([np.interp(.005,arc,p[:,k]) for k in range(3)])-p[0]
    region=(r[:,0]>.005)&(r[:,0]<.07)&(r[:,1]>-.01)&(r[:,1]<.035)&(r[:,2]>1.475)
    f=np.flatnonzero(region&(g<40));rear=np.flatnonzero(region&(g==40))
    d=np.linalg.norm(r[f,None]-r[None,rear],axis=2);j=d.argmin(1);near=d[np.arange(len(f)),j]<.0015
    pairs=np.column_stack((f,rear[j]))[near];dist=np.linalg.norm(r[pairs[:,0]]-r[pairs[:,1]],axis=1)
    u=delta5[pairs[:,0]];v=delta5[pairs[:,1]]
    angle=np.degrees(np.arccos(np.clip(np.sum(u*v,axis=1)/np.linalg.norm(u,axis=1)/np.linalg.norm(v,axis=1),-1.,1.)))
    end_dist=np.linalg.norm(r[pairs[:,0]]+u-r[pairs[:,1]]-v,axis=1)
    rows=[]
    for y in np.arange(-.005,.03001,.005):
        mask=(r[:,0]>.005)&(r[:,0]<.07)&(r[:,1]>=y)&(r[:,1]<y+.005)&(r[:,2]>1.475)
        rows.append({'yStartMeters':float(y),'yEndMeters':float(y+.005),'roots':int(mask.sum()),
                     'front':int((mask&(g<40)).sum()),'gather':int((mask&(g==40)).sum())})
    same=np.array_equal(a['interpolate_world'][a['interpolate_starts']],b['interpolate_world'][b['interpolate_starts']])
    report={'scope':'Read-only scientific diagnostic using native exported coordinates; not a render or modification.',
            'sourceFiles':{k:{'path':str((W/v).relative_to(P)),'sha256':sha(W/v)} for k,v in FILES.items()},
            'actualQuarterImage':'previews/reconstruction-r02/upper-lid-roll-quarter-r11/frame-0001.png',
            'all26046NativeInterpolationRootsExactSinceV6':same,'positiveCrownRegionMeters':{'x':[.005,.070],'zMinimum':1.475},
            'rootSlabs':rows,'crossRoleNearestPairsUnder1_5mm':len(pairs),
            'rootDistanceMetersP0P50P95P100':np.percentile(dist,[0,50,95,100]).tolist(),
            'initial5mmArcDirectionAngleDegreesP0P50P95P100':np.percentile(angle,[0,50,95,100]).tolist(),
            'initial5mmArcEndpointSeparationMetersP0P50P95P100':np.percentile(end_dist,[0,50,95,100]).tolist(),
            'pairs':[{'frontCurve':int(i),'gatherCurve':int(j),'rootSeparationMeters':float(dd),'after5mmSeparationMeters':float(ee),'angleDegrees':float(aa)} for (i,j),dd,ee,aa in zip(pairs,dist,end_dist,angle)],
            'finding':'The positive-crown band contains dense roots throughout. Adjacent front and gather roots fan apart immediately. The v6 hard role boundary at root Y=14mm has become an unintended broad curved part.',
            'limits':['The growth mesh export predates v6, whose installer preserves its vertices. Actual current dense interpolation roots are checked bitexact against v6.',
                      'Plots show native point projections, not visibility or shading; quarter render independently establishes visible bare skin.',
                      'No gather-reconnection plan, source geometry, material, radius or frozen evidence was modified.']}
    im=Image.new('RGB',(1800,1120),'white');draw=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16);big=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26)
    except OSError:font=small=big=ImageFont.load_default()
    blue=(35,90,170);orange=(205,100,35);grey=(213,215,218);black=(35,35,35)
    draw.text((30,15),'Actual crown roots and initial 5 mm of hair flow',fill=black,font=big)
    draw.text((30,52),'Native v10e data; front locks blue, gathers orange. Top projection: X horizontal, posterior Y upward. All distances in mm.',fill=black,font=font)
    def panel(box,limits,title):
        x0,y0,x1,y1=box;lx,hx,ly,hy=limits
        draw.text((x0,y0-30),title,fill=black,font=font);draw.rectangle(box,outline=(150,150,150))
        def project(point):return (x0+(point[0]-lx)/(hx-lx)*(x1-x0),y1-(point[1]-ly)/(hy-ly)*(y1-y0))
        for x in np.arange(np.ceil(lx*100)/100,hx+.00001,.01):
            a1=project([x,ly]);a2=project([x,hy]);draw.line((a1,a2),fill=(237,237,237));draw.text((a1[0]-10,y1+4),str(round(x*1000)),fill=black,font=small)
        for y in np.arange(np.ceil(ly*100)/100,hy+.00001,.01):
            a1=project([lx,y]);a2=project([hx,y]);draw.line((a1,a2),fill=(237,237,237));draw.text((x0-34,a1[1]-7),str(round(y*1000)),fill=black,font=small)
        return project
    pa=panel((65,135,850,590),(-.078,.078,-.035,.055),'A. Existing growth mesh and current raw guide roots')
    scalp=m['scalp_world'];edges=m['scalp_edges']
    for edge in edges:
        p=scalp[edge]
        if np.min(p[:,2])>1.475 and np.all((p[:,1]>-.035)&(p[:,1]<.055)):draw.line([pa(v) for v in p],fill=grey,width=1)
    rw=raw['main_world'];rs=raw['main_starts'];rl=raw['main_lengths'];rg=raw['main_role'];rr=rw[rs]
    for root,role in zip(rr,rg):
        if root[2]>1.475 and -.035<root[1]<.055:
            x,y=pa(root);color=blue if role<40 else orange;draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
    yline=pa([0,.014])[1];draw.line((65,yline,850,yline),fill=(90,90,90),width=2);draw.text((68,yline-24),'v6 raw role boundary Y = +14',fill=black,font=small)
    pb=panel((965,135,1750,590),(-.078,.078,-.035,.055),'B. Actual dense roots: continuous across the boundary')
    mask=(r[:,2]>1.475)&(r[:,1]>-.035)&(r[:,1]<.055)
    for i in np.flatnonzero(mask):
        x,y=pb(r[i]);draw.ellipse((x-1,y-1,x+1,y+1),fill=blue if g[i]<40 else orange)
    yline=pb([0,.014])[1];draw.line((965,yline,1750,yline),fill=(80,80,80),width=1)
    pc=panel((65,695,850,1045),(.005,.070,-.005,.035),'C. 47 adjacent front/gather pairs and their first 5 mm')
    for i,j in pairs:
        draw.line((pc(r[i]),pc(r[j])),fill=(150,150,150),width=1)
        for k,color in ((i,blue),(j,orange)):
            a1=pc(r[k]);a2=pc(r[k]+delta5[k]);draw.line((a1,a2),fill=color,width=2)
            x,y=a1;draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            direction=np.array(a2)-np.array(a1);direction/=max(np.linalg.norm(direction),1e-9);perp=np.array([-direction[1],direction[0]])
            draw.polygon([a2,tuple(np.array(a2)-direction*6+perp*3),tuple(np.array(a2)-direction*6-perp*3)],fill=color)
    draw.text((960,668),'D. Root counts per 5 mm Y strip (X=5..70; Z>1475)',fill=black,font=font)
    for k,row in enumerate(rows):
        y=715+k*35;draw.text((960,y-5),f"{row['yStartMeters']*1000:3.0f} to {row['yEndMeters']*1000:2.0f}",fill=black,font=small)
        x=1060;scale=1.6;draw.rectangle((x,y,x+row['front']*scale,y+21),fill=blue);x+=row['front']*scale
        draw.rectangle((x,y,x+row['gather']*scale,y+21),fill=orange);draw.text((1510,y-4),str(row['roots'])+' roots',fill=black,font=small)
    draw.text((960,1012),f"Root gap median {np.median(dist)*1000:.2f} mm; after 5 mm of flow: {np.median(end_dist)*1000:.2f} mm",fill=black,font=font)
    draw.text((960,1042),f"Direction angle median {np.median(angle):.1f} deg; P95 {np.percentile(angle,95):.1f} deg",fill=black,font=font)
    draw.text((30,1090),'Geometry diagnostic only. No model changes. The quarter render establishes the visible bare-skin arc; these plots explain its native root/flow cause.',fill=black,font=small)
    im.save(out.with_suffix('.png'))
    out.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'report':str(out.with_suffix('.json')),'plot':str(out.with_suffix('.png')),'pairs':len(pairs),'v6RootsExact':same}))

if __name__=='__main__':main()
