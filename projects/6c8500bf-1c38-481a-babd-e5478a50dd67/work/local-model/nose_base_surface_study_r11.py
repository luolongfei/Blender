"""Read-only current nasal-base depth map; no generated-reference calibration."""
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=Path(__file__).resolve().parents[2];d=p/'work/reconstruction-r02'
    source=d/'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz'
    a=np.load(source);w=a['after_0.0_1.0World'];tri=a['after_0.0_1.0Triangles']
    # The region contains the entire nasal base and adjoining visible skin;
    # first-hit depth remains exact for the original full-head triangle IDs.
    vertices=w[tri];keep=(vertices[:,:,0].max(1)>-.023)&(vertices[:,:,0].min(1)<.023)&(vertices[:,:,2].max(1)>1.382)&(vertices[:,:,2].min(1)<1.414)
    tri_ids=np.flatnonzero(keep);selected=tri[keep]
    cls=runpy.run_path(str(Path(__file__).with_name('adult_blink_inner_thickness_support_r11.py')))['FrontTriangleIndex']
    index=cls(w,selected)
    xs=np.arange(-.022, .0220001, .00025);zs=np.arange(1.382,1.4140001,.00025)
    xx,zz=np.meshgrid(xs,zs);points=np.column_stack([xx.ravel(),np.zeros(xx.size),zz.ravel()])
    yy,ids,tested=index.front(points);depth=yy.reshape(xx.shape)
    if not np.isfinite(depth).all():raise RuntimeError('Complete actual first-hit nasal depth map required')
    local=np.ones_like(depth,bool);local[[0,-1],:]=False;local[:,[0,-1]]=False
    for dx,dz in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
        local[1:-1,1:-1]&=depth[1:-1,1:-1]>=depth[1+dz:len(zs)-1+dz,1+dx:len(xs)-1+dx]
    candidates=[]
    for iz,ix in zip(*np.nonzero(local & (abs(xx)>.004)&(abs(xx)<.016)&(zz>1.386)&(zz<1.401))):
        ring=[]
        for dx,dz in [(4,0),(-4,0),(0,4),(0,-4),(3,3),(3,-3),(-3,3),(-3,-3)]:
            ring.append(depth[iz+dz,ix+dx])
        candidates.append({'world':[float(xs[ix]),float(depth[iz,ix]),float(zs[iz])],
            'oneMillimeterNeighborhoodRecessMeanMeters':float(depth[iz,ix]-np.mean(ring)),
            'oneMillimeterNeighborhoodRecessMinimumMeters':float(depth[iz,ix]-max(ring)),
            'actualTriangleId':int(tri_ids[ids.reshape(xx.shape)[iz,ix]])})
    candidates.sort(key=lambda r:r['oneMillimeterNeighborhoodRecessMeanMeters'],reverse=True)
    anchors=[8009,8010,8011,8012,8013,8014,8015,2832,8016,8017,8018,8068,8019]
    cage=a['neutralCageWorld']
    report={'sourceSha256':'29e3e6b59a08487643d28642975a6ab1f92ca4af830f484af291f25434e08df8',
        'actualArraysSha256':sha(source),'scope':'Exact front-axis triangle first-hit depth and local cavity candidates on current neutral nasal base. No deformation, image-based scaling or ancestry template.',
        'gridStepMeters':.00025,'gridShape':list(depth.shape),'originalTriangleSubsetCount':len(selected),
        'localDepthMaximaCandidates':candidates,
        'currentNasalCageCenterline':[{'originalVertexId':i,'world':cage[i].tolist()} for i in anchors],
        'interpretation':['Local maxima are actual anterior-view recess candidates, not certified nostril boundaries.',
            'Front cavity prominence is a profile/coverage issue and does not by itself imply the whole nose is too large.',
            'Wait for true side views before selecting an alar rim, nasal-base inclination or tip-volume candidate.'],
        'sourceSaved':False,'geometryModified':False}
    out=d/'nose-base-current-surface-study-r11.json';out.write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(out.with_suffix('.npz'),x=xs,z=zs,frontY=depth,originalTriangleIds=tri_ids[ids].reshape(xx.shape))
    print(json.dumps({'candidates':candidates,'jsonSha256':sha(out)},indent=2))


if __name__=='__main__':main()
