"""Actual donor/target cap angular domain audit; no Blender execution."""
import hashlib,json,math,runpy
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[2];W=P/'work/reconstruction-r02'
m=runpy.run_path(str(Path(__file__).with_name('donor_groom_adult_r11.py')))
d=np.load(W/'hair-donor-guides-r10.npz');a=np.load(W/'adult-crown-domain-r11.npz')
v=d['growth_local'];corners=d['growth_loop_vertex'];starts=d['growth_polygon_start'];sizes=d['growth_polygon_size'];tri=[]
for start,size in zip(starts,sizes):
    f=corners[start:start+size]
    for j in range(1,int(size)-1):tri.append([f[0],f[j],f[j+1]])
tri=np.asarray(tri,dtype=np.int32);center=np.array([0.,-.47,15.35])
sa,sl,se=m['_polar_domain'](v,tri,center,True)
ta,tl,te=m['_polar_domain'](a['root_world'],a['root_triangles'],a['center_world'],False)
delta=v-center;direction=delta/np.linalg.norm(delta,axis=1,keepdims=True)
phi=np.arctan2(direction[:,1],direction[:,0]);theta=np.arccos(direction[:,2])
source_limits=np.interp(phi,sa,sl);target_limits=np.interp(phi,ta,tl)-.012
ratio=np.minimum(1.,target_limits/source_limits);newtheta=theta*ratio
rows=[]
for i in range(len(v)):
    rows.append({'vertex':i,'sourceTheta':float(theta[i]),'targetTheta':float(newtheta[i]),'phi':float(phi[i]),
                 'targetSafeLimit':float(target_limits[i]),'ratio':float(ratio[i]),'safeMarginRadians':float(target_limits[i]-newtheta[i])})
report={'helperSha256':hashlib.sha256(Path(m['__file__']).read_bytes()).hexdigest(),
        'actualTargetArraysSha256':hashlib.sha256((W/'adult-crown-domain-r11.npz').read_bytes()).hexdigest(),
        'sourceGrowthVertices':len(v),'sourceBoundaryEdges':se,'targetBoundaryEdges':te,
        'originalOutsideTargetDomain':int(np.count_nonzero(theta>target_limits+1e-6)),
        'mappedOutsideTargetDomain':int(np.count_nonzero(newtheta>target_limits+1e-6)),
        'sourceEnvelopeContainmentError':float(np.max(theta-source_limits)),
        'minimumRatio':float(ratio.min()),'minimumAngularMarginBeyondSafeLimit':float(np.min(target_limits-newtheta)),
        'vertex16':rows[16],'minimumTargetSafetyAppliedRadians':.012,
        'method':'Exact boundary-edge/azimuth-plane intersections on actual source and target triangles, conservative angular envelopes, proportional polar row mapping. Actual target non-ear skin rays still must run in Blender.',
        'blenderStarted':False,'sourceEdited':False,'appearanceAccepted':False}
(W/'donor-scalp-domain-audit-r11.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,indent=2))
