"""Offline visible-rim direction study on the real current character surface.

Not an installer. The selected mouth source must be sampled and its nasal
dependency rechecked before this direction can become an actual candidate.
"""
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np

P=Path(__file__).resolve().parents[2]
D=P/'work/reconstruction-r02'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def smooth(t):t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))


def plan(world,faces,groups,triangles,evidence,angle_degrees=10.,philtrum_forward=.00035):
    w=np.asarray(world,float);x,y,z=w.T
    if w.shape!=(238889,3) or np.asarray(faces).shape!=(238672,4):raise RuntimeError('Current actual fixed-L3 head required')
    # Actual upper apparent-contour evidence determines the pivot separately
    # for each side. These are local cover arcs, not a complete anatomical ring.
    mids=evidence['midpoints'];upper=evidence['upperCapMask']
    offset=np.zeros_like(w);angle=np.zeros(len(w));pivots={}
    for sign,label in [(1,'positiveX'),(-1,'negativeX')]:
        arc=mids[upper&(mids[:,0]*sign>0)]
        if len(arc)<10:raise RuntimeError('Actual visible upper rim evidence missing')
        top=arc[arc[:,2]>arc[:,2].max()-.0006]
        cy=float(np.median(top[:,1])+.003);cz=float(np.median(top[:,2])-.0018)
        pivots[label]={'worldY':cy,'worldZ':cz,'observedUpperArcPoints':len(arc),'observedTopArcPoints':len(top)}
        lateral=smooth((abs(x)-.003)/.004)*(1-smooth((abs(x)-.012)/.006))
        # Stop the angular transport above the actual upper-rim cap; a higher
        # support would add unwanted projection to the lateral tip dome.
        vertical=smooth((z-1.3855)/.0065)*(1-smooth((z-1.3978)/.0052))
        # Keep the projecting tip dome itself unchanged, including the flank
        # beyond the central 3 mm strip; transport starts behind that dome.
        depth=smooth((y+.1225)/.003)*(1-smooth((y+.105)/.009))
        theta=np.deg2rad(angle_degrees)*lateral*vertical*depth*(x*sign>0)
        dy=y-cy;dz=z-cz
        offset[:,1]+=dy*(np.cos(theta)-1)-dz*np.sin(theta)
        offset[:,2]+=dy*np.sin(theta)+dz*(np.cos(theta)-1)
        angle+=theta
    # A small, broad change to the basal/philtral recess, without moving the
    # upper vermilion or the nasal tip. The current forward skin, not a cheek
    # average, defines which layer is affected.
    study=api('whole_face_proportion_study_r11')
    sag=study['plane_segments'](w,triangles,0,.000001)
    profile=study['samples_at_z'](sag,np.arange(1.3835,1.3930001,.00025))
    fy=np.interp(z,profile[:,2],profile[:,1])
    fill=philtrum_forward*(1-smooth((abs(x)-.003)/.006))*smooth((z-1.3835)/.004)*(1-smooth((z-1.3875)/.0055))*(1-smooth((y-fy-.001)/.003))
    offset[:,1]-=fill
    offset[np.linalg.norm(offset,axis=1)<1e-10]=0.
    protected=np.unique(faces[np.isin(groups,[2,3,4,5,7,8,22])])
    facial=np.zeros(len(w),bool);facial[np.unique(faces[groups==17])]=True
    shared=np.intersect1d(np.flatnonzero(facial),protected)
    if np.any(offset[shared]):raise RuntimeError('Nonzero motion reaches a shared anatomical boundary; do not clip it into a hard seam')
    # The upper oral roof is spatially close to the nose cavity but is a
    # separate surface layer. Its true interface already lies outside this
    # field and has exactly zero motion, so facial-only selection is smooth.
    excluded=np.flatnonzero(~facial&np.any(offset!=0,axis=1))
    offset[~facial]=0.;angle[~facial]=0.;fill[~facial]=0.
    active=np.any(offset!=0,axis=1)
    tip=(abs(x)<=.003)&(z>=1.394)
    if np.any(offset[protected]) or np.any(offset[tip]):
        conflict={}
        for group in [2,3,4,5,7,8,22]:
            ids=np.unique(faces[groups==group]);moved=ids[np.any(offset[ids]!=0,axis=1)]
            if len(moved):conflict[str(group)]={'count':len(moved),'bounds':[w[moved].min(0).tolist(),w[moved].max(0).tolist()], 'maximumMeters':float(np.linalg.norm(offset[moved],axis=1).max())}
        raise RuntimeError('Candidate crosses protected domain: '+json.dumps(conflict))
    after=w+offset
    quality=api('upper_lid_roll_install_r11')['quality'](w,after,triangles,active)
    cross=api('adult_blink_midpath_contact_r11')['changed_intersections']
    old=cross(w,triangles,active);new=cross(after,triangles,active)
    report={'sourceScope':'Pure direction study using actual29e3; not installed, not proof for the selected future mouth combination.',
        'angularCoverDegrees':angle_degrees,'maximumPhiltralForwardMeters':float(fill.max()),'pivots':pivots,
        'activeVertices':int(active.sum()),'activeBounds':[w[active].min(0).tolist(),w[active].max(0).tolist()],
        'separateNonfacialLayerExcludedVertices':len(excluded),'sharedAnatomicalInterfaceDisplacementExactZero':True,
        'maximumDisplacementMeters':float(np.linalg.norm(offset,axis=1).max()),'quality':quality,
        'oldChangedTriangleCrossings':len(old),'newChangedTriangleCrossings':len(new),'addedCrossings':len(new-old),
        'centralNasalTipExact':bool(np.array_equal(w[tip],after[tip])),
        'globalAnteriorProjectionExact':bool(w[:,1].min()==after[:,1].min()),
        'widthExact':bool(np.array_equal(w[:,0],after[:,0])),
        'protectedAnatomyExact':bool(np.array_equal(w[protected],after[protected])),
        'appearanceAccepted':False,'technicalAcceptance':False,
        'limits':['The prior arcs are partial front apparent contours, not complete nostril boundaries.',
            'The contour must be remeasured on after geometry, not only transformed from old endpoints.',
            'A meaningful change in frontal coverage does not certify final likeness or appropriate visible darkness.',
            'Current approved mouth geometry and all source dependencies require actual reevaluation before installation.']}
    arrays={'beforeWorld':w,'offsets':offset,'afterWorld':after,'faces':faces,'groups':groups,'triangles':triangles,
        'angleRadians':angle,'philtralForward':fill,'protectedTip':tip}
    return arrays,report


def run():
    actual=D/'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz';a=np.load(actual)
    groups=np.load(D/'head-neck-current-posture-r11-v2.npz')['rest_head_faceGroups']
    evidence_path=D/'nose-rim-silhouette-diagnostic-r11.npz';e=np.load(evidence_path)
    arrays,report=plan(a['after_0.0_1.0World'],a['fullFaces'],groups,a['after_0.0_1.0Triangles'],e)
    report.update(sourceSha256='29e3e6b59a08487643d28642975a6ab1f92ca4af830f484af291f25434e08df8',
        sourceActualArraysSha256=sha(actual),apparentContourEvidenceSha256=sha(evidence_path),helperSha256=sha(__file__))
    out=D/'nasal-rim-direction-study-r11.json';payload=out.with_suffix('.npz')
    np.savez_compressed(payload,**arrays);report.update(arrays=str(payload.relative_to(P)),arraysSha256=sha(payload))
    out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))


if __name__=='__main__':run()
