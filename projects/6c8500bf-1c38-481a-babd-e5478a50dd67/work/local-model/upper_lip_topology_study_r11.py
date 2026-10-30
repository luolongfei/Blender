"""Offline-only upper-vermilion contour study, NOT a Blender apply helper.

Use the observed complete female topology. Existing outer control contour has
one central high point rather than paired Cupid-bow peaks. A small candidate
is confined to that outer row and its first noseward neighbor row. Mid-body,
opening, lower-lip and oral rows are exactly fixed. This does not yet establish
that the evaluated contact surface stays fixed; actual Multires and images on
the adopted lower-face source are required before any application is written.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
KEY_PROPOSAL='R11 soft upper vermilion contour'
DRAFT_PARAMETERS={'centralOuterDropMeters':.00020,'pairedPeakRiseMeters':.00045,
    'lateralShoulderRiseMeters':.00020,'maximumForwardMeters':.00010,
    'firstNosewardRowFraction':.25,'maximumTotalWorldDisplacementMeters':.0006}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _smoother(x):
    t=np.clip(np.asarray(x,float),0.,1.)
    return t*t*t*(t*(6*t-15)+10)


def _interpolate_piecewise(distance,abscissa,values):
    """Compact smooth interpolation along the actual ordered surface controls."""
    x=np.asarray(abscissa,float);y=np.asarray(values,float)
    if np.any(np.diff(x)<=0):raise RuntimeError('Outer-arc anchors are not ordered')
    d=np.asarray(distance,float);out=np.zeros_like(d)
    for j in range(len(x)-1):
        take=(d>=x[j])&(d<x[j+1]);t=_smoother((d[take]-x[j])/(x[j+1]-x[j]))
        out[take]=y[j]*(1-t)+y[j+1]*t
    out[d==x[-1]]=y[-1]
    return out


def study(cage_world):
    skin=_api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']()
    v=np.asarray(cage_world,float)
    if v.shape!=(10582,3):raise RuntimeError('Complete original female index space required')
    mouth=land['mouth'];qa=_api('face_contact_audit_r11')
    outer=list(map(int,mouth['outerVermilionLoopCandidate']['sourceVertexIndices']))
    mid=list(map(int,mouth['vermilionMidLoop']['sourceVertexIndices']))
    opening=list(map(int,mouth['openingRimCandidate']['sourceVertexIndices']))
    left=min(outer,key=lambda i:v[i,0]);right=max(outer,key=lambda i:v[i,0])
    upper,lower=qa['_split'](outer,v,left,right)
    if 8007 not in upper or 10444 not in upper or 10485 not in upper:raise RuntimeError('Observed upper-lip contour identity changed')
    center=upper.index(8007);arcs={'R':list(reversed(upper[:center+1])),'L':upper[center:]}
    anchors={'R':[8007,10444,1636,1752,1806], 'L':[8007,10485,6769,6885,6939]}
    per_side={};delta=np.zeros_like(v);p=DRAFT_PARAMETERS
    for side,arc in arcs.items():
        if any(i not in arc for i in anchors[side]):raise RuntimeError('Expected anatomical contour sequence missing')
        distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(v[arc],axis=0),axis=1))]
        ds=[float(distances[arc.index(i)]) for i in anchors[side]]
        heights=[-p['centralOuterDropMeters'],p['pairedPeakRiseMeters'],p['lateralShoulderRiseMeters'],0.,0.]
        forward=[.4*p['maximumForwardMeters'],p['maximumForwardMeters'],.5*p['maximumForwardMeters'],0.,0.]
        dz=_interpolate_piecewise(distances,ds,heights);dy=-_interpolate_piecewise(distances,ds,forward)
        delta[arc,2]=dz;delta[arc,1]=dy
        peak=anchors[side][1]
        per_side[side]={'orderedUpperOuterArc':arc,'sourceArcDistancesMeters':distances.tolist(),
            'shapeAnchors':anchors[side],'anchorArcDistancesMeters':ds,
            'currentPeakMinusCenterZ':float(v[peak,2]-v[8007,2]),
            'candidatePeakMinusCenterZ':float((v[peak]+delta[peak])[2]-(v[8007]+delta[8007])[2]),
            'currentOuterAnchorWorld':v[anchors[side]].tolist(),'proposedAnchorDeltaWorld':delta[anchors[side]].tolist()}
    strip=set(mouth['vermilionStripVertexIndices']);outer_set=set(outer)
    adjacency=[set() for _ in v]
    for a,b in donor['edges']:adjacency[int(a)].add(int(b));adjacency[int(b)].add(int(a))
    noseward={}
    for index in upper:
        if np.linalg.norm(delta[index])<1e-12:continue
        candidates=[i for i in adjacency[index] if i not in strip and i not in outer_set and v[i,2]>v[index,2]]
        if len(candidates)!=1:raise RuntimeError('No unique real noseward quad continuation at '+str(index)+': '+str(candidates))
        other=candidates[0]
        if other in noseward:raise RuntimeError('Multiple rows unexpectedly meet at noseward control')
        noseward[other]=index
        delta[other]=delta[index]*p['firstNosewardRowFraction']
    # Exact source-data protection: no soft mask touching the contact rows.
    fixed=set(mid)|set(opening)|set(lower)
    oral=np.flatnonzero(skin['_vertices_of_groups'](donor,{7,8}));fixed.update(map(int,oral))
    if np.any(delta[list(fixed)]):raise RuntimeError('Draft touched mid-lip, lower-lip or oral/contact cage')
    selected=np.linalg.norm(delta,axis=1)>1e-12
    if np.linalg.norm(delta,axis=1).max()>p['maximumTotalWorldDisplacementMeters'] or not 8<int(selected.sum())<45:
        raise RuntimeError('Upper-contour draft exceeded its bounded scope')
    edges=donor['edges'];old=v[edges[:,1]]-v[edges[:,0]];new=old+delta[edges[:,1]]-delta[edges[:,0]]
    ratio=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-15);active=selected[edges].any(1)
    triangles=[]
    for start,size in zip(donor['face_starts'],donor['face_lengths']):
        f=donor['corner_vertices'][start:start+size]
        if selected[f].any():triangles.extend((int(f[0]),int(f[j]),int(f[j+1])) for j in range(1,len(f)-1))
    triangles=np.asarray(triangles,np.int32);a=v[triangles];b=(v+delta)[triangles]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    aa=np.linalg.norm(na,axis=1);bb=np.linalg.norm(nb,axis=1);valid=aa>1e-12
    alignment=np.sum(na[valid]*nb[valid],axis=1)/(aa[valid]*bb[valid])
    if alignment.min()<=0 or ratio[active].min()<.75 or ratio[active].max()>1.25:raise RuntimeError('Draft raw contour deforms too strongly')
    report={'status':'OFFLINE DRAFT; wait for the adopted lower-face images before choosing amplitude',
        'provenance':provenance,'proposedKey':KEY_PROPOSAL,'draftParameters':p,'sides':per_side,
        'changedIndices':np.flatnonzero(selected).tolist(),'firstNosewardConnections':{str(k):v for k,v in noseward.items()},
        'changedRawVertices':int(selected.sum()),'maximumWorldDeltaMeters':float(np.linalg.norm(delta,axis=1).max()),
        'fixedControlSets':{'upperLowerOpening':opening,'wholeMidBodyLoop':mid,'lowerOuterArc':lower,'oralVertexCount':len(oral)},
        'rawEdgeRatioQuantiles':np.quantile(ratio[active],[0,.05,.5,.95,1]).tolist(),
        'rawAreaRatioRange':[float((bb[valid]/aa[valid]).min()),float((bb[valid]/aa[valid]).max())],
        'rawNormalAlignmentMinimum':float(alignment.min()),'noMaterialChange':True,
        'actualContactUnchanged':None,'appearanceAccepted':False,'blenderApplied':False,
        'anatomy':'The upper outer-vermilion control loop currently has a central crest; its immediately lateral controls are about 0.15 mm lower. This draft introduces two small outer peaks while retaining the existing deeper central lip tubercle and every mid/opening/oral cage point.',
        'limits':['Outer control loop is anatomical support, not a guaranteed exact visible vermilion edge or wet line.',
            'Pinned cage contact points do not prove an unchanged evaluated Multires contact surface. Fresh evaluated central and lateral lip sections and 3D local clearance are required.',
            'No black-seam tint, increased red saturation, mouth narrowing, lower-lip filling, whole-philtrum scaling or corner lift is proposed.',
            'Actual front/quarter/side and mouth close-up with the shortened lower face decide amplitude; reject a sharp W, lipstick-like ridge, lip thinning or new opening.']}
    return delta.astype(np.float32),report


def write_study(output_stem):
    folder=project_path()/'work/reconstruction-r02';stem=(project_path()/output_stem).resolve();stem.relative_to(folder.resolve())
    paths=[stem.with_suffix('.json'),stem.with_suffix('.npz')]
    if any(p.exists() for p in paths):raise FileExistsError('Retain existing study outputs')
    source=folder/'adult-neck-observation-r11.npz';meta=json.loads((folder/'adult-neck-observation-r11.json').read_text(encoding='utf-8'))
    with np.load(source) as a:cage=a['rest_00_raw_world'].copy()
    delta,report=study(cage)
    report.update(sourceObservation=str(source.relative_to(project_path())),sourceObservationSha256=_sha(source),
        observedBlend=meta['source'],observedBlendSha256=meta['sourceSha256'],helperSha256=_sha(__file__),
        currentAdoptedSourceMeasured=False)
    np.savez_compressed(paths[1],observed_cage_world=cage,proposed_delta_world=delta,changed_indices=np.flatnonzero(np.linalg.norm(delta,axis=1)>1e-12))
    report['draftArraysSha256']=_sha(paths[1]);paths[0].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
