"""Offline, topology-bound neutral outer-upper-lid relief draft.

Only an actual R11 L3 observation may be used. No Blender access or source
mutation. The protected visible border is not asserted to be a physical wet
line. Installation and intermediate blink/contact checks remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

PARAMETERS=dict(maximumInwardMeters=.00025, visibleBorderProtectedMeters=.0008,
    fullInfluenceStartsMeters=.0015, fullInfluenceEndsMeters=.0025,
    influenceEndsMeters=.004, cornerProtectedMeters=.002, cornerFadeMeters=.003,
    conservativeSphereProtectedMeters=.0007, conservativeSphereFullMeters=.0012)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(value):
    value=np.clip(value,0.,1.)
    return value*value*(3.-2.*value)


def plan(meta,data):
    study=runpy.run_path(str(Path(__file__).with_name('upper_lid_surface_study_r11.py')))
    world=np.asarray(data['head_world'],float);tri=np.asarray(data['head_triangles'],np.int32)
    normal=study['normals'](world,tri)
    delta=np.zeros_like(world);weights={};records={}
    for side in ('L','R'):
        eye=meta['eyes'][side];center=np.asarray(eye['centerWorld'],float)
        globe=np.asarray(data['eye_'+side+'_world'],float)
        radius=float(np.linalg.norm(globe-center,axis=1).max())
        contour=[p for p in eye['contour'] if not p['upper']['unboundedAtGlobeRim'] and p['upper'].get('headOutside120Micrometers')]
        x=np.asarray([p['x'] for p in contour]);z=np.asarray([p['upper']['worldZ'] for p in contour])
        y=np.asarray([p['upper']['headOutside120Micrometers']['world'][1] for p in contour])
        height=world[:,2]-np.interp(world[:,0],x,z)
        front_offset=world[:,1]-np.interp(world[:,0],x,y)
        clearance=np.linalg.norm(world-center,axis=1)-radius
        # Source normals select only the exterior-facing crown. No protected
        # near-globe point is displaced; no socket/contact line is inferred
        # from the coarse cage or a hand-drawn circle.
        w=smooth((height-.0008)/.0007)*(1.-smooth((height-.0025)/.0015))
        w*=smooth((world[:,0]-x.min()-.002)/.003)*smooth((x.max()-.002-world[:,0])/.003)
        w*=smooth((clearance-.0007)/.0005)
        w*=smooth((-normal[:,1]-.25)/.4)
        w*=1.-smooth((abs(front_offset)-.003)/.002)
        side_delta=-normal*(PARAMETERS['maximumInwardMeters']*w[:,None])
        changed=w>1e-9
        if not 100<int(changed.sum())<3000:
            raise RuntimeError('Unexpected actual outer-upper-lid support size: '+str(changed.sum()))
        if np.max(np.linalg.norm(delta[changed],axis=1))>0:
            raise RuntimeError('The two physical eyelid domains overlap')
        after=world+side_delta
        radial_after=np.linalg.norm(after-center,axis=1)-radius
        if radial_after[changed].min()<.0007-1e-9:
            raise RuntimeError('Draft consumes the protected conservative globe margin')
        delta+=side_delta;weights[side]=w.astype(np.float32)
        records[side]=dict(affectedEvaluatedVertices=int(changed.sum()), centerWorld=center.tolist(),
            conservativeBoundingRadiusMeters=radius,
            maximumInwardMeters=float(np.linalg.norm(side_delta,axis=1).max()),
            minimumChangedHeightAboveVisibleBoundaryMeters=float(height[changed].min()),
            minimumConservativeClearanceBeforeMeters=float(clearance[changed].min()),
            minimumConservativeClearanceAfterMeters=float(radial_after[changed].min()),
            changedWorldBounds=[world[changed].min(0).tolist(),world[changed].max(0).tolist()],
            maximumVisibleProtectedBandDisplacementMeters=float(np.linalg.norm(side_delta[height<=.0008],axis=1).max()),
            maximumConservativeGlobeProtectedDisplacementMeters=float(np.linalg.norm(side_delta[clearance<=.0007],axis=1).max()))
    active=np.linalg.norm(delta,axis=1)>1e-12;local_tri=tri[active[tri].any(1)]
    a=world[local_tri];b=(world+delta)[local_tri]
    n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    area0=np.linalg.norm(n0,axis=1);area1=np.linalg.norm(n1,axis=1)
    valid=area0>1e-14;alignment=np.sum(n0[valid]*n1[valid],axis=1)/(area0[valid]*area1[valid])
    if alignment.min()<=0 or (area1[valid]/area0[valid]).min()<.5:
        raise RuntimeError('Offline relief folds or strongly collapses an actual triangle')
    return dict(deltaWorld=delta.astype(np.float32),normalWorld=normal.astype(np.float32),
        weight_L=weights['L'],weight_R=weights['R'],originalVertexIds=data['head_originalVertexIds'],
        neutralWorld=world,triangles=tri),dict(eyes=records,parameters=PARAMETERS,
        affectedEvaluatedVertices=int(active.sum()),minimumTriangleNormalAlignment=float(alignment.min()),
        triangleAreaRatioRange=[float((area1[valid]/area0[valid]).min()),float((area1[valid]/area0[valid]).max())],
        scope='Fine evaluated outer crown, not coarse cage. Only neutral relief is planned; no key, source, material, globe or blink path is modified.',
        proposedDriverInterface={'controls':['R11 blink L','R11 blink R'],
            'perSideExpression':'strength * (1 - b) * (1 - b) * (1 + 2 * b)',
            'note':'Smoothstep fade of a separate terminal geometry modifier; default strength 0 until real current-source comparison. Half-blink must be checked for return of the thick ridge, not just endpoint safety.'},
        installationRequirements=['Verify actual neutral ROI coordinates/indices and external globe are unchanged from the observation before assigning weights.',
            'Keep actual socket/inner-eye regions and verified physical contact/wet line unchanged; the 0.8mm visible band alone does not identify them.',
            'Do not add until the blink owner has completed current closure geometry and confirmed modifier order.',
            'Use native same-topology final geometry and dynamic normals after the lid deformation; preserve all existing keys and fibre binding.',
            'Compare real neutral/half-blink/full-blink quarter and eye macro, then verify actual protected geometry and local intersections.'],
        sourceSaved=False,blenderExecuted=False,appearanceAccepted=False)


def run(observation_relative,output_relative):
    root=project_path();observation=(root/observation_relative).resolve();output=(root/output_relative).resolve()
    for path in (observation,output):path.relative_to((root/'work/reconstruction-r02').resolve())
    arrays_path=output.with_suffix('.npz')
    if output.exists() or arrays_path.exists():raise RuntimeError('Fresh independent plan paths required')
    meta=json.loads(observation.read_text(encoding='utf-8'));data_path=observation.with_suffix('.npz')
    if sha(data_path)!=meta['arraysSha256']:raise RuntimeError('Observation checksum mismatch')
    with np.load(data_path) as data:arrays,report=plan(meta,data)
    np.savez_compressed(arrays_path,**arrays)
    report.update(actualSource=meta['source'],actualSourceSha256=meta['sourceSha256'],
        observation=str(observation.relative_to(root)),observationSha256=sha(observation),
        observationArraysSha256=sha(data_path),helperSha256=sha(__file__),arraysSha256=sha(arrays_path))
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),helperSha256=report['helperSha256'],
        affected=report['affectedEvaluatedVertices'],minNormal=report['minimumTriangleNormalAlignment'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.observation,args.output)
