"""Fit a two-part sleeve capsule profile to measured mesh plane sections."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def distance(points,a,b):
    a=np.asarray(a);b=np.asarray(b);delta=b-a
    t=np.clip((points-a)@delta/(delta@delta),0,1)
    return np.linalg.norm(points-a-t[:,None]*delta,axis=1)

def run():
    p=project_path();root=p/'work/runtime-r06';source=root/'r09-sleeve-cross-sections.json';data=json.loads(source.read_text())
    base=json.loads((root/'r09-tail-fit-measured-candidate.json').read_text());capsules=[];fit=[]
    for row in data['rows']:
        # Bone-local X maps approximately to world forward, Y along the arm.
        # Use a changing axial profile, not an unsupported flattened XZ guess.
        sections=row['sections'];rows=[]
        for label,low,high,a,b in [('upper',-.02,.14,[0,0,.001],[0,.14,-.013]),('lower',.14,.22,[0,.14,-.013],[0,.22,-.011])]:
            points=np.array([[xz[0],s['y'],xz[1]] for s in sections if low-1e-7<=s['y']<=high+1e-7 for segment in s['planeSegmentsXZ'] for xz in segment])
            measured=distance(points,a,b);radius=float(np.ceil((measured.max()+.001)*1000)/1000)
            capsule={'bone':row['bone'],'offset':a,'tail':b,'radius':radius};capsules.append(capsule)
            rows.append({'region':label,'measuredYRange':[low,high],'samplePoints':len(points),'maxDistanceToAxisMeters':float(measured.max()),
                'fitRadiusMeters':radius,'minimumMeasuredSurfacePaddingMeters':float(radius-measured.max()),'capsule':capsule})
        points=np.array([[xz[0],s['y'],xz[1]] for s in sections for segment in s['planeSegmentsXZ'] for xz in segment])
        signed=np.min(np.stack([distance(points,r['capsule']['offset'],r['capsule']['tail'])-r['fitRadiusMeters'] for r in rows]),axis=0)
        fit.append({'side':row['side'],'regions':rows,'maximumMeasuredPointOutsideUnionMeters':float(signed.max()),'pointsOutside':int((signed>0).sum())})
    base['sleeveCapsules']=capsules
    cfg=root/'r09-tail-profile-candidate.json';cfg.open('x').write(json.dumps(base,indent=2))
    report={'sourceSha256':data['inputSha'],'sectionReport':str(source),'sectionReportSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'candidate':str(cfg),'candidateSha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),'fits':fit,
        'scope':'Both capsule radii derived from actual boundary endpoints at measured y planes; 1mm padding rounded up to 1mm. No claims about unsampled planes/arm poses.'}
    (root/'r09-sleeve-profile-fit.json').open('x').write(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':run()
