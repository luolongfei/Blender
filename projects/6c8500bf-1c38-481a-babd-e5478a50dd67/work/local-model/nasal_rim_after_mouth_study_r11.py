"""Rebuild the nasal direction on the actually evaluated selected corner rest.

The input is the successful static afterWorld capture; the render's separate
hand-state failure is retained explicitly. No saved Blend/production claim.
"""
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np

P=Path(__file__).resolve().parents[2];D=P/'work/reconstruction-r02'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))


def run():
    folder=P/'previews/reconstruction-r02/relaxed-mouth-corners-preview-r11'
    rpath=folder/'render-report.json';r=json.loads(rpath.read_text(encoding='utf8'))
    apath=P/r['arrays']
    if sha(apath)!=r['arraysSha256']:raise RuntimeError('Actual selected-mouth snapshot changed')
    checks=r['actualChecks']
    if not checks['frozenDependencyExact'] or not checks['topologyExact'] or checks['newLocalCrossings'] or not r['postRenderHeadExact'] or not r['sourceFileExact']:
        raise RuntimeError('Selected mouth actual surface evidence is incomplete')
    if any(not row['exact'] for row in checks['protected'].values()):raise RuntimeError('Mouth protected anatomy moved')
    a=np.load(apath);w=a['afterWorld'];tri=a['afterTriangles'];faces=a['afterFaces'];groups=a['afterGroups']
    evidence,contour_before=api('nasal_apparent_contour_r11')['measure'](w,tri)
    arrays,report=api('nasal_rim_direction_study_r11')['plan'](w,faces,groups,tri,evidence)
    nose_active=np.any(arrays['offsets']!=0,axis=1);mouth_active=np.any(a['fullWorldOffsets']!=0,axis=1)
    intersect=nose_active&mouth_active
    report.update(sourceScope='Pure nasal direction rebuilt on the actual native selected-mouth afterWorld snapshot; not merely composed theoretical offsets.',
        mouthSnapshotSourceSha256=r['sourceSha256'],mouthSnapshotReport=str(rpath.relative_to(P)),mouthSnapshotReportSha256=sha(rpath),
        mouthActualArrays=str(apath.relative_to(P)),mouthActualArraysSha256=sha(apath),mouthControl='R11 relaxed mouth corners',mouthControlValue=1.,
        warmSmileValue=0.,retainedRenderGuardErrors=r['guardErrors'],retainedRenderHeadExact=True,
        noseMouthActiveVertexOverlap=int(intersect.sum()),
        maximumNoseOffsetOnPriorMouthSupportMeters=float(np.linalg.norm(arrays['offsets'][intersect],axis=1).max()) if intersect.any() else 0.,
        contourBefore=contour_before,helperSha256=sha(__file__),
        requiredCurrentSourceGuard='The future saved selected-mouth source must reproduce actual nasal dependency coordinates and anatomical/topological identity exactly before preview.')
    dependency=nose_active.copy()
    for _ in range(2):dependency[np.unique(faces[dependency[faces].any(1)])]=True
    arrays['dependencyOriginalVertexIds']=np.flatnonzero(dependency).astype(np.int32)
    arrays['mouthOriginalWorldOffsets']=a['fullWorldOffsets']
    arrays['nasalOriginalUpperCapVertexIds']=evidence['upperCapVertexIds']
    out=D/'nasal-rim-after-mouth-study-r11.json';payload=out.with_suffix('.npz')
    np.savez_compressed(payload,**arrays);report.update(arrays=str(payload.relative_to(P)),arraysSha256=sha(payload))
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ['activeVertices','maximumDisplacementMeters','quality','newChangedTriangleCrossings','noseMouthActiveVertexOverlap','maximumNoseOffsetOnPriorMouthSupportMeters','arraysSha256']},indent=2))


if __name__=='__main__':run()
