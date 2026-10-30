"""Read-only discrete quality and actual contour comparison of one bridge test."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
W=P/'work/reconstruction-r02'

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30)

def metrics(old,new,tri):
    q0,q1=old[tri],new[tri]
    e01,e02=q0[:,1]-q0[:,0],q0[:,2]-q0[:,0]
    e11,e12=q1[:,1]-q1[:,0],q1[:,2]-q1[:,0]
    cross0,cross1=np.cross(e01,e02),np.cross(e11,e12)
    n0,n1=unit(cross0),unit(cross1)
    u0=unit(e01);v0=np.cross(n0,u0)
    r00=np.linalg.norm(e01,axis=1);r01=(e02*u0).sum(1);r11=(e02*v0).sum(1)
    f0=e11/r00[:,None];f1=(e12-f0*r01[:,None])/r11[:,None]
    sv=np.linalg.svd(np.stack([f0,f1],axis=2),compute_uv=False)
    return sv,n0,n1,np.linalg.norm(cross0,axis=1)*.5,np.linalg.norm(cross1,axis=1)*.5

def run():
    source=W/'adult-outer-eye-bridge-angle-evidence-r11.npz'
    target=W/'adult-outer-eye-bridge-angle-quality-r11'
    if target.with_suffix('.json').exists():raise RuntimeError('Preserve evidence')
    data=dict(np.load(source));meta=json.loads(source.with_suffix('.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['arraysSha256']:raise RuntimeError('Candidate hash')
    actual_path=P/'previews/reconstruction-r02/outer-eye-neutral-direction-preview-r11/neutral-geometry-preview.npz'
    actual=dict(np.load(actual_path));plan=json.loads(actual_path.with_suffix('.json').read_text())['plan']
    world,tri,ids=data['beforeWorld'],data['triangles'],data['originalVertexIds']
    old=world+data['parentDeltaWorld'];new=world+data['deltaWorld']
    sv,n0,n1,a0,a1=metrics(world,new,tri)
    svold,_,nold,_,_=metrics(world,old,tri)
    centroid=world[tri].mean(1)
    front=runpy.run_path(str(HERE/'adult_blink_inner_thickness_support_r11.py'))['FrontTriangleIndex']
    index=front(world,tri)
    changed=np.linalg.norm(new-old,axis=1)>1e-9
    faces=np.flatnonzero(changed[tri].any(1))
    first,_,_=index.front(centroid[faces]);cover=np.full(len(tri),np.nan);cover[faces]=centroid[faces,1]-first
    visible=(n0[:,1]<-.1)&(abs(cover)<.00002)
    report={'candidateSha256':meta['arraysSha256'],'actualInputSha256':meta['actualInputSha256'],
            'helperSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'allPhysicalContactPassed':False,'appearanceAccepted':False,'productionReady':False,
            'note':'Source tessellation only. Normal rotation is reported, not used as an absolute topological veto.',
            'originalCandidateReportScopeCorrection':'Its R sourceInsideGlobeXPreservedExactly included the other eye halfspace; corrected eye-local masks are recorded here. Candidate coordinates are unchanged.',
            'sides':{},'failedFaces':[]}
    payload={}
    def row(i):
        return {'triangleRow':int(i),'globalVertexIds':ids[tri[i]],'beforeAreaMm2':a0[i]*1e6,'afterAreaMm2':a1[i]*1e6,
                'principalStretch':sv[i],'parentPrincipalStretch':svold[i],
                'normalDot':float(n0[i]@n1[i]),'parentNormalDot':float(n0[i]@nold[i]),
                'frontCoverMeters':cover[i],'beforeNormal':n0[i],'centroidWorld':centroid[i],
                'originalEdgesMillimeters':np.linalg.norm(world[tri[i]]-np.roll(world[tri[i]],1,axis=0),axis=1)*1000}
    for i in (174822,174832,174835):report['failedFaces'].append(row(i))
    sl=runpy.run_path(str(HERE/'adult_outer_eye_visible_contour_r11.py'))
    before_slices,after_slices=sl['Slices'](old,tri),sl['Slices'](new,tri)
    for side in ('L','R'):
        md=plan['sides'][side];c=np.array(md['centerWorld']);sign=md['outwardSign']
        outside=sign*(world[:,0]-md['actualGlobeOuterX'])
        local_eye=(abs(world[:,0]-c[0])<.013)&(outside<=0.)
        face_scope=changed[tri].any(1)&(sign*centroid[:,0]>0.)
        select=np.flatnonzero(face_scope)
        vis=np.flatnonzero(face_scope&visible)
        globe=sl['Slices'](actual['eye_'+side+'_world'],actual['eye_'+side+'_triangles'])
        xs=md['actualGlobeOuterX']-sign*np.r_[np.arange(.0012,.000019,-.000025),.00001]
        rows=[]
        for x in xs:
            b=sl['intervals'](before_slices,globe,float(x));a=sl['intervals'](after_slices,globe,float(x))
            rows.append([x,b[0][0] if b else np.nan,b[-1][1] if b else np.nan,
                         a[0][0] if a else np.nan,a[-1][1] if a else np.nan,len(b),len(a)])
        rows=np.array(rows);payload[side+'_sampledVisibleContours']=rows
        diff=np.abs(rows[:,1:3]-rows[:,3:5])
        report['sides'][side]={
            'candidateInsideEyeGlobeXVerticesExactlyPreserved':bool(np.array_equal(new[local_eye],old[local_eye])),
            'changedTriangleCount':len(select),'frontVisibleChangedTriangles':len(vis),
            'worstStretch':row(select[np.argmax(sv[select,0])]),
            'worstFrontVisibleStretch':row(vis[np.argmax(sv[vis,0])]) if len(vis) else None,
            'minimumAreaFace':row(select[np.argmin(a1[select])]),
            'normalDotRange':[float(np.min(np.sum(n0[select]*n1[select],axis=1))),float(np.max(np.sum(n0[select]*n1[select],axis=1)))],
            'sampledVisibleBoundaryMaxDifferenceMeters':float(np.nanmax(diff)),
            'sampledVisibleColumnsChangedCount':int(np.any(diff>2e-6,axis=1).sum()),
            'sampledIntervalsCountChanged':int(np.any(rows[:,5]!=rows[:,6]))}
    contact=runpy.run_path(str(HERE/'adult_blink_midpath_contact_r11.py'))
    eyes={s:{'world':actual['eye_'+s+'_world'],'triangles':actual['eye_'+s+'_triangles']} for s in ('L','R')}
    layers=runpy.run_path(str(HERE/'adult_neutral_eye_outerends_radial_install_r11_v2.py'))['actual_layers']
    report['actualGlobeLayers'],report['pendingInnerGlobalVertexIds']=layers(
        {'world':world,'triangles':tri,'originalVertexIds':ids},{'world':new,'triangles':tri,'originalVertexIds':ids},eyes,contact)
    payload.update(principalStretch=sv,parentPrincipalStretch=svold,changedTriangleRows=faces,frontVisibleMask=visible,
                   normalDot=np.sum(n0*n1,axis=1),beforeArea=a0,afterArea=a1)
    np.savez_compressed(target.with_suffix('.npz'),**payload)
    report['arraysSha256']=hashlib.sha256(target.with_suffix('.npz').read_bytes()).hexdigest()
    target.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,default=serial),flush=True)

if __name__=='__main__':run()
