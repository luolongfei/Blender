"""Read-only actual neutral radial-eye strain and surface-transition diagnosis."""
import hashlib,json,runpy
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
P=HERE.parent.parent
W=P/'work/reconstruction-r02'
SOURCE=W/'reference-adult-outer-eye-radial-r11-v2.outer-eye.npz'
OUT=W/'adult-outer-eye-actual-strain-r11'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30)
def js(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def smooth(x):
    u=np.clip(x,0,1);return u*u*(3-2*u)

def run():
    if OUT.with_suffix('.json').exists():raise RuntimeError('Preserve previous diagnosis')
    meta=json.loads(SOURCE.with_suffix('.json').read_text(encoding='utf8'))
    if sha(SOURCE)!=meta['arraysSha256']:raise RuntimeError('Actual capture checksum differs')
    with np.load(SOURCE) as a:
        ids=a['originalVertexIds'];tri=a['afterTriangles_blink_0'];global_tri=a['afterOriginalTriangleIds_blink_0']
        old=a['before_blink_0'][ids];new=a['after_blink_0'][ids]
        fields={s:a['field_'+s] for s in 'LR'}
        edges={s:{k:a['edge_'+s+'_'+k+'_headWorld'] for k in ('upper','lower')} for s in 'LR'}
    q0,q1=old[tri],new[tri];e01=q0[:,1]-q0[:,0];e02=q0[:,2]-q0[:,0]
    e11=q1[:,1]-q1[:,0];e12=q1[:,2]-q1[:,0]
    cross0=np.cross(e01,e02);cross1=np.cross(e11,e12)
    n0,n1=unit(cross0),unit(cross1)
    area0=np.linalg.norm(cross0,axis=1)*.5;area1=np.linalg.norm(cross1,axis=1)*.5
    # Isometric old tangent coordinates; singular values of the exact affine map.
    u0=unit(e01);v0=np.cross(n0,u0);r00=np.linalg.norm(e01,axis=1)
    r01=(e02*u0).sum(1);r11=(e02*v0).sum(1)
    f0=e11/r00[:,None];f1=(e12-f0*r01[:,None])/r11[:,None]
    F=np.stack([f0,f1],axis=2);sv=np.linalg.svd(F,compute_uv=False)
    rotation=np.degrees(np.arccos(np.clip((n0*n1).sum(1),-1,1)))
    centroid0=q0.mean(1);centroid1=q1.mean(1);delta=new-old
    changed=np.linalg.norm(delta,axis=1)>1e-9;affected=changed[tri].any(1)
    possible=np.flatnonzero(affected&(n0[:,1]<-.1))
    front=runpy.run_path(str(HERE/'adult_blink_inner_thickness_support_r11.py'))['FrontTriangleIndex']
    old_index,new_index=front(old,tri),front(new,tri)
    beforefront,_,_=old_index.front(centroid0[possible]);afterfront,_,_=new_index.front(centroid1[possible])
    cover0=np.full(len(tri),np.nan);cover1=cover0.copy()
    cover0[possible]=centroid0[possible,1]-beforefront;cover1[possible]=centroid1[possible,1]-afterfront
    visible=affected&(n0[:,1]<-.1)&(np.abs(cover0)<.00002)
    # Actual post-evaluation adjacency. Same connectivity is used for the old reference.
    ee=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    face_rows=np.tile(np.arange(len(tri)),3);order=np.lexsort((ee[:,1],ee[:,0]));ee=ee[order];face_rows=face_rows[order]
    adjacent=np.flatnonzero(np.all(ee[1:]==ee[:-1],axis=1));fa,fb=face_rows[adjacent],face_rows[adjacent+1]
    edge=ee[adjacent]
    angle0=np.degrees(np.arccos(np.clip((n0[fa]*n0[fb]).sum(1),-1,1)))
    angle1=np.degrees(np.arccos(np.clip((n1[fa]*n1[fb]).sum(1),-1,1)))
    old_len=np.linalg.norm(old[edge[:,1]]-old[edge[:,0]],axis=1)
    new_len=np.linalg.norm(new[edge[:,1]]-new[edge[:,0]],axis=1)
    def face_record(i):
        return dict(actualRoiTriangleRow=int(i),actualGlobalLoopTriangleId=int(global_tri[i]),globalVertexIds=ids[tri[i]],
                    centroidBeforeWorld=centroid0[i],centroidAfterWorld=centroid1[i],
                    principalStretchMax=float(sv[i,0]),principalStretchMin=float(sv[i,1]),
                    normalTurnDegrees=float(rotation[i]),oldNormal=n0[i],newNormal=n1[i],
                    beforeAreaMm2=float(area0[i]*1e6),afterAreaMm2=float(area1[i]*1e6),
                    oldEdgeLengthsMm=np.linalg.norm(q0[i]-np.roll(q0[i],1,axis=0),axis=1)*1000,
                    deltaVerticesMm=delta[tri[i]]*1000,frontCoverBeforeMm=float(cover0[i]*1000),frontCoverAfterMm=float(cover1[i]*1000))
    report=dict(actualArray=str(SOURCE.relative_to(P)),actualArraySha256=sha(SOURCE),sourceSha256=meta['sourceSha256'],
                actualSavedCandidateSha256=meta['outputSha256'],helperSha256=sha(__file__),
                method='Exact affine deformation per actual post-evaluation triangle, old isometric tangent basis, two principal stretches; actual edge-adjacent normal turn and front-axis visibility. Post-evaluation triangles applied to both states avoid old/new quad diagonal mismatches.',
                limits=['Front visibility is an orthographic negative-Y geometric test, not camera-space pixel attribution.',
                        'Normal rotation alone does not establish a topological flip. No new physical or appearance acceptance.',
                        'Reference fields are metadata only; measurements use saved actual neutral geometry.'],
                regions={},globalMaximumDisplacementMm=float(np.linalg.norm(delta,axis=1).max()*1000),geometryModified=False,blenderStarted=False)
    masks={};extra={}
    for s in 'LR':
        md=meta['plan']['sides'][s];sign=md['outwardSign'];center=np.array(md['centerWorld']);endpoint=md['endpointX']
        e=edges[s];bounds={}
        for label in ('upper','lower'):
            a=e[label];ok=np.isfinite(a).all(1);a=a[ok];sort=np.argsort(a[:,0]);a=a[sort]
            bounds[label]=np.interp(centroid0[:,0],a[:,0],a[:,2])
        out_x=sign*(centroid0[:,0]-endpoint)*1000
        upper=(centroid0[:,2]-bounds['upper'])*1000;lower=(bounds['lower']-centroid0[:,2])*1000
        domain=visible&(out_x>=-4)&(out_x<=4)
        regions={'upperCrown':domain&(upper>=-.25)&(upper<=6),
                 'lowerCheek':domain&(lower>=-.25)&(lower<=12),
                 'cornerBand':domain&(upper<-.25)&(lower<-.25)}
        for label,mask in regions.items():
            name=s+'_'+label;ii=np.flatnonzero(mask);masks[name]=mask
            if not len(ii):continue
            by_stretch=ii[np.argsort(sv[ii,0])[-5:][::-1]];by_turn=ii[np.argsort(rotation[ii])[-5:][::-1]]
            by_compress=ii[np.argsort(sv[ii,1])[:5]]
            adjacent_mask=(mask[fa]|mask[fb])&visible[fa]&visible[fb]
            aj=np.flatnonzero(adjacent_mask);rank=aj[np.argsort((angle1-angle0)[aj])[-5:][::-1]]
            er=[]
            for k in rank:
                er.append(dict(globalVertexIds=ids[edge[k]],actualRoiTriangleRows=[int(fa[k]),int(fb[k])],
                    oldEdgeLengthMm=float(old_len[k]*1000),newEdgeLengthMm=float(new_len[k]*1000),
                    oldDihedralDegrees=float(angle0[k]),newDihedralDegrees=float(angle1[k]),
                    meanCentroidSeparationBeforeMm=float(np.linalg.norm(centroid0[fa[k]]-centroid0[fb[k]])*1000),
                    midpointBeforeWorld=old[edge[k]].mean(0),midpointAfterWorld=new[edge[k]].mean(0)))
            report['regions'][name]=dict(triangleCount=len(ii),oldAreaMm2=float(area0[ii].sum()*1e6),
                originalGlobalVertexIds=ids[np.unique(tri[ii])],principalStretchQuantiles=np.quantile(sv[ii],[0,.05,.5,.95,1],axis=0),
                normalTurnQuantilesDegrees=np.quantile(rotation[ii],[0,.05,.5,.95,1]),
                maximumVertexDisplacementMm=float(np.linalg.norm(delta[np.unique(tri[ii])],axis=1).max()*1000),
                mostStretched=[face_record(i) for i in by_stretch],mostCompressed=[face_record(i) for i in by_compress],
                largestNormalTurns=[face_record(i) for i in by_turn],largestNewDihedrals=er)
        extra[s+'_centroidOuterXmm']=out_x;extra[s+'_upperEdgeDistanceMm']=upper;extra[s+'_lowerEdgeDistanceMm']=lower
        # The maximum-moved original visible outer-skin vertex and its support values.
        vi=np.unique(tri[domain]);v=vi[np.argmax(np.linalg.norm(delta[vi],axis=1))]
        f=fields[s][v];relative=old[v]-center;theta=np.arctan2(relative[2],-relative[1]);angle=theta-f[1]
        report.setdefault('visibleMaximumPoints',{})[s]=dict(globalVertexId=int(ids[v]),relativeToEyeMm=relative*1000,
            deltaMm=delta[v]*1000,xEnvelope=float(f[0]),thetaRadians=float(theta),thetaMid=float(f[1]),angularCoreHalf=float(f[2]),
            angularWeight=float(1-smooth((abs(angle)-f[2])/meta['plan']['angularFadeRadians'])))
    arrays=dict(originalVertexIds=ids,actualTriangles=tri,actualGlobalLoopTriangleIds=global_tri,beforeWorld=old,afterWorld=new,
                principalStretches=sv,normalTurnDegrees=rotation,visibleBeforeMask=visible,beforeFrontCover=cover0,afterFrontCover=cover1,
                sharedEdgeVertexIds=ids[edge],sharedEdgeAdjacentTriangleRows=np.stack([fa,fb],1),beforeDihedralDegrees=angle0,afterDihedralDegrees=angle1,
                **extra,**{k+'_mask':v for k,v in masks.items()})
    np.savez_compressed(OUT.with_suffix('.npz'),**arrays);report['arraysSha256']=sha(OUT.with_suffix('.npz'))
    OUT.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=js),encoding='utf8')
    print(json.dumps({'report':str(OUT.with_suffix('.json')),'sha256':sha(OUT.with_suffix('.json')),'arraysSha256':report['arraysSha256'],
        'regions':{k:{'count':r['triangleCount'],'stretch':r['principalStretchQuantiles'],'maxMove':r['maximumVertexDisplacementMm'],
                      'topDihedral':r['largestNewDihedrals'][:1]} for k,r in report['regions'].items()}},default=js),flush=True)
    return report,arrays

if __name__=='__main__':run()
