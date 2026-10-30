"""Unapplied six-piece sock candidate from actual original sock and leg-foot sections.

The target is the convex horizontal envelope of an offset anatomical surface,
appropriate to cloth spanning toe clefts. Original topology, UVs, weights and
material data are untouched. This is not a native thickness/contact verdict.
"""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def smooth(t):
    t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))

def triangles(loops,starts,sizes):
    return np.asarray([(int(loops[s]),int(loops[s+k]),int(loops[s+k+1])) for s,n in zip(starts,sizes) for k in range(1,n-1)],np.int32)

def normals(v,t):
    n=np.zeros_like(v);p=v[t];f=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for k in range(3):np.add.at(n,t[:,k],f)
    return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-30)

def cubic(t,p0,p1,p2,p3):
    return .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)

def sample(grid,z,angle,zmin,zmax):
    nz,nt=grid.shape;u=np.clip((z-zmin)/(zmax-zmin)*(nz-1),0,nz-1);i=np.floor(u).astype(int);fz=u-i
    u=(angle+np.pi)/(2*np.pi)*nt;j=np.floor(u).astype(int);ft=u-j;rows=[]
    for dz in (-1,0,1,2):
        ii=np.clip(i+dz,0,nz-1);rows.append(cubic(ft,*(grid[ii,(j+dt)%nt] for dt in (-1,0,1,2))))
    return cubic(fz,*rows)

def sample1(grid,z,zmin,zmax):
    u=np.clip((z-zmin)/(zmax-zmin)*(len(grid)-1),0,len(grid)-1);i=np.floor(u).astype(int);t=u-i
    return cubic(t,*(grid[np.clip(i+k,0,len(grid)-1)] for k in (-1,0,1,2)))

def main():
    p=project_path();s=p/'work/reconstruction-r02';cap=s/'foot-proportion-capture-r11.json';leg=s/'foot-leg-unified-candidate-r11-v2.json'
    pins={cap:'c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f',leg:'8977e539f122dc0cdf501b3a8f3fef2f72acab6ef94337aadc8229046a178a11'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('Frozen actual input changed')
    cj=json.loads(cap.read_text());lj=json.loads(leg.read_text())
    for j in (cj,lj):
        if sha(p/j['arrays'])!=j['arraysSha256']:raise RuntimeError('Frozen arrays changed')
    a=np.load(p/cj['arrays']);b=np.load(p/lj['arrays']);by={r['name']:r for r in cj['objects']}
    helper=Path(__file__).with_name('footwear_225_shared_map_r11_v4.py')
    if sha(helper)!='d415a59b2597cdc14f9b6b42862b782b55c631e4e2a6ceede7ad1260ae711a93':raise RuntimeError('Frozen cross-section utility changed')
    api=runpy.run_path(str(helper));out=s/'sock_225_section_fit_r11_v2.json';outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve prior sock candidate')
    packed={};reports=[];sides={}
    for side in ('L','R'):
        row=by['Sock_Ribbed_'+side];pre=row['arrayPrefix'];m=np.asarray(row['matrixWorld']);sv=a[pre+'_raw_positions']@m[:3,:3].T+m[:3,3]
        st=triangles(a[pre+'_raw_loops'],a[pre+'_raw_polygonStarts'],a[pre+'_raw_polygonSizes'])
        preleg='legFoot_'+side;tv=b[preleg+'_positions'];tt=triangles(b[preleg+'_loops'],b[preleg+'_polygonStarts'],b[preleg+'_polygonSizes'])
        offset=.0009+.0004*smooth((tv[:,2]-.100)/.035);envelope=tv+normals(tv,tt)*offset[:,None]
        floor_delta=float(envelope[:,2].min()-sv[:,2].min());zmin=float(sv[:,2].min()+.00003);zmax=float(sv[:,2].max()-.00001)
        zs=np.linspace(zmin,zmax,321);angles=np.linspace(-np.pi,np.pi,256,endpoint=False)
        oc=[];nc=[];orr=[];nrr=[]
        sh=api['hull2d'](sv[sv[:,2]<.110,:2]);th=api['hull2d'](envelope[envelope[:,2]<.110,:2])
        sc=(sh.min(0)+sh.max(0))*.5;tc=(th.min(0)+th.max(0))*.5
        sr=api['radial'](sh,sc,angles);tr=api['radial'](th,tc,angles)
        for z in zs:
            nz=z+floor_delta*(1-float(smooth((z-.031)/.025)))
            blend=float(smooth((z-.040)/.015))
            if blend==0:c0,c1,r0,r1=sc,tc,sr.copy(),tr.copy()
            else:
                h0,c0=api['section_xy'](sv,st,z);h1,c1=api['section_xy'](envelope,tt,max(float(envelope[:,2].min()+.00002),nz))
                r0=api['radial'](h0,c0,angles);r1=api['radial'](h1,c1,angles)
                c0=sc*(1-blend)+c0*blend;c1=tc*(1-blend)+c1*blend
                r0=sr*(1-blend)+r0*blend;r1=tr*(1-blend)+r1*blend
            # Exclude the original high-frequency knit ribs from the reference
            # radius, so their signed residual survives common transport.
            f=np.fft.rfft(r0);f[13:]=0;r0=np.fft.irfft(f,n=len(r0))
            if r0.min()<=0 or r1.min()<=0:raise RuntimeError('Invalid actual sock section')
            oc.append(c0);nc.append(c1);orr.append(r0);nrr.append(r1)
        oc=np.asarray(oc);nc=np.asarray(nc);orr=np.asarray(orr);nrr=np.asarray(nrr)
        def apply(v):
            v=np.asarray(v);z=v[:,2];c0=np.column_stack([sample1(oc[:,k],z,zmin,zmax) for k in range(2)]);c1=np.column_stack([sample1(nc[:,k],z,zmin,zmax) for k in range(2)])
            d=v[:,:2]-c0;r=np.linalg.norm(d,axis=1);angle=np.arctan2(d[:,1],d[:,0]);r0=sample(orr,z,angle,zmin,zmax);r1=sample(nrr,z,angle,zmin,zmax)
            residual=r-r0
            # Positive interior radial scale transports cap interiors without
            # collapsing them onto a skin envelope. Outside offsets remain exact.
            radial=np.where(r<=r0,r*r1/r0,r1+residual)
            q=v.copy();q[:,:2]=c1+d/np.maximum(r[:,None],1e-20)*radial[:,None]
            q[:,2]=z+floor_delta*(1-smooth((z-.031)/.025))
            return q
        for key,value in {'z':zs,'theta':angles,'sourceCenterXY':oc,'targetCenterXY':nc,'sourceRadiusXY':orr,'targetRadiusXY':nrr,'floorDelta':np.asarray(floor_delta)}.items():packed[side+'_field_'+key]=value
        packed[side+'_targetEnvelopeWorld']=envelope;packed[side+'_targetTriangles']=tt
        for name in ('Sock_Ribbed_'+side,'Sock_Cuff_Lower_'+side,'Sock_Cuff_Top_'+side):
            row=by[name];pre=row['arrayPrefix'];m=np.asarray(row['matrixWorld']);raw=a[pre+'_raw_positions'];v=raw@m[:3,:3].T+m[:3,3];q=apply(v);inv=np.linalg.inv(m);local=q@inv[:3,:3].T+inv[:3,3]
            packed[pre+'_sourceRaw']=raw;packed[pre+'_targetRaw']=local;packed[pre+'_targetWorld']=q
            sample_ids=np.unique(np.linspace(0,len(v)-1,min(len(v),1500),dtype=int));probe=v[sample_ids];eps=1e-6;jac=[]
            for axis in range(3):
                step=np.zeros(3);step[axis]=eps;jac.append((apply(probe+step)-apply(probe-step))/(2*eps))
            det=np.linalg.det(np.stack(jac,axis=2));t=triangles(a[pre+'_raw_loops'],a[pre+'_raw_polygonStarts'],a[pre+'_raw_polygonSizes']);area=np.linalg.norm(np.cross(q[t[:,1]]-q[t[:,0]],q[t[:,2]]-q[t[:,0]]),axis=1)*.5
            reports.append({'name':name,'side':side,'arrayPrefix':pre,'vertices':len(raw),'rawBoundsMeters':[v.min(0).tolist(),v.max(0).tolist()],'candidateBoundsMeters':[q.min(0).tolist(),q.max(0).tolist()],
                'maximumDisplacementMeters':float(np.linalg.norm(q-v,axis=1).max()),'sampledMinimumJacobian':float(det.min()),'sampledNonpositiveJacobianCount':int((det<=0).sum()),'minimumRawTriangleAreaM2':float(area.min()),'rawDegenerateTriangles':int((area<1e-14).sum()),'cuffHeightExact':bool(np.array_equal(v[:,2],q[:,2])) if 'Cuff' in name else None})
        sides[side]={'floorMidplaneShiftMeters':floor_delta,'sourceRawSockUsedForField':by['Sock_Ribbed_'+side]['name'],'footMidplaneEnvelopeOffsetMeters':.0009,'calfMidplaneEnvelopeOffsetMeters':.0013,'actualThicknessMeters':.0008,
            'method':'Original real sock cross-sections to the horizontal convex envelope of normal-offset unified anatomical mesh. C1 cubic common field, same transform for all three pieces. Positive source residual preserved; interior radial coordinates retain positive proportional scale. Below .040 m both source and target use their actual whole-foot footprint, smoothly transitioning to actual horizontal sections by .055 m. The toe envelope bridges clefts without altering skin geometry.',
            'limits':'Target unified candidate is not yet natively subdivided. Offset/envelope values are construction choices, not anatomical standards. Native Solidify, source topology, local self-contact, skin separation, shoe separation and poses require real evaluation.'}
    np.savez_compressed(outa,**packed)
    result={'schemaVersion':1,'mode':'offline-unapplied-six-piece-sock-fit','helperSha256':sha(__file__),'crossSectionHelperSha256':sha(helper),'sourceCaptureSha256':sha(cap),'legCandidateSha256':sha(leg),'arrays':str(outa.relative_to(p)),'arraysSha256':sha(outa),'sides':sides,'objects':reports,'executedBlender':False,'savedBlend':False,'sourceGeometryEdited':False,'technicalAcceptance':False,'appearanceAccepted':False,
        'preservedByPlan':['All six original component names, meshes, topology, UVs, weights and materials','Original cuff heights and existing thickness','Common spatial deformation for sock body and cuff pieces'],
        'rejectedPriorCandidate':'v1 cap-level section centres jumped and tanh radial saturation caused near-degeneracy; preserved and never installed.',
        'pending':['Actual source guard and native Solidify evaluation against actual final unified leg/foot','Continuous triangle and thickness-aware shoe contact','Foot/ankle/knee motion and original-design image review']}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'output':str(out),'sha256':sha(out),'arraysSha256':sha(outa),'objects':reports}))

if __name__=='__main__':main()
