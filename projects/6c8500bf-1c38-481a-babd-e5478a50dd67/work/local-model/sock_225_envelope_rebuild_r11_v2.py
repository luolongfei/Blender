"""Ordinary sock foot chamber and calf rebuild from measured anatomical surfaces.

Replaces only the two failed old sock bodies, retaining their materials,
thickness/modifier order and original group definitions. Four cuff components
are preserved through the shared measured circumferential fit.
"""
import hashlib,json,runpy,sys
from pathlib import Path
from itertools import combinations
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def smooth(t):
    t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))

def triangles(l,s,n):
    return np.asarray([(l[a],l[a+k],l[a+k+1]) for a,z in zip(s,n) for k in range(1,z-1)],np.int32)

def convex_hull(points):
    """Incremental oriented 3D hull, with a fixed interior witness."""
    points=np.unique(np.asarray(points),axis=0);i0=int(np.argmin(points[:,0]));i1=int(np.argmax(np.linalg.norm(points-points[i0],axis=1)))
    direction=points[i1]-points[i0];i2=int(np.argmax(np.linalg.norm(np.cross(points-points[i0],direction),axis=1)))
    normal=np.cross(points[i1]-points[i0],points[i2]-points[i0]);i3=int(np.argmax(abs((points-points[i0])@normal)))
    initial=[i0,i1,i2,i3];inside=points[initial].mean(0)
    def orient(face):
        q=points[list(face)];return tuple(face) if np.dot(np.cross(q[1]-q[0],q[2]-q[0]),inside-q[0])<0 else (face[1],face[0],face[2])
    faces=np.asarray([orient(f) for f in combinations(initial,3)],np.int32)
    order=np.random.default_rng(225).permutation(len(points))
    for index in order:
        if index in initial:continue
        q=points[faces];n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);length=np.linalg.norm(n,axis=1);visible=np.einsum('ij,ij->i',n,points[index]-q[:,0])>length*1e-11
        if not visible.any():continue
        edges={}
        for face in faces[visible]:
            for a,b in zip(face,np.roll(face,-1)):
                key=tuple(sorted((int(a),int(b))))
                if key in edges:del edges[key]
                else:edges[key]=(int(a),int(b))
        new=[orient((a,b,int(index))) for a,b in edges.values()];faces=np.concatenate((faces[~visible],np.asarray(new,np.int32)))
    if not len(faces):raise RuntimeError('Empty convex foot chamber')
    used=np.unique(faces);remap=np.full(len(points),-1,np.int32);remap[used]=np.arange(len(used));return points[used],remap[faces]

def expanded_hull(v,t,offset):
    p=v[t];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n/=np.linalg.norm(n,axis=1)[:,None];vn=np.zeros_like(v)
    for k in range(3):np.add.at(vn,t[:,k],n)
    vn/=np.linalg.norm(vn,axis=1)[:,None];minimum=np.ones(len(v))
    for k in range(3):np.minimum.at(minimum,t[:,k],np.einsum('ij,ij->i',vn[t[:,k]],n))
    if minimum.min()<=.05:raise RuntimeError('Convex foot hull has unstable corner offsets')
    return convex_hull(v+vn*(offset/minimum)[:,None])

def main():
    p=project_path();s=p/'work/reconstruction-r02';cp=s/'foot-proportion-capture-r11.json';lp=s/'foot-leg-unified-candidate-r11-v2.json';previous=s/'sock_225_section_fit_r11_v4.json'
    pins={cp:'c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f',lp:'8977e539f122dc0cdf501b3a8f3fef2f72acab6ef94337aadc8229046a178a11',previous:'6ab1bfafa5be1ed61dc7b9d02391f8c2f50bef12b784926747006c02c8378fdf'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('Frozen sock input changed')
    cj=json.loads(cp.read_text());lj=json.loads(lp.read_text());pj=json.loads(previous.read_text())
    for j in (cj,lj,pj):
        if sha(p/j['arrays'])!=j['arraysSha256']:raise RuntimeError('Frozen input array changed')
    a=np.load(p/cj['arrays']);b=np.load(p/lj['arrays']);old=np.load(p/pj['arrays']);by={r['name']:r for r in cj['objects']};prior={r['name']:r for r in pj['objects']}
    utility=Path(__file__).with_name('footwear_225_shared_map_r11_v4.py')
    if sha(utility)!='d415a59b2597cdc14f9b6b42862b782b55c631e4e2a6ceede7ad1260ae711a93':raise RuntimeError('Frozen section helper changed')
    api=runpy.run_path(str(utility));packed={};rows=[];sides={}
    for side in ('L','R'):
        pl='legFoot_'+side;skin=b[pl+'_positions'];skintri=triangles(b[pl+'_loops'],b[pl+'_polygonStarts'],b[pl+'_polygonSizes'])
        hv,ht=convex_hull(skin[skin[:,2]<.141]);hv,ht=expanded_hull(hv,ht,.00105)
        packed[side+'_footEnvelopeWorld']=hv;packed[side+'_footEnvelopeTriangles']=ht
        floor=float(hv[:,2].min());zs=np.r_[np.linspace(floor+.00010,.061,81),np.linspace(.061,.141,41)[1:],np.linspace(.141,.2879999876022339,41)[1:]];theta=np.arange(160)*2*np.pi/160-np.pi;rings=[]
        for z in zs:
            if z<.135:
                hull,center=api['section_xy'](hv,ht,z)
                if z>.075:
                    ignored,skin_center=api['section_xy'](skin,skintri,z);blend_center=float(smooth((z-.075)/.025));center=center*(1-blend_center)+skin_center*blend_center
                rh=api['radial'](hull,center,theta)
            if z>=.100:
                true,cc=api['section_xy'](skin,skintri,z);rt=api['radial'](true,cc,theta)+.00130
                blend=float(smooth((z-.100)/.035))
                if blend<1:
                    # Same centre makes radial convex interpolation contain
                    # the true skin at every angle during the ankle transition.
                    rh=api['radial'](hull,cc,theta);radius=rh*(1-blend)+rt*blend;center=cc
                else:radius=rt;center=cc
            else:radius=rh
            knit=.00018*np.cos(40*theta)*smooth((z-.082)/.025)
            wrinkle=.00025*np.sin((z-.074)/.214*12*np.pi+1.2*np.sin(theta))*np.exp(-(((z-.074)/.214-.22)/.2)**2)*smooth((z-.100)/.025)
            radius=radius+knit+wrinkle
            rings.append(np.column_stack((center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),np.full(160,z))))
        rings=np.asarray(rings);bottom=hv[int(np.argmin(hv[:,2]))].copy();world=np.vstack((bottom,rings.reshape(-1,3)));faces=[];uv=[]
        # Cylindrical chart uses longitudinal surface arc length, not height,
        # so the foot sole receives useful texture space. Repack is still due.
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(rings,axis=0),axis=2).mean(1))];arc=.07+.93*arc/arc[-1]
        for j in range(160):faces.append([0,1+(j+1)%160,1+j]);uv.append([[.5,.025],[.5+.02*np.cos(theta[(j+1)%160]),.025+.02*np.sin(theta[(j+1)%160])],[.5+.02*np.cos(theta[j]),.025+.02*np.sin(theta[j])]])
        for ring in range(len(rings)-1):
            for j in range(160):
                faces.append([1+ring*160+j,1+ring*160+(j+1)%160,1+(ring+1)*160+(j+1)%160,1+(ring+1)*160+j]);uv.append([[j/160,arc[ring]],[(j+1)/160,arc[ring]],[(j+1)/160,arc[ring+1]],[j/160,arc[ring+1]]])
        name='Sock_Ribbed_'+side;row=by[name];pre=row['arrayPrefix'];m=np.asarray(row['matrixWorld']);inv=np.linalg.inv(m);local=world@inv[:3,:3].T+inv[:3,3]
        sizes=np.asarray([len(f) for f in faces],np.int32);starts=np.r_[0,np.cumsum(sizes[:-1])].astype(np.int32);loops=np.concatenate(faces).astype(np.int32);newtri=triangles(loops,starts,sizes);areas=np.linalg.norm(np.cross(world[newtri[:,1]]-world[newtri[:,0]],world[newtri[:,2]]-world[newtri[:,0]]),axis=1)*.5
        edges={};balance={}
        for face in faces:
            for aa,bb in zip(face,np.roll(face,-1)):
                key=tuple(sorted((int(aa),int(bb))));edges[key]=edges.get(key,0)+1;balance[key]=balance.get(key,0)+(1 if aa<bb else -1)
        boundary=[k for k,v in edges.items() if v==1]
        if len(boundary)!=160 or any(v>2 for v in edges.values()) or any(balance[k]!=0 for k,v in edges.items() if v==2):raise RuntimeError('Sock boundary/winding invalid')
        if any(min(key)<1+(len(rings)-1)*160 for key in boundary):raise RuntimeError('Sock opening occurs outside top cuff')
        capuv=np.asarray(uv[:160]);caparea=abs((capuv[:,1,0]-capuv[:,0,0])*(capuv[:,2,1]-capuv[:,0,1])-(capuv[:,1,1]-capuv[:,0,1])*(capuv[:,2,0]-capuv[:,0,0]))*.5
        if areas.min()<1e-14 or caparea.min()<1e-12:raise RuntimeError('Degenerate geometry or cap UV')
        names=[g[0] for g in sorted(row['groups'],key=lambda g:g[1])];expected=[('left' if side=='L' else 'right')+x for x in ('Foot','LowerLeg')]
        if names!=expected:raise RuntimeError('Original sock bone group order changed')
        calf=smooth((world[:,2]-.095)/.065);weights=np.column_stack((1-calf,calf));packed.update({pre+'_sourceRaw':a[pre+'_raw_positions'],pre+'_targetRaw':local,pre+'_targetWorld':world,pre+'_loops':loops,pre+'_polygonStarts':starts,pre+'_polygonSizes':sizes,pre+'_cornerUV':np.concatenate(uv),pre+'_weights':weights,pre+'_materialIndex':np.zeros(len(faces),np.int32)})
        rows.append({'name':name,'side':side,'arrayPrefix':pre,'mode':'replaceMesh','coordinateSpace':'targetRaw is original-object local, targetWorld is rest world','sourceVertices':len(a[pre+'_raw_positions']),'vertices':len(world),'faces':len(faces),'groupNames':names,'maximumWeightSumError':float(abs(weights.sum(1)-1).max()),'candidateBoundsMeters':[world.min(0).tolist(),world.max(0).tolist()],'minimumRawTriangleAreaM2':float(areas.min()),'boundaryEdges':len(boundary),'consistentWinding':True,'minimumFootCapUvTriangleArea':float(caparea.min()),'rawDegenerateTriangles':int((areas<1e-14).sum()),'boundary':'Exactly 160 top-cuff boundary edges; foot is closed. Original Solidify creates the real inner wall and cuff rim.','uvStatus':'New continuous cylindrical corner chart weighted by surface arc length, with explicit back seam and foot-cap chart; original material retained, final UV packing/texture scale review still due.'})
        for label in ('Sock_Cuff_Lower_','Sock_Cuff_Top_'):
            name=label+side;row=by[name];pre=row['arrayPrefix'];r=dict(prior[name]);r.update({'mode':'coordinates','groupNames':[g[0] for g in sorted(row['groups'],key=lambda g:g[1])]});rows.append(r)
            for key in ('sourceRaw','targetRaw','targetWorld'):packed[pre+'_'+key]=old[pre+'_'+key]
        sides[side]={'ordinaryFootChamber':'Convex envelope of actual complete toe/forefoot/heel points, offset outward by support-face-aware normals. Toe clefts and plantar arch are bridged by ordinary sock cloth; no five-toe sock topology.','footEnvelopeOffsetMeters':.00105,'calfNominalRadialMidplaneOffsetMeters':.0013,'actualSolidifyThicknessMeters':.0008,'knitAmplitudeMeters':.00018,'wrinkleAmplitudeMeters':.00025,'upperSockHeightMeters':float(zs[-1]),'ankleTransitionWorldZ':[.100,.135],'noAnatomyModified':True}
        print('SOCK_ENVELOPE_BUILT',side,len(world),len(faces),flush=True)
    out=s/'sock_225_envelope_rebuild_r11_v2.json';outa=out.with_suffix('.npz')
    if out.exists() or outa.exists():raise FileExistsError('Preserve candidate evidence')
    np.savez_compressed(outa,**packed);result={'schemaVersion':2,'mode':'offline-unapplied-two-body-sock-rebuild-four-cuff-fit','helperSha256':sha(__file__),'sourceCaptureSha256':sha(cp),'legCandidateSha256':sha(lp),'arrays':str(outa.relative_to(p)),'arraysSha256':sha(outa),'objects':rows,'sides':sides,'executedBlender':False,'savedBlend':False,'technicalAcceptance':False,'appearanceAccepted':False,'sourceGeometryEdited':False,'rejectedPriorCandidates':'All section-fit v1-v4 old foot-topology attempts are rejected and preserved; v4 has 2556 raw continuous skin intersections.','pending':['Actual continuous raw-surface skin check and image inspection','Native final leg Subdivision and sock Solidify surfaces/skin/shoe contact','New joint-axis, active ankle movement and final source UV/weights/material appearance']};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(outa)}))

if __name__=='__main__':main()
