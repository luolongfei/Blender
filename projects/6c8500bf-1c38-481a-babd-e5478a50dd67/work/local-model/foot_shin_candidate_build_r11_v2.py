"""Build a closed offline foot/shin candidate from actual captured surfaces.

Retains the evaluated donor foot, five toes, nail faces and original corner UVs
below a proximal shin plane. Only the upper shin is continued into measured
current-shin sections. The output is an unapplied candidate, never acceptance.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def smooth(x):
    x=np.clip(x,0,1);return x*x*x*(x*(x*6-15)+10)

def section_segments(points,triangles,z):
    q=points[triangles];mask=(q[:,:,2].min(1)<z)&(q[:,:,2].max(1)>z);q=q[mask]
    segments=[]
    for face in q:
        hits=[]
        for a,b in zip(face,np.roll(face,-1,axis=0)):
            if (a[2]<z)!=(b[2]<z):hits.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
        if len(hits)==2 and np.linalg.norm(hits[0]-hits[1])>1e-12:segments.append(hits)
    if not segments:raise RuntimeError('Missing actual shin section')
    return np.asarray(segments)

def profile(segments,angles):
    q=segments.reshape(-1,2);center=(q.min(0)+q.max(0))*.5
    p=segments[:,0]-center;edge=segments[:,1]-segments[:,0]
    results=[]
    for angle in angles:
        ray=np.array([np.cos(angle),np.sin(angle)]);det=ray[0]*edge[:,1]-ray[1]*edge[:,0]
        okay=np.abs(det)>1e-12
        r=np.full(len(p),np.nan);t=np.full(len(p),np.nan)
        r[okay]=(p[okay,0]*edge[okay,1]-p[okay,1]*edge[okay,0])/det[okay]
        t[okay]=(p[okay,0]*ray[1]-p[okay,1]*ray[0])/det[okay]
        hit=r[(r>0)&(t>=-1e-9)&(t<=1+1e-9)]
        if not len(hit):raise RuntimeError('Shin section not star-shaped at requested radial sample')
        results.append(center+ray*hit.max())
    return np.asarray(results)

def clip_donor(vertices,faces,uvs,groups,plane):
    positions=[v.copy() for v in vertices];crossed={};newfaces=[];newuv=[];newgroups=[]
    for face,uv,group in zip(faces,uvs,groups):
        ids=[];cuv=[]
        for k in range(len(face)):
            i=face[k-1];j=face[k];a=vertices[i];b=vertices[j];ai=a[2]<=plane;bi=b[2]<=plane
            if ai!=bi:
                edge=tuple(sorted((i,j)));fraction=(plane-a[2])/(b[2]-a[2])
                if edge not in crossed:crossed[edge]=len(positions);positions.append(a+(b-a)*fraction)
                ids.append(crossed[edge]);cuv.append(uv[k-1]+(uv[k]-uv[k-1])*fraction)
            if bi:ids.append(int(j));cuv.append(uv[k])
        if len(ids)>=3:newfaces.append(ids);newuv.append(np.asarray(cuv));newgroups.append(int(group))
    edge_faces={};uv_by_edge={}
    for face,uv in zip(newfaces,newuv):
        for k,(i,j) in enumerate(zip(face,np.roll(face,-1))):
            edge_faces.setdefault(tuple(sorted((i,int(j)))),[]).append((i,int(j)))
            uv_by_edge[(i,int(j))]=(uv[k].copy(),uv[(k+1)%len(uv)].copy())
    boundary=[directions[0] for directions in edge_faces.values() if len(directions)==1]
    if not boundary:raise RuntimeError('Clip did not expose an upper join loop')
    next_id={i:j for i,j in boundary}
    if len(next_id)!=len(boundary):raise RuntimeError('Upper join is not a simple directed loop')
    ring=[boundary[0][0]]
    while next_id[ring[-1]]!=ring[0]:
        ring.append(next_id[ring[-1]])
        if len(ring)>len(boundary):raise RuntimeError('Open/disconnected proximal join')
    if len(ring)!=len(boundary):raise RuntimeError('Unexpected multiple boundaries after clipping')
    return positions,newfaces,newuv,newgroups,ring,uv_by_edge

def build():
    p=project_path();stage=p/'work/reconstruction-r02';plan_path=stage/'foot-225-placement-plan-r11.json'
    if sha(plan_path)!='52e7e5f3a4b45abc2e4857c2c05adb9b1b9b4ede95a4575cdeffe56d7407d4f2':raise RuntimeError('Placement changed')
    plan=json.loads(plan_path.read_text());placement=p/plan['arrays']
    if sha(placement)!=plan['arraysSha256']:raise RuntimeError('Placement arrays changed')
    for name,digest in plan['inputSha256'].items():
        if sha(stage/name)!=digest:raise RuntimeError('Captured input changed: '+name)
    out=stage/'foot-shin-candidate-r11-v2.json';out_npz=out.with_suffix('.npz')
    if out.exists() or out_npz.exists():raise FileExistsError('Preserve prior candidate')
    a=np.load(stage/'foot-asset-observation-r11.npz');b=np.load(stage/'foot-proportion-capture-r11.npz');q=np.load(placement)
    loops=a['asset_0_evaluated_loopVertex'];starts=a['asset_0_evaluated_polygonStart'];sizes=a['asset_0_evaluated_polygonSize'];uv=a['asset_0_evaluated_uv_0']
    faces=[loops[s:s+n].tolist() for s,n in zip(starts,sizes)];uvs=[uv[s:s+n].astype(float) for s,n in zip(starts,sizes)]
    groups=a['asset_0_evaluated_attr_5'];packed={};reports={}
    for side,index in (('L',0),('R',1)):
        donor=q['donor_'+side+'_evaluatedWorld'];sidefaces=faces if side=='L' else [list(reversed(f)) for f in faces]
        sideuvs=uvs if side=='L' else [f[::-1] for f in uvs]
        cut=.185;positions,polys,corners,sets,ring,edgeuv=clip_donor(donor,sidefaces,sideuvs,groups,cut)
        base=np.asarray(positions)[ring];center=(base[:,:2].min(0)+base[:,:2].max(0))*.5
        angles=np.arctan2(base[:,1]-center[1],base[:,0]-center[0])
        below=profile(section_segments(donor,a['asset_0_evaluated_triangles'],cut-.003),angles)
        slope=(base[:,:2]-below)/.003
        old=b['footwear_'+str(index)+'_restWorld'];oldtris=b['footwear_'+str(index)+'_restTriangles']
        ztop=float(old[:,2].max());levels=[cut+.002,cut+.006,.196,.204,.214,.226,.24,.26,.28,.30,ztop]
        end=profile(section_segments(old,oldtris,.302),angles)
        end_slope=(end-profile(section_segments(old,oldtris,.292),angles))/.010
        length=ztop-cut
        current=ring.copy();ring_records=[]
        for z in levels:
            t=(z-cut)/length
            h00=1-10*t**3+15*t**4-6*t**5;h10=t-6*t**3+8*t**4-3*t**5
            h01=10*t**3-15*t**4+6*t**5;h11=-4*t**3+7*t**4-3*t**5
            xy=h00*base[:,:2]+h10*length*slope+h01*end+h11*length*end_slope
            blend=float(h01)
            new=[]
            for point in xy:new.append(len(positions));positions.append(np.array([point[0],point[1],z]))
            for k,(ia,ib) in enumerate(zip(current,np.roll(current,-1))):
                na,nb=new[k],new[(k+1)%len(new)];oa,ob=ring[k],ring[(k+1)%len(ring)]
                ua,ub=edgeuv[(oa,ob)];prior_z=positions[ia][2]
                va=ua+np.array([0,(prior_z-cut)/.3]);vb=ub+np.array([0,(prior_z-cut)/.3])
                van=ua+np.array([0,(z-cut)/.3]);vbn=ub+np.array([0,(z-cut)/.3])
                polys.append([ib,ia,na,nb]);corners.append(np.asarray([vb,va,van,vbn]));sets.append(13)
            current=new;ring_records.append({'z':z,'blendToMeasuredOldShin':blend,'xyBounds':[xy.min(0).tolist(),xy.max(0).tolist()]})
        cap=list(reversed(current));capco=np.asarray(positions)[cap,:2]
        polys.append(cap);corners.append((capco-capco.min(0))/np.maximum(np.ptp(capco,axis=0),1e-12));sets.append(13)
        used=np.unique(np.concatenate([np.asarray(f) for f in polys]));remap=np.full(len(positions),-1,int);remap[used]=np.arange(len(used))
        coords=np.asarray(positions)[used];polys=[remap[np.asarray(f)] for f in polys]
        flat=np.concatenate(polys).astype(np.int32);counts=np.array([len(f) for f in polys],np.int32);offsets=np.r_[0,np.cumsum(counts[:-1])].astype(np.int32)
        edges={};directions={}
        for f in polys:
            for aa,bb in zip(f,np.roll(f,-1)):
                edge=tuple(sorted((int(aa),int(bb))));edges[edge]=edges.get(edge,0)+1;directions[edge]=directions.get(edge,0)+(1 if aa<bb else -1)
        if any(n!=2 for n in edges.values()) or any(n!=0 for n in directions.values()):raise RuntimeError('Candidate is not closed with consistent winding')
        original_kept=used[used<len(donor)];original_map=remap[original_kept]
        if not np.array_equal(coords[original_map],donor[original_kept]):raise RuntimeError('Original donor anatomy changed below cut')
        lower=smooth((coords[:,2]-.108)/.067)
        toe_center=np.asarray(plan['sides'][side]['provisionalCommonToeFlexZoneWorld'])
        toes=(1-lower)*smooth((toe_center[1]+.010-coords[:,1])/.035)
        weights=np.column_stack((lower,1-lower-toes,toes))
        if np.min(weights)<-1e-12 or np.max(abs(weights.sum(1)-1))>1e-12:raise RuntimeError('Candidate weights not normalized')
        prefix='foot_'+side
        packed.update({prefix+'_positions':coords,prefix+'_loops':flat,prefix+'_polygonStarts':offsets,prefix+'_polygonSizes':counts,
            prefix+'_uv':np.concatenate(corners),prefix+'_faceSets':np.asarray(sets,np.int32),prefix+'_materialIndex':(np.asarray(sets)==2).astype(np.int32),
            prefix+'_weights':weights,prefix+'_preservedDonorVertexIds':original_kept,prefix+'_preservedCandidateVertexIds':original_map})
        reports[side]={'vertices':len(coords),'polygons':len(polys),'boundaryEdges':0,'nonManifoldEdges':0,'consistentWinding':True,
            'proximalCutWorldZ':cut,'proximalBoundaryVertices':len(ring),'donorVerticesPreservedExactly':len(original_kept),
            'keptOriginalDonorRegion':'All actual donor evaluated surface below proximal shin plane; foot, ankle and five toe forms unchanged.',
            'extensionRings':ring_records,'continuationMethod':'One continuous quintic per radial direction, joining donor proximal position/tangent to the existing upper-shin overlap section/tangent; does not force the old narrow intermediate shin profile.','sourceShinMaximumZ':ztop,'materialSlots':['current anatomical body skin','current refined natural nail, with foot attributes still to author'],
            'weightColumns':['LowerLeg','Foot','Toes'],'weightsStatus':'Preliminary smooth candidate only; no actual deformation or motion validation.',
            'uvStatus':'Original donor corner UVs preserved below cut; proximal extension reuses boundary UV with explicit axial continuation.',
            'continuousSelfIntersectionChecked':False,'shoeSockFitPassed':False,'actualBlenderEvaluationPassed':False}
    np.savez_compressed(out_npz,**packed)
    result={'schemaVersion':1,'mode':'offline-unapplied-foot-shin-225mm-candidate','placementPlanSha256':sha(plan_path),
        'arrays':str(out_npz.relative_to(p)),'arraysSha256':sha(out_npz),'helperSha256':sha(__file__),'sides':reports,
        'executedBlender':False,'sourceGeometryEdited':False,'savedBlend':False,'appearanceAccepted':False,
        'rejectedPriorCandidate':'foot-shin-candidate-r11.json: local blend into old thin intermediate shin created a visible neck; kept as rejected offline evidence.',
        'limits':['No claim that the preliminary malleolar midpoint is a clinically verified joint centre.',
            'Current Foot/Toes rest axes and controls need explicit adaptation and fresh action/ground tests.',
            'Complete shoe/sock family fit and toe/nail material attributes are still pending.',
            'Closed topology and preserved donor positions do not certify continuous self-contact or appearance.']}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(out_npz),'sides':reports}))

if __name__=='__main__':build()
