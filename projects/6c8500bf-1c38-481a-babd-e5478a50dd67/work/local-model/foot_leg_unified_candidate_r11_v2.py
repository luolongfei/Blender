"""Offline native-loop continuation of real donor feet into retained legs.

Discards the donor artificial cap and its inward beveled termination.
Joins the .185 m actual shin section to the actual retained-leg 48-point row at world Z=.34 m.
Retains all original upper-leg raw coordinates, UVs and weights above that row.
No Blender execution, source mutation, rig editing or acceptance claim.
"""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def loop_for(faces):
    rows={}
    for f in faces:
        for a,b in zip(f,np.roll(f,-1)):rows.setdefault(tuple(sorted((int(a),int(b)))),[]).append((int(a),int(b)))
    edges=[directions[0] for directions in rows.values() if len(directions)==1];successor={a:b for a,b in edges}
    if len(successor)!=len(edges):raise RuntimeError('Boundary branches')
    ring=[edges[0][0]]
    while successor[ring[-1]]!=ring[0]:
        ring.append(successor[ring[-1]])
        if len(ring)>len(edges):raise RuntimeError('Boundary did not close')
    if len(ring)!=len(edges):raise RuntimeError('Multiple unexpected open boundaries')
    return ring

def area_xy(q):
    return np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))*.5

def samples(q,n):
    t=np.arange(n)*len(q)/n;i=np.floor(t).astype(int);u=t-i
    return q[i]*(1-u[:,None])+q[(i+1)%len(q)]*u[:,None]

def stitch(a,b):
    i=j=0;faces=[]
    while i<len(a) or j<len(b):
        da=(i+1)*len(b);db=(j+1)*len(a);aa=a[i%len(a)];bb=b[j%len(b)]
        if da==db:faces.append([a[(i+1)%len(a)],aa,bb,b[(j+1)%len(b)]]);i+=1;j+=1
        elif da<db:faces.append([a[(i+1)%len(a)],aa,bb]);i+=1
        else:faces.append([aa,bb,b[(j+1)%len(b)]]);j+=1
    return faces

def build():
    p=project_path();s=p/'work/reconstruction-r02';placement_path=s/'foot-225-placement-plan-r11.json';leg_path=s/'foot-leg-join-capture-r11.json'
    pins={placement_path:'52e7e5f3a4b45abc2e4857c2c05adb9b1b9b4ede95a4575cdeffe56d7407d4f2',
        leg_path:'adb90f20b6369b8228d0a19ddc634d0598ab8b151ea3e1ca856d505efa1f1578',
        s/'foot-asset-observation-r11.npz':'202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564'}
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path.name)
    plan=json.loads(placement_path.read_text());legs=json.loads(leg_path.read_text())
    for row in (plan,legs):
        if sha(p/row['arrays'])!=row['arraysSha256']:raise RuntimeError('Input array changed')
    a=np.load(s/'foot-asset-observation-r11.npz');b=np.load(p/legs['arrays']);c=np.load(p/plan['arrays'])
    old_helper=Path(__file__).with_name('foot_shin_candidate_build_r11.py')
    utility=runpy.run_path(str(old_helper));profile=utility['profile'];section=utility['section_segments'];smooth=utility['smooth']
    out=s/'foot-leg-unified-candidate-r11-v2.json';out_npz=out.with_suffix('.npz')
    if out.exists() or out_npz.exists():raise FileExistsError('Preserve prior candidate')
    raw_cap=json.loads((s/'foot-anatomy-measure-r11.json').read_text())['donor']['shinEndCap']['vertexIds']
    adj=[set() for _ in a['asset_0_evaluated_positions']]
    for i,j in a['asset_0_evaluated_edges']:adj[i].add(int(j));adj[j].add(int(i))
    edge_points=[]
    for i,j in zip(raw_cap,np.roll(raw_cap,-1)):
        common=adj[i]&adj[j]
        if len(common)!=1:raise RuntimeError('Native donor cap edge subdivision changed')
        edge_points.append(next(iter(common)))
    center=set.intersection(*(adj[i] for i in edge_points))
    if len(center)!=1:raise RuntimeError('Native donor cap centre ambiguous')
    cap_center=next(iter(center));fl=a['asset_0_evaluated_loopVertex'];fst=a['asset_0_evaluated_polygonStart'];fsz=a['asset_0_evaluated_polygonSize'];fuv=a['asset_0_evaluated_uv_0'];fgroup=a['asset_0_evaluated_attr_5']
    foot_faces=[fl[start:start+n].tolist() for start,n in zip(fst,fsz)];remove=[i for i,f in enumerate(foot_faces) if cap_center in f]
    if len(remove)!=32:raise RuntimeError('Expected exactly 32 proximal cap faces')
    kept_foot=[i for i in range(len(foot_faces)) if i not in set(remove)];packed={};reports={}
    for side in ('L','R'):
        donor=c['donor_'+side+'_evaluatedWorld'];source_uv=[fuv[start:start+n].astype(float) for start,n in zip(fst,fsz)]
        source_faces=foot_faces
        if side=='R':source_faces=[list(reversed(f)) for f in foot_faces];source_uv=[u[::-1] for u in source_uv]
        positions,polys,uv,sets,foot_ring,unused_edgeuv=utility['clip_donor'](donor,source_faces,source_uv,fgroup,.185)
        region=[0]*len(polys);materials=[1 if group==2 else 0 for group in sets]
        if len(foot_ring)!=66:raise RuntimeError('Actual .185 m proximal shin section changed')
        pre='leg_'+side;row=next(o for o in legs['legObjects'] if o['name'].startswith('Leg '+side+' '));matrix=np.asarray(row['matrixWorld'])
        native=b[pre+'_raw_positions'];world=native@matrix[:3,:3].T+matrix[:3,3];ll=b[pre+'_raw_loopVertex'];lst=b[pre+'_raw_polygonStart'];lsz=b[pre+'_raw_polygonSize'];luv=b[pre+'_raw_uv_0']
        all_leg=[ll[start:start+n].tolist() for start,n in zip(lst,lsz)];keep=[i for i,f in enumerate(all_leg) if world[f,2].min()>=.34-1e-6]
        local_leg=[all_leg[i] for i in keep];leg_ring=loop_for(local_leg)
        if len(leg_ring)!=48 or np.max(abs(world[leg_ring,2]-.34))>1e-6:raise RuntimeError('Actual .34 m native row changed')
        if np.sign(area_xy(np.asarray(positions)[foot_ring]))!=np.sign(area_xy(world[leg_ring])):leg_ring=list(reversed(leg_ring))
        foot0=np.asarray(positions)[foot_ring];leg0=world[leg_ring]
        fc=(foot0[:,:2].min(0)+foot0[:,:2].max(0))*.5;lc=(leg0[:,:2].min(0)+leg0[:,:2].max(0))*.5
        target_angle=np.arctan2(foot0[0,1]-fc[1],foot0[0,0]-fc[0]);leg_angles=np.arctan2(leg0[:,1]-lc[1],leg0[:,0]-lc[0])
        shift=int(np.argmin(abs(np.angle(np.exp(1j*(leg_angles-target_angle))))));leg_ring=np.roll(leg_ring,-shift).tolist();leg0=world[leg_ring]
        offset=len(positions);positions.extend(q.copy() for q in world)
        for i in keep:
            polys.append([offset+v for v in all_leg[i]]);uv.append(luv[lst[i]:lst[i]+lsz[i]].astype(float));region.append(2);materials.append(0)
        leg_top=[offset+i for i in leg_ring];n=96;base=samples(foot0,n);end=samples(leg0,n)
        angles=np.arctan2(base[:,1]-fc[1],base[:,0]-fc[0]);below=profile(section(donor,a['asset_0_evaluated_triangles'],.182),angles)
        m0=(base[:,:2]-below)/(base[:,2,None]-.182)
        eworld=b[pre+'_evaluated_positions'];em=np.asarray(row['evaluatedMatrixWorld']);eworld=eworld@em[:3,:3].T+em[:3,3];et=b[pre+'_evaluated_triangles']
        ecenter=(end[:,:2].min(0)+end[:,:2].max(0))*.5;eangles=np.arctan2(end[:,1]-ecenter[1],end[:,0]-ecenter[0])
        above=profile(section(eworld,et,.35),eangles);at=profile(section(eworld,et,.34),eangles);m1=(above-at)/.01
        current=foot_ring;bridge_count=0
        for step in range(1,17):
            if step==16:new=leg_top
            else:
                t=step/16;length=end[:,2]-base[:,2];h00=1-10*t**3+15*t**4-6*t**5;h10=t-6*t**3+8*t**4-3*t**5;h01=10*t**3-15*t**4+6*t**5;h11=-4*t**3+7*t**4-3*t**5
                xy=h00*base[:,:2]+h10*length[:,None]*m0+h01*end[:,:2]+h11*length[:,None]*m1;zz=base[:,2]+length*t
                new=[]
                for point,z in zip(xy,zz):new.append(len(positions));positions.append(np.array([point[0],point[1],z]))
            for face in stitch(current,new):
                polys.append(face);points=np.asarray(positions)[face];theta=np.unwrap(np.arctan2(points[:,1]-fc[1],points[:,0]-fc[0]));uv.append(np.column_stack(((theta+np.pi)/(2*np.pi),(points[:,2]-.19)/.15)));region.append(1);materials.append(0);bridge_count+=1
            current=new
        used=np.unique(np.concatenate([np.asarray(f) for f in polys]));remap=np.full(len(positions),-1,int);remap[used]=np.arange(len(used));co=np.asarray(positions)[used];faces=[remap[np.asarray(f)] for f in polys]
        edges={};balance={}
        for f in faces:
            for aa,bb in zip(f,np.roll(f,-1)):
                key=tuple(sorted((int(aa),int(bb))));edges[key]=edges.get(key,0)+1;balance[key]=balance.get(key,0)+(1 if aa<bb else -1)
        if any(v!=2 for v in edges.values()) or any(v!=0 for v in balance.values()):raise RuntimeError('Unified leg/foot did not close consistently')
        donor_kept=used[used<len(donor)];leg_kept=used[(used>=offset)&(used<offset+len(world))]-offset
        if not np.array_equal(co[remap[donor_kept]],donor[donor_kept]) or not np.array_equal(co[remap[leg_kept+offset]],world[leg_kept]):raise RuntimeError('Protected native anatomy changed')
        weights=np.zeros((len(positions),4));low=smooth((donor[:,2]-.115)/.070);toe_center=np.asarray(plan['sides'][side]['provisionalCommonToeFlexZoneWorld']);toes=(1-low)*smooth((toe_center[1]+.010-donor[:,1])/.035)
        weights[:len(donor),0]=low;weights[:len(donor),2]=1-low-toes;weights[:len(donor),3]=toes;weights[len(donor):offset,0]=1;weights[offset+len(world):,0]=1
        leg_weights=np.zeros((len(world),2));vi=b[pre+'_weight_vertex'];gi=b[pre+'_weight_group'];values=b[pre+'_weight_value']
        if [g['name'] for g in row['vertexGroups']]!=[('left' if side=='L' else 'right')+'LowerLeg',('left' if side=='L' else 'right')+'UpperLeg']:raise RuntimeError('Native leg group order changed')
        leg_weights[vi,gi]=values;weights[offset:offset+len(world),:2]=leg_weights;weights=weights[used]
        if np.max(abs(weights.sum(1)-1))>2e-6:raise RuntimeError('Candidate weights not complete')
        flat=np.concatenate(faces).astype(np.int32);sizes=np.asarray([len(f) for f in faces],np.int32);starts=np.r_[0,np.cumsum(sizes[:-1])].astype(np.int32)
        prefix='legFoot_'+side;packed.update({prefix+'_positions':co,prefix+'_loops':flat,prefix+'_polygonStarts':starts,prefix+'_polygonSizes':sizes,
            prefix+'_uvSource':np.concatenate(uv),prefix+'_faceRegion':np.asarray(region,np.int32),prefix+'_materialIndex':np.asarray(materials,np.int32),prefix+'_weights':weights,
            prefix+'_donorOriginalIds':donor_kept,prefix+'_donorCandidateIds':remap[donor_kept],prefix+'_legOriginalIds':leg_kept,prefix+'_legCandidateIds':remap[leg_kept+offset]})
        reports[side]={'vertices':len(co),'faces':len(faces),'donorArtificialCapAndBeveledTerminationRemoved':True,'donorProximalCutWorldZ':.185,'donorSectionBoundaryPoints':66,'retainedLegNativeRingPoints':48,
            'bridgeFaces':bridge_count,'boundaryEdges':0,'nonManifoldEdges':0,'consistentWinding':True,'donorVerticesUnchanged':len(donor_kept),'retainedLegRawVerticesUnchanged':len(leg_kept),
            'legRawWeightsExactAboveJoin':True,'legJoinWorldZ':.34,'oldLegRawSurfaceAboveJoinPreserved':True,
            'modifierPlan':'Retain the original Armature then Subdivision order. Further native evaluation is required: donor L1 is now base geometry, and new bridge topology changes local subdivision near the join.',
            'uvStatus':'Source donor and retained-leg corner UVs preserved in the candidate payload. Bridge has a separate cylindrical chart; packed final texture layout remains pending.',
            'continuousSelfContactChecked':False,'actualEvaluatedShapeChecked':False,'ankleWeightsAndRigValidated':False}
    np.savez_compressed(out_npz,**packed);result={'schemaVersion':1,'mode':'offline-unapplied-unified-leg-foot-candidate','helperSha256':sha(__file__),
        'placementPlanSha256':sha(placement_path),'legCaptureSha256':sha(leg_path),'legCaptureArraysSha256':legs['arraysSha256'],
        'source':legs['source'],'sourceSha256':legs['sourceSha256'],'arrays':str(out_npz.relative_to(p)),'arraysSha256':sha(out_npz),'sides':reports,
        'executedBlender':False,'savedBlend':False,'sourceGeometryEdited':False,'appearanceAccepted':False,'deliveryReady':False,
        'supersedesForFurtherFit':'The .306 m foot-shin-only candidates and direct artificial-cap continuation are retained as rejected experiments; this joins actual donor shin before its artificial cap bevel to the actual calf at .34 m.',
        'next':['Actual native evaluated foot/leg and knee/ankle poses, self-contact and source preservation checks.',
            'Explicit new ankle/toe rest axes and foot-controller adaptation; no old-action compatibility assumption.',
            'Complete footwear/sock fit against the new evaluated surface and fresh ground-contact motion.']}
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(out_npz),'sides':reports}))

if __name__=='__main__':build()
