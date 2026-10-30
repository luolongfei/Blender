"""Uniform 225 mm donor placement from captured surface landmarks.

Creates an independent offline candidate package. No source edit or Blender
execution. Explicitly exposes old-rig/footwear incompatibilities instead of
squashing the donor to the old short shoes or certifying an inferred joint.
"""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

PIN={
 'foot-asset-observation-r11.json':'67d6afbc595cb757c533d66f53bd494e24b29f172356bda251fb1530b98b336c',
 'foot-asset-observation-r11.npz':'202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564',
 'foot-proportion-capture-r11.json':'c39e4147eb1962b845899a25a3716e59c5272554fa0b983a165cb82c5654ee5f',
 'foot-proportion-capture-r11.npz':'6481734f699921c8f64872a46a5d38cdae3aa3fefb26812f498b84d0f8b11ee3',
 'foot-landmark-atlas-r11.json':'b53652b1f892e23fc8323531cb6fc10fda43893dffa8327fbca4d3a43a4b150d'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def bounds(v):
    return {'minimum':v.min(0).tolist(),'maximum':v.max(0).tolist(),'dimensions':np.ptp(v,axis=0).tolist()}

def rings(raw,a):
    loops=a['asset_0_raw_loopVertex'];starts=a['asset_0_raw_polygonStart'];sizes=a['asset_0_raw_polygonSize']
    groups=a['asset_0_raw_attr_3'];edges={}
    for pi,(start,size) in enumerate(zip(starts,sizes)):
        f=loops[start:start+size]
        for x,y in zip(f,np.roll(f,-1)):edges.setdefault(tuple(sorted((int(x),int(y)))),[]).append(pi)
    remaining={e for e,ps in edges.items() if len(ps)==2 and 12 in groups[ps] and any(x in (8,9,10,11) for x in groups[ps])}
    parts=[]
    while remaining:
        e=remaining.pop();ids=set(e);changed=True
        while changed:
            touched={e for e in remaining if set(e)&ids};changed=bool(touched);remaining-=touched
            for e in touched:ids.update(e)
        parts.append(sorted(ids))
    if len(parts)!=5:raise RuntimeError('Expected five actual proximal toe-root loops')
    rows=[]
    for index,ids in enumerate(sorted(parts,key=lambda ids:raw[ids,0].mean())):
        row={'toeFromMedial':index+1,'rawVertexIds':ids,'centerLocal':raw[ids].mean(0).tolist(),
             'extentLocal':np.ptp(raw[ids],axis=0).tolist(),'meaning':'Actual source face-set 12 / proximal-toe boundary, not a verified internal MTP joint centre.'}
        rows.append(row)
    return rows

def build():
    p=project_path();folder=p/'work/reconstruction-r02';out=folder/'foot-225-placement-plan-r11.json';arrays=out.with_suffix('.npz')
    if out.exists() or arrays.exists():raise FileExistsError('Preserve existing candidate package')
    for name,digest in PIN.items():
        if sha(folder/name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    captured=json.loads((folder/'foot-proportion-capture-r11.json').read_text());atlas=json.loads((folder/'foot-landmark-atlas-r11.json').read_text())
    donor=json.loads((folder/'foot-asset-observation-r11.json').read_text())
    a=np.load(folder/'foot-asset-observation-r11.npz');b=np.load(folder/'foot-proportion-capture-r11.npz')
    v=a['asset_0_evaluated_positions'].astype(float);raw=a['asset_0_raw_positions'].astype(float)
    supports=np.asarray([row['centroidLocalMeters'] for row in atlas['plantarRegions'].values()])
    slope=np.linalg.solve(np.column_stack((supports[:,:2],np.ones(3))),supports[:,2])
    normal=np.array([-slope[0],-slope[1],1.]);normal/=np.linalg.norm(normal);up=np.array([0.,0.,1.])
    axis=np.cross(normal,up);K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    rotation=np.eye(3)+K+K@K/(1+normal@up)
    if abs(np.linalg.det(rotation)-1)>1e-12 or np.max(np.abs(rotation@normal-up))>1e-12:raise RuntimeError('Support orientation lost rigidity')
    scale=.225/np.ptp((v@rotation.T)[:,1]);rotated=v@rotation.T*scale
    malleoli=np.asarray([atlas['landmarks'][name]['evaluatedLocalMeters'] for name in ('medial-malleolar-surface','lateral-malleolar-surface')])
    center=malleoli.mean(0)@rotation.T*scale;toe_rings=rings(raw,a)
    packed={};sides={}
    objects={o['name']:o for o in captured['objects']}
    rig=np.asarray(captured['rigMatrixWorld'])
    bones=captured['bones']
    for side,prefix in (('L','left'),('R','right')):
        body=objects['Body_Complete_Shin_Foot_'+side];original=b[body['arrayPrefix']+'_restWorld'];floor=float(original[:,2].min())
        foot=bones[prefix+'Foot'];ankle=np.asarray(foot['headWorld'])
        mirror=np.diag([1 if side=='L' else -1,1,1]);linear=mirror@rotation*scale
        candidate=v@linear.T
        c=mirror@center;translate=np.array([ankle[0]-c[0],ankle[1]-c[1],floor-candidate[:,2].min()])
        matrix=np.eye(4);matrix[:3,:3]=linear;matrix[:3,3]=translate
        candidate+=translate;raw_new=raw@linear.T+translate
        packed['donor_'+side+'_matrixWorld']=matrix
        packed['donor_'+side+'_evaluatedWorld']=candidate;packed['donor_'+side+'_rawWorld']=raw_new
        packed['source_'+side+'_shinFootRaw']=b[body['arrayPrefix']+'_raw_positions']
        new_malleoli=malleoli@linear.T+translate;surface_axis_center=new_malleoli.mean(0)
        ring_world=[np.asarray(row['centerLocal'])@linear.T+translate for row in toe_rings]
        # The central two toe-root loops suggest where to inspect a shared
        # toe-flex zone. A single pivot is not automatically accepted.
        toe_zone=np.mean(ring_world[1:3],axis=0)
        cap=next(row for row in json.loads((folder/'foot-anatomy-measure-r11.json').read_text())['donor'].items() if row[0]=='shinEndCap')[1]
        cap_world=raw_new[np.asarray(cap['vertexIds'])]
        shoe=[o for o in captured['objects'] if o['name'].startswith('Shoe_') and re.search('_'+side+r'(?:_|$)',o['name'])]
        sock=[o for o in captured['objects'] if o['name'].startswith('Sock_') and re.search('_'+side+r'(?:_|$)',o['name'])]
        if len(shoe)!=37 or len(sock)!=3:raise RuntimeError('Footwear family changed')
        old_toe=float(original[:,1].min());new_toe=float(candidate[:,1].min())
        old_heel=float(original[:,1].max());new_heel=float(candidate[:,1].max())
        sides[side]={'matrixWorld':matrix.tolist(),'mirroredWindingMustBeReversed':side=='R','actualEvaluatedBounds':bounds(candidate),
            'sourceBareFootBounds':bounds(original),'targetBareFootMinimumZ':floor,
            'malleolarSurfaceAxisWorld':new_malleoli.tolist(),'provisionalAnkleAxisCenterWorld':surface_axis_center.tolist(),
            'oldAnkleRigHeadWorld':ankle.tolist(),'surfaceCenterMinusOldRigHead':(surface_axis_center-ankle).tolist(),
            'functionalJointCentreCertified':False,'provisionalCommonToeFlexZoneWorld':toe_zone.tolist(),
            'toeRootCentersWorld':[x.tolist() for x in ring_world],
            'upperDonorCapWorldBounds':bounds(cap_world),'capIsAnkleJoint':False,
            'oldShinUpperWorldZ':float(original[:,2].max()),
            'plannedBodyJoin':'Retain donor foot and ankle unchanged below the proximal shin join. Extend the donor upper-shin ring to the original shin top using actual old-shin cross sections; no overlapping old toes or hidden short-foot deformation.',
            'sharedLongitudinalFootwearMap':{'oldToeY':old_toe,'newToeY':new_toe,'oldHeelY':old_heel,'newHeelY':new_heel,
                'ankleY':float(ankle[1]),'forefootExtensionMeters':old_toe-new_toe,'heelExtensionMeters':new_heel-old_heel,
                'rule':'Use a common monotone map for the complete side family; outside toe/heel bounds retain decorative offsets. Fit X/Z from actual foot/sock cross sections before installing. No per-piece bounding-box resize.'},
            'shoeObjectNames':[o['name'] for o in shoe],'sockObjectNames':[o['name'] for o in sock],
            'shoeShellThicknessMeters':.0013000000035390258,'sockShellThicknessMeters':.0007999999797903001,
            'footwearFitPassed':False,'skinJoinPassed':False,'rigRetargetPassed':False}
    if not np.allclose(packed['donor_L_evaluatedWorld'][:,0],-packed['donor_R_evaluatedWorld'][:,0],atol=2e-7):
        raise RuntimeError('Bilateral placement symmetry failed')
    for side in ('L','R'):
        q=packed['donor_'+side+'_evaluatedWorld']
        if abs(np.ptp(q[:,1])-.225)>1e-12:raise RuntimeError('Uniform measured target length mismatch')
    np.savez_compressed(arrays,**packed)
    result={'schemaVersion':1,'mode':'offline-225mm-placement-candidate','targetLongitudinalSpanMeters':.225,
        'targetMeaning':'Root-selected initial artistic candidate; not a population, footwear or medical standard.',
        'uniformScale':float(scale),'supportAlignmentDegrees':float(np.degrees(np.arccos(normal@up))),
        'supportPlaneNativeZfromXY':slope.tolist(),'supportRotation':rotation.tolist(),
        'sourceCapture':captured['source'],'sourceCaptureSha256':captured['sourceSha256'],
        'bundle':donor['bundle'],'bundleSha256':donor['bundleSha256'],'inputSha256':PIN,'helperSha256':sha(__file__),
        'toeRootLoops':toe_rings,'sides':sides,'arrays':str(arrays.relative_to(p)),'arraysSha256':sha(arrays),
        'landmarkAtlasInspected':True,'sourceGeometryEdited':False,'executedBlender':False,'savedBlend':False,
        'appearanceAccepted':False,'deliveryReady':False,
        'nextActualChecks':['Current source feet, full footwear raw geometry/weights and leg rest matrices must match the capture.',
            'New donor + proximal shin continuation with preserved original anatomical sole and five toe forms.',
            'New actual shoe/sock inner surfaces, decoration attachment, ankle motion, toe flex and ground contact.',
            'Explicit ankle/toe rest-axis change with current action retargeting; do not call inherited old actions compatible.']}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'plan':str(out),'planSha256':sha(out),'arraysSha256':sha(arrays),'uniformScale':scale,'sides':{k:{'ankleDeltaZ':v['surfaceCenterMinusOldRigHead'][2],'toeExtension':v['sharedLongitudinalFootwearMap']['forefootExtensionMeters']} for k,v in sides.items()}}))

if __name__=='__main__':build()
