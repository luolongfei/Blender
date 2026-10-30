"""One bounded actual-L3 external lower-lip / mental-slope continuity field.

Only anteroposterior coordinates change. Actual upper lip body/contact envelope,
oral face sets7/8, the rounded chin base and all other head points remain fixed.
The reference contour is measured from the current cage, not from an image.
"""
import numpy as np


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(x*(x*6.-15.)+10.)


def triangles(faces):
    f=np.asarray(faces,np.int32)
    return np.concatenate([f[:,[0,1,2]],f[:,[0,2,3]]])


def normals(world,faces):
    t=triangles(faces);p=world[t];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);out=np.zeros_like(world,float)
    for j in range(3):np.add.at(out,t[:,j],n)
    return out/np.maximum(np.linalg.norm(out,axis=1)[:,None],1e-15)


def state(data,prefix):
    return {key[len(prefix)+1:]:data[key] for key in data if key.startswith(prefix+'_')}


def lower_contour(cage,land):
    ids=np.asarray(land['mouth']['outerVermilionLoopCandidate']['sourceVertexIndices'])
    if len(ids)!=40 or ids[20]!=7983:raise RuntimeError('Current traced lower-vermilion ring required')
    q=cage[ids[14:27]];x=q[:,0]-cage[7972,0]
    # Millimeter-scaled polynomial is smooth, gently asymmetric and well
    # conditioned; it follows only the central lower external border.
    coeff=np.polynomial.polynomial.polyfit(x*1000,(q[:,2]-cage[7972,2])*1000,4)
    return coeff


def line_distance(points,line):
    a=np.asarray(line);v=np.roll(a,-1,axis=0)-a;length2=np.einsum('ij,ij->i',v,v);out=np.empty(len(points))
    for start in range(0,len(points),256):
        p=points[start:start+256];t=np.clip(np.einsum('nfi,fi->nf',p[:,None,:]-a[None,:,:],v)/np.maximum(length2,1e-20),0,1)
        q=a[None,:,:]+t[:,:,None]*v[None,:,:]
        out[start:start+len(p)]=np.sqrt(((p[:,None,:]-q)**2).sum(2).min(1))
    return out


def envelope(snapshot,land):
    w=snapshot['world'];f=snapshot['faces'];g=snapshot['groups'];c=snapshot['currentMixedCageWorld']
    chin=c[7972];x=w[:,0]-chin[0];z=w[:,2]-chin[2]
    coefficients=lower_contour(c,land)
    border_z=np.polynomial.polynomial.polyval(x*1000,coefficients)/1000+chin[2]
    depth=border_z-w[:,2];n=normals(w,f)
    protected=np.zeros(len(w),bool)
    protected[np.unique(f[np.isin(g,[2,3,4,5,7,8])])]=True
    # The upper half of the lower vermilion and the complete lip contact band
    # are above this conservative measured border-relative cutoff. Colour
    # support alone would incorrectly treat the outer lip ring as bare skin.
    protected|=(depth<=-.0015)|(snapshot['lipSupport']>=.60)|(z<=.0055)
    protected|=(abs(x)>=.032)|(w[:,1]>=-.080)|(n[:,1]>=-.10)
    opening_distance=line_distance(w,snapshot['openingCageWorld'])
    protected|=opening_distance<=.003
    e=smooth((depth+.0015)/.0025)
    e*=1-smooth((snapshot['lipSupport']-.22)/.38)
    e*=smooth((z-.0055)/.0040)
    e*=1-smooth((abs(x)-.019)/.013)
    e*=smooth((-n[:,1]-.10)/.55)
    e*=1-smooth((w[:,1]+.087)/.007)
    e*=smooth((opening_distance-.003)/.002)
    e[protected]=0.
    return e,protected,depth,coefficients


def quality(before,after,faces,active):
    t=triangles(faces);t=t[active[t].any(1)]
    p=before[t];q=after[t]
    a=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);b=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
    valid=la>1e-14
    cosine=np.sum(a[valid]*b[valid],axis=1)/(la[valid]*lb[valid]);ratios=lb[valid]/la[valid]
    e=np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
    lengths0=np.linalg.norm(before[e[:,1]]-before[e[:,0]],axis=1);lengths1=np.linalg.norm(after[e[:,1]]-after[e[:,0]],axis=1)
    ratios_edge=lengths1/np.maximum(lengths0,1e-15)
    return {'minimumNormalAlignment':float(cosine.min()),'triangleAreaRatioRange':[float(ratios.min()),float(ratios.max())],
        'edgeLengthRatioRange':[float(ratios_edge.min()),float(ratios_edge.max())],
        'nonpositiveNormalCount':int((cosine<=0).sum())}


def plan(data,land):
    a=state(data,'state_0');b=state(data,'state_1');w=a['world'];ids=a['originalVertexIds'];faces=a['faces']
    if len(np.unique(ids))!=len(ids):raise RuntimeError('Current actual ID uniqueness required')
    e,protected,depth,coeff=envelope(a,land)
    e1,p1,_,_=envelope(b,land);matching=np.searchsorted(b['originalVertexIds'],ids)
    present=(matching<len(b['originalVertexIds']))
    present[present]&=b['originalVertexIds'][matching[present]]==ids[present]
    # Static correspondence must remain outside the protected mouth in both
    # observed expressions. Their other ROI differences are not dependencies.
    other=np.zeros(len(ids));other[present]=e1[matching[present]]
    protected|=~present
    protected[present]|=p1[matching[present]]
    envelope_both=np.minimum(e,other);envelope_both[protected]=0.
    # Main visible correction: pull the outer underside back <0.8mm and
    # fill the steep plane below by <1.35mm. No direct chin-pad retraction.
    base=.00085*np.exp(-((depth-.00015)/.00185)**2)-.00135*np.exp(-((depth-.0052)/.0034)**2)
    lateral=np.exp(-((w[:,0]-a['currentMixedCageWorld'][7972,0])/.025)**4)
    amount=base*envelope_both*lateral
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1),axis=0)
    ends=edges.ravel();other_ends=edges[:,::-1].ravel();degree=np.maximum(np.bincount(ends,minlength=len(ids)),1)
    for _ in range(6):
        average=np.bincount(ends,weights=amount[other_ends],minlength=len(ids))/degree
        amount=(.65*amount+.35*average)
        # Preserve the C2 boundary envelope without diffusing into contact.
        amount=np.clip(amount,-.00135*envelope_both,.00085*envelope_both)
        amount[protected]=0.
    amount[abs(amount)<1e-9]=0.
    offset=np.zeros_like(w);offset[:,1]=amount
    active=amount!=0
    if not 300<int(active.sum())<15000:raise RuntimeError('Unexpected actual lower-face support count '+str(active.sum()))
    if np.any(offset[protected]):raise RuntimeError('Protected mouth/neck/chin-base entered field')
    checks=[]
    for s in (a,b):
        lookup=np.searchsorted(ids,s['originalVertexIds']);found=lookup<len(ids);found[found]&=ids[lookup[found]]==s['originalVertexIds'][found]
        delta=np.zeros_like(s['world']);delta[found]=offset[lookup[found]]
        check=quality(s['world'],s['world']+delta,s['faces'],np.linalg.norm(delta,axis=1)>0)
        if check['nonpositiveNormalCount'] or check['minimumNormalAlignment']<.75 or check['triangleAreaRatioRange'][0]<.55 or check['triangleAreaRatioRange'][1]>1.7:
            raise RuntimeError('Lower-face candidate folds or strains: '+str(check))
        checks.append(check)
    arrays={'originalVertexIds':ids,'neutralWorld':w,'faces':faces,'worldOffsets':offset.astype(np.float32),
        'protected':protected,'envelope':envelope_both.astype(np.float32),'depthBelowLowerExternalBorder':depth.astype(np.float32)}
    report={'affectedActualVertices':int(active.sum()),'minimumAnteriorOffsetMeters':float(amount.min()),'maximumPosteriorOffsetMeters':float(amount.max()),
        'lowerContourPolynomialMillimeters':coeff.tolist(),'chinBaseProtectedHeightMeters':.0055,
        'upperLowerLipProtectionOffsetAboveOuterBorderMeters':.0015,'zCoordinatesUnchanged':True,'xCoordinatesUnchanged':True,
        'directChinPadRetractionMeters':0.,'sourceStates':'Actual current neutral and warm closed smile1; support uses their protected-domain union.',
        'offlineStateChecks':checks,'appearanceAccepted':False,
        'scope':'One continuous external lower-lip underside to anterior mental slope field. Preserves lip contact/upper lip body, actual inner oral groups7/8, eye/ear, chin base, neck and all other head vertices.'}
    return arrays,report
