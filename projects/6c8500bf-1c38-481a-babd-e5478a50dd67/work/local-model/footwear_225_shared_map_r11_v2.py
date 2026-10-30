"""Offline common shoe-last map from actual old/new foot cross sections.

All 37 shoe pieces per side receive the same rest-space map. This produces
unapplied raw-coordinate candidates and diagnostics; Blender's resulting
subdivision/thickness, actual contact and visual shape remain unverified.
"""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def smoother(t):
    t=np.clip(t,0,1);return t*t*t*(t*(t*6-15)+10)

def hull2d(points):
    points=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[];upper=[]
    for point in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=1e-15:lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=1e-15:upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1]+upper[:-1])

def monotone_map(x,y):
    h=np.diff(x);delta=np.diff(y)/h;m=np.zeros(len(x))
    for i in range(1,len(x)-1):
        if delta[i-1]*delta[i]>0:
            w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1];m[i]=(w1+w2)/(w1/delta[i-1]+w2/delta[i])
    for end,hh,dd in ((0,h,delta),(-1,h[::-1],delta[::-1])):
        value=((2*hh[0]+hh[1])*dd[0]-hh[0]*dd[1])/(hh[0]+hh[1])
        m[end]=min(1.,3*dd[0])  # retain positive exterior offset scale at both shoe ends
    def evaluate(q):
        q=np.asarray(q,float);i=np.clip(np.searchsorted(x,q,side='right')-1,0,len(x)-2);t=(q-x[i])/h[i]
        value=(2*t**3-3*t**2+1)*y[i]+(t**3-2*t**2+t)*h[i]*m[i]+(-2*t**3+3*t**2)*y[i+1]+(t**3-t**2)*h[i]*m[i+1]
        low=q<x[0];high=q>x[-1];value[low]=y[0]+(q[low]-x[0])*m[0];value[high]=y[-1]+(q[high]-x[-1])*m[-1]
        return value
    return evaluate

def grid_sample(x,y,grid,points):
    q=np.asarray(points);i=np.clip(np.searchsorted(x,q[:,0],side='right')-1,0,len(x)-2);j=np.clip(np.searchsorted(y,q[:,1],side='right')-1,0,len(y)-2)
    tx=(q[:,0]-x[i])/(x[i+1]-x[i]);ty=(q[:,1]-y[j])/(y[j+1]-y[j])
    return (1-tx)*(1-ty)*grid[i,j]+tx*(1-ty)*grid[i+1,j]+(1-tx)*ty*grid[i,j+1]+tx*ty*grid[i+1,j+1]

def section_hull(vertices,triangles,y,top=.145):
    q=vertices[triangles];q=q[(q[:,:,1].min(1)<=y)&(q[:,:,1].max(1)>=y)];points=[]
    for triangle in q:
        hits=[]
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            if (a[1]<y)!=(b[1]<y):hits.append(a+(b-a)*(y-a[1])/(b[1]-a[1]))
        if len(hits)!=2:continue
        a,b=hits;ia=a[2]<=top;ib=b[2]<=top
        if ia:points.append(a[[0,2]])
        if ib:points.append(b[[0,2]])
        if ia!=ib:points.append((a+(b-a)*(top-a[2])/(b[2]-a[2]))[[0,2]])
    points=np.unique(np.asarray(points),axis=0)
    if len(points)<3:raise RuntimeError('No usable actual foot section')
    hull=hull2d(points);center=(hull.min(0)+hull.max(0))*.5
    return hull,center

def section_xy(vertices,triangles,z):
    q=vertices[triangles];q=q[(q[:,:,2].min(1)<=z)&(q[:,:,2].max(1)>=z)];points=[]
    for triangle in q:
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            if (a[2]<z)!=(b[2]<z):points.append((a+(b-a)*(z-a[2])/(b[2]-a[2]))[:2])
    hull=hull2d(points);center=(hull.min(0)+hull.max(0))*.5
    return hull,center

def radial(hull,center,angles):
    p=hull-center;edge=np.roll(hull,-1,axis=0)-hull;out=[]
    for theta in angles:
        u=np.array([np.cos(theta),np.sin(theta)]);det=u[0]*edge[:,1]-u[1]*edge[:,0];okay=abs(det)>1e-12
        r=np.full(len(p),np.nan);t=r.copy();r[okay]=(p[okay,0]*edge[okay,1]-p[okay,1]*edge[okay,0])/det[okay]
        t[okay]=(p[okay,0]*u[1]-p[okay,1]*u[0])/det[okay]
        hit=r[(r>0)&(t>=-1e-8)&(t<=1+1e-8)]
        if not len(hit):raise RuntimeError('Radial hull sample missing')
        out.append(hit.max())
    return np.asarray(out)

def make_field(old,old_tri,new,new_tri,old_upper,ankle_y):
    old_toe,old_heel=old[:,1].min(),old[:,1].max();new_toe,new_heel=new[:,1].min(),new[:,1].max()
    knots=np.array([old_upper[:,1].min(),old_toe,ankle_y,old_heel,old_upper[:,1].max()])
    targets=np.array([new_toe-.012,new_toe,ankle_y,new_heel,new_heel+.008])
    if np.any(np.diff(knots)<=0) or np.any(np.diff(targets)<=0):raise RuntimeError('Shoe/foot bounds cannot define monotone map')
    ymap=monotone_map(knots,targets)
    if np.any(np.diff(ymap(np.linspace(knots[0]-.01,knots[-1]+.01,10000)))<0):raise RuntimeError('Longitudinal map reversed')
    ys=np.linspace(old_toe+.0005,old_heel-.0005,161);angles=np.linspace(-np.pi,np.pi,129)
    centers0=[];centers1=[];radii0=[];radii1=[]
    for y in ys:
        yn=float(ymap(np.array([y]))[0]);h0,c0=section_hull(old,old_tri,y);h1,c1=section_hull(new,new_tri,yn)
        centers0.append(c0);centers1.append(c1);radii0.append(radial(h0,c0,angles));radii1.append(radial(h1,c1,angles))
    centers0=np.asarray(centers0);centers1=np.asarray(centers1);radii0=np.asarray(radii0);radii1=np.asarray(radii1)
    zs=np.linspace(.065,.115,81);axc0=[];axc1=[];axr0=[];axr1=[]
    for z in zs:
        h0,c0=section_xy(old,old_tri,z);h1,c1=section_xy(new,new_tri,z)
        axc0.append(c0);axc1.append(c1);axr0.append(radial(h0,c0,angles));axr1.append(radial(h1,c1,angles))
    axc0=np.asarray(axc0);axc1=np.asarray(axc1);axr0=np.asarray(axr0);axr1=np.asarray(axr1)
    def apply(points):
        q=np.asarray(points,float);y=q[:,1];sample_y=np.clip(y,ys[0],ys[-1]);c0=np.column_stack([np.interp(sample_y,ys,centers0[:,k]) for k in range(2)])
        c1=np.column_stack([np.interp(sample_y,ys,centers1[:,k]) for k in range(2)])
        d=q[:,[0,2]]-c0;distance=np.linalg.norm(d,axis=1);angle=np.arctan2(d[:,1],d[:,0]);samples=np.column_stack((sample_y,angle));r0=grid_sample(ys,angles,radii0,samples);r1=grid_sample(ys,angles,radii1,samples)
        radius=np.maximum(0,distance+r1-r0);direction=d/np.maximum(distance[:,None],1e-12);xz=c1+direction*radius[:,None]
        result=q.copy();result[:,0]=xz[:,0];result[:,1]=ymap(y)
        # The original outsole/heel-bottom thickness is preserved vertically.
        # This is a construction constraint, not a shoe-interior contact test.
        blend=smoother((q[:,2]-.025)/.025);result[:,2]=q[:,2]+(xz[:,1]-q[:,2])*blend
        sample_z=np.clip(q[:,2],zs[0],zs[-1]);ac0=np.column_stack([np.interp(sample_z,zs,axc0[:,k]) for k in range(2)]);ac1=np.column_stack([np.interp(sample_z,zs,axc1[:,k]) for k in range(2)])
        ad=q[:,:2]-ac0;ar=np.linalg.norm(ad,axis=1);angle=np.arctan2(ad[:,1],ad[:,0]);samples=np.column_stack((sample_z,angle))
        rr0=grid_sample(zs,angles,axr0,samples);rr1=grid_sample(zs,angles,axr1,samples)
        axial=q.copy();axial[:,:2]=ac1+ad/np.maximum(ar[:,None],1e-12)*np.maximum(0,ar+rr1-rr0)[:,None]
        collar=smoother((q[:,2]-.075)/.030)
        result=result*(1-collar[:,None])+axial*collar[:,None]
        return result
    table={'y':ys,'theta':angles,'oldCentersXZ':centers0,'newCentersXZ':centers1,'oldRadius':radii0,'newRadius':radii1,
           'longitudinalKnots':knots,'longitudinalTargets':targets,'collarZ':zs,'collarOldCentersXY':axc0,'collarNewCentersXY':axc1,'collarOldRadius':axr0,'collarNewRadius':axr1}
    return apply,table

def build():
    p=project_path();s=p/'work/reconstruction-r02';pp=s/'foot-225-placement-plan-r11.json'
    if sha(pp)!='52e7e5f3a4b45abc2e4857c2c05adb9b1b9b4ede95a4575cdeffe56d7407d4f2':raise RuntimeError('Placement changed')
    plan=json.loads(pp.read_text())
    for name,digest in plan['inputSha256'].items():
        if sha(s/name)!=digest:raise RuntimeError('Actual input changed')
    capture=json.loads((s/'foot-proportion-capture-r11.json').read_text());a=np.load(s/'foot-proportion-capture-r11.npz')
    donor=np.load(s/'foot-asset-observation-r11.npz');placed=np.load(p/plan['arrays'])
    if sha(p/plan['arrays'])!=plan['arraysSha256']:raise RuntimeError('Placed foot arrays changed')
    out=s/'footwear-225-shared-map-r11-v2.json';out_npz=out.with_suffix('.npz')
    if out.exists() or out_npz.exists():raise FileExistsError('Preserve existing fit candidate')
    objects={o['name']:o for o in capture['objects']};packed={};rows=[];side_reports={}
    for side in ('L','R'):
        oldrow=objects['Body_Complete_Shin_Foot_'+side];old=a[oldrow['arrayPrefix']+'_restWorld'];oldt=a[oldrow['arrayPrefix']+'_restTriangles']
        new=placed['donor_'+side+'_evaluatedWorld'];newt=donor['asset_0_evaluated_triangles'];upper=objects['Shoe_Leather_Upper_'+side]
        fn,table=make_field(old,oldt,new,newt,a[upper['arrayPrefix']+'_restWorld'],plan['sides'][side]['oldAnkleRigHeadWorld'][1])
        for k,v in table.items():packed[side+'_field_'+k]=v
        all_before=[];all_after=[]
        for row in capture['objects']:
            if not(row['name'].startswith('Shoe_') and re.search('_'+side+r'(?:_|$)',row['name'])):continue
            if row['type']!='MESH':raise RuntimeError('A curve component needs an explicit handle-preserving route')
            prefix=row['arrayPrefix'];raw=a[prefix+'_raw_positions'];m=np.asarray(row['matrixWorld']);world=raw@m[:3,:3].T+m[:3,3]
            target=fn(world);inverse=np.linalg.inv(m);local=target@inverse[:3,:3].T+inverse[:3,3]
            if not np.isfinite(local).all():raise RuntimeError('Nonfinite common-map coordinates')
            packed[prefix+'_sourceRaw']=raw;packed[prefix+'_targetRaw']=local
            old_floor=world[:,2]<=.025
            if not np.array_equal(target[old_floor,2],world[old_floor,2]):raise RuntimeError('Sole-bottom vertical construction changed')
            sample=world[np.linspace(0,len(world)-1,min(200,len(world)),dtype=int)];eps=1e-5;jac=[]
            for axis in range(3):
                step=np.zeros(3);step[axis]=eps;jac.append((fn(sample+step)-fn(sample-step))/(2*eps))
            determinants=np.linalg.det(np.stack(jac,axis=2))
            rows.append({'name':row['name'],'arrayPrefix':prefix,'side':side,'vertices':len(raw),'maximumDisplacementMeters':float(np.linalg.norm(target-world,axis=1).max()),
                'sampledJacobianMinimum':float(determinants.min()),'sampledNegativeJacobianCount':int((determinants<=0).sum()),
                'soleBottomVerticesVerticallyExact':int(old_floor.sum()),'originalUVMaterialsWeightsTopologyPreservedByPlan':True})
            all_before.append(world);all_after.append(target)
        oldall=np.concatenate(all_before);newall=np.concatenate(all_after)
        side_reports[side]={'objects':len(all_before),'beforeBounds':[oldall.min(0).tolist(),oldall.max(0).tolist()],
            'afterBounds':[newall.min(0).tolist(),newall.max(0).tolist()],
            'targetUpperExteriorLengthMeters':.245,'toeAndHeelExteriorAllowancesMeters':[.012,.008],
            'allowanceMeaning':'Provisional construction allowances selected for actual fitting; not a population size or certification of interior room.'}
    np.savez_compressed(out_npz,**packed)
    result={'schemaVersion':1,'mode':'offline-unapplied-common-225mm-shoe-map','placementPlanSha256':sha(pp),'helperSha256':sha(__file__),
        'sides':side_reports,'objects':rows,'arrays':str(out_npz.relative_to(p)),'arraysSha256':sha(out_npz),
        'sourceCaptureSha256':sha(s/'foot-proportion-capture-r11.json'),'executedBlender':False,'sourceGeometryEdited':False,
        'savedBlend':False,'appearanceAccepted':False,'technicalAcceptance':False,
        'rejectedPriorCandidate':'v1 spuriously lifted the back collar toward the entire Achilles cross-section and flattened exterior Y endpoints; never install v1.',
        'mapping':'Height-preserving actual XY ankle-section field for the collar, smoothly joined to one monotone Y and radial X/Z cross-section field for every shoe component on each side. Radial envelopes come from actual old/new foot slice hulls below Z=.145 m; they are shoe-last construction envelopes, not substituted anatomy.',
        'unverified':['Native subdivision/thickness and all continuous inside/outside contact.',
            'Final shape of laces, eyelet rings, welt and stitches; equal space mapping alone does not certify natural material strain.',
            'Sock continuation requires the final new calf shape; no sock source changed by this package.',
            'New ankle rest axes, inherited action retargeting and ground motion.']}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(out_npz),'objects':len(rows),'negativeJacobianSamples':sum(r['sampledNegativeJacobianCount'] for r in rows)}))

if __name__=='__main__':build()
