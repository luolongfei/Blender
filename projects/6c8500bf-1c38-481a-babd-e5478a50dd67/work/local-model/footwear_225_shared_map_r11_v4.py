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
    angles=np.linspace(-np.pi,np.pi,129);zs=np.linspace(.025,.115,181)
    footprint0=hull2d(old[old[:,2]<.145,:2]);footprint1=hull2d(new[new[:,2]<.145,:2])
    fc0=(footprint0.min(0)+footprint0.max(0))*.5;fc1=(footprint1.min(0)+footprint1.max(0))*.5
    fr0=radial(footprint0,fc0,angles);fr1=radial(footprint1,fc1,angles)
    centers0=[];centers1=[];radii0=[];radii1=[]
    for z in zs:
        mix=float(smoother((z-.040)/.012))
        if mix==0:c0,c1,r0,r1=fc0,fc1,fr0,fr1
        else:
            h0,c0=section_xy(old,old_tri,z);h1,c1=section_xy(new,new_tri,z)
            r0=radial(h0,c0,angles);r1=radial(h1,c1,angles)
            c0=fc0*(1-mix)+c0*mix;c1=fc1*(1-mix)+c1*mix
            r0=fr0*(1-mix)+r0*mix;r1=fr1*(1-mix)+r1*mix
        centers0.append(c0);centers1.append(c1);radii0.append(r0);radii1.append(r1)
    centers0=np.asarray(centers0);centers1=np.asarray(centers1);radii0=np.asarray(radii0);radii1=np.asarray(radii1)
    def apply(points):
        q=np.asarray(points,float);z=np.clip(q[:,2],zs[0],zs[-1])
        c0=np.column_stack([np.interp(z,zs,centers0[:,k]) for k in range(2)]);c1=np.column_stack([np.interp(z,zs,centers1[:,k]) for k in range(2)])
        d=q[:,:2]-c0;r=np.linalg.norm(d,axis=1);angle=np.arctan2(d[:,1],d[:,0]);sample=np.column_stack((z,angle))
        r0=grid_sample(zs,angles,radii0,sample);r1=grid_sample(zs,angles,radii1,sample)
        last_radius=r1+.003
        mapped=np.where(r<=r0,r*last_radius/r0,last_radius+(r-r0)*.45)
        result=q.copy();result[:,:2]=c1+d/np.maximum(r[:,None],1e-12)*mapped[:,None]
        return result
    table={'height':zs,'theta':angles,'oldCentersXY':centers0,'newCentersXY':centers1,'oldRadiusXY':radii0,'newRadiusXY':radii1,'exteriorOffsetScale':np.array(.45),'constructionRadialOffsetMeters':np.array(.003)}
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
    out=s/'footwear-225-shared-map-r11-v4.json';out_npz=out.with_suffix('.npz')
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
            'nominalExteriorLengthArtDirectionMeters':.245,'actualExteriorLengthRequiresNativeEvaluation':True,'oldRadialExteriorOffsetScale':.45,
            'allowanceMeaning':'Provisional common last envelope with reduced old excessive radial offsets; not a footwear size or certification of interior room.'}
    np.savez_compressed(out_npz,**packed)
    result={'schemaVersion':1,'mode':'offline-unapplied-common-225mm-shoe-map','placementPlanSha256':sha(pp),'helperSha256':sha(__file__),
        'sides':side_reports,'objects':rows,'arrays':str(out_npz.relative_to(p)),'arraysSha256':sha(out_npz),
        'sourceCaptureSha256':sha(s/'foot-proportion-capture-r11.json'),'executedBlender':False,'sourceGeometryEdited':False,
        'savedBlend':False,'appearanceAccepted':False,'technicalAcceptance':False,
        'rejectedPriorCandidate':'v1 lifted back collar/flattened Y endpoints; v2 reversed local volume; v3 raw mesh preview still exposed skin at heel/toe. All preserved, none accepted for installation.',
        'mapping':'One height-preserving radial XY map for every shoe component. New/old radii use actual horizontal foot-section hulls; near the sole they transition to the full anatomical footprint. The map preserves all Z values, maps interior radii positively to a provisional 3 mm offset last and scales old exterior offsets by .45. These are shoe-last envelopes, not modified anatomy.',
        'unverified':['Native subdivision/thickness and all continuous inside/outside contact.',
            'Final shape of laces, eyelet rings, welt and stitches; equal space mapping alone does not certify natural material strain.',
            'Sock continuation requires the final new calf shape; no sock source changed by this package.',
            'New ankle rest axes, inherited action retargeting and ground motion.']}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'report':str(out),'sha256':sha(out),'arraysSha256':sha(out_npz),'objects':len(rows),'negativeJacobianSamples':sum(r['sampledNegativeJacobianCount'] for r in rows)}))

if __name__=='__main__':build()
