"""Measured reversible relaxed lip contact; pure NumPy, no Blender execution."""
from pathlib import Path
import runpy
import numpy as np

def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def smooth(t):
 t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))
def rim_edges(faces,groups,aa,bb):
 e=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]);g=np.tile(groups,4);e=np.sort(e,axis=1)
 u,inv=np.unique(e,axis=0,return_inverse=True);a=np.zeros(len(u),bool);b=a.copy();a[inv[g==aa]]=True;b[inv[g==bb]]=True
 return u[a&b]
def cut_edges(w,edges,x):
 q=w[edges];ok=(q[:,0,0]-x)*(q[:,1,0]-x)<=0;ok&=abs(q[:,1,0]-q[:,0,0])>1e-12;q=q[ok]
 return q[:,0]+(q[:,1]-q[:,0])*((x-q[:,0,0])/(q[:,1,0]-q[:,0,0]))[:,None]
def cut_triangles(w,faces,x):
 q=w[faces];q=q[(q[:,:,0].min(1)<=x)&(q[:,:,0].max(1)>=x)];t=np.concatenate([q[:,[0,1,2]],q[:,[0,2,3]]]);out=[]
 for tri in t:
  hits=[]
  for a,b in zip(tri,np.roll(tri,-1,axis=0)):
   if (a[0]-x)*(b[0]-x)<0:hits.append((a+(b-a)*((x-a[0])/(b[0]-a[0])))[[1,2]])
  if len(hits)==2:out.append(hits)
 return np.asarray(out).reshape(-1,2,2)
def sample_floor(seg,ys,zseed):
 a=seg[:,0];dd=seg[:,1]-a;out=[]
 for y in ys:
  ok=(seg[:,:,0].min(1)<=y)&(seg[:,:,0].max(1)>=y)&(abs(dd[:,0])>1e-12)
  zz=a[ok,1]+(y-a[ok,0])*dd[ok,1]/dd[ok,0];zz=zz[(zz>zseed-.004)&(zz<zseed+.008)]
  out.append(zz.min() if len(zz) else np.nan)
 return np.asarray(out)
def grid_sample(xs,ys,grid,points):
 x=points[:,0];y=points[:,1];gx=np.clip((x-xs[0])/(xs[1]-xs[0]),0,len(xs)-1-1e-10);gy=np.clip((y-ys[0])/(ys[1]-ys[0]),0,len(ys)-1-1e-10)
 ix=gx.astype(int);iy=gy.astype(int);fx=gx-ix;fy=gy-iy
 return (1-fx)*((1-fy)*grid[ix,iy]+fy*grid[ix,iy+1])+fx*((1-fy)*grid[ix+1,iy]+fy*grid[ix+1,iy+1])
def plan(world,faces,groups,cage,land,amount=1.,angle=.38):
 w=np.asarray(world,float);rim=rim_edges(faces,groups,7,17)
 xs=np.linspace(-.020,.020,161);ys=np.linspace(-.119,-.090,291);fields=[];rims=[]
 for x in xs:
  rr=cut_edges(w,rim,x)
  if not len(rr):raise RuntimeError('Missing actual upper-vermilion/inner interface')
  r=rr[np.argmin(rr[:,1])];seg=cut_triangles(w,faces[np.isin(groups,[7,17])],x);u=sample_floor(seg,ys,r[2]);valid=np.isfinite(u)
  if valid.sum()<30:raise RuntimeError('Missing anterior upper-lip section')
  u=np.interp(ys,ys[valid],u[valid]);target=r[2]-.00012
  # Lift the existing opposing contact valley together; the upper border and
  # the lower-vermilion base remain fixed. This is not two lips squeezing.
  lift=np.maximum(target-u,0.);lift*=smooth(lift/.00025)
  lift*=smooth((ys-r[1])/.0007)*(1-smooth((ys-r[1]-.006)/.004))
  fields.append(lift);rims.append(r)
 grid=np.array(fields);rims=np.array(rims)
 kernel=np.exp(-.5*(np.arange(-6,7)/2.5)**2);kernel/=kernel.sum();pad=np.pad(grid,((6,6),(0,0)),mode='edge');grid=sum(k*pad[i:i+len(xs)] for i,k in enumerate(kernel))
 lift=grid_sample(xs,ys,grid,w)
 midids=np.asarray(land['mouth']['vermilionMidLoop']['sourceVertexIndices']);lp=cage[midids[13:28]];order=np.argsort(lp[:,0]);lp=lp[order]
 base=np.interp(w[:,0],lp[:,0],lp[:,2]);rimz=np.interp(w[:,0],xs,rims[:,2]);rimy=np.interp(w[:,0],xs,rims[:,1])
 upper=np.zeros(len(w),bool);lower=upper.copy()
 upper[np.unique(faces[np.isin(groups,[7,17])])]=True;lower[np.unique(faces[np.isin(groups,[8,22])])]=True
 both=upper&lower;upper[both]=False;lower[both]=False
 lateral=1-smooth((abs(w[:,0])-.014)/.006)
 support=smooth((w[:,2]-base)/.0038)*(1-smooth((w[:,2]-rimz-.0006)/.0032))
 support*=lateral*smooth((w[:,1]-rimy+.001)/.001)*(1-smooth((w[:,1]-rimy-.008)/.003))
 offset=np.zeros_like(w);offset[upper,2]=lift[upper]*support[upper]*amount
 py=np.interp(w[:,0],lp[:,0],lp[:,1]);pz=base
 roll=smooth((w[:,2]-pz)/.0035)*(1-smooth((w[:,1]-rimy-.006)/.007))*lateral
 theta=angle*roll*amount
 dy=w[:,1]-py;dz=w[:,2]-pz
 offset[lower,1]=(np.cos(theta)*dy-np.sin(theta)*dz-dy)[lower]
 offset[lower,2]=(np.sin(theta)*dy+np.cos(theta)*dz-dz)[lower]
 offset[np.linalg.norm(offset,axis=1)<1e-9]=0.
 return {'worldOffsets':offset,'profileXs':xs,'profileYs':ys,'upperFloorLift':grid,'upperInterface':rims,'support':support,'lowerBaseZ':base,'lowerPivotY':py,'lowerRollWeight':roll,'lowerMembership':lower,'upperMembership':upper}

def rotate_lower(w,field,angle):
 angle=np.asarray(angle)*field['lowerRollWeight'];dy=w[:,1]-field['lowerPivotY'];dz=w[:,2]-field['lowerBaseZ'];off=np.zeros_like(w);lo=field['lowerMembership']
 off[lo,1]=(np.cos(angle)*dy-np.sin(angle)*dz-dy)[lo];off[lo,2]=(np.sin(angle)*dy+np.cos(angle)*dz-dz)[lo]
 return off

def section_recipe(w,faces,x):
 q=w[faces];rows=np.flatnonzero((q[:,:,0].min(1)<=x)&(q[:,:,0].max(1)>=x));tri=np.concatenate([faces[rows][:,[0,1,2]],faces[rows][:,[0,2,3]]]);edges=[];weights=[]
 for t in tri:
  hit=[];tt=[]
  for a,b in zip(t,np.roll(t,-1)):
   if (w[a,0]-x)*(w[b,0]-x)<0:hit.append([a,b]);tt.append((x-w[a,0])/(w[b,0]-w[a,0]))
  if len(hit)==2:edges.append(hit);weights.append(tt)
 return np.asarray(edges,np.int32),np.asarray(weights,float)
def section_from(w,recipe):
 e,t=recipe;return w[e[:,:,0],1:]*(1-t[:,:,None])+w[e[:,:,1],1:]*t[:,:,None]
def z_envelope(ss,ys,lower,zlo,zhi):
 a=ss[:,0];d=ss[:,1]-a
 safe=abs(d[:,0])>1e-12;a=a[safe];d=d[safe];ss=ss[safe]
 mask=(ys[:,None]>=ss[:,:,0].min(1))&(ys[:,None]<=ss[:,:,0].max(1));z=a[None,:,1]+(ys[:,None]-a[None,:,0])*d[None,:,1]/d[None,:,0]
 mask&=(z>zlo)&(z<zhi);z=np.where(mask,z,-np.inf if lower else np.inf)
 return z.max(1) if lower else z.min(1)
def bounded_angles(states,faces,groups,field):
 w=states[0];xs=field['profileXs'];upperdelta=field['worldOffsets'].copy();upperdelta[field['lowerMembership']]=0.;safe=[];rows=[]
 for x in xs:
  cases=[]
  for wi in states:
   recipe_u=section_recipe(wi,faces[np.isin(groups,[7,17])],x);recipe_l=section_recipe(wi,faces[np.isin(groups,[8,22])],x)
   yz=section_from(wi+upperdelta,recipe_u);ry=np.interp(x,xs,field['upperInterface'][:,1]);rz=np.interp(x,xs,field['upperInterface'][:,2]);ys=np.arange(ry-.001,ry+.012,.000025)
   u=z_envelope(yz,ys,False,rz-.005,rz+.006);cases.append((wi,recipe_l,ys,u,rz))
  def gap(theta):
   delta=rotate_lower(w,field,np.full(len(w),theta));answer=1.
   for wi,rec,ys,u,rz in cases:
    l=z_envelope(section_from(wi+delta,rec),ys,True,rz-.005,rz+.006);valid=np.isfinite(u)&np.isfinite(l)
    if valid.any():answer=min(answer,float((u[valid]-l[valid]).min()))
   return answer
  g0=gap(0.);threshold=min(.00006,g0-.000002)
  # Preserve pre-existing corner overlap instead of inventing zero-clearance.
  lo=0.;hi=.50
  for _ in range(17):
   mid=(lo+hi)/2
   if gap(mid)>=threshold:lo=mid
   else:hi=mid
  safe.append(lo);rows.append([x,g0,threshold,gap(lo)])
 safe=np.asarray(safe);radius=4;pad=np.pad(safe,(radius,radius),mode='edge');low=np.min(np.array([pad[i:i+len(safe)] for i in range(radius*2+1)]),axis=0)
 ker=np.array([1,2,3,4,5,4,3,2,1],float);ker/=ker.sum();pad=np.pad(low,(radius,radius),mode='edge');result=sum(k*pad[i:i+len(low)] for i,k in enumerate(ker))
 return result,np.asarray(rows)
