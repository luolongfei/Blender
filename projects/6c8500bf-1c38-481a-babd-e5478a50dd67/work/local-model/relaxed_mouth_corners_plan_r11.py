"""Pure-array neutral rest commissure candidate on actual29e3; no Blender.

This is a new independent study. Frozen seal and prior source are untouched.
"""
from pathlib import Path
import runpy, json, hashlib
import numpy as np
P=next(q for q in Path(__file__).resolve().parents if (q/'project-state.json').exists());OUT=P/'work/reconstruction-r02'
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))
def smooth(t):t=np.clip(t,0.,1.);return t*t*t*(10+t*(-15+6*t))
def cross2(a,b):return a[0]*b[1]-a[1]*b[0]
def overlap_polygon(a,b):
 a=a[:,:2].copy();b=b[:,:2].copy()
 if cross2(a[1]-a[0],a[2]-a[0])<0:a=a[::-1]
 if cross2(b[1]-b[0],b[2]-b[0])<0:b=b[::-1]
 poly=list(a)
 for q,r in zip(b,np.roll(b,-1,axis=0)):
  new=[]
  for s,e in zip(poly,poly[1:]+poly[:1]):
   ds=cross2(r-q,s-q);de=cross2(r-q,e-q)
   if ds>=-1e-15 and de>=-1e-15:new.append(e)
   elif ds>=-1e-15 and de< -1e-15:new.append(s+(e-s)*ds/(ds-de))
   elif ds< -1e-15 and de>=-1e-15:new.extend([s+(e-s)*ds/(ds-de),e])
  poly=new
  if len(poly)<3:return np.empty((0,2))
 return np.asarray(poly)
def z_at(tri,xy):
 M=np.column_stack([tri[1,:2]-tri[0,:2],tri[2,:2]-tri[0,:2]])
 if abs(np.linalg.det(M))<1e-14:return None
 uv=np.linalg.solve(M,(xy-tri[0,:2]).T).T;return tri[0,2]+uv[:,0]*(tri[1,2]-tri[0,2])+uv[:,1]*(tri[2,2]-tri[0,2])
def pair_depth(world,triangles,groups,pair):
 i,j=pair;gi,gj=int(groups[i]),int(groups[j])
 if gi in (7,17) and gj in (8,22):u,l=i,j
 elif gj in (7,17) and gi in (8,22):u,l=j,i
 else:return {'pair':list(pair),'unsupportedGroups':[gi,gj]}
 ut=world[triangles[u]];lt=world[triangles[l]];xy=overlap_polygon(ut,lt)
 if len(xy)<3:return {'pair':list(pair),'noProjectedOverlap':True}
 uz=z_at(ut,xy);lz=z_at(lt,xy)
 if uz is None or lz is None:return {'pair':list(pair),'verticalDegenerate':True}
 gap=lz-uz;maximum=float(gap.max());k=int(gap.argmax())
 return {'pair':list(pair),'upperTriangle':int(u),'lowerTriangle':int(l),'maximumZPenetrationMeters':maximum,'maximumAtXY':xy[k].tolist(),'upperZ':float(uz[k]),'lowerZ':float(lz[k]),'projectedPolygon':xy.tolist()}
def common_field(world,faces,groups):
 surf=api('relaxed_lip_seal_surface_r11');edges=surf['rim_edges'](faces,groups,7,17);xs=np.linspace(-.0205,.0205,165);rims=[]
 for x in xs:
  rr=surf['cut_edges'](world,edges,x)
  if not len(rr):raise RuntimeError('Actual upper mouth interface missing')
  rims.append(rr[np.argmin(rr[:,1])])
 rims=np.asarray(rims);anchor_y=np.interp(world[:,0],xs,rims[:,1]);anchor_z=np.interp(world[:,0],xs,rims[:,2]);x=abs(world[:,0]);dy=world[:,1]-anchor_y;dz=world[:,2]-anchor_z
 wx=smooth((x-.010)/.011)*(1-smooth((x-.021)/.009))
 wy=smooth((dy+.010)/.005)*(1-smooth((dy-.004)/.007))
 wz=smooth((dz+.013)/.010)*(1-smooth((dz-.003)/.010))
 delta=np.zeros_like(world);delta[:,2]=.0022*wx*wy*wz;delta[abs(delta[:,2])<1e-10]=0.
 return delta,{'xs':xs,'rim':rims,'anchorY':anchor_y,'anchorZ':anchor_z,'commonWeight':wx*wy*wz}
def load():
 d=np.load(OUT/'reference-adult-relaxed-lip-seal-r11.relaxed-lip-seal.npz');g=np.load(OUT/'head-neck-current-posture-r11-v2.npz');w=d['after_0.0_1.0World'];tri=d['after_0.0_1.0Triangles'];faces=d['fullFaces'];groups=g['rest_head_faceGroups'];poly=d['beforeWarm0.0TrianglePolygonIds']
 if not np.all((tri[:,:,None]==faces[poly,None,:]).any(2)):raise RuntimeError('Actual triangle/polygon IDs differ')
 return d,w,tri,faces,groups,poly
def graph_edges(faces):
 return np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1),axis=0)
def diffuse_floor(required,allowed,edges,iterations=12):
 value=required.copy();a=edges[:,0];b=edges[:,1];count=np.bincount(np.r_[a,b],minlength=len(value)).astype(float);count=np.maximum(count,1.)
 for _ in range(iterations):
  total=np.bincount(a,weights=value[b],minlength=len(value))+np.bincount(b,weights=value[a],minlength=len(value))
  value=np.maximum(required,.90*total/count);value[~allowed]=0.
 for _ in range(8):
  total=np.bincount(a,weights=value[b],minlength=len(value))+np.bincount(b,weights=value[a],minlength=len(value))
  value=np.maximum(required,value*.6+total/count*.4);value[~allowed]=0.
 return value

def plan(world,faces,groups,triangles,triangle_polygon_ids):
 w=np.asarray(world,float);t=np.asarray(triangles,np.int32);f=np.asarray(faces,np.int32);g=np.asarray(groups,np.int32);tg=g[triangle_polygon_ids]
 common,field=common_field(w,f,g);current=w+common;active=np.any(common,1);fn=api('adult_blink_midpath_contact_r11')['changed_intersections'];first=fn(current,t,active);edges=graph_edges(f);up=np.zeros(len(w),bool);low=up.copy();up[np.unique(f[np.isin(g,[7,17])])]=True;low[np.unique(f[np.isin(g,[8,22])])]=True;shared=up&low;up[shared]=False;low[shared]=False
 region=np.zeros(len(w),bool)
 for pair in first:region[t[list(pair)].ravel()]=True
 for _ in range(12):region[np.unique(t[region[t].any(1)])]=True
 # Keep true attached corner boundary coherent. The original crossing strips
 # are 20+ quad edges apart, so six-ring boundary fade retains useful support.
 distance=np.full(len(w),99,np.int32);distance[shared]=0
 for level in range(1,7):
  touched=distance<level;hit=np.unique(edges[touched[edges].any(1)]);distance[hit]=np.minimum(distance[hit],level)
 allowed=region&(abs(w[:,0])>.010)&(abs(w[:,0])<.026)
 weights=smooth(distance/6.)
 correction=np.zeros_like(w);rounds=[];index=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex']
 for iteration in range(6):
  hits=fn(current,t,active|np.any(correction,1));rounds.append({'iteration':iteration,'crossings':len(hits)});print('CORNER_CLEARANCE',iteration,len(hits),flush=True)
  if not hits:break
  # Actual current front upper/lower strips, bounded by the old contact graph
  # region, produce vertical envelopes; no guessed tooth/cavity geometry.
  tf=region[t].any(1);ut=t[tf&np.isin(tg,[7,17])];lt=t[tf&np.isin(tg,[8,22])]
  upper_world=current[:,[0,2,1]];lower_world=upper_world.copy();lower_world[:,1]*=-1
  upper_index=index(upper_world,ut);lower_index=index(lower_world,lt)
  ui=np.flatnonzero(allowed&up);li=np.flatnonzero(allowed&low);uz,_,_=upper_index.front(current[li][:,[0,2,1]]);lz,_,_=lower_index.front(current[ui][:,[0,2,1]]);lz=-lz
  req_up=np.zeros(len(w));req_low=req_up.copy();ok=np.isfinite(lz);req_up[ui[ok]]=np.maximum(lz[ok]+.00006-current[ui[ok],2],0.)*.58
  ok=np.isfinite(uz);req_low[li[ok]]=np.maximum(current[li[ok],2]+.00006-uz[ok],0.)*.42
  req_up*=weights;req_low*=weights
  # The two sides remain independent but share a gently propagated support.
  du=diffuse_floor(req_up,allowed&up,edges);dl=diffuse_floor(req_low,allowed&low,edges)
  if max(du.max(),dl.max())>.0015:raise RuntimeError('Contact solver requests more than1.5mm; diagnose actual envelope before use')
  step=du-dl;correction[:,2]+=step;current[:,2]+=step
  rounds[-1].update(maximumUpperStepMeters=float(du.max()),maximumLowerStepMeters=float(dl.max()),maximumAccumulatedCorrectionMeters=float(abs(correction[:,2]).max()))
 final=fn(current,t,active|np.any(correction,1));offset=current-w;quality=api('lower_face_continuity_surface_r11')['quality'](w,current,f,np.any(offset,1))
 arrays={'fullWorldBefore':w,'triangles':t,'faces':f,'groups':g,'trianglePolygonIds':triangle_polygon_ids,'offsets':offset,'commonOffsets':common,'contactOffsets':correction,'protectedCentral':abs(w[:,0])<=.010,'contactRegion':region,'upperExclusive':up,'lowerExclusive':low}
 dependency=np.any(offset,1).copy()
 for _ in range(2):dependency[np.unique(f[dependency[f].any(1)])]=True
 arrays['dependencyOriginalVertexIds']=np.flatnonzero(dependency).astype(np.int32)
 report={'sourceScope':'Actual29e3 neutral input + independent theoretical rest-corner field, no Blender execution','control':'R11 relaxed mouth corners','centralExact':bool(np.array_equal(w[abs(w[:,0])<=.010],current[abs(w[:,0])<=.010])),'rounds':rounds,'remainingCrossingPairs':[list(p) for p in sorted(final)],'quality':quality,'maximumDisplacementMeters':float(np.linalg.norm(offset,axis=1).max()),'affectedVertices':int(np.any(offset,1).sum()),'appearanceAccepted':False,'technicalAcceptance':False}
 return offset,arrays,report
if __name__=='__main__':
 d,w,t,f,g,poly=load();offset,arrays,report=plan(w,f,g,t,poly)
 np.savez_compressed(OUT/'relaxed-rest-mouth-corners-plan-r11.npz',**arrays);(OUT/'relaxed-rest-mouth-corners-plan-r11.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='remainingCrossingPairs'},indent=2));print('remaining',len(report['remainingCrossingPairs']))
