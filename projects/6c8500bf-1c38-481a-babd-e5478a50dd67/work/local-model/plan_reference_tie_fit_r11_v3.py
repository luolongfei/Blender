"""Offline conforming-band proposal from real rings, pins and all true gather.

No Blender import or source write. Hair is a conservative swept-radius
CONTROL-POLYLINE envelope, not proof of the native interpolated curves.
All input geometry is observed in REST then placed by the frozen rigid T.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unit(x):return x/max(float(np.linalg.norm(x)),1e-15)
def stats(x):return dict(zip(('min','p05','median','p95','max'),np.percentile(x,[0,5,50,95,100]).tolist()))
def transform(p,tr):return (p-np.asarray(tr['originWorld']))@np.asarray(tr['rotationWorld']).T+np.asarray(tr['targetWorld'])
def world_to_local(p,m):
 inv=np.linalg.inv(m);return p@inv[:3,:3].T+inv[:3,3]
def capsule_distance(q,a,b,ra,rb):
 """Exact query to linear center/radius sweep among the selected segments."""
 v=b-a;length=np.linalg.norm(v,axis=1);good=length>1e-12
 if not np.all(good):a=a[good];b=b[good];ra=ra[good];rb=rb[good];v=v[good];length=length[good]
 u=v/length[:,None];slope=(rb-ra)/length
 if np.any(abs(slope)>=1):raise RuntimeError('Radius slope >= 1 needs separate sphere envelope')
 gap=[];nearest=[]
 for start in range(0,len(q),24):
  x=q[start:start+24,None]-a[None];along=np.sum(x*u,axis=2);h2=np.maximum(np.sum(x*x,axis=2)-along*along,0)
  t=np.clip(along+slope*np.sqrt(h2/np.maximum(1-slope*slope,1e-30)),0,length)
  val=np.sqrt((t-along)**2+h2)-(ra+slope*t);ix=val.argmin(1)
  gap.extend(val[np.arange(len(ix)),ix]);nearest.extend(ix)
 return np.asarray(gap),np.asarray(nearest,np.int32)
def area_points(triangles,n=4):
 bary=[]
 for i in range(n):
  for j in range(n-i):
   bary.append([1-(i+j+2/3)/n,(i+1/3)/n,(j+1/3)/n])
   if i+j<n-1:bary.append([1-(i+j+4/3)/n,(i+2/3)/n,(j+2/3)/n])
 bary=np.asarray(bary);q=np.einsum('bi,tij->tbj',bary,triangles).reshape(-1,3)
 areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)*.5
 return q,np.repeat(areas/(n*n),n*n)
def signed_volume(points,tri):
 t=points[tri]-points.mean(0);return float(np.sum(t[:,0]*np.cross(t[:,1],t[:,2]))/6)
def convex_cross(points):
 edges=np.roll(points,-1,axis=0)-points;next_edges=np.roll(edges,-1,axis=0);return edges[:,0]*next_edges[:,1]-edges[:,1]*next_edges[:,0]
def collect_segments(obs,gather,side,roles):
 tail=obs['tail_'+side+'_world'].reshape(2480,35,3);tail=transform(tail,roles['transform'])
 radius=obs['tail_'+side+'_radius'].reshape(2480,35)*roles['tailRadiusScale']
 aa=[tail[:,:2].reshape(-1,3)];bb=[tail[:,1:3].reshape(-1,3)];ra=[radius[:,:2].reshape(-1)];rb=[radius[:,1:3].reshape(-1)]
 owner=[np.repeat(np.arange(2480,dtype=np.int32),2)];kind=[np.zeros(2480*2,np.int8)];arc_a=[np.zeros(2480*2)];arc_b=[np.zeros(2480*2)]
 world=gather['target_world'];starts=gather['source_starts'];lengths=gather['source_lengths'];groups=gather['source_group']%256;r=gather['source_radius']
 for curve in gather['gather_curve_ids'][groups[gather['gather_curve_ids']]==roles['role']]:
  s=int(starts[curve]);n=int(lengths[curve]);aa.append(world[s:s+n-1]);bb.append(world[s+1:s+n]);ra.append(r[s:s+n-1]);rb.append(r[s+1:s+n]);owner.append(np.full(n-1,curve,np.int32));kind.append(np.ones(n-1,np.int8))
  arc=np.r_[np.cumsum(np.linalg.norm(np.diff(world[s:s+n],axis=0),axis=1)[::-1])[::-1],0.];arc_a.append(arc[:-1]);arc_b.append(arc[1:])
 return dict(a=np.concatenate(aa).astype(float),b=np.concatenate(bb).astype(float),ra=np.concatenate(ra),rb=np.concatenate(rb),owner=np.concatenate(owner),kind=np.concatenate(kind),tip_arc_a=np.concatenate(arc_a),tip_arc_b=np.concatenate(arc_b),tail_first3=tail[:,:3],tail_first3_radius=radius[:,:3])
def clipped_envelope(seg,c,n,half_width,terminal_only=False):
 z0=(seg['a']-c)@n;z1=(seg['b']-c)@n;maxr=np.maximum(seg['ra'],seg['rb']);lo=-half_width-maxr;hi=half_width+maxr
 valid=(np.minimum(z0,z1)<=hi)&(np.maximum(z0,z1)>=lo)
 if terminal_only:
  # The wrapped bundle is the final connected approach toward the tail,
  # not an earlier bend that revisits this infinite plane. Keep all of the
  # connected final non-reversing axial run; no nearest-root quantile trim.
  forward=(z1-z0)>=-1e-9;eligible=np.ones(len(z0),bool)
  boundaries=np.r_[0,np.flatnonzero((np.diff(seg['owner'])!=0)|(np.diff(seg['kind'])!=0))+1,len(z0)]
  for start,end in zip(boundaries[:-1],boundaries[1:]):
   if seg['kind'][start]==0:continue
   reverse=np.flatnonzero(~forward[start:end])
   if len(reverse):eligible[start:start+int(reverse[-1])+1]=False
  valid &= eligible
 ids=np.flatnonzero(valid);delta=z1[ids]-z0[ids]
 t0=np.zeros(len(ids));t1=np.ones(len(ids));moving=abs(delta)>1e-12
 x0=(lo[ids][moving]-z0[ids][moving])/delta[moving];x1=(hi[ids][moving]-z0[ids][moving])/delta[moving]
 t0[moving]=np.clip(np.minimum(x0,x1),0,1);t1[moving]=np.clip(np.maximum(x0,x1),0,1)
 if terminal_only:
  # Fit only the actual terminal bundle, not distant upstream hair that
  # happens to cross this infinite plane. All upstream segments remain in
  # the separate physical ring-to-hair collision query below.
  isgather=seg['kind'][ids]==1;arc_delta=seg['tip_arc_a'][ids]-seg['tip_arc_b'][ids]
  minimum=np.zeros(len(ids));minimum[isgather]=(seg['tip_arc_a'][ids][isgather]-.030)/np.maximum(arc_delta[isgather],1e-15)
  t0=np.maximum(t0,minimum);valid=t0<=t1
  ids=ids[valid];t0=t0[valid];t1=t1[valid]
 a=seg['a'][ids];v=seg['b'][ids]-a;ra=seg['ra'][ids];dr=seg['rb'][ids]-ra
 points=np.concatenate([a+v*t0[:,None],a+v*t1[:,None]]);radii=np.r_[ra+dr*t0,ra+dr*t1]
 return ids,points,radii
def solve_support_polygon(points,radii,u,v,c,theta,margin):
 plane=np.c_[(points-c)@u,(points-c)@v];angle=(theta+np.r_[theta[1:],theta[0]+2*np.pi])*.5
 normals=np.c_[np.cos(angle),np.sin(angle)];support=np.max(plane@normals.T+radii[:,None],axis=0)+margin
 poly=np.array([np.linalg.solve(np.stack([normals[i-1],normals[i]]),[support[i-1],support[i]]) for i in range(32)])
 violation=float(np.max(poly@normals.T-support));cross=convex_cross(poly)
 if violation>1e-7 or cross.min()<-1e-10:raise RuntimeError('Support contour needs active-plane simplification; do not fit an invalid polygon')
 outward=np.array([unit(normals[i-1]+normals[i]) for i in range(32)])
 containment=float(np.max(plane@normals.T+radii[:,None]-support))
 return poly,outward,normals,support,plane,containment,violation
def plot(path,rows,arr):
 im=Image.new('RGB',(1400,1450),(247,246,242));d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
 d.text((24,18),'ACTUAL ARRAY CROSS-SECTIONS - candidate bands, not a Blender render',font=font,fill=(20,20,20))
 d.text((24,50),'Gray: old inner/outer rings | Black: fitted inner/outer band | Green: tail pins | Blue: true gather in full-width slab',font=small,fill=(30,30,30))
 for index,row in enumerate(rows):
  col=index%2;rr=index//2;left=col*700+32;top=rr*650+108;cx=left+310;cy=top+270;scale=14000;prefix=row['prefix']
  def xy(p):return (cx+float(p[0])*scale,cy-float(p[1])*scale)
  d.text((left,top-20),row['ring']+' / same fixed axis plane',font=font,fill=(20,20,20))
  points=arr[prefix+'_clipped_hair_plane'];kind=arr[prefix+'_clipped_hair_kind']
  for k in [0,1]:
   sample=points[kind==k];step=max(1,len(sample)//5000)
   for p in sample[::step]:x,y=xy(p);d.ellipse((x-1,y-1,x+1,y+1),fill=(45,133,73) if k==0 else (60,115,176))
  for key,color,width in [('old_inner_plane',(175,175,175),2),('old_outer_plane',(175,175,175),2),('new_inner_plane',(15,15,15),3),('new_outer_plane',(15,15,15),3)]:
   p=arr[prefix+'_'+key];d.line([xy(x) for x in np.vstack([p,p[0]])],fill=color,width=width)
  d.text((left,top+545),f'Band width {row["axialWidthMm"]:.2f} mm; radial thickness {row["radialThicknessMm"]:.2f} mm',font=small,fill=(40,40,40))
  d.text((left,top+568),'Fixed hair: complete terminal approach + pins; other hair still collision-checked.',font=small,fill=(40,40,40))
 d.text((24,1408),'Control-polyline proposal only. Reobserve actual native curves after tie installation; appearance and production acceptance pending.',font=small,fill=(30,30,30));im.save(path)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--observation',required=True);ap.add_argument('--tie-plan',required=True);ap.add_argument('--gather-plan',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 obs_path,plan_path,gather_path,out=[(P/x).resolve() for x in (args.observation,args.tie_plan,args.gather_plan,args.output)];npz=out.with_suffix('.npz');png=out.with_name(out.stem+'-geometry.png')
 for p in [obs_path,plan_path,gather_path,out,npz,png]:p.relative_to((P/'work/reconstruction-r02').resolve())
 if any(p.exists() for p in [out,npz,png]):raise RuntimeError('Independent unused output paths required')
 obsj=json.loads(obs_path.read_text());plan=json.loads(plan_path.read_text());gj=json.loads(gather_path.read_text());obsfile=P/obsj['arrays'];gfile=P/gj['arrays']
 if sha(obsfile)!=obsj['arraysSha256'] or sha(gfile)!=gj['arraysSha256'] or sha(plan_path)!=gj['tiePlanSha256'] or sha(obs_path)!=plan['observationSha256']:raise RuntimeError('Frozen actual geometry identity mismatch')
 obs=np.load(obsfile);g=np.load(gfile);arr={};rows=[];errors=[];width=.0044;thickness=.0014;bevel=.0003;margin=.00015
 geom=runpy.run_path(str(HERE/'plan_reference_tie_transport_r11.py'));near=runpy.run_path(str(HERE/'analyze_reference_gather_transport_r11.py'))['oriented_near'];head=obs['head_world'][obs['head_triangles']]
 for side,role in [('positive',40),('negative',41)]:
  tailrow=next(x for x in obsj['tails'] if x['prefix']=='tail_'+side);seg=collect_segments(obs,g,side,{'transform':plan['transforms'][side],'role':role,'tailRadiusScale':max(tailrow['matrixSingularValues'])});arr['tail_'+side+'_first3_world']=seg['tail_first3'];arr['tail_'+side+'_first3_radius']=seg['tail_first3_radius']
  for ring in ['a','b']:
   prefix='tie_'+side+'_'+ring;raw=prefix+'_raw';oldlocal=obs[raw+'_local'];oldworld=obs[raw+'_world'];matrix=obs[raw+'_matrix'];before=transform(oldworld,plan['transforms'][side]);blocks=before.reshape(32,8,3);c=before.mean(0);centerline=blocks.mean(1)
   _,vec=np.linalg.eigh((before-c).T@(before-c));n=vec[:,0]
   if n@(seg['tail_first3'][:,2].mean(0)-seg['tail_first3'][:,0].mean(0))<0:n=-n
   u=unit(centerline[0]-c-n*np.dot(centerline[0]-c,n));v=np.cross(n,u);theta=np.unwrap(np.arctan2((centerline-c)@v,(centerline-c)@u))
   if np.mean(np.diff(theta))<0:v=-v;theta=np.unwrap(np.arctan2((centerline-c)@v,(centerline-c)@u))
   selected,points,radii=clipped_envelope(seg,c,n,width/2,terminal_only=True)
   poly,outward,normals,support,plane,containment,violation=solve_support_polygon(points,radii,u,v,c,theta,margin)
   axis_sign=1 if np.mean((blocks[:,2]-centerline)@n)>0 else -1
   radial=np.array([thickness,thickness,thickness*.5,0,0,0,thickness*.5,thickness]);axial=np.array([0,width/2-bevel,width/2,width/2-bevel,0,-width/2+bevel,-width/2,-width/2+bevel])*axis_sign
   p2=poly[:,None,:]+outward[:,None,:]*radial[None,:,None];candidate=c+p2[:,:,0,None]*u+p2[:,:,1,None]*v+axial[None,:,None]*n;candidate=candidate.reshape(256,3);local=world_to_local(candidate,matrix).astype(np.float32);rounded=local@matrix[:3,:3].T+matrix[:3,3];tri=obs[raw+'_triangles'];innerface=np.all((tri%8>=2)&(tri%8<=6),axis=1)
   oldq,oldweights=area_points(before[tri[innerface]],4);newq,newweights=area_points(rounded[tri[innerface]],4)
   # Select all segments touching the widest old/new axial slab. Distances to
   # other segments cannot establish inner-wall support at this band plane.
   collisionids,_,_=clipped_envelope(seg,c,n,max(width/2,float(np.max(abs((before-c)@n))))+.001)
   ss={k:seg[k][collisionids] for k in ['a','b','ra','rb']};oldgap,_=capsule_distance(oldq,**ss);newgap,nearest=capsule_distance(newq,**ss)
   headgap,_=near(np.concatenate([rounded,rounded[tri].mean(1)]),head,head.mean(1),24,geom['closest_triangles'],batch=16)
   trianglearea=np.linalg.norm(np.cross(rounded[tri[:,1]]-rounded[tri[:,0]],rounded[tri[:,2]]-rounded[tri[:,0]]),axis=1)*.5
   oldarea=np.linalg.norm(np.cross(before[tri[:,1]]-before[tri[:,0]],before[tri[:,2]]-before[tri[:,0]]),axis=1)*.5
   oldinner=np.c_[(blocks[:,4]-c)@u,(blocks[:,4]-c)@v];oldouter=np.c_[(blocks[:,0]-c)@u,(blocks[:,0]-c)@v];newinner=np.c_[(rounded.reshape(32,8,3)[:,4]-c)@u,(rounded.reshape(32,8,3)[:,4]-c)@v];newouter=np.c_[(rounded.reshape(32,8,3)[:,0]-c)@u,(rounded.reshape(32,8,3)[:,0]-c)@v]
   if containment>1e-8 or newgap.min()<-.0001:errors.append(prefix+': observed control capsule intersects the proposed inner band')
   if headgap.min()<.0005:errors.append(prefix+': sampled head clearance below 0.5 mm')
   if trianglearea.min()<1e-12 or signed_volume(rounded,tri)*signed_volume(before,tri)<=0:errors.append(prefix+': degenerate or reversed mesh')
   if convex_cross(newinner).min()<-1e-10 or convex_cross(newouter).min()<-1e-10:errors.append(prefix+': nonconvex inner/outer loop')
   row={'ring':side+'_'+ring,'prefix':prefix,'object':next(x['object'] for x in obsj['ties'] if x['raw']['prefix']==raw),'vertices':256,'triangles':len(tri),'centerWorld':c.tolist(),'axisWorld':n.tolist(),'planeUWorld':u.tolist(),'planeVWorld':v.tolist(),'axialWidthMm':width*1000,'radialThicknessMm':thickness*1000,'fitMarginMm':margin*1000,
    'supportContainmentMaximumMeters':containment,'supportPolygonViolationMeters':violation,'selectedEnvelopeSegments':len(selected),'selectedTailSegments':int(np.sum(seg['kind'][selected]==0)),'selectedTrueGatherSegments':int(np.sum(seg['kind'][selected]==1)),
    'oldInnerWallGapMm':stats(oldgap*1000),'newInnerWallGapMm':stats(newgap*1000),'oldInnerWallAreaFractionOver1mm':float(oldweights[oldgap>.001].sum()/oldweights.sum()),'newInnerWallAreaFractionOver1mm':float(newweights[newgap>.001].sum()/newweights.sum()),'newInnerWallAreaFractionUnder05mm':float(newweights[newgap<.0005].sum()/newweights.sum()),
    'headSampledMinimumGapMm':float(headgap.min()*1000),'centroidShiftMm':((rounded.mean(0)-c)*1000).tolist(),'maximumVertexMovementMm':float(np.linalg.norm(rounded-before,axis=1).max()*1000),'localRoundoffMaximumMicrometers':float(np.linalg.norm(rounded-candidate,axis=1).max()*1e6),
    'innerPlaneDiametersMmBefore':(np.ptp(oldinner,axis=0)*1000).tolist(),'innerPlaneDiametersMmAfter':(np.ptp(newinner,axis=0)*1000).tolist(),'triangleMinimumAreaM2':float(trianglearea.min()),'triangleAreaRatioMinMax':(float((trianglearea/oldarea).min()),float((trianglearea/oldarea).max())),
    'selfIntersectionEvidence':'Ordered convex inner/outer loops, noncrossing positive-thickness cross profile, positive triangle areas and same signed volume orientation. Full triangle-pair intersection is not run.',
    'hairEnvelopeScope':'Every true gather final connected non-reversing axial approach, up to 30 mm arc from its exact tail root, plus all tail first two pin segments intersecting the full band slab. Earlier disconnected plane crossings remain in the separate complete capsule collision check, not the wrapped bundle. No face-curtain exclusions reinstated.',
    'allSlabCrossingCollisionSegments':len(collisionids),'nonterminalSegmentsRetainedForCollision':int(np.sum(~np.isin(collisionids,selected)))}
   values={'source_original_local':oldlocal,'source_expected_after_transport_world':before,'source_expected_after_transport_local':world_to_local(before,matrix),'candidate_world':rounded,'candidate_local':local,'matrix':matrix,'triangles':tri,'vertex_correspondence':np.arange(256,dtype=np.int32),'old_inner_plane':oldinner,'old_outer_plane':oldouter,'new_inner_plane':newinner,'new_outer_plane':newouter,'clipped_hair_plane':plane,'clipped_hair_radius':radii,'clipped_hair_kind':np.tile(seg['kind'][selected],2),'envelope_segment_ids':selected,'envelope_segment_kind':seg['kind'][selected],'envelope_segment_owner':seg['owner'][selected],'envelope_segment_a_world':seg['a'][selected],'envelope_segment_b_world':seg['b'][selected],'envelope_segment_radius_a':seg['ra'][selected],'envelope_segment_radius_b':seg['rb'][selected],'support_normals_plane':normals,'support_distances':support,'inner_wall_samples_world':newq,'inner_wall_area_weights':newweights,'inner_wall_hair_clearance':newgap,'inner_wall_nearest_segment':collisionids[nearest],'head_sample_clearance':headgap}
   for key,val in values.items():arr[prefix+'_'+key]=val
   rows.append(row);print('BAND_FIT '+json.dumps({'ring':row['ring'],'segments':len(selected),'gather':row['selectedTrueGatherSegments'],'gapMedianMm':row['newInnerWallGapMm']['median'],'inside':float(newgap.min()),'headGapMm':row['headSampledMinimumGapMm']}),flush=True)
 pairs=[]
 for side in ['positive','negative']:
  aa=next(r for r in rows if r['ring']==side+'_a');bb=next(r for r in rows if r['ring']==side+'_b');separation=abs(np.dot(np.asarray(bb['centerWorld'])-aa['centerWorld'],aa['axisWorld']));gap=separation-width;pairs.append({'side':side,'axisCenterSeparationMm':separation*1000,'axialSlabSeparationMm':gap*1000})
  if gap<.0004:errors.append(side+': bands have less than 0.4 mm axial separation')
 with npz.open('xb') as f:np.savez_compressed(f,**arr)
 plot(png,rows,arr)
 report={'schemaVersion':1,'status':'blocked' if errors else 'proposed','errors':errors,'appearanceAccepted':False,'productionCollisionAccepted':False,'savedBlend':False,'rendered':False,'source':obsj['source'],'sourceSha256':obsj['sourceSha256'],'observation':str(obs_path.relative_to(P)),'observationSha256':sha(obs_path),'observationArraysSha256':sha(obsfile),'tiePlan':str(plan_path.relative_to(P)),'tiePlanSha256':sha(plan_path),'gatherPlan':str(gather_path.relative_to(P)),'gatherPlanSha256':sha(gather_path),'gatherArraysSha256':sha(gfile),'helperSha256':sha(__file__),'arrays':str(npz.relative_to(P)),'arraysSha256':sha(npz),'geometryDiagram':str(png.relative_to(P)),'geometryDiagramSha256':sha(png),'rings':rows,'pairs':pairs,
  'hairModified':False,'parentingOrObjectMatricesModified':False,'positionsConvention':'Candidate local uses the observed unchanged object matrix; the frozen transport writes vertices through this same matrix. Installer must verify current expected-after-transport coordinates before applying.',
  'nativeCurveEnvelopeVerified':False,'mandatoryBeforeInstallation':['Reobserve the actually saved successful tie/gather source; this planner does not assume it already exists.','Match all ring vertex IDs, source matrices and after-transport world positions; protect all tail/pin/gather coordinates.','Evaluate actual native tail/gather curves at sufficient resolution and all radii near both band widths; reject or refit if their envelope exceeds this proposed ring.','Check current visible head/ear and full mesh intersections; this offline sampled head check is not a full collision proof.','Render front and quarter to judge reference-like black band width and fit.'],
  'limits':['Envelope guarantees apply to the measured straight control segments and varying radii, not undocumented native interpolation.','No hair pressure/compression solver was run. Ring-to-hair support gaps are evidence, not a claimed physical attachment simulation.','No material changes, source writes, cache actions or Blender renders.']}
 with out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
 print('BAND_PLAN '+json.dumps({'status':report['status'],'errors':errors,'output':str(out.relative_to(P)),'arraysSha256':report['arraysSha256']}),flush=True)
if __name__=='__main__':main()
