"""Read-only residual lip-corner topology on actual31df + frozen seal field.

No Blender, no source modification. Positions after seal are a theoretical
composition of actual observed 31df and frozen full-strength vectors. Pair
indices use declared polygon-fan triangulation, not a new actual Blender run.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
P=next(q for q in Path(__file__).resolve().parents if (q/'project-state.json').exists());OUT=P/'work/reconstruction-r02'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=OUT/'relaxed-lip-seal-plan-r11.npz';reference=OUT/'reference-adult-lower-face-continuity-v2-r11.lower-face-continuity-v2.npz';preflight=OUT/'relaxed-lip-seal-preflight-r11.json'
 a=np.load(plan);v=np.load(reference);rows=json.loads(preflight.read_text());ids=a['originalVertexIds'];faces=a['faces'];groups=a['groups'];full=a['fullFaces'];faceids=a['roiOriginalFaceIds'];offset=a['worldOffsets'];n=len(v['afterWarm0.0World'])
 tri=np.concatenate([faces[:,[0,1,2]],faces[:,[0,2,3]]]);tripoly=np.tile(np.arange(len(faces),dtype=np.int32),2);globaltri=ids[tri]
 fullgroups=np.full(len(full),-1,np.int32);fullgroups[faceids]=groups
 edges=np.concatenate([full[:,[0,1]],full[:,[1,2]],full[:,[2,3]],full[:,[3,0]]]);edges=np.unique(np.sort(edges,axis=1),axis=0)
 directed=np.concatenate([edges,edges[:,::-1]]);directed=directed[np.lexsort((directed[:,1],directed[:,0]))];starts=np.r_[0,np.cumsum(np.bincount(directed[:,0],minlength=n))];neighbors=directed[:,1]
 vertexgroups=[set() for _ in range(n)]
 for face,group in zip(full,fullgroups):
  for vertex in face:vertexgroups[int(vertex)].add(int(group))
 cached={};seen=np.zeros(n,np.int32);dist=np.empty(n,np.int16);prev=np.empty(n,np.int32);stamp=0
 def shortest(aa,bb):
  nonlocal stamp
  key=(tuple(sorted(map(int,aa))),tuple(sorted(map(int,bb))))
  if key in cached:return cached[key]
  stamp+=1;targets=set(map(int,bb));queue=collections.deque()
  for vertex in aa:vertex=int(vertex);seen[vertex]=stamp;dist[vertex]=0;prev[vertex]=-1;queue.append(vertex)
  end=None
  while queue:
   u=queue.popleft()
   if u in targets:end=u;break
   for vv in neighbors[starts[u]:starts[u+1]]:
    vv=int(vv)
    if seen[vv]!=stamp:seen[vv]=stamp;dist[vv]=dist[u]+1;prev[vv]=u;queue.append(vv)
  if end is None:raise RuntimeError('No full-mesh topological connection between lip triangles')
  path=[];cur=end
  while cur!=-1:path.append(cur);cur=int(prev[cur])
  path=path[::-1];r={'minimumQuadEdgeCount':int(dist[end]),'globalVertexPath':path,'pathVertexFaceGroups':[sorted(vertexgroups[k]) for k in path]};cached[key]=r;return r
 result={'source':'work/reconstruction-r02/reference-adult-lower-face-continuity-v2-r11.blend','sourceSha256':'31df467c65168468b2a246e169e165a3f3e870bcb0eccabac8d0377c565c8b4b','positionScope':'Actual31df three warm states plus frozen full-seal delta, theoretical after geometry; NOT newly evaluated actual5f30 seal installation.',
 'triangulation':'Declared quad polygon fan [0,1,2] and [0,2,3], same as offline preflight; not Blender loop-triangle evidence.','globalIndexScope':'Original evaluated fixed-L3 head vertex and quad polygon IDs, not raw cage IDs.','inputSha256':{p.name:sha(p) for p in [plan,reference,preflight]},'fullVertexCount':n,'fullQuadCount':len(full),'fullUniqueQuadEdges':len(edges),'states':[],'BlenderExecuted':False,'geometryModified':False}
 for row in rows:
  if row['seal']!=1.:continue
  warm=row['warm'];contact=row['contact'];residual=(set(map(tuple,contact['baselinePairs']))-set(map(tuple,contact['removedPairs'])))|set(map(tuple,contact['newPairs']));world=v['afterWarm'+str(warm)+'World'].copy();world[ids]+=offset
  records=[]
  for i,pair in enumerate(sorted(residual)):
   polys=faceids[tripoly[list(pair)]];tv=globaltri[list(pair)];quads=full[polys];shared_t=sorted(set(map(int,tv[0]))&set(map(int,tv[1])));shared_q=sorted(set(map(int,quads[0]))&set(map(int,quads[1])))
   edgesets=[{tuple(sorted(map(int,e))) for e in zip(q,np.roll(q,-1))} for q in quads];common_edges=sorted(edgesets[0]&edgesets[1]);center=world[tv].mean((0,1));path=shortest(tv[0],tv[1])
   records.append({'offlineTrianglePair':list(pair),'originalPolygonIds':polys.tolist(),'faceGroups':groups[tripoly[list(pair)]].tolist(),'globalTriangleVertexIds':tv.tolist(),'globalQuadVertexIds':quads.tolist(),'sharedTriangleVertices':shared_t,'sharedQuadVertices':shared_q,'sharedQuadEdges':common_edges,'worldCentroid':center.tolist(),'side':'world_negative_X' if center[0]<0 else 'world_positive_X','shortestFullHeadTopology':path})
   if i%100==0:print('TOPOLOGY',warm,i,'of',len(residual),flush=True)
  hist=collections.Counter('/'.join(map(str,sorted(r['faceGroups']))) for r in records);by_side={}
  for side in ['world_negative_X','world_positive_X']:
   part=[r for r in records if r['side']==side];points=np.array([r['worldCentroid'] for r in part]);by_side[side]={'pairs':len(part),'centroidBounds':None if not len(part) else [points.min(0).tolist(),points.max(0).tolist()],'minimumQuadEdgeCountRange':None if not part else [min(r['shortestFullHeadTopology']['minimumQuadEdgeCount'] for r in part),max(r['shortestFullHeadTopology']['minimumQuadEdgeCount'] for r in part)],'examplePair':None if not part else min(part,key=lambda r:r['shortestFullHeadTopology']['minimumQuadEdgeCount'])}
  result['states'].append({'warm':warm,'seal':1.,'residualPairCount':len(records),'groupCombinations':dict(hist),'sharedTriangleVertexPairCount':sum(bool(r['sharedTriangleVertices']) for r in records),'sharedQuadVertexPairCount':sum(bool(r['sharedQuadVertices']) for r in records),'sharedQuadEdgePairCount':sum(bool(r['sharedQuadEdges']) for r in records),'sides':by_side,'sectionEvidenceFromActual31dfPlusFrozenDelta':contact['sections'],'pairs':records})
  (OUT/'relaxed-lip-corner-topology-study-r11.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('DONE_STATE',warm,by_side['world_negative_X']['pairs'],by_side['world_positive_X']['pairs'],dict(hist),flush=True)
 print('COMPLETE',sha(OUT/'relaxed-lip-corner-topology-study-r11.json'))
if __name__=='__main__':main()
