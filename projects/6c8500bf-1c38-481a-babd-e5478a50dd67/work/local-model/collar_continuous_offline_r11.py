"""Actual-array offline oracle for continuous collar fitting. Never opens Blender."""
import argparse
import heapq
import json
from pathlib import Path
import runpy
import sys
import time
import types
import numpy as np


class Surface:
    def __init__(self,points,triangles):
        self.points=np.asarray(points,float);self.faces=np.asarray(triangles,np.int32)
        self.tri=self.points[self.faces];self.lo=self.tri.min(1);self.hi=self.tri.max(1)
        self.center=(self.lo+self.hi)*.5;self.nodes=[]
        self._build(np.arange(len(self.tri)))

    def _build(self,ids):
        lo=self.lo[ids].min(0);hi=self.hi[ids].max(0);idx=len(self.nodes);self.nodes.append(None)
        if len(ids)<=16:self.nodes[idx]=(lo,hi,ids,None)
        else:
            axis=np.argmax(self.center[ids].max(0)-self.center[ids].min(0));order=ids[np.argsort(self.center[ids,axis])];m=len(order)//2
            left=self._build(order[:m]);right=self._build(order[m:]);self.nodes[idx]=(lo,hi,left,right)
        return idx

    @staticmethod
    def _box(point,lo,hi):
        d=np.maximum(np.maximum(lo-point,point-hi),0.);return float(d@d)

    def _nearest_leaf(self,point,ids):
        t=self.tri[ids];a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a;n=np.cross(ab,ac);n2=np.sum(n*n,axis=1)
        proj=point-(np.sum((point-a)*n,axis=1)/np.maximum(n2,1e-30))[:,None]*n;ap=proj-a
        d00=np.sum(ab*ab,1);d01=np.sum(ab*ac,1);d11=np.sum(ac*ac,1);d20=np.sum(ap*ab,1);d21=np.sum(ap*ac,1)
        den=np.maximum(d00*d11-d01*d01,1e-30);u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den
        inside=(u>=0)&(v>=0)&(u+v<=1)&(n2>1e-24);qs=[proj];scores=[np.where(inside,np.sum((proj-point)**2,1),np.inf)]
        for i,j in ((0,1),(1,2),(2,0)):
            e=t[:,j]-t[:,i];f=np.clip(np.sum((point-t[:,i])*e,1)/np.maximum(np.sum(e*e,1),1e-30),0,1)
            q=t[:,i]+f[:,None]*e;qs.append(q);scores.append(np.sum((q-point)**2,1))
        scores=np.asarray(scores);k,i=np.unravel_index(np.argmin(scores),scores.shape)
        return float(scores[k,i]),qs[k][i],n[i]/np.sqrt(max(n2[i],1e-30)),int(ids[i])

    def find_nearest(self,point):
        point=np.asarray(point);queue=[(0.,0)];best=(float('inf'),None,None,None)
        while queue:
            d,idx=heapq.heappop(queue)
            if d>best[0]:break
            lo,hi,left,right=self.nodes[idx]
            if right is None:
                candidate=self._nearest_leaf(point,left)
                if candidate[0]<best[0]:best=candidate
            else:
                for child in (left,right):
                    l,h,_,_=self.nodes[child];d=self._box(point,l,h)
                    if d<=best[0]:heapq.heappush(queue,(d,child))
        return best[1],best[2],best[3],np.sqrt(best[0])

    def ray_cast(self,origin,direction,maximum):
        origin=np.asarray(origin);direction=np.asarray(direction);best=float(maximum);found=None;stack=[0]
        while stack:
            idx=stack.pop();lo,hi,left,right=self.nodes[idx];near=0.;far=best;hit=True
            for axis in range(3):
                if abs(direction[axis])<1e-14:
                    if origin[axis]<lo[axis] or origin[axis]>hi[axis]:hit=False;break
                else:
                    p=(lo[axis]-origin[axis])/direction[axis];q=(hi[axis]-origin[axis])/direction[axis]
                    near=max(near,min(p,q));far=min(far,max(p,q))
                    if near>far:hit=False;break
            if not hit:continue
            if right is not None:stack.extend((left,right));continue
            tri=self.tri[left];a=tri[:,0];e1=tri[:,1]-a;e2=tri[:,2]-a;h=np.cross(direction,e2);det=np.sum(e1*h,1)
            valid=np.abs(det)>1e-15;inv=np.where(valid,1./np.where(valid,det,1.),0.);s=origin-a;u=inv*np.sum(s*h,1);q=np.cross(s,e1);v=inv*np.sum(direction*q,1);d=inv*np.sum(e2*q,1)
            valid&=(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)&(d>=0)&(d<=best)
            if valid.any():
                j=np.argmin(np.where(valid,d,np.inf));best=float(d[j]);n=np.cross(e1[j],e2[j]);n/=np.linalg.norm(n);found=(origin+best*direction,n,int(left[j]),best)
        return found or (None,None,None,None)


def run(observation,output):
    a=runpy.run_path(str(Path(__file__).with_name('collar_fit_r11_v2.py')))
    stub=types.ModuleType('mathutils');stub.Vector=np.asarray;sys.modules['mathutils']=stub
    record=json.loads(observation.read_text(encoding='utf8'));payload=observation.with_suffix('.npz');data=np.load(payload,allow_pickle=False)
    if record['arraysSha256']!=a['_sha'](payload):raise RuntimeError('Actual probe payload hash mismatch')
    skin=[]
    for i in range(2):
        p=data[f'skin_{i}_world'];t=data[f'skin_{i}_triangles'];tp=p[t];keep=(tp.max(1)[:,2]>=1.23)&(tp.min(1)[:,2]<=1.31)
        skin.append((str(i),Surface(p,t[keep])))
    body=Surface(data['garment_0_failed_target'],data['garment_0_triangles'])
    results=[];arrays={};start=time.time()
    for i in (1,2):
        original=data[f'garment_{i}_original'];failed=data[f'garment_{i}_failed_target'];tri=data[f'garment_{i}_triangles'];_,basis,projector,_=a['_collar_basis']()
        projected=original+basis@(projector@(failed-original));old_cos=a['_normal_cosines'](original,failed,tri);smooth_cos=a['_normal_cosines'](original,projected,tri)
        print(json.dumps({'collar':i,'failedMinimumNormalCosine':float(old_cos.min()),'directSmoothMinimumNormalCosine':float(smooth_cos.min()),'failedFaces':np.flatnonzero(old_cos<=.02).tolist()}),flush=True)
        try:
            sewn=a['_sewn_edge_targets'](original[:49],data['garment_0_original'],data['garment_0_failed_target'],data['garment_0_triangles'])
            candidate,details=a['_project_collar'](original,original,tri,body,skin,sewn);guard=a['_shape_guard'](original,candidate,tri,str(i));arrays[f'collar_{i}_target']=candidate
            a['_transport'].__globals__['_bvh']=Surface
            stitch=data[f'garment_{i+2}_original'];stitch_tri=data[f'garment_{i+2}_triangles'];stitch_target=a['_transport'](stitch,original,candidate,tri)
            stitch_guard=a['_shape_guard'](stitch,stitch_target,stitch_tri,a['SEAMS'][i-1]);arrays[f'collar_{i}_stitch_target']=stitch_target
            edges=np.unique(np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
            crossings={}
            for surface_name,surface in [('body',body)]+skin:
                hits=[]
                for edge in edges:
                    if surface_name=='body' and np.max(edge//49)<3:continue
                    point=candidate[edge[0]];direction=candidate[edge[1]]-point;length=np.linalg.norm(direction)
                    h,n,face,distance=surface.ray_cast(point,direction/length,length)
                    if face is not None and 1e-5<distance<length-1e-5:hits.append(edge.tolist())
                crossings[surface_name]=hits
            results.append({'collar':i,'status':'offline-fit-passed','guard':guard,'stitchGuard':stitch_guard,'actualRawEdgesCrossingSurface':crossings,'details':details})
        except Exception as e:results.append({'collar':i,'status':'offline-fit-rejected','error':str(e)})
        print(json.dumps({k:v for k,v in results[-1].items() if k!='details'}),flush=True)
    report={'sourceSha256':record['sourceSha256'],'observationSha256':a['_sha'](observation),'payloadSha256':a['_sha'](payload),'helperSha256':a['_sha'](Path(__file__).with_name('collar_fit_r11_v2.py')),'elapsedSeconds':time.time()-start,'results':results,'savedBlend':False,'limitations':['Offline triangle oracle validates raw-cage fitting only. Actual Blender evaluation, solidify, stitches and appearance remain required.']}
    with output.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    if arrays:np.savez_compressed(output.with_suffix('.npz'),**arrays)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--observation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.observation,args.output)
