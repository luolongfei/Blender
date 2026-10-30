"""Offline exact local Catmull-Clark L1/limit transfer reconstruction.

Match predicted local positions against actual Blender-observed positions
before using the same linear operator to diagnose a wrist correction. This is
not an inferred acceptance result and does not launch or write Blender.
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def topology(loops,starts,sizes,nv):
    faces=[loops[int(s):int(s+n)].tolist() for s,n in zip(starts,sizes)]
    edge_map={};edges=[];edge_faces=[];vertex_edges=[[] for _ in range(nv)];vertex_faces=[[] for _ in range(nv)];face_edges=[]
    for fi,face in enumerate(faces):
        local=[]
        for a,b in zip(face,face[1:]+face[:1]):
            pair=(min(a,b),max(a,b));index=edge_map.get(pair)
            if index is None:
                index=len(edges);edge_map[pair]=index;edges.append(pair);edge_faces.append([])
                vertex_edges[a].append(index);vertex_edges[b].append(index)
            edge_faces[index].append(fi);vertex_faces[a].append(fi);local.append(index)
        face_edges.append(local)
    ne=len(edges);quads=[]
    for fi,(face,es) in enumerate(zip(faces,face_edges)):
        for i,a in enumerate(face):quads.append((a,nv+es[i],nv+ne+fi,nv+es[i-1]))
    return {'faces':faces,'edges':np.asarray(edges,np.int32),'edgeFaces':edge_faces,'vertexEdges':vertex_edges,
            'vertexFaces':vertex_faces,'quads':np.asarray(quads,np.int32),'nv':nv,'ne':ne}


def subdivide_limit(values,top):
    nv=top['nv'];edges=top['edges'];faces=top['faces'];columns=values.shape[1]
    face_points=np.asarray([values[f].mean(0) for f in faces]);edge_points=np.empty((len(edges),columns),np.float64)
    midpoint=values[edges].mean(1)
    for i,(a,b) in enumerate(edges):
        fs=top['edgeFaces'][i]
        edge_points[i]=(values[a]+values[b]+face_points[fs].sum(0))/4 if len(fs)==2 else midpoint[i]
    points=np.empty_like(values)
    for i in range(nv):
        es=top['vertexEdges'][i];fs=top['vertexFaces'][i];n=len(es)
        if n==0:points[i]=values[i];continue
        if any(len(top['edgeFaces'][e])!=2 for e in es):points[i]=np.nan;continue
        points[i]=(face_points[fs].mean(0)+2*midpoint[es].mean(0)+(n-3)*values[i])/n
    level=np.concatenate((points,edge_points,face_points));quads=top['quads']
    new_edges=np.unique(np.sort(np.concatenate([quads[:,[0,1]],quads[:,[1,2]],quads[:,[2,3]],quads[:,[3,0]]]),axis=1),axis=0)
    neighbor_sum=np.zeros_like(level);degree=np.zeros(len(level),np.int32)
    np.add.at(neighbor_sum,new_edges[:,0],level[new_edges[:,1]]);np.add.at(neighbor_sum,new_edges[:,1],level[new_edges[:,0]])
    np.add.at(degree,new_edges[:,0],1);np.add.at(degree,new_edges[:,1],1)
    opposite_sum=np.zeros_like(level)
    for c in range(4):np.add.at(opposite_sum,quads[:,c],level[quads[:,(c+2)%4]])
    n=degree[:,None];limit=(n*n*level+4*neighbor_sum+opposite_sum)/(n*(n+5))
    return level,limit


def face_limit_stencil(top,face_index):
    """Sparse original-cage weights for an L1 face point at the limit."""
    cache={}
    def add(out,source,factor):
        for key,value in source.items():out[key]=out.get(key,0.)+factor*value
    def face(fi):return {int(v):1/len(top['faces'][fi]) for v in top['faces'][fi]}
    def edge(ei):
        a,b=top['edges'][ei];out={int(a):.25,int(b):.25}
        for fi in top['edgeFaces'][ei]:add(out,face(fi),.25)
        return out
    def vertex(vi):
        if vi in cache:return cache[vi]
        es=top['vertexEdges'][vi];fs=top['vertexFaces'][vi];n=len(es);out={vi:(n-3)/n}
        for fi in fs:add(out,face(fi),1/(n*len(fs)))
        for ei in es:
            a,b=top['edges'][ei];add(out,{int(a):.5,int(b):.5},2/(n*len(es)))
        cache[vi]=out;return out
    points=top['faces'][face_index];n=len(points);out={};den=n*(n+5)
    add(out,face(face_index),n*n/den)
    edge_lookup={tuple(pair):i for i,pair in enumerate(top['edges'])}
    for a,b in zip(points,points[1:]+points[:1]):add(out,edge(edge_lookup[(min(a,b),max(a,b))]),4/den)
    for vi in points:add(out,vertex(vi),1/den)
    return out


def analyze():
    folder=project_path()/'work/reconstruction-r02'
    probe=json.loads((folder/'hand-integrated-wrist-probe-r11-v3.json').read_text(encoding='utf-8'))
    parent=json.loads((folder/'hand-integrated-wrist-fit-plan-r11-v2.json').read_text(encoding='utf-8'))
    v4=json.loads((folder/'hand-integrated-wrist-fit-plan-r11-v4.json').read_text(encoding='utf-8'))
    with np.load(project_path()/probe['arrays']) as observed,np.load(project_path()/parent['arrays']) as p,np.load(project_path()/v4['arrays']) as q:
        nv=probe['bodyBeforeIntegration']['rawVertices'];top=topology(observed['raw_body_loops'],observed['raw_body_polygonStarts'],observed['raw_body_polygonSizes'],nv)
        values=np.zeros((nv,6),np.float64);values[:,:3]=np.nan;values[observed['raw_R_ids'],:3]=observed['raw_R_mix']
        values[p['R_ids'],3:]=q['R_delta'].astype(np.float64)-p['R_delta'].astype(np.float64)
        level,limit=subdivide_limit(values,top);finite=np.flatnonzero(np.all(np.isfinite(limit[:,:3]),axis=1))
        core=observed['relaxed_R_body_core'];rest=observed['relaxed_R_body_unposed'][core]
        rows=[]
        for index in (767,916):
            dist=np.linalg.norm(limit[finite,:3]-rest[index],axis=1);j=int(finite[np.argmin(dist)])
            record={'coreIndex':index,'actualBodyIndex':probe['poses'][0]['sides']['R']['coreOriginalVertexIds'][index],
                         'matchedL1Index':j,'L1Role':'vertex' if j<nv else 'edge' if j<nv+top['ne'] else 'face',
                         'limitPositionErrorMeters':float(dist.min()),'actualRestPosition':rest[index].tolist(),
                         'predictedLimitPosition':limit[j,:3].tolist(),'v4PredictedRestDisplacement':limit[j,3:].tolist(),
                         'v4PredictedDisplacementMeters':float(np.linalg.norm(limit[j,3:]))}
            if j>=nv+top['ne']:
                stencil=face_limit_stencil(top,j-nv-top['ne']);predicted=sum(weight*values[vi] for vi,weight in stencil.items())
                if np.max(np.abs(predicted-limit[j]))>1e-12:raise RuntimeError('Sparse stencil differs from the independently evaluated linear operator')
                ordered=sorted(stencil.items(),key=lambda pair:-pair[1])
                record.update({'originalFaceIndex':j-nv-top['ne'],'originalFaceVertexIds':top['faces'][j-nv-top['ne']],
                    'supportCount':len(stencil),'stencilSum':sum(stencil.values()),'topRawSupports':[
                        {'id':vi,'weight':weight,'restPosition':values[vi,:3].tolist(),'v4RawMovementMeters':float(np.linalg.norm(values[vi,3:]))}
                        for vi,weight in ordered[:8]],
                    'completeRawStencil':[[vi,weight] for vi,weight in ordered]})
            rows.append(record)
        package=json.loads((project_path()/probe['package']).read_text(encoding='utf-8'));bones=package['bones']
        lower=np.asarray(bones['rightLowerArm']['pose'])@np.linalg.inv(bones['rightLowerArm']['rest'])
        hand=np.asarray(bones['rightHand']['pose'])@np.linalg.inv(bones['rightHand']['rest'])
        index=767;point=rest[index];actual_point=observed['relaxed_R_body_actual'][core][index]
        h=(hand@np.r_[point,1])[:3];l=(lower@np.r_[point,1])[:3]
        hand_weight=float(np.dot(actual_point-l,h-l)/np.dot(h-l,h-l))
        displacement=np.asarray(rows[0]['v4PredictedRestDisplacement']);old=observed['relaxed_R_hand_positions']
        distal=np.abs(observed['relaxed_R_handRest_positions'][:,0])>.68;X=old[distal].astype(np.float64)
        projection=[]
        for pose in probe['poses']:
            prefix=pose['pose'].replace('-','_')+'_R';Y=observed[prefix+'_hand_positions'][distal].astype(np.float64)
            xm=X.mean(0);ym=Y.mean(0);u,_,vt=np.linalg.svd((X-xm).T@(Y-ym));rotation=vt.T@u.T;offset=ym-rotation@xm
            matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=offset;current_hand=matrix@hand
            old_predicted=hand_weight*(current_hand@np.r_[point,1])[:3]+(1-hand_weight)*(lower@np.r_[point,1])[:3]
            old_actual=observed[prefix+'_body_actual'][observed[prefix+'_body_core']][index]
            moved=(hand_weight*current_hand[:3,:3]+(1-hand_weight)*lower[:3,:3])@displacement
            normal=observed[prefix+'_closestNormal'][index];projection.append({'pose':pose['pose'],
                'distalHandRigidFitMaximumErrorMeters':float(np.linalg.norm(X@rotation.T+offset-Y,axis=1).max()),
                'bodyPointReconstructionErrorMeters':float(np.linalg.norm(old_predicted-old_actual)),
                'predictedWorldDisplacement':moved.tolist(),'actualNearestOutwardNormal':normal.tolist(),
                'predictedClearanceGainMeters':float(-moved@normal)})
        result={'sourceObservation':probe['source'],'finiteLocalLimitVertices':len(finite),'matchedRows':rows,
                'worstPointInferredActualLinearHandWeight':hand_weight,'poseProjection':projection,
                'observedV4ClearanceGainsMeters':[.00003139604814350605,.000031410665542352944,.00003136043233198462],
                'diagnosis':'Actual worst point is a subdivided face point dominated by raw triangle 3143/3144/3145. A spatial brush centered on that new face point scarcely moved its true raw support. X-lock is not the primary attenuation.',
                'scope':'Offline subdivision transfer diagnosis. Exact applicability requires observed point matches at float precision. No actual new pose pass claim.'}
        report=folder/'hand-wrist-subdivision-transfer-r11-v2.json'
        if not report.exists():
            with report.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        print(json.dumps({**result,'matchedRows':[{k:v for k,v in row.items() if k!='completeRawStencil'} for row in rows]},ensure_ascii=False,indent=2))
        return result,top,values,level,limit


if __name__=='__main__':analyze()
