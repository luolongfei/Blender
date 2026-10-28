"""Local fabric allowance against the complete body in fixed motion probes.

This does not simulate cloth or claim arbitrary-motion collision freedom.
Original cloth UVs/weights and the complete anatomical surfaces are retained.
"""
import json
from pathlib import Path
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def _matrix(values):
    return Matrix([values[i:i+4] for i in range(0,16,4)]).transposed()


def _blend(weights, matrices):
    result=Matrix(((0,0,0,0),)*4)
    for name,weight in weights.items():
        result += matrices.get(name,Matrix.Identity(4))*weight
    return result


def fit(plans, anatomy, probe_path):
    probes=json.loads(Path(probe_path).read_text())
    poses=[{'name':'source bind','matrices':{}}]
    for row in probes['poses']:
        poses.append({'name':row['name'],'matrices':{
            name:_matrix(values) for name,values in row['matrices'].items()}})
    # All seam copies are one unknown so fitting cannot reopen a sewn edge.
    positions=[];weights=[];members=[];indices={};triangles=[]
    for obj,mesh,skin in plans:
        if not (obj.name.startswith('Blouse_') or obj.name.startswith('Sock_')):
            continue
        lookup={}
        for vertex,w in zip(mesh.vertices,skin):
            key=tuple(round(v,7) for v in vertex.co)
            if key not in indices:
                indices[key]=len(positions);positions.append(vertex.co.copy())
                weights.append(w);members.append([])
            index=indices[key];lookup[vertex.index]=index
            members[index].append((mesh,vertex.index))
        mesh.calc_loop_triangles()
        triangles.extend(tuple(lookup[i] for i in t.vertices) for t in mesh.loop_triangles)
    start=[p.copy() for p in positions]
    active=[]
    for p in positions:
        x,y,z=p;ax=abs(x)
        active.append(z<.076 or (1.14<z<1.32 and .11<ax<.49))
    adjacency=[set() for p in positions]
    for tri in triangles:
        for i in tri:adjacency[i].update(j for j in tri if i!=j)
    active_triangles=[t for t in triangles if all(active[i] for i in t)]
    body_points=[];body_weights=[];body_tris=[]
    for source in anatomy:
        offset=len(body_points)
        body_points.extend(source['points']);body_weights.extend(source['weights'])
        body_tris.extend(tuple(offset+i for i in t) for t in source['triangles'])
    for pose in poses:
        pose['skinMatrices']=[_blend(w,pose['matrices']) for w in weights]
        pose['inverseLinear']=[m.to_3x3().inverted_safe() for m in pose['skinMatrices']]
        points=[_blend(w,pose['matrices'])@p for p,w in zip(body_points,body_weights)]
        pose['tree']=BVHTree.FromPolygons(points,body_tris,all_triangles=True)
    history=[]
    for iteration in range(24):
        correction=[Vector((0,0,0)) for p in positions]
        counts=[0. for p in positions]
        priorities=[0. for p in positions]
        worst={}
        for pose in poses:
            posed=[m@p for m,p in zip(pose['skinMatrices'],positions)]
            maximum=0.;violations=0
            for index,p in enumerate(posed):
                if not active[index]:continue
                hit,normal,_,distance=pose['tree'].find_nearest(p)
                signed=(p-hit).dot(normal)
                margin=.0016 if positions[index].z>.1 else .0009
                deficit=margin-signed
                if deficit<=.00002:continue
                maximum=max(maximum,deficit);violations+=1
                if positions[index].z<.1 or abs(positions[index].x)>.28:
                    correction[index]+=pose['inverseLinear'][index]@(normal*deficit)
                    counts[index]+=1
                elif deficit>priorities[index]:
                    correction[index]=pose['inverseLinear'][index]@(normal*deficit)
                    priorities[index]=deficit
                    counts[index]=1
            for tri in active_triangles:
                p=sum((posed[i] for i in tri),Vector())/3
                hit,normal,_,distance=pose['tree'].find_nearest(p)
                signed=(p-hit).dot(normal)
                margin=.0016 if positions[tri[0]].z>.1 else .0009
                deficit=margin-signed
                if deficit<=.00002:continue
                maximum=max(maximum,deficit);violations+=1
                for index in tri:
                    if positions[index].z<.1 or abs(positions[index].x)>.28:
                        correction[index]+=pose['inverseLinear'][index]@(normal*deficit)
                        counts[index]+=1
                    elif deficit>priorities[index]:
                        correction[index]=pose['inverseLinear'][index]@(normal*deficit)
                        priorities[index]=deficit
                        counts[index]=1
            worst[pose['name']]={'maximumDeficitMeters':maximum,'probes':violations}
        history.append(worst)
        print('CLOTH_FIT',iteration,worst,flush=True)
        if max(v['maximumDeficitMeters'] for v in worst.values())<.00005:break
        raw=[d/max(c,1) for d,c in zip(correction,counts)]
        # Spread added ease to adjacent rings instead of isolated spike edits.
        smooth=[]
        for i,d in enumerate(raw):
            nearby=[raw[j] for j in adjacency[i] if active[j]]
            average=sum(nearby,Vector())/len(nearby) if nearby else Vector()
            smooth.append(d*.8+average*.2 if counts[i] else average*.35)
        for i,delta in enumerate(smooth):
            if not active[i]:continue
            if delta.length>.003:delta=delta.normalized()*.003
            positions[i]+=delta*.9
    # A folded sleeve's inner elbow can lie between two differently moving
    # skin surfaces. Moving the rest fabric alone then alternates between
    # opposing closest normals. Solve only the local upper/lower-arm blend
    # with the actual linear derivative, keeping rest shape and total weight.
    elbow=[.32<abs(p.x)<.46 and 1.17<p.z<1.30 for p in positions]
    elbow_triangles=[t for t in triangles if all(elbow[i] for i in t)]
    original_elbow={i:weights[i].get(('left' if positions[i].x>=0 else 'right')+'LowerArm',0)
                    for i in range(len(positions)) if elbow[i]}
    weight_history=[]
    for iteration in range(24):
        delta=[0.]*len(positions);count=[0]*len(positions);worst={}
        for pose in poses[1:]:
            posed=[pose['skinMatrices'][i]@p for i,p in enumerate(positions)]
            derivatives={}
            for i in original_elbow:
                side='left' if positions[i].x>=0 else 'right'
                upper=pose['matrices'].get(side+'UpperArm',Matrix.Identity(4))
                lower=pose['matrices'].get(side+'LowerArm',Matrix.Identity(4))
                derivatives[i]=lower@positions[i]-upper@positions[i]
            maximum=0;violations=0
            for tri in [(i,) for i in original_elbow]+elbow_triangles:
                point=sum((posed[i] for i in tri),Vector())/len(tri)
                hit,normal,_,_=pose['tree'].find_nearest(point)
                deficit=.00125-(point-hit).dot(normal)
                if deficit<.00003:continue
                gradients=[normal.dot(derivatives[i])/len(tri) for i in tri]
                denominator=sum(g*g for g in gradients)
                if denominator<1e-9:continue
                maximum=max(maximum,deficit);violations+=1
                for i,g in zip(tri,gradients):
                    delta[i]+=max(-.025,min(.025,deficit*g/denominator))
                    count[i]+=1
            worst[pose['name']]={'maximumDeficitMeters':maximum,'probes':violations}
        weight_history.append(worst)
        print('ELBOW_WEIGHT_FIT',iteration,worst,flush=True)
        if max((r['maximumDeficitMeters'] for r in worst.values()),default=0)<.00005:
            break
        raw=[d/max(1,c) for d,c in zip(delta,count)]
        for i in original_elbow:
            side='left' if positions[i].x>=0 else 'right'
            upper,lower=side+'UpperArm',side+'LowerArm'
            w=weights[i];total=w.get(upper,0)+w.get(lower,0)
            neighbors=[raw[j] for j in adjacency[i] if elbow[j]]
            average=sum(neighbors)/len(neighbors) if neighbors else 0
            step=raw[i]*.8+average*.2
            base=original_elbow[i]
            desired=max(0.,min(total,base+.12,max(base-.12,w.get(lower,0)+step*.8)))
            w[lower]=desired;w[upper]=total-desired
            for pose in poses:
                pose['skinMatrices'][i]=_blend(w,pose['matrices'])
                pose['inverseLinear'][i]=pose['skinMatrices'][i].to_3x3().inverted_safe()
    rows=[];largest=[]
    for i,p in enumerate(positions):
        delta=(p-start[i]).length
        if delta:
            for mesh,index in members[i]:mesh.vertices[index].co=p
            rows.append(delta)
            largest.append({'distanceMeters':delta,'before':list(start[i]),
                            'after':list(p)})
    for _,mesh,_ in plans:mesh.update()
    return {'method':'local seam-coherent fabric allowance; maximum-violation projection over fixed linear-skin probes',
            'probeSource':probes['source'],'poses':[p['name'] for p in poses],
            'modifiedVertices':len(rows),'maximumDisplacementMeters':max(rows,default=0),
            'meanModifiedDisplacementMeters':sum(rows)/max(1,len(rows)),
            'largestChanges':sorted(largest,key=lambda row:row['distanceMeters'],reverse=True)[:8],
            'elbowWeightRefinement':{'method':'bounded local linear-skin weight derivative',
                'maximumAbsoluteWeightChange':max((abs(weights[i].get(
                    ('left' if positions[i].x>=0 else 'right')+'LowerArm',0)-value)
                    for i,value in original_elbow.items()),default=0),
                'history':weight_history},
            'history':history,'bodyUnchanged':True}
