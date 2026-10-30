"""Offline topology evidence from the actual HBM female export; no Blender.

Indices refer to the complete original 10582-point mesh, before any Mask or
crop. Labels are geometric control candidates, not pre-existing animation or
proof that the authored level-three surface has closed eyelids/lips.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path


def run():
    folder=project_path()/'work/reconstruction-r02';source=folder/'female-basemesh-actual-r11.json'
    arrays=folder/'female-basemesh-actual-r11.npz';out=folder/'female-face-landmarks-r11.json'
    if out.exists():raise FileExistsError(out)
    meta=json.loads(source.read_text(encoding='utf-8'))
    digest=hashlib.sha256(arrays.read_bytes()).hexdigest()
    if digest!=meta['npz_sha256']:raise RuntimeError('Actual female array identity changed')
    with np.load(arrays) as data:
        v=data['object_00_local'].astype(float);corners=data['object_00_corner_vertices']
        faces=[corners[s:s+n].tolist() for s,n in zip(data['object_00_face_starts'],data['object_00_face_lengths'])]
        groups=data['object_00_face_sets'];edge_faces=collections.defaultdict(list);adj=[set() for _ in v]
        for i,face in enumerate(faces):
            for a,b in zip(face,face[1:]+face[:1]):
                edge_faces[tuple(sorted((a,b)))].append(i);adj[a].add(b);adj[b].add(a)
        def cycle(edges):
            ring=collections.defaultdict(set)
            for a,b in edges:ring[a].add(b);ring[b].add(a)
            if any(len(n)!=2 for n in ring.values()):raise RuntimeError('Expected simple boundary cycle')
            first=min(ring);result=[first];last=None;current=first
            for _ in range(len(ring)+1):
                nxt=min(ring[current]-({last} if last is not None else set()))
                if nxt==first:break
                result.append(nxt);last,current=current,nxt
            if len(result)!=len(ring):raise RuntimeError('Boundary has more than one component')
            return result
        def loop(start,nxt):
            result=[start,nxt]
            for _ in range(1000):
                previous,current=result[-2:];exclude={previous}
                for fi in edge_faces[tuple(sorted((previous,current)))]:
                    face=faces[fi];j=face.index(current);exclude.update((face[j-1],face[(j+1)%len(face)]))
                choices=adj[current]-exclude
                if len(choices)!=1:return result,{'closed':False,'stopVertex':current,'degree':len(adj[current]),'branches':sorted(choices)}
                target=choices.pop()
                if target==start:return result,{'closed':True}
                if target in result:raise RuntimeError('Unexpected loop intersection')
                result.append(target)
            raise RuntimeError('Unbounded loop')
        def split(ring,start,end):
            a=ring.index(start);ordered=ring[a:]+ring[:a];b=ordered.index(end)
            paths=[ordered[:b+1],[ordered[0]]+list(reversed(ordered[b:]))]
            upper=max(paths,key=lambda ids:v[ids,2].mean());lower=min(paths,key=lambda ids:v[ids,2].mean())
            return {'upper':upper,'lower':lower}
        def describe(ring):
            q=v[ring]
            return {'sourceVertexIndices':ring,'sourceLocalCoordinates':q.tolist(),
                    'boundsLocal':[q.min(0).tolist(),q.max(0).tolist()]}
        eyes={}
        for side,group,obj_index in (('L',2,1),('R',3,2)):
            edges=[e for e,fs in edge_faces.items() if len(fs)==2 and set(int(groups[i]) for i in fs)=={group,17}]
            ring=cycle(edges);inner=min(ring,key=lambda i:abs(v[i,0]));outer=max(ring,key=lambda i:abs(v[i,0]))
            paths=split(ring,inner,outer)
            world=data[f'object_{obj_index:02d}_world'].astype(float)
            inv=np.linalg.inv(meta['objects'][0]['matrix_world']);eye=world@inv[:3,:3].T+inv[:3,3]
            center=(eye.min(0)+eye.max(0))*.5;radii=(eye.max(0)-eye.min(0))*.5
            normalized=np.linalg.norm((v[ring]-center)/radii,axis=1);clearance=(normalized-1)*radii.mean()
            eyes[side]={**describe(ring),'socketFaceSet':group,'faceSetInterface':[group,17],
                'innerCanthusCandidate':inner,'outerCanthusCandidate':outer,'orderedUpperInnerToOuter':paths['upper'],
                'orderedLowerInnerToOuter':paths['lower'],'sourceEyeObject':meta['objects'][obj_index]['name'],
                'sourceEyeBoundsLocalToBody':[eye.min(0).tolist(),eye.max(0).tolist()],
                'boundsEllipsoidCenter':center.tolist(),'boundsEllipsoidRadii':radii.tolist(),
                'rawLoopEllipsoidClearanceMeters':{'min':float(clearance.min()),'max':float(clearance.max()),
                    'median':float(np.median(clearance)),'perVertex':clearance.tolist()},
                'scope':'Raw cage loop against the real eye bounds ellipsoid proxy. This is not the level-three eyelid surface or an exact eye-triangle intersection test.'}
        opening,status=loop(8005,10541);middle,middle_status=loop(8006,10565);outer,outer_status=loop(8007,10485)
        if not all(x['closed'] for x in (status,middle_status,outer_status)):raise RuntimeError('Observed lip loops stopped being closed')
        if [len(x) for x in (opening,middle,outer)]!=[40,40,40]:raise RuntimeError('Observed lip loop topology changed')
        left=min(opening,key=lambda i:v[i,0]);right=max(opening,key=lambda i:v[i,0]);opening_paths=split(opening,left,right)
        # Flood the actual two quad strips between opening and outer vermilion.
        blocked={tuple(sorted((a,b))) for ring in (opening,outer) for a,b in zip(ring,ring[1:]+ring[:1])}
        seed=next(i for i,face in enumerate(faces) if 8005 in face and 8006 in face)
        pending=[seed];support={seed}
        while pending:
            fi=pending.pop();face=faces[fi]
            for a,b in zip(face,face[1:]+face[:1]):
                edge=tuple(sorted((a,b)))
                if edge in blocked:continue
                for neighbor in edge_faces[edge]:
                    if neighbor not in support:support.add(neighbor);pending.append(neighbor)
            if len(support)>200:raise RuntimeError('Lip strip escaped its verified boundary')
        support_vertices=sorted({i for fi in support for i in faces[fi]})
        upper_inner,upper_inner_status=loop(8004,10574);lower_inner,lower_inner_status=loop(7986,6670)
        caps=sorted({i for fs in edge_faces.values() if len(fs)==2 and set(int(groups[j]) for j in fs)=={1,8}
                     for i in fs if groups[i]==1})
        mouth={'openingRimCandidate':{**describe(opening),'leftCornerCandidate':left,'rightCornerCandidate':right,
                    'orderedUpperLeftToRight':opening_paths['upper'],'orderedLowerLeftToRight':opening_paths['lower']},
            'vermilionMidLoop':describe(middle),'outerVermilionLoopCandidate':describe(outer),
            'vermilionStripFaceIndices':sorted(support),'vermilionStripVertexIndices':support_vertices,
            'lipColorSupportPolicy':'Use only this two-quad-strip anatomical region, fading at the outer loop. Do not colour an axis-aligned Gaussian box across perioral skin.',
            'innerUpperRightHalfArch':{**describe(upper_inner),'status':upper_inner_status},
            'innerLowerRightHalfArch':{**describe(lower_inner),'status':lower_inner_status},
            'oralUpperFaceIndices':np.flatnonzero(groups==7).tolist(),'oralLowerFaceIndices':np.flatnonzero(groups==8).tolist(),
            'requiredPosteriorClosureFaceIndices':caps,
            'rawCentralLipGapMeters':{'openingRim8005to7985':float(np.linalg.norm(v[8005]-v[7985])),
                'innerFold8004to7986':float(np.linalg.norm(v[8004]-v[7986]))},
            'interpretation':'A deep oral bag and real lip strips exist. The raw cage central lips are not exactly closed. Neutral closure and subsequent jaw motion need new evaluated checks; these are not inherited animation controls.'}
        result={'schemaVersion':1,'sourceObservation':str(source.relative_to(project_path())),
            'sourceObservationSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sourceArraysSha256':digest,
            'sourceBundleSha256':meta['source_sha256'],'mesh':'GEO-body_female_realistic','originalVertexCount':len(v),
            'indexSpace':'Original complete female mesh, preserved by post-Multires Mask; do not use these IDs on reordered crops without explicit old_to_new mapping.',
            'eyes':eyes,'mouth':mouth,'existingShapeKeys':0,'bindingOrExpressionCompatibilityPassed':False,
            'limitations':['Geometry labels are source-topology candidates; authored Multires3 silhouette and contact must be examined.',
                'The full-body UV includes multiple tiles; unchanged UV cannot be assumed to be a packed 0..1 facial atlas.',
                'The female asset contains no blink/jaw/smile keys or current rig weights. Build new controls after neutral likeness is accepted, without retaining failed old shapes for compatibility.']}
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'report':str(out.relative_to(project_path())),'eyeLoops':{k:len(x['sourceVertexIndices']) for k,x in eyes.items()},
        'lipStripFaces':len(support),'lipStripVertices':len(support_vertices),'posteriorMouthCaps':caps},ensure_ascii=False))
    return result


if __name__=='__main__':run()
