"""Read-only R11 neutral eye/oral audit and complete-head camera fitting.

Use the existing candidate rendering pipeline with fit_head_camera(). This
module does not render, change materials, create expressions, or save Blend.
The CLI temporarily evaluates armatures in REST, then restores their mode.
The eye and lip loops come from the actual evaluated topology, not inherited
indices from the old face, and not the first N vertices of a Multires mesh.
"""
import argparse
import collections
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
EYES={s:'R11 Adult sclera '+s for s in ('L','R')}
VIEWS={'front':(0,-1,0),'quarter':(.65,-1,0),'side':(1,0,0)}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _transform(points,matrix):
    return np.asarray(points)@matrix[:3,:3].T+matrix[:3,3]


def _matrix(head):
    matrix=np.asarray(json.loads(head['R11_female_to_target']),dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or np.linalg.det(matrix[:3,:3])<=0:
        raise RuntimeError('Explicit finite female body-space to target REST matrix required')
    return matrix


def _cycle(edges):
    adjacency=collections.defaultdict(set)
    for a,b in edges:adjacency[int(a)].add(int(b));adjacency[int(b)].add(int(a))
    if not adjacency or any(len(x)!=2 for x in adjacency.values()):
        raise RuntimeError('Evaluated landmark boundary is not a simple cycle')
    first=min(adjacency);result=[first];previous=None;current=first
    for _ in range(len(adjacency)+1):
        nxt=min(adjacency[current]-({previous} if previous is not None else set()))
        if nxt==first:break
        result.append(nxt);previous,current=current,nxt
    if len(result)!=len(adjacency):raise RuntimeError('Evaluated landmark has multiple cycles')
    return result


def _group_boundary(faces,groups,group):
    counts=collections.Counter()
    for face in faces[np.asarray(groups)==group]:
        for a,b in zip(face,np.roll(face,-1)):counts[tuple(sorted((int(a),int(b))))]+=1
    return _cycle([e for e,count in counts.items() if count==1])


def _split(ring,coordinates,a,b):
    start=ring.index(a);order=ring[start:]+ring[:start];end=order.index(b)
    paths=[order[:end+1],[order[0]]+list(reversed(order[end:]))]
    return sorted(paths,key=lambda p:float(coordinates[p,2].mean()),reverse=True)


def _polyline(points,count=41):
    points=np.asarray(points);d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if d[-1]<=0:raise RuntimeError('Degenerate landmark arc')
    return np.column_stack([np.interp(np.linspace(0,d[-1],count),d,points[:,i]) for i in range(3)])


def evaluated_landmarks(coordinates,faces,groups,cage):
    """Pure array topology extraction; coordinates and cage share body space.

    Boundary groups 2/3 each form a real 160-point level-three socket/outer-face
    interface. This stable control loop is not the visible wet-line contour.
    The oral ring is traced through quads from the actual 7/17 upper-lip edge.
    No numeric index correspondence with evaluated points is assumed.
    """
    eyes={}
    for side,group in (('L',2),('R',3)):
        ring=_group_boundary(faces,groups,group)
        inner=min(ring,key=lambda i:abs(coordinates[i,0]));outer=max(ring,key=lambda i:abs(coordinates[i,0]))
        upper,lower=_split(ring,coordinates,inner,outer)
        eyes[side]={'ring':ring,'upper':upper,'lower':lower,'inner':inner,'outer':outer}
    # Bounded patch follows actual cage lip position, not a fixed world box.
    seed=np.asarray(cage[8005]);half=np.array([.055,.045,.035])
    mask=np.all(np.abs(coordinates-seed)<=half,axis=1)
    selected=np.flatnonzero(mask[faces].all(axis=1))
    edge_faces=collections.defaultdict(list);adjacency=collections.defaultdict(set)
    for fi in selected:
        face=faces[fi]
        for a,b in zip(face,np.roll(face,-1)):
            a=int(a);b=int(b);edge_faces[tuple(sorted((a,b)))].append(int(fi))
            adjacency[a].add(b);adjacency[b].add(a)
    candidates=[edge for edge,fs in edge_faces.items()
                if len(fs)==2 and set(int(groups[i]) for i in fs)=={7,17}]
    if not candidates:raise RuntimeError('Actual upper lip 7/17 interface missing')
    edge=min(candidates,key=lambda e:float(np.linalg.norm(coordinates[list(e)].mean(0)-seed)))
    distance=float(np.linalg.norm(coordinates[list(edge)].mean(0)-seed))
    if distance>.006:raise RuntimeError('Actual upper-lip seed is more than 6 mm from current cage seed')
    ring=list(edge)
    for _ in range(2000):
        previous,current=ring[-2:];excluded={previous}
        for fi in edge_faces[tuple(sorted((previous,current)))]:
            face=faces[fi].tolist();j=face.index(current)
            excluded.update((face[j-1],face[(j+1)%len(face)]))
        choices=adjacency[current]-excluded
        if len(choices)!=1:raise RuntimeError('Evaluated lip quad tracing stopped at a pole or patch border')
        nxt=choices.pop()
        if nxt==ring[0]:break
        if nxt in ring:raise RuntimeError('Evaluated lip loop self-intersects topologically')
        ring.append(nxt)
    else:raise RuntimeError('Evaluated lip loop exceeded bounded tracing limit')
    left=min(ring,key=lambda i:coordinates[i,0]);right=max(ring,key=lambda i:coordinates[i,0])
    upper,lower=_split(ring,coordinates,left,right)
    return {'eyes':eyes,'mouth':{'ring':ring,'upper':upper,'lower':lower,
            'left':left,'right':right,'seedDistanceBodySpaceMeters':distance,'patchFaces':len(selected)}}


def lip_section_contacts(coordinates,faces,groups,cage,matrix):
    """Actual lip-body sections, independent of the 320-point control loop.

    The source lip bulges almost meet behind that loop: its roughly 3 mm arc
    distance is NOT a 3 mm open mouth. These bounded sections measure the real
    intervening surface and distinguish local crossings from positive gaps.
    They are not a complete three-dimensional collision/airtightness test.
    """
    seed=np.asarray(cage[8005]);q=coordinates[faces]
    base=(q[:,:,1].min(1)<seed[1]+.018)&(q[:,:,2].min(1)>seed[2]-.009)&(q[:,:,2].max(1)<seed[2]+.006)
    base&=np.isin(groups,[7,17,8,22])
    results=[]
    for sx in np.linspace(-.016,.016,9):
        selected=np.flatnonzero(base&(q[:,:,0].min(1)<=sx)&(q[:,:,0].max(1)>=sx))
        segments={0:[],1:[]}
        for fi in selected:
            # Split each quad to straight triangles for this geometric probe.
            for tri in (q[fi,[0,1,2]],q[fi,[0,2,3]]):
                points=[]
                for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                    if (a[0]-sx)*(b[0]-sx)<0:
                        points.append((a+(b-a)*(sx-a[0])/(b[0]-a[0]))[[1,2]])
                if len(points)==2:
                    segments[0 if groups[fi] in (7,17) else 1].append(np.asarray(points,dtype=float))
        if not all(segments.values()):raise RuntimeError('A bounded actual upper/lower lip section is missing')
        best=float('inf');nearest=None;crossings=[]
        for upper in segments[0]:
            for lower in segments[1]:
                aa=upper[1]-upper[0];bb=lower[1]-lower[0];system=np.column_stack([aa,-bb])
                if abs(np.linalg.det(system))>1e-20:
                    parameters=np.linalg.solve(system,lower[0]-upper[0])
                    if np.all(parameters>=0) and np.all(parameters<=1):
                        point=upper[0]+parameters[0]*aa
                        if not any(np.linalg.norm(point-p)<1e-7 for p in crossings):crossings.append(point)
                        best=0.;nearest=(point,point)
                for first,second,reverse in ((upper,lower,False),(lower,upper,True)):
                    edge=second[1]-second[0];denom=float(np.dot(edge,edge))
                    if denom<=1e-20:continue
                    for p in first:
                        other=second[0]+np.clip(np.dot(p-second[0],edge)/denom,0,1)*edge
                        distance=float(np.linalg.norm(p-other))
                        if distance<best:
                            best=distance;nearest=(other,p) if reverse else (p,other)
        points=np.asarray([[sx,*p] for p in nearest]);world=_transform(points,matrix)
        results.append({'bodySectionX':float(sx),'upperSegments':len(segments[0]),'lowerSegments':len(segments[1]),
            'minimumSurfaceGapMeters':float(np.linalg.norm(world[0]-world[1])),
            'upperNearestWorld':world[0].tolist(),'lowerNearestWorld':world[1].tolist(),
            'sectionCrossings':len(crossings),'crossingWorld':_transform(np.asarray([[sx,*p] for p in crossings]).reshape(-1,3),matrix).tolist()})
    return {'method':'Nine body-space sagittal cuts, actual evaluated quad triangles, upper lip sets7/17 versus lower22/8 inside a bounded perioral patch.',
            'sections':results,'limitations':'Crossings near the corners can include the joining lip surfaces. These sections do not prove whole-mouth watertightness or global self-collision absence.'}


def _eye_section_segments(coordinates,faces,sx,zlo,zhi,ymax):
    q=coordinates[faces]
    selected=(q[:,:,0].min(1)<=sx)&(q[:,:,0].max(1)>=sx)&(q[:,:,1].min(1)<ymax)
    selected&=(q[:,:,2].min(1)<=zhi)&(q[:,:,2].max(1)>=zlo)
    segments=[]
    for quad in q[selected]:
        for tri in (quad[[0,1,2]],quad[[0,2,3]]):
            points=[]
            for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                if (a[0]-sx)*(b[0]-sx)<0:
                    points.append((a+(b-a)*(sx-a[0])/(b[0]-a[0]))[[1,2]])
            if len(points)==2:segments.append(points)
    if not segments:raise RuntimeError('Actual eye/head section has no surface segments')
    return np.asarray(segments)


def frontal_eye_aperture_sections(head_coordinates,head_faces,eye_coordinates,eye_faces,matrix):
    """Visible globe from actual head/globe geometry along REST body -Y.

    A socket-interface control-ring distance must not be treated as physical
    lid contact. This complementary geometric visibility measurement retains
    the eyelid bulges that narrow the actual opening inside that control ring.
    """
    mn=eye_coordinates.min(0);mx=eye_coordinates.max(0);center=(mn+mx)*.5
    sections=[];step=.000025;zs=np.arange(mn[2],mx[2]+step*.5,step)
    def front(segments):
        dz=segments[:,1,1]-segments[:,0,1]
        valid=np.abs(dz)>1e-12
        t=(zs[:,None]-segments[None,:,0,1])/np.where(valid,dz,1.)[None,:]
        y=segments[None,:,0,0]+t*(segments[None,:,1,0]-segments[None,:,0,0])
        return np.where(valid[None,:]&(t>=0)&(t<=1),y,np.inf).min(axis=1)
    for offset in (-.006,-.003,0.,.003,.006):
        sx=float(center[0]+offset)
        hs=_eye_section_segments(head_coordinates,head_faces,sx,mn[2],mx[2],mx[1]+.005)
        es=_eye_section_segments(eye_coordinates,eye_faces,sx,mn[2],mx[2],mx[1]+.005)
        hy=front(hs);ey=front(es);indices=np.flatnonzero(np.isfinite(ey)&(ey<hy-1e-7))
        groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []
        runs=[]
        for ids in groups:
            if not len(ids):continue
            endpoints=np.asarray([[sx,ey[ids[0]],zs[ids[0]]],[sx,ey[ids[-1]],zs[ids[-1]]]])
            world=_transform(endpoints,matrix)
            runs.append({'bottomWorld':world[0].tolist(),'topWorld':world[1].tolist(),
                         'bodyVerticalHeightMeters':float(zs[ids[-1]]-zs[ids[0]]),
                         'targetVerticalHeightMeters':float((zs[ids[-1]]-zs[ids[0]])*np.linalg.norm(matrix[:3,2]))})
        sections.append({'bodyX':sx,'bodyOffsetFromEyeCenter':offset,'visibleRuns':runs})
    return {'method':'Five sagittal sections, actual evaluated head and sclera triangles; closest front surface along REST body -Y, sampled every 25 micrometers in source Z.',
            'sourceVerticalStepMeters':step,'targetVerticalStepMeters':step*float(np.linalg.norm(matrix[:3,2])),
            'sections':sections,'limitations':'Static geometric visibility, not an iris optical render, wet-line thickness measure, blink seal, or gaze test. Socket-interface loop is recorded separately.'}


def _evaluated_mesh(obj):
    import bpy
    graph=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(graph)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        co=np.empty((len(mesh.vertices),3),dtype=np.float64);mesh.vertices.foreach_get('co',co.ravel())
        corners=np.empty(len(mesh.loops),dtype=np.int32);mesh.loops.foreach_get('vertex_index',corners)
        starts=np.empty(len(mesh.polygons),dtype=np.int32);mesh.polygons.foreach_get('loop_start',starts)
        lengths=np.empty(len(mesh.polygons),dtype=np.int32);mesh.polygons.foreach_get('loop_total',lengths)
        groups=None;attr=mesh.attributes.get('.sculpt_face_set')
        if attr and attr.domain=='FACE' and attr.data_type=='INT':
            groups=np.empty(len(mesh.polygons),dtype=np.int32);attr.data.foreach_get('value',groups)
        faces=[corners[s:s+n].tolist() for s,n in zip(starts,lengths)]
        return {'world':_transform(co,np.asarray(ev.matrix_world,dtype=float)),
                'faces':faces,'groups':groups,'matrix':np.asarray(ev.matrix_world,dtype=float)}
    finally:ev.to_mesh_clear()


@contextlib.contextmanager
def rest_evaluation():
    """Only armature pose-position flags change; actions and bone poses stay intact."""
    import bpy
    armatures={o.data:o.data.pose_position for o in bpy.context.scene.objects if o.type=='ARMATURE'}
    try:
        for data in armatures:data.pose_position='REST';data.update_tag()
        bpy.context.view_layer.update()
        yield
    finally:
        for data,value in armatures.items():data.pose_position=value;data.update_tag()
        bpy.context.view_layer.update()


def fit_head_camera(camera,head_name=HEAD,view='side',padding=1.12):
    """Fit all actual evaluated head/neck points; caller sets REST and resolution.

    Does not touch lights, exposure, visibility, or source geometry. Keeps full
    ear, chin, posterior skull and the selected neck base in every view.
    """
    import bpy
    from mathutils import Matrix,Vector
    if view not in VIEWS or not 1.06<=padding<=1.3:raise ValueError('Explicit bounded head view required')
    head=bpy.data.objects[head_name];T=_matrix(head);mesh=_evaluated_mesh(head);points=mesh['world']
    body=_transform(points,np.linalg.inv(T))
    if body[:,2].min()<1.20:raise RuntimeError('Head Mask appears disabled; refusing full-body head fit')
    forward=T[:3,:3]@np.asarray(VIEWS[view],dtype=float);forward/=np.linalg.norm(forward)
    up=T[:3,2]/np.linalg.norm(T[:3,2]);right=np.cross(up,forward);right/=np.linalg.norm(right);up=np.cross(forward,right)
    rotation=np.column_stack([right,up,forward]);projected=points@rotation
    mn=projected.min(0);mx=projected.max(0);target=((mn+mx)*.5)@rotation.T
    camera.matrix_world=Matrix.Translation(Vector(target+forward*1.5))@Matrix(rotation.tolist()).to_4x4()
    camera.data.type='ORTHO';camera.data.dof.use_dof=False;camera.data.ortho_scale=1.
    bpy.context.view_layer.update()
    frame=camera.data.view_frame(scene=bpy.context.scene)
    halfx=max(abs(v.x) for v in frame);halfy=max(abs(v.y) for v in frame)
    camera.data.ortho_scale=padding*max((mx[0]-mn[0])/(2*halfx),(mx[1]-mn[1])/(2*halfy))
    bpy.context.view_layer.update()
    return {'view':view,'head':head_name,'evaluatedVertices':len(points),'boundsWorld':[points.min(0).tolist(),points.max(0).tolist()],
            'targetWorld':target.tolist(),'orthoScale':float(camera.data.ortho_scale),'padding':padding,
            'cameraMatrix':[list(r) for r in camera.matrix_world],'depthOfField':False,
            'scope':'All actual evaluated new head/neck points, including posterior skull; no eye-center crop.'}


def _co_hash(obj):
    a=np.empty((len(obj.data.vertices),3),dtype=np.float64);obj.data.vertices.foreach_get('co',a.ravel())
    return hashlib.sha256(a.tobytes()).hexdigest()


def _stats(values):
    a=np.asarray(values,dtype=float)
    return {'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}


def audit(source_relative,report_relative,expected_source_sha=None,head_name=HEAD,eye_names=None):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    P=project_path().resolve();source=(P/source_relative).resolve();report=(P/report_relative).resolve()
    source.relative_to(P);report.relative_to(P)
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Load the explicit source in a separate background Object Mode process')
    if report.exists():raise FileExistsError(report)
    source_sha=_sha(source)
    if expected_source_sha and source_sha!=expected_source_sha:raise RuntimeError('Explicit source SHA differs')
    head=bpy.data.objects[head_name];eye_names=eye_names or EYES
    eyes={side:bpy.data.objects[name] for side,name in eye_names.items()}
    if len(head.data.vertices)!=10582:raise RuntimeError('Original complete female cage index space required')
    multires=[m for m in head.modifiers if m.type=='MULTIRES']
    if len(multires)!=1 or not multires[0].show_viewport or multires[0].levels!=3 or multires[0].render_levels!=3:
        raise RuntimeError('Inspect unchanged authored Multires L3 in both viewport and render')
    if head.data.shape_keys and any(abs(k.value)>1e-7 for k in list(head.data.shape_keys.key_blocks)[1:]):
        raise RuntimeError('This audit is for the new neutral face, not unreviewed expression values')
    before={o.name:_co_hash(o) for o in [head]+list(eyes.values())};ids=set(bpy.data.objects.keys())
    frame=(bpy.context.scene.frame_current,bpy.context.scene.frame_subframe)
    result={'schemaVersion':1,'source':source.relative_to(P).as_posix(),'sourceSha256':source_sha,
            'head':head_name,'poseScope':'Temporary armature REST only, original actions and pose channels retained',
            'blender':bpy.app.version_string,'eyes':{},'appearanceAccepted':False,'expressionCompatibilityPassed':False}
    with rest_evaluation():
        T=_matrix(head);inv=np.linalg.inv(T);data=_evaluated_mesh(head)
        if data['groups'] is None:raise RuntimeError('Actual evaluated sculpt face sets required; no inferred landmark fallback')
        if any(len(f)!=4 for f in data['faces']):raise RuntimeError('Expected actual level-three all-quad surface')
        world=data['world'];body=_transform(world,inv);faces=np.asarray(data['faces'],dtype=np.int32)
        cage=np.asarray([v.co[:] for v in head.data.vertices],dtype=float)
        cage=_transform(_transform(cage,data['matrix']),inv)
        marks=evaluated_landmarks(body,faces,data['groups'],cage)
        result['femaleToTargetRestWorld']=T.tolist();result['evaluatedHead']={'vertices':len(world),'faces':len(faces),
            'boundsWorld':[world.min(0).tolist(),world.max(0).tolist()]}
        for side,obj in eyes.items():
            eye=_evaluated_mesh(obj);bvh=BVHTree.FromPolygons([Vector(p) for p in eye['world']],eye['faces'],all_triangles=False)
            if obj.parent is None or not obj.parent.get('R11_eye_fit_record'):
                raise RuntimeError('Explicit new adult-eye fit provenance required')
            fit=json.loads(obj.parent['R11_eye_fit_record'])
            bounds=np.asarray([eye['world'].min(0),eye['world'].max(0)])
            rest_error=float(np.max(np.abs(bounds-np.asarray(fit['target_sclera_world_bounds']))))
            if rest_error>2e-6:
                raise RuntimeError('New eye is not at its documented target REST placement: '+side+' '+str(rest_error))
            donor_index=obj.data.attributes.get('R11_donor_vertex_index')
            if donor_index is None or len(donor_index.data)!=546 or len(eye['world'])!=546:
                raise RuntimeError('Actual female eye vertex provenance changed')
            actual_ids=np.empty(546,dtype=np.int32);donor_index.data.foreach_get('value',actual_ids)
            if not np.array_equal(actual_ids,np.arange(546)):
                raise RuntimeError('Actual female eye vertex order changed')
            ring=marks['eyes'][side];rows=[]
            if len(ring['ring'])!=160:raise RuntimeError('Level-three eye boundary count differs from actual donor 160')
            for i in ring['ring']:
                p=Vector(world[i]);q,n,fi,d=bvh.find_nearest(p)
                if q is None:raise RuntimeError('Actual eye surface nearest query failed')
                signed=float(d)*(1. if (p-q).dot(n)>=0 else -1.)
                rows.append({'evaluatedHeadVertex':i,'pointWorld':list(p),'eyeNearestWorld':list(q),
                             'eyeFace':int(fi),'signedDistanceMeters':signed})
            upper=_polyline(world[ring['upper']]);lower=_polyline(world[ring['lower']])
            result['eyes'][side]={'object':obj.name,'evaluatedVertices':len(eye['world']),'evaluatedFaces':len(eye['faces']),
                'worldBounds':[eye['world'].min(0).tolist(),eye['world'].max(0).tolist()],
                'documentedRestBoundsMaximumErrorMeters':rest_error,
                'actualSocketInterfaceLoopVertices':len(ring['ring']),'innerControlCornerWorld':world[ring['inner']].tolist(),
                'outerControlCornerWorld':world[ring['outer']].tolist(),
                'signedSocketInterfaceToActualEyeTriangleDistanceMeters':_stats([r['signedDistanceMeters'] for r in rows]),
                'upperLowerArcDistanceMeters':_stats(np.linalg.norm(upper-lower,axis=1)),
                'signedDistanceConvention':'Negative follows the actual closest eye polygon normal into the eye. The socket/outer-face interface is a topology control loop, not the visible wet-line or proof of physical lid contact.',
                'frontalVisibleApertureSections':frontal_eye_aperture_sections(body,faces,_transform(eye['world'],inv),np.asarray(eye['faces'],dtype=np.int32),T),
                'loopSamples':rows,'upperArcWorld':upper.tolist(),'lowerArcWorld':lower.tolist()}
        mouth=marks['mouth']
        if len(mouth['ring'])!=320:raise RuntimeError('Level-three lip quad loop count differs from actual donor 320')
        upper=_polyline(world[mouth['upper']]);lower=_polyline(world[mouth['lower']]);gaps=np.linalg.norm(upper-lower,axis=1)
        result['mouth']={'actualLipControlLoopVertices':len(mouth['ring']),'method':'Closed quad edge loop from actual 7/17 upper lip interface; lower lip traced through jaw set22. This is a topology control loop, not the physical wet-line contact contour.',
            'seedDistanceBodySpaceMeters':mouth['seedDistanceBodySpaceMeters'],'patchFaces':mouth['patchFaces'],
            'controlRingWorld':world[mouth['ring']].tolist(),'upperArcWorld':upper.tolist(),'lowerArcWorld':lower.tolist(),
            'equalArcFractionControlDistanceMeters':_stats(gaps),'centralHalfControlDistanceMeters':_stats(gaps[10:31]),
            'gapInterpretation':'Paired control arcs can be about 3 mm apart while fuller lip surfaces almost touch farther inside. Never use this control distance as visible mouth opening or a direct closure displacement.',
            'actualLipSurfaceSections':lip_section_contacts(body,faces,data['groups'],cage,T),
            'oralSurfaceFaces':{str(g):int((data['groups']==g).sum()) for g in (7,8)},
            'expressionKeys':[k.name for k in head.data.shape_keys.key_blocks] if head.data.shape_keys else []}
    after={o.name:_co_hash(o) for o in [head]+list(eyes.values())}
    if before!=after or ids!=set(bpy.data.objects.keys()) or frame!=(bpy.context.scene.frame_current,bpy.context.scene.frame_subframe):
        raise RuntimeError('Read-only neutral audit changed source geometry, object IDs, or time')
    if _sha(source)!=source_sha:raise RuntimeError('Source disk file changed')
    result['sourceGeometryUnchanged']=True;result['sourceSaved']=False
    result['reviewRequired']=[
        'Neutral front/quarter/side: full skull, ear, jaw and neck, same new eyes and head; no DOF or exposure increase.',
        'Neutral eye close-up: eyelid thickness, canthi, sclera/iris exposure, focus and catchlight; distance alone is not appearance.',
        'Neutral mouth quarter: actual lip contact versus deliberate slight parting, corner continuity and visible oral depth; no black cut across cheeks.',
        'After new controls exist: closed blink front and quarter must seal without globe exposure or eyelid volume loss.',
        'After new controls exist: small smile and jaw-open at quarter/side must preserve lip rim, corner volume and jaw arc; inspect teeth/gums/tongue independently.',
        'No old blink/jaw/smile keys or old validation reports establish compatibility for this replacement head.'
    ]
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'report':report.relative_to(P).as_posix(),'sourceSha256':source_sha,
        'eyeLoops':{s:r['actualSocketInterfaceLoopVertices'] for s,r in result['eyes'].items()},
        'mouthLoop':result['mouth']['actualLipControlLoopVertices'],'sourceSaved':False},ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--source-sha')
    parser.add_argument('--report',required=True);parser.add_argument('--head',default=HEAD)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    audit(args.source,args.report,args.source_sha,args.head)


if __name__=='__main__':main()
