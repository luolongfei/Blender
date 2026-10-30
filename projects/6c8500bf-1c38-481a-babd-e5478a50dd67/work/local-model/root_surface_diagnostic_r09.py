"""Read-only scalp-root faceting and hypothetical smooth-surface clearance audit.

Call run() in a fresh background process opened on the explicit R09 source.
No vertices, attributes, normals, modifiers, shaders or hair are written.
The PN-triangle array is a diagnostic hypothesis derived from the ACTUAL head
vertices and vertex normals, not a fitted ellipsoid and not an adopted fix.
All 39,950-ish root vertices are examined. Hair clearance uses bounded,
separately reported original/appended strand samples; it is not full collision QA.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SHELL='R06 attached scalp root substrate'
SCALP='R06 continuous scalp flow'
ALPHA='R06_root_edge_alpha'
CENTER=np.array([0.,.00202,1.441285])


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _path(relative,suffix):
    path=(project_path()/relative).resolve();path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix!=suffix:raise ValueError('Unexpected diagnostic output type')
    return path


def _array(collection,prop,width,dtype=np.float32):
    a=np.empty(len(collection)*width,dtype);collection.foreach_get(prop,a);return a.reshape(-1,width)


def _world(a,matrix):
    m=np.array(matrix,dtype=float);return a@m[:3,:3].T+m[:3,3]


def _summary(a):
    a=np.asarray(a,dtype=float)
    if not a.size:return {'count':0}
    if not np.isfinite(a).all():raise RuntimeError('Non-finite geometric diagnostic')
    return {'count':a.size,'minimum':float(a.min()),'p05':float(np.quantile(a,.05)),
            'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),
            'p99':float(np.quantile(a,.99)),'maximum':float(a.max())}


def _barycentric(point,triangle):
    a,b,c=triangle;e0=b-a;e1=c-a;d=point-a
    aa=np.dot(e0,e0);ab=np.dot(e0,e1);bb=np.dot(e1,e1);den=aa*bb-ab*ab
    if abs(den)<1e-22:raise RuntimeError('Degenerate surface triangle')
    v=(bb*np.dot(d,e0)-ab*np.dot(d,e1))/den
    w=(aa*np.dot(d,e1)-ab*np.dot(d,e0))/den
    return np.array([1-v-w,v,w])


def _pn_point(vertices,normals,bary):
    """Cubic point-normal triangle position, using shared real corner normals."""
    a,b,c=vertices;na,nb,nc=normals;u,v,w=bary
    ab=(2*a+b-np.dot(b-a,na)*na)/3
    ba=(2*b+a-np.dot(a-b,nb)*nb)/3
    bc=(2*b+c-np.dot(c-b,nb)*nb)/3
    cb=(2*c+b-np.dot(b-c,nc)*nc)/3
    ca=(2*c+a-np.dot(a-c,nc)*nc)/3
    ac=(2*a+c-np.dot(c-a,na)*na)/3
    edge=(ab+ba+bc+cb+ca+ac)/6;vertex=(a+b+c)/3
    middle=edge+(edge-vertex)/2
    return (a*u**3+b*v**3+c*w**3+3*(ab*u*u*v+ba*u*v*v+bc*v*v*w+
            cb*v*w*w+ca*w*w*u+ac*u*u*w)+6*middle*u*v*w)


def _mesh_surface(obj,graph=None):
    evaluated=obj.evaluated_get(graph) if graph is not None else None
    mesh=evaluated.to_mesh() if evaluated else obj.data
    matrix=evaluated.matrix_world if evaluated else obj.matrix_world
    try:
        mesh.calc_loop_triangles()
        vertices=_world(_array(mesh.vertices,'co',3),matrix)
        triangles=np.asarray([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
        normals=_array(mesh.vertices,'normal',3).astype(float)@np.linalg.inv(np.array(matrix)[:3,:3])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        polygons=[tuple(p.vertices) for p in mesh.polygons]
        summary={'vertices':len(vertices),'polygons':len(polygons),'triangles':len(triangles),
                 'smoothPolygons':sum(p.use_smooth for p in mesh.polygons),
                 'hasCustomNormals':getattr(mesh,'has_custom_normals',None),
                 'bounds':[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()]}
        return vertices,triangles,normals,polygons,summary
    finally:
        if evaluated:evaluated.to_mesh_clear()


def _tree(vertices,triangles):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([Vector(v) for v in vertices],triangles.tolist(),all_triangles=True)


def _boundary(polygons):
    edges={}
    for face in polygons:
        for a,b in zip(face,face[1:]+face[:1]):
            key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
    if any(count>2 for count in edges.values()):raise RuntimeError('Nonmanifold root requires a separate topology audit')
    return np.asarray(sorted({v for edge,count in edges.items() if count==1 for v in edge}),np.int32)


def _patch_metrics(vertices,triangles):
    """Connected coplanar triangle groups; not a rounded-normal histogram."""
    tri=vertices[triangles];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(length[:,None],1e-20);area=length*.5
    parent=np.arange(len(triangles));edges={};angles=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=int(parent[i])
        return i
    threshold=math.cos(math.radians(.1))
    for i,indices in enumerate(triangles):
        for a,b in zip(indices,np.roll(indices,-1)):
            edge=tuple(sorted((int(a),int(b))))
            if edge not in edges:edges[edge]=i;continue
            j=edges[edge];dot=float(np.clip(np.dot(normals[i],normals[j]),-1,1));angles.append(math.degrees(math.acos(dot)))
            plane_error=float(np.max(np.abs((tri[j]-tri[i,0])@normals[i])))
            if dot>=threshold and plane_error<=1e-5:
                pi,pj=find(i),find(j)
                if pi!=pj:parent[pj]=pi
    groups={}
    for i in range(len(triangles)):groups.setdefault(find(i),[]).append(i)
    ordered=sorted(groups.values(),key=lambda ids:float(area[ids].sum()),reverse=True)
    largest=[]
    for ids in ordered[:12]:
        points=tri[ids].reshape(-1,3);bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()]
        largest.append({'triangles':len(ids),'areaSquareMeters':float(area[ids].sum()),'bounds':bounds,
                        'maximumAxisSpanMeters':float((points.max(axis=0)-points.min(axis=0)).max())})
    return {'criteria':{'adjacentNormalAngleDegrees':.1,'planeDistanceMeters':1e-5},
            'triangleCount':len(triangles),'connectedCoplanarGroups':len(groups),
            'adjacentTriangleAngleDegrees':_summary(angles),'largestPatches':largest,
            'patchesOverOneSquareCentimeter':sum(float(area[ids].sum())>1e-4 for ids in groups.values())}


def _smooth_hypothesis(root,root_normals,alpha,boundary,head,head_tri,head_normals):
    from mathutils import Vector
    tree=_tree(head,head_tri);proposed=[];signed_gap=[];distance=[];normal_angles=[];pn_bulge=[]
    for i,q in enumerate(root):
        hit,_,index,d=tree.find_nearest(Vector(q))
        if hit is None:raise RuntimeError('Actual head nearest-surface query missed')
        hit=np.asarray(hit);ids=head_tri[index];bary=_barycentric(hit,head[ids])
        n=bary@head_normals[ids];n/=max(np.linalg.norm(n),1e-12)
        if np.dot(n,hit-CENTER)<0:n=-n
        gap=float(np.dot(q-hit,n));smooth=_pn_point(head[ids],head_normals[ids],bary)
        # Keep the original signed normal stand-off and tangential offset; only
        # replace the triangle's flat interpolation with its measured-normal
        # cubic patch. Alpha fades the hypothesis to the unchanged open edge.
        delta=(smooth-hit)*float(alpha[i])**2
        proposed.append(q+delta);signed_gap.append(gap);distance.append(d)
        normal_angles.append(math.degrees(math.acos(float(np.clip(abs(np.dot(root_normals[i],n)),-1,1)))))
        pn_bulge.append(float(np.dot(smooth-hit,n)))
    proposed=np.array(proposed);proposed[boundary]=root[boundary]
    return proposed,{'rootToActualHeadDistanceMeters':_summary(distance),
        'rootToActualHeadSignedNormalGapMeters':_summary(signed_gap),
        'rootVersusInterpolatedHeadNormalDegrees':_summary(normal_angles),
        'headPNTriangleSignedBulgeMeters':_summary(pn_bulge),
        'hypotheticalRootDisplacementMeters':_summary(np.linalg.norm(proposed-root,axis=1)),
        'openBoundaryMaximumDisplacementMeters':float(np.abs(proposed[boundary]-root[boundary]).max())}


def _hair_sample(obj,first,last,max_curves=4096):
    selected=np.unique(np.linspace(first,last-1,min(last-first,max_curves),dtype=int));indices=[];curve_ids=[]
    for i in selected:
        curve=obj.data.curves[int(i)]
        for t in (0.,1/3,2/3,1.):
            indices.append(curve.first_point_index+round(t*(curve.points_length-1)));curve_ids.append(int(i))
    indices=np.asarray(indices,np.int32)
    points=_world(_array(obj.data.position_data,'vector',3)[indices],obj.matrix_world)
    radius=_array(obj.data.attributes['radius'].data,'value',1).ravel()[indices]
    return points,radius,indices,np.asarray(curve_ids,np.int32)


def _clearance(points,radii,root,triangles,alpha):
    from mathutils import Vector
    tree=_tree(root,triangles);clearance=[];opacity=[];distances=[]
    for q,radius in zip(points,radii):
        hit,n,index,d=tree.find_nearest(Vector(q))
        if hit is None:raise RuntimeError('Root clearance nearest query missed')
        hit=np.asarray(hit);n=np.asarray(n)
        if np.dot(n,hit-CENTER)<0:n=-n
        bary=_barycentric(hit,root[triangles[index]])
        clearance.append(float(np.dot(q-hit,n))-float(radius));opacity.append(float(bary@alpha[triangles[index]]));distances.append(d)
    return np.asarray(clearance),np.asarray(opacity),np.asarray(distances)


def _compare_hair(obj,root,proposal,triangles,alpha,offset):
    rows={};arrays={}
    for label,lo,hi in (('originalScalp',0,offset),('appendedPosterior',offset,len(obj.data.curves))):
        points,radius,indices,curve_ids=_hair_sample(obj,lo,hi)
        before,old_alpha,old_distance=_clearance(points,radius,root,triangles,alpha)
        after,new_alpha,new_distance=_clearance(points,radius,proposal,triangles,alpha)
        opaque=(new_alpha>=.95)&(new_distance<.01)
        newly_buried=opaque&(before>=0)&(after<0)
        worst=np.argsort(after)[:12]
        rows[label]={'strandRange':[lo,hi],'sampledCurves':len(np.unique(curve_ids)),'sampledPoints':len(points),
            'opaqueNearbySampleCount':int(opaque.sum()),'beforeClearanceMeters':_summary(before[opaque]),
            'hypothesisClearanceMeters':_summary(after[opaque]),'newlyBuriedSampleCount':int(newly_buried.sum()),
            'buriedBeforeCount':int((opaque&(before<0)).sum()),'buriedAfterCount':int((opaque&(after<0)).sum()),
            'worstSamples':[{'curve':int(curve_ids[i]),'pointIndex':int(indices[i]),'world':points[i].tolist(),
                'beforeMeters':float(before[i]),'afterMeters':float(after[i]),'rootAlpha':float(new_alpha[i]),
                'nearbyOpaque':bool(opaque[i])} for i in worst]}
        arrays.update({label+'_world':points.astype(np.float32),label+'_radius':radius,
                       label+'_point_indices':indices,label+'_curve_indices':curve_ids,
                       label+'_before_clearance':before.astype(np.float32),label+'_after_clearance':after.astype(np.float32)})
    return rows,arrays


def run(source_relative='work/reconstruction-r02/reference-posterior-groom-r09.blend',
        report_relative='work/reconstruction-r02/root-surface-diagnostic-r09.json',export_arrays=True):
    import bpy
    source=_path(source_relative,'.blend');output=_path(report_relative,'.json');numeric=output.with_suffix('.npz')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the exact independent source in background Object Mode')
    if output.exists() or export_arrays and numeric.exists():raise RuntimeError('Preserve existing diagnostic outputs; choose a new report path')
    obj=bpy.data.objects[SHELL];scalp=bpy.data.objects[SCALP];rig=bpy.data.objects['Character_Rig']
    if any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Root modifiers need an explicit evaluated-to-original mapping')
    if ALPHA not in obj.data.attributes:raise RuntimeError('Existing root edge alpha required')
    original=bpy.data.hair_curves.get(scalp.get('R09_posterior_original_data',''))
    if original is None:raise RuntimeError('Retained original scalp identifies the append boundary')
    offset=len(original.curves);pose=rig.data.pose_position;uv=_api('production_uv_r08');groom=_api('posterior_groom_r09')
    bpy.context.view_layer.update();protected=groom['_protected'](bpy,uv);source_sha=_sha(source)
    report=None;export={}
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update();graph=bpy.context.evaluated_depsgraph_get()
        root,root_tri,root_normals,polygons,root_summary=_mesh_surface(obj)
        head,head_tri,head_normals,_,head_summary=_mesh_surface(bpy.data.objects['R01 Head'],graph)
        alpha=_array(obj.data.attributes[ALPHA].data,'value',1).ravel().astype(float)
        boundary=_boundary(polygons)
        if not len(boundary):raise RuntimeError('Expected the open scalp substrate boundary')
        print('R09_ROOT_DIAGNOSTIC_PROJECTION '+str(len(root)),flush=True)
        proposal,comparison=_smooth_hypothesis(root,root_normals,alpha,boundary,head,head_tri,head_normals)
        selected=np.all(head[head_tri][:,:,2]>1.35,axis=1)
        print('R09_ROOT_DIAGNOSTIC_FACET_PATCHES',flush=True)
        root_patches=_patch_metrics(root,root_tri);head_patches=_patch_metrics(head,head_tri[selected])
        proposal_patches=_patch_metrics(proposal,root_tri)
        print('R09_ROOT_DIAGNOSTIC_HAIR_CLEARANCE',flush=True)
        hair,hair_arrays=_compare_hair(scalp,root,proposal,root_tri,alpha,offset)
        old_mesh=bpy.data.meshes.get(obj.get('R06_root_pre_feather_mesh',''))
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'blenderVersion':bpy.app.version_string,'rootObject':SHELL,'rootMesh':obj.data.name,
            'rootDataSha256':groom['_data_hash'](obj.data),'retainedPreFeatherMesh':old_mesh.name if old_mesh else None,
            'retainedPreFeatherVertexCount':len(old_mesh.vertices) if old_mesh else None,
            'rootSurface':root_summary,'evaluatedHead':head_summary,
            'headModifiers':[{'name':m.name,'type':m.type,'viewport':m.show_viewport,'render':m.show_render,
                             'levels':getattr(m,'levels',None),'renderLevels':getattr(m,'render_levels',None)} for m in bpy.data.objects['R01 Head'].modifiers],
            'rootCurrentCoplanarPatches':root_patches,'upperHeadCurrentCoplanarPatches':head_patches,
            'rootHypothesisCoplanarPatches':proposal_patches,'surfaceComparison':comparison,
            'sampledHairClearance':hair,'hypothesisAdopted':False,'sourceGeometryWritten':False,'savedBlend':False,
            'hypothesis':'Cubic point-normal interpolation of actual evaluated head triangles using actual smooth vertex normals; no ellipsoid. Preserve original per-vertex stand-off; alpha^2 taper; pin all open boundary points.',
            'limits':['Hair check samples up to 4096 original plus 4096 appended curves, four points each. Not all fibre points or continuous segment collision.',
                      'A smoother hypothetical root can intersect existing fibres that were laid on planar head triangles. Newly buried counts explicitly block blind adoption.',
                      'UV, alpha, material and all source data remain unchanged; proposed coordinates exist only in temporary arrays.',
                      'Any adopted root geometry change requires matched actual renders and runtime root normal rebake, even if UV/alpha remain identical.']}
        export={'root_world':root.astype(np.float32),'root_proposed_world':proposal.astype(np.float32),
            'root_triangles':root_tri,'root_alpha':alpha.astype(np.float32),'root_boundary_indices':boundary,
            'head_world':head.astype(np.float32),'head_triangles':head_tri,'head_normals':head_normals.astype(np.float32),**hair_arrays}
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    if groom['_protected'](bpy,uv)!=protected:raise RuntimeError('Original source data/pose changed during read-only diagnostic')
    if _sha(source)!=source_sha:raise RuntimeError('Source file changed externally during diagnostic')
    report['originalDataAndPoseUnchanged']=True
    if export_arrays:
        with numeric.open('xb') as stream:np.savez_compressed(stream,**export)
        report['diagnosticArrays']={'path':str(numeric.relative_to(project_path())),'sha256':_sha(numeric),
                                    'bytes':numeric.stat().st_size,'scope':'Temporary geometry hypothesis and finite samples, not an adopted production mesh.'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


if __name__=='__main__':print('Loaded only. Explicit run() is a read-only diagnostic; no automatic repair, save or render.')
