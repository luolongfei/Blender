"""Reversible adult neck proportion with exact-boundary bridge transport.

Only run on a fresh independent R11 source. Existing head keys, body, rig,
bridge topology/weights and materials are retained. The new shape key and the
copied bridge form ONE candidate and must be restored together. This does not
claim a deforming anatomical neck or an accepted appearance. Existing bridge
tangent defects are explicitly retained. The draft tangent_bridge function is
NOT used: actual c2 offline observations show that the current 432-to-128
transition needs a different topology to match tangents without intersections.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
BODY='R01 Head'
BRIDGE='R11 Continuous neck bridge'
KEY='R11 natural middle neck proportion'
MARKER='R11_neck_proportion_transport_candidate'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(data,field='co',width=3,dtype=np.float32):
    a=np.empty((len(data),width),dtype);data.foreach_get(field,a.ravel());return a


def _unit(a):
    a=np.asarray(a,float);n=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(n<1e-12):raise RuntimeError('Zero direction in actual neck geometry')
    return a/n


def _smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def _topology(mesh):
    return {'loops':_array(mesh.loops,'vertex_index',1,np.int32).ravel(),
        'face_starts':_array(mesh.polygons,'loop_start',1,np.int32).ravel(),
        'face_sizes':_array(mesh.polygons,'loop_total',1,np.int32).ravel()}


def _triangles(mesh):
    return np.asarray([(int(v[0]),int(v[k]),int(v[k+1]))
        for s,n in zip(mesh['face_starts'],mesh['face_sizes'])
        for v in [mesh['loops'][s:s+n]] for k in range(1,len(v)-1)],np.int32)


def neck_field(current_world,donor,strength):
    """Narrow lateral neck only; anatomically tagged face/oral/ear cage is fixed."""
    if not 0<=strength<=1:raise ValueError('Explicit strength in [0, 1] required')
    world=np.asarray(current_world,float);x,y,z=world.T
    skin=_api('natural_skin_adult_r11')
    protected=skin['_vertices_of_groups'](donor,{2,3,4,5,7,8,22})
    # Two true edge rings prevent the neck edit from reaching lip/jaw support.
    edges=donor['edges']
    for _ in range(2):
        grown=protected.copy();grown[edges[protected[edges].any(1)].ravel()]=True;protected=grown
    graph=skin['_graph'](world,edges,np.ones(len(world),bool))
    distance=skin['_distance'](graph,np.flatnonzero(protected),.020)
    protection_fade=_smooth(distance/.016)
    selection_path=project_path()/'work/reconstruction-r02/female-head-selection-r11.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if selection['source_npz_sha256']!=_sha(project_path()/selection['source_npz']):raise RuntimeError('Actual author head selection source changed')
    author_head=np.zeros(len(world),bool);author_head[np.asarray(selection['original_vertex_indices'],np.int32)]=True
    support=author_head.astype(float);grown=author_head.copy()
    # Multires evaluates the full donor cage before Mask. Keep two immediate
    # support rings with a smooth taper, but never edit distant hidden torso
    # or shoulder vertices merely because they have a similar Z coordinate.
    for weight in (.67,.17):
        expanded=grown.copy();expanded[edges[grown[edges].any(1)].ravel()]=True
        support[expanded&~grown]=weight;grown=expanded
    fade=_smooth((z-1.255)/.040)*(1.-_smooth((z-1.328)/.037))
    side=_smooth((abs(x)-.012)/.033)
    displacement=np.zeros_like(world)
    displacement[:,0]=-np.sign(x)*.006*fade*side*protection_fade*support*float(strength)
    displacement[protected]=0.
    displacement[(z<=1.255)|(z>=1.365)]=0.
    selected=np.linalg.norm(displacement,axis=1)>1e-12
    return displacement,{'strength':float(strength),'maximumSideReductionMeters':.006*float(strength),
        'affectedCageVertices':int(selected.sum()),'protectedCageVertices':int(protected.sum()),
        'protectedFaceOralEarExact':bool(np.all(displacement[protected]==0)),
        'protectedSurfaceFadeMeters':.016,
        'authorHeadSelectionSha256':_sha(selection_path),
        'outsideAuthorHeadSupportVerticesAffected':int((selected&~author_head).sum()),
        'distantHiddenBodyCageUnchanged':bool(np.all(displacement[~grown]==0)),
        'fieldWorldZ':[1.255,1.295,1.328,1.365],
        'maximumDisplacementMeters':float(np.linalg.norm(displacement,axis=1).max()),
        'scope':'Lateral lower/middle neck only. No Y/Z displacement; tagged jaw, oral cavity, sockets and ears plus two adjacency rings stay fixed.'}


def _geometry_guard(before,after,triangles,selected=None):
    tri=np.asarray(triangles,np.int32)
    if selected is not None:tri=tri[np.asarray(selected)[tri].any(1)]
    p=np.asarray(before)[tri];q=np.asarray(after)[tri]
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1)
    valid=a0>1e-12
    if not valid.any() or np.min(a1[valid])<1e-12:raise RuntimeError('Degenerate candidate neck triangles')
    align=np.sum(n0[valid]*n1[valid],axis=1)/(a0[valid]*a1[valid])
    if align.min()<=0:raise RuntimeError('Candidate reverses an actual neck triangle')
    return {'triangles':len(tri),'minimumOriginalToCandidateNormalCosine':float(align.min()),
        'areaRatioQuantiles':np.quantile(a1[valid]/a0[valid],[0,.05,.5,.95,1]).tolist()}


def _down_tangent(ring):
    tangent=_unit(np.roll(ring['points'],-1,axis=0)-np.roll(ring['points'],1,axis=0))
    direction=_unit(np.cross(tangent,ring['geometric_normals']))
    direction[direction[:,2]>0]*=-1
    return direction


def _incident_direction(mesh,ring):
    """Actual incident-face inward tangent, not a guessed normal direction."""
    ids=np.asarray(ring['ids']);excluded=set(map(int,ids));rows={int(i):set() for i in ids}
    for start,size in zip(mesh['face_starts'],mesh['face_sizes']):
        face=set(map(int,mesh['loops'][start:start+size]));intersection=face&excluded
        for i in intersection:rows[i].update(face-excluded)
    tangent=_unit(np.roll(ring['points'],-1,axis=0)-np.roll(ring['points'],1,axis=0))
    direction=[];fallback=[]
    normals=_down_tangent(ring)
    for k,i in enumerate(ids):
        row=sorted(rows[int(i)])
        if row:
            d=mesh['world'][row].mean(0)-mesh['world'][i]
            d-=tangent[k]*np.dot(d,tangent[k])
        else:d=np.zeros(3)
        if np.linalg.norm(d)<1e-12:
            # Explicit geometric-normal fallback at a topological corner.
            d=normals[k];fallback.append(int(i))
        d=d/np.linalg.norm(d)
        if d[2]>0:d=-d
        direction.append(d)
    return np.asarray(direction),fallback


def tangent_bridge(top_mesh,bottom_mesh,old_bridge,known_rings=None):
    """Same 432/7x128/224 topology; bounded Hermite surface interior."""
    core=_api('adult_neck_bridge_r11');periodic=core['_periodic']
    if known_rings is None:
        top=core['boundary_ring'](top_mesh);bottom=core['boundary_ring'](bottom_mesh,neck_only=True)
    else:
        top,bottom=known_rings
    nt=len(top['ids']);nb=len(bottom['ids']);n=128;rows=7
    if (nt,nb,len(old_bridge['world']))!=(432,224,1552):raise RuntimeError('Expected original R11 neck boundary/topology')
    for ring,mesh in ((top,top_mesh),(bottom,bottom_mesh)):
        ring['geometric_normals']=ring['normals'].copy()
        ring['normals']=mesh['shading_normals'][ring['ids']]
    da,fallback_top=_incident_direction(top_mesh,top);db,fallback_bottom=_incident_direction(bottom_mesh,bottom)
    a=np.arange(n)*(2*np.pi/n);ts=np.arange(1,rows+1)/8.
    pa=periodic(top['points'],top['angles'],a);pb=periodic(bottom['points'],bottom['angles'],a)
    chord=pb-pa;length=np.linalg.norm(chord,axis=1);gap=pa[:,2]-pb[:,2]
    if gap.min()<=.0001:raise RuntimeError('Actual neck boundaries cross in Z')
    da=_unit(periodic(da,top['angles'],a));db=_unit(periodic(db,bottom['angles'],a))
    # Preserve source tangent direction. Bound its Hermite derivative to the
    # actual gap/chord scale, rather than letting nearly horizontal tangents
    # produce an unbounded overshoot. No tangent is silently blended to pass.
    magnitude_a=np.minimum(gap/np.maximum(-da[:,2],.05),length*1.35)
    magnitude_b=np.minimum(gap/np.maximum(-db[:,2],.05),length*1.35)
    ma=da*magnitude_a[:,None];mb=db*magnitude_b[:,None]
    interior=[]
    for t in ts:
        interior.append((2*t**3-3*t**2+1)*pa+(t**3-2*t**2+t)*ma+(-2*t**3+3*t**2)*pb+(t**3-t**2)*mb)
    candidate=np.vstack([top['points'],*interior,bottom['points']])
    triangles=_triangles(old_bridge)
    guard=_geometry_guard(old_bridge['world'],candidate,triangles)
    intersections=_api('neck_bridge_geometry_audit_r11')['triangle_intersections'](candidate,triangles)
    if len(intersections['strictCrossingPairs']) or len(intersections['coplanarOverlapPairs']):
        raise RuntimeError('Hermite bridge intersects; source retained: '+json.dumps({
            'strict':intersections['strictCrossingPairs'][:8].tolist(),
            'coplanar':intersections['coplanarOverlapPairs'][:8].tolist()}))
    normals=np.zeros_like(candidate);p=candidate[triangles];fn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for k in range(3):np.add.at(normals,triangles[:,k],fn)
    normals=_unit(normals)
    # Endpoint normals exactly follow the actual source; inner normals are
    # generated by the new geometry. No flat override on the interior.
    normals[:nt]=top['normals'];normals[-nb:]=bottom['normals']
    return candidate,normals,{'geometryGuard':guard,'topBoundaryIds':top['ids'].tolist(),
        'bottomBoundaryIds':bottom['ids'].tolist(),'topNormalFallbackVertexIds':fallback_top,
        'bottomNormalFallbackVertexIds':fallback_bottom,'strictCrossingCount':0,'coplanarOverlapCount':0,
        'maxBridgeDisplacementMeters':float(np.linalg.norm(candidate-old_bridge['world'],axis=1).max()),
        'method':'Actual incident-face tangents with explicit geometric-normal corner fallback; bounded cubic Hermite between unchanged source boundary cycles, same original 7 interior rings and unchanged bridge topology/weights.'}


def _snapshot(obj):
    import bpy
    core=_api('adult_neck_bridge_r11');mesh,_,normals=core['_snapshot'](bpy,obj)
    mesh['shading_normals']=normals;return mesh


def transport_bridge(before_head,after_head,bottom_mesh,old_bridge):
    """Move the old top boundary with the head; original lower 224 stay fixed."""
    core=_api('adult_neck_bridge_r11');periodic=core['_periodic']
    before=core['boundary_ring'](before_head);after=core['boundary_ring'](after_head)
    bottom=core['boundary_ring'](bottom_mesh,neck_only=True)
    if not np.array_equal(before['ids'],after['ids']):
        # The canonical angular start can move by one vertex during narrowing.
        where={int(i):k for k,i in enumerate(after['ids'])}
        if set(where)!=set(map(int,before['ids'])):raise RuntimeError('Head boundary topology changed')
        after_points=np.asarray([after['points'][where[int(i)]] for i in before['ids']])
    else:after_points=after['points']
    nt=len(before['ids']);nb=len(bottom['ids']);old=np.asarray(old_bridge['world'])
    if (nt,nb,len(old))!=(432,224,1552):raise RuntimeError('Original bridge topology changed')
    if np.max(np.linalg.norm(old[:nt]-before['points'],axis=1))>1e-6:raise RuntimeError('Actual old bridge/head vertex correspondence differs')
    if np.max(np.linalg.norm(old[-nb:]-bottom['points'],axis=1))>1e-6:raise RuntimeError('Actual old bridge/body vertex correspondence differs')
    delta=after_points-before['points'];candidate=old.copy();candidate[:nt]=after_points
    angles=np.arange(128)*2*np.pi/128;row_delta=periodic(delta,before['angles'],angles)
    for row in range(7):
        t=(row+1)/8.;candidate[nt+row*128:nt+(row+1)*128]+=row_delta*(1.-_smooth(t))
    if not np.array_equal(candidate[-nb:],old[-nb:]):raise RuntimeError('Original lower bridge boundary moved')
    triangles=_triangles(old_bridge);guard=_geometry_guard(old,candidate,triangles)
    audit=_api('neck_bridge_geometry_audit_r11')['triangle_intersections'](candidate,triangles)
    if len(audit['strictCrossingPairs']) or len(audit['coplanarOverlapPairs']):raise RuntimeError('Transported bridge intersects; source retained')
    normals=np.zeros_like(candidate);p=candidate[triangles];fn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for k in range(3):np.add.at(normals,triangles[:,k],fn)
    normals=_unit(normals);normals[:nt]=after_head['shading_normals'][before['ids']];normals[-nb:]=bottom_mesh['shading_normals'][bottom['ids']]
    return candidate,normals,{'geometryGuard':guard,'topBoundaryIds':before['ids'].tolist(),
        'bottomBoundaryIds':bottom['ids'].tolist(),'originalBottomBoundaryMaximumDisplacementMeters':0.,
        'strictCrossingCount':0,'coplanarOverlapCount':0,
        'maxBridgeDisplacementMeters':float(np.linalg.norm(candidate-old,axis=1).max()),
        'method':'Exact actual new head boundary; periodic smooth transport of original 7x128 interior, original body boundary/topology/weights fixed.',
        'tangentContinuityRepaired':False,
        'knownLimitation':'Existing head/bridge and body/bridge geometric tangent defects remain. Matching custom shading normals does not establish true continuity.'}


def _weights(obj):
    return {'groups':[(g.name,g.index,g.lock_weight) for g in obj.vertex_groups],
        'values':[[(g.group,float(g.weight)) for g in v.groups] for v in obj.data.vertices]}


def apply(strength):
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.context.scene.get(MARKER):raise RuntimeError('Fresh independent background candidate required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if 'R11_low_neck_contour_candidate' not in bpy.context.scene:raise RuntimeError('Requires the actual saved low-neck base')
    head=bpy.data.objects[HEAD];bridge=bpy.data.objects[BRIDGE];body=bpy.data.objects[BODY]
    if len(head.data.vertices)!=10582 or len(bridge.data.vertices)!=1552:raise RuntimeError('Source topology changed')
    if bridge.data.shape_keys is not None:raise RuntimeError('Preserve existing bridge shape work; independent merge required')
    uv=_api('production_uv_r08');jaw=_api('adult_jaw_contour_r11');skin=_api('natural_skin_adult_r11');binding=_api('face_fibre_binding_r11')
    donor,_,provenance=skin['_donor']();keys_before=jaw['_keys'](head.data);weights_before=_weights(bridge)
    body_hash=_api('posterior_groom_r09')['_data_hash'](body.data);pose=uv['_pose_contract']();source_sha=_sha(source)
    basis=_array(head.data.shape_keys.reference_key.data);current,mix=uv['current_mix'](head)
    T=np.asarray(json.loads(head['R11_female_to_target']),float);world=current@T[:3,:3].T+T[:3,3]
    displacement,report=neck_field(world,donor,strength);delta=displacement@np.linalg.inv(T[:3,:3]).T
    head_triangles=_triangles({'loops':donor['corner_vertices'],'face_starts':donor['face_starts'],'face_sizes':donor['face_lengths']})
    raw_guard=_geometry_guard(world,world+displacement,head_triangles,np.linalg.norm(displacement,axis=1)>1e-12)
    old_mesh=bridge.data;old_fake=old_mesh.use_fake_user;old_mesh.use_fake_user=True
    key=None;new_mesh=None;success=False;active=head.active_shape_key_index
    try:
        with _api('face_contact_audit_r11')['rest_evaluation']():before_head=_snapshot(head)
        key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key;key.value=1.;key.slider_min=0.;key.slider_max=1.
        key.data.foreach_set('co',(basis+delta).astype(np.float32).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
        with _api('face_contact_audit_r11')['rest_evaluation']():
            top_mesh=_snapshot(head);bottom_mesh=_snapshot(body);old_snapshot=_snapshot(bridge)
            candidate,normals,build=transport_bridge(before_head,top_mesh,bottom_mesh,old_snapshot)
            new_mesh=old_mesh.copy();new_mesh.name='R11 natural neck proportion transported bridge';bridge.data=new_mesh
            if _weights(bridge)!=weights_before:raise RuntimeError('Mesh copy altered original bridge weight/name mapping')
            matrix=np.asarray(bridge.matrix_world,float);inverse=np.linalg.inv(matrix)
            local=candidate@inverse[:3,:3].T+inverse[:3,3]
            new_mesh.vertices.foreach_set('co',local.astype(np.float32).ravel())
            normal_local=_unit(normals@matrix[:3,:3]);new_mesh.update()
            new_mesh.normals_split_custom_set([tuple(normal_local[i]) for i in _topology(new_mesh)['loops']])
            bpy.context.view_layer.update();actual=_snapshot(bridge)
            error=float(np.linalg.norm(actual['world']-candidate,axis=1).max())
            if error>4e-7:raise RuntimeError('Candidate REST evaluated bridge differs from intended geometry')
            if not np.array_equal(actual['loops'],old_snapshot['loops']):raise RuntimeError('Bridge topology changed')
        after=jaw['_keys'](head.data)
        for name,before in keys_before.items():
            now=after[name]
            if not np.array_equal(before['coords'],now['coords']) or any(before[k]!=now[k] for k in before if k!='coords'):raise RuntimeError('Original head key changed: '+name)
        if _weights(bridge)!=weights_before or _api('posterior_groom_r09')['_data_hash'](body.data)!=body_hash:raise RuntimeError('Body or original bridge weights changed')
        if uv['_pose_contract']()!=pose or _sha(source)!=source_sha:raise RuntimeError('Original pose/source changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
            provenance=provenance,currentMix=mix,rawGeometryGuard=raw_guard,bridgeBuild=build,
            candidateBridgeRestErrorMeters=error,originalBridgeData=old_mesh.name,originalBridgeFakeUser=old_fake,
            candidateBridgeData=new_mesh.name,originalBridgeWeights=weights_before,key=KEY,keyValue=1.,
            keySha256=skin['_ah'](_array(key.data)),originalActiveShapeKeyIndex=active,
            candidateBridgeSha256=_api('posterior_groom_r09')['_data_hash'](new_mesh),
            actualFibreBindingValidation=binding['validate'](neutral=False),savedBlend=False,appearanceAccepted=False,
            limits=['New front/side/quarter and actual neck pose review required.',
                'Original geometric tangent seams remain; this candidate improves width, not seam topology.',
                'Material graphs and every bridge colour/roughness/rest-coordinate attribute are retained; this geometry pass does not claim to fix shader seams.',
                'Head is still rigidly parented; anatomical neck deformation and full dynamic approval remain separate work.',
                'Restore the head key and copied bridge together; merely turning the key off disconnects the candidate bridge.'])
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            bridge.data=old_mesh;old_mesh.use_fake_user=old_fake
            if key is not None and head.data.shape_keys and KEY in head.data.shape_keys.key_blocks:head.shape_key_remove(head.data.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[HEAD];bridge=bpy.data.objects[BRIDGE]
    key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['keySha256']:raise RuntimeError('Preserve subsequently edited neck key before restore')
    if _api('posterior_groom_r09')['_data_hash'](bridge.data)!=report['candidateBridgeSha256']:raise RuntimeError('Preserve subsequently edited bridge before restore')
    bridge.data.use_fake_user=True;bridge.data=bpy.data.meshes[report['originalBridgeData']]
    bridge.data.use_fake_user=report['originalBridgeFakeUser'];head.shape_key_remove(key)
    head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    if _weights(bridge)!=report['originalBridgeWeights']:
        # JSON changes tuples to lists; compare canonical serialized values.
        if json.dumps(_weights(bridge),sort_keys=True)!=json.dumps(report['originalBridgeWeights'],sort_keys=True):raise RuntimeError('Restored original bridge weights differ')
    return {'restored':True,'savedBlend':False}


def prepare(source_relative,output_relative,strength):
    import bpy
    root=(project_path()/'work/reconstruction-r02').resolve()
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();report_path=output.with_suffix('.neck-continuity.json')
    for p in (source,output):p.relative_to(root)
    if source.suffix!='.blend' or output.suffix!='.blend' or source==output or output.exists() or report_path.exists():raise RuntimeError('New explicit independent output required')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh specified source required')
    report=apply(strength)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NECK_CONTINUITY_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--strength',type=float,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.strength)
