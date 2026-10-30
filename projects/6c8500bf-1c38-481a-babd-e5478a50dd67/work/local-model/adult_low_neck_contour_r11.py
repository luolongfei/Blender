"""Independent low-neck contour repair after the actual R11 bridge review.

The previous bridge joins endpoints but leaves a shoulder ledge and posterior
point. This candidate changes the lower donor neck and rebuilds the annulus
from its ACTUAL evaluated edge. The head Mask and all head topology stay fixed,
so existing brow/lash triangle bindings keep their indices. No import-time bpy.
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
HEAD='R11 Female head study';BRIDGE='R11 Continuous neck bridge';OLD='R01 Head'
KEY='R11 continuous low neck contour';MARKER='R11_low_neck_contour_candidate'
ENVELOPE='work/reconstruction-r02/adult-low-neck-envelope-plan-r11.json'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a


def _smooth(x):
    x=np.clip(x,0.,1.);return x*x*(3-2*x)


def field(world):
    """One smooth low-neck field derived from the observed section envelope."""
    record=json.loads((project_path()/ENVELOPE).read_text());rows=np.asarray(record['rows'])
    points=np.asarray(world,float);x,y,z=points.T
    side=np.interp(z,rows[:,0],rows[:,3],left=0.,right=0.)
    side*=1.-_smooth((z-1.309)/.007);side[z<=1.26]=0.
    posterior=np.interp(z,rows[:,0],rows[:,4],left=0.,right=0.)
    delta=np.zeros_like(points)
    delta[:,0]=-np.sign(x)*side*_smooth((abs(x)-.020)/.040)
    delta[:,1]=-posterior*_smooth((y-.005)/.040)
    # Exact head/face/body exclusion, with zero-valued envelope at both Z ends.
    delta[z<=1.255]=0.;delta[z>=1.355]=0.
    return delta


def plan(basis,T):
    skin=_api('natural_skin_adult_r11');donor,_,provenance=skin['_donor']()
    basis=np.asarray(basis,np.float32);T=np.asarray(T,float)
    if basis.shape!=(10582,3):raise RuntimeError('Complete original female cage required')
    world=basis@T[:3,:3].T+T[:3,3];displacement=field(world);delta=displacement@np.linalg.inv(T[:3,:3]).T
    selected=np.linalg.norm(delta,axis=1)>1e-10
    edges=donor['edges'];before=basis[edges[:,1]].astype(float)-basis[edges[:,0]];after=before+delta[edges[:,1]]-delta[edges[:,0]]
    affected=selected[edges].any(1);ratio=np.linalg.norm(after,axis=1)/np.maximum(np.linalg.norm(before,axis=1),1e-12)
    tris=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face=donor['corner_vertices'][start:start+count]
        if selected[face].any():tris.extend((int(face[0]),int(face[k]),int(face[k+1])) for k in range(1,len(face)-1))
    tris=np.asarray(tris,np.int32);a=basis[tris].astype(float);b=(basis+delta)[tris]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);aa=np.linalg.norm(na,axis=1);bb=np.linalg.norm(nb,axis=1);valid=aa>1e-12
    alignment=np.sum(na[valid]*nb[valid],axis=1)/(aa[valid]*bb[valid])
    if alignment.min()<=0 or not np.isfinite(delta).all():raise RuntimeError('Raw low-neck field folds a face')
    if np.linalg.norm(displacement,axis=1).max()>.023:raise RuntimeError('Low-neck displacement exceeds the explicitly reviewed envelope')
    return delta.astype(np.float32),{'provenance':provenance,'envelopePlan':ENVELOPE,'envelopeSha256':_sha(project_path()/ENVELOPE),
        'rawAffectedVertices':int(selected.sum()),'maximumWorldDisplacementMeters':float(np.linalg.norm(displacement,axis=1).max()),
        'componentMaximumWorldMeters':np.abs(displacement).max(0).tolist(),
        'rawEdgeRatioQuantiles':np.quantile(ratio[affected],[0,.05,.5,.95,1]).tolist(),
        'rawNormalAlignmentMinimum':float(alignment.min()),'headMaskTopologyUnchanged':True,
        'scope':'Lateral lower neck below 1.316 m and posterior lower neck below 1.355 m. The visible face/eyes/mouth, skull, retained original body and rig are not edited.',
        'appearanceAccepted':False,'actualMultiresAndBridgePending':True}


def build_low_bridge(top,bottom):
    core=_api('adult_neck_bridge_r11');_periodic=core['_periodic'];_unit=core['_unit'];TAU=2*np.pi
    # Keep both true boundary cycles. A separate uniformly spaced interior
    # avoids extruding the old seam's local angular backtracks into the neck.
    n=128;rows_count=7;a=np.arange(n)*TAU/n
    pa=_periodic(top['points'],top['angles'],a);pb=_periodic(bottom['points'],bottom['angles'],a)
    gap=pa[:,2]-pb[:,2]
    if gap.min()<=.0001:raise RuntimeError('Upper/lower neck order crosses; needs another anatomical fit')
    ts=np.linspace(0.,1.,rows_count+2)[1:-1];nt=len(top['points']);nb=len(bottom['points'])
    vertices=np.vstack([top['points']]+[(1-t)*pa+t*pb for t in ts]+[bottom['points']]);faces=[]
    def zipper(upper_start,upper_angles,lower_start,lower_angles):
        i=j=0;nu=len(upper_angles);nl=len(lower_angles)
        while i<nu or j<nl:
            next_u=upper_angles[(i+1)%nu]+(TAU if i+1>=nu else 0.) if i<nu else float('inf')
            next_l=lower_angles[(j+1)%nl]+(TAU if j+1>=nl else 0.) if j<nl else float('inf')
            u=upper_start+i%nu;l=lower_start+j%nl
            if abs(next_u-next_l)<1e-10:
                faces.append((u,l,lower_start+(j+1)%nl,upper_start+(i+1)%nu));i+=1;j+=1
            elif next_u<next_l:faces.append((u,l,upper_start+(i+1)%nu));i+=1
            else:faces.append((u,l,lower_start+(j+1)%nl));j+=1
    zipper(0,top['angles'],nt,a)
    for row in range(rows_count-1):zipper(nt+row*n,a,nt+(row+1)*n,a)
    lower_start=nt+rows_count*n;zipper(nt+(rows_count-1)*n,a,lower_start,bottom['angles'])
    triangles=np.asarray([(f[0],f[k],f[k+1]) for f in faces for k in range(1,len(f)-1)],np.int32)
    edges=np.unique(np.sort(np.vstack([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1),axis=0)
    ends=edges.ravel();other=edges[:,::-1].ravel();degree=np.bincount(ends,minlength=len(vertices))
    initial=vertices.copy()
    # Finite, positive-weight fairing of new interior only. Neither endpoint
    # cycle, original head nor torso is modified. This removes seam-induced
    # zigzags rather than changing the old source to satisfy a validator.
    for iteration in range(50):
        sums=np.zeros_like(vertices);np.add.at(sums,ends,vertices[other]);target=sums/degree[:,None]
        vertices[nt+n:lower_start-n]+=.5*(target[nt+n:lower_start-n]-vertices[nt+n:lower_start-n])
    t=np.r_[np.zeros(nt),np.repeat(ts,n),np.ones(nb)]
    angles=np.r_[top['angles'],np.tile(a,rows_count),bottom['angles']]
    normals=np.zeros_like(vertices);normals[:nt]=top['normals'];normals[lower_start:]=bottom['normals']
    p=vertices[triangles];fn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(fn,axis=1)*.5
    if area.min()<1e-12:raise RuntimeError('Degenerate neck bridge triangle')
    audit=_api('neck_bridge_geometry_audit_r11')['triangle_intersections'](vertices,triangles)
    if len(audit['strictCrossingPairs']) or len(audit['coplanarOverlapPairs']):
        raise RuntimeError('Neck bridge intersects: '+str(audit['strictCrossingPairs'][:12].tolist())+' coplanar '+str(audit['coplanarOverlapPairs'][:12].tolist()))
    cent=p.mean(1);axis=top['centerXY'][None,:]*(1-t[triangles].mean(1)[:,None])+bottom['centerXY'][None,:]*t[triangles].mean(1)[:,None]
    radial=np.c_[cent[:,:2]-axis,np.zeros(len(cent))]
    radial_cos=np.einsum('ij,ij->i',_unit(fn),_unit(radial))
    sizes=np.asarray([len(f) for f in faces],np.int32);starts=np.r_[0,np.cumsum(sizes[:-1])].astype(np.int32)
    return ({'vertices':vertices,'loops':np.concatenate([np.asarray(f,np.int32) for f in faces]),
        'face_starts':starts,'face_sizes':sizes,'t':t,'angles':angles,'boundary_normals':normals,
        'top_boundary_ids':np.arange(nt,dtype=np.int32),'bottom_boundary_ids':np.arange(lower_start,len(vertices),dtype=np.int32),
        'top_angles':top['angles'],'bottom_angles':bottom['angles'],'top_source_ids':top['ids'],'bottom_source_ids':bottom['ids']},
        {'vertices':len(vertices),'faces':len(faces),'topBoundaryCount':nt,'bottomBoundaryCount':nb,
         'minimumTriangleAreaSquareMeters':float(area.min()),'minimumRadialNormalCosineDiagnosticOnly':float(radial_cos.min()),
         'strictTriangleCrossingCount':len(audit['strictCrossingPairs']),'coplanarOverlapCount':len(audit['coplanarOverlapPairs']),
         'selfIntersectionMethod':audit['method'],'maximumNewInteriorFairingMeters':float(np.linalg.norm(vertices-initial,axis=1).max()),
         'originalBoundaryMaximumDisplacementMeters':0.,'verticalGapRangeMeters':[float(gap.min()),float(gap.max())],
         'method':'Exact original 432/224 boundary cycles, seven uniformly parameterized 128-point interior rings, finite positive-weight interior fairing with the first and last collar rows fixed against seam-notch folding, ordered unequal-count annuli. Endpoint shading normals match actual source; exact tangent continuity and dynamic poses require visual review.'})


def _path(relative):
    path=(project_path()/relative).resolve();path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix!='.blend':raise ValueError('Explicit independent Blend required')
    return path


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.context.scene.get(MARKER):raise RuntimeError('Independent background candidate required')
    if 'R11_continuous_neck_bridge' not in bpy.context.scene:raise RuntimeError('Requires the actually saved R11 neck bridge')
    head=bpy.data.objects[HEAD];bridge=bpy.data.objects[BRIDGE];body=bpy.data.objects[OLD]
    if len(head.data.vertices)!=10582 or len(bridge.data.vertices)!=1552:raise RuntimeError('Inspect changed head/bridge topology before continuing')
    core=_api('adult_neck_bridge_r11');jaw=_api('adult_jaw_contour_r11');skin=_api('natural_skin_adult_r11')
    qa=_api('face_contact_audit_r11');binding=_api('face_fibre_binding_r11');source=Path(bpy.data.filepath).resolve();source_sha=_sha(source)
    if not head.data.shape_keys:raise RuntimeError('Preserve the actual R11 Basis and existing face keys')
    keys_before=jaw['_keys'](head.data);basis=_array(head.data.shape_keys.reference_key.data)
    T=np.asarray(json.loads(head['R11_female_to_target']));delta,report=plan(basis,T)
    original_bridge=bridge.data;original_fake=original_bridge.use_fake_user;original_bridge.use_fake_user=True
    groups_before=[(g.name,g.index) for g in bridge.vertex_groups]
    key=None;candidate_mesh=None;success=False;added_groups=[]
    active=head.active_shape_key_index;pose_contract=_api('production_uv_r08')['_pose_contract']()
    head_topology_before=_api('face_fibre_binding_r11')['_hash']({'loops':_array(head.data.loops,'vertex_index',1,np.int32).ravel().tolist(),
        'mask':[(m.name,m.vertex_group,m.threshold,m.invert_vertex_group) for m in head.modifiers if m.type=='MASK']})
    try:
        key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=1.
        key.data.foreach_set('co',(basis+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
        with qa['rest_evaluation']():
            top_mesh,_,top_normals=core['_snapshot'](bpy,head);body_mesh,weights,body_normals=core['_snapshot'](bpy,body)
            top=core['boundary_ring'](top_mesh);bottom=core['boundary_ring'](body_mesh,neck_only=True)
            if len(top['points'])!=432 or len(bottom['points'])!=224:raise RuntimeError('Expected exact original boundary topology')
            top['normals']=top_normals[top['ids']];bottom['normals']=body_normals[bottom['ids']]
            arrays,build=build_low_bridge(top,bottom)
            if len(arrays['vertices'])!=1552:raise RuntimeError('Unexpected bridge vertex parameterization')
            faces=[arrays['loops'][s:s+n].tolist() for s,n in zip(arrays['face_starts'],arrays['face_sizes'])]
            candidate_mesh=bpy.data.meshes.new('R11 low-neck continuous bridge mesh')
            candidate_mesh.from_pydata(arrays['vertices'].tolist(),[],faces);candidate_mesh.update()
            for slot in original_bridge.materials:candidate_mesh.materials.append(slot)
            for name,width,kind in [('R01_region_color',4,'FLOAT_COLOR'),('R04 skin roughness',1,'FLOAT'),('R04 skin rest position',3,'FLOAT_VECTOR')]:
                old=original_bridge.attributes.get(name)
                if old is None or old.domain!='POINT':raise RuntimeError('Bridge material input changed: '+name)
                attr=candidate_mesh.attributes.new(name,kind,'POINT')
                member='color' if width==4 else 'value' if width==1 else 'vector'
                values=_array(old.data,member,width)
                if width==3:values=arrays['vertices'].astype(np.float32)
                attr.data.foreach_set(member,values.ravel())
            bridge.data=candidate_mesh
            # Blender versions can carry deform-group names on the mesh ID.
            # Inspect the actual swap before rebuilding weights; never assume
            # that assigning a fresh mesh preserves object.vertex_groups.
            groups_after_swap=[(g.name,g.index) for g in bridge.vertex_groups]
            recreated_original_groups=False
            if groups_after_swap==[] and groups_before:
                for old_name,old_index in groups_before:
                    recovered=bridge.vertex_groups.new(name=old_name)
                    if recovered.index!=old_index or recovered.name!=old_name:
                        raise RuntimeError('Cannot reconstruct original bridge group mapping '+json.dumps({'expected':[old_name,old_index],'actual':[recovered.name,recovered.index]}))
                recreated_original_groups=True
            elif groups_after_swap!=groups_before:
                raise RuntimeError('Unexpected bridge groups immediately after data swap '+json.dumps({'before':groups_before,'afterSwap':groups_after_swap}))
            group_swap_diagnostic={'before':groups_before,'afterMeshSwap':groups_after_swap,'recreatedOriginalMapping':recreated_original_groups,
                'afterMappingInitialization':[(g.name,g.index) for g in bridge.vertex_groups]}
            print('R11_LOW_NECK_GROUP_SWAP '+json.dumps(group_swap_diagnostic),flush=True)
            names={g.index:g.name for g in body.vertex_groups};bottom_weights={}
            for j,index in enumerate(bottom['ids']):
                for group_id,value in weights[int(index)]:
                    if value>0:bottom_weights.setdefault(names[group_id],np.zeros(len(bottom['ids'])))[j]=value
            t=arrays['t'];blend=_smooth(t);all_weights={name:core['_periodic'](values,bottom['angles'],arrays['angles'])*blend for name,values in bottom_weights.items()}
            all_weights['head']=all_weights.get('head',np.zeros(len(t)))+1-blend
            if np.max(np.abs(sum(all_weights.values())-1))>2e-5:raise RuntimeError('Actual bridge weights lost normalization')
            for name,values in all_weights.items():
                group=bridge.vertex_groups.get(name)
                if group is None:
                    rig=bpy.data.objects['Character_Rig']
                    if name not in rig.data.bones:raise RuntimeError('Actual body weight refers to a missing rig bone: '+name)
                    group=bridge.vertex_groups.new(name=name);added_groups.append({'name':group.name,'index':group.index})
                for index in np.flatnonzero(values>1e-8):group.add([int(index)],float(values[index]),'REPLACE')
            for poly in candidate_mesh.polygons:poly.use_smooth=True
            candidate_mesh.normals_split_custom_set([tuple(arrays['boundary_normals'][loop.vertex_index]) for loop in candidate_mesh.loops])
            bpy.context.view_layer.update();actual,_,_=core['_snapshot'](bpy,bridge)
            error=float(np.max(np.abs(actual['world']-arrays['vertices'])))
            if error>3e-7:raise RuntimeError('REST bridge basis error: '+str(error))
        keys_after=jaw['_keys'](head.data)
        for name,old in keys_before.items():
            now=keys_after[name]
            if not np.array_equal(old['coords'],now['coords']) or any(old[k]!=now[k] for k in old if k!='coords'):
                raise RuntimeError('Existing original key changed: '+name)
        actual_groups=[(g.name,g.index) for g in bridge.vertex_groups]
        if actual_groups[:len(groups_before)]!=groups_before or actual_groups[len(groups_before):]!=[(g['name'],g['index']) for g in added_groups]:
            raise RuntimeError('Original bridge group indices changed or unexpected group was added '+json.dumps({'before':groups_before,'actual':actual_groups,'added':added_groups,'swap':group_swap_diagnostic}))
        head_topology_after=binding['_hash']({'loops':_array(head.data.loops,'vertex_index',1,np.int32).ravel().tolist(),
            'mask':[(m.name,m.vertex_group,m.threshold,m.invert_vertex_group) for m in head.modifiers if m.type=='MASK']})
        if head_topology_before!=head_topology_after:raise RuntimeError('Head topology or Mask changed')
        if _api('production_uv_r08')['_pose_contract']()!=pose_contract or _sha(source)!=source_sha:raise RuntimeError('Rig pose or saved source changed')
        fibre=binding['validate'](neutral=False) if 'R11_face_fibre_binding' in bpy.context.scene else {'notPresent':True}
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(Path(__file__)),key=KEY,keyValue=1.,
            bridgeBuild=build,bridgeRestMaximumErrorMeters=error,originalBridgeData=original_bridge.name,originalBridgeFakeUser=original_fake,
            candidateBridgeData=candidate_mesh.name,bridgeGroupSwapDiagnostic=group_swap_diagnostic,addedBridgeVertexGroups=added_groups,originalBridgeVertexGroups=groups_before,
            oldKeysUnchanged=list(keys_before),headMaskAndTopologyUnchanged=True,
            actualFibreBindingValidation=fibre,originalActiveShapeKeyIndex=active,keySha256=skin['_ah'](_array(key.data)),savedBlend=False,
            limits=['A further actual front/quarter/side comparison is required; no shape is accepted merely because boundaries meet.',
                'The original body colour and optics remain unchanged. The existing bridge pigment blend is retained in this geometry candidate.',
                'Dynamic neck rotations, seam continuity and shoulder/garment clearance still require actual pose tests.'])
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            bridge.data=original_bridge;original_bridge.use_fake_user=original_fake
            for added in reversed(added_groups):
                group=bridge.vertex_groups.get(added['name'])
                if group is not None:bridge.vertex_groups.remove(group)
            if key is not None and head.data.shape_keys and KEY in head.data.shape_keys.key_blocks:head.shape_key_remove(head.data.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[HEAD];bridge=bpy.data.objects[BRIDGE]
    key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['keySha256']:raise RuntimeError('Preserve edited low-neck key before restoring')
    candidate=bridge.data;candidate.use_fake_user=True;bridge.data=bpy.data.meshes[report['originalBridgeData']]
    bridge.data.use_fake_user=report['originalBridgeFakeUser']
    for added in reversed(report.get('addedBridgeVertexGroups',[])):
        group=bridge.vertex_groups.get(added['name'])
        if group is not None:
            if group.index!=added['index']:raise RuntimeError('Bridge group ordering changed; preserve that work before restoring')
            bridge.vertex_groups.remove(group)
    if [(g.name,g.index) for g in bridge.vertex_groups]!=[tuple(x) for x in report['originalBridgeVertexGroups']]:
        raise RuntimeError('Restored bridge original group indices differ')
    head.shape_key_remove(key);head.active_shape_key_index=report['originalActiveShapeKeyIndex']
    del bpy.context.scene[MARKER];bpy.context.view_layer.update();return {'restored':True,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists() or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh explicit source and new independent output required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_LOW_NECK_CONTOUR_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
