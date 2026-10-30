"""A continuous new-neck-to-retained-body bridge in an independent R11 copy.

Plan is pure NumPy on actual observations. Prepare keeps the female raw cage,
Multires, face, eyes and brows intact, restores the anatomical display Mask,
and builds an annulus between the exact evaluated boundaries. No rendering,
head slimming, material opacity, camera cropping or master write.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';OLD='R01 Head';BRIDGE='R11 Continuous neck bridge'
OBS='work/reconstruction-r02/adult-neck-anatomical-observation-r11.json'
PLAN='work/reconstruction-r02/adult-neck-bridge-plan-r11.json'
MARKER='R11_continuous_neck_bridge'
TAU=2*np.pi


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _read(relative):return json.loads((project_path()/relative).read_text(encoding='utf-8'))


def _path(relative,suffix):
    p=project_path().resolve();f=(p/relative).resolve();f.relative_to(p/'work/reconstruction-r02')
    if f.suffix!=suffix:raise ValueError('Unexpected independent work artifact')
    return f


def _array(data,field,width=3,dtype=np.float32):
    a=np.empty(len(data)*width,dtype);data.foreach_get(field,a);return a.reshape(-1,width)


def _world(co,matrix):
    m=np.asarray(matrix,float);return np.asarray(co)@m[:3,:3].T+m[:3,3]


def _unit(v):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    if np.any(length<1e-12):raise RuntimeError('Degenerate surface direction')
    return v/length


def _isotonic_phase(theta):
    """Least-squares monotone phase only; never moves a boundary vertex."""
    epsilon=1e-5;n=len(theta)
    extended=np.r_[theta-TAU,theta,theta+TAU]
    values=extended-epsilon*np.arange(len(extended));blocks=[]
    for index,value in enumerate(values):
        blocks.append([float(value),1,index,index+1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            right=blocks.pop();left=blocks.pop();count=left[1]+right[1]
            blocks.append([(left[0]*left[1]+right[0]*right[1])/count,count,left[2],right[3]])
    result=np.empty(len(extended))
    for mean,count,start,end in blocks:result[start:end]=mean
    result+=epsilon*np.arange(len(extended));result=result[n:2*n]
    if np.max(np.abs(result-theta))>.04:raise RuntimeError('Boundary angular reversal exceeds a small seam irregularity')
    return result


def _mesh_arrays(mesh,matrix):
    mesh.calc_loop_triangles()
    return {'world':_world(_array(mesh.vertices,'co'),matrix),
        'loops':_array(mesh.loops,'vertex_index',1,np.int32).ravel(),
        'face_starts':_array(mesh.polygons,'loop_start',1,np.int32).ravel(),
        'face_sizes':_array(mesh.polygons,'loop_total',1,np.int32).ravel(),
        'triangles':_array(mesh.loop_triangles,'vertices',3,np.int32)}


def boundary_ring(mesh,neck_only=False):
    """Actual cycle order; monotone polar phase preserves tiny seam zigzags."""
    v=mesh['world'];loops=mesh['loops'];starts=mesh['face_starts'];sizes=mesh['face_sizes']
    nxt=np.arange(len(loops))+1;nxt[starts+sizes-1]=starts
    edges=np.sort(np.column_stack([loops,loops[nxt]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True)
    if np.any(count>2):raise RuntimeError('Nonmanifold source surface')
    boundary=unique[count==1]
    if neck_only:boundary=boundary[np.all(v[boundary,2]<1.31,axis=1)]
    ids=np.unique(boundary)
    if len(ids)<50:raise RuntimeError('No complete neck boundary')
    degree=np.bincount(boundary.ravel(),minlength=len(v))
    if np.any(degree[ids]!=2):raise RuntimeError('Neck boundary is not a closed simple cycle')
    adjacency={int(i):[] for i in ids}
    for x,y in boundary:adjacency[int(x)].append(int(y));adjacency[int(y)].append(int(x))
    start=int(ids[np.argmax(v[ids,0])]);cycle=[start];previous=None;current=start
    while True:
        candidates=sorted(i for i in adjacency[current] if i!=previous)
        following=candidates[0]
        if following==start:break
        if following in cycle:raise RuntimeError('Self-repeating boundary traversal')
        cycle.append(following);previous,current=current,following
    if len(cycle)!=len(ids):raise RuntimeError('Boundary contains disconnected cycles')
    ids=np.asarray(cycle,np.int32);points=v[ids]
    area=np.sum(points[:,0]*np.roll(points[:,1],-1)-np.roll(points[:,0],-1)*points[:,1])
    if area<0:ids=np.r_[ids[:1],ids[:0:-1]];points=v[ids]
    center=(points[:,:2].min(0)+points[:,:2].max(0))*.5
    theta=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
    monotone=_isotonic_phase(theta);phase_error=float(np.max(np.abs(monotone-theta)))
    angles=np.mod(monotone,TAU);start=int(np.argmin(angles))
    ids=np.roll(ids,-start);angles=np.roll(angles,-start)
    expected={tuple(e) for e in boundary.tolist()}
    ordered={tuple(sorted((int(x),int(y)))) for x,y in zip(ids,np.roll(ids,-1))}
    if expected!=ordered:raise RuntimeError('Ordered cycle does not preserve every original edge')
    if np.min(np.diff(np.r_[angles,angles[0]+TAU]))<1e-8:raise RuntimeError('Degenerate angular neck parameter')
    triangles=mesh['triangles'];p=v[triangles]
    face_normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal=np.zeros_like(v)
    for corner in range(3):np.add.at(normal,triangles[:,corner],face_normals)
    normal=_unit(normal[ids])
    return {'ids':ids,'points':v[ids],'angles':angles,'normals':normal,'centerXY':center,
        'phaseMethod':'Actual topological cycle, least-squares monotone polar phase; coordinates untouched',
        'maximumPhaseAdjustmentRadians':phase_error}


def _periodic(values,angles,target):
    values=np.asarray(values)
    if values.ndim==1:return np.interp(target,angles,values,period=TAU)
    return np.column_stack([np.interp(target,angles,values[:,j],period=TAU) for j in range(values.shape[1])])


def _down_tangent(ring):
    tangent=_unit(np.roll(ring['points'],-1,axis=0)-np.roll(ring['points'],1,axis=0))
    direction=_unit(np.cross(tangent,ring['normals']))
    direction[direction[:,2]>0]*=-1
    return direction


def build_bridge(top,bottom):
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
        vertices[nt:lower_start]+=.5*(target[nt:lower_start]-vertices[nt:lower_start])
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
         'method':'Exact original 432/224 boundary cycles, seven uniformly parameterized 128-point interior rings, finite positive-weight interior fairing, ordered unequal-count annuli. Endpoint shading normals match actual source; exact tangent continuity and dynamic poses require visual review.'})


def plan(output_relative=PLAN):
    record=_read(OBS);payload=project_path()/record['arrays']['path']
    if _sha(payload)!=record['arrays']['sha256']:raise RuntimeError('Anatomical observation changed')
    with np.load(payload) as f:
        source={name:{key:f[prefix+'_'+key] for key in ('world','loops','face_starts','face_sizes','triangles')}
            for name,prefix in (('top','anatomical_new_head'),('bottom','old_displayed_body'))}
    top=boundary_ring(source['top']);bottom=boundary_ring(source['bottom'],neck_only=True)
    arrays,diagnostics=build_bridge(top,bottom)
    output=_path(output_relative,'.json');npz=output.with_suffix('.npz')
    if output.exists() or npz.exists():raise RuntimeError('Retain prior bridge plan')
    with npz.open('xb') as stream:np.savez_compressed(stream,**arrays)
    report={'observation':OBS,'observationArraysSha256':record['arrays']['sha256'],
        'observationSourceSha256':record['sourceSha256'],'femaleToTargetRestWorld':record['femaleToTargetRestWorld'],
        'helperSha256AtPlan':_sha(Path(__file__)),'diagnostics':diagnostics,
        'arrays':{'path':str(npz.relative_to(project_path())),'sha256':_sha(npz)},
        'appearanceAccepted':False,'rendered':False,'scope':'Offline annular neck bridge design only; requires actual source matching, Blender preparation and side/quarter renders.'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2));return report


def _snapshot(bpy,obj):
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
    try:
        geometry=_mesh_arrays(mesh,ev.matrix_world)
        weights=[[(g.group,float(g.weight)) for g in v.groups] for v in mesh.vertices]
        normals=_array(mesh.vertices,'normal').astype(float)
        normal_matrix=np.linalg.inv(np.asarray(ev.matrix_world,float)[:3,:3]).T
        normals=_unit(normals@normal_matrix.T)
        return geometry,weights,normals
    finally:ev.to_mesh_clear()


def _body_only_group(bpy,old):
    modifier=old.modifiers['R11 hide rejected head surface'];original=modifier.node_group
    original.use_fake_user=True
    group=bpy.data.node_groups.new('R11 Retained body without rejected face or oral pocket','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    inp=group.nodes.new('NodeGroupInput');out=group.nodes.new('NodeGroupOutput')
    selection=group.nodes.new('GeometryNodeMaterialSelection');selection.inputs['Material'].default_value=bpy.data.materials['Character / anatomical body skin']
    invert=group.nodes.new('FunctionNodeBooleanMath');invert.operation='NOT'
    delete=group.nodes.new('GeometryNodeDeleteGeometry');delete.domain='FACE';delete.mode='ALL'
    group.links.new(inp.outputs['Geometry'],delete.inputs['Geometry']);group.links.new(selection.outputs['Selection'],invert.inputs[0])
    group.links.new(invert.outputs[0],delete.inputs['Selection']);group.links.new(delete.outputs['Geometry'],out.inputs['Geometry'])
    modifier.node_group=group;return original.name


def prepare(source_relative,output_relative,plan_relative=PLAN):
    import bpy
    from mathutils import Matrix
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if output.exists() or report_path.exists() or output==source:raise RuntimeError('New independent candidate/report required')
    if not bpy.app.background or bpy.data.is_dirty or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh explicit independent R11 source required')
    if bpy.data.objects.get(BRIDGE) or bpy.context.scene.get(MARKER):raise RuntimeError('Preserve existing bridge candidate first')
    report_plan=_read(plan_relative);npz=project_path()/report_plan['arrays']['path']
    if _sha(npz)!=report_plan['arrays']['sha256']:raise RuntimeError('Bridge plan changed')
    with np.load(npz) as f:planned={name:f[name] for name in f.files}
    uv=_api('production_uv_r08');head=bpy.data.objects[HEAD];old=bpy.data.objects[OLD];rig=bpy.data.objects['Character_Rig']
    if len(head.data.vertices)!=10582:raise RuntimeError('Female full source topology must remain available')
    bpy.context.view_layer.update();source_sha=_sha(source);pose_before=uv['_pose_contract']()
    head_raw=_array(head.data.vertices,'co').copy();head_uv={x.name:_array(x.data,'uv',2).copy() for x in head.data.uv_layers}
    old_data_hash=_api('posterior_groom_r09')['_data_hash'](old.data)
    scene=bpy.context.scene;frame=scene.frame_current;sub=scene.frame_subframe;pose=rig.data.pose_position
    mask=next(m for m in head.modifiers if m.type=='MASK');old_group=mask.vertex_group
    old_head_data=head.data;old_head_fake=old_head_data.use_fake_user;old_head_data.use_fake_user=True
    saved_levels=[(m.name,m.levels) for m in old.modifiers if m.type=='SUBSURF']
    bridge=None
    try:
        scene.frame_set(1);rig.data.pose_position='REST';bpy.context.view_layer.update()
        head.data=old_head_data.copy();head.data.name='R11 Female cage with anatomical neck display'
        selection=_read('work/reconstruction-r02/female-head-selection-r11.json')
        group=head.vertex_groups.new(name='R11 anatomical head neck for true bridge');group.add(selection['original_vertex_indices'],1.,'REPLACE')
        mask.vertex_group=group.name
        for m in old.modifiers:
            if m.type=='SUBSURF':m.levels=m.render_levels
        old_node_group=_body_only_group(bpy,old);bpy.context.view_layer.update()
        top_mesh,_,top_normals=_snapshot(bpy,head);body_mesh,body_weights,body_normals=_snapshot(bpy,old)
        top=boundary_ring(top_mesh);bottom=boundary_ring(body_mesh,neck_only=True)
        ti=planned['top_boundary_ids'];bi=planned['bottom_boundary_ids']
        for label,actual,expected in (('head',top['points'],planned['vertices'][ti]),('body',bottom['points'],planned['vertices'][bi])):
            if actual.shape!=expected.shape or np.max(np.abs(actual-expected))>2e-7:
                raise RuntimeError('Actual '+label+' neck boundary differs from the observed plan')
        # Normal/weight data are taken from this actual source, never an old rig.
        top['normals']=top_normals[top['ids']];bottom['normals']=body_normals[bottom['ids']]
        arrays,build_report=build_bridge(top,bottom)
        faces=[arrays['loops'][s:s+n].tolist() for s,n in zip(arrays['face_starts'],arrays['face_sizes'])]
        mesh=bpy.data.meshes.new(BRIDGE+' mesh');mesh.from_pydata(arrays['vertices'].tolist(),[],faces);mesh.update()
        bridge=bpy.data.objects.new(BRIDGE,mesh);head.users_collection[0].objects.link(bridge)
        bridge.parent=rig;bridge.parent_type='OBJECT';bridge.matrix_parent_inverse=rig.matrix_world.inverted();bridge.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update()
        if np.max(np.abs(np.asarray(bridge.matrix_world)-np.eye(4)))>2e-7:raise RuntimeError('Bridge world basis changed')
        group_names={g.index:g.name for g in old.vertex_groups};bottom_weights={}
        for j,source_index in enumerate(bottom['ids']):
            for index,value in body_weights[int(source_index)]:
                if value>0:bottom_weights.setdefault(group_names[index],np.zeros(len(bottom['ids'])))[j]=value
        if not bottom_weights or np.min(sum(bottom_weights.values()))<.999:
            raise RuntimeError('Actual evaluated body boundary lacks complete deformation weights')
        t=arrays['t'];blend=t*t*(3-2*t);all_weights={}
        for name,values in bottom_weights.items():
            all_weights[name]=_periodic(values,bottom['angles'],arrays['angles'])*blend
        all_weights['head']=all_weights.get('head',np.zeros(len(t)))+(1-blend)
        total=sum(all_weights.values())
        if np.max(np.abs(total-1))>2e-5:raise RuntimeError('Bridge weights are not normalized')
        for name,values in all_weights.items():
            if name not in rig.data.bones:raise RuntimeError('Unknown actual body group: '+name)
            vg=bridge.vertex_groups.new(name=name)
            for index in np.flatnonzero(values>1e-8):vg.add([int(index)],float(values[index]),'REPLACE')
        arm=bridge.modifiers.new('R11 continuous neck skin weights','ARMATURE');arm.object=rig;arm.use_deform_preserve_volume=False
        material=bpy.data.materials.get('R11 Female head neutral skin study')
        if material is None:raise RuntimeError('Retained neutral source skin material is required for the bridge')
        material=material.copy();material.name='R11 Continuous neck neutral skin'
        mesh.materials.append(material)
        body_material=bpy.data.materials['Character / anatomical body skin'];bs=next(n for n in body_material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        if bs.inputs['Base Color'].is_linked:raise RuntimeError('Body colour source changed; require explicit bridge colour inspection')
        base=np.asarray((.49,.31,.225));body_colour=np.asarray(bs.inputs['Base Color'].default_value[:3])
        pigment=t**6;col=np.ones((len(t),4),np.float32);col[:,:3]=base*(1-pigment[:,None])+body_colour*pigment[:,None]
        ca=mesh.color_attributes.new(name='R01_region_color',type='FLOAT_COLOR',domain='POINT');ca.data.foreach_set('color',col.ravel())
        ra=mesh.attributes.new(name='R04 skin roughness',type='FLOAT',domain='POINT');ra.data.foreach_set('value',(.50*(1-pigment)+float(bs.inputs['Roughness'].default_value)*pigment).astype(np.float32))
        rest=mesh.attributes.new(name='R04 skin rest position',type='FLOAT_VECTOR',domain='POINT');rest.data.foreach_set('vector',arrays['vertices'].astype(np.float32).ravel())
        for polygon in mesh.polygons:polygon.use_smooth=True
        mesh.normals_split_custom_set([tuple(arrays['boundary_normals'][loop.vertex_index]) for loop in mesh.loops])
        bridge['R11_static_fit_not_full_animation_approval']=True
        bridge['R11_actual_boundary_observation']=OBS
        bridge['R11_boundary_top_count']=len(ti);bridge['R11_boundary_bottom_count']=len(bi)
        bridge['R11_original_head_data']=old_head_data.name
        bridge['R11_original_head_mask_group']=old_group
        bpy.context.view_layer.update()
        rest_mesh,_,_=_snapshot(bpy,bridge)
        rest_error=float(np.max(np.abs(rest_mesh['world']-arrays['vertices'])))
        if rest_error>3e-7:raise RuntimeError('REST armature changed bridge geometry: '+str(rest_error))
    finally:
        rig.data.pose_position=pose;scene.frame_set(frame,subframe=sub);bpy.context.view_layer.update()
    if uv['_pose_contract']()!=pose_before:raise RuntimeError('Original body rig pose changed')
    if not np.array_equal(_array(head.data.vertices,'co'),head_raw):raise RuntimeError('New face raw geometry changed')
    if any(not np.array_equal(_array(head.data.uv_layers[n].data,'uv',2),v) for n,v in head_uv.items()):raise RuntimeError('Female author UV changed')
    if _api('posterior_groom_r09')['_data_hash'](old.data)!=old_data_hash:raise RuntimeError('Retained body raw mesh/attributes changed')
    if _sha(source)!=source_sha:raise RuntimeError('Saved source changed')
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(Path(__file__)),
        'plan':plan_relative,'planArraysSha256':report_plan['arrays']['sha256'],'bridge':bridge.name,
        'build':build_report,'restMaximumErrorMeters':rest_error,
        'oldHeadDataBackup':old_head_data.name,'oldHeadDataOriginalFakeUser':old_head_fake,'oldHeadMaskGroup':old_group,
        'newHeadData':head.data.name,'newHeadMaskGroup':group.name,
        'oldBodyDisplayNodeGroupBackup':old_node_group,'oldBodyViewportSubdivisionLevels':saved_levels,
        'faceRawCageAndAuthorUVUnchanged':True,'bodyRawDataUnchanged':True,'bodyRigPoseUnchanged':True,
        'bodyOralResidueHidden':True,'appearanceAccepted':False,'savedBlend':False,
        'scope':'Anatomical author neck region plus an explicit continuous open-ended annulus connecting its exact evaluated edge to the retained torso edge. No opacity/crop/capped stump.',
        'limits':['Three editable surface components share exact REST boundary positions; they are not yet welded into one final mesh.',
            'Head-side boundary uses head weight 1; body-side uses actual evaluated torso weights. Dynamic seam/neck poses still require actual testing.',
            'Current body and head skin have different base colours. The bridge blends only near the torso end; final full-body skin matching remains a separate visual task.',
            'Groom, eyes, brows, lips, clothing and their caches are not modified or approved by this geometry pass.']}
    scene[MARKER]=json.dumps(report)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent neck candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_NECK_BRIDGE_SAVED '+json.dumps({'output':str(output),'outputSha256':report['outputSha256']}),flush=True)
    return report



def restore():
    """Exact data/display-stack restoration in a disposable candidate session.

    Does not save, delete history or change the source file on disk.
    """
    import bpy
    scene=bpy.context.scene
    if MARKER not in scene:raise RuntimeError('No R11 neck bridge candidate to restore')
    record=json.loads(scene[MARKER]);head=bpy.data.objects[HEAD];old=bpy.data.objects[OLD]
    previous=bpy.data.meshes[record['oldHeadDataBackup']];candidate=head.data;candidate.use_fake_user=True
    head.data=previous;previous.use_fake_user=record['oldHeadDataOriginalFakeUser']
    next(m for m in head.modifiers if m.type=='MASK').vertex_group=record['oldHeadMaskGroup']
    vg=head.vertex_groups.get(record['newHeadMaskGroup'])
    if vg is not None:head.vertex_groups.remove(vg)
    old.modifiers['R11 hide rejected head surface'].node_group=bpy.data.node_groups[record['oldBodyDisplayNodeGroupBackup']]
    for name,level in record['oldBodyViewportSubdivisionLevels']:old.modifiers[name].levels=level
    bridge=bpy.data.objects[record['bridge']];bridge.hide_render=True;bridge.hide_viewport=True
    scene['R11_continuous_neck_bridge_restored_history']=scene[MARKER];del scene[MARKER]
    bpy.context.view_layer.update();return {'restored':True,'savedBlend':False,'retainedCandidateData':candidate.name}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('plan','prepare'),required=True)
    p.add_argument('--source');p.add_argument('--output');p.add_argument('--plan',default=PLAN)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    if args.mode=='plan':plan(args.output or PLAN)
    else:
        if not args.source or not args.output:p.error('--source and --output required')
        prepare(args.source,args.output,args.plan)
