"""Actual upper-neck tangent collar, preserving the original lower bridge.

The old bridge directly zipped a 432-vertex irregular head edge to 128 points.
This adds three 432-point collar rows before the old third uniform row, using
the real incident-face direction. It preserves the head and body data and all
original lower bridge points, material attributes and weights. New topology is
2592 vertices, not the previous 1552. Bottom/body tangent defects are retained.
No Blender starts, renders, live-scene edits or master writes are performed.
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
HEAD='R11 Female head study';BODY='R01 Head';BRIDGE='R11 Continuous neck bridge'
MARKER='R11_upper_neck_seam_candidate'
ENDROW=2
FRACTIONS=(.15,.40,.70)


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
    if np.any(n<1e-12):raise RuntimeError('Undefined actual surface direction')
    return a/n


def _inward(mesh,ring):
    ids=np.asarray(ring['ids']);excluded=set(map(int,ids));neighbors={int(i):set() for i in ids}
    for s,n in zip(mesh['face_starts'],mesh['face_sizes']):
        face=set(map(int,mesh['loops'][s:s+n]))
        for i in face&excluded:neighbors[i].update(face-excluded)
    tangent=_unit(np.roll(ring['points'],-1,axis=0)-np.roll(ring['points'],1,axis=0));result=[]
    for k,i in enumerate(ids):
        row=sorted(neighbors[int(i)])
        if not row:raise RuntimeError('No actual inward face neighbor at '+str(i))
        d=mesh['world'][row].mean(0)-mesh['world'][i];d-=tangent[k]*np.dot(d,tangent[k]);result.append(d)
    return _unit(result)


def _zipper(ua,la,us,ls):
    out=[];i=j=0;nu=len(ua);nl=len(la);tau=2*np.pi
    while i<nu or j<nl:
        up=ua[(i+1)%nu]+(tau if i+1>=nu else 0.) if i<nu else np.inf
        low=la[(j+1)%nl]+(tau if j+1>=nl else 0.) if j<nl else np.inf
        u=us+i%nu;l=ls+j%nl
        if abs(up-low)<1e-10:out.append((u,l,ls+(j+1)%nl,us+(i+1)%nu));i+=1;j+=1
        elif up<low:out.append((u,l,us+(i+1)%nu));i+=1
        else:out.append((u,l,ls+(j+1)%nl));j+=1
    return out


def _topology(faces):
    sizes=np.asarray([len(f) for f in faces],np.int32)
    loops=np.concatenate([np.asarray(f,np.int32) for f in faces])
    return {'loops':loops,'face_sizes':sizes,'face_starts':np.r_[0,np.cumsum(sizes[:-1])].astype(np.int32),
        'triangles':np.asarray([(f[0],f[k],f[k+1]) for f in faces for k in range(1,len(f)-1)],np.int32)}


def _geometric_normals(v,triangles):
    p=v[triangles];fn=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal=np.zeros_like(v)
    for k in range(3):np.add.at(normal,triangles[:,k],fn)
    return _unit(normal),np.linalg.norm(fn,axis=1)*.5


def _seam_stats(head,ring,bridge):
    first=_inward(head,ring);bridge_ring={'ids':np.arange(432),'points':bridge['world'][:432]}
    second=_inward(bridge,bridge_ring)
    angle=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',first,-second),-1,1)))
    return {'actualSamples':len(angle),'angleQuantilesDeg':np.quantile(angle,[0,.5,.95,1]).tolist(),
        'maximumPositionGapMeters':float(np.linalg.norm(ring['points']-bridge['world'][:432],axis=1).max())}


def build(head,bridge,known_ring=None):
    core=_api('adult_neck_bridge_r11');periodic=core['_periodic']
    ring=core['boundary_ring'](head) if known_ring is None else known_ring
    old=np.asarray(bridge['world']);nt=432;n=128;start=nt+ENDROW*n
    if len(old)!=1552 or len(ring['ids'])!=nt:raise RuntimeError('Requires the original 1552/432 bridge geometry')
    if np.max(np.linalg.norm(old[:nt]-ring['points'],axis=1))>1e-6:raise RuntimeError('Original bridge vertex order does not match actual head boundary')
    u=np.arange(n)*2*np.pi/n;a=ring['angles'];top=ring['points']
    target=periodic(old[start:start+n],u,a);chord=target-top;length=np.linalg.norm(chord,axis=1)
    ma=-_inward(head,ring)*length[:,None]
    mb=periodic(old[start+n:start+2*n]-old[start-n:start],u,a)*(ENDROW+1)/2.
    pieces=[top]
    for t in FRACTIONS:
        pieces.append((2*t**3-3*t**2+1)*top+(t**3-2*t**2+t)*ma+(-2*t**3+3*t**2)*target+(t**3-t**2)*mb)
    pieces.append(old[start:]);v=np.vstack(pieces);faces=[]
    for j in range(3):faces+=_zipper(a,a,j*nt,(j+1)*nt)
    faces+=_zipper(a,u,3*nt,4*nt);shift=4*nt-start;retained_face_start=len(faces);retained_face_ids=[]
    for face_id,(s,nf) in enumerate(zip(bridge['face_starts'],bridge['face_sizes'])):
        face=bridge['loops'][s:s+nf]
        if int(face.min())>=start:faces.append((face+shift).tolist());retained_face_ids.append(face_id)
    topology=_topology(faces);normals,areas=_geometric_normals(v,topology['triangles'])
    if len(v)!=2592 or areas.min()<1e-12 or not np.isfinite(v).all():raise RuntimeError('Invalid upper neck collar geometry')
    audit=_api('neck_bridge_geometry_audit_r11')['triangle_intersections'](v,topology['triangles'])
    if len(audit['strictCrossingPairs']) or len(audit['coplanarOverlapPairs']):raise RuntimeError('Actual new upper-neck collar self-intersects')
    new={'world':v,**topology};before=_seam_stats(head,ring,bridge);after=_seam_stats(head,ring,new)
    if after['angleQuantilesDeg'][2]>=before['angleQuantilesDeg'][2]:raise RuntimeError('Actual upper seam 95th-percentile tangent mismatch did not improve')
    if not np.array_equal(v[4*nt:],old[start:]):raise RuntimeError('Original lower bridge points changed')
    return new,normals,{'originalVertices':1552,'newVertices':2592,'newFaces':len(faces),
        'headBoundaryIds':ring['ids'].tolist(),'sourceAngles':a.tolist(),'originalRetainedStart':start,
        'newRetainedStart':4*nt,'newRowFractions':FRACTIONS,'minimumTriangleAreaSquareMeters':float(areas.min()),
        'originalRetainedFaceIds':retained_face_ids,'newRetainedFaceStart':retained_face_start,
        'strictCrossingCount':0,'coplanarOverlapCount':0,'upperSeamBefore':before,'upperSeamAfter':after,
        'originalLowerBridgePointsExact':True,'originalBodyBoundaryExact':True,
        'bottomSeamRepaired':False,'appearanceAccepted':False,
        'scope':'Three full 432-point tangent collar rows replace the first two uniform rings; original third uniform ring through body boundary retained exactly.'}


def resample_point_data(values,angles):
    """Exact retained rows; new collar interpolates existing actual data only."""
    values=np.asarray(values);start=432+ENDROW*128
    target=_api('adult_neck_bridge_r11')['_periodic'](values[start:start+128],np.arange(128)*2*np.pi/128,angles)
    return np.concatenate([values[:432]]+[(1-t)*values[:432]+t*target for t in FRACTIONS]+[values[start:]],axis=0)


def _snapshot(obj):
    import bpy
    mesh,_,normals=_api('adult_neck_bridge_r11')['_snapshot'](bpy,obj);mesh['shading_normals']=normals;return mesh


def _contract(obj):
    uv=_api('production_uv_r08');groom=_api('posterior_groom_r09');mesh=obj.data
    return {'mesh':groom['_data_hash'](mesh),'keys':[(k.name,_sha_array(_array(k.data)),float(k.value),bool(k.mute),k.relative_key.name)
        for k in mesh.shape_keys.key_blocks] if mesh.shape_keys else None,
        'materials':[(s.link,s.material.name if s.material else None) for s in obj.material_slots],
        'matrix':uv['_plain'](obj.matrix_world),'parent':uv['_plain'](obj.parent),'animation':uv['_animation'](obj)}


def _sha_array(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(a.tobytes()).hexdigest()


def _weight_matrix(obj):
    groups=[(g.name,g.index,g.lock_weight) for g in obj.vertex_groups];weights=np.zeros((len(obj.data.vertices),len(groups)),np.float32)
    for v in obj.data.vertices:
        for g in v.groups:weights[v.index,g.group]=g.weight
    return groups,weights


def _native_shading_attributes(old,new,report):
    """Preserve retained sharp flags and regenerate packed normals via bpy.

    custom_normal stores Blender's encoded corner normals. Interpolating its
    integer components would be incorrect; normals_split_custom_set later
    rebuilds it from actual candidate geometric/boundary normals.
    """
    info={};start=report['originalRetainedStart'];new_start=report['newRetainedStart']
    for name,domain in [('sharp_face','FACE'),('sharp_edge','EDGE'),('custom_normal','CORNER')]:
        attr=old.attributes.get(name)
        if attr is not None:
            if attr.domain!=domain:raise RuntimeError('Unexpected native shading attribute domain: '+name)
            info[name]={'originalDomain':attr.domain,'originalDataType':attr.data_type}
    # New collar faces are smooth by design. Every surviving original face
    # carries its original smooth flag and material index without guessing.
    for face in new.polygons:face.use_smooth=True
    for k,old_id in enumerate(report['originalRetainedFaceIds']):
        target=new.polygons[report['newRetainedFaceStart']+k]
        target.use_smooth=old.polygons[old_id].use_smooth
        target.material_index=old.polygons[old_id].material_index
    info.setdefault('sharp_face',{}).update(method='New collar smooth; original retained face flags/material indices copied exactly',retainedFaces=len(report['originalRetainedFaceIds']))
    attr=old.attributes.get('sharp_edge')
    if attr is not None:
        if attr.data_type!='BOOLEAN':raise RuntimeError('Unexpected native sharp_edge data type')
        old_values=_array(attr.data,'value',1,np.bool_).ravel();old_edges=_array(old.edges,'vertices',2,np.int32)
        new_edges=_array(new.edges,'vertices',2,np.int32);lookup={tuple(sorted(map(int,e))):k for k,e in enumerate(new_edges)}
        values=np.zeros(len(new_edges),np.bool_);retained=removed=0
        def mapped(i):return int(i) if i<432 else int(i-start+new_start) if i>=start else None
        for edge,value in zip(old_edges,old_values):
            pair=[mapped(i) for i in edge]
            if None in pair:removed+=1;continue
            index=lookup.get(tuple(sorted(pair)))
            if index is None:removed+=1;continue
            values[index]=value;retained+=1
        target=new.attributes.get('sharp_edge') or new.attributes.new('sharp_edge','BOOLEAN','EDGE')
        target.data.foreach_set('value',values)
        info['sharp_edge'].update(method='Exact surviving edge correspondence; new collar edges smooth',retainedEdges=retained,removedOldEdges=removed,
            originalTrueEdges=int(old_values.sum()),newTrueEdges=int(values.sum()))
    info.setdefault('custom_normal',{}).update(method='Re-encoded by normals_split_custom_set from actual world geometric normals; actual head/body shading normals at endpoints; no interpolation of packed integer values')
    return info


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.context.scene.get(MARKER):raise RuntimeError('Independent fresh background candidate required')
    head=bpy.data.objects[HEAD];body=bpy.data.objects[BODY];obj=bpy.data.objects[BRIDGE];old=obj.data
    if old.shape_keys is not None or len(old.vertices)!=1552:raise RuntimeError('Preserve existing bridge topology/shape work before a separate integration')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve());source_sha=_sha(source)
    uv=_api('production_uv_r08');pose=uv['_pose_contract']();head_before=_contract(head);body_before=_contract(body)
    groups,weights=_weight_matrix(obj);old_fake=old.use_fake_user;old.use_fake_user=True;new=None;success=False
    try:
        with _api('face_contact_audit_r11')['rest_evaluation']():
            h=_snapshot(head);b=_snapshot(obj);mesh,normals,report=build(h,b)
            angles=np.asarray(report['sourceAngles']);new_weights=resample_point_data(weights,angles)
            if np.max(np.abs(new_weights.sum(1)-1))>2e-5:raise RuntimeError('New collar weights are not normalized')
            matrix=np.asarray(obj.matrix_world,float);inv=np.linalg.inv(matrix);local=mesh['world']@inv[:3,:3].T+inv[:3,3]
            faces=[mesh['loops'][s:s+n].tolist() for s,n in zip(mesh['face_starts'],mesh['face_sizes'])]
            new=bpy.data.meshes.new('R11 upper neck tangent collar mesh');new.from_pydata(local.tolist(),[],faces);new.update()
            for material in old.materials:new.materials.append(material)
            attrs=('R01_region_color','R04 skin roughness','R04 skin rest position')
            native=('sharp_face','sharp_edge','custom_normal')
            unknown=[a.name for a in old.attributes if not a.name.startswith('.') and a.name not in attrs+native and a.name!='position']
            if unknown:raise RuntimeError('Preserve unexpected old bridge attributes explicitly: '+str(unknown))
            attr_hashes={}
            for name,width,kind,field in [('R01_region_color',4,'FLOAT_COLOR','color'),('R04 skin roughness',1,'FLOAT','value'),('R04 skin rest position',3,'FLOAT_VECTOR','vector')]:
                attr=old.attributes.get(name)
                if attr is None or attr.domain!='POINT' or attr.data_type!=kind:raise RuntimeError('Original bridge material attribute differs: '+name)
                values=_array(attr.data,field,width);mapped=resample_point_data(values,angles).astype(np.float32)
                target=new.attributes.new(name,kind,'POINT');target.data.foreach_set(field,mapped.ravel());attr_hashes[name]=_sha_array(mapped)
            obj.data=new
            existing=[(g.name,g.index,g.lock_weight) for g in obj.vertex_groups]
            if existing==[]:
                for name,index,lock in groups:
                    group=obj.vertex_groups.new(name=name);group.lock_weight=lock
                    if group.name!=name or group.index!=index:raise RuntimeError('Cannot preserve original bridge group order')
            elif existing!=groups:raise RuntimeError('Unexpected vertex group remapping')
            for name,index,lock in groups:
                group=obj.vertex_groups[name]
                for i in np.flatnonzero(new_weights[:,index]>0):group.add([int(i)],float(new_weights[i,index]),'REPLACE')
            native_shading=_native_shading_attributes(old,new,report)
            # Top and bottom source normals remain exact; all new interior
            # normals come from geometry, not an artificial flat normal strip.
            normals[:432]=h['shading_normals'][np.asarray(report['headBoundaryIds'])]
            normals[-224:]=b['shading_normals'][-224:]
            normal_local=_unit(normals@matrix[:3,:3]);new.normals_split_custom_set([tuple(normal_local[i]) for i in mesh['loops']])
            for name in native:
                attr=new.attributes.get(name)
                native_shading.setdefault(name,{})['newAttribute']=None if attr is None else {'domain':attr.domain,'dataType':attr.data_type,'elements':len(attr.data)}
            bpy.context.view_layer.update();actual=_snapshot(obj)
            error=float(np.linalg.norm(actual['world']-mesh['world'],axis=1).max())
            if error>4e-7 or not np.array_equal(actual['loops'],mesh['loops']):raise RuntimeError('Actual evaluated new bridge differs from the prepared mesh')
            actual_groups,actual_weights=_weight_matrix(obj)
            if actual_groups!=groups or not np.array_equal(actual_weights,new_weights.astype(np.float32)):raise RuntimeError('Actual assigned weights differ')
            report['actualRestMaximumErrorMeters']=error
        if _contract(head)!=head_before or _contract(body)!=body_before or uv['_pose_contract']()!=pose or _sha(source)!=source_sha:raise RuntimeError('Original head/body/pose/source changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
            originalBridgeData=old.name,originalBridgeFakeUser=old_fake,candidateBridgeData=new.name,
            candidateBridgeSha256=_api('posterior_groom_r09')['_data_hash'](new),originalGroups=groups,
            originalWeightsSha256=_sha_array(weights),newWeightsSha256=_sha_array(new_weights.astype(np.float32)),newAttributeHashes=attr_hashes,
            nativeShadingAttributeHandling=native_shading,
            originalHeadBodyDataKeysUnchanged=True,savedBlend=False,
            limits=['Actual same-light front/side/back and dynamic neck images are required.',
                'Bottom bridge/body tangent discontinuity and different skin shader/rest coordinates remain.',
                'New 2592-point bridge requires dependent mesh-count checks to use the actual current topology; no old whole-neck acceptance report is reused.',
                'This is an upper-seam geometry candidate, not a complete neck deformation rig.'])
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:obj.data=old;old.use_fake_user=old_fake;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);obj=bpy.data.objects[BRIDGE]
    if _api('posterior_groom_r09')['_data_hash'](obj.data)!=report['candidateBridgeSha256']:raise RuntimeError('Preserve subsequent new-bridge edits before restoring')
    obj.data.use_fake_user=True;obj.data=bpy.data.meshes[report['originalBridgeData']];obj.data.use_fake_user=report['originalBridgeFakeUser']
    groups,weights=_weight_matrix(obj)
    if json.dumps(groups)!=json.dumps(report['originalGroups']) or _sha_array(weights)!=report['originalWeightsSha256']:raise RuntimeError('Original bridge groups/weights did not restore exactly')
    del bpy.context.scene[MARKER];bpy.context.view_layer.update();return {'restored':True,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();report_path=output.with_suffix('.upper-neck-seam.json')
    for path in (source,output):path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if source.suffix!='.blend' or output.suffix!='.blend' or source==output or output.exists() or report_path.exists():raise RuntimeError('New independent output required')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh specified source required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_UPPER_NECK_SEAM_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output)
