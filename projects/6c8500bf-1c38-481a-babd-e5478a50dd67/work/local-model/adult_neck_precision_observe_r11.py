"""Read-only, finite-ROI audit of the ACTUAL adult neck, bridge and body.

Run in a fresh background source. Exports current-pose and temporary-REST
geometry, adjacent boundary rows, geometric/shading normals and material inputs.
No mesh/key/material changes, no save and no rendering. Old observations are
never substituted for the current evaluated source.
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

NAMES=('R11 Female head study','R11 Continuous neck bridge','R01 Head')
ROI={'minimum':[-.145,-.12,1.20],'maximum':[.145,.12,1.375]}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(data,field,width=3,dtype=np.float32):
    a=np.empty((len(data),width),dtype);data.foreach_get(field,a.ravel());return a


def _unit(v):
    v=np.asarray(v,float);return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-15)


def _hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def _value(v):
    if isinstance(v,(str,bool,int,float)) or v is None:return v
    try:return list(v)
    except TypeError:return str(v)


def _material(mat):
    if mat is None:return None
    rec={'name':mat.name,'useNodes':mat.use_nodes,'nodes':[],'links':[]}
    if not mat.use_nodes:return rec
    for node in mat.node_tree.nodes:
        rec['nodes'].append({'name':node.name,'type':node.bl_idname,
            'attributeName':getattr(node,'attribute_name',None),'layerName':getattr(node,'layer_name',None),
            'operation':getattr(node,'operation',None),'blendType':getattr(node,'blend_type',None),
            'inputs':[{'index':i,'name':s.name,'linked':s.is_linked,'default':_value(s.default_value)}
                      for i,s in enumerate(node.inputs) if hasattr(s,'default_value')]})
    rec['links']=[(x.from_node.name,x.from_socket.name,x.to_node.name,x.to_socket.name) for x in mat.node_tree.links]
    return rec


def _contract():
    import bpy
    result={}
    for name in NAMES:
        obj=bpy.data.objects[name];mesh=obj.data
        result[name]={'data':mesh.name,'vertices':_hash(_array(mesh.vertices,'co')),
            'loops':_hash(_array(mesh.loops,'vertex_index',1,np.int32)),
            'keys':{k.name:{'value':float(k.value),'mute':bool(k.mute),'relative':k.relative_key.name,'co':_hash(_array(k.data,'co'))}
                    for k in mesh.shape_keys.key_blocks} if mesh.shape_keys else {},
            'matrix':[list(row) for row in obj.matrix_world],
            'materials':[_material(m) for m in mesh.materials],
            'groups':[(g.name,g.index) for g in obj.vertex_groups]}
    return result


def _attributes(mesh,vertex_ids,corner_ids,face_ids):
    result={};index={'POINT':vertex_ids,'CORNER':corner_ids,'FACE':face_ids}
    fields={'FLOAT_COLOR':('color',4,np.float32),'BYTE_COLOR':('color',4,np.float32),
            'FLOAT':('value',1,np.float32),'FLOAT_VECTOR':('vector',3,np.float32),
            'INT':('value',1,np.int32),'BOOLEAN':('value',1,np.bool_)}
    for attr in mesh.attributes:
        if attr.domain not in index or attr.data_type not in fields:continue
        if not any(t in attr.name.lower() for t in ('color','colour','rough','rest position','sculpt_face_set')):continue
        field,width,dtype=fields[attr.data_type]
        result[attr.name]={'domain':attr.domain,'type':attr.data_type,
            'values':_array(attr.data,field,width,dtype)[index[attr.domain]]}
    return result


def _capture(obj):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        T=np.asarray(ev.matrix_world,float);normalT=np.linalg.inv(T[:3,:3]).T
        local=_array(mesh.vertices,'co');world=local@T[:3,:3].T+T[:3,3]
        loops=_array(mesh.loops,'vertex_index',1,np.int32).ravel()
        starts=_array(mesh.polygons,'loop_start',1,np.int32).ravel()
        sizes=_array(mesh.polygons,'loop_total',1,np.int32).ravel()
        face_index=np.repeat(np.arange(len(sizes)),sizes)
        nxt=np.arange(len(loops))+1;nxt[starts+sizes-1]=starts
        edges=np.unique(np.sort(np.c_[loops,loops[nxt]],axis=1),axis=0)
        mesh.calc_loop_triangles();tri=_array(mesh.loop_triangles,'vertices',3,np.int32)
        p=world[tri];area_normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        geometric=np.zeros_like(world)
        for column in range(3):np.add.at(geometric,tri[:,column],area_normal)
        geometric=_unit(geometric)
        # corner_normals includes the actual custom split normals on the bridge.
        if hasattr(mesh,'corner_normals') and len(mesh.corner_normals)==len(loops):
            corner=_array(mesh.corner_normals,'vector').astype(float)
        else:
            corner=_array(mesh.loops,'normal').astype(float)
        corner=_unit(corner@normalT.T);shading=np.zeros_like(world);np.add.at(shading,loops,corner);shading=_unit(shading)
        core=_api('adult_neck_bridge_r11')
        full={'world':world,'loops':loops,'face_starts':starts,'face_sizes':sizes,'triangles':tri}
        if obj.name==NAMES[0]:boundary=core['boundary_ring'](full)
        elif obj.name==NAMES[2]:boundary=core['boundary_ring'](full,neck_only=True)
        else:boundary=None
        lo=np.asarray(ROI['minimum']);hi=np.asarray(ROI['maximum'])
        inside=((world>=lo)&(world<=hi)).all(1)
        touched=np.bincount(face_index,weights=inside[loops],minlength=len(sizes))>0
        faces=np.flatnonzero(touched);corners=np.flatnonzero(touched[face_index]);ids=np.unique(loops[corners])
        mapping=np.full(len(world),-1,np.int32);mapping[ids]=np.arange(len(ids));new_sizes=sizes[faces]
        arrays={'world':world[ids],'original_vertex_ids':ids,'loops':mapping[loops[corners]],
            'face_starts':np.r_[0,np.cumsum(new_sizes[:-1])].astype(np.int32),'face_sizes':new_sizes,
            'original_face_ids':faces,'original_corner_ids':corners,'corner_normals_world':corner[corners],
            'geometric_normals_world':geometric[ids],'shading_normals_world':shading[ids],
            'face_material_indices':_array(mesh.polygons,'material_index',1,np.int32).ravel()[faces]}
        attributes=_attributes(mesh,ids,corners,faces)
        return {'object':obj.name,'world':world,'edges':edges,'loops':loops,'face_starts':starts,'face_sizes':sizes,
            'geometric':geometric,'shading':shading,'boundary':boundary,
            'arrays':arrays,'attributes':attributes,'materials':[_material(m) for m in mesh.materials],
            'matrixWorld':T.tolist(),'fullCounts':[len(world),len(sizes)],'roiCounts':[len(ids),len(faces)],
            'modifiers':[{'name':m.name,'type':m.type,'viewport':m.show_viewport,'render':m.show_render,
                'levels':getattr(m,'levels',None),'renderLevels':getattr(m,'render_levels',None)} for m in obj.modifiers]}
    finally:ev.to_mesh_clear()


def _neighbors(edges,count):
    result=[[] for _ in range(count)]
    for a,b in edges:result[int(a)].append(int(b));result[int(b)].append(int(a))
    return result


def _seam(one,two,two_boundary_ids):
    ids=np.asarray(one['boundary']['ids'],np.int32);p=one['world'][ids]
    q=two['world'][two_boundary_ids];dist=((p[:,None,:]-q[None,:,:])**2).sum(2)
    match=np.asarray(two_boundary_ids)[dist.argmin(1)];error=np.sqrt(dist.min(1))
    if len(np.unique(match))!=len(ids):raise RuntimeError('Seam correspondence is not one-to-one; inspect topology')
    # Cross-boundary directions use original edge adjacency, excluding seam
    # edges; normal projection removes uneven tangential sample spacing.
    normals={};inward=[];validity=[];counts={}
    for label,surface,indices in [('head_or_body',one,ids),('bridge',two,match)]:
        neighbors=_neighbors(surface['edges'],len(surface['world']));excluded=set(int(x) for x in indices)
        # An original quad corner can have both EDGE neighbors on the seam,
        # while the opposite quad corner is a valid inward surface neighbor.
        # Use all incident-face vertices, without inventing a second-ring
        # tangent where the actual incident surface gives no inward sample.
        loops=surface['loops'];starts=surface['face_starts'];sizes=surface['face_sizes']
        corner_faces=np.repeat(np.arange(len(sizes)),sizes)
        incident=np.unique(corner_faces[np.isin(loops,indices)])
        face_neighbors={int(i):set() for i in indices}
        for face in incident:
            fv=set(int(i) for i in loops[starts[face]:starts[face]+sizes[face]])
            for index in fv&excluded:face_neighbors[index].update(fv-excluded)
        tangent=_unit(np.roll(p,-1,axis=0)-np.roll(p,1,axis=0));directions=[];adjacent=set();edge_counts=[];face_counts=[];valid=[]
        for k,index in enumerate(indices):
            edge_row=[x for x in neighbors[int(index)] if x not in excluded]
            row=sorted(face_neighbors[int(index)]);edge_counts.append(len(edge_row));face_counts.append(len(row))
            if not row:
                directions.append(np.zeros(3));valid.append(False);continue
            adjacent.update(row);d=surface['world'][row].mean(0)-surface['world'][index]
            d-=tangent[k]*np.dot(d,tangent[k]);directions.append(d);valid.append(np.linalg.norm(d)>1e-12)
        normals[label]={'ids':indices,'adjacent_ids':np.asarray(sorted(adjacent),np.int32),
            'geometry':surface['geometric'][indices],'shading':surface['shading'][indices]}
        inward.append(_unit(directions));validity.append(np.asarray(valid,bool));counts[label]=(np.asarray(edge_counts,np.int32),np.asarray(face_counts,np.int32))
        if not all(valid):
            bad=np.asarray(indices)[~np.asarray(valid,bool)]
            print('R11_NECK_TANGENT_UNMEASURED '+json.dumps({'object':surface['object'],'boundaryRole':label,
                'vertexIndices':bad.tolist(),'world':surface['world'][bad].tolist(),
                'reason':'No non-boundary vertex in any incident face, or a zero cross-seam direction. No continuity is inferred at these samples.'}),flush=True)
    def angle(a,b):return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1),-1,1)))
    ga=angle(normals['head_or_body']['geometry'],normals['bridge']['geometry'])
    sa=angle(normals['head_or_body']['shading'],normals['bridge']['shading'])
    ta=angle(inward[0],-inward[1]);valid=validity[0]&validity[1];ta[~valid]=np.nan
    return {'world':p,'first_ids':ids,'bridge_ids':match,'position_error':error,
        'geometry_normal_angle_deg':ga,'shading_normal_angle_deg':sa,'cross_seam_tangent_angle_deg':ta,
        'first_inward_unit':inward[0],'bridge_inward_unit':inward[1],
        'tangent_valid':valid,'first_edge_inward_counts':counts['head_or_body'][0],'first_face_inward_counts':counts['head_or_body'][1],
        'bridge_edge_inward_counts':counts['bridge'][0],'bridge_face_inward_counts':counts['bridge'][1],
        'first_adjacent_ids':normals['head_or_body']['adjacent_ids'],'bridge_adjacent_ids':normals['bridge']['adjacent_ids']}


def _sections(snap):
    world=snap['world'];rows=[]
    for z in np.arange(1.255,1.371,.005):
        p=world[(abs(world[:,2]-z)<.00075)&(abs(world[:,0])<.11)]
        if len(p)<10:continue
        rows.append({'worldZ':float(z),'verticesIn1p5mmBand':len(p),'min':p.min(0).tolist(),
            'max':p.max(0).tolist(),'xWidth':float(np.ptp(p[:,0])),'yDepth':float(np.ptp(p[:,1]))})
    return rows


def observe(source_relative,output_relative):
    import bpy
    root=(project_path()/'work/reconstruction-r02').resolve()
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();npz=output.with_suffix('.npz')
    for p in (source,output):p.relative_to(root)
    if source.suffix!='.blend' or output.suffix!='.json' or output.exists() or npz.exists():raise RuntimeError('Explicit source and fresh JSON/NPZ outputs required')
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh background source required; no live/source writes')
    source_sha=_sha(source);before=_contract();pose=_api('production_uv_r08')['_pose_contract']();frame=[bpy.context.scene.frame_current,bpy.context.scene.frame_subframe]
    arrays={};report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
        'roiWorld':ROI,'frame':frame,'states':{},'savedBlend':False,'appearanceAccepted':False,
        'scope':'Actual finite lower-neck ROI in saved current pose and temporary REST, no geometry/shader edits. Boundary normal/tangent angles are diagnostics, not acceptance thresholds.'}
    def capture_state(state):
        snaps={name:_capture(bpy.data.objects[name]) for name in NAMES};record={'objects':{},'seams':{}}
        for i,name in enumerate(NAMES):
            snap=snaps[name];prefix=f'{state}_{i}'
            for key,value in snap['arrays'].items():arrays[prefix+'_'+key]=value
            attributes={}
            for j,(attr,item) in enumerate(snap['attributes'].items()):
                key=prefix+'_attribute_'+str(j);arrays[key]=item['values']
                attributes[attr]={'domain':item['domain'],'type':item['type'],'array':key}
            if snap['boundary'] is not None:
                arrays[prefix+'_boundary_original_ids']=np.asarray(snap['boundary']['ids'],np.int32)
                arrays[prefix+'_boundary_world']=snap['boundary']['points']
            record['objects'][name]={'prefix':prefix,'fullCounts':snap['fullCounts'],'roiCounts':snap['roiCounts'],
                'matrixWorld':snap['matrixWorld'],'modifiers':snap['modifiers'],'materials':snap['materials'],
                'attributes':attributes,'sectionBands':_sections(snap)}
        bridge=snaps[NAMES[1]];nt=int(bpy.data.objects[NAMES[1]]['R11_boundary_top_count']);nb=int(bpy.data.objects[NAMES[1]]['R11_boundary_bottom_count'])
        for label,first,ids in [('head_bridge',snaps[NAMES[0]],np.arange(nt)),
                ('body_bridge',snaps[NAMES[2]],np.arange(len(bridge['world'])-nb,len(bridge['world'])))]:
            seam=_seam(first,bridge,ids);prefix=state+'_'+label
            for key,value in seam.items():arrays[prefix+'_'+key]=value
            record['seams'][label]={'prefix':prefix,'count':len(seam['world']),
                'positionErrorMaxMeters':float(seam['position_error'].max()),
                'geometricNormalAngleQuantilesDeg':np.quantile(seam['geometry_normal_angle_deg'],[0,.5,.95,1]).tolist(),
                'shadingNormalAngleQuantilesDeg':np.quantile(seam['shading_normal_angle_deg'],[0,.5,.95,1]).tolist(),
                'crossSeamTangentAngleQuantilesDeg':np.quantile(seam['cross_seam_tangent_angle_deg'][seam['tangent_valid']],[0,.5,.95,1]).tolist() if seam['tangent_valid'].any() else None,
                'crossSeamTangentMeasuredCount':int(seam['tangent_valid'].sum()),
                'crossSeamTangentUnmeasuredFirstVertexIds':seam['first_ids'][~seam['tangent_valid']].tolist(),
                'crossSeamTangentUnmeasuredBridgeVertexIds':seam['bridge_ids'][~seam['tangent_valid']].tolist(),
                'edgeOnlyMissingFirstVertexIds':seam['first_ids'][seam['first_edge_inward_counts']==0].tolist(),
                'edgeOnlyMissingBridgeVertexIds':seam['bridge_ids'][seam['bridge_edge_inward_counts']==0].tolist(),
                'tangentMethod':'Mean of non-boundary vertices in actual incident faces, projected perpendicular to seam tangent. Missing samples explicitly remain unmeasured, never counted as continuous.'}
        report['states'][state]=record
    capture_state('current')
    with _api('face_contact_audit_r11')['rest_evaluation']():capture_state('rest')
    if _contract()!=before or _api('production_uv_r08')['_pose_contract']()!=pose or _sha(source)!=source_sha:
        raise RuntimeError('Observation changed source content or pose; refusing success report')
    np.savez_compressed(npz,**arrays);report['arraysSha256']=_sha(npz);report['sourceContractUnchanged']=True
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NECK_PRECISION_OBSERVED '+json.dumps({'output':str(output),'arraysSha256':report['arraysSha256'],'sourceSha256':source_sha}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);observe(a.source,a.output)
