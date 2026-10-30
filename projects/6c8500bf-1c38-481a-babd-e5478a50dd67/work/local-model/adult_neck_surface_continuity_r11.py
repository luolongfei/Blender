"""Bridge-only skin transition with exact current head/body shader endpoints.

Copies the current head skin graph, samples its actual evaluated boundary
attributes, and mixes continuously to the existing body Principled surface at
the lower seam. Does not alter head/body data, geometry, normals, rig or keys.
The source bridge mesh/material remain recoverable. No rendering or live edits.
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
MARKER='R11_neck_surface_continuity_candidate'
BLEND='R11 neck body surface blend'


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(data,field='co',width=3,dtype=np.float32):
    a=np.empty((len(data),width),dtype);data.foreach_get(field,a.ravel());return a


def _source_surface(obj,body=False):
    import bpy
    core=_api('adult_neck_bridge_r11');dg=bpy.context.evaluated_depsgraph_get();ev=obj.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        geometry=core['_mesh_arrays'](mesh,ev.matrix_world);ring=core['boundary_ring'](geometry,neck_only=body)
        face_index=np.repeat(np.arange(len(mesh.polygons)),geometry['face_sizes'])
        incident=np.unique(face_index[np.isin(geometry['loops'],ring['ids'])])
        slots=np.unique(_array(mesh.polygons,'material_index',1,np.int32).ravel()[incident])
        if len(slots)!=1:raise RuntimeError('Actual skin boundary has multiple materials: '+obj.name)
        material=mesh.materials[int(slots[0])]
        if material is None or not material.use_nodes:raise RuntimeError('Node-based boundary skin required')
        used={}
        for node in material.node_tree.nodes:
            if node.type=='VERTEX_COLOR':
                name=node.layer_name
                if not name:
                    if mesh.color_attributes.active_color is None:raise RuntimeError('No actual default colour layer')
                    name=mesh.color_attributes.active_color.name
                used[node.name]={'attribute':name,'nodeType':node.type}
            elif node.type=='ATTRIBUTE':
                if node.attribute_type!='GEOMETRY':raise RuntimeError('Non-geometry attribute requires explicit bridge adaptation')
                used[node.name]={'attribute':node.attribute_name,'nodeType':node.type}
        samples={};fields={'FLOAT_COLOR':('color',4),'BYTE_COLOR':('color',4),'FLOAT_VECTOR':('vector',3),'FLOAT':('value',1)}
        for item in used.values():
            name=item['attribute'];attr=mesh.attributes.get(name)
            if attr is None:
                samples[name]={'absent':True};continue
            if attr.domain!='POINT' or attr.data_type not in fields:raise RuntimeError('Unsupported actual skin input: '+name+'/'+attr.domain+'/'+attr.data_type)
            field,width=fields[attr.data_type]
            samples[name]={'absent':False,'type':attr.data_type,'field':field,'width':width,
                'values':_array(attr.data,field,width)[ring['ids']]}
        return {'material':material,'ring':ring,'nodeAttributes':used,'samples':samples}
    finally:ev.to_mesh_clear()


def _bridge_geometry(obj):
    import bpy
    return _api('adult_neck_bridge_r11')['_snapshot'](bpy,obj)[0]


def _parameterize(mesh,top,bottom):
    n=len(mesh['world']);nt=len(top['ids']);nb=len(bottom['ids'])
    if np.max(np.linalg.norm(mesh['world'][:nt]-top['points'],axis=1))>1e-6 or np.max(np.linalg.norm(mesh['world'][-nb:]-bottom['points'],axis=1))>1e-6:
        raise RuntimeError('Actual head/body and bridge boundary correspondence differs')
    loops=mesh['loops'];nxt=np.arange(len(loops))+1;nxt[mesh['face_starts']+mesh['face_sizes']-1]=mesh['face_starts']
    edges=np.unique(np.sort(np.c_[loops,loops[nxt]],axis=1),axis=0);ends=edges.ravel();other=edges[:,::-1].ravel()
    degree=np.bincount(ends,minlength=n)
    if (degree==0).any():raise RuntimeError('Bridge contains an isolated vertex')
    fixed=np.r_[np.arange(nt),np.arange(n-nb,n)];interior=np.ones(n,bool);interior[fixed]=False
    angles=np.r_[top['angles'],bottom['angles']]
    values=np.zeros((n,3));values[:,0]=.5;values[fixed,0]=np.r_[np.zeros(nt),np.ones(nb)]
    values[fixed,1]=np.cos(angles);values[fixed,2]=np.sin(angles)
    for iteration in range(3000):
        target=np.column_stack([np.bincount(ends,weights=values[other,k],minlength=n)/degree for k in range(3)])
        change=.8*(target-values);change[~interior]=0.;values+=change
        error=float(np.abs(change).max())
        if error<1e-10:break
    if error>1e-7:raise RuntimeError('Bridge harmonic material parameter did not converge')
    if np.min(np.linalg.norm(values[:,1:],axis=1))<.1:raise RuntimeError('Bridge angular parameter became singular')
    phase=np.mod(np.arctan2(values[:,2],values[:,1]),2*np.pi)
    t=np.clip(values[:,0],0,1);blend=t*t*(3-2*t);blend[:nt]=0.;blend[-nb:]=1.
    return blend,phase,{'iterations':iteration+1,'maximumFinalUpdate':error,'vertices':n,'topCount':nt,'bottomCount':nb,
        'method':'Actual bridge edge graph harmonic coordinates; exact 0/1 boundary values, smoothstep shader mix and periodic angular sampling.'}


def _copy_body_principled(material,nodes):
    outputs=[n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
    if len(outputs)!=1 or len(outputs[0].inputs['Surface'].links)!=1:raise RuntimeError('Inspect actual body surface graph before adaptation')
    source=outputs[0].inputs['Surface'].links[0].from_node
    if source.type!='BSDF_PRINCIPLED' or any(s.is_linked for s in source.inputs):raise RuntimeError('Body now has a linked/complex shader; preserve it and extend bridge adapter explicitly')
    target=nodes.new('ShaderNodeBsdfPrincipled');target.name='R11 exact retained body surface'
    for prop in ('distribution','subsurface_method'):
        if hasattr(source,prop):setattr(target,prop,getattr(source,prop))
    copied=[]
    for i,socket in enumerate(source.inputs):
        if not hasattr(socket,'default_value'):continue
        dst=target.inputs[i]
        if dst.name!=socket.name:raise RuntimeError('Principled input interface mismatch')
        value=socket.default_value
        try:value=list(value)
        except TypeError:pass
        dst.default_value=value;copied.append({'index':i,'name':socket.name,'value':value})
    return target,copied


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.context.scene.get(MARKER):raise RuntimeError('Fresh independent background candidate required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve());source_sha=_sha(source)
    head=bpy.data.objects[HEAD];body=bpy.data.objects[BODY];bridge=bpy.data.objects[BRIDGE]
    guard=_api('adult_upper_neck_seam_r11_v2');uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    head_before=guard['_contract'](head);body_before=guard['_contract'](body);pose=uv['_pose_contract']()
    old=bridge.data;groups,weights=guard['_weight_matrix'](bridge);old_fake=old.use_fake_user;old.use_fake_user=True
    if len(old.materials)!=1 or old.shape_keys is not None or bridge.material_slots[0].link!='DATA':raise RuntimeError('Inspect unexpected bridge material/shape state')
    original_material=old.materials[0];original_material_hash=face['_material_contract'](original_material)
    new=None;candidate=None;success=False
    try:
        with _api('face_contact_audit_r11')['rest_evaluation']():
            hs=_source_surface(head);bs=_source_surface(body,body=True);geometry=_bridge_geometry(bridge)
            blend,phase,parameter=_parameterize(geometry,hs['ring'],bs['ring']);periodic=_api('adult_neck_bridge_r11')['_periodic']
            new=old.copy();new.name='R11 neck continuous skin attribute mesh';bridge.data=new
            if guard['_weight_matrix'](bridge)[0]!=groups or not np.array_equal(guard['_weight_matrix'](bridge)[1],weights):raise RuntimeError('Bridge mesh copy changed original weight mapping')
            candidate=hs['material'].copy();candidate.name='R11 continuous head to body neck skin';new.materials[0]=candidate
            samples={};mapping={};nt=len(hs['ring']['ids'])
            for ordinal,(name,item) in enumerate(hs['samples'].items()):
                if item['absent']:
                    if new.attributes.get(name) is not None:raise RuntimeError('A missing head attribute would resolve differently on the bridge: '+name)
                    samples[name]={'headAndBridgeAttributeAbsent':True};continue
                target_name='R11 neck sampled source '+str(ordinal);mapping[name]=target_name
                values=periodic(item['values'],hs['ring']['angles'],phase)
                if name=='R04 skin rest position':
                    top_world=periodic(hs['ring']['points'],hs['ring']['angles'],phase)
                    values+=geometry['world']-top_world
                values[:nt]=item['values'];values=values.astype(np.float32)
                kind='FLOAT_COLOR' if item['type']=='BYTE_COLOR' else item['type']
                target=new.attributes.new(target_name,kind,'POINT');target.data.foreach_set(item['field'],values.ravel())
                samples[name]={'target':target_name,'type':kind,'sourceBoundaryMaximumDifference':float(np.abs(values[:nt]-item['values']).max()),
                    'valuesSha256':guard['_sha_array'](values),'restCoordinatePolicy':'Actual head boundary rest coordinates plus physical bridge displacement' if name=='R04 skin rest position' else 'Periodic actual head boundary samples'}
            for node_name,item in hs['nodeAttributes'].items():
                if item['attribute'] not in mapping:continue
                node=candidate.node_tree.nodes[node_name]
                if item['nodeType']=='VERTEX_COLOR':node.layer_name=mapping[item['attribute']]
                else:node.attribute_name=mapping[item['attribute']]
            attr=new.attributes.new(BLEND,'FLOAT','POINT');attr.data.foreach_set('value',blend.astype(np.float32))
            tree=candidate.node_tree;outputs=[n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
            if len(outputs)!=1 or len(outputs[0].inputs['Surface'].links)!=1:raise RuntimeError('Current head surface graph is ambiguous')
            surface=outputs[0].inputs['Surface'];original_link=surface.links[0];original_socket=original_link.from_socket
            body_node,body_inputs=_copy_body_principled(bs['material'],tree.nodes)
            factor=tree.nodes.new('ShaderNodeAttribute');factor.name='R11 head body neck transition';factor.attribute_name=BLEND
            mix=tree.nodes.new('ShaderNodeMixShader');mix.name='R11 continuous boundary surface mix'
            tree.links.new(factor.outputs['Fac'],mix.inputs[0]);tree.links.new(original_socket,mix.inputs[1]);tree.links.new(body_node.outputs['BSDF'],mix.inputs[2])
            tree.links.remove(original_link);tree.links.new(mix.outputs[0],surface)
            bpy.context.view_layer.update();actual=_bridge_geometry(bridge)
            if not np.array_equal(actual['world'],geometry['world']) or not np.array_equal(actual['loops'],geometry['loops']):raise RuntimeError('Material-only pass changed evaluated bridge geometry')
            # Existing custom normals and all old attributes are copied bit for
            # bit. New data only enters the copied bridge's shader graph.
            original_attrs=_api('posterior_groom_r09')['_attributes'](old);new_attrs=_api('posterior_groom_r09')['_attributes'](new)
            for name,item in original_attrs.items():
                other=new_attrs[name]
                if item['domain']!=other['domain'] or item['type']!=other['type'] or not np.array_equal(item['values'],other['values']):raise RuntimeError('Original bridge attribute changed: '+name)
        if guard['_contract'](head)!=head_before or guard['_contract'](body)!=body_before or uv['_pose_contract']()!=pose or _sha(source)!=source_sha:raise RuntimeError('Original head/body/pose/source changed')
        if face['_material_contract'](original_material)!=original_material_hash:raise RuntimeError('Original bridge material changed')
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
            'originalBridgeData':old.name,'originalBridgeFakeUser':old_fake,'originalBridgeMaterial':original_material.name,
            'candidateBridgeData':new.name,'candidateMaterial':candidate.name,'candidateMaterialSha256':face['_material_contract'](candidate),
            'candidateBridgeSha256':_api('posterior_groom_r09')['_data_hash'](new),
            'headMaterial':hs['material'].name,'bodyMaterial':bs['material'].name,'bodyPrincipledInputsCopied':body_inputs,
            'parameterization':parameter,'sampledHeadAttributes':samples,'topShaderWeight':0.,'bottomShaderWeight':1.,
            'originalHeadBodyDataKeysMaterialsUnchanged':True,'originalBridgeGeometryNormalsWeightsAttributesUnchanged':True,
            'savedBlend':False,'appearanceAccepted':False,
            'limits':['Actual side/back/front rendered comparison required; endpoint shader equivalence is not visual acceptance.',
                'Bottom bridge/body geometric tangent discontinuity is retained; this pass repairs only optical input continuity.',
                'Original head and body colours differ; this pass supplies a gradual bridge transition and does not recolour the body.',
                'No new head/neck dynamic deformation approval.']}
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:bridge.data=old;old.use_fake_user=old_fake;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);obj=bpy.data.objects[BRIDGE]
    if _api('posterior_groom_r09')['_data_hash'](obj.data)!=report['candidateBridgeSha256']:raise RuntimeError('Preserve subsequent bridge edits before restore')
    if _api('runtime_face_refresh_r08')['_material_contract'](bpy.data.materials[report['candidateMaterial']])!=report['candidateMaterialSha256']:raise RuntimeError('Preserve subsequent shader edits before restore')
    obj.data.use_fake_user=True;obj.data=bpy.data.meshes[report['originalBridgeData']];obj.data.use_fake_user=report['originalBridgeFakeUser']
    del bpy.context.scene[MARKER];bpy.context.view_layer.update();return {'restored':True,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve();report=output.with_suffix('.neck-surface.json')
    for path in (source,output):path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if source.suffix!='.blend' or output.suffix!='.blend' or source==output or output.exists() or report.exists():raise RuntimeError('Explicit fresh independent output required')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh specified source required')
    result=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    result.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('R11_NECK_SURFACE_SAVED '+json.dumps({'output':result['output'],'outputSha256':result['outputSha256']}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output)
