"""One neutral actual lash/eyelid observation; no render, save or shader test.

Records complete material node routes, actual four lash curves, retained/raw
curve positions, exact root mapping and current neutral eye-region triangles.
Designed for offline growth-direction, background and burial attribution.
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def scalar(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if hasattr(value,'name_full'):return {'dataBlock':value.name_full}
    try:return [scalar(v) for v in value]
    except TypeError:return str(value)


def material_record(mat):
    if mat is None:return None
    result={'name':mat.name,'useNodes':mat.use_nodes,'diffuseColor':list(mat.diffuse_color),
            'surfaceRenderMethod':getattr(mat,'surface_render_method',None),
            'useBackfaceCulling':mat.use_backface_culling,'nodes':[],'links':[]}
    def tree_record(tree,seen):
        if tree is None or tree.as_pointer() in seen:return None
        seen.add(tree.as_pointer());nodes=[]
        for node in tree.nodes:
            row={'name':node.name,'type':node.type,'blIdname':node.bl_idname,'mute':node.mute,
                 'inputs':[{'index':i,'name':s.name,'identifier':s.identifier,'linked':s.is_linked,
                            'default':scalar(s.default_value) if hasattr(s,'default_value') else None}
                           for i,s in enumerate(node.inputs)]}
            for prop in ('target','is_active_output','operation','model','parametrization'):
                if hasattr(node,prop):row[prop]=scalar(getattr(node,prop))
            if getattr(node,'node_tree',None):row['groupTree']=tree_record(node.node_tree,seen)
            nodes.append(row)
        return {'name':tree.name,'nodes':nodes,
                'links':[{'fromNode':l.from_node.name,'fromSocket':l.from_socket.name,'fromIdentifier':l.from_socket.identifier,
                          'toNode':l.to_node.name,'toSocket':l.to_socket.name,'toIdentifier':l.to_socket.identifier,
                          'valid':l.is_valid,'muted':l.is_muted} for l in tree.links]}
    result['nodeTree']=tree_record(mat.node_tree,set()) if mat.use_nodes else None
    return result


def observe(source_relative,output_relative):
    import bpy
    p=project_path().resolve();source=(p/source_relative).resolve();out=(p/output_relative).resolve();npz=out.with_suffix('.npz')
    for path in (source,out,npz):path.relative_to(p/'work/reconstruction-r02')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists() or npz.exists():
        raise RuntimeError('Fresh independent source and unused observation outputs required')
    binding=api('face_fibre_binding_r11');mid=api('adult_blink_midpath_observe_r11');pose_api=api('production_uv_r08')
    source_sha=sha(source);pose_before=pose_api['_pose_contract']();scene=bpy.context.scene;head=bpy.data.objects[binding['HEAD']]
    public=('R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile','R11 warm closed smile')
    keys=head.data.shape_keys.key_blocks;old={name:float(keys[name].value) for name in public}
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
            'evaluation':'Current frame, temporary armature REST, public smiles and blinks zero; other static keys/relief preserved.',
            'sourceSaved':False,'appearanceAccepted':False,'materials':{},'lashes':{},'eyes':{},
            'viewLayerMaterialOverrides':{layer.name:layer.material_override.name if layer.material_override else None for layer in scene.view_layers},
            'cyclesCurveShape':getattr(scene.cycles,'curve_shape',None),
            'limits':'Only actual neutral observation. No new collision approval, full binding validation, rendering or geometry modification.'}
    arrays={};objects=[bpy.data.objects['R11 Adult lashes '+part+' '+side] for side in ('L','R') for part in ('upper','lower')]
    raw_contracts={o.name:binding['_curve_contract'](o.data) for o in objects}
    record=json.loads(scene[binding['MARKER']])
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            for name in public:keys[name].value=0.
            bpy.context.view_layer.update()
            if any(abs(float(keys[name].value))>1e-7 for name in public):raise RuntimeError('Neutral public control is overridden')
            mesh=mid['capture_mesh'](head,binding)
            eyes={side:mid['capture_mesh'](bpy.data.objects['R11 Adult sclera '+side],binding) for side in ('L','R')}
            mappings={o.name:binding['_mapping'](o.data) for o in objects}
            extra=np.unique(np.concatenate([v['vertices'].ravel() for v in mappings.values()]))
            ids,faces,index_map,boxes=mid['fixed_roi'](mesh,eyes,extra)
            triangles,tri_ids,tri_polygons=mid['roi_triangles'](mesh,faces,index_map)
            arrays.update(head_world=mesh['world'][ids],head_triangles=triangles,head_originalVertexIds=ids,
                          head_originalTriangleIds=tri_ids,head_trianglePolygonIds=tri_polygons)
            report.update(headCounts=mesh['counts'],headTopologySha256=mesh['topologySha256'],headMatrixWorld=mesh['matrix'].tolist(),
                          roiBoxes=boxes,roiVertexCount=len(ids),allKeyValues={k.name:float(k.value) for k in keys},
                          headModifierOrder=[m.name for m in head.modifiers])
            for side,eye in eyes.items():
                for field in ('world','triangles','matrix'):arrays['eye_'+side+'_'+field]=eye[field]
                report['eyes'][side]={'centerWorld':((eye['world'].min(0)+eye['world'].max(0))*.5).tolist(),
                                     'counts':eye['counts'],'topologySha256':eye['topologySha256']}
            for obj in objects:
                tag=obj.name.replace('R11 Adult lashes ','').replace(' ','_');mapping=mappings[obj.name]
                evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=evaluated.data
                actual_local=binding['_array'](data.position_data,'vector');actual_world=binding['_transform'](actual_local,evaluated.matrix_world)
                counts=np.asarray([c.points_length for c in data.curves],np.int32)
                if not np.array_equal(counts,mapping['counts']):raise RuntimeError('Actual lash count differs from recorded mapping')
                arrays[tag+'_world']=actual_world;arrays[tag+'_matrix']=np.asarray(evaluated.matrix_world)
                arrays[tag+'_rawLocal']=binding['_array'](obj.data.position_data,'vector')
                arrays[tag+'_counts']=counts;arrays[tag+'_radius']=binding['_array'](data.attributes['radius'].data,'value',1).ravel()
                arrays[tag+'_boundTrianglesWorld']=mesh['world'][mapping['vertices']]
                for field,value in mapping.items():arrays[tag+'_mapping_'+field]=value
                attrs={}
                for name in ('material_index','curve_type','normal_mode'):
                    attr=data.attributes.get(name)
                    if attr is not None:
                        values=binding['_array'](attr.data,'value',1,np.int32).ravel();arrays[tag+'_'+name]=values
                        attrs[name]={'domain':attr.domain,'type':attr.data_type,'uniqueValues':np.unique(values).tolist()}
                report['lashes'][obj.name]={'arrayTag':tag,'curves':len(counts),'points':len(actual_world),
                        'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hideGet':obj.hide_get(),
                        'visibleCamera':obj.visible_camera,'parent':obj.parent.name if obj.parent else None,
                        'modifiers':[{'name':m.name,'type':m.type,'showRender':m.show_render,'showViewport':m.show_viewport,
                                      'nodeGroup':m.node_group.name if m.type=='NODES' and m.node_group else None} for m in obj.modifiers],
                        'dataMaterials':[m.name if m else None for m in obj.data.materials],
                        'actualMaterials':[m.name if m else None for m in data.materials],
                        'objectSlots':[{'link':s.link,'material':s.material.name if s.material else None} for s in obj.material_slots],
                        'actualAttributes':attrs,'recordedBindingItem':record['objects'][obj.name]}
                for material in list(obj.data.materials)+list(data.materials):
                    if material is not None:report['materials'][material.name]=material_record(material)
    finally:
        for name,value in old.items():keys[name].value=value
        bpy.context.view_layer.update()
    if pose_api['_pose_contract']()!=pose_before or sha(source)!=source_sha:raise RuntimeError('Neutral observation altered pose or saved source')
    if any(binding['_curve_contract'](o.data)!=raw_contracts[o.name] for o in objects):raise RuntimeError('Neutral observation changed raw lash data')
    np.savez_compressed(npz,**arrays);report.update(arrays=str(npz.relative_to(p)),arraysSha256=sha(npz),controlsAndPoseRestored=True,rawLashesUnchanged=True)
    with out.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_LASH_NEUTRAL_OBSERVED '+json.dumps({'output':output_relative,'arraysSha256':report['arraysSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);observe(args.source,args.output)
