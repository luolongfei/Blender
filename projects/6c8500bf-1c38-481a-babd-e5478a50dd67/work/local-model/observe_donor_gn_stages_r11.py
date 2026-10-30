"""Read-only actual donor GN stage geometry, with render branches selected.

Only temporary copied node trees are rewired. Original node trees, curve data,
body pose and the input blend are restored/left unsaved. No beauty render.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import bpy
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
MAIN='R11 Donor / long hair main';SECONDARY='R11 Donor / long hair strands'
STAGES=(('roll','Group.002'),('profile','Group.001'),('interpolate','Group.005'),
        ('parting','Group.014'),('trim','Group'),('big-clump','Group.008'),
        ('small-clump','Group.006'),('noise-subset','Group.004'),
        ('shrinkwrap-subset','Group.007'),('main-joined','Join Geometry'),
        ('secondary-profile','Group.003'),('secondary-duplicate','Group.010'),
        ('secondary-clump','Group.011'),('secondary-noise','Group.012'),
        ('full','Join Geometry.001'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_value(v):
    if v is None or isinstance(v,(str,bool,int,float)):return v
    if isinstance(v,bpy.types.ID):return {'type':v.bl_rna.identifier,'name':v.name}
    try:return [json_value(q) for q in v]
    except TypeError:return str(v)


def node_inputs(group):
    result=[]
    for node in group.nodes:
        row={'name':node.name,'type':node.bl_idname,'label':node.label,
             'group':node.node_tree.name if node.type=='GROUP' and node.node_tree else None,'mute':node.mute,
             'inputs':[],'constantOutputs':[]}
        for socket in node.inputs:
            row['inputs'].append({'name':socket.name,'identifier':socket.identifier,'type':socket.type,
                'linked':socket.is_linked,'value':json_value(socket.default_value) if hasattr(socket,'default_value') else None})
        for socket in node.outputs:
            if hasattr(socket,'default_value'):row['constantOutputs'].append({'name':socket.name,'identifier':socket.identifier,'value':json_value(socket.default_value)})
        result.append(row)
    return result


def capture(obj,prefix,arrays,evaluated=True):
    e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()) if evaluated else obj
    if e.type!='CURVES':raise RuntimeError('Expected actual CURVES output at '+prefix)
    data=e.data;q=np.empty(len(data.position_data)*3,np.float32);data.position_data.foreach_get('vector',q);q=q.reshape(-1,3)
    matrix=np.asarray(e.matrix_world);world=q@matrix[:3,:3].T+matrix[:3,3]
    starts=np.array([c.first_point_index for c in data.curves],np.int32);lengths=np.array([c.points_length for c in data.curves],np.int32)
    arrays[prefix+'_world']=world.astype(np.float32);arrays[prefix+'_starts']=starts;arrays[prefix+'_lengths']=lengths
    radius=data.attributes.get('radius');radius_report=None
    if radius:
        values=np.empty(len(radius.data),np.float32);radius.data.foreach_get('value',values)
        values*=float(np.linalg.svd(matrix[:3,:3],compute_uv=False).max());arrays[prefix+'_radius']=values
        radius_report={'domain':radius.domain,'worldMetersP0P50P95P100':np.percentile(values,[0,50,95,100]).tolist()}
    arcs=[];turns=[]
    for start,length in zip(starts,lengths):
        path=world[start:start+length];segments=np.diff(path,axis=0);segment_lengths=np.linalg.norm(segments,axis=1)
        arcs.append(float(segment_lengths.sum()))
        if len(segments)>1:
            unit=segments/np.maximum(segment_lengths[:,None],1e-12)
            turns.extend(np.degrees(np.arccos(np.clip(np.sum(unit[:-1]*unit[1:],axis=1),-1,1))).tolist())
    if not np.isfinite(world).all():raise RuntimeError('Nonfinite GN coordinates at '+prefix)
    return {'prefix':prefix,'object':obj.name,'evaluated':evaluated,'curves':len(starts),'points':len(q),
            'worldBounds':[world.min(0).tolist(),world.max(0).tolist()],
            'radius':radius_report,'arcLengthMetersP5P50P95':np.percentile(arcs,[5,50,95]).tolist(),
            'segmentTurnDegreesP50P95P99Max':np.percentile(turns,[50,95,99,100]).tolist() if turns else [],
            'pointCountPerCurveP0P50P100':np.percentile(lengths,[0,50,100]).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=(P/a.source).resolve();out=(P/a.output).resolve();npz=out.with_suffix('.npz')
    for path in (source,out,npz):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or out.exists() or npz.exists():raise RuntimeError('Fresh source and unused outputs required')
    before=sha(source);rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;obj=bpy.data.objects[MAIN]
    modifiers=[m for m in obj.modifiers if m.type=='NODES' and m.show_render]
    if len(modifiers)!=1:raise RuntimeError('Expected one active main styling GN modifier')
    modifier=modifiers[0];tree=modifier.node_group;arrays={};rows=[];memo={};replacements=[]
    helpers=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        rows.append(capture(obj,'main-raw',arrays,evaluated=False))
        rows.append(capture(bpy.data.objects[SECONDARY],'secondary-raw',arrays,evaluated=False))
        rows.append(capture(bpy.data.objects[SECONDARY],'secondary-direct-evaluated',arrays))
        clone=helpers['copy_render_branch'](tree,memo,replacements);modifier.node_group=clone
        outputs=[n for n in clone.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output]
        if len(outputs)!=1:raise RuntimeError('Expected one active geometry output')
        output=next(s for s in outputs[0].inputs if s.type=='GEOMETRY')
        for label,node_name in STAGES:
            node=clone.nodes.get(node_name)
            if node is None:raise RuntimeError('Recorded stage node missing: '+node_name)
            sockets=[s for s in node.outputs if s.type=='GEOMETRY']
            if len(sockets)!=1:raise RuntimeError('Ambiguous stage geometry output: '+node_name)
            for link in list(output.links):clone.links.remove(link)
            clone.links.new(sockets[0],output);clone.update_tag();obj.update_tag();bpy.context.view_layer.update()
            row=capture(obj,'main-'+label,arrays);row['stageNode']=node_name;row['stageGroup']=node.node_tree.name if node.type=='GROUP' else None
            rows.append(row);print('R11_GN_STAGE '+json.dumps({'stage':label,'curves':row['curves'],'points':row['points']}),flush=True)
        # Capture actual head once for matched projections and contact studies.
        head=bpy.data.objects['R11 Female head study'].evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=head.to_mesh()
        try:
            q=np.empty(len(mesh.vertices)*3,np.float32);mesh.vertices.foreach_get('co',q);q=q.reshape(-1,3);m=np.asarray(head.matrix_world)
            arrays['head_world']=(q@m[:3,:3].T+m[:3,3]).astype(np.float32);mesh.calc_loop_triangles()
            arrays['head_triangles']=np.array([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
        finally:head.to_mesh_clear()
        original_inputs=node_inputs(tree)
        metadata=runpy.run_path(str(HERE/'export_hair_donor_guides_r10.py'))
        secondary_mods=[{'name':m.name,'enabled':m.show_render,'group':m.node_group.name if m.type=='NODES' and m.node_group else None,
                         'properties':metadata['json_value'](m)} for m in bpy.data.objects[SECONDARY].modifiers]
    finally:
        modifier.node_group=tree;rig.data.pose_position=pose;obj.update_tag();bpy.context.view_layer.update()
    if sha(source)!=before:raise RuntimeError('Saved source changed')
    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
    report={'source':a.source,'sourceSha256':before,'helperSha256':sha(__file__),'arrays':str(npz.relative_to(P)),
            'arraysSha256':sha(npz),'stages':rows,'originalMainNodeInputs':original_inputs,
            'secondaryModifiers':secondary_mods,'renderBranchReplacements':replacements,
            'savedBlend':False,'originalNodesAndPoseRestored':True,
            'scope':'Actual per-stage dense GN geometry, including raw guides, roll/profile/interpolation/clump/noise and each joined branch. Subset stages are explicitly labelled; no final hairstyle or physics acceptance.'}
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_GN_STAGES_OBSERVED '+json.dumps({'output':a.output,'arraysSha256':report['arraysSha256']}),flush=True)
if __name__=='__main__':main()
