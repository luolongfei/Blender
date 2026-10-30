"""Independent groom repair from measured raw-volume and GN-stage failures.

Retains editable guides and the author's interpolation/parting/clump/noise
structure, while disabling the inappropriate roll and limiting secondary
displacement. Original curves/node tree remain retained data. No import work.
"""
import hashlib,json,math,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
MAIN='R11 Donor / long hair main';HEAD='R11 Female head study'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)


def compress_back_volume(path,kind,nearest):
    """Soften only excessive rear/side scalp lift, preserving crown layering."""
    path=np.asarray(path,dtype=float);delta=np.zeros_like(path);before=[]
    for j,p in enumerate(path):
        if j==0 or (kind=='gather' and j>=len(path)-2):continue
        if kind not in ('gather','author-crown-layer'):continue
        if p[2]<1.405 or (p[1]<.005 and abs(p[0])<.065):continue
        hit,normal,signed=nearest(p)
        upper=float(smooth((p[2]-1.405)/.045))
        # Higher crown keeps 14 mm volume; lower posterior gathers use 8 mm.
        # Excess is compressed, not flattened to one constant shell.
        allowed=.008+.006*float(smooth((p[2]-1.43)/.08))
        if signed>allowed:
            amount=(signed-allowed)*.85*upper
            delta[j]=-np.asarray(normal)*amount;before.append(float(signed))
    for unused in range(2):
        candidate=delta.copy();candidate[1:-1]=.25*delta[:-2]+.5*delta[1:-1]+.25*delta[2:]
        candidate[0]=0.
        if kind=='gather':candidate[-2:]=0.
        delta=candidate
    result=path+delta;maxpush=0.
    for j,p in enumerate(result):
        if j==0 or (kind=='gather' and j>=len(path)-2):continue
        hit,n,signed=nearest(p)
        if signed<.00020:
            push=.00025-signed
            if push>.012:raise RuntimeError('Posterior volume pass exceeds existing skin correction limit')
            result[j]=p+np.asarray(n)*push;maxpush=max(maxpush,float(push))
    result[0]=path[0]
    if kind=='gather':result[-2:]=path[-2:]
    return result,{'compressedSourcePoints':len(before),'largestOriginalExcessGapMeters':max(before,default=0.),
                   'maximumStoredMoveMeters':float(np.linalg.norm(result-path,axis=1).max()),'maximumSkinCorrectionMeters':maxpush}


def socket(node,name):
    found=[s for s in node.inputs if s.name==name]
    if len(found)!=1:raise RuntimeError('Expected unique input '+node.name+'/'+name)
    return found[0]


def geometry_socket(collection):
    found=[s for s in collection if s.type=='GEOMETRY']
    if len(found)!=1:raise RuntimeError('Expected one geometry socket')
    return found[0]


def sample_position(node):
    found=[s for s in node.inputs if s.type=='VECTOR' and s.name in ('Sample Position','Source Position','Position')]
    if len(found)!=1:raise RuntimeError('Expected one sample-position vector on '+node.bl_idname)
    return found[0]


def constant(tree,node_name,input_name,value,changes):
    node=tree.nodes[node_name];s=socket(node,input_name);links=list(s.links)
    before={'value':list(s.default_value) if hasattr(s.default_value,'__len__') else s.default_value,
            'links':[{'node':l.from_node.name,'socket':l.from_socket.identifier} for l in links]}
    for link in links:tree.links.remove(link)
    s.default_value=value;changes.append({'node':node_name,'input':input_name,'before':before,'after':value})


def factor_envelope(tree,changes):
    nodes=tree.nodes;links=tree.links
    parameter=nodes.new('GeometryNodeSplineParameter');parameter.label='R11 actual curve parameter'
    inverse=nodes.new('ShaderNodeMath');inverse.operation='SUBTRACT';inverse.inputs[0].default_value=1.;links.new(parameter.outputs['Factor'],inverse.inputs[1])
    product=nodes.new('ShaderNodeMath');product.operation='MULTIPLY';links.new(parameter.outputs['Factor'],product.inputs[0]);links.new(inverse.outputs[0],product.inputs[1])
    for name,amount in (('Group.008',.05),('Group.006',.12),('Group.004',.8),('Group.012',.8)):
        s=socket(nodes[name],'Factor');previous=[{'node':l.from_node.name,'socket':l.from_socket.identifier} for l in s.links]
        for link in list(s.links):links.remove(link)
        scale=nodes.new('ShaderNodeMath');scale.operation='MULTIPLY';scale.inputs[1].default_value=4.*amount;links.new(product.outputs[0],scale.inputs[0]);links.new(scale.outputs[0],s)
        scale.label='R11 pinned-end factor '+name
        changes.append({'node':name,'input':'Factor','previousLinks':previous,'after':'4*t*(1-t)*'+str(amount),'rootAndTipFactor':0.})
    return parameter


def contact_and_profile(tree,head,unit_scale,parameter):
    nodes=tree.nodes;links=tree.links
    outputs=[n for n in nodes if n.type=='GROUP_OUTPUT' and n.is_active_output]
    output=next(s for s in outputs[0].inputs if s.type=='GEOMETRY')
    if len(output.links)!=1:raise RuntimeError('Expected joined source geometry')
    geometry=output.links[0].from_socket;links.remove(output.links[0])
    info=nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE';info.inputs['Object'].default_value=head;info.inputs['As Instance'].default_value=False;info.label='R11 actual full head contact surface'
    position=nodes.new('GeometryNodeInputPosition')
    nearest=nodes.new('GeometryNodeProximity');nearest.target_element='FACES';links.new(info.outputs['Geometry'],geometry_socket(nearest.inputs));links.new(position.outputs['Position'],sample_position(nearest))
    normal=nodes.new('GeometryNodeInputNormal')
    sampled=nodes.new('GeometryNodeSampleNearestSurface');sampled.data_type='FLOAT_VECTOR';links.new(info.outputs['Geometry'],geometry_socket(sampled.inputs));links.new(normal.outputs['Normal'],sampled.inputs['Value']);links.new(position.outputs['Position'],sample_position(sampled))
    difference=nodes.new('ShaderNodeVectorMath');difference.operation='SUBTRACT';links.new(position.outputs['Position'],difference.inputs[0]);links.new(nearest.outputs['Position'],difference.inputs[1])
    signed=nodes.new('ShaderNodeVectorMath');signed.operation='DOT_PRODUCT';links.new(difference.outputs['Vector'],signed.inputs[0]);links.new(sampled.outputs['Value'],signed.inputs[1])
    required=nodes.new('ShaderNodeMath');required.operation='SUBTRACT';required.inputs[0].default_value=.00020/unit_scale;links.new(signed.outputs['Value'],required.inputs[1])
    positive=nodes.new('ShaderNodeMath');positive.operation='MAXIMUM';positive.inputs[1].default_value=0.;links.new(required.outputs[0],positive.inputs[0])
    offset=nodes.new('ShaderNodeVectorMath');offset.operation='SCALE';links.new(sampled.outputs['Value'],offset.inputs[0]);links.new(positive.outputs[0],offset.inputs['Scale'])
    corrected=nodes.new('GeometryNodeSetPosition');corrected.name='R11 full head clearance';links.new(geometry,corrected.inputs['Geometry']);links.new(offset.outputs['Vector'],corrected.inputs['Offset'])
    taper=nodes.new('ShaderNodeMapRange');taper.clamp=True;taper.interpolation_type='SMOOTHERSTEP'
    taper.inputs['From Min'].default_value=.65;taper.inputs['From Max'].default_value=1.;taper.inputs['To Min'].default_value=.00055;taper.inputs['To Max'].default_value=.00004
    links.new(parameter.outputs['Factor'],taper.inputs['Value'])
    profile=nodes.new('GeometryNodeSetCurveRadius');profile.name='R11 strand tip taper';links.new(corrected.outputs['Geometry'],geometry_socket(profile.inputs));links.new(taper.outputs['Result'],profile.inputs['Radius']);links.new(geometry_socket(profile.outputs),output)
    return {'clearanceNode':corrected.name,'profileNode':profile.name,'worldSkinClearanceMeters':.00020,
            'radiusWorldRootMeters':.00055*unit_scale,'radiusWorldTipMeters':.00004*unit_scale,'taperStartsAtCurveParameter':.65,
            'sourceGeometryNode':geometry.node.name}


def apply(*,expected_source_relative):
    import bpy
    from mathutils import Vector
    source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source or bpy.context.mode!='OBJECT':raise RuntimeError('Explicit independent background source required')
    rig=bpy.data.objects['Character_Rig']
    if rig.data.pose_position!='REST':raise RuntimeError('Caller must enter REST')
    obj=bpy.data.objects[MAIN];head=bpy.data.objects[HEAD]
    if not obj.get('R11_donor_restyle') or obj.get('R11_donor_gn_finish_v3'):raise RuntimeError('Fresh restyled donor required')
    fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));coords=fit['_coords'];world=fit['_world']
    source_sha=sha(source);before=coords(obj);world_before=world(before,obj.matrix_world);head_before=hashlib.sha256(coords(head).tobytes()).hexdigest()
    scalp,skin,head_report=fit['_head_trees'](head)
    def nearest(p):
        hit,n,index,distance=skin.find_nearest(Vector(p))
        if hit is None:raise RuntimeError('No actual head nearest point')
        return np.asarray(hit),np.asarray(n),float((Vector(p)-hit).dot(n))
    style=json.loads(bpy.context.scene['R11_actual_donor_restyle'])
    kinds={r['guideIndex']:r['kind'] for r in style['guideReports'] if r['object']==MAIN}
    starts=np.array([c.first_point_index for c in obj.data.curves]);lengths=np.array([c.points_length for c in obj.data.curves]);new=world_before.copy();rows=[]
    for k,(start,length) in enumerate(zip(starts,lengths)):
        result,row=compress_back_volume(world_before[start:start+length],kinds[k],nearest)
        row.update(guide=k,kind=kinds[k]);rows.append(row);new[start:start+length]=result
    local=world(new,np.asarray(obj.matrix_world.inverted())).astype(np.float32);local[starts]=before[starts]
    for k,(start,length) in enumerate(zip(starts,lengths)):
        if kinds[k]=='gather':local[start+length-2:start+length]=before[start+length-2:start+length]
    retained=obj.data;retained.use_fake_user=True;obj.data=retained.copy();obj.data.name='R11 groom v3 softened posterior volume'
    obj.data.position_data.foreach_set('vector',local.ravel());obj.data.update_tag()
    mods=[m for m in obj.modifiers if m.type=='NODES' and m.show_render]
    if len(mods)!=1:raise RuntimeError('Expected sole active author styling modifier')
    modifier=mods[0];original=modifier.node_group;original.use_fake_user=True;tree=original.copy();tree.name='R11 donor groom v3 controlled styling';modifier.node_group=tree;changes=[]
    constant(tree,'Group.002','Factor',0.,changes)
    constant(tree,'Group','Random Offset',0.,changes)
    constant(tree,'Group.006','Clump Offset',0.,changes)
    constant(tree,'Group.004','Distance',.015,changes)
    constant(tree,'Group.012','Distance',.008,changes)
    constant(tree,'Group.004','Preserve Length',False,changes)
    constant(tree,'Group.012','Preserve Length',False,changes)
    constant(tree,'Group.005','Density',8000.,changes)
    constant(tree,'Group.005','Interpolation Guides',2,changes)
    radius=tree.nodes['Value'].outputs[0];old_radius=float(radius.default_value);radius.default_value=.00055
    changes.append({'node':'Value','output':'Value','before':old_radius,'after':.00055})
    parameter=factor_envelope(tree,changes)
    scale=float(np.linalg.svd(np.asarray(obj.matrix_world)[:3,:3],compute_uv=False).max())
    contact=contact_and_profile(tree,head,scale,parameter)
    tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
    if hashlib.sha256(coords(head).tobytes()).hexdigest()!=head_before:raise RuntimeError('Protected head cage changed')
    if not np.array_equal(coords(obj)[starts],before[starts]):raise RuntimeError('Original guide roots changed')
    if sha(source)!=source_sha:raise RuntimeError('Input source changed')
    obj['R11_donor_gn_finish_v3']='Measured posterior volume; roll disabled; bounded clump/noise, pinned ends, actual-head contact and tip taper; appearance pending'
    return {'source':expected_source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'stage':'Independent repair of measured raw volume and native GN distortion, actual images pending',
        'headObservation':head_report,'posteriorVolume':rows,'nodeChanges':changes,'actualHeadContactAndTipProfile':contact,
        'retainedOriginalGuideData':retained.name,'retainedOriginalNodeTree':original.name,'originalGuideRootsExact':True,
        'guideTopologyUnchanged':True,'headAndTailGeometryEdited':False,'savedBlend':False,'appearanceAccepted':False,
        'limits':['C55 and its GN styling are visibly rejected. This candidate requires actual dense-output geometry and front/quarter/side review.',
                  'Interpolated hair is corrected against the full actual head after styling; its maximum actual correction and final contact must be measured.',
                  'No new hair dynamics, collision-cache acceptance, realtime export or master promotion.']}


if __name__=='__main__':print('Explicit apply(expected_source_relative=...) in an independent saved source; caller owns save/review.')
