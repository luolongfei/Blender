"""Local measured v10: remove two upward gather loops and taper two front locks."""
import hashlib,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent;MAIN='R11 Donor / long hair main';GROUP='R11 groom interpolation group'
GATHER_SHA='5340c545a0d1103db82ee7e8c559172a0d80ebfc18349a8a2d4da909579292ed'
FAN_SHA='710bcda5d46ace79cb54302de571374381d5de4fd8e48f25cb0e11ebefb530df'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def math(tree,operation,a,b):
    n=tree.nodes.new('ShaderNodeMath');n.operation=operation
    for i,value in enumerate((a,b)):
        if isinstance(value,(int,float,np.integer,np.floating)):n.inputs[i].default_value=float(value)
        else:tree.links.new(value,n.inputs[i])
    return n.outputs[0]

def install_fan(obj,rules,start_factor):
    modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);old=modifier.node_group;old.use_fake_user=True
    tree=old.copy();tree.name='R11 original-reference front detail v10';modifier.node_group=tree
    skin=tree.nodes['R11 full head clearance'];geometry=next(s for s in skin.inputs if s.type=='GEOMETRY')
    if len(geometry.links)!=1 or geometry.links[0].from_node.name!='Join Geometry.001':raise RuntimeError('Measured before-contact geometry changed')
    source=geometry.links[0].from_socket
    for link in list(geometry.links):tree.links.remove(link)
    template=tree.nodes['Group'].node_tree;tip_group=template.nodes['Group.006'].node_tree.nodes['Group.002'].node_tree
    if 'Tip Position' not in [s.name for s in tip_group.interface.items_tree if getattr(s,'in_out',None)=='OUTPUT']:raise RuntimeError('Measured author curve-tip field changed')
    matrix=np.asarray(obj.matrix_world);scales=np.diag(matrix[:3,:3]);translation=matrix[:3,3]
    if np.any(scales<=0.) or np.max(np.abs(matrix[:3,:3]-np.diag(scales)))>1e-6:raise RuntimeError('Expected actual axis-aligned donor rest transform')
    attr=tree.nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='INT';attr.inputs['Name'].default_value=GROUP
    role_field=math(tree,'MODULO',attr.outputs['Attribute'],256.)
    for role,rule in sorted(rules.items()):
        tip=tree.nodes.new('GeometryNodeGroup');tip.node_tree=tip_group;tip.name='R11 v10 own-curve tip '+str(role)
        split=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(tip.outputs['Tip Position'],split.inputs['Vector'])
        offsets=[]
        for letter,axis in (('x',0),('z',2)):
            focus=(rule[letter+'Focus']-translation[axis])/scales[axis];center=(rule[letter+'OldCenter']-translation[axis])/scales[axis]
            original=split.outputs[letter.upper()]
            desired=math(tree,'ADD',focus,math(tree,'MULTIPLY',math(tree,'SUBTRACT',original,float(center)),rule[letter+'Scale']))
            offsets.append(math(tree,'SUBTRACT',desired,original))
        vector=tree.nodes.new('ShaderNodeCombineXYZ');tree.links.new(offsets[0],vector.inputs['X']);tree.links.new(offsets[1],vector.inputs['Z'])
        parameter=tree.nodes.new('GeometryNodeSplineParameter')
        ramp=tree.nodes.new('ShaderNodeMapRange');ramp.data_type='FLOAT';ramp.interpolation_type='SMOOTHSTEP';ramp.clamp=True
        ramp.inputs['From Min'].default_value=start_factor;ramp.inputs['From Max'].default_value=1.;ramp.inputs['To Min'].default_value=0.;ramp.inputs['To Max'].default_value=1.
        tree.links.new(parameter.outputs['Factor'],ramp.inputs['Value'])
        offset=tree.nodes.new('ShaderNodeVectorMath');offset.operation='SCALE';tree.links.new(vector.outputs['Vector'],offset.inputs[0]);tree.links.new(ramp.outputs['Result'],offset.inputs['Scale'])
        select_node=tree.nodes.new('ShaderNodeMath');select_node.operation='COMPARE';tree.links.new(role_field,select_node.inputs[0]);select_node.inputs[1].default_value=role;select_node.inputs[2].default_value=.1
        setpos=tree.nodes.new('GeometryNodeSetPosition');setpos.name='R11 v10 own-lock taper '+str(role);tree.links.new(source,setpos.inputs['Geometry']);tree.links.new(select_node.outputs[0],setpos.inputs['Selection']);tree.links.new(offset.outputs['Vector'],setpos.inputs['Offset']);source=setpos.outputs['Geometry']
    setpos.name='R11 reference local front detail';tree.links.new(source,geometry)
    tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
    return {'targetRoles':sorted(rules),'lastArcFraction':1.-start_factor,'rules':rules,'sourceTreeRetained':old.name,
        'newTree':tree.name,'ownCurveTipOnly':True,'crossLockClumping':False,'stage':'Before actual head clearance and existing role-12 native trim'}

def apply(*,expected_source_relative):
    source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Explicit independent source required')
    obj=bpy.data.objects[MAIN]
    if not obj.get('R11_reference_local_v9') or obj.get('R11_reference_detail_v10'):raise RuntimeError('Actual saved v9-derived source required')
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Caller must enter REST')
    if sha(HERE/'reference_left_gather_v10_r11.py')!=GATHER_SHA or sha(HERE/'reference_central_fan_v10_r11.py')!=FAN_SHA:raise RuntimeError('Frozen v10 detail plan changed')
    gather=runpy.run_path(str(HERE/'reference_left_gather_v10_r11.py'));fan=runpy.run_path(str(HERE/'reference_central_fan_v10_r11.py'));fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'))
    _,skin,head=fit['_head_trees'](bpy.data.objects['R11 Female head study'])
    original=obj.data;old_local=fit['_coords'](obj);world=fit['_world'](old_local,obj.matrix_world);local=old_local.copy();inverse=np.asarray(obj.matrix_world.inverted())
    starts=np.array([c.first_point_index for c in original.curves],np.int32);lengths=np.array([c.points_length for c in original.curves],np.int32)
    roles=np.empty(len(starts),np.int32);original.attributes['R11 groom role'].data.foreach_get('value',roles)
    if (len(starts),len(local))!=(346,6770):raise RuntimeError('Measured raw topology changed')
    selected=np.zeros(len(local),bool);rows=[];max_push=0.
    for i in gather['GUIDES']:
        if roles[i]!=41:raise RuntimeError('Measured left gather role changed')
        start=int(starts[i]);length=int(lengths[i]);old_path=world[start:start+length];new,row=gather['flatten_takeoff'](old_path,i)
        for j in row['changedPoints']:
            hit,n,_,distance=skin.find_nearest(Vector(new[j]));signed=float((Vector(new[j])-hit).dot(n))
            if signed<.00025:
                push=.00028-signed
                if push>.004:raise RuntimeError('Left gather requires excessive local skin correction')
                new[j]+=np.asarray(n)*push;max_push=max(max_push,push)
            local[start+j]=fit['_world'](new[j:j+1],inverse)[0].astype(np.float32);selected[start+j]=True
        rows.append(row)
    if not np.array_equal(local[~selected],old_local[~selected]):raise RuntimeError('Raw edit changed non-target points')
    original.use_fake_user=True;obj.data=original.copy();obj.data.name='R11 measured left takeoff detail v10';obj.data.position_data.foreach_set('vector',local.ravel());obj.data.update_tag()
    field_report=install_fan(obj,fan['RULES'],fan['START_FACTOR'])
    obj['R11_reference_detail_v10']='Original portrait: two measured left upward loops removed, own-lock central end taper, right opening/tails untouched'
    return {'source':expected_source_relative,'sourceSha256':sha(source),'helperSha256':sha(__file__),'gatherHelperSha256':GATHER_SHA,'fanHelperSha256':FAN_SHA,
        'reference':'references/user-original-portrait-2026-10-28.png','head':head,'rawGatherGuides':rows,'changedRawPoints':int(selected.sum()),'maximumRawSkinCorrectionMeters':max_push,
        'allOtherRawPointsExact':True,'allRawRootsAndGatherLowerApproachesExact':True,'guideRolesCarriersDensityTailsFaceEdited':False,
        'centralOwnLockField':field_report,'retainedMainGuideData':original.name,'appearanceAccepted':False,'stage':'Local v10 front detail candidate; actual final geometry and image review required'}

if __name__=='__main__':print('Explicit saved v9-derived candidate apply only.')
