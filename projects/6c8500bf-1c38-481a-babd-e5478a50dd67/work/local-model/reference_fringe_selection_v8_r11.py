"""Explicit native selection trim for the measured overlong crown fringe layer."""
import hashlib,runpy
from pathlib import Path
import bpy,numpy as np
HERE=Path(__file__).resolve().parent
GROUP='R11 groom interpolation group'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def math_node(tree,operation,a=None,b=None):
    n=tree.nodes.new('ShaderNodeMath');n.operation=operation
    for i,value in enumerate((a,b)):
        if value is None:continue
        if isinstance(value,(int,float)):n.inputs[i].default_value=value
        else:tree.links.new(value,n.inputs[i])
    return n

def apply(*,expected_source_relative):
    base=HERE/'reference_front_groom_v6_r11.py'
    if sha(base)!='297b6b6533321f6893b4f211d992c290927b11a91444a593118eaa4844acaf2c':raise RuntimeError('Measured v6 reconstruction changed')
    report=runpy.run_path(str(base))['apply'](expected_source_relative=expected_source_relative)
    obj=bpy.data.objects['R11 Donor / long hair main'];tree=next(m.node_group for m in obj.modifiers if m.type=='NODES' and m.show_render)
    template=tree.nodes['Group'].node_tree
    root=tree.nodes.new('GeometryNodeGroup');root.node_tree=template.nodes['Group.007'].node_tree;root.name='R11 reference layer root position'
    length=tree.nodes.new('GeometryNodeGroup');length.node_tree=template.nodes['Group.006'].node_tree;length.name='R11 reference actual curve length'
    split=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(root.outputs['Root Position'],split.inputs['Vector'])
    matrix=np.asarray(obj.matrix_world);scale=float(matrix[1,1])
    if scale<=0 or np.max(np.abs(matrix[:3,:3]-np.diag(np.diag(matrix[:3,:3]))))>1e-6:raise RuntimeError('Expected measured axis-aligned donor rest transform')
    ramp=tree.nodes.new('ShaderNodeMapRange');ramp.data_type='FLOAT';ramp.interpolation_type='SMOOTHSTEP';ramp.clamp=True
    ramp.inputs['From Min'].default_value=(-.012-float(matrix[1,3]))/scale;ramp.inputs['From Max'].default_value=(.008-float(matrix[1,3]))/scale
    ramp.inputs['To Min'].default_value=0.;ramp.inputs['To Max'].default_value=1.;tree.links.new(split.outputs['Y'],ramp.inputs['Value'])
    removed=math_node(tree,'MULTIPLY',ramp.outputs['Result'],.009/scale)
    attr=tree.nodes.new('GeometryNodeInputNamedAttribute');attr.data_type='INT';attr.inputs['Name'].default_value=GROUP
    role=math_node(tree,'MODULO',attr.outputs['Attribute'],256.);selected=math_node(tree,'COMPARE',role.outputs[0],12.);selected.inputs[2].default_value=.1
    # Use the Blender kernel node with an actual curve Selection. The author
    # asset's Mask still redistributed all unselected curves in the v7 run.
    root_selected=math_node(tree,'GREATER_THAN',split.outputs['Y'],ramp.inputs['From Min'].default_value)
    selection=tree.nodes.new('FunctionNodeBooleanMath');selection.operation='AND';tree.links.new(selected.outputs[0],selection.inputs[0]);tree.links.new(root_selected.outputs[0],selection.inputs[1])
    end=math_node(tree,'SUBTRACT',length.outputs['Length'],removed.outputs[0])
    trim=tree.nodes.new('GeometryNodeTrimCurve');trim.mode='LENGTH';trim.name='R11 reference fringe native selection trim'
    start_socket=next(s for s in trim.inputs if s.identifier=='Start_001');end_socket=next(s for s in trim.inputs if s.identifier=='End_001');start_socket.default_value=0.
    tree.links.new(selection.outputs[0],trim.inputs['Selection']);tree.links.new(end.outputs[0],end_socket)
    profile=tree.nodes['R11 strand tip taper'];geometry=next(s for s in profile.inputs if s.type=='GEOMETRY')
    if len(geometry.links)!=1 or geometry.links[0].from_node.name!='R11 full head clearance':raise RuntimeError('Measured post-skin / pre-radius stage changed')
    source=geometry.links[0].from_socket
    for link in list(geometry.links):tree.links.remove(link)
    tree.links.new(source,trim.inputs['Curve']);tree.links.new(trim.outputs['Curve'],geometry)
    tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
    report['fringeLengthCorrection']={'role':12,'bothNativeIslands':True,'geometryStage':'After actual head clearance, before final physical radius taper',
        'restRootWorldYTransitionMeters':[-.012,.008],'maximumArcLengthRemovedMeters':.009,
        'rule':'Smoothly shorten only the crown-origin layer of the measured longer left lock along its existing curve; front-origin fine ends and all other locks retain their lengths.',
        'nativeImplementationReference':'https://github.com/blender/blender/blob/main/source/blender/geometry/intern/trim_curves.cc','basis':'Actual failed-v6 26046 dense curves; 11 final segments entered the R eye envelope. Offline same-curve truncation changed 236/26046 curves and gave minimum R gap 2.744 mm; native output still requires validation.',
        'nativeKernelTrimNode':'GeometryNodeTrimCurve','unselectedCurvesDirectlyCopied':True,'authorTrimGroupNotUsed':True,'notAPlaneCut':True,'guideRolesAndRootDistributionUnchanged':True,'sourceCanBeAnyNeutralV5DerivedCandidate':True}
    report['lengthHelperSha256']=sha(__file__);report['stage']='Reference v8 measured crown-layer length correction; actual dense eye and visual checks pending'
    obj['R11_reference_fringe_v8']='Native explicit-Selection trim of role-12 crown layer; exact protection of other curves required'
    return report

if __name__=='__main__':print('Explicit independent candidate apply only.')
