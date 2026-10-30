"""Keep fringe, curtains and each gather destination separate through native GN.

Native scalp-island IDs are measured from the author's actual graph. Each
island uses its own root lookup mesh so nearest-root roles cannot jump across
the part. Both interpolation and clump-guide selection retain that grouping.
"""
import hashlib,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
ROLE='R11 groom role';ISLAND='R11 original scalp island';GROUP='R11 groom interpolation group'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def geom(sockets):return next(s for s in sockets if s.type=='GEOMETRY')


def set_link(tree,output,input):
    for link in list(input.links):tree.links.remove(link)
    tree.links.new(output,input)


def named(tree,name):
    n=tree.nodes.new('GeometryNodeInputNamedAttribute');n.data_type='INT';n.inputs['Name'].default_value=name
    return n.outputs['Attribute']


def store(tree,geometry,name,value,domain='CURVE'):
    n=tree.nodes.new('GeometryNodeStoreNamedAttribute');n.data_type='INT';n.domain=domain;n.inputs['Name'].default_value=name
    tree.links.new(geometry,n.inputs['Geometry']);tree.links.new(value,n.inputs['Value']);return n.outputs['Geometry']


def group_number(tree,island,role):
    mul=tree.nodes.new('ShaderNodeMath');mul.operation='MULTIPLY';mul.inputs[1].default_value=16.;tree.links.new(island,mul.inputs[0])
    add=tree.nodes.new('ShaderNodeMath');add.operation='ADD';tree.links.new(mul.outputs[0],add.inputs[0]);tree.links.new(role,add.inputs[1]);return add.outputs[0]


def output_socket(tree):return geom(next(n for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output).inputs)


def eval_curves(obj):
    obj.update_tag();bpy.context.view_layer.update();e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if e.type!='CURVES':raise RuntimeError('Expected actual curves in role-group observation')
    return e


def read_int(data,name):
    attr=data.attributes.get(name)
    if attr is None or attr.data_type!='INT' or attr.domain!='CURVE':raise RuntimeError('Required actual curve attribute missing: '+name)
    a=np.empty(len(attr.data),np.int32);attr.data.foreach_get('value',a);return a


def install(obj,tree):
    if any(bpy.data.objects.get('R11 groom role roots / island '+str(i)) for i in (0,1)):raise RuntimeError('Preserve existing role root meshes')
    style=json.loads(bpy.context.scene['R11_actual_donor_restyle']);rows={r['guideIndex']:r for r in style['guideReports'] if r['object']==obj.name}
    starts=np.array([c.first_point_index for c in obj.data.curves],np.int32);lengths=np.array([c.points_length for c in obj.data.curves],np.int32)
    raw=np.empty(len(obj.data.position_data)*3,np.float32);obj.data.position_data.foreach_get('vector',raw);raw=raw.reshape(-1,3)
    roles=[]
    for i,(start,length) in enumerate(zip(starts,lengths)):
        kind=rows[i]['kind'];right=raw[start+length-1,0]>0
        roles.append(1 if kind=='fringe' else (2 if right else 3) if kind=='temple-curtain' else (4 if right else 5) if kind=='gather' else 6)
    roles=np.asarray(roles,np.int32)
    if obj.data.attributes.get(ROLE):raise RuntimeError('Preserve previous guide role assignment')
    attr=obj.data.attributes.new(ROLE,'INT','CURVE');attr.data.foreach_set('value',roles);obj.data.update_tag()
    parent_node=tree.nodes['Group.005'];previous_group=parent_node.node_tree;previous_group.use_fake_user=True
    child=previous_group.copy();child.name='R11 role-aware native interpolation';parent_node.node_tree=child
    core=child.nodes['Interpolate Curves']
    if core.bl_idname!='GeometryNodeInterpolateCurves':raise RuntimeError('Expected native interpolation core')
    guide_input=core.inputs['Guide Curves'];guide_group_input=core.inputs['Guide Group ID'];point_group_input=core.inputs['Point Group ID']
    if len(guide_input.links)!=1 or len(guide_group_input.links)!=1 or len(point_group_input.links)!=1:raise RuntimeError('Unrecognized original island graph')
    guide_geo=guide_input.links[0].from_socket;guide_island=guide_group_input.links[0].from_socket;point_island=point_group_input.links[0].from_socket
    measured=store(child,guide_geo,ISLAND,guide_island)
    top_output=output_socket(tree);old_top=top_output.links[0].from_socket;child_output=output_socket(child);old_child=child_output.links[0].from_socket
    try:
        set_link(child,measured,child_output);set_link(tree,geom(parent_node.outputs),top_output);tree.update_tag();child.update_tag()
        actual=eval_curves(obj);islands=read_int(actual.data,ISLAND);observed_roles=read_int(actual.data,ROLE)
        if len(islands)!=len(starts) or not np.array_equal(observed_roles,roles):raise RuntimeError('Observed guide identity/role order changed')
        coordinates=np.empty(len(actual.data.position_data)*3,np.float32);actual.data.position_data.foreach_get('vector',coordinates)
        if np.max(np.abs(coordinates.reshape(-1,3)-raw))>2e-6:raise RuntimeError('Native guide-role observation unexpectedly changes positions')
    finally:
        set_link(child,old_child,child_output);set_link(tree,old_top,top_output);tree.update_tag();child.update_tag();eval_curves(obj)
    values=sorted(int(x) for x in np.unique(islands))
    if len(values)!=2:raise RuntimeError('Expected the two actual source scalp islands, got '+str(values))
    carriers=[];sampled_roles=[]
    role_field=named(child,ROLE)
    for island in values:
        mask=islands==island;mesh=bpy.data.meshes.new('R11 groom role root points / '+str(island));mesh.from_pydata(raw[starts[mask]].tolist(),[],[]);mesh.update()
        attribute=mesh.attributes.new(ROLE,'INT','POINT');attribute.data.foreach_set('value',roles[mask])
        carrier=bpy.data.objects.new('R11 groom role roots / island '+str(island),mesh);obj.users_collection[0].objects.link(carrier)
        # Match the donor transform without parenting to the object whose
        # geometry consumes this carrier (avoids a dependency cycle).
        carrier.parent=obj.parent;carrier.matrix_world=obj.matrix_world.copy()
        carrier.hide_render=True;carrier.hide_set(True);carrier.display_type='WIRE'
        carrier['R11_groom_role_root_source']=obj.name;carrier['R11_original_scalp_island']=island
        info=child.nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE';info.inputs['Object'].default_value=carrier;info.inputs['As Instance'].default_value=False
        nearest=child.nodes.new('GeometryNodeSampleNearest');nearest.domain='POINT';child.links.new(info.outputs['Geometry'],nearest.inputs['Geometry'])
        position=child.nodes.new('GeometryNodeInputPosition');child.links.new(position.outputs['Position'],nearest.inputs['Sample Position'])
        sample=child.nodes.new('GeometryNodeSampleIndex');sample.data_type='INT';sample.domain='POINT';sample.clamp=True
        child.links.new(info.outputs['Geometry'],sample.inputs['Geometry']);child.links.new(role_field,sample.inputs['Value']);child.links.new(nearest.outputs['Index'],sample.inputs['Index'])
        sampled_roles.append(sample.outputs['Value']);carriers.append({'object':carrier.name,'island':island,'points':int(mask.sum()),'roleCounts':{str(k):int(np.sum(roles[mask]==k)) for k in np.unique(roles[mask])}})
    compare=child.nodes.new('FunctionNodeCompare');compare.data_type='INT';compare.operation='EQUAL'
    a=next(s for s in compare.inputs if s.name=='A' and s.type=='INT');b=next(s for s in compare.inputs if s.name=='B' and s.type=='INT')
    child.links.new(point_island,a);b.default_value=values[1]
    switch=child.nodes.new('GeometryNodeSwitch');switch.input_type='INT';child.links.new(compare.outputs['Result'],switch.inputs['Switch'])
    child.links.new(sampled_roles[0],switch.inputs['False']);child.links.new(sampled_roles[1],switch.inputs['True'])
    guide_number=group_number(child,named(child,ISLAND),role_field)
    grouped_guides=store(child,measured,GROUP,guide_number)
    set_link(child,grouped_guides,guide_input);set_link(child,named(child,GROUP),guide_group_input)
    set_link(child,group_number(child,point_island,switch.outputs['Output']),point_group_input)
    # Store the chosen guide's group on each resulting curve explicitly;
    # downstream clumps must not rely on implicit attribute propagation.
    sample=child.nodes.new('GeometryNodeSampleIndex');sample.data_type='INT';sample.domain='CURVE';sample.clamp=True
    child.links.new(grouped_guides,sample.inputs['Geometry']);child.links.new(named(child,GROUP),sample.inputs['Value']);child.links.new(core.outputs['Closest Index'],sample.inputs['Index'])
    original_links=list(core.outputs['Curves'].links)
    grouped_curves=store(child,core.outputs['Curves'],GROUP,sample.outputs['Value'])
    for link in original_links:
        dest=link.to_socket;child.links.remove(link);child.links.new(grouped_curves,dest)
    # The native auxiliary branch copies exactly one guide at its own root.
    # Keep that path and explicitly carry the same group into its output.
    auxiliary=child.nodes['Interpolate Curves.003']
    if auxiliary.inputs['Max Neighbors'].is_linked or auxiliary.inputs['Max Neighbors'].default_value!=1:raise RuntimeError('Auxiliary native branch is no longer exact one-guide copying')
    set_link(child,grouped_guides,auxiliary.inputs['Guide Curves'])
    sample_aux=child.nodes.new('GeometryNodeSampleIndex');sample_aux.data_type='INT';sample_aux.domain='CURVE';sample_aux.clamp=True
    child.links.new(grouped_guides,sample_aux.inputs['Geometry']);child.links.new(named(child,GROUP),sample_aux.inputs['Value']);child.links.new(auxiliary.outputs['Closest Index'],sample_aux.inputs['Index'])
    auxiliary_links=list(auxiliary.outputs['Curves'].links)
    grouped_auxiliary=store(child,auxiliary.outputs['Curves'],GROUP,sample_aux.outputs['Value'])
    for link in auxiliary_links:
        dest=link.to_socket;child.links.remove(link);child.links.new(grouped_auxiliary,dest)
    # Restore four local neighbours, now restricted to the actual role/side.
    parent_node.inputs['Interpolation Guides'].default_value=4
    # The old free-hair parting snap displaces newly distributed roots by up to
    # 9 mm. Native island grouping now preserves the part without that offset.
    tree.nodes['Group.014'].inputs['Mix'].default_value=0.
    # Both clump passes recompute their guide map within the same group.
    clumps=[]
    for name in ('Group.008','Group.006'):
        node=tree.nodes[name];original=node.node_tree;original.use_fake_user=True;copy=original.copy();copy.name='R11 role-aware '+name+' clumps';node.node_tree=copy
        internal=copy.nodes.get('Group.003')
        if internal is None or internal.type!='GROUP' or not internal.node_tree.name.startswith('Create Guide Index Map'):raise RuntimeError('Unexpected clump guide-map core')
        set_link(copy,named(copy,GROUP),internal.inputs['Group ID'])
        existing=node.inputs['Existing Guide Map']
        for link in list(existing.links):tree.links.remove(link)
        existing.default_value=False
        # Keep actual selected-guide group IDs in the evaluated output so the
        # wrapper proves both maps stay inside their advertised groups.
        map_geo=internal.outputs['Geometry'];map_links=list(map_geo.links)
        selected=copy.nodes.new('GeometryNodeSampleIndex');selected.data_type='INT';selected.domain='CURVE';selected.clamp=True
        copy.links.new(map_geo,selected.inputs['Geometry']);copy.links.new(named(copy,GROUP),selected.inputs['Value']);copy.links.new(internal.outputs['Guide Index'],selected.inputs['Index'])
        audit_name='R11 '+name+' selected guide group'
        audited=store(copy,map_geo,audit_name,selected.outputs['Value'])
        for link in map_links:
            dest=link.to_socket;copy.links.remove(link);copy.links.new(audited,dest)
        clumps.append({'node':name,'group':copy.name,'mapGroupInput':internal.name+'/Group ID','existingMap':False,'selectedGuideGroupAttribute':audit_name})
    tree.update_tag();child.update_tag();eval_curves(obj)
    return {'roleAttribute':ROLE,'groupAttribute':GROUP,'nativeScalpIslandIds':values,'rootCarriers':carriers,
            'roles':{'1':'fringe','2':'positive-X curtain','3':'negative-X curtain','4':'positive-X gather','5':'negative-X gather','6':'short crown layer'},
            'groupRule':'actual native scalp island * 16 + real guide style/destination role',
            'rootRoleLookup':'Nearest actual guide root separately inside each original scalp island',
            'interpolationNeighboursWithinGroup':4,'clumpGroups':clumps,
            'partingSnapMix':0.,'auxiliaryGuideCopies':'Native nearest-one at exact raw-guide roots, with explicit group propagation',
            'guideGroupCounts':{str(k):int(np.sum(islands*16+roles==k)) for k in np.unique(islands*16+roles)},
            'originalInterpolationGroupRetained':previous_group.name,'carrierPointCount':int(sum(c['points'] for c in carriers))}


def apply(*,expected_source_relative):
    base=HERE/'donor_groom_finish_v3_r11.py'
    if sha(base)!='ee8eae92f6163ef9c9667d681547acb99557badb2f5a89117288d27e54a9b072':raise RuntimeError('Measured v3 foundation changed')
    report=runpy.run_path(str(base))['apply'](expected_source_relative=expected_source_relative)
    obj=bpy.data.objects['R11 Donor / long hair main'];tree=next(m.node_group for m in obj.modifiers if m.type=='NODES' and m.show_render)
    report['roleGrouping']=install(obj,tree);report['groupHelperSha256']=sha(__file__)
    report['stage']='Role/island-aware native interpolation and clumping after measured v3 base repair; dense check/images pending'
    obj['R11_donor_gn_finish_v4']='Role and native-island restricted interpolation/clumps; raw roots and separate gather destinations preserved'
    return report


if __name__=='__main__':print('Explicit apply on a fresh independent candidate; no import work.')
