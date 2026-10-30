"""Add reversible native adult jaw motion after facial sculpt deformation.

The production Geometry Nodes motion rotates the current surface, so smile
and lip deltas participate in the mandibular arc. It runs before the actual
head/neck Armature modifier. A separate cranial-parented mandibular carrier
uses the same angle/translation for future lower teeth, gingiva and tongue.
No tooth asset is fitted by this helper. Public control defaults to zero.
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
P=project_path()
HEAD='R11 Female head study'
KEY='R11 jaw open'
ATTRIBUTE='R11 mandibular rotation weight'
MODIFIER='R11 anatomical jaw rotation'
CARRIER='R11 Mandibular attachment space'
MARKER='R11_native_mandibular_motion'
FIELD_HELPER='adult_jaw_motion_field_r11.py'
FIELD_SHA='294a24603936ce6717c79e04782b2de4e8ba9034b0a443bb97c370b9e4fb55dd'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a
def transform(v,m):
    m=np.asarray(m,float);return np.asarray(v)@m[:3,:3].T+m[:3,3]


def driver(target,path,keys,expression,index=None):
    fc=target.driver_add(path) if index is None else target.driver_add(path,index)
    d=fc.driver;d.type='SCRIPTED';d.expression=expression
    var=d.variables.new();var.name='j';var.type='SINGLE_PROP'
    t=var.targets[0];t.id_type='KEY';t.id=keys;t.data_path=keys.key_blocks[KEY].path_from_id('value')
    return fc


def make_nodes(head,pivot_local,axis_local,translation_local):
    import bpy
    tree=bpy.data.node_groups.new(MODIFIER,'GeometryNodeTree')
    tree.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    n=tree.nodes;l=tree.links;inp=n.new('NodeGroupInput');out=n.new('NodeGroupOutput')
    attr=n.new('GeometryNodeInputNamedAttribute');attr.data_type='FLOAT';attr.inputs['Name'].default_value=ATTRIBUTE
    value=n.new('ShaderNodeValue');value.name='R11 jaw normalized control';value.outputs[0].default_value=0.
    driver(value.outputs[0],'default_value',head.data.shape_keys,'min(max(j,0),1)')
    radians=n.new('ShaderNodeMath');radians.operation='MULTIPLY';radians.inputs[1].default_value=float(np.deg2rad(18.));l.new(value.outputs[0],radians.inputs[0])
    weighted_angle=n.new('ShaderNodeMath');weighted_angle.operation='MULTIPLY';l.new(attr.outputs['Attribute'],weighted_angle.inputs[0]);l.new(radians.outputs[0],weighted_angle.inputs[1])
    pos=n.new('GeometryNodeInputPosition');rotate=n.new('ShaderNodeVectorRotate');rotate.rotation_type='AXIS_ANGLE'
    rotate.inputs['Center'].default_value=tuple(pivot_local);rotate.inputs['Axis'].default_value=tuple(axis_local)
    l.new(pos.outputs['Position'],rotate.inputs['Vector']);l.new(weighted_angle.outputs[0],rotate.inputs['Angle'])
    sq=n.new('ShaderNodeMath');sq.operation='MULTIPLY';l.new(value.outputs[0],sq.inputs[0]);l.new(value.outputs[0],sq.inputs[1])
    transweight=n.new('ShaderNodeMath');transweight.operation='MULTIPLY';l.new(sq.outputs[0],transweight.inputs[0]);l.new(attr.outputs['Attribute'],transweight.inputs[1])
    trans=n.new('ShaderNodeVectorMath');trans.operation='SCALE';trans.inputs[0].default_value=tuple(translation_local);l.new(transweight.outputs[0],trans.inputs['Scale'])
    add=n.new('ShaderNodeVectorMath');add.operation='ADD';l.new(rotate.outputs['Vector'],add.inputs[0]);l.new(trans.outputs['Vector'],add.inputs[1])
    selected=n.new('ShaderNodeMath');selected.operation='GREATER_THAN';selected.inputs[1].default_value=1e-8;l.new(attr.outputs['Attribute'],selected.inputs[0])
    setpos=n.new('GeometryNodeSetPosition');l.new(inp.outputs['Geometry'],setpos.inputs['Geometry']);l.new(selected.outputs[0],setpos.inputs['Selection']);l.new(add.outputs['Vector'],setpos.inputs['Position'])
    enabled=n.new('ShaderNodeMath');enabled.operation='GREATER_THAN';enabled.inputs[1].default_value=1e-8;l.new(value.outputs[0],enabled.inputs[0])
    switch=n.new('GeometryNodeSwitch');switch.input_type='GEOMETRY';l.new(enabled.outputs[0],switch.inputs['Switch']);l.new(inp.outputs['Geometry'],switch.inputs['False']);l.new(setpos.outputs['Geometry'],switch.inputs['True']);l.new(switch.outputs['Output'],out.inputs['Geometry'])
    return tree


def cranial_parent(head):
    import bpy
    choices=[o for o in bpy.data.objects if o.get('R11_original_head_space_source')==HEAD]
    if len(choices)>1:raise RuntimeError('Ambiguous original cranial space')
    if choices:return choices[0]
    if head.parent_type=='BONE' and head.parent_bone=='head' and not any(m.type=='ARMATURE' for m in head.modifiers):return head
    raise RuntimeError('New skinned head requires its explicit original cranial carrier')


def evaluated(head):
    import bpy
    ev=head.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())
    try:
        attribute=mesh.attributes.get(ATTRIBUTE)
        return {'world':transform(array(mesh.vertices),ev.matrix_world),'local':array(mesh.vertices),
                'loops':array(mesh.loops,'vertex_index',1,np.int32).ravel(),
                'weights':array(attribute.data,'value',1).ravel() if attribute is not None else None,
                'matrix':np.asarray(ev.matrix_world,float)}
    finally:ev.to_mesh_clear()


def head_contract(head):
    return api('adult_head_bind_observe_r11')['data_contract'](head)


def prepare(source_relative,output_relative):
    import bpy
    from mathutils import Matrix,Vector
    source=(P/source_relative).resolve();output=(P/output_relative).resolve();report_path=output.with_suffix('.jaw-open.json')
    for p in (source,output,report_path):p.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent saved source required')
    if any(p.exists() for p in (output,report_path)):raise RuntimeError('Preserve earlier candidate')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys
    if len(head.data.vertices)!=10582 or not keys or KEY in keys.key_blocks or ATTRIBUTE in head.data.attributes or MODIFIER in head.modifiers or MARKER in bpy.context.scene or CARRIER in bpy.data.objects:
        raise RuntimeError('Fresh compatible full adult cage required')
    if any(abs(k.value)>1e-7 for k in keys.key_blocks if k.name.startswith(('R11 blink','R11 natural soft smile','R11 gentle'))):raise RuntimeError('Install jaw with neutral expressions')
    if sha(Path(__file__).with_name(FIELD_HELPER))!=FIELD_SHA:raise RuntimeError('Review changed mandibular field helper before installation')
    qa=api('face_contact_audit_r11');uv=api('production_uv_r08');field=api('adult_jaw_motion_field_r11');skin=api('natural_skin_adult_r11')
    donor,_,_=skin['_donor']()
    if not np.array_equal(array(head.data.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):raise RuntimeError('Actual adult topology changed')
    original=head_contract(head);pose_before=uv['_pose_contract']();key_values={k.name:float(k.value) for k in keys.key_blocks};active=head.active_shape_key_index
    key=None;attr=None;mod=None;tree=None;carrier=None;success=False;report={}
    try:
        with qa['rest_evaluation']():
            parent=cranial_parent(head);mix,mixmeta=uv['current_mix'](head);T=np.asarray(head.matrix_world,float);parent_T=np.asarray(parent.matrix_world,float)
            if np.max(abs(T-parent_T))>2e-6:raise RuntimeError('Cranial carrier and neutral head local spaces differ; inspect binding before jaw installation')
            world=transform(mix,T);planned,report=field['plan'](world)
            if any(row['nonpositiveNormalCount'] for row in report['rawCageOpeningChecks']):raise RuntimeError('Current raw jaw field folds; do not install')
            before=evaluated(head);baseline=array(keys.reference_key.data)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.;key.data.foreach_set('co',baseline.ravel())
            attr=head.data.attributes.new(ATTRIBUTE,'FLOAT','POINT');attr.data.foreach_set('value',planned['weights'])
            inv=np.linalg.inv(T);pivot=transform(planned['pivot_world'][None,:],inv)[0];axis=inv[:3,:3]@np.array([1.,0.,0.]);axis/=np.linalg.norm(axis)
            translation=inv[:3,:3]@np.asarray(report['maximumCondyleTranslationWorld'])
            tree=make_nodes(head,pivot,axis,translation);mod=head.modifiers.new(MODIFIER,'NODES');mod.node_group=tree
            # Keep true neck/head skinning after mandibular deformation.
            armature_indices=[i for i,m in enumerate(head.modifiers) if m.type=='ARMATURE']
            if armature_indices:
                index=armature_indices[0];head.modifiers.move(len(head.modifiers)-1,index)
            else:index=len(head.modifiers)-1
            carrier=bpy.data.objects.new(CARRIER,None);head.users_collection[0].objects.link(carrier);carrier.empty_display_type='PLAIN_AXES';carrier.empty_display_size=.012
            carrier.parent=parent;carrier.matrix_parent_inverse=Matrix.Identity(4);carrier.location=Vector(pivot);carrier.rotation_mode='QUATERNION'
            carrier['R11_mandibular_attachment_source']=HEAD;carrier['R11_jaw_control_key']=KEY;carrier['R11_jaw_pivot_estimated']=True
            limit=float(np.deg2rad(report['maximumAngleDegrees']))
            for i in range(4):driver(carrier,'rotation_quaternion',keys,('cos('+repr(limit)+'*min(max(j,0),1)/2)' if i==0 else repr(float(axis[i-1]))+'*sin('+repr(limit)+'*min(max(j,0),1)/2)'),i)
            for i in range(3):driver(carrier,'location',keys,repr(float(pivot[i]))+'+'+repr(float(translation[i]))+'*min(max(j,0),1)**2',i)
            head.active_shape_key_index=active;bpy.context.view_layer.update();neutral=evaluated(head)
            if not np.array_equal(neutral['world'],before['world']) or not np.array_equal(neutral['loops'],before['loops']):raise RuntimeError('Inactive jaw changed actual neutral geometry')
            if neutral['weights'] is None or len(neutral['weights'])!=len(neutral['world']):raise RuntimeError('Mandibular field did not survive actual Multires and Mask')
            carrier_rest=np.asarray(carrier.matrix_world,float)
            check_points=world[planned['rigid_ids'][::max(1,len(planned['rigid_ids'])//20)]]
            carrier_local=transform(check_points,np.linalg.inv(carrier_rest))
            checks=[]
            for strength in (.25,.5,.75,1.):
                key.value=strength;bpy.context.view_layer.update();now=evaluated(head)
                if not np.array_equal(now['loops'],before['loops']):raise RuntimeError('Jaw changed actual head topology')
                expected=before['world']+field['rotation_displacement'](before['world'],neutral['weights'],planned['pivot_world'],limit*strength,np.asarray(report['maximumCondyleTranslationWorld'])*strength*strength)
                # The selection intentionally fixes extremely small interpolated weights.
                expected[neutral['weights']<=1e-8]=before['world'][neutral['weights']<=1e-8]
                error=np.linalg.norm(now['world']-expected,axis=1);maximum=float(error.max())
                if maximum>2e-6:raise RuntimeError('Native jaw/evaluated rotation mismatch: '+str(maximum))
                pinned=neutral['weights']<=1e-8;pin_error=float(np.linalg.norm(now['world'][pinned]-before['world'][pinned],axis=1).max())
                if pin_error>2e-6:raise RuntimeError('Jaw moved pinned upper face or low neck')
                carried=transform(carrier_local,carrier.matrix_world)
                carried_expected=check_points+field['rotation_displacement'](check_points,np.ones(len(check_points)),planned['pivot_world'],limit*strength,np.asarray(report['maximumCondyleTranslationWorld'])*strength*strength)
                carrier_error=float(np.linalg.norm(carried-carried_expected,axis=1).max())
                if carrier_error>2e-6:raise RuntimeError('Lower dental carrier differs from skin mandibular arc: '+str(carrier_error))
                checks.append({'strength':strength,'actualEvaluatedVertices':len(now['world']),'nativeRotationMaximumErrorMeters':maximum,
                    'pinnedSurfaceMaximumDisplacementMeters':pin_error,'maximumDisplacementMeters':float(np.linalg.norm(now['world']-before['world'],axis=1).max()),
                    'lowerDentalCarrierMaximumErrorMeters':carrier_error,
                    'mandibularCarrierMatrixWorld':[list(r) for r in carrier.matrix_world],
                    'fibreBinding':api('face_fibre_binding_r11')['validate'](neutral=False)})
            key.value=0.;bpy.context.view_layer.update()
            if not np.array_equal(evaluated(head)['world'],before['world']):raise RuntimeError('Jaw neutral restoration changed actual geometry')
            report.update(source=source_relative,sourceSha256=sha(source),controlKey=KEY,controlDefault=0.,weightAttribute=ATTRIBUTE,
                modifierName=MODIFIER,modifierIndex=index,modifierOrder=[m.name for m in head.modifiers],
                attachmentCarrier=CARRIER,cranialParent=parent.name,estimatedPivotLocal=pivot.tolist(),axisLocal=axis.tolist(),translationLocal=translation.tolist(),
                currentMixMetadata=mixmeta,actualNativeChecks=checks,neutralEvaluatedCoordinatesExact=True,
                fieldHelperSha256=sha(Path(__file__).with_name(FIELD_HELPER)),helperSha256=sha(__file__),
                originalHeadContract=original,originalActiveShapeKey=active,lowerTeethOrTongueFitted=False,
                scope='Native production jaw rotation and lower-oral attachment space only. Teeth/gingiva/tongue fitting, visual oral clearance, speech and runtime baking remain pending.')
        # New modifier/attribute/key are additive; all original raw geometry and key coordinates survive.
        after=head_contract(head);stripped=dict(after)
        stripped['keys']=[k for k in stripped['keys'] if k['name']!=KEY]
        stripped['modifiers']=[m for m in stripped['modifiers'] if m['name']!=MODIFIER]
        if stripped!=original:raise RuntimeError('Jaw install changed original head geometry, keys, rig or materials')
        if uv['_pose_contract']()!=pose_before:raise RuntimeError('Jaw install changed original rig pose')
        bpy.context.scene[MARKER]=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Jaw candidate save failed')
        report.update(output=output_relative,outputSha256=sha(output),savedBlend=True)
        with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        success=True
        print('R11_NATIVE_JAW_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256'],'keyDefault':0.}),flush=True)
        return report
    finally:
        if not success:
            if carrier is not None and carrier.name in bpy.data.objects:bpy.data.objects.remove(carrier,do_unlink=True)
            if mod is not None and head.modifiers.get(MODIFIER):head.modifiers.remove(mod)
            if tree is not None and tree.name in bpy.data.node_groups:bpy.data.node_groups.remove(tree,do_unlink=True)
            if attr is not None and head.data.attributes.get(ATTRIBUTE):head.data.attributes.remove(attr)
            if key is not None and keys.key_blocks.get(KEY):head.shape_key_remove(key)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            for name,value in key_values.items():keys.key_blocks[name].value=value
            head.active_shape_key_index=active;bpy.context.view_layer.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(a.source,a.output)
