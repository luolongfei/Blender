"""Read-only numerical diagnosis of R11 blink / native fibre transport.

Installs the frozen blink path temporarily in a fresh background process,
captures the exact float32 head positions entering a relative Object Info
node, and compares float64 and stepwise float32 triangle-frame references.
No Blend save, raw/mapping edit, rebind, rendering or public helper mutation.
The old parity threshold is bypassed only in a private imported diagnostic
namespace so every fibre can be measured; topology/raw/mapping/radius guards
remain active. This is precision evidence, never contact/appearance approval.
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

FROZEN_PATH_SHA='71d90373ff146fb2c07a9ea92856f31ab1d8300bcf958216b0aab3652568d272'


def _api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def rebuild_float32(points,mapping):
    """Match the GN node arithmetic order with a rounding at each vector op."""
    tri=np.asarray(points,np.float32)[mapping['vertices']]
    e1=(tri[:,1]-tri[:,0]).astype(np.float32)
    e2=(tri[:,2]-tri[:,0]).astype(np.float32)
    tangent=e1/np.sqrt(np.sum(e1*e1,axis=1,dtype=np.float32))[:,None]
    cross=np.cross(e1,e2).astype(np.float32)
    normal=cross/np.sqrt(np.sum(cross*cross,axis=1,dtype=np.float32))[:,None]
    bitangent=np.cross(normal,tangent).astype(np.float32)
    weights=np.asarray(mapping['weights'],np.float32)
    result=(tri[:,0]*weights[:,0,None]).astype(np.float32)
    result=(result+tri[:,1]*weights[:,1,None]).astype(np.float32)
    result=(result+tri[:,2]*weights[:,2,None]).astype(np.float32)
    result=np.repeat(result,mapping['counts'],axis=0)
    offsets=np.asarray(mapping['offsets'],np.float32)
    for index,frame in enumerate((tangent,bitangent,normal)):
        term=(np.repeat(frame,mapping['counts'],axis=0)*offsets[:,index,None]).astype(np.float32)
        result=(result+term).astype(np.float32)
    return result


def _input_capture(head):
    import bpy
    mesh=bpy.data.meshes.new('R11 temporary GN input capture')
    obj=bpy.data.objects.new('R11 temporary GN input capture',mesh)
    bpy.context.scene.collection.objects.link(obj)
    group=bpy.data.node_groups.new('R11 temporary relative input capture','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    info=group.nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE'
    info.inputs['Object'].default_value=head;info.inputs['As Instance'].default_value=False
    output=group.nodes.new('NodeGroupOutput');group.links.new(info.outputs['Geometry'],output.inputs['Geometry'])
    modifier=obj.modifiers.new('Temporary capture','NODES');modifier.node_group=group
    return obj,mesh,group


def inspect_current(binding,probe):
    import bpy
    head=bpy.data.objects[binding['HEAD']];current=binding['_head_mesh'](head)
    # Isolated private module namespace. All non-parity exact checks execute;
    # infinity here means collect the measured errors, NOT claim acceptance.
    validate=binding['validate'];previous=validate.__globals__['TOLERANCE']
    validate.__globals__['TOLERANCE']=float('inf')
    try:
        structural=validate(neutral=False)
    finally:
        validate.__globals__['TOLERANCE']=previous
    record=json.loads(bpy.context.scene[binding['MARKER']]);rows={};arrays={};capture_cache={}
    for name,item in record['objects'].items():
        obj=bpy.data.objects[name];mapping=binding['_mapping'](obj.data)
        world_matrix=np.asarray(obj.matrix_world,float)
        key=world_matrix.astype(np.float32).tobytes()
        if key not in capture_cache:
            probe.matrix_world=obj.matrix_world.copy();bpy.context.view_layer.update()
            ev=probe.evaluated_get(bpy.context.evaluated_depsgraph_get())
            if ev.type!='MESH' or len(ev.data.vertices)!=current['counts']['points']:
                raise RuntimeError('GN relative input capture changed head point count')
            capture_cache[key]=binding['_array'](ev.data.vertices,'co')
        input32=capture_cache[key]
        double_input=binding['_transform'](current['world'],np.linalg.inv(world_matrix))
        reference64=binding['rebuild'](double_input,mapping['vertices'],mapping['weights'],mapping['offsets'],mapping['counts'])
        exact_input64=binding['rebuild'](input32,mapping['vertices'],mapping['weights'],mapping['offsets'],mapping['counts'])
        reference32=rebuild_float32(input32,mapping)
        evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        actual=binding['_array'](evaluated.data.position_data,'vector')
        world=lambda p:binding['_transform'](p,world_matrix)
        actual_world=world(actual);ref64_world=world(reference64);input64_world=world(exact_input64);ref32_world=world(reference32)
        error=np.linalg.norm(actual_world-ref64_world,axis=1)
        arithmetic=np.linalg.norm(ref32_world-input64_world,axis=1)
        transform=np.linalg.norm(input64_world-ref64_world,axis=1)
        residual32=np.linalg.norm(actual_world-ref32_world,axis=1)
        ulp=np.spacing(np.abs(actual).astype(np.float32)).astype(float)
        ulp_world=np.linalg.norm(ulp@np.abs(world_matrix[:3,:3]).T,axis=1)
        bound=max(2e-6,float(transform.max()+arithmetic.max()+8.*ulp_world.max()))
        strand_for_point=np.repeat(np.arange(len(mapping['counts'])),mapping['counts'])
        worst=int(np.argmax(error));strand=int(strand_for_point[worst])
        tri=input32[mapping['vertices']];edge=tri[:,1]-tri[:,0];other=tri[:,2]-tri[:,0]
        area2=np.linalg.norm(np.cross(edge,other),axis=1)
        lengths=np.linalg.norm(edge,axis=1)
        rows[name]={'actualVersusOriginalFloat64MaxMeters':float(error.max()),
            'actualVersusExactGNInputFloat64MaxMeters':float(np.linalg.norm(actual_world-input64_world,axis=1).max()),
            'actualVersusStepwiseFloat32MaxMeters':float(residual32.max()),
            'GNInputRoundingAmplificationMaxMeters':float(transform.max()),
            'stepwiseArithmeticRoundingMaxMeters':float(arithmetic.max()),
            'maximumOneULPOutputWorldMeters':float(ulp_world.max()),
            'derivedParityBoundMeters':bound,'hardPrecisionCapMeters':1e-5,
            'withinDerivedBoundAnd10MicrometerCap':bool(error.max()<=bound and bound<=1e-5),
            'original2MicrometerThresholdMeters':previous,
            'worstPoint':worst,'worstStrand':strand,'worstTriangleVertexIds':mapping['vertices'][strand].tolist(),
            'worstTriangleEdgeLengthMeters':float(lengths[strand]),'worstTriangleTwiceAreaSquareMeters':float(area2[strand]),
            'minimumTriangleEdgeLengthMeters':float(lengths.min()),'minimumTriangleTwiceAreaSquareMeters':float(area2.min()),
            'worstErrorVectorWorld':(actual_world[worst]-ref64_world[worst]).tolist(),
            'unchangedRawGeometryMappingAndRadii':structural['objects'][name]}
        prefix='object_'+str(len(rows)-1)+'_'
        arrays.update({prefix+'actualLocal':actual,prefix+'float64Local':reference64,
            prefix+'exactGNInputFloat64Local':exact_input64,prefix+'float32Local':reference32,
            prefix+'boundTrianglesGNInputLocal':tri,prefix+'boundTrianglesFloat64InputLocal':double_input[mapping['vertices']],
            prefix+'triangleVertexIds':mapping['vertices'],prefix+'counts':mapping['counts'],prefix+'worldMatrix':world_matrix})
    return {'objects':rows,'precisionOnly':True,'contactVerified':False,'appearanceAccepted':False},arrays


def main():
    import bpy
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    path=_api('adult_blink_path_r11');helper=Path(__file__).with_name('adult_blink_path_r11.py')
    if _sha(helper)!=FROZEN_PATH_SHA:
        raise RuntimeError('Frozen blink source changed; inspect before diagnosing')
    source=path['_path'](args.source,'.blend');output=path['_path'](args.output,'.json');payload=output.with_suffix('.npz')
    observation_path=path['_path'](args.observation,'.json')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or output.exists() or payload.exists():
        raise RuntimeError('Fresh source and independent new diagnostic paths required')
    source_sha=_sha(source);head=bpy.data.objects[path['HEAD']];keys=head.data.shape_keys
    if any(name in keys.key_blocks for name in path['CONTROLS'].values()):
        raise RuntimeError('Use pre-blink source; do not replace installed editable controls')
    old=path['_api']('adult_jaw_contour_r11')['_keys'](head.data)
    active=head.active_shape_key_index;pose=path['_api']('production_uv_r08')['_pose_contract']()
    observation,observed=path['_load_observation'](observation_path)
    created=[];probe=None;mesh=None;group=None;report={'states':[]};arrays={}
    try:
        with path['_api']('face_contact_audit_r11')['rest_evaluation']():
            path['_assert_current_observation'](head,observation,observed)
            paths,plan=path['build_plan'](observation,observed)
            if not all(r['rawGeometrySafe'] for r in plan['sides'].values()):
                raise RuntimeError('Frozen path raw geometry differs')
            basis=path['_array'](keys.reference_key.data)
            for side,deltas in paths.items():
                full=deltas[-1];key=head.shape_key_add(name=path['CONTROLS'][side],from_mix=False);created.append(key.name)
                key.relative_key=keys.reference_key;key.value=0.;key.slider_min=0.;key.slider_max=1.
                key.data.foreach_set('co',(basis+full).astype(np.float32).ravel())
                for step in range(1,path['INTERVALS']):
                    time=step/path['INTERVALS'];key=head.shape_key_add(name='R11 blink path '+side+' '+str(step).zfill(2),from_mix=False)
                    created.append(key.name);key.relative_key=keys.reference_key;key.value=0.;key.slider_min=0.;key.slider_max=1.
                    key.data.foreach_set('co',(basis+deltas[step]-time*full).astype(np.float32).ravel())
                    path['_add_corrective_driver'](key,path['CONTROLS'][side],time)
            binding=_api('face_fibre_binding_r11');probe,mesh,group=_input_capture(head)
            for step,value in enumerate((0.,.25,.5,.75,1.)):
                for name in path['CONTROLS'].values():
                    keys.key_blocks[name].value=value
                bpy.context.view_layer.update();state,data=inspect_current(binding,probe)
                state['blinkValues']={'L':value,'R':value};report['states'].append(state)
                arrays.update({'state_'+str(step)+'_'+k:v for k,v in data.items()})
                print('R11_BLINK_PRECISION '+json.dumps({'value':value,'objects':{name:{k:v for k,v in row.items() if k in ('actualVersusOriginalFloat64MaxMeters','actualVersusStepwiseFloat32MaxMeters','derivedParityBoundMeters','withinDerivedBoundAnd10MicrometerCap')} for name,row in state['objects'].items()}}),flush=True)
    finally:
        if probe is not None:
            bpy.data.objects.remove(probe,do_unlink=True)
        if mesh is not None and mesh.users==0:
            bpy.data.meshes.remove(mesh)
        if group is not None and group.users==0:
            bpy.data.node_groups.remove(group)
        for name in reversed(created):
            key=head.data.shape_keys.key_blocks.get(name)
            if key is not None:
                key.driver_remove('value');head.shape_key_remove(key)
        head.active_shape_key_index=active;bpy.context.view_layer.update()
    now=path['_api']('adult_jaw_contour_r11')['_keys'](head.data)
    if set(now)!=set(old) or any(not np.array_equal(old[n]['coords'],now[n]['coords']) or any(old[n][k]!=now[n][k] for k in old[n] if k!='coords') for n in old):
        raise RuntimeError('Temporary precision diagnostic did not restore original keys')
    if path['_api']('production_uv_r08')['_pose_contract']()!=pose or _sha(source)!=source_sha:
        raise RuntimeError('Source or pose changed during precision diagnostic')
    np.savez_compressed(payload,**arrays)
    report.update(source=args.source,sourceSha256=source_sha,helperSha256=_sha(__file__),frozenBlinkHelperSha256=FROZEN_PATH_SHA,
        observation=args.observation,observationSha256=_sha(observation_path),arraysSha256=_sha(payload),sourceSaved=False,
        temporaryKeysAndProbeRemoved=True,publicBindingHelperChanged=False,appearanceAccepted=False,
        boundFormula='max(2 um, maximum propagated actual GN input coordinate rounding + stepwise float32 arithmetic difference + eight output float32 ULPs); additionally require bound <= 10 um.',
        scope='Numerical transport parity only. Exact raw data, mapping, topology, fibre counts and radii remain guarded. No actual eyelid or lash-contact acceptance.')
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_BLINK_PRECISION_SAVED '+str(output),flush=True)


if __name__=='__main__':
    main()
