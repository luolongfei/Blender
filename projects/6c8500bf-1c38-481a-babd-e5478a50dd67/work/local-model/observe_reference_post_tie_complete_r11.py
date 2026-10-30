"""One read-only post-tie capture for crown-flow and fitted-band work.

Exports actual reconnection/final curves, 128 native samples in the positive
crown region before and after head contact, complete native gather and tail
paths, current head/ears/rings, and actual growth/carrier scalar attributes. No rendering,
model installation or blend save. Use one fresh background source after tie v2.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import bpy
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
MAIN='R11 Donor / long hair main';CARRIER='R11 reference tie gather target positions'
STAGE='R11 reference gather tie transport';GROUP='R11 groom interpolation group'
GROWTH='R11 Donor / long hair growth mesh'
ROI_MIN=np.array([0.,-.025,1.460]);ROI_MAX=np.array([.085,.055,1.55]);COUNT=128
PINS={
    'observe_reference_tie_alignment_r11_v2.py':'c488fccec705d4a805f142dab3c2fa8fe902bf42d3a59fd6307d04f248cf7c5c',
    'prepare_reference_tie_reconnection_v2_r11.py':'ef079aaa840043037181d6a55561394591bc14934271d9e97c40564162b9925d',
    'adult_full_groom_review_r11.py':'699d0a96cd332682e9bc9ddf8e6d3d731053e19889196f7f92a65c569fa7f6ae',
    'observe_donor_gn_stages_r11.py':'3c5da0c03dbbc9beb7c44884d271115c793a75fb5ca366ae2d7d8d1457f7e8cb',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def scoped(relative):
    result=(P/str(relative).replace('\\','/')).resolve();result.relative_to((P/'work/reconstruction-r02').resolve());return result


def math(tree,operation,a,b):
    n=tree.nodes.new('ShaderNodeMath');n.operation=operation
    for i,value in enumerate((a,b)):
        if isinstance(value,(int,float,np.integer,np.floating)):n.inputs[i].default_value=float(value)
        else:tree.links.new(value,n.inputs[i])
    return n.outputs[0]


def point_mesh(obj,prefix,arrays,observer):
    if obj.type!='MESH':raise RuntimeError('Actual point carrier must be MESH: '+obj.name)
    local=observer['coordinates'](obj);matrix=np.asarray(obj.matrix_world,np.float64)
    arrays[prefix+'_local']=local;arrays[prefix+'_matrix']=matrix;arrays[prefix+'_world']=observer['world_positions'](local,matrix)
    attributes=[]
    for i,attr in enumerate(obj.data.attributes):
        row={'name':attr.name,'type':attr.data_type,'domain':attr.domain,'elements':len(attr.data)}
        if attr.data_type in ('FLOAT','INT','BOOLEAN'):
            key=prefix+'_scalar_'+str(i)
            row.update(observer['scalar_attribute'](obj.data,attr.name,key,arrays,required=True))
        else:row['valuesExportedSeparately']=attr.name=='position'
        attributes.append(row)
    return {'object':obj.name,'prefix':prefix,'evaluated':False,'vertices':len(local),'polygons':len(obj.data.polygons),
            'attributes':attributes,'positionsSha256':observer['array_sha'](local),'visibility':observer['visibility'](obj)}


def native_crown(arrays,label,node_name,capture_api):
    obj=bpy.data.objects[MAIN];modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    old=modifier.node_group;replacements=[]
    try:
        tree=runpy.run_path(str(HERE/'adult_full_groom_review_r11.py'))['copy_render_branch'](old,{},replacements);modifier.node_group=tree
        output=next(s for n in tree.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output for s in n.inputs if s.type=='GEOMETRY')
        source=next(s for s in tree.nodes[node_name].outputs if s.type=='GEOMETRY')
        template=tree.nodes['Group'].node_tree
        root=tree.nodes.new('GeometryNodeGroup');root.node_tree=template.nodes['Group.007'].node_tree
        if root.outputs.get('Root Position') is None:raise RuntimeError('Measured native root-position field is unavailable')
        xyz=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(root.outputs['Root Position'],xyz.inputs['Vector'])
        matrix=np.asarray(obj.matrix_world,np.float64);comparisons=[]
        for axis in range(3):
            terms=[math(tree,'MULTIPLY',xyz.outputs[letter],matrix[axis,j]) for j,letter in enumerate('XYZ')]
            world=math(tree,'ADD',math(tree,'ADD',math(tree,'ADD',terms[0],terms[1]),terms[2]),matrix[axis,3])
            comparisons.extend([math(tree,'GREATER_THAN',world,ROI_MIN[axis]),math(tree,'LESS_THAN',world,ROI_MAX[axis])])
        selected=comparisons[0]
        for value in comparisons[1:]:
            boolean=tree.nodes.new('FunctionNodeBooleanMath');boolean.operation='AND'
            tree.links.new(selected,boolean.inputs[0]);tree.links.new(value,boolean.inputs[1]);selected=boolean.outputs[0]
        sample=tree.nodes.new('GeometryNodeResampleCurve');mode=sample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Measured Blender 5.2 Resample Mode menu missing')
        mode.default_value='Count';sample.inputs['Count'].default_value=COUNT
        if mode.default_value!='Count' or sample.inputs['Count'].default_value!=COUNT:raise RuntimeError('Native path sample settings did not persist')
        tree.links.new(source,sample.inputs['Curve']);tree.links.new(selected,sample.inputs['Selection'])
        for link in list(output.links):tree.links.remove(link)
        tree.links.new(sample.outputs['Curve'],output);tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());data=e.data
        values=np.empty(len(data.position_data)*3,np.float32);data.position_data.foreach_get('vector',values)
        m=np.asarray(e.matrix_world,np.float64);world=values.reshape(-1,3)@m[:3,:3].T+m[:3,3]
        starts=np.array([c.first_point_index for c in data.curves],np.int32);lengths=np.array([c.points_length for c in data.curves],np.int32)
        groups=capture_api['integer_attribute'](obj,GROUP)
        ids=np.flatnonzero(lengths==COUNT)
        if not len(ids):raise RuntimeError('Native positive-crown region is empty')
        measured_prefix='post-tie' if node_name==STAGE else 'final'
        base_starts=arrays[measured_prefix+'_starts'];base_world=arrays[measured_prefix+'_world'];base_groups=arrays[measured_prefix+'_group']
        if not np.array_equal(groups,base_groups):raise RuntimeError('Native sampling changed ordered curve identities')
        roots=base_world[base_starts];inside=np.all(roots>ROI_MIN+2e-6,axis=1)&np.all(roots<ROI_MAX-2e-6,axis=1)
        outside=np.any(roots<ROI_MIN-2e-6,axis=1)|np.any(roots>ROI_MAX+2e-6,axis=1)
        mask=np.zeros(len(starts),bool);mask[ids]=True
        if np.any(inside&~mask) or np.any(outside&mask):raise RuntimeError('Native ROI selection does not match the actual world-root region')
        paths=np.concatenate([world[starts[i]:starts[i]+COUNT] for i in ids]).astype(np.float32)
        if np.linalg.norm(paths[::COUNT]-roots[ids],axis=1).max()>2e-6:raise RuntimeError('Native sampled path roots differ from the actual stage')
        arrays[label+'_world']=paths;arrays[label+'_starts']=np.arange(len(ids),dtype=np.int32)*COUNT
        arrays[label+'_lengths']=np.full(len(ids),COUNT,np.int32);arrays[label+'_group']=groups[ids];arrays[label+'_original_curve_ids']=ids.astype(np.int32)
        return {'prefix':label,'stage':node_name,'curves':len(ids),'samplesPerCurve':COUNT,'samples':len(paths),
                'worldRootRegion':[ROI_MIN.tolist(),ROI_MAX.tolist()],'boundaryComparisonToleranceMeters':2e-6,
                'realNativeNode':'GeometryNodeResampleCurve / Mode=Count','temporaryOnly':True}
    finally:
        modifier.node_group=old;obj.update_tag();bpy.context.view_layer.update()


def native_complete_gather(arrays,capture_api):
    """Reuse the production tie sampler, retaining actual ordered curve IDs."""
    scratch={'after-final_'+suffix:arrays['final_'+suffix] for suffix in ('world','starts','lengths','group','radius')}
    row=capture_api['capture_native_gather'](scratch)
    prefix='gather_native_after_'
    for key,value in scratch.items():
        if key.startswith(prefix):
            suffix=key[len(prefix):]
            if suffix=='original_curve':suffix='original_curve_ids'
            arrays['native_gather_final_'+suffix]=value
    row['prefix']='native_gather_final'
    row['includesVerifiedTempleCurtainCurves']=True
    return row


def native_tail(obj,prefix,raw_prefix,arrays):
    """Temporary native sampling of all current static tail paths, not a proxy."""
    if obj.type!='CURVES' or any(m.show_viewport or m.show_render for m in obj.modifiers):
        raise RuntimeError('Current static tail must have no active deformation: '+obj.name)
    tree=None;modifier=None
    try:
        tree=bpy.data.node_groups.new('TEMP R11 post-tie native tail observation','GeometryNodeTree')
        tree.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        tree.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        source=tree.nodes.new('NodeGroupInput');output=tree.nodes.new('NodeGroupOutput')
        sample=tree.nodes.new('GeometryNodeResampleCurve');mode=sample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Measured Blender 5.2 Resample Mode menu missing')
        mode.default_value='Count';sample.inputs['Count'].default_value=COUNT;sample.inputs['Selection'].default_value=True
        if mode.default_value!='Count' or sample.inputs['Count'].default_value!=COUNT:raise RuntimeError('Native tail sample settings did not persist')
        tree.links.new(source.outputs['Geometry'],sample.inputs['Curve']);tree.links.new(sample.outputs['Curve'],output.inputs['Geometry'])
        modifier=obj.modifiers.new('TEMP R11 native tail observation','NODES');modifier.node_group=tree
        tree.update_tag();obj.update_tag();bpy.context.view_layer.update()
        row=runpy.run_path(str(HERE/'observe_donor_gn_stages_r11.py'))['capture'](obj,prefix,arrays)
        if row['curves']!=2480 or not np.all(arrays[prefix+'_lengths']==COUNT):raise RuntimeError('Actual native tail topology differs from the static source')
        radius=arrays.get(prefix+'_radius')
        if radius is None or len(radius)!=2480*COUNT or not np.isfinite(radius).all() or radius.min()<0:
            raise RuntimeError('Native tail requires its actual finite point radius')
        roots=arrays[raw_prefix+'_world'][arrays[raw_prefix+'_starts']]
        difference=float(np.linalg.norm(arrays[prefix+'_world'][arrays[prefix+'_starts']]-roots,axis=1).max())
        if difference>2e-6:raise RuntimeError('Native tail sampling changed measured roots')
        arrays[prefix+'_original_curve_ids']=np.arange(2480,dtype=np.int32)
        row.update({'samplesPerCurve':COUNT,'temporaryOnly':True,'maximumRootDifferenceMeters':difference,
                    'radiusPolicy':'Actual native interpolated radius, converted to world units by maximum object singular scale'})
        return row
    finally:
        if modifier is not None:obj.modifiers.remove(modifier)
        if tree is not None and tree.users==0:bpy.data.node_groups.remove(tree)
        obj.update_tag();bpy.context.view_layer.update()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);source=scoped(args.source);report_path=scoped(args.output).with_suffix('.json');array_path=report_path.with_suffix('.npz')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or report_path.exists() or array_path.exists():raise RuntimeError('Fresh explicit source and unused observation outputs required')
    for name,digest in PINS.items():
        if sha(HERE/name)!=digest:raise RuntimeError('Frozen observer dependency changed: '+name)
    observer=runpy.run_path(str(HERE/'observe_reference_tie_alignment_r11_v2.py'))
    capture_api=runpy.run_path(str(HERE/'prepare_reference_tie_reconnection_v2_r11.py'))
    donor=bpy.data.objects[MAIN]
    if not donor.get('R11_reference_gather_reconnection_v2') or bpy.data.objects.get(CARRIER) is None:raise RuntimeError('An actually saved tie-reconnected source is required')
    carriers=sorted((o for o in bpy.data.objects if o.get('R11_groom_role_root_source')==MAIN),key=lambda o:o.name)
    names=[MAIN,CARRIER,'R11 Female head study','R11 Donor / reference face wisps',GROWTH]
    names.extend(name for _,name in observer['TAILS']);names.extend(name for _,name in observer['TIES']);names.extend(o.name for o in carriers)
    objects={name:bpy.data.objects[name] for name in names}
    bpy.context.view_layer.update();before={name:observer['raw_contract'](obj) for name,obj in objects.items()}
    rig=bpy.data.objects['Character_Rig'];pose=rig.data.pose_position;scene=bpy.context.scene;frame=(scene.frame_current,scene.frame_subframe)
    action=rig.animation_data.action if rig.animation_data else None;digest=sha(source);arrays={}
    report={'source':args.source,'sourceSha256':digest,'helperSha256':sha(__file__),'savedBlend':False,'rendered':False,
            'status':'collecting','sourcePose':pose,'observationPose':'REST','blenderVersion':bpy.app.version_string,
            'scope':'One current post-tie read-only observation shared by crown start-flow and ring-band fit; exported geometry does not itself certify either future modification.'}
    error=None
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        print('R11_POST_TIE_ACTUAL_GROOM_STAGES',flush=True)
        report['actualStages']=capture_api['capture_stages'](arrays,(('post-tie',STAGE),('final','R11 strand tip taper')))
        print('R11_POST_TIE_NATIVE_CROWN_PATHS',flush=True)
        report['nativeCrownBeforeHeadContact']=native_crown(arrays,'native_post_tie',STAGE,capture_api)
        report['nativeCrownFinal']=native_crown(arrays,'native_final','R11 strand tip taper',capture_api)
        print('R11_POST_TIE_NATIVE_COMPLETE_GATHER',flush=True)
        report['nativeGatherFinal']=native_complete_gather(arrays,capture_api)
        print('R11_POST_TIE_CURRENT_HEAD_TAILS_RINGS_ROOT_FIELDS',flush=True)
        report['head']=observer['collect_mesh'](objects['R11 Female head study'],'head',arrays,require_face_sets=True,require_actual_evaluation=True)
        report['ears']=[observer['collect_ear'](arrays,4,'ear_positive'),observer['collect_ear'](arrays,5,'ear_negative')]
        report['tails']=[observer['collect_curves'](objects[name],prefix,arrays,expected_curves=2480,expected_points_per_curve=35) for prefix,name in observer['TAILS']]
        print('R11_POST_TIE_NATIVE_COMPLETE_TAILS',flush=True)
        report['nativeTails']=[native_tail(objects[name],'native_'+prefix,prefix,arrays) for prefix,name in observer['TAILS']]
        report['ties']=[{'object':name,'raw':observer['collect_mesh'](objects[name],prefix+'_raw',arrays,evaluated=False),
                         'evaluated':observer['collect_mesh'](objects[name],prefix+'_eval',arrays,require_actual_evaluation=True)} for prefix,name in observer['TIES']]
        report['mainRaw']=observer['collect_curves'](donor,'main_raw',arrays,expected_curves=346,require_roles=True)
        report['faceWisps']=observer['collect_curves'](objects['R11 Donor / reference face wisps'],'face_wisps_raw',arrays)
        report['growthMesh']=observer['collect_mesh'](objects[GROWTH],'growth_raw',arrays,evaluated=False)
        report['growthDensity']=observer['scalar_attribute'](objects[GROWTH].data,'R11 reference scalp density','growth_actual_density',arrays,required=True,expected_type='FLOAT',expected_domain='POINT')
        report['targetCarrier']=point_mesh(objects[CARRIER],'target_carrier',arrays,observer)
        report['rootCarriers']=[point_mesh(obj,'root_carrier_'+str(i),arrays,observer) for i,obj in enumerate(carriers)]
        report['status']='collected'
    except Exception as exc:
        error=exc;report['status']='failed';report['error']=str(exc)
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()
    after={name:observer['raw_contract'](obj) for name,obj in objects.items()}
    changes={name:{key:{'before':before[name][key],'after':after[name][key]} for key in before[name] if before[name][key]!=after[name][key]} for name in before if before[name]!=after[name]}
    report['readOnlyRestoredPoseChanges']=changes
    report['sourceUnchanged']=sha(source)==digest;report['sourcePoseRestored']=rig.data.pose_position==pose
    report['sourceFrameAndActionUnchanged']=(scene.frame_current,scene.frame_subframe)==frame and (rig.animation_data.action if rig.animation_data else None)==action
    if changes or not all(report[k] for k in ('sourceUnchanged','sourcePoseRestored','sourceFrameAndActionUnchanged')):
        error=error or RuntimeError('Read-only source/pose contract changed');report['status']='failed';report['error']=str(error)
    if arrays:
        with array_path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        report['arrays']=str(array_path.relative_to(P));report['arraysSha256']=sha(array_path)
        report['arrayManifest']={key:{'shape':list(value.shape),'dtype':str(value.dtype),'sha256':observer['array_sha'](value)} for key,value in arrays.items()}
    with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_POST_TIE_OBSERVED '+json.dumps({'report':args.output,'status':report['status'],'arrays':len(arrays),'savedBlend':False}),flush=True)
    if error:raise error


if __name__=='__main__':main()
