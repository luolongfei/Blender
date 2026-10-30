"""Rebuild only the two upper lash root lines on the actual neutral eyelid.

Preserves physical radii, black shader, lower lashes, brows and old datablocks.
Roots sit 0.30-0.60 mm above the actual front-visible aperture; late curl is
restrained and varied. Rebinds the authored paths to actual current triangles.
Actual resampled shafts are checked at late anchors, their midpoints and warm
combinations. New-source upper-skin fields preserve blink <= .5 exactly.
Requires a fresh actual neutral observation after the final eye/midpath changes.
V4 adds early .2/.35 and warm .2 checks because the new midpath changes that interval;
geometry authoring, pinned dependencies, contact limits and the v3 late checks remain unchanged.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

MARKER='R11_upper_lash_actual_margin_regroom'
NAMES=tuple('R11 Adult lashes upper '+s for s in ('L','R'))
PUBLIC=('R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile','R11 warm closed smile')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


PINNED_HELPERS={
    'face_fibre_binding_r11_v2':'47483f7a09d535214cfa43b20b766ca95adf66790b3a02b6fb1c0f221133ce49',
    'upper_lash_corrective_validation_r11':'36a8713e93a962504b6f1546fb466afd70e3a7af394c449b40176b32999e3c0a',
    'upper_lash_late_surface_r11':'2257f0c73f7b21d2bdbcb22cfb9320563c665189f0ba6b6c980777abc73efbeb',
}
STATES=((0.,0.,0.),(.2,.2,0.),(.35,.35,0.),(.5,.5,0.),(.575,.575,0.),(.65,.65,0.),(.725,.725,0.),
        (.8,.8,0.),(.85,.85,0.),(.9,.9,0.),(.95,.95,0.),(1.,1.,0.),
        (.2,.2,1.),(.5,.5,1.),(.8,.8,1.),(.9,.5,1.),(1.,1.,1.))


def api(name):
    for helper,expected in PINNED_HELPERS.items():
        if sha(Path(__file__).with_name(helper+'.py'))!=expected:raise RuntimeError('Frozen upper-lash dependency differs '+helper)
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def external_contract(bpy,binding):
    # Exactly the public external scene contract, except two explicitly
    # authored lookup objects and their recorded final modifiers. Their full
    # fields, nodes and driver contracts are separately validated on every pose.
    scene=bpy.context.scene;late=json.loads(scene['R11_upper_lash_late_surface']) if 'R11_upper_lash_late_surface' in scene else {'objects':{}}
    if set(late['objects'])-set(NAMES):raise RuntimeError('Unexpected corrective object exemption')
    lookup_names={entry['lookupObject'] for entry in late['objects'].values()};rows=[]
    for obj in scene.objects:
        if obj.name in lookup_names:continue
        expected_late=late['objects'].get(obj.name,{}).get('modifier')
        row={'name':obj.name,'object':obj.as_pointer(),'data':None if obj.name in binding['OBJECTS'] else getattr(getattr(obj,'data',None),'as_pointer',lambda:None)(),
             'parent':obj.parent.as_pointer() if obj.parent else None,'world':[list(r) for r in obj.matrix_world],
             'materials':[slot.material.as_pointer() if slot.material else None for slot in obj.material_slots],
             'modifiers':[(modifier.name,modifier.type) for modifier in obj.modifiers if modifier.name!=binding['MODIFIER'] and modifier.name!=expected_late]}
        if obj.type=='ARMATURE':row['pose']=[(bone.name,[list(r) for r in bone.matrix_basis]) for bone in obj.pose.bones]
        rows.append(row)
    return binding['_hash']({'objects':rows,'actions':[(action.name,action.as_pointer()) for action in bpy.data.actions],
                             'frame':[scene.frame_current,scene.frame_subframe]})


def quantile(v):return np.quantile(np.asarray(v),[0,.1,.5,.9,1]).tolist()


def bvh(mesh):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([Vector(p) for p in mesh['world']],[tuple(t) for t in mesh['triangles']],all_triangles=True)


def assert_dependencies(observation,arrays,head,eyes):
    """Strict actual peri-lid surface/globe comparison; excludes unrelated jaw."""
    if head['topologySha256']!=observation['headTopologySha256']:raise RuntimeError('Current head topology differs from observed eyelid topology')
    global_ids=arrays['head_originalVertexIds'];reference=arrays['head_world'];selected=np.zeros(len(reference),bool)
    for side in ('L','R'):
        center=np.asarray(observation['eyes'][side]['centerWorld']);d=reference-center
        selected|=(abs(d[:,0])<=.015)&(d[:,2]>=-.009)&(d[:,2]<=.014)&(d[:,1]>=-.025)&(d[:,1]<=.010)
        if eyes[side]['world'].shape!=arrays['eye_'+side+'_world'].shape:raise RuntimeError('Current eye topology changed')
        error=float(np.linalg.norm(eyes[side]['world']-arrays['eye_'+side+'_world'],axis=1).max())
        if error>2e-7 or not np.array_equal(eyes[side]['triangles'],arrays['eye_'+side+'_triangles']):raise RuntimeError('Current actual globe differs from neutral observation '+side)
    if selected.sum()<1000:raise RuntimeError('Insufficient actual eyelid dependency points')
    delta=np.linalg.norm(head['world'][global_ids[selected]]-reference[selected],axis=1)
    if delta.max()>2e-7:raise RuntimeError('Actual neutral peri-lid dependency changed '+str(float(delta.max())))
    return {'actualNeutralDependencyPoints':int(selected.sum()),'maximumDifferenceMeters':float(delta.max()),'toleranceMeters':2e-7,
            'scope':'Both actual peri-lid front surfaces (15 mm lateral, -9/+14 mm vertical, -25/+10 mm depth around observed globe centers) and whole actual globes; unrelated jaw excluded.'}


def new_paths(side,points,counts,radii,skin,globe,eye_world):
    from mathutils import Vector
    if len(counts)!=156 or not np.all(counts==9):raise RuntimeError('Expected current 156 nine-point upper lashes per side')
    old=points.reshape(156,9,3);radii=radii.reshape(156,9);rng=random.Random(2026102931+(side=='R'))
    sign=1. if side=='L' else -1.;x_span=sign*old[:,0,0];low,high=float(x_span.min()),float(x_span.max())
    lo,hi=eye_world[:,2].min(),eye_world[:,2].max();new=[];roots=[];lengths=[]
    def ray(surface,x,z):return surface.ray_cast(Vector((float(x),-.35,float(z))),Vector((0.,1.,0.)),.7)
    def visible(x,z):
        g=ray(globe,x,z)[0]
        if g is None:return False
        h=ray(skin,x,z)[0]
        return h is None or g.y<h.y-1e-7
    for i,path in enumerate(old):
        x=float(path[0,0]);t=(sign*x-low)/(high-low)
        zs=np.linspace(lo,hi,129);flags=np.asarray([visible(x,z) for z in zs]);yes=np.flatnonzero(flags)
        if len(yes)<2 or yes[-1]>=len(zs)-1:raise RuntimeError('No bounded actual visible upper aperture at lash root '+side+' '+str(i))
        inside=float(zs[yes[-1]]);outside=float(zs[yes[-1]+1])
        for _ in range(16):
            mid=(inside+outside)*.5
            if visible(x,mid):inside=mid
            else:outside=mid
        aperture=(inside+outside)*.5
        gap=.00034+.00017*rng.random()+.000045*abs(2*t-1)
        if not .00030<=gap<=.00060:raise RuntimeError('Authored front-margin interval exceeded')
        location,normal,index,distance=ray(skin,x,aperture+gap)
        if location is None or normal.y>=-.05:raise RuntimeError('Actual anterior ciliary surface unavailable '+side+' '+str(i))
        root=np.asarray(location)+np.asarray(normal)*max(.000025,float(radii[i,0])*.55)
        old_length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum());length=old_length*.92
        forward=length*.91
        fan=float((path[-1,0]-path[0,0])*.82)
        rise=(.00100+.00082*np.sin(np.pi*t))*(.67+.40*rng.random())
        # Root first exits forward with only a small vertical component. The
        # independent late tangent creates restrained varied curl, not a comb.
        d0=np.asarray((fan/max(length,1e-9)*.7,-1.,-.055+.07*rng.random()));d0/=np.linalg.norm(d0)
        if float(d0@np.asarray(normal))<=.08:raise RuntimeError('New root tangent does not exit actual skin '+side+' '+str(i))
        tip=root+np.asarray((fan,-forward,rise))
        tip_slope=.55+.35*rng.random();d1=np.asarray((fan/max(length,1e-9)*.55,-.75,tip_slope));d1/=np.linalg.norm(d1)
        p1=root+d0*length*.34;p2=tip-d1*length*.28
        u=np.linspace(0.,1.,9)[:,None];curve=(1-u)**3*root+3*(1-u)**2*u*p1+3*(1-u)*u*u*p2+u**3*tip
        new.append(curve);lengths.append(float(np.linalg.norm(np.diff(curve,axis=0),axis=1).sum()))
        roots.append({'strand':i,'oldRoot':path[0].tolist(),'newRoot':root.tolist(),'skinPoint':list(location),'skinNormal':list(normal),
                      'actualApertureUpperZ':aperture,'authoredVerticalMarginMeters':gap,'rootSkinTriangle':int(index),
                      'initialDirectionDotSkinNormal':float(d0@np.asarray(normal)),'targetRiseMeters':rise})
    return np.asarray(new).reshape(-1,3),{'side':side,'roots':roots,'polylineLengthMetersQuantiles':quantile(lengths),
                    'rootVerticalMarginMetersQuantiles':quantile([r['authoredVerticalMarginMeters'] for r in roots]),
                    'targetRiseMetersQuantiles':quantile([r['targetRiseMeters'] for r in roots])}


def sample_actual(obj,binding,count=33):
    """Sample native interpolated curves after the real surface binding."""
    import bpy
    probe=None;group=None
    try:
        probe=obj.copy();probe.name='TEMP upper lash shaft sampling';bpy.context.scene.collection.objects.link(probe);probe.hide_render=True
        group=bpy.data.node_groups.new('TEMP upper lash actual resample','GeometryNodeTree')
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        source=group.nodes.new('NodeGroupInput');sample=group.nodes.new('GeometryNodeResampleCurve')
        # Blender 5.2.3 actual RNA inspection: Mode is a MENU socket, value 'Count'.
        mode=sample.inputs.get('Mode')
        if mode is None or mode.type!='MENU':raise RuntimeError('Expected actual Blender 5.2 Resample Mode menu socket')
        mode.default_value='Count'
        if mode.default_value!='Count':raise RuntimeError('Actual Resample Mode did not retain Count')
        sample.inputs['Count'].default_value=count
        output=group.nodes.new('NodeGroupOutput');group.links.new(source.outputs['Geometry'],sample.inputs['Curve']);group.links.new(sample.outputs['Curve'],output.inputs['Geometry'])
        modifier=probe.modifiers.new('TEMP actual native shaft resample','NODES');modifier.node_group=group;bpy.context.view_layer.update()
        e=probe.evaluated_get(bpy.context.evaluated_depsgraph_get());data=e.data
        counts=np.asarray([c.points_length for c in data.curves],np.int32)
        if len(counts)!=156 or not np.all(counts==count):raise RuntimeError('Actual resampled upper lash topology unexpected')
        points=binding['_transform'](binding['_array'](data.position_data,'vector'),e.matrix_world)
        scale=float(np.linalg.svd(np.asarray(e.matrix_world)[:3,:3],compute_uv=False).max())
        radii=binding['_array'](data.attributes['radius'].data,'value',1).ravel()*scale
        return points,radii,counts
    finally:
        if probe is not None:bpy.data.objects.remove(probe,do_unlink=True)
        if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        bpy.context.view_layer.update()


def contact_report(points,radii,counts,skin,globe):
    from mathutils import Vector
    starts=np.r_[0,np.cumsum(counts)[:-1]];shaft=np.ones(len(points),bool);shaft[starts]=False
    report={}
    for name,surface,tolerance in (('head',skin,.00005),('globe',globe,.00003)):
        signed=[]
        for point in points:
            q,n,face,distance=surface.find_nearest(Vector(point))
            if q is None:raise RuntimeError('Actual shaft nearest-surface measurement failed')
            signed.append(float(distance)*(1. if (Vector(point)-q).dot(n)>=0 else -1.))
        signed=np.asarray(signed);clearance=signed-radii
        report[name]={'minimumNonRootSurfaceClearanceMeters':float(clearance[shaft].min()),
                      'minimumRootCenterSignedDistanceMeters':float(signed[starts].min()),
                      'maximumRootCenterSignedDistanceMeters':float(signed[starts].max()),
                      'nonRootPointsBeyondTolerance':int((clearance[shaft]<-tolerance).sum()),
                      'rootPointsBeyondTolerance':int((clearance[starts]<-tolerance).sum()) if name=='globe' else int(((signed[starts]<-.00005)|(signed[starts]>.00010)).sum()),
                      'rootPolicy':'Globe includes physical radius; head follicle center must remain -50 to +100 micrometers from actual skin (authored target +25 micrometers).',
                      'toleranceMeters':tolerance}
    return report


def prepare(source_relative,observation_relative,output_relative):
    import bpy
    from mathutils import Vector
    p=project_path().resolve();source=(p/source_relative).resolve();observation_path=(p/observation_relative).resolve();out=(p/output_relative).resolve()
    report_path=out.with_suffix('.upper-lash-regroom.json');array_out=out.with_suffix('.upper-lash-regroom.npz')
    for path in (source,observation_path,out,report_path,array_out):path.relative_to(p/'work/reconstruction-r02')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or MARKER in bpy.context.scene or 'R11_upper_lash_late_surface' in bpy.context.scene or any(f.exists() for f in (out,report_path,array_out)):
        raise RuntimeError('Fresh independent source and unused outputs required, without prior upper-lash regroom')
    observation=json.loads(observation_path.read_text(encoding='utf8'));array_path=(p/observation['arrays']).resolve()
    if sha(array_path)!=observation['arraysSha256']:raise RuntimeError('Neutral lash observation arrays changed')
    reference=np.load(array_path);binding=api('face_fibre_binding_r11_v2');late_api=api('upper_lash_late_surface_r11');scene=bpy.context.scene;head=bpy.data.objects[binding['HEAD']]
    before_validation=binding['validate'](neutral=False);old_record=json.loads(scene[binding['MARKER']]);record=copy.deepcopy(old_record)
    head_protected=binding['_protected'](head);external=external_contract(bpy,binding);keys=head.data.shape_keys.key_blocks
    previous={name:float(keys[name].value) for name in PUBLIC};source_sha=sha(source)
    preserved={};old_data={};created=[];report={'source':source_relative,'sourceSha256':source_sha,'observation':observation_relative,
                'observationSha256':sha(observation_path),'helperSha256':sha(__file__),'savedBlend':False,'appearanceAccepted':False,
                'bindingBefore':before_validation,'pinnedHelpers':dict(PINNED_HELPERS),'changes':[],'states':[]};payload={};success=False
    unaffected=tuple(name for name in binding['OBJECTS'] if name not in NAMES)
    def values(left,right=None,warm=0.):
        right=left if right is None else right
        for name in PUBLIC:keys[name].value=left if name=='R11 blink L' else right if name=='R11 blink R' else warm if name=='R11 warm closed smile' else 0.
        bpy.context.view_layer.update()
    def current_points(name):
        o=bpy.data.objects[name];e=o.evaluated_get(bpy.context.evaluated_depsgraph_get())
        return binding['_transform'](binding['_array'](e.data.position_data,'vector'),e.matrix_world)
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            baseline_unaffected={}
            for pose in STATES:
                values(*pose);baseline_unaffected[pose]={name:current_points(name) for name in unaffected}
            values(0.);hd=binding['_head_mesh'](head,triangulate=True)
            eyes={s:binding['_head_mesh'](bpy.data.objects['R11 Adult sclera '+s],triangulate=True) for s in ('L','R')}
            report['dependencies']=assert_dependencies(observation,reference,hd,eyes);skin=bvh(hd);globes={s:bvh(eyes[s]) for s in ('L','R')}
            for side in ('L','R'):
                name='R11 Adult lashes upper '+side;obj=bpy.data.objects[name];old_data[name]=obj.data;prior_item=copy.deepcopy(record['objects'][name])
                old_reference=bpy.data.hair_curves[prior_item['originalData']]
                for data in (obj.data,old_reference):
                    if data.name not in preserved:preserved[data.name]={'data':data,'fakeUser':data.use_fake_user,'contract':binding['_curve_contract'](data)};data.use_fake_user=True
                counts=np.asarray([c.points_length for c in obj.data.curves],np.int32);raw_radius=binding['_array'](obj.data.attributes['radius'].data,'value',1).ravel()
                matrix=np.asarray(obj.matrix_world);scale=float(np.linalg.svd(matrix[:3,:3],compute_uv=False).max())
                world,design=new_paths(side,current_points(name),counts,raw_radius*scale,skin,globes[side],eyes[side]['world'])
                payload[side+'_newNeutralWorld']=world;payload[side+'_radius']=raw_radius;payload[side+'_counts']=counts
                new_reference=old_reference.copy();created.append(new_reference);new_reference.name=old_reference.name+' - R11 actual margin regroom reference';new_reference.use_fake_user=True
                new_reference.position_data.foreach_set('vector',np.asarray(binding['_transform'](world,np.linalg.inv(matrix)),np.float32).ravel())
                new_reference.materials.clear()
                for mat in obj.data.materials:new_reference.materials.append(mat)
                new_bound=new_reference.copy();created.append(new_bound);new_bound.name=new_reference.name+' bound'
                obj.data=new_reference
                mapping,plan=binding['_plan'](obj,hd,skin)
                for i in range(3):
                    for label,array in (('vertex',mapping['vertices']),('corner',mapping['corners'])):
                        new_bound.attributes.new(binding['P']+label+'_'+str(i),'INT','CURVE').data.foreach_set('value',np.ascontiguousarray(array[:,i]))
                new_bound.attributes.new(binding['P']+'barycentric','FLOAT_VECTOR','CURVE').data.foreach_set('vector',mapping['weights'].ravel())
                new_bound.attributes.new(binding['P']+'frame_offset','FLOAT_VECTOR','POINT').data.foreach_set('vector',mapping['offsets'].ravel())
                obj.data=new_bound
                if not np.array_equal(binding['_array'](new_bound.attributes['radius'].data,'value',1).ravel(),raw_radius):raise RuntimeError('Upper-lash physical radii changed')
                if binding['_curve_contract'](new_reference)!=binding['_curve_contract'](new_bound):raise RuntimeError('Authored reference/bound path contracts differ')
                item={**plan,'originalData':new_reference.name,'candidateData':new_bound.name,'originalFakeUser':True,
                      'originalContractSha256':binding['_hash'](binding['_curve_contract'](new_reference)),
                      'authoredPathRevision':{'helperSha256':sha(__file__),'sourceSha256':source_sha,'priorOriginalData':prior_item['originalData'],
                                             'priorBoundData':prior_item['candidateData'],'reason':'Actual anterior lash margin and restrained late curl; physical radius and existing black material preserved.'}}
                record['objects'][name]=item
                report['changes'].append({'object':name,'priorBindingItem':prior_item,'authoredBindingItem':copy.deepcopy(item),'design':design,
                                          'priorBoundContract':preserved[old_data[name].name]['contract'],'retainedOldData':old_data[name].name})
            scene[binding['MARKER']]=json.dumps(record,ensure_ascii=False);bpy.context.view_layer.update()
            unchanged_poses=tuple(pose for pose in STATES if max(pose[:2])<=.5);base_early={}
            for pose in unchanged_poses:
                values(*pose);base_early[pose]={name:current_points(name) for name in NAMES}
            values(0.)
            report['lateSurface']=late_api['install'](head,binding,report['changes'],payload)
            report['actualNeutralBinding']=binding['validate'](neutral=False)
            report['exactEarlyPosePreservation']=[]
            for pose in unchanged_poses:
                values(*pose)
                for name in NAMES:
                    if not np.array_equal(current_points(name),base_early[pose][name]):raise RuntimeError('Late field changed an exact-preserved early pose '+str(pose)+' '+name)
                report['exactEarlyPosePreservation'].append({'blinkL':pose[0],'blinkR':pose[1],'warm':pose[2],'pointwiseExact':True})
            for pose_index,pose in enumerate(STATES):
                values(*pose);current_head=binding['_head_mesh'](head,triangulate=True);current_skin=bvh(current_head)
                state={'blinkL':pose[0],'blinkR':pose[1],'warm':pose[2],'unaffectedFibreParity':{},'upperLashes':{}}
                for name in unaffected:
                    if not np.array_equal(current_points(name),baseline_unaffected[pose][name]):raise RuntimeError('Brow/lower-lash actual movement changed '+name)
                    state['unaffectedFibreParity'][name]=True
                for side in ('L','R'):
                    obj=bpy.data.objects['R11 Adult lashes upper '+side];points,radii,counts=sample_actual(obj,binding)
                    contacts=contact_report(points,radii,counts,current_skin,globes[side]);state['upperLashes'][side]=contacts
                    tag='pose_'+str(pose_index).zfill(2)+'_'+side
                    payload[tag+'_world']=points;payload[tag+'_radius']=radii;payload[tag+'_counts']=counts
                report['states'].append(state)
                print('R11_UPPER_LASH_ACTUAL_STATE '+json.dumps(state),flush=True)
                if any(info['nonRootPointsBeyondTolerance'] or info['rootPointsBeyondTolerance'] for side in state['upperLashes'].values() for info in side.values()):
                    raise RuntimeError('Actual resampled upper shaft/root contact exceeds limit at pose '+str(pose))
                state['binding']=binding['validate'](neutral=False)
            values(0.)
        for name,value in previous.items():keys[name].value=value
        bpy.context.view_layer.update()
        if binding['_protected'](head)!=head_protected or external_contract(bpy,binding)!=external or sha(source)!=source_sha:
            raise RuntimeError('Regroom altered protected head, other scene objects, pose or source')
        for row in preserved.values():
            if binding['_curve_contract'](row['data'])!=row['contract']:raise RuntimeError('Retained old lash data changed')
        report['retainedDataFakeUserBefore']={name:row['fakeUser'] for name,row in preserved.items()}
        report['bindingAfterRestorePose']=binding['validate'](neutral=False)
        report['scope']='Only two upper-lash paths, actual root maps and explicitly recorded late-blink fields authored on this source; thin radii, black shader, brows/lower lashes and head preserved. Seventeen finite blink/anchor/intermediate/warm states are not all-expression or appearance approval.'
        scene[MARKER]=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(out)):raise RuntimeError('Upper-lash regroom candidate save failed')
        report.update(output=output_relative,outputSha256=sha(out),savedBlend=True);success=True
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        for name,value in previous.items():keys[name].value=value
        if not success:
            if 'R11_upper_lash_late_surface' in scene:late_api['remove_current']()
            for name,data in old_data.items():bpy.data.objects[name].data=data
            scene[binding['MARKER']]=json.dumps(old_record,ensure_ascii=False)
            for row in preserved.values():row['data'].use_fake_user=row['fakeUser']
            for data in created:
                data.use_fake_user=False
                if data.users==0:bpy.data.hair_curves.remove(data)
            if MARKER in scene:del scene[MARKER]
        bpy.context.view_layer.update()
        if payload:
            np.savez_compressed(array_out,**payload);report.update(actualArrays=str(array_out.relative_to(p)),actualArraysSha256=sha(array_out))
        with report_path.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_UPPER_LASH_REGROOM_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--observation',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(args.source,args.observation,args.output)
