"""Bounded, reversible eye/globe/lid proportion study for the adult R11 head.

Globe and iris scale uniformly about their actual unchanged centres. A single
static head key adapts both lid sheets and the adjacent orbit. The new head
topology, nose, mouth, rig, materials, images and prior keys remain intact.
This is one visual candidate, not a diagnosis or a completed blink system.
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
HEAD='R11 Female head study';KEY='R11 adult eye proportion';MARKER='R11_adult_eye_proportion'
PARAMETERS={'globeMeanXZDiameterMeters':.0246,'lidHorizontalScale':1.17,'lidVerticalScale':1.07,
            'tissueCoreGeodesicMeters':.003,'tissueOuterGeodesicMeters':.018}
EYES=tuple('R11 Adult '+part+' '+side for side in ('L','R') for part in ('sclera','iris'))


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def _array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a


def _mixed(mesh):
    keys=mesh.shape_keys
    if not keys or not keys.use_relative:raise RuntimeError('Actual relative R11 face keys required')
    result=_array(keys.reference_key.data).astype(float)
    for key in keys.key_blocks:
        if key!=keys.reference_key and not key.mute and key.value:
            result+=key.value*(_array(key.data).astype(float)-_array(key.relative_key.data).astype(float))
    return result


def _transform(co,matrix):
    m=np.asarray(matrix,float);return np.asarray(co)@m[:3,:3].T+m[:3,3]


def _surface(obj,override_world=None):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.data
    world=_transform(_array(mesh.vertices),ev.matrix_world) if override_world is None else override_world
    faces=[tuple(p.vertices) for p in mesh.polygons]
    return BVHTree.FromPolygons([Vector(p) for p in world],faces),world


def _ray(bvh,x,z):
    from mathutils import Vector
    return bvh.ray_cast(Vector((float(x),-.35,float(z))),Vector((0,1,0)),.7)[0]


def _measure_rest(head):
    import bpy
    skin,_=_surface(head);result={}
    for side in ('L','R'):
        globe,world=_surface(bpy.data.objects['R11 Adult sclera '+side])
        lo=world.min(0);hi=world.max(0);center=(lo+hi)*.5
        xs=np.linspace(lo[0]+.0001,hi[0]-.0001,151);zs=np.linspace(lo[2]+.0001,hi[2]-.0001,121)
        visible=[];central=[]
        for i,x in enumerate(xs):
            ids=[]
            for j,z in enumerate(zs):
                g=_ray(globe,x,z)
                if g is None:continue
                h=_ray(skin,x,z)
                if h is None or g.y<h.y-1e-7:visible.append((i,j));ids.append(j)
            if i==len(xs)//2 and ids:central=[float(zs[min(ids)]),float(zs[max(ids)])]
        if not visible:raise RuntimeError('No actual visible globe: '+side)
        ij=np.asarray(visible,np.int32);dx=xs[1]-xs[0];dz=zs[1]-zs[0]
        # A separate diagnostic for an exposed deep socket outside the globe;
        # it is not silently counted as a larger, successful white-eye opening.
        hole=0;hx=np.arange(center[0]-.016,center[0]+.0161,.00025);hz=np.arange(center[2]-.005,center[2]+.0051,.00025)
        for x in hx:
            for z in hz:
                if _ray(globe,x,z) is not None:continue
                h=_ray(skin,x,z)
                if h is None or h.y>center[1]+.003:hole+=1
        result[side]={'centerWorld':center.tolist(),'diametersWorld':(hi-lo).tolist(),
            'frontVisibleGlobeWidthMeters':float(xs[ij[:,0].max()]-xs[ij[:,0].min()]),
            'frontVisibleGlobeAreaSquareMeters':float(len(visible)*dx*dz),
            'centralOpeningZ':central,'centralOpeningHeightMeters':float(central[1]-central[0]) if central else None,
            'outsideGlobeDeepSocketAreaSquareMeters':float(hole*.00025**2),
            'measurement':'Actual head/eye front ray grid, 151 by 121; visible globe is distinct from the author canthus-control span. Socket-area diagnostic uses 0.25 mm cells.'}
    return result


def _plan_rest(head):
    import bpy
    skin=_api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']()
    layers=json.loads((project_path()/'work/reconstruction-r02/female-eyelid-layers-r11.json').read_text())
    mixed=_mixed(head.data);T=np.asarray(head.matrix_world,float);world=_transform(mixed,T)
    allowed=skin['_vertices_of_groups'](donor,{2,3,17});graph=skin['_graph'](world,donor['edges'],allowed)
    delta_world=np.zeros_like(world);reports={};scales={}
    for side in ('L','R'):
        obj=bpy.data.objects['R11 Adult sclera '+side];old_bvh,points=_surface(obj)
        lo=points.min(0);hi=points.max(0);center=(lo+hi)*.5;diameter=float(((hi-lo)[0]+(hi-lo)[2])*.5)
        if not .0215<diameter<.023:raise RuntimeError('Unexpected current adult eye size; do not stack proportion studies')
        scale=PARAMETERS['globeMeanXZDiameterMeters']/diameter;scales[side]=scale
        new_bvh,_=_surface(obj,center+(points-center)*scale)
        seeds=sorted({int(i) for layer in layers['layers'][side][:3] for component in layer['components'] for i in component['indices']})
        distance=skin['_distance'](graph,seeds,PARAMETERS['tissueOuterGeodesicMeters'])
        weight=1.-skin['_smooth']((distance-PARAMETERS['tissueCoreGeodesicMeters'])/(PARAMETERS['tissueOuterGeodesicMeters']-PARAMETERS['tissueCoreGeodesicMeters']))
        geodesic_weight=weight.copy()
        weight*=skin['_smooth']((abs(world[:,0])-.010)/.007)
        weight*=1.-skin['_smooth']((abs(world[:,0]-center[0])-.0115)/.0085)
        outside=(~allowed)|(world[:,2]<center[2]-.028)|(world[:,2]>center[2]+.030)
        weight[outside]=0.;geodesic_weight[outside]=0.
        relative=world-center;candidate=world.copy()
        sign=1. if side=='L' else -1.;along=sign*relative[:,0];side_x=sign*world[:,0]
        edge_x=.0115;amount=(PARAMETERS['lidHorizontalScale']-1.)*edge_x
        horizontal=(PARAMETERS['lidHorizontalScale']-1.)*along
        medial=along<-edge_x;lateral=along>edge_x
        horizontal[medial]=-amount*skin['_smooth']((side_x[medial]-.009)/(abs(center[0])-edge_x-.009))
        horizontal[lateral]=amount*(1.-skin['_smooth']((along[lateral]-edge_x)/.016))
        candidate[:,0]=world[:,0]+sign*horizontal*geodesic_weight
        candidate[:,2]=center[2]+relative[:,2]*(1.+(PARAMETERS['lidVerticalScale']-1.)*weight)
        candidate[:,1]=center[1]+relative[:,1]*(1.+(scale-1.)*weight)
        projected=[];fallback=[]
        for index in np.flatnonzero(weight>1e-10):
            old_hit=_ray(old_bvh,world[index,0],world[index,2]);new_hit=_ray(new_bvh,candidate[index,0],candidate[index,2])
            if old_hit is None or new_hit is None or relative[index,1]>.003:
                fallback.append(int(index));continue
            radial=np.linalg.norm(candidate[index,[0,2]]-center[[0,2]])
            edge=skin['_smooth']((PARAMETERS['globeMeanXZDiameterMeters']*.5-radial-.0003)/.0020)
            correction=float(new_hit.y-old_hit.y)-relative[index,1]*(scale-1.)
            candidate[index,1]+=correction*weight[index]*edge
            projected.append(int(index))
        change=candidate-world;delta_world+=change;selected=np.linalg.norm(change,axis=1)>1e-10
        canthi=[landmarks['eyes'][side]['innerCanthusCandidate'],landmarks['eyes'][side]['outerCanthusCandidate']]
        reports[side]={'sourceGlobeDiameterMeters':diameter,'uniformGlobeScale':scale,
            'targetGlobeMeanXZDiameterMeters':PARAMETERS['globeMeanXZDiameterMeters'],'centerUnchanged':center.tolist(),
            'affectedCageVertices':int(selected.sum()),'maximumTissueDisplacementMeters':float(np.linalg.norm(change,axis=1).max()),
            'authorCanthusControlIndices':canthi,'authorCanthusControlSpanBeforeX':float(np.ptp(world[canthi,0])),
            'authorCanthusControlSpanAfterX':float(np.ptp(candidate[canthi,0])),
            'actualFrontSphereClearanceTransportVertices':projected,'smoothOrbitalFallbackVertices':fallback}
    delta=delta_world@np.linalg.inv(T[:3,:3]).T;selected=np.linalg.norm(delta,axis=1)>1e-10
    edges=donor['edges'];before=mixed[edges[:,1]]-mixed[edges[:,0]];after=before+delta[edges[:,1]]-delta[edges[:,0]]
    affected=selected[edges].any(1);ratio=np.linalg.norm(after,axis=1)/np.maximum(np.linalg.norm(before,axis=1),1e-12)
    tris=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face=donor['corner_vertices'][start:start+count]
        if selected[face].any():tris.extend((int(face[0]),int(face[k]),int(face[k+1])) for k in range(1,len(face)-1))
    tris=np.asarray(tris,np.int32);a=mixed[tris];b=(mixed+delta)[tris]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);aa=np.linalg.norm(na,axis=1);bb=np.linalg.norm(nb,axis=1);valid=aa>1e-12
    alignment=np.sum(na[valid]*nb[valid],axis=1)/(aa[valid]*bb[valid])
    if alignment.min()<=0 or np.linalg.norm(delta_world,axis=1).max()>.0055 or ratio[affected].min()<.65 or ratio[affected].max()>1.50:
        raise RuntimeError('Eye proportion tissue folds or exceeds bounded deformation '+json.dumps({'normalMinimum':float(alignment.min()),'maximumMovement':float(np.linalg.norm(delta_world,axis=1).max()),'edgeRatioRange':[float(ratio[affected].min()),float(ratio[affected].max())]}))
    return delta.astype(np.float32),scales,{'parameters':PARAMETERS,'provenance':provenance,'sides':reports,
        'affectedCageVertices':int(selected.sum()),'rawNormalAlignmentMinimum':float(alignment.min()),
        'rawEdgeRatioQuantiles':np.quantile(ratio[affected],[0,.05,.5,.95,1]).tolist(),
        'topologyLayerReportSha256':_sha(project_path()/'work/reconstruction-r02/female-eyelid-layers-r11.json'),
        'sourceMixSha256':skin['_ah'](mixed),'sourceMixValues':{k.name:float(k.value) for k in head.data.shape_keys.key_blocks},
        'scope':'Uniform globe/iris scaling, horizontal/vertical lid tissue widening with actual front-sphere clearance transport and outer-orbit fade. Eye centres, nose, lips, ears, body and rig are unchanged.',
        'appearanceAccepted':False,'blinkRebuildRequired':True}


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or bpy.context.scene.get(MARKER):raise RuntimeError('New independent background eye candidate required')
    if 'R11_face_fibre_binding' not in bpy.context.scene:raise RuntimeError('Actual R11 face fibre binding required')
    head=bpy.data.objects[HEAD];mesh=head.data
    if len(mesh.vertices)!=10582 or mesh.users!=1:raise RuntimeError('Actual independent female head cage required')
    keys=mesh.shape_keys
    if not keys or KEY in keys.key_blocks or abs(keys.key_blocks['R11 gentle upper lid coverage'].value-1)>1e-6:raise RuntimeError('Calibrate on the actual upper-lid default 1')
    for name in ('R11 natural soft smile','R11 gentle smile','R11 blink L','R11 blink R'):
        if name in keys.key_blocks and abs(keys.key_blocks[name].value)>1e-8:raise RuntimeError('Neutral expression required for eye calibration: '+name)
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve());source_sha=_sha(source)
    qa=_api('face_contact_audit_r11');jaw=_api('adult_jaw_contour_r11');skin=_api('natural_skin_adult_r11');binding=_api('face_fibre_binding_r11')
    before_keys=jaw['_keys'](mesh);active=head.active_shape_key_index;basis=_array(keys.reference_key.data)
    pose=_api('production_uv_r08')['_pose_contract']();backups={};key=None;success=False
    try:
        with qa['rest_evaluation']():
            visible_before=_measure_rest(head);delta,scales,report=_plan_rest(head)
            for side in ('L','R'):
                for part in ('sclera','iris'):
                    obj=bpy.data.objects['R11 Adult '+part+' '+side];old=obj.data
                    if old.shape_keys:raise RuntimeError('Inspect unexpected eye shape keys')
                    backups[obj.name]={'data':old,'fake':old.use_fake_user,'coordinates':_array(old.vertices),
                        'world':[list(row) for row in obj.matrix_world],'materials':[m.as_pointer() if m else None for m in old.materials],
                        'uv':{uv.name:_array(uv.data,'uv',2) for uv in old.uv_layers}}
                    old.use_fake_user=True;obj.data=old.copy();obj.data.name='R11 proportion study '+part+' '+side
                    obj.data.vertices.foreach_set('co',(backups[obj.name]['coordinates']*scales[side]).astype(np.float32).ravel());obj.data.update()
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=1.
            key.data.foreach_set('co',(basis+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            visible_after=_measure_rest(head)
            for side in ('L','R'):
                old=visible_before[side];new=visible_after[side]
                if np.max(np.abs(np.asarray(old['centerWorld'])-new['centerWorld']))>3e-7:raise RuntimeError('Eye centre moved '+side)
                actual=float((new['diametersWorld'][0]+new['diametersWorld'][2])*.5)
                if abs(actual-PARAMETERS['globeMeanXZDiameterMeters'])>3e-7:raise RuntimeError('Actual globe size mismatch '+side)
                if not new['frontVisibleGlobeWidthMeters']>old['frontVisibleGlobeWidthMeters']*1.07:
                    raise RuntimeError('Actual lid/eye opening did not widen sufficiently '+side)
                allowance=old['outsideGlobeDeepSocketAreaSquareMeters']*scales[side]**2+.5e-6
                if new['outsideGlobeDeepSocketAreaSquareMeters']>allowance:
                    raise RuntimeError('Wider lid exposes deep socket outside globe '+json.dumps({'side':side,'before':old,'after':new,'allowanceSquareMeters':allowance}))
            fibre=binding['validate'](neutral=False)
        after_keys=jaw['_keys'](mesh)
        for name,old in before_keys.items():
            new=after_keys[name]
            if not np.array_equal(old['coords'],new['coords']) or any(old[k]!=new[k] for k in old if k!='coords'):raise RuntimeError('Existing key changed '+name)
        for name,old in backups.items():
            obj=bpy.data.objects[name]
            if not np.array_equal(_array(old['data'].vertices),old['coordinates']):raise RuntimeError('Eye backup coordinates changed')
            if [m.as_pointer() if m else None for m in obj.data.materials]!=old['materials']:raise RuntimeError('Eye materials changed')
            if any(not np.array_equal(_array(obj.data.uv_layers[n].data,'uv',2),v) for n,v in old['uv'].items()):raise RuntimeError('Eye UV changed')
        if _api('production_uv_r08')['_pose_contract']()!=pose or _sha(source)!=source_sha:raise RuntimeError('Rig pose or source file changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(Path(__file__)),
            key=KEY,keyValue=1.,headTopologyUnchanged=True,existingKeysUnchanged=list(before_keys),
            actualBefore=visible_before,actualAfter=visible_after,actualFibreBindingValidation=fibre,
            originalActiveShapeKeyIndex=active,newKeySha256=skin['_ah'](_array(key.data)),
            eyeBackups={name:{'data':old['data'].name,'fakeUser':old['fake'],'candidateData':bpy.data.objects[name].data.name} for name,old in backups.items()},
            savedBlend=False,limits=['Static proportion candidate only. Review front, quarter, full side and eye macro before adopting.',
                'The visible globe width is not a clinical or calibrated canthus measurement.',
                'Static upper-lid coverage remains at 1. Blink and smiling-lid combinations must be rebuilt/verified for this changed eye geometry.',
                'Turning off only this head key leaves larger eye meshes; use restore() to undo the complete candidate.'])
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            for name,old in backups.items():bpy.data.objects[name].data=old['data'];old['data'].use_fake_user=old['fake']
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[HEAD];key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['newKeySha256']:raise RuntimeError('Retain edited proportion key before restoring')
    for name,old in report['eyeBackups'].items():
        obj=bpy.data.objects[name];obj.data.use_fake_user=True;obj.data=bpy.data.meshes[old['data']];obj.data.use_fake_user=old['fakeUser']
    head.shape_key_remove(key);head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'restoredCompleteEyeAndLidCandidate':True,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    root=(project_path()/'work/reconstruction-r02').resolve();source=(project_path()/source_relative).resolve();output=(project_path()/output_relative).resolve()
    for path in (source,output):
        path.relative_to(root)
        if path.suffix!='.blend':raise ValueError('Explicit independent Blend paths required')
    report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists() or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh source and new independent output required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent eye proportion save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_EYE_PROPORTION_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
