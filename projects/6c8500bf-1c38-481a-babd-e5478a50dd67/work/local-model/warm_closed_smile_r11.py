"""A visibly stronger closed smile, retaining all existing R11 expressions.

The real rendered natural smile merely flattened the downturned neutral line.
This alternative raises the continuous lip/corner field about 3.8 mm at the
corners, coordinates approximately 2 mm cheek rise, and copies the already
installed lower-lid component with independent same-side blink attenuation.
No neutral geometry, existing key, globe, material or closure key is edited.
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
HEAD='R11 Female head study'
KEY='R11 warm closed smile'
LID_KEYS={s:'R11 warm smile lower lid '+s for s in ('L','R')}
OLD_LID_KEYS={s:'R11 smile lower lid '+s for s in ('L','R')}
OLD_SMILE='R11 natural soft smile';MOUTH='R11 gentle smile'
MARKER='R11_warm_closed_smile_candidate'
PARAMETERS={'mouthWorldAxisScales':[1.10,1.0,2.60],'cheekComponentScale':2.1052631578947367,
    'lowerLidComponentScale':1.0,'maximumTotalWorldMeters':.0046,
    'intendedCornerRiseMeters':.003848,'intendedCheekRiseMeters':.002}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _array(items):
    out=np.empty((len(items),3),np.float32);items.foreach_get('co',out.ravel());return out


def plan(current,gentle,natural_main,lids,T):
    """Explicit decomposition of real installed fields; no whole-face scaling."""
    skin=_api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']()
    current=np.asarray(current,np.float32);gentle=np.asarray(gentle,np.float32);natural_main=np.asarray(natural_main,np.float32)
    T=np.asarray(T,float);linear=T[:3,:3];inverse=np.linalg.inv(linear)
    if any(a.shape!=(10582,3) for a in (current,gentle,natural_main)):
        raise RuntimeError('Actual complete installed R11 fields required')
    scale=np.linalg.norm(linear[:,0])
    if not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):
        raise RuntimeError('Uniform actual head frame required')
    world=current@linear.T+T[:3,3]
    mouth_world=gentle@linear.T
    cheek_world=(natural_main-gentle).astype(float)@linear.T
    # Float32 shape coordinates can leave <0.2um subtraction residue on points
    # whose intended cheek delta is zero; retain exact unchanged support there.
    cheek_world[np.linalg.norm(cheek_world,axis=1)<3e-7]=0.
    mouth=mouth_world*np.asarray(PARAMETERS['mouthWorldAxisScales'])
    cheek=cheek_world*PARAMETERS['cheekComponentScale']
    main=((mouth+cheek)@inverse.T).astype(np.float32)
    lid_delta={side:np.asarray(lids[side],np.float32).copy() for side in ('L','R')}
    if any(a.shape!=current.shape for a in lid_delta.values()):raise RuntimeError('Actual split lower-lid fields required')
    total=main+lid_delta['L']+lid_delta['R'];motion=total.astype(float)@linear.T
    active=np.linalg.norm(total,axis=1)>1e-9
    if not 500<int(active.sum())<1800 or np.linalg.norm(motion,axis=1).max()>PARAMETERS['maximumTotalWorldMeters']:
        raise RuntimeError('Unexpected coordinated warm-smile support or amplitude')
    edges=donor['edges'];old=current[edges[:,1]].astype(float)-current[edges[:,0]]
    new=old+total[edges[:,1]]-total[edges[:,0]];selected_edges=active[edges].any(1)
    ratios=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
    tri=_api('lower_face_proportion_r11')['_triangles'](donor['corner_vertices'],donor['face_starts'],donor['face_lengths'])
    tri=tri[active[tri].any(1)];p=current[tri].astype(float);q=(current+total)[tri].astype(float)
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1);valid=a0>1e-14
    align=np.sum(n0[valid]*n1[valid],axis=1)/(a0[valid]*a1[valid]);area=a1[valid]/a0[valid]
    if align.min()<=0 or ratios[selected_edges].min()<.65 or ratios[selected_edges].max()>1.5:
        raise RuntimeError('Warm smile raw local field folds or strains excessively: '+str([align.min(),ratios[selected_edges].min(),ratios[selected_edges].max()]))
    mouth_ring=np.asarray(land['mouth']['openingRimCandidate']['sourceVertexIndices'],np.int32)
    corners=[int(mouth_ring[np.argmin(world[mouth_ring,0])]),int(mouth_ring[np.argmax(world[mouth_ring,0])])]
    # All ocular motion is the unchanged existing lower-lid component. New
    # amplitude enters the established perioral and cheek support only.
    ocular=skin['_vertices_of_groups'](donor,{2,3,4,5})
    if np.max(np.linalg.norm(main[ocular],axis=1))>4e-7:
        raise RuntimeError('Stronger mouth/cheek field entered ocular/ear controls')
    report={'parameters':PARAMETERS,'provenance':provenance,'key':KEY,'cornerCageIndices':corners,
        'actualCornerTargetDisplacementsMeters':motion[corners].tolist(),
        'maximumFullWorldDisplacementMeters':float(np.linalg.norm(motion,axis=1).max()),
        'maximumCheekComponentRiseMeters':float(cheek[:,2].max()),'affectedCageVertices':int(active.sum()),
        'rawEdgeRatioQuantiles':np.quantile(ratios[selected_edges],[0,.05,.5,.95,1]).tolist(),
        'rawTriangleMinimumNormalAlignment':float(align.min()),'rawTriangleAreaRatioRange':[float(area.min()),float(area.max())],
        'lowerLidFieldsCopiedExactly':True,'neutralGeometryChanged':False,'appearanceAccepted':False,
        'scope':'Alternative expression, not additive to the old natural or gentle smile. Preserve both lip sheets in the continuous corner field; amplify cheek support, retain central lip volume and original lower-lid field.',
        'requiredReview':['Same current-light front and quarter at 0, 0.5 and 1: corners must form a shallow continuous smile without hooks or a long flat seam.',
            'Check cheek volume/nasal transition and lower-lid fullness: no hard nasolabial groove or eye bag.',
            'Actual oral contact at 0/0.5/1 including pre-existing inner-upper/lower-outer overlap must be inspected and corrected by the oral pass.',
            'Actual half/full blink combinations remain required; unchanged lower-lid delta is not combination acceptance.']}
    return main,lid_delta,report


def _driver(block,keys,side):
    curve=block.driver_add('value');driver=curve.driver;driver.type='SCRIPTED'
    driver.expression='min(max(smile,0),1)*(1-min(max(blink,0),1))'
    for name,key in (('smile',KEY),('blink','R11 blink '+side)):
        variable=driver.variables.new();variable.name=name;variable.type='SINGLE_PROP'
        target=variable.targets[0];target.id_type='KEY';target.id=keys;target.data_path=keys.key_blocks[key].path_from_id('value')
    return curve.data_path


def apply():
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or MARKER in bpy.context.scene:
        raise RuntimeError('Fresh independent background candidate required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    required=(OLD_SMILE,MOUTH,*OLD_LID_KEYS.values(),'R11 blink L','R11 blink R')
    if not keys or any(name not in keys.key_blocks for name in required) or any(name in keys.key_blocks for name in (KEY,*LID_KEYS.values())):
        raise RuntimeError('Actual installed natural smile and split blink controls required')
    if head.library or mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582:raise RuntimeError('Independent complete R11 head required')
    if any(abs(block.value)>1e-7 for block in keys.key_blocks if any(word in block.name.lower() for word in ('smile','blink'))):
        raise RuntimeError('Keep existing smiles/blinks zero; retain all static face keys')
    natural=_api('natural_smile_adult_r11');uv=_api('production_uv_r08');skin=_api('natural_skin_adult_r11');face=_api('runtime_face_refresh_r08')
    binding=_api('face_fibre_binding_r11');gentle=_api('gentle_smile_adult_r11');qa=_api('face_contact_audit_r11')
    saved=json.loads(bpy.context.scene['R11_natural_soft_smile_candidate'])
    if skin['_ah'](_array(keys.key_blocks[OLD_SMILE].data))!=saved['newKeyCoordinatesSha256']:
        raise RuntimeError('Existing natural smile changed; inspect before component reuse')
    for side,name in OLD_LID_KEYS.items():
        if skin['_ah'](_array(keys.key_blocks[name].data))!=saved['internalLowerLidKeyHashes'][side]:raise RuntimeError('Existing smile-lid field changed')
    baseline=_array(keys.reference_key.data)
    for name in required[:4]:
        block=keys.key_blocks[name]
        if block.relative_key!=keys.reference_key or block.vertex_group or block.mute:raise RuntimeError('Unexpected retained key semantics: '+name)
    bpy.context.view_layer.update();prior=natural['_key_records'](mesh,uv);nonkey=natural['_nonkey_state'](head,uv,face)
    external=gentle['_external_identity'](bpy,head);source_sha=_sha(source);active_index=head.active_shape_key_index;created=[];own=[];success=False
    try:
        with qa['rest_evaluation']():
            current,_=uv['current_mix'](head);T=np.asarray(head.matrix_world,float)
            main,lids,report=plan(current,_array(keys.key_blocks[MOUTH].data)-baseline,
                _array(keys.key_blocks[OLD_SMILE].data)-baseline,
                {s:_array(keys.key_blocks[n].data)-baseline for s,n in OLD_LID_KEYS.items()},T)
            before=binding['_head_mesh'](head)
            for name,delta in [(KEY,main),*[(LID_KEYS[s],lids[s]) for s in ('L','R')]]:
                block=head.shape_key_add(name=name,from_mix=False);created.append(name)
                block.relative_key=keys.reference_key;block.slider_min=0.;block.slider_max=1.;block.value=0.
                block.data.foreach_set('co',(baseline+delta).ravel())
                if name!=KEY:own.append(_driver(block,keys,next(s for s,n in LID_KEYS.items() if n==name)))
            head.active_shape_key_index=active_index;bpy.context.view_layer.update();after=binding['_head_mesh'](head)
            if before['topologySha256']!=after['topologySha256'] or not np.array_equal(before['co'],after['co']):raise RuntimeError('Inactive warm smile changed actual neutral geometry')
            neutral_binding=binding['validate'](neutral=False)
        now=natural['_key_records'](mesh,uv)
        for name in created:now['blocks'].pop(name)
        records=[row for row in now['animation']['drivers'] if row['path'] in own]
        now['animation']=natural['_without_own_drivers'](now['animation'],own,prior['animation'])
        if now!=prior or natural['_hash'](natural['_nonkey_state'](head,uv,face))!=natural['_hash'](nonkey):
            raise RuntimeError('Existing key/driver or original non-key head changed')
        if gentle['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('External objects/rig/source changed')
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
            newKeyCoordinatesSha256=skin['_ah'](_array(keys.key_blocks[KEY].data)),keyValue=0.,
            internalLowerLidKeys=LID_KEYS,internalKeyHashes={s:skin['_ah'](_array(keys.key_blocks[n].data)) for s,n in LID_KEYS.items()},
            internalDriverPaths=own,internalDriverContracts=natural['_portable_contract'](records),
            existingKeysPreserved=list(prior['blocks']),actualInactiveGeometryDifferenceMeters=0.,actualInactiveFibreValidation=neutral_binding,
            originalActiveShapeKeyIndex=active_index,technicalAcceptance=False,savedBlend=False,
            alternativeControlsMustRemainZero=[MOUTH,OLD_SMILE])
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            for name in reversed(created):
                if mesh.shape_keys and name in mesh.shape_keys.key_blocks:
                    block=mesh.shape_keys.key_blocks[name]
                    if name!=KEY:block.driver_remove('value')
                    head.shape_key_remove(block)
            head.active_shape_key_index=active_index;bpy.context.view_layer.update()


def prepare(source_relative,output_relative):
    import bpy
    natural=_api('natural_smile_adult_r11');source=natural['_path'](source_relative);output=natural['_path'](output_relative)
    record=output.with_suffix('.warm-smile.json')
    if output.exists() or record.exists() or source==output or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh actual source and independent output required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Warm-smile candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),savedBlend=True)
    with record.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print('R11_WARM_SMILE_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256'],'key':KEY,'value':0.}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
