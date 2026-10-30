"""Install limited upper-lid outer-crown relief from the current real surface.

No old observation is a prerequisite. The current neutral head and actual
globes are measured inside this independent background run. Actual inner-facing
and near-globe tissue, 0.8mm space around the edge, and corners are protected.
A native field retains at least0.7mm actual globe distance and becomes exactly
zero at blink>=0.8, after which the separate late-corner system is authoritative.
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
MARKER='R11_upper_lid_outer_relief_candidate'
CONTROL='R11 upper lid relief'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def path(relative,suffix):
    result=(project_path()/relative).resolve();result.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if result.suffix!=suffix:raise RuntimeError('Independent reconstruction path required')
    return result


def external_contract(bpy,head,owned=None):
    rows=[]
    for obj in bpy.context.scene.objects:
        if obj==head or obj==owned:continue
        row={'name':obj.name,'object':obj.as_pointer(),'data':getattr(getattr(obj,'data',None),'as_pointer',lambda:None)(),
            'parent':obj.parent.as_pointer() if obj.parent else None,'world':[list(r) for r in obj.matrix_world],
            'modifiers':[(m.name,m.type) for m in obj.modifiers],'materials':[s.material.as_pointer() if s.material else None for s in obj.material_slots]}
        if obj.type=='ARMATURE':row['pose']=[(b.name,[list(r) for r in b.matrix_basis]) for b in obj.pose.bones]
        rows.append(row)
    return {'objects':rows,'frame':bpy.context.scene.frame_current,'subframe':bpy.context.scene.frame_subframe,
        'actions':[(a.name,a.as_pointer()) for a in bpy.data.actions]}


def polygon_contract(head):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();evaluated=head.evaluated_get(dg)
    mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        h=hashlib.sha256()
        for data,field in ((mesh.loops,'vertex_index'),(mesh.polygons,'loop_start'),(mesh.polygons,'loop_total')):
            array=np.empty(len(data),np.int32);data.foreach_get(field,array);h.update(array.tobytes())
        return {'vertices':len(mesh.vertices),'loops':len(mesh.loops),'polygons':len(mesh.polygons),'connectionSha256':h.hexdigest()}
    finally:evaluated.to_mesh_clear()


def modifier_contract(head,uv,excluded=None):
    rows=[]
    for item in head.modifiers:
        if item==excluded:continue
        state=uv['_scalar_rna'](item)
        state.pop('is_active',None)  # The active stack UI selection is not deformation state.
        group=getattr(item,'node_group',None)
        rows.append((item.name,item.type,group.as_pointer() if group else None,state))
    return rows


def actual_check(before,after,weights,eyes,surface,blink,maximum):
    if before['fullCounts']!=after['fullCounts'] or not np.array_equal(before['originalVertexIds'],after['originalVertexIds']):
        raise RuntimeError('Relief changed evaluated topology/ROI order')
    delta=after['world']-before['world'];amount=np.linalg.norm(delta,axis=1)
    selected=(weights['L']>1e-8)|(weights['R']>1e-8)
    outside=float(amount[~selected].max())
    if outside>2e-6:raise RuntimeError('Actual protected/outside crown surface changed: '+str(outside))
    if blink>=.8 and not np.array_equal(before['fullWorld'],after['fullWorld']):raise RuntimeError('Relief failed to preserve all actual late-blink vertices exactly')
    if amount.max()>maximum+2e-6:raise RuntimeError('Actual relief exceeds its physical limit')
    report={'blink':blink,'maximumActualDisplacementMeters':float(amount.max()),'outsideSelectedMaximumMeters':outside,
        'lateBlinkFullHeadVerticesExact':bool(np.array_equal(before['fullWorld'],after['fullWorld'])) if blink>=.8 else None,
        'retessellatedTriangleRows':int(np.count_nonzero((before['triangles']!=after['triangles']).any(1))),'eyes':{}}
    active=amount>2e-7
    for side in ('L','R'):
        ids=np.flatnonzero((weights[side]>1e-8)&active)
        if not len(ids):
            report['eyes'][side]={'changedAbove0_2Micrometers':0};continue
        e=eyes[side]
        old=surface['closest_surface'](before['world'][ids],e['world'],e['triangles'])
        new=surface['closest_surface'](after['world'][ids],e['world'],e['triangles'])
        if np.any(new['signed']<=0) or new['distance'].min()<.0007-3e-6:
            raise RuntimeError('Actual crown reduction consumed globe protection: '+side)
        report['eyes'][side]={'changedAbove0_2Micrometers':len(ids),'minimumActualGlobeDistanceBeforeMeters':float(old['distance'].min()),
            'minimumActualGlobeDistanceAfterMeters':float(new['distance'].min())}
    if active.any():
        tri=before['triangles'][active[before['triangles']].any(1)];p=before['world'][tri];q=after['world'][tri]
        n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);a0=np.linalg.norm(n0,axis=1);a1=np.linalg.norm(n1,axis=1);valid=a0>1e-14
        align=np.sum(n0[valid]*n1[valid],axis=1)/(a0[valid]*a1[valid]);ratio=a1[valid]/a0[valid]
        report['minimumActualNormalAlignment']=float(align.min());report['actualTriangleAreaRatioRange']=[float(ratio.min()),float(ratio.max())]
        if align.min()<=.5 or ratio.min()<.45 or ratio.max()>1.7:raise RuntimeError('Actual upper-lid relief has a fold or excessive strain')
    return report


def prepare(source_relative,output_relative,maximum=.00030):
    import bpy
    source=path(source_relative,'.blend');output=path(output_relative,'.blend')
    report_path=output.with_suffix('.upper-lid-relief.json');attempt_path=output.with_suffix('.upper-lid-relief-attempt.json');array_path=output.with_suffix('.upper-lid-relief.npz')
    if source==output or any(p.exists() for p in (output,report_path,attempt_path,array_path)) or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh current source and explicit new independent output required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys
    if MARKER in bpy.context.scene or CONTROL in head:raise RuntimeError('Retain or explicitly restore the existing relief candidate')
    if 'R11_blink_corner_surface_refinement' not in bpy.context.scene:raise RuntimeError('Install after the current actual late-corner refinement')
    if any(abs(k.value)>1e-7 for k in keys.key_blocks if 'blink' in k.name.lower() or 'smile' in k.name.lower()):raise RuntimeError('Current neutral expression required')
    capture=api('adult_blink_recompute_r11');surface=api('upper_lid_relief_surface_r11');qa=api('face_contact_audit_r11')
    natural=api('natural_smile_adult_r11');uv=api('production_uv_r08');face=api('runtime_face_refresh_r08');binding=api('face_fibre_binding_r11');groom=api('posterior_groom_r09')
    before_keys=natural['_key_records'](head.data,uv);before_data=groom['_data_hash'](head.data)
    before_materials=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    before_modifiers=modifier_contract(head,uv)
    before_external=external_contract(bpy,head);source_sha=sha(source);original_active=head.active_shape_key_index
    modifier=group=cache=mesh=None;success=False;observed={};report={}
    def values(value):
        for side in ('L','R'):keys.key_blocks['R11 blink '+side].value=value
        bpy.context.view_layer.update()
    try:
        with qa['rest_evaluation']():
            print('R11_UPPER_RELIEF_CAPTURE_CURRENT_NEUTRAL',flush=True)
            values(0.);baseline={0.:capture['_snapshot'](head)};topology=polygon_contract(head)
            eyes={s:capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+s]) for s in ('L','R')}
            eye_info={s:capture['_measure'](baseline[0.],eyes[s]) for s in ('L','R')}
            actual=qa['_evaluated_mesh'](head);faces=np.asarray(actual['faces'],np.int32);regions=np.asarray(actual['groups'])
            if len(actual['world'])!=baseline[0.]['fullCounts'][0] or np.max(abs(np.asarray(actual['world'])-baseline[0.]['fullWorld']))>2e-7:
                raise RuntimeError('Current topology/region capture is inconsistent')
            # Face sets2/3 include eyelid surfaces; freezing their whole tagged
            # box would also freeze the external crown. Select physical outer
            # tissue using orientation, true globe distance and the 3D edge.
            protected=np.ones(len(actual['world']),bool)
            allowed=np.unique(faces[np.isin(regions,[2,3,17])]);protected[allowed]=False
            protected[np.unique(faces[np.isin(regions,[4,5,7,8])])]=True
            roi_protected=protected[baseline[0.]['originalVertexIds']]
            weights,arrays,plan=surface['plan'](baseline[0.]['world'],baseline[0.]['triangles'],eye_info,eyes,roi_protected,maximum)
            arrays.update(originalVertexIds=baseline[0.]['originalVertexIds'],neutralWorld=baseline[0.]['world'],triangles=baseline[0.]['triangles'],headMatrixWorld=np.asarray(head.matrix_world))
            np.savez_compressed(array_path,**arrays)
            report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperHashes':{name:sha(Path(__file__).with_name(name)) for name in ('upper_lid_relief_install_r11.py','upper_lid_relief_surface_r11.py','upper_lid_relief_nodes_r11.py')},
                'arrays':str(array_path.relative_to(project_path())),'arraysSha256':sha(array_path),'plan':plan,'currentNeutralEyeObservation':eye_info,
                'sourceSaved':False,'technicalAcceptance':False,'appearanceAccepted':False,'actualChecks':[],
                'limits':['Front-visible edge is a protected spatial proxy, not a certified anatomical wet-line trace; inner-facing and near-globe tissue are separately fixed by actual physical-surface tests.',
                    'Current neutral and four blink snapshots are local evidence, not all animation/contact certification.',
                    'Real same-light neutral/half-blink/closed eye macros and warm smile must be inspected; no visual acceptance from a successful save.']}
            attempt_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            for blink in (.4,.8,.9,1.):
                values(blink);baseline[blink]=capture['_snapshot'](head)
                if polygon_contract(head)!=topology:raise RuntimeError('Existing blink changes polygon/loop connectivity')
            values(0.)
            head[CONTROL]=1.;head.id_properties_ui(CONTROL).update(min=0.,max=1.,description='Limited outer upper-lid relief. Automatically fades to zero by blink 0.8.')
            matrix=np.asarray(head.matrix_world,float);scale=float(np.linalg.norm(matrix[:3,0]))
            if not np.allclose(matrix[:3,:3],np.eye(3)*scale,atol=2e-7):raise RuntimeError('Current upright uniform REST head frame required')
            print('R11_UPPER_RELIEF_INSTALL_NATIVE_FIELD',flush=True)
            modifier,group,cache,mesh=api('upper_lid_relief_nodes_r11')['build'](head,baseline[0.]['fullCounts'][0],baseline[0.]['originalVertexIds'],weights,scale,maximum=maximum,protected_gap=.0007,fade_end=.8)
            for blink in (0.,.4,.8,.9,1.):
                values(blink);now=capture['_snapshot'](head)
                if polygon_contract(head)!=topology:raise RuntimeError('Relief changed actual polygon/loop connectivity')
                result=actual_check(baseline[blink],now,weights,eyes,surface,blink,maximum)
                result['actualPolygonLoopConnectionExact']=True
                if blink==0.:
                    changed_info={s:capture['_measure'](now,eyes[s]) for s in ('L','R')}
                    result['neutralVisibleApertureAreaChangeSquareMeters']={s:changed_info[s]['frontVisibleAreaSquareMeters']-eye_info[s]['frontVisibleAreaSquareMeters'] for s in ('L','R')}
                    if any(abs(x)>1e-7 for x in result['neutralVisibleApertureAreaChangeSquareMeters'].values()):raise RuntimeError('Outer-crown relief changed neutral globe aperture')
                result['actualFibreBindingValidation']=binding['validate'](neutral=False)
                report['actualChecks'].append(result);attempt_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print('R11_UPPER_RELIEF_ACTUAL_SAMPLE '+json.dumps({'blink':blink,'maximumMeters':result['maximumActualDisplacementMeters']}),flush=True)
            values(0.);head.active_shape_key_index=original_active
        if natural['_key_records'](head.data,uv)!=before_keys or groom['_data_hash'](head.data)!=before_data:
            raise RuntimeError('Original keys/drivers or raw head data changed')
        if [face['_material_contract'](s.material) if s.material else None for s in head.material_slots]!=before_materials:
            raise RuntimeError('Original skin/lid materials changed')
        if modifier_contract(head,uv,modifier)!=before_modifiers:
            raise RuntimeError('Existing modifier state/order changed')
        if external_contract(bpy,head,cache)!=before_external or sha(source)!=source_sha:raise RuntimeError('External objects/pose/source changed')
        report.update(control=CONTROL,controlValue=1.,modifier=modifier.name,nodeGroup=group.name,lookupObject=cache.name,lookupMesh=mesh.name,
            neutralActualCrownRecomputedFromCurrentSource=True,originalKeysDriversRawDataMaterialsPreserved=True,lateBlinkCompletelyExcluded=True,polygonContract=topology,
            modifierOrder=[m.name for m in head.modifiers],candidateSaved=False)
        bpy.context.scene[MARKER]=json.dumps(report)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent upper-lid relief save failed')
        report.update(output=str(output.relative_to(project_path())),outputSha256=sha(output),candidateSaved=True)
        with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        success=True
        print('R11_UPPER_RELIEF_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
        return report
    except Exception as exc:
        report.update(sourceSaved=False,candidateSaved=False,guardError=str(exc))
        if report.get('arrays'):attempt_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    finally:
        if not success:
            values(0.);head.active_shape_key_index=original_active
            if modifier is not None:head.modifiers.remove(modifier)
            if CONTROL in head:del head[CONTROL]
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
            if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:bpy.data.node_groups.remove(group)
            bpy.context.view_layer.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--maximum',type=float,default=.00030)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output,args.maximum)
