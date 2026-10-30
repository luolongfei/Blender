"""One fresh-source whole-face diagnostic of a neutral corner release.

No Blend save and no warm-expression sweep. Actual neutral closure, triangle
crossings and continuity must pass before rendering. This file is a preview
entry point; the independently frozen surface plan owns the shape design.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

BASE=runpy.run_path(str(Path(__file__).with_name('chin_throat_contour_preview_r11.py')))
api,sha,array,external_geometry=(BASE[k] for k in ['api','sha','array','external_geometry'])
project_path=BASE['project_path'];HEAD=BASE['HEAD']
KEY='R11 relaxed mouth corners'
REFERENCE='previews/reconstruction-r02/natural-rest-overall-r11/render-report.json'
PLAN='work/reconstruction-r02/relaxed-rest-mouth-corners-plan-r11.json'
PLAN_SHA='929509d98703f4c23689c7271714e5e79df8a369ab8b4a34a0b401c702b6084a'
PLAN_ARRAYS_SHA='eeeba21987e9e39839d1120ca5ec2606c407bd04615bd7d42fe7e4af988d8c36'


def snapshot(head):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();ev=head.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        sizes=array(mesh.polygons,'loop_total',1,np.int32).ravel()
        if not np.all(sizes==4):raise RuntimeError('Complete actual quad topology required')
        matrix=np.asarray(ev.matrix_world,float)
        world=array(mesh.vertices,'co')@matrix[:3,:3].T+matrix[:3,3]
        faces=array(mesh.loops,'vertex_index',1,np.int32).ravel().reshape(-1,4)
        group=mesh.attributes.get('.sculpt_face_set')
        if group is None or group.domain!='FACE' or group.data_type!='INT':raise RuntimeError('Actual FACE regions required')
        groups=array(group.data,'value',1,np.int32).ravel()
        mesh.calc_loop_triangles()
        return {'world':world,'faces':faces,'groups':groups,'matrix':matrix,
            'triangles':array(mesh.loop_triangles,'vertices',3,np.int32),
            'trianglePolygonIds':array(mesh.loop_triangles,'polygon_index',1,np.int32).ravel()}
    finally:ev.to_mesh_clear()


def preview(source_relative,output_relative,samples=24):
    import bpy
    from mathutils import Matrix
    p=project_path();source=(p/source_relative).resolve();out=(p/output_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve());out.relative_to((p/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or source.suffix!='.blend':raise RuntimeError('Fresh saved independent source required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve previous diagnostic output')
    if not 8<=samples<=32:raise ValueError('Bounded diagnostic samples required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks;scene=bpy.context.scene
    if KEY in keys or (scene.frame_current,scene.frame_subframe)!=(1,0.):raise RuntimeError('Fresh frame-one source without corner candidate required')
    bpy.context.view_layer.update()
    for name in ['R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile','R11 warm closed smile','R11 jaw open']:
        if name in keys and float(keys[name].value)!=0.:raise RuntimeError('Neutral expression required; do not increase warm smile to hide the resting corner shape')
    if 'R11 relaxed lip seal' not in keys or float(keys['R11 relaxed lip seal'].value)!=1.:raise RuntimeError('Current central relaxed seal must remain enabled')
    original_keys={k.name:float(k.value) for k in keys}
    if float(head.get('R11 lower face continuity',0.))!=0.:raise RuntimeError('Rejected original continuity must remain disabled')
    refpath=p/REFERENCE;ref=json.loads(refpath.read_text(encoding='utf8'));front=next(f for f in ref['frames'] if f['view']=='front')
    plan_path=p/PLAN
    if sha(plan_path)!=PLAN_SHA:raise RuntimeError('Inspect changed frozen corner plan')
    plan=json.loads(plan_path.read_text(encoding='utf8'));plan_arrays=p/plan['arrays']
    if sha(plan_arrays)!=PLAN_ARRAYS_SHA or plan['arraysSha256']!=PLAN_ARRAYS_SHA or plan['remainingCrossingPairs'] or not plan['centralExact']:raise RuntimeError('Frozen neutral corner field is incomplete or changed')
    with np.load(plan_arrays) as npz:frozen={k:npz[k].copy() for k in npz.files}
    detail=api('final_detail_review_r09');uv=api('production_uv_r08');states=api('chin_throat_contour_preview_r11_v2')
    lights=detail['_lights'](scene);world=detail['_world'](scene)
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    if lights!=ref['lights'] or world!=ref['world'] or colour!=ref['colourManagement']:raise RuntimeError('Current illumination differs from the pinned whole-face review')
    source_sha=sha(source);saved_state=states['state'](head,uv,excluded_key=KEY)
    old_camera=scene.camera;old_active=head.active_shape_key_index
    modifier=group=cache=mesh=key=camera=None
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'referenceReport':REFERENCE,'referenceReportSha256':sha(refpath),'sourceSaved':False,
        'frozenPlan':PLAN,'frozenPlanSha256':PLAN_SHA,'frozenPlanArraysSha256':PLAN_ARRAYS_SHA,
        'diagnosticPreview':True,'technicalAcceptance':False,'appearanceAccepted':False,
        'scope':'Current neutral geometry only, one whole-face image. Existing central seal and all original controls stay fixed. No warm/jaw/motion acceptance.',
        'savedState':saved_state,'lights':lights,'world':world,'colourManagement':colour,
        'preservedKeyValues':original_keys,'actualChecks':{},'guardErrors':[],
        'helperHashes':{n:sha(Path(__file__).with_name(n)) for n in ['relaxed_mouth_corner_preview_r11.py','relaxed_mouth_corners_plan_r11.py','relaxed_mouth_corner_nodes_r11.py']}}
    data={};out.mkdir(parents=True,exist_ok=True);payload=out/'actual-diagnostic.npz';record=out/'render-report.json'
    def persist():
        np.savez_compressed(payload,**data);report.update(arrays=str(payload.relative_to(p)),arraysSha256=sha(payload))
        record.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    def value(v):keys[KEY].value=v;head.update_tag();bpy.context.view_layer.update()
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            before=snapshot(head);data.update({('before'+k[0].upper()+k[1:]):v for k,v in before.items()})
            dependency=frozen['dependencyOriginalVertexIds']
            if not np.array_equal(before['faces'],frozen['faces']) or not np.array_equal(before['groups'],frozen['groups']):raise RuntimeError('Current actual polygon/anatomical correspondence differs from frozen plan')
            if not np.array_equal(before['world'][dependency],frozen['fullWorldBefore'][dependency]):raise RuntimeError('Current actual corner support/neighborhood differs from frozen shape; disjoint hair/eye/neck changes remain allowed')
            offset=frozen['offsets'];arrays={k:frozen[k] for k in ['offsets','commonOffsets','contactOffsets','dependencyOriginalVertexIds','protectedCentral','contactRegion','upperExclusive','lowerExclusive']}
            active=np.any(offset!=0,axis=1);ids=np.flatnonzero(active).astype(np.int32)
            data.update(arrays);data['fullWorldOffsets']=offset;report['plan']=plan
            report['actualChecks']['frozenDependencyExact']=True;report['actualChecks']['dependencyVertices']=len(dependency);persist()
            if np.asarray(offset).shape!=before['world'].shape or not np.isfinite(offset).all():raise RuntimeError('Finite complete actual offset field required')
            ext_before=external_geometry({head})
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key
            key.data.foreach_set('co',array(head.data.shape_keys.reference_key.data,'co',3,np.float32).ravel());key.slider_min=0.;key.slider_max=1.;key.value=0.
            modifier,group,cache,mesh=api('relaxed_mouth_corner_nodes_r11')['build'](head,len(offset),ids,offset[ids],before['matrix'])
            value(0.);zero=snapshot(head);data['disabledWorld']=zero['world']
            if not np.array_equal(before['world'],zero['world']) or not np.array_equal(before['faces'],zero['faces']):raise RuntimeError('Zero corner control does not return exact current head')
            value(1.);after=snapshot(head);data.update({('after'+k[0].upper()+k[1:]):v for k,v in after.items()});persist()
            delta=after['world']-before['world']
            topology=all(np.array_equal(before[k],after[k]) for k in ['faces','groups'])
            protected={
                'centralSealAndCentralLipChin':np.flatnonzero(abs(before['world'][:,0])<=.010),
                'eyesEars':np.unique(before['faces'][np.isin(before['groups'],[2,3,4,5])]),
                'lowNeck':np.flatnonzero(before['world'][:,2]<1.335),
                'outsideContinuousSupport':np.flatnonzero(~active)}
            protected_report={n:{'vertices':len(v),'exact':bool(np.array_equal(before['world'][v],after['world'][v])),'maximumMeters':float(np.linalg.norm(delta[v],axis=1).max()) if len(v) else 0.} for n,v in protected.items()}
            quality=api('upper_lid_roll_install_r11')['quality'](before['world'],after['world'],after['triangles'],active)
            report['actualChecks'].update(zeroExact=True,topologyExact=topology,protected=protected_report,
                expectedMaximumErrorMeters=float(np.linalg.norm(delta-offset,axis=1).max()),maximumActualMeters=float(np.linalg.norm(delta,axis=1).max()),quality=quality)
            persist()
            if not topology or any(not x['exact'] for x in protected_report.values()) or report['actualChecks']['expectedMaximumErrorMeters']>3e-7:raise RuntimeError('Actual neutral corner field differs or moves protected anatomy')
            if quality['nonpositiveNormals'] or quality['minimumNormalAlignment']<.75 or quality['areaRatioRange'][0]<.50 or quality['areaRatioRange'][1]>1.8:raise RuntimeError('Actual corner field folds or compresses the continuous support excessively')
            cross=api('adult_blink_midpath_contact_r11')['changed_intersections']
            old_cross=cross(before['world'],after['triangles'],active);new_cross=cross(after['world'],after['triangles'],active)
            report['actualChecks'].update(oldLocalCrossings=len(old_cross),newLocalCrossings=len(new_cross),newlyAddedCrossings=len(new_cross-old_cross))
            # This candidate is intended to clear the real corner defect, not
            # merely translate it. Any remaining active-domain crossing fails.
            ext_after=external_geometry({head,cache});extdiff=states['exact_diff'](ext_before,ext_after)
            report['actualChecks']['externalGeometryDiff']=extdiff;persist()
            if new_cross or extdiff:raise RuntimeError('Actual corner intersection remains or external geometry moved')
        pre=states['state'](head,uv,excluded_key=KEY);report['savedToPreviewStateDiff']=states['exact_diff'](saved_state,pre);persist()
        if report['savedToPreviewStateDiff']:raise RuntimeError('Original controls/pose changed before whole-face preview')
        camera=bpy.data.objects.new('R11 relaxed corner whole-face review',bpy.data.cameras.new('R11 relaxed corner whole-face review'))
        scene.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.ortho_scale=front['camera']['orthographicScale'];camera.data.dof.use_dof=False
        camera.matrix_world=Matrix(front['camera']['matrixWorld']);scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.resolution_x,scene.render.resolution_y=map(int,ref['resolution']);scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
        bpy.context.view_layer.update();pre=states['state'](head,uv,excluded_key=KEY);pre_head=snapshot(head);pre_ext=external_geometry({head,cache})
        report['preRenderState']=pre;report['savedToPreRenderDiff']=states['exact_diff'](saved_state,pre)
        data['preRenderPosedWorld']=pre_head['world'];persist()
        if report['savedToPreRenderDiff']:raise RuntimeError('Original controls/pose changed before rendering')
        image=out/'front-1.png';scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        if not image.is_file():raise RuntimeError('Actual whole-face candidate image missing')
        report.update(image=str(image.relative_to(p)),imageSha256=sha(image),cameraMatrixWorld=[list(r) for r in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world],resolution=ref['resolution'],samples=samples)
        report['postRenderBeforeUpdateState']=states['state'](head,uv,excluded_key=KEY);persist()
        bpy.context.view_layer.update();post=states['state'](head,uv,excluded_key=KEY);post_head=snapshot(head);post_ext=external_geometry({head,cache})
        report['postRenderState']=post;report['authorDiff']=states['exact_diff'](pre['author'],post['author']);report['derivedDiff']=states['exact_diff'](pre['derived'],post['derived'])
        report['externalGeometryDiff']=states['exact_diff'](pre_ext,post_ext);report['postRenderHeadExact']=all(np.array_equal(pre_head[k],post_head[k]) for k in ['world','faces','groups'])
        data['postRenderPosedWorld']=post_head['world'];report['sourceFileExact']=sha(source)==source_sha;persist()
        if report['authorDiff'] or report['externalGeometryDiff'] or not report['postRenderHeadExact'] or not report['sourceFileExact']:raise RuntimeError('Render changed original inputs or actual geometry; precise diagnostics retained')
        report.update(complete=True,localNeutralGeometryChecksPassed=True);persist()
        print('R11_RELAXED_CORNER_PREVIEW '+json.dumps({'image':str(image),'sourceSha256':source_sha}),flush=True)
        return report
    except BaseException as exc:
        report['guardErrors'].append(str(exc));persist();raise
    finally:
        scene.camera=old_camera
        if modifier is not None:head.modifiers.remove(modifier)
        if key is not None:head.shape_key_remove(key)
        if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
        if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
        if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        if camera is not None:
            cd=camera.data;bpy.data.objects.remove(camera,do_unlink=True)
            if cd.users==0:bpy.data.cameras.remove(cd)
        head.active_shape_key_index=old_active;bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=24)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);preview(a.source,a.output,a.samples)
