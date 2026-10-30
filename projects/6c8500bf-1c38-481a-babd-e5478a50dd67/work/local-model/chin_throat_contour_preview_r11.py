"""Read-only three-image diagnostic of one chin / front-throat candidate.

No Blend is saved. Only current neutral REST surface checks are run, then
the saved frame-one pose is restored for the exact quarter and true side A/B.
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
KEY='R11 chin throat contour'
REFERENCE='previews/reconstruction-r02/natural-rest-overall-r11/render-report.json'
OBSERVATION='work/reconstruction-r02/head-neck-current-posture-r11-v2.json'


def api(name): return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def array(items,field,width=3,dtype=np.float64):
    out=np.empty((len(items),width),dtype)
    items.foreach_get(field,out.ravel())
    return out


def digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()


def snapshot(head):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();ev=head.evaluated_get(dg)
    mesh=ev.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    try:
        sizes=array(mesh.polygons,'loop_total',1,np.int32).ravel()
        if not np.all(sizes==4): raise RuntimeError('Actual all-quad fixed-L3 head required')
        faces=array(mesh.loops,'vertex_index',1,np.int32).ravel().reshape(-1,4)
        m=np.asarray(ev.matrix_world,float)
        w=array(mesh.vertices,'co')@m[:3,:3].T+m[:3,3]
        attr=mesh.attributes.get('.sculpt_face_set')
        if attr is None or attr.domain!='FACE' or attr.data_type!='INT': raise RuntimeError('Current actual FACE regions required')
        groups=array(attr.data,'value',1,np.int32).ravel()
        mesh.calc_loop_triangles()
        tri=array(mesh.loop_triangles,'vertices',3,np.int32)
        return {'world':w,'faces':faces,'groups':groups,'triangles':tri,'matrix':m}
    finally: ev.to_mesh_clear()


def external_geometry(excluded):
    """Actual evaluated geometry, including every hair/eyelash curve object."""
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();out={}
    for original in bpy.context.scene.objects:
        if original in excluded: continue
        obj=original.evaluated_get(dg);data=obj.data
        row={'type':obj.type,'matrixWorld':[list(r) for r in obj.matrix_world]}
        if obj.type=='CURVES':
            row.update(positionHash=digest(array(data.position_data,'vector')),pointCount=len(data.position_data))
        elif obj.type=='MESH':
            mesh=obj.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
            try:
                row.update(positionHash=digest(array(mesh.vertices,'co')),pointCount=len(mesh.vertices),
                    loopHash=digest(array(mesh.loops,'vertex_index',1,np.int32)),
                    faceSizesHash=digest(array(mesh.polygons,'loop_total',1,np.int32)))
            finally:obj.to_mesh_clear()
        elif obj.type=='CURVE':
            row['splines']=[]
            for s in data.splines:
                v={'type':s.type,'points':digest(array(s.points,'co',4))}
                if len(s.bezier_points):
                    for field in ['co','handle_left','handle_right']:
                        v[field]=digest(array(s.bezier_points,field))
                row['splines'].append(v)
        else: continue
        out[original.name]=row
    return out


def preview(source_relative,output_relative,amplitude=.004,samples=24):
    import bpy
    from mathutils import Matrix,Vector
    project=project_path();source=(project/source_relative).resolve();out=(project/output_relative).resolve()
    source.relative_to((project/'work/reconstruction-r02').resolve());out.relative_to((project/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or source.suffix!='.blend':
        raise RuntimeError('Fresh background saved source required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve earlier preview output')
    if not 8<=samples<=32: raise ValueError('Bounded diagnostic samples required')
    out.mkdir(parents=True,exist_ok=True)
    ref_path=project/REFERENCE;ref=json.loads(ref_path.read_text(encoding='utf8'))
    observed=json.loads((project/OBSERVATION).read_text(encoding='utf8'))
    observed_npz=project/observed['arrays']
    if not observed['technicalAcceptance'] or sha(observed_npz)!=observed['arraysSha256']:raise RuntimeError('Current exact observation required')
    observation=np.load(observed_npz)
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks;scene=bpy.context.scene
    if KEY in keys:raise RuntimeError('Fresh in-memory candidate required')
    old_frame=(scene.frame_current,scene.frame_subframe);old_camera=scene.camera;old_active=head.active_shape_key_index
    if old_frame!=(1,0.):scene.frame_set(1)
    bpy.context.view_layer.update()
    expected=next(r for r in ref['frames'] if r['view']=='quarter')
    old_keys={k.name:float(k.value) for k in keys}
    if old_keys!=expected['actualKeyValues']:raise RuntimeError('Current static/default expression controls differ from exact quarter reference')
    uv=api('production_uv_r08');old_pose=uv['_pose_contract']()
    detail=api('final_detail_review_r09')
    lights=detail['_lights'](scene);world=detail['_world'](scene)
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    if lights!=ref['lights'] or world!=ref['world'] or colour!=ref['colourManagement']:
        raise RuntimeError('Exact saved lighting/colour match required; do not compare a relit candidate')
    source_sha=sha(source)
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'helperHashes':{name:sha(Path(__file__).with_name(name)) for name in ['chin_throat_contour_preview_r11.py','chin_throat_contour_surface_r11.py','chin_throat_contour_nodes_r11.py','whole_face_proportion_study_r11.py']},
        'referenceReport':REFERENCE,'referenceReportSha256':sha(ref_path),'observedSourceSha256':observed['sourceSha256'],
        'sourceSaved':False,'technicalAcceptance':False,'appearanceAccepted':False,'diagnosticPreview':True,
        'scope':'One static neutral contour candidate; actual local surface guards followed by 3 images. No animation/production acceptance.',
        'lights':lights,'world':world,'colourManagement':colour,'lightingMatchesReferenceExactly':True,
        'preservedKeyValues':old_keys,'frames':[],'guardErrors':[],'actualChecks':{},'samples':samples,'resolution':ref['resolution']}
    payload={};mod=group=cache=mesh=key=camera=None
    original_external_pose=external_geometry({head})
    report_path=out/'render-report.json';npz_path=out/'actual-diagnostic.npz'
    def persist():
        np.savez_compressed(npz_path,**payload)
        report['arrays']=str(npz_path.relative_to(project));report['arraysSha256']=sha(npz_path)
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    def value(v):
        keys[KEY].value=v;head.update_tag();bpy.context.view_layer.update()
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            before=snapshot(head);payload.update(beforeWorld=before['world'],faces=before['faces'],groups=before['groups'],beforeTriangles=before['triangles'])
            if not np.array_equal(before['groups'],observation['rest_head_faceGroups']):raise RuntimeError('Actual FACE group identity differs from current probe')
            boundary=observation['rest_head_boundaryVertexIds']
            if not np.array_equal(before['world'][boundary],observation['rest_head_boundaryWorld']):raise RuntimeError('Current low neck interface differs from current observation')
            offset,field,plan=api('chin_throat_contour_surface_r11')['plan'](before['world'],before['faces'],before['groups'],before['triangles'],amplitude)
            payload.update(field);report['plan']=plan;persist()
            original_external=external_geometry({head})
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key
            raw=array(head.data.shape_keys.reference_key.data,'co',3,np.float32);key.data.foreach_set('co',raw.ravel())
            key.slider_min=0.;key.slider_max=1.;key.value=0.
            mod,group,cache,mesh=api('chin_throat_contour_nodes_r11')['build'](head,len(before['world']),field['originalVertexIds'],field['worldOffsets'],before['matrix'])
            value(0.);zero=snapshot(head);payload['disabledWorld']=zero['world'];persist()
            if not np.array_equal(zero['world'],before['world']) or not np.array_equal(zero['faces'],before['faces']):raise RuntimeError('Key0 does not preserve exact current geometry')
            value(1.);after=snapshot(head);payload['afterWorld']=after['world'];payload['afterTriangles']=after['triangles'];persist()
            if not np.array_equal(after['faces'],before['faces']) or not np.array_equal(after['groups'],before['groups']):raise RuntimeError('Actual topology/anatomical identity changed')
            active=np.any(offset!=0,axis=1);delta=after['world']-before['world']
            protected={
                'lipNoseFaceAbove335':np.flatnonzero(before['world'][:,2]>=1.335),
                'eyesEarsInnerOral':np.unique(before['faces'][np.isin(before['groups'],[2,3,4,5,7,8])]),
                'lowNeckSeam':boundary,'chinFrontAnchor':np.array([3214]),
                'outsideContinuousSupport':np.flatnonzero(~active)}
            checks={n:{'vertices':len(ids),'exact':bool(np.array_equal(before['world'][ids],after['world'][ids])),'maximumMeters':float(np.linalg.norm(delta[ids],axis=1).max()) if len(ids) else 0.} for n,ids in protected.items()}
            report['actualChecks'].update(protected=checks,zeroExact=True,maximumMeters=float(np.linalg.norm(delta,axis=1).max()),
                expectedErrorMeters=float(np.linalg.norm(delta-offset,axis=1).max()),xZExact=bool(np.array_equal(before['world'][:,[0,2]],after['world'][:,[0,2]])))
            persist()
            if any(not r['exact'] for r in checks.values()) or report['actualChecks']['expectedErrorMeters']>3e-7 or not report['actualChecks']['xZExact']:raise RuntimeError('Actual contour differs from local continuous Y field or moves protected geometry')
            quality=api('upper_lid_roll_install_r11')['quality'](before['world'],after['world'],after['triangles'],active)
            report['actualChecks']['quality']=quality
            if quality['nonpositiveNormals'] or quality['minimumNormalAlignment']<.85 or quality['areaRatioRange'][0]<.65 or quality['areaRatioRange'][1]>1.5:raise RuntimeError('Actual candidate folds or strains')
            sections=api('chin_throat_contour_surface_r11')['section_check'](before['world'],after['world'],after['triangles'])
            report['actualChecks']['sections']=sections
            if sections['newReversedSteps']:raise RuntimeError('Actual field creates a new throat section waist')
            contact=api('adult_blink_midpath_contact_r11')['changed_intersections']
            previous=contact(before['world'],after['triangles'],active);current=contact(after['world'],after['triangles'],active)
            report['actualChecks'].update(priorCrossings=len(previous),afterCrossings=len(current),newCrossings=len(current-previous))
            current_external=external_geometry({head,cache})
            changes=[n for n in sorted(set(original_external)|set(current_external)) if original_external.get(n)!=current_external.get(n)]
            report['actualChecks'].update(externalEvaluatedGeometryExact=not changes,externalChangedObjects=changes,externalActualObjects=len(original_external),externalGeometryBefore=original_external)
            persist()
            if current-previous or changes:raise RuntimeError('New local intersection or actual external/hair/eye geometry movement')
        # The exact saved POSE, not REST, is used for all rendering.
        if uv['_pose_contract']()!=old_pose:raise RuntimeError('Saved pose did not restore exactly after static capture')
        if {k.name:float(k.value) for k in keys if k.name!=KEY}!=old_keys:raise RuntimeError('Original static/expression controls changed')
        external_pose=external_geometry({head,cache})
        pose_changes=[n for n in sorted(set(original_external_pose)|set(external_pose)) if original_external_pose.get(n)!=external_pose.get(n)]
        report['actualChecks'].update(posedExternalEvaluatedGeometryExact=not pose_changes,posedExternalChangedObjects=pose_changes)
        if pose_changes:raise RuntimeError('Actual hair/eyes/external geometry changed in saved review POSE')
        report['actualChecks']['localStaticChecksPassed']=True;persist()
        target=np.asarray(expected['camera']['targetWorld'],float)
        quarter_matrix=np.asarray(expected['camera']['matrixWorld'],float)
        camera=bpy.data.objects.new('R11 chin throat diagnostic camera',bpy.data.cameras.new('R11 chin throat diagnostic camera'))
        scene.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.ortho_scale=float(expected['camera']['orthographicScale']);camera.data.dof.use_dof=False;scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.resolution_x,scene.render.resolution_y=map(int,ref['resolution']);scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
        for name,strength in [('quarter-1',1.),('side-0',0.),('side-1',1.)]:
            value(strength)
            if name.startswith('quarter'):camera.matrix_world=Matrix(quarter_matrix.tolist())
            else:
                camera.location=Vector(target+np.array([2.,0.,0.]));camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()
            m=np.asarray(camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world,float)
            if name.startswith('quarter') and np.max(abs(m-quarter_matrix))>2e-7:raise RuntimeError('Quarter camera changed from actual reference')
            if {k.name:float(k.value) for k in keys if k.name!=KEY}!=old_keys or uv['_pose_contract']()!=old_pose:raise RuntimeError('Original controls/pose changed while reviewing')
            if detail['_lights'](scene)!=lights or detail['_world'](scene)!=world:raise RuntimeError('Lighting changed while reviewing')
            image=out/(name+'.png');scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
            if not image.is_file():raise RuntimeError('Actual rendered preview missing')
            report['frames'].append({'view':name,'control':strength,'image':image.name,'imageSha256':sha(image),
                'cameraMatrixWorld':m.tolist(),'targetWorld':target.tolist(),'orthographicScale':camera.data.ortho_scale,'poseExact':True,'originalKeysExact':True})
            persist();print('R11_CHIN_THROAT_PREVIEW '+json.dumps({'image':str(image),'strength':strength}),flush=True)
        if sha(source)!=source_sha:raise RuntimeError('Saved source changed during diagnostic')
        report['complete']=True;persist()
        return report
    except BaseException as exc:
        report['guardErrors'].append(str(exc));persist();raise
    finally:
        scene.camera=old_camera
        if mod is not None:head.modifiers.remove(mod)
        if key is not None:head.shape_key_remove(key)
        if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
        if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
        if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        if camera is not None:
            cdata=camera.data;bpy.data.objects.remove(camera,do_unlink=True)
            if cdata.users==0:bpy.data.cameras.remove(cdata)
        head.active_shape_key_index=old_active
        if (scene.frame_current,scene.frame_subframe)!=old_frame:scene.frame_set(old_frame[0],subframe=old_frame[1])
        head.update_tag();bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--amplitude',type=float,default=.004);p.add_argument('--samples',type=int,default=24)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    preview(args.source,args.output,args.amplitude,args.samples)
