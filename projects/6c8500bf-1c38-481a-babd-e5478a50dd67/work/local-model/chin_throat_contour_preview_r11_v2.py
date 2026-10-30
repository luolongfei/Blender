"""Finish one side diagnostic per fresh process and preserve exact render diffs.

Reuses the completed quarter and the actual successful local surface checks.
No Blend save, no matrix writeback, no unobserved cache-equivalence assumption.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

BASE_PATH=Path(__file__).with_name('chin_throat_contour_preview_r11.py')
BASE_SHA='7e0777082dacf1069c4ae195517e3ee2c121376ce0cee99cb7cb6208b5575e90'
base=runpy.run_path(str(BASE_PATH))
api,sha,array,snapshot,external_geometry=(base[k] for k in ['api','sha','array','snapshot','external_geometry'])
HEAD,KEY,REFERENCE=(base[k] for k in ['HEAD','KEY','REFERENCE'])
project_path=base['project_path']
PRIOR='previews/reconstruction-r02/chin-throat-contour-preview-r11/render-report.json'


def exact_diff(before,after,path='$'):
    return api('head_neck_posture_observe_r11_v2')['leaf_diff'](before,after,path)


def authored_rna(block,uv):
    out={}
    for prop in block.bl_rna.properties:
        if prop.identifier=='rna_type' or prop.is_readonly:continue
        if prop.type in {'BOOLEAN','INT','FLOAT','ENUM','STRING','POINTER'}:
            try:out[prop.identifier]=uv['_plain'](getattr(block,prop.identifier))
            except (AttributeError,TypeError,ValueError):pass
    return out


def state(head,uv,excluded_key=KEY):
    import bpy
    scene=bpy.context.scene
    author={'frame':scene.frame_current,'subframe':scene.frame_subframe,
        'originalHeadKeys':{k.name:float(k.value) for k in head.data.shape_keys.key_blocks if k.name!=excluded_key},
        'headScalars':{name:uv['_plain'](head[name]) for name in head.keys()},'rigs':{}}
    rows=[]
    def channels(block):
        active='rotation_quaternion' if block.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if block.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
        return {n:uv['_plain'](getattr(block,n)) for n in ('location','rotation_mode','scale',active)}
    for rig in scene.objects:
        if rig.type!='ARMATURE':continue
        delta_rotation='delta_rotation_quaternion' if rig.rotation_mode=='QUATERNION' else 'delta_rotation_euler'
        row={'posePosition':rig.data.pose_position,'channels':channels(rig),
            'deltaChannels':{n:uv['_plain'](getattr(rig,n)) for n in ('delta_location','delta_scale',delta_rotation)},
            'animation':uv['_animation'](rig),'custom':uv['_custom'](rig),
            'constraints':[authored_rna(c,uv) for c in rig.constraints],'bones':{}}
        for b in rig.pose.bones:
            row['bones'][b.name]={'channels':channels(b),
                'custom':uv['_custom'](b),'constraints':[authored_rna(c,uv) for c in b.constraints]}
        author['rigs'][rig.name]=row
        rows.append({'name':rig.name,'world':uv['_plain'](rig.matrix_world),'animation':uv['_animation'](rig),
            'bones':[{'name':b.name,'basis':uv['_plain'](b.matrix_basis),'matrix':uv['_plain'](b.matrix),'custom':uv['_custom'](b)} for b in rig.pose.bones]})
    return {'author':author,'derived':{'poseContract':uv['_pose_contract'](),'poseRows':rows,
        'headMatrixWorld':[list(r) for r in head.matrix_world]}}


def preview(source_relative,output_relative,view,samples=24):
    import bpy
    from mathutils import Vector
    if sha(BASE_PATH)!=BASE_SHA:raise RuntimeError('Frozen first-preview helper changed')
    if view not in ['side-0','side-1'] or not 8<=samples<=32:raise ValueError('One explicit side control and bounded samples required')
    project=project_path();source=(project/source_relative).resolve();out=(project/output_relative).resolve()
    source.relative_to((project/'work/reconstruction-r02').resolve());out.relative_to((project/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('A fresh saved-source background process is required for each side image')
    paths={n:out/(view+suffix) for n,suffix in [('image','.png'),('report','.json'),('arrays','-diagnostic.npz')]}
    if any(p.exists() for p in paths.values()):raise RuntimeError('Preserve previous side output')
    prior_path=project/PRIOR;prior=json.loads(prior_path.read_text(encoding='utf8'))
    actual_path=project/prior['arrays'];quarter=next(r for r in prior['frames'] if r['view']=='quarter-1')
    quarter_image=prior_path.parent/quarter['image']
    if sha(actual_path)!=prior['arraysSha256'] or sha(quarter_image)!=quarter['imageSha256']:raise RuntimeError('Completed actual diagnostics/quarter changed')
    checks=prior['actualChecks']
    if not (checks['localStaticChecksPassed'] and checks['externalEvaluatedGeometryExact'] and checks['posedExternalEvaluatedGeometryExact'] and checks['zeroExact'] and checks['newCrossings']==0 and checks['sections']['newReversedSteps']==0):raise RuntimeError('Prior static surface checks are incomplete')
    source_sha=sha(source)
    if source_sha!=prior['sourceSha256']:raise RuntimeError('Side completion must use the exact same source as completed quarter')
    data=np.load(actual_path);ref=json.loads((project/REFERENCE).read_text(encoding='utf8'))
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks;scene=bpy.context.scene
    if KEY in keys:raise RuntimeError('Fresh source without in-memory preview Key required')
    if (scene.frame_current,scene.frame_subframe)!=(1,0.):raise RuntimeError('Exact saved frame-one source required; no silent recanonicalization')
    bpy.context.view_layer.update()
    original_keys={k.name:float(k.value) for k in keys}
    if original_keys!=prior['preservedKeyValues']:raise RuntimeError('Saved original expression/sculpt controls changed')
    uv=api('production_uv_r08');detail=api('final_detail_review_r09');saved_state=state(head,uv)
    lights=detail['_lights'](scene);world=detail['_world'](scene)
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    if lights!=prior['lights'] or world!=prior['world'] or colour!=prior['colourManagement']:raise RuntimeError('Saved lighting differs from completed quarter')
    out.mkdir(parents=True,exist_ok=True)
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'priorReport':PRIOR,'priorReportSha256':sha(prior_path),'reusedQuarter':str(quarter_image.relative_to(project)),
        'reusedQuarterSha256':sha(quarter_image),'reusedActualArraysSha256':sha(actual_path),
        'sourceSaved':False,'diagnosticPreview':True,'technicalAcceptance':False,'appearanceAccepted':False,
        'view':view,'samples':samples,'resolution':ref['resolution'],'guardErrors':[],
        'savedState':saved_state,'actualChecks':{},'scope':'One fresh-process side image, reusing the successful actual contour and completed quarter. Render pre/post input, derived and actual geometry differences are retained.'}
    diagnostic={};mod=group=cache=mesh=key=camera=None;old_camera=scene.camera;old_active=head.active_shape_key_index
    def persist():
        np.savez_compressed(paths['arrays'],**diagnostic)
        report['arrays']=str(paths['arrays'].relative_to(project));report['arraysSha256']=sha(paths['arrays'])
        paths['report'].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    def value(v):keys[KEY].value=v;head.update_tag();bpy.context.view_layer.update()
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            before=snapshot(head)
            report['actualChecks']['currentRestBeforeEqualsPrior']=all(np.array_equal(before[k],data[n]) for k,n in [('world','beforeWorld'),('faces','faces'),('groups','groups')])
            if not report['actualChecks']['currentRestBeforeEqualsPrior']:raise RuntimeError('Current actual neutral head differs from completed static diagnostic')
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key
            key.data.foreach_set('co',array(head.data.shape_keys.reference_key.data,'co',3,np.float32).ravel());key.slider_min=0.;key.slider_max=1.;key.value=0.
            mod,group,cache,mesh=api('chin_throat_contour_nodes_r11')['build'](head,len(before['world']),data['originalVertexIds'],data['worldOffsets'],before['matrix'])
            strength=float(view[-1]);value(strength);changed=snapshot(head)
            reference_world=data['afterWorld'] if strength else data['beforeWorld']
            report['actualChecks']['currentRestCandidateEqualsPrior']=np.array_equal(changed['world'],reference_world) and np.array_equal(changed['faces'],data['faces']) and np.array_equal(changed['groups'],data['groups'])
            diagnostic.update(restBeforeWorld=before['world'],restCandidateWorld=changed['world'])
            persist()
            if not report['actualChecks']['currentRestCandidateEqualsPrior']:raise RuntimeError('Actual current native field differs from completed successful geometry')
        pre_camera_state=state(head,uv)
        report['preCameraState']=pre_camera_state
        report['savedToPreCameraDiff']=exact_diff(saved_state,pre_camera_state)
        persist()
        if report['savedToPreCameraDiff']:raise RuntimeError('Original saved author or derived pose changed before rendering')
        target=np.asarray(quarter['targetWorld'],float)
        camera=bpy.data.objects.new('R11 chin throat side completion',bpy.data.cameras.new('R11 chin throat side completion'))
        scene.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.ortho_scale=quarter['orthographicScale'];camera.data.dof.use_dof=False
        camera.location=Vector(target+np.array([2.,0.,0.]));camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler();scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.resolution_x,scene.render.resolution_y=map(int,ref['resolution']);scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
        bpy.context.view_layer.update()
        pre=state(head,uv);pre_head=snapshot(head);pre_external=external_geometry({head,cache})
        report['preRenderState']=pre;report['savedToPreRenderDiff']=exact_diff(saved_state,pre)
        report['cameraMatrixWorld']=[list(r) for r in camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world]
        report.update(targetWorld=target.tolist(),orthographicScale=camera.data.ortho_scale,control=strength,
            preRenderExternalGeometry=pre_external,lightingMatchesReferenceExactly=True)
        diagnostic['preRenderPosedHeadWorld']=pre_head['world'];persist()
        if report['savedToPreRenderDiff']:raise RuntimeError('Original saved author or derived pose changed before side render')
        scene.render.filepath=str(paths['image']);bpy.ops.render.render(write_still=True)
        if not paths['image'].is_file():raise RuntimeError('Actual side image missing')
        report.update(image=str(paths['image'].relative_to(project)),imageSha256=sha(paths['image']))
        # Do not repair or overwrite any original channels. First retain the
        # render-return state, then the result of an ordinary viewport update.
        post_raw=state(head,uv);report['postRenderBeforeUpdateState']=post_raw
        report['renderReturnExactDiff']=exact_diff(pre,post_raw);persist()
        bpy.context.view_layer.update()
        post=state(head,uv);post_head=snapshot(head);post_external=external_geometry({head,cache})
        diagnostic['postRenderPosedHeadWorld']=post_head['world']
        report['postRenderState']=post
        report['authorDiff']=exact_diff(pre['author'],post['author'])
        report['derivedDiff']=exact_diff(pre['derived'],post['derived'])
        report['externalGeometryDiff']=exact_diff(pre_external,post_external)
        report['postRenderHeadExact']=all(np.array_equal(pre_head[k],post_head[k]) for k in ['world','faces','groups'])
        report['postRenderHeadMaximumDifferenceMeters']=float(np.linalg.norm(pre_head['world']-post_head['world'],axis=1).max())
        report['sourceSha256After']=sha(source);report['sourceFileExact']=report['sourceSha256After']==source_sha
        report['postRenderStateClassification']='exact' if not report['authorDiff'] and not report['derivedDiff'] else 'unresolved'
        if not report['authorDiff'] and report['postRenderHeadExact'] and not report['externalGeometryDiff'] and report['derivedDiff']:
            report['postRenderStateClassification']='derived-matrix-only difference with exact authored inputs and complete actual geometry'
        persist()
        if report['authorDiff'] or not report['postRenderHeadExact'] or report['externalGeometryDiff'] or not report['sourceFileExact']:
            raise RuntimeError('Render changed original input or actual geometry; precise before/after diagnostics retained')
        report['renderInputsVerified']=True;report['complete']=True;persist()
        print('R11_CHIN_THROAT_SIDE_COMPLETE '+json.dumps({'view':view,'image':str(paths['image']),'postRenderStateClassification':report['postRenderStateClassification']}),flush=True)
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
        head.active_shape_key_index=old_active;bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--view',required=True,choices=['side-0','side-1']);p.add_argument('--samples',type=int,default=24)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    preview(a.source,a.output,a.view,a.samples)
