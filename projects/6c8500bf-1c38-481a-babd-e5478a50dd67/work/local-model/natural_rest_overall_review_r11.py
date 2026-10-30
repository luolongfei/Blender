"""Two comparable whole-face views of the saved relaxed-rest candidate.

The quarter camera is copied from the actual 31df review report; the front
uses its same target, scale and resolution. Only explicitly named public
expression controls are changed. Static sculpt Keys, internal driven Keys,
groom, skin and lighting remain as saved. No Blend is saved.
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
SEAL='R11 relaxed lip seal'
WARM='R11 warm closed smile'
PUBLIC_REST_CONTROLS=('R11 blink L','R11 blink R','R11 gentle smile',
                      'R11 natural soft smile',WARM,'R11 jaw open')
DEFAULT_PLAN='work/reconstruction-r02/natural-rest-overall-review-r11.json'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def camera_target(matrix,distance):
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Finite actual reference camera matrix required')
    axis=matrix[:3,2];axis=axis/np.linalg.norm(axis)
    return matrix[:3,3]-axis*float(distance)


def review(source_relative,output_relative,plan_relative=DEFAULT_PLAN,warm=None):
    import bpy
    from mathutils import Matrix,Vector
    project=project_path();source=(project/source_relative).resolve();out=(project/output_relative).resolve();planpath=(project/plan_relative).resolve()
    source.relative_to((project/'work/reconstruction-r02').resolve());out.relative_to((project/'previews/reconstruction-r02').resolve());planpath.relative_to(project.resolve())
    if source.suffix!='.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved independent candidate required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve prior review files')
    plan=json.loads(planpath.read_text(encoding='utf8'));reference_path=(project/plan['referenceRenderReport']).resolve();reference_path.relative_to(project.resolve())
    if sha(reference_path)!=plan['referenceRenderReportSha256']:raise RuntimeError('Inspect changed reference framing before review')
    reference=json.loads(reference_path.read_text(encoding='utf8'))
    views=plan['views'];warm=float(plan['warmSmile'] if warm is None else warm)
    if views!=['front','quarter'] or not 0<=warm<=.2:raise ValueError('Two fixed rest views, with only an explicit optional slight-smile override')
    width,height=map(int,reference['resolution']);samples=int(plan['samples'])
    if not 8<=samples<=32:raise ValueError('Bounded review samples required')
    reference_matrix=np.asarray(reference['camera']['matrixWorld'],float)
    target=camera_target(reference_matrix,plan['referenceCameraDistanceMeters'])
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if SEAL not in keys or WARM not in keys:raise RuntimeError('Saved seal and warm controls required')
    if float(head.get('R11 lower face continuity',0.))!=0.:raise RuntimeError('Rejected first continuity layer remains enabled')
    scene=bpy.context.scene;source_frame=[scene.frame_current,scene.frame_subframe]
    saved_values={key.name:float(key.value) for key in keys}
    old_camera=scene.camera;scene.frame_set(1);bpy.context.view_layer.update()
    at_frame_one={key.name:float(key.value) for key in keys}
    drivers=head.data.shape_keys.animation_data
    driven_paths=set() if drivers is None else {curve.data_path for curve in drivers.drivers}
    driven={key.name for key in keys if key.path_from_id('value') in driven_paths}
    requested={name:0. for name in PUBLIC_REST_CONTROLS if name in keys};requested[WARM]=warm;requested[SEAL]=1.
    if driven.intersection(requested):raise RuntimeError('Requested public rest control is driven; inspect its actual expression interface')
    preserved={name:value for name,value in at_frame_one.items() if name not in requested and name not in driven}
    detail=api('final_detail_review_r09');lights=detail['_lights'](scene);world=detail['_world'](scene)
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
            'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    source_sha=sha(source)
    refs=[]
    for row in plan['designReferences']:
        path=(project/row['path']).resolve();path.relative_to(project.resolve())
        refs.append({**row,'sha256':sha(path)})
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'plan':plan_relative,'planSha256':sha(planpath),'referenceReport':plan['referenceRenderReport'],
        'referenceReportSha256':sha(reference_path),'referenceSourceSha256':reference['sourceSha256'],
        'designReferences':refs,'resolution':[width,height],'samples':samples,'sourceSceneFrame':source_frame,
        'reviewSceneFrame':1,'savedKeyValues':saved_values,'frameOneKeyValues':at_frame_one,
        'explicitPublicControls':requested,'preservedUndrivenKeys':preserved,'internallyDrivenKeys':sorted(driven),
        'headScalarControls':{str(name):float(head[name]) for name in ('R11 upper lid relief','R11 lower face continuity') if name in head},
        'lights':lights,'world':world,'colourManagement':colour,
        'referenceLightingMatches':lights==reference['lights'] and world==reference['world'] and colour==reference['colourManagement'],
        'groomVisibilityChanged':False,'appearanceAccepted':False,'sourceSaved':False,'frames':[],
        'scope':'Whole-face default-rest review. Quarter reuses the actual 31df camera; front shares its target, scale and crop. Current saved sculpture and groom remain visible. Generated art references are direction only, not geometry calibration.'}
    camera=None
    try:
        for name,value in requested.items():keys[name].value=value
        head.update_tag();bpy.context.view_layer.update()
        camera=bpy.data.objects.new('R11 natural rest overall review',bpy.data.cameras.new('R11 natural rest overall review'))
        scene.collection.objects.link(camera);camera.data.type='ORTHO';camera.data.ortho_scale=float(reference['camera']['orthographicScale']);camera.data.dof.use_dof=False;scene.camera=camera
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
        out.mkdir(parents=True,exist_ok=True)
        for number,view in enumerate(views,1):
            if view=='quarter':camera.matrix_world=Matrix(reference_matrix.tolist())
            else:
                camera.location=Vector(target+np.array([0.,-float(plan['referenceCameraDistanceMeters']),0.]))
                camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
            bpy.context.view_layer.update()
            actual={name:float(keys[name].value) for name in requested}
            if any(abs(actual[name]-value)>1e-7 for name,value in requested.items()):raise RuntimeError('Animation or a driver overrides the explicit rest request')
            changed={name:[value,float(keys[name].value)] for name,value in preserved.items() if float(keys[name].value)!=value}
            if changed:raise RuntimeError('An unrequested static Key changed: '+json.dumps(changed))
            matrix=camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
            if view=='quarter' and np.max(abs(np.asarray(matrix)-reference_matrix))>2e-7:raise RuntimeError('Quarter framing differs from the actual reference camera')
            image=out/(view+'.png');scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
            if not image.is_file():raise RuntimeError('Actual whole-face render missing')
            row={'view':view,'image':image.name,'imageSha256':sha(image),'keys':actual,
                 'staticKeysPreservedExactly':True,'actualKeyValues':{key.name:float(key.value) for key in keys},
                 'camera':{'matrixWorld':[list(r) for r in matrix],'orthographicScale':camera.data.ortho_scale,'depthOfField':False,'targetWorld':target.tolist()}}
            report['frames'].append(row)
            with image.with_suffix('.json').open('x',encoding='utf8') as f:json.dump({'sourceSha256':source_sha,**row},f,ensure_ascii=False,indent=2)
            print('R11_NATURAL_REST_OVERALL '+json.dumps({'view':view,'image':str(image),'warm':warm}),flush=True)
        if sha(source)!=source_sha:raise RuntimeError('Saved source changed during read-only review')
        with (out/'render-report.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    finally:
        scene.camera=old_camera
        for name in requested:keys[name].value=at_frame_one[name]
        scene.frame_set(source_frame[0],subframe=source_frame[1]);head.update_tag();bpy.context.view_layer.update()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',default=DEFAULT_PLAN);p.add_argument('--warm',type=float,default=None)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source,args.output,args.plan,args.warm)
