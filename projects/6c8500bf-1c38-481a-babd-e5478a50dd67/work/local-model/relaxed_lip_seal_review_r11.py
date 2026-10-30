"""Read-only same-source 0/1 review of the relaxed lip-seal candidate.

The quarter camera deliberately matches the accepted lower-face-v2 review.
No source save, relighting, material change, field installation or automatic
appearance acceptance occurs. Existing static sculpt controls remain active.
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
CONTROL='R11 relaxed lip seal'
WARM='R11 warm closed smile'
PUBLIC_EXPRESSIONS=('R11 blink L','R11 blink R','R11 natural soft smile',WARM,'R11 jaw open')


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def review(source_relative,output_relative,view='quarter',warm=0.,samples=32):
    import bpy
    from mathutils import Vector
    project=project_path();source=(project/source_relative).resolve();out=(project/output_relative).resolve()
    source.relative_to((project/'work/reconstruction-r02').resolve())
    out.relative_to((project/'previews/reconstruction-r02').resolve())
    if source.suffix!='.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved independent candidate required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve existing review outputs')
    if view not in ('front','quarter','side') or not 0<=warm<=1 or not 8<=samples<=64:
        raise ValueError('Explicit bounded view, warm strength and samples required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if CONTROL not in keys or WARM not in keys:raise RuntimeError('Saved relaxed-seal and warm-smile controls required')
    if float(head.get('R11 lower face continuity',0.))!=0.:raise RuntimeError('Rejected first continuity layer must stay disabled')
    if 'R11 lower face contour v2' not in keys or abs(float(keys['R11 lower face contour v2'].value)-1.)>1e-7:
        raise RuntimeError('Accepted v2 lip-chin contour must remain active')
    original={key.name:float(key.value) for key in keys};source_sha=sha(source)
    scene=bpy.context.scene;scene.frame_set(1)
    for name in PUBLIC_EXPRESSIONS:
        if name in keys:keys[name].value=0.
    keys[WARM].value=warm
    head.update_tag();bpy.context.view_layer.update()
    mixed,_=api('production_uv_r08')['current_mix'](head)
    matrix=np.asarray(head.matrix_world,float);world=np.asarray(mixed)@matrix[:3,:3].T+matrix[:3,3]
    target=(world[8005]+world[7972])*.5;target[2]+=.004
    direction=np.asarray({'front':(0.,-1.,0.),'quarter':(.6,-.8,0.),'side':(1.,0.,0.)}[view])
    camera=bpy.data.objects.new('R11 relaxed lip seal review',bpy.data.cameras.new('R11 relaxed lip seal review'))
    scene.collection.objects.link(camera);camera.location=Vector(target+direction*2.)
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=.115;camera.data.dof.use_dof=False;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
    scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    out.mkdir(parents=True,exist_ok=True)
    detail=api('final_detail_review_r09');bpy.context.view_layer.update()
    evaluated_camera=camera.evaluated_get(bpy.context.evaluated_depsgraph_get())
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
        'controlKey':CONTROL,'view':view,'warmSmile':warm,'jawOpen':0.,'resolution':[960,960],'samples':samples,
        'camera':{'matrixWorld':[list(row) for row in evaluated_camera.matrix_world],
                  'orthographicScale':.115,'depthOfField':False,'targetWorld':target.tolist()},
        'lights':detail['_lights'](scene),'world':detail['_world'](scene),
        'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
                            'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
        'frames':[],'sourceSaved':False,'appearanceAccepted':False,
        'scope':'Actual native relaxed-seal 0/1 only; identical saved source, camera, lights, static face controls and expression. Not a contact or collision certificate.'}
    try:
        for number,strength in enumerate((0.,1.),1):
            keys[CONTROL].value=strength;head.update_tag();bpy.context.view_layer.update()
            if abs(float(keys[CONTROL].value)-strength)>1e-7 or abs(float(keys[WARM].value)-warm)>1e-7:
                raise RuntimeError('A driver overrides the requested review control')
            image=out/('frame-%04d.png'%number);scene.render.filepath=str(image)
            bpy.ops.render.render(write_still=True)
            if not image.is_file():raise RuntimeError('Actual render image missing')
            row={'frame':number,'relaxedSeal':strength,'warmSmile':float(keys[WARM].value),
                 'imageSha256':sha(image),'actualKeyValues':{key.name:float(key.value) for key in keys}}
            report['frames'].append(row)
            with image.with_suffix('.json').open('x',encoding='utf-8') as f:
                json.dump({'sourceSha256':source_sha,**row},f,ensure_ascii=False,indent=2)
            print('R11_RELAXED_LIP_SEAL_REVIEW '+json.dumps({'frame':number,'relaxedSeal':strength,'imageSha256':row['imageSha256']}),flush=True)
        if sha(source)!=source_sha:raise RuntimeError('Source changed during read-only review')
        with (out/'render-report.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    finally:
        for name,value in original.items():keys[name].value=value
        head.update_tag();bpy.context.view_layer.update()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--view',choices=('front','quarter','side'),default='quarter')
    p.add_argument('--warm',type=float,default=0.);p.add_argument('--samples',type=int,default=32)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    review(args.source,args.output,args.view,args.warm,args.samples)
