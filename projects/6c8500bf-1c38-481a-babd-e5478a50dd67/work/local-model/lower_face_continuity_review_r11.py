"""Two genuine same-source/same-light lower-face images with continuity 0/1.

Read-only saved candidate review. Camera frames nose base, mouth and complete
chin at fixed scale. No depth of field, beauty lighting change, source save or
shape-key installation. Optional warm strength is identical in both images.
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
CONTROL='R11 lower face continuity';WARM='R11 warm closed smile'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def review(source_relative,output_relative,view='quarter',warm=0.,samples=32):
    import bpy
    from mathutils import Vector
    project=project_path();source=(project/source_relative).resolve();out=(project/output_relative).resolve()
    source.relative_to((project/'work/reconstruction-r02').resolve());out.relative_to((project/'previews/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or source.suffix!='.blend':raise RuntimeError('Fresh saved independent source required')
    if out.exists() and any(out.iterdir()):raise RuntimeError('Preserve earlier review outputs')
    if view not in ('front','quarter','side') or not 0<=warm<=1 or not 8<=samples<=64:raise ValueError('Bounded explicit review requested')
    out.mkdir(parents=True,exist_ok=True);head=bpy.data.objects['R11 Female head study'];keys=head.data.shape_keys.key_blocks
    if CONTROL not in head or WARM not in keys:raise RuntimeError('Saved lower-face continuity and warm-smile candidate required')
    original={k.name:float(k.value) for k in keys};original_control=float(head[CONTROL]);source_sha=sha(source)
    scene=bpy.context.scene;scene.frame_set(1)
    for key in keys:
        if 'blink' in key.name.lower() or 'smile' in key.name.lower():key.value=0.
    keys[WARM].value=warm;head.update_tag();bpy.context.view_layer.update()
    mixed,_=api('production_uv_r08')['current_mix'](head);T=np.asarray(head.matrix_world,float);world=np.asarray(mixed)@T[:3,:3].T+T[:3,3]
    target=(world[8005]+world[7972])*.5;target[2]+=.004
    direction=np.asarray({'front':(0.,-1.,0.),'quarter':(.6,-.8,0.),'side':(1.,0.,0.)}[view])
    camera=bpy.data.objects.new('R11 lower-face continuity review',bpy.data.cameras.new('R11 lower-face continuity review'))
    scene.collection.objects.link(camera);camera.location=Vector(target+direction*2.)
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=.115;camera.data.dof.use_dof=False;scene.camera=camera
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
    scene.render.resolution_x=960;scene.render.resolution_y=960;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG'
    detail=api('final_detail_review_r09')
    report={'source':source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),'view':view,'warmSmile':warm,
        'resolution':[960,960],'samples':samples,'camera':{'matrixWorld':[list(r) for r in camera.matrix_world],'orthographicScale':.115,'depthOfField':False},
        'lights':detail['_lights'](scene),'world':detail['_world'](scene),'frames':[],
        'colourManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma},
        'appearanceAccepted':False,'sourceSaved':False,'scope':'Current saved-source native control0/1 A/B, identical warm strength, camera, lights, skin, geometry stack and scene frame.'}
    try:
        for number,value in enumerate((0.,1.),1):
            head[CONTROL]=value;head.update_tag();bpy.context.view_layer.update()
            image=out/('frame-%04d.png'%number);scene.render.filepath=str(image)
            bpy.ops.render.render(write_still=True)
            if not image.is_file():raise RuntimeError('Review image was not written')
            row={'frame':number,'continuityControl':value,'warmSmile':float(keys[WARM].value),'imageSha256':sha(image)};report['frames'].append(row)
            with image.with_suffix('.json').open('x',encoding='utf-8') as f:json.dump({'sourceSha256':source_sha,**row},f,indent=2)
            print('R11_LOWER_FACE_CONTINUITY_REVIEW '+json.dumps(row),flush=True)
        if sha(source)!=source_sha:raise RuntimeError('Saved source changed during read-only rendering')
        with (out/'render-report.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    finally:
        for name,value in original.items():keys[name].value=value
        head[CONTROL]=original_control;head.update_tag();bpy.context.view_layer.update()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--view',choices=('front','quarter','side'),default='quarter');p.add_argument('--warm',type=float,default=0.);p.add_argument('--samples',type=int,default=32)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);review(a.source,a.output,a.view,a.warm,a.samples)
