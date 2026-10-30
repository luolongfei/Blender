"""Render actual matched R09 body, native cloth and native hair together."""
import argparse, json, runpy, sys, time
from pathlib import Path
import bpy
from mathutils import Vector

FOLDER=Path(__file__).resolve().parent
M=runpy.run_path(str(FOLDER/'cloth_merge_r09.py'))
C=M['C']

def render(source_relative, mode='sparse', label=None):
    if not bpy.app.background:raise RuntimeError('Run in an isolated background process')
    source=C['_path'](source_relative)
    if Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Open the exact requested source first')
    if mode not in ('sparse','continuous'):raise ValueError('Unknown review mode')
    label=label or 'body-combined-'+mode+'-r09'
    if Path(label).name!=label:raise ValueError('Label must be one directory name')
    out=C['project_path']()/'previews/reconstruction-r02'/label
    if out.exists():raise RuntimeError('Preserve existing visual evidence')
    source_sha=C['_sha'](source);rig=C['_rig']();scene=bpy.context.scene
    cloth=bpy.data.objects[C['NATIVE']]
    if cloth.hide_render or cloth.hide_viewport or cloth.hide_get():raise RuntimeError('Matched R09 cloth is hidden')
    if rig.animation_data.action.name!=C['ACTION']:raise RuntimeError('Select the matching R09 body action')
    hair=M['_matched_hair'](rig)
    verification=M['verify_loaded'](C['_path']('work/reconstruction-r02/cloth-body-r09'),cloth)
    native=runpy.run_path(str(FOLDER/'native_hair_playback_r04.py'))
    hair_record=json.loads(bpy.data.objects[hair['guide']][native['RECORD']])
    hair_parity=native['validate_evaluated'](cache_relative=hair_record['cacheRelativePath'],frames=[1,55,109.375,151,247,301])
    camera=bpy.data.objects.new('R09 combined body review camera',bpy.data.cameras.new('R09 combined body review camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.data.type='ORTHO';camera.data.ortho_scale=1.90;camera.data.dof.use_dof=False
    camera.location=(3.,-4.,1.12);target=Vector((0.,0.,.83))
    camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
    scene.render.resolution_x=480;scene.render.resolution_y=640;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.use_sequencer=False
    frames=list(range(1,302)) if mode=='continuous' else [1,55,151,247,301]
    fps=scene.render.fps/scene.render.fps_base
    if abs(fps-30)>1e-6:raise RuntimeError('Preserve authored 30 fps')
    out.mkdir(parents=True);rows=[];started=time.monotonic()
    for frame in frames:
        C['_set_frame'](frame)
        if cloth.hide_render:raise RuntimeError('Matched cloth was unexpectedly hidden')
        for name in hair_record['cacheMetadata']['objects']:
            obj=bpy.data.objects.get(name)
            if obj is None or obj.type!='CURVES' or obj.hide_render:
                raise RuntimeError('A matched production tail is missing or hidden: '+name)
        path=out/f'frame-{frame:04d}.png';scene.render.filepath=str(path)
        bpy.ops.render.render(write_still=True)
        rows.append({'frame':frame,'path':str(path.relative_to(C['project_path']()))})
        print('R09_COMBINED_FRAME '+str(frame),flush=True)
    if C['_sha'](source)!=source_sha:raise RuntimeError('Saved source changed during review')
    report={'source':str(source),'sha256':source_sha,'mode':mode,'fps':fps,'frames':rows,
        'resolution':[480,640],'samples':12,'cameraPosition':list(camera.location),'cameraTarget':list(target),
        'clothVerification':verification,'matchedHair':hair,'nativeHairParity':hair_parity,
        'wallSeconds':time.monotonic()-started,'savedBlend':False,'appearanceAccepted':False,
        'scope':'Actual authored R09 squat, bend and reach with matching embedded cloth and hair. No old cloth or static-hair substitution. Continuous mode renders every authored frame; sparse mode proves only its listed poses.'}
    (out/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--mode',choices=('sparse','continuous'),default='sparse')
    parser.add_argument('--label')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    render(args.source,args.mode,args.label)
