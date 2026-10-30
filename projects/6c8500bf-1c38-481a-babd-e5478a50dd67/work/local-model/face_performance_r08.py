"""Two explicit entry points for an independent R08 portrait performance.

Background Blender CLI:
  reference-face-flow-r08.blend --python face_performance_r08.py -- prepare
  reference-face-performance-r08.blend --python face_performance_r08.py -- sparse
  reference-face-performance-r08.blend --python face_performance_r08.py -- continuous

prepare() validates and saves only the new portrait copy. render(mode) never
saves a blend. Sparse fresh-file proof for the exact source SHA precedes the
continuous 181-frame render. No Python hair handler and no walk cloth playback.
After continuous completion, use encode_review_r07.py --folder
previews/reconstruction-r02/face-performance-continuous-r08
--expected-count 181 --output face-performance-r08.mp4
"""
import argparse, hashlib, json, math, re, runpy, sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SOURCE='work/reconstruction-r02/reference-face-flow-r08.blend'
TARGET='work/reconstruction-r02/reference-face-performance-r08.blend'
PREPARED='work/reconstruction-r02/face-performance-source-r08.json'
CACHE='work/reconstruction-r02/hair-batch-r07/R03-Performance-review.guides.npz'
BODY_ACTION='R03 Performance review'
FACE_ACTION='R03 Silent vowels and expression combinations'
SPARSE=[1,20,42,63,82,98,110,128,146,162,181]
PARITY=[1,19.375,42,81.5,121,145.375,162,181]
STATIC_SKIRT=('Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _guard(relative):
    if bpy is None or not bpy.app.background:raise RuntimeError('Use a fresh disposable background Blender process')
    expected=(project_path()/relative).resolve()
    if Path(bpy.data.filepath).resolve()!=expected or not expected.is_file():
        raise RuntimeError('Exact saved review source required: '+relative)
    if bpy.data.is_dirty:raise RuntimeError('Fresh-file process required; source already has unsaved changes')
    if bpy.context.mode!='OBJECT':raise RuntimeError('Object mode required')
    for handlers in (bpy.app.handlers.frame_change_pre,bpy.app.handlers.frame_change_post):
        if any(getattr(h,'_r02_secondary_hair',False) for h in handlers):
            raise RuntimeError('Old Python hair playback handler is present; use a fresh process without it')
    return expected


def _native():
    return runpy.run_path(str(Path(__file__).with_name('native_hair_playback_r04.py')))


def _bind(owner,action_name,id_type):
    action=bpy.data.actions[action_name]
    slots=[slot for slot in action.slots if slot.target_id_type==id_type]
    if len(slots)!=1:raise RuntimeError(f'Expected one {id_type} slot in {action_name}')
    owner.animation_data_create();ad=owner.animation_data
    if ad.action:ad.action.use_fake_user=True
    muted=[track.name for track in ad.nla_tracks if not track.mute]
    for track in ad.nla_tracks:track.mute=True
    ad.action=action;ad.action_slot=slots[0]
    return {'action':action.name,'slot':slots[0].identifier,'targetIdType':id_type,'mutedNlaTracks':muted,
            'inheritedActionRange':[float(v) for v in action.frame_range]}


def _check_actions():
    rows=[]
    for owner,name,kind in ((bpy.data.objects['Character_Rig'],BODY_ACTION,'OBJECT'),
            (bpy.data.objects['R01 Head'].data.shape_keys,FACE_ACTION,'KEY')):
        ad=owner.animation_data
        if not ad or not ad.action or ad.action.name!=name or not ad.action_slot or ad.action_slot.target_id_type!=kind:
            raise RuntimeError(f'Active action/slot mismatch: {name} / {kind}')
        if any(not t.mute for t in ad.nla_tracks):raise RuntimeError('Active NLA is outside portrait review scope')
        rows.append({'action':name,'slot':ad.action_slot.identifier,'targetIdType':kind})
    return rows


def _portrait_cloth(change):
    records=[]
    if bpy.data.objects.get(STATIC_SKIRT[0]) is None:raise RuntimeError('Preserved original static skirt is missing')
    for obj in bpy.context.scene.objects:
        dynamic=(any(m.type=='CLOTH' for m in obj.modifiers)
            or obj.name in ('R02 Cloth Skirt control','R06 Walk cloth')
            or ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
        if dynamic and obj.name in STATIC_SKIRT:raise RuntimeError('Original skirt unexpectedly contains simulation')
        if not dynamic and obj.name not in STATIC_SKIRT:continue
        if change:
            obj.hide_render=dynamic;obj.hide_viewport=dynamic;obj.hide_set(dynamic)
        if obj.hide_render!=dynamic or obj.hide_viewport!=dynamic:
            raise RuntimeError('Portrait cloth visibility mismatch: '+obj.name)
        records.append({'object':obj.name,'role':'hidden historical simulation' if dynamic else 'visible original static skirt',
                        'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport})
    return records


def _frame(value):
    integer=math.floor(value);bpy.context.scene.frame_set(integer,subframe=float(value)-integer)
    bpy.context.view_layer.update()


def _face_probe(native):
    keys=bpy.data.objects['R01 Head'].data.shape_keys.key_blocks
    expectations={1:{'R02_blink_L':0.,'R02_jaw_open':0.},20:{'R02_blink_L':1.,'R02_blink_R':1.},
        42:{'R02_jaw_open':.65},63:{'R02_mouth_wide':.75},82:{'R02_pucker':.6},
        98:{'R02_pucker':.85},110:{'R02_lip_press':.85},128:{'R02_smile':1.},
        146:{'R02_brow_furrow':.6},162:{'R02_brow_raise':.65,'R02_jaw_open':.35},
        181:{'R02_blink_L':0.,'R02_jaw_open':0.}}
    face_action=bpy.data.actions[FACE_ACTION]
    times=[float(k.co.x) for fc in native['_fcurves'](face_action) for k in fc.keyframe_points]
    if not times or min(times)<1 or max(times)>181:raise RuntimeError('Facial action exceeds the authored 1–181 range')
    rows=[]
    try:
        for frame,values in expectations.items():
            _frame(frame);actual={name:float(keys[name].value) for name in values}
            if any(abs(actual[name]-value)>2e-5 for name,value in values.items()):
                raise RuntimeError('Facial action did not evaluate as authored: '+json.dumps({'frame':frame,'actual':actual,'expected':values}))
            for name in ('R07_cheek_surface_continuity','R08_cheek_perioral_transition'):
                if abs(keys[name].value-1.)>1e-6:raise RuntimeError('Accepted local facial correction changed: '+name)
            for name in ('R02 Soft brow -1','R02 Soft brow 1'):
                brow=bpy.data.objects[name]
                if not brow.get('R08_brow_arc_flow'):raise RuntimeError('Accepted R08 brow geometry is missing')
                for channel in ('R02_brow_raise','R02_brow_furrow'):
                    if abs(brow.data.shape_keys.key_blocks[channel].value-keys[channel].value)>2e-5:
                        raise RuntimeError('Brow expression driver mismatch: '+name+' / '+channel)
            rows.append({'frame':frame,'values':actual})
    finally:_frame(1)
    return {'authoredFacialKeyRange':[min(times),max(times)],'samples':rows,'driverAndStaticCorrectionChecks':'passed'}


def prepare():
    """Create the new portrait-only source after actual evaluated validation."""
    source=_guard(SOURCE);p=project_path();target=p/TARGET;report_path=p/PREPARED
    if target.exists() or report_path.exists():raise RuntimeError('Retain existing R08 performance source/report; no overwrite')
    source_sha=_sha(source);native=_native();rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    _,meta,arrays,_,_=native['load_cache'](CACHE)
    if meta['action']!=BODY_ACTION or meta['fps']!=30 or arrays['frames'][0]>1 or arrays['frames'][-1]<181:
        raise RuntimeError('Wrong performance guide cache')
    # Both old and new native caches keep the authored strands unchanged.
    restored=native['restore']() if native['RIG_GUIDE'] in rig else {'previousCache':'none'}
    actions=[_bind(rig,BODY_ACTION,'OBJECT'),_bind(bpy.data.objects['R01 Head'].data.shape_keys,FACE_ACTION,'KEY')]
    rig.data.pose_position='POSE';scene.render.fps=30;scene.render.fps_base=1.;scene.frame_start=1;scene.frame_end=181
    cloth=_portrait_cloth(True);_frame(1)
    native['run'](cache_relative=CACHE)
    native['enable_for_action'](action_name=BODY_ACTION,select=False,frame=1)
    parity=native['validate_evaluated'](cache_relative=CACHE,frames=PARITY)
    face=_face_probe(native);_check_actions();_frame(1)
    if _sha(source)!=source_sha:raise RuntimeError('Original source changed on disk during preparation')
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    result={'source':SOURCE,'sourceSha256':source_sha,'output':TARGET,'sha256':_sha(target),
        'blender':bpy.app.version_string,'actions':actions,'sceneFrames':[1,181],'fps':30,'fpsBase':1,
        'frameCount':181,'durationSeconds':181/30,'cache':CACHE,'cacheRange':meta['frameRange'],
        'priorNativeCacheRestore':restored,'hairParity':parity,'faceProbe':face,'cloth':cloth,
        'scope':'Portrait only: authored idle/head/gaze and silent expressions, embedded performance twin-tail playback, static original skirt. No walk cloth, new mocap, speech audio or appearance acceptance claim.'}
    with report_path.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('R08_FACE_PERFORMANCE_SAVED '+json.dumps({'output':TARGET,'sha256':result['sha256'],'frames':181}),flush=True)
    return result


def render(mode='sparse',label='r08'):
    """Fresh-file sparse proof, then continuous 1..181; neither saves source."""
    if mode not in ('sparse','continuous'):raise ValueError('Use sparse or continuous')
    if not re.fullmatch(r'[a-z0-9-]+',label):raise ValueError('Safe lowercase review label required')
    source=_guard(TARGET);p=project_path();digest=_sha(source)
    prepared=json.loads((p/PREPARED).read_text(encoding='utf-8'))
    if prepared['sha256']!=digest:raise RuntimeError('Saved performance source differs from its preparation proof')
    proof=p/'work/reconstruction-r02'/f'face-performance-reopen-{label}.json'
    out=p/'previews/reconstruction-r02'/f'face-performance-{mode}-{label}'
    if out.exists():raise RuntimeError('Retain the existing output folder; use a new --label')
    if mode=='sparse' and proof.exists():raise RuntimeError('Retain the existing sparse proof; use a new --label')
    if mode=='continuous':
        if not proof.is_file():raise RuntimeError('Fresh-file sparse review proof must precede continuous rendering')
        previous=json.loads(proof.read_text(encoding='utf-8'))
        if previous['sha256']!=digest or previous['status']!='sparse-rendered':raise RuntimeError('Sparse proof does not match this source SHA')
        if previous['helperSha256']!=_sha(__file__):raise RuntimeError('Review helper changed after sparse proof')
        for item in previous['images']:
            if _sha(p/item['path'])!=item['sha256']:raise RuntimeError('Sparse image changed after proof')
    native=_native();actions=_check_actions();cloth=_portrait_cloth(False)
    scene=bpy.context.scene
    if (scene.frame_start,scene.frame_end,scene.render.fps,scene.render.fps_base)!=(1,181,30,1.):
        raise RuntimeError('Saved scene is not the 181-frame 30 fps portrait clip')
    native['enable_for_action'](action_name=BODY_ACTION,select=False,frame=1)
    checks={}
    if mode=='sparse':
        checks={'hairParity':native['validate_evaluated'](cache_relative=CACHE,frames=PARITY),'faceProbe':_face_probe(native)}
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_persistent_data=True
    scene.render.resolution_x=480;scene.render.resolution_y=540;scene.render.resolution_percentage=100
    scene.render.use_sequencer=False;scene.render.use_file_extension=True;scene.render.use_overwrite=False
    if hasattr(scene.render.image_settings,'media_type'):scene.render.image_settings.media_type='IMAGE'
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    camera=bpy.data.objects.new('R08 Portrait performance camera',bpy.data.cameras.new('R08 Portrait performance camera'))
    scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='ORTHO';camera.data.ortho_scale=.36
    camera.data.dof.use_dof=False;camera.location=(0,-3,1.429)
    camera.rotation_euler=(Vector((0,-.005,1.429))-camera.location).to_track_quat('-Z','Y').to_euler()
    out.mkdir(parents=True);frames=range(1,182) if mode=='continuous' else SPARSE
    rows=[];images=[]
    for frame in frames:
        _frame(frame);path=out/f'frame-{frame:04d}.png';scene.render.filepath=str(path)
        bpy.ops.render.render(write_still=True)
        if not path.is_file() or path.stat().st_size==0:raise RuntimeError('Render did not produce its PNG: '+str(path))
        rows.append({'frame':frame});images.append({'path':path.relative_to(p).as_posix(),'sha256':_sha(path)})
        print('R08_FACE_PERFORMANCE_FRAME '+str(frame),flush=True)
    if _sha(source)!=digest:raise RuntimeError('Saved source changed while rendering')
    report={'source':str(source),'sha256':digest,'helperSha256':_sha(__file__),'fps':30,'fpsBase':1,
        'framePattern':'frame-%04d.png','frames':rows,'mode':mode,'resolution':[480,540],'samples':12,
        'actions':actions,'cloth':cloth,'cache':CACHE,'images':images,'checks':checks,
        'sourceSavedByRender':False,'continuousDurationSeconds':181/30,
        'scope':'Actual portrait performance frames: authored silent expressions and embedded matched twin-tail playback. Static original skirt; no walk cloth or Python hair handler. Requires visual review.'}
    with (out/'render-report.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    if mode=='sparse':
        with proof.open('x',encoding='utf-8') as f:json.dump(dict(report,status='sparse-rendered'),f,ensure_ascii=False,indent=2)
    print('R08_FACE_PERFORMANCE_COMPLETE '+json.dumps({'mode':mode,'frames':len(rows),'folder':out.relative_to(p).as_posix()}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','sparse','continuous'])
    parser.add_argument('--label',default='r08')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    prepare() if args.mode=='prepare' else render(args.mode,args.label)
