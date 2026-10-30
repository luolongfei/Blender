"""One complete front portrait: saved skin candidate + balanced light + optional smile.

Strict fresh-source checks run before all temporary changes. Reuses v6 actual
render-branch bounds and the shared .55 m camera; never crops head/tails. This
is a combined preview, not a single-variable comparison or final acceptance.
No static sculpt-key list: all non-expression keys retain actual frame-one
values, including later chin sculpts. No source Blend is saved.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HERE=Path(__file__).resolve().parent
HEAD='R11 Female head study'
CAMERA_SPEC='previews/reconstruction-r02/adult-full-groom-v5-r11/full-groom-front.json'
CONTROLS=('R11 gentle smile','R11 natural soft smile','R11 warm closed smile','R11 blink L','R11 blink R','R11 jaw open')
INTERNAL=tuple(
    ['R11 blink path '+s+' '+str(i).zfill(2) for s in ('L','R') for i in range(1,16)]+
    ['R11 blink closure '+s for s in ('L','R')]+
    ['R11 blink closure path '+s+' '+str(i).zfill(2) for s in ('L','R') for i in range(1,16)]+
    ['R11 smile lower lid '+s for s in ('L','R')]+
    ['R11 warm smile lower lid '+s for s in ('L','R')])
WARM_INTERNAL=('R11 warm smile lower lid L','R11 warm smile lower lid R')
DEPENDENCIES={
    'adult_full_groom_review_v6_r11.py':'c819e1788a3f76b24cda692e11bc6e10a87f5eb49fe5e264b395114ef1685d96',
    'adult_balanced_lighting_review_r11.py':'c005ad95cf9587ab490340f6c218b77af7737a0153dfe78edca43d1f82dfc7e2',
    'adult_neutral_lighting_review_r11_v2.py':'5378ca57e92d6edbdb2554723e36f8bc1cfd0ed1f85f57cb1d210ceeb473521e'}


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def _api(name):return runpy.run_path(str(HERE/(name+'.py')))
def _values(keys):return {k.name:float(k.value) for k in keys.key_blocks}
def _structure(keys):
    h=hashlib.sha256()
    for k in keys.key_blocks:
        h.update(json.dumps([k.name,k.relative_key.name,k.vertex_group,k.mute,k.slider_min,k.slider_max,k.interpolation]).encode())
        a=np.empty(len(k.data)*3,np.float32);k.data.foreach_get('co',a);h.update(a.tobytes())
    return h.hexdigest()


def _verify_expression(keys,warm,preserved):
    values=_values(keys); expected={n:0. for n in CONTROLS+INTERNAL if n in values}
    expected['R11 warm closed smile']=warm
    for n in WARM_INTERNAL:
        if n not in values:raise RuntimeError('Required actual warm-smile derived key missing: '+n)
        expected[n]=warm
    bad={n:[target,values.get(n)] for n,target in expected.items() if n not in values or not np.isfinite(values[n]) or abs(values[n]-target)>1e-7}
    if bad:raise RuntimeError('Actual expression values differ from requested warm smile: '+json.dumps(bad))
    changed={n:[v,values.get(n)] for n,v in preserved.items() if n not in values or not np.isfinite(values[n]) or abs(values[n]-v)>1e-7}
    if changed:raise RuntimeError('A preserved sculpt/other key changed: '+json.dumps(changed))
    return values


@contextlib.contextmanager
def expression_session(bpy,warm,audit):
    if not np.isfinite(warm) or not 0.<=warm<=.6:raise ValueError('Restrained warm smile range is 0..0.6')
    scene=bpy.context.scene;keys=bpy.data.objects[HEAD].data.shape_keys
    if keys is None or not keys.use_relative:raise RuntimeError('Existing relative adult keys required')
    for n in CONTROLS[:-1]:
        if n not in keys.key_blocks:raise RuntimeError('Required public expression control missing: '+n)
    bpy.context.view_layer.update();entry_frame=(scene.frame_current,scene.frame_subframe);entry=_values(keys)
    geometry=_structure(keys);ad=keys.animation_data;action=ad.action if ad else None
    slot=getattr(ad,'action_slot',None) if ad else None;use_nla=ad.use_nla if ad else None
    tracks=[(t,t.mute) for t in ad.nla_tracks] if ad else [];drivers=[(f,f.mute) for f in ad.drivers] if ad else []
    controller_paths={keys.key_blocks[n].path_from_id('value') for n in CONTROLS if n in keys.key_blocks}
    audit.update(entryFrame=list(entry_frame),entryKeyValues=entry,entryAction=action.name if action else None,
        entryActionSlot=getattr(slot,'identifier',None),entryUseNla=use_nla,
        entryDriverMute=[{'path':f.data_path,'mute':m} for f,m in drivers],requestedWarmSmile=warm,
        publicControlsOnly=True,staticKeyEnumerationUsed=False,internalDriversKept=True,restored=False)
    try:
        scene.frame_set(1);bpy.context.view_layer.update();frame_one=_values(keys)
        if not all(np.isfinite(v) for v in frame_one.values()):raise RuntimeError('Non-finite source key value')
        preserved={n:v for n,v in frame_one.items() if n not in CONTROLS and n not in INTERNAL}
        audit.update(frameOneKeyValues=frame_one,preservedOtherKeys=preserved)
        if ad:ad.action=None;ad.use_nla=False
        for f,_ in drivers:
            if f.data_path in controller_paths:f.mute=True
        for n,v in preserved.items():keys.key_blocks[n].value=v
        for n in CONTROLS:
            if n in keys.key_blocks:keys.key_blocks[n].value=warm if n=='R11 warm closed smile' else 0.
        keys.update_tag();bpy.context.view_layer.update()
        audit['actualRenderKeyValues']=_verify_expression(keys,warm,preserved)
        yield audit
        _verify_expression(keys,warm,preserved)
    finally:
        errors=[]
        def attempt(label,fn):
            try:fn()
            except Exception as exc:errors.append(label+': '+str(exc))
        if ad:
            attempt('action',lambda:setattr(ad,'action',action))
            if action is not None and hasattr(ad,'action_slot'):attempt('action slot',lambda:setattr(ad,'action_slot',slot))
            attempt('NLA',lambda:setattr(ad,'use_nla',use_nla))
        for f,m in drivers:attempt('driver '+f.data_path,lambda f=f,m=m:setattr(f,'mute',m))
        attempt('frame',lambda:scene.frame_set(entry_frame[0],subframe=entry_frame[1]))
        for n,v in entry.items():attempt('key '+n,lambda n=n,v=v:setattr(keys.key_blocks[n],'value',v))
        attempt('key tag',keys.update_tag);attempt('evaluation',bpy.context.view_layer.update)
        restored=_values(keys)
        if set(restored)!=set(entry) or any(abs(restored.get(n,float('inf'))-v)>1e-7 for n,v in entry.items()):errors.append('Key values did not restore')
        if ad and (ad.action!=action or getattr(ad,'action_slot',None)!=slot or ad.use_nla!=use_nla):errors.append('Action/slot/NLA did not restore')
        if any(t.mute!=m for t,m in tracks) or any(f.mute!=m for f,m in drivers):errors.append('Driver/NLA track mute changed')
        if (scene.frame_current,scene.frame_subframe)!=entry_frame or _structure(keys)!=geometry:errors.append('Frame/key geometry changed')
        audit.update(restoredKeyValues=restored,restorationErrors=errors,restored=not errors)
        if errors:raise RuntimeError('Portrait expression restoration failed: '+'; '.join(errors))


def review(source_relative,expected_sha,output_relative,width=1000,samples=24,warm=0.,camera_spec=CAMERA_SPEC):
    import bpy
    from mathutils import Vector
    P=project_path().resolve();source=(P/source_relative).resolve();source.relative_to(P/'work/reconstruction-r02')
    out=(P/output_relative).resolve();out.relative_to(P/'previews/reconstruction-r02')
    # This guard deliberately precedes expression, lighting, framing or camera mutations.
    if source.suffix!='.blend' or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or _sha(source)!=expected_sha:
        raise RuntimeError('Fresh exact independently saved source required before combined preview changes')
    if width not in (800,1000,1200) or not 16<=samples<=64:raise ValueError('Width 800/1000/1200 and 16..64 samples required')
    if not np.isfinite(warm) or not 0.<=warm<=.6:raise ValueError('Warm smile must be 0..0.6')
    image_path=out/'combined-portrait-front.png';report_path=image_path.with_suffix('.json');session_path=out/'combined-portrait-session.json'
    if any(p.exists() for p in (image_path,report_path,session_path)):raise RuntimeError('Preserve prior combined portrait outputs')
    dependencies={}
    for name,expected in DEPENDENCIES.items():
        actual=_sha(HERE/name)
        if expected is not None and actual!=expected:raise RuntimeError('Frozen dependency changed: '+name)
        dependencies[name]=actual
    spec_path=(P/camera_spec).resolve();spec_path.relative_to(P/'previews/reconstruction-r02')
    spec=json.loads(spec_path.read_text(encoding='utf-8'))['camera']
    if spec['sensorFit']!='VERTICAL' or spec['depthOfField'] is not False or abs(spec['orthographicScale']-.55)>1e-8:
        raise RuntimeError('Recorded .55 m vertical complete-portrait camera required')
    spec=dict(spec);spec['resolution']=[width,round(width*7/6)];spec['pixelAspect']=[1.,1.]
    groom=_api('adult_full_groom_review_v6_r11');light=_api('adult_balanced_lighting_review_r11');base=light['_base']()
    scene=bpy.context.scene;original_camera=scene.camera;camera=None;expression={};lighting=None
    original_lights=base._light_records(scene);original_world=base._world_record(scene.world);colour=base._colour(scene)
    audit={'source':source_relative,'sourceSha256':expected_sha,'helperSha256':_sha(__file__),
        'dependencies':dependencies,'expression':expression,'savedBlend':False,'restored':False}
    report=None;out.mkdir(parents=True,exist_ok=True)
    try:
        with expression_session(bpy,warm,expression):
            # Measure full render branch after expression; original GN trees restore before rendering.
            before=base._nonlight_contract(scene,bpy.data.objects[HEAD])
            rows,geometry=groom['measure'](scene)
            if base._nonlight_contract(scene,bpy.data.objects[HEAD])!=before:raise RuntimeError('Full-hair bounds changed original non-light data')
            target=np.asarray(spec['target'],float);cloud=groom['corners'](rows);scale=float(spec['orthographicScale'])
            portrait_width=scale*spec['resolution'][0]/spec['resolution'][1]
            x=np.abs(cloud[:,0]-target[0]);y=np.abs(cloud[:,2]-target[2])
            if x.max()>portrait_width*.5-.002 or y.max()>scale*.5-.002:
                raise RuntimeError('Actual head/tails would be cropped by shared .55 m frame; do not render a cropped portrait')
            camera=bpy.data.objects.new('R11 combined portrait camera',bpy.data.cameras.new('R11 combined portrait camera'))
            scene.collection.objects.link(camera);camera.location=Vector(target+np.array([0.,-float(spec['distance']),0.]))
            camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
            camera.data.type='ORTHO';camera.data.sensor_fit='VERTICAL';camera.data.ortho_scale=scale;camera.data.dof.use_dof=False;scene.camera=camera
            scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.seed=67
            scene.cycles.use_animated_seed=False;scene.cycles.use_denoising=True
            scene.render.resolution_x=spec['resolution'][0];scene.render.resolution_y=spec['resolution'][1];scene.render.resolution_percentage=100
            scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1.
            scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.filepath=str(image_path)
            bpy.context.view_layer.update();frame=np.asarray([list(v) for v in camera.data.view_frame(scene=scene)])
            if abs(float(np.ptp(frame[:,1]))-scale)>1e-5 or abs(float(np.ptp(frame[:,0]))-portrait_width)>1e-5:raise RuntimeError('Actual portrait camera differs from computed frame')
            plan=light['make_plan'](bpy);protected=base._nonlight_contract(scene,bpy.data.objects[HEAD])
            raw_skin=scene.get('R11_natural_skin_refinement_candidate')
            skin=json.loads(raw_skin) if isinstance(raw_skin,str) else None
            with light['diagnostic_lights'](bpy,plan) as lighting:
                if base._colour(scene)!=colour or base._nonlight_contract(scene,bpy.data.objects[HEAD])!=protected:raise RuntimeError('Lighting context altered protected character or colour management')
                report={'recordedAtUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'source':source_relative,'sourceSha256':expected_sha,'helperSha256':_sha(__file__),'dependencies':dependencies,
                    'case':'front','frame':1,'camera':spec,'cameraSpecReused':{'path':camera_spec,'sha256':_sha(spec_path)},
                    'actualCameraMatrix':[list(row) for row in camera.matrix_world],'actualCameraLocalFrame':frame.tolist(),
                    'actualFramingObjects':rows,'denseGeometryMeasurement':geometry,
                    'minimumBoundsBorderMeters':{'horizontal':float(portrait_width*.5-x.max()),'vertical':float(scale*.5-y.max())},
                    'actualLightsDuringRender':base._light_records(scene),'actualWorldDuringRender':base._world_record(scene.world),
                    'lightingChange':lighting,'balancedPlan':plan,'colourManagementUnchanged':colour,
                    'savedSkinCandidate':skin,'skinCandidatePresentInSavedSource':skin is not None,
                    'requestedWarmSmile':warm,'actualKeyValuesDuringRender':_values(bpy.data.objects[HEAD].data.shape_keys),
                    'combinedChanges':['saved-source skin candidate' if skin else 'saved-source skin retained','temporary balanced lighting','temporary warm smile '+str(warm)],
                    'singleFactorComparison':False,'renderedImages':1,'engine':'Cycles CPU','samples':samples,'seed':67,'denoising':True,
                    'savedBlend':False,'appearanceAccepted':False,
                    'scope':'Combined complete-head portrait. Saved skin, temporary balanced lighting and requested expression are shown together. Changes cannot be attributed to material alone. No physical-motion or final appearance acceptance.'}
                bpy.ops.render.render(write_still=True)
                if not image_path.is_file() or _sha(source)!=expected_sha:raise RuntimeError('Render or unchanged source check failed')
                if base._colour(scene)!=colour or base._nonlight_contract(scene,bpy.data.objects[HEAD])!=protected:raise RuntimeError('Render changed protected character state')
                report['imageSha256']=_sha(image_path);report['imageStats']=base._render_stats(bpy,image_path)
        if base._light_records(scene)!=original_lights or base._world_record(scene.world)!=original_world or base._colour(scene)!=colour:raise RuntimeError('Original lighting/colour restoration failed')
        audit.update(originalLightingRestored=True,restored=True)
    except BaseException as exc:
        audit['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        scene.camera=original_camera
        if camera is not None:
            data=camera.data;bpy.data.objects.remove(camera,do_unlink=True)
            if data.users==0:bpy.data.cameras.remove(data)
        audit['sourceFileUnchanged']=_sha(source)==expected_sha
        audit['originalLightingRestored']=base._light_records(scene)==original_lights and base._world_record(scene.world)==original_world
        audit['renderSettingsScope']='Only disposable background-session render settings remain; no Blend save. Original camera, expression, lights and world restore.'
        with session_path.open('x',encoding='utf-8') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
    if not audit['restored'] or not expression.get('restored') or not audit['sourceFileUnchanged']:raise RuntimeError('Combined review restoration incomplete')
    report['sessionRestoration']=audit
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_COMBINED_PORTRAIT_DONE '+json.dumps({'image':str(image_path),'sourceSha256':expected_sha,'warmSmile':warm,'resolution':spec['resolution']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--expected-source-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--width',type=int,default=1000);parser.add_argument('--samples',type=int,default=24);parser.add_argument('--warm',type=float,default=0.);parser.add_argument('--camera-spec',default=CAMERA_SPEC)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    review(a.source,a.expected_source_sha256,a.output,a.width,a.samples,a.warm,a.camera_spec)
