"""One fresh-source R09 detail still per explicit background invocation.

Supports an adopted independent candidate or work/character.blend. Preserves
the real R09 body action at frame 1 and its matched native hair/cloth; never
calls the old _neutral/rest/static-cloth substitution path. No blend save.
Core views: hand-left, hand-right, eye-mouth and three fixed full-body views.
Additional bounded views: hairline, side-light, back-light, garment and a full
head/shoulder/twin-tail portrait using current evaluated frame-one geometry.
"""
import argparse
import contextlib
import hashlib
import io
import json
import re
import runpy
import sys
import time
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
CASES=('hand-left','hand-right','eye-mouth','face-side','hairline','side-light','back-light',
       'full-front','full-side','full-back','garment','shoulder-portrait')
FULL_DIRECTIONS={'full-front':(0.,-4.,.90),'full-side':(4.,0.,.90),'full-back':(0.,4.,.90)}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _write_new(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def _lights(scene):
    return [{'name':o.name,'type':o.data.type,'energy':o.data.energy,'colour':list(o.data.color),
             'matrixWorld':[list(row) for row in o.matrix_world],'hideRender':o.hide_render,
             'size':getattr(o.data,'size',None),'sizeY':getattr(o.data,'size_y',None),
             'useNodes':o.data.use_nodes}
            for o in scene.objects if o.type=='LIGHT']


def _world(scene):
    result={'name':scene.world.name if scene.world else None,'backgroundNodes':[]}
    if scene.world and scene.world.use_nodes:
        result['backgroundNodes']=[{'name':n.name,'colour':list(n.inputs['Color'].default_value),
            'strength':n.inputs['Strength'].default_value,
            'colourLinked':n.inputs['Color'].is_linked,'strengthLinked':n.inputs['Strength'].is_linked}
            for n in scene.world.node_tree.nodes if n.type=='BACKGROUND']
    return result


def _pose_state(bpy,rig):
    head=bpy.data.objects['R01 Head'];keys=head.data.shape_keys
    return {'bodyAction':rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None,
        'bodyActionSlot':getattr(getattr(rig.animation_data,'action_slot',None),'identifier',None),
        'headAction':keys.animation_data.action.name if keys.animation_data and keys.animation_data.action else None,
        'posePosition':rig.data.pose_position,'nativeHairEnabled':float(rig.get('R04_native_hair_enabled',0)),
        'keyValues':{key.name:float(key.value) for key in keys.key_blocks},
        'bonePoseMatrices':{bone.name:[list(row) for row in bone.matrix_basis] for bone in rig.pose.bones}}


def _matched_standing(bpy):
    """Only frame-one native parity, not a re-bake or alternate-pose setup."""
    merge=_api('cloth_merge_r09');C=merge['C'];native=merge['H']
    master=Path(bpy.data.filepath).resolve()==(project_path()/'work/character.blend').resolve()
    rig=C['_rig'](readonly_master=master)
    if not rig.animation_data or not rig.animation_data.action or rig.animation_data.action.name!=C['ACTION']:
        raise RuntimeError('Use the adopted matched R09 body action, not another clip or rest pose')
    scene=bpy.context.scene
    if abs(scene.render.fps/scene.render.fps_base-30)>1e-6:raise RuntimeError('Matched R09 source must retain 30 fps')
    scene.frame_set(1);bpy.context.view_layer.update()
    cloth=bpy.data.objects[C['NATIVE']]
    if cloth.hide_render or cloth.hide_viewport or cloth.hide_get() or cloth.parent!=rig:
        raise RuntimeError('The actual R09 native cloth must remain visible and attached')
    provenance=json.loads(cloth['r09_combined_cloth_provenance'])
    for name in provenance['hiddenOldClothObjects']:
        obj=bpy.data.objects.get(name)
        if obj and (not obj.hide_render or not obj.hide_viewport):raise RuntimeError('Historical/static skirt is visible: '+name)
    capture,points,matrix=C['_load'](project_path()/'work/reconstruction-r02/cloth-body-r09')
    if C['_action_hash'](rig.animation_data.action)!=capture['actionSha256']:
        raise RuntimeError('Current body action differs from recorded cloth')
    native_report=json.loads((project_path()/'work/reconstruction-r02/cloth-body-r09/native-report.json').read_text(encoding='utf-8'))
    if C['_topology'](cloth.data)!=native_report['topologySha256'] or C['_uv_hash'](cloth.data)!=native_report['uvSha256']:
        raise RuntimeError('Native cloth topology/UV differs from source capture')
    if not cloth.data.shape_keys or len(cloth.data.shape_keys.key_blocks)!=302:
        raise RuntimeError('Expected embedded Basis plus 301 cloth frames')
    cloth_action=cloth.data.shape_keys.animation_data.action
    if C['_action_hash'](cloth_action)!=native_report['shapeActionSha256']:raise RuntimeError('Native cloth shape action changed')
    modifiers=merge['modifier_record'](cloth)
    if modifiers!=provenance['verification']['modifiers']:raise RuntimeError('Native cloth surface settings changed')
    cloth_parity=C['_parity'](cloth,points,frames=(1.,))
    identity=merge['_matched_hair'](rig)
    guide=native['_active_guide'](rig);before_record=guide[native['RECORD']];record=json.loads(before_record)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            hair_parity=native['validate_evaluated'](cache_relative=record['cacheRelativePath'],frames=[1])
    finally:guide[native['RECORD']]=before_record
    for name,modifier_name in record['modifiers'].items():
        obj=bpy.data.objects[name];modifier=obj.modifiers[modifier_name]
        if obj.hide_render or not modifier.show_render or not modifier.show_viewport:
            raise RuntimeError('Actual matched tail playback is disabled: '+name)
    for handlers in (bpy.app.handlers.frame_change_pre,bpy.app.handlers.frame_change_post):
        if any(getattr(h,'_r02_secondary_hair',False) for h in handlers):raise RuntimeError('Unexpected old Python hair handler')
    return rig,{'frame':1,'bodyAction':C['ACTION'],'actionSha256':capture['actionSha256'],
        'clothObject':cloth.name,'clothPayloadSha256':capture['payloadSha256'],'clothPointParityMeters':cloth_parity,
        'matchedHair':identity,'hairPointParity':hair_parity,
        'staticHairOrClothSubstituted':False,'actionsCleared':False,
        'scope':'Actual matched standing frame only; does not certify the continuous clip.'}


def _eye_center(bpy,utility):
    points=[]
    for side in ('L','R'):
        obj=bpy.data.objects.get('R01 Sclera '+side)
        if obj is None or obj.hide_render:raise RuntimeError('Current visible eye geometry required')
        bounds=np.asarray([list(p) for p in utility['_bounds']([obj])])
        points.append((bounds.min(axis=0)+bounds.max(axis=0))*.5)
    center=np.mean(points,axis=0)
    if not .03<float(np.linalg.norm(points[0]-points[1]))<.12:
        raise RuntimeError('Unexpected actual inter-eye separation; inspect before framing')
    return center


def _shoulder_portrait_camera(camera,bpy,rig,utility):
    """Fit the actual head, shoulder caps and ALL native-tail points only.

    R01 Head contains merged body geometry, so its whole-object bounding box
    must never drive this camera. No whole-scene/whole-body fallback is used.
    """
    from mathutils import Vector
    scene=bpy.context.scene;scene.render.resolution_x=scene.render.resolution_y=768
    eyes=_eye_center(bpy,utility);graph=bpy.context.evaluated_depsgraph_get()
    shoulders=np.asarray([list(rig.matrix_world@rig.pose.bones[side+'UpperArm'].head)
                          for side in ('left','right')],dtype=float)
    shoulder_z=float(shoulders[:,2].mean());center_x=float(eyes[0])
    half_shoulders=float(np.max(np.abs(shoulders[:,0]-center_x)))
    if not .09<half_shoulders<.25 or not .10<eyes[2]-shoulder_z<.28:
        raise RuntimeError('Actual shoulder anchors do not match compact R09 head/shoulder framing')
    major_hair=('R06 attached scalp root substrate','R06 continuous scalp flow',
                'R03 Forelocks from crown','R06 full gathered roots L','R06 full gathered roots R')
    tails=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
    garments=('Blouse_Tucked_Bodice','Blouse_Puff_Sleeve_L','Blouse_Puff_Sleeve_R')
    names=('R01 Head',)+major_hair+tails+garments
    rows=[];limits=[];tail_records=[]
    for name in names:
        obj=bpy.data.objects.get(name)
        if obj is None or obj.hide_render or obj.hide_viewport or obj.hide_get():
            raise RuntimeError('Required current head/shoulder object is missing/hidden: '+name)
        evaluated=obj.evaluated_get(graph);matrix=np.asarray(evaluated.matrix_world,dtype=float)
        if evaluated.type=='CURVES':
            local=np.empty(len(evaluated.data.position_data)*3,np.float32)
            evaluated.data.position_data.foreach_get('vector',local);local=local.reshape(-1,3)
        elif evaluated.type=='MESH':
            mesh=evaluated.to_mesh()
            try:
                local=np.empty(len(mesh.vertices)*3,np.float32)
                mesh.vertices.foreach_get('co',local);local=local.reshape(-1,3)
            finally:evaluated.to_mesh_clear()
        else:raise RuntimeError('Unsupported evaluated portrait-bound object: '+name)
        points=local.astype(float)@matrix[:3,:3].T+matrix[:3,3]
        if not len(points) or not np.isfinite(points).all():raise RuntimeError('Invalid actual evaluated portrait points: '+name)
        count=len(points);region='all evaluated points'
        if name=='R01 Head':
            points=points[(points[:,2]>=eyes[2]-.16)&(np.abs(points[:,0]-center_x)<=half_shoulders+.025)]
            region='merged mesh: head/neck above eyeZ-160mm, inside shoulder-joint width+25mm; arms/body excluded'
        elif name in garments:
            points=points[(points[:,2]>=shoulder_z-.10)&(np.abs(points[:,0]-center_x)<=half_shoulders+.12)]
            region='actual upper bodice/shoulder cap above shoulderZ-100mm; lower sleeves/waist excluded'
        if not len(points):raise RuntimeError('Requested visible head/shoulder region has no points: '+name)
        low,high=points.min(0),points.max(0)
        rows.append({'object':name,'evaluatedPointCount':count,'framingPointCount':len(points),
                     'region':region,'worldBounds':[low.tolist(),high.tolist()]})
        limits.extend((low,high))
        if name in tails:
            curves=evaluated.data.curves
            if len(curves)!=2480 or any(c.points_length!=35 or c.first_point_index!=i*35 for i,c in enumerate(curves)):
                raise RuntimeError('Current native-tail evaluated topology differs from its preserved strands')
            strand=points.reshape(2480,35,3);roots=strand[:,:3].mean(1);tips=strand[:,-1]
            tail_records.append({'object':name,'pointCount':len(points),'allPointsIncluded':True,
                'medianPinCenterWorld':np.median(roots,axis=0).tolist(),
                'medianTipWorld':np.median(tips,axis=0).tolist(),
                'tipBoundsWorld':[tips.min(0).tolist(),tips.max(0).tolist()],
                'medianArcLengthMeters':float(np.median(np.linalg.norm(np.diff(strand,axis=1),axis=2).sum(1))),
                'geometrySource':'Actual evaluated native playback at frame 1; no raw-rest substitution'})
    limits=np.asarray(limits);low,high=limits.min(0),limits.max(0)
    target=Vector((center_x,float((low[1]+high[1])*.5),1.34))
    # A deliberate compact base framing, enlarged only by these head/shoulder
    # regions. A malformed cache cannot silently shrink the entire character.
    margin=.025
    required=2*max(abs(low[0]-target.x)+margin,abs(high[0]-target.x)+margin,
                   abs(low[2]-target.z)+margin,abs(high[2]-target.z)+margin)
    scale=max(.62,float(required))
    if scale>.74 or high[2]-low[2]>.62 or high[0]-low[0]>.69:
        raise RuntimeError('Current head/shoulder/tail bounds exceed compact portrait scope: '+json.dumps(
            {'bounds':[low.tolist(),high.tolist()],'requiredScale':scale,'objects':rows}))
    utility['_aim'](camera,target+Vector((0.,-3.,0.)),target);camera.data.ortho_scale=scale
    minimum_border=(scale*.5-max(abs(low[0]-target.x),abs(high[0]-target.x),
                                abs(low[2]-target.z),abs(high[2]-target.z)))*768/scale
    if minimum_border<20.:raise RuntimeError('Complete hair/shoulder bounds are too close to the portrait border')
    return {'target':list(target),'evaluatedEyeCenter':eyes.tolist(),
        'shoulderBoneAnchorsWorld':shoulders.tolist(),'wholeBodyBBoxUsed':False,
        'selectedCurrentObjects':rows,'portraitWorldBounds':[low.tolist(),high.tolist()],
        'allTailEvaluatedPointsIncluded':True,'actualNativeTailMeasurements':tail_records,
        'baseOrthographicScale':.62,'minimumProjectedBorderPixels':float(minimum_border),
        'scope':'Actual R09 matched frame-one head through chest and both complete tails; pose, native hair/cloth, lights and source remain unchanged. This camera is for assessing tail silhouette, not authorizing a reshape.'}


def _camera(case,bpy,rig,utility):
    from mathutils import Vector
    scene=bpy.context.scene
    camera=bpy.data.objects.new('R09 current detail camera',bpy.data.cameras.new('R09 current detail camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.data.type='ORTHO';camera.data.dof.use_dof=False;camera.data.clip_start=.01;camera.data.clip_end=100.
    if case in FULL_DIRECTIONS:
        scene.render.resolution_x,scene.render.resolution_y=640,848
        target=Vector((0.,0.,.80));utility['_aim'](camera,FULL_DIRECTIONS[case],target)
        camera.data.ortho_scale=1.90
        detail={'target':list(target),'fixedCharacterSpace':True,'fullBodyFitUsed':False}
    elif case.startswith('hand-'):
        scene.render.resolution_x=scene.render.resolution_y=768
        side='L' if case=='hand-left' else 'R';hand=bpy.data.objects.get('Hand '+side+' · continuous anatomy')
        if hand is None or hand.hide_render:raise RuntimeError('Expected current visible separate continuous hand: '+side)
        bounds=np.asarray([list(point) for point in utility['_bounds']([hand])])
        diagonal=float(np.linalg.norm(bounds.max(axis=0)-bounds.min(axis=0)))
        if not .06<diagonal<.35:raise RuntimeError('Hand-only bounds are invalid; no whole-body framing fallback')
        # Pure camera utility: real hand/finger bones, evaluated hand/nail
        # bounds and actual dorsal plate normals. Does not call _neutral().
        if 'R10_hbm_hand_provenance' in hand:
            detail=_api('hand_asset_fit_r10')['hand_camera'](camera,rig,side,utility)
        else:
            detail=utility['_hand_camera'](camera,rig,side)
        if not .07<camera.data.ortho_scale<.40:raise RuntimeError('Hand framing is not compact')
        detail.update(handObject=hand.name,handBoundsDiagonalMeters=diagonal,wholeBodyBBoxUsed=False)
    elif case=='shoulder-portrait':
        detail=_shoulder_portrait_camera(camera,bpy,rig,utility)
    elif case=='garment':
        scene.render.resolution_x,scene.render.resolution_y=640,848
        target=Vector((0.,0.,.94));utility['_aim'](camera,(2.4,-4.,1.10),target)
        camera.data.ortho_scale=1.20;detail={'target':list(target),'actualNativeCloth':True}
    else:
        scene.render.resolution_x=scene.render.resolution_y=768
        eyes=_eye_center(bpy,utility);target=Vector(eyes)
        if case=='eye-mouth':target.z-=.035;scale=.235;offset=Vector((0.,-3.,0.))
        elif case=='face-side':
            target.y+=.055;target.z-=.010;scale=.39;offset=Vector((3.,0.,0.))
        elif case=='hairline':target.z+=.065;scale=.265;offset=Vector((0.,-3.,.03))
        else:target.z+=.015;scale=.36;offset=Vector((.45,-3.,.025))
        utility['_aim'](camera,target+offset,target);camera.data.ortho_scale=scale
        detail={'target':list(target),'evaluatedEyeCenter':eyes.tolist(),'wholeBodyBBoxUsed':False}
    return camera,detail


def _light_variant(case,bpy,target,utility):
    from mathutils import Vector
    if case not in ('side-light','back-light'):return {'mode':'source-lights-unchanged'}
    lights=[obj for obj in bpy.context.scene.objects if obj.type=='LIGHT' and not obj.hide_render and obj.data.type in ('AREA','POINT','SPOT')]
    if not lights:raise RuntimeError('No existing local light available for bounded relighting')
    if any(obj.data.use_nodes for obj in lights):raise RuntimeError('Node-driven light power requires explicit handling; no silent bypass')
    key=max(lights,key=lambda obj:obj.data.energy)
    offset=Vector((-.75,-.30,.35) if case=='side-light' else (.35,.70,.45))
    key.matrix_world.translation=Vector(target)+offset
    key.rotation_euler=(Vector(target)-key.matrix_world.translation).to_track_quat('-Z','Y').to_euler()
    factor=.55 if case=='side-light' else .65
    for light in lights:
        if light!=key:light.data.energy*=factor
    return {'mode':case,'repositionedExistingLight':key.name,'keyPowerMultiplier':1.,
            'remainingLocalLightPowerMultiplier':factor,'colourChanged':False,'worldChanged':False,'exposureChanged':False}


def run(source_relative,case,label,samples=24):
    import bpy
    if case not in CASES or samples not in (24,32):raise ValueError('Choose one case and 24 or 32 samples')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,63}',label):raise ValueError('Use an explicit new safe review label')
    p=project_path().resolve();source=(p/source_relative).resolve();source.relative_to(p/'work')
    if source!=p/'work/character.blend':source.relative_to(p/'work/reconstruction-r02')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh exact candidate/master in background Object Mode required')
    source_sha=_sha(source);out=p/'previews/reconstruction-r02'/label
    png=out/(case+'.png');report_path=out/(case+'-report.json');scope_file=out/'review-source.json'
    if png.exists() or report_path.exists():raise RuntimeError('Preserve existing case images/reports; use a new label')
    scope={'source':source.relative_to(p).as_posix(),'sourceSha256':source_sha,'frame':1,'review':'R09 actual matched native body/hair/cloth details'}
    if out.exists():
        if not scope_file.is_file() or json.loads(scope_file.read_text(encoding='utf-8'))!=scope:
            raise RuntimeError('Existing label belongs to another source/review')
    else:out.mkdir(parents=True);_write_new(scope_file,scope)
    scene=bpy.context.scene;original_camera=scene.camera;original_frame=(scene.frame_current,scene.frame_subframe)
    old_lights=[(obj,obj.matrix_world.copy(),float(obj.data.energy),tuple(obj.data.color)) for obj in scene.objects if obj.type=='LIGHT']
    before_lights=_lights(scene);before_world=_world(scene)
    colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
            'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
    camera=None;started=time.monotonic()
    try:
        rig,matched=_matched_standing(bpy);before_pose=_pose_state(bpy,rig)
        if np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6:raise RuntimeError('Fixed review space expects observed identity rig transform')
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples;scene.cycles.use_denoising=True
        scene.render.threads_mode='FIXED';scene.render.threads=6;scene.render.use_sequencer=False
        scene.render.resolution_percentage=100;scene.render.use_file_extension=True;scene.render.use_overwrite=False
        scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
        utility=_api('reference_final_stills_r08');camera,framing=_camera(case,bpy,rig,utility)
        lighting=_light_variant(case,bpy,framing['target'],utility);bpy.context.view_layer.update()
        camera_record={'matrixWorld':[list(row) for row in camera.matrix_world],'orthographicScale':camera.data.ortho_scale,
                       'depthOfField':False,**framing}
        lights_during=_lights(scene);scene.render.filepath=str(png)
        bpy.ops.render.render(write_still=True)
        if not png.is_file() or not png.stat().st_size:raise RuntimeError('Expected actual rendered image missing')
        if _pose_state(bpy,rig)!=before_pose:raise RuntimeError('Matched action, pose or facial values changed during detail review')
        if _sha(source)!=source_sha:raise RuntimeError('Saved source changed during review')
        after_colour={'view':scene.view_settings.view_transform,'look':scene.view_settings.look,
                      'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma}
        if after_colour!=colour:raise RuntimeError('Source exposure/colour-management changed')
        report={**scope,'case':case,'image':png.relative_to(p).as_posix(),'imageSha256':_sha(png),
            'sourceUnchangedOnDisk':True,'savedBlend':False,'geometryEdited':False,'appearanceAccepted':False,
            'matchingStandingEvidence':matched,'pose':before_pose,'camera':camera_record,
            'lightsBefore':before_lights,'lightsDuring':lights_during,'lightingVariant':lighting,
            'world':before_world,'colourManagement':colour,'resolution':[scene.render.resolution_x,scene.render.resolution_y],
            'samples':samples,'blender':bpy.app.version_string,'engine':'Cycles CPU','wallSeconds':time.monotonic()-started,
            'scopeLimit':'Single actual R09 frame-one detail. No old rest/static skirt substitution; not full motion, contact, finger-flex or final likeness acceptance.'}
        _write_new(report_path,report)
        print('R09_FINAL_DETAIL '+json.dumps({'case':case,'image':report['image'],'report':report_path.relative_to(p).as_posix(),'sourceSha256':source_sha}),flush=True)
        return report
    finally:
        for light,matrix,energy,colour_value in old_lights:
            light.matrix_world=matrix;light.data.energy=energy;light.data.color=colour_value
        scene.camera=original_camera
        if camera is not None:
            data=camera.data;bpy.data.objects.remove(camera,do_unlink=True)
            if data.users==0:bpy.data.cameras.remove(data)
        scene.frame_set(original_frame[0],subframe=original_frame[1]);bpy.context.view_layer.update()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--label',required=True);parser.add_argument('--samples',type=int,choices=(24,32),default=24)
    args=parser.parse_args(argv);return run(args.source,args.case,args.label,args.samples)


if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
