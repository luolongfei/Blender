"""One actual R08 still per explicit background invocation; never saves Blend.

With reference-face-flow-r08.blend open in a disposable background process:
  --python reference_final_stills_r08.py -- --case front
Cases: front, quarter, side, back, hand, hand-right, portrait.
Use --label final-stills-r08b for a new review; existing PNG/JSON are never
overwritten. Optional --warm is portrait-only and only lowers green/blue light
components. No extra exposure, depth of field, mesh edit or production save.
"""
import argparse, hashlib, json, math, re, runpy, sys
from pathlib import Path
import numpy as np
try:
    import bpy
    from mathutils import Matrix,Vector
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SOURCE='work/reconstruction-r02/reference-face-flow-r08.blend'
FULL_VIEWS={'front':(0.,-1.,0.),'quarter':(.6,-.8,0.),'side':(1.,0.,0.),'back':(0.,1.,0.)}
CASES=tuple(FULL_VIEWS)+('hand','hand-right','portrait')


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _clear_action(owner,records):
    ad=owner.animation_data
    if not ad:return
    if ad.action:
        records.append({'owner':owner.name,'action':ad.action.name})
        ad.action.use_fake_user=True;ad.action=None
    for track in ad.nla_tracks:track.mute=True


def _neutral(rig):
    scene=bpy.context.scene;cleared=[];disabled_hair=[]
    if 'R04_native_hair_enabled' in rig:rig['R04_native_hair_enabled']=0.
    _clear_action(rig,cleared)
    for keys in bpy.data.shape_keys:_clear_action(keys,cleared)
    # In this disposable still process explicitly mute the two visibility
    # drivers as well, so stale driver evaluation cannot enable a mismatched
    # cache after the pose changes. Native strand points remain untouched.
    for obj in scene.objects:
        if not obj.get('r04_native_hair_owner'):continue
        for modifier in obj.modifiers:
            if modifier.type!='NODES' or not modifier.name.startswith('R04 Native hair playback'):continue
            for prop in ('show_viewport','show_render'):
                fc=obj.animation_data.drivers.find(modifier.path_from_id(prop)) if obj.animation_data else None
                if fc:fc.mute=True
                setattr(modifier,prop,False)
            disabled_hair.append({'object':obj.name,'modifier':modifier.name})
    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    master=rig.pose.bones.get('CTRL_R02_Master')
    if master:
        for name in ('leg_L','leg_R','arm_L','arm_R'):
            if 'ik_'+name in master:master['ik_'+name]=0.
    for prop in ('R03_look_yaw','R03_look_pitch'):
        if prop in rig:rig[prop]=0.
    keys=bpy.data.objects['R01 Head'].data.shape_keys.key_blocks
    expressions=('R02_blink_L','R02_blink_R','R02_jaw_open','R02_mouth_wide','R02_pucker',
        'R02_lip_press','R02_frown','R02_brow_raise','R02_brow_furrow','R03_squint_L','R03_squint_R')
    for name in expressions:
        if name in keys:keys[name].value=0.
    keys['R02_smile'].value=.45
    # The accepted likeness, gentle smile, R07 and R08 sculpt keys stay active.
    for name in ('R07_cheek_surface_continuity','R08_cheek_perioral_transition'):
        if name not in keys or abs(keys[name].value-1.)>1e-6:raise RuntimeError('Accepted R08 face missing: '+name)
    static=('Skirt_Sixteen_Pleats','Skirt_Turned_Hem_Stitch','Skirt_Center_Back_Seam')
    if static[0] not in bpy.data.objects:raise RuntimeError('Original static skirt missing')
    cloth=[]
    for obj in scene.objects:
        dynamic=(any(m.type=='CLOTH' for m in obj.modifiers) or obj.name in ('R02 Cloth Skirt control','R06 Walk cloth')
            or ('Native' in obj.name and 'Cloth' in obj.name and 'playback' in obj.name))
        if dynamic or obj.name in static:
            if dynamic and obj.name in static:raise RuntimeError('Original static skirt unexpectedly simulated')
            obj.hide_render=dynamic;obj.hide_viewport=dynamic;obj.hide_set(dynamic)
            cloth.append({'name':obj.name,'hidden':dynamic})
    rig.data.pose_position='POSE';scene.frame_set(1);rig.update_tag();bpy.context.view_layer.update()
    helper=runpy.run_path(str(Path(__file__).with_name('head_neck_pose_review_r07.py')))
    pose=helper['_pose'](rig,'neutral')
    return {'clearedActions':cleared,'clothVisibility':cloth,'nativeHairEnabled':float(rig.get('R04_native_hair_enabled',0)),
            'temporarilyDisabledHairModifiers':disabled_hair,
            'pose':pose,'expressionValues':{k.name:float(k.value) for k in keys if k.name.startswith(('R02_','R03_','R04_','R05_','R07_','R08_'))}}


def _portrait_roll(rig,degrees=-8.):
    """World -Y-axis roll: screen-right eye rises, head top leans screen-left."""
    head=rig.pose.bones['head']
    if head.constraints:raise RuntimeError('Unexpected head constraints require explicit pose review')
    before=rig.matrix_world@head.matrix;pivot=rig.matrix_world@head.head
    rotation=Matrix.Rotation(math.radians(degrees),4,'Y')
    desired_world=Matrix.Translation(pivot)@rotation@Matrix.Translation(-pivot)@before
    desired=rig.matrix_world.inverted()@desired_world
    head.matrix_basis=head.bone.convert_local_to_pose(desired,head.bone.matrix_local,
        parent_matrix=head.parent.matrix,parent_matrix_local=head.parent.bone.matrix_local,invert=True)
    rig.update_tag();bpy.context.view_layer.update();after=rig.matrix_world@head.head
    drift=float((after-pivot).length)
    if drift>2e-6:raise RuntimeError(f'Head pivot moved during roll: {drift*1000:.6f} mm')
    top=rotation.to_3x3()@Vector((0,0,1));right=rotation.to_3x3()@Vector((1,0,0))
    if top.x>=0 or right.z<=0:raise RuntimeError('Portrait roll sign is opposite the reference')
    return {'worldAxis':[0,1,0],'degrees':degrees,'pivotWorldBefore':list(pivot),'pivotWorldAfter':list(after),
        'pivotDriftMeters':drift,'worldTopDirectionAfter':list(top),'worldScreenRightDirectionAfter':list(right),
        'scope':'Temporary art-display pose around the calibrated head pivot; no rest/mesh change'}


def _attached(obj,rig):
    parent=obj.parent
    while parent:
        if parent==rig:return True
        parent=parent.parent
    return any(m.type=='ARMATURE' and m.object==rig for m in obj.modifiers)


def _bounds(objects):
    graph=bpy.context.evaluated_depsgraph_get();points=[]
    for obj in objects:
        ev=obj.evaluated_get(graph)
        bounds=[Vector(v) for v in ev.bound_box]
        if all(v==Vector((-1,-1,-1)) for v in bounds):continue
        points.extend(ev.matrix_world@v for v in bounds)
    if not points:raise RuntimeError('No evaluated character bounds')
    return points


def _aim(camera,position,target,up=None):
    camera.location=position;direction=Vector(target)-camera.location
    if up is None:camera.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
    else:
        z=-direction.normalized();y=(up-z*up.dot(z)).normalized();x=y.cross(z).normalized();y=z.cross(x).normalized()
        camera.rotation_euler=Matrix((x,y,z)).transposed().to_euler()
    bpy.context.view_layer.update()


def _fit_scale(camera,points,padding=1.08):
    camera.data.ortho_scale=1.
    frame=camera.data.view_frame(scene=bpy.context.scene)
    halfx=max(abs(v.x) for v in frame);halfy=max(abs(v.y) for v in frame)
    inverse=camera.matrix_world.inverted();local=[inverse@p for p in points]
    return padding*max(max(abs(v.x)/halfx for v in local),max(abs(v.y)/halfy for v in local))


def _hand_camera(camera,rig,side):
    prefix='left' if side=='L' else 'right';graph=bpy.context.evaluated_depsgraph_get()
    hand=bpy.data.objects.get('Hand '+side+' · continuous anatomy')
    if hand is None or hand.hide_render:raise RuntimeError('Visible continuous hand required')
    wrist=rig.matrix_world@rig.pose.bones[prefix+'Hand'].head
    tip=rig.matrix_world@rig.pose.bones[prefix+'MiddleDistal'].tail
    index=rig.matrix_world@rig.pose.bones[prefix+'IndexProximal'].head
    little=rig.matrix_world@rig.pose.bones[prefix+'LittleProximal'].head
    axis=(tip-wrist).normalized();across=(little-index).normalized()
    nails=[];normals=[];centers=[]
    for finger in ('Index','Middle','Ring','Little'):
        obj=bpy.data.objects.get('Nail '+side+' '+finger+' · polished')
        if obj is None or obj.hide_render:raise RuntimeError('Visible original nail required: '+finger)
        ev=obj.evaluated_get(graph);mesh=ev.to_mesh()
        try:
            # The first source-count polygons are the upper plate. Including
            # Solidify's underside would cancel the normals of the closed shell.
            upper_count=len(obj.data.polygons)
            if len(mesh.polygons)<upper_count:raise RuntimeError('Unexpected nail topology')
            normals.append((ev.matrix_world.to_3x3().inverted().transposed()@
                sum((p.normal*p.area for p in mesh.polygons[:upper_count]),Vector())).normalized())
            centers.append(ev.matrix_world@mesh.vertices[0].co)
        finally:ev.to_mesh_clear()
        nails.append(obj)
    normal=sum(normals,Vector()).normalized()
    if abs(normal.dot(axis))>.6:raise RuntimeError('Nail normal does not describe the dorsal hand surface')
    target=(wrist+tip)*.5
    position=target+normal*.65-axis*.12+across*.04
    _aim(camera,position,target,axis)
    camera.data.ortho_scale=_fit_scale(camera,_bounds([hand]+nails),1.12)
    visibility=[]
    for obj,point in zip(nails,centers):
        direction=point-camera.location
        hit,_,_,_,first,_=bpy.context.scene.ray_cast(graph,camera.location,direction.normalized(),distance=direction.length+.003)
        name=first.original.name if hit else None
        visibility.append({'nail':obj.name,'firstVisibleObject':name,'centerRayVisible':name==obj.name})
    if sum(v['centerRayVisible'] for v in visibility)<3:
        raise RuntimeError('Hand camera does not expose at least three nail centers: '+json.dumps(visibility))
    return {'target':list(target),'boneAnchors':{'wrist':list(wrist),'middleDistalTip':list(tip),
        'indexKnuckle':list(index),'littleKnuckle':list(little)},'dorsalNormal':list(normal),'nailCenterVisibility':visibility}


def run(case,source_relative=SOURCE,label='final-stills-r08',samples=None,warm=False):
    if bpy is None or not bpy.app.background:raise RuntimeError('Disposable background Blender process required')
    if case not in CASES:raise ValueError('Select exactly one named case')
    if warm and case!='portrait':raise ValueError('--warm applies only to portrait')
    if not re.fullmatch(r'[a-z0-9-]+',label):raise ValueError('Safe lowercase output label required')
    p=project_path();source=(p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not source.name.endswith('r08.blend') or source!=Path(bpy.data.filepath).resolve():raise RuntimeError('Exact independent R08 source required')
    if bpy.data.is_dirty or bpy.context.mode!='OBJECT':raise RuntimeError('Fresh saved source in Object mode required')
    for handlers in (bpy.app.handlers.frame_change_pre,bpy.app.handlers.frame_change_post):
        if any(getattr(h,'_r02_secondary_hair',False) for h in handlers):raise RuntimeError('Old Python hair handler present')
    out=p/'previews/reconstruction-r02'/label;png=out/(case+'.png');report_path=out/(case+'-report.json')
    if png.exists() or report_path.exists():raise RuntimeError('Existing case output retained; select a new --label')
    source_sha=_sha(source);rig=bpy.data.objects['Character_Rig']
    if 'R06_head_pivot_calibration' not in rig or np.max(np.abs(np.asarray(rig.matrix_world)-np.eye(4)))>1e-6:
        raise RuntimeError('Expected calibrated R08 head and identity rig object transform')
    setup=_neutral(rig);pose={'name':'neutral','headRollWorldDegrees':0.}
    if case=='portrait':pose=_portrait_roll(rig)
    scene=bpy.context.scene
    for light in (o for o in scene.objects if o.type=='LIGHT'):
        light.data.color=(1.,.96,.90) if warm else (1.,1.,1.)
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=samples or (24 if case in FULL_VIEWS else 32)
    if scene.cycles.samples not in (24,32):raise ValueError('Use 24 or 32 samples')
    scene.cycles.use_denoising=True;scene.render.threads_mode='FIXED';scene.render.threads=6
    scene.render.use_sequencer=False;scene.render.use_file_extension=True;scene.render.use_overwrite=False
    scene.render.resolution_percentage=100;scene.render.film_transparent=False
    scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0.;scene.view_settings.gamma=1.
    if hasattr(scene.render.image_settings,'media_type'):scene.render.image_settings.media_type='IMAGE'
    scene.render.image_settings.file_format='PNG'
    scene.render.resolution_x=960 if case in FULL_VIEWS else 1200
    scene.render.resolution_y=1200
    camera=bpy.data.objects.new('R08 Final still camera',bpy.data.cameras.new('R08 Final still camera'))
    scene.collection.objects.link(camera);scene.camera=camera;camera.data.type='ORTHO';camera.data.dof.use_dof=False
    detail={}
    if case in FULL_VIEWS:
        actor=[o for o in scene.objects if o.type in ('MESH','CURVE','CURVES') and not o.hide_render and _attached(o,rig)]
        points=_bounds(actor);arr=np.asarray(points);target=Vector((arr.min(axis=0)+arr.max(axis=0))*.5)
        scales=[]
        for direction in FULL_VIEWS.values():
            _aim(camera,target+Vector(direction)*4.5,target);scales.append(_fit_scale(camera,points))
        _aim(camera,target+Vector(FULL_VIEWS[case])*4.5,target);camera.data.ortho_scale=max(scales)
        detail={'target':list(target),'sharedFullBodyScale':max(scales),'actorBoundsObjects':[o.name for o in actor]}
    elif case in ('hand','hand-right'):
        detail=_hand_camera(camera,rig,'L' if case=='hand' else 'R')
    else:
        target=Vector((-.010,-.005,1.365));_aim(camera,(0,-3,1.365),target);camera.data.ortho_scale=.43
        detail={'target':list(target),'referenceIntent':'Head leans screen-left; screen-right eye higher; unchanged blouse, face and twin-tail geometry'}
    out.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(png)
    bpy.ops.render.render(write_still=True)
    if not png.is_file() or not png.stat().st_size:raise RuntimeError('Expected still PNG missing')
    if _sha(source)!=source_sha:raise RuntimeError('Saved source changed during review')
    lights=[{'name':o.name,'type':o.data.type,'energy':o.data.energy,'color':list(o.data.color),
        'matrixWorld':[list(r) for r in o.matrix_world],'hideRender':o.hide_render,
        'size':getattr(o.data,'size',None)} for o in scene.objects if o.type=='LIGHT']
    world={'name':scene.world.name if scene.world else None,'backgrounds':[]}
    if scene.world and scene.world.use_nodes:
        world['backgrounds']=[{'name':n.name,'color':list(n.inputs['Color'].default_value),
            'strength':float(n.inputs['Strength'].default_value)} for n in scene.world.node_tree.nodes if n.type=='BACKGROUND']
    report={'source':source.relative_to(p).as_posix(),'sourceSha256':source_sha,'sourceUnchangedOnDisk':True,
        'case':case,'image':png.relative_to(p).as_posix(),'imageSha256':_sha(png),'blender':bpy.app.version_string,
        'pose':pose,'neutralSetup':setup,'poseBoneMatrices':{b.name:[list(r) for r in b.matrix_basis] for b in rig.pose.bones},
        'camera':{'type':'ORTHO','matrixWorld':[list(r) for r in camera.matrix_world],
            'orthoScale':camera.data.ortho_scale,'depthOfField':False,**detail},'lights':lights,'world':world,
        'resolution':[scene.render.resolution_x,scene.render.resolution_y],'samples':scene.cycles.samples,
        'engine':'Cycles CPU','colorManagement':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':0.,'gamma':1.},
        'portraitWarmLight':warm,'savedBlend':False,'geometryEdited':False,
        'scope':'Actual static appearance/inspection render from the independent R08 source. Neutral full-body cases share lighting/framing. Static original skirt and authored-rest hair replace unmatched motion caches. Portrait roll is a temporary display pose, not a new model or acceptance claim.'}
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R08_FINAL_STILL '+json.dumps({'case':case,'image':report['image'],'report':report_path.relative_to(p).as_posix()}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=CASES)
    parser.add_argument('--source',default=SOURCE);parser.add_argument('--label',default='final-stills-r08')
    parser.add_argument('--samples',type=int,choices=[24,32]);parser.add_argument('--warm',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    run(args.case,args.source,args.label,args.samples,args.warm)
