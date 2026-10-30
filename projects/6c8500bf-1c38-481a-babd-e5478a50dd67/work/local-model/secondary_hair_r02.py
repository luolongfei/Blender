"""Small deterministic PBD guide caches for the two existing R02 native tails.

This file never saves a Blend, launches Blender, imports motion, or renders.
Call bake_actions() explicitly in a saved candidate. Then attach(cache_path)
and call apply_frame() after scene.frame_set() and before each render. Optional
preview handlers are session-only; the .blend alone does not contain the cache.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:  # Permit numerical checks in a normal Python process.
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

VERSION=1
RIG='Character_Rig'
OBJECTS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
MARKER='r02_secondary_hair_session'
PARAMS={'fps':30,'guides_per_tail':24,'substeps':4,'iterations':6,'pinned_points':3,
        'gravity':[0.0,0.0,-9.81],'damping_per_second':9.0,
        'root_frequency_hz':12.0,'tip_frequency_hz':4.0,'bend_stiffness':0.32,
        'collision_margin':0.0015,'contact_friction':0.18,'warmup_seconds':0.6,
        'loop_preroll_cycles':2,'max_cache_frames':600}
LIMITS=['Only the two twin tails move; scalp/fringe/temple strands stay attached to head.',
        'Art-directed elastic guide dynamics, not measured material properties or mocap.',
        'Head/shoulder/chest spheres approximate anatomy; no garment mesh or strand self-collision.',
        'Guide padding is conservative; interpolated individual strands still require visual collision checks.',
        'No guarantee of a seamless loop; inspect the measured end/start guide displacement.',
        'A cache is valid only for the recorded action, rest geometry, controls, and sampled body motion.',
        'Playback changes native curve positions in memory. Export/reopen requires explicit cache tooling.',
        'Render explicitly after apply_frame(); interactive handler/thread safety and motion blur require testing.']
_PLAYBACK=None


def _digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        q=np.ascontiguousarray(a)
        h.update(str(q.shape).encode());h.update(q.dtype.str.encode());h.update(q.tobytes())
    return h.hexdigest()


def _path(relative):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\','/').split('/'):
        raise ValueError('Use a project-relative path without ..')
    p=(project_path()/relative).resolve()
    p.relative_to((project_path()/'work').resolve())
    return p


def _guard(allow_authored_ik=False):
    if bpy is None: raise RuntimeError('This operation needs the existing Blender session')
    path=Path(bpy.data.filepath).resolve()
    if not bpy.data.filepath or path==(project_path()/'work/character.blend').resolve():
        raise RuntimeError('Use a saved independent candidate, never the formal main source')
    path.relative_to((project_path()/'work').resolve())
    rig=bpy.data.objects.get(RIG)
    if rig is None or rig.type!='ARMATURE' or rig.library: raise RuntimeError('Local Character_Rig required')
    scene=bpy.context.scene
    if abs(scene.render.fps/scene.render.fps_base-30)>1e-6:
        raise RuntimeError('The imported actions were made at 30 fps; use a 30 fps scene')
    if bpy.context.mode not in {'OBJECT','POSE'}: raise RuntimeError('Leave edit/sculpt mode first')
    master=rig.pose.bones.get('CTRL_R02_Master')
    if not allow_authored_ik and master and any(abs(float(master.get('ik_'+key,0)))>1e-7 for key in ('leg_L','leg_R','arm_L','arm_R')):
        raise RuntimeError('Use FK playback, or explicitly bake with allow_authored_ik=True for a controlled review action')
    if rig.animation_data and any(not t.mute for t in rig.animation_data.nla_tracks):
        raise RuntimeError('Active NLA needs a separate explicit bake scope')
    return rig


def _read_tail(name):
    o=bpy.data.objects.get(name)
    if o is None or o.type!='CURVES': raise RuntimeError('Native curves object missing: '+name)
    if o.library or o.data.library or o.data.users!=1: raise RuntimeError('Use local single-user hair data')
    if o.parent is None or o.parent.name!=RIG or o.parent_type!='BONE' or o.parent_bone!='head':
        raise RuntimeError('Expected existing head-bone attachment: '+name)
    if any(m.show_viewport or m.show_render for m in o.modifiers):
        raise RuntimeError('Integrate hair modifiers explicitly before caching: '+name)
    sizes=np.array([c.points_length for c in o.data.curves],dtype=np.int32)
    offsets=np.array([c.first_point_index for c in o.data.curves],dtype=np.int32)
    if len(sizes)<24 or np.any(sizes!=sizes[0]) or sizes[0]<8:
        raise RuntimeError('This cache requires equal-length existing strand point arrays')
    if not np.array_equal(offsets,np.arange(len(sizes))*sizes[0]):
        raise RuntimeError('Unexpected native curve point ordering')
    co=np.empty(len(o.data.position_data)*3,dtype=np.float32)
    o.data.position_data.foreach_get('vector',co)
    if not np.all(np.isfinite(co)): raise RuntimeError('Non-finite hair coordinates')
    return o,co.reshape(len(sizes),int(sizes[0]),3)


def _guide_map(rest,count):
    """Deterministic farthest-point sampling over whole-strand shape descriptors."""
    sample=np.linspace(0,rest.shape[1]-1,7,dtype=int)
    desc=rest[:,sample].reshape(len(rest),-1).astype(np.float64)
    ids=[int(np.argmin(np.linalg.norm(desc-desc.mean(axis=0),axis=1)))]
    distance=np.full(len(rest),np.inf)
    for _ in range(count-1):
        distance=np.minimum(distance,np.sum((desc-desc[ids[-1]])**2,axis=1))
        distance[ids]=-1;ids.append(int(np.argmax(distance)))
    d2=np.sum((desc[:,None,:]-desc[np.array(ids)][None,:,:])**2,axis=2)
    nearest=np.argsort(d2,axis=1)[:,:3]
    d=np.take_along_axis(d2,nearest,axis=1)
    weights=1.0/np.maximum(d,1e-10);weights/=weights.sum(axis=1,keepdims=True)
    guides=rest[ids].copy()
    # Radius of the assigned guide bundle, used to inflate sphere clearance.
    padding=np.zeros(count,dtype=np.float32)
    for g in range(count):
        members=rest[nearest[:,0]==g]
        if len(members): padding[g]=np.percentile(np.linalg.norm(members-guides[g],axis=2),95)
    padding=np.minimum(padding,0.018)
    return guides,nearest.astype(np.uint16),weights.astype(np.float32),padding


def _world(points,mats):
    return np.einsum('tgpi,tji->tgpj',points,mats[:,:3,:3])+mats[:,None,None,:3,3]


def _blend_matrices(a,b,t):
    m=(1-t)*a+t*b
    u,_,vh=np.linalg.svd(m[:,:3,:3])
    m[:,:3,:3]=u@vh
    return m


def _project_distances(x,rest,invmass,step,strength):
    for phase in range(step*2):
        i=np.arange(phase,x.shape[1]-step,step*2);j=i+step
        if not len(i): continue
        diff=x[:,j]-x[:,i];length=np.linalg.norm(diff,axis=2)
        w=invmass[i]+invmass[j]
        correction=diff*((length-rest[:,i])/np.maximum(length,1e-12)
                         *strength/np.maximum(w,1.0))[...,None]
        x[:,i]+=correction*invmass[i][None,:,None]
        x[:,j]-=correction*invmass[j][None,:,None]


def _collide(x,centers,radii,padding,pinned,margin):
    contact=np.zeros(x.shape[:2],dtype=bool)
    for center,radius in zip(centers,radii):
        diff=x-center;distance=np.linalg.norm(diff,axis=2)
        limit=radius+padding[:,None]+margin
        hit=distance<limit;hit[:,:pinned]=False
        direction=diff/np.maximum(distance[...,None],1e-12)
        direction[distance<1e-12]=[0,0,1]
        x+=direction*np.where(hit,limit-distance,0)[...,None]
        contact|=hit
    return contact


def _anchored_length_pass(x,length,pinned):
    # One root-to-tip projection avoids accumulated short-segment stretch after
    # the symmetric PBD iterations. A final collision projection takes priority;
    # any resulting contact stretch is measured rather than hidden.
    for j in range(pinned,x.shape[1]):
        direction=x[:,j]-x[:,j-1]
        direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-12)
        x[:,j]=x[:,j-1]+direction*length[:,j-1,None]


def _simulate(guides,matrices,centers,radii,padding,params,loop=False):
    """NumPy-only Verlet/PBD kernel; no Blender access and no random forcing."""
    T,G,P,_=guides.shape;F=len(matrices);pin=params['pinned_points']
    sub=params['substeps'];dt=1.0/(params['fps']*sub)
    target=_world(guides,matrices[0]).reshape(T*G,P,3)
    x=target.copy();previous=x.copy()
    length=np.linalg.norm(np.diff(target,axis=1),axis=2)
    bend=np.linalg.norm(target[:,2:]-target[:,:-2],axis=2)
    if np.min(length)<1e-5: raise RuntimeError('Degenerate guide segment')
    invmass=np.ones(P);invmass[:pin]=0
    t=np.linspace(0,1,P)
    freq=params['root_frequency_hz']*(1-t)+params['tip_frequency_hz']*t
    stiffness=(2*math.pi*freq)**2
    gravity=np.asarray(params['gravity'],dtype=np.float64)
    damping=math.exp(-params['damping_per_second']*dt)
    flat_padding=padding.reshape(T*G)
    def step(goal,spheres):
        nonlocal x,previous
        old=x.copy()
        x=x+(x-previous)*damping+(gravity+(goal-x)*stiffness[None,:,None])*dt*dt
        contact=np.zeros(x.shape[:2],dtype=bool)
        for _ in range(params['iterations']):
            x[:,:pin]=goal[:,:pin]
            _project_distances(x,length,invmass,1,1.0)
            _project_distances(x,bend,invmass,2,params['bend_stiffness'])
            _project_distances(x,length,invmass,1,1.0)
            contact|=_collide(x,spheres,radii,flat_padding,pin,params['collision_margin'])
        x[:,:pin]=goal[:,:pin]
        _anchored_length_pass(x,length,pin)
        contact|=_collide(x,spheres,radii,flat_padding,pin,params['collision_margin'])
        velocity=x-old
        velocity[contact]*=1-params['contact_friction']
        previous=x-velocity
        previous[:,:pin]=goal[:,:pin]
    def interval(f):
        for s in range(1,sub+1):
            t=s/sub;m=_blend_matrices(matrices[f-1],matrices[f],t)
            goal=_world(guides,m).reshape(T*G,P,3)
            step(goal,centers[f-1]*(1-t)+centers[f]*t)
    for _ in range(round(params['warmup_seconds']/dt)): step(target,centers[0])
    cycles=params['loop_preroll_cycles'] if loop else 0
    for _ in range(cycles):
        for f in range(1,F): interval(f)
    # Even a near-loop can have a tiny sampled endpoint discrepancy. Pins always
    # follow the actual first frame, never the previous loop's last transform.
    x[:,:pin]=target[:,:pin];previous[:,:pin]=target[:,:pin]
    output=np.empty((F,T,G,P,3),dtype=np.float32)
    max_stretch=0.0;max_displacement=0.0;max_root=0.0;max_penetration=0.0
    for f in range(F):
        if f: interval(f)
        wanted=_world(guides,matrices[f])
        delta=x.reshape(T,G,P,3)-wanted
        inverse=np.linalg.inv(matrices[f,:,:3,:3])
        output[f]=np.einsum('tgpi,tji->tgpj',delta,inverse)
        max_displacement=max(max_displacement,float(np.max(np.linalg.norm(delta,axis=3))))
        max_root=max(max_root,float(np.max(np.linalg.norm(delta[:,:,:pin],axis=3))))
        max_stretch=max(max_stretch,float(np.max(np.abs(np.linalg.norm(np.diff(x,axis=1),axis=2)/length-1))))
        for center,radius in zip(centers[f],radii):
            penetration=radius+flat_padding[:,None]+params['collision_margin']-np.linalg.norm(x[:,pin:]-center,axis=2)
            max_penetration=max(max_penetration,float(np.max(penetration)))
    if not np.all(np.isfinite(output)) or max_displacement>0.25 or max_stretch>0.35:
        raise RuntimeError(f'Unstable/incompatible guide setup: displacement={max_displacement}, stretch={max_stretch}')
    packed=output.astype(np.float16)
    report={'maxRootDriftMeters':max_root,'maxGuideDisplacementMeters':max_displacement,
            'maxSegmentRelativeStretch':max_stretch,'maxGuideSpherePenetrationMeters':max(0,max_penetration),
            'float16MaxErrorMeters':float(np.max(np.abs(packed.astype(np.float32)-output))),
            'endStartGuideDeltaMeters':float(np.max(np.linalg.norm(output[-1]-output[0],axis=3))),
            'loopPrerollCycles':cycles,'simulatedGuides':T*G,'pointsPerGuide':P}
    return packed,report


def _collider_definitions(rig):
    b=rig.data.bones
    span=(b['leftUpperArm'].head_local-b['rightUpperArm'].head_local).length
    # Bone-relative centers move with the actual sampled body pose.
    head_length=b['head'].length
    specs=[('head','head',(0,head_length*.67,0),min(span*.225,head_length*.48)),
           ('neck','neck',(0,b['neck'].length*.5,0),span*.085),
           ('left shoulder','leftShoulder',(0,b['leftShoulder'].length*.70,0),span*.165),
           ('right shoulder','rightShoulder',(0,b['rightShoulder'].length*.70,0),span*.165),
           ('upper chest','upperChest',(0,b['upperChest'].length*.12,0),span*.285)]
    return [{'name':name,'bone':bone,'localCenter':list(center),'radius':float(radius)}
            for name,bone,center,radius in specs]


def _action_fingerprint(action):
    curves=[]
    if hasattr(action,'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip,'channelbags'):
                    for bag in strip.channelbags: curves.extend(bag.fcurves)
    if not curves and hasattr(action,'fcurves'): curves=list(action.fcurves)
    if not curves: raise RuntimeError('Cannot inspect action curves: '+action.name)
    h=hashlib.sha256()
    for c in sorted(curves,key=lambda c:(c.data_path,c.array_index)):
        h.update(f'{c.data_path}:{c.array_index}'.encode())
        a=np.array([(*k.co,*k.handle_left,*k.handle_right) for k in c.keyframe_points],dtype=np.float32)
        h.update(a.tobytes());h.update('|'.join(k.interpolation for k in c.keyframe_points).encode())
    return h.hexdigest()


def _assign_action(rig,action):
    rig.animation_data_create();rig.animation_data.action=action
    if action and hasattr(action,'slots'):
        slots=[s for s in action.slots if s.target_id_type=='OBJECT']
        if len(slots)!=1: raise RuntimeError('Expected one object slot in imported body action')
        rig.animation_data.action_slot=slots[0]


def _save_npz(path,arrays,meta):
    temporary=path.with_suffix('.writing.npz')
    try:
        with temporary.open('wb') as f: np.savez_compressed(f,meta=np.array(json.dumps(meta,ensure_ascii=False)),**arrays)
        temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def _action_entries(actions):
    inventory=json.loads((project_path()/'work/reconstruction-r02/imported-motions.json').read_text(encoding='utf-8'))['actions']
    known={e['action']:e for e in inventory}
    if isinstance(actions,str): raise ValueError('Pass a list of action names, not one string')
    names=list(known) if actions is None else list(actions)
    if not names or len(set(names))!=len(names): raise ValueError('Choose one or more unique action names')
    entries=[]
    for name in names:
        if bpy.data.actions.get(name) is None: raise RuntimeError('Existing body action missing: '+name)
        if name in known:
            entries.append(dict(known[name],fromInventory=True,
                provenance='Existing imported project-authored VRMA; not mocap'))
        else:
            entries.append({'action':name,'file':None,'fromInventory':False,
                'provenance':'Explicit existing Blender action, evaluated as currently authored; no import provenance claimed'})
    return entries


def plan(actions=None,allow_authored_ik=False,**overrides):
    """Read-only cost/scope report, without sampling animation or changing hair."""
    _guard(allow_authored_ik);params=dict(PARAMS);params.update(overrides)
    entries=_action_entries(actions)
    objects=[_read_tail(n) for n in OBJECTS]
    frames=sum(math.ceil(bpy.data.actions[e['action']].frame_range[1])-math.floor(bpy.data.actions[e['action']].frame_range[0])+1 for e in entries)
    return {'objects':[{'name':o.name,'strands':len(rest),'pointsPerStrand':rest.shape[1]} for o,rest in objects],
            'actions':[e['action'] for e in entries],'sampledFrames':frames,
            'dynamicBytesBeforeCompression':frames*len(OBJECTS)*params['guides_per_tail']*objects[0][1].shape[1]*3*2,
            'staticBytesBeforeCompression':sum(rest.nbytes for _,rest in objects),
            'params':params,'limitations':LIMITS,'savedBlend':False}


def bake_actions(actions=None,frame_ranges=None,output='work/reconstruction-r02/hair-cache',
                 overwrite=False,colliders=None,allow_authored_ik=False,**overrides):
    """Sample existing body actions once/frame, then write small guide caches.

    For a limited first check: actions=[the idle action],
    frame_ranges={the idle action:(1,61)}, loop_preroll_cycles=0.
    Explicit names may include an existing R03/custom action. Such entries are
    recorded as current Blender actions, never fabricated VRMA imports. The
    default selection remains the four known imported actions. Custom actions
    are treated as non-looping; any other animated IDs keep their current actions.
    All actions are restored on exit; no native strand positions are changed.
    """
    rig=_guard(allow_authored_ik)
    if _PLAYBACK or any(o.get(MARKER) for o in bpy.data.objects if o.name in OBJECTS):
        raise RuntimeError('detach() and restore authored hair before baking')
    params=dict(PARAMS)
    unknown=set(overrides)-set(params)
    if unknown: raise ValueError('Unknown simulation settings: '+str(sorted(unknown)))
    params.update(overrides)
    if params['fps']!=30 or not (8<=params['guides_per_tail']<=48) or not (2<=params['substeps']<=12):
        raise ValueError('Use 30 fps, 8..48 guides per tail and 2..12 substeps')
    if not (2<=params['iterations']<=16) or not (1<=params['pinned_points']<=4):
        raise ValueError('Invalid constraint iteration or pin count')
    if not (0<=params['warmup_seconds']<=2) or not (0<=params['loop_preroll_cycles']<=4):
        raise ValueError('Keep preroll bounded: at most two seconds static and four cycles')
    if not (0<=params['damping_per_second']<=60) or not (0<=params['bend_stiffness']<=1):
        raise ValueError('Invalid damping or bend stiffness')
    if max(params['root_frequency_hz'],params['tip_frequency_hz'])*2*math.pi/(30*params['substeps'])>1.2:
        raise ValueError('Elastic stiffness is too high for the chosen integration step')
    entries=_action_entries(actions)
    dest=_path(output);dest.mkdir(parents=True,exist_ok=True)
    output_paths={e['action']:dest/(re.sub('[^a-zA-Z0-9_-]+','-',e['action']).strip('-')+'.guides.npz') for e in entries}
    if not overwrite and any(p.exists() for p in output_paths.values()): raise FileExistsError('Cache exists; choose another output or explicitly overwrite')
    pairs=[_read_tail(n) for n in OBJECTS]
    if len({r.shape[1] for _,r in pairs})!=1: raise RuntimeError('Both tails must have equal samples per strand')
    static={};all_guides=[];all_padding=[];geometry_hashes={}
    for i,(obj,rest) in enumerate(pairs):
        guides,ids,weights,pad=_guide_map(rest,params['guides_per_tail'])
        static.update({f'rest_{i}':rest,f'ids_{i}':ids,f'weights_{i}':weights})
        geometry_hashes[obj.name]=_digest(rest);all_guides.append(guides);all_padding.append(pad)
    guide_array=np.array(all_guides,dtype=np.float64);padding=np.array(all_padding,dtype=np.float64)
    static['guides']=guide_array.astype(np.float32);static['padding']=padding.astype(np.float32)
    base_hash=_digest(*[static[k] for k in sorted(static)])
    base_path=dest/('hair-static-'+base_hash[:16]+'.npz')
    static_meta={'version':VERSION,'baseHash':base_hash,'objects':list(OBJECTS),'geometryHashes':geometry_hashes}
    if not base_path.exists(): _save_npz(base_path,static,static_meta)
    specs=colliders if colliders is not None else _collider_definitions(rig)
    for c in specs:
        if c['bone'] not in rig.pose.bones or c['radius']<=0: raise ValueError('Invalid collider definition')
    radii=np.array([c['radius'] for c in specs],dtype=np.float64)
    scene=bpy.context.scene;old_frame=scene.frame_current;old_subframe=scene.frame_subframe
    old_action=rig.animation_data.action if rig.animation_data else None
    old_slot=rig.animation_data.action_slot if rig.animation_data and hasattr(rig.animation_data,'action_slot') else None
    reports=[]
    try:
        for entry in entries:
            action=bpy.data.actions.get(entry['action'])
            if action is None: raise RuntimeError('Action missing: '+entry['action'])
            action_hash=_action_fingerprint(action)
            a,b=(frame_ranges or {}).get(action.name,action.frame_range)
            frames=np.arange(math.floor(a),math.ceil(b)+1,dtype=np.int32)
            if len(frames)<2 or len(frames)>params['max_cache_frames']: raise ValueError('Frame range exceeds bounded cache scope')
            _assign_action(rig,action)
            matrices=np.empty((len(frames),len(pairs),4,4),dtype=np.float64)
            centers=np.empty((len(frames),len(specs),3),dtype=np.float64)
            ik_switches=np.zeros((len(frames),4),dtype=np.float32)
            for f,frame in enumerate(frames):
                scene.frame_set(int(frame));bpy.context.view_layer.update()
                master=rig.pose.bones.get('CTRL_R02_Master')
                if master: ik_switches[f]=[master.get('ik_'+k,0) for k in ('leg_L','leg_R','arm_L','arm_R')]
                if not allow_authored_ik and np.max(np.abs(ik_switches[f]))>1e-7:
                    raise RuntimeError('Action uses animated IK controls; explicitly pass allow_authored_ik=True')
                for t,(obj,_) in enumerate(pairs):
                    matrices[f,t]=np.array(obj.matrix_world,dtype=np.float64)
                    if np.max(np.abs(matrices[f,t,:3,:3].T@matrices[f,t,:3,:3]-np.eye(3)))>2e-4:
                        raise RuntimeError('Rigid unit-scale hair transforms required; animated scaling needs explicit support')
                for j,c in enumerate(specs):
                    m=np.array(rig.matrix_world@rig.pose.bones[c['bone']].matrix,dtype=np.float64)
                    centers[f,j]=m[:3,:3]@np.asarray(c['localCenter'])+m[:3,3]
                if f%60==0: print(f'R02_HAIR_SAMPLE {action.name}: {f+1}/{len(frames)}')
            loop=(entry['fromInventory'] and entry['file'] in
                  {'natural-idle.vrma','natural-walk-in-place.vrma','friendly-greeting.vrma'}
                  and action.name not in (frame_ranges or {}))
            seam=float(np.max(np.abs(matrices[-1]-matrices[0])))
            center_seam=float(np.max(np.linalg.norm(centers[-1]-centers[0],axis=1))) if len(specs) else 0.0
            if seam>2e-4 or center_seam>0.0005: loop=False
            delta,metrics=_simulate(guide_array,matrices,centers,radii,padding,params,loop=loop)
            meta={'version':VERSION,'baseFile':base_path.name,'baseHash':base_hash,'objects':list(OBJECTS),
                  'action':action.name,'actionSha256':action_hash,'sourceMotion':entry['file'],
                  'sourceProvenance':entry['provenance'],'sourceFromImportInventory':entry['fromInventory'],
                  'allowAuthoredIK':bool(allow_authored_ik),
                  'sourceBlend':str(Path(bpy.data.filepath).relative_to(project_path())),
                  'frameRange':[int(frames[0]),int(frames[-1])],'fps':30,'params':params,
                  'colliders':specs,'metrics':metrics,'bodyLoopMatrixSeam':seam,'colliderLoopSeamMeters':center_seam,
                  'limitations':LIMITS,'savedBlend':False}
            path=output_paths[action.name]
            _save_npz(path,{'frames':frames,'guide_delta':delta,'object_matrices':matrices.astype(np.float32),
                            'collider_centers':centers.astype(np.float32),'ik_switches':ik_switches},meta)
            report=dict(meta,cache=str(path.relative_to(project_path())),compressedBytes=path.stat().st_size)
            reports.append(report);print(json.dumps(report,ensure_ascii=False))
    finally:
        _assign_action(rig,old_action)
        if old_slot: rig.animation_data.action_slot=old_slot
        scene.frame_set(old_frame,subframe=old_subframe);bpy.context.view_layer.update()
    result={'version':VERSION,'staticCache':str(base_path.relative_to(project_path())),
            'staticCompressedBytes':base_path.stat().st_size,'actions':reports,'savedBlend':False}
    (dest/'cache-report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def attach(cache,preview_handler=False):
    """Load one cache. Caller selects its action/frame; nothing is saved."""
    global _PLAYBACK
    rig=_guard(allow_authored_ik=True)
    if _PLAYBACK: detach()
    if any(getattr(h,'_r02_secondary_hair',False) for h in bpy.app.handlers.frame_change_post):
        raise RuntimeError('Detach the previous module instance before reloading this script')
    path=_path(cache)
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files};meta=json.loads(str(arrays.pop('meta')))
    if meta['version']!=VERSION: raise RuntimeError('Unsupported cache format')
    if not meta.get('allowAuthoredIK',False): _guard()
    base_path=path.parent/meta['baseFile'];base_path.resolve().relative_to(path.parent.resolve())
    with np.load(base_path,allow_pickle=False) as f:
        base={k:f[k].copy() for k in f.files};base_meta=json.loads(str(base.pop('meta')))
    if _digest(*[base[k] for k in sorted(base)])!=meta['baseHash'] or base_meta['baseHash']!=meta['baseHash']:
        raise RuntimeError('Static cache hash mismatch')
    if rig.animation_data is None or rig.animation_data.action is None or rig.animation_data.action.name!=meta['action']:
        raise RuntimeError('Select the cache body action first: '+meta['action'])
    if _action_fingerprint(rig.animation_data.action)!=meta['actionSha256']:
        raise RuntimeError('Action changed after cache bake; rebake')
    objects=[]
    for i,name in enumerate(meta['objects']):
        obj,rest=_read_tail(name)
        if obj.get(MARKER): raise RuntimeError('Restore the previous cache session before attachment')
        if _digest(rest)!=base_meta['geometryHashes'][name]: raise RuntimeError('Authored groom changed; rebake')
        objects.append(obj)
    _PLAYBACK={'path':path,'meta':meta,'arrays':arrays,'base':base,'objects':objects,
               'lockInterface':bpy.context.scene.render.use_lock_interface,'error':None}
    try:
        apply_frame()
        for obj in objects: obj[MARKER]=str(path.relative_to(project_path()))
        if preview_handler:
            bpy.context.scene.render.use_lock_interface=True
            bpy.app.handlers.frame_change_post.append(_frame_handler)
    except Exception:
        detach();raise
    return {'cache':cache,'previewHandler':preview_handler,'savedBlend':False,'limitations':LIMITS}


def apply_frame(frame=None):
    """Apply cached local guide offsets after scene.frame_set(), before render."""
    if _PLAYBACK is None: raise RuntimeError('attach() a matching cache first')
    state=_PLAYBACK;rig=_guard(state['meta'].get('allowAuthoredIK',False));scene=bpy.context.scene
    current=scene.frame_current+scene.frame_subframe
    frame=current if frame is None else float(frame)
    if abs(frame-current)>1e-6: raise RuntimeError('Set the scene frame first')
    if rig.animation_data.action.name!=state['meta']['action']: raise RuntimeError('Wrong body action for this cache')
    a=state['arrays'];frames=a['frames']
    if frame<frames[0]-1e-6 or frame>frames[-1]+1e-6: raise RuntimeError('Frame outside cached range')
    pos=np.clip(frame-frames[0],0,len(frames)-1);lo=int(math.floor(pos));hi=min(lo+1,len(frames)-1);t=pos-lo
    if 'ik_switches' in a:
        master=rig.pose.bones.get('CTRL_R02_Master')
        current_switches=np.array([master.get('ik_'+k,0) if master else 0 for k in ('leg_L','leg_R','arm_L','arm_R')])
        expected_switches=a['ik_switches'][lo]*(1-t)+a['ik_switches'][hi]*t
        if np.max(np.abs(current_switches-expected_switches))>2e-4:
            raise RuntimeError('IK switch state changed after the cache was baked')
    expected=_blend_matrices(a['object_matrices'][lo].astype(float),a['object_matrices'][hi].astype(float),t)
    for i,obj in enumerate(state['objects']):
        if np.max(np.abs(np.array(obj.matrix_world)-expected[i]))>2e-4:
            raise RuntimeError('Head/body attachment changed; rebake instead of replaying a stale cache')
    delta=a['guide_delta'][lo].astype(np.float32)*(1-t)+a['guide_delta'][hi].astype(np.float32)*t
    for i,obj in enumerate(state['objects']):
        b=state['base'];ids=b[f'ids_{i}'];weights=b[f'weights_{i}']
        displacement=np.einsum('nkpq,nk->npq',delta[i][ids],weights)
        result=np.ascontiguousarray(b[f'rest_{i}']+displacement,dtype=np.float32)
        pin=state['meta']['params']['pinned_points'];result[:,:pin]=b[f'rest_{i}'][:,:pin]
        obj.data.position_data.foreach_set('vector',result.ravel());obj.data.update_tag();obj.update_tag()
    return {'frame':frame,'cache':str(state['path'].relative_to(project_path())),'savedBlend':False}


def _frame_handler(scene,dependency_graph=None):
    try: apply_frame()
    except Exception as exc:
        if _PLAYBACK and _PLAYBACK.get('error')!=str(exc):
            _PLAYBACK['error']=str(exc);print('R02_HAIR_PLAYBACK_STOP: '+str(exc))
        # Fail visibly by restoring the authored groom instead of leaving a stale pose.
        detach()
_frame_handler._r02_secondary_hair=True


def detach():
    """Remove this session's handler and restore exact authored strand points."""
    global _PLAYBACK
    if bpy is None: return
    if _frame_handler in bpy.app.handlers.frame_change_post: bpy.app.handlers.frame_change_post.remove(_frame_handler)
    if _PLAYBACK:
        for i,obj in enumerate(_PLAYBACK['objects']):
            obj.data.position_data.foreach_set('vector',_PLAYBACK['base'][f'rest_{i}'].ravel())
            obj.data.update_tag();obj.update_tag()
            if MARKER in obj: del obj[MARKER]
        bpy.context.scene.render.use_lock_interface=_PLAYBACK['lockInterface']
    _PLAYBACK=None
    return {'restoredAuthoredGroom':True,'savedBlend':False}


if __name__=='__main__':
    print('Loaded only. Use plan(), then explicitly call bake_actions(); no Blender state changed.')
