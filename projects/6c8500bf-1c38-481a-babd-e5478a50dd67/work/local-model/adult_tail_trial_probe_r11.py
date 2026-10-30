"""Read-only short actual-input probe before a new R11 tail simulation.

Creates a new envelope proposal plus JSON/NPZ observations, never a Blend,
simulation, cache, hair edit or render. Captures every requested action frame;
no old cached transforms or tail coordinates enter this probe. The final bound
source needs its own fresh probe and bake even after an earlier static trial.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

TAILS = ('R02 Strands / R02 Twin tail L', 'R02 Strands / R02 Twin tail R')
HEAD = 'R11 Female head study'
RIG = 'Character_Rig'
DYNAMICS_SHA = '69ce9482dfdac6dcb85c7f4cdb1e48b57657e81c169c753eb664a871e65f3254'


def api(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def path(relative, suffix):
    if Path(relative).is_absolute() or '..' in str(relative).replace('\\','/').split('/'):
        raise ValueError('Explicit project-relative reconstruction path required')
    q = (project_path()/relative).resolve()
    q.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if q.suffix != suffix:
        raise ValueError('Wrong source/output suffix')
    return q


def world(points, matrix):
    m = np.asarray(matrix, float)
    return np.asarray(points) @ m[:3,:3].T + m[:3,3]


def pins_exit_slack(guides_world, sphere_centers, radii, padding, margin, pins=3):
    """Necessary, not sufficient, feasibility for the first unpinned segment.

    A free point one rest-segment away from its fixed predecessor cannot escape
    a sphere when distance(anchor,center)+segment_length < expanded_radius.
    No numerical simulation or anatomical inside classification is involved.
    """
    guides = np.asarray(guides_world, float)
    centers = np.asarray(sphere_centers, float)
    anchor = guides[:, pins-1]
    length = np.linalg.norm(guides[:, pins]-anchor, axis=1)
    expanded = np.asarray(radii)[None,:] + np.asarray(padding)[:,None] + margin
    distance = np.linalg.norm(anchor[:,None,:]-centers[None,:,:], axis=2)
    slack = distance + length[:,None] - expanded
    free_gap = np.linalg.norm(guides[:,pins,None,:]-centers[None,:,:],axis=2)-expanded
    return slack, free_gap


def attachment_kind(obj, rig):
    if obj.parent == rig and obj.parent_type == 'BONE' and obj.parent_bone == 'head':
        return 'existing-rig-head-bone'
    parent = obj.parent
    if (parent is not None and parent.type == 'EMPTY' and
        parent.get('R11_original_head_space_source') == HEAD and
        parent.parent == rig and parent.parent_type == 'BONE' and parent.parent_bone == 'head' and
        obj.parent_type == 'OBJECT'):
        return 'explicit-cranial-carrier-needs-secondary-reader-adapter'
    return 'unsupported-current-attachment'


def preflight(bpy, rig, native):
    rows = {}; blockers = []
    for name in TAILS:
        obj = bpy.data.objects[name]
        row = {'type': obj.type, 'data': obj.data.name, 'dataUsers': obj.data.users,
            'objectLibrary': obj.library.filepath if obj.library else None,
            'dataLibrary': obj.data.library.filepath if obj.data.library else None,
            'parent': obj.parent.name if obj.parent else None, 'parentType': obj.parent_type,
            'parentBone': obj.parent_bone, 'attachmentKind': attachment_kind(obj,rig),
            'matrixParentInverse': [list(r) for r in obj.matrix_parent_inverse],
            'matrixBasis': [list(r) for r in obj.matrix_basis],
            'modifiers': [{'name': m.name, 'type': m.type, 'viewport': m.show_viewport,
                'render': m.show_render} for m in obj.modifiers],
            'oldOwner': obj.get(native['OWNER']), 'oldPythonSession': obj.get('r02_secondary_hair_session'),
            'reservedAttributes': [n for n in native['ATTRS'] if obj.data.attributes.get(n)],
            'staticRestMarker': obj.get('R11_tail_rest_shape'), 'dynamicValidity': obj.get('R11_tail_dynamics_valid')}
        counts = np.asarray([c.points_length for c in obj.data.curves],np.int32)
        starts = np.asarray([c.first_point_index for c in obj.data.curves],np.int32)
        row.update(curves=len(counts), pointCount=len(obj.data.position_data),
            equal35PointStrands=bool(len(counts)==2480 and np.all(counts==35)),
            contiguousPointOrdering=bool(np.array_equal(starts,np.arange(len(starts))*35)))
        if obj.library or obj.data.library or obj.data.users != 1:
            blockers.append(name+': requires local single-user curve data')
        if row['attachmentKind'] != 'existing-rig-head-bone':
            blockers.append(name+': existing secondary reader requires direct rig/head bone parenting')
        if row['modifiers'] or row['reservedAttributes'] or row['oldOwner'] or row['oldPythonSession']:
            blockers.append(name+': native installer requires zero modifiers and retired playback owners/attributes')
        if not row['equal35PointStrands'] or not row['contiguousPointOrdering']:
            blockers.append(name+': actual retained strand topology changed')
        rows[name] = row
    flags = {n: rig[n] for n in (native['RIG_GUIDE'], native['ENABLE']) if n in rig}
    if flags:
        blockers.append('Rig still has native cache/toggle properties; preserve and explicitly retire before rebuilding')
    handlers = [getattr(h,'__name__',repr(h)) for h in bpy.app.handlers.frame_change_post
                if getattr(h,'_r02_secondary_hair',False)]
    if handlers:
        blockers.append('Legacy Python hair playback handler is active')
    return {'tails': rows, 'rigNativeProperties': flags, 'oldPythonHandlers': handlers,
            'blockers': blockers, 'legacyNativeInputCompatible': not blockers}


def probe(source_relative, output_relative, action_name, first=1, last=37):
    import bpy
    source = path(source_relative,'.blend'); out = path(output_relative,'.json')
    npz = out.with_suffix('.npz'); proposal_path = out.with_suffix('.envelope-plan.json')
    if any(p.exists() for p in (out,npz,proposal_path)):
        raise RuntimeError('Retain previous trial observations; select a new explicit output name')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the exact fresh saved independent source')
    if not isinstance(first,int) or not isinstance(last,int) or not 2<=last-first+1<=61:
        raise ValueError('Choose 2..61 consecutive actual frames; initial recommendation 1..37')
    dynamics_path = Path(__file__).with_name('adult_tail_dynamics_r11.py')
    if sha(dynamics_path) != DYNAMICS_SHA:
        raise RuntimeError('Review changed R11 envelope/bake helper before this probe')
    dynamics = api('adult_tail_dynamics_r11'); native = api('native_hair_playback_r04')
    secondary = api('secondary_hair_r02'); uv = api('production_uv_r08')
    _,_,rig = dynamics['guard'](source_relative)
    source_sha = sha(source)
    before = dynamics['contract'](bpy,rig); pose_before = uv['_pose_contract']()
    check = preflight(bpy,rig,native)
    report = {'source': source_relative, 'sourceSha256': source_sha, 'helperSha256': sha(__file__),
        'frozenDynamicsHelperSha256': DYNAMICS_SHA, 'preflight': check,
        'action': action_name, 'requestedFrameRange': [first,last],
        'savedBlend': False, 'simulationRun': False, 'cacheInstalled': False,
        'surfaceCollisionAccepted': False, 'appearanceAccepted': False,
        'headSkinBindingPresent': 'R11_head_neck_skin_bind' in bpy.context.scene,
        'requiresFreshFinalBoundSourceProbe': 'R11_head_neck_skin_bind' not in bpy.context.scene,
        'limits': ['Only actual current input measurements; no previous cached motion/points are reused.',
            'Sphere/padded-guide feasibility is a conservative proposal diagnostic, not actual mesh collision.',
            'A static unbound-source probe cannot certify a later bound head, garment edit or different action.',
            'Crown/fringe/face-wisp nodes and all editable meshes/curves remain unchanged.']}
    if check['blockers']:
        report['status'] = 'blocked-before-frame-sampling'
        dynamics['write_new'](out,report)
        return report
    action = bpy.data.actions.get(action_name)
    if action is None or not native['_fcurves'](action):
        raise RuntimeError('Existing explicit body action required')
    if first < math.floor(action.frame_range[0]) or last > math.ceil(action.frame_range[1]):
        raise RuntimeError('Probe frame range exceeds actual action')
    old_action,old_slot = rig.animation_data.action,rig.animation_data.action_slot
    old_frame = (bpy.context.scene.frame_current,bpy.context.scene.frame_subframe)
    old_position = rig.data.pose_position
    transforms = {b.name:{prop:tuple(getattr(b,prop)) for prop in
        ('location','rotation_euler','rotation_quaternion','rotation_axis_angle','scale')} for b in rig.pose.bones}
    master = rig.pose.bones.get('CTRL_R02_Master')
    switches = {k:master[k] for k in ('ik_leg_L','ik_leg_R','ik_arm_L','ik_arm_R') if master and k in master}
    unit_errors = []
    try:
        proposal_relative = proposal_path.relative_to(project_path()).as_posix()
        dynamics['plan'](source_relative,proposal_relative)
        proposal = json.loads(proposal_path.read_text(encoding='utf8'))
        arrays = {}; tail_rows = {}; guides = []; padding = []
        for index,name in enumerate(TAILS):
            obj,rest = secondary['_read_tail'](name)
            gs,ids,weights,pad = secondary['_guide_map'](rest,proposal['simulationParameters']['guides_per_tail'])
            arrays.update({f'tail_{index}_rest_local':rest, f'tail_{index}_guides_local':gs,
                           f'tail_{index}_guide_map_ids':ids, f'tail_{index}_guide_map_weights':weights,
                           f'tail_{index}_solver_padding':pad})
            measured = []
            for g in range(len(gs)):
                members = rest[ids[:,0]==g]
                distance = np.linalg.norm(members-gs[g],axis=2) if len(members) else np.zeros((1,1))
                measured.append({'guide':g, 'assignedStrands':len(members), 'uncappedP95Meters':float(np.percentile(distance,95)),
                    'maximumBundleRadiusMeters':float(distance.max()), 'solverPaddingMeters':float(pad[g])})
            tail_rows[name] = {'currentRestSha256':secondary['_digest'](rest), 'guideSha256':secondary['_digest'](gs),
                'rootThreeLocalSha256':secondary['_digest'](rest[:,:3]), 'guideCount':len(gs), 'bundlePadding':measured,
                'rootThreeWorldAtFirst':None}
            guides.append(gs); padding.append(pad)
        frames = np.arange(first,last+1,dtype=np.int32)
        matrices = np.empty((len(frames),2,4,4),np.float64)
        centers = np.empty((len(frames),len(proposal['colliders']),3),np.float64)
        pins_world = np.empty((len(frames),2,2480,3,3),np.float32)
        radii = np.asarray([c['radius'] for c in proposal['colliders']],float)
        slack = np.empty((len(frames),2,len(guides[0]),len(radii)),np.float32)
        free_gap = np.empty_like(slack)
        rig.data.pose_position='POSE'; secondary['_assign_action'](rig,action)
        for fi,frame in enumerate(frames):
            bpy.context.scene.frame_set(int(frame)); bpy.context.view_layer.update()
            for ci,spec in enumerate(proposal['colliders']):
                matrix = np.asarray(rig.matrix_world @ rig.pose.bones[spec['bone']].matrix,float)
                centers[fi,ci] = matrix[:3,:3] @ np.asarray(spec['localCenter'])+matrix[:3,3]
            for ti,name in enumerate(TAILS):
                matrix = np.asarray(bpy.data.objects[name].matrix_world,float); matrices[fi,ti]=matrix
                unit_errors.append(float(np.max(abs(matrix[:3,:3].T @ matrix[:3,:3]-np.eye(3)))))
                pins_world[fi,ti] = world(arrays[f'tail_{ti}_rest_local'][:,:3],matrix)
                gs_world = world(guides[ti],matrix)
                slack[fi,ti],free_gap[fi,ti] = pins_exit_slack(gs_world,centers[fi],radii,padding[ti],
                    proposal['simulationParameters']['collision_margin'])
            print('R11_TAIL_INPUT_FRAME '+str(int(frame)),flush=True)
    finally:
        secondary['_assign_action'](rig,old_action)
        if old_slot is not None: rig.animation_data.action_slot=old_slot
        rig.data.pose_position=old_position
        bpy.context.scene.frame_set(old_frame[0],subframe=old_frame[1])
        # A saved source can contain an unkeyed manual pose over an action.
        # Restore exact channels after frame evaluation, without matrix
        # decomposition round trips or a final frame_set overwriting them.
        for name,channels in transforms.items():
            for prop,value in channels.items(): setattr(rig.pose.bones[name],prop,value)
        for name,value in switches.items(): master[name]=value
        rig.update_tag(); bpy.context.view_layer.update()
    if dynamics['contract'](bpy,rig)!=before or uv['_pose_contract']()!=pose_before or sha(source)!=source_sha:
        raise RuntimeError('Read-only probe changed source/geometry/action/pose')
    arrays.update(frames=frames,tail_object_matrices=matrices,collider_centers=centers,
        actual_three_pin_world=pins_world,first_free_exit_slack=slack,first_free_sphere_gap=free_gap)
    bad = np.argwhere(slack < -.0002)
    worst = sorted(bad.tolist(),key=lambda at:float(slack[tuple(at)]))[:20]
    worst_rows = [{'frame':int(frames[f]),'tail':TAILS[t],'guide':g,'collider':proposal['colliders'][c]['name'],
        'unavoidableMinimumExitDeficitMeters':float(-slack[f,t,g,c])} for f,t,g,c in worst]
    dt=1./30.; displacement=pins_world-pins_world[:1]
    for ti,name in enumerate(TAILS):
        tail_rows[name].update(rootThreeWorldAtFirst=pins_world[0,ti].mean(axis=0).tolist(),
            maximumActualPinnedPointTravelMeters=float(np.linalg.norm(displacement[:,ti],axis=-1).max()),
            maximumActualPinnedPointSpeedMetersPerSecond=float(np.linalg.norm(np.diff(pins_world[:,ti],axis=0)/dt,axis=-1).max()))
    np.savez_compressed(npz,**arrays)
    needs_adaptation = bool(len(bad) or max(unit_errors)>2e-4)
    report.update(status='actual-inputs-observed; collider-adaptation-required' if needs_adaptation else 'actual-inputs-observed; new-short-simulation-pending',
        actualFrameRange=[int(frames[0]),int(frames[-1])],actualFrameCount=len(frames),fps=30,
        actionSha256=native['_action_fingerprint'](action), currentTails=tail_rows,
        envelopePlan=proposal_relative,envelopePlanSha256=sha(proposal_path),
        actualColliders=proposal['colliders'],maximumTailRigidUnitMatrixError=max(unit_errors),
        impossiblePinnedExitSamplesBeyond02Millimeter=len(bad),worstPinnedExitCandidates=worst_rows,
        minimumFirstFreeSphereGapMeters=float(free_gap.min()),
        arrays=npz.relative_to(project_path()).as_posix(),arraysSha256=sha(npz),
        sourceGeometryActionAndPoseUnchanged=True,
        next='Inspect blocked root/envelope pairs before any solver run; do not shrink spheres blindly.' if needs_adaptation else
             'Run a new bounded PBD trial on this exact source/plan, then actual native parity and surface checks. Any later source/binding/action change requires fresh input planning.')
    dynamics['write_new'](out,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--action',default='R09 Body performance - authored squat bend reach')
    parser.add_argument('--first',type=int,default=1); parser.add_argument('--last',type=int,default=37)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    report=probe(args.source,args.output,args.action,args.first,args.last)
    print('R11_TAIL_TRIAL_PROBE '+json.dumps({'status':report['status'],'output':args.output,'simulationRun':False}),flush=True)
