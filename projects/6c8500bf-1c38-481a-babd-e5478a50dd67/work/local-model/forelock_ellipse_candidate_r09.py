"""One new candidate from accepted small forelock flow, NEVER rejected volume.

Seven original cohorts retain count/order/radii and their last 20% point paths.
Each main lock has a full continuous elliptical volume, staggered posterior root
patch, different bow phase, mild cross-section roll and varied transition phase.
No added fibres, materials, tail/rig changes, save, render or cache operation.
"""
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SOURCE='work/reconstruction-r02/reference-root-flow-r09.blend'
OBJECT='R03 Forelocks from crown'
MARKER='R09_forelock_ellipse_candidate'
BACKUP='R09_forelock_ellipse_original_data'
META='R09_forelock_ellipse_restore_record'
ROOT_BACK=np.array([.007,.017,.011,.022,.005,.015,.009])
ROOT_SIDE=np.array([-.0018,.0012,-.0010,.0005,.0014,-.0004,.0016])
PEAK=np.array([.30,.43,.35,.47,.28,.40,.33])
LIFT=np.array([.0025,.0045,.0034,.0050,.0028,.0040,.0024])
HALF_DEPTH=np.array([.0017,.0020,.0018,.0021,.0016,.0019,.0017])
WIDTH=np.array([.98,1.06,.96,1.04,.94,1.07,.97])
ROLL=np.radians([-9.,6.,-7.,10.,-5.,7.,-6.])
CONTACT_END=np.array([.14,.20,.16,.22,.13,.19,.15])


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _smooth(t):
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)


def _sample(path,t):
    u=np.clip(t,0.,1.)*(len(path)-1);i=np.minimum(u.astype(np.int32),len(path)-2);f=u-i
    return path[i]*(1.-f[:,None])+path[i+1]*f[:,None]


def _unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)


def _bell(t,peak):
    # Distinct smooth beta-shaped bows, zero value and tangent at t=0/.8.
    u=np.clip(t/.8,0.,1.);p=peak/.8;a=5.*p;b=5.*(1.-p)
    return (u**a*(1.-u)**b)/(p**a*(1.-p)**b)


def propose(points,starts,sizes,center):
    if len(starts)!=3360 or sizes.min()<40 or sizes.max()>60:
        raise RuntimeError('Expected existing seven 480-fibre cohorts')
    volume=_api('forelock_volume_candidate_r09')
    points=np.asarray(points,float);out=points.copy();fixed=np.ones(len(points),bool)
    t_values=np.zeros(len(points));root_weight=np.zeros(len(points));clearance=np.zeros(len(points))
    grid=np.linspace(0.,1.,101);rows=[]
    circle_u=np.linspace(-1.,1.,4097)
    circle_cdf=(circle_u*np.sqrt(np.maximum(0.,1.-circle_u**2))+np.arcsin(circle_u)+np.pi/2)/np.pi
    for group in range(7):
        ids=list(range(group*480,(group+1)*480))
        paths=np.stack([_sample(points[starts[i]:starts[i]+sizes[i]],grid) for i in ids])
        core=paths.mean(axis=0);tangent,across,normal=volume['_frame'](core,center)
        bow=_bell(grid,PEAK[group]);root_fade=1.-_smooth(grid/(.39+.02*(group%3)))
        root_delta=np.array([ROOT_SIDE[group],ROOT_BACK[group],0.])
        main=core+root_delta[None,:]*root_fade[:,None]+normal*(LIFT[group]*bow)[:,None]
        _,across_new,normal_new=volume['_frame'](main,center)
        roll=ROLL[group]*bow
        axis_u=across_new*np.cos(roll)[:,None]+normal_new*np.sin(roll)[:,None]
        axis_v=normal_new*np.cos(roll)[:,None]-across_new*np.sin(roll)[:,None]
        offsets=paths-core[None,:,:]
        old_u=np.sum(offsets*across[None,:,:],axis=2)
        halfwidth=np.maximum(.0002,np.quantile(np.abs(old_u),.975,axis=0))
        halfwidth*=1.+(WIDTH[group]-1.)*bow
        order=np.argsort(old_u[:,0],kind='stable');rank=np.empty(480)
        rank[order]=(np.arange(480)+.5)/480.
        u_rank=np.interp(rank,circle_cdf,circle_u)
        depth_rank=(np.arange(480)*.618033988749895+group*.371)%1.
        phase_rank=(np.arange(480)*.754877666246693+group*.217)%1.
        for within,i in enumerate(ids):
            start=int(starts[i]);count=int(sizes[i]);t=np.linspace(0.,1.,count)
            d=depth_rank[within];p=phase_rank[within];u=u_rank[within]
            # Slightly different path-length phase, without moving a single final
            # tip. Distinct blending endpoints also avoid a transverse curtain.
            phase=(p-.5)*.027*_bell(t,PEAK[group])
            warped=np.clip(t+phase,0.,1.)
            c=_sample(main,warped);old_core=_sample(core,t)
            a=_unit(_sample(axis_u,warped));n=_unit(_sample(axis_v,warped))
            width=np.interp(warped,grid,halfwidth)
            thickness=HALF_DEPTH[group]*(.35+.65*_bell(warped,PEAK[group]))
            # Uniform area distribution through a filled ellipse. It is not a
            # single quadratic surface and has no discrete depth sub-bands.
            transverse=width*u
            depth=thickness*np.sqrt(max(0.,1.-u*u))*(2.*d-1.)
            cross=a*transverse[:,None]+n*(depth+.00055)[:,None]
            root_patch=tangent[0][None,:]*((p-.5)*.005*(1.-_smooth(t/.28)))[:,None]
            blend_start=.39+.085*d;blend_end=.69+.09*p
            blend=1.-_smooth((t-blend_start)/(blend_end-blend_start))
            old=points[start:start+count]
            q=c+(old-old_core)*(1.-blend[:,None])+(cross+root_patch)*blend[:,None]
            keep=t>=.8;q[keep]=old[keep]
            out[start:start+count]=q;fixed[start:start+count]=keep;t_values[start:start+count]=t
            end=CONTACT_END[group]+(p-.5)*.024
            root_weight[start:start+count]=1.-_smooth((t-.018)/(end-.018))
            clearance[start:start+count]=.00012+.00030*d
        rows.append({'group':group,'rootBackMm':float(ROOT_BACK[group]*1000.),
            'rootSideMm':float(ROOT_SIDE[group]*1000.),'mainBowPeakT':float(PEAK[group]),
            'mainLiftMm':float(LIFT[group]*1000.),'centerFullEllipseThicknessMm':float(HALF_DEPTH[group]*2000.),
            'peakWidthFactor':float(WIDTH[group]),'crossSectionRollDegrees':float(np.degrees(ROLL[group])),
            'contactTransitionEndT':float(CONTACT_END[group]),
            'sourceGuideSamples':core[::10].tolist(),'proposedGuideSamples':main[::10].tolist()})
    return out,fixed,t_values,root_weight,clearance,rows


def _seat(old,candidate,fixed,t,weights,clearance,nearest):
    out=candidate.copy();root_errors=[];max_push=0.;pushed=0
    for i in np.flatnonzero(~fixed):
        point=out[i];hit,n,depth=nearest(point)
        if weights[i]>0.:
            anchored=hit+n*clearance[i]
            if np.linalg.norm(anchored-point)>.012:
                raise RuntimeError('Ellipse root seating exceeds 12 mm at point '+str(i))
            point=point*(1.-weights[i])+anchored*weights[i]
        correction=0.
        for unused in range(2):
            hit,n,depth=nearest(point);target=min(.00020,clearance[i])
            if depth>=target-.000015:break
            step=target-depth+.000003;correction+=step;point+=n*step
        hit,n,depth=nearest(point)
        if depth<min(.00020,clearance[i])-.000015 or correction>.004:
            raise RuntimeError('Ellipse contact guard: '+json.dumps({'point':int(i),'t':float(t[i]),
                               'clearanceMm':depth*1000.,'correctionMm':correction*1000.}))
        out[i]=point;max_push=max(max_push,correction);pushed+=correction>0.
        if t[i]==0.:root_errors.append(depth-clearance[i])
    out[fixed]=old[fixed]
    return out,{'rootCount':len(root_errors),'maximumRootClearanceErrorMm':float(np.max(np.abs(root_errors))*1000.),
                'extraPushPoints':int(pushed),'maximumExtraContactPushMm':max_push*1000.}


def _checks(old,new,starts,sizes,fixed,t):
    if not np.isfinite(new).all() or not np.array_equal(old[fixed],new[fixed]):
        raise RuntimeError('Finite/original-tip invariant failed')
    distance=np.linalg.norm(new-old,axis=1)
    # Larger displacement is explicitly confined to the requested posterior
    # redistribution of roots. This does not authorize greater crown inflation.
    if distance.max()>.027 or distance[t>=.25].max()>.014:
        raise RuntimeError('Root/main-shape displacement bounds exceeded: '+str([float(distance.max()),float(distance[t>=.25].max())]))
    largest=0.;added=0.;turns=0.;lengths=[]
    for i,(start,count) in enumerate(zip(starts,sizes)):
        a=old[start:start+count];b=new[start:start+count]
        da=np.diff(a,axis=0);db=np.diff(b,axis=0);la=np.linalg.norm(da,axis=1);lb=np.linalg.norm(db,axis=1)
        ta=np.degrees(np.arccos(np.clip(np.sum(_unit(da)[1:]*_unit(da)[:-1],axis=1),-1,1)))
        tb=np.degrees(np.arccos(np.clip(np.sum(_unit(db)[1:]*_unit(db)[:-1],axis=1),-1,1)))
        if lb.max()>.012 or (lb-la).max()>.004 or np.any(tb>np.maximum(ta+40.,70.)):
            j=int(np.argmax(lb-la))
            raise RuntimeError('Ellipse continuity guard: '+json.dumps({'strand':i,'segment':j,
                'oldStepMm':float(la[j]*1000.),'newStepMm':float(lb[j]*1000.),'maximumTurnDegrees':float(tb.max()),
                'oldPoints':a[j:j+2].tolist(),'newPoints':b[j:j+2].tolist()}))
        ratio=float(lb.sum()/la.sum())
        if not .80<ratio<1.38:raise RuntimeError('Unexpected path-length change: '+str((i,ratio)))
        largest=max(largest,float(lb.max()));added=max(added,float((lb-la).max()))
        turns=max(turns,float((tb-ta).max()));lengths.append(ratio)
    return {'maximumTotalDisplacementMm':float(distance.max()*1000.),
            'maximumDisplacementBeyondT025Mm':float(distance[t>=.25].max()*1000.),
            'exactLast20PercentPoints':int(fixed.sum()),'maximumSegmentMm':largest*1000.,
            'maximumExtraSegmentMm':added*1000.,'maximumAdditionalTurnDegrees':turns,
            'lengthRatioRange':[min(lengths),max(lengths)]}


def run(source_relative=SOURCE,diagnostics_only=False):
    import bpy
    p=project_path();source=(p/source_relative).resolve()
    source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Exact fresh saved independent current source required')
    obj=bpy.data.objects[OBJECT];rig=bpy.data.objects['Character_Rig'];head=bpy.data.objects['R01 Head']
    if not obj.get('R09_forelock_flow') or not head.get('R09_lower_face_candidate'):
        raise RuntimeError('Accepted small forelock flow and R09 proportion required')
    if obj.get('R09_forelock_volume_candidate'):
        raise RuntimeError('Rejected volume candidate is NOT an allowed input; reopen root-flow/bodycombined')
    if any(key in obj for key in (MARKER,BACKUP,META)):raise RuntimeError('Ellipse candidate already exists')
    if obj.type!='CURVES' or obj.library or obj.data.library or obj.data.users!=1:
        raise RuntimeError('Local single-user native forelocks required')
    if any(m.show_render or m.show_viewport for m in obj.modifiers) or obj.animation_data or obj.data.animation_data:
        raise RuntimeError('Static original forelock data required')
    util=_api('posterior_groom_r09');flow=_api('forelock_flow_r09')
    volume=_api('forelock_volume_candidate_r09');uv=_api('production_uv_r08')
    original=obj.data;original_hash=util['_data_hash'](original)
    if json.loads(obj['R09_forelock_flow'])['candidateDataSha256']!=original_hash:
        raise RuntimeError('Forelocks no longer match accepted small-pass data; do not layer on unknown edits')
    source_sha=util['_sha'](source);before=flow['_protected'](bpy,util,uv)
    attachment={'parent':uv['_plain'](obj.parent),'type':obj.parent_type,'bone':obj.parent_bone,
                'basis':uv['_plain'](obj.matrix_basis),'inverse':uv['_plain'](obj.matrix_parent_inverse)}
    restore_record={'originalHash':original_hash,'originalFakeUser':original.use_fake_user}
    pose=rig.data.pose_position;prepared=None;committed=False
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        local=util['_values'](original.position_data,'vector');matrix=obj.matrix_world.copy();world=util['_world'](local,matrix)
        starts=np.array([c.first_point_index for c in original.curves],np.int32)
        sizes=np.array([c.points_length for c in original.curves],np.int32)
        with volume['_render_surface'](head) as subdivision:nearest,top=flow['_surface'](bpy,util)
        proposed,fixed,t,weights,clearance,groups=propose(world,starts,sizes,np.array([0.,.003,top-.10]))
        candidate,contact=_seat(world,proposed,fixed,t,weights,clearance,nearest)
        checks=_checks(world,candidate,starts,sizes,fixed,t)
        eyes=volume['_eyes'](bpy,util,world,candidate,fixed)
        new_local=util['_world'](candidate,matrix.inverted()).astype(np.float32);new_local[fixed]=local[fixed]
        if np.abs(util['_world'](new_local,matrix)-candidate).max()>2e-6:
            raise RuntimeError('World/local conversion exceeds two micrometres')
        prepared=original.copy();prepared.name='R09 elliptical layered forelocks candidate'
        prepared.position_data.foreach_set('vector',new_local.ravel())
        attrs=util['_attributes'](original);newattrs=util['_attributes'](prepared)
        for name,row in attrs.items():
            if name not in newattrs or row['domain']!=newattrs[name]['domain'] or row['type']!=newattrs[name]['type']:
                raise RuntimeError('Forelock attribute layout changed: '+name)
            if name!='position' and not np.array_equal(row['values'],newattrs[name]['values']):
                raise RuntimeError('Non-position attribute changed: '+name)
        if not np.array_equal(sizes,[c.points_length for c in prepared.curves]):raise RuntimeError('Curve order/count changed')
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        if flow['_protected'](bpy,util,uv)!=before:raise RuntimeError('Protected face/posterior/tail/rig changed')
        report={'source':str(source.relative_to(p)),'sourceSha256':source_sha,'sourceData':original.name,
            'sourceDataSha256':original_hash,'candidateDataSha256':util['_data_hash'](prepared),
            'object':OBJECT,'groups':groups,'geometry':checks,'contact':contact,'eyes':eyes,'headSurface':subdivision,
            'scope':'Accepted small-pass input only. Filled continuous elliptical main-lock volumes, posterior root patches and varied phase/roll. Last20% original paths exact.',
            'strandsAdded':0,'radiiMaterialsGroupOrderChanged':False,'posteriorTailRigChanged':False,
            'backupMetadataKey':BACKUP,'savedBlend':False,'writtenInMemory':not diagnostics_only,'appearanceAccepted':False,
            'requiredEvidence':'One same-light front and quarter pair. Reject if elliptical locks still read as plates/ropes, crown highlights line up, or roots leave gaps. No further blind inflation.',
            'runtimeConsequence':'Only if adopted, reproject seven fringe groups using this original data backup; no tail-cache changes.'}
        if diagnostics_only:return report
        original.use_fake_user=True;obj[BACKUP]=original.name;obj[META]=json.dumps(restore_record)
        obj.data=prepared;obj[MARKER]=json.dumps(report);obj.data.update_tag();obj.update_tag();bpy.context.view_layer.update()
        after_attachment={'parent':uv['_plain'](obj.parent),'type':obj.parent_type,'bone':obj.parent_bone,
                          'basis':uv['_plain'](obj.matrix_basis),'inverse':uv['_plain'](obj.matrix_parent_inverse)}
        if after_attachment!=attachment or flow['_protected'](bpy,util,uv)!=before:
            raise RuntimeError('Protected data/binding changed after candidate swap')
        if util['_data_hash'](original)!=original_hash or util['_sha'](source)!=source_sha:raise RuntimeError('Retained source changed')
        report['protectedDataUnchanged']=True;committed=True;return report
    finally:
        rig.data.pose_position=pose
        if not committed:
            if obj.data!=original:obj.data=original
            original.use_fake_user=restore_record['originalFakeUser']
            for key in (MARKER,BACKUP,META):
                if key in obj:del obj[key]
            if prepared is not None and prepared.users==0:bpy.data.hair_curves.remove(prepared)
        bpy.context.view_layer.update()


def restore():
    import bpy
    obj=bpy.data.objects[OBJECT]
    if not obj.get(MARKER):return False
    util=_api('posterior_groom_r09');record=json.loads(obj[META]);original=bpy.data.hair_curves.get(obj.get(BACKUP,''))
    if original is None or util['_data_hash'](original)!=record['originalHash']:
        raise RuntimeError('Retained accepted forelocks missing/changed; no blind restore')
    candidate=obj.data;candidate.use_fake_user=True;obj.data=original;original.use_fake_user=record['originalFakeUser']
    for key in (MARKER,BACKUP,META):del obj[key]
    bpy.context.view_layer.update()
    return {'restored':original.name,'retainedCandidate':candidate.name,'savedBlend':False}


if __name__=='__main__':print('Loaded only. Call run(source_relative=...) explicitly; no automatic save/render.')
