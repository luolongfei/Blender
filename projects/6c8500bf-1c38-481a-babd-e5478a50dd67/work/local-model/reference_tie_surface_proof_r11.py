"""Read-only current-source tie/gather/tail surface proof for a REST wrapper.

prove(arrays, observer_namespace) returns a report with errors[]. The namespace
must be the reviewed observe_reference_tie_alignment_r11_v2.py runpy namespace.
No Blender process, render, model correction, scene change or save is performed.

Inputs: after-final_{world,starts,lengths,group,radius} (radius in world metres);
tail_{positive,negative}_after_{world,starts,lengths,radius,matrix} (observer raw
local radius), or explicit *_radius_world instead of radius/matrix; each observer
TIES prefix + _after_eval_{world,triangles}. Optional gather_native_after_*
world/starts/lengths/group may contain all curves or the ordered gather subset.
Its radius is conservatively bounded by each corresponding actual final curve's
maximum world radius if no explicit native radius_world is supplied.

Compact queried signed distances and sample/curve provenance are appended to
arrays under tieproof_*. Current collider geometry is freshly collected; hashes,
visibility, evaluation and AABBs are returned, without reusing old surface data.
"""
from pathlib import Path
import hashlib
import numpy as np

HAIR_LIMIT = -.0005
RING_LIMIT = -.0002
ROOT_ARC = .001
RAY_BUDGET_PER_SURFACE = 4096
RAY_EPSILON = .000002


def _put(arrays,key,value):
    if key in arrays:raise RuntimeError('Duplicate surface-proof evidence: '+key)
    arrays[key]=np.ascontiguousarray(value)


def _radius(arrays,prefix,lengths,*,world=False,required=True):
    explicit=prefix+'_radius_world';ordinary=prefix+'_radius'
    if explicit in arrays:values=np.asarray(arrays[explicit],float).reshape(-1)
    elif ordinary in arrays:
        values=np.asarray(arrays[ordinary],float).reshape(-1)
        if not world:
            if prefix+'_matrix' not in arrays:raise RuntimeError('Raw radius requires its actual object matrix: '+prefix)
            matrix=np.asarray(arrays[prefix+'_matrix'],float)
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise RuntimeError('Invalid radius matrix: '+prefix)
            values=values*float(np.linalg.svd(matrix[:3,:3],compute_uv=False).max())
    elif required:raise RuntimeError('Actual radius evidence missing: '+prefix)
    else:return None
    if len(values)==len(lengths):values=np.repeat(values,lengths)
    if len(values)!=int(np.sum(lengths)) or not np.isfinite(values).all() or np.any(values<0.):raise RuntimeError('Invalid actual radius domain/count/values: '+prefix)
    return values


def _bundle(arrays,prefix,*,world_radius=False,required_radius=True):
    q=np.asarray(arrays[prefix+'_world'],float);s=np.asarray(arrays[prefix+'_starts'],np.int64);n=np.asarray(arrays[prefix+'_lengths'],np.int64)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise RuntimeError('Invalid world points: '+prefix)
    if s.ndim!=1 or s.shape!=n.shape or not len(s) or np.any(n<2) or int(n.sum())!=len(q):raise RuntimeError('Invalid complete curve topology: '+prefix)
    if not np.array_equal(s,np.cumsum(np.r_[0,n[:-1]],dtype=np.int64)):raise RuntimeError('Noncontiguous curve starts: '+prefix)
    radius=_radius(arrays,prefix,n,world=world_radius,required=required_radius)
    arc=np.empty(len(q),float)
    for start,count in zip(s,n):arc[start:start+count]=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q[start:start+count],axis=0),axis=1))]
    group=np.asarray(arrays[prefix+'_group'],np.int64) if prefix+'_group' in arrays else None
    if group is not None and group.shape!=s.shape:raise RuntimeError('Invalid curve groups: '+prefix)
    return {'prefix':prefix,'q':q,'s':s,'n':n,'radius':radius,'arc':arc,'group':group}


def _samples(bundle,ids,*,tail=False):
    q=bundle['q'];s=bundle['s'];n=bundle['n'];rad=bundle['radius'];arc=bundle['arc']
    indices=np.concatenate([np.arange(s[i],s[i]+n[i],dtype=np.int64) for i in ids])
    owners=np.repeat(ids,n[ids]);within=indices-np.repeat(s[ids],n[ids])
    if tail and np.any(n<3):raise RuntimeError('Native tail attachment requires three actual pins')
    boundaries=(arc[s+2] if tail else np.full(len(s),ROOT_ARC))
    free=arc[indices]>boundaries[owners]+1e-12
    points={'q':q[indices],'radius':rad[indices],'free':free,'curve':owners,'within':within,'source':indices,'arc':arc[indices],'kind':'point'}
    left=np.concatenate([np.arange(s[i],s[i]+n[i]-1,dtype=np.int64) for i in ids]);own=np.repeat(ids,n[ids]-1)
    distance=arc[left+1]-arc[left];midarc=arc[left]+distance*.5
    mids={'q':(q[left]+q[left+1])*.5,'radius':(rad[left]+rad[left+1])*.5,'free':midarc>boundaries[own]+1e-12,
          'curve':own,'within':left-s[own],'source':left,'arc':midarc,'kind':'segment_midpoint'}
    clip=np.clip((boundaries[own]-arc[left])/np.maximum(distance,1e-20),0.,1.)
    aa=q[left]+clip[:,None]*(q[left+1]-q[left]);bb=q[left+1]
    segments={'a':aa,'b':bb,'radius':np.maximum(rad[left],rad[left+1]),'free':(distance>1e-12)&(clip<1.),
              'curve':own,'within':left-s[own],'source':left,'arc':np.maximum(arc[left],boundaries[own]),'kind':'segment'}
    return points,mids,segments


def _witness(samples,index,signed,effective):
    return {'sample':int(index),'curve':int(samples['curve'][index]),'pointOrSegment':int(samples['within'][index]),
            'sourcePointOrVertex':int(samples['source'][index]),'sampleKind':samples['kind'],'arcMeters':float(samples['arc'][index]),
            'world':samples['q'][index].tolist(),'worldRadiusMeters':float(samples['radius'][index]),
            'signedCenterMeters':float(signed),'signedSurfaceMeters':float(effective)}


def _scan_points(samples,surface,label,arrays,limit):
    from mathutils import Vector
    q=samples['q'];r=samples['radius'];lo=surface['lo'];hi=surface['hi'];margin=r+abs(limit)
    candidate=np.all(q>=lo-margin[:,None],axis=1)&np.all(q<=hi+margin[:,None],axis=1)
    indices=np.flatnonzero(candidate);signed=np.empty(len(indices),np.float32)
    for j,index in enumerate(indices):
        hit,normal,_,distance=surface['tree'].find_nearest(Vector(q[index]))
        if hit is None:raise RuntimeError('Current surface nearest query failed: '+surface['name'])
        signed[j]=float((Vector(q[index])-hit).dot(normal.normalized()))
    effective=signed.astype(float)-r[indices];free=samples['free'][indices]
    prefix='tieproof_'+label+'_against_'+surface['prefix']
    _put(arrays,prefix+'_sample_index',indices.astype(np.int32));_put(arrays,prefix+'_signed_center',signed);_put(arrays,prefix+'_signed_surface',effective.astype(np.float32))
    row={'surface':surface['name'],'surfacePrefix':surface['prefix'],'samples':len(q),'aabbQueriedSamples':len(indices),'aabbExcludedSamples':int((~candidate).sum()),
         'minimumAllowedSignedSurfaceMeters':limit,'evidencePrefix':prefix,'freeSamples':int(samples['free'].sum()),'attachmentSamples':int((~samples['free']).sum())}
    for name,mask in (('free',free),('attachment',~free)):
        tested=np.flatnonzero(mask);bad=tested[effective[tested]<limit]
        row[name+'QueriedSamples']=len(tested);row[name+'SamplesBelowLimit']=len(bad)
        row[name+'MinimumSignedCenterMeters']=float(signed[tested].min()) if len(tested) else None
        row[name+'MinimumSignedSurfaceMeters']=float(effective[tested].min()) if len(tested) else None
        worst=int(tested[np.argmin(effective[tested])]) if len(tested) else None
        row[name+'Worst']=_witness(samples,int(indices[worst]),signed[worst],effective[worst]) if worst is not None else None
    return row


def _scan_rays(segments,surface,label,arrays,limit):
    """Bounded real segment crossings; depth probes retain the same tolerance."""
    from mathutils import Vector
    aa=segments['a'];bb=segments['b'];length=np.linalg.norm(bb-aa,axis=1)
    eligible=segments['free']&(length>RAY_EPSILON*3.)&np.all(np.minimum(aa,bb)<=surface['hi'],axis=1)&np.all(np.maximum(aa,bb)>=surface['lo'],axis=1)
    candidates=np.flatnonzero(eligible)
    tested=candidates if len(candidates)<=RAY_BUDGET_PER_SURFACE else candidates[np.linspace(0,len(candidates)-1,RAY_BUDGET_PER_SURFACE,dtype=np.int64)]
    hits=[];hit_world=[];deep=[];minimum=None
    for index in tested:
        direction=(bb[index]-aa[index])/length[index]
        hit,normal,face,distance=surface['tree'].ray_cast(Vector(aa[index]+direction*RAY_EPSILON),Vector(direction),float(length[index]-RAY_EPSILON*2.))
        if hit is None:continue
        along=float(distance)+RAY_EPSILON;hits.append(int(index));hit_world.append(tuple(hit))
        for step in (.00025,.0005,.001):
            if along+step>=length[index]-RAY_EPSILON:continue
            point=aa[index]+direction*(along+step);nearest,n,_,_=surface['tree'].find_nearest(Vector(point))
            signed=float((Vector(point)-nearest).dot(n.normalized()))-float(segments['radius'][index])
            minimum=signed if minimum is None else min(minimum,signed)
            if signed<limit:deep.append({'segment':int(index),'curve':int(segments['curve'][index]),'withinCurve':int(segments['within'][index]),'world':point.tolist(),'signedSurfaceMeters':signed})
    prefix='tieproof_'+label+'_against_'+surface['prefix']+'_ray'
    _put(arrays,prefix+'_tested_segment_index',tested.astype(np.int32));_put(arrays,prefix+'_crossing_segment_index',np.asarray(hits,np.int32));_put(arrays,prefix+'_hit_world',np.asarray(hit_world,np.float32).reshape(-1,3))
    return {'surface':surface['name'],'candidateFreeSegments':len(candidates),'testedSegments':len(tested),'exhaustiveForAabbCandidates':len(tested)==len(candidates),
            'centerlineCrossings':len(hits),'postCrossingDepthSamplesBelowLimit':len(deep),'postCrossingMinimumSignedSurfaceMeters':minimum,
            'belowLimitExamples':deep[:24],'evidencePrefix':prefix,'scope':'Deterministic bounded real centerline ray queries, clipped outside the root attachment zone; crossings alone do not measure intrusion depth.'}


def _family(name,samples,segments,surfaces,arrays,errors,limit):
    for suffix,value in (('curve',samples['curve']),('within_curve',samples['within']),('source_point_or_vertex',samples['source']),('arc',samples['arc'])):
        _put(arrays,'tieproof_'+name+'_'+suffix,np.asarray(value,np.float32 if suffix=='arc' else np.int32))
    rows=[];rays=[]
    for surface in surfaces:
        row=_scan_points(samples,surface,name,arrays,limit);rows.append(row)
        if row['freeSamplesBelowLimit']:errors.append(name+' / '+surface['name']+': '+str(row['freeSamplesBelowLimit'])+' free samples below '+str(limit)+' m')
        if segments is not None:
            ray=_scan_rays(segments,surface,name,arrays,limit);rays.append(ray)
            if ray['postCrossingDepthSamplesBelowLimit']:errors.append(name+' / '+surface['name']+': '+str(ray['postCrossingDepthSamplesBelowLimit'])+' real crossing depth probes below '+str(limit)+' m')
    print('R11_TIE_SURFACE_FAMILY '+name+' samples='+str(len(samples['q']))+' errors='+str(sum(r['freeSamplesBelowLimit'] for r in rows)),flush=True)
    return {'sampleKind':samples['kind'],'samples':len(samples['q']),'surfaces':rows,'segmentRayEvidence':rays}


def _ring_samples(q,tri):
    if q.ndim!=2 or q.shape[1]!=3 or tri.ndim!=2 or tri.shape[1]!=3 or not len(q) or not len(tri):raise RuntimeError('Actual evaluated ring geometry required')
    if not np.isfinite(q).all() or np.any(tri<0) or np.any(tri>=len(q)):raise RuntimeError('Invalid evaluated ring mesh')
    edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)
    def make(points,kind):
        count=len(points);return {'q':points,'radius':np.zeros(count),'free':np.ones(count,bool),'curve':np.full(count,-1,np.int32),
                                  'within':np.arange(count),'source':np.arange(count),'arc':np.zeros(count),'kind':kind}
    segments={'a':q[edges[:,0]],'b':q[edges[:,1]],'radius':np.zeros(len(edges)),'free':np.ones(len(edges),bool),
              'curve':np.full(len(edges),-1,np.int32),'within':np.arange(len(edges)),'source':np.arange(len(edges)),'arc':np.zeros(len(edges)),'kind':'mesh_edge'}
    return [('vertices',make(q,'mesh_vertex'),None),('triangle_centers',make(q[tri].mean(1),'triangle_center'),None),
            ('edge_midpoints',make(q[edges].mean(1),'unique_edge_midpoint'),segments)],edges


def prove(arrays,observer_namespace):
    import bpy
    from mathutils.bvhtree import BVHTree
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Current surface proof requires the wrapper neutral REST state')
    for name in ('COLLIDERS','TIES','visibility','collect_mesh'):
        if name not in observer_namespace:raise RuntimeError('Reviewed observer namespace missing '+name)
    errors=[];surfaces=[];surface_rows=[];excluded=[];graph=bpy.context.evaluated_depsgraph_get()
    for prefix,name in observer_namespace['COLLIDERS']:
        obj=bpy.data.objects.get(name)
        if obj is None or obj.type!='MESH':raise RuntimeError('Current required collider object/type missing: '+name)
        visible=observer_namespace['visibility'](obj)
        if not visible['renderEligibleByCollectedFlags']:
            excluded.append({'object':name,'prefix':prefix,'visibility':visible,'reason':'Not currently render eligible; no exclusion by name'});continue
        temporary={};row=observer_namespace['collect_mesh'](obj,'current_'+prefix,temporary,graph=graph,require_actual_evaluation=True)
        if not row.get('evaluated'):raise RuntimeError('Render-eligible collider lacks current evaluated geometry: '+name)
        q=np.asarray(temporary['current_'+prefix+'_world'],float);tri=np.asarray(temporary['current_'+prefix+'_triangles'],np.int32)
        surfaces.append({'name':name,'prefix':prefix,'lo':q.min(0),'hi':q.max(0),'tree':BVHTree.FromPolygons(q.tolist(),tri.tolist(),all_triangles=True)})
        surface_rows.append(row)
    if not surfaces or not any(s['prefix']=='head' for s in surfaces):raise RuntimeError('Current evaluated render-eligible head is required')
    main=_bundle(arrays,'after-final',world_radius=True);groups=main['group']
    if groups is None:raise RuntimeError('Actual final dense curve groups required')
    gather=np.flatnonzero(np.isin(groups%256,(40,41)))
    if not len(gather):raise RuntimeError('Actual final gather curves missing')
    report={'errors':errors,'savedBlend':False,'rendered':False,'helperSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'currentCollisionSurfaces':surface_rows,'excludedCollisionSurfaces':excluded,'mandatorySurfaceNames':[s['name'] for s in surfaces],
            'hairFreeSurfaceLimitMeters':HAIR_LIMIT,'ringSurfaceLimitMeters':RING_LIMIT,'gatherAttachmentArcMeters':ROOT_ARC,
            'tailAttachmentPolicy':'First three actual native pins and their two internal segments; the next segment is free.',
            'signedConvention':'Nearest current triangle normal signed center distance minus actual world hair radius; rings use actual sampled surface positions.',
            'aabbPolicy':'Only exclude samples outside the surface AABB expanded by actual radius plus the absolute applicable limit. Interior AABB samples are always queried.',
            'families':{},'limitations':['Nearest-normal signed clearance is not a watertight-mesh containment proof. All requested point/midpoint samples are checked; bounded ray evidence is explicitly reported.',
                                         'Evaluated render-eligible collector surfaces are used without exclusions by object name; hidden surfaces are reported separately.',
                                         'Static REST surface proof does not certify subsequent animation or customer appearance acceptance.']}
    families=report['families']
    pp,mm,ss=_samples(main,np.arange(len(main['s'])))
    families['all_main_control_points_current_head']=_family('all_main_control_points_current_head',pp,None,[s for s in surfaces if s['prefix']=='head'],arrays,errors,HAIR_LIMIT)
    pp,mm,ss=_samples(main,gather)
    families['gather_control_points']=_family('gather_control_points',pp,None,surfaces,arrays,errors,HAIR_LIMIT)
    families['gather_control_midpoints']=_family('gather_control_midpoints',mm,ss,surfaces,arrays,errors,HAIR_LIMIT)
    for side in ('positive','negative'):
        prefix='tail_'+side+'_after';tail=_bundle(arrays,prefix)
        pp,mm,ss=_samples(tail,np.arange(len(tail['s'])),tail=True)
        families[prefix+'_points']=_family(prefix+'_points',pp,None,surfaces,arrays,errors,HAIR_LIMIT)
        families[prefix+'_midpoints']=_family(prefix+'_midpoints',mm,ss,surfaces,arrays,errors,HAIR_LIMIT)
    native_key='gather_native_after'
    if native_key+'_world' in arrays:
        native=_bundle(arrays,native_key,world_radius=True,required_radius=False);ng=native['group']
        if ng is None:raise RuntimeError('Native sampled gather group provenance required')
        if np.array_equal(ng,groups):mapping=np.arange(len(groups))
        elif np.array_equal(ng,groups[gather]):mapping=gather
        else:raise RuntimeError('Native gather order/role identity does not match current final geometry')
        root_error=np.linalg.norm(native['q'][native['s']]-main['q'][main['s'][mapping]],axis=1)
        if float(root_error.max())>2e-6:raise RuntimeError('Native gather sampled roots no longer match current final roots')
        selected=np.flatnonzero(np.isin(ng%256,(40,41)))
        if not len(selected):raise RuntimeError('Native path dataset has no gather curves')
        if np.any(native['n'][selected]!=64):raise RuntimeError('Actual native gather paths must contain the requested 64 samples per selected curve')
        _put(arrays,'tieproof_native_to_main_curve',mapping.astype(np.int32))
        if native['radius'] is None:
            maximum=np.asarray([main['radius'][start:start+count].max() for start,count in zip(main['s'],main['n'])])
            native['radius']=np.repeat(maximum[mapping],native['n']);report['nativeRadiusPolicy']='Actual corresponding final curve maximum world radius; conservative along the full native path'
        else:report['nativeRadiusPolicy']='Explicit actual world native radius'
        pp,mm,ss=_samples(native,selected)
        families['gather_native_points']=_family('gather_native_points',pp,None,surfaces,arrays,errors,HAIR_LIMIT)
        families['gather_native_midpoints']=_family('gather_native_midpoints',mm,ss,surfaces,arrays,errors,HAIR_LIMIT)
        report['nativeGatherPathsChecked']=True;report['nativeSamplesPerGatherCurve']=64;report['nativeRootMatchMaximumMeters']=float(root_error.max())
    else:report['nativeGatherPathsChecked']=False
    for prefix,name in observer_namespace['TIES']:
        key=prefix+'_after_eval';q=np.asarray(arrays[key+'_world'],float);tri=np.asarray(arrays[key+'_triangles'],np.int64)
        batches,edges=_ring_samples(q,tri);_put(arrays,'tieproof_'+prefix+'_unique_edges',edges.astype(np.int32))
        for suffix,samples,segments in batches:
            label=prefix+'_'+suffix;families[label]=_family(label,samples,segments,surfaces,arrays,errors,RING_LIMIT)
    report['allRequestedSampleGatesPassed']=not errors
    return report
