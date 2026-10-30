"""Reversible posterior-only fibres appended to the recognized R06 scalp object.

In a new background process opened on the explicitly selected production copy:
    run(source_relative='work/reconstruction-r02/reference-production-uv-r08.blend')
    restore()  # restore retained original Curves data and runtime metadata

No save/render on import or call. Existing 6,720 curves are byte-preserved;
320 spatially compact groups of 30 projected fibres are appended. Object name,
parent, material, front fibres, tail geometry and native tail playback remain.
The root substrate is not modified. Actual matched back/quarter/top renders are
required before adopting this candidate; geometric checks are not appearance QA.
"""
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SCALP='R06 continuous scalp flow'
SHELL='R06 attached scalp root substrate'
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
MARKER='R09_posterior_groom'
BACKUP='R09_posterior_original_data'
META='R09_posterior_restore_record'
GROUPS='RT09_runtime_group_counts'
PROVENANCE='RT09_runtime_group_provenance'
OLD_GROUPS=[32]*192+[12]*48
ROWS=80;CHILDREN=30;POINTS=51;RADIUS=.000045
FIELDS={'FLOAT':('value',1,np.float32),'INT':('value',1,np.int32),
    'INT8':('value',1,np.int32),'BOOLEAN':('value',1,np.bool_),
    'INT16_2D':('value',2,np.int32),
    'FLOAT_VECTOR':('vector',3,np.float32),'FLOAT2':('vector',2,np.float32),
    'FLOAT_COLOR':('color',4,np.float32),'BYTE_COLOR':('color',4,np.float32)}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def _array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def _values(collection,field,width=3):
    a=np.empty(len(collection)*width,np.float32);collection.foreach_get(field,a)
    return a.reshape(-1,width)


def _attributes(data):
    rows={}
    for attr in data.attributes:
        if hasattr(data,'polygons') and attr.name in {'.edge_verts','.corner_vert','.corner_edge'}:continue
        if attr.data_type not in FIELDS:raise RuntimeError('Unreviewed original attribute: '+attr.name+' / '+attr.data_type)
        field,width,dtype=FIELDS[attr.data_type]
        values=np.empty(len(attr.data)*width,dtype=dtype);attr.data.foreach_get(field,values)
        rows[attr.name]={'domain':attr.domain,'type':attr.data_type,'values':values.reshape(-1,width)}
    return rows


def _data_hash(data):
    h=hashlib.sha256()
    for name,row in sorted(_attributes(data).items()):
        h.update((name+row['domain']+row['type']).encode());h.update(row['values'].tobytes())
    if hasattr(data,'curves'):
        h.update(np.array([c.points_length for c in data.curves],np.int32).tobytes())
    if hasattr(data,'polygons'):
        for face in data.polygons:h.update(np.array(list(face.vertices),np.int32).tobytes())
    return h.hexdigest()


def _world(points,matrix):
    m=np.array(matrix,dtype=float);return points@m[:3,:3].T+m[:3,3]


def _smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def _protected(bpy,uv):
    rows={}
    for obj in bpy.context.scene.objects:
        if obj.type=='CURVES' and obj.name!=SCALP or obj.name==SHELL:
            rows[obj.name]={'data':obj.data.name,'hash':_data_hash(obj.data),
                'world':uv['_plain'](obj.matrix_world),'parent':uv['_plain'](obj.parent),
                'animation':uv['_animation'](obj),'modifiers':[uv['_scalar_rna'](m) for m in obj.modifiers]}
    head=bpy.data.objects['R01 Head'];keys=head.data.shape_keys
    rows['head']={'mesh':_data_hash(head.data),
        'keys':[{'name':key.name,'co':_array_hash(_values(key.data,'co')),
                 'value':key.value,'mute':key.mute,'relative':key.relative_key.name} for key in keys.key_blocks],
        'animation':uv['_animation'](keys)}
    rows['rig']=uv['_pose_contract']()
    return rows


def _surface(bpy,scalp,rig):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    head=bpy.data.objects['R01 Head'];graph=bpy.context.evaluated_depsgraph_get()
    evaluated=head.evaluated_get(graph);mesh=evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices=_world(_values(mesh.vertices,'co'),evaluated.matrix_world)
        triangles=np.asarray([triangle.vertices[:] for triangle in mesh.loop_triangles],dtype=np.int32)
        normals=_values(mesh.vertices,'normal').astype(float)
        normals=normals@np.linalg.inv(np.array(evaluated.matrix_world)[:3,:3])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    finally:evaluated.to_mesh_clear()
    roots=[]
    for name in TAILS:
        obj=bpy.data.objects[name];sizes=[curve.points_length for curve in obj.data.curves]
        if len(sizes)!=2480 or set(sizes)!={35}:raise RuntimeError('Unexpected original tail layout: '+name)
        points=_world(_values(obj.data.position_data,'vector'),obj.matrix_world).reshape(2480,35,3)
        roots.append(points[:,0].mean(axis=0))
    roots=np.array(roots);top=float(vertices[:,2].max());scale=(top-float(roots[:,2].mean()))/.158
    if not .85<scale<1.35:raise RuntimeError('Head/tail scale escaped the reviewed scalp relationship')
    center=np.array([roots[:,0].mean(),roots[:,1].mean()+.007*scale,top-.108*scale])
    tree=BVHTree.FromPolygons([Vector(v) for v in vertices],triangles.tolist(),all_triangles=True)
    stats={'rays':0,'misses':0,'normalFlips':0}

    def surface(theta,phi,side,lift):
        direction=np.array([side*math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)])
        hit,face_normal,index,_=tree.ray_cast(Vector(center+direction*.40),Vector(-direction),.80)
        stats['rays']+=1
        if hit is None:
            stats['misses']+=1;raise RuntimeError('Posterior radial ray missed the actual head; no fallback')
        hit=np.asarray(hit);triangle=triangles[index];a,b,c=vertices[triangle]
        e0=b-a;e1=c-a;d=hit-a
        d00=np.dot(e0,e0);d01=np.dot(e0,e1);d11=np.dot(e1,e1)
        den=d00*d11-d01*d01
        if abs(den)<1e-20:raise RuntimeError('Degenerate head surface triangle')
        v=(d11*np.dot(d,e0)-d01*np.dot(d,e1))/den
        w=(d00*np.dot(d,e1)-d01*np.dot(d,e0))/den
        n=normals[triangle[0]]*(1-v-w)+normals[triangle[1]]*v+normals[triangle[2]]*w
        n/=max(np.linalg.norm(n),1e-12)
        if np.dot(n,direction)<0:n=-n;stats['normalFlips']+=1
        return hit+n*lift,n

    def boundary(point):
        d=point-center;phi=math.atan2(d[1],abs(d[0]))
        return center[2]+scale*(.017+.030*max(0,-math.sin(phi))-.040*max(0,math.sin(phi))
                    +.007*math.exp(-((phi-.18)/.28)**2))

    phi_table=np.linspace(.48,math.pi/2,97);limits={}
    for side in (1,-1):
        row=[]
        for phi in phi_table:
            lo,hi=.02,2.05
            for _ in range(11):
                mid=(lo+hi)/2;q,_=surface(mid,float(phi),side,0.)
                if q[2]>=boundary(q):lo=mid
                else:hi=mid
            row.append(lo)
        limits[side]=row
    return surface,center,scale,phi_table,limits,stats


def _build(surface,center,scale,phi_table,limits,rows=ROWS):
    """Spatial group order is also the runtime card-conversion group order."""
    rng=np.random.default_rng(909028);paths=[];groups=[];max_step=0.;lift_bounds=[1.,0.]
    for side in (1,-1):
        for root_band in ('part','outer'):
            for row in range(rows):
                fraction=(row+.5)/rows
                phi0=(math.pi/2-.014 if root_band=='part' else
                      1.02+.43*((row*.61803398875+(.11 if side==1 else .43))%1))
                end_phi=.66+.06*math.sin(fraction*math.pi)+(.02 if root_band=='outer' else 0)
                end_theta=1.61+.09*fraction
                curve_offset=len(paths)
                for child in range(CHILDREN):
                    phi_start=float(np.clip(phi0+rng.uniform(-.011,.011),.98,math.pi/2-.002))
                    root_limit=float(np.interp(phi_start,phi_table,limits[side]))-.045
                    u=np.clip(fraction+rng.uniform(-.46,.46)/rows,0,1)
                    theta_start=.030+(root_limit-.030)*u
                    terminal_phi=end_phi+rng.uniform(-.008,.008)
                    terminal_theta=end_theta+rng.uniform(-.009,.009)
                    phase=rng.uniform(0,math.tau);depth=rng.uniform(.00005,.00055)*scale
                    path=[]
                    for j in range(POINTS):
                        t=j/(POINTS-1)
                        # Strong early sideways combing covers both banks of the
                        # narrow posterior part; a small late bend enters the
                        # existing broad gather region rather than a single pin.
                        drift=t*(1.18-.18*t)
                        phi=phi_start+(terminal_phi-phi_start)*drift
                        theta=theta_start+(terminal_theta-theta_start)*t
                        theta+=.018*math.sin(math.pi*t)*math.sin(phase)*scale
                        theta=min(theta,float(np.interp(phi,phi_table,limits[side]))-.026)
                        lift=(.00073+.00048*math.sin(math.pi*t))*scale+depth
                        lift+=.00005*scale*math.sin(phase+t*5)*math.sin(math.pi*t)
                        q,_=surface(theta,phi,side,lift);path.append(q)
                        lift_bounds=[min(lift_bounds[0],lift),max(lift_bounds[1],lift)]
                    path=np.array(path);step=float(np.linalg.norm(np.diff(path,axis=0),axis=1).max())
                    if not np.isfinite(path).all() or step>.015:
                        raise RuntimeError('Discontinuous posterior fibre: '+json.dumps({'side':side,'row':row,'step':step}))
                    max_step=max(max_step,step);paths.append(path)
                groups.append({'side':side,'rootBand':root_band,'row':row,'appendCurveOffset':curve_offset,
                               'strands':CHILDREN,'pointsPerStrand':POINTS})
    return np.asarray(paths,dtype=np.float64),groups,{'maximumPointStepMeters':max_step,'surfaceLiftRangeMeters':lift_bounds}


def _posterior_density(world,center,scale):
    # Point-density evidence only; no claim of raster coverage or occlusion.
    x=(world[:,0]-center[0])/scale;y=(world[:,1]-center[1])/scale;z=(world[:,2]-center[2])/scale
    mask=(np.abs(x)<.028)&(y>.045)&(z>-.025)&(z<.092)
    hist,_,_=np.histogram2d(x[mask],z[mask],bins=(16,32),range=((- .028,.028),(-.025,.092)))
    return {'centralPosteriorPointSamples':int(mask.sum()),'occupiedBins':int((hist>0).sum()),
            'totalBins':512,'emptyBins':int((hist==0).sum()),'method':'Fixed XZ point-density bins, restricted to posterior Y. Not visible coverage.'}


def run(source_relative='work/reconstruction-r02/reference-production-uv-r08.blend',diagnostics_only=False):
    import bpy
    p=project_path();source=(p/source_relative).resolve();source.relative_to((p/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the explicitly selected independent production source in background Object Mode')
    obj=bpy.data.objects[SCALP];rig=bpy.data.objects['Character_Rig']
    if any(name in obj for name in (MARKER,BACKUP,META)):
        raise RuntimeError('Posterior candidate/restore metadata already exists; preserve or restore before retrying')
    if obj.type!='CURVES' or obj.library or obj.data.library or obj.data.users!=1:
        raise RuntimeError('Expected local single-user original scalp Curves')
    if any(m.show_viewport or m.show_render for m in obj.modifiers) or obj.animation_data or obj.data.animation_data:
        raise RuntimeError('Static original scalp expected; preserve unexpected animation/modifiers')
    if len(obj.data.curves)!=sum(OLD_GROUPS) or len(obj.data.materials)!=1:
        raise RuntimeError('Original 6720-fibre R06 scalp/material layout changed')
    if GROUPS in obj and json.loads(obj[GROUPS])!=OLD_GROUPS:raise RuntimeError('Existing runtime group metadata differs from original layout')
    uv=_api('production_uv_r08');bpy.context.view_layer.update()
    before=_protected(bpy,uv);source_sha=_sha(source);original=obj.data
    restore_record={'originalHash':_data_hash(original),'originalFakeUser':original.use_fake_user,
                    'metadata':{name:obj.get(name) for name in (GROUPS,PROVENANCE)}}
    old_hash=_data_hash(original);old_attributes=_attributes(original)
    old_lengths=[curve.points_length for curve in original.curves]
    attachment={'parent':uv['_plain'](obj.parent),'type':obj.parent_type,'bone':obj.parent_bone,
                'basis':uv['_plain'](obj.matrix_basis),'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
    pose=rig.data.pose_position;prepared=None;committed=False
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        world_matrix=np.asarray(obj.matrix_world);scales=np.linalg.norm(world_matrix[:3,:3],axis=0)
        if np.max(np.abs(scales-1))>1e-5:raise RuntimeError('45 micrometre radii require the existing unit scalp transform')
        surface,center,scale,phis,limits,ray_stats=_surface(bpy,obj,rig)
        paths,groups,numeric=_build(surface,center,scale,phis,limits)
        if paths.shape!=(9600,POINTS,3):raise RuntimeError('Unexpected appended fibre count')
        old_world=_world(_values(original.position_data,'vector'),obj.matrix_world)
        local=_world(paths.reshape(-1,3),obj.matrix_world.inverted()).astype(np.float32)
        roundtrip=_world(local,obj.matrix_world)
        conversion_error=float(np.abs(roundtrip-paths.reshape(-1,3)).max())
        if conversion_error>2e-6:raise RuntimeError('World/local attachment conversion lost precision')
        prepared=original.copy();prepared.name='R09 posterior-complete scalp flow'
        prepared.add_curves([POINTS]*len(paths))
        old_positions=old_attributes['position']['values']
        prepared.position_data.foreach_set('vector',np.concatenate((old_positions,local)).ravel())
        radius=prepared.attributes.get('radius')
        if radius is None or radius.data_type!='FLOAT' or radius.domain!='POINT':raise RuntimeError('Original point radii missing')
        t=np.linspace(0,1,POINTS);taper=np.maximum(.08,(1-t**4.5)**.48)*(.64+.36*_smooth(t/.07))
        variation=.84+.27*((np.arange(len(paths))*.61803398875)%1)
        radii=(RADIUS*variation[:,None]*taper[None,:]).astype(np.float32).ravel()
        radius.data.foreach_set('value',np.concatenate((old_attributes['radius']['values'].ravel(),radii)))
        # add_curves normally uses the same native default as the R06 creator.
        # Preserve any explicit original curve type on the appended curves too.
        curve_type=prepared.attributes.get('curve_type')
        if curve_type and 'curve_type' in old_attributes:
            values=old_attributes['curve_type']['values'].ravel()
            if len(np.unique(values))!=1:raise RuntimeError('Mixed original curve types require explicit append handling')
            curve_type.data.foreach_set('value',np.concatenate((values,np.full(len(paths),values[0],values.dtype))))
        current_attributes=_attributes(prepared)
        for name,record in old_attributes.items():
            new=current_attributes.get(name)
            if not new or new['domain']!=record['domain'] or new['type']!=record['type'] or not np.array_equal(
                new['values'][:len(record['values'])],record['values']):
                raise RuntimeError('An existing scalp attribute prefix changed: '+name)
        if [curve.points_length for curve in prepared.curves][:len(old_lengths)]!=old_lengths:
            raise RuntimeError('Existing scalp curve order/length changed')
        report={'source':str(source.relative_to(p)),'sourceSha256':source_sha,'object':SCALP,
            'sourceCurvesData':original.name,'sourceCurvesSha256':old_hash,'candidateCurvesSha256':_data_hash(prepared),
            'oldStrands':len(old_lengths),'newStrands':len(prepared.curves),'appendedStrands':len(paths),
            'appendCurveOffset':len(old_lengths),'appendPointOffset':len(old_positions),'newPointsPerStrand':POINTS,
            'oldGroupCounts':OLD_GROUPS,'newGroupCounts':[CHILDREN]*len(groups),'newGroups':groups,
            'grouping':'Per side: 80 compact posterior-part groups then 80 compact outer groups. Thirty adjacent projected fibres per group.',
            'radiusMeters':RADIUS,'actualHeadScale':scale,'projection':ray_stats,
            'projectionMethod':'Ray hit on current evaluated REST head; barycentric smooth vertex normal; bounded outward lift. No ellipsoid fallback.',
            'localConversionMaximumErrorMeters':conversion_error,**numeric,
            'posteriorDensityBefore':_posterior_density(old_world,center,scale),
            'posteriorDensityAfter':_posterior_density(np.concatenate((old_world,paths.reshape(-1,3))),center,scale),
            'originalAttributePrefixesUnchanged':True,'oldScalpCurveOrderUnchanged':True,
            'rootSubstrateBeforeSha256':before[SHELL]['hash'],'rootSubstrateAfterSha256':before[SHELL]['hash'],
            'rootGeometryUVAlphaMaterialsChanged':False,'existingTailPointsChanged':0,
            'savedBlend':False,'writtenInMemory':not diagnostics_only,
            'requiredEvidence':['Same-camera/light back, both rear-quarter and crown renders before and after.',
                'Check narrow part width, central posterior coverage, nape edge, no triangular striping or new halo.',
                'Check original front portrait and side silhouette; inspect head turn with existing native tail playback.',
                'Runtime scalp cards/atlas require explicit refresh using RT09_runtime_group_counts; no automatic export claim.'],
            'appearanceAccepted':False}
        # Restore evaluation before any persistent object data swap and validate
        # everything outside the scalp candidate against the initial state.
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        if _protected(bpy,uv)!=before:raise RuntimeError('Protected face/root/tail/rig data changed during preparation')
        if _data_hash(original)!=old_hash:raise RuntimeError('Original scalp data changed')
        if diagnostics_only:return report
        original.use_fake_user=True;obj[BACKUP]=original.name;obj[META]=json.dumps(restore_record)
        obj.data=prepared;obj[MARKER]=True
        obj[GROUPS]=json.dumps(OLD_GROUPS+[CHILDREN]*len(groups))
        obj[PROVENANCE]='R09 posterior-only supplement: original 240 groups retained; 320 spatially compact projected groups x 30, ordered side/root-band/latitude. Source '+source_sha
        obj.data.update_tag();obj.update_tag();bpy.context.view_layer.update()
        after_attachment={'parent':uv['_plain'](obj.parent),'type':obj.parent_type,'bone':obj.parent_bone,
                'basis':uv['_plain'](obj.matrix_basis),'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}
        if attachment!=after_attachment or _protected(bpy,uv)!=before:
            raise RuntimeError('Attachment or protected data changed after candidate assignment')
        report['protectedDataUnchanged']=True;report['retainedOriginalData']=original.name
        committed=True;return report
    finally:
        rig.data.pose_position=pose
        if not committed:
            if obj.data!=original:obj.data=original
            original.use_fake_user=restore_record['originalFakeUser']
            for name,value in restore_record['metadata'].items():
                if value is None:
                    if name in obj:del obj[name]
                else:obj[name]=value
            for name in (MARKER,BACKUP,META):
                if name in obj:del obj[name]
        if not committed and prepared is not None and prepared.users==0:bpy.data.hair_curves.remove(prepared)
        bpy.context.view_layer.update()


def restore():
    import bpy
    obj=bpy.data.objects[SCALP]
    if not obj.get(MARKER):return False
    original=bpy.data.hair_curves.get(obj.get(BACKUP,''));record=json.loads(obj[META])
    if original is None or _data_hash(original)!=record['originalHash']:
        raise RuntimeError('Retained original scalp data missing or changed; refuse blind restore')
    candidate=obj.data;candidate.use_fake_user=True;obj.data=original;original.use_fake_user=record['originalFakeUser']
    for name,value in record['metadata'].items():
        if value is None:
            if name in obj:del obj[name]
        else:obj[name]=value
    for name in (MARKER,BACKUP,META):del obj[name]
    bpy.context.view_layer.update()
    return {'restored':original.name,'retainedCandidate':candidate.name,'object':SCALP,'savedBlend':False}


if __name__=='__main__':print('Loaded only. Call run(source_relative=...) in an independent production copy; no auto save/render.')
