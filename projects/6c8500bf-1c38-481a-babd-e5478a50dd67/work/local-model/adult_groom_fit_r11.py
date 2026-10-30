"""Refit the existing groom to the real adult skull in an independent file.

The scalp root mesh follows the actual new skull at 0.25 mm; it is not inflated
to create apparent hair volume. Existing fibre offsets are transported by the
same continuous root correspondence. Only free strand midsections receive a
small, varying lift. Gather final two samples and complete native tails remain
bit-identical. The old groom datablocks are retained for a reversible candidate.

No import-time work, head/neck edits, rig changes, cache solve, render, or save.
Caller must already have the rig in REST while invoking fit_adult_groom().
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SHELL='R06 attached scalp root substrate'
SCALP='R06 continuous scalp flow'
FRINGE='R03 Forelocks from crown'
GATHERS=('R06 full gathered roots L','R06 full gathered roots R')
WISPS='R03 curved cheek wisps'
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
NAMES=(SHELL,SCALP,FRINGE)+GATHERS+(WISPS,)
MARKER='R11_adult_groom_fit'
OLD_CENTER=(0.,.00202,1.441285)
NEW_CENTER_DONOR=(0.,-.047,1.535)
ROOT_GAP=.00025
SOURCE_ARRAY_SHA='530a9079d845eb0df1aaaba006a38aa41a01a9791de3943f3df3a0da5e612de2'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _smooth(t):
    import numpy as np
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)


def _read_positions(obj):
    import numpy as np
    items,field=(obj.data.vertices,'co') if obj.type=='MESH' else (obj.data.position_data,'vector')
    a=np.empty(len(items)*3,dtype=np.float32);items.foreach_get(field,a)
    return a.reshape(-1,3)


def _world(a,matrix):
    import numpy as np
    m=np.asarray(matrix,dtype=np.float64);return a@m[:3,:3].T+m[:3,3]


def _summary(a):
    import numpy as np
    a=np.asarray(a,dtype=np.float64)
    return {'min':float(a.min()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def _head_surfaces(head,depsgraph):
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    evaluated=head.evaluated_get(depsgraph);me=evaluated.data
    co=np.empty(len(me.vertices)*3,dtype=np.float64);me.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    world=_world(co,evaluated.matrix_world)
    polygons=[tuple(p.vertices) for p in me.polygons]
    fs=me.attributes.get('.sculpt_face_set')
    if fs is None or fs.domain!='FACE' or fs.data_type!='INT':
        raise RuntimeError('Actual adult head face sets required to exclude ear surfaces from scalp anchoring')
    labels=np.empty(len(fs.data),dtype=np.int32);fs.data.foreach_get('value',labels)
    # The ear lobes, eye pouches and oral interior must never become substitute
    # scalp roots. The complete head remains the separate collision surface.
    skull_polys=[f for f,label in zip(polygons,labels) if int(label) not in (2,3,4,5,7,8)]
    points=[Vector(v) for v in world]
    skull=BVHTree.FromPolygons(points,skull_polys,all_triangles=False)
    skin=BVHTree.FromPolygons(points,polygons,all_triangles=False)
    return skull,skin,{'object':head.name,'evaluatedVertices':len(world),'evaluatedFaces':len(polygons),
                       'skullAnchorFaces':len(skull_polys),'excludedAnchorFaceSets':[2,3,4,5,7,8],
                       'worldBounds':[world.min(axis=0).tolist(),world.max(axis=0).tolist()],
                       'matrixWorld':np.asarray(evaluated.matrix_world).tolist()}


def _fit_root(old_root,triangles,skull,new_center):
    import numpy as np
    from mathutils import Vector
    old_center=np.asarray(OLD_CENTER);new_root=np.empty_like(old_root);normals=np.empty_like(old_root)
    for i,p in enumerate(old_root):
        direction=p-old_center;direction/=np.linalg.norm(direction)
        hit,normal,index,distance=skull.ray_cast(Vector(new_center),Vector(direction),.5)
        if hit is None:
            raise RuntimeError('New skull radial anchor missed at old root vertex '+str(i))
        n=np.asarray(normal,dtype=np.float64)
        if np.dot(n,direction)<0:n=-n
        normals[i]=n;new_root[i]=np.asarray(hit)+n*ROOT_GAP
    a,b,c=old_root[triangles[:,0]],old_root[triangles[:,1]],old_root[triangles[:,2]]
    old_cross=np.cross(b-a,c-a);old_out=np.einsum('ij,ij->i',old_cross,(a+b+c)/3.-old_center)
    aa,bb,cc=new_root[triangles[:,0]],new_root[triangles[:,1]],new_root[triangles[:,2]]
    new_cross=np.cross(bb-aa,cc-aa);new_out=np.einsum('ij,ij->i',new_cross,(aa+bb+cc)/3.-new_center)
    nondeg=np.linalg.norm(old_cross,axis=1)>1e-12
    inverted=nondeg&(old_out>1e-12)&(new_out< -1e-12)
    if np.any(inverted):
        raise RuntimeError('Radial skull fit inverted '+str(int(inverted.sum()))+' previously outward root triangles')
    displacement=np.linalg.norm(new_root-old_root,axis=1)
    if displacement.max()>.065:
        raise RuntimeError('Skull refit exceeds 65 mm; wrong source, pose, or target transform')
    return new_root,normals,{'vertices':len(old_root),'maximumDisplacementMeters':float(displacement.max()),
                             'displacementMeters':_summary(displacement),'newlyInvertedTriangles':int(inverted.sum()),
                             'rootGapMeters':ROOT_GAP,'rootVolumePolicy':'Actual skin + fixed 0.25 mm only; no volume inflation',
                             'newWorldBounds':[new_root.min(axis=0).tolist(),new_root.max(axis=0).tolist()]}


class _Correspondence:
    def __init__(self,old_root,new_root,triangles,new_normals):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        self.old=[Vector(p) for p in old_root];self.new=[Vector(p) for p in new_root]
        self.normals=[Vector(p) for p in new_normals]
        self.triangles=triangles;self.old_center=Vector(OLD_CENTER)
        self.tree=BVHTree.FromPolygons(self.old,[tuple(int(i) for i in f) for f in triangles],all_triangles=True)

    def map(self,point,radius=0.,clamp_near_root=False):
        from mathutils import Vector
        from mathutils.geometry import barycentric_transform
        p=Vector(point);hit,old_normal,index,distance=self.tree.find_nearest(p)
        if hit is None:raise RuntimeError('Old root correspondence query failed')
        if old_normal.dot(hit-self.old_center)<0:old_normal=-old_normal
        ids=self.triangles[index];a,b,c=(self.old[int(i)] for i in ids)
        na,nb,nc=(self.new[int(i)] for i in ids)
        qa,qb,qc=(self.normals[int(i)] for i in ids)
        new_hit=barycentric_transform(hit,a,b,c,na,nb,nc)
        new_normal=barycentric_transform(hit,a,b,c,qa,qb,qc).normalized()
        signed=(p-hit).dot(old_normal)
        normal_offset=signed
        if clamp_near_root and distance<.012:
            normal_offset=max(signed,radius+.00010)
        # Preserve the vector outside the old root boundary. Only its normal
        # component follows the new scalp frame; free tangent lengths survive.
        mapped=p+(new_hit-hit)-old_normal*signed+new_normal*normal_offset
        return mapped,new_normal,float(distance),float(signed)


def _transport_curves(name,arrays,prefix,correspondence,skin,eye_delta_z):
    import numpy as np
    from mathutils import Vector
    old=arrays[prefix+'_world'];starts=arrays[prefix+'_starts'];lengths=arrays[prefix+'_lengths']
    radii=arrays[prefix+'_radii'];output=np.empty_like(old);normal=np.empty_like(old)
    clearance_before=np.empty(len(old));collisions=0;largest_push=0.;first_bad=[]
    for i,p in enumerate(old):
        mapped,n,distance,signed=correspondence.map(p,float(radii[i]),name==SCALP)
        output[i]=mapped;normal[i]=n;clearance_before[i]=signed-radii[i]
        if i and i%250000==0:print('R11_GROOM_TRANSPORT '+name+' '+str(i)+'/'+str(len(old)),flush=True)
    pinned=np.zeros(len(old),dtype=bool)
    tip_adjustments=[]
    for k,(start,length) in enumerate(zip(starts,lengths)):
        start=int(start);length=int(length);sl=slice(start,start+length)
        u=np.linspace(0.,1.,length)
        if name in GATHERS:
            # Root correspondence agrees with the transported scalp endpoints;
            # a smooth fade retains the exact original tail/cache join.
            fade=1.-_smooth((u-.10)/.83)
            output[sl]=old[sl]+(output[sl]-old[sl])*fade[:,None]
            pinned[start+length-2:start+length]=True
        elif name==WISPS:
            fade=1.-_smooth((u-.12)/.66)
            output[sl]=old[sl]+(output[sl]-old[sl])*fade[:,None]
        elif name==FRINGE:
            # Keep this character's original uneven bang lengths relative to
            # the eyes, now using the adult eye line. This is hair styling, not
            # transfer of the rejected old facial geometry or large eye size.
            desired_tip_z=float(old[start+length-1,2]+eye_delta_z)
            dz=desired_tip_z-float(output[start+length-1,2])
            output[sl,2]+=dz*_smooth((u-.28)/.72)
            tip_adjustments.append(dz)
        # Real volume belongs to the free shaft, with zero lift at both ends.
        # Amplitude stays millimetric and varies in coherent groups of fibres.
        if name==SCALP:
            bundle=k//120
            amplitude=.00075+.00110*(.5+.5*math.sin(bundle*1.61803398875))
            envelope=np.sin(np.pi*u)**1.55
            envelope*=_smooth(u/.10)*(1.-_smooth((u-.87)/.13))
            output[sl]+=normal[sl]*(amplitude*envelope[:,None])
        elif name==FRINGE:
            bundle=k//120
            amplitude=.00045+.00065*(.5+.5*math.sin(bundle*2.39996323))
            envelope=np.sin(np.pi*u)**1.8
            output[sl]+=normal[sl]*(amplitude*envelope[:,None])
    output[pinned]=old[pinned]
    # The actual complete head supplies non-penetration constraints, including
    # ears. No head vertex is displaced to satisfy the groom.
    for i,p in enumerate(output):
        if pinned[i]:continue
        hit,n,index,distance=skin.find_nearest(Vector(p))
        if hit is None:raise RuntimeError('Actual new head clearance query failed')
        signed=(Vector(p)-hit).dot(n)-float(radii[i])
        if signed<.00008 and distance<.035:
            movement=.00012-signed
            output[i]=p+np.asarray(n)*movement
            collisions+=1;largest_push=max(largest_push,float(movement))
            if movement>.018 and len(first_bad)<8:first_bad.append({'point':i,'pushMeters':float(movement)})
        if i and i%250000==0:print('R11_GROOM_CLEARANCE '+name+' '+str(i)+'/'+str(len(output)),flush=True)
    if first_bad:
        raise RuntimeError('Refit requires excessive local skin correction: '+json.dumps(first_bad))
    output[pinned]=old[pinned]
    old_lengths=[];new_lengths=[];max_extra_step=0.;max_new_step=0.
    for start,length in zip(starts,lengths):
        sl=slice(int(start),int(start+length))
        a=np.linalg.norm(np.diff(old[sl],axis=0),axis=1);b=np.linalg.norm(np.diff(output[sl],axis=0),axis=1)
        old_lengths.append(a.sum());new_lengths.append(b.sum())
        max_extra_step=max(max_extra_step,float((b-a).max()));max_new_step=max(max_new_step,float(b.max()))
    ratios=np.asarray(new_lengths)/np.asarray(old_lengths)
    if not np.isfinite(output).all() or max_extra_step>.007 or ratios.min()<.35 or ratios.max()>1.85:
        raise RuntimeError('Refit lost strand continuity: '+json.dumps({'object':name,'ratio':_summary(ratios),'extraStep':max_extra_step,'maxStep':max_new_step}))
    return output,pinned,{'object':name,'curves':len(starts),'points':len(old),
                          'maximumDisplacementMeters':float(np.linalg.norm(output-old,axis=1).max()),
                          'strandLengthRatio':_summary(ratios),'maximumSegmentMeters':max_new_step,
                          'maximumExtraSegmentMeters':max_extra_step,'skinConstraintCorrections':collisions,
                          'largestSkinCorrectionMeters':largest_push,'gatherPinnedPointCount':int(pinned.sum()),
                          'gatherPinsExact':bool(np.array_equal(output[pinned],old[pinned])),
                          'forelockTipDeltaFromTransportMeters':_summary(tip_adjustments) if tip_adjustments else None,
                          'newWorldBounds':[output.min(axis=0).tolist(),output.max(axis=0).tolist()]}


def fit_adult_groom(head_object,female_to_target_matrix,*,expected_source_relative,diagnostics_only=False):
    import bpy
    import numpy as np
    P=project_path()
    source=(P/expected_source_relative).resolve()
    source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Exact explicit independent candidate in background Object Mode required')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.data.pose_position!='REST':
        raise RuntimeError('Caller must enter REST for this source-space groom fit and restore its previous pose afterward')
    if isinstance(head_object,str):head_object=bpy.data.objects.get(head_object)
    if head_object is None or head_object.type!='MESH' or head_object.name=='R01 Head':
        raise RuntimeError('Actual new adult head required')
    if any(bpy.data.objects[n].get(MARKER) for n in NAMES):
        raise RuntimeError('Preserve existing fitted groom candidate before any repeat')
    T=np.asarray(female_to_target_matrix,dtype=np.float64)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-8):
        raise ValueError('Explicit finite female body-space to target matrix required')
    linear=T[:3,:3];scale=float(np.linalg.norm(linear[:,0]))
    if not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8) or np.linalg.det(linear)<=0:
        raise ValueError('Uniform positive target scale required')
    obs=json.loads((P/'work/reconstruction-r02/crown-observation-r10-v2.json').read_text(encoding='utf8'))
    arrays_path=P/obs['arrays']['path']
    if _sha(arrays_path)!=SOURCE_ARRAY_SHA or obs['arrays']['sha256']!=SOURCE_ARRAY_SHA:
        raise RuntimeError('Actual source groom observation changed')
    with np.load(arrays_path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    bpy.context.view_layer.update()
    objects={n:bpy.data.objects[n] for n in NAMES};original={};before_contract={}
    for name,obj in objects.items():
        pre=obs['objects'][name]['prefix'];positions=_read_positions(obj)
        if not np.array_equal(positions,arrays[pre+'_local']):
            raise RuntimeError('Groom data no longer matches observed 1c source: '+name)
        current_world=_world(positions,obj.matrix_world)
        error=float(np.max(np.abs(current_world-arrays[pre+'_world'])))
        if error>2e-6:raise RuntimeError('REST groom transform no longer matches source: '+name+' '+str(error))
        if obj.library or obj.data.library or obj.data.users!=1:
            raise RuntimeError('Local single-user source groom data required: '+name)
        if any(m.show_viewport or m.show_render for m in obj.modifiers):
            raise RuntimeError('Existing groom modifier needs explicit evaluation strategy: '+name)
        if getattr(obj.data,'shape_keys',None):raise RuntimeError('Do not overwrite keyed groom: '+name)
        original[name]=obj.data
        before_contract[name]={'matrix':np.asarray(obj.matrix_world).tolist(),'parent':obj.parent.name if obj.parent else None,
                               'parentType':obj.parent_type,'parentBone':obj.parent_bone,
                               'sourceDataName':obj.data.name,'sourcePositionSha256':hashlib.sha256(positions.tobytes()).hexdigest()}
    tail_before={n:hashlib.sha256(_read_positions(bpy.data.objects[n]).tobytes()).hexdigest() for n in TAILS}
    source_sha=_sha(source)
    skull,skin,head_report=_head_surfaces(head_object,bpy.context.evaluated_depsgraph_get())
    new_center=linear@np.asarray(NEW_CENTER_DONOR)+T[:3,3]
    root,new_normals,root_report=_fit_root(arrays['root_world'],arrays['root_triangles'],skull,new_center)
    correspondence=_Correspondence(arrays['root_world'],root,arrays['root_triangles'],new_normals)
    # The same adult-eye source used by eye_adult_fit; no old globe size is used.
    female=json.loads((P/'work/reconstruction-r02/female-basemesh-actual-r11.json').read_text(encoding='utf8'))
    female_rows={o['name']:o for o in female['objects']}
    donor_eye_z=float(female_rows['GEO-body_female_realistic.eye.L']['world_bounds'][0][2]+female_rows['GEO-body_female_realistic.eye.L']['world_bounds'][1][2])*.5
    new_eye_z=float((linear@np.array([0.,0.,donor_eye_z])+T[:3,3])[2])
    old_eye_z=float(arrays['eye_centers_world'][:,2].mean());eye_delta_z=new_eye_z-old_eye_z
    prepared={SHELL:root};pinned={};reports=[]
    for name in NAMES[1:]:
        pre=obs['objects'][name]['prefix']
        w,pin,report=_transport_curves(name,arrays,pre,correspondence,skin,eye_delta_z)
        prepared[name]=w;pinned[name]=pin;reports.append(report)
    report={'source':expected_source_relative,'sourceSha256':source_sha,'head':head_report,
            'sourceGroomArraySha256':SOURCE_ARRAY_SHA,'femaleToTargetMatrix':T.tolist(),
            'oldRadialCenter':list(OLD_CENTER),'newRadialCenter':new_center.tolist(),
            'root':root_report,'curves':reports,'forelockEyeLineShiftMeters':eye_delta_z,
            'tailDataSha256':tail_before,'tailsEdited':False,'tailCachesEdited':False,
            'headNeckRigEyesBrowsEdited':False,'geometryWrites':not diagnostics_only,'savedBlend':False,'appearanceAccepted':False,
            'sourceContracts':before_contract,
            'limits':['This is a static groom refit candidate, not yet visual approval or a new cloth/hair simulation cache.',
                      'The donor Hair Styles guides are retained for a later deliberate style rebuild; this module does not claim to have adapted them.',
                      'Root topology and alpha feather remain unchanged; actual front/quarter/back review must confirm natural boundaries.',
                      'Hair shaft clearance is solved against this exact head; new combined dynamic verification remains necessary.']}
    if diagnostics_only:return report
    copies={};old_fake={name:data.use_fake_user for name,data in original.items()}
    try:
        for name,obj in objects.items():
            copied=original[name].copy();copied.name='R11 adult fitted / '+name;copies[name]=copied
            inverse=np.asarray(obj.matrix_world.inverted(),dtype=np.float64)
            local=(prepared[name]@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32)
            if name in GATHERS:
                pre=obs['objects'][name]['prefix'];local[pinned[name]]=arrays[pre+'_local'][pinned[name]]
            items,field=(copied.vertices,'co') if obj.type=='MESH' else (copied.position_data,'vector')
            items.foreach_set(field,local.ravel())
            if obj.type=='MESH':copied.update()
            else:copied.update_tag()
            actual=_world(local,obj.matrix_world)
            error=float(np.max(np.abs(actual-prepared[name])))
            if error>2e-6:raise RuntimeError('New fitted data lost requested REST coordinates: '+name)
            original[name].use_fake_user=True
            obj.data=copied
            obj[MARKER]=json.dumps({'previousData':original[name].name,'sourceSha256':source_sha,
                                    'sourcePositionSha256':before_contract[name]['sourcePositionSha256'],
                                    'localFloatMaximumErrorMeters':error,'staticCandidate':True})
        bpy.context.view_layer.update()
        for name in TAILS:
            if hashlib.sha256(_read_positions(bpy.data.objects[name]).tobytes()).hexdigest()!=tail_before[name]:
                raise RuntimeError('Protected native tail changed unexpectedly')
        if _sha(source)!=source_sha:raise RuntimeError('Saved independent input changed')
        report['newDataNames']={n:d.name for n,d in copies.items()}
        report['retainedOriginalDataNames']={n:d.name for n,d in original.items()}
        return report
    except Exception:
        for name,obj in objects.items():
            obj.data=original[name];original[name].use_fake_user=old_fake[name]
            if MARKER in obj:del obj[MARKER]
        for data in copies.values():
            if data.users==0:
                if isinstance(data,bpy.types.Mesh):bpy.data.meshes.remove(data)
                else:bpy.data.hair_curves.remove(data)
        bpy.context.view_layer.update()
        raise


if __name__=='__main__':
    print('Loaded only. Explicit fit_adult_groom(new_head, donor_to_target, expected_source_relative=..., diagnostics_only=False); caller owns REST state, save and renders.')
