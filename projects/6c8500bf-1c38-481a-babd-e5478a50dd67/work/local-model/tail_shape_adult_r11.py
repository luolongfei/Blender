"""Independent static C-shaped low-tail study based on the actual native groom.

Preserves every source strand and its first three pinned points. The original
native cache is retained as history and must be rebuilt for the new rest shape.
No import-time Blender work, saving, rendering, or source-master modification.
"""
import hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
RAW_SHA=('db35bd519eee1906cfc4213c4b4ef4513cc5f4c0a2d359cdcf7b6ada6c1403de',
         '02e0206c16e515ad9537b418e6cd69cb38593a6f509c3efe449a4439b0e07161')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _smooth(t):
    q=np.clip(t,0.,1.);return q*q*(3.-2.*q)
def _unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def _world(a,m):
    m=np.asarray(m,dtype=float);return np.asarray(a)@m[:3,:3].T+m[:3,3]
def _data_coords(data):
    a=np.empty(len(data.position_data)*3,dtype=np.float32)
    data.position_data.foreach_get('vector',a);return a.reshape(-1,3)
def _coords(obj):return _data_coords(obj.data)


def _frames(center):
    """Rotation-minimizing local cross-sections, never fixed world XY scales."""
    tangent=_unit(np.gradient(center,axis=0));normals=[]
    n=np.array([1.,0.,0.]);n=_unit(n-tangent[0]*np.dot(n,tangent[0]))
    normals.append(n)
    for a,b in zip(tangent[:-1],tangent[1:]):
        cross=np.cross(a,b);s=float(np.linalg.norm(cross));c=float(np.clip(np.dot(a,b),-1.,1.))
        if s>1e-9:
            axis=cross/s;n=n*c+np.cross(axis,n)*s+axis*np.dot(axis,n)*(1.-c)
        elif c<0:raise RuntimeError('Tail centreline reversed direction')
        n=_unit(n-b*np.dot(n,b));normals.append(n)
    normal=np.asarray(normals);binormal=_unit(np.cross(tangent,normal))
    return np.stack([normal,binormal,tangent],axis=1)


def _hermite(t,k,v,start_slope):
    slopes=np.empty_like(v);slopes[0]=start_slope;slopes[-1]=(v[-1]-v[-2])/(k[-1]-k[-2])
    slopes[1:-1]=(v[2:]-v[:-2])/(k[2:,None]-k[:-2,None])
    j=np.clip(np.searchsorted(k,t,side='right')-1,0,len(k)-2)
    h=k[j+1]-k[j];u=(t-k[j])/h
    return ((2*u**3-3*u**2+1)[:,None]*v[j]+(u**3-2*u**2+u)[:,None]*h[:,None]*slopes[j]
        +(-2*u**3+3*u**2)[:,None]*v[j+1]+(u**3-u**2)[:,None]*h[:,None]*slopes[j+1])


def design(world,side_index):
    """Pure NumPy geometry; input is the actual current REST world positions."""
    old=np.asarray(world,dtype=float).reshape(8,310,35,3)
    center=old.mean((0,1));root=center[0];side=float(np.sign(root[0]))
    if abs(root[0])<.06 or not 1.32<root[2]<1.39:
        raise RuntimeError('Unexpected actual adult low-tail pin location')
    t=np.arange(35)/34.;knots=np.array([2/34,.24,.46,.68,.86,1.])
    target=np.array([center[2],root+[side*.041,-.058,-.051],root+[side*.049,-.096,-.092],
        root+[side*.043,-.119,-.128],root+[side*.027,-.128,-.157],root+[side*.007,-.122,-.170]])
    new_center=center.copy()
    new_center[2:]=_hermite(t[2:],knots,target,(center[2]-center[1])*34.)
    before_frame=_frames(center);after_frame=_frames(new_center)
    offsets=old-center[None,None,:,:]
    components=np.einsum('gcpi,pji->gcpj',offsets,before_frame)
    blend=_smooth((t-2/34)/.26);lower=_smooth((t-.52)/.35)
    # A compact upper belly with fuller lower layers. Existing strands and
    # clumps supply volume; no opaque volume shell or added root expansion.
    transverse=(1.-.20*blend)*(1.+.37*lower)
    depth=(1.-.28*blend)*(1.+.18*lower)
    longitudinal=1.-.28*blend
    components[...,0]*=transverse;components[...,1]*=depth;components[...,2]*=longitudinal
    rng=np.random.default_rng(1107+int(side_index)*29)
    # Original 310 strands within each clump share almost the same final point.
    # Small independent cut lengths and lateral scatter remove eight hard pins.
    tip=np.empty((8,310,3));tip[...,0]=rng.uniform(-.0018,.0018,(8,310))
    tip[...,1]=rng.uniform(-.0012,.0012,(8,310));tip[...,2]=rng.uniform(-.006,.006,(8,310))
    components+=tip[:,:,None,:]*_smooth((t-.62)/.38)[None,None,:,None]
    new=new_center[None,None,:,:]+np.einsum('gcpj,pji->gcpi',components,after_frame)
    new[:,:,:3]=old[:,:,:3]
    steps=np.linalg.norm(np.diff(new,axis=2),axis=3)
    if not np.isfinite(new).all() or steps.min()<1e-6 or steps.max()>.025:
        raise RuntimeError('Invalid independent tail shape continuity '+json.dumps({'side':int(side_index),
            'minimumSegmentMeters':float(steps.min()),'maximumSegmentMeters':float(steps.max())}))
    old_arc=np.linalg.norm(np.diff(old,axis=2),axis=3).sum(2)
    new_arc=steps.sum(2)
    before_exit=_unit(new[:,:,2]-new[:,:,1]);after_exit=_unit(new[:,:,3]-new[:,:,2])
    exit_angle=np.degrees(np.arccos(np.clip(np.sum(before_exit*after_exit,axis=-1),-1.,1.)))
    if exit_angle.max()>15.:raise RuntimeError('Tail shape creates an abrupt bend after the protected pin segment')
    row={'side':int(side_index),'strands':2480,'pointsPerStrand':35,'existingClumps':8,
         'rootThreePointsExactlyRetained':bool(np.array_equal(new[:,:,:3],old[:,:,:3])),
         'oldMeanTipWorld':center[-1].tolist(),'newMeanTipWorld':new.mean((0,1))[-1].tolist(),
         'oldArcLengthPercentilesMeters':np.percentile(old_arc,[5,50,95]).tolist(),
         'newArcLengthPercentilesMeters':np.percentile(new_arc,[5,50,95]).tolist(),
         'newMinimumSegmentMeters':float(steps.min()),'newMaximumSegmentMeters':float(steps.max()),
         'pinExitAngleDegreesP50P95Max':np.percentile(exit_angle,[50,95,100]).tolist(),
         'oldWorldBounds':[old.reshape(-1,3).min(0).tolist(),old.reshape(-1,3).max(0).tolist()],
         'newWorldBounds':[new.reshape(-1,3).min(0).tolist(),new.reshape(-1,3).max(0).tolist()],
         'newCenterline':new_center.tolist(),'tipScatterMeters':{'transverse':.0018,'depth':.0012,'length':.006}}
    return new.reshape(-1,3),row


def apply_adult_tail_shape(*,expected_source_relative):
    """Static independent source only. Caller owns save and visual evaluation."""
    import bpy
    P=project_path();source=(P/expected_source_relative).resolve()
    source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Explicit independent saved reconstruction source required')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.data.pose_position!='REST':raise RuntimeError('Caller must set actual rig REST')
    source_hash=_sha(source);bpy.context.view_layer.update()
    objects=[bpy.data.objects[n] for n in TAILS];prepared={};records={};old_data={};radii_before={}
    for i,obj in enumerate(objects):
        raw=_coords(obj)
        if hashlib.sha256(raw.tobytes()).hexdigest()!=RAW_SHA[i]:raise RuntimeError('Original native tail groom changed: '+obj.name)
        if len(obj.data.curves)!=2480 or any(c.points_length!=35 for c in obj.data.curves):
            raise RuntimeError('Unexpected original tail strand topology')
        matrix=np.asarray(obj.matrix_world,dtype=float)
        new,row=design(_world(raw,matrix),i)
        local=_world(new,np.linalg.inv(matrix)).astype(np.float32)
        local.reshape(2480,35,3)[:,:3]=raw.reshape(2480,35,3)[:,:3]
        if not np.array_equal(local.reshape(2480,35,3)[:,:3],raw.reshape(2480,35,3)[:,:3]):
            raise RuntimeError('Exact original native pins changed')
        prepared[obj]=local;records[obj.name]=row;old_data[obj]=obj.data
        radii=obj.data.attributes.get('radius')
        if radii:
            values=np.empty(len(radii.data),np.float32);radii.data.foreach_get('value',values)
            radii_before[obj.name]=hashlib.sha256(values.tobytes()).hexdigest()
    present=[obj.modifiers.get('R04 Native hair playback') is not None for obj in objects]
    if any(present) and not all(present):raise RuntimeError('Inconsistent existing native playback; inspect before rest-shape edit')
    for obj in objects:
        old_data[obj].use_fake_user=True;obj.data=old_data[obj].copy();obj.data.name='R11 shorter C tail / '+old_data[obj].name
    if all(present):
        native=runpy.run_path(str(Path(__file__).with_name('native_hair_playback_r04.py')))
        retired=native['restore']()
    else:
        retired={'alreadyAbsent':True,'reason':'Neither native-tail playback modifier is present; no cache or other rig data removed.'}
    for obj in objects:
        obj.data.position_data.foreach_set('vector',prepared[obj].ravel());obj.data.update_tag();obj.update_tag()
        obj['R11_tail_rest_shape']='Shorter layered C curve; exact original three-point pins; static candidate'
        obj['R11_tail_dynamics_valid']=False;obj['R11_tail_original_data']=old_data[obj].name
    bpy.context.view_layer.update()
    for obj in objects:
        if not np.array_equal(_coords(obj),prepared[obj]):raise RuntimeError('Stored tail coordinate mismatch')
        if hashlib.sha256(_data_coords(old_data[obj]).tobytes()).hexdigest()!=RAW_SHA[objects.index(obj)]:
            raise RuntimeError('Original retained tail data changed')
        radii=obj.data.attributes.get('radius')
        if radii and obj.name in radii_before:
            values=np.empty(len(radii.data),np.float32);radii.data.foreach_get('value',values)
            if hashlib.sha256(values.tobytes()).hexdigest()!=radii_before[obj.name]:raise RuntimeError('Tail radii changed')
    if _sha(source)!=source_hash:raise RuntimeError('Saved input source was changed')
    return {'schemaVersion':1,'source':expected_source_relative,'sourceSha256':source_hash,'helperSha256':_sha(__file__),
        'stage':'Static shorter low twin-tail candidate; not a physics cache or final hairstyle','changes':records,
        'retainedOriginalData':{obj.name:data.name for obj,data in old_data.items()},'retiredNativePlayback':retired,
        'sourceGuidesAndCacheFilesEdited':False,'tailTopologyAndRadiiPreserved':True,'nativeDynamicsReady':False,
        'headDonorGroomAndTieGeometryEdited':False,'savedBlend':False,'appearanceAccepted':False,
        'limitations':['All old tail static-rest hashes, representative guide selection/maps, per-action simulations and native playback parity are invalid for the new rest shape.',
            'Rebake body, walk and performance tail motion against the adopted new head/neck and actual garment collision envelope.',
            'Cloth caches and body animation are not edited; combined hair/cloth collision and appearance still require new review.',
            'Existing realtime tail cards, skinning/dynamic bones, bounds and model exports require adaptation and revalidation.',
            'Actual front/quarter/profile, tie continuity and dense-strand contact have not been checked by this static build.']}


if __name__=='__main__':print('Loaded only. Use design() offline or explicit apply_adult_tail_shape() on an independent Blender source.')
