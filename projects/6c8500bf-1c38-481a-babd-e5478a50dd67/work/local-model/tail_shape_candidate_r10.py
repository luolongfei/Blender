"""Bounded tail rest-shape study. Old native dynamics are retained as history.

Only independent work copies are eligible. This is explicitly a STATIC study;
new rest geometry must receive new physics caches before production adoption.
"""
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path()
SOURCE_SHA='1c0b0210114170f0ce6e09b266b4a9ded58f29bf1df45a6e2be13e9b5c77aa72'
OBS_SHA='9ddb5f6ffe1fcc640256782451fcd54b327785c62270cd9ea749d33f5ad2c879'
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
def api(n):return runpy.run_path(str(Path(__file__).with_name(n+'.py')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def smooth(x):
    t=np.clip(x,0,1);return t*t*(3-2*t)
def interpolate(t,k,v,start_slope):
    slopes=np.empty_like(v);slopes[0]=start_slope;slopes[-1]=(v[-1]-v[-2])/(k[-1]-k[-2])
    slopes[1:-1]=(v[2:]-v[:-2])/(k[2:,None]-k[:-2,None])
    j=np.minimum(np.searchsorted(k,t,side='right')-1,len(k)-2);j=np.maximum(j,0)
    h=k[j+1]-k[j];u=(t-k[j])/h
    return ((2*u**3-3*u**2+1)[:,None]*v[j]+(u**3-2*u**2+u)[:,None]*h[:,None]*slopes[j]
        +(-2*u**3+3*u**2)[:,None]*v[j+1]+(u**3-u**2)[:,None]*h[:,None]*slopes[j+1])
def design(world):
    old=world.reshape(8,310,35,3);mean=old.mean((0,1));side=float(np.sign(mean[0,0]));z=mean[0,2]
    t=np.arange(35)/34;knots=np.array([2/34,.22,.43,.67,.87,1.])
    target=np.array([mean[2],[side*.139,-.073,z-.060],[side*.150,-.115,z-.105],
        [side*.149,-.136,z-.140],[side*.142,-.143,z-.160],[side*.137,-.140,z-.171]])
    core=interpolate(t[2:],knots,target,(mean[2]-mean[1])*34)
    center=mean.copy();center[2:]=core
    group=old.mean(1);offset=group-mean[None,:,:];within=old-group[:,None,:,:]
    blend=smooth((t-2/34)/.24)
    # Bring the tail forward over the shoulder, reduce excessive broad upper
    # sweep and retain fuller lower clumps instead of long single-point spikes.
    transverse=(1-.28*blend)*(1+.24*smooth((t-.60)/.29))
    offset[:,:,:2]*=transverse[None,:,None]
    within[:,:,:,:2]*=transverse[None,None,:,None]
    offset[:,:,2]*=(1-.35*blend)[None,:]
    within[:,:,:,2]*=(1-.20*blend)[None,None,:]
    # Small distinct end gestures remain tied to the eight existing clumps.
    end=smooth((t-.70)/.30)
    curls=np.array([-.005,.004,-.002,.007,-.004,.005,-.006,.003])*side
    offset[:,:,0]+=curls[:,None]*end[None,:]
    new=center[None,None,:,:]+offset[:,None,:,:]+within
    new[:,:,:3]=old[:,:,:3]
    step=np.linalg.norm(np.diff(new,axis=2),axis=3)
    if not np.isfinite(new).all() or step.max()>.025 or step.min()<1e-6:
        raise RuntimeError('Invalid tail strand continuity')
    return new.reshape(-1,3),{'maximumSegmentMm':float(step.max()*1000),
        'rootThreePointsExactlyRetained':bool(np.array_equal(new[:,:,:3],old[:,:,:3])),
        'oldMeanEndWorld':mean[-1].tolist(),'newMeanEndWorld':new.mean((0,1))[-1].tolist(),
        'newWorldBounds':[new.reshape(-1,3).min(0).tolist(),new.reshape(-1,3).max(0).tolist()]}
def prepare(source_relative,output_relative):
    import bpy
    source=(P/source_relative).resolve();output=(P/output_relative).resolve()
    for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or sha(source)!=SOURCE_SHA:
        raise RuntimeError('Fresh exact adopted independent R09 source required')
    if output.exists() or output.with_suffix('.tail.json').exists():raise RuntimeError('Preserve prior candidates')
    obs=P/'work/reconstruction-r02/tail-observation-r10.npz'
    if sha(obs)!=OBS_SHA:raise RuntimeError('Observed geometry changed')
    data=np.load(obs,allow_pickle=False);review=api('final_detail_review_r09')
    rig,matching=review['_matched_standing'](bpy);groom=api('posterior_groom_r09')
    outputs={};records={}
    for i,name in enumerate(TAILS):
        obj=bpy.data.objects[name];raw=groom['_values'](obj.data.position_data,'vector')
        matrix=np.asarray(obj.matrix_world,dtype=float)
        if not np.array_equal(raw.astype(np.float32),data['tail_'+str(i)+'_local']):raise RuntimeError('Tail source differs from actual observation')
        if not np.allclose(matrix,data['tail_'+str(i)+'_matrix'],rtol=0,atol=1e-8):raise RuntimeError('Attachment changed')
        new,record=design(data['tail_'+str(i)+'_world'])
        inv=np.linalg.inv(matrix);local=(new-matrix[:3,3])@inv[:3,:3].T
        local[:]=local.astype(np.float32)
        local.reshape(2480,35,3)[:,:3]=raw.reshape(2480,35,3)[:,:3]
        outputs[name]=local;records[name]=record
    for name in TAILS:
        obj=bpy.data.objects[name];old=obj.data;old.use_fake_user=True
        obj['R10_tail_original_data']=old.name;obj.data=old.copy();obj.data.name='R10 softer tail / '+old.name
    retired=api('native_hair_playback_r04')['restore']()
    for name in TAILS:
        obj=bpy.data.objects[name];obj.data.position_data.foreach_set('vector',outputs[name].ravel())
        obj.data.update_tag();obj.update_tag();obj['R10_tail_static_candidate']=True
    bpy.context.view_layer.update()
    result={'source':source_relative,'sourceSha256':SOURCE_SHA,'observedArraysSha256':OBS_SHA,
        'candidate':output_relative,'changes':records,'retiredNativePlayback':retired,
        'baselineStandingEvidence':matching,'newDynamicsReady':False,'appearanceAccepted':False,
        'scope':'Static rest-shape proposal only. Existing native R09 cloth remains; old hair cache disabled and retained. New body, walk, performance hair caches and runtime tail adaptation are mandatory if adopted.'}
    bpy.context.scene['R10_tail_static_candidate']=json.dumps(result,ensure_ascii=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(output),check_existing=False,relative_remap=True)
    result['candidateSha256']=sha(output)
    output.with_suffix('.tail.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('R10_STATIC_TAIL='+json.dumps({'candidate':output_relative,'sha256':result['candidateSha256']}))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(args.source,args.output)
if __name__=='__main__':main()
