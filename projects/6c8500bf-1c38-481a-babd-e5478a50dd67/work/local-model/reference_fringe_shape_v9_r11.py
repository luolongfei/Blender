"""Small original-reference shape corrections after the measured v8 groom."""
import hashlib,math,runpy
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'reference_fringe_layout_v6_r11.py'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!='e6b1879cc9773e73f277cdc30ca374303f55899cf7377226b22f03ff8437a8b1':raise RuntimeError('Reviewed v6 layout changed')
BASE=runpy.run_path(str(SOURCE));CENTER=BASE['CENTER'];normalize=BASE['normalize'];smooth=BASE['smooth'];resample=BASE['resample']
TARGET_ROLES=(10,13,14,15)

def design(root,index,count,old_role,project_ray,project_front):
    if old_role not in TARGET_ROLES:raise RuntimeError('Not an authorized local v9 lock')
    k=int(old_role)-10;low=float(BASE['EDGES'][k]);high=float(BASE['EDGES'][k+1]);tip_x=float(BASE['TIP_X'][k]);tip_z=float(BASE['TIP_Z'][k]);width=.0021;feather=.010;new_role=int(old_role)
    if old_role==13:
        if root[0]<-.0015:
            low=-.022;high=-.0015;new_role=18;tip_x=-.011;tip_z=1.442;width=.0027;feather=.008
        else:
            low=-.0015;high=.008;new_role=19;tip_x=.004;tip_z=1.438;width=.0008;feather=.006
    elif old_role==14:tip_x=.008;tip_z=1.448;width=.0016;feather=.008
    elif old_role==15:tip_x=.053;tip_z=1.444;width=.0019
    lateral=float(np.clip((root[0]-(low+high)*.5)/((high-low)*.5),-1.,1.))
    end_x=tip_x+lateral*width;end_z=tip_z+feather*abs(lateral)**1.35+.0015*math.sin(index*2.39996)
    if root[2]<1.478:end_z=min(end_z,root[2]-.026)
    target,_=project_front(float(end_x),float(end_z))
    if target is None:raise RuntimeError('Local reference tip misses actual skin '+str(index))
    a=normalize(root-CENTER);b=normalize(np.asarray(target)-CENTER);angle=math.acos(float(np.clip(np.dot(a,b),-.999999,1.)))
    gap_scale=.55 if old_role==10 else 1.
    root_hit,root_normal=project_ray(a)
    if root_hit is None:raise RuntimeError('Local reference root ray missed actual skin')
    root_offset=root-(np.asarray(root_hit)+np.asarray(root_normal)*.00030*gap_scale)
    path=[]
    for u in np.linspace(0.,1.,max(96,count*5)):
        direction=normalize((math.sin((1-u)*angle)*a+math.sin(u*angle)*b)/max(math.sin(angle),1e-10));hit,normal=project_ray(direction)
        if hit is None:raise RuntimeError('Local reference surface ray missed '+str(index))
        gap=(.00030+.0062*math.sin(math.pi*u)**1.15+.0060*float(smooth((u-.58)/.42)))*gap_scale
        point=np.asarray(hit)+np.asarray(normal)*gap
        if old_role==10:point+=root_offset*(1.-float(smooth(u/.24)))
        if u==0.:point=root.copy()
        path.append(point)
    result=resample(np.asarray(path),count);result[0]=root
    return result,{'guide':int(index),'oldRole':int(old_role),'newRole':new_role,'tip':result[-1].tolist(),
        'referenceChange':('left temple lower takeoff and half lift' if old_role==10 else 'central wide-left / fine-right split' if old_role==13 else 'move right opening boundary'),
        'gapScale':gap_scale,'rootOffsetBlendedMeters':float(np.linalg.norm(root_offset)) if old_role==10 else 0.}
