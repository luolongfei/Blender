"""Remove the measured upward takeoff of two left low-side gather guides."""
import numpy as np
GUIDES={88:8,94:5}

def flatten_takeoff(path,index):
    """Keep root and the complete existing lower gather/tie approach exact."""
    if index not in GUIDES:raise RuntimeError('Not a measured left-side upward guide')
    join=GUIDES[index];old=np.asarray(path);root=old[0];end=old[join]
    if root[0]>=-.065 or not 1.45<root[2]<1.47 or not -.01<root[1]<.015:raise RuntimeError('Actual low-side root left reviewed envelope')
    if old[:join+1,2].max()-root[2]<.007:raise RuntimeError('Measured upward takeoff is no longer present')
    if end[2]>=root[2] or end[1]<=root[1]:raise RuntimeError('Gather join is not behind/below root')
    # A short rearward takeoff which bends down into the already authored
    # lower path; avoid the old 10-27 mm upward loop and preserve its endpoint.
    c1=root+np.array([-.0015,(end[1]-root[1])*.35,-.001])
    tangent=old[join+1]-end
    tangent=tangent/np.linalg.norm(tangent)
    handle=min(.007,(root[2]-.0015-end[2])/max(-tangent[2],1e-9))
    if handle<=0.:raise RuntimeError('Insufficient vertical room for smooth gather join')
    c2=end-tangent*handle
    t=np.linspace(0.,1.,join+1)[:,None]
    curve=(1.-t)**3*root+3.*(1.-t)**2*t*c1+3.*(1.-t)*t*t*c2+t**3*end
    new=old.copy();new[1:join]=curve[1:join]
    if not np.array_equal(new[0],old[0]) or not np.array_equal(new[join:],old[join:]):raise RuntimeError('Protected gather root/lower approach changed')
    return new,{'guide':int(index),'joinPoint':join,'changedPoints':list(range(1,join)),
        'oldUpwardExcursionMeters':float(old[:join+1,2].max()-root[2]),
        'newUpwardExcursionMeters':float(new[:join+1,2].max()-root[2]),
        'rootAndCompleteLowerGatherExact':True}
