"""Original-reference central fringe fan, in world metres; pure NumPy.

Only dense interpolation roles 18/19 change. Crown, roots and all other roles
remain exact. The current-source head-contact stage follows this field.
"""
import numpy as np

START_FACTOR = .60
RULES = {
    18: {'xFocus': -.016, 'xOldCenter': -.01266, 'xScale': .26,
         'zFocus': 1.439, 'zOldCenter': 1.44493, 'zScale': .65},
    19: {'xFocus': .0015, 'xOldCenter': .00579, 'xScale': .16,
         'zFocus': 1.440, 'zOldCenter': 1.44004, 'zScale': .50},
}

def apply(points, starts, lengths, groups):
    """Return a same-dtype copy; groups are native island * 256 + role IDs.

    The tip's X/Z coordinates receive a role-local affine contraction, retaining
    unequal lengths and finite endpoint spread. Y is unchanged. Its displacement
    is blended over the final 40% of each original curve's cumulative arc length.
    No resampling, cross-curve attraction, radius change, or point removal occurs.
    """
    points=np.asarray(points);starts=np.asarray(starts);lengths=np.asarray(lengths);groups=np.asarray(groups)
    if points.ndim!=2 or points.shape[1]!=3 or points.dtype.kind!='f':raise ValueError('Floating world XYZ points required')
    if starts.ndim!=1 or starts.shape!=lengths.shape or starts.shape!=groups.shape:raise ValueError('One start, length and group per curve required')
    if not np.isfinite(points).all():raise ValueError('Finite world points required')
    if np.any(lengths<2) or np.any(starts<0) or np.any(starts+lengths>len(points)):raise ValueError('Invalid curve slices')
    out=points.copy();roles=groups.astype(np.int64)%256
    for index in np.flatnonzero(np.isin(roles,list(RULES))):
        start=int(starts[index]);count=int(lengths[index]);p=points[start:start+count].astype(np.float64);tip=p[-1];rule=RULES[int(roles[index])]
        desired=tip.copy()
        desired[0]=rule['xFocus']+rule['xScale']*(tip[0]-rule['xOldCenter'])
        desired[2]=rule['zFocus']+rule['zScale']*(tip[2]-rule['zOldCenter'])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if arc[-1]<=1e-12:raise ValueError('Zero-length selected curve')
        t=np.clip((arc/arc[-1]-START_FACTOR)/(1.-START_FACTOR),0.,1.)
        weight=t*t*(3.-2.*t)
        out[start:start+count]=(p+weight[:,None]*(desired-tip)).astype(points.dtype)
        out[start]=points[start]
    return out
