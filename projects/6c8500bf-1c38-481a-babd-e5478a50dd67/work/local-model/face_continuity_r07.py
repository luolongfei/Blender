"""Bounded, reversible cheek-continuity candidate from the saved neutral sample.

Offline compute() uses the actual exported key combination and mesh adjacency.
run() only adds one explicitly Basis-initialized Y correction key. No rest mesh,
existing key value, X/Z coordinate, bone, material, cache or file is changed.
This is a visual candidate, not an accepted facial anatomy or expression result.
"""
import json
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None

KEY='R07_cheek_surface_continuity'
INPUT=Path(__file__).resolve().parent.parent/'reconstruction-r02/face-contour-input-r06.json'


def smooth(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def compute(path=INPUT,cap_mm=1.8):
    d=json.loads(Path(path).read_text(encoding='utf8'))
    ids=np.array(d['ids'],dtype=np.int32)
    basis=np.array(d['keys']['Basis']['co'],dtype=float)
    deltas={k:np.array(v['co'],dtype=float)-basis for k,v in d['keys'].items()}
    current=basis+sum(deltas[k]*v['value'] for k,v in d['keys'].items() if k!='Basis')
    x,y,z=current.T
    weight=(smooth((np.abs(x)-.025)/.012)*(1-smooth((np.abs(x)-.066)/.014))
            *smooth((z-1.342)/.014)*(1-smooth((z-1.389)/.010))
            *smooth((-y-.055)/.020))
    mouth=np.sqrt((x/.036)**2+((z-1.356)/.017)**2)
    weight*=smooth((mouth-1.0)/.4)
    # The export also contains rear socket/oral layers at the same X/Z. Use a
    # conservative front envelope so fitting never drags those hidden layers.
    exterior=np.ones(len(ids),dtype=bool)
    for i in range(len(ids)):
        near=((x-x[i])/.004)**2+((z-z[i])/.004)**2<=1
        exterior[i]=y[i]-float(y[near].min())<=.0045
    weight[~exterior]=0
    # Exact zero on every vertex materially touched by contact/aperture keys.
    eye_protected=np.zeros(len(ids),dtype=bool)
    for name in ('R02_globe_contact','R05_aperture_L','R05_aperture_R'):
        if name in deltas:
            eye_protected |= np.linalg.norm(deltas[name],axis=1)>.00005
    weight[eye_protected]=0
    weight[z>=1.399]=0
    index={int(old):i for i,old in enumerate(ids)}
    adjacent=[set() for _ in ids]
    for a,b in d['edges']:
        if a in index and b in index:
            a,b=index[a],index[b];adjacent[a].add(b);adjacent[b].add(a)
    fits={}
    for i in np.flatnonzero(weight>1e-5):
        reached={int(i)};frontier={int(i)}
        for _ in range(6):
            frontier=set().union(*(adjacent[j] for j in frontier))-reached if frontier else set()
            reached|=frontier
        candidates=np.array(sorted(reached),dtype=int)
        u=(x[candidates]-x[i])/.020;v=(z[candidates]-z[i])/.018
        keep=((u*u+v*v)<2.8)&(np.abs(y[candidates]-y[i])<.023)&(y[candidates]<-.052)&exterior[candidates]
        candidates=candidates[keep];u=u[keep];v=v[keep]
        if len(candidates)<12:continue
        design=np.column_stack((np.ones(len(u)),u,v,u*u,u*v,v*v))
        w=np.exp(-(u*u+v*v)/1.1)
        # A local quadratic retains the broad cheek curvature; the residual
        # reduces the abrupt ridge/groove instead of flattening the entire face.
        regularizer=np.diag([1e-9,1e-8,1e-8,1e-4,1e-4,1e-4])
        solve=np.linalg.solve(design.T@(design*w[:,None])+regularizer,design.T*w[None,:])
        fits[int(i)]=(candidates,solve[0])
    work=y.copy()
    for _ in range(3):
        updated=work.copy()
        for i,(neighbors,intercept) in fits.items():
            target=float(intercept@work[neighbors])
            updated[i]=work[i]+.72*weight[i]*(target-work[i])
        work=y+np.clip(updated-y,-cap_mm*.001,cap_mm*.001)
    delta=np.zeros_like(basis);delta[:,1]=work-y
    delta[weight==0]=0
    report={'affected_over_01mm':int((np.abs(delta[:,1])>.0001).sum()),
            'maximum_forward_mm':float(max(0,-delta[:,1].min())*1000),
            'maximum_back_mm':float(max(0,delta[:,1].max())*1000),
            'maximum_absolute_mm':float(np.abs(delta[:,1]).max()*1000),
            'protected_contact_vertices':int(eye_protected.sum()),
            'exact_x_z_motion':0,
            'method':'Three local quadratic residual passes, 1.8 mm bound, fixed ocular/oral/jaw margins',
            'visual_validation':'Pending neutral front/quarter plus smile/blink; no render made by this script'}
    return d,ids,basis,delta,report


def run(strength=1.0,diagnostics_only=False):
    if bpy is None:raise RuntimeError('run() requires the integrator Blender candidate')
    if not 0<=strength<=1:raise ValueError('Candidate strength is bounded to 0..1')
    if not bpy.data.filepath or Path(bpy.data.filepath).name=='character.blend':
        raise RuntimeError('Use the saved independent candidate')
    h=bpy.data.objects['R01 Head'];keys=h.data.shape_keys.key_blocks
    if keys.get(KEY):raise RuntimeError('Candidate already exists; adjust its value or use restore()')
    d,ids,basis,delta,report=compute()
    actual=np.empty(len(keys[0].data)*3,dtype=float);keys[0].data.foreach_get('co',actual)
    actual=actual.reshape(-1,3)
    if int(ids.max())>=len(actual) or float(np.abs(actual[ids]-basis).max())>2e-6:
        raise RuntimeError('Facial Basis differs from the exported sample; refresh the sample before adapting')
    for name in ('R02_reference_likeness','R02_soft_tissue','R03_facial_balance',
                 'R04_reference_softness','R04_orbital_depth','R05_cheek_contour'):
        if abs(keys[name].value-d['keys'][name]['value'])>1e-5:
            raise RuntimeError('Static face combination changed: '+name)
    report.update({'new_key':KEY,'strength':strength,'writesPerformed':not diagnostics_only})
    if diagnostics_only:print(json.dumps(report));return report
    target=actual.copy();target[ids]+=delta
    key=h.shape_key_add(name=KEY,from_mix=False)
    key.relative_key=keys[0]
    # Never trust the initial contents of a newly allocated key.
    key.data.foreach_set('co',np.asarray(target,dtype=np.float32).ravel())
    key.slider_min=0;key.slider_max=1;key.value=float(strength)
    h.data.update_tag();bpy.context.view_layer.update()
    print(json.dumps(report));return report


def restore():
    key=bpy.data.objects['R01 Head'].data.shape_keys.key_blocks.get(KEY)
    if key:key.value=0
    bpy.context.view_layer.update()


if __name__=='__main__':
    print(json.dumps(compute()[-1],indent=2))
