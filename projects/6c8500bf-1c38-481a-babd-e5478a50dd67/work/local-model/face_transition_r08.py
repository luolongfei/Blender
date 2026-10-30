"""Local cheek-to-perioral transition candidate; explicit run(), no IO/render.

Uses the saved R06 facial data plus the accepted R07 local correction. Leaves
lip contact, nasal center, ocular contact, silhouette and every old key intact.
Only a new additive, Basis-initialized depth key is created. It is not accepted
until actual neutral/quarter and expression images have been checked.
"""
import json, runpy
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None

KEY='R08_cheek_perioral_transition'
PREVIOUS=Path(__file__).with_name('face_continuity_r07.py')


def smooth(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def compute():
    old=runpy.run_path(str(PREVIOUS))
    data,ids,basis,accepted,_=old['compute']()
    deltas={k:np.asarray(v['co'])-basis for k,v in data['keys'].items()}
    current=basis+sum(deltas[k]*v['value'] for k,v in data['keys'].items() if k!='Basis')+accepted
    x,y,z=current.T
    index={int(i):j for j,i in enumerate(ids)}
    adjacency=[set() for _ in ids]
    for a,b in data['edges']:
        if a in index and b in index:
            a,b=index[a],index[b];adjacency[a].add(b);adjacency[b].add(a)
    exterior=np.ones(len(ids),dtype=bool)
    for i in range(len(ids)):
        near=((x-x[i])/.004)**2+((z-z[i])/.004)**2<=1
        exterior[i]=y[i]-float(y[near].min())<=.0045
    ocular=np.zeros(len(ids),dtype=bool)
    for name in ('R02_globe_contact','R05_aperture_L','R05_aperture_R'):
        ocular|=np.linalg.norm(deltas[name],axis=1)>.00005
    lip=(np.abs(x)<.034)&(z>1.343)&(z<1.363)
    protected=ocular|lip|(~exterior)|(np.abs(x)<=.018)|(np.abs(x)>=.068)|(z<=1.358)|(z>=1.400)|(y>=-.060)
    mask=(smooth((np.abs(x)-.018)/.010)*(1-smooth((np.abs(x)-.055)/.013))
          *smooth((z-1.358)/.012)*(1-smooth((z-1.390)/.010)))
    # The previous patch excluded a 36 x 17 mm oral ellipse, leaving a wide
    # untouched collar. This patch explicitly includes that cheek-side collar,
    # while the actual lip band remains fixed and its transition is gradual.
    mouth=np.sqrt((x/.034)**2+((z-1.353)/.010)**2)
    mask*=smooth((mouth-1)/.55)
    mask[protected]=0
    # Graph-distance falloff avoids a correction step at fixed ocular vertices.
    frontier=set(np.flatnonzero(ocular|lip));reached=frontier.copy()
    for ring,factor in ((1,.25),(2,.65)):
        frontier=set().union(*(adjacency[i] for i in frontier))-reached
        reached|=frontier
        mask[list(frontier)]*=factor
    fits={}
    for i in np.flatnonzero(mask>1e-5):
        reached={int(i)};frontier={int(i)}
        for _ in range(6):
            frontier=set().union(*(adjacency[j] for j in frontier))-reached if frontier else set()
            reached|=frontier
        ns=np.array(sorted(reached));u=(x[ns]-x[i])/.019;v=(z[ns]-z[i])/.017
        keep=(u*u+v*v<2.6)&exterior[ns]&(y[ns]<-.058)&(np.abs(y[ns]-y[i])<.022)
        ns=ns[keep];u=u[keep];v=v[keep]
        if len(ns)<12:continue
        design=np.column_stack((np.ones(len(u)),u,v,u*u,u*v,v*v))
        w=np.exp(-(u*u+v*v)/1.2)
        reg=np.diag([1e-9,1e-8,1e-8,2e-4,2e-4,2e-4])
        fits[int(i)]=(ns,np.linalg.solve(design.T@(design*w[:,None])+reg,design.T*w)[0])
    depth=y.copy()
    before=[]
    for i,(ns,fit) in fits.items():before.append(float(fit@y[ns]-y[i]))
    for _ in range(3):
        out=depth.copy()
        for i,(ns,fit) in fits.items():
            out[i]+=.7*mask[i]*(float(fit@depth[ns])-depth[i])
        depth=y+np.clip(out-y,-.0011,.0015)
        depth[protected]=y[protected]
    delta=np.zeros_like(current);delta[:,1]=depth-y
    changed=np.flatnonzero(np.abs(delta[:,1])>1e-7)
    after=[float(fit@depth[ns]-depth[i]) for i,(ns,fit) in fits.items()]
    report={'newKey':KEY,'affectedOver01mm':int((np.abs(delta[:,1])>.0001).sum()),
        'maximumForwardMm':float(-delta[:,1].min()*1000),'maximumBackMm':float(delta[:,1].max()*1000),
        'changedBoundsXYZ':[[float(v) for v in current[changed].min(axis=0)],[float(v) for v in current[changed].max(axis=0)]],
        'fixedOcularVertices':int(ocular.sum()),'fixedLipBandVertices':int(lip.sum()),
        'exactXZChange':0,'protectedMaximumDelta':float(np.abs(delta[protected]).max()),
        'localResidualRmsMmBeforeAfter':[float(np.sqrt(np.mean(np.square(a)))*1000) for a in (before,after)],
        'method':'Current R07 exterior depth residual fitted to broad local quadratic patches; oral collar included, exact lip/eye boundaries fixed with two-ring falloff',
        'scope':'Only additive cheek/perioral Y depth. Old face keys, eyes, nose center, mouth contact, outline, neck and rest untouched.',
        'visualStatus':'Unrendered candidate. Neutral/quarter, smile and blink must be reviewed.'}
    return data,ids,basis,current,delta,report


def run(strength=1.,diagnostics_only=False):
    if bpy is None:raise RuntimeError('Explicitly run in the saved independent Blender candidate')
    if not bpy.data.filepath or Path(bpy.data.filepath).name=='character.blend':
        raise RuntimeError('Saved independent candidate required')
    if not 0<=strength<=1:raise ValueError('Strength is bounded to 0..1')
    h=bpy.data.objects['R01 Head'];keys=h.data.shape_keys.key_blocks
    if keys.get(KEY):raise RuntimeError('Patch already exists; set its value or restore()')
    data,ids,basis,reference,delta,report=compute()
    arrays={}
    for key in keys:
        a=np.empty(len(key.data)*3);key.data.foreach_get('co',a);arrays[key.name]=a.reshape(-1,3)
    actual=arrays[keys[0].name].copy()
    for key in list(keys)[1:]:
        if key.vertex_group:raise RuntimeError('Vertex-group key needs explicit evaluation: '+key.name)
        if not key.mute:actual+=(arrays[key.name]-arrays[key.relative_key.name])*key.value
    if np.abs(arrays[keys[0].name][ids]-basis).max()>2e-6:
        raise RuntimeError('Facial Basis changed; refresh sample')
    face=(reference[:,1]<-.055)&(reference[:,2]>1.340)&(reference[:,2]<1.410)
    error=float(np.abs(actual[ids][face]-reference[face]).max())
    if error>3e-6:raise RuntimeError(f'Current facial combination differs from audited R07: {error*1000:.6f} mm; no writes')
    report.update({'strength':strength,'currentMixMaximumErrorMm':error*1000,'writesPerformed':not diagnostics_only})
    if diagnostics_only:print(json.dumps(report));return report
    target=arrays[keys[0].name].copy();target[ids]+=delta
    key=h.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys[0]
    key.data.foreach_set('co',target.astype(np.float32).ravel());key.value=float(strength)
    key.slider_min=0.;key.slider_max=1.
    h.data.update_tag();bpy.context.view_layer.update();print(json.dumps(report));return report


def restore():
    key=bpy.data.objects['R01 Head'].data.shape_keys.key_blocks.get(KEY)
    if key:key.value=0.
    bpy.context.view_layer.update()


if __name__=='__main__':print(json.dumps(compute()[-1],indent=2))
