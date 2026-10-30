"""Add one reversible posterior-neck contour key using the current shape mix.

Only the new relative key is authored; Basis, existing keys, weights, bones,
poses, hair, eyes and garments are untouched. Explicit run() does not render or
save. apply_and_save() saves a new independent candidate. No Blender launch.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np
try:
    import bpy
except ImportError:
    bpy=None

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
KEY='R07_posterior_neck_contour'
REPORT='work/reconstruction-r02/posterior-neck-contour-r07.json'
TARGET_PROFILE=np.asarray([[1.270,.046],[1.285,.0475],[1.300,.048],
    [1.320,.050],[1.340,.055],[1.355,.065],[1.370,.077],[1.380,.083]])
PARAMETERS={'posteriorFixedY':.020,'posteriorFullY':.045,
    'sideFullAbsX':.045,'sideFixedAbsX':.070,
    'lowerFixedZ':1.270,'lowerFullZ':1.285,
    'upperFullZ':1.365,'upperFixedZ':1.380,
    'lateralContractionFraction':.10}


def _smooth(a,b,x):
    t=np.clip((np.asarray(x)-a)/(b-a),0.,1.)
    return t*t*(3.-2.*t)


def _pchip(xs,ys,query):
    """C1 shape-preserving cubic profile, including non-monotone source data."""
    xs=np.asarray(xs);ys=np.asarray(ys);h=np.diff(xs);d=np.diff(ys)/h
    if len(xs)<3 or np.any(h<=0):raise ValueError('At least three distinct ordered profile knots required')
    m=np.zeros_like(ys)
    for i in range(1,len(xs)-1):
        if d[i-1]*d[i]>0:
            a=2*h[i]+h[i-1];b=h[i]+2*h[i-1]
            m[i]=(a+b)/(a/d[i-1]+b/d[i])
    for i,j,k in ((0,0,1),(-1,-1,-2)):
        m[i]=((2*h[j]+h[k])*d[j]-h[j]*d[k])/(h[j]+h[k])
        if m[i]*d[j]<=0:m[i]=0.
        elif d[j]*d[k]<0 and abs(m[i])>abs(3*d[j]):m[i]=3*d[j]
    q=np.clip(np.asarray(query),xs[0],xs[-1]);i=np.clip(np.searchsorted(xs,q,side='right')-1,0,len(xs)-2)
    t=(q-xs[i])/h[i]
    return ((2*t**3-3*t*t+1)*ys[i]+(t**3-2*t*t+t)*h[i]*m[i]+
            (-2*t**3+3*t*t)*ys[i+1]+(t**3-t*t)*h[i]*m[i+1])


def plan_from_data(data):
    """Pure NumPy contour plan: proportional posterior-half ellipse compression.

    Current mix includes enabled shape keys, before skin deformation. The new
    key contains only the correction delta relative to the existing Basis, so
    it does not double-apply the present facial/likeness keys.
    """
    p=PARAMETERS;matrix=np.asarray(data['matrixWorld'],dtype=float)
    local=np.asarray([v['currentMix'] for v in data['vertices']],dtype=float)
    world=local@matrix[:3,:3].T+matrix[:3,3];x,y,z=world.T
    mid=(np.abs(x)<.003)&(y>.02)&(z>=1.267)&(z<=1.395)
    candidates=world[mid];candidates=candidates[np.argsort(candidates[:,2])]
    # This existing cage has a sparse, nearly central rear edge chain. Keep the
    # outermost point if a future saved mesh contains coincident-height seams.
    profile=[]
    for q in candidates:
        if profile and q[2]-profile[-1][0]<1.e-5:profile[-1][1]=max(profile[-1][1],float(q[1]))
        else:profile.append([float(q[2]),float(q[1])])
    profile=np.asarray(profile)
    if not 6<=len(profile)<=40 or profile[0,0]>1.275 or profile[-1,0]<1.380:
        raise RuntimeError('Unexpected posterior centerline; do not infer a contour from different topology')
    envelope=_pchip(profile[:,0],profile[:,1],z)
    desired=_pchip(TARGET_PROFILE[:,0],TARGET_PROFILE[:,1],z)
    # Never expand to the target. Shrinking the whole rear half by a shared
    # section ratio retains elliptical curvature instead of clipping to a Y cap.
    compression=np.clip((envelope-desired)/np.maximum(envelope-p['posteriorFixedY'],.005),0.,.65)
    mask=(_smooth(p['lowerFixedZ'],p['lowerFullZ'],z)*
          (1.-_smooth(p['upperFullZ'],p['upperFixedZ'],z))*
          (1.-_smooth(p['sideFullAbsX'],p['sideFixedAbsX'],np.abs(x)))*
          _smooth(p['posteriorFixedY'],p['posteriorFullY'],y))
    delta=np.zeros_like(world);delta[:,1]=-(y-p['posteriorFixedY'])*compression*mask
    delta[:,0]=-x*p['lateralContractionFraction']*compression*mask
    lengths=np.linalg.norm(delta,axis=1);changed=np.flatnonzero(lengths>1.e-8)
    if not 20<=len(changed)<=600 or lengths.max()>.040 or np.abs(delta[:,0]).max()>.003:
        raise RuntimeError('Posterior contour exceeds this bounded edit scope')
    # Exact zero outside the declared spatial region, including all front face,
    # eyes, shoulder/arms, original neck base and upper skull.
    protected=(y<=p['posteriorFixedY'])|(np.abs(x)>=p['sideFixedAbsX'])|(z<=p['lowerFixedZ'])|(z>=p['upperFixedZ'])
    if np.any(delta[protected]!=0):raise RuntimeError('Protected region has a nonzero offset')
    local_delta=delta@np.linalg.inv(matrix[:3,:3]).T
    records=[{'id':int(data['vertices'][i]['id']),'beforeWorld':world[i].tolist(),
              'afterWorld':(world[i]+delta[i]).tolist(),'localDelta':local_delta[i].tolist(),
              'mask':float(mask[i])} for i in changed]
    posterior_center=[{'id':int(data['vertices'][i]['id']),'before':world[i].tolist(),
                      'after':(world[i]+delta[i]).tolist()} for i in np.flatnonzero(mid)]
    return {'parameters':dict(p),'sourceCenterlineProfileZY':profile.tolist(),
        'desiredCenterlineProfileZY':TARGET_PROFILE.tolist(),'changedVertexCount':len(changed),
        'maximumDisplacementMeters':float(lengths.max()),'maximumLateralDisplacementMeters':float(np.abs(delta[:,0]).max()),
        'protectedVertexCount':int(protected.sum()),'posteriorCenterlineSamples':posterior_center,'changes':records,
        'selectionAndDeformationScope':'Current pre-skin shape mix. Smooth proportional Y compression of posterior half-sections plus one-tenth proportional X contraction. Z never changes; no clipping plane.'}


def _original_hash(obj,names):
    helper=runpy.run_path(str(Path(__file__).with_name('neck_weights_r07.py')))
    h=hashlib.sha256();h.update(helper['_coords'](obj.data.vertices).tobytes())
    for name in names:
        key=obj.data.shape_keys.key_blocks[name];h.update(name.encode());h.update(helper['_coords'](key.data).tobytes())
    return h.hexdigest()


def run(report_relative=REPORT):
    """Create the one additive key at value 1, without rendering or saving.

    Setting R07_posterior_neck_contour.value=0 fully disables this edit. Use the
    existing finite pose-render script in a fresh folder to inspect the result.
    """
    if bpy is None:raise RuntimeError('Execute explicitly in the existing Blender candidate')
    helper=runpy.run_path(str(Path(__file__).with_name('neck_weights_r07.py')))
    source,rig,obj=helper['_guard']()
    if helper['MARKER'] not in obj:raise RuntimeError('Apply the separate neck weight correction first')
    if KEY in obj.data.shape_keys.key_blocks:raise RuntimeError('Contour key already exists; do not accumulate this patch')
    path=(project_path()/report_relative).resolve();path.relative_to((project_path()/'work').resolve())
    if path.exists():raise RuntimeError('Use a new report path; prior evidence is retained')
    data=helper['_read'](obj);plan=plan_from_data(data)
    names=list(obj.data.shape_keys.key_blocks.keys());old_active=obj.active_shape_key_index
    original_hash=_original_hash(obj,names);weights=helper['_weights'](obj);rest=helper['_rest'](rig)
    basis=helper['_coords'](obj.data.shape_keys.reference_key.data)
    report={'sourceBlend':str(source),'sourceFileSha256':helper['_file_hash'](source),
        'sourceHadUnsavedChanges':bool(bpy.data.is_dirty),'blenderVersion':bpy.app.version_string,
        'object':obj.name,'key':KEY,'originalKeyNames':names,'originalMeshAndKeyCoordinatesSha256':original_hash,
        'shapeKeyValuesAtSelection':data['shapeKeys'],'plan':plan,'applied':False,'savedBlend':False,
        'reviewRequired':'Actual neutral, low and raised-head side renders are required. This is a localized sculpt correction, not a skin/collision acceptance score.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    key=None
    try:
        key=obj.shape_key_add(name=KEY,from_mix=False)
        # Critical: shape_key_add may copy stale mesh.vertices. Explicitly copy
        # the existing Basis first, then add only our current-mix-derived delta.
        points=basis.copy()
        for row in plan['changes']:points[row['id']]+=np.asarray(row['localDelta'])
        key.data.foreach_set('co',points.ravel());key.relative_key=obj.data.shape_keys.reference_key
        key.slider_min=0.;key.slider_max=1.;key.value=1.;key.mute=False
        obj.data.update();obj.update_tag();bpy.context.view_layer.update()
        if _original_hash(obj,names)!=original_hash:raise RuntimeError('Original mesh/Basis/existing key geometry changed')
        if helper['_weights'](obj)!=weights:raise RuntimeError('Skin weights changed')
        if helper['_rest'](rig)!=rest:raise RuntimeError('Bone rest data changed')
        after=helper['_read'](obj)
        if [r for r in after['shapeKeys'] if r['name']!=KEY]!=data['shapeKeys']:
            raise RuntimeError('An existing shape-key state changed')
        original_mix=np.asarray([v['currentMix'] for v in data['vertices']]);new_mix=np.asarray([v['currentMix'] for v in after['vertices']])
        expected=np.zeros_like(original_mix)
        for row in plan['changes']:expected[row['id']]=row['localDelta']
        error=float(np.max(np.linalg.norm(new_mix-original_mix-expected,axis=1)))
        if error>2e-6:raise RuntimeError('New additive key does not reproduce the intended local correction')
        report.update(applied=True,keyValue=1.,maximumCurrentMixReconstructionErrorMeters=error,
                      originalRawBasisAndKeysExactlyUnchanged=True,weightsAndAllBoneRestExactlyUnchanged=True,
                      otherObjectsAndCachesUntouchedByScope=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as error:
        if key is not None:obj.shape_key_remove(key)
        obj.data.update();obj.update_tag();bpy.context.view_layer.update()
        report.update(applied=False,rolledBack=True,error=str(error))
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    finally:
        obj.active_shape_key_index=old_active
    return report


def apply_and_save(output_relative='work/reconstruction-r02/reference-neck-contour-r07.blend',report_relative=REPORT):
    target=(project_path()/output_relative).resolve();target.relative_to((project_path()/'work').resolve())
    if target.exists() or target.suffix.lower()!='.blend':raise RuntimeError('A new independent .blend path is required')
    report=run(report_relative);bpy.ops.wm.save_as_mainfile(filepath=str(target))
    report.update(savedBlend=True,outputBlend=str(target))
    (project_path()/report_relative).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    print('Loaded only. run() adds one reversible posterior-neck key; apply_and_save() saves a new candidate.')
