"""Restyle the fitted real donor guides into bangs and existing low twin tails.

Works only on the independent fitted-donor study. Keeps its real 346/30 guide
roots, UV, two-island surface and author GN interpolation/style system. The
upper authored paths remain the basis; no synthetic few-bundle replacement.
No import-time scene work, save, render, source-head or native-tail edits.
"""
import hashlib,json,math,runpy,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
NAMES=('R11 Donor / long hair main','R11 Donor / long hair strands')
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
SOURCE_GUIDES_SHA256='5a6c5e1b1965339da2116b5a4af07c75d2449de9200503319dddc6df9cc1c0a3'
VISIBLE_ARC_SECONDARY=(11,12,13,22,27,28)


def _smooth(t):
    import numpy as np
    t=np.clip(t,0.,1.);return t*t*(3.-2.*t)


def _resample(points,count,max_length=None):
    import numpy as np
    a=np.asarray(points,dtype=float);d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(a,axis=0),axis=1))]
    if d[-1]<1e-8:raise RuntimeError('Degenerate stored donor guide')
    total=d[-1] if max_length is None else min(float(max_length),float(d[-1]))
    targets=np.linspace(0.,total,count)
    return np.column_stack([np.interp(targets,d,a[:,j]) for j in range(3)])


def _cut_at_z(path,z):
    import numpy as np
    for i in range(1,len(path)):
        if path[i,2]<=z and path[i-1,2]>z:
            f=(z-path[i-1,2])/(path[i,2]-path[i-1,2])
            return np.vstack([path[:i],path[i-1]+f*(path[i]-path[i-1])])
    if path[0,2]<=z:return path[:1].copy()
    return path.copy()


def _forelock(path,source_root,index,weight,skin,count):
    import numpy as np
    from mathutils import Vector
    root=path[0].copy()
    x=float(np.clip(root[0]+.0030*math.sin(index*1.917+source_root[1]*3.),-.060,.060))
    # The continuous transition creates shorter bangs centrally and longer
    # temple/cheek curtains. No equal row of pin-shaped lock endings.
    outer=float(_smooth((abs(x)-.034)/.026))
    z=1.449-.010*outer+.0024*math.sin(index*2.39996)
    z=1.355+(z-1.355)*weight
    if weight<.55:x=math.copysign(.061+.004*math.sin(index*1.1),x if abs(x)>.005 else source_root[0]+.001)
    hit,normal,face,distance=skin.ray_cast(Vector((x,-.32,z)),Vector((0,1,0)),.50)
    if hit is None:
        if weight>=.55:raise RuntimeError('Actual front endpoint ray missed '+json.dumps({'guide':index,'xz':[x,z]}))
        endpoint=np.array([x,-.075,z])
        nearest,n,idx,dist=skin.find_nearest(Vector(endpoint))
        if nearest is None or (Vector(endpoint)-nearest).dot(n)<.001:
            raise RuntimeError('Free lateral curtain endpoint is not outside actual skin')
    else:
        if normal.y>0:normal=-normal
        endpoint=np.asarray(hit)+np.asarray(normal)*(.0020+.0006*math.sin(index*1.7)**2)
    desired=float(np.linalg.norm(endpoint-root))
    base=_resample(path,max(80,count*4),max(.045,desired*1.35))
    old_vector=Vector(base[-1]-root);new_vector=Vector(endpoint-root)
    if old_vector.length<.010:raise RuntimeError('Author forelock basis is too short')
    rotation=old_vector.rotation_difference(new_vector);scale=new_vector.length/old_vector.length
    rotated=np.asarray([np.asarray(rotation@Vector(p-root))*scale+root for p in base])
    u=np.linspace(0.,1.,len(base));built=[];ray_controls=0;largest_style_distance=0.
    for j,(original,q) in enumerate(zip(base,rotated)):
        if j==0:built.append(root.copy());continue
        old_hit,old_n,idx,dist=skin.find_nearest(Vector(original))
        old_clear=max(0.,(Vector(original)-old_hit).dot(old_n)) if old_hit is not None else .003
        gap=.00030+min(.010,old_clear*.75+.0015)*math.sin(math.pi*u[j])**1.25+.0017*_smooth((u[j]-.70)/.30)
        p,n,face,dist=skin.ray_cast(Vector((float(q[0]),-.32,float(q[2]))),Vector((0,1,0)),.50)
        if p is not None:
            if n.y>0:n=-n
            surface=np.asarray(p)+np.asarray(n)*gap
            blend=float(_smooth(u[j]/.16))
            result=q*(1.-blend)+surface*blend;ray_controls+=1
        else:
            # A free crown point above the head or a lateral cheek point may
            # have no forward-ray intersection, but must already be external.
            p,n,face,dist=skin.find_nearest(Vector(q))
            if p is None:raise RuntimeError('Actual forelock nearest surface missing')
            signed=(Vector(q)-p).dot(n)
            result=q if signed>=gap else np.asarray(p)+np.asarray(n)*gap
        largest_style_distance=max(largest_style_distance,float(np.linalg.norm(result-q)));built.append(result)
    if largest_style_distance>.025:
        raise RuntimeError('Author-based forelock styling requires excessive surface displacement '+json.dumps({'guide':index,'meters':largest_style_distance}))
    result=_resample(np.asarray(built),count);result[0]=root;result[-1]=endpoint
    return result,{'kind':'fringe' if weight>=.55 else 'temple-curtain','sourceFrontWeight':weight,
                   'authorBasisScale':float(scale),'actualFrontSurfaceSamples':ray_controls,
                   'maximumStyleSurfaceDistanceMeters':largest_style_distance,'targetTip':endpoint.tolist()}


def _gather(path,index,pins,count):
    import numpy as np
    # The real donor contains deliberately short crown/parting guides. Their
    # end is not a missing ponytail join: preserve that layer instead of
    # stretching a short author curl down to a tie.
    if float(path[:,2].min())>1.435:
        return path.copy(),{'kind':'author-crown-layer','retainedUpperAuthorPoints':len(path),
                            'targetTip':path[-1].tolist()}
    prefix=_cut_at_z(path,1.4315)
    q=prefix[-1];side=0 if q[0]>=0 else 1
    roots,directions=pins[side]
    # Distributed real native pins, not one merged artificial endpoint.
    slot=(index*619+137)%len(roots);pin=roots[slot];direction=directions[slot]
    if len(prefix)>1:tangent=prefix[-1]-prefix[-2];tangent/=max(np.linalg.norm(tangent),1e-12)
    else:tangent=np.array([math.copysign(.25,q[0]),.1,-1.]);tangent/=np.linalg.norm(tangent)
    c1=q+tangent*min(.014,np.linalg.norm(pin-q)*.22)
    c2=pin-direction*.018
    t=np.linspace(0.,1.,32)[:,None]
    extension=(1.-t)**3*q+3.*(1.-t)**2*t*c1+3.*(1.-t)*t*t*c2+t**3*pin
    joined=np.vstack([prefix,extension[1:]])
    arc=float(np.linalg.norm(np.diff(joined,axis=0),axis=1).sum())
    # Reserve the final short tangent sample before resampling. Overwriting
    # two points of a uniformly sampled 15-point guide would skip a long final
    # interval and create a visible 4 cm kink near the tie.
    leading=_resample(joined,count-1,max_length=max(.002,arc-.0018))
    leading[-1]=pin-direction*.0018
    result=np.vstack([leading,pin]);result[0]=path[0]
    return result,{'kind':'gather','nativeTail':TAILS[side],'nativePinIndex':int(slot),
                   'retainedUpperAuthorPoints':len(prefix),'targetTip':pin.tolist()}


def _skin_check(path,skin,index,protected_end=False):
    import numpy as np
    from mathutils import Vector
    largest=0.;count=0
    for j,p in enumerate(path):
        if j==0 or (protected_end and j>=len(path)-2):continue
        hit,n,face,distance=skin.find_nearest(Vector(p))
        if hit is None:raise RuntimeError('Missing current head collision surface')
        signed=(Vector(p)-hit).dot(n)
        if signed<.00016:
            push=.00020-signed
            if push>.012:
                raise RuntimeError('Restyled donor guide exceeds skin correction limit '+json.dumps({
                    'guide':index,'point':j,'world':p.tolist(),'nearest':list(hit),'normal':list(n),'pushMeters':push}))
            path[j]=p+np.asarray(n)*push;largest=max(largest,float(push));count+=1
    steps=np.linalg.norm(np.diff(path,axis=0),axis=1)
    if not np.isfinite(path).all() or steps.max()>.035:
        raise RuntimeError('Restyled donor guide lost continuity '+json.dumps({'guide':index,'maximumSegmentMeters':float(steps.max())}))
    return {'skinCorrectedGuidePoints':count,'maximumSkinCorrectionMeters':largest,'maximumGuideSegmentMeters':float(steps.max())}


def _segment_eye_gap(path,center,radius):
    import numpy as np
    a=path[:-1];d=path[1:]-a
    t=np.clip(np.sum((center-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-16),0.,1.)
    return float(np.linalg.norm(a+t[:,None]*d-center,axis=1).min()-radius)


def restyle_fitted_donor(head_object,*,expected_source_relative):
    import bpy,numpy as np
    P=project_path();source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Explicit independent fitted donor source required')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.data.pose_position!='REST':raise RuntimeError('Caller must enter neutral REST')
    if isinstance(head_object,str):head_object=bpy.data.objects.get(head_object)
    if head_object is None or head_object.name!='R11 Female head study':raise RuntimeError('Actual adult head required')
    fit=runpy.run_path(str(Path(__file__).with_name('donor_groom_adult_r11.py')))
    sha=fit['_sha'];coords=fit['_coords'];world=fit['_world'];source_sha=sha(source)
    objects=[bpy.data.objects[n] for n in NAMES]
    if any(o.get('R11_donor_restyle') for o in objects):raise RuntimeError('Preserve previous style before retry')
    bpy.context.view_layer.update();scalp,skin,head_report=fit['_head_trees'](head_object)
    head_hash=hashlib.sha256(coords(head_object).tobytes()).hexdigest()
    growth=bpy.data.objects['R11 Donor / long hair growth mesh']
    growth_hash=hashlib.sha256(coords(growth).tobytes()).hexdigest()
    eye_rows=[]
    for side in ('L','R'):
        eye=bpy.data.objects['R11 Adult sclera '+side];v=world(coords(eye),eye.matrix_world)
        eye_rows.append(((v.min(0)+v.max(0))*.5,float((v.max(0)-v.min(0)).max())*.5))
    source_arrays=P/'work/reconstruction-r02/hair-donor-guides-r10.npz'
    if sha(source_arrays)!=SOURCE_GUIDES_SHA256:raise RuntimeError('Original author guide observation changed')
    with np.load(source_arrays,allow_pickle=False) as z:
        original={i:(z[f'guide_{i}_local'].copy(),z[f'guide_{i}_starts'].copy(),z[f'guide_{i}_lengths'].copy()) for i in (0,1)}
    pins=[];tail_contract={}
    for name in TAILS:
        obj=bpy.data.objects[name];local=coords(obj);a=world(local,obj.matrix_world)
        starts=np.array([c.first_point_index for c in obj.data.curves]);roots=a[starts];d=a[starts+1]-roots
        d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12);pins.append((roots,d))
        tail_contract[name]=hashlib.sha256(local.tobytes()).hexdigest()
    prepared={};reports=[]
    for group,obj in enumerate(objects):
        local=coords(obj);positions=world(local,obj.matrix_world);new=np.empty_like(positions)
        donor,starts,lengths=original[group]
        if len(local)!=len(donor) or len(obj.data.curves)!=len(starts):raise RuntimeError('Author guide topology changed unexpectedly')
        for k,(start,length) in enumerate(zip(starts,lengths)):
            start=int(start);length=int(length);path=positions[start:start+length];source_root=donor[start]
            front=float(_smooth((-source_root[1]-.86)/.38))
            width=float(1.-_smooth((abs(source_root[0])-.46)/.23))
            weight=front*width
            if group==1 and k in VISIBLE_ARC_SECONDARY:
                # These are the real short visible crown arcs after the fit
                # removed tails proven to be buried in the original head.
                # They are not newly available long fringe or gather guides.
                result=path.copy();row={'kind':'author-crown-layer','retainedUpperAuthorPoints':length,
                                       'preservedActualTrimmedVisibleArc':True,'targetTip':path[-1].tolist()}
            elif weight>.12 and source_root[2]>15.70:
                result,row=_forelock(path,source_root,k+group*1000,weight,skin,length)
            else:result,row=_gather(path,k+group*1000,pins,length)
            row.update(_skin_check(result,skin,k+group*1000,row['kind']=='gather'))
            eye_gap=min(_segment_eye_gap(result,center,radius) for center,radius in eye_rows)
            if eye_gap<.001:
                raise RuntimeError('Restyled donor enters actual adult eye envelope '+json.dumps({'guide':k,'group':group,'gapMeters':eye_gap}))
            row['minimumEyeEnvelopeGapMeters']=eye_gap
            row.update(object=obj.name,guideIndex=k,storedPoints=length,sourceRootDonorUnits=source_root.tolist())
            if not np.array_equal(result[0],path[0]):raise RuntimeError('Restyling displaced a real donor root')
            new[start:start+length]=result;reports.append(row)
        inverse=np.asarray(obj.matrix_world.inverted());prepared[obj]=world(new,inverse).astype(np.float32)
        prepared[obj][starts]=local[starts]
    originals={o:o.data for o in objects};copies={}
    for obj in objects:
        originals[obj].use_fake_user=True;copy=originals[obj].copy();copy.name='R11 twin tail style / '+obj.name
        copy.position_data.foreach_set('vector',prepared[obj].ravel());copy.update_tag();copies[obj]=copy
    for obj in objects:
        obj.data=copies[obj];obj['R11_donor_restyle']='Author-derived distributed fringe and original native tail pins; static candidate'
    for name in TAILS:
        obj=bpy.data.objects[name];obj.hide_render=False;obj.hide_viewport=False;obj.hide_set(False)
    bpy.context.view_layer.update()
    for name,digest in tail_contract.items():
        if hashlib.sha256(coords(bpy.data.objects[name]).tobytes()).hexdigest()!=digest:raise RuntimeError('Native tail data changed')
    if hashlib.sha256(coords(head_object).tobytes()).hexdigest()!=head_hash or hashlib.sha256(coords(growth).tobytes()).hexdigest()!=growth_hash:
        raise RuntimeError('Protected adult head or donor growth surface changed')
    if sha(source)!=source_sha:raise RuntimeError('Input source changed')
    return {'schemaVersion':1,'source':expected_source_relative,'sourceSha256':source_sha,'helperSha256':sha(__file__),
            'stage':'Actual donor guide restyle candidate; visual acceptance pending','head':head_report,
            'guideReports':reports,'styleCounts':{kind:sum(r['kind']==kind for r in reports) for kind in ('fringe','temple-curtain','gather','author-crown-layer')},
            'authorRootsUnchanged':True,'preservedActualTrimmedSecondaryGuides':list(VISIBLE_ARC_SECONDARY),
            'eyeEnvelopeCheck':'Closest point on every stored segment, not only guide control points',
            'nativeTailHashes':tail_contract,'nativeTailsEdited':False,
            'growthMeshUVAndGNUnchanged':True,'retainedFittedGuideData':{o.name:d.name for o,d in originals.items()},
            'savedBlend':False,'appearanceAccepted':False,
            'limits':['Stored guide constraints do not verify final dense GN hair/skin contact; actual front, quarter and profile are required.',
                      'Original native tail shape/cache remains unchanged and has not been accepted as final matching the design reference.',
                      'Author off-centre part remains in this first mature-groom style; no unsupported root-cap inflation or hidden opaque substrate.',
                      'Static source only; new dynamics, root deformation binding, realtime hair and exports remain pending.',
                      'Adapted author hair remains subject to Daniel Bystedt attribution and official CC BY-SA terms; no license version is invented.']}


if __name__=='__main__':print('Loaded only. Explicit restyle_fitted_donor(adult_head, expected_source_relative=...) on an independent neutral REST study.')
