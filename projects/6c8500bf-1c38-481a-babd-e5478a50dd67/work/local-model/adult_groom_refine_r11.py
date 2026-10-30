"""A new front/crown comb for the adult R11 head, in an independent candidate.

The successful attached root substrate, gathered roots and native tails are
protected. The failed short fringe and front scalp flow are retained hidden.
New fibres have distributed roots, a slightly offset part, varied free arches,
and staggered, narrow locks near the brows. All coordinates are real REST world
metres for this particular adult head, not generic demographic measurements.

No import-time scene work, render, save, head/rig edit, or cache calculation.
Caller must enter REST, invoke build_adult_groom_refinement(), then restore pose.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

PREFIX='R11 Recombed'
SHELL='R06 attached scalp root substrate'
SCALP='R06 continuous scalp flow'
FRINGE='R03 Forelocks from crown'
WISPS='R03 curved cheek wisps'
PROTECTED=(SHELL,'R06 full gathered roots L','R06 full gathered roots R',
           'R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
SEED=202610291103


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _smooth(x):
    import numpy as np
    x=np.clip(x,0.,1.);return x*x*(3.-2.*x)


def _unit(a):
    import numpy as np
    a=np.asarray(a,dtype=float);return a/max(float(np.linalg.norm(a)),1e-12)


def _coords(obj):
    import numpy as np
    items,key=(obj.data.vertices,'co') if obj.type=='MESH' else (obj.data.position_data,'vector')
    a=np.empty(len(items)*3,dtype=np.float32);items.foreach_get(key,a)
    return a.reshape(-1,3)


def _world(a,matrix):
    import numpy as np
    m=np.asarray(matrix,dtype=float);return a@m[:3,:3].T+m[:3,3]


def _contract(obj):
    import numpy as np
    a=_coords(obj)
    return {'data':obj.data.name,'positionSha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'matrixWorld':np.asarray(obj.matrix_world).tolist(),
            'parent':obj.parent.name if obj.parent else None,'parentBone':obj.parent_bone,
            'materials':[m.name if m else None for m in obj.data.materials]}


def _tree(obj,evaluated=False):
    import bpy
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get()) if evaluated else obj
    co=_world(_coords(e),e.matrix_world)
    tree=BVHTree.FromPolygons([Vector(v) for v in co],[tuple(p.vertices) for p in e.data.polygons],all_triangles=False)
    return tree,co


def _ray_root(tree,xy):
    from mathutils import Vector
    import numpy as np
    p,n,idx,dist=tree.ray_cast(Vector((float(xy[0]),float(xy[1]),1.70)),Vector((0,0,-1)),.4)
    if p is None:return None
    if n.z<0:n=-n
    return np.asarray(p),np.asarray(n),int(idx)


def _surface_arc(tree,center,directions,context=None):
    import numpy as np
    from mathutils import Vector
    points=[];normals=[]
    for point_index,direction in enumerate(directions):
        direction=_unit(direction)
        p,n,idx,dist=tree.ray_cast(Vector(center),Vector(direction),.3)
        if p is None:
            detail={'curve':context,'pointIndex':point_index,'pointCount':len(directions),
                    'centerWorld':np.asarray(center).tolist(),'unitDirection':direction.tolist(),
                    'lastSurfaceWorld':list(points[-1]) if points else None}
            print('R11_CROWN_DOMAIN_MISS '+json.dumps(detail),flush=True)
            raise RuntimeError('New crown path left the attached scalp domain; no floating fallback: '+json.dumps(detail))
        if n.dot(Vector(direction))<0:n=-n
        points.append(tuple(p));normals.append(tuple(n))
    return np.asarray(points),np.asarray(normals)


def _catmull(controls,count=64):
    """Centrally parameterized smooth guide; sampled densely, no mesh ribbon."""
    import numpy as np
    c=np.asarray(controls,dtype=float)
    # Chord-length knot coordinates prevent short crown controls from receiving
    # the same parameter interval as the much longer hanging cheek section.
    knots=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
    tangents=np.empty_like(c)
    tangents[0]=(c[1]-c[0])/(knots[1]-knots[0])
    tangents[-1]=(c[-1]-c[-2])/(knots[-1]-knots[-2])
    for i in range(1,len(c)-1):tangents[i]=(c[i+1]-c[i-1])/(knots[i+1]-knots[i-1])
    out=[]
    for distance in np.linspace(0.,knots[-1],count):
        i=min(int(np.searchsorted(knots,distance,side='right')-1),len(c)-2)
        i=max(0,i);h=knots[i+1]-knots[i];u=(distance-knots[i])/h
        out.append((2*u**3-3*u*u+1)*c[i]+(u**3-2*u*u+u)*h*tangents[i]
                   +(-2*u**3+3*u*u)*c[i+1]+(u**3-u*u)*h*tangents[i+1])
    return np.asarray(out)


def fringe_guide_plans():
    """Editable authored guides. Tips are at/above brows or outside the eyes."""
    plans=[]
    # The two sides differ slightly in spacing/length, not gross curl direction.
    for side in (1,-1):
        asym=.0007 if side==1 else -.0005
        # Inner locks are sparse enough to show irregular skin gaps. The outer
        # locks become a curtain, with no transverse horizontal hinge.
        rows=[
            (.005,1.449,95,.0017,-.032),
            (.014,1.454,125,.0021,-.026),
            (.025,1.4505,145,.0023,-.021),
            (.037,1.456,160,.0026,-.015),
            (.049,1.4475,195,.0030,-.008),
            (.060,1.432,210,.0032,.000),
        ]
        for k,(tx,tz,count,width,ry) in enumerate(rows):
            rx=.0013+side*(.0022+.0010*k)
            root=[rx,ry,1.53]
            # Front surface of the real R11 forehead is around y=-0.10 m.
            # The guide gradually separates from skin, instead of translating
            # old fringe tips upward with the old/new eye-line delta.
            end_y=-.108+.038*(tx/.064)**2
            controls=[root,
                      [side*(.005+.18*tx)+asym,-.059+.018*(tx/.064),1.522-.005*(tx/.064)],
                      [side*(.25*tx+.004),-.091+.035*(tx/.064)**2,1.498],
                      [side*(.76*tx+.002),-.105+.037*(tx/.064)**2,1.475],
                      [side*tx,end_y,tz+asym],
                      [side*(tx-.0009),end_y+.0004,tz-.002+asym]]
            plans.append({'name':f'front_{side}_{k}','side':side,'kind':'fringe','count':count,
                          'halfWidth':width,'halfDepth':.0010+.00015*k,'controls':controls})
        # These broad-but-fine cheek locks replace the mismatched old wisps.
        # Their midsection is outside the lateral orbital rim and follows
        # gravity; only a restrained inward bend occurs near the jaw.
        for k in range(3):
            x=.0615+.0026*k
            controls=[[.001+side*(.010+.003*k),-.003+.009*k,1.529],
                      [side*.039,-.042,1.516],
                      [side*(x-.003),-.064+.003*k,1.481],
                      [side*(x+.001),-.068+.005*k,1.437],
                      [side*(x+.0015),-.070+.005*k,1.393],
                      [side*(x-.003),-.072+.005*k,1.354-.006*k],
                      [side*(x-.007),-.070+.006*k,1.336-.009*k]]
            plans.append({'name':f'cheek_{side}_{k}','side':side,'kind':'cheek','count':150-25*k,
                          'halfWidth':.0022-.0003*k,'halfDepth':.0010,'controls':controls})
    return plans


def _strand_radii(count,radius):
    import numpy as np
    u=np.linspace(0.,1.,count)
    return radius*(.78+.22*_smooth(u/.09))*np.maximum(.018,(1.-u**3.2)**.70)


def _new_crown(root_tree,center,rng,count_per_side=2900):
    import numpy as np
    all_paths=[];all_radii=[];root_faces=[];max_lift=0.
    u=np.linspace(0.,1.,43)
    for side in (1,-1):
        for i in range(count_per_side):
            # Near-part roots plus distributed upper scalp roots prevent a
            # row of identical bundles beginning at a single visible pole.
            hit=None
            for attempt in range(40):
                near_part=rng.random()<.62
                x=.0013+side*(rng.uniform(.0012,.020) if near_part else rng.uniform(.014,.056))
                y=rng.uniform(-.046,.029) if near_part else rng.uniform(-.070,.020)
                hit=_ray_root(root_tree,(x,y))
                if hit is not None and hit[0][2]>1.478:break
                hit=None
            if hit is None:raise RuntimeError('Could not place a root on the actual upper scalp')
            root,normal,face=hit;root_faces.append(face)
            # Actual root boundary at posterior-temple azimuth ~0.56 rad has
            # polar limit 1.682 rad. The former z=1.421 endpoint crossed that
            # ear-side notch. Finish the scalp layer 9 mm higher, inside the
            # real domain, overlapping retained posterior flow/gather roots.
            # This changes the guide, not the ray test or the scalp geometry.
            endpoint=np.array([side*rng.uniform(.048,.065),rng.uniform(.026,.052),rng.uniform(1.430,1.449)])
            middle=np.array([side*(.057+.010*rng.random()),y*.35+.006,1.493+.007*rng.random()])
            a,b,c=(_unit(v-center) for v in (root,middle,endpoint))
            directions=(1.-u[:,None])**2*a+2.*u[:,None]*(1.-u[:,None])*b+u[:,None]**2*c
            surface,normals=_surface_arc(root_tree,center,directions,
                {'side':side,'strandIndex':i,'rootWorld':root.tolist(),'rootFace':face,
                 'middleWorld':middle.tolist(),'endpointWorld':endpoint.tolist()})
            # Every fibre has its own three-dimensional arch. Lift is modest
            # and vanishes into roots/terminal scalp; the substrate never moves.
            amp=rng.uniform(.0008,.0048)*(1.+.12*math.sin(y*68.))
            envelope=np.sin(np.pi*u)**1.50
            lift=.00010+amp*envelope
            lift+=rng.uniform(0.,.00035)*_smooth(u/.1)*(1.-_smooth((u-.8)/.2))
            tangent=np.gradient(surface,axis=0);across=np.cross(normals,tangent)
            across/=np.maximum(np.linalg.norm(across,axis=1,keepdims=True),1e-12)
            phase=rng.uniform(0.,math.tau)
            wave=.00010*np.sin(u*5.+phase)*np.sin(np.pi*u)
            path=surface+normals*lift[:,None]+across*wave[:,None]
            radius=rng.uniform(.000026,.000043)
            all_paths.append(path);all_radii.append(_strand_radii(len(path),radius));max_lift=max(max_lift,float(lift.max()))
        print('R11_RECOMB_CROWN '+str(side)+' '+str(count_per_side),flush=True)
    return all_paths,all_radii,{'curves':len(all_paths),'rootFaceCount':len(set(root_faces)),
                              'maxFreeLiftAboveSubstrateMeters':max_lift,'rootsDistributed':True}


def _new_front(root_tree,skin,rng):
    import numpy as np
    from mathutils import Vector
    paths=[];radii=[];guide_rows=[]
    for spec in fringe_guide_plans():
        controls=np.asarray(spec['controls'],dtype=float)
        root_hit=_ray_root(root_tree,controls[0,:2])
        if root_hit is None:raise RuntimeError('Authored front root outside actual attached scalp: '+spec['name'])
        controls[0]=root_hit[0]+root_hit[1]*.00011
        surface_controls=[]
        # The authored X/Z layout is only a styling plan. Its depth must come
        # from this actual head: a side-scalp Y coordinate is not valid while
        # a lock's X still crosses the central forehead. Cast from in front of
        # the actual face and add finite normal lift before interpolation.
        # The lower cheek curtain is free-hanging and is not flattened to skin.
        for j in range(1,len(controls)):
            if spec['kind']=='cheek' and j>=4:continue
            origin=Vector((float(controls[j,0]),-.32,float(controls[j,2])))
            hit,normal,face,distance=skin.ray_cast(origin,Vector((0,1,0)),.50)
            if hit is None:
                # Outside the lateral facial silhouette, a cheek curtain point
                # is allowed only if it is already outside actual skin. This
                # is not a missing root or scalp-ray fallback.
                if spec['kind']=='cheek' and j==3:
                    nearest,n,idx,dist=skin.find_nearest(Vector(controls[j]))
                    signed=(Vector(controls[j])-nearest).dot(n) if nearest else -1.
                    if nearest is not None and signed>.002:
                        surface_controls.append({'index':j,'freeOutsideLateralSilhouette':True,
                                                 'nearestSignedDistanceMeters':float(signed),
                                                 'world':controls[j].tolist()})
                        continue
                detail={'guide':spec['name'],'controlIndex':j,'authoredWorld':controls[j].tolist()}
                raise RuntimeError('Actual anterior head control ray missed: '+json.dumps(detail))
            if normal.y>0:normal=-normal
            lift=([0.,.0035,.0042,.0030,.0022,.0018][j] if spec['kind']=='fringe'
                  else [0.,.0036,.0040,.0030][j])
            authored=controls[j].copy();controls[j]=np.asarray(hit)+np.asarray(normal)*lift
            surface_controls.append({'index':j,'authoredWorld':authored.tolist(),'skinWorld':list(hit),
                                     'normalWorld':list(normal),'faceIndex':int(face),'normalLiftMeters':lift,
                                     'world':controls[j].tolist()})
        guide=_catmull(controls,76 if spec['kind']=='cheek' else 61)
        n=len(guide);u=np.linspace(0.,1.,n);tangent=np.gradient(guide,axis=0)
        outward=np.empty_like(guide)
        outward[:]=[0.,-1.,0.]
        lateral=np.cross(tangent,outward);lateral/=np.maximum(np.linalg.norm(lateral,axis=1,keepdims=True),1e-12)
        depth=np.cross(lateral,tangent);depth/=np.maximum(np.linalg.norm(depth,axis=1,keepdims=True),1e-12)
        # The root patch opens gradually into a volumetric oval clump, then
        # narrows softly. Fibres end at different distances instead of sharing
        # one knife-like tip. Width and depth are independent physical axes.
        for k in range(spec['count']):
            q=math.sqrt(rng.random());angle=rng.uniform(0.,math.tau)
            across=q*math.cos(angle);deep=q*math.sin(angle)
            fraction=rng.uniform(.935,1.0)
            sample_u=u*fraction
            path=np.column_stack([np.interp(sample_u,u,guide[:,j]) for j in range(3)])
            width=(.50+.50*np.sin(np.pi*u)**.8)*(1.-.80*_smooth((u-.62)/.38))
            depth_width=(.60+.40*np.sin(np.pi*u))*(1.-.82*_smooth((u-.70)/.30))
            path+=lateral*(across*spec['halfWidth']*width[:,None])
            path+=depth*(deep*spec['halfDepth']*depth_width[:,None])
            # Re-anchor the real root patch without projecting the free middle
            # onto skin. A root displacement fades smoothly after its first 20%.
            hit=_ray_root(root_tree,path[0,:2])
            if hit is None:raise RuntimeError('Front fibre root patch left scalp')
            requested=hit[0]+hit[1]*.00011
            path+=(requested-path[0])[None,:]*(1.-_smooth(u/.22))[:,None]
            phase=rng.uniform(0.,math.tau)
            path+=lateral*(rng.uniform(.00004,.00016)*np.sin(u*6.+phase)*np.sin(np.pi*u))[:,None]
            radius=rng.uniform(.000024,.000041)*(1. if spec['kind']=='fringe' else .94)
            paths.append(path);radii.append(_strand_radii(n,radius))
        guide_rows.append({'name':spec['name'],'kind':spec['kind'],'fibres':spec['count'],
                           'guideWorld':guide.tolist(),'freeGuideTip':guide[-1].tolist(),
                           'halfWidthMeters':spec['halfWidth'],'halfDepthMeters':spec['halfDepth'],
                           'actualSurfaceControls':surface_controls})
    return paths,radii,guide_rows


def _clearance(paths,radii,skin,eye_rows,group='unspecified'):
    import numpy as np
    from mathutils import Vector
    count=0;maximum=0.;min_eye=float('inf');min_skin=float('inf');max_segment=0.
    for k,(path,rr) in enumerate(zip(paths,radii)):
        for i,p in enumerate(path):
            hit,n,idx,distance=skin.find_nearest(Vector(p))
            if hit is None:raise RuntimeError('Actual new skin query failed')
            signed=float((Vector(p)-hit).dot(n))-float(rr[i])
            if signed<.00011 and distance<.020:
                push=.00014-signed
                if push>.012:
                    detail={'group':group,'curveIndex':k,'pointIndex':i,'pointCount':len(path),
                            'pointWorld':np.asarray(p).tolist(),'nearestSkinWorld':list(hit),'normalWorld':list(n),
                            'nearestFace':int(idx),'nearestDistanceMeters':float(distance),
                            'signedClearanceMeters':signed,'requiredPushMeters':float(push),
                            'curveRootWorld':path[0].tolist(),'curveTipWorld':path[-1].tolist()}
                    print('R11_FRONT_CLEARANCE_FAILURE '+json.dumps(detail),flush=True)
                    raise RuntimeError('Authored guide needs more than 12 mm skin correction: '+json.dumps(detail))
                path[i]=p+np.asarray(n)*push;count+=1;maximum=max(maximum,push)
                signed=.00014
            min_skin=min(min_skin,signed)
        for center,radius in eye_rows:
            gap=np.linalg.norm(path-center,axis=1)-radius-rr
            min_eye=min(min_eye,float(gap.min()))
            if gap.min()<.001:
                raise RuntimeError('New free hair enters the adult eye clearance envelope')
        steps=np.linalg.norm(np.diff(path,axis=0),axis=1)
        max_segment=max(max_segment,float(steps.max()))
        if not np.isfinite(path).all() or steps.max()>.009:
            raise RuntimeError('New hair guide has a discontinuity')
        if k and k%3000==0:print('R11_RECOMB_CLEARANCE '+str(k),flush=True)
    return {'correctedPoints':count,'maximumCorrectionMeters':maximum,'minimumSignedSkinClearanceMeters':min_skin,
            'minimumEyeSphereClearanceMeters':min_eye,'maximumSegmentMeters':max_segment}


def _posterior_subset(obj):
    import numpy as np
    local=_coords(obj);world=_world(local,obj.matrix_world)
    radius=obj.data.attributes.get('radius')
    if radius is None:raise RuntimeError('Expected native source radii')
    rr=np.empty(len(radius.data),dtype=np.float32);radius.data.foreach_get('value',rr)
    paths=[];radii=[];indices=[]
    for k,curve in enumerate(obj.data.curves):
        start=curve.first_point_index;end=start+curve.points_length;p=world[start:end]
        # Only actual posterior paths survive. Any curve that traverses the
        # failed front/upper crown is removed in full, not cut into loose ends.
        if p[:,1].min()>=-.003 and p[:,2].max()<=1.526:
            paths.append(p.copy());radii.append(rr[start:end].copy());indices.append(k)
    if not paths:raise RuntimeError('No genuine posterior source paths survived classification')
    return paths,radii,indices


def _material(bpy):
    m=bpy.data.materials.new(PREFIX+' black brown fibres');m.use_nodes=True
    nodes=m.node_tree.nodes;nodes.clear();links=m.node_tree.links
    out=nodes.new('ShaderNodeOutputMaterial');hair=nodes.new('ShaderNodeBsdfHairPrincipled')
    hair.parametrization='COLOR';hair.inputs['Color'].default_value=(.0058,.0044,.0033,1.)
    hair.inputs['Roughness'].default_value=.52
    if 'Radial Roughness' in hair.inputs:hair.inputs['Radial Roughness'].default_value=.55
    if 'IOR' in hair.inputs:hair.inputs['IOR'].default_value=1.55
    info=nodes.new('ShaderNodeHairInfo');ramp=nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(.0032,.0025,.0020,1.)
    ramp.color_ramp.elements[1].color=(.0090,.0066,.0048,1.)
    links.new(info.outputs['Random'],ramp.inputs['Fac']);links.new(ramp.outputs['Color'],hair.inputs['Color'])
    links.new(hair.outputs[0],out.inputs['Surface']);m.diffuse_color=(.006,.0045,.0033,1.)
    m['R11_source']='Current workspace authored material, no image dependency or donor shader copied'
    return m


def _create_curves(collection,parent,name,paths,radii,material):
    import bpy
    import numpy as np
    from mathutils import Matrix
    data=bpy.data.hair_curves.new(name);data.add_curves([len(p) for p in paths])
    coords=np.concatenate(paths).astype(np.float32)
    data.position_data.foreach_set('vector',coords.ravel())
    radius=data.attributes.new('radius','FLOAT','POINT');radius.data.foreach_set('value',np.concatenate(radii).astype(np.float32))
    data.materials.append(material)
    obj=bpy.data.objects.new(name,data);collection.objects.link(obj)
    obj.parent=parent;bpy.context.view_layer.update()
    obj.matrix_parent_inverse=parent.matrix_world.inverted();obj.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    actual=_world(_coords(obj),obj.matrix_world)
    error=float(np.max(np.abs(actual-coords)))
    if error>5e-7:raise RuntimeError('New groom object parent changed requested REST world coordinates')
    obj['R11_recombed_static_candidate']=True
    obj['R11_root_follow']='Rigid object-parent motion only; facial/root deformation and dynamics need later binding'
    return obj,{'name':name,'data':data.name,'curves':len(paths),'points':len(coords),
                'worldBounds':[actual.min(axis=0).tolist(),actual.max(axis=0).tolist()],
                'localPositionSha256':hashlib.sha256(coords.tobytes()).hexdigest(),'parentWorldErrorMeters':error}


def build_adult_groom_refinement(candidate_collection,head_object,female_to_target_matrix,
                                 *,expected_source_relative,diagnostics_only=False):
    import bpy
    import numpy as np
    P=project_path();source=(P/expected_source_relative).resolve()
    source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Exact explicit independent background candidate required')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.data.pose_position!='REST':raise RuntimeError('Caller must temporarily enter REST')
    if isinstance(head_object,str):head_object=bpy.data.objects.get(head_object)
    if isinstance(candidate_collection,str):candidate_collection=bpy.data.collections.get(candidate_collection)
    if head_object is None or head_object.name!='R11 Female head study' or head_object.type!='MESH':
        raise RuntimeError('The actual new R11 adult head is required')
    if candidate_collection is None or candidate_collection.library:raise RuntimeError('Existing local candidate collection required')
    T=np.asarray(female_to_target_matrix,dtype=float);s=1.53119993/1.6414071321487427
    expected=np.diag([s,s,s,1.]);expected[1,3]=.030
    if T.shape!=(4,4) or not np.allclose(T,expected,atol=1e-7):
        raise RuntimeError('This authored comb is explicitly calibrated to the verified adult REST transform')
    new_names=[PREFIX+' posterior retained',PREFIX+' crown flow',PREFIX+' fringe and cheek locks']
    if any(bpy.data.objects.get(n) for n in new_names):raise RuntimeError('Preserve the existing recombed candidate before retrying')
    bpy.context.view_layer.update();source_sha=_sha(source)
    protected={n:_contract(bpy.data.objects[n]) for n in PROTECTED}
    for n in (SHELL,SCALP,FRINGE,WISPS):
        if not bpy.data.objects[n].get('R11_adult_groom_fit'):raise RuntimeError('Actual successful adult refit is required: '+n)
    visibility={n:{'hideRender':bpy.data.objects[n].hide_render,'hideViewport':bpy.data.objects[n].hide_viewport,
                   'hideSet':bpy.data.objects[n].hide_get()} for n in (SCALP,FRINGE,WISPS)}
    root_tree,root_world=_tree(bpy.data.objects[SHELL]);skin,head_world=_tree(head_object,True)
    center=T[:3,:3]@np.array([0.,-.047,1.535])+T[:3,3]
    eye_rows=[]
    for side in ('L','R'):
        eye=bpy.data.objects['R11 Adult sclera '+side];v=_world(_coords(eye),eye.matrix_world)
        eye_rows.append(((v.min(axis=0)+v.max(axis=0))*.5,float((v.max(axis=0)-v.min(axis=0)).max())*.5))
    posterior,posterior_radii,posterior_indices=_posterior_subset(bpy.data.objects[SCALP])
    rng=np.random.default_rng(SEED)
    crown,crown_radii,crown_info=_new_crown(root_tree,center,rng)
    front,front_radii,guide_rows=_new_front(root_tree,skin,rng)
    crown_clearance=_clearance(crown,crown_radii,skin,eye_rows,'crown')
    front_clearance=_clearance(front,front_radii,skin,eye_rows,'front')
    report={'schemaVersion':1,'source':expected_source_relative,'sourceSha256':source_sha,'helperSha256':_sha(__file__),
            'status':'prepared geometry; actual renders decide adoption','seed':SEED,'femaleToTargetMatrix':T.tolist(),
            'rootSubstrateEdited':False,'protectedContracts':protected,
            'replacedObjectsRetainedHidden':visibility,'posteriorOriginalCurveIndices':posterior_indices,
            'posteriorFibresRetained':len(posterior),'crown':crown_info,'crownClearance':crown_clearance,
            'frontClearance':front_clearance,'frontGuides':guide_rows,'frontFibres':len(front),
            'geometryWrites':not diagnostics_only,'savedBlend':False,'appearanceAccepted':False,
            'donorHairGeometryUsed':False,
            'sourceAttribution':'New workspace authored paths. Bystedt donor was inspected for guide/parting workflow; no donor points/nodes/images are copied here.',
            'limits':['Static candidate only; no new simulation, collision cache, facial deformation binding, runtime cards or VRM refresh.',
                      'Protected posterior/tail geometry retains its previous limitations; new front/quarter/back views remain necessary.',
                      'The successful attached root substrate is unchanged; crown volume comes only from the free fibre paths.',
                      'New material belongs to new objects only; continuity with retained gathers/tails must be judged in actual light.']}
    if diagnostics_only:return report
    created=[];material=None
    try:
        material=_material(bpy);object_reports=[]
        for name,paths,radii in [(new_names[0],posterior,posterior_radii),(new_names[1],crown,crown_radii),(new_names[2],front,front_radii)]:
            obj,row=_create_curves(candidate_collection,head_object,name,paths,radii,material);created.append(obj);object_reports.append(row)
        for n in (SCALP,FRINGE,WISPS):
            obj=bpy.data.objects[n];obj.hide_render=True;obj.hide_viewport=True;obj.hide_set(True)
        bpy.context.view_layer.update()
        for name,expected_contract in protected.items():
            if _contract(bpy.data.objects[name])!=expected_contract:raise RuntimeError('Protected groom changed: '+name)
        if _sha(source)!=source_sha:raise RuntimeError('Input file was changed externally during construction')
        report['objects']=object_reports;report['protectedUnchanged']=True
        return report
    except Exception:
        for obj in created:
            data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if data.users==0:bpy.data.hair_curves.remove(data)
        if material and material.users==0:bpy.data.materials.remove(material)
        for n,row in visibility.items():
            obj=bpy.data.objects[n];obj.hide_render=row['hideRender'];obj.hide_viewport=row['hideViewport'];obj.hide_set(row['hideSet'])
        bpy.context.view_layer.update();raise


if __name__=='__main__':
    print('Loaded only. Explicit build_adult_groom_refinement(collection, adult_head, T, expected_source_relative=...) on a REST candidate; caller owns save/render.')
