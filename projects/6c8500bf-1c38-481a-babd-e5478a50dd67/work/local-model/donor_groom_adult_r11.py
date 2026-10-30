"""Fit the real Bystedt guide/surface/GN system to the actual adult head.

First stage is an explicitly labelled donor-fit study, not the final twin-tail
style. It retains all 346+30 author guides and the actual interpolation,
parting-density texture, clump, noise and profile nodes. No donor head is used.
The original root cap and old groom remain in the file, hidden for this study.
Caller owns neutral REST state, save and actual rendering. No import-time work.
"""
import hashlib,json,math,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
SOURCE_REL='assets/blender-hair-styles-bystedt-2023/hair_nodes-female_hair_styles.blend'
SOURCE_SHA='1ad6202095c1793678fee7d69a7e9f8b5fdb6e5c293d300eb1062d2d437e8d48'
DONOR_NAMES=('long hair growth mesh','long hair main','long hair strands')
OLD_GROOM=('R06 attached scalp root substrate','R06 continuous scalp flow','R03 Forelocks from crown',
           'R03 curved cheek wisps','R06 full gathered roots L','R06 full gathered roots R',
           'R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _coords(obj):
    import numpy as np
    items,key=(obj.data.vertices,'co') if obj.type=='MESH' else (obj.data.position_data,'vector')
    a=np.empty(len(items)*3,dtype=np.float32);items.foreach_get(key,a);return a.reshape(-1,3)


def _world(a,matrix):
    import numpy as np
    m=np.asarray(matrix,dtype=float);return a@m[:3,:3].T+m[:3,3]


def _head_trees(head):
    import bpy,numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    e=head.evaluated_get(bpy.context.evaluated_depsgraph_get());v=_world(_coords(e),e.matrix_world)
    faces=[tuple(p.vertices) for p in e.data.polygons]
    fs=e.data.attributes.get('.sculpt_face_set')
    if fs is None:raise RuntimeError('Actual author head face sets required for scalp/ear separation')
    labels=np.empty(len(fs.data),dtype=np.int32);fs.data.foreach_get('value',labels)
    scalp=[p for p,label in zip(faces,labels) if int(label) not in (2,3,4,5,7,8)]
    points=[Vector(p) for p in v]
    return (BVHTree.FromPolygons(points,scalp,all_triangles=False),
            BVHTree.FromPolygons(points,faces,all_triangles=False),
            {'vertices':len(v),'faces':len(faces),'scalpFaces':len(scalp),
             'worldBounds':[v.min(0).tolist(),v.max(0).tolist()]})


class SurfaceMap:
    def __init__(self,old,new,triangles,old_center,new_normals):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        self.old=[Vector(p) for p in old];self.new=[Vector(p) for p in new]
        self.normals=[Vector(p) for p in new_normals];self.center=Vector(old_center)
        self.triangles=triangles
        self.tree=BVHTree.FromPolygons(self.old,[tuple(int(i) for i in f) for f in triangles],all_triangles=True)
    def point(self,position,root=False):
        from mathutils import Vector
        from mathutils.geometry import barycentric_transform
        p=Vector(position);hit,n,index,distance=self.tree.find_nearest(p)
        if hit is None:raise RuntimeError('Stored donor growth surface query failed')
        if n.dot(hit-self.center)<0:n=-n
        ids=self.triangles[index];a,b,c=(self.old[int(i)] for i in ids)
        na,nb,nc=(self.new[int(i)] for i in ids)
        qa,qb,qc=(self.normals[int(i)] for i in ids)
        new_hit=barycentric_transform(hit,a,b,c,na,nb,nc)
        new_normal=barycentric_transform(hit,a,b,c,qa,qb,qc).normalized()
        rotation=n.rotation_difference(new_normal)
        # Both source and mapped data remain in donor units. Their shared
        # object matrix supplies the real metre scale, including GN radii.
        result=new_hit if root else new_hit+rotation@(p-hit)
        return result,float(distance),int(index)


def _polar_domain(vertices,triangles,center,outermost):
    """Real boundary intersections, excluding the donor's internal part seam.

    Taking the outermost donor intersection gives its outer cap envelope even
    though its two islands also have seam boundaries. The target root cap has
    a single actual outer boundary; its innermost intersection is conservative.
    """
    import numpy as np
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    edge,count=np.unique(edges,axis=0,return_counts=True);edge=edge[count==1]
    q=np.asarray(vertices)[edge]-np.asarray(center)
    angles=np.linspace(-math.pi,math.pi,8193);limits=[]
    for phi in angles:
        h=q[:,:,0]*math.sin(phi)-q[:,:,1]*math.cos(phi);den=h[:,1]-h[:,0]
        f=np.divide(-h[:,0],den,out=np.full(len(h),-1.),where=np.abs(den)>1e-15)
        cut=q[:,0]+f[:,None]*(q[:,1]-q[:,0])
        valid=(f>=0)&(f<=1)&((cut[:,0]*math.cos(phi)+cut[:,1]*math.sin(phi))>0)
        if not np.any(valid):raise RuntimeError('Actual scalp polar boundary has an unobserved direction')
        theta=np.arccos(np.clip(cut[valid,2]/np.linalg.norm(cut[valid],axis=1),-1.,1.))
        limits.append(float(theta.max() if outermost else theta.min()))
    limits=np.asarray(limits)
    # A conservative five-sample envelope covers finite angular interpolation.
    envelope=np.stack([np.roll(limits,k) for k in (-2,-1,0,1,2)])
    limits=envelope.max(axis=0) if outermost else envelope.min(axis=0)
    return angles,limits,len(edge)


def _fit_growth(old,triangles,tree,T,M,target_root):
    import numpy as np
    from mathutils import Vector
    # Shared anatomical centre in donor's ten-times body units and target REST.
    donor_center=np.array([0.,-.47,15.35]);target_center=T[:3,:3]@np.array([0.,-.047,1.535])+T[:3,3]
    inverse=np.linalg.inv(M);new=[];normals=[];rays=[]
    target_root.data.calc_loop_triangles()
    target_tri=np.array([tuple(t.vertices) for t in target_root.data.loop_triangles],dtype=np.int32)
    target_world=_world(_coords(target_root),target_root.matrix_world)
    source_angles,source_limits,source_edges=_polar_domain(old,triangles,donor_center,True)
    target_angles,target_limits,target_edges=_polar_domain(target_world,target_tri,target_center,False)
    compressions=[];margin=.012
    for i,p in enumerate(old):
        direction=p-donor_center;direction/=np.linalg.norm(direction)
        phi=math.atan2(float(direction[1]),float(direction[0]));theta=math.acos(float(np.clip(direction[2],-1.,1.)))
        old_limit=float(np.interp(phi,source_angles,source_limits))
        new_limit=float(np.interp(phi,target_angles,target_limits))-margin
        ratio=min(1.,new_limit/old_limit)
        mapped_theta=theta*ratio
        direction=np.array([math.sin(mapped_theta)*math.cos(phi),math.sin(mapped_theta)*math.sin(phi),math.cos(mapped_theta)])
        compressions.append({'vertex':i,'sourceTheta':theta,'targetTheta':mapped_theta,'phi':phi,'ratio':ratio})
        hit,n,face,dist=tree.ray_cast(Vector(target_center),Vector(direction),.35)
        if hit is None:
            raise RuntimeError('Donor growth scalp ray miss '+json.dumps({'vertex':i,'source':p.tolist(),'direction':direction.tolist()}))
        if n.dot(Vector(direction))<0:n=-n
        w=np.asarray(hit)+np.asarray(n)*.00025
        new.append(inverse[:3,:3]@w+inverse[:3,3]);normals.append(np.asarray(n));rays.append(int(face))
    return np.asarray(new),np.asarray(normals),donor_center,{'growthVertices':len(old),'skinGapMeters':.00025,
              'actualScalpFaceCount':len(set(rays)),'sourceCenter':donor_center.tolist(),'targetCenter':target_center.tolist(),
              'domainMapping':'Source outer cap polar envelope to actual validated target scalp domain; proportional rows, no ear roots or floating fallback',
              'sourceBoundaryEdges':source_edges,'targetBoundaryEdges':target_edges,'angularSafetyMarginRadians':margin,
              'compressedVertices':sum(r['ratio']<.99999 for r in compressions),
              'minimumPolarRatio':min(r['ratio'] for r in compressions),'vertex16Diagnostic':compressions[16]}


def _author_visible_inputs(P,objects):
    """Use observed SurfaceDeform inputs; cut only proven buried helper tails.

    Six secondary short crown guides in the official file terminate inside the
    author's head (up to 0.34 donor units). Those are not visible free hairs.
    Preserve their visible authored arc and cut at its final skin entry before
    fitting; never push the buried remainder outside to make an artificial spike.
    """
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    W=P/'work/reconstruction-r02'
    probe=json.loads((W/'hair-donor-surface-deform-r11.json').read_text(encoding='utf8'))
    actual=W/'hair-donor-surface-deform-r11.npz'
    if probe['sourceSha256']!=SOURCE_SHA or _sha(actual)!=probe['arraysSha256']:
        raise RuntimeError('Verified actual donor SurfaceDeform observation required')
    observed=json.loads((W/'hair-donor-guides-r10.json').read_text(encoding='utf8'))
    stored=W/'hair-donor-guides-r10.npz'
    if observed['sourceSha256']!=SOURCE_SHA or _sha(stored)!=observed['arrays']['sha256']:
        raise RuntimeError('Verified original donor topology observation required')
    with np.load(stored,allow_pickle=False) as z:
        v=z['reference_head_local'];matrix=z['reference_head_matrix'];vertices=_world(v,matrix)
        corners=z['reference_head_loop_vertex'];starts=z['reference_head_polygon_start'];sizes=z['reference_head_polygon_size']
        faces=[tuple(int(x) for x in corners[int(st):int(st+n)]) for st,n in zip(starts,sizes)]
    if probe['headEvaluatedVertices']!=len(vertices) or probe['growth']['modifiers']:
        raise RuntimeError('Source source-head/growth evaluation requires a new topology strategy')
    tree=BVHTree.FromPolygons([Vector(p) for p in vertices],faces,all_triangles=False)
    def signed(p):
        hit,n,idx,distance=tree.find_nearest(Vector(p))
        if hit is None:raise RuntimeError('Original donor head query failed')
        return float((Vector(p)-hit).dot(n))
    def resample(p,count):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if arc[-1]<1e-8:raise RuntimeError('Degenerate original visible guide')
        u=np.linspace(0.,arc[-1],count)
        return np.column_stack([np.interp(u,arc,p[:,j]) for j in range(3)])
    prepared={};trimmed=[]
    with np.load(actual,allow_pickle=False) as z:
        for group,obj in enumerate(objects):
            raw=_coords(obj)
            if not np.array_equal(raw,z[f'guide_{group}_raw']):raise RuntimeError('Appended guides differ from actual source observation')
            data=z[f'guide_{group}_surface_deformed'].copy()
            starts=z[f'guide_{group}_starts'];lengths=z[f'guide_{group}_lengths']
            if group==1:
                for k,(st,length) in enumerate(zip(starts,lengths)):
                    st=int(st);length=int(length);original=data[st:st+length].copy();tail_signed=signed(original[-1])
                    if tail_signed>=-.015:continue
                    dense=resample(original,length*10);dist=np.array([signed(p) for p in dense]);clearance=.003
                    outside=np.flatnonzero(dist>=clearance)
                    if not len(outside):raise RuntimeError('Secondary author guide has no visible arc to retain: '+str(k))
                    last=int(outside[-1])
                    if last>=len(dense)-1:raise RuntimeError('Inconsistent trailing burial observation')
                    low=dense[last].copy();high=dense[last+1].copy()
                    for _ in range(24):
                        mid=(low+high)*.5
                        if signed(mid)>=clearance:low=mid
                        else:high=mid
                    visible=np.vstack([dense[:last+1],low]);new=resample(visible,length);new[0]=original[0]
                    old_length=float(np.linalg.norm(np.diff(original,axis=0),axis=1).sum())
                    new_length=float(np.linalg.norm(np.diff(new,axis=0),axis=1).sum())
                    if new_length/old_length<.25:raise RuntimeError('Too little original helper arc remains to adapt safely')
                    data[st:st+length]=new
                    trimmed.append({'secondaryGuide':k,'originalTailSignedDistanceDonorUnits':tail_signed,
                                    'sourceVisibleEntryPoint':low.tolist(),'newTailSignedDistanceDonorUnits':signed(new[-1]),
                                    'originalLengthDonorUnits':old_length,'visibleLengthDonorUnits':new_length,
                                    'retainedLengthFraction':new_length/old_length,'storedPointCountPreserved':length})
            prepared[obj]=data
    return prepared,{'actualSurfaceDeformArraysSha256':probe['arraysSha256'],
                     'surfaceDeformBakedFromActualObservation':True,'trimmedBuriedSecondaryTails':trimmed,
                     'reason':'Original author helper tails were inside the original head, not visible free shafts; retain visible arcs instead of raising the target correction limit.'}


def _map_guides(obj,mapper,M,skin,source_positions):
    import numpy as np
    from mathutils import Vector
    inverse=np.linalg.inv(M);old=source_positions.copy();new=np.empty_like(old)
    roots={int(c.first_point_index) for c in obj.data.curves}
    root_distances=[];largest_push=0.;corrected=0
    for i,p in enumerate(old):
        result,distance,face=mapper.point(p,i in roots)
        local=np.asarray(result);world=M[:3,:3]@local+M[:3,3]
        if i in roots:root_distances.append(distance)
        else:
            hit,n,index,dist=skin.find_nearest(Vector(world))
            if hit is None:raise RuntimeError('Actual target head nearest query failed')
            signed=(Vector(world)-hit).dot(n)
            if signed<.00014 and dist<.025:
                push=.00018-signed
                if push>.012:
                    raise RuntimeError('Donor guide requires excessive target-skin correction '+json.dumps({
                        'object':obj.name,'point':i,'world':world.tolist(),'nearest':list(hit),'normal':list(n),'push':push}))
                world+=np.asarray(n)*push;local=inverse[:3,:3]@world+inverse[:3,3]
                largest_push=max(largest_push,float(push));corrected+=1
        new[i]=local
    if not np.isfinite(new).all():raise RuntimeError('Non-finite donor mapping')
    root_max=max(root_distances)
    if root_max>.08:raise RuntimeError('Unexpected stored author root/growth distance in donor units: '+str(root_max))
    return new,{'object':obj.name,'authorGuides':len(obj.data.curves),'authorPoints':len(old),
                'storedRootMaxDistanceDonorUnits':root_max,'correctedGuidePoints':corrected,
                'maximumGuideCorrectionMeters':largest_push,'allAuthorPointsMapped':True,
                'sourcePositionsSha256':hashlib.sha256(old.astype(np.float32).tobytes()).hexdigest()}


def _black_author_material(bpy,new_groups,curve_objects):
    originals={m for obj in curve_objects for m in obj.data.materials if m}
    copies=[]
    for original in originals:
        copy=original.copy();copy.name='R11 donor black / '+original.name;changed=[]
        if copy.use_nodes:
            for node in copy.node_tree.nodes:
                if node.type=='GROUP' and node.node_tree and node.node_tree.name.startswith('Hair shader EEVEE'):
                    for key,value in [('Color 1',(.012,.009,.0065,1.)),('Color 2',(.0028,.0022,.0018,1.))]:
                        socket=node.inputs.get(key)
                        if socket is None or socket.is_linked:raise RuntimeError('Unexpected author hair colour interface')
                        changed.append({'input':key,'before':list(socket.default_value),'after':list(value)})
                        socket.default_value=value
        if not changed:raise RuntimeError('Could not identify native author colour controls; do not substitute an unverified shader')
        for obj in curve_objects:
            for i,m in enumerate(obj.data.materials):
                if m==original:obj.data.materials[i]=copy
        for group in new_groups:
            for node in group.nodes:
                for socket in node.inputs:
                    if socket.bl_idname=='NodeSocketMaterial' and socket.default_value==original:
                        socket.default_value=copy
        copies.append({'original':original.name,'new':copy.name,'changed':changed,
                       'authorOpticsPreserved':True,'newImages':False})
    return copies


def build_donor_adult_groom(candidate_collection,head_object,female_to_target_matrix,*,expected_source_relative):
    import bpy,numpy as np
    from mathutils import Matrix
    P=project_path();source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    donor=P/SOURCE_REL
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Explicit independent background candidate required')
    if _sha(donor)!=SOURCE_SHA:raise RuntimeError('Official licensed donor changed')
    rig=bpy.data.objects.get('Character_Rig')
    if rig is None or rig.data.pose_position!='REST':raise RuntimeError('Caller must temporarily use neutral REST')
    if isinstance(head_object,str):head_object=bpy.data.objects.get(head_object)
    if isinstance(candidate_collection,str):candidate_collection=bpy.data.collections.get(candidate_collection)
    if head_object is None or head_object.name!='R11 Female head study' or candidate_collection is None:
        raise RuntimeError('Actual adult head and local candidate collection required')
    T=np.asarray(female_to_target_matrix,dtype=float);s=1.53119993/1.6414071321487427
    expected=np.diag([s,s,s,1.]);expected[1,3]=.030
    if T.shape!=(4,4) or not np.allclose(T,expected,atol=1e-7):raise RuntimeError('Unexpected adult transform')
    if any(bpy.data.objects.get(n) for n in DONOR_NAMES) or any(o.name.startswith('R11 Donor /') for o in bpy.data.objects):
        raise RuntimeError('Preserve prior donor import instead of ambiguous re-import')
    source_sha=_sha(source);bpy.context.view_layer.update()
    prior_data={n:hashlib.sha256(_coords(bpy.data.objects[n]).tobytes()).hexdigest() for n in OLD_GROOM if bpy.data.objects.get(n)}
    old_visibility={n:{'hideRender':bpy.data.objects[n].hide_render,'hideViewport':bpy.data.objects[n].hide_viewport,
                       'hideSet':bpy.data.objects[n].hide_get()} for n in prior_data}
    head_hash=hashlib.sha256(_coords(head_object).tobytes()).hexdigest()
    scalp,skin,head_report=_head_trees(head_object)
    before_objects=set(bpy.data.objects);before_groups=set(bpy.data.node_groups)
    with bpy.data.libraries.load(str(donor),link=False) as (available,requested):
        requested.objects=list(DONOR_NAMES)
    imported=dict(zip(DONOR_NAMES,requested.objects))
    if any(v is None for v in imported.values()):raise RuntimeError('Donor objects missing')
    new_groups=[g for g in bpy.data.node_groups if g not in before_groups]
    growth=imported[DONOR_NAMES[0]];curves=[imported[n] for n in DONOR_NAMES[1:]]
    if growth.type!='MESH' or any(o.type!='CURVES' for o in curves):raise RuntimeError('Unexpected donor types')
    if [len(o.data.curves) for o in curves]!=[346,30]:raise RuntimeError('Unexpected actual author guide counts')
    # Keep donor coordinates and all GN distance/density units. Only the shared
    # object transform scales their metres, radii, wavelength and scalp area.
    M=T.copy();M[:3,:3]*=.1
    original_growth=_coords(growth).astype(float);growth.data.calc_loop_triangles()
    triangles=np.array([tuple(t.vertices) for t in growth.data.loop_triangles],dtype=np.int32)
    target_root=bpy.data.objects.get('R06 attached scalp root substrate')
    if target_root is None or not target_root.get('R11_adult_groom_fit'):
        raise RuntimeError('Verified attached target root domain required; its old opaque render surface stays hidden')
    mapped_growth,normals,donor_center,growth_report=_fit_growth(original_growth,triangles,scalp,T,M,target_root)
    correspondence=SurfaceMap(original_growth,mapped_growth,triangles,donor_center,normals)
    author_inputs,author_input_report=_author_visible_inputs(P,curves)
    mapped={};guide_reports=[]
    for obj in curves:
        positions,row=_map_guides(obj,correspondence,M,skin,author_inputs[obj]);mapped[obj]=positions;guide_reports.append(row)
    # Preserve stored author data as internal history; all mutations are copies.
    retained={}
    for original_name,obj in imported.items():
        obj.data.use_fake_user=True;retained[original_name]=obj.data.name
        obj.data=obj.data.copy();obj.data.name='R11 fitted donor / '+original_name
        if obj.type=='MESH':obj.data.vertices.foreach_set('co',mapped_growth.astype(np.float32).ravel());obj.data.update()
        else:obj.data.position_data.foreach_set('vector',mapped[obj].astype(np.float32).ravel());obj.data.update_tag()
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
        candidate_collection.objects.link(obj)
        obj.name='R11 Donor / '+original_name
        # Disable only the original rest-surface deform: guide data has already
        # been mapped with the growth mesh. Its interpolation/style GN survives.
        for mod in obj.modifiers:
            if mod.type=='NODES' and mod.node_group and mod.node_group.name.startswith('Surface Deform'):
                mod.show_viewport=False;mod.show_render=False
        obj.parent=head_object;bpy.context.view_layer.update()
        obj.matrix_parent_inverse=head_object.matrix_world.inverted();obj.matrix_basis=Matrix(M.tolist())
        obj.hide_set(False);obj.hide_viewport=False;obj.hide_render=(obj==growth)
        obj['R11_donor_source']=SOURCE_REL;obj['R11_donor_author']='Daniel Bystedt'
        obj['R11_donor_license']='CC BY-SA; version unspecified by official evidence'
        obj['R11_donor_stage']='Actual head fit study; twin-tail restyling and fringe cut still pending'
    bpy.context.view_layer.update()
    for obj in imported.values():
        err=float(np.max(np.abs(np.asarray(obj.matrix_world)-M)))
        if err>2e-7:raise RuntimeError('Donor parent changed shared unit transform '+obj.name+' '+str(err))
    material_report=_black_author_material(bpy,new_groups,curves)
    for name in old_visibility:
        obj=bpy.data.objects[name];obj.hide_render=True;obj.hide_viewport=True;obj.hide_set(True)
    bpy.context.view_layer.update()
    evaluated=[];deps=bpy.context.evaluated_depsgraph_get()
    for obj in curves:
        ev=obj.evaluated_get(deps)
        row={'name':obj.name,'evaluatedType':ev.type,'modifiers':[{'name':m.name,'type':m.type,'showRender':m.show_render,
              'nodeGroup':m.node_group.name if m.type=='NODES' and m.node_group else None} for m in obj.modifiers]}
        if ev.type=='CURVES':
            row.update(curves=len(ev.data.curves),points=len(ev.data.position_data))
            if row['curves']>250000 or row['points']>18000000:raise RuntimeError('Author GN exceeds evaluated memory budget')
        evaluated.append(row)
    for name,digest in prior_data.items():
        if hashlib.sha256(_coords(bpy.data.objects[name]).tobytes()).hexdigest()!=digest:raise RuntimeError('Old groom geometry changed')
    if hashlib.sha256(_coords(head_object).tobytes()).hexdigest()!=head_hash:raise RuntimeError('Head geometry changed')
    if _sha(source)!=source_sha or _sha(donor)!=SOURCE_SHA:raise RuntimeError('Input file changed')
    return {'schemaVersion':1,'source':expected_source_relative,'sourceSha256':source_sha,
            'helperSha256':_sha(__file__),'donor':{'path':SOURCE_REL,'sha256':SOURCE_SHA,'author':'Daniel Bystedt',
              'license':'CC BY-SA (version unspecified in official page/embedded credits)',
              'url':'https://www.blender.org/download/demo-files/#hair','actualGeometryAndNodesUsed':True},
            'stage':'donor-head-fit study; not finished twin-tail style','head':head_report,'objectTransform':M.tolist(),
            'growth':growth_report,'authorInputPreparation':author_input_report,'guides':guide_reports,'materials':material_report,'evaluated':evaluated,
            'retainedAuthorData':retained,'hiddenOldObjects':old_visibility,'oldGeometryUnchanged':True,
            'headGeometryUnchanged':True,'savedBlend':False,'appearanceAccepted':False,
            'limits':['Author head was not imported as a character replacement; only hair surface/curves/nodes are used.',
                      'This full donor fit intentionally precedes actual twin-tail gathering/fringe cutting; it is not a final hairstyle.',
                      'Only stored guides have target skin clearance checks; evaluated GN/interpolated hair requires actual visual review.',
                      'Original growth UV, two islands and density/clump/noise/profile system are preserved; author Surface Deform is disabled after explicit remapping.',
                      'Static head parenting only; root attachment under deformation, hair dynamics, realtime cards and VRM remain pending.',
                      'Redistribution of adapted donor hair must retain Daniel Bystedt attribution, official source and applicable CC BY-SA terms; version is not invented.']}


if __name__=='__main__':print('Loaded only. Explicit build_donor_adult_groom(...) in an independent neutral REST source; caller owns save/render.')
