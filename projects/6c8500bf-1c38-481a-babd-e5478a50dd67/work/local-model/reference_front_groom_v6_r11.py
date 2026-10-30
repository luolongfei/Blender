"""Rebuild the front of the actual v5 groom around the user's anime design."""
import hashlib,heapq,json,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
ROLE='R11 groom role';GROUP='R11 groom interpolation group';ISLAND='R11 original scalp island';DENSITY='R11 reference scalp density'
WISPS='R11 Donor / reference face wisps'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def named(tree,name,data_type='INT'):
    n=tree.nodes.new('GeometryNodeInputNamedAttribute');n.data_type=data_type;n.inputs['Name'].default_value=name;return n.outputs['Attribute']
def replace_link(tree,source,dest):
    for link in list(dest.links):tree.links.remove(link)
    tree.links.new(source,dest)
def number(tree,island,role):
    m=tree.nodes.new('ShaderNodeMath');m.operation='MULTIPLY';m.inputs[1].default_value=256.;tree.links.new(island,m.inputs[0])
    a=tree.nodes.new('ShaderNodeMath');a.operation='ADD';tree.links.new(m.outputs[0],a.inputs[0]);tree.links.new(role,a.inputs[1]);return a.outputs[0]

def coverage_mask(obj):
    original=obj.data;original.use_fake_user=True;obj.data=original.copy();obj.data.name='R11 reference continuous scalp density'
    mesh=obj.data;local=np.array([tuple(v.co) for v in mesh.vertices]);matrix=np.asarray(obj.matrix_world);world=local@matrix[:3,:3].T+matrix[:3,3]
    counts={};adj=[[] for _ in world]
    for p in mesh.polygons:
        ids=list(p.vertices)
        for i,a in enumerate(ids):key=tuple(sorted((int(a),int(ids[(i+1)%len(ids)]))));counts[key]=counts.get(key,0)+1
    for edge in mesh.edges:
        a,b=map(int,edge.vertices);length=float(np.linalg.norm(world[a]-world[b]));adj[a].append((b,length));adj[b].append((a,length))
    boundary={v for pair,count in counts.items() if count==1 for v in pair}
    # Feather the real lower/temple outline. Do not re-create the donor's
    # unrelated crown parting by fading its upper disconnected island seam.
    seeds=[i for i in boundary if world[i,2]<1.487]
    if not seeds:raise RuntimeError('No actual scalp outline for reference density feather')
    distance=np.full(len(world),np.inf);queue=[]
    for i in seeds:distance[i]=0.;heapq.heappush(queue,(0.,i))
    while queue:
        dist,i=heapq.heappop(queue)
        if dist!=distance[i]:continue
        for j,length in adj[i]:
            new=dist+length
            if new<distance[j]:distance[j]=new;heapq.heappush(queue,(new,j))
    x=np.clip(distance/.0045,0.,1.);density=(.08+.92*x*x*(3.-2.*x)).astype(np.float32)
    attr=mesh.attributes.new(DENSITY,'FLOAT','POINT');attr.data.foreach_set('value',density);mesh.update()
    mesh.calc_loop_triangles();tri=np.array([tuple(t.vertices) for t in mesh.loop_triangles]);p=local[tri]
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5;weighted=float(np.sum(area*density[tri].mean(1)))
    if not .1<weighted<30.:raise RuntimeError('Unexpected actual donor-unit scalp area '+str(weighted))
    amount=26000./weighted
    return amount,{'attribute':DENSITY,'actualBoundaryVertices':len(boundary),'featherSeedVertices':len(seeds),'featherWidthMeters':.0045,
        'minimumDensity':float(density.min()),'weightedAreaDonorSquareMeters':weighted,'densityPerDonorSquareMeter':amount,'targetDistributedCurvesApproximate':26000,
        'originalDensityMaskRetainedInDisconnectedNodes':True,'retainedOriginalMesh':original.name,'scalpVerticesUnchanged':True}

def make_wisps(head,obj,layout,ray,skin,eyes):
    if bpy.data.objects.get(WISPS):raise RuntimeError('Preserve existing reference face wisps')
    paths=[];radii=[];roles=[];rng=np.random.default_rng(71028);rows=[]
    for side in (-1,1):
        for kind,count in enumerate((52,18,9)):
            base=layout['wisp_center'](side,kind,ray);side_paths=[]
            for k in range(count):
                t=np.linspace(0.,1.,len(base));lateral=rng.normal(0.,.00038);depth=rng.normal(0.,.00025)
                # A narrow, coherent long strand and two finer short companions.
                q=base.copy();envelope=np.sin(np.pi*t)**.65;q[:,0]+=lateral*(.6+.4*envelope);q[:,1]+=depth*(.6+.4*envelope)
                q[:,2]+=(rng.random()-.5)*.0025*layout['smooth']((t-.65)/.35)
                for j,point in enumerate(q):
                    hit,n,_,dist=skin.find_nearest(Vector(point));signed=float((Vector(point)-hit).dot(n))
                    if signed<.00025:
                        push=.00028-signed
                        if push>.003:raise RuntimeError('Face wisp requires excessive skin correction')
                        q[j]+=np.asarray(n)*push
                gap=min(segment_eye(q,c,r) for c,r in eyes)
                if gap<.001:raise RuntimeError('Face wisp enters actual eye envelope')
                paths.append(q);radii.extend((.000027*(1.-.94*layout['smooth']((t-.56)/.44))).tolist());roles.append(70+kind);side_paths.append(q)
            rows.append({'side':side,'kind':kind,'curves':count,'root':base[0].tolist(),'tip':base[-1].tolist(),'widthSigmaMeters':.00038})
    data=bpy.data.hair_curves.new('R11 reference fine face strands');data.add_curves([len(q) for q in paths]);world=np.concatenate(paths);inv=np.asarray(obj.matrix_world.inverted());local=world@inv[:3,:3].T+inv[:3,3];data.position_data.foreach_set('vector',local.astype(np.float32).ravel())
    scale=float(np.linalg.svd(np.asarray(obj.matrix_world)[:3,:3],compute_uv=False).max());attr=data.attributes.get('radius') or data.attributes.new('radius','FLOAT','POINT');attr.data.foreach_set('value',(np.asarray(radii)/scale).astype(np.float32))
    for material in obj.data.materials:data.materials.append(material)
    new=bpy.data.objects.new(WISPS,data);obj.users_collection[0].objects.link(new);new.parent=obj.parent;new.matrix_world=obj.matrix_world.copy()
    new['R11_design_reference']='User original portrait 2026-10-28; each side one long fine strand and two shorter companions'
    return {'object':new.name,'curves':len(paths),'points':len(world),'worldRadiusMaxMeters':max(radii),'groups':rows,'dynamicsAccepted':False}

def segment_eye(path,center,radius):
    a=path[:-1];d=path[1:]-a;t=np.clip(np.sum((center-a)*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-15),0.,1.)
    return float(np.linalg.norm(a+t[:,None]*d-center,axis=1).min()-radius)

def apply(*,expected_source_relative):
    source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Explicit independent candidate source required')
    obj=bpy.data.objects['R11 Donor / long hair main']
    if not obj.get('R11_donor_gn_finish_v5') or obj.get('R11_reference_front_v6'):raise RuntimeError('Fresh actual v5-derived source required')
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Caller must enter neutral REST')
    if sha(HERE/'reference_fringe_layout_v6_r11.py')!='e6b1879cc9773e73f277cdc30ca374303f55899cf7377226b22f03ff8437a8b1':raise RuntimeError('Reference layout changed after offline geometry review')
    layout=runpy.run_path(str(HERE/'reference_fringe_layout_v6_r11.py'));fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'));head=bpy.data.objects['R11 Female head study'];_,skin,head_report=fit['_head_trees'](head)
    center=layout['CENTER']
    def ray(direction):
        d=Vector(direction);hit,normal,_,distance=skin.ray_cast(Vector(center)+d*.30,-d,.60)
        return (None,None) if hit is None else (np.asarray(hit),np.asarray(normal))
    def front(x,z):
        hit,normal,_,distance=skin.ray_cast(Vector((x,-.60,z)),Vector((0.,1.,0.)),1.20)
        return (None,None) if hit is None else (np.asarray(hit),np.asarray(normal))
    eyes=[]
    for side in ('L','R'):
        eye=bpy.data.objects['R11 Adult sclera '+side];q=fit['_world'](fit['_coords'](eye),eye.matrix_world);eyes.append(((q.min(0)+q.max(0))*.5,float((q.max(0)-q.min(0)).max())*.5))
    original=obj.data;old_local=fit['_coords'](obj);old_world=fit['_world'](old_local,obj.matrix_world);starts=np.array([c.first_point_index for c in original.curves],np.int32);lengths=np.array([c.points_length for c in original.curves],np.int32)
    world=old_world.copy();roles=np.zeros(len(starts),np.int32);rows=[];max_push=0.
    for i,(start,length) in enumerate(zip(starts,lengths)):
        path=old_world[start:start+length];kind=layout['role'](path[0])
        if kind:
            new,row=layout['front_path'](path[0],i,int(length),ray,front)
            for j in range(1,len(new)):
                hit,n,_,distance=skin.find_nearest(Vector(new[j]));signed=float((Vector(new[j])-hit).dot(n))
                if signed<.00025:
                    push=.00028-signed
                    if push>.004:raise RuntimeError('Reference front guide requires excessive skin correction '+str((i,j,push)))
                    new[j]+=np.asarray(n)*push;max_push=max(max_push,push)
            eye_gap=min(segment_eye(new,c,r) for c,r in eyes)
            if eye_gap<.001:raise RuntimeError('Reference fringe enters actual eye envelope '+str((i,eye_gap)))
            world[start:start+length]=new;roles[i]=kind;row.update(guide=i,eyeEnvelopeGapMeters=eye_gap);rows.append(row)
        else:roles[i]=40 if path[-1,0]>0 else 41
    original.use_fake_user=True;obj.data=original.copy();obj.data.name='R11 original-reference continuous crown fringe'
    inverse=np.asarray(obj.matrix_world.inverted());local=fit['_world'](world,inverse).astype(np.float32);local[starts]=old_local[starts]
    obj.data.position_data.foreach_set('vector',local.ravel());obj.data.attributes[ROLE].data.foreach_set('value',roles);obj.data.update_tag()
    # Existing carriers map each original scalp island to the updated roles.
    carriers=[]
    for carrier in sorted((o for o in bpy.data.objects if o.get('R11_groom_role_root_source')==obj.name),key=lambda o:o.name):
        old=carrier.data;positions=np.asarray([tuple(v.co) for v in old.vertices]);dist=np.linalg.norm(positions[:,None,:]-old_local[starts][None,:,:],axis=2);indices=dist.argmin(1)
        if float(dist[np.arange(len(indices)),indices].max())>2e-5:raise RuntimeError('Actual root carrier no longer matches source guides')
        old.use_fake_user=True;carrier.data=old.copy();carrier.data.name='R11 reference lock roles / '+old.name;carrier.data.attributes[ROLE].data.foreach_set('value',roles[indices]);carrier.data.update();carriers.append({'object':carrier.name,'points':len(indices),'newRoles':sorted(map(int,np.unique(roles[indices])))})
    modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render);old_tree=modifier.node_group;old_tree.use_fake_user=True;tree=old_tree.copy();tree.name='R11 original-reference continuous front groom';modifier.node_group=tree
    interpolation=tree.nodes['Group.005'];old_child=interpolation.node_tree;old_child.use_fake_user=True;child=old_child.copy();child.name='R11 reference lock-aware interpolation';interpolation.node_tree=child;core=child.nodes['Interpolate Curves']
    grouped=core.inputs['Guide Curves'].links[0].from_node
    if grouped.bl_idname!='GeometryNodeStoreNamedAttribute' or grouped.inputs['Name'].default_value!=GROUP:raise RuntimeError('Measured v4 grouped-guide storage changed')
    replace_link(child,number(child,named(child,ISLAND),named(child,ROLE)),grouped.inputs['Value'])
    point_add=core.inputs['Point Group ID'].links[0].from_node
    point_role=point_add.inputs[1].links[0].from_socket
    replace_link(child,number(child,child.nodes['Reroute.059'].outputs[0],point_role),core.inputs['Point Group ID'])
    scalp=bpy.data.objects['R11 Donor / long hair growth mesh'];density,mask=coverage_mask(scalp)
    replace_link(tree,named(tree,DENSITY,'FLOAT'),interpolation.inputs['Density Mask']);interpolation.inputs['Density'].default_value=density
    # Retire the old loose rolled accent branch; new fine face strands are
    # authored from the reference rather than duplicate the donor's loose curls.
    removed=[]
    for link in list(tree.nodes['Join Geometry.001'].inputs['Geometry'].links):
        if link.from_node.name=='Group.012':removed.append(link.from_node.name);tree.links.remove(link)
    if len(removed)!=1:raise RuntimeError('Unexpected legacy loose accent branch')
    secondary=bpy.data.objects['R11 Donor / long hair strands'];secondary.hide_render=True;secondary.hide_set(True)
    wisps=make_wisps(head,obj,layout,ray,skin,eyes)
    tree.update_tag();child.update_tag();obj.update_tag();bpy.context.view_layer.update()
    obj['R11_reference_front_v6']='Original anime design: continuous crown-to-fringe flow, unequal tapered locks, fine long face-side strands'
    return {'source':expected_source_relative,'sourceSha256':sha(source),'helperSha256':sha(__file__),'layoutHelperSha256':sha(HERE/'reference_fringe_layout_v6_r11.py'),
        'reference':'references/user-original-portrait-2026-10-28.png','head':head_report,'rebuiltFrontGuides':len(rows),'frontGuides':rows,
        'maximumRawSkinCorrectionMeters':max_push,'originalRootsPreservedExact':True,'nativeTailGeometryAndPinsEdited':False,
        'roleCounts':{str(k):int(np.sum(roles==k)) for k in np.unique(roles)},'roleRule':'native island * 256 + unequal reference lock or left/right gather',
        'roleCarriers':carriers,'densityCoverage':mask,'retiredLooseAccentBranch':removed,'fineFaceWisps':wisps,'retainedMainGuideData':original.name,
        'appearanceAccepted':False,'stage':'Reference-led v6 front hair candidate; actual dense geometry and images required.'}

if __name__=='__main__':print('Explicit independent candidate apply only.')
