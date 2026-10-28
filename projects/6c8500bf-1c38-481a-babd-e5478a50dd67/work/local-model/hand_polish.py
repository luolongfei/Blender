"""Original continuous hands fitted to the existing character's finger bones.

No donor geometry or rig changes.  apply(collection, rig, mats) builds replacement
hands first, validates topology/weights, and only then removes exact original names.
Designed for use on a saved working copy; caller remains responsible for its backup.
"""
import bpy
import bmesh
import math
import json
from mathutils import Vector, Quaternion
from mathutils.bvhtree import BVHTree

VERSION=1
DIGITS=(
    ("Index",-.020,.058,.0060),
    ("Middle",-.005,.064,.0063),
    ("Ring",.010,.058,.0058),
    ("Little",.024,.045,.0049),
)


def _smooth(a,b,x):
    f=max(0.0,min(1.0,(x-a)/(b-a)))
    return f*f*(3-2*f)


def _lerp_profile(table,x):
    if x<=table[0][0]:
        return table[0][1]
    for i,(a,b) in enumerate(zip(table,table[1:])):
        if a[0]<=x<=b[0]:
            t=_smooth(a[0],b[0],x)
            return a[1]*(1-t)+b[1]*t
    return table[-1][1]


class _Geometry:
    def __init__(self):
        self.verts=[]
        self.faces=[]

    def ellipsoid(self,center,scale,nu=28,nv=16):
        offset=len(self.verts)
        self.verts.append((center[0],center[1],center[2]+scale[2]))
        for j in range(1,nv):
            lat=math.pi*j/nv
            for i in range(nu):
                th=math.tau*i/nu
                self.verts.append((center[0]+scale[0]*math.sin(lat)*math.cos(th),
                                   center[1]+scale[1]*math.sin(lat)*math.sin(th),
                                   center[2]+scale[2]*math.cos(lat)))
        bottom=len(self.verts)
        self.verts.append((center[0],center[1],center[2]-scale[2]))
        for i in range(nu):
            self.faces.append((offset,offset+1+i,offset+1+(i+1)%nu))
        for j in range(nv-2):
            for i in range(nu):
                n=(i+1)%nu
                self.faces.append((offset+1+j*nu+i,offset+1+(j+1)*nu+i,
                                   offset+1+(j+1)*nu+n,offset+1+j*nu+n))
        last=offset+1+(nv-2)*nu
        for i in range(nu):
            self.faces.append((bottom,last+(i+1)%nu,last+i))

    def loft(self,centers,yr,zr,nu=24):
        start=len(self.verts)
        for j,p in enumerate(centers):
            if j==0:
                direction=Vector(centers[1])-Vector(p)
            elif j==len(centers)-1:
                direction=Vector(p)-Vector(centers[j-1])
            else:
                direction=Vector(centers[j+1])-Vector(centers[j-1])
            direction.normalize()
            vertical=Vector((0,0,1))
            across=vertical.cross(direction).normalized()
            vertical=direction.cross(across).normalized()
            for i in range(nu):
                th=math.tau*i/nu
                co=Vector(p)+across*yr[j]*math.cos(th)+vertical*zr[j]*math.sin(th)
                self.verts.append(tuple(co))
        for j in range(len(centers)-1):
            for i in range(nu):
                n=(i+1)%nu
                self.faces.append((start+j*nu+i,start+j*nu+n,
                                   start+(j+1)*nu+n,start+(j+1)*nu+i))
        self.faces.append(tuple(reversed([start+i for i in range(nu)])))
        self.faces.append(tuple(start+(len(centers)-1)*nu+i for i in range(nu)))


def _finger_point(index,name,y,length,t):
    return Vector((.672+length*t,y+(index-1.2)*.003*t,1.235-.003*t*t))


def _thumb_point(t):
    points=[
        (0, (.625,-.015,1.230)),(.20,(.633,-.022,1.231)),
        (.47,(.647,-.037,1.230)),(.73,(.661,-.047,1.2285)),
        (1,(.677,-.054,1.228))
    ]
    if t<=0:
        return Vector(points[0][1])
    if t>=1:
        end=Vector(points[-1][1])
        direction=(end-Vector(points[-2][1])).normalized()
        return end+direction*((t-1)*.060)
    for i,(a,b) in enumerate(zip(points,points[1:])):
        if a[0]<=t<=b[0]:
            f=(t-a[0])/(b[0]-a[0])
            before=points[max(0,i-1)]
            after=points[min(len(points)-1,i+2)]
            va,vb,vp,vn=map(Vector,(a[1],b[1],before[1],after[1]))
            h=b[0]-a[0]
            ma=(vb-vp)/(b[0]-before[0])
            mb=(vn-va)/(after[0]-a[0])
            return ((2*f**3-3*f*f+1)*va+(f**3-2*f*f+f)*h*ma+
                    (-2*f**3+3*f*f)*vb+(f**3-f*f)*h*mb)


def _original_geometry():
    geo=_Geometry()
    # A shaped palm: tapered wrist, broad knuckles, gently arched back and pads.
    profile=[
        (.597,.0174,.0120),(.607,.0197,.0129),(.620,.0238,.0135),
        (.635,.0274,.0136),(.649,.0292,.0128),(.662,.0294,.0109),
        (.672,.0278,.0084),(.680,.0200,.0043)
    ]
    centers=[];ys=[];zs=[]
    for j in range(49):
        x=.597+(.680-.597)*j/48
        yr=_lerp_profile([(a,b) for a,b,c in profile],x)
        zr=_lerp_profile([(a,c) for a,b,c in profile],x)
        centers.append((x,.0003+.0008*math.sin(math.pi*j/48),
                        1.235+.0010*math.sin(math.pi*j/48)))
        ys.append(yr);zs.append(zr)
    geo.loft(centers,ys,zs,40)
    geo.ellipsoid((.638,-.0160,1.2305),(.024,.0130,.0093))
    geo.ellipsoid((.647,.0165,1.2307),(.024,.0110,.0085))
    # Fingers extend inside the palm before exiting, so the remesh creates natural webs.
    for index,(name,y,length,r) in enumerate(DIGITS):
        ts=[-.23+j*(1.255/48) for j in range(49)]
        centers=[];ys=[];zs=[]
        for t in ts:
            p=_finger_point(index,name,y,length,t)
            factor=_lerp_profile([
                (-.23,1.12),(-.10,1.13),(0,1.04),(.12,.99),(.27,.93),
                (.40,1.015),(.48,.98),(.60,.87),(.73,.90),
                (.80,.84),(.90,.78),(.94,.74)
            ],t)
            if t>=.94:
                cap=max(0,min(1,(t-.94)/.085))
                factor=max(.025,.74*math.sqrt(1-cap*cap))
            centers.append(p)
            ys.append(r*factor)
            zs.append(r*factor*.82)
        geo.loft(centers,ys,zs,24)
        # The loft itself carries the knuckle volume; extra spheres would create
        # isolated-looking nodules rather than the reference's soft anatomy.
    # Thin soft webs differ slightly along the knuckle arc.
    for x,y,r in ((.676,-.0125,.0051),(.674,.0025,.0050),(.671,.0170,.0046)):
        geo.ellipsoid((x,y,1.2345),(.0081,r,.0068),24,12)
    centers=[];ys=[];zs=[]
    for j in range(48):
        t=1.035*j/47
        centers.append(_thumb_point(t))
        radius=_lerp_profile([(0,.0110),(.24,.0094),(.42,.0085),(.50,.0086),
                              (.64,.0072),(.75,.00725),(.86,.0061),
                              (.90,.0056)],t)
        if t>=.90:
            cap=max(0,min(1,(t-.90)/.135))
            radius=max(.00018,.0056*math.sqrt(1-cap*cap))
        ys.append(radius)
        zs.append(radius*.85)
    geo.loft(centers,ys,zs,26)
    return geo


def _active(obj):
    if bpy.context.object and bpy.context.object.mode!="OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active=obj


def _make_skin(col,side,material):
    geo=_original_geometry()
    sign=1 if side=="L" else -1
    verts=[(sign*x,y,z) for x,y,z in geo.verts]
    faces=[tuple(reversed(f)) if sign<0 else f for f in geo.faces]
    mesh=bpy.data.meshes.new("Continuous hand "+side+" topology")
    mesh.from_pydata(verts,[],faces);mesh.update()
    obj=bpy.data.objects.new("Hand "+side+" · continuous anatomy",mesh)
    col.objects.link(obj)
    _tag(obj,side,"hand")
    mesh.materials.append(material)
    _active(obj)
    remesh=obj.modifiers.new("Union original palm fingers webs","REMESH")
    remesh.mode="VOXEL"
    remesh.voxel_size=.00145
    remesh.use_smooth_shade=True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    smooth=obj.modifiers.new("Relax union surface","SMOOTH")
    smooth.factor=.42
    smooth.iterations=5
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    if len(obj.data.vertices)>5900:
        decimate=obj.modifiers.new("Balanced final hand topology","DECIMATE")
        decimate.decimate_type="COLLAPSE"
        decimate.ratio=min(1.0,5750/len(obj.data.vertices))
        decimate.use_collapse_triangulate=False
        bpy.ops.object.modifier_apply(modifier=decimate.name)
    bm=bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(obj.data);bm.free()
    obj.data.update()
    for p in obj.data.polygons:
        p.use_smooth=True
    # A genuine non-overlapping UV layout for portable materials.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66),island_margin=.015)
    bpy.ops.object.mode_set(mode="OBJECT")
    if obj.data.uv_layers.active:
        obj.data.uv_layers.active.name="UVMap"
    return obj


def _tag(obj,side,part,finger=None):
    obj["part"]=part
    obj["deform_role"]="hand_polished" if part=="hand" else "nail"
    obj["side"]=side
    obj["original_mesh"]=True
    obj["hand_polish_version"]=VERSION
    if finger:
        obj["finger"]=finger


def _thumb_weights(point,side):
    controls=[Vector((.633,-.022,1.231)),Vector((.647,-.037,1.230)),
              Vector((.661,-.047,1.2285)),Vector((.677,-.054,1.228))]
    lengths=[(b-a).length for a,b in zip(controls,controls[1:])]
    nearest=None
    acc=0
    for a,b,length in zip(controls,controls[1:],lengths):
        t=max(0,min(1,(point-a).dot(b-a)/(length*length)))
        p=a+(b-a)*t
        d=(point-p).length_squared
        if nearest is None or d<nearest[0]:
            nearest=(d,acc+t*length)
        acc+=length
    s=nearest[1]
    names=[side+"ThumbMetacarpal",side+"ThumbProximal",side+"ThumbDistal"]
    if s<lengths[0]+.0038:
        f=_smooth(lengths[0]-.0042,lengths[0]+.0038,s)
        return {names[0]:1-f,names[1]:f}
    f=_smooth(sum(lengths[:2])-.0038,sum(lengths[:2])+.0038,s)
    return {names[1]:1-f,names[2]:f}


def _weights(co,label):
    side="left" if label=="L" else "right"
    x,y,z=abs(co.x),co.y,co.z
    hand=side+"Hand";forearm=side+"LowerArm"
    wrist=_smooth(.604,.625,x)
    base={forearm:1-wrist,hand:wrist}
    amount=_smooth(.657,.686,x)
    if amount>0:
        scores=[]
        for idx,(name,fy,length,r) in enumerate(DIGITS):
            t=(x-.672)/length
            center_y=fy+(idx-1.2)*.003*max(0,t)
            score=math.exp(-((y-center_y)/.0062)**2)
            scores.append((name,length,t,score))
        total=sum(a[3] for a in scores)
        if total>1e-14:
            result={k:v*(1-amount) for k,v in base.items()}
            for name,length,t,score in scores:
                share=amount*score/total
                if t<.57:
                    f=_smooth(.33,.52,t)
                    segments={"Proximal":1-f,"Intermediate":f}
                else:
                    f=_smooth(.64,.83,t)
                    segments={"Intermediate":1-f,"Distal":f}
                for seg,w in segments.items():
                    key=side+name+seg
                    result[key]=result.get(key,0)+share*w
            base=result
    thumb_amount=_smooth(.021,.036,-y)*(1-_smooth(.679,.691,x))
    if thumb_amount>0:
        result={k:v*(1-thumb_amount) for k,v in base.items()}
        for key,w in _thumb_weights(Vector((x,y,z)),side).items():
            result[key]=result.get(key,0)+thumb_amount*w
        base=result
    top=sorted(((k,v) for k,v in base.items() if v>1e-7),
               key=lambda item:item[1],reverse=True)[:4]
    total=sum(v for k,v in top)
    return {k:v/total for k,v in top}


def _bind(obj,rig,label,nail=None):
    groups={}
    side="left" if label=="L" else "right"
    for v in obj.data.vertices:
        weights={side+nail+"Distal":1.0} if nail else _weights(v.co,label)
        for name,value in weights.items():
            if name not in rig.data.bones:
                raise ValueError("Required existing bone missing: "+name)
            if name not in groups:
                groups[name]=obj.vertex_groups.new(name=name)
            groups[name].add([v.index],value,"REPLACE")
    obj.parent=rig
    modifier=obj.modifiers.new("Humanoid deformation","ARMATURE")
    modifier.object=rig
    modifier.use_deform_preserve_volume=True
    obj.modifiers.move(len(obj.modifiers)-1,0)
    if nail is None:
        finish=obj.modifiers.new("Smooth finger flexion","SUBSURF")
        finish.levels=1
        finish.render_levels=1


def _make_nails(skin,col,side,material):
    sign=1 if side=="L" else -1
    bvh=BVHTree.FromPolygons([v.co.copy() for v in skin.data.vertices],
                            [tuple(p.vertices) for p in skin.data.polygons])
    out=[]
    specs=[]
    for idx,(name,y,length,r) in enumerate(DIGITS):
        a=_finger_point(idx,name,y,length,.79)
        b=_finger_point(idx,name,y,length,.94)
        c=(a+b)/2
        axis=(b-a).normalized()
        specs.append((name,c,axis,.0076 if name!="Little" else .0062,r*.62))
    a=_thumb_point(.77);b=_thumb_point(.94)
    specs.append(("Thumb",(a+b)/2,(b-a).normalized(),.0073,.0037))
    for name,center,axis,length,width in specs:
        axis.z=0;axis.normalize()
        across=Vector((-axis.y,axis.x,0))
        verts=[];uv=[];faces=[]
        # Concentric rounded rectangle nail surface, with a soft oval cuticle.
        ns=32;nr=4
        samples=[(0,0,0)]
        for row in range(1,nr+1):
            radius=row/nr
            for i in range(ns):
                th=math.tau*i/ns
                cs,ss=math.cos(th),math.sin(th)
                u=math.copysign(abs(cs)**.72,cs)*length*.5*radius
                v=math.copysign(abs(ss)**.86,ss)*width*radius
                samples.append((u,v,radius))
        for u,v,radius in samples:
            target=center+axis*u+across*v
            target.x*=sign
            p,n,face,dist=bvh.ray_cast(Vector((target.x,target.y,1.34)),
                                     Vector((0,0,-1)),.2)
            if p is None:
                raise ValueError("Nail projection missed hand: "+side+" "+name)
            if n.z<0:n=-n
            p+=n*(.00027+.00013*(1-radius*radius))
            verts.append(tuple(p))
            uv.append((.5+u/length,.5+v/(2*width)))
        for i in range(ns):
            f=(0,1+i,1+(i+1)%ns)
            faces.append(tuple(reversed(f)) if sign<0 else f)
        for j in range(nr-1):
            for i in range(ns):
                n=(i+1)%ns
                f=(1+j*ns+i,1+(j+1)*ns+i,1+(j+1)*ns+n,1+j*ns+n)
                faces.append(tuple(reversed(f)) if sign<0 else f)
        mesh=bpy.data.meshes.new("Nail "+side+" "+name+" surface")
        mesh.from_pydata(verts,[],faces);mesh.update()
        mesh.materials.append(material)
        layer=mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            poly.use_smooth=True
            for li in poly.loop_indices:
                layer.data[li].uv=uv[mesh.loops[li].vertex_index]
        obj=bpy.data.objects.new("Nail "+side+" "+name+" · polished",mesh)
        col.objects.link(obj);_tag(obj,side,"nail",name)
        thick=obj.modifiers.new("Nail plate thickness","SOLIDIFY")
        thick.thickness=.00020;thick.offset=-1
        out.append(obj)
    return out


def _validate_skin(obj):
    bm=bmesh.new();bm.from_mesh(obj.data)
    bad=sum(not e.is_manifold for e in bm.edges)
    components=0
    unseen=set(bm.verts)
    while unseen:
        components+=1;stack=[unseen.pop()]
        while stack:
            vertex=stack.pop()
            for e in vertex.link_edges:
                other=e.other_vert(vertex)
                if other in unseen:
                    unseen.remove(other);stack.append(other)
    bm.free()
    if bad or components!=1:
        raise ValueError("%s topology: %d non-manifold edges, %d components"%(obj.name,bad,components))
    for v in obj.data.vertices:
        if not 1<=len(v.groups)<=4 or abs(sum(g.weight for g in v.groups)-1)>1e-5:
            raise ValueError("Invalid normalized hand weights: "+obj.name)
    if len(obj.data.vertices)>6300:
        raise ValueError("Hand vertex budget exceeded: "+str(len(obj.data.vertices)))
    return {"object":obj.name,"vertices":len(obj.data.vertices),
            "faces":len(obj.data.polygons),"nonManifoldEdges":bad,
            "connectedComponents":components,"maxInfluences":max(len(v.groups) for v in obj.data.vertices)}


def apply(collection,rig,mats):
    """Replace exact original generated hand objects, returning the 12 new meshes.

    Requires skin and nail materials. Existing rig is used, never changed.
    New hand skin is continuous, manifold, UV unwrapped, and has <=4 normalized weights.
    """
    already=[o for o in collection.objects if o.get("hand_polish_version")==VERSION]
    if len(already)==12:
        return []
    old_names=[]
    for label in ("L","R"):
        old_names.append("Hand "+label+" · palm")
        for name in ("Index","Middle","Ring","Little","Thumb"):
            old_names.extend(("Hand "+label+" · "+name,"Nail "+label+" "+name))
    old=[o for o in collection.objects if o.name in old_names]
    if len(old)!=22:
        raise ValueError("Expected precisely 22 original hand meshes; found "+str(len(old)))
    if any(o.type!="MESH" for o in old):
        raise ValueError("Original hand object type changed; refusing replacement.")
    selected=[o.name for o in bpy.context.selected_objects]
    active=bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else None
    new=[];reports=[]
    try:
        for label in ("L","R"):
            skin=_make_skin(collection,label,mats["skin"])
            new.append(skin)
            _bind(skin,rig,label)
            nails=_make_nails(skin,collection,label,mats["nail"])
            new.extend(nails)
            for nail in nails:
                _bind(nail,rig,label,nail["finger"])
            reports.append(_validate_skin(skin))
        for obj in old:
            bpy.data.objects.remove(obj,do_unlink=True)
        collection["hand_polish_report"]=json.dumps(reports)
        return new
    except Exception:
        for obj in list(collection.objects):
            if obj not in already and obj.get("hand_polish_version")==VERSION:
                bpy.data.objects.remove(obj,do_unlink=True)
        raise
    finally:
        if bpy.context.object and bpy.context.object.mode!="OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for name in selected:
            obj=bpy.data.objects.get(name)
            if obj:
                obj.select_set(True)
        if active and bpy.data.objects.get(active):
            bpy.context.view_layer.objects.active=bpy.data.objects[active]
