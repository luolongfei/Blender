"""Replace floating original shoe detail with detail fitted to the edited upper.

Operates on the two original Shoe_Leather_Upper meshes in the supplied collection.
The upper, sole, heel, collar, toe-cap seam and their existing rig are preserved.
Returned objects are meshes. If rig is supplied they receive rigid foot weights.
"""
import bpy
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

VERSION=2
OLD_PREFIXES=(
    "Shoe_Quarter_Seam_","Shoe_Quarter_Stitch_","Shoe_Padded_Tongue_",
    "Shoe_Eyelet_Facing_","Shoe_Facing_Stitch_","Shoe_Eyelet_",
    "Shoe_Lace_Bottom_","Shoe_Cross_Lace_","Shoe_Lace_Knot_",
    "Shoe_Bow_Loop_","Shoe_Lace_End_","Shoe_Aglet_",
)


def _mesh(name,verts,faces,col,material,side,uv=None):
    data=bpy.data.meshes.new(name+"_mesh")
    data.from_pydata(verts,[],faces)
    data.update()
    if material:
        data.materials.append(material)
    for p in data.polygons:
        p.use_smooth=True
    if uv:
        layer=data.uv_layers.new(name="UVMap")
        for p in data.polygons:
            for li in p.loop_indices:
                layer.data[li].uv=uv[data.loops[li].vertex_index]
    obj=bpy.data.objects.new(name,data)
    col.objects.link(obj)
    obj["part"]="lower_body"
    obj["deform_role"]="shoe"
    obj["side"]=side
    obj["original_mesh"]=True
    obj["shoe_detail_refinement_version"]=VERSION
    return obj


def _tube(name,points,radius,col,material,side,cyclic=False,sides=8):
    pts=[Vector(p) for p in points]
    if cyclic and (pts[0]-pts[-1]).length<1e-7:
        pts.pop()
    verts,faces,uv=[],[],[]
    previous=None
    for i,p in enumerate(pts):
        if cyclic:
            tangent=(pts[(i+1)%len(pts)]-pts[(i-1)%len(pts)]).normalized()
        elif i==0:
            tangent=(pts[1]-p).normalized()
        elif i==len(pts)-1:
            tangent=(p-pts[-2]).normalized()
        else:
            tangent=(pts[i+1]-pts[i-1]).normalized()
        if previous is None:
            axis=Vector((0,0,1))
            if abs(axis.dot(tangent))>.92:
                axis=Vector((0,1,0))
            a=tangent.cross(axis).normalized()
        else:
            a=(previous-tangent*previous.dot(tangent)).normalized()
            if a.length<.1:
                a=tangent.cross(Vector((0,1,0))).normalized()
        b=tangent.cross(a).normalized()
        previous=a
        for j in range(sides):
            th=math.tau*j/sides
            verts.append(tuple(p+radius*(a*math.cos(th)+b*math.sin(th))))
            uv.append((j/sides,i/max(1,len(pts)-1)))
    for i in range(len(pts) if cyclic else len(pts)-1):
        ni=(i+1)%len(pts)
        for j in range(sides):
            faces.append((i*sides+j,i*sides+(j+1)%sides,
                          ni*sides+(j+1)%sides,ni*sides+j))
    if not cyclic:
        faces.append(tuple(reversed(range(sides))))
        faces.append(tuple((len(pts)-1)*sides+i for i in range(sides)))
    return _mesh(name,verts,faces,col,material,side,uv)


def _torus(name,point,normal,radius,wire,col,material,side):
    n=Vector(normal).normalized()
    a=n.cross(Vector((1,0,0))).normalized()
    if a.length<.1:
        a=n.cross(Vector((0,1,0))).normalized()
    b=n.cross(a).normalized()
    c=Vector(point)
    verts,faces,uv=[],[],[]
    nu,nv=20,8
    for j in range(nv):
        v=math.tau*j/nv
        for i in range(nu):
            u=math.tau*i/nu
            radial=a*math.cos(u)+b*math.sin(u)
            verts.append(tuple(c+radial*(radius+wire*math.cos(v))+n*wire*math.sin(v)))
            uv.append((i/nu,j/nv))
    for j in range(nv):
        for i in range(nu):
            faces.append((j*nu+i,j*nu+(i+1)%nu,
                          ((j+1)%nv)*nu+(i+1)%nu,((j+1)%nv)*nu+i))
    return _mesh(name,verts,faces,col,material,side,uv)


def _smooth_path(points,samples=8,closed=False):
    controls=[Vector(p) for p in points]
    if closed and (controls[0]-controls[-1]).length<1e-7:
        controls.pop()
    result=[]
    n=len(controls)
    for i in range(n if closed else n-1):
        p0=controls[(i-1)%n] if closed or i>0 else controls[i]
        p1=controls[i]
        p2=controls[(i+1)%n]
        p3=controls[(i+2)%n] if closed or i+2<n else p2
        for j in range(samples):
            t=j/samples
            result.append(.5*((2*p1)+(-p0+p2)*t+
                               (2*p0-5*p1+4*p2-p3)*t*t+
                               (-p0+3*p1-3*p2+p3)*t*t*t))
    if not closed:
        result.append(controls[-1])
    return result


class _Surface:
    def __init__(self,obj):
        # Geometry is in the original/rest object coordinate frame. Using data,
        # rather than a posed evaluated mesh, prevents double-pose deformation.
        verts=[v.co.copy() for v in obj.data.vertices]
        polys=[tuple(p.vertices) for p in obj.data.polygons]
        self.bvh=BVHTree.FromPolygons(verts,polys,all_triangles=False)
        self.cx=sum(v.x for v in verts)/len(verts)
        self.high=max(v.z for v in verts)+.12

    def top(self,x,y,lift=.001):
        p,n,face,dist=self.bvh.ray_cast(Vector((x,y,self.high)),Vector((0,0,-1)),.5)
        if p is None:
            raise ValueError("Shoe vamp ray missed at %.6f, %.6f; upper geometry changed."%(x,y))
        if n.z<0:
            n=-n
        return p+n*lift,n

    def near(self,p,lift=.001):
        point,normal,face,dist=self.bvh.find_nearest(Vector(p))
        if point is None or dist>.025:
            raise ValueError("Shoe side projection too far from actual leather surface.")
        outward=Vector((point.x-self.cx,point.y+.025,max(.008,point.z-.035)))
        if normal.dot(outward)<0:
            normal=-normal
        return point+normal*lift,normal


def _patch(name,surface,col,mat,side,width_inner,width_outer,y0,y1,
           lift=.001,sign=1,nx=6,ny=22):
    verts,faces,uv=[],[],[]
    for j in range(ny+1):
        v=j/ny
        y=y0+(y1-y0)*v
        for i in range(nx+1):
            u=i/nx
            dx=width_inner*(1-u)+width_outer*u
            # Gently round the front/back corners without a hanging tongue edge.
            corner=1-.065*(abs(2*v-1)**6)
            p,n=surface.top(surface.cx+sign*dx*corner,y,lift)
            verts.append(tuple(p))
            uv.append((u,v))
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i
            face=(a,a+1,a+nx+2,a+nx+1)
            faces.append(tuple(reversed(face)) if sign<0 else face)
    obj=_mesh(name,verts,faces,col,mat,side,uv)
    solid=obj.modifiers.new("Thin leather facing","SOLIDIFY")
    solid.thickness=.00065
    solid.offset=-1
    return obj


def _build_side(upper,col,mats,side):
    surf=_Surface(upper)
    cx=surf.cx
    objects=[]
    prefix="Shoe_Refined_"
    # Brown tongue ends in front of the ankle opening. It never crosses the sock.
    objects.append(_patch(prefix+"Tongue_"+side,surf,col,mats["leather"],side,
                          -.014,.014,-.091,-.035,.00115,nx=12))
    for s in (-1,1):
        objects.append(_patch(prefix+"Facing_%s_%s"%(side,s),surf,col,
                              mats["leather"],side,.0125,.026,
                              -.086,-.037,.00165,sign=s,nx=6))
        stitch=[]
        for j in range(29):
            y=-.083+.044*j/28
            p,n=surf.top(cx+s*.0245,y,.00205)
            stitch.append(p)
        objects.append(_tube(prefix+"Facing_Stitch_%s_%s"%(side,s),stitch,.00020,
                             col,mats["lace"],side,sides=6))
        # Quarter seams are nearest-surface projections, not floating design-space lines.
        controls=[(cx+s*.027,.002,.097),(cx+s*.031,-.004,.079),
                  (cx+s*.037,-.020,.055),(cx+s*.040,-.038,.036)]
        seam=[surf.near(p,.00085)[0] for p in _smooth_path(controls,10)]
        objects.append(_tube(prefix+"Quarter_Seam_%s_%s"%(side,s),seam,
                             .00028,col,mats["sole"],side,sides=6))

    rows=[]
    for i,y in enumerate((-.080,-.068,-.056,-.044)):
        row=[]
        for s in (-1,1):
            x=cx+s*(.0200-.001*i/3)
            p,n=surf.top(x,y,.00210)
            objects.append(_torus(prefix+"Eyelet_%s_%s_%s"%(side,i,s),
                                  p,n,.00155,.00036,col,mats["metal"],side))
            row.append((x,y))
        rows.append(row)

    def fitted_xy_path(controls,lift=.0028,bulge=.0004,closed=False):
        points=_smooth_path([(x,y,0) for x,y in controls],10,closed)
        result=[]
        for i,p in enumerate(points):
            rise=bulge*math.sin(math.pi*i/max(1,len(points)-1))**2
            result.append(surf.top(p.x,p.y,lift+rise)[0])
        return result

    bottom=fitted_xy_path([rows[0][0],(cx,-.080),rows[0][1]])
    objects.append(_tube(prefix+"Lace_Bottom_"+side,bottom,.00072,col,mats["lace"],side))
    for row in range(3):
        for s in (0,1):
            a=rows[row][s]
            b=rows[row+1][1-s]
            middle=((a[0]+b[0])/2,(a[1]+b[1])/2)
            pts=fitted_xy_path([a,middle,b],.0027+(.00025 if s else 0),.00055)
            objects.append(_tube(prefix+"Cross_Lace_%s_%s_%s"%(side,row,s),
                                 pts,.00073,col,mats["lace"],side))

    # Small, low bow on the vamp, 15mm ahead of the shoe opening.
    knot_xy=(cx,-.046)
    for s in (-1,1):
        controls=[knot_xy,(cx+s*.010,-.038),(cx+s*.019,-.041),
                  (cx+s*.019,-.049),(cx+s*.009,-.054),knot_xy]
        pts=fitted_xy_path(controls,.00315,.00065,closed=True)
        objects.append(_tube(prefix+"Bow_Loop_%s_%s"%(side,s),pts,.00077,
                             col,mats["lace"],side,cyclic=True))
        controls=[knot_xy,(cx+s*.004,-.053),(cx+s*.010,-.063),(cx+s*.013,-.071)]
        pts=fitted_xy_path(controls,.0033,.00025)
        objects.append(_tube(prefix+"Lace_End_%s_%s"%(side,s),pts,.00073,
                             col,mats["lace"],side))
        objects.append(_tube(prefix+"Aglet_%s_%s"%(side,s),pts[-5:],.00077,
                             col,mats["metal"],side))
    knotpts=fitted_xy_path([(cx-.0023,-.046),(cx,-.0447),(cx+.0023,-.046)],
                          .0040,.0002)
    objects.append(_tube(prefix+"Lace_Knot_"+side,knotpts,.0013,
                         col,mats["lace"],side,sides=10))
    return objects


def _bind(obj,rig,side,source):
    # Vertex coordinates use the same original mesh frame as the original upper.
    obj.parent=rig
    obj.matrix_world=source.matrix_world.copy()
    name=("left" if side=="L" else "right")+"Foot"
    if name not in rig.data.bones:
        raise ValueError("Missing required shoe bone "+name)
    group=obj.vertex_groups.new(name=name)
    group.add(list(range(len(obj.data.vertices))),1.0,"REPLACE")
    mod=obj.modifiers.new("Humanoid deformation","ARMATURE")
    mod.object=rig
    mod.use_deform_preserve_volume=True
    # New details are dense final meshes; their thin-facing Solidify may stay last.
    obj.modifiers.move(len(obj.modifiers)-1,0)


def refine_shoes(col,mats,rig=None):
    """Replace only explicitly named original shoe decorations; return new meshes.

    Required material keys: leather, sole, lace, metal. Passing rig binds all new
    vertices to their matching Foot bone. Existing upper geometry/weights remain.
    A repeated successful call is a no-op and returns [].
    """
    uppers={}
    for side in ("L","R"):
        upper=next((o for o in col.objects
                    if o.name=="Shoe_Leather_Upper_"+side and o.type=="MESH"),None)
        if upper is None:
            raise ValueError("Original shoe upper not found: "+side)
        uppers[side]=upper
    existing=[o for o in col.objects if o.get("shoe_detail_refinement_version")==VERSION]
    if all(o.get("shoe_details_refined_version")==VERSION for o in uppers.values()) and existing:
        return []
    if rig is not None:
        for name in ("leftFoot","rightFoot"):
            if name not in rig.data.bones:
                raise ValueError("Required foot bone missing: "+name)
    new=[]
    try:
        for side,upper in uppers.items():
            objects=_build_side(upper,col,mats,side)
            new.extend(objects)
            for obj in objects:
                if rig is not None:
                    _bind(obj,rig,side,upper)
                else:
                    obj.matrix_world=upper.matrix_world.copy()
        # Delay removing the known bad detail until all replacement projections succeed.
        old=[o for o in list(col.objects)
             if o not in new and o.name.startswith("Shoe_")
             and (o.name.startswith(OLD_PREFIXES) or o in existing)
             and (o.get("deform_role")=="shoe" or o.get("original_mesh"))]
        for obj in old:
            bpy.data.objects.remove(obj,do_unlink=True)
        for upper in uppers.values():
            upper["shoe_details_refined_version"]=VERSION
        col["shoe_detail_refinement_removed_count"]=len(old)
        col["shoe_detail_refinement_added_count"]=len(new)
        return new
    except Exception:
        # Source shoe objects and original details stay intact on a projection failure.
        for obj in new:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj,do_unlink=True)
        # _build_side can fail before returning its list; remove only this run's fresh tag.
        for obj in list(col.objects):
            if obj not in existing and obj.get("shoe_detail_refinement_version")==VERSION:
                bpy.data.objects.remove(obj,do_unlink=True)
        raise
