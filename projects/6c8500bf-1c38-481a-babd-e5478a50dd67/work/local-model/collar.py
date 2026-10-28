"""Broad, rounded original Peter Pan collar following the approved character.

Call build_collar(collection, mats) after removing the old collar and edge seams.
The module does not reset a scene or touch any existing objects.
"""
import bpy
import math
from mathutils import Vector


INNER = [
    (0.00, (.005, -.047, 1.261)),
    (.18, (.022, -.043, 1.266)),
    (.40, (.040, -.025, 1.276)),
    (.57, (.044, .000, 1.280)),
    (.80, (.033, .032, 1.280)),
    (1.00, (.000, .043, 1.280)),
]
OUTER = [
    (0.00, (.007, -.092, 1.236)),
    (.18, (.045, -.091, 1.213)),
    (.40, (.085, -.055, 1.242)),
    (.57, (.072, .000, 1.274)),
    (.80, (.055, .043, 1.278)),
    (1.00, (.000, .060, 1.277)),
]


def _path(points, t):
    if t <= points[0][0]:
        return Vector(points[0][1])
    if t >= points[-1][0]:
        return Vector(points[-1][1])
    for i in range(len(points)-1):
        ta, a = points[i]
        tb, b = points[i+1]
        if ta <= t <= tb:
            tp, p = points[max(0, i-1)]
            tn, n = points[min(len(points)-1, i+2)]
            a, b, p, n = map(Vector, (a,b,p,n))
            h = tb-ta
            ma = (b-p)/(tb-tp)
            mb = (n-a)/(tn-ta)
            f = (t-ta)/h
            return ((2*f**3-3*f*f+1)*a + (f**3-2*f*f+f)*h*ma
                    +(-2*f**3+3*f*f)*b + (f**3-f*f)*h*mb)
    return Vector(points[-1][1])


def collar_point(u,v,sign=1):
    inner = _path(INNER,u)
    outer = _path(OUTER,u)
    p = inner.lerp(outer,v)
    # A softly rolled surface, without a raised vertical neckband.
    front = max(0.0,1-u/.67)
    roll = math.sin(math.pi*v)*(.0015+.0014*front)
    p.z += roll
    p.y -= roll*.35*front
    p.x *= sign
    return p


def _normal(u,v,sign):
    eps=.0001
    du = collar_point(min(1,u+eps),v,sign)-collar_point(max(0,u-eps),v,sign)
    dv = collar_point(u,min(1,v+eps),sign)-collar_point(u,max(0,v-eps),sign)
    normal=du.cross(dv)*(-sign)
    return normal.normalized() if normal.length else Vector((0,0,1))


def _tags(obj,side):
    obj["part"]="upper_body"
    obj["deform_role"]="blouse"
    obj["side"]=side
    obj["original_mesh"]=True
    obj["collar_refined"]=True
    obj["collar_geometry_version"]=2
    return obj


def build_collar(collection,mats):
    """Return two broad collar flap meshes and their subtle outer seam curves.

    Required materials: blouse and trim. All coordinates are metres; front is -Y.
    This is a replacement for the original small Blouse_PeterPan_Collar objects.
    """
    objects=[]
    nu=48
    # Closely spaced support rows retain the carefully drawn outer contour.
    rows=(0,.015,.055,.14,.27,.42,.58,.73,.86,.945,.985,1)
    nv=len(rows)-1
    for sign,side in ((1,"L"),(-1,"R")):
        verts=[]
        faces=[]
        uv=[]
        for v in rows:
            for i in range(nu+1):
                u=i/nu
                verts.append(tuple(collar_point(u,v,sign)))
                uv.append((u,v))
        for j in range(nv):
            for i in range(nu):
                a=j*(nu+1)+i
                face=(a,a+1,a+nu+2,a+nu+1)
                faces.append(tuple(reversed(face)) if sign>0 else face)
        mesh=bpy.data.meshes.new("PeterPan_Broad_"+side+"_mesh")
        mesh.from_pydata(verts,[],faces)
        mesh.update()
        layer=mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            poly.use_smooth=True
            for li in poly.loop_indices:
                layer.data[li].uv=uv[mesh.loops[li].vertex_index]
        mesh.materials.append(mats["blouse"])
        obj=bpy.data.objects.new("Blouse_PeterPan_Collar_"+side,mesh)
        collection.objects.link(obj)
        _tags(obj,side)
        sub=obj.modifiers.new("Soft tailored collar","SUBSURF")
        sub.levels=1
        sub.render_levels=1
        solid=obj.modifiers.new("Collar fabric thickness","SOLIDIFY")
        solid.thickness=.0015
        solid.offset=0
        solid.use_even_offset=True
        objects.append(obj)

        data=bpy.data.curves.new("Collar_Edge_Stitch_"+side+"_curve","CURVE")
        data.dimensions="3D"
        data.resolution_u=2
        data.bevel_depth=.00024
        data.bevel_resolution=2
        data.use_fill_caps=True
        spline=data.splines.new("POLY")
        spline.points.add(72)
        for i,point in enumerate(spline.points):
            u=i/72
            p=collar_point(u,.955,sign)+_normal(u,.955,sign)*.00092
            point.co=(*p,1)
        data.materials.append(mats["trim"])
        seam=bpy.data.objects.new("Collar_Edge_Stitch_"+side,data)
        collection.objects.link(seam)
        _tags(seam,side)
        objects.append(seam)
    return objects
