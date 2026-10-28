"""Original reference-guided garments for the twin-ponytail character.

Coordinates are metres, +Z up, character front is -Y, anatomical left is +X.
No scene changes occur on import. Call build_garments(collection, materials).
All geometry here is original; no supplied donor body is used.
"""
import bpy
import math
from mathutils import Vector

PI = math.pi
TAU = 2.0 * PI


def _smooth(a, b, x):
    if a == b:
        return float(x >= b)
    t = max(0.0, min(1.0, (x-a)/(b-a)))
    return t*t*(3.0-2.0*t)


def _interp(table, x):
    if x <= table[0][0]:
        return table[0][1]
    for i in range(1, len(table)):
        x1, v1 = table[i]
        x0, v0 = table[i-1]
        if x <= x1:
            t = _smooth(x0, x1, x)
            return v0*(1.0-t)+v1*t
    return table[-1][1]


def _tag(obj, part, role, side=None):
    obj["part"] = part
    obj["deform_role"] = role
    obj["original_mesh"] = True
    if side:
        obj["side"] = side
    return obj


def _mesh(name, verts, faces, collection, material, part, role,
          uv=None, smooth=True, side=None):
    mesh = bpy.data.meshes.new(name+"_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    if material:
        mesh.materials.append(material)
    for p in mesh.polygons:
        p.use_smooth = smooth
    if uv is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        for poly in mesh.polygons:
            us = [uv[mesh.loops[li].vertex_index][0] for li in poly.loop_indices]
            seam = max(us)-min(us) > 0.5
            for li in poly.loop_indices:
                u, v = uv[mesh.loops[li].vertex_index]
                if seam and u < 0.5:
                    u += 1.0
                layer.data[li].uv = (u, v)
    return _tag(obj, part, role, side)


def _shell(obj, thickness=0.0012, subdiv=1):
    if subdiv:
        modifier = obj.modifiers.new("Soft cloth surface", "SUBSURF")
        modifier.subdivision_type = "CATMULL_CLARK"
        modifier.levels = subdiv
        modifier.render_levels = subdiv
    modifier = obj.modifiers.new("Real fabric thickness", "SOLIDIFY")
    modifier.thickness = thickness
    modifier.offset = 0.0
    modifier.use_even_offset = True
    return obj


def _flip(obj):
    for polygon in obj.data.polygons:
        polygon.flip()
    obj.data.update()
    return obj


def _curve(name, points, collection, material, radius=0.0007,
           cyclic=False, part="upper_body", role="blouse", side=None,
           bevel_resolution=2, smooth=True):
    data = bpy.data.curves.new(name+"_curve", "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 3
    data.bevel_depth = radius
    data.bevel_resolution = bevel_resolution
    data.resolution_u = 6
    if smooth:
        spline = data.splines.new("BEZIER")
        spline.bezier_points.add(len(points)-1)
        for p, co in zip(spline.bezier_points, points):
            p.co = co
            p.handle_left_type = "AUTO"
            p.handle_right_type = "AUTO"
        spline.use_cyclic_u = cyclic
        spline.resolution_u = 4
    else:
        spline = data.splines.new("POLY")
        spline.points.add(len(points)-1)
        for p, co in zip(spline.points, points):
            p.co = (*co, 1.0)
        spline.use_cyclic_u = cyclic
    data.use_fill_caps = not cyclic
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    if material:
        data.materials.append(material)
    return _tag(obj, part, role, side)


def _grid(name, point, nu, nv, collection, material, part, role,
          wrap=False, side=None):
    verts, uv, faces = [], [], []
    for j in range(nv+1):
        for i in range(nu if wrap else nu+1):
            u, v = i/nu, j/nv
            verts.append(point(u, v))
            uv.append((u, v))
    stride = nu if wrap else nu+1
    for j in range(nv):
        for i in range(nu):
            ni = (i+1) % stride
            faces.append((j*stride+i, j*stride+ni,
                          (j+1)*stride+ni, (j+1)*stride+i))
    return _mesh(name, verts, faces, collection, material, part, role,
                 uv=uv, side=side)


def _ellipsoid(name, location, scale, collection, material, part, role,
               side=None, segments=24, rings=12):
    verts = [(location[0], location[1], location[2]+scale[2])]
    uv = [(.5, 0)]
    for j in range(1, rings):
        lat = PI*j/rings
        for i in range(segments):
            lon = TAU*i/segments
            verts.append((location[0]+scale[0]*math.sin(lat)*math.cos(lon),
                          location[1]+scale[1]*math.sin(lat)*math.sin(lon),
                          location[2]+scale[2]*math.cos(lat)))
            uv.append((i/segments, j/rings))
    bottom = len(verts)
    verts.append((location[0], location[1], location[2]-scale[2]))
    uv.append((.5, 1))
    faces = [(0,1+i,1+(i+1)%segments) for i in range(segments)]
    for j in range(rings-2):
        for i in range(segments):
            ni = (i+1) % segments
            faces.append((1+j*segments+i, 1+(j+1)*segments+i,
                          1+(j+1)*segments+ni, 1+j*segments+ni))
    last = 1+(rings-2)*segments
    faces.extend((bottom,last+(i+1)%segments,last+i) for i in range(segments))
    return _mesh(name, verts, faces, collection, material, part, role,
                 uv=uv, side=side)


def _torus(name, center, normal, major, minor, collection, material,
           part, role, side=None, nu=20, nv=8):
    n = Vector(normal).normalized()
    ref = Vector((1, 0, 0))
    if abs(n.dot(ref)) > 0.95:
        ref = Vector((0, 1, 0))
    a = n.cross(ref).normalized()
    b = n.cross(a).normalized()
    c = Vector(center)
    verts, faces, uv = [], [], []
    for j in range(nv):
        v = TAU*j/nv
        for i in range(nu):
            u = TAU*i/nu
            r = a*math.cos(u)+b*math.sin(u)
            p = c+r*(major+minor*math.cos(v))+n*minor*math.sin(v)
            verts.append(tuple(p))
            uv.append((i/nu, j/nv))
    for j in range(nv):
        for i in range(nu):
            faces.append((j*nu+i, j*nu+(i+1)%nu,
                          ((j+1)%nv)*nu+(i+1)%nu, ((j+1)%nv)*nu+i))
    return _mesh(name, verts, faces, collection, material, part, role,
                 uv=uv, side=side)


def _torso_point(theta, t):
    z = 0.985 + 0.298*t
    xr = _interp([(0, .108), (.10, .110), (.16, .123), (.25, .124), (.42, .132),
                  (.59, .151), (.73, .166), (.84, .181), (.90, .162),
                  (.94, .121), (.97, .081), (1, .0475)], t)
    yr = _interp([(0, .070), (.10, .0735), (.16, .082), (.27, .078), (.46, .086),
                  (.62, .096), (.77, .084), (.85, .068),
                  (.93, .048), (1, .041)], t)
    s, c = math.sin(theta), math.cos(theta)
    front = max(0.0, -s)
    back = max(0.0, s)
    fold_env = (1.0-_smooth(.78, 1.0, t))
    hem_gather = math.exp(-((t-.19)/.085)**2)
    shoulder_gather = math.exp(-((t-.82)/.15)**2)
    longfold = math.sin(11*theta + .32*math.sin(t*5.0))
    ripple = ((.0019*longfold + .0012*math.cos(17*theta + t*3))
              *fold_env + .0034*math.cos(12*theta+.6)*hem_gather)
    ripple += .0018*math.cos(15*theta)*shoulder_gather
    # Conceal the tucked fabric all the way under the waistband.  In particular,
    # radial gathers must not cross the belt and appear as an exposed scalloped hem.
    ripple *= _smooth(.095, .155, t)
    # The tucked hem and broad, soft folds are actual surface relief.
    x = (xr+ripple)*c
    y = (yr+ripple)*s
    y -= .009*front**5*math.exp(-((t-.58)/.17)**2)
    y += .0025*back*math.sin(theta*8.0+t*3.0)*fold_env*_smooth(.10,.16,t)
    # Neckline is lower at the throat; shoulders slope gently into the sleeves.
    z -= .019*front**3*_smooth(.87, 1, t)
    z += .0011*math.sin(9*theta)*math.sin(PI*t)**2
    return (x, y, z)


def _make_blouse(collection, mats, out):
    torso = _grid("Blouse_Tucked_Bodice", lambda u, v:_torso_point(TAU*u, v),
                  96, 44, collection, mats["blouse"], "upper_body", "blouse",
                  wrap=True)
    _shell(torso, .0015, 1)
    out.append(torso)

    # Front button placket follows the draped bodice, rather than floating flat.
    def placket(u, v):
        t = .015 + .91*v
        c = _torso_point(-PI/2, t)
        return ((u-.5)*.013, c[1]-.0019-.001*math.sin(PI*u), c[2])
    obj = _grid("Blouse_Button_Placket", placket, 4, 42, collection,
                mats["blouse"], "upper_body", "blouse")
    _shell(obj, .0009, 1)
    out.append(obj)
    for sx in (-1, 1):
        pts = []
        for j in range(26):
            p = placket(.13 if sx < 0 else .87, j/25)
            pts.append((p[0], p[1]-.0004, p[2]))
        out.append(_curve("Blouse_Placket_Edge_"+str(sx), pts, collection,
                          mats["trim"], .00028, role="blouse"))

    for i, z in enumerate((1.035, 1.077, 1.119, 1.161, 1.203, 1.246)):
        t = (z-.985)/.298
        p = _torso_point(-PI/2, t)
        y = p[1]-.0043
        out.append(_ellipsoid("Blouse_Button_%02d"%(i+1), (0, y, z),
                              (.00335, .00125, .00335), collection,
                              mats["button"], "upper_body", "blouse",
                              segments=24, rings=10))
        for dx in (-.0007, .0007):
            out.append(_curve("Button_%02d_Thread_%s"%(i+1, dx),
                              [(dx,y-.0012,z-.0007), (dx,y-.0014,z+.0007)],
                              collection, mats["trim"], .00018,
                              role="blouse", smooth=False, bevel_resolution=1))

    # Soft Peter Pan collar, split at the throat, curving all around the neck.
    for side, sign in (("L", 1), ("R", -1)):
        def collar_point(u, v, sign=sign):
            th = PI*u
            front = max(0.0, math.cos(th))
            width = .012 + .019*front**.6
            # Front tips are rounded off, leaving a clean center slit.
            near_tip = math.exp(-(u/.13)**2)
            rx = .048+width*v
            ry = .041+width*1.03*v
            x = sign*(rx*math.sin(th)+.004*near_tip*v)
            y = -ry*math.cos(th)
            z = 1.282-.018*front-.020*v*front-.005*v
            z += .0025*math.sin(PI*v)
            return (x,y,z)
        obj = _grid("Blouse_PeterPan_Collar_"+side, collar_point, 34, 8,
                    collection, mats["blouse"], "upper_body", "blouse", side=side)
        if sign > 0:
            _flip(obj)
        _shell(obj, .0016, 1)
        out.append(obj)
        pts = [collar_point(i/38, .93) for i in range(39)]
        out.append(_curve("Collar_Edge_Stitch_"+side,
                          [(x,y-.0001,z+.0011) for x,y,z in pts],
                          collection, mats["trim"], .00028,
                          role="blouse", side=side))

    for side, sign in (("L",1), ("R",-1)):
        def sleeve_point(u,v,sign=sign):
            th = TAU*u
            x = .112+.469*v
            ry = _interp([(0,.020),(.05,.035),(.115,.048),(.32,.046),(.52,.045),
                          (.68,.052),(.82,.054),(.92,.044),(1,.0276)],v)
            rz = _interp([(0,.020),(.05,.034),(.115,.049),(.35,.045),(.55,.050),
                          (.72,.060),(.83,.059),(.93,.041),(1,.029)],v)
            centerz = 1.235 - .011*(1-_smooth(0,.12,v))-.013*math.sin(PI*v)**1.6
            gather = (.0016*math.exp(-((v-.15)/.10)**2)
                      +.0052*math.exp(-((v-.91)/.085)**2))
            folds = gather*math.cos(10*th+.4*math.sin(v*6))
            folds += .0016*math.sin(6*th+v*3)*math.sin(PI*v)**2
            # Different drape at the underside retains the blouse's soft balloon shape.
            y = (ry+folds)*math.cos(th)
            z = centerz+(rz+folds)*math.sin(th)
            return (sign*x, y, z)
        obj = _grid("Blouse_Puff_Sleeve_"+side, sleeve_point, 64, 40,
                    collection,mats["blouse"],"upper_body","sleeve",wrap=True,side=side)
        if sign < 0:
            _flip(obj)
        _shell(obj,.0013,1)
        out.append(obj)
        for v,label in ((.145,"Shoulder"),(.987,"Gather")):
            pts = [sleeve_point(i/48,v) for i in range(48)]
            out.append(_curve("Sleeve_"+label+"_Seam_"+side,pts,collection,
                              mats["trim"],.00045,cyclic=True,
                              role="sleeve",side=side))
        # Tailored cuffs leave the wrist opening hollow.
        def cuff(u,v,sign=sign):
            th = TAU*u
            x = .576+.039*v
            bulge = .0008*math.sin(PI*v)
            return (sign*x, (.0292+bulge)*math.cos(th),
                    1.235+(.0308+bulge)*math.sin(th))
        obj=_grid("Blouse_Cuff_"+side,cuff,48,7,collection,mats["blouse"],
                  "upper_body","cuff",wrap=True,side=side)
        if sign < 0:
            _flip(obj)
        _shell(obj,.0018,1)
        out.append(obj)
        for v in (.1,.9):
            pts=[cuff(i/48,v) for i in range(48)]
            out.append(_curve("Cuff_Edge_%s_%s"%(side,v),pts,collection,
                              mats["trim"],.00032,cyclic=True,
                              role="cuff",side=side))
        out.append(_ellipsoid("Cuff_Button_"+side,(sign*.595,-.0307,1.231),
                              (.003,.0012,.003),collection,mats["button"],
                              "upper_body","cuff",side=side,segments=20,rings=8))


def _skirt_point(theta, t):
    # t=0 waist; t=1 hem.
    z = .994-.379*t
    rx = .117+.120*t**.88
    ry = .081+.064*t**.85
    step = TAU/16
    q = (theta+PI/2)/step+.375
    panel = math.floor(q)
    phase = q-panel
    start = (panel-.375)*step-PI/2
    depth = .0035+.0100*t
    def ellipse(angle, inset=0):
        return ((rx-inset)*math.cos(angle),(ry-inset)*math.sin(angle))
    a = ellipse(start)
    b = ellipse(start+.8125*step)
    valley = ellipse(start+.9375*step,depth)
    next_a = ellipse(start+step)
    # Broad panel is a true planar chord. Its narrow turned-under return forms
    # the pleat; unlike a corrugated cone this does not sinusoidally curve every panel.
    if phase <= .8125:
        f=phase/.8125
        x=a[0]*(1-f)+b[0]*f
        y=a[1]*(1-f)+b[1]*f
        bulge=.0008*math.sin(PI*f)*math.sin(PI*t)
        x+=bulge*math.cos(theta)
        y+=bulge*math.sin(theta)
    elif phase <= .9375:
        f=(phase-.8125)/.125
        x=b[0]*(1-f)+valley[0]*f
        y=b[1]*(1-f)+valley[1]*f
    else:
        f=(phase-.9375)/.0625
        x=valley[0]*(1-f)+next_a[0]*f
        y=valley[1]*(1-f)+next_a[1]*f
    z += .0030*math.cos(theta)**2*_smooth(.75,1,t)
    return (x,y,z)


def _make_skirt(collection,mats,out):
    obj=_grid("Skirt_Sixteen_Pleats",lambda u,v:_skirt_point(TAU*u,v),
              256,30,collection,mats["skirt"],"lower_body","skirt",wrap=True)
    _flip(obj)
    # Crease only the long fold corners, leaving the broad cloth panels soft.
    crease=obj.data.attributes.new("crease_edge","FLOAT","EDGE")
    for edge in obj.data.edges:
        ia,ib=edge.vertices
        if ia%256==ib%256:
            theta=TAU*(ia%256)/256
            phase=((theta+PI/2)/(TAU/16)+.375)%1
            if min(abs(phase-p) for p in (.0,.8125,.9375,1.0)) < .035:
                crease.data[edge.index].value=.72
    _shell(obj,.0018,1)
    out.append(obj)
    # Waistband is a separate tailored band, overlapping the tucked-in blouse.
    def belt(u,v):
        th=TAU*u
        z=.988+.029*v
        rx=.1178+.0016*math.sin(PI*v)
        ry=.0825+.0013*math.sin(PI*v)
        return (rx*math.cos(th),ry*math.sin(th),z)
    obj=_grid("Skirt_Fitted_Waistband",belt,96,7,collection,mats["skirt"],
              "lower_body","waistband",wrap=True)
    _shell(obj,.0023,1)
    out.append(obj)
    for v,label in ((.08,"Lower"),(.92,"Upper")):
        pts=[belt(i/96,v) for i in range(96)]
        out.append(_curve("Waistband_"+label+"_Stitch",pts,collection,
                          mats["skirt_shadow"],.00028,cyclic=True,
                          part="lower_body",role="waistband",smooth=False))
    pts=[_skirt_point(TAU*i/256,.982) for i in range(256)]
    out.append(_curve("Skirt_Turned_Hem_Stitch",pts,collection,mats["skirt_shadow"],
                      .00032,cyclic=True,part="lower_body",role="skirt",
                      bevel_resolution=1,smooth=False))
    # A discreet center-back seam and waistband fastening.
    pts=[_skirt_point(PI/2,.03+.94*i/32) for i in range(33)]
    out.append(_curve("Skirt_Center_Back_Seam",
                      [(x,y+.0006,z) for x,y,z in pts],collection,
                      mats["skirt_shadow"],.00045,part="lower_body",role="skirt"))
    out.append(_ellipsoid("Waistband_Back_Button",(0,.0845,1.004),
                          (.0035,.0012,.0035),collection,mats["skirt_shadow"],
                          "lower_body","waistband",segments=20,rings=8))


def _make_socks(collection,mats,out):
    skin_profile=[
        (.085,.022,.026,.0),(.13,.023,.026,.0),(.22,.026,.027,.005),
        (.29,.033,.033,.009),(.35,.042,.040,.009)
    ]
    def skin_value(z,index):
        if z<=skin_profile[0][0]:
            return skin_profile[0][index]
        for i,(a,b) in enumerate(zip(skin_profile,skin_profile[1:])):
            if a[0]<=z<=b[0]:
                f=(z-a[0])/(b[0]-a[0])
                before=skin_profile[max(0,i-1)]
                after=skin_profile[min(len(skin_profile)-1,i+2)]
                ma=(b[index]-before[index])/(b[0]-before[0])
                mb=(after[index]-a[index])/(after[0]-a[0])
                h=b[0]-a[0]
                return ((2*f**3-3*f**2+1)*a[index]+(f**3-2*f*f+f)*h*ma
                        +(-2*f**3+3*f*f)*b[index]+(f**3-f*f)*h*mb)
        return skin_profile[-1][index]
    for side,sign in (("L",1),("R",-1)):
        def sock(u,v,sign=sign):
            th=TAU*u
            z=.074+.214*v
            rx=skin_value(z,1)+.0020
            ry=skin_value(z,2)+.0020
            cy=skin_value(z,3)
            cx=sign*(.068+.006*math.exp(-((z-.36)/.12)**2))
            rib=.00022*math.cos(40*th)
            wrinkle=.00065*math.sin(v*12*PI+1.2*math.sin(th))*math.exp(-((v-.22)/.2)**2)
            return (cx+(rx+rib+wrinkle)*math.cos(th),
                    cy+(ry+rib+wrinkle)*math.sin(th),z)
        obj=_grid("Sock_Ribbed_"+side,sock,160,32,collection,mats["sock"],
                  "lower_body","sock",wrap=True,side=side)
        _shell(obj,.0008,0)
        out.append(obj)
        for v,label in ((.965,"Cuff_Lower"),(1.0,"Cuff_Top")):
            pts=[sock(i/96,v) for i in range(96)]
            out.append(_curve("Sock_"+label+"_"+side,pts,collection,
                              mats["sock"],.00075,cyclic=True,
                              part="lower_body",role="sock",side=side,
                              bevel_resolution=2,smooth=False))


def _shoe_outline(theta, scale=1.0):
    s,c=math.sin(theta),math.cos(theta)
    # Rounded toe and a narrower heel, with a modest medial arch.
    width=.0445*(1-.15*_smooth(0,1,s))
    return (scale*width*c, -.040+scale*.105*s)


def _shoe_upper(theta,t):
    z=_interp([(0,.027),(.18,.042),(.40,.061),(.62,.082),(.81,.098),(1,.108)],t)
    xr=_interp([(0,.0440),(.18,.0433),(.40,.0417),(.62,.0365),(.81,.0292),(1,.0254)],t)
    yr=_interp([(0,.102),(.18,.100),(.40,.086),(.62,.062),(.81,.040),(1,.0310)],t)
    cy=_interp([(0,-.040),(.18,-.040),(.40,-.034),(.62,-.020),(.81,.003),(1,.021)],t)
    s,c=math.sin(theta),math.cos(theta)
    x=xr*c*(1-.10*max(0,s)*(1-t))
    y=cy+yr*s
    # At the toe the leather rounds into the welt with no pinched pole.
    z+=.0018*math.sin(PI*t)*math.cos(theta*2)
    return (x,y,z)


def _make_shoe(side,sign,collection,mats,out):
    cx=sign*.068
    role="shoe"
    # Multiple real layers make the welt readable in close renders.
    for name,levels,mat in (
        ("Outsole",[(.006,.965),(.008,1.0),(.021,1.0),(.025,.985)],mats["sole"]),
        ("Leather_Welt",[(.023,.985),(.025,1.01),(.0275,1.0),(.029,.982)],mats["leather"]),
    ):
        verts,faces,uv=[],[],[]
        n=80
        for j,(z,scale) in enumerate(levels):
            for i in range(n):
                x,y=_shoe_outline(TAU*i/n,scale)
                verts.append((cx+x,y,z))
                uv.append((i/n,j/(len(levels)-1)))
        for j in range(len(levels)-1):
            for i in range(n):
                faces.append((j*n+i,j*n+(i+1)%n,(j+1)*n+(i+1)%n,(j+1)*n+i))
        faces.append(tuple(reversed(range(n))))
        faces.append(tuple((len(levels)-1)*n+i for i in range(n)))
        obj=_mesh("Shoe_"+name+"_"+side,verts,faces,collection,mat,
                  "lower_body",role,uv=uv,side=side)
        bevel=obj.modifiers.new("Rounded leather edges","BEVEL")
        bevel.width=.0009
        bevel.segments=2
        out.append(obj)

    # A distinct low heel follows the rear footprint instead of being a cube.
    verts,faces=[],[]
    heelpoly=[(-.028,-.002),(-.033,.018),(-.030,.052),(-.020,.062),
              (.020,.062),(.030,.052),(.033,.018),(.028,-.002)]
    heelpoly.reverse()
    for z,sc in ((.001,.96),(.004,1),(.018,1)):
        for x,y in heelpoly:
            verts.append((cx+x*sc,.03+(y-.03)*sc,z))
    for j in range(2):
        for i in range(8):
            faces.append((j*8+i,j*8+(i+1)%8,(j+1)*8+(i+1)%8,(j+1)*8+i))
    faces.extend([tuple(reversed(range(8))),tuple(16+i for i in range(8))])
    obj=_mesh("Shoe_Stacked_Heel_"+side,verts,faces,collection,mats["sole"],
              "lower_body",role,side=side)
    bevel=obj.modifiers.new("Heel soft corners","BEVEL")
    bevel.width=.002
    bevel.segments=3
    out.append(obj)

    obj=_grid("Shoe_Leather_Upper_"+side,
              lambda u,v:(cx+_shoe_upper(TAU*u,v)[0],
                          _shoe_upper(TAU*u,v)[1],
                          _shoe_upper(TAU*u,v)[2]),
              80,24,collection,mats["leather"],"lower_body",role,wrap=True,side=side)
    _shell(obj,.0013,1)
    out.append(obj)
    # An inset dark collar defines the opening around the sock.
    def inner(u,v):
        th=TAU*u
        rx=.0255-.002*v
        ry=.0310-.002*v
        return (cx+rx*math.cos(th),.021+ry*math.sin(th),.108-.012*v)
    obj=_grid("Shoe_Inner_Collar_"+side,inner,64,4,collection,mats["sole"],
              "lower_body",role,wrap=True,side=side)
    out.append(obj)
    pts=[(cx+_shoe_upper(TAU*i/64,1)[0],
          _shoe_upper(TAU*i/64,1)[1],
          _shoe_upper(TAU*i/64,1)[2]) for i in range(64)]
    out.append(_curve("Shoe_Rolled_Collar_"+side,pts,collection,mats["leather"],
                      .0015,cyclic=True,part="lower_body",role=role,side=side,
                      smooth=False))
    # Welt stitch ring is subtle and physically follows the shoe perimeter.
    pts=[]
    for i in range(100):
        x,y=_shoe_outline(TAU*i/100,1.002)
        pts.append((cx+x,y,.0270))
    out.append(_curve("Shoe_Welt_Stitch_"+side,pts,collection,mats["lace"],
                      .00028,cyclic=True,part="lower_body",role=role,side=side,
                      smooth=False,bevel_resolution=1))

    # Toe-cap seam: sweep across the front dome at a nearly constant upper ring.
    pts=[]
    for i in range(33):
        th=PI+PI*i/32
        x,y,z=_shoe_upper(th,.39+.06*math.cos(th)**2)
        pts.append((cx+x,y-.0005,z+.00075))
    out.append(_curve("Shoe_Toe_Cap_Seam_"+side,pts,collection,mats["sole"],
                      .00065,part="lower_body",role=role,side=side))
    # Curved side-quarter seams and double row stitching.
    for ss in (-1,1):
        pts=[(cx+ss*.023,.035,.105),(cx+ss*.031,.009,.088),
             (cx+ss*.036,-.015,.061),(cx+ss*.038,-.032,.039)]
        out.append(_curve("Shoe_Quarter_Seam_%s_%s"%(side,ss),pts,collection,
                          mats["sole"],.00065,part="lower_body",role=role,side=side))
        out.append(_curve("Shoe_Quarter_Stitch_%s_%s"%(side,ss),
                          [(x+ss*.0012,y-.0006,z+.0006) for x,y,z in pts],
                          collection,mats["lace"],.00023,
                          part="lower_body",role=role,side=side,bevel_resolution=1))

    # Tongue and lacing flaps sit over the instep, on its sloping plane.
    def tongue(u,v):
        y=-.071+.082*v
        width=.0205*(.90+.10*math.sin(PI*v))
        x=cx+(2*u-1)*width
        z=.086+.030*v+.0025*math.sin(PI*u)
        return (x,y,z)
    obj=_grid("Shoe_Padded_Tongue_"+side,tongue,10,18,collection,mats["leather"],
              "lower_body",role,side=side)
    _shell(obj,.0016,1)
    out.append(obj)
    for ss in (-1,1):
        def flap(u,v,ss=ss):
            y=-.066+.067*v
            x=cx+ss*(.014+.016*u-.002*v)
            z=.088+.025*v-.010*u+.002*math.sin(PI*u)
            return (x,y,z)
        obj=_grid("Shoe_Eyelet_Facing_%s_%s"%(side,ss),flap,5,16,
                  collection,mats["leather"],"lower_body",role,side=side)
        if ss < 0:
            _flip(obj)
        _shell(obj,.0014,1)
        out.append(obj)
        pts=[flap(.10,i/20) for i in range(21)]
        out.append(_curve("Shoe_Facing_Stitch_%s_%s"%(side,ss),
                          [(x,y,z+.001) for x,y,z in pts],collection,
                          mats["lace"],.00023,part="lower_body",role=role,
                          side=side,bevel_resolution=1))

    eyelets=[]
    for row in range(4):
        v=row/3
        y=-.060+.047*v
        z=.091+.020*v
        pair=[]
        for ss in (-1,1):
            x=cx+ss*(.022-.002*v)
            out.append(_torus("Shoe_Eyelet_%s_%d_%s"%(side,row,ss),
                              (x,y,z),(0,-.35,1),.00205,.00048,
                              collection,mats["metal"],"lower_body",role,side))
            pair.append((x,y,z+.001))
        eyelets.append(pair)
    out.append(_curve("Shoe_Lace_Bottom_"+side,
                      [eyelets[0][0],(cx,-.060,.095),eyelets[0][1]],
                      collection,mats["lace"],.00088,part="lower_body",
                      role=role,side=side))
    for row in range(3):
        for ss in (0,1):
            a=eyelets[row][ss]
            b=eyelets[row+1][1-ss]
            mid=((a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2+.003)
            out.append(_curve("Shoe_Cross_Lace_%s_%d_%d"%(side,row,ss),
                              [a,mid,b],collection,mats["lace"],.00088,
                              part="lower_body",role=role,side=side))
    knot=(cx,-.011,.117)
    out.append(_ellipsoid("Shoe_Lace_Knot_"+side,knot,(.003,.0022,.002),
                          collection,mats["lace"],"lower_body",role,side,
                          segments=16,rings=8))
    for ss in (-1,1):
        pts=[knot,(cx+ss*.011,-.005,.122),(cx+ss*.023,-.009,.123),
             (cx+ss*.021,-.018,.119),(cx+ss*.008,-.020,.118),knot]
        out.append(_curve("Shoe_Bow_Loop_%s_%s"%(side,ss),pts,collection,
                          mats["lace"],.0009,part="lower_body",role=role,side=side))
        pts=[knot,(cx+ss*.006,-.022,.117),(cx+ss*.010,-.035,.110),
             (cx+ss*.014,-.046,.104)]
        out.append(_curve("Shoe_Lace_End_%s_%s"%(side,ss),pts,collection,
                          mats["lace"],.00083,part="lower_body",role=role,side=side))
        out.append(_curve("Shoe_Aglet_%s_%s"%(side,ss),pts[-2:],collection,
                          mats["metal"],.0009,part="lower_body",role=role,side=side,
                          smooth=False))


def build_garments(collection, mats):
    """Create all garments and footwear; return created objects.

    Required material keys: blouse, blouse_shadow, trim, button, skirt,
    skirt_shadow, sock, leather, sole, lace, metal.
    Every object contains part/deform_role tags; side L is anatomical left (+X).
    """
    out=[]
    _make_blouse(collection,mats,out)
    _make_skirt(collection,mats,out)
    _make_socks(collection,mats,out)
    _make_shoe("L",1,collection,mats,out)
    _make_shoe("R",-1,collection,mats,out)
    return out
