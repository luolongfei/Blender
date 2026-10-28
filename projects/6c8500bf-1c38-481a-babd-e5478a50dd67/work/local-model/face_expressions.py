"""Project-specific facial reconstruction, with editable geometric expressions.

The previous painted face is preserved in expression-redesign-start/character.blend.
This file acts only on the user's confirmed character, not on kit tooling.
"""
from pathlib import Path
import json
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

P = Path(__file__).resolve().parents[2]
VERSION = "expressive-face-05"
KEYS = (
    "happy_eye", "angry_eye", "sad_eye", "relaxed_eye", "surprised_eye",
    "blink", "blink_L", "blink_R",
    "happy_mouth", "angry_mouth", "sad_mouth", "relaxed_mouth",
    "surprised_mouth", "aa", "ih", "ou", "ee", "oh",
    "brow_happy", "brow_angry", "brow_sad", "brow_relaxed", "brow_surprised",
)
EYE_X = .0435
EYE_Z = 1.4308
EYE_RX = .0235
MOUTH_Z = 1.3688


def smooth(a, b, x):
    t = min(1.0, max(0.0, (x-a)/(b-a)))
    return t*t*(3-2*t)


def mat(name, color, rough=.6, spec=.18, subsurface=0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = rough
    shader.inputs["Specular IOR Level"].default_value = spec
    shader.inputs["Subsurface Weight"].default_value = subsurface
    shader.inputs["Subsurface Radius"].default_value = (.7, .3, .15)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(shader.outputs["BSDF"], out.inputs["Surface"])
    m.diffuse_color = (*color, 1)
    return m


def recalc(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


class FaceBuilder:
    def __init__(self):
        self.face = bpy.data.objects["Face · sculpted planes & original illustration"]
        if self.face.get("facial_rebuild_version"):
            raise RuntimeError("Use the retained checkpoint for deterministic rebuild.")
        self.col = bpy.data.collections["CHARACTER · Original geometry"]
        self.rig = bpy.data.objects["Character_Rig"]
        self.original = [v.co.copy() for v in self.face.data.vertices]
        for co in self.original:
            front=1-smooth(-.065,-.015,co.y)
            co.y-=front*(.0020*math.exp(-(co.x/.011)**2-((co.z-1.395)/.010)**2)
                         +.0009*math.exp(-((abs(co.x)-.012)/.005)**2-((co.z-1.386)/.006)**2))
        self.polys = [list(p.vertices) for p in self.face.data.polygons]
        self.bvh = BVHTree.FromPolygons(self.original, self.polys)
        self.cache = {}
        self.parts = []
        self.skin = mat("Character / expressive neutral skin", (.89, .665, .495), .68, .14, .025)
        self.lip = mat("Character / subtle lip tissue", (.80, .53, .40), .7, .12)
        self.lash = mat("Character / soft charcoal lashes", (.029, .019, .016), .73, .1)
        self.brow = mat("Character / warm charcoal brows", (.059, .040, .031), .8, .08)
        self.white = mat("Character / warm sclera", (.79, .75, .66), .34, .26)
        self.pupil = mat("Character / deep brown pupil", (.019, .010, .006), .34, .25)
        self.iris_colors = [
            mat("Character / iris limbus", (.045, .025, .014), .43, .2),
            mat("Character / iris umber", (.13, .068, .035), .43, .2),
            mat("Character / iris warm brown", (.23, .13, .067), .44, .2),
            mat("Character / iris amber fibers", (.31, .19, .101), .44, .2),
        ]
        self.spark = mat("Character / eye catchlight", (.97, .95, .87), .18, .3)
        self.iris_paint=mat("Character / approved brown iris",(.2,.12,.07),.5,.15)
        image=bpy.data.images.load(str(P/"textures/face-neutral-projection.png"),check_existing=True)
        image.pack()
        nt=self.iris_paint.node_tree
        texture=nt.nodes.new("ShaderNodeTexImage")
        texture.image=image
        texture.extension="EXTEND"
        shader=next(n for n in nt.nodes if n.type=="BSDF_PRINCIPLED")
        nt.links.new(texture.outputs["Color"],shader.inputs["Base Color"])
        self.oral = mat("Character / oral lining", (.105, .021, .025), .78, .08)
        self.tooth = mat("Character / soft ivory teeth", (.84, .78, .65), .5, .12)
        self.tongue = mat("Character / tongue", (.48, .17, .15), .65, .13)

    def y(self, x, z):
        key = (round(x, 7), round(z, 7))
        if key not in self.cache:
            hit = self.bvh.ray_cast(Vector((x, -.5, z)), Vector((0, 1, 0)))
            self.cache[key] = hit[0].y if hit[0] else -.065
        return self.cache[key]

    def eye_y(self, x, z, side):
        # A shallow, obliquely oriented ocular dome. Its lateral surface follows
        # the actual 3D facial plane, so it cannot act as a floating flat decal.
        u = (x-side*EYE_X)/.030
        v = (z-EYE_Z)/.026
        dome = max(0, 1-u*u-v*v)
        return self.y(x, z) - .0032*dome

    def lid_y(self,x,z,side,t):
        q=math.sqrt(((x-side*EYE_X)/.019)**2+((z-EYE_Z)/.017)**2)
        coverage=1-smooth(.80,1.55,q)
        blend=max(coverage,smooth(0,.70,t))
        base=self.y(x,z)
        return base+(self.eye_y(x,z,side)-.00085-base)*blend

    def eye_rim(self, a, side, key="Basis"):
        u, v = math.cos(a), math.sin(a)
        x = side*(EYE_X + EYE_RX*u)
        arch = abs(v)**.88
        zmid = EYE_Z + .0006*u
        hu, hl = .0126, .0090
        if key == "happy_eye":
            hu, hl = .0110, -.0022
            zmid += .0005*arch
            zmid += .0008*u
        elif key == "relaxed_eye":
            hu, hl = .0085, .0070
        elif key == "angry_eye":
            zmid += .0034*u
            hu, hl = .0090, .0075
        elif key == "sad_eye":
            zmid -= .0027*u
            hu, hl = .0100, .0068
        elif key == "surprised_eye":
            hu, hl = .0155, .0105
        closing = key == "blink" or key == ("blink_L" if side > 0 else "blink_R")
        if closing:
            z = zmid - .0048*arch + (.00009 if v >= 0 else -.00009)*arch
        else:
            z = zmid + (hu if v >= 0 else -hl)*arch
        return Vector((x, self.eye_y(x, z, side)-.00090, z))

    def mouth_rim(self, a, key="Basis"):
        rx, hu, hl, corner = .020, .00022, .00025, .0012
        if key == "happy_mouth":
            rx, hu, hl, corner = .023, .0015, .0068, .0034
        elif key == "angry_mouth":
            rx, hu, hl, corner = .020, .0010, .0010, -.0020
        elif key == "sad_mouth":
            rx, hu, hl, corner = .019, .0003, .0004, -.0020
        elif key == "relaxed_mouth":
            rx, hu, hl, corner = .0205, .0003, .0003, .0015
        elif key == "surprised_mouth":
            rx, hu, hl, corner = .0115, .0055, .0120, 0
        elif key == "aa":
            rx, hu, hl, corner = .018, .0035, .0125, .0004
        elif key == "ih":
            rx, hu, hl, corner = .024, .0012, .0038, .0007
        elif key == "ou":
            rx, hu, hl, corner = .0083, .0030, .0047, 0
        elif key == "ee":
            rx, hu, hl, corner = .0245, .0017, .0065, .0012
        elif key == "oh":
            rx, hu, hl, corner = .0112, .0048, .0095, 0
        u, v = math.cos(a), math.sin(a)
        arc = abs(v)**.85
        x = rx*u
        z = MOUTH_Z + corner*(abs(u)**1.8) + (hu if v >= 0 else -hl)*arc
        # Mild paired cupid bow, much smaller than realistic lipstick anatomy.
        if v >= 0:
            z += .0005*math.exp(-((abs(u)-.32)/.20)**2)*arc
        y = self.y(x, MOUTH_Z) - .0007
        if key in ("ou", "oh"):
            y -= .0018*arc
        return Vector((x, y, z))

    def new_obj(self, name, coords, polys, materials, material_ids=None, keys=None, role="face"):
        mesh = bpy.data.meshes.new(name+" · editable topology")
        mesh.from_pydata(coords, [], polys)
        mesh.update()
        recalc(mesh)
        obj = bpy.data.objects.new(name, mesh)
        self.col.objects.link(obj)
        for m in materials:
            mesh.materials.append(m)
        for i, poly in enumerate(mesh.polygons):
            poly.use_smooth = True
            if material_ids:
                poly.material_index = material_ids[i]
        obj["part"] = "head"
        obj["facial_role"] = role
        obj["facial_rebuild_version"] = VERSION
        group = obj.vertex_groups.new(name="head")
        group.add(list(range(len(mesh.vertices))), 1, "REPLACE")
        mod = obj.modifiers.new("Humanoid deformation", "ARMATURE")
        mod.object = self.rig
        if keys:
            obj.shape_key_add(name="Basis",from_mix=False)
            for key, data in keys.items():
                block = obj.shape_key_add(name=key,from_mix=False)
                block.data.foreach_set("co", np.asarray(data, dtype=np.float32).ravel())
        self.parts.append(obj)
        return obj

    def rebuild_skin(self):
        # Explicitly selected grid rectangles encompass the painted eye/mouth
        # regions. Their boundary vertices stay shared with the retained head.
        patches = [
            {"role":"eye", "side":1, "r0":36, "r1":54, "s0":2, "s1":17},
            {"role":"eye", "side":-1, "r0":36, "r1":54, "s0":-17, "s1":-2},
            {"role":"mouth", "side":0, "r0":10, "r1":27, "s0":-8, "s1":8},
        ]
        def idx(r, s):
            return r*112 + s%112
        excluded = set()
        for patch in patches:
            for r in range(patch["r0"], patch["r1"]):
                for s in range(patch["s0"], patch["s1"]):
                    excluded.add(frozenset((idx(r,s),idx(r,s+1),idx(r+1,s+1),idx(r+1,s))))
        faces = [f for f in self.polys if frozenset(f) not in excluded]
        used = sorted({i for f in faces for i in f})
        remap = {old:new for new,old in enumerate(used)}
        coords = [self.original[i].copy() for i in used]
        faces = [[remap[i] for i in f] for f in faces]
        materials = [0]*len(faces)
        records = []
        landmarks = {}
        for p in patches:
            r0,r1,s0,s1 = [p[k] for k in ("r0","r1","s0","s1")]
            boundary = ([idx(r0,s) for s in range(s0,s1)] +
                        [idx(r,s1) for r in range(r0,r1)] +
                        [idx(r1,s) for s in range(s1,s0,-1)] +
                        [idx(r,s0) for r in range(r1,r0,-1)])
            cx, cz = (p["side"]*EYE_X,EYE_Z) if p["role"]=="eye" else (0,MOUTH_Z)
            rx, rz = (EYE_RX,.013) if p["role"]=="eye" else (.020,.014)
            angles = []
            for old in boundary:
                co = self.original[old]
                xlocal = (co.x-cx)*(p["side"] or 1)
                angles.append(math.atan2((co.z-cz)/rz, xlocal/rx))
            last = [remap[i] for i in boundary]
            for j in range(1, 10):
                t = j/9
                current = []
                for old,a in zip(boundary, angles):
                    outer = self.original[old]
                    inner = self.eye_rim(a,p["side"]) if p["role"]=="eye" else self.mouth_rim(a)
                    x = outer.x*(1-t)+inner.x*t
                    z = outer.z*(1-t)+inner.z*t
                    y0 = self.y(x,z)
                    if p["role"]=="eye":
                        y = self.lid_y(x,z,p["side"],t)
                    else:
                        y = y0 + (inner.y-self.y(inner.x,inner.z))*smooth(.35,1,t)
                        # A softly raised lip roll within the continuous face.
                        y -= .00065*math.exp(-((t-.91)/.09)**2)*abs(math.sin(a))
                    co = Vector((x,y,z))
                    vi = len(coords)
                    coords.append(co)
                    current.append(vi)
                    records.append((vi,p,a,t,outer.copy()))
                n = len(current)
                for k in range(n):
                    faces.append([last[k],last[(k+1)%n],current[(k+1)%n],current[k]])
                    materials.append(1 if p["role"]=="mouth" and j==9 else 0)
                last=current
            landmarks[p["role"]+str(p["side"])] = {
                "rimVertices":last,"rimAngles":angles,
                "boundaryVertices":[remap[i] for i in boundary],
            }
        mesh = bpy.data.meshes.new("Face · continuous eyelid and lip loops")
        mesh.from_pydata(coords,[],faces)
        mesh.update()
        recalc(mesh)
        mesh.materials.append(self.skin)
        mesh.materials.append(self.lip)
        lipweight=mesh.attributes.new("LipWeight","FLOAT","POINT")
        for vi,p,a,t,outer in records:
            if p["role"]=="mouth":
                lipweight.data[vi].value=smooth(.80,1,t)*abs(math.sin(a))**.65
        for poly,mi in zip(mesh.polygons,materials):
            poly.use_smooth=True
            poly.material_index=mi
        oldmesh=self.face.data
        self.face.data=mesh
        self.face.vertex_groups.clear()
        g=self.face.vertex_groups.new(name="head")
        g.add(list(range(len(coords))),1,"REPLACE")
        for mod in list(self.face.modifiers):
            if mod.type!="ARMATURE":
                self.face.modifiers.remove(mod)
        sub=self.face.modifiers.new("Facial surface continuity","SUBSURF")
        sub.levels=1
        sub.render_levels=1
        self.face["part"]="head"
        self.face["facial_role"]="continuous_skin_lids_lips"
        self.face["facial_rebuild_version"]=VERSION
        self.face["default_blush"]=False
        self.face["painted_features"]="Replaced with anatomical apertures, eyelids, irises, lips and independent brows"
        self.face.shape_key_add(name="Basis",from_mix=False)
        for key in KEYS:
            block=self.face.shape_key_add(name=key,from_mix=False)
            data=[co.copy() for co in coords]
            for vi,p,a,t,outer in records:
                if p["role"]=="eye" and (key.endswith("_eye") or key.startswith("blink")):
                    neutral=self.eye_rim(a,p["side"])
                    posed=self.eye_rim(a,p["side"],key)
                    w=smooth(0,1,t)
                    data[vi] += (posed-neutral)*w
                    if t>.05:
                        x,y,z=data[vi]
                        data[vi].y=self.lid_y(x,z,p["side"],t)
                if p["role"]=="mouth" and (key.endswith("_mouth") or key in ("aa","ih","ou","ee","oh")):
                    neutral=self.mouth_rim(a)
                    posed=self.mouth_rim(a,key)
                    data[vi]+=(posed-neutral)*smooth(0,1,t)
            # Broad cheek participation keeps smile movement from being
            # confined to the corners. This field is zero at eye apertures.
            if key=="happy_mouth":
                for i,co in enumerate(data):
                    weight=math.exp(-((abs(co.x)-.041)/.022)**2-((co.z-1.392)/.014)**2)
                    frontal=1-smooth(-.065,-.025,co.y)
                    co.z+=.0018*weight*frontal
                    co.y-=.0008*weight*frontal
            opening={"aa":.0040,"oh":.0030,"ou":.0010,"ee":.0018,
                     "ih":.0008,"surprised_mouth":.0040,"happy_mouth":.0012}.get(key,0)
            if opening:
                for co in data:
                    jaw=(1-smooth(1.339,1.366,co.z))*(1-smooth(-.035,.045,co.y))
                    co.z-=opening*jaw
                    co.y+=opening*.28*jaw
            block.data.foreach_set("co",np.asarray(data,dtype=np.float32).ravel())
        self.parts.append(self.face)
        self.landmarks=landmarks
        # Align all exposed skin to the same neutral palette.
        for obj in self.col.all_objects:
            if obj==self.face or obj.type!="MESH":
                continue
            for i,m in enumerate(obj.data.materials):
                if m and m.name=="Character / skin":
                    obj.data.materials[i]=self.skin

    def make_eyes(self):
        for side,label in ((1,"L"),(-1,"R")):
            # Broad ocular surface tucked under the surrounding eyelid skin.
            n=96
            faces=[]
            def sclera(key):
                cv=[]
                for j in range(17):
                    r=j/16
                    for k in range(n):
                        a=math.tau*k/n
                        # The ocular margin stops underneath the actual rim,
                        # preventing white crescents outside the eye socket.
                        rim=self.eye_rim(a,side,key)
                        x=side*EYE_X+(rim.x-side*EYE_X)*r*1.025
                        z=EYE_Z+(rim.z-EYE_Z)*r*1.025
                        y=self.eye_y(x,z,side)+.00010+.0002*smooth(.85,1,r)
                        cv.append((x,y,z))
                return cv
            for j in range(16):
                for k in range(n):
                    faces.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
            whitekeys={k:sclera(k) for k in KEYS if k.endswith("_eye")}
            for key in ("blink","blink_L" if side>0 else "blink_R"):
                whitekeys[key]=[(x,y+.004,z) for x,y,z in sclera("Basis")]
            self.new_obj("Eye sclera "+label,sclera("Basis"),faces,[self.white],
                         keys=whitekeys,
                         role="eye_sclera")
            # Iris includes a dark limbus, warm radial fibers and a brown pupil.
            irx,irz=.0120,.0126
            coords=[]
            faces=[]
            irisuv=[]
            for j in range(13):
                r=j/12
                for k in range(n):
                    a=math.tau*k/n
                    x=side*EYE_X+irx*r*math.cos(a)
                    z=EYE_Z+.0007+irz*r*math.sin(a)
                    coords.append((x,self.eye_y(x,z,side)-.00010,z))
                    px,py=(744,578) if side>0 else (513,584)
                    irisuv.append(((px+30*r*math.cos(a))/1254,1-(py-31*r*math.sin(a))/1254))
            for j in range(12):
                for k in range(n):
                    faces.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
            iriskeys={key:[(x,y+.004,z) for x,y,z in coords] for key in ("blink","blink_L" if side>0 else "blink_R")}
            iris=self.new_obj("Iris brown "+label,coords,faces,[self.iris_paint],keys=iriskeys,role="eye_iris")
            uv=iris.data.uv_layers.new(name="ApprovedIrisUV")
            for loop in iris.data.loops:
                uv.data[loop.index].uv=irisuv[loop.vertex_index]
            def disk(name,cx,cz,rx,rz,material,offset):
                cv=[(cx,self.eye_y(cx,cz,side)+offset,cz)]
                for k in range(64):
                    a=math.tau*k/64
                    x=cx+rx*math.cos(a)
                    z=cz+rz*math.sin(a)
                    cv.append((x,self.eye_y(x,z,side)+offset,z))
                fs=[(0,k+1,(k+1)%64+1) for k in range(64)]
                return self.new_obj(name,cv,fs,[material],role="eye_detail")
            # The original illustration supplies the iris/pupil detail only.
            # It is mapped onto a separate domed surface, never onto facial skin.

    def make_lashes(self):
        for side,label in ((1,"L"),(-1,"R")):
            for upper in (True,False):
                n=65
                def geometry(key):
                    cv=[]
                    for j in range(3):
                        v=j/2
                        for k in range(n):
                            t=k/(n-1)
                            a=math.pi*t*(1 if upper else -1)
                            co=self.eye_rim(a,side,key)
                            taper=math.sin(math.pi*t)**.5
                            width=(.0026*(1-.55*t) if upper else .00016)*taper
                            co.z+=width*v*(1 if upper else -1)
                            co.y-=.00016+.00008*math.sin(v*math.pi)
                            cv.append(co)
                    return cv
                fs=[]
                for j in range(2):
                    for k in range(n-1):
                        fs.append((j*n+k,j*n+k+1,(j+1)*n+k+1,(j+1)*n+k))
                keys={key:geometry(key) for key in KEYS if key.endswith("_eye") or key.startswith("blink")}
                self.new_obj(("Upper lashes " if upper else "Lower lid edge ")+label,geometry("Basis"),fs,[self.lash if upper else self.lip],keys=keys,role="eye_lash")
            # Small tapered outer lash, rather than a uniformly outlined oval.
            def wing(key):
                corner=self.eye_rim(0,side,key)
                inner=self.eye_rim(.27,side,key)
                tip=corner+Vector((side*.0033,-.0002,.0014))
                raised=inner+Vector((side*.0003,-.0002,.0014))
                cv=[corner,tip,raised,inner]
                for co in cv:
                    co.y-=.00025
                return cv
            self.new_obj("Outer lash wing "+label,wing("Basis"),[(0,1,2,3)],[self.lash],
                         keys={k:wing(k) for k in KEYS if k.endswith("_eye") or k.startswith("blink")},
                         role="eye_lash")
            crease_mat=mat("Character / eyelid soft crease",(.54,.36,.25),.8,.08)
            def crease(key):
                cv=[]
                for j in range(2):
                    for k in range(45):
                        t=k/44
                        a=.2+(math.pi-.4)*t
                        co=self.eye_rim(a,side,key)
                        co.z+=.0031*math.sin(math.pi*t)**.5+j*.00025*math.sin(math.pi*t)
                        co.y=self.lid_y(co.x,co.z,side,.78)-.00018
                        cv.append(co)
                return cv
            fs=[(k,k+1,45+k+1,45+k) for k in range(44)]
            self.new_obj("Upper eyelid crease "+label,crease("Basis"),fs,[crease_mat],
                         keys={k:crease(k) for k in KEYS if k.endswith("_eye") or k.startswith("blink")},
                         role="eye_lid_crease")

    def make_brows(self):
        for side,label in ((1,"L"),(-1,"R")):
            def geometry(key):
                cv=[]
                for j in range(3):
                    w=j/2-.5
                    for k in range(49):
                        t=k/48
                        x=side*(.019+.049*t)
                        z=1.4555+.0048*math.sin(math.pi*t)-.0016*t
                        if key=="brow_happy":
                            z+=.0018*math.sin(math.pi*t)+.0007
                        elif key=="brow_surprised":
                            z+=.006+.0015*math.sin(math.pi*t)
                        elif key=="brow_angry":
                            z+=-.0040+.0080*t
                        elif key=="brow_sad":
                            z+=.0040-.0080*t
                        elif key=="brow_relaxed":
                            z-=.001
                        width=.00285*(math.sin(math.pi*t)**.4)*(1-.65*t)
                        z+=w*width
                        cv.append((x,self.y(x,z)-.00055,z))
                return cv
            fs=[]
            for j in range(2):
                for k in range(48):
                    fs.append((j*49+k,j*49+k+1,(j+1)*49+k+1,(j+1)*49+k))
            keys={k:geometry(k) for k in KEYS if k.startswith("brow_")}
            self.new_obj("Expressive brow "+label,geometry("Basis"),fs,[self.brow],keys=keys,role="brow")

    def make_mouth(self):
        n=96
        mouthkeys=[k for k in KEYS if k.endswith("_mouth") or k in ("aa","ih","ou","ee","oh")]
        def lining(key):
            cv=[]
            for j in range(6):
                t=j/5
                for k in range(n):
                    a=math.tau*k/n
                    rim=self.mouth_rim(a,key)
                    # Enclosed oral cavity; the lip rim is the actual aperture.
                    x=rim.x*(1-.32*t)
                    z=rim.z*(1-t*.30)+(MOUTH_Z-.004)*t*.30
                    cv.append((x,rim.y+.0005+.013*t,z))
            return cv
        fs=[]
        for j in range(5):
            for k in range(n):
                fs.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
        fs.append(tuple(range(5*n,6*n)))
        self.new_obj("Mouth interior",lining("Basis"),fs,[self.oral],keys={k:lining(k) for k in mouthkeys},role="mouth_interior")
        def teeth(key):
            cv=[]
            for j in range(3):
                t=j/2
                for k in range(49):
                    u=-.91+1.82*k/48
                    a=math.acos(u)
                    rim=self.mouth_rim(a,key)
                    cv.append((rim.x*.94,rim.y+.0035+.0015*u*u,rim.z-.0002-.0028*t))
            return cv
        fs=[(j*49+k,j*49+k+1,(j+1)*49+k+1,(j+1)*49+k) for j in range(2) for k in range(48)]
        self.new_obj("Upper dental arch",teeth("Basis"),fs,[self.tooth],keys={k:teeth(k) for k in mouthkeys},role="mouth_teeth")
        # Tongue stays behind the lower lip; its visible portion follows vowels.
        def tongue(key):
            cv=[]
            low=self.mouth_rim(1.5*math.pi,key)
            for j in range(12):
                r=j/11
                for k in range(48):
                    a=math.tau*k/48
                    x=.010*r*math.cos(a)
                    y=low.y+.007+.003*r*math.sin(a)
                    z=low.z+.0018-.0016*r*r
                    cv.append((x,y,z))
            return cv
        fs=[(j*48+k,j*48+(k+1)%48,(j+1)*48+(k+1)%48,(j+1)*48+k) for j in range(11) for k in range(48)]
        self.new_obj("Tongue surface",tongue("Basis"),fs,[self.tongue],keys={k:tongue(k) for k in mouthkeys},role="mouth_tongue")

    def make_nose(self):
        shade=mat("Character / nostril soft shade",(.38,.205,.13),.83,.08)
        for side,label in ((1,"L"),(-1,"R")):
            cv=[]
            fs=[]
            cx,cz=side*.0065,1.3850
            cv.append((cx,self.y(cx,cz)-.00013,cz))
            for k in range(48):
                a=math.tau*k/48
                x=cx+.00155*math.cos(a)
                z=cz+.00055*math.sin(a)+side*.00018*math.cos(a)
                cv.append((x,self.y(x,z)-.00013,z))
            for k in range(48):
                fs.append((0,k+1,(k+1)%48+1))
            self.new_obj("Nostril subtle "+label,cv,fs,[shade],role="nose")

    def build(self):
        self.rebuild_skin()
        self.make_eyes()
        self.make_lashes()
        self.make_brows()
        self.make_mouth()
        # The skin-only source provides restrained nostril colour; nose volume
        # is actual sculpted mesh, avoiding duplicate painted/geometric dots.
        for obj in self.parts:
            if obj.data.shape_keys:
                for block in obj.data.shape_keys.key_blocks:
                    block.value=0
        self.face["expression_keys"]=json.dumps(KEYS)
        (P/"work/expression-authoring").mkdir(parents=True,exist_ok=True)
        (P/"work/expression-authoring/facial-construction.json").write_text(json.dumps({
            "version":VERSION,"source":"work/checkpoints/expression-redesign-start/character.blend",
            "objects":[o.name for o in self.parts],"keys":KEYS,"landmarks":self.landmarks,
            "defaultBlush":False,"neutralShapeKeyValues":0,
            "method":"Continuous eye and mouth apertures; shared annular skin loops with curved eyelid/lip motion; independent iris/lashes/brows; enclosed mouth",
        },ensure_ascii=False,indent=2))
        return {"version":VERSION,"objects":len(self.parts),"skinVertices":len(self.face.data.vertices),"keys":len(KEYS)}


def apply():
    return FaceBuilder().build()


def pose(values=None):
    values=values or {}
    for obj in bpy.context.scene.objects:
        if obj.type=="MESH" and obj.get("facial_role") and obj.data.shape_keys:
            for key in obj.data.shape_keys.key_blocks:
                if key.name!="Basis":
                    key.value=float(values.get(key.name,0))
    bpy.context.view_layer.update()


def presets():
    return {
        "neutral":{},
        "smile":{"happy_eye":.55,"happy_mouth":.7,"brow_happy":.5},
        "happy":{"happy_eye":1,"happy_mouth":1,"brow_happy":.65},
        "blink":{"blink":1},
        "wink":{"blink_L":1,"happy_mouth":.55,"brow_happy":.25},
        "sad":{"sad_eye":.7,"sad_mouth":.8,"brow_sad":.8},
        "angry":{"angry_eye":.65,"angry_mouth":.7,"brow_angry":.75},
        "surprised":{"surprised_eye":.8,"surprised_mouth":.8,"brow_surprised":.8},
        "relaxed":{"relaxed_eye":.65,"relaxed_mouth":.6,"brow_relaxed":.5},
        "aa":{"aa":1},"ih":{"ih":1},"ou":{"ou":1},"ee":{"ee":1},"oh":{"oh":1},
    }
