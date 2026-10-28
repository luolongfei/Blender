"""Original mesh construction for the approved twin-ponytail character.

Coordinates: metres, +Z up, character faces -Y.  Run inside Blender through MCP.
User references remain unchanged.  No donor geometry is used.
"""
import bpy
import bmesh
import math
import json
import importlib.util
from mathutils import Vector, Quaternion
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
HERE = Path(__file__).resolve().parent
OUT = ROOT / "exports"
PREVIEWS = ROOT / "previews"
SCENE_NAME = "双马尾绿裙 · Local Atelier"
COL_NAME = "CHARACTER · Original geometry"


def load_module(name):
    spec = importlib.util.spec_from_file_location("char_local_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear_hex(value):
    value = value.lstrip("#")
    v = [int(value[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= .04045 else ((c + .055)/1.055)**2.4 for c in v) + (1,)


def material(name, color, rough=.6, metal=0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes.clear()
    out = m.node_tree.nodes.new("ShaderNodeOutputMaterial")
    p = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Base Color"].default_value = linear_hex(color)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Specular IOR Level"].default_value = .22
    m.node_tree.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    m.diffuse_color = linear_hex(color)
    return m


def materials():
    palette = {
        "skin": ("F8D7B6", .66, 0), "skin_shadow": ("D49F80", .65, 0),
        "nail": ("EFCBB8", .34, 0), "blouse": ("F0E4D4", .76, 0),
        "blouse_shadow": ("D8C9B8", .78, 0), "trim": ("EFE1D1", .74, 0),
        "button": ("DCCDBA", .35, .05), "skirt": ("4C5645", .82, 0),
        "skirt_shadow": ("3E4839", .85, 0), "sock": ("E8E2D3", .88, 0),
        "leather": ("4D3E32", .38, 0), "sole": ("302820", .73, 0),
        "lace": ("3B3028", .78, 0), "metal": ("7A705C", .3, .55),
        "hair": ("35322E", .45, 0), "hair_highlight": ("403B34", .46, 0),
        "hair_shadow": ("292724", .50, 0), "hair_tie": ("414743", .66, 0),
    }
    result = {key: material("Character / " + key, *values) for key, values in palette.items()}
    for key in ("blouse", "skirt", "sock"):
        m = result[key]
        n = m.node_tree.nodes
        links = m.node_tree.links
        p = next(x for x in n if x.type == "BSDF_PRINCIPLED")
        coord = n.new("ShaderNodeTexCoord")
        noise = n.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 700 if key == "sock" else 520
        noise.inputs["Detail"].default_value = 2
        bump = n.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = .12
        bump.inputs["Distance"].default_value = .00035
        links.new(coord.outputs["Generated"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return result


def mesh_obj(name, vertices, faces, col, mat, part, sub=0, uv=None):
    mesh = bpy.data.meshes.new(name + " geometry")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    if mat:
        mesh.materials.append(mat)
    for p in mesh.polygons:
        p.use_smooth = True
    if uv is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            layer.data[loop.index].uv = uv[loop.vertex_index]
    obj["part"] = part
    obj["source"] = "Original local Blender modeling from approved reference images"
    if sub:
        mod = obj.modifiers.new("Surface refinement", "SUBSURF")
        mod.levels = sub
        mod.render_levels = sub
    return obj


def tube(name, points, radii, col, mat, part, sides=12):
    verts, faces, uv = [], [], []
    points = [Vector(p) for p in points]
    for j, (p, r) in enumerate(zip(points, radii)):
        tangent = points[min(j+1, len(points)-1)] - points[max(0, j-1)]
        tangent.normalize()
        up = Vector((0, 0, 1))
        if abs(tangent.dot(up)) > .92:
            up = Vector((0, 1, 0))
        a = tangent.cross(up).normalized()
        b = tangent.cross(a).normalized()
        for k in range(sides):
            t = 2*math.pi*k/sides
            verts.append(tuple(p + r*(a*math.cos(t) + b*math.sin(t))))
            uv.append((k/sides, j/(len(points)-1)))
    for j in range(len(points)-1):
        for k in range(sides):
            kn = (k+1) % sides
            faces.append((j*sides+k, j*sides+kn, (j+1)*sides+kn, (j+1)*sides+k))
    faces.append(tuple(reversed(range(sides))))
    faces.append(tuple((len(points)-1)*sides+k for k in range(sides)))
    return mesh_obj(name, verts, faces, col, mat, part, 1, uv)


def sphere(name, loc, scale, col, mat, part, segments=40, rings=24):
    verts, faces, uv = [], [], []
    for j in range(rings+1):
        phi = math.pi*j/rings
        for k in range(segments):
            t = math.tau*k/segments
            verts.append((loc[0]+scale[0]*math.sin(phi)*math.cos(t),
                          loc[1]+scale[1]*math.sin(phi)*math.sin(t),
                          loc[2]+scale[2]*math.cos(phi)))
            uv.append((k/segments, 1-j/rings))
    for j in range(rings):
        for k in range(segments):
            faces.append((j*segments+k, j*segments+(k+1)%segments,
                          (j+1)*segments+(k+1)%segments, (j+1)*segments+k))
    return mesh_obj(name, verts, faces, col, mat, part, 0, uv)


def interp(profile, z, index):
    for a, b in zip(profile, profile[1:]):
        if a[0] <= z <= b[0]:
            f = (z-a[0]) / (b[0]-a[0])
            # Cubic interpolation with neighbouring control-point tangents.
            ia = profile.index(a)
            before = profile[max(0, ia-1)]
            after = profile[min(len(profile)-1, ia+2)]
            ma = (b[index]-before[index])/(b[0]-before[0])
            mb = (after[index]-a[index])/(after[0]-a[0])
            h = b[0]-a[0]
            return (2*f**3-3*f**2+1)*a[index]+(f**3-2*f*f+f)*h*ma+(-2*f**3+3*f*f)*b[index]+(f**3-f*f)*h*mb
    return profile[0 if z < profile[0][0] else -1][index]


def face_material(mats):
    m = material("Character / painted face from approved source", "F8D7B6", .72)
    nodes, links = m.node_tree.nodes, m.node_tree.links
    p = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    image = bpy.data.images.load(str(ROOT/"images/head/front.png"), check_existing=True)
    image.name = "APPROVED · head front · unchanged"
    image.pack()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    tex.extension = "EXTEND"
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "FaceProjection"
    mult = nodes.new("ShaderNodeMath")
    mult.operation = "MULTIPLY"
    links.new(attr.outputs["Fac"], mult.inputs[0])
    links.new(tex.outputs["Alpha"], mult.inputs[1])
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[1].default_value = linear_hex("F8D7B6")
    links.new(mult.outputs[0], mix.inputs[0])
    links.new(tex.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs[0], p.inputs["Base Color"])
    return m


def build_head(col, mats):
    profile = [
        (1.318, .002, .041, .040), (1.326, .020, .061, .047),
        (1.340, .039, .074, .051), (1.360, .064, .084, .061),
        (1.385, .086, .088, .080), (1.410, .098, .087, .095),
        (1.440, .104, .088, .104), (1.470, .104, .088, .106),
        (1.500, .100, .083, .103), (1.530, .084, .068, .086),
        (1.555, .053, .044, .056), (1.572, .002, .002, .002)
    ]
    count, rows = 112, 100
    vertices, faces, uvs, masks = [], [], [], []
    for j in range(rows+1):
        z = 1.318 + (1.572-1.318)*j/rows
        r = interp(profile,z,1)
        front, back = interp(profile,z,2), interp(profile,z,3)
        for k in range(count):
            t = math.tau*k/count
            s, c = math.sin(t), math.cos(t)
            x = r*s
            y = .005 - front*max(c,0)**.57 if c >= 0 else .005 + back*(-c)
            if c > 0:
                # Broad smooth facial planes, nose bridge and restrained tip.
                nose = .014*math.exp(-(x/.012)**2-((z-1.390)/.015)**2)
                bridge = .007*math.exp(-(x/.009)**2-((z-1.415)/.028)**2)
                lips = .003*math.exp(-(x/.024)**4-((z-1.365)/.008)**2)
                cheek = .003*(math.exp(-((x-.055)/.025)**2)+math.exp(-((x+.055)/.025)**2))*math.exp(-((z-1.387)/.020)**2)
                eyes = .0024*(math.exp(-((x-.044)/.024)**4)+math.exp(-((x+.044)/.024)**4))*math.exp(-((z-1.428)/.014)**4)
                y -= (nose+bridge+lips+cheek-eyes)*c**4
            vertices.append((x,y,z))
            uvs.append(((627+x*2590)/1254, 1-(858-(z-1.32)*2590)/1254))
            angle = abs((t+math.pi) % math.tau-math.pi)
            f = max(0,min(1,(1.35-angle)/.5))
            masks.append(f*f*(3-2*f))
    for j in range(rows):
        for k in range(count):
            faces.append((j*count+k,j*count+(k+1)%count,(j+1)*count+(k+1)%count,(j+1)*count+k))
    faces.extend((tuple(reversed(range(count))), tuple(rows*count+k for k in range(count))))
    obj = mesh_obj("Face · sculpted planes & original illustration",vertices,faces,col,face_material(mats),"head",1,uvs)
    attr = obj.data.attributes.new("FaceProjection", "FLOAT", "POINT")
    for a, val in zip(attr.data,masks):
        a.value=val
    obj["painted_features"] = "Fixed neutral gaze; approved front source projected on curved facial geometry"
    objs = [obj]
    # Neck is an original visible cylinder, shaped at jaw and collar.
    nprofile=[(1.252,.039,.034),(1.27,.037,.033),(1.30,.034,.032),(1.329,.035,.033),(1.35,.045,.040)]
    vs,fs,uv=[],[],[]
    n=48
    for j,(z,rx,ry) in enumerate(nprofile):
        for k in range(n):
            t=math.tau*k/n
            vs.append((rx*math.sin(t),.006+ry*math.cos(t),z))
            uv.append((k/n,j/(len(nprofile)-1)))
    for j in range(len(nprofile)-1):
        for k in range(n):
            fs.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
    objs.append(mesh_obj("Neck",vs,fs,col,mats["skin"],"neck",2,uv))
    for side,label in ((1,"L"),(-1,"R")):
        ear=sphere("Ear "+label,(side*.103,.002,1.405),(.017,.014,.032),col,mats["skin"],"head")
        objs.append(ear)
        pts=[]
        for j in range(21):
            t=-1.3+5.0*j/20
            pts.append((side*(.110+.007*math.cos(t)),-.009,1.407+.021*math.sin(t)))
        objs.append(tube("Ear helix "+label,pts,[.0017]*len(pts),col,mats["skin_shadow"],"head",8))
        objs.append(sphere("Ear concha "+label,(side*.111,-.011,1.403),(.006,.002,.011),col,mats["skin_shadow"],"head",24,16))
    return objs


def build_legs(col,mats):
    objs=[]
    profile=[
        (.085,.022,.026,.0),(.13,.023,.026,.0),(.22,.026,.027,.005),
        (.29,.033,.033,.009),(.35,.042,.040,.009),(.40,.043,.039,.002),
        (.46,.035,.034,-.006),(.50,.038,.038,-.006),(.56,.045,.046,-.002),
        (.63,.054,.051,.004),(.71,.061,.055,.009),(.81,.065,.058,.012),(.85,.061,.055,.009)
    ]
    for side,label in ((1,"L"),(-1,"R")):
        vs,fs,uv=[],[],[]
        rows,n=96,48
        for j in range(rows+1):
            z=.085+(.85-.085)*j/rows
            rx,ry,cy=[interp(profile,z,i) for i in (1,2,3)]
            cx=side*(.068+.006*math.exp(-((z-.36)/.12)**2))
            for k in range(n):
                t=math.tau*k/n
                x=cx+rx*math.sin(t)
                y=cy-ry*math.cos(t)
                # Soft kneecap prominence; asymmetric calf contour.
                y-=.0028*max(math.cos(t),0)**6*math.exp(-((z-.466)/.028)**2)
                vs.append((x,y,z))
                uv.append((k/n,j/rows))
        for j in range(rows):
            for k in range(n):
                fs.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
        fs.extend((tuple(reversed(range(n))),tuple(rows*n+k for k in range(n))))
        obj=mesh_obj("Leg "+label+" · knee calf ankle",vs,fs,col,mats["skin"],"leg",1,uv)
        obj["side"]=label
        objs.append(obj)
    return objs


def build_hands(col,mats):
    objs=[]
    for side,label in ((1,"L"),(-1,"R")):
        # The oval palm flows out of the cuff; dorsal surface points upward.
        palmprof=[(.600,.019,.015),(.615,.023,.015),(.635,.027,.016),(.655,.029,.0155),(.675,.025,.013),(.682,.022,.010)]
        vs,fs,uv=[],[],[]
        n=32
        for j,(x,ry,rz) in enumerate(palmprof):
            for k in range(n):
                t=math.tau*k/n
                vs.append((side*x,ry*math.cos(t),1.235+rz*math.sin(t)))
                uv.append((k/n,j/(len(palmprof)-1)))
        for j in range(len(palmprof)-1):
            for k in range(n):
                fs.append((j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k))
        fs.extend((tuple(reversed(range(n))),tuple((len(palmprof)-1)*n+k for k in range(n))))
        palm=mesh_obj("Hand "+label+" · palm",vs,fs,col,mats["skin"],"hand",2,uv)
        palm["side"]=label
        objs.append(palm)
        for idx,(name,y,length,r) in enumerate([("Index",-.020,.058,.0073),("Middle",-.005,.064,.0077),("Ring",.010,.058,.0072),("Little",.024,.045,.0062)]):
            pts=[]
            radii=[]
            start=.672
            for j in range(10):
                t=j/9
                pts.append((side*(start+length*t),y+(idx-1.2)*.003*t,1.235-.003*t*t))
                radii.append(r*(1-.22*t)*(1 if j<9 else .4))
            finger=tube("Hand "+label+" · "+name,pts,radii,col,mats["skin"],"finger",12)
            finger["side"]=label; finger["finger"]=name
            objs.append(finger)
            nail=sphere("Nail "+label+" "+name,(side*(start+length*.86),pts[-1][1],1.235+r*.76-.002),
                        (.009 if name!="Little" else .007,.0047 if name!="Little" else .0039,.001),col,mats["nail"],"nail",24,12)
            nail["side"]=label;nail["finger"]=name
            objs.append(nail)
        pts=[(side*.633,-.022,1.231),(side*.642,-.032,1.230),(side*.654,-.043,1.229),(side*.666,-.050,1.228),(side*.677,-.054,1.228)]
        thumb=tube("Hand "+label+" · Thumb",pts,[.010,.0096,.0085,.0075,.0035],col,mats["skin"],"finger",14)
        thumb["side"]=label;thumb["finger"]="Thumb";objs.append(thumb)
        nail=sphere("Nail "+label+" Thumb",(side*.670,-.051,1.234),(.007,.0045,.0012),col,mats["nail"],"nail",24,12)
        nail["side"]=label;nail["finger"]="Thumb";objs.append(nail)
    return objs


def studio(scene):
    studio_col=bpy.data.collections.new("STUDIO · lights & cameras")
    scene.collection.children.link(studio_col)
    floor=material("Studio / warm porcelain","D8D5CC",.87)
    o=mesh_obj("Cyclorama floor",[(-200,-200,-.001),(200,-200,-.001),(200,200,-.001),(-200,200,-.001)],[(0,1,2,3)],studio_col,floor,"studio")
    def area(name,loc,energy,size,color):
        data=bpy.data.lights.new(name,"AREA")
        data.energy=energy;data.shape="DISK";data.size=size;data.color=color
        obj=bpy.data.objects.new(name,data);studio_col.objects.link(obj);obj.location=loc
        obj.rotation_euler=(Vector((0,0,.95))-obj.location).to_track_quat("-Z","Y").to_euler()
    area("Key · large softbox",(-2.8,-3.5,4.2),250,4.0,(1,.89,.77))
    area("Fill · cool softbox",(2.5,-2.0,2.2),160,3.0,(.80,.88,1))
    area("Rim · hair separation",(1.5,2.5,3.0),300,2.2,(1,.90,.78))
    world=bpy.data.worlds.new("Atelier ambient")
    world.use_nodes=True
    world.node_tree.nodes["Background"].inputs[0].default_value=(.32,.34,.39,1)
    world.node_tree.nodes["Background"].inputs[1].default_value=.4
    scene.world=world
    def camera(name,loc,target,ortho,lens=65):
        data=bpy.data.cameras.new(name);obj=bpy.data.objects.new(name,data);studio_col.objects.link(obj)
        obj.location=loc;obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()
        data.type="ORTHO";data.ortho_scale=ortho;data.lens=lens
        return obj
    camera("Camera · front",(0,-5,.85),(0,0,.85),1.84)
    camera("Camera · three-quarter",(2.6,-5.2,2.4),(0,0,.84),1.83)
    camera("Camera · back",(0,5,.85),(0,0,.85),1.84)
    camera("Camera · side",(5,0,.85),(0,0,.85),1.84)
    camera("Camera · portrait",(.45,-2.4,1.54),(0,0,1.41),.53)
    camera("Camera · detail shoes",(.5,-1,.35),(0,-.03,.16),.45)
    scene.camera=bpy.data.objects["Camera · front"]
    scene.render.engine="CYCLES"
    scene.cycles.samples=48
    scene.cycles.use_denoising=True
    scene.render.resolution_x=1600;scene.render.resolution_y=1600;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.film_transparent=False
    scene.view_settings.view_transform="Standard"
    scene.view_settings.look="None"
    scene.view_settings.exposure=0
    scene.render.image_settings.color_mode="RGBA"
    scene.render.image_settings.color_depth="8"
    scene.render.fps=30
    return studio_col


def references(scene):
    col=bpy.data.collections.new("REFERENCES · approved untouched inputs")
    scene.collection.children.link(col)
    col.hide_render=True
    for name,file,loc,rot in [
        ("front","IMG_7225.PNG",(0,.4,.8),(math.pi/2,0,0)),
        ("back","IMG_7227.PNG",(0,.45,.8),(math.pi/2,0,math.pi)),
        ("left","IMG_7226.PNG",(.5,0,.8),(math.pi/2,0,math.pi/2))
    ]:
        img=bpy.data.images.load(str(WORK/"inputs"/file),check_existing=True);img.pack()
        obj=bpy.data.objects.new("Reference / "+name,None)
        col.objects.link(obj);obj.empty_display_type="IMAGE";obj.data=img
        obj.empty_display_size=1.74;obj.location=loc;obj.rotation_euler=rot
        obj.color[3]=.35;obj.hide_render=True;obj.hide_viewport=True


def create():
    WORK.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);PREVIEWS.mkdir(exist_ok=True)
    existing=bpy.data.scenes.get(SCENE_NAME)
    if existing:
        raise RuntimeError("Scene already exists; use explicit rebuild functions, not a destructive reset.")
    scene=bpy.data.scenes.new(SCENE_NAME)
    bpy.context.window.scene=scene
    col=bpy.data.collections.new(COL_NAME)
    scene.collection.children.link(col)
    mats=materials()
    head=build_head(col,mats)
    legs=build_legs(col,mats)
    hands=build_hands(col,mats)
    studio(scene);references(scene)
    scene["reference_project"]="6c8500bf-1c38-481a-babd-e5478a50dd67"
    scene["production_route"]="Original local Blender geometry; no Tripo generation"
    scene["status"]="Model in progress — visual and deformation review required"
    bpy.context.view_layer.objects.active=head[0]
    head[0].select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(WORK/"character.blend"))
    return {"scene":scene.name,"objects":len(col.objects),"head_vertices":len(head[0].data.vertices)}


def add_components():
    col=bpy.data.collections[COL_NAME]
    mats={k:bpy.data.materials["Character / "+k] for k in ("skin","skin_shadow","nail","blouse","blouse_shadow","trim","button","skirt","skirt_shadow","sock","leather","sole","lace","metal","hair","hair_highlight","hair_shadow","hair_tie")}
    result={}
    for name,fun in (("garments","build_garments"),("hair","build_hair")):
        module=load_module(name)
        objs=getattr(module,fun)(col,mats)
        result[name]=[o.name for o in objs]
    bpy.ops.wm.save_as_mainfile(filepath=str(WORK/"character.blend"))
    return result


def render(view="front",name=None,res=1200,samples=32):
    scene=bpy.context.scene
    scene.camera=bpy.data.objects["Camera · "+view]
    scene.render.resolution_x=res;scene.render.resolution_y=res;scene.render.resolution_percentage=100
    scene.cycles.samples=samples
    scene.render.filepath=str(PREVIEWS/(name or (view+".png")))
    bpy.ops.render.render(write_still=True)
    return scene.render.filepath


def refine_presentation():
    scene=bpy.context.scene
    scene.view_settings.view_transform="AgX"
    scene.view_settings.look="AgX - Medium High Contrast"
    world=scene.world
    nodes=world.node_tree.nodes
    nodes.clear()
    links=world.node_tree.links
    out=nodes.new("ShaderNodeOutputWorld")
    light=nodes.new("ShaderNodeBackground")
    light.inputs[0].default_value=(.55,.60,.68,1)
    light.inputs[1].default_value=.30
    bg=nodes.new("ShaderNodeBackground")
    bg.inputs[0].default_value=(.70,.68,.64,1)
    bg.inputs[1].default_value=.85
    ray=nodes.new("ShaderNodeLightPath")
    mix=nodes.new("ShaderNodeMixShader")
    links.new(ray.outputs["Is Camera Ray"],mix.inputs[0])
    links.new(light.outputs[0],mix.inputs[1])
    links.new(bg.outputs[0],mix.inputs[2])
    links.new(mix.outputs[0],out.inputs["Surface"])
    for name,power in [("Key · large softbox",180),("Fill · cool softbox",110),("Rim · hair separation",180)]:
        bpy.data.lights[name].energy=power
    for key in ("hair","hair_highlight","hair_shadow"):
        mat=bpy.data.materials["Character / "+key]
        p=next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
        p.inputs["Roughness"].default_value=.65
        p.inputs["Specular IOR Level"].default_value=.14
    # Do not retain occluded shin geometry under the opaque socks.
    for o in bpy.data.collections[COL_NAME].objects:
        if o.get("part")=="leg":
            bm=bmesh.new();bm.from_mesh(o.data)
            bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.co.z<.278],context="VERTS")
            bm.to_mesh(o.data);bm.free();o.data.update()
        if o.get("part") in ("hand","finger","nail") and o.get("side")=="R":
            bm=bmesh.new();bm.from_mesh(o.data)
            bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
            bm.to_mesh(o.data);bm.free()
    bpy.ops.wm.save_as_mainfile(filepath=str(WORK/"character.blend"))


if __name__=="__main__":
    print(json.dumps(create(),ensure_ascii=False))
