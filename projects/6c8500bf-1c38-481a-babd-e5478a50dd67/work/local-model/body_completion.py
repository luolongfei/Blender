"""Original clothed-body completion, authored in the current rig's rest space.

Adds real trunk/pelvis/arm anatomy and both missing lower-shin/ankle/foot volumes.
Existing neck, hands, legs, face, clothes, UVs, keys, bones and springs survive.
New components are closed, connected surfaces; their joins to retained parts
overlap inside the clothing. This is not claimed to be a single welded body.
"""
import bpy
import bmesh
import math
import json
import runpy
from pathlib import Path
from mathutils import Vector

VERSION='body-completion-02'
PREFIX='Body_Complete_'
NAMES=(PREFIX+'Trunk_Arms_Pelvis',PREFIX+'Shin_Foot_L',PREFIX+'Shin_Foot_R')

def smooth(a,b,v):
    t=max(0.,min(1.,(v-a)/(b-a)))
    return t*t*(3-2*t)

def interp(table,x,index):
    if x<=table[0][0]:return table[0][index]
    if x>=table[-1][0]:return table[-1][index]
    for i,(a,b) in enumerate(zip(table,table[1:])):
        if a[0]<=x<=b[0]:
            p=table[max(0,i-1)];n=table[min(len(table)-1,i+2)]
            t=(x-a[0])/(b[0]-a[0]);h=b[0]-a[0]
            ma=(b[index]-p[index])/(b[0]-p[0])
            mb=(n[index]-a[index])/(n[0]-a[0])
            value=(2*t**3-3*t*t+1)*a[index]+(t**3-2*t*t+t)*h*ma+(-2*t**3+3*t*t)*b[index]+(t**3-t*t)*h*mb
            return max(min(a[index],b[index]),min(max(a[index],b[index]),value))

class Geometry:
    def __init__(self):
        self.vertices=[];self.faces=[]
    def loft(self,centers,r1,r2,e1,e2,sides=48,shape=None):
        start=len(self.vertices);a=Vector(e1);b=Vector(e2)
        for j,center in enumerate(centers):
            center=Vector(center)
            for k in range(sides):
                theta=math.tau*k/sides
                p=center+a*r1[j]*math.cos(theta)+b*r2[j]*math.sin(theta)
                if shape:p=Vector(shape(p,theta,j))
                self.vertices.append(tuple(p))
        for j in range(len(centers)-1):
            for k in range(sides):
                n=(k+1)%sides
                self.faces.append((start+j*sides+k,start+j*sides+n,start+(j+1)*sides+n,start+(j+1)*sides+k))
        self.faces.append(tuple(start+k for k in reversed(range(sides))))
        self.faces.append(tuple(start+(len(centers)-1)*sides+k for k in range(sides)))
    def ellipsoid(self,center,scale,nu=28,nv=16):
        centers=[];r1=[];r2=[]
        # Tiny closed end rings avoid a degeneracy in the volume remesher.
        for j in range(nv+1):
            th=.001+(math.pi-.002)*j/nv
            centers.append((center[0],center[1],center[2]-scale[2]*math.cos(th)))
            r1.append(scale[0]*math.sin(th));r2.append(scale[1]*math.sin(th))
        self.loft(centers,r1,r2,(1,0,0),(0,1,0),nu)

def trunk_geometry():
    g=Geometry()
    # Z, lateral radius, front depth, back depth. All dimensions fit inside
    # the existing blouse and skirt; landmarks follow the authored skeleton.
    profile=[
        (.768,.018,.023,.027),(.790,.078,.049,.061),
        (.815,.110,.067,.079),(.855,.121,.077,.084),
        (.900,.111,.070,.073),(.945,.098,.061,.060),
        (.990,.096,.058,.056),(1.035,.103,.064,.058),
        (1.085,.116,.072,.060),(1.135,.132,.079,.062),
        (1.170,.140,.084,.062),(1.210,.148,.071,.059),
        (1.240,.154,.047,.046),(1.259,.121,.036,.037),
        (1.274,.047,.029,.031),(1.285,.034,.027,.029)]
    centers=[];xr=[];yr=[]
    for j in range(141):
        z=.768+(1.285-.768)*j/140
        rx,front,back=[interp(profile,z,k) for k in (1,2,3)]
        centers.append((0,.004+(back-front)/2,z));xr.append(rx);yr.append((front+back)/2)
    def body_surface(p,theta,j):
        x,y,z=p
        front=max(0.,-math.sin(theta))
        back=max(0.,math.sin(theta))
        # Broad pectoral/rib contour and abdomen; no sexual or private detail.
        chest=(math.exp(-((x-.061)/.047)**2)+math.exp(-((x+.061)/.047)**2))
        y-=.0042*chest*math.exp(-((z-1.149)/.036)**2)*front**2
        y+=.0013*math.exp(-(x/.014)**2)*math.exp(-((z-1.174)/.044)**2)*front
        y-=.0020*math.exp(-(x/.057)**2)*math.exp(-((z-1.024)/.033)**2)*front
        # Subtle scapular planes and a shallow spine channel.
        scapula=(math.exp(-((x-.061)/.035)**2)+math.exp(-((x+.061)/.035)**2))
        y+=.0023*scapula*math.exp(-((z-1.192)/.035)**2)*back
        y-=.0011*math.exp(-(x/.012)**2)*math.exp(-((z-1.125)/.100)**2)*back
        return (x,y,z)
    g.loft(centers,xr,yr,(1,0,0),(0,1,0),64,body_surface)
    arm_profile=[
        (.119,.022,.021,1.235),(.145,.030,.025,1.235),
        (.177,.034,.030,1.235),(.220,.032,.0255,1.234),
        (.272,.029,.030,1.233),(.322,.0255,.0265,1.234),
        (.365,.024,.024,1.235),(.385,.0245,.0245,1.235),
        (.416,.0265,.028,1.235),(.455,.0255,.027,1.235),
        (.500,.022,.023,1.235),(.547,.0195,.019,1.235),
        (.585,.0187,.014,1.235),(.610,.0188,.0124,1.235),
        (.629,.0190,.0120,1.235)]
    for sign in (-1,1):
        centers=[];ry=[];rz=[]
        for j in range(121):
            ax=.119+(.629-.119)*j/120
            a,b,z=[interp(arm_profile,ax,k) for k in (1,2,3)]
            centers.append((sign*ax,0,z));ry.append(a);rz.append(b)
        # Mirroring changes the winding; bmesh recalculates after the union.
        g.loft(centers,ry,rz,(0,1,0),(0,0,1),48)
        # Proximal thighs merge into the pelvis and overlap retained leg tops.
        centers=[];rx=[];ry=[]
        thigh_profile=[(.765,.060,.052,.009),(.805,.063,.056,.012),
                       (.850,.061,.056,.009),(.883,.053,.048,.006)]
        for j in range(35):
            z=.765+(.883-.765)*j/34
            a,b,cy=[interp(thigh_profile,z,k) for k in (1,2,3)]
            centers.append((sign*.068,cy,z));rx.append(a);ry.append(b)
        g.loft(centers,rx,ry,(1,0,0),(0,1,0),48)
    return g

def shin_foot_geometry(sign):
    g=Geometry()
    # The retained lower leg was trimmed at .284m; this restores real skin
    # under the sock and overlaps its original boundary above that height.
    profile=[(.094,.0200,.0225,.001),(.130,.0215,.0240,.000),
             (.180,.0230,.0242,.002),(.220,.0248,.0256,.005),
             (.260,.0275,.0280,.007),(.292,.0310,.0310,.009),
             (.306,.0324,.0321,.009)]
    centers=[];rx=[];ry=[]
    for j in range(61):
        z=.094+(.306-.094)*j/60
        a,b,cy=[interp(profile,z,k) for k in (1,2,3)]
        centers.append((sign*(.068+.006*math.exp(-((z-.36)/.12)**2)),cy,z))
        rx.append(a);ry.append(b)
    g.loft(centers,rx,ry,(1,0,0),(0,1,0),48)
    # A formed foot with heel, arch, instep and forefoot, fitted to the shoe.
    foot_profile=[(-.129,.025,.0085,.045),(-.110,.032,.012,.047),
        (-.082,.033,.016,.052),(-.050,.029,.021,.058),
        (-.020,.023,.031,.067),(.008,.021,.036,.070),
        (.030,.023,.030,.062),(.047,.017,.018,.054)]
    centers=[];rz=[];rx=[]
    for j in range(57):
        y=-.129+(.047+.129)*j/56
        a,b,z=[interp(foot_profile,y,k) for k in (1,2,3)]
        centers.append((sign*.068,y,z));rx.append(a);rz.append(b)
    g.loft(centers,rz,rx,(0,0,1),(1,0,0),48)
    # Five softly joined toe tips, retained as real geometry inside the shoe.
    for dx,y,rx,rlen in ((-.021,-.128,.0065,.013),
                         (-.008,-.130,.0053,.011),
                         (.004,-.127,.0050,.010),
                         (.015,-.120,.0047,.010),
                         (.024,-.112,.0042,.009)):
        g.ellipsoid((sign*(.068+dx),y,.043),(rx,rlen,.0065),24,14)
    # The retained rounded shoe last narrows sharply at its front. Preserve
    # the formed forefoot and toe silhouettes while fitting their length to
    # its real interior rather than projecting them through the leather tip.
    g.vertices=[(x, -.060+(y+.060)*.75 if y<-.060
                else y-.006*smooth(.022,.047,y),z) for x,y,z in g.vertices]
    g.vertices=[(sign*.068+(x-sign*.068)*(1-.14*(1-smooth(-.115,-.078,y))),y,z)
                for x,y,z in g.vertices]
    return g

def activate(obj):
    if bpy.context.object and bpy.context.object.mode!='OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True);bpy.context.view_layer.objects.active=obj

def topology(obj):
    bm=bmesh.new();bm.from_mesh(obj.data)
    bad=sum(not e.is_manifold for e in bm.edges)
    vertices=set(bm.verts);components=0
    while vertices:
        components+=1;stack=[vertices.pop()]
        while stack:
            v=stack.pop()
            for e in v.link_edges:
                n=e.other_vert(v)
                if n in vertices:vertices.remove(n);stack.append(n)
    volume=bm.calc_volume(signed=False)
    bm.free()
    return {'vertices':len(obj.data.vertices),'faces':len(obj.data.polygons),
            'nonmanifold_edges':bad,'connected_components':components,'volume_m3':volume}

def mesh_volume(name,geometry,collection,material,voxel,maximum):
    mesh=bpy.data.meshes.new(name+' Mesh')
    mesh.from_pydata(geometry.vertices,[],geometry.faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);collection.objects.link(obj)
    mesh.materials.append(material);activate(obj)
    bm=bmesh.new();bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(mesh);bm.free()
    remesh=obj.modifiers.new('Continuous original anatomical union','REMESH')
    remesh.mode='VOXEL';remesh.voxel_size=voxel;remesh.use_smooth_shade=True
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    relax=obj.modifiers.new('Soft anatomical surface','SMOOTH')
    relax.factor=.32;relax.iterations=3
    bpy.ops.object.modifier_apply(modifier=relax.name)
    if len(obj.data.vertices)>maximum:
        dec=obj.modifiers.new('Balanced body topology','DECIMATE')
        dec.ratio=maximum/len(obj.data.vertices)
        bpy.ops.object.modifier_apply(modifier=dec.name)
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(obj.data);bm.free()
    for p in obj.data.polygons:p.use_smooth=True
    activate(obj)
    bpy.ops.object.mode_set(mode='EDIT');bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(68),island_margin=.012)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.data.uv_layers.active.name='BodyAnatomyUV'
    obj.data.uv_layers.active.active_render=True
    return obj

def normalize(weights):
    entries=sorted(((n,w) for n,w in weights.items() if w>1e-7),
                   key=lambda a:a[1],reverse=True)[:4]
    total=sum(w for n,w in entries)
    if not total:raise ValueError('Unweighted completed body vertex')
    return {n:w/total for n,w in entries}

def body_weights(co,fields):
    x,y,z=co;ax=abs(x);side='left' if x>=0 else 'right'
    if z<.925:
        # Continuous pelvis-to-thigh deformation across the hip fold.
        thigh=(1-smooth(.796,.892,z))*smooth(.022,.060,ax)
        base={'hips':1-thigh,side+'UpperLeg':thigh}
        return base
    if ax>.22:
        weights=fields['_sleeve'](co,side)
        wrist=smooth(.604,.628,ax)
        weights={n:w*(1-wrist) for n,w in weights.items()}
        weights[side+'Hand']=wrist
        return weights
    if z>1.250 and ax<.060:
        neck=smooth(1.258,1.300,z)*.72
        return {'upperChest':1-neck,'neck':neck}
    return fields['_bodice'](co,side)

def foot_weights(co,side):
    x,y,z=co
    # The closed derby upper and outsole are Foot-rigid in the retained rig.
    # Enclosed toe tips follow that same rigid last; independent toe curling
    # would force them through the unchanged sole. Ankle skin follows socks.
    shin=smooth(.095,.150,z)
    return {side+'LowerLeg':shin,side+'Foot':1-shin}

def skin(obj,rig,field):
    planned=[normalize(field(v.co)) for v in obj.data.vertices]
    names=sorted({n for w in planned for n in w})
    if any(n not in rig.data.bones for n in names):raise ValueError('Missing expected humanoid bone')
    groups={n:obj.vertex_groups.new(name=n) for n in names}
    for index,weights in enumerate(planned):
        for name,weight in weights.items():groups[name].add([index],weight,'REPLACE')
    obj.parent=rig
    modifier=obj.modifiers.new('Humanoid skinning','ARMATURE')
    modifier.object=rig;modifier.use_deform_preserve_volume=False
    return {'bones':names,'max_influences':max(map(len,planned)),
            'max_normalization_error':max(abs(sum(w.values())-1) for w in planned)}

def apply(collection=None,rig=None):
    collection=collection or bpy.data.collections['CHARACTER · Original geometry']
    rig=rig or bpy.data.objects['Character_Rig']
    found=[bpy.data.objects.get(n) for n in NAMES]
    if any(found):
        if all(found) and all(o.get('body_completion_version')==VERSION for o in found):
            return json.loads(found[0]['body_completion_report'])
        raise ValueError('Partial or different body completion exists; preserve and inspect before replacement')
    if len(rig.data.bones)!=108:raise ValueError('Expected the retained 108-bone rig')
    retained_names=['Neck','Hand L · continuous anatomy','Hand R · continuous anatomy',
                    'Leg L · knee calf ankle','Leg R · knee calf ankle']
    if any(not bpy.data.objects.get(n) for n in retained_names):raise ValueError('Missing retained anatomical anchors')
    # A distinct copy lets the final material pass tune hidden anatomy together
    # without altering the retained neck or facial material network.
    material=bpy.data.objects['Neck'].data.materials[0].copy()
    material.name='Character / anatomical body skin'
    fields=runpy.run_path(str(Path(__file__).with_name('body_motion_refine.py')))
    previous_selection=list(bpy.context.selected_objects)
    previous_active=bpy.context.view_layer.objects.active
    built=[]
    try:
        specs=[(NAMES[0],trunk_geometry(),.0028,24000,lambda co:body_weights(co,fields),'trunk')]
        for label,sign,side in (('L',1,'left'),('R',-1,'right')):
            specs.append((PREFIX+'Shin_Foot_'+label,shin_foot_geometry(sign),.0018,6500,
                          lambda co,side=side:foot_weights(co,side),'shin_foot'))
        rows=[]
        for name,geometry,voxel,maximum,field,region in specs:
            obj=mesh_volume(name,geometry,collection,material,voxel,maximum)
            built.append(obj)
            topo=topology(obj)
            if topo['nonmanifold_edges'] or topo['connected_components']!=1:
                raise ValueError(name+': anatomical volume is not closed and connected '+str(topo))
            weights=skin(obj,rig,field)
            obj['part']='body'
            obj['deform_role']='completed_anatomy'
            obj['anatomy_region']=region
            obj['body_completion_version']=VERSION
            obj['original_mesh']=True
            obj['source']='Original anatomical lofts and volume union; no donor geometry'
            rows.append({'name':name,**topo,'skin':weights,'uv':obj.data.uv_layers.active.name})
        report={'version':VERSION,'objects':rows,
            'retained_geometry':retained_names,
            'original_objects_modified':False,
            'new_meshes_are_closed_connected':True,
            'connection_strategy':'Anatomically shaped overlap inside garments, not a globally welded replacement body.',
            'overlap_regions_m':{
                'neck_z':[1.252,1.285],'wrist_abs_x':[.597,.629],
                'upper_thigh_z':[.765,.850],'lower_shin_z':[.284,.306]},
            'covered_regions':['trunk, chest, back, shoulders, abdomen, pelvis',
                'upper arms, elbows and forearms','hip folds and proximal thighs',
                'missing lower shins, ankles, heels, arches, insteps, forefeet and five toe tips'],
            'preserved':['all original geometry and UVs','all facial expression keys',
                'existing hand/neck/leg skinning','108-bone hierarchy','spring and collider settings',
                'all clothing, shoes, socks and source materials'],
            'limits':['Retained body pieces remain independent meshes with concealed overlaps.',
                'Enclosed toes follow Foot with the existing rigid shoe; independent toe curls need compatible footwear deformation.',
                'No genital or nipple details are authored.',
                'Clothing clearance must be checked in the final motion and export; a closed body alone does not prove collision-free clothing.']}
        encoded=json.dumps(report)
        for obj in built:obj['body_completion_report']=encoded
        return report
    except Exception:
        # Only this call's newly created objects may be rolled back.
        for name in NAMES:
            obj=bpy.data.objects.get(name)
            if obj is not None:bpy.data.objects.remove(obj,do_unlink=True)
        if material.users==0:bpy.data.materials.remove(material)
        raise
    finally:
        if bpy.context.object and bpy.context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in previous_selection:
            if obj.name in bpy.data.objects:obj.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active=previous_active
