"""Continuous original ear cartilage for the existing 34-degree ear frame.

The surface contains the conchal bowl, antihelix forks, tragus and rolled helix.
It replaces the primitive pinna surface and its four separate decorative parts.
Face geometry, expressions, head scale, bones and spring settings are untouched.
Apply before the parent's unified head/ear width transformation.
"""
import bpy,bmesh,math,json
from mathutils import Vector

VERSION='ear-cartilage-04'

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)))
    return t*t*(3-2*t)

def gauss(u,v,cu,cv,ru,rv):
    return math.exp(-((u-cu)/ru)**2-((v-cv)/rv)**2)

def distance_path(u,v,points):
    result=1e6
    for (a,b),(c,d) in zip(points,points[1:]):
        dx=c-a;dy=d-b
        t=max(0.,min(1.,((u-a)*dx+(v-b)*dy)/(dx*dx+dy*dy)))
        result=min(result,math.hypot(u-a-t*dx,v-b-t*dy))
    return result

HELIX=[(-.011,.011),(-.010,.020),(-.006,.026),
       (.000,.029),(.007,.026),(.012,.019),(.014,.008),
       (.013,-.004),(.010,-.015),(.005,-.023)]
ANTIHELIX=[(.003,-.015),(.006,-.008),(.007,.001),
           (.005,.009),(.001,.016),(-.004,.020)]
FORK=[(.005,.009),(.009,.016),(.009,.022)]

def front_surface(u,v,r):
    # Depth is measured toward the back of the local pinna. Broad depressions
    # and softly seated ridges share one mesh, so no floating tubes appear.
    y=-.0040
    y+=.0060*gauss(u,v,.0025,-.005,.0050,.0090)
    y+=.0016*gauss(u,v,.0015,-.007,.0025,.0040)
    y+=.0016*gauss(u,v,.0045,.017,.0034,.0050)
    y-=.0049*math.exp(-(distance_path(u,v,HELIX)/.00225)**2)
    main=.0030*math.exp(-(distance_path(u,v,ANTIHELIX)/.0026)**2)
    fork=.0023*math.exp(-(distance_path(u,v,FORK)/.0023)**2)
    y-=main+fork-main*fork/.0030
    # A broad low tragus guards the auditory depression, with a smaller
    # antitragus under it and a shallow intertragic notch.
    y-=.0043*gauss(u,v,-.0010,-.005,.0024,.0045)
    y-=.0025*gauss(u,v,.003,-.016,.0032,.0034)
    y+=.0015*gauss(u,v,.000,-.014,.0020,.0026)
    y-=.0016*gauss(u,v,.000,-.025,.0065,.0065)
    # The outer contour is a thin rolled sheet; the posterior surface shares
    # this exact boundary. Preserve a quiet root region against the head.
    edge=smooth(.92,1.,r)
    y=y*(1-edge)+(-.0010)*edge
    return min(y,.0028)

def ear_geometry(sign):
    rows=28;sides=88
    verts=[];faces=[];uv=[]
    angle=math.radians(34)
    def point(u,v,y):
        # Identical orientation and centre to the already accepted body pass.
        attachment=smooth(-.012,-.002,u)
        # The old pinna centre was embedded in the retained wide head, which
        # hid the conchal bowl behind the cheek mesh. Seat the free cartilage
        # laterally while its inner attachment remains in the old position.
        x=.1025+u+.0085*attachment;z=1.405+v
        y-=.0045*attachment
        dx=x-.103;dy=y
        return (sign*(.103+math.cos(angle)*dx-math.sin(angle)*dy),
                .002+math.sin(angle)*dx+math.cos(angle)*dy,z)
    # Front/back each have one central vertex and radial quad rows.
    for front in (True,False):
        start=len(verts)
        anterior=front_surface(0,0,0)
        y=anterior if front else anterior+.0025-.35*anterior
        verts.append(point(0,0,y));uv.append((.5,.5))
        for row in range(1,rows+1):
            r=row/rows
            for k in range(sides):
                theta=math.tau*k/sides
                u=r*.016*(1+.045*math.sin(theta)) * math.cos(theta)
                v=r*.0305*(1+.045*math.cos(theta)) * math.sin(theta)
                anterior=front_surface(u,v,r)
                # Thin posterior cartilage follows the front relief softly;
                # it is not the flat back of a thick elliptical slab.
                y=anterior if front else (
                    anterior+(.0025-.35*anterior)*(1-smooth(.90,1.,r)))
                verts.append(point(u,v,y))
                uv.append((.5+u/.035,.5+v/.067))
        for k in range(sides):
            faces.append((start,start+1+k,start+1+(k+1)%sides))
        for row in range(rows-1):
            base=start+1+row*sides;nxt=base+sides
            for k in range(sides):
                kn=(k+1)%sides
                faces.append((base+k,nxt+k,nxt+kn,base+kn))
    # Weld shared perimeter indices, not a zero-thickness band of coincident
    # triangles. Central rows remain separate and give the pinna real volume.
    per=1+rows*sides
    front_edge=1+(rows-1)*sides;back_edge=per+front_edge
    mapping={back_edge+k:front_edge+k for k in range(sides)}
    faces=[tuple(mapping.get(i,i) for i in f) for f in faces]
    used=sorted({i for f in faces for i in f});index={old:i for i,old in enumerate(used)}
    return ([verts[i] for i in used],[tuple(index[i] for i in f) for f in faces],
            [uv[i] for i in used])

def build_mesh(name,sign):
    verts,faces,uv=ear_geometry(sign)
    mesh=bpy.data.meshes.new(name+' · continuous cartilage')
    mesh.from_pydata(verts,[],faces);mesh.update()
    bm=bmesh.new();bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bad=sum(not e.is_manifold for e in bm.edges)
    volume=bm.calc_volume(signed=False)
    bm.to_mesh(mesh);bm.free()
    if bad or volume<=0:
        bpy.data.meshes.remove(mesh)
        raise ValueError('Ear shell must be closed with positive volume')
    layer=mesh.uv_layers.new(name='EarAnatomyUV')
    for loop in mesh.loops:layer.data[loop.index].uv=uv[loop.vertex_index]
    for face in mesh.polygons:face.use_smooth=True
    return mesh,{'vertices':len(mesh.vertices),'faces':len(mesh.polygons),
                 'nonmanifold_edges':bad,'volume_m3':volume}

def apply(collection=None,rig=None):
    collection=collection or bpy.data.collections['CHARACTER · Original geometry']
    rig=rig or bpy.data.objects['Character_Rig']
    ears=[bpy.data.objects['Ear '+s] for s in ('L','R')]
    if all(o.get('ear_cartilage_version')==VERSION for o in ears):
        return json.loads(ears[0]['ear_cartilage_report'])
    if any(o.get('ear_cartilage_version') for o in ears):
        raise ValueError('Different/partial ear cartilage version exists')
    if any(o.data.shape_keys for o in ears):
        raise ValueError('Do not replace an ear carrying expression keys')
    if 'head' not in rig.data.bones:raise ValueError('Retained head bone missing')
    planned=[]
    for obj,sign in zip(ears,(1,-1)):
        mesh,stats=build_mesh(obj.name,sign)
        for mat in obj.data.materials:mesh.materials.append(mat)
        planned.append((obj,mesh,stats))
    report={'version':VERSION,'objects':[],
            'features':['continuous rolled helix','conchal bowl',
                        'antihelix and superior fork','tragus and antitragus',
                        'soft earlobe and posterior pinna'],
            'orientation_degrees':34,
            'preserved':['face and all expressions','head scale',
                         'rig/bone hierarchy','spring/collider settings'],
            'replaced_details':[]}
    for obj,mesh,stats in planned:
        old=obj.data
        old.use_fake_user=True
        obj['ear_cartilage_original_mesh']=old.name
        obj.data=mesh
        obj.vertex_groups.clear()
        group=obj.vertex_groups.new(name='head')
        group.add(list(range(len(mesh.vertices))),1.,'REPLACE')
        # The source already has an Armature modifier; retain it and avoid
        # unnecessary subdivision over this sufficiently sampled surface.
        for modifier in list(obj.modifiers):
            if modifier.type=='SUBSURF':obj.modifiers.remove(modifier)
        if not any(m.type=='ARMATURE' for m in obj.modifiers):
            modifier=obj.modifiers.new('Head deformation','ARMATURE')
            modifier.object=rig
        obj.parent=rig;obj['part']='head'
        obj['ear_cartilage_version']=VERSION
        report['objects'].append({'name':obj.name,**stats,
                                  'original_mesh':old.name,'weight':{'head':1.}})
    # Removal is narrowly scoped and occurs only after both valid ear shells
    # are present. Original mesh datablocks are retained as source backups.
    for label in ('L','R'):
        for prefix in ('Ear helix ','Ear concha '):
            obj=bpy.data.objects.get(prefix+label)
            if obj:
                obj.data.use_fake_user=True
                report['replaced_details'].append(obj.name)
                bpy.data.objects.remove(obj,do_unlink=True)
    encoded=json.dumps(report)
    for obj in ears:obj['ear_cartilage_report']=encoded
    return report
