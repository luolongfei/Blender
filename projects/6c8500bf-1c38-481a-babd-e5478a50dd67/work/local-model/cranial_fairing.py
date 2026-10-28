"""Restricted cranial contour fairing after anatomy_material_refine.apply.

Only the scalp above Z1.49 and posterior lower skull Y>.02/Z1.34..1.42
are authored. All facial keys receive the identical per-vertex Basis delta.
Hair, ears and hair rest joints follow the old-skull triangle displacement
field, with a smooth distance fade away from the skull. No live/file IO.
"""
import bpy, math, json
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

VERSION='cranial-fairing-02'
FACE='Face · sculpted planes & original illustration'

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)))
    return t*t*(3-2*t)

def pchip(xs,ys,value):
    """Monotone cubic interpolation, no new ripples between contour anchors."""
    if value<=xs[0]:return ys[0]
    if value>=xs[-1]:return ys[-1]
    h=np.diff(xs);d=np.diff(ys)/h
    m=np.zeros(len(xs))
    m[0]=d[0];m[-1]=d[-1]
    for i in range(1,len(xs)-1):
        if d[i-1]*d[i]>0:
            w1=2*h[i]+h[i-1];w2=h[i]+2*h[i-1]
            m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i])
    i=int(np.searchsorted(xs,value)-1)
    t=(value-xs[i])/h[i]
    return ((2*t**3-3*t*t+1)*ys[i]+(t**3-2*t*t+t)*h[i]*m[i]
            +(-2*t**3+3*t*t)*ys[i+1]+(t**3-t*t)*h[i]*m[i+1])

def prepare_contour(points):
    mid=points[(abs(points[:,0])<.00020)&(points[:,1]>.001)]
    order=np.argsort(mid[:,2]);mid=mid[order]
    if len(mid)<30:raise ValueError('Cannot identify the retained skull midline')
    zs=mid[:,2];ys=mid[:,1]
    apex=float(points[:,2].max())
    band=points[(points[:,2]>1.487)&(points[:,2]<1.493)]
    if len(band)<30:raise ValueError('Missing cap transition contour')
    cy=.005;cz=1.455
    scale=math.sqrt(1-((1.490-cz)/(apex-cz))**2)
    rx=float(abs(band[:,0]).max())/scale
    front=(cy-float(band[:,1].min()))/scale
    back=(float(band[:,1].max())-cy)/scale
    neck=bpy.data.objects.get('Neck')
    if neck:
        samples=np.array([v.co[:] for v in neck.data.vertices])
        ring=sorted(set(round(float(z),5) for z in samples[:,2]))
        neck_back=[]
        for z in ring:
            part=samples[abs(samples[:,2]-z)<.0001]
            neck_back.append((z,float(part[:,1].max())))
        ny=float(np.interp(1.348,[a[0] for a in neck_back],
                                  [a[1] for a in neck_back]))
    else:ny=.0385
    # Preserve each horizontal ring's original angular section.  Applying an
    # XY ellipse pointwise to the squarer forehead creates a diagonal ledge.
    # The shared per-ring height instead rounds the apex without shearing rings.
    cap_z=[];cap_target=[]
    for level in sorted(set(float(v) for v in points[points[:,2]>1.488,2])):
        ring_points=points[abs(points[:,2]-level)<1e-8]
        if len(ring_points)<16:continue
        ring_radius=float(abs(ring_points[:,0]).max())
        target=cz+(apex-cz)*math.sqrt(max(0.,1-(ring_radius/rx)**2))
        cap_z.append(level)
        cap_target.append(level+smooth(1.490,1.530,level)*(target-level))
    if len(cap_z)<15:raise ValueError('Cannot identify upper scalp rings')
    if any(b<=a for a,b in zip(cap_target,cap_target[1:])):
        raise ValueError('Fairing would invert upper scalp rings')
    # Remove the local posterior reversal.  Do not push a new shelf outside
    # the retained neck: the hidden lower skull already overlaps that neck.
    nx=[1.340,1.360,1.380,1.400,1.420]
    baseline=lambda z:float(np.interp(z,zs,ys))
    target=[baseline(z) for z in nx]
    if any(b<a for a,b in zip(target,target[1:])):
        raise ValueError('Nape target must be monotone')
    return {'apex':apex,'cap_center_y':cy,'cap_center_z':cz,
            'cap_x_radius':rx,'cap_front_radius':front,'cap_back_radius':back,
            'cap_ring_z':cap_z,'cap_ring_target_z':cap_target,
            'back_z':zs,'back_y':ys,'target_z':nx,'target_y':target,
            'neck_back_y_at_1_348':ny}

def fair_point(co,contour):
    x,y,z=map(float,co);out=Vector((x,y,z))
    if z>1.490:
        out.z=float(np.interp(z,contour['cap_ring_z'],contour['cap_ring_target_z']))
    if y>.020 and 1.340<z<1.420:
        old=float(np.interp(z,contour['back_z'],contour['back_y']))
        target=pchip(contour['target_z'],contour['target_y'],z)
        weight=smooth(.020,.028,y)*smooth(1.340,1.348,z)
        weight*=1-smooth(1.405,1.420,z)
        out.y+=(target-old)*weight
    return out

class Transfer:
    def __init__(self,face,old,delta):
        self.face=face;self.old=old;self.delta=delta
        face.data.calc_loop_triangles()
        self.triangles=[tuple(t.vertices) for t in face.data.loop_triangles]
        self.tree=BVHTree.FromPolygons(old.tolist(),self.triangles,all_triangles=True)
    def sample(self,point):
        hit,normal,index,distance=self.tree.find_nearest(point)
        if index is None or distance>=.105:return Vector()
        ids=self.triangles[index]
        a,b,c=(self.old[i] for i in ids)
        q=np.array(hit);v0=b-a;v1=c-a;v2=q-a
        d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1
        den=d00*d11-d01*d01
        if abs(den)<1e-18:
            w=np.array([1.,0,0])
        else:
            v=(d11*d20-d01*d21)/den
            u=(d00*d21-d01*d20)/den
            w=np.maximum(0.,np.array([1-v-u,v,u]))
            w/=w.sum()
        displacement=sum((self.delta[i]*weight for i,weight in zip(ids,w)),
                         start=np.zeros(3))
        return Vector(displacement)*(1-smooth(.028,.105,distance))

def coordinates(obj):
    points=(obj.data.shape_keys.key_blocks[0].data if obj.data.shape_keys
            else obj.data.vertices)
    return np.array([v.co[:] for v in points],dtype=np.float64)

def write_delta(obj,delta):
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            a=np.array([p.co[:] for p in key.data],dtype=np.float64)
            key.data.foreach_set('co',(a+delta).astype(np.float32).ravel())
        obj.data.vertices.foreach_set('co',np.array(
            [p.co[:] for p in obj.data.shape_keys.key_blocks[0].data],
            dtype=np.float32).ravel())
    else:
        obj.data.vertices.foreach_set('co',(coordinates(obj)+delta).astype(np.float32).ravel())
    obj.data.update()

def is_hair_bone(bone):
    """Include subdivided-chain joints and generated terminal descendants."""
    current=bone
    while current is not None:
        if current.name.startswith('hair_'):return True
        current=current.parent
    return False

def apply(collection=None,rig=None):
    collection=collection or bpy.data.collections['CHARACTER · Original geometry']
    rig=rig or bpy.data.objects['Character_Rig']
    face=bpy.data.objects[FACE]
    if face.get('cranial_fairing_version')==VERSION:
        return json.loads(face['cranial_fairing_report'])
    if face.get('cranial_fairing_version'):
        raise ValueError('Different cranial fairing already exists')
    if face.get('anatomy_material_version')!='anatomy-material-03':
        raise ValueError('Apply after anatomy_material_refine version 03')
    old=coordinates(face);contour=prepare_contour(old)
    new=np.array([fair_point(co,contour)[:] for co in old])
    delta=new-old
    protect=(old[:,2]<=1.490)&~((old[:,1]>.020)&(old[:,2]>1.340)&(old[:,2]<1.420))
    if np.max(abs(delta[protect]))>1e-12:
        raise ValueError('Protected face region moved')
    transfer=Transfer(face,old,delta)
    face_inverse=face.matrix_world.inverted()
    plans=[]
    for obj in list(collection.all_objects):
        if obj==face or obj.type!='MESH':continue
        if not (obj.get('part')=='hair' or obj.name.startswith(('Hair_','Ear '))):
            continue
        to_face=face_inverse@obj.matrix_world
        from_face=(obj.matrix_world.inverted()@face.matrix_world).to_3x3()
        source=coordinates(obj)
        moved=np.array([(from_face@transfer.sample(to_face@Vector(p)))[:] for p in source])
        if not np.isfinite(moved).all():raise ValueError('Nonfinite attachment displacement')
        plans.append((obj,moved))
    bones=[]
    to_face=face_inverse@rig.matrix_world
    from_face=(rig.matrix_world.inverted()@face.matrix_world).to_3x3()
    for bone in rig.data.bones:
        if not is_hair_bone(bone):continue
        hd=from_face@transfer.sample(to_face@bone.head_local)
        td=from_face@transfer.sample(to_face@bone.tail_local)
        bones.append((bone.name,bone.head_local.copy()+hd,bone.tail_local.copy()+td,
                      max(hd.length,td.length)))
    # The planned field is validated before any authored mesh is changed.
    write_delta(face,delta)
    rows=[]
    for obj,moved in plans:
        if np.max(abs(moved))>0:write_delta(obj,moved)
        rows.append({'object':obj.name,'moved_vertices':int(np.sum(np.linalg.norm(moved,axis=1)>1e-8)),
                     'max_displacement_m':float(np.linalg.norm(moved,axis=1).max())})
        obj['cranial_fairing_version']=VERSION
    previous_active=bpy.context.view_layer.objects.active
    selected=list(bpy.context.selected_objects)
    if bpy.context.object and bpy.context.object.mode!='OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    try:
        bpy.ops.object.select_all(action='DESELECT')
        rig.select_set(True);bpy.context.view_layer.objects.active=rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name,head,tail,maximum in bones:
            bone=rig.data.edit_bones[name];bone.head=head;bone.tail=tail
        bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if bpy.context.object and bpy.context.object.mode!='OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=previous_active
    report={'version':VERSION,'face_vertices_moved':int(np.sum(np.linalg.norm(delta,axis=1)>1e-8)),
            'max_face_displacement_m':float(np.linalg.norm(delta,axis=1).max()),
            'max_cap_displacement_m':float(np.linalg.norm(delta[old[:,2]>1.49],axis=1).max()),
            'max_nape_displacement_m':float(np.linalg.norm(delta[(old[:,2]<1.42)&(old[:,1]>.02)],axis=1).max()),
            'protected_face_max_displacement_m':float(abs(delta[protect]).max()),
            'cap_parameters':{k:v for k,v in contour.items() if k.startswith('cap_') or k=='apex'},
            'nape_anchors':list(zip(contour['target_z'],contour['target_y'])),
            'neck_back_reference_y':contour['neck_back_y_at_1_348'],
            'attachment_objects':rows,
            'hair_rest_bones':[{'name':n,'max_displacement_m':m} for n,h,t,m in bones],
            'transfer':'Nearest old-skull triangle barycentric displacement; full field within 28mm, smooth fade to zero at105mm.',
            'expression_method':'Identical Basis per-vertex delta added to every existing key.',
            'preserved':['eyes/nose/mouth/chin and anterior neck','all topology/UV/materials/skin weights',
                         'all humanoid rest bones','spring settings and collider parameters']}
    face['cranial_fairing_version']=VERSION
    face['cranial_fairing_report']=json.dumps(report)
    return report
