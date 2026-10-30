"""Clean, fitted Peter Pan collar leaves for the actual retained blouse.

Creates two closed, editable quad meshes. The existing body is only sampled,
never cut or saved. Side ends meet the current rebuilt rear collar band.
"""
import math
import statistics
import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _smoothstep(a,b,x):
    t=max(0.0,min(1.0,(x-a)/(b-a)))
    return t*t*(3-2*t)


def _curve(points,t):
    # Catmull-Rom with clamped end controls: rounded hem, not a pointed shirt collar.
    s=t*(len(points)-1); i=min(len(points)-2,int(s)); f=s-i
    a=Vector(points[max(0,i-1)]); b=Vector(points[i])
    c=Vector(points[i+1]); d=Vector(points[min(len(points)-1,i+2)])
    return .5*((2*b)+(-a+c)*f+(2*a-5*b+4*c-d)*f*f+(-a+3*b-3*c+d)*f*f*f)


def _signed_volume(vertices,faces):
    value=0.0
    for f in faces:
        a=Vector(vertices[f[0]])
        for i in range(1,len(f)-1):
            value+=a.dot(Vector(vertices[f[i]]).cross(Vector(vertices[f[i+1]])))/6
    return value


def add_clean_front_collar(body, material=None):
    """Return a serializable report; body geometry and original UVs are untouched."""
    if body.type!='MESH':
        raise TypeError('Expected the current source-retained blouse mesh')
    if any(bpy.data.objects.get('Rebuilt_FrontCollar_'+side) for side in ('L','R')):
        raise RuntimeError('Clean front collar already exists; preserve the previous stage')
    if material is None:
        material=bpy.data.materials.get('Rebuilt_blouse')
    if material is None:
        import materials
        material=materials.principled_material('Rebuilt_blouse',(.940,.890,.805),.88)
    bm=bmesh.new();bm.from_mesh(body.data)
    bmesh.ops.transform(bm,matrix=body.matrix_world,verts=list(bm.verts))
    tree=BVHTree.FromBMesh(bm)
    # Original design: two round front leaves, each ending beside the neck rather
    # than becoming an oversized shoulder cape. Width remains inside +/-0.118.
    outer_controls=[(.003,-.061,.589),(.025,-.093,.568),
                    (.075,-.091,.565),(.111,-.060,.583),
                    (.102,-.009,.609),(.076,.031,.622)]
    U,V=48,12
    created=[]; reports=[]
    for sign,label in ((1,'L'),(-1,'R')):
        top=[]; projections=0
        for iu in range(U+1):
            u=iu/U
            inner=Vector((.003+.052*math.sin(u*math.pi/2),
                          .030-.058*math.cos(u*math.pi/2),.615+.005*u))
            outer=_curve(outer_controls,u)
            contact=1-_smoothstep(.78,1,u)
            for iv in range(V+1):
                v=iv/V
                p=inner.lerp(outer,v)
                p.x*=sign
                # Slight rounded roll across the real cloth width.
                p.y-=.0018*math.sin(math.pi*v)*math.sin(math.pi*u)
                samples=[]
                for dx,dz in ((0,0),(-.002,0),(.002,0),(0,-.002),(0,.002)):
                    hit,n,index,d=tree.ray_cast(Vector((p.x+dx,-.8,p.z+dz)),Vector((0,1,0)))
                    if hit is not None and n.y<.35:
                        samples.append(hit.y)
                if samples and contact>0:
                    target=min(p.y,statistics.median(samples)-.0040)
                    p.y=p.y*(1-contact)+target*contact
                    projections+=1
                top.append(p)
        # Smooth the contact sampling in the longitudinal direction only; no body
        # vertices are changed. Side attachments and rounded hem shape remain fixed.
        for iteration in range(3):
            new=[p.copy() for p in top]
            for iu in range(1,U):
                for iv in range(V+1):
                    k=iu*(V+1)+iv
                    new[k].y=.6*top[k].y+.2*top[k-(V+1)].y+.2*top[k+(V+1)].y
            top=new
        # Final direct front-surface clearance on the front 78% of each leaf.
        for iu in range(U+1):
            if iu/U>.78:continue
            for iv in range(V+1):
                p=top[iu*(V+1)+iv]
                hit,n,index,d=tree.ray_cast(Vector((p.x,-.8,p.z)),Vector((0,1,0)))
                if hit is not None and n.y<.35:
                    p.y=min(p.y,hit.y-.0035)
        normals=[]
        for iu in range(U+1):
            for iv in range(V+1):
                du=top[min(U,iu+1)*(V+1)+iv]-top[max(0,iu-1)*(V+1)+iv]
                dv=top[iu*(V+1)+min(V,iv+1)]-top[iu*(V+1)+max(0,iv-1)]
                normal=du.cross(dv)
                if normal.length<1e-10:normal=Vector((0,-1,0))
                else:normal.normalize()
                if normal.y>0:normal=-normal
                normals.append(normal)
        thickness=.0015
        vertices=[tuple(p) for p in top]+[tuple(p-n*thickness) for p,n in zip(top,normals)]
        S=len(top);faces=[]
        for underside in (0,1):
            offset=underside*S
            for iu in range(U):
                for iv in range(V):
                    a=offset+iu*(V+1)+iv;b=a+V+1;c=b+1;d=a+1
                    faces.append((d,c,b,a) if underside else (a,b,c,d))
        for iu in range(U):
            a=iu*(V+1);b=(iu+1)*(V+1)
            faces.append((a,S+a,S+b,b))
            a=iu*(V+1)+V;b=(iu+1)*(V+1)+V
            faces.append((a,b,S+b,S+a))
        for iv in range(V):
            a=iv;b=iv+1;faces.append((a,b,S+b,S+a))
            a=U*(V+1)+iv;b=a+1;faces.append((a,S+a,S+b,b))
        if _signed_volume(vertices,faces)<0:faces=[tuple(reversed(f)) for f in faces]
        mesh=bpy.data.meshes.new('FrontCollar_'+label+'_EditableQuads')
        mesh.from_pydata(vertices,[],faces);mesh.update()
        layer=mesh.uv_layers.new(name='FrontCollar_UV')
        for poly in mesh.polygons:
            poly.use_smooth=True
            for li,vi in zip(poly.loop_indices,poly.vertices):
                base=vi%S;layer.data[li].uv=((base//(V+1))/U,(base%(V+1))/V)
        mesh.materials.append(material)
        cb=bmesh.new();cb.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(cb,faces=list(cb.faces))
        boundary=sum(e.is_boundary for e in cb.edges)
        nonmanifold=sum(not e.is_manifold for e in cb.edges)
        cb.to_mesh(mesh);cb.free();mesh.update()
        obj=bpy.data.objects.new('Rebuilt_FrontCollar_'+label,mesh)
        body.users_collection[0].objects.link(obj)
        obj['reconstruction_note']='New rounded Peter Pan front collar leaf, fitted to actual blouse, closed 1.5mm thickness in original source units. Existing body is uncut.'
        obj['inferred_detail']='Hidden seam/underside construction inferred; visible round collar design follows supplied front reference.'
        created.append(obj.name)
        reports.append({'object':obj.name,'vertices':len(vertices),'quads':len(faces),
                        'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,
                        'projected_samples':projections,'nominal_thickness':thickness,
                        'front_top_clearance':.0035,
                        'bounds':{'min':[min(p[k] for p in vertices) for k in range(3)],
                                  'max':[max(p[k] for p in vertices) for k in range(3)]}})
    bm.free()
    return {'created_objects':created,'leaves':reports,'body_geometry_changed':False,
            'body_uv_changed':False,'rear_collar_changed':False,
            'material':material.name,'bones_added':False,
            'body_texture_cleanup_region':{'absX':.16,'minZ':.555,'maxY':.035,
                'note':'Remove obsolete drawn collar fringe from the body atlas beneath new leaves with a feathered cream repair; preserve buttons below this region.'},
            'validation_note':'Closed topology checked here; root must inspect actual front, side and quarter views before delivery.'}
