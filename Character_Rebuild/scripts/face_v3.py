"""Reference-constrained stylised face construction; no old character dependency.

This module never reads/saves a .blend or changes any existing object. The caller
must supply measured head profiles and facial landmarks, then review the result.
Coordinates are Blender metres, X lateral, Z up, face toward -Y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=8)
def _spline_second_derivatives(rows):
    n=len(rows);lo=[0.]*n;diag=[1.]*n;hi=[0.]*n;rhs=[0.]*n
    for i in range(1,n-1):
        h0=rows[i][0]-rows[i-1][0];h1=rows[i+1][0]-rows[i][0]
        lo[i]=h0;diag[i]=2*(h0+h1);hi[i]=h1
        rhs[i]=6*((rows[i+1][1]-rows[i][1])/h1-(rows[i][1]-rows[i-1][1])/h0)
    for i in range(1,n):
        f=lo[i]/diag[i-1];diag[i]-=f*hi[i-1];rhs[i]-=f*rhs[i-1]
    m=[0.]*n;m[-1]=rhs[-1]/diag[-1]
    for i in range(n-2,-1,-1):m[i]=(rhs[i]-hi[i]*m[i+1])/diag[i]
    return tuple(m)


def _smooth_front(rows,z):
    rows=tuple(tuple(r) for r in rows)
    if z<=rows[0][0]:return rows[0][1]
    if z>=rows[-1][0]:return rows[-1][1]
    m=_spline_second_derivatives(rows)
    i=next(k for k in range(len(rows)-1) if rows[k][0]<=z<=rows[k+1][0])
    a,b=rows[i],rows[i+1];h=b[0]-a[0];u=(b[0]-z)/h;v=(z-a[0])/h
    return u*a[1]+v*b[1]+((u*u*u-u)*m[i]+(v*v*v-v)*m[i+1])*h*h/6


def _lerp(a, b, t):
    return a + (b - a) * t


def _profile_at(rows, z):
    """C2 interpolation; no per-interval value clamps or zeroed knot slopes."""
    return tuple(_smooth_front([[r[0],r[c]] for r in rows],z)
                 for c in range(1,len(rows[0])))


@dataclass
class FaceSurface:
    """profile rows: [z, halfWidth, centreY, frontDepth, backDepth]."""
    profile: list
    nose: dict
    cheek: dict | None = None
    front_curve: list | None = None
    width_curve: list | None = None
    rear_curve: list | None = None
    side_curve: list | None = None

    def section(self,z):
        w,cy,fd,bd=_profile_at(self.profile,z)
        if self.width_curve:w=_smooth_front(self.width_curve,z)
        front=_smooth_front(self.front_curve,z) if self.front_curve else cy-fd
        rear=_smooth_front(self.rear_curve,z) if self.rear_curve else cy+bd
        if self.side_curve:cy=_smooth_front(self.side_curve,z)
        if w<=0 or rear<=front:raise ValueError('Invalid faired head cross-section')
        return w,cy,(rear-front)/2,(front+rear)/2

    def front(self, x, z):
        w,cy,radius,mid=self.section(z)
        c=math.sqrt(max(0,1-(x/max(w,1e-8))**2))
        # Unified smooth circumference. Sparse C2 front, rear, side and width
        # controls remove the horizontal curvature bands inherited from noisy
        # cross-section columns without changing the eye/mouth landmarks.
        y = cy-radius*c-(cy-mid)*c*c
        fade=c*c
        n = self.nose
        if n:
            # A shaped bridge/tip, not a uniformly swollen facial centre.
            for comp in n.get('lobes', []):
                nx,nz,sx,sz,depth = comp
                y -= fade*depth*math.exp(-((x-nx)/sx)**2-((z-nz)/sz)**2)
        if self.cheek:
            for cx in self.cheek['centres_x']:
                y -= fade*self.cheek['depth']*math.exp(-((x-cx)/self.cheek['width'])**2
                     -((z-self.cheek['z'])/self.cheek['height'])**2)
        return y

    def point(self, theta, z):
        w,cy,radius,mid=self.section(z)
        x = w*math.sin(theta)
        c=math.cos(theta)
        y=self.front(x,z) if c>=0 else cy-radius*c-(cy-mid)*c*c
        return (x,y,z)


def _almond(feature, theta, scale=1.):
    """Feature perimeter with independently specified upper/lower lid height."""
    c,s = math.cos(theta), math.sin(theta)
    x = feature['x']+feature['half_width']*c*scale
    h = feature['upper'] if s>=0 else feature['lower']
    shaped_s=math.copysign(abs(s)**feature.get('almond_power',1.),s)
    z = (feature['z']+h*shaped_s*scale+feature.get('tilt',0)*c*scale
         +feature.get('corner_lift',0)*c*c*scale*scale)
    return x,z


def _find_rectangle(feature, thetas, zs, surface, margin=1.55):
    z0 = feature['z']-(feature['lower']+abs(feature.get('tilt',0)))*margin
    z1 = feature['z']+(feature['upper']+abs(feature.get('tilt',0))+feature.get('corner_lift',0))*margin
    width = surface.section(feature['z'])[0]
    x0,x1 = feature['x']-feature['half_width']*margin,feature['x']+feature['half_width']*margin
    t0,t1=math.asin(max(-.94,x0/width)),math.asin(min(.94,x1/width))
    lo=lambda values,value: max(i for i,v in enumerate(values) if v<=value)
    hi=lambda values,value: min(i for i,v in enumerate(values) if v>=value)
    return lo(thetas,t0),hi(thetas,t1),lo(zs,z0),hi(zs,z1)


def geometry(spec):
    """Return new mesh dictionaries, explicitly no scene mutation.

    Required: profile, nose, eyes (two landmark dicts), mouth, neck/hair end caps.
    Eye/mouth dicts: name,x,z,half_width,upper,lower,tilt,bulge.
    Front diffuse UVs are provided in a separate UV layer by the Blender wrapper.
    """
    profile=spec['profile']
    assert len(profile)>=6 and all(profile[i][0]<profile[i+1][0] for i in range(len(profile)-1))
    assert len(spec['eyes'])==2
    surface=FaceSurface(profile,spec.get('nose',{}),spec.get('cheek'),spec.get('front_curve'),spec.get('width_curve'),spec.get('rear_curve'),spec.get('side_curve'))
    nt,nz=spec.get('radial_segments',96),spec.get('height_segments',96)
    if nt%4: raise ValueError('radial_segments must be divisible by four')
    thetas=[-math.pi+2*math.pi*i/nt for i in range(nt)]
    zs=[_lerp(profile[0][0],profile[-1][0],j/nz) for j in range(nz+1)]
    features=spec['eyes']+[spec['mouth']]
    rectangles=[_find_rectangle(f,thetas,zs,surface,1.65 if i<2 else 2.3) for i,f in enumerate(features)]
    for a in range(len(rectangles)):
        for b in range(a):
            i0,i1,j0,j1=rectangles[a]; k0,k1,l0,l1=rectangles[b]
            if max(i0,k0)<min(i1,k1) and max(j0,l0)<min(j1,l1):
                raise ValueError('Feature transition regions overlap; adjust margin or landmarks')
    verts=[surface.point(t,z) for z in zs for t in thetas]
    faces=[]
    for j in range(nz):
        for i in range(nt):
            if any(a<=i<b and c<=j<d for a,b,c,d in rectangles): continue
            faces.append((j*nt+i,j*nt+(i+1)%nt,(j+1)*nt+(i+1)%nt,(j+1)*nt+i))
    ring_data=[]
    for feature,rect in zip(features,rectangles):
        a,b,c,d=rect
        boundary=([c*nt+i for i in range(a,b)]+[j*nt+b for j in range(c,d)]
                 +[d*nt+i for i in range(b,a,-1)]+[j*nt+a for j in range(d,c,-1)])
        old=boundary
        # Preserve existing quadrilateral grid boundary. Inner rings establish
        # eyelid/lip contour flow, rather than merely subdividing noisy triangles.
        xmin,xmax=min(verts[i][0] for i in boundary),max(verts[i][0] for i in boundary)
        zmin,zmax=min(verts[i][2] for i in boundary),max(verts[i][2] for i in boundary)
        angles=[math.atan2((verts[i][2]-(zmin+zmax)/2)/((zmax-zmin)/2),
                           (verts[i][0]-(xmin+xmax)/2)/((xmax-xmin)/2)) for i in boundary]
        rim=feature.get('rim_recess',.00035)
        # Support the new quad patch over a continuous radial transition. The
        # extra local rings sample curvature; they do not change head silhouette.
        transition=[]
        for t in [.20,.40,.60,.75,.86,.93,.98,1.]:
            q=max(0,min(1,(t-.60)/.40));recess=rim*q*q*(3-2*q)
            transition.append((t,recess))
        for amount,recess in transition:
            new=[]
            for k,angle in enumerate(angles):
                tx,tz=_almond(feature,angle,1.)
                bx,by,bz=verts[boundary[k]]
                x,z=_lerp(bx,tx,amount),_lerp(bz,tz,amount)
                new.append(len(verts));verts.append((x,surface.front(x,z)+recess,z))
            faces.extend((old[k],old[(k+1)%len(old)],new[(k+1)%len(old)],new[k]) for k in range(len(old)))
            old=new
        # A short rearward socket wall hides the seam and gives the rim thickness.
        inner=[]
        for index in old:
            x,y,z=verts[index];inner.append(len(verts));verts.append((x,y+feature.get('socket_depth',.002),z))
        faces.extend((old[k],old[(k+1)%len(old)],inner[(k+1)%len(old)],inner[k]) for k in range(len(old)))
        ring_data.append({'feature':feature,'angles':angles,'rim':old,'socket':inner})
    # Top is covered by hair; neck bottom is intentionally open for body junction.
    top=len(verts);w,cy,fd,bd=_profile_at(profile,profile[-1][0]);verts.append((0,cy,profile[-1][0]+spec.get('crown_cap_height',.002)))
    faces.extend((nz*nt+i,nz*nt+(i+1)%nt,top) for i in range(nt))
    if spec.get('close_chin', False):
        bottom=len(verts);w,cy,fd,bd=_profile_at(profile,profile[0][0])
        verts.append((0,cy,profile[0][0]-.0002))
        faces.extend(((i+1)%nt,i,bottom) for i in range(nt))
    # Remove unused vertices left inside the removed eye/mouth grid rectangles.
    used=sorted(set(i for f in faces for i in f));mapping={v:i for i,v in enumerate(used)}
    skin={'name':'Head_Skin_Quad_Retopo','vertices':[verts[i] for i in used],
          'faces':[tuple(mapping[i] for i in f) for f in faces], 'role':'skin'}
    result=[skin]
    if spec.get('neck_profile'):
        result.append(_neck_geometry(spec['neck_profile']))
    for n,eye in enumerate(spec['eyes']):
        result.extend(_eye_geometry(surface,eye,n))
        result.append(_upper_lash_geometry(surface,eye,n))
    for n,brow in enumerate(spec.get('brows', [])):
        result.append(_brow_geometry(surface,brow,n))
    result.append(_mouth_geometry(surface,spec['mouth']))
    return result,surface


def _neck_geometry(profile):
    """Conservative separate neck sleeve, rooted in the input neck section.

    The upper ring is inserted inside the rear lower head. Keeping this editable
    shell separate avoids widening the reference's pointed chin into a cylinder.
    """
    nr,nt=24,64;vertices=[];faces=[]
    for j in range(nr+1):
        z=_lerp(profile[0][0],profile[-1][0],j/nr)
        w,cy,fd,bd=_profile_at(profile,z)
        for i in range(nt):
            a=2*math.pi*i/nt;c=math.cos(a)
            vertices.append((w*math.sin(a),cy-.5*(fd+bd)*c-.5*(fd-bd)*c*c,z))
    for j in range(nr):
        for i in range(nt):faces.append((j*nt+i,j*nt+(i+1)%nt,(j+1)*nt+(i+1)%nt,(j+1)*nt+i))
    return {'name':'Neck_Clean_Quad_Sleeve','vertices':vertices,'faces':faces,'role':'skin_back'}


def _eye_surface(surface,eye,x,z):
    u=(x-eye['x'])/eye['half_width']
    h=eye['upper'] if z>=eye['z'] else eye['lower']
    v=(z-eye['z']-eye.get('tilt',0)*u)/h
    q=u*u+abs(v)**(2/eye.get('almond_power',1.))
    return surface.front(x,z)+eye.get('rim_recess',.00035)-eye['bulge']*max(0,1-q)


def _eye_geometry(surface,eye,index):
    label=eye.get('name','L' if index==0 else 'R')
    nr,nt=12,64
    v=[(eye['x'],_eye_surface(surface,eye,eye['x'],eye['z']),eye['z'])];f=[]
    for j in range(1,nr+1):
        for i in range(nt):
            x,z=_almond(eye,2*math.pi*i/nt,j/nr)
            v.append((x,_eye_surface(surface,eye,x,z),z))
    for i in range(nt):f.append((0,1+i,1+(i+1)%nt))
    for j in range(nr-1):
        a=1+j*nt;b=a+nt
        f.extend((a+i,b+i,b+(i+1)%nt,a+(i+1)%nt) for i in range(nt))
    out=[{'name':f'Eye_Sclera_{label}','vertices':v,'faces':f,'role':'sclera'}]
    # Iris is a separate curved disk, clipped at the actual eyelid aperture.
    iris=eye.get('iris',{})
    ix,iz=eye['x']+iris.get('dx',0),eye['z']+iris.get('dz',0)
    rx,rz=iris.get('radius_x',eye['half_width']*.57),iris.get('radius_z',max(eye['upper'],eye['lower'])*1.02)
    iv=[(ix,_eye_surface(surface,eye,ix,iz)-.00004,iz)];iff=[]
    for j in range(1,9):
        for i in range(nt):
            angle=2*math.pi*i/nt
            low,high=0.,1.
            for step in range(24):
                r=(low+high)/2
                tx,tz=ix+rx*math.cos(angle)*r,iz+rz*math.sin(angle)*r
                u=(tx-eye['x'])/eye['half_width'];h=eye['upper'] if tz>=eye['z'] else eye['lower']
                q=u*u+abs((tz-eye['z']-eye.get('tilt',0)*u)/h)**(2/eye.get('almond_power',1.))
                if q>1:high=r
                else:low=r
            r=low
            r*=j/8
            x,z=ix+rx*math.cos(angle)*r,iz+rz*math.sin(angle)*r
            iv.append((x,_eye_surface(surface,eye,x,z)-.00005,z))
    for i in range(nt):iff.append((0,1+i,1+(i+1)%nt))
    for j in range(7):
        a=1+j*nt;b=a+nt;iff.extend((a+i,b+i,b+(i+1)%nt,a+(i+1)%nt) for i in range(nt))
    out.append({'name':f'Eye_Iris_{label}','vertices':iv,'faces':iff,'role':'iris',
                'iris_uv':{'x':ix,'z':iz,'rx':rx,'rz':rz}})
    return out


def _mouth_geometry(surface,mouth):
    # A strip follows the shallow smile; a triangle fan would fill across the
    # concave curve and turn a fine closed mouth into a thick dark wedge.
    v=[];rows=[];f=[];n=48
    for i in range(n+1):
        u=-1+2*i/n;x=mouth['x']+mouth['half_width']*u
        z=mouth['z']+mouth.get('corner_lift',0)*u*u+mouth.get('tilt',0)*u
        h=math.sqrt(max(0,1-u*u))
        if i in (0,n):
            rows.append([len(v)]);v.append((x,surface.front(x,z)+mouth.get('rim_recess',.00035)+.0002,z))
        else:
            rows.append([len(v),len(v)+1])
            for zz in [z-mouth['lower']*h,z+mouth['upper']*h]:
                v.append((x,surface.front(x,zz)+mouth.get('rim_recess',.00035)+.0002,zz))
    f.append((rows[0][0],rows[1][0],rows[1][1]))
    for i in range(1,n-1):f.append((rows[i][0],rows[i+1][0],rows[i+1][1],rows[i][1]))
    f.append((rows[-2][0],rows[-1][0],rows[-2][1]))
    return {'name':'Mouth_Closed_Line','vertices':v,'faces':f,'role':'mouth'}


def _upper_lash_geometry(surface,eye,index):
    """A tapered, skin-following upper lash ribbon in exportable quads."""
    label=eye.get('name','L' if index==0 else 'R')
    vertices=[];faces=[];n=40
    for i in range(n+1):
        a=math.pi*i/n
        x,z=_almond(eye,a)
        width=eye.get('lash_width',.0008)*(.035+.965*max(0,math.sin(a))**.45)
        rim=eye.get('rim_recess',.00035)
        vertices.extend([(x,surface.front(x,z)+rim-.00008,z),
                         (x,surface.front(x,z+width)+rim*.30-.00010,z+width)])
    faces.extend((2*i,2*i+1,2*i+3,2*i+2) for i in range(n))
    return {'name':f'Eye_UpperLash_{label}','vertices':vertices,'faces':faces,'role':'lash'}


def _brow_geometry(surface,brow,index):
    """Measured Bézier eyebrow, thin stylised ribbon rather than hair particles."""
    control=brow['points']
    if len(control)!=3:raise ValueError('Brow needs three reference-measured x,z control points')
    vertices=[];faces=[];n=32
    for i in range(n+1):
        t=i/n
        x=(1-t)**2*control[0][0]+2*(1-t)*t*control[1][0]+t*t*control[2][0]
        z=(1-t)**2*control[0][1]+2*(1-t)*t*control[1][1]+t*t*control[2][1]
        w=brow['width']*(.14+.86*math.sin(math.pi*t)**.5)
        vertices.extend([(x,surface.front(x,z-w/2)-.00012,z-w/2),
                         (x,surface.front(x,z+w/2)-.00012,z+w/2)])
    faces.extend((2*i,2*i+1,2*i+3,2*i+2) for i in range(n))
    return {'name':f'Brow_{brow.get("name",index)}','vertices':vertices,'faces':faces,'role':'brow'}


def build_face(spec, collection, materials):
    """Build into caller-owned collection, return objects and construction report.

    materials maps skin/skin_back/sclera/iris/mouth/lash/brow to prepared materials.
    skin is a masked/repainted face texture, NOT an unmasked full reference image.
    The rear hemisphere uses skin_back to prohibit facial projection on the back.
    spec['front_uv'] is {left,right,bottom,top}, measured in Blender metres.
    No external files, scene replacement, automatic source saves or rendering.
    """
    import bpy
    import bmesh
    pieces,surface=geometry(spec)
    objects=[]
    uvbox=spec['front_uv']
    for part in pieces:
        mesh=bpy.data.meshes.new(part['name']+'_Mesh')
        mesh.from_pydata(part['vertices'],[],part['faces']);mesh.update()
        bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
        obj=bpy.data.objects.new(part['name'],mesh);collection.objects.link(obj)
        material=materials.get(part['role'])
        if material:mesh.materials.append(material)
        if part['role']=='skin':
            if 'skin_back' not in materials:raise ValueError('skin_back material required; never project front features on back')
            mesh.materials.append(materials['skin_back'])
        uv=mesh.uv_layers.new(name='Face_Reference_UV' if part['role']!='iris' else 'Iris_UV')
        for poly in mesh.polygons:
            poly.use_smooth=True
            if part['role']=='skin':
                cy=surface.section(poly.center.z)[1]
                # Back and far lateral faces use clean base skin, never front UV.
                if poly.center.y>=cy:
                    poly.material_index=1
            for li in poly.loop_indices:
                x,y,z=mesh.vertices[mesh.loops[li].vertex_index].co
                if part['role']=='iris':
                    q=part['iris_uv'];u=.5+(x-q['x'])/(2*q['rx']);v=.5+(z-q['z'])/(2*q['rz'])
                else:
                    u=(x-uvbox['left'])/(uvbox['right']-uvbox['left']);v=(z-uvbox['bottom'])/(uvbox['top']-uvbox['bottom'])
                uv.data[li].uv=(u,v)
        obj['rebuild_source']='New measured GLB and anime reference only'
        obj['rebuild_role']=part['role'];obj['review_status']='unreviewed construction candidate'
        objects.append(obj)
    return objects,{'new_mesh_objects':[o.name for o in objects],
                    'vertices':sum(len(o.data.vertices) for o in objects),
                    'polygons':sum(len(o.data.polygons) for o in objects),
                    'quad_faces':sum(len(p.vertices)==4 for o in objects for p in o.data.polygons),
                    'neck_boundary':'intentionally open for body integration',
                    'saved_source':False,'appearance_accepted':False}
