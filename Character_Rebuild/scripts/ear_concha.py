"""Isolated conservative anime ear-body refinement for the actual E4 scene.

Keeps ear objects/transforms, the YZ silhouette and the original ellipsoid
envelope; replaces the convex pill with an integral concha depression.
No other model, material, camera or scene setting is changed by apply().
"""
from math import sin,cos,pi,sqrt,exp


def apply():
    import bpy,bmesh
    report={'objects':[],'removedInnerPills':[],'style':'simplified anime concha, helix and tragus; no pores',
            'changesOutsideEars':False,'sceneSaved':False}
    for label in ('L','R'):
        ob=bpy.data.objects['Ear_'+label]
        if ob.type!='MESH' or ob.modifiers:
            raise ValueError('Expected unchanged mesh ear without modifiers: '+ob.name)
        old=ob.data
        mins=[min(v.co[a] for v in old.vertices) for a in range(3)]
        maxs=[max(v.co[a] for v in old.vertices) for a in range(3)]
        mid=[(a+b)/2 for a,b in zip(mins,maxs)]
        rx,ry,rz=[(b-a)/2 for a,b in zip(mins,maxs)]
        if not (.008<rx<.03 and .01<ry<.03 and .02<rz<.05):
            raise ValueError('Unexpected actual ear envelope')
        sign=1 if ob.location.x>0 else -1
        count=48
        radial=(.06,.14,.22,.30,.38,.46,.54,.62,.70,.78,.86,.93,.975,1.0)
        vertices=[];faces=[];uv_faces=[]
        def add(x,y,z):
            vertices.append((mid[0]+sign*x,mid[1]+ry*y,mid[2]+rz*z))
            return len(vertices)-1
        def uv(y,z,back=False):
            return ((.75 if back else .25)+.23*y,.5+.45*z)
        def depth(y,z):
            r=sqrt(y*y+z*z)
            # Intrinsic bowl volume, with an untouched outer silhouette band.
            bowl=.0090*exp(-((y+.10)/.50)**4-((z+.07)/.58)**4)
            q=max(0,min(1,(.88-r)/.13));support=q*q*(3-2*q)
            bowl*=support
            # Two broad internal folds model shape rather than pasted pieces.
            gate=max(0,min(1,(y+.12)/.35));gate=gate*gate*(3-2*gate)
            anti=.0032*exp(-((r-.45)/.12)**2)*gate
            tragus=.0032*exp(-((y+.44)/.18)**2-((z+.12)/.27)**2)
            return max(0,bowl-anti-tragus)
        def front_x(y,z):
            base=rx*sqrt(max(0,1-y*y-z*z))
            return max(base*.20,base-depth(y,z))
        center=add(front_x(0,0),0,0)
        prev=None
        for ri,r in enumerate(radial):
            ring=[]
            for j in range(count):
                a=j*2*pi/count;y=r*cos(a);z=r*sin(a)
                ring.append(add(front_x(y,z),y,z))
            if prev is None:
                for j in range(count):
                    k=(j+1)%count
                    faces.append((center,ring[j],ring[k]))
                    uv_faces.append((uv(0,0),uv(r*cos(j*2*pi/count),r*sin(j*2*pi/count)),uv(r*cos(k*2*pi/count),r*sin(k*2*pi/count))))
            else:
                for j in range(count):
                    k=(j+1)%count
                    faces.append((prev[j],ring[j],ring[k],prev[k]))
                    rp=radial[ri-1]
                    uv_faces.append((uv(rp*cos(j*2*pi/count),rp*sin(j*2*pi/count)),uv(r*cos(j*2*pi/count),r*sin(j*2*pi/count)),uv(r*cos(k*2*pi/count),r*sin(k*2*pi/count)),uv(rp*cos(k*2*pi/count),rp*sin(k*2*pi/count))))
            prev=ring
        # Exact same silhouette ring is shared by front and retained convex back.
        rprev=1.0
        for r in (.975,.93,.86,.76,.64,.50,.35,.18):
            ring=[]
            for j in range(count):
                a=j*2*pi/count;ring.append(add(-rx*sqrt(1-r*r),r*cos(a),r*sin(a)))
            for j in range(count):
                k=(j+1)%count
                faces.append((prev[j],ring[j],ring[k],prev[k]))
                uv_faces.append((uv(rprev*cos(j*2*pi/count),rprev*sin(j*2*pi/count),True),uv(r*cos(j*2*pi/count),r*sin(j*2*pi/count),True),uv(r*cos(k*2*pi/count),r*sin(k*2*pi/count),True),uv(rprev*cos(k*2*pi/count),rprev*sin(k*2*pi/count),True)))
            prev=ring;rprev=r
        center_back=add(-rx,0,0)
        for j in range(count):
            k=(j+1)%count;faces.append((prev[j],center_back,prev[k]))
            uv_faces.append((uv(rprev*cos(j*2*pi/count),rprev*sin(j*2*pi/count),True),uv(0,0,True),uv(rprev*cos(k*2*pi/count),rprev*sin(k*2*pi/count),True)))
        mesh=bpy.data.meshes.new(ob.name+'.ConchaQuadSurface')
        mesh.from_pydata(vertices,[],faces);mesh.update()
        layer=mesh.uv_layers.new(name='Ear_Surface_UV')
        for polygon,coords in zip(mesh.polygons,uv_faces):
            polygon.use_smooth=True
            for li,p in zip(polygon.loop_indices,coords):layer.data[li].uv=p
        for mat in old.materials:mesh.materials.append(mat)
        bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
        ob.data=mesh
        ob['inferred_detail']='Conservative stylized concha/helix carved inside original ear envelope; details inferred from anime design.'
        inner=bpy.data.objects.get('Ear_Inner_'+label)
        if inner:
            report['removedInnerPills'].append(inner.name)
            bpy.data.objects.remove(inner,do_unlink=True)
        report['objects'].append({'name':ob.name,'location':list(ob.location),
              'sourceLocalBounds':[mins,maxs],'radii':[rx,ry,rz],
              'vertices':len(mesh.vertices),'polygons':len(mesh.polygons),
              'uv':'Ear_Surface_UV','materials':[m.name for m in mesh.materials],
              'shape':'Integral concha depression; original YZ silhouette exact; all new front vertices within old ellipsoid envelope'})
    return report
