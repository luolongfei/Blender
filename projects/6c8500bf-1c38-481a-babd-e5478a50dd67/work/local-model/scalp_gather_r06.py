"""Replace sparse R03 scalp coverage and the thin ear-to-tail connections.

Candidate-only, no save/render/cache bake. Uses the CURRENT evaluated upper head
and actual native tail pin coordinates. Leaves forelocks, wisps and all tail
points unchanged. New objects inherit the existing head-bone attachment.
The real head surface supplies a recessed root substrate; native fibres supply
the visible directional layer. Broad gather bundles end on each tail root.
"""
import math
import json
import random
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

OWNER = 'R06_scalp_gather'
TAILS = ('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
REPLACE = ('R03 Scalp gather guides','R03 Crown soft coverage',
           'R03 scalp root substrate','R03 soft transition fibres')


def _smooth(x):
    x=max(0.,min(1.,float(x)))
    return x*x*(3-2*x)


def _unit(x):
    return x/max(float(np.linalg.norm(x)),1e-10)


def _read_tail(obj):
    sizes=[c.points_length for c in obj.data.curves]
    if len(sizes)!=2480 or set(sizes)!={35}:
        raise RuntimeError('Expected current 2480 x 35 native tail: '+obj.name)
    raw=np.empty(len(obj.data.position_data)*3,dtype=np.float64)
    obj.data.position_data.foreach_get('vector',raw)
    local=raw.reshape(-1,3); matrix=np.array(obj.matrix_world)
    world=local@matrix[:3,:3].T+matrix[:3,3]
    return world.reshape(2480,35,3)


def _attach(obj, example):
    obj.parent=example.parent;obj.parent_type=example.parent_type
    obj.parent_bone=example.parent_bone
    obj.matrix_parent_inverse=example.matrix_parent_inverse.copy()
    obj.matrix_basis=example.matrix_basis.copy()


def _local(world, example):
    inv=np.array(example.matrix_world.inverted())
    a=np.asarray(world,dtype=np.float64)
    return a@inv[:3,:3].T+inv[:3,3]


def _curves(name, paths, material, example, radius, taper, created):
    data=bpy.data.hair_curves.new(name)
    data.add_curves([len(p) for p in paths])
    data.position_data.foreach_set('vector',_local(np.concatenate(paths),example).ravel())
    attr=data.attributes.new('radius','FLOAT','POINT')
    values=[]
    for i,path in enumerate(paths):
        variation=.83+.29*((i*.61803398875)%1)
        for j in range(len(path)):
            t=j/(len(path)-1)
            end=max(.08,(1-t**2.4)**.6) if taper else .9+.1*math.sin(math.pi*t)
            root=.62+.38*_smooth(t/.09)
            values.append(radius*variation*root*end)
    attr.data.foreach_set('value',values)
    data.materials.append(material)
    obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj)
    _attach(obj,example);obj[OWNER]=True
    obj.hide_render=True;obj.hide_set(True);created.append(obj)
    return obj


def run(diagnostics_only=False):
    if not bpy.data.filepath or bpy.data.filepath.replace('\\','/').endswith('/work/character.blend'):
        raise RuntimeError('Use the current saved independent R05 work copy')
    if any(o.get(OWNER) for o in bpy.context.scene.objects):
        raise RuntimeError('R06 scalp/gather already exists; use restore() before retrying')
    head=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig']
    tails=[bpy.data.objects[name] for name in TAILS]
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    created=[]
    try:
        pins=[_read_tail(o) for o in tails]
        roots=np.array([p[:,0].mean(axis=0) for p in pins])
        evaluated=head.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        try:
            vertices=np.array([evaluated.matrix_world@v.co for v in mesh.vertices])
            faces=[tuple(p.vertices) for p in mesh.polygons]
        finally:evaluated.to_mesh_clear()
        top=float(vertices[:,2].max())
        scale=(top-float(roots[:,2].mean()))/.158
        if not .85<scale<1.35:
            raise RuntimeError('Head/tail root relationship outside the intended candidate range')
        center=np.array([float(roots[:,0].mean()),float(roots[:,1].mean())+.007*scale,
                         top-.108*scale])
        bvh=BVHTree.FromPolygons([Vector(v) for v in vertices],faces)
        rng=random.Random(606028)

        def boundary(q):
            d=q-center
            phi=math.atan2(d[1],abs(d[0]))
            front=max(0.,-math.sin(phi));back=max(0.,math.sin(phi))
            ear_guard=.007*math.exp(-((phi-.18)/.28)**2)
            return center[2]+scale*(.017+.030*front-.040*back+ear_guard)

        def surface(theta,phi,side,lift=0.):
            direction=Vector((side*math.sin(theta)*math.cos(phi),
                              math.sin(theta)*math.sin(phi),math.cos(theta)))
            hit,normal,_,_=bvh.ray_cast(Vector(center)+direction*.40,-direction)
            if hit is None:
                raise RuntimeError('R06 scalp radial ray missed; no projection fallback')
            return np.array(hit+normal*lift),np.array(normal)

        # Each side gets a continuous angular hairline table. Rays are confined
        # to the actual head dome; there is no per-point front/height fallback.
        phi_table=np.linspace(-math.pi/2+.006,math.pi/2-.006,129)
        limits={}
        for side in (1,-1):
            values=[]
            for phi in phi_table:
                lo,hi=.02,2.05
                for _ in range(11):
                    mid=(lo+hi)/2;q,_=surface(mid,float(phi),side)
                    if q[2]>=boundary(q):lo=mid
                    else:hi=mid
                values.append(lo)
            limits[side]=values

        def guide_point(theta,phi,side,lift):
            limit=float(np.interp(phi,phi_table,limits[side]))-.018
            return surface(min(theta,limit),phi,side,lift)

        # The substrate follows actual evaluated geometry. Its anterior edge is
        # hidden under the existing forelocks; the ear guard leaves cartilage bare.
        selected=[]
        for face in faces:
            if all(vertices[i,2]>=boundary(vertices[i])-.00025*scale for i in face):
                selected.append(face)
        used=sorted({i for face in selected for i in face})
        if len(used)<200:
            raise RuntimeError('Insufficient actual scalp surface selected')
        remap={old:new for new,old in enumerate(used)}
        shell=[]
        for i in used:
            q=vertices[i];shell.append(q+_unit(q-center)*(.00035*scale))

        # Uniformly distributed roots and continuously curving surface guides.
        # Terminal guides occupy a broad posterior patch rather than one point.
        scalp=[]
        for side in (1,-1):
            for sector in range(12):
                phi0=-1.50+(sector+.5)/12*3.0
                limit=float(np.interp(phi0,phi_table,limits[side]))
                for row in range(8):
                    theta0=.025+(limit-.12)*math.sqrt((row+.3)/8)
                    end_phi=.53+.31*((sector*5+row*3)%13)/12
                    end_theta=1.57+.19*((row*5+sector)%11)/10
                    guide=[];normals=[]
                    for j in range(47):
                        t=j/46;ease=t*t*(3-2*t)
                        phi=phi0+(end_phi-phi0)*ease
                        theta=theta0+(end_theta-theta0)*t
                        q,n=guide_point(theta,phi,side,(.0010+.0006*math.sin(math.pi*t))*scale)
                        guide.append(q);normals.append(n)
                    guide=np.array(guide);normals=np.array(normals)
                    tangent=np.gradient(guide,axis=0)
                    lateral=np.cross(normals,tangent)
                    lateral/=np.maximum(np.linalg.norm(lateral,axis=1,keepdims=True),1e-9)
                    for strand in range(32):
                        across=rng.uniform(-.0018,.0018)*scale
                        depth=rng.uniform(.0001,.0007)*scale
                        phase=rng.uniform(0,math.tau)
                        path=[]
                        for j,q in enumerate(guide):
                            t=j/46
                            path.append(q+lateral[j]*(across+.00012*scale*math.sin(phase+t*4))
                                        +normals[j]*depth)
                        scalp.append(np.array(path))
        # Cover the pole with short, differently oriented overlaps, not a top seam.
        for side in (1,-1):
            for group in range(24):
                phi=-1.50+3.0*(group+.5)/24
                for strand in range(12):
                    path=[]
                    for j in range(28):
                        t=j/27
                        q,n=guide_point(.003+t*.72,phi+(.65-phi)*t*.3,side,
                                        (.0013+.0006*(strand/11))*scale)
                        q+=np.array([math.sin(group+strand)*.00035*scale,0,0])*(1-t)
                        path.append(q)
                    scalp.append(np.array(path))

        # One static connecting strand for every original tail root. The final
        # two samples follow the actual first tail segment, so there is no air
        # gap or single-wire bridge between ear-back hair and the tail mass.
        gathers=[]
        gather_reports=[]
        for index,side in enumerate((1,-1)):
            tail=pins[index];paths=[]
            for group in range(8):
                phi=.55+.27*((group*3)%8)/7
                theta=1.57+.17*((group*5)%8)/7
                anchor,normal=guide_point(theta,phi,side,.0014*scale)
                patch_u=_unit(np.cross(normal,np.array([0,0,1.])))
                patch_v=_unit(np.cross(normal,patch_u))
                for k in range(310):
                    strand=group*310+k
                    root=tail[strand,0]
                    direction=_unit(tail[strand,1]-root)
                    # Root offsets are spread over an actual posterior scalp
                    # patch; lower controls remain freely curved behind the ear.
                    a=anchor+patch_u*rng.uniform(-.006,.006)*scale+patch_v*rng.uniform(-.005,.005)*scale
                    nearest,n,_,_=bvh.find_nearest(Vector(a))
                    if nearest is None:raise RuntimeError('Gather anchor lost scalp contact')
                    a=np.array(nearest)+np.array(n)*(.0014*scale)
                    b=a+np.array([side*.010,-.004,-.012])*scale
                    c=root-direction*(.019*scale)
                    path=[]
                    for j in range(31):
                        t=j/30
                        path.append((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*root)
                    # Explicit tangent/position join to the unchanged tail pin.
                    path[-2]=root-direction*(.0014*scale)
                    path[-1]=root.copy()
                    paths.append(np.array(path))
            gathers.append(paths)
            gather_reports.append({'side':side,'strands':len(paths),
                                   'root_join_max_error_mm':max(float(np.linalg.norm(p[-1]-tail[i,0])*1000)
                                                                 for i,p in enumerate(paths))})

        max_step=0.
        for name,paths in [('scalp',scalp),('gather L',gathers[0]),('gather R',gathers[1])]:
            for i,path in enumerate(paths):
                steps=np.linalg.norm(np.diff(path,axis=0),axis=1);segment=int(np.argmax(steps))
                max_step=max(max_step,float(steps[segment]))
                if steps[segment]>.020:
                    raise RuntimeError('R06 guide discontinuity '+json.dumps({'group':name,'strand':i,
                        'segment':segment,'step_mm':float(steps[segment]*1000),
                        'world_points':path[segment:segment+2].tolist(),'writesPerformed':False}))
        report={'scale_from_current_head_roots':scale,'scalp_vertices':len(used),
                'scalp_fibres':len(scalp),'gathers':gather_reports,'maximum_step_mm':max_step*1000,
                'tail_points_modified':0,'writesPerformed':not diagnostics_only,
                'validation':'Continuous candidate geometry; visual and ear-clearance review still required'}
        if diagnostics_only:print(json.dumps(report));return report
        example=tails[0]
        source=bpy.data.materials.get('R03 softer charcoal fibres') or tails[0].data.materials[0]
        hair=source.copy();hair.name='R06 scalp and gathered root fibres'
        for node in hair.node_tree.nodes:
            if node.type=='BSDF_HAIR_PRINCIPLED':
                if node.inputs.get('Roughness'):node.inputs['Roughness'].default_value=.60
                if node.inputs.get('Radial Roughness'):node.inputs['Radial Roughness'].default_value=.53
        shell_mesh=bpy.data.meshes.new('R06 actual scalp substrate')
        shell_mesh.from_pydata(_local(shell,example),[],[tuple(remap[i] for i in f) for f in selected])
        shell_mesh.update()
        mat=bpy.data.materials.new('R06 recessed textured dark roots');mat.use_nodes=True
        tree=mat.node_tree;bsdf=next(n for n in tree.nodes if n.type=='BSDF_PRINCIPLED')
        bsdf.inputs['Base Color'].default_value=(.0045,.0040,.0037,1)
        bsdf.inputs['Roughness'].default_value=.79
        noise=tree.nodes.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=720
        noise.inputs['Detail'].default_value=1.0
        bump=tree.nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12
        bump.inputs['Distance'].default_value=.00006*scale
        tree.links.new(noise.outputs['Fac'],bump.inputs['Height']);tree.links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
        shell_mesh.materials.append(mat)
        for face in shell_mesh.polygons:face.use_smooth=True
        obj=bpy.data.objects.new('R06 attached scalp root substrate',shell_mesh)
        bpy.context.scene.collection.objects.link(obj);_attach(obj,example)
        obj[OWNER]=True;obj.hide_render=True;obj.hide_set(True);created.append(obj)
        _curves('R06 continuous scalp flow',scalp,hair,example,.000045,True,created)
        for i,paths in enumerate(gathers):
            _curves('R06 full gathered roots '+('L' if i==0 else 'R'),paths,hair,tails[i],.000080,False,created)
        hidden=[]
        for name in REPLACE:
            old=bpy.data.objects.get(name)
            if old and not old.hide_render:
                old['R06_previous_visibility']=json.dumps({'render':old.hide_render,'viewport':old.hide_get()})
                old.hide_render=True;old.hide_set(True);hidden.append(name)
        for obj in created:obj.hide_render=False;obj.hide_set(False)
        rig['R06_static_roots_changed_review_hair_cache']=True
        report['new_objects']=[o.name for o in created];report['hidden_existing']=hidden
        bpy.context.view_layer.update();print(json.dumps(report));return report
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()


def restore():
    """Retain new objects as hidden history and restore old scalp visibility."""
    for obj in list(bpy.context.scene.objects):
        if obj.get(OWNER):
            obj.hide_render=True;obj.hide_set(True)
            obj.name=obj.name+' restored-away';del obj[OWNER]
        if 'R06_previous_visibility' in obj:
            state=json.loads(obj['R06_previous_visibility'])
            obj.hide_render=state['render'];obj.hide_set(state['viewport'])
            del obj['R06_previous_visibility']
    bpy.context.view_layer.update()


if __name__=='__main__':run()
