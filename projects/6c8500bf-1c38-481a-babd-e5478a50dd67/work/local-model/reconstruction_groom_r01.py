"""Deterministic editable static curve groom; no dynamics or baked simulation."""
import bpy, math, random
from mathutils import Vector
from mathutils.bvhtree import BVHTree
R=random.Random(1127)

def curve_object(name,paths,radius,material):
    d=bpy.data.curves.new(name,'CURVE'); d.dimensions='3D'; d.resolution_u=2
    d.bevel_depth=radius; d.bevel_resolution=0; d.resolution_u=2
    for path in paths:
        s=d.splines.new('POLY'); s.points.add(len(path)-1)
        for i,(point,co) in enumerate(zip(s.points,path)):
            point.co=(*co,1); point.radius=max(.08,(1-i/(len(path)-1))**.3)
    o=bpy.data.objects.new(name,d); bpy.context.scene.collection.objects.link(o); d.materials.append(material)
    o['R01_scope']='Static authored groom, unbound and unsimulated'
    return o

def run():
    if bpy.data.objects.get('R01 Groom scalp'): raise RuntimeError('Already built')
    head=bpy.data.objects['R01 Head']
    # Fit neck inside original blouse. The shoulder-to-body seam is not yet welded.
    for v in head.data.vertices:
        x,y,z=v.co
        a=max(0,min(1,(1.338-z)/.080)); a=a*a*(3-2*a)
        v.co.x=x*(1-.69*a)
        v.co.y=y*(1-.72*a)+.003*a
    bpy.context.view_layer.update()
    bvh=BVHTree.FromObject(head,bpy.context.evaluated_depsgraph_get())
    def front_y(x,z):
        hit,normal,idx,dist=bvh.ray_cast(Vector((x,-.5,z)),Vector((0,1,0)))
        return hit.y if hit else -.07
    # Lips receive only restrained natural color; no pore/noise displacement in form pass.
    attr=head.data.color_attributes.new(name='R01_region_color',type='FLOAT_COLOR',domain='POINT')
    for v,c in zip(head.data.vertices,attr.data):
        x,y,z=v.co
        lip=math.exp(-((x/.025)**6)-((z-1.378)/.008)**4)*max(0,min(1,(-y-.09)/.007))
        c.color=(.48-.08*lip,.29-.12*lip,.205-.07*lip,1)
    mat=head.data.materials[0]; n=mat.node_tree.nodes.new('ShaderNodeVertexColor'); n.layer_name=attr.name
    bsdf=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    mat.node_tree.links.new(n.outputs['Color'],bsdf.inputs['Base Color'])
    # Reuse the project's existing iris texture on a new curved anterior patch.
    for side in ['L','R']:
        bpy.data.objects['R01 Iris '+side].hide_render=True; bpy.data.objects['R01 Iris '+side].hide_set(True)
        eye=bpy.data.objects['R01 Sclera '+side]
        vs=[v.co for v in eye.data.vertices]
        cen=Vector([(min(v[i] for v in vs)+max(v[i] for v in vs))/2 for i in range(3)])
        rad=.01142; ir=.00535; verts=[]; uv=[]; faces=[]
        for j in range(9):
            r=ir*j/8
            for k in range(64):
                a=k*math.tau/64; xx=r*math.cos(a); zz=r*math.sin(a)
                verts.append((cen.x+xx,cen.y-math.sqrt(rad*rad-r*r)-.00025,cen.z+zz))
                uv.append((.5+xx/ir*.5,.5+zz/ir*.5))
        for j in range(8):
            for k in range(64): faces.append((j*64+k,j*64+(k+1)%64,(j+1)*64+(k+1)%64,(j+1)*64+k))
        me=bpy.data.meshes.new('R01 iris surface'); me.from_pydata(verts,[],faces); me.update()
        o=bpy.data.objects.new('R01 Textured iris '+side,me); bpy.context.scene.collection.objects.link(o)
        me.materials.append(bpy.data.objects['Iris brown '+side].data.materials[0]); u=me.uv_layers.new()
        for poly in me.polygons:
            poly.use_smooth=True
            for li in poly.loop_indices: u.data[li].uv=uv[me.loops[li].vertex_index]
    # Replace slab construction with native editable curves in separate scalp/tail groups.
    for o in bpy.data.objects:
        if o.name.startswith('Hair_') and not o.name.startswith('Hair_Tie'):
            o.hide_render=True; o.hide_set(True); o['R01_superseded']=True
    hair=bpy.data.materials.new('R01 soft black hair'); hair.use_nodes=True
    nt=hair.node_tree; nt.nodes.clear(); out=nt.nodes.new('ShaderNodeOutputMaterial'); h=nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    h.parametrization=next(e.identifier for e in h.bl_rna.properties['parametrization'].enum_items if e.identifier=='COLOR')
    h.inputs['Color'].default_value=(.012,.008,.006,1); h.inputs['Roughness'].default_value=.35
    nt.links.new(h.outputs[0],out.inputs['Surface'])
    def scalp(theta,phi,side,offset=0):
        c=Vector((0,.002,1.446)); v=Vector((side*.079*math.sin(theta)*math.cos(phi),.095*math.sin(theta)*math.sin(phi),.113*math.cos(theta)))
        direction=v.normalized(); hit,normal,idx,dist=bvh.ray_cast(c+direction*.30,-direction)
        radius=v.length
        if hit: radius=max(radius,(hit-c).length+.0025)
        return c+direction*(radius+offset)
    paths=[]
    for side in [-1,1]:
        for i in range(3300):
            th0=R.uniform(.045,1.45); ph0=R.uniform(-1.4,1.5); width=R.uniform(-.004,.004)
            arr=[]
            for j in range(25):
                t=j/24; th=th0+(1.95-th0)*t; ph=ph0+(.55-ph0)*t**1.5
                q=scalp(th,ph,side,R.uniform(.0001,.0006)+width*math.sin(math.pi*t)**2)
                if t>.78:
                    blend=((t-.78)/.22)**2
                    end=Vector((side*(.091+width),.041,1.387))
                    q=q.lerp(end,blend)
                arr.append(q)
            paths.append(arr)
    curve_object('R01 Groom scalp',paths,.00010,hair)
    # Side-swept fringe, independently editable; different endpoints break the sheet edge.
    paths=[]
    for i in range(1500):
        side=1 if R.random()>.43 else -1
        x0=R.uniform(-.018,.018); xe=side*R.uniform(.025,.071)
        ze=R.uniform(1.469,1.497)-.015*(abs(xe)/.071)
        arr=[]
        for j in range(24):
            t=j/23; x=x0*(1-t)+xe*t; z=1.544*(1-t)+ze*t
            y=front_y(x,z)-.003-R.uniform(0,.0015)-.007*math.sin(math.pi*t)
            arr.append((x,y,z))
        paths.append(arr)
    curve_object('R01 Groom fringe',paths,.000095,hair)
    for side,name in [(1,'L'),(-1,'R')]:
        tie=bpy.data.objects['Hair_Tie_'+name]
        for v in tie.data.vertices: v.co.x-=side*.009; v.co.y+=.017; v.co.z+=.005
        paths=[]
        for i in range(2200):
            ang=R.random()*math.tau; r=math.sqrt(R.random()); length=R.uniform(.255,.305)
            phase=R.uniform(0,math.tau); arr=[]
            for j in range(32):
                t=j/31; spread=.011+.008*math.sin(math.pi*t); taper=1-.65*t**2
                x=side*(.091+.019*math.sin(math.pi*t)-.004*t)+math.cos(ang)*r*spread*taper
                y=.041+.085*math.sin(t*math.pi/2)+math.sin(ang)*r*spread*taper
                z=1.388-length*t
                x+=.002*math.sin(t*10+phase)*math.sin(math.pi*t)
                arr.append((x,y,z))
            paths.append(arr)
        curve_object('R01 Groom tail '+name,paths,.00011,hair)
    browmat=bpy.data.materials.new('R01 brow hair'); browmat.use_nodes=True
    b=next(n for n in browmat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'); b.inputs['Base Color'].default_value=(.018,.009,.005,1); b.inputs['Roughness'].default_value=.62
    for side in [-1,1]:
        paths=[]
        for i in range(240):
            t=R.random(); x=side*(.014+.037*t); z=1.464+.008*math.sin(t*math.pi*.9)+R.uniform(-.0014,.0014)*(1-.65*t)
            arr=[]
            for j in range(4):
                u=j/3; xx=x+side*.0032*u; zz=z+.0022*math.sin(u*1.8)
                arr.append((xx,front_y(xx,zz)-.0006,zz))
            paths.append(arr)
        curve_object('R01 Brow '+str(side),paths,.00010,browmat)
    bpy.context.view_layer.update(); print('R01_GROOM_BUILT')

if __name__=='__main__': run()
