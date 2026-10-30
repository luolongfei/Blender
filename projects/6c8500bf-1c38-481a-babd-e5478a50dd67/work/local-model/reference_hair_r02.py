"""Reference fringe, loose side locks and curved, layered low twin tails."""
import bpy,math,random,runpy
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def run():
    if bpy.data.objects.get('R02 Fringe'): raise RuntimeError('Already groomed')
    make=runpy.run_path(str(Path(__file__).with_name('reconstruction_groom_r01.py')))['curve_object']
    h=bpy.data.objects['R01 Head']; dg=bpy.context.evaluated_depsgraph_get(); bvh=BVHTree.FromObject(h,dg)
    def front(x,z):
        loc,no,idx,d=bvh.ray_cast(Vector((x,-.5,z)),Vector((0,1,0)))
        return loc.y if loc else -.02
    hair=bpy.data.materials['R01 soft black hair']
    n=next(n for n in hair.node_tree.nodes if n.type=='BSDF_HAIR_PRINCIPLED'); n.inputs['Roughness'].default_value=.47
    n.inputs['Color'].default_value=(.007,.006,.0055,1)
    for o in bpy.data.objects:
        if o.name.startswith(('R01 Groom fringe','R01 Groom tail','Hair_Tie')):
            o.hide_render=True; o.hide_set(True)
    # Existing scalp coverage is retained; gather ends move forward to the new dark ties.
    for name in ['R01 Groom scalp','R01 Groom back coverage']:
        o=bpy.data.objects[name]
        for s in o.data.splines:
            for i,p in enumerate(s.points):
                t=i/(len(s.points)-1); a=max(0,(t-.78)/.22)**2
                side=1 if p.co.x>0 else -1
                p.co.x-=side*.001*a; p.co.y-=.045*a; p.co.z+=.008*a
    rng=random.Random(20028)
    def bezier(a,b,c,d,t): return (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d
    # Seven distinct forelocks with an irregular fringe silhouette.
    paths=[]
    guides=[(-.024,-.065,1.449,.013),(-.018,-.046,1.477,.014),(-.008,-.024,1.469,.014),(.002,.002,1.474,.016),(.013,.026,1.481,.014),(.023,.046,1.476,.012),(.030,.064,1.447,.011)]
    for rootx,tipx,tipz,width in guides:
        for i in range(470):
            off=rng.uniform(-1,1); thick=rng.uniform(.0005,.0020); endz=tipz+rng.uniform(-.003,.004)
            arr=[]
            for j in range(29):
                t=j/28
                x=bezier(rootx,rootx*.9,tipx-.008,tipx,t)+off*width*(1-t**1.8)
                z=bezier(1.551,1.535,1.493,endz,t)
                y=front(x,z)-.003-thick-.004*math.sin(math.pi*t)
                arr.append((x,y,z))
            paths.append(arr)
    make('R02 Fringe',paths,.00010,hair)
    # Longer narrow temple locks framing the cheeks, independently editable.
    paths=[]
    for side in [-1,1]:
        for i in range(440):
            offset=rng.uniform(-.004,.004); endz=rng.uniform(1.330,1.355)
            arr=[]
            for j in range(30):
                t=j/29; x=side*(.065+.009*math.sin(t*math.pi)-.009*t*t)+offset*(1-.7*t)
                z=1.495+(endz-1.495)*t
                y=front(x,z)-.004 if z>1.37 else -.061-.025*t
                arr.append((x,y,z))
            paths.append(arr)
    make('R02 Temple locks',paths,.000095,hair)
    # Eight separate coherent guides per tail; each has its own arc, length and tapered end.
    for side,side_name in [(1,'L'),(-1,'R')]:
        paths=[]
        for group in range(8):
            angle=group*math.tau/8
            length=.245+rng.uniform(-.025,.025)
            bend=rng.uniform(-.008,.008)
            for i in range(310):
                offx=rng.uniform(-.008,.008); offy=rng.uniform(-.006,.006)
                arr=[]
                for j in range(35):
                    t=j/34; envelope=math.sin(math.pi*t)**.7
                    cx=.090+.039*math.sin(math.pi*t)+.014*math.sin(2*math.pi*t)+bend*t
                    cx+=math.cos(angle)*(.004+.016*envelope)*(1-.7*t)
                    y=-.004-.145*math.sin(t*math.pi/2)+math.sin(angle)*(.004+.015*envelope)
                    x=side*cx+offx*(1-t**3)
                    y+=offy*(1-t**3)
                    z=1.395-length*t+.006*math.cos(angle)*envelope
                    # Separate ends curl inward instead of sharing a straight cut.
                    x-=side*.022*max(0,(t-.77)/.23)**2
                    arr.append((x,y,z))
                paths.append(arr)
        make('R02 Twin tail '+side_name,paths,.00011,hair)
    band=bpy.data.materials.new('R02 black cloth hair ties');band.use_nodes=True
    bn=next(n for n in band.node_tree.nodes if n.type=='BSDF_PRINCIPLED');bn.inputs['Base Color'].default_value=(.012,.014,.014,1);bn.inputs['Roughness'].default_value=.85
    direction=Vector((.05,-.14,-.25)).normalized()
    for side in [-1,1]:
        for j in [-1,1]:
            loc=Vector((side*.090,-.004,1.395))+direction*(j*.0028)
            bpy.ops.mesh.primitive_torus_add(major_radius=.0135,minor_radius=.0019,major_segments=32,minor_segments=8,location=loc)
            o=bpy.context.object;o.name=f'R02 Tie {side} {j}';o.rotation_euler=direction.to_track_quat('Z','Y').to_euler();o.data.materials.append(band)
            for f in o.data.polygons:f.use_smooth=True
    brow=bpy.data.materials['R01 brow hair']
    for side in [-1,1]:
        paths=[]
        for i in range(260):
            t=rng.random(); x=side*(.016+.036*t)
            z=1.468+.0045*math.sin(t*math.pi)+rng.uniform(-.0012,.0012)*(1-t*.6)
            arr=[]
            for j in range(5):
                u=j/4;xx=x+side*.0028*u;zz=z+.0015*math.sin(u*2)
                arr.append((xx,front(xx,zz)-.0007,zz))
            paths.append(arr)
        make('R02 Soft brow '+str(side),paths,.00010,brow)
    bpy.context.view_layer.update();print('R02_GROOM_BUILT')

if __name__=='__main__':run()
