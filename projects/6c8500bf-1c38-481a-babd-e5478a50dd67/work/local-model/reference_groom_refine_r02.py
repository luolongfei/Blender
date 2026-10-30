"""Surface-projected scalp and softly separated bangs, after render inspection."""
import bpy,math,random,runpy
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def run():
    make=runpy.run_path(str(Path(__file__).with_name('reconstruction_groom_r01.py')))['curve_object']
    h=bpy.data.objects['R01 Head']
    h.data.normals_split_custom_set([(0,0,0)]*len(h.data.loops))
    bpy.context.view_layer.update(); bvh=BVHTree.FromObject(h,bpy.context.evaluated_depsgraph_get())
    for o in bpy.data.objects:
        if o.name.startswith(('R01 Groom','R02 Fringe','R02 Temple')): o.hide_render=True;o.hide_set(True)
    mat=bpy.data.materials['R01 soft black hair']; n=next(n for n in mat.node_tree.nodes if n.type=='BSDF_HAIR_PRINCIPLED');n.inputs['Roughness'].default_value=.62
    def surface(theta,phi,side):
        c=Vector((0,.002,1.445));v=Vector((side*math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)))
        hit,no,idx,d=bvh.ray_cast(c+v*.3,-v)
        return hit+no*.002 if hit else c+v*.09
    rng=random.Random(2247);paths=[]
    for side in [-1,1]:
        for i in range(4000):
            theta0=rng.uniform(.005,1.7);phi0=rng.uniform(-.24,1.58);arr=[]
            for j in range(28):
                t=j/27;theta=theta0+(1.93-theta0)*t
                phi=phi0+(.55-phi0)*t*t
                q=surface(theta,phi,side)
                if t>.78:q=q.lerp(Vector((side*.09,-.004,1.395)),((t-.78)/.22)**2)
                arr.append(q)
            paths.append(arr)
    make('R02 Surface scalp',paths,.000095,mat)
    def frontpos(x,z,lift):
        # A requested point above the narrow crown is lowered onto actual scalp.
        for step in range(24):
            zz=z-step*.001;hit,no,idx,d=bvh.ray_cast(Vector((x,-.5,zz)),Vector((0,1,0)))
            if hit:return hit+Vector((0,-lift,0))
        return Vector((x,-.03,z-.02))
    paths=[]
    guides=[(-.032,-.062,1.448,.012),(-.022,-.043,1.470,.013),(-.013,-.022,1.465,.012),(-.002,.003,1.470,.015),(.01,.026,1.475,.012),(.025,.047,1.469,.012),(.034,.063,1.448,.010)]
    for rx,tx,tz,width in guides:
        for i in range(600):
            off=rng.uniform(-1,1);startz=rng.uniform(1.538,1.554);tipz=tz+rng.uniform(-.005,.005);tipx=tx+rng.uniform(-.0025,.0025);arr=[]
            for j in range(31):
                t=j/30;x=rx*(1-t)+tipx*t-.006*math.sin(math.pi*t)+off*width*(1-t)**.65
                z=startz+(tipz-startz)*t
                arr.append(frontpos(x,z,.002+rng.uniform(0,.001)+.003*math.sin(math.pi*t)))
            paths.append(arr)
    make('R02 Surface bangs',paths,.00010,mat)
    paths=[]
    for side in [-1,1]:
        for i in range(450):
            off=rng.uniform(-.0035,.0035);endz=rng.uniform(1.334,1.351);arr=[]
            a=Vector((side*.063,-.058,1.495));b=Vector((side*.079,-.071,1.43));c=Vector((side*.068,-.074,1.362));d=Vector((side*.057,-.068,endz))
            for j in range(32):
                t=j/31;q=a*(1-t)**3+b*3*t*(1-t)**2+c*3*t*t*(1-t)+d*t**3
                q.x+=off*(1-t)**.6;q.y-=rng.uniform(0,.001);arr.append(q)
            paths.append(arr)
    make('R02 Curved temple wisps',paths,.00009,mat)
    print('R02_SURFACE_GROOM_REFINED')

if __name__=='__main__':run()
