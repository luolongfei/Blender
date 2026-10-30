"""Visible correction after first render; preserves A/B face direction controls."""
import bpy, math, random
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def run():
    h=bpy.data.objects['R01 Head']
    if h.data.shape_keys: raise RuntimeError('Already finished')
    for v in h.data.vertices:
        a=max(0,min(1,(1.335-v.co.z)/.055)); a=a*a*(3-2*a)
        v.co.y-=.020*a
    # Root tint only under hair, resolving exposed pale crown on the first render.
    attr=h.data.color_attributes['R01_region_color']
    for v,c in zip(h.data.vertices,attr.data):
        a=max(0,min(1,(v.co.z-1.527)/.015))
        c.color=tuple(c.color[i]*(1-a)+(.035,.022,.015,1)[i]*a for i in range(4))
    # Smaller fabric ties stay centered on the actual bundles.
    for side,name in [(1,'L'),(-1,'R')]:
        tie=bpy.data.objects['Hair_Tie_'+name]
        center=sum((v.co for v in tie.data.vertices),Vector())/len(tie.data.vertices)
        for v in tie.data.vertices:
            v.co.x=center.x+(v.co.x-center.x)*.72
            v.co.z=center.z+(v.co.z-center.z)*.72
    h.shape_key_add(name='Basis')
    key=h.shape_key_add(name='R01_B_soft_oval')
    for v,k in zip(h.data.vertices,key.data):
        x,y,z=v.co
        front=max(0,min(1,(-y+.01)/.06))
        jaw=math.exp(-((z-1.355)/.042)**2)*front
        k.co.z+=.006*jaw
        cheek=math.exp(-((z-1.408)/.024)**2)*front
        k.co.x=x*(1+.035*cheek-.035*jaw)
        k.co.y+=.0018*math.exp(-(x/.018)**2-((z-1.414)/.022)**2)*front
    key.value=0
    h['R01_A']='Longer oval, more defined lower face'
    h['R01_B']='Softer, shorter lower face; key value 1'
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

def repair_crown():
    """Extend fringe roots over crown after identifying a horizontal scalp gap."""
    o=bpy.data.objects['R01 Groom fringe']
    if o.get('R01_root_repaired'): return
    paths=[]
    for s in o.data.splines:
        old=[p.co.xyz.copy() for p in s.points]; end=old[0]
        root=Vector((end.x*.25,.006,1.557))
        arr=[]
        for j in range(7):
            t=j/7; q=root.lerp(end,t)
            q.z+=.004*math.sin(math.pi*t)
            arr.append(q)
        paths.append(arr+old)
    o.data.splines.clear()
    for path in paths:
        s=o.data.splines.new('POLY'); s.points.add(len(path)-1)
        for i,(pt,co) in enumerate(zip(s.points,path)):
            pt.co=(*co,1); pt.radius=max(.08,(1-i/(len(path)-1))**.3)
    o['R01_root_repaired']=True

def repair_back():
    import runpy
    from pathlib import Path
    if bpy.data.objects.get('R01 Groom back coverage'): return
    h=bpy.data.objects['R01 Head']
    for i,v in enumerate(h.data.vertices):
        x,y,z=v.co
        a=max(0,min(1,(1.305-z)/.055)); a=a*a*(3-2*a)
        delta=Vector((x*.40*a,((y+.021)*.8-.005)*a,-.013*a))
        v.co+=delta
        for key in h.data.shape_keys.key_blocks: key.data[i].co+=delta
    make=runpy.run_path(str(Path(__file__).with_name('reconstruction_groom_r01.py')))['curve_object']
    r=random.Random(209); paths=[]
    bvh=BVHTree.FromObject(h,bpy.context.evaluated_depsgraph_get())
    for side in [-1,1]:
        for i in range(1800):
            theta0=r.uniform(.08,1.65); phi0=r.uniform(1.25,1.59); arr=[]
            for j in range(28):
                t=j/27; theta=theta0+(2.02-theta0)*t
                phi=phi0+(.50-phi0)*max(0,(t-.30)/.70)**2
                c=Vector((0,.002,1.446)); v=Vector((side*.079*math.sin(theta)*math.cos(phi),.096*math.sin(theta)*math.sin(phi),.113*math.cos(theta)))
                direct=v.normalized(); hit,normal,idx,dist=bvh.ray_cast(c+direct*.30,-direct)
                radius=max(v.length,(hit-c).length+.003 if hit else 0)
                q=c+direct*(radius+r.uniform(0,.0008))
                if t>.85: q=q.lerp(Vector((side*.091,.041,1.387)),((t-.85)/.15)**2)
                arr.append(q)
            paths.append(arr)
    make('R01 Groom back coverage',paths,.00010,bpy.data.materials['R01 soft black hair'])

if __name__=='__main__':
    run()
    repair_crown()
    repair_back()
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
