"""Local soft-tissue smoothing and corrected scalp coverage after review."""
import bpy,math
from mathutils import Vector
def run():
    h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
    if ks.get('R02_soft_tissue'):raise RuntimeError('Already softened')
    co=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        co.append(q)
    neighbors=[set() for _ in co]
    for e in h.data.edges:
        a,b=e.vertices;neighbors[a].add(b);neighbors[b].add(a)
    work=[q.copy() for q in co]
    weights=[]
    for q in co:
        x,y,z=q
        w=math.exp(-((abs(x)-.043)/.026)**4-((z-1.405)/.031)**4)*max(0,min(1,(-y-.05)/.02))
        weights.append(w)
    for _ in range(4):
        new=[]
        for i,q in enumerate(work):
            avg=sum((work[j] for j in neighbors[i]),Vector())/len(neighbors[i]) if neighbors[i] else q
            new.append(q.lerp(avg,.32*weights[i]))
        work=new
    key=h.shape_key_add(name='R02_soft_tissue',from_mix=False)
    for i,(v,k) in enumerate(zip(ks[0].data,key.data)):
        k.co=v.co+(work[i]-co[i]);x,y,z=co[i]
        fullness=.004*math.exp(-((abs(x)-.044)/.019)**2-((z-1.403)/.024)**2)*max(0,min(1,(-y-.055)/.02))
        k.co.y-=fullness
        neck=math.exp(-((z-1.30)/.04)**4)
        k.co.x-=x*.1*neck
    key.value=1
    # Reuse the previously complete cranial coverage; reroute front strays to the temples.
    for name in ['R01 Groom scalp','R01 Groom back coverage']:
        o=bpy.data.objects[name];o.hide_render=False;o.hide_set(False)
        for s in o.data.splines:
            side=1 if s.points[-1].co.x>0 else -1
            for p in s.points:
                x,y,z,w=p.co
                if z<1.49 and y<-.025:
                    a=max(0,min(1,(1.49-z)/.025))*max(0,min(1,(-y-.025)/.04))
                    target=side*max(abs(x),.072)
                    p.co.x=x+(target-x)*a
    o=bpy.data.objects['R02 Surface scalp'];o.hide_render=True;o.hide_set(True)
    # Feather the rectangular root band into the upper scalp.
    fringe=bpy.data.objects['R02 Surface bangs']
    for s in fringe.data.splines:
        count=len(s.points)
        for i,p in enumerate(s.points):
            t=i/(count-1);p.radius*=min(1,.15+t*7)
    print('R02_SOFTENED')
if __name__=='__main__':run()
