"""Topology-led globe contact, independent blinks and matching eyelash controls."""
import bpy,math,random,runpy
from pathlib import Path
from mathutils import Vector
UP=[1491,307,963,277,271,862,276,272,273,274,1182,278]
LOW=[1491,275,279,960,267,864,268,269,866,270,1440,305,1461,278]

def run():
    h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
    if ks.get('R02_globe_contact'):raise RuntimeError('Already added')
    def composed():
        out=[]
        for i,v in enumerate(ks[0].data):
            q=v.co.copy()
            for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
            out.append(q)
        return out
    neighbors=[set() for _ in h.data.vertices]
    for e in h.data.edges:
        a,b=e.vertices;neighbors[a].add(b);neighbors[b].add(a)
    co=composed();regions={};centers={}
    contact=h.shape_key_add(name='R02_globe_contact',from_mix=False)
    for side,offset in [('R',0),('L',1519)]:
        up=[i+offset for i in UP];lo=[i+offset for i in LOW];ring=set(up+lo)
        eye=bpy.data.objects['R01 Sclera '+side]; vs=[v.co for v in eye.data.vertices]
        cen=Vector([(min(v[i] for v in vs)+max(v[i] for v in vs))/2 for i in range(3)]); centers[side]=cen
        radius=(max(v.x for v in vs)-min(v.x for v in vs))/2
        dist={i:0 for i in ring};frontier=ring
        for step in range(1,5):
            nxt={j for i in frontier for j in neighbors[i] if j not in dist}
            for j in nxt:dist[j]=step
            frontier=nxt
        regions[side]=(up,lo,dist,radius)
        weights=[1,.90,.55,.20,.04]
        for i,d in dist.items():
            q=co[i]; r2=(q.x-cen.x)**2+(q.z-cen.z)**2
            if r2<radius**2 and q.y<cen.y+.003:
                surface=cen.y-math.sqrt(radius*radius-r2)-.0006
                # Only bring surrounding eyelid outwards; no collapse of the socket wall.
                delta=min(0,surface-q.y)*weights[d]
                contact.data[i].co.y+=delta
    contact.value=1;co=composed()
    make=runpy.run_path(str(Path(__file__).with_name('reconstruction_groom_r01.py')))['curve_object']
    mat=bpy.data.materials['R01 brow hair'];rng=random.Random(2028)
    for side,(up,lo,dist,radius) in regions.items():
        cen=centers[side];left=co[up[0]];right=co[up[-1]]
        def close(q):
            t=max(0,min(1,(q.x-left.x)/(right.x-left.x)))
            z=left.z*(1-t)+right.z*t-.0020*math.sin(math.pi*t)
            r2=(q.x-cen.x)**2+(z-cen.z)**2
            y=cen.y-math.sqrt(max(.000001,radius*radius-r2))-.0008 if r2<radius*radius else q.y
            return Vector((q.x,y,z))
        key=h.shape_key_add(name='R02_blink_'+side,from_mix=False)
        rim=up+lo[1:-1];weights=[1,.92,.66,.33,.10]
        for i,d in dist.items():
            nearest=min(rim,key=lambda j:(co[j]-co[i]).length_squared)
            delta=(close(co[nearest])-co[nearest])*weights[d]
            key.data[i].co+=delta
        key.value=0
        def interp(ids,t):
            lengths=[0]
            for a,b in zip(ids,ids[1:]):lengths.append(lengths[-1]+(co[a]-co[b]).length)
            s=t*lengths[-1]
            for j in range(len(ids)-1):
                if lengths[j+1]>=s:
                    u=(s-lengths[j])/max(.000001,lengths[j+1]-lengths[j]);return co[ids[j]].lerp(co[ids[j+1]],u)
            return co[ids[-1]].copy()
        paths=[];closedpaths=[]
        for i in range(100):
            t=.06+.90*i/99;root=interp(up,t);root.y-=.0005
            end=close(root);length=.0025+.0025*math.sin(math.pi*t)+rng.uniform(-.0005,.0005)
            arr=[];closed=[]
            for j in range(5):
                u=j/4;off=Vector(((1 if side=='L' else -1)*.001*u,-length*u,.0012*math.sin(u*1.8)))
                arr.append(root+off);closed.append(end+off)
            paths.append(arr);closedpaths.append(closed)
        lash=make('R02 Upper lashes '+side,paths,.000115,mat)
        lash.shape_key_add(name='Basis',from_mix=False);lk=lash.shape_key_add(name='Blink',from_mix=False)
        flat=[q for path in closedpaths for q in path]
        for v,q in zip(lk.data,flat):v.co=q
        drv=lk.driver_add('value').driver;drv.type='AVERAGE';var=drv.variables.new();var.name='blink';var.targets[0].id_type='KEY';var.targets[0].id=h.data.shape_keys;var.targets[0].data_path=f'key_blocks["R02_blink_{side}"].value'
        # Subtle dark upper rim on the actual eyelid edge, not a floating eye decal.
        path=[interp(up,i/50)+Vector((0,-.00045,0)) for i in range(51)]
        rimobj=make('R02 Upper lid rim '+side,[path],.00020,mat)
        rimobj.shape_key_add(name='Basis',from_mix=False);rk=rimobj.shape_key_add(name='Blink',from_mix=False)
        for v,q in zip(rk.data,path):v.co=close(q)
        dr=rk.driver_add('value').driver;dr.type='AVERAGE';vv=dr.variables.new();vv.targets[0].id_type='KEY';vv.targets[0].id=h.data.shape_keys;vv.targets[0].data_path=f'key_blocks["R02_blink_{side}"].value'
    bpy.context.view_layer.update();print('R02_EYELIDS_CREATED')

if __name__=='__main__':run()
