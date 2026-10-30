"""Small, globe-conforming lid follow, fading away during closed blinks."""
import bpy,json,math,sys
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def driver(key,rig,h,side,direction):
    d=key.driver_add('value').driver;d.type='SCRIPTED'
    for name,id,path in [('pitch',rig,'["R03_look_pitch"]'),('blink',h.data.shape_keys,f'key_blocks["R02_blink_{side}"].value'),
                         ('squint',h.data.shape_keys,f'key_blocks["R03_squint_{side}"].value')]:
        v=d.variables.new();v.name=name;v.type='SINGLE_PROP';v.targets[0].id_type='OBJECT' if id==rig else 'KEY'
        v.targets[0].id=id;v.targets[0].data_path=path
    d.expression=f'min(1,max(0,{-1 if direction=="up" else 1}*pitch/18))*max(0,1-blink-0.36*squint)'

def run():
    p=project_path();h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];ks=h.data.shape_keys.key_blocks
    if Path(bpy.data.filepath).name=='character.blend' or ks.get('R05_gaze_up_L'):raise RuntimeError('Use untouched R05 work copy')
    if abs(rig.get('R03_look_pitch',0))>1e-6:raise RuntimeError('Neutral pitch required')
    mapping=json.loads((p/'work/reconstruction-r02/eye-aperture-r05.json').read_text())
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    co=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        co.append(q)
    neighbors=[set() for _ in co]
    for e in h.data.edges:a,b=e.vertices;neighbors[a].add(b);neighbors[b].add(a)
    for side in ['L','R']:
        upper=mapping[side]['ringIndices']['upper'];lower=mapping[side]['ringIndices']['lower'];ring=set(upper+lower)
        cen=Vector(mapping[side]['center']);radius=mapping[side]['radius'];xa,xb=co[upper[0]].x,co[upper[-1]].x
        dist={i:0 for i in ring};frontier=ring
        for step in range(1,5):
            nxt={j for i in frontier for j in neighbors[i] if j not in dist};dist.update({j:step for j in nxt});frontier=nxt
        for direction in ['up','down']:
            sign=1 if direction=='up' else -1
            def shift(q,isupper):
                t=max(0,min(1,(q.x-xa)/(xb-xa)));dz=sign*(.0024 if isupper else .0011)*math.sin(math.pi*t)**.6
                target=q.copy();target.z+=dz
                r0=(q.x-cen.x)**2+(q.z-cen.z)**2;r1=(q.x-cen.x)**2+(target.z-cen.z)**2
                if max(r0,r1)<radius*radius:
                    clearance=max(.00065,cen.y-math.sqrt(radius*radius-r0)-q.y)
                    target.y=cen.y-math.sqrt(radius*radius-r1)-clearance
                return target-q
            delta={i:shift(co[i],i in upper) for i in ring};k=h.shape_key_add(name=f'R05_gaze_{direction}_{side}',from_mix=False)
            for i,v in enumerate(k.data):v.co=ks[0].data[i].co
            for i,depth in dist.items():
                near=min(ring,key=lambda j:(co[i]-co[j]).length_squared);k.data[i].co+=delta[near]*[1,.85,.5,.2,.04][depth]
            k.value=0;driver(k,rig,h,side,direction)
            for name in ['R02 Upper lashes '+side,'R02 Upper lid rim '+side]:
                o=bpy.data.objects[name];ok=o.data.shape_keys.key_blocks;basis=ok[0];opened=ok['R05 Aperture'];k=o.shape_key_add(name='R05 Gaze '+direction,from_mix=False)
                inv=o.matrix_world.inverted().to_3x3()
                for i,v in enumerate(k.data):
                    v.co=basis.data[i].co;root=(i//5)*5 if 'lashes' in name else i
                    q=o.matrix_world@opened.data[root].co;v.co+=inv@shift(q,True)
                k.value=0;driver(k,rig,h,side,direction)
    rig.data.pose_position=pose;bpy.context.view_layer.update()
    h['R05_gaze_follow']='Upper/lower lids follow pitch up to 2.4/1.1 mm at 18 degrees; fade at blink. Requires combined visual review.'
    print('R05_GAZE_FOLLOW_CREATED')
