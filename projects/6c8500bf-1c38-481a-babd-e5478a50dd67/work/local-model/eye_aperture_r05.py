"""Round the neutral eyelid aperture, retaining the checked closed-eye shape."""
import bpy,math,json,sys
from pathlib import Path
from mathutils import Vector,kdtree
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
UP=[1491,307,963,277,271,862,276,272,273,274,1182,278]
LOW=[1491,275,279,960,267,864,268,269,866,270,1440,305,1461,278]

def _driver(key,rig,head,side,kind):
    d=key.driver_add('value').driver;d.type='SCRIPTED'
    while d.variables:d.variables.remove(d.variables[0])
    for name,id,path in [('blink',head.data.shape_keys,f'key_blocks["R02_blink_{side}"].value'),
                         ('squint',head.data.shape_keys,f'key_blocks["R03_squint_{side}"].value'),
                         ('amount',rig,'["R05_eye_openness"]')]:
        v=d.variables.new();v.name=name;v.type='SINGLE_PROP';v.targets[0].id_type='KEY' if id==head.data.shape_keys else 'OBJECT'
        v.targets[0].id=id;v.targets[0].data_path=path
    d.expression={'open':'amount * max(0,1-blink-0.36*squint)',
                  'excess':'max(0,blink+0.36*squint-1)',
                  'close':'min(1,blink+0.36*squint)'}[kind]

def run():
    if Path(bpy.data.filepath).name=='character.blend':raise RuntimeError('Use work copy')
    h=bpy.data.objects['R01 Head'];rig=bpy.data.objects['Character_Rig'];ks=h.data.shape_keys.key_blocks
    if ks.get('R05_aperture_L'):raise RuntimeError('Already applied')
    if not h.get('R05_reference_proportion'):raise RuntimeError('R05 proportions required')
    if any(abs(ks['R02_blink_'+s].value)>1e-6 or abs(ks['R03_squint_'+s].value)>1e-6 for s in ('L','R')):
        raise RuntimeError('Neutral eyes required')
    pose=rig.data.pose_position;rig.data.pose_position='REST';bpy.context.view_layer.update()
    old=bpy.data.objects['R03 prejoin Head preserved'];tree=kdtree.KDTree(len(ks[0].data))
    for i,v in enumerate(ks[0].data):tree.insert(v.co,i)
    tree.balance();prop=json.loads(h['R05_reference_proportion']);center=Vector(prop['center'])
    def index(oi):
        q=old.data.shape_keys.key_blocks[0].data[oi].co.copy();q=center+(q-center)*prop['scale'];q.z-=prop['drop']
        _,i,d=tree.find(q)
        if d>1e-5:raise RuntimeError('Exact eyelid mapping lost')
        return i
    co=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for key in ks[1:]:q+=(key.data[i].co-key.relative_key.data[i].co)*key.value
        co.append(q)
    neighbors=[set() for _ in co]
    for e in h.data.edges:a,b=e.vertices;neighbors[a].add(b);neighbors[b].add(a)
    rig['R05_eye_openness']=1.;rig.id_properties_ui('R05_eye_openness').update(min=0.,max=1.)
    report={}
    for side,offset in [('R',0),('L',1519)]:
        upper=[index(i+offset) for i in UP];lower=[index(i+offset) for i in LOW]
        eye=bpy.data.objects['R01 Sclera '+side];pts=[eye.matrix_world@v.co for v in eye.data.vertices]
        cen=Vector([(min(v[j] for v in pts)+max(v[j] for v in pts))/2 for j in range(3)])
        radius=(max(v.x for v in pts)-min(v.x for v in pts))/2
        xa,xb=co[upper[0]].x,co[upper[-1]].x
        def displacement(q,upper_lid=True):
            t=max(0.,min(1.,(q.x-xa)/(xb-xa)));dz=(.0020 if upper_lid else -.0011)*math.sin(math.pi*t)**.8
            target=q.copy();target.z+=dz
            oldr=(q.x-cen.x)**2+(q.z-cen.z)**2;newr=(q.x-cen.x)**2+(target.z-cen.z)**2
            if max(oldr,newr)<radius*radius:
                clearance=max(.00065,cen.y-math.sqrt(radius*radius-oldr)-q.y)
                target.y=cen.y-math.sqrt(radius*radius-newr)-clearance
            return target-q
        ring=set(upper+lower);dist={i:0 for i in ring};frontier=ring
        for step in range(1,5):
            nxt={j for i in frontier for j in neighbors[i] if j not in dist}
            dist.update({j:step for j in nxt});frontier=nxt
        delta={i:displacement(co[i],i in upper) for i in ring};weights=[1,.82,.48,.18,.035]
        key=h.shape_key_add(name='R05_aperture_'+side,from_mix=False)
        for i,v in enumerate(key.data):v.co=ks[0].data[i].co
        for i,depth in dist.items():
            nearest=min(ring,key=lambda j:(co[j]-co[i]).length_squared)
            key.data[i].co+=delta[nearest]*weights[depth]
        key.value=0;_driver(key,rig,h,side,'open')
        excess=h.shape_key_add(name='R05_blink_squint_limit_'+side,from_mix=False);blink=ks['R02_blink_'+side]
        for i,v in enumerate(excess.data):v.co=ks[0].data[i].co-(blink.data[i].co-blink.relative_key.data[i].co)
        excess.value=0;_driver(excess,rig,h,side,'excess')
        for name in ['R02 Upper lashes '+side,'R02 Upper lid rim '+side]:
            o=bpy.data.objects[name];bases=o.data.shape_keys.key_blocks[0].data;inv=o.matrix_world.inverted()
            new=o.shape_key_add(name='R05 Aperture',from_mix=False)
            for i,v in enumerate(new.data):
                v.co=bases[i].co;root=(i//5)*5 if 'lashes' in name else i
                q=o.matrix_world@bases[root].co;v.co+=inv.to_3x3()@displacement(q,True)
            new.value=0;_driver(new,rig,h,side,'open')
            _driver(o.data.shape_keys.key_blocks['Blink'],rig,h,side,'close')
        report[side]={'ringIndices':{'upper':upper,'lower':lower},'affectedVertices':len(dist),'radius':radius,'center':list(cen)}
    rig.data.pose_position=pose;bpy.context.view_layer.update()
    (project_path()/'work/reconstruction-r02/eye-aperture-r05.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));return report
