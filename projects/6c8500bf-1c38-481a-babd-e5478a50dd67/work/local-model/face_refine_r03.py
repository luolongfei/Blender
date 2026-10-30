"""Local fairing of the adapted face; never replay against the migration master."""
import bpy, math
from mathutils import Vector

def run():
    h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
    if ks.get('R03_facial_balance'):raise RuntimeError('Already applied')
    coords=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        coords.append(q)
    adjacency=[set() for _ in coords]
    for e in h.data.edges:
        a,b=e.vertices;adjacency[a].add(b);adjacency[b].add(a)
    work=[q.copy() for q in coords]
    weights=[]
    for q in coords:
        x,y,z=q
        cheek=math.exp(-((abs(x)-.043)/.026)**4-((z-1.405)/.027)**4)
        front=max(0,min(1,(-y-.03)/.035))
        eyeprotect=1-math.exp(-((z-1.445)/.009)**4)
        mouthprotect=1-math.exp(-((z-1.389)/.01)**4-(x/.026)**6)
        weights.append(cheek*front*eyeprotect*mouthprotect)
    for _ in range(14):
        nxt=[]
        for i,q in enumerate(work):
            avg=sum((work[j] for j in adjacency[i]),Vector())/len(adjacency[i]) if adjacency[i] else q
            nxt.append(q.lerp(avg,.20*weights[i]))
        work=nxt
    key=h.shape_key_add(name='R03_facial_balance',from_mix=False)
    for i,q in enumerate(coords):
        x,y,z=q;front=max(0,min(1,(-y-.045)/.025))
        delta=work[i]-q
        cheek=math.exp(-((abs(x)-.042)/.023)**2-((z-1.408)/.023)**2)*front
        delta+=Vector((math.copysign(.0012,x),-.002,0))*cheek
        nose=math.exp(-(x/.019)**4-((z-1.418)/.021)**4)*front
        delta+=Vector((-x*.075,.0018,0))*nose
        key.data[i].co+=delta
    key.value=1
    bpy.context.view_layer.update()
    print('R03_FACE_BALANCE_CREATED')

if __name__=='__main__':run()
