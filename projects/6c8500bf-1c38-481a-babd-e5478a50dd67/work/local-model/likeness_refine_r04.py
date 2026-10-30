"""Reference-led small-nose/soft-cheek study, with preserved eye contact loops."""
import bpy,math,json,sys
from pathlib import Path
from mathutils import Vector,kdtree
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

def run():
    h=bpy.data.objects['R01 Head'];ks=h.data.shape_keys.key_blocks
    if ks.get('R04_reference_softness'):raise RuntimeError('Already studied')
    coords=[]
    for i,v in enumerate(ks[0].data):
        q=v.co.copy()
        for k in ks[1:]:q+=(k.data[i].co-k.relative_key.data[i].co)*k.value
        coords.append(q)
    key=h.shape_key_add(name='R04_reference_softness',from_mix=False)
    for i,q in enumerate(coords):
        x,y,z=q
        if z<1.32:continue
        front=max(0,min(1,(-y-.045)/.025))
        nose=math.exp(-(x/.019)**4-((z-1.401)/.019)**4)*front
        mouth=math.exp(-(x/.031)**6-((z-1.371)/.009)**4)*front
        cheek=math.exp(-((abs(x)-.042)/.022)**2-((z-1.387)/.018)**2)*front
        eye_lock=max(0,min(1,(1.414-z)/.010));eye_lock=eye_lock*eye_lock*(3-2*eye_lock)
        delta=Vector((-x*.16,.0032,0))*nose+Vector((-x*.12,0,0))*mouth
        delta+=Vector((math.copysign(.0007,x),-.0025,0))*cheek
        key.data[i].co+=delta*eye_lock
    key.value=.8
    # Use the actual preserved lip-ring topology for restrained natural colour.
    old=bpy.data.objects['R03 prejoin Head preserved'];tree=kdtree.KDTree(len(h.data.vertices))
    for i,v in enumerate(ks[0].data):tree.insert(v.co,i)
    tree.balance();lm=json.loads((project_path()/'work/reconstruction-r02/lip-landmarks.json').read_text())
    rings=[(lm['contact_rim_cycle'],.72),(lm['outer_vermilion_cycle'],.40)]
    attr=h.data.color_attributes['R01_region_color'];colored=set()
    for ring,blend in rings:
        for oi in ring:
            _,ni,d=tree.find(old.data.shape_keys.key_blocks[0].data[oi].co)
            if d>1e-5:raise RuntimeError('Lip topology mapping no longer exact')
            if ni in colored:continue
            col=attr.data[ni].color;target=(.43,.225,.175,1.)
            attr.data[ni].color=tuple(col[j]*(1-blend)+target[j]*blend for j in range(4));colored.add(ni)
    bpy.context.view_layer.update();print('R04_REFERENCE_STUDY',len(colored))

if __name__=='__main__':run()
