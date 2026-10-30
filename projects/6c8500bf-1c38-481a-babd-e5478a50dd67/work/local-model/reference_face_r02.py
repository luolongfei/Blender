"""Reference-led face proportion study on an isolated R02 copy."""
import bpy,math
from mathutils import Vector

def run():
    h=bpy.data.objects['R01 Head']
    if h.data.shape_keys.key_blocks.get('R02_reference_likeness'): raise RuntimeError('R02 already applied')
    h.data.shape_keys.key_blocks['R01_B_soft_oval'].value=0
    def shape(co):
        x,y,z=co; front=max(0,min(1,(-y+.01)/.06))
        if 1.335<=z<1.445: zz=1.445+(z-1.445)*.82
        elif z<1.335: zz=z+.0198*max(0,(z-1.24)/.095)
        else: zz=z
        cheek=math.exp(-((z-1.397)/.027)**2)*front
        jaw=math.exp(-((z-1.35)/.024)**2)*front
        xx=x*(1+.075*cheek-.12*jaw)
        yy=y
        nose=math.exp(-(x/.023)**4-((z-1.416)/.036)**4)*front
        yy+=.008*nose
        xx*=1-.11*nose
        for s in [-1,1]:
            cx=s*.03068; cz=1.44505; dx=x-cx; dz=z-cz
            f=math.exp(-((dx/.028)**2+(dz/.030)**2))*max(0,min(1,(-y-.035)/.035))
            xx+=f*(.30*dx+s*.0017)
            zz+=f*(.90*dz)
            yy-=.0025*f
        return Vector((xx,yy,zz))
    k=h.shape_key_add(name='R02_reference_likeness')
    for v,kv in zip(h.data.vertices,k.data): kv.co=shape(v.co)
    k.value=1
    # Matched globes and iris patches enlarge with the eyelid volume.
    for side,s in [('L',1),('R',-1)]:
        eye=bpy.data.objects['R01 Sclera '+side]
        vs=[v.co for v in eye.data.vertices]
        cen=Vector([(min(v[i] for v in vs)+max(v[i] for v in vs))/2 for i in range(3)])
        target=cen+Vector((s*.0017,-.002,0))
        for name in ['R01 Sclera '+side,'R01 Textured iris '+side]:
            o=bpy.data.objects[name]
            for v in o.data.vertices:
                d=v.co-cen
                v.co=target+d*1.30
    # Mild closed-mouth smile, separate from the likeness control.
    smile=h.shape_key_add(name='R02_gentle_smile')
    for v,sv in zip(h.data.vertices,smile.data):
        sv.co=v.co
        x,y,z=v.co
        lip=math.exp(-((z-1.378)/.012)**4-(abs(x)/.033)**6)*max(0,min(1,(-y-.075)/.016))
        corner=math.exp(-((abs(x)-.023)/.010)**2-((z-1.378)/.013)**2)*max(0,min(1,(-y-.07)/.02))
        sv.co.z+=.0035*corner
        if 1.373<z<1.380: sv.co.z+=.0010*lip
        smile.relative_key=h.data.shape_keys.key_blocks['Basis']
    smile.value=.65
    # Clear the provisional dark scalp color; next groom provides its own scalp coverage.
    attr=h.data.color_attributes['R01_region_color']
    for v,c in zip(h.data.vertices,attr.data):
        x,y,z=v.co
        lip=math.exp(-((x/.025)**6)-((z-1.378)/.008)**4)*max(0,min(1,(-y-.09)/.007))
        c.color=(.49-.07*lip,.31-.105*lip,.225-.065*lip,1)
    for o in bpy.data.objects:
        if o.name.startswith('R01 Brow'): o.hide_render=True; o.hide_set(True)
    bpy.context.view_layer.update()
    print('R02_REFERENCE_FACE')

if __name__=='__main__': run()
