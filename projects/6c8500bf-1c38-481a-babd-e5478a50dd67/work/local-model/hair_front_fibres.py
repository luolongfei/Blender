"""Local forehead-fibre clearance repair. No nape, tail, rig or material changes."""
import bpy,hashlib,struct
from mathutils.bvhtree import BVHTree

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)));return t*t*(3-2*t)

def digest(points):
    h=hashlib.sha256()
    for p in points:h.update(struct.pack('<3f',*p))
    return h.hexdigest()

def apply(col=None,rig=None):
    """Approved-scope local repair without applying the rejected nape candidate."""
    col=col or bpy.data.collections['CHARACTER · Original geometry']
    rig=rig or bpy.data.objects['Character_Rig']
    fib=bpy.data.objects['Hair_Realism_ScalpFibres']
    key='HairFrontFloatOriginal';attr=fib.data.attributes.get(key)
    if attr is None:
        attr=fib.data.attributes.new(key,'FLOAT_VECTOR','POINT')
        for v,d in zip(fib.data.vertices,attr.data):d.vector=v.co
    if len(attr.data)!=len(fib.data.vertices):raise RuntimeError('Fibre topology changed.')
    old=[d.vector.copy() for d in attr.data]
    previous=fib.get('hair_front_float_output_sha')
    if previous and digest(v.co for v in fib.data.vertices) not in {previous,digest(old)}:
        raise RuntimeError('Front fibres were reshaped after the local checkpoint.')
    vertices=[];polygons=[]
    for obj in col.all_objects:
        if obj.type!='MESH':continue
        if obj.name!='Hair_ScalpFoundation' and not (
                obj.name.startswith('Hair_Fringe_') and '_Flow' not in obj.name):continue
        offset=len(vertices);vertices.extend(tuple(v.co) for v in obj.data.vertices)
        polygons.extend(tuple(i+offset for i in f.vertices) for f in obj.data.polygons)
    bvh=BVHTree.FromPolygons(vertices,polygons);changed=0;maximum=0.
    for vertex,p in zip(fib.data.vertices,old):
        q=p.copy()
        if p.y<-.025 and 1.495<p.z<1.555:
            hit,normal,_,distance=bvh.find_nearest(p)
            if hit is not None and distance>.0025:
                if normal.y>0:normal=-normal
                if (p-hit).dot(normal)>0:
                    region=smooth(1.495,1.505,p.z)*(1-smooth(1.545,1.555,p.z))
                    amount=smooth(.0025,.005,distance)*region
                    q=p.lerp(hit+(p-hit).normalized()*.0008,amount)
                    changed+=1;maximum=max(maximum,distance)
        vertex.co=q
    fib.data.update();fib['hair_front_float_output_sha']=digest(v.co for v in fib.data.vertices)
    fib['hair_front_float_version']='local-forehead-fibres-01'
    bpy.context.view_layer.update()
    return {'version':'local-forehead-fibres-01','object':fib.name,
            'changedVertices':changed,'maximumOriginalFloatMeters':maximum,
            'napeChanged':False,'ponytailsTiesRestBonesPhysicsChanged':False}
