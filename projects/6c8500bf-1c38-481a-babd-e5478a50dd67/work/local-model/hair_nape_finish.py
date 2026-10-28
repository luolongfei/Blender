"""Post-integration fixed nape gathering, with local floated-fibre correction.
Never replays prior head/hair deformation modules. Existing topology, UVs,
materials, weights, ponytails, ties, rest bones and physics are preserved.
"""
from pathlib import Path
import bpy,json,math,hashlib,struct,sys
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import closest_point_on_tri

VERSION='gathered-fixed-nape-02'
BACKUP='HairNapeFinishOriginal'
UV_BACKUP='HairNapeFlowUVOriginal'
P=Path(__file__).resolve().parents[2]

def smooth(a,b,x):
    t=max(0.,min(1.,(x-a)/(b-a)));return t*t*(3-2*t)

def digest(points):
    h=hashlib.sha256()
    for p in points:h.update(struct.pack('<3f',*p))
    return h.hexdigest()

def params(obj):
    uv=obj.data.uv_layers['UVMap'];r={}
    for l in obj.data.loops:r.setdefault(l.vertex_index,tuple(uv.data[l.index].uv))
    return r

class Surface:
    def __init__(self,obj,points):
        self.points=points;pp=params(obj);rows={}
        for i,(u,t) in pp.items():rows.setdefault(round(t,6),[]).append(i)
        self.rows=[(t,sorted(indices,key=lambda i:pp[i][0])) for t,indices in sorted(rows.items())]
        self.n=len(self.rows[0][1])
    def point(self,u,t):
        t=max(0,min(1,t));row=min(len(self.rows)-2,max(0,int(t*(len(self.rows)-1))))
        ta,ia=self.rows[row];tb,ib=self.rows[row+1];f=max(0,min(1,(t-ta)/max(1e-9,tb-ta)))
        s=(u%1)*self.n;i=int(s)%self.n;j=(i+1)%self.n
        a=self.points[ia[i]].lerp(self.points[ia[j]],s-int(s))
        b=self.points[ib[i]].lerp(self.points[ib[j]],s-int(s))
        return a.lerp(b,f)

def checkpoint(obj):
    attr=obj.data.attributes.get(BACKUP)
    if attr is None:
        attr=obj.data.attributes.new(BACKUP,'FLOAT_VECTOR','POINT')
        for v,d in zip(obj.data.vertices,attr.data):d.vector=v.co
    elif len(attr.data)!=len(obj.data.vertices):raise RuntimeError('Nape topology changed: '+obj.name)
    old=digest(d.vector for d in attr.data);now=digest(v.co for v in obj.data.vertices)
    previous=obj.get('hair_nape_finish_output_sha')
    if previous and now not in {previous,old}:raise RuntimeError('Nape geometry edited after checkpoint: '+obj.name)
    return [d.vector.copy() for d in attr.data]

def tree(obj,points):
    return BVHTree.FromPolygons([tuple(p) for p in points],[tuple(f.vertices) for f in obj.data.polygons])

def transfer(point,cap,old,new,bvh):
    hit,normal,face,distance=bvh.find_nearest(point)
    if hit is None or distance>.007:return None
    ids=list(cap.data.polygons[face].vertices);best=None
    for k in range(1,len(ids)-1):
        inds=(ids[0],ids[k],ids[k+1]);a,b,c=(old[i] for i in inds)
        q=closest_point_on_tri(point,a,b,c);err=(q-point).length_squared
        ab,ac,aq=b-a,c-a,q-a;den=ab.dot(ab)*ac.dot(ac)-ab.dot(ac)**2
        if abs(den)<1e-18:continue
        v=(ac.dot(ac)*aq.dot(ab)-ab.dot(ac)*aq.dot(ac))/den
        w=(ab.dot(ab)*aq.dot(ac)-ab.dot(ac)*aq.dot(ab))/den
        weights=(1-v-w,v,w)
        delta=sum(((new[i]-old[i])*f for i,f in zip(inds,weights)),Vector())
        if best is None or err<best[0]:best=(err,point+delta)
    if best is None:return None
    mapped=best[1]
    if (mapped-point).length<1e-8:return point.copy()
    # The backing shell stays completely unchanged. Keep transported detail
    # on that same shell instead of cutting through its convex silhouette.
    hit2,normal2,_,_=bvh.find_nearest(mapped)
    if normal.y<0:normal=-normal
    if normal2.y<0:normal2=-normal2
    offset=max(-.0002,min(.0035,(point-hit).dot(normal)))
    return hit2+normal2*offset

def flow_u(u,t):
    back=smooth(.23,.30,u)*(1-smooth(.70,.77,u))
    amount=.88*smooth(.64,1,t)*back
    target=.5+.20*math.tanh((u-.5)/.065)
    return u+(target-u)*amount

def inverse_flow_u(u,t):
    low,high=0.,1.
    for _ in range(28):
        middle=(low+high)*.5
        if flow_u(middle,t)<u:low=middle
        else:high=middle
    return (low+high)*.5

def invariant(collection,rig):
    h=hashlib.sha256()
    for o in sorted(collection.all_objects,key=lambda x:x.name):
        if o.type!='MESH':continue
        if o.name.startswith('Hair_') and o.get('hair_region') not in {'ponytail','tie'}:continue
        h.update(o.name.encode());h.update(digest(v.co for v in o.data.vertices).encode())
        if o.data.shape_keys:
            for key in o.data.shape_keys.key_blocks:
                h.update(key.name.encode());h.update(digest(v.co for v in key.data).encode());h.update(str(key.value).encode())
    for b in rig.data.bones:
        h.update(b.name.encode());h.update(digest((b.head_local,b.tail_local)).encode())
    return h.hexdigest()

def apply_nape_candidate(col=None,rig=None):
    col=col or bpy.data.collections['CHARACTER · Original geometry']
    rig=rig or bpy.data.objects['Character_Rig'];cap=bpy.data.objects['Hair_ScalpFoundation']
    if cap.get('hair_refinement_version')!='fine-gathered-mobile-hair-07':raise RuntimeError('Requires integrated hair refinement 07.')
    protected=invariant(col,rig)
    fixed=[o for o in col.all_objects if o.type=='MESH' and o.name.startswith('Hair_') and o.get('hair_region') not in {'ponytail','tie'}]
    old={o.name:checkpoint(o) for o in fixed};new={n:[v.copy() for v in p] for n,p in old.items()}
    sf=Surface(cap,old[cap.name]);cap_tree=tree(cap,old[cap.name]);pp=params(cap)
    transported_cap=[p.copy() for p in old[cap.name]]
    for i,(u,t) in pp.items():
        if t>.64 and .23<u<.77:
            transported_cap[i]=sf.point(flow_u(u,t),t)
    # Keep every shell vertex and the entire old hairline exactly in place.
    # Only the existing detail strands and the matching longitudinal texture
    # flow are gathered toward the ties.
    uv=cap.data.uv_layers['HairRealismUV']
    uv_backup=cap.data.attributes.get(UV_BACKUP)
    if uv_backup is None:
        uv_backup=cap.data.attributes.new(UV_BACKUP,'FLOAT_VECTOR','CORNER')
        for d,a in zip(uv.data,uv_backup.data):a.vector=(d.uv.x,d.uv.y,0)
    if len(uv_backup.data)!=len(uv.data):raise RuntimeError('Nape UV topology changed.')
    uv_changed=0
    for loop,d,a in zip(cap.data.loops,uv.data,uv_backup.data):
        u,t=pp[loop.vertex_index];delta=0.
        if t>.64 and .23<u<.77:delta=inverse_flow_u(u,t)-u
        d.uv=(a.vector.x+delta,a.vector.y)
        if abs(delta)>1e-7:uv_changed+=1
    mapped_counts={}
    for obj in fixed:
        if obj==cap:continue
        count=0
        for i,p in enumerate(old[obj.name]):
            if p.y>.003 and p.z<1.522:
                q=transfer(p,cap,old[cap.name],transported_cap,cap_tree)
                if q is not None and (q-p).length>1e-8:new[obj.name][i]=q;count+=1
        if count:mapped_counts[obj.name]=count
    # Nearest visible fixed surfaces, excluding the fine fibres themselves.
    vertices=[];polygons=[]
    for obj in fixed:
        if obj!=cap and not (obj.name.startswith('Hair_Fringe_') and '_Flow' not in obj.name):continue
        offset=len(vertices);vertices.extend(tuple(p) for p in new[obj.name]);polygons.extend(tuple(i+offset for i in f.vertices) for f in obj.data.polygons)
    front_tree=BVHTree.FromPolygons(vertices,polygons)
    floats=[];max_float=0.;fib=bpy.data.objects['Hair_Realism_ScalpFibres']
    for i,p in enumerate(new[fib.name]):
        if fib.get('hair_front_float_version')=='local-forehead-fibres-01':continue
        if not (p.y<-.025 and 1.495<p.z<1.555):continue
        hit,normal,_,distance=front_tree.find_nearest(p)
        if hit is None or distance<=.0025:continue
        if normal.y>0:normal=-normal
        if (p-hit).dot(normal)<=0:continue
        region=smooth(1.495,1.505,p.z)*(1-smooth(1.545,1.555,p.z))
        amount=smooth(.0025,.005,distance)*region
        target=hit+(p-hit).normalized()*.0008
        new[fib.name][i]=p.lerp(target,amount);floats.append(i);max_float=max(max_float,distance)
    changed=[];max_delta=0.
    for obj in fixed:
        count=0
        for v,a,b in zip(obj.data.vertices,old[obj.name],new[obj.name]):
            d=(b-a).length;max_delta=max(max_delta,d)
            if d>1e-8:count+=1
            v.co=b
        obj.data.update();obj['hair_nape_finish_version']=VERSION
        obj['hair_nape_finish_output_sha']=digest(v.co for v in obj.data.vertices)
        if count:changed.append({'name':obj.name,'vertices':count})
    bpy.context.view_layer.update()
    if invariant(col,rig)!=protected:raise RuntimeError('Protected geometry, face keys or rest bones changed.')
    assert digest(new[cap.name])==digest(old[cap.name]),'Nape shell and coverage must be exact.'
    report={'version':VERSION,'changed':changed,'maximumDisplacementMeters':max_delta,'mappedFixedDetailVertices':mapped_counts,'localFrontFibreVertices':len(floats),'maximumOriginalFrontFibreFloatMeters':max_float,'capGeometryAndHairlineExact':True,'capFlowUVLoopsChanged':uv_changed,'topologyMaterialsWeightsPreserved':True,'ponytailsTiesFaceBodyRestBonesUnchanged':True,'newMeshes':0,'newTriangles':0}
    cap['hair_nape_finish_report']=json.dumps(report)
    return report

def restore(col=None):
    col=col or bpy.data.collections['CHARACTER · Original geometry']
    for obj in col.all_objects:
        if obj.type!='MESH' or BACKUP not in obj.data.attributes:continue
        for v,d in zip(obj.data.vertices,obj.data.attributes[BACKUP].data):v.co=d.vector
        if UV_BACKUP in obj.data.attributes:
            for d,a in zip(obj.data.uv_layers['HairRealismUV'].data,obj.data.attributes[UV_BACKUP].data):
                d.uv=(a.vector.x,a.vector.y)
        obj.data.update()
    bpy.context.view_layer.update()

def apply_front_fibres(col=None,rig=None):
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

def apply(col=None,rig=None):
    """The default entry point is the safe, isolated foreground-fibre repair."""
    return apply_front_fibres(col,rig)

def review():
    if not bpy.app.background:raise RuntimeError('QA requires disposable background Blender.')
    out=P/'work/qa-hair-nape';source=Path(bpy.data.filepath).resolve()
    if source!=out/'source-snapshot.blend':raise RuntimeError('Only frozen QA source is supported.')
    sha=hashlib.sha256(source.read_bytes()).hexdigest();col=bpy.data.collections['CHARACTER · Original geometry']
    initial={o.name:digest(v.co for v in o.data.vertices) for o in col.all_objects if o.type=='MESH'}
    r=apply_nape_candidate();first={o.name:digest(v.co for v in o.data.vertices) for o in col.all_objects if o.type=='MESH'}
    apply_nape_candidate();assert first=={o.name:digest(v.co for v in o.data.vertices) for o in col.all_objects if o.type=='MESH'}
    restore();assert initial=={o.name:digest(v.co for v in o.data.vertices) for o in col.all_objects if o.type=='MESH'}
    r=apply_nape_candidate();r.update(source=str(source),sourceSha256=sha,idempotent=True,restoreExact=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'nape-review.blend'))
    (out/'report.json').write_text(json.dumps(r,indent=2))
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12;scene.cycles.use_denoising=True
    scene.render.threads_mode='FIXED';scene.render.threads=2
    scene.render.resolution_x=600;scene.render.resolution_y=700;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    d=bpy.data.cameras.new('Fixed nape review camera');cam=bpy.data.objects.new('Fixed nape review camera',d);scene.collection.objects.link(cam);scene.camera=cam
    d.type='ORTHO';d.ortho_scale=.40;target=Vector((0,.005,1.445))
    views=[('back',(0,3,1.445))]
    if '--back-only' not in sys.argv:views.append(('profile',(3,0,1.445)))
    for name,location in views:
        cam.location=location;cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    print('NAPE_READY',json.dumps(r))

if __name__=='__main__':review()
