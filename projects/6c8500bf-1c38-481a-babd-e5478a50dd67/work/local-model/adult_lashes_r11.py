"""Static lash candidate rooted on the actual visible adult eyelid surface.

The socket control loop is not used as a lash line. Frontal visibility of the
actual globe against the actual evaluated face determines the aperture. Roots
are then placed on skin just outside that silhouette. This remains a neutral
candidate; a facial surface binding is required before blink animation.
"""
import argparse, hashlib, json, math, random, runpy, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PREFIX='R11 Adult lashes'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))

def build(head,collection):
    import bpy
    import numpy as np
    from mathutils import Matrix,Vector
    from mathutils.bvhtree import BVHTree
    if not bpy.app.background or bpy.data.objects.get(PREFIX+' upper L'):raise RuntimeError('New independent background candidate required')
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Explicit REST required')
    if head.name!='R11 Female head study':raise RuntimeError('Explicit new adult head required')
    dg=bpy.context.evaluated_depsgraph_get()
    def surface(obj):
        ev=obj.evaluated_get(dg);me=ev.data
        v=np.empty((len(me.vertices),3));me.vertices.foreach_get('co',v.ravel());m=np.asarray(ev.matrix_world)
        v=v@m[:3,:3].T+m[:3,3]
        return BVHTree.FromPolygons([Vector(p) for p in v],[tuple(p.vertices) for p in me.polygons]),v
    skin,_=surface(head);forward=Vector((0,1,0))
    def ray(bvh,x,z):return bvh.ray_cast(Vector((float(x),-.35,float(z))),forward,.7)
    material=bpy.data.materials.new(PREFIX+' soft black');material.use_nodes=True
    nt=material.node_tree;nt.nodes.clear();out=nt.nodes.new('ShaderNodeOutputMaterial')
    hair=nt.nodes.new('ShaderNodeBsdfHairPrincipled');hair.parametrization='COLOR'
    hair.inputs['Color'].default_value=(.003,.002,.0015,1);hair.inputs['Roughness'].default_value=.48
    nt.links.new(hair.outputs[0],out.inputs['Surface'])
    report={'head':head.name,'objects':{},'facialSurfaceBindingComplete':False,'appearanceAccepted':False,
            'method':'Actual front-visible globe contour, bisection to <1 micrometer, root on actual skin outside contour; not socket control ring.',
            'limits':['Neutral static candidate only. Object parenting follows head motion; eyelid muscle and blink surface binding remains required.']}
    report['actualHairShader']={'model':getattr(hair,'model',None),'parametrization':hair.parametrization,
        'inputs':{s.name:(list(s.default_value) if hasattr(s.default_value,'__len__') else s.default_value) for s in hair.inputs if hasattr(s,'default_value')}}
    for side in ('L','R'):
        globe,verts=surface(bpy.data.objects['R11 Adult sclera '+side]);lo=verts.min(0);hi=verts.max(0)
        def visible(x,z):
            g=ray(globe,x,z)[0]
            if g is None:return False
            h=ray(skin,x,z)[0]
            return h is None or g.y<h.y-1e-7
        contours=[]
        for x in np.linspace(lo[0]+.00035,hi[0]-.00035,151):
            zs=np.linspace(lo[2]+.0001,hi[2]-.0001,121)
            flags=[visible(x,z) for z in zs];ids=np.flatnonzero(flags)
            if len(ids)<3 or ids[0]==0 or ids[-1]==len(zs)-1:continue
            if not np.all(np.diff(ids)==1):raise RuntimeError('Non-contiguous visible eye aperture')
            row=[float(x)]
            for j,outside in ((int(ids[0]),int(ids[0]-1)),(int(ids[-1]),int(ids[-1]+1))):
                inside_z=float(zs[j]);outside_z=float(zs[outside])
                for _ in range(12):
                    mid=(inside_z+outside_z)*.5
                    if visible(x,mid):inside_z=mid
                    else:outside_z=mid
                row.append((inside_z+outside_z)*.5)
            contours.append(row)
        contour=np.asarray(contours)
        if len(contour)<80:raise RuntimeError('Insufficient actual visible eyelid span')
        sign=1 if side=='L' else -1
        # Medial to lateral growth, excluding the bare tear-duct interval.
        medial=contour[0,0] if sign==1 else contour[-1,0]
        lateral=contour[-1,0] if sign==1 else contour[0,0]
        rng=random.Random(11028+(side=='R'))
        for part,count,dzsign,column in (('upper',156,1,2),('lower',48,-1,1)):
            paths=[];radii=[];roots=[];lengths=[]
            for i in range(count):
                t=.08+.86*(i+.2+.6*rng.random())/count;x=medial+(lateral-medial)*t
                aperture_z=float(np.interp(x,contour[:,0],contour[:,column]))
                # The wet inner rim faces into the eye. Lashes arise from the
                # outer ciliary margin, beyond that rounded lid thickness.
                # Find that actual outward-facing skin instead of burying a
                # strand behind the rim and counting its mere existence.
                for margin in np.linspace(.00025,.00240,87):
                    z=aperture_z+dzsign*float(margin);p,n,fi,d=ray(skin,x,z)
                    if p is not None and n.y<-.50 and n.z*dzsign>-.25:break
                else:raise RuntimeError('Actual outer ciliary margin missing '+side+' '+part+' '+str(i))
                # Stagger the follicles over the real narrow ciliary strip,
                # instead of a single mathematically regular comb row.
                extra=rng.uniform(0.,.00026 if part=='upper' else .00014)
                z+=dzsign*extra;p,n,fi,d=ray(skin,x,z)
                if p is None or n.y>-.45:raise RuntimeError('Staggered ciliary root lost outer skin')
                root=p+n*.000025
                length=(.0033+.0040*math.sin(math.pi*t)**.7)*rng.uniform(.72,1.10)
                if part=='lower':length*=.43
                radius=rng.uniform(.000032,.000050)*(1 if part=='upper' else .62)
                lateral_fan=sign*((t-.32)*.8+rng.uniform(-.09,.09))
                # Root growth exits the local lid plane before curling. A
                # fixed upward vector can travel through the convex upper lid.
                direction=(n*.65+Vector((lateral_fan,-1,dzsign*.10))).normalized()
                path=[]
                for j in range(9):
                    u=j/8;curl=.50 if part=='upper' else .28
                    point=root+direction*length*u+Vector((0,.13*length*u*u,dzsign*curl*length*u*u))
                    # Thin physical shaft cannot be intentionally buried in skin.
                    q,normal,index,distance=skin.find_nearest(point)
                    if j>0 and (point-q).dot(normal)<-radius:
                        raise RuntimeError('Lash shaft enters face '+json.dumps({'side':side,'part':part,'strand':i,'point':j,'root':list(root),'rootNormal':list(n),'pointWorld':list(point),'nearestWorld':list(q),'nearestNormal':list(normal),'signedDistance':(point-q).dot(normal)}))
                    path.append(tuple(point));radii.append(radius*max(.025,(1-u)**.62))
                paths.extend(path);lengths.append(length)
                roots.append({'world':list(root),'surface':list(p),'normal':list(n),'evaluatedFace':int(fi),'apertureFraction':t,'outerMarginDistance':float(margin)+extra})
            data=bpy.data.hair_curves.new(PREFIX+' '+part+' '+side);data.add_curves([9]*count)
            data.position_data.foreach_set('vector',np.asarray(paths,dtype=np.float32).ravel())
            data.attributes.new('radius','FLOAT','POINT').data.foreach_set('value',np.asarray(radii,dtype=np.float32))
            data.materials.append(material);obj=bpy.data.objects.new(data.name,data);collection.objects.link(obj)
            obj.parent=head;obj.matrix_parent_inverse=head.matrix_world.inverted();obj.matrix_basis=Matrix.Identity(4)
            obj['R11_lash_root_record']=json.dumps(roots);obj['R11_surface_binding']='pending; neutral only'
            report['objects'][obj.name]={'curves':count,'points':count*9,'minimumLength':min(lengths),'maximumLength':max(lengths),'roots':roots}
    bpy.context.view_layer.update()
    return report

def main():
    import bpy
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);P=project_path();source=(P/a.source).resolve();output=(P/a.output).resolve()
    for path in (source,output):path.relative_to((P/'work/reconstruction-r02').resolve())
    if bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source or sha(source)!=a.source_sha:raise RuntimeError('Fresh explicit source required')
    if output.exists() or output.with_suffix('.lashes.json').exists():raise RuntimeError('Preserve prior candidates')
    head=bpy.data.objects['R11 Female head study'];rig=bpy.data.objects['Character_Rig'];scene=bpy.context.scene
    geometry=api('body_hair_driver_r09')['_geometry'];before=geometry(head)
    pose=rig.data.pose_position;frame=(scene.frame_current,scene.frame_subframe)
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        report=build(head,bpy.data.collections['R11 Adult female face candidate'])
    finally:
        rig.data.pose_position=pose;scene.frame_set(frame[0],subframe=frame[1]);bpy.context.view_layer.update()
    if geometry(head)!=before or sha(source)!=a.source_sha:raise RuntimeError('Protected head or input changed')
    report.update(source=a.source,sourceSha256=a.source_sha,output=a.output,helperSha256=sha(__file__),headUnchanged=True)
    scene['R11_adult_lash_candidate']=json.dumps(report)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Save failed')
    report['outputSha256']=sha(output)
    with output.with_suffix('.lashes.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print('R11_ADULT_LASHES_SAVED '+json.dumps({'output':a.output,'sha256':report['outputSha256']}),flush=True)
if __name__=='__main__':main()
