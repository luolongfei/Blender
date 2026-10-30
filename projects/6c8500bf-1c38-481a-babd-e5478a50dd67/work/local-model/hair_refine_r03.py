"""R03 scalp, forelocks and curved side wisps for the supplied character.

Run only in an independent candidate. No file IO, save, render, or simulation.
Existing twin tails, ties, brows and lashes are retained. New strands are native
CURVES in world space and must be attached by the sole live-scene integrator.

Sparse guide controls are projected onto the evaluated head in world space.
Smooth guide interpolation and local clearance correction replace the previous
per-point front-ray/fallback-height technique. Randomness is coherent per strand.
"""
import math
import random
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SEED = 302028
PREFIX = 'R03 '
CENTER = Vector((0.0, 0.003, 1.445))
REFERENCE_HEAD_TOP = 1.553


def _evaluated_surface(head):
    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = head.evaluated_get(dg)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        vertices = [matrix @ v.co for v in mesh.vertices]
        polygons = [tuple(p.vertices) for p in mesh.polygons]
        # A portable guard against applying this character-specific guide layout
        # to an unrelated object or a non-neutral pose.
        top = max(v.z for v in vertices)
        if not 1.49 < top < 1.60:
            raise RuntimeError(f'Unexpected head height {top:.4f}; use the R02 rest candidate')
        return BVHTree.FromPolygons(vertices, polygons), vertices, polygons
    finally:
        evaluated.to_mesh_clear()


def _catmull(points, samples=55):
    """Interpolating C1 guide, with duplicated endpoint tangents."""
    padded = [points[0]] + list(points) + [points[-1]]
    result = []
    spans = len(points)-1
    for j in range(samples):
        u = j/(samples-1)*spans
        i = min(int(u), spans-1)
        t = u-i
        p0,p1,p2,p3 = padded[i:i+4]
        result.append((2*p1+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t)*0.5)
    return result


def _native(name, paths, material, radius, created):
    sizes = [len(p) for p in paths]
    data = bpy.data.hair_curves.new(name)
    data.add_curves(sizes)
    data.position_data.foreach_set('vector', [f for p in paths for q in p for f in q])
    radii = data.attributes.new('radius', 'FLOAT', 'POINT')
    values = []
    for pi,path in enumerate(paths):
        variation = 0.82 + 0.34*((pi*0.61803398875)%1)
        for i in range(len(path)):
            t=i/(len(path)-1)
            values.append(radius*variation*max(0.055,(1-t**2.2)**0.55))
    radii.data.foreach_set('value', values)
    data.materials.append(material)
    obj=bpy.data.objects.new(name,data)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_render=True
    obj.hide_set(True)
    obj['R03_scope']='Static editable guide-derived native strands; new attachment/dynamics not tested'
    obj['R03_seed']=SEED
    obj['R03_groom_pass']=3
    data.update_tag()
    created.append(obj)
    return obj


def _strand_bundle(guide, count, width, depth, rng, *, center=CENTER, tip_taper=True, lift_bias=0.0):
    """Coherent clump cross-sections; no independent point jitter."""
    paths=[]
    frames=[]
    for j,q in enumerate(guide):
        tangent=guide[min(j+1,len(guide)-1)]-guide[max(0,j-1)]
        tangent.normalize()
        # A horizontal-oriented cross-section for the wide forelocks; its normal
        # faces away from the head so thickness does not jump across the surface.
        lateral=Vector((1,0,0))-tangent*tangent.x
        if lateral.length < 0.1:
            lateral=Vector((0,1,0))-tangent*tangent.y
        lateral.normalize()
        normal=tangent.cross(lateral).normalized()
        if normal.dot(q-center)<0:normal.negate()
        frames.append((lateral,normal))
    for _ in range(count):
        xoff=rng.uniform(-1,1)
        noff=rng.uniform(0.1,1.0)
        phase=rng.uniform(0,math.tau)
        length=rng.uniform(.94,1.0) if tip_taper else 1.0
        stop=max(8,int((len(guide)-1)*length)+1)
        arr=[]
        for j,q in enumerate(guide[:stop]):
            t=j/(len(guide)-1)
            # Roots overlap instead of sharing an exposed straight border.
            spread=(.72+.30*math.sin(math.pi*t))
            if tip_taper:spread*=max(.04,1-t**1.55)
            lateral,normal=frames[j]
            wave=math.sin(phase+t*5.8)*.00025*math.sin(math.pi*t)
            arr.append(q+lateral*(width*xoff*spread+wave)+normal*(depth*noff+lift_bias))
        paths.append(arr)
    return paths


def _underlay(vertices, polygons, created, center, head_shift_z):
    """Subtle actual-scalp root substrate, entirely beneath the guide hairline."""
    selected=[]
    for face in polygons:
        valid=True
        for i in face:
            q=vertices[i]; delta=q-center
            radial=max(.01,math.hypot(delta.x,delta.y))
            front=max(0,-delta.y/radial)
            side=abs(delta.x)/radial
            threshold=1.452+head_shift_z+.060*front+.014*side
            if q.z<threshold:
                valid=False;break
        if valid:selected.append(face)
    used=sorted({i for f in selected for i in f})
    remap={old:new for new,old in enumerate(used)}
    coords=[]
    for i in used:
        q=vertices[i]
        coords.append(q+(q-center).normalized()*.00045)
    mesh=bpy.data.meshes.new('R03 root substrate')
    mesh.from_pydata(coords,[],[tuple(remap[i] for i in f) for f in selected])
    mesh.update()
    mat=bpy.data.materials.new('R03 matte dark roots')
    mat.use_nodes=True
    n=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    n.inputs['Base Color'].default_value=(.006,.005,.0048,1)
    n.inputs['Roughness'].default_value=.88
    mesh.materials.append(mat)
    for f in mesh.polygons:f.use_smooth=True
    obj=bpy.data.objects.new('R03 scalp root substrate',mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.hide_render=True;obj.hide_set(True)
    obj['R03_scope']='Actual evaluated scalp root substrate; boundary is inset above visible hairline'
    obj['R03_groom_pass']=3
    created.append(obj)


def run(head_shift_z=None):
    """Build once in the lowered or original head candidate.

    None infers the displacement from the actual evaluated crown, relative to
    the reference crown at Z=1.553 m. An explicit value must agree within 3 mm.
    This shifts only new groom controls, never the head or preserved twin tails.
    """
    if bpy.data.filepath.replace('\\','/').endswith('/work/character.blend'):
        raise RuntimeError('Use an independent work copy; never groom the master directly')
    if any(o.get('R03_groom_pass') == 3 for o in bpy.context.scene.objects):
        raise RuntimeError('R03 groom objects already exist; do not stack this pass')
    head=bpy.data.objects.get('R01 Head')
    if not head:raise RuntimeError('R01 Head is required')
    bpy.context.view_layer.update()
    bvh,vertices,polygons=_evaluated_surface(head)
    inferred_shift=max(v.z for v in vertices)-REFERENCE_HEAD_TOP
    if head_shift_z is None:
        head_shift_z=inferred_shift
    else:
        head_shift_z=float(head_shift_z)
        if abs(head_shift_z-inferred_shift)>.003:
            raise RuntimeError(f'Requested groom shift {head_shift_z:.4f} disagrees with evaluated crown shift {inferred_shift:.4f}')
    shift=Vector((0,0,head_shift_z))
    center=CENTER+shift
    def point(q):
        return Vector(q)+shift
    rng=random.Random(SEED)
    created=[]
    source=bpy.data.materials.get('R01 soft black hair')
    if not source:raise RuntimeError('Source hair material is required')
    material=source.copy();material.name='R03 natural charcoal strands'
    node=next((n for n in material.node_tree.nodes if n.type=='BSDF_HAIR_PRINCIPLED'),None)
    if node:
        if node.inputs.get('Color'):node.inputs['Color'].default_value=(.0065,.0055,.0050,1)
        if node.inputs.get('Roughness'):node.inputs['Roughness'].default_value=.53
        if node.inputs.get('Radial Roughness'):node.inputs['Radial Roughness'].default_value=.40

    def radial(direction,lift=.0024):
        direction=Vector(direction).normalized()
        hit,normal,index,distance=bvh.ray_cast(center+direction*.35,-direction)
        if hit is None:
            raise RuntimeError('A scalp control ray missed; no arbitrary front/height fallback is allowed')
        return hit+normal*lift

    def sphere(theta,phi,side,lift=.0024):
        return radial((side*math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)),lift)

    def keypoint(q,lift=.0025):
        return radial(point(q)-center,lift)

    def safe_guide(controls,samples=55):
        guide=_catmull(controls,samples)
        # Contact-only correction of a continuous guide. Free-hanging portions
        # stay free; no camera-facing ray is used and no height is changed on miss.
        for j,q in enumerate(guide):
            hit,normal,index,distance=bvh.find_nearest(q)
            if hit is not None and (q-hit).dot(normal)<.0016 and distance<.012:
                guide[j]=hit+normal*.0022
        # One smoothing pass removes small nearest-normal changes around ears.
        smoothed=[guide[0]]
        for j in range(1,len(guide)-1):
            smoothed.append(guide[j]*.72+(guide[j-1]+guide[j+1])*.14)
        smoothed.append(guide[-1])
        if max((b-a).length for a,b in zip(smoothed,smoothed[1:]))>.018:
            raise RuntimeError('Discontinuous groom guide rejected')
        return smoothed

    # Side and back surface guides fan continuously from variable crown/side roots
    # into the existing low-tail attachment points. Last controls are free and
    # smooth over the back of each ear, rather than sliding per-point across it.
    scalp=[]
    for side in [-1,1]:
        for group in range(86):
            theta0=rng.uniform(.015,1.56)
            phi0=rng.uniform(-1.34,1.59)
            # Frontal roots stay above the eyebrows; the separate forelocks take over.
            if phi0<-.60:theta0=min(theta0,.95)
            controls=[]
            for j in range(7):
                t=j/6
                theta=theta0+(1.77-theta0)*t
                phi=phi0+(.61-phi0)*(t*t*(3-2*t))
                controls.append(sphere(theta,phi,side,.0024+.0012*math.sin(math.pi*t)))
            controls.extend([point((side*.091,.020,1.414)),point((side*.090,-.004,1.395))])
            guide=safe_guide(controls,62)
            scalp.extend(_strand_bundle(guide,34,.0030,.0008,rng,center=center,tip_taper=False))

    # Dedicated pole coverage has no minimum-angle gap; the dark scalp substrate
    # is only a root-density backup beneath this layer, never a visible hair cap.
    crown=[]
    for side in [-1,1]:
        for group in range(18):
            phi0=rng.uniform(-1.50,1.58)
            controls=[sphere(.001,phi0,side,.0026)]
            for j in range(1,8):
                t=j/7
                controls.append(sphere(.001+1.76*t,phi0+(.61-phi0)*t*t,side,.0028))
            controls.extend([point((side*.091,.020,1.414)),point((side*.090,-.004,1.395))])
            crown.extend(_strand_bundle(safe_guide(controls,62),20,.0014,.0007,rng,center=center,tip_taper=False))

    # Seven coherent asymmetric bangs. Only the three sparse scalp controls are
    # surface-projected; the final locks are continuous free-curving guides.
    specs=[
        (-.031,-.067,1.438,.0078),(-.023,-.043,1.464,.0092),
        (-.012,-.022,1.456,.0088),(-.001,.004,1.461,.0102),
        (.011,.027,1.468,.0090),(.024,.047,1.459,.0083),
        (.034,.066,1.433,.0068)]
    bangs=[]
    for rx,tx,tz,width in specs:
        root=keypoint((rx*.55,-.010,1.552),.0028)
        shoulder=keypoint((rx,-.050,1.538),.005)
        forehead=keypoint((rx+(tx-rx)*.42,-.089,1.505),.007)
        tipy=-.102+.026*(abs(tx)/.067)**1.6
        controls=[root,shoulder,forehead,
                  point((tx-.004,-.104+.023*(abs(tx)/.067),max(tz+.015,1.477))),
                  point((tx,tipy,tz))]
        guide=safe_guide(controls,52)
        bangs.extend(_strand_bundle(guide,480,width,.0012,rng,center=center,tip_taper=True))

    # Temple pieces have a distributed root blended into the scalp and a loose
    # forward arc with separate wispy tips. They no longer begin as vertical sticks.
    wisps=[]
    for side in [-1,1]:
        for lock in range(3):
            off=(lock-1)*.0032
            controls=[keypoint((side*.041,-.030,1.536),.003),
                      keypoint((side*.064,-.057,1.502),.0045),
                      point((side*(.077+off),-.073,1.461)),
                      point((side*(.076+off),-.087,1.409)),
                      point((side*(.064+off),-.081,1.365)),
                      point((side*(.053+off),-.068,1.341+lock*.007))]
            guide=safe_guide(controls,60)
            wisps.extend(_strand_bundle(guide,105,.0016,.00065,rng,center=center,tip_taper=True))
    # Build scene objects only after every guide has passed continuity checks.
    _native('R03 Scalp gather guides',scalp,material,.000065,created)
    _native('R03 Crown soft coverage',crown,material,.000060,created)
    _native('R03 Forelocks from crown',bangs,material,.000066,created)
    _native('R03 curved cheek wisps',wisps,material,.000055,created)
    _underlay(vertices,polygons,created,center,head_shift_z)

    old_hidden=[]
    for obj in bpy.context.scene.objects:
        if obj in created or obj.hide_render:continue
        name=obj.name.lower()
        # Tail assets and expressive facial curves remain exactly as supplied.
        if any(s in name for s in ['tail','tie','brow','lash']):continue
        if obj.name.startswith(('R01 Groom','R02 Surface','R02 Fringe','R02 Temple','R02 Curved temple','R02 Strands /')):
            if any(s in name for s in ['scalp','crown','fringe','bang','temple','nape','coverage']):
                obj.hide_render=True;obj.hide_set(True);old_hidden.append(obj.name)
    for obj in created:
        obj['R03_groom_pass']=3
        obj['R03_groom_head_shift_z']=head_shift_z
        obj.hide_render=False;obj.hide_set(False)
    bpy.context.view_layer.update()
    report={'new_objects':[o.name for o in created],'hidden_existing':old_hidden,
            'new_strands':len(scalp)+len(crown)+len(bangs)+len(wisps),
            'head_shift_z':head_shift_z,
            'inferred_crown_shift_z':inferred_shift,
            'preserved':'existing twin tails, ties, brows, lashes, head geometry and expressions',
            'attachment':'new R03 objects are world-space static geometry; integrator must attach to head controls',
            'validation':'syntax only before execution; no scene render or dynamics claim'}
    print(report)
    return report


if __name__=='__main__':run()
