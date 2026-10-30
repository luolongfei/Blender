"""Editable, exportable stylised hair locks for the Hunyuan model rebuild.

This module neither opens nor saves files and never deletes source geometry.
All placements are supplied in the CURRENT Blender world coordinates.  Root
integrator must fit them to the new source and inspect front, sides and back.
"""

from math import sin, cos, pi, sqrt


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _mul(a, b):
    return tuple(x * b for x in a)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _unit(a):
    d = sqrt(_dot(a, a))
    if d < 1e-12:
        raise ValueError("Degenerate hair tangent or normal")
    return _mul(a, 1.0 / d)


def _lerp(a, b, t):
    return _add(_mul(a, 1 - t), _mul(b, t))


def _profile(values, t):
    """Monotone linear profile: never overshoots into negative strand radii."""
    q = min(max(t, 0), 1) * (len(values) - 1)
    k = min(int(q), len(values) - 2)
    return values[k] * (1 - q + k) + values[k + 1] * (q - k)


def sample_path(points, count=33):
    """Centripetal-ish chord parameter with safe Hermite tangents.

    The path is in physical space.  Endpoint positions are exact.  Knot
    tangents are limited so short control spans cannot make looping spikes.
    """
    if len(points) < 2 or count < 3:
        raise ValueError("A hair lock needs >=2 control points and >=3 rings")
    pts = [tuple(map(float, p)) for p in points]
    distances = [sqrt(_dot(_sub(b, a), _sub(b, a))) for a, b in zip(pts, pts[1:])]
    if min(distances) < 1e-8:
        raise ValueError("Consecutive hair control points coincide")
    knots = [0.0]
    for d in distances:
        knots.append(knots[-1] + d)
    total = knots[-1]
    tangents = []
    for k in range(len(pts)):
        if k == 0:
            tangents.append(_unit(_sub(pts[1], pts[0])))
        elif k == len(pts) - 1:
            tangents.append(_unit(_sub(pts[-1], pts[-2])))
        else:
            tangents.append(_mul(_add(_unit(_sub(pts[k], pts[k - 1])),
                                      _unit(_sub(pts[k + 1], pts[k]))), 0.5))
    result = []
    segment = 0
    for j in range(count):
        d = total * j / (count - 1)
        while segment < len(distances) - 1 and d > knots[segment + 1]:
            segment += 1
        t = (d - knots[segment]) / distances[segment]
        h00, h10 = 2*t*t*t - 3*t*t + 1, t*t*t - 2*t*t + t
        h01, h11 = -2*t*t*t + 3*t*t, t*t*t - t*t
        result.append(_add(_add(_mul(pts[segment], h00), _mul(pts[segment+1], h01)),
                           _add(_mul(tangents[segment], h10 * distances[segment]),
                                _mul(tangents[segment+1], h11 * distances[segment]))))
    return result


def lock_geometry(points, widths, depths, outward=(0, -1, 0), rings=33,
                  sides=12, ridge=0.04):
    """A closed, quad-sided volumetric leaf, with nonzero capped tip.

    Width/depth values are half-extents in model units.  Radii remain positive;
    tip convergence is controlled by the caller, not hidden zero-area faces.
    Returns vertices/faces/loop UVs and never changes Blender scene state.
    """
    if len(widths) < 2 or len(depths) < 2 or min(list(widths) + list(depths)) <= 0:
        raise ValueError("Hair widths and depths must be positive profiles")
    path = sample_path(points, rings)
    verts, faces, face_uvs = [], [], []
    old_width = None
    for i, p in enumerate(path):
        tangent = _unit(_sub(path[min(i+1, rings-1)], path[max(i-1, 0)]))
        guide = outward
        if abs(_dot(_unit(guide), tangent)) > 0.96:
            guide = (0, 0, 1) if abs(tangent[2]) < 0.9 else (0, 1, 0)
        width_axis = _unit(_cross(guide, tangent))
        if old_width is not None and _dot(width_axis, old_width) < 0:
            width_axis = _mul(width_axis, -1)
        depth_axis = _unit(_cross(tangent, width_axis))
        old_width = width_axis
        t = i / (rings - 1)
        w, d = _profile(widths, t), _profile(depths, t)
        for j in range(sides):
            a = 2*pi*j/sides
            # Broad flattened section, subtle planar central ridge only.
            radial_depth = sin(a) * (1 + ridge * cos(4*a))
            verts.append(_add(p, _add(_mul(width_axis, w*cos(a)),
                                      _mul(depth_axis, d*radial_depth))))
    for i in range(rings-1):
        for j in range(sides):
            k = (j+1) % sides
            faces.append((i*sides+j, i*sides+k, (i+1)*sides+k, (i+1)*sides+j))
            face_uvs.append(((j/sides, i/(rings-1)), ((j+1)/sides, i/(rings-1)),
                             ((j+1)/sides, (i+1)/(rings-1)), (j/sides, (i+1)/(rings-1))))
    faces.append(tuple(reversed(range(sides))))
    face_uvs.append(tuple((0.5 + 0.45*cos(2*pi*j/sides),
                          0.5 + 0.45*sin(2*pi*j/sides)) for j in reversed(range(sides))))
    faces.append(tuple((rings-1)*sides+j for j in range(sides)))
    face_uvs.append(tuple((0.5 + 0.45*cos(2*pi*j/sides),
                          0.5 + 0.45*sin(2*pi*j/sides)) for j in range(sides)))
    return verts, faces, face_uvs


def _mesh_object(name, geometry, collection, material):
    import bpy
    verts, faces, face_uvs = geometry
    mesh = bpy.data.meshes.new(name + "_EditableMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name="HairUV")
    for poly, values in zip(mesh.polygons, face_uvs):
        for loop, value in zip(poly.loop_indices, values):
            uv.data[loop].uv = value
        poly.use_smooth = len(poly.vertices) == 4
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(material)
    obj["rebuild_role"] = "stylised mesh hair; editable control-ring topology"
    return obj


def material(name="Hair_DeepWarmBlack", color=(0.027, 0.024, 0.021, 1)):
    """Simple glTF-compatible material, no renderer-only hair shader."""
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.48
    bsdf.inputs["Metallic"].default_value = 0
    mat.diffuse_color = color
    return mat


def build_locks(lock_specs, collection=None, hair_material=None, cap_sampler=None):
    """Build caller-fitted locks.  Does not replace or hide original hair."""
    import bpy
    if collection is None:
        collection = bpy.data.collections.new("Rebuilt_Hair_Editable")
        bpy.context.scene.collection.children.link(collection)
    if hair_material is None:
        hair_material = material()
    result = []
    for spec in lock_specs:
        geometry = lock_geometry(spec["points"], spec["widths"], spec["depths"],
                                 spec.get("outward", (0, -1, 0)), spec.get("rings", 33),
                                 spec.get("sides", 12), spec.get("ridge", 0.04))
        if spec.get('conformToCap') and cap_sampler is not None:
            geometry=conform_geometry_to_cap(geometry,spec,cap_sampler)
        obj = _mesh_object(spec["name"], geometry, collection, hair_material)
        obj["control_path"] = [v for p in spec["points"] for v in p]
        obj["profile_widths"] = spec["widths"]
        obj["profile_depths"] = spec["depths"]
        obj["source_basis"] = "new Hunyuan source and original anime references"
        result.append(obj)
    return result


class CapSampler:
    """Exact ray/triangle interpolation on outer cap, indexed in the XZ plane."""
    def __init__(self,cap):
        s=cap['surface'];self.triangles=[];self.cells={};self.step=.016
        for i in range(len(s)-1):
            for j in range(len(s[0])):
                k=(j+1)%len(s[0]);a,b,c,d=s[i][j],s[i+1][j],s[i+1][k],s[i][k]
                for tri in ((a,b,c),(a,c,d)):
                    idx=len(self.triangles);self.triangles.append(tri)
                    xmin,xmax=min(p[0] for p in tri),max(p[0] for p in tri)
                    zmin,zmax=min(p[2] for p in tri),max(p[2] for p in tri)
                    for ix in range(int(xmin//self.step),int(xmax//self.step)+1):
                        for iz in range(int(zmin//self.step),int(zmax//self.step)+1):
                            self.cells.setdefault((ix,iz),[]).append(idx)

    def sample(self,x,z,side='back'):
        values=[]
        for idx in self.cells.get((int(x//self.step),int(z//self.step)),[]):
            a,b,c=self.triangles[idx]
            det=(b[2]-c[2])*(a[0]-c[0])+(c[0]-b[0])*(a[2]-c[2])
            if abs(det)<1e-14:continue
            u=((b[2]-c[2])*(x-c[0])+(c[0]-b[0])*(z-c[2]))/det
            v=((c[2]-a[2])*(x-c[0])+(a[0]-c[0])*(z-c[2]))/det
            w=1-u-v
            if min(u,v,w)>=-1e-7:values.append(u*a[1]+v*b[1]+w*c[1])
        if not values:return None
        return max(values) if side=='back' else min(values)


def conform_geometry_to_cap(geometry,spec,sampler):
    """Embed broad rear locks in the cap; avoid unsupported floating strips."""
    verts,faces,uvs=geometry;new=[]
    rings,sides=spec.get('rings',33),spec.get('sides',12)
    side=spec.get('capSide','back');sign=1 if side=='back' else -1
    scale=spec.get('capDepthScale',.45);embed=spec.get('capEmbed',-.00025)
    for n,p in enumerate(verts):
        i,j=divmod(n,sides);t=i/(rings-1)
        y=sampler.sample(p[0],p[2],side)
        if y is None:new.append(p);continue
        depth=_profile(spec['depths'],t)*scale
        new.append((p[0],y+sign*(embed+depth*(1+sin(2*pi*j/sides))),p[2]))
    return new,faces,uvs


def tail_bundle_specs(name, spine, widths, depths, lock_count=5, front=(0, -1, 0)):
    """Partition a caller-fitted ponytail envelope into modest broad locks.

    This does not guess ear, shoulder or head positions.  The spine and
    half-extents must be measured against the source/reference.  The outer
    envelope is retained by distributing small ribbon centres within it;
    staggered inward tips separate the silhouette without splaying the hair.
    """
    if len(spine) < 4:
        raise ValueError("Ponytail spine needs tie, upper, lower and tip controls")
    result = []
    n = len(spine)
    guide = _unit(front)
    for k in range(lock_count):
        lateral = (k / max(lock_count - 1, 1) - 0.5) * 1.25
        phase = (k - (lock_count-1)/2) / max(lock_count, 1)
        pts, ws, ds = [], [], []
        for i, p in enumerate(spine):
            t = i / (n - 1)
            tangent = _unit(_sub(spine[min(i+1, n-1)], spine[max(i-1, 0)]))
            axis = _unit(_cross(guide, tangent))
            w, d = _profile(widths, t), _profile(depths, t)
            # Locks converge at the tie. Their tips remain within the same
            # silhouette and curl subtly inward instead of flaring outward.
            spread = sin(pi * t) ** 0.6
            offset = _add(_mul(axis, lateral * w * spread),
                          _mul(guide, d * (0.30 + 0.10 * cos(k*pi)) * spread))
            if i == n-1:
                offset = _add(offset, _mul(tangent, phase * widths[0] * 0.85))
                offset = _add(offset, _mul(axis, -lateral * widths[0] * 0.1))
            pts.append(_add(p, offset))
            ws.append(max(w * (0.48 if lock_count > 3 else 0.65), widths[0]*0.025))
            ds.append(max(d * 0.60, widths[0]*0.018))
        result.append(dict(name=f"{name}_Lock_{k+1:02d}", points=pts,
                           widths=ws, depths=ds, outward=front, rings=41, sides=12))
    return result


def tie_geometry(center, tangent, width_radius, depth_radius, thickness,
                 around=48, section=10):
    """Closed elliptical fabric band perpendicular to measured tail axis."""
    tangent = _unit(tangent)
    guide = (0, -1, 0) if abs(tangent[1]) < 0.95 else (1, 0, 0)
    width_axis = _unit(_cross(guide, tangent))
    depth_axis = _unit(_cross(tangent, width_axis))
    verts, faces, uv = [], [], []
    for i in range(around):
        a = 2*pi*i/around
        radial = _unit(_add(_mul(width_axis, cos(a)), _mul(depth_axis, sin(a))))
        ring_center = _add(center, _add(_mul(width_axis, width_radius*cos(a)),
                                       _mul(depth_axis, depth_radius*sin(a))))
        for j in range(section):
            b = 2*pi*j/section
            verts.append(_add(ring_center, _add(_mul(radial, thickness*cos(b)),
                                                _mul(tangent, thickness*1.25*sin(b)))))
    for i in range(around):
        for j in range(section):
            faces.append((i*section+j, ((i+1)%around)*section+j,
                          ((i+1)%around)*section+(j+1)%section,
                          i*section+(j+1)%section))
            uv.append(((i/around,j/section),((i+1)/around,j/section),
                       ((i+1)/around,(j+1)/section),(i/around,(j+1)/section)))
    return verts, faces, uv


def cap_geometry(cap):
    """Build a closed thin shell from the measured source-envelope grid."""
    surface=cap['surface']; rows=len(surface); around=len(surface[0])
    if any(len(row)!=around for row in surface):
        raise ValueError('Cap rows have different sampling counts')
    center=tuple(cap['center']); thickness=cap.get('thickness',.003)
    outer=[tuple(p) for row in surface for p in row]
    inner=[_sub(p,_mul(_unit(_sub(p,center)),thickness)) for p in outer]
    verts=outer+inner
    count=len(outer); faces=[]; uvs=[]
    for layer in range(2):
        offset=layer*count
        for i in range(rows-1):
            for j in range(around):
                k=(j+1)%around
                f=(offset+i*around+j,offset+(i+1)*around+j,
                   offset+(i+1)*around+k,offset+i*around+k)
                uv=((j/around,i/(rows-1)),(j/around,(i+1)/(rows-1)),
                    ((j+1)/around,(i+1)/(rows-1)),((j+1)/around,i/(rows-1)))
                if layer:f=tuple(reversed(f));uv=tuple(reversed(uv))
                faces.append(f);uvs.append(uv)
        pole=tuple(sum(verts[offset+j][k] for j in range(around))/around for k in range(3))
        pole=tuple(v+(0.0001 if layer==0 else -0.0001) if k==2 else v for k,v in enumerate(pole))
        poleid=len(verts);verts.append(pole)
        for j in range(around):
            k=(j+1)%around
            f=(poleid,offset+j,offset+k);uv=((.5,0),(j/around,.001),((j+1)/around,.001))
            if layer:f=tuple(reversed(f));uv=tuple(reversed(uv))
            faces.append(f);uvs.append(uv)
    lower=(rows-1)*around
    for j in range(around):
        k=(j+1)%around
        faces.append((lower+j,count+lower+j,count+lower+k,lower+k))
        uvs.append(((j/around,0),(j/around,1),((j+1)/around,1),((j+1)/around,0)))
    return verts,faces,uvs


def build_from_spec(spec, collection=None, hair_material=None, tie_material=None):
    """Install measured candidate in caller's scene, without changing source.

    The caller owns replacement of fused source geometry and verification of
    head, shoulder and garment clearances.  No file is saved here.
    """
    import bpy
    if collection is None:
        collection=bpy.data.collections.new('Rebuilt_Hair_Editable')
        bpy.context.scene.collection.children.link(collection)
    hair_material=hair_material or material()
    tie_material=tie_material or material('Hair_Ties_Fabric',(.018,.020,.019,1))
    result=[_mesh_object('Hair_Crown_SourceEnvelope',cap_geometry(spec['cap']),collection,hair_material)]
    result[0]['source_sha256']=spec.get('sourceSha256','')
    result[0]['construction']='radial sampling of new source hair; local regularisation; closed thickness shell'
    locks=list(spec.get('locks',[]))
    for tail in spec.get('tails',[]):
        if tail.get('locks'):
            locks.extend(tail['locks'])
        else:
            locks.extend(tail_bundle_specs(tail['name'],tail['spine'],tail['widths'],tail['depths'],
                                           tail.get('lock_count',5),tail.get('front',(0,-1,0))))
        tangent=_sub(tail['spine'][1],tail['spine'][0])
        result.append(_mesh_object(tail['name']+'_FabricTie',
            tie_geometry(tail['spine'][0],tangent,tail['widths'][0]*1.02,
                         tail['depths'][0]*1.02,.0038),collection,tie_material))
    result.extend(build_locks(locks,collection,hair_material,CapSampler(spec['cap'])))
    return result
