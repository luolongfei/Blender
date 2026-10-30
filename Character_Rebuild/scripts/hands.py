"""Editable stylized hands for the new Hunyuan input (not the old R11 model).

The caller supplies measured wrist frames/dimensions. This module never opens,
saves, cuts, or exports a scene. Five finger branches are part of one connected
quad-dominant palm surface. It is deliberately independent of old character data.

Local axes: U = wrist toward middle finger, V = little toward index/thumb,
N = palm toward hand dorsum. A mirrored world frame is supported.
"""
from math import cos, sin, pi, sqrt, exp


DEFAULT_DIMENSIONS = {
    "palm_length": 0.064,
    "palm_width": 0.058,
    "palm_depth": 0.019,
    "wrist_width": 0.036,
    "wrist_depth": 0.021,
    # Ordered little, ring, middle, index; dimensions must be calibrated to input.
    "finger_lengths": (0.045, 0.061, 0.066, 0.058),
    "finger_widths": (0.0102, 0.0120, 0.0125, 0.0117),
    "finger_depths": (0.0100, 0.0115, 0.0120, 0.0113),
    "finger_splay": (-0.025, -0.010, 0.006, 0.024),
    "thumb_length": 0.043,
    "thumb_width": 0.0132,
    "thumb_depth": 0.0120,
    "thumb_forward": 0.68,
    "thumb_outward": 0.69,
    "curl": 0.0012,
    "finger_root_offsets": (0.0,0.0,0.0,0.0),
    "thenar_pad": 0.0,
    "hypothenar_pad": 0.0,
    "palm_cup": 0.0,
    "dorsal_knuckle": 0.0,
    "finger_arch": 0.0,
    "fuller_finger_profile": False,
    "thumb_base_flare": 0.0,
    "thumb_root_clearance": 0.0,
}


def hand_mesh_data(dimensions=None):
    """Return local vertices, polygons, UVs, named regions and audit data.

    No Blender import is needed, so connectivity can be checked independently.
    Fingertip closing fans are the only intentional triangles. Wrist is capped
    below the sleeve; it can be removed by the integrator for a skin connection.
    """
    d = dict(DEFAULT_DIMENSIONS)
    if dimensions:
        d.update(dimensions)
    for k in ("palm_length", "palm_width", "palm_depth", "wrist_width", "wrist_depth"):
        if d[k] <= 0:
            raise ValueError("Positive hand dimension required: " + k)
    vertices, faces, uv_faces, regions = [], [], [], {}
    lattice = {}
    us = (0.0, 0.10, 0.26, 0.43, 0.61, 0.80, 1.0)
    nu, nv, nn = len(us)-1, 11, 4
    L, W, D = d["palm_length"], d["palm_width"], d["palm_depth"]

    def root_offset(j):
        # Each socket stays planar. The single skin-web column between
        # sockets interpolates the offset; palm-wise support is smooth below.
        return d["finger_root_offsets"][min(3,j//3)]

    def gauss(value, center, width):
        return exp(-((value-center)/width)**2)

    def vertex(co, region=None):
        i = len(vertices)
        vertices.append(tuple(float(c) for c in co))
        if region:
            regions.setdefault(region, []).append(i)
        return i

    def face(ids, uv):
        if len(set(ids)) != len(ids):
            raise ValueError("Repeated index in hand polygon")
        faces.append(tuple(ids))
        uv_faces.append(tuple(uv))

    def lattice_vertex(i, j, k):
        key = i, j, k
        if key in lattice:
            return lattice[key]
        u = us[i]
        s, t = 2*j/nv-1, 2*k/nn-1
        # Wrist-to-metacarpal expansion and rounded rectangular cross-section.
        width = d["wrist_width"] + (W-d["wrist_width"])*sin(u*pi/2)**0.72
        depth = d["wrist_depth"] + (D-d["wrist_depth"])*u
        v = width*0.5*s*sqrt(max(0, 1-0.20*t*t))
        n = depth*0.5*t*sqrt(max(0, 1-0.30*s*s))
        # Conservative pad volume; no photoreal skin grooves/noise.
        n -= 0.0013*sin(pi*u)*max(0, 1-s*s)*(1-t)*0.5
        # Optional v2 macrostructure: palm pads and a gentle dorsal knuckle arch.
        # The support is zero at the wrist, preserving the already fitted cuff.
        wrist_support=sin(pi*u/2)**1.5
        palmar=max(0,-t)**1.5
        n -= wrist_support*palmar*(
            d["thenar_pad"]*gauss(u,.43,.27)*gauss(s,.48,.42)
            +d["hypothenar_pad"]*gauss(u,.46,.34)*gauss(s,-.70,.38))
        n += wrist_support*palmar*d["palm_cup"]*gauss(u,.58,.25)*gauss(s,-.04,.36)
        n += max(0,t)**1.5*d["dorsal_knuckle"]*gauss(u,.92,.20)*sum(
            gauss(j,c,.92) for c in (1,4,7,10))
        f=max(0,min(1,(u-.55)/.45)); f=f*f*(3-2*f)
        co = (L*u+f*root_offset(j), v, n)
        lattice[key] = vertex(co, "palm")
        if i == 0:
            regions.setdefault("wrist", []).append(lattice[key])
        return lattice[key]

    # Palm dorsum / palm side. Sharing lattice indices closes every patch seam.
    for k in (0, nn):
        for i in range(nu):
            for j in range(nv):
                ids = [lattice_vertex(i,j,k), lattice_vertex(i+1,j,k),
                       lattice_vertex(i+1,j+1,k), lattice_vertex(i,j+1,k)]
                uv = [(0.02+us[a]*0.44, 0.02+(b/nv)*0.21+(0.25 if k else 0))
                      for a,b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))]
                face(ids,uv)
    # Side walls. The positive V wall contains a true thumb socket.
    thumb_i0, thumb_i1 = 2, 4
    for j in (0,nv):
        for i in range(nu):
            for k in range(nn):
                if j == nv and thumb_i0 <= i < thumb_i1 and 1 <= k < 3:
                    continue
                ids = [lattice_vertex(i,j,k),lattice_vertex(i+1,j,k),
                       lattice_vertex(i+1,j,k+1),lattice_vertex(i,j,k+1)]
                uv = [(0.50+us[a]*0.46,0.02+(b/nn)*0.10+(0.13 if j else 0))
                      for a,b in ((i,k),(i+1,k),(i+1,k+1),(i,k+1))]
                face(ids,uv)
    # Wrist cap, independent of the reasonable open clothing cuff.
    for j in range(nv):
        for k in range(nn):
            face([lattice_vertex(0,j,k),lattice_vertex(0,j+1,k),
                  lattice_vertex(0,j+1,k+1),lattice_vertex(0,j,k+1)],
                 [(0.50+j/nv*.21,0.28+k/nn*.12),
                  (0.50+(j+1)/nv*.21,0.28+k/nn*.12),
                  (0.50+(j+1)/nv*.21,0.28+(k+1)/nn*.12),
                  (0.50+j/nv*.21,0.28+(k+1)/nn*.12)])
    finger_j0 = (0,3,6,9)
    # Distal web grid: separate openings leave skin webs BETWEEN fingers.
    # Outer openings need one side support column. Use sockets based on
    # eleven columns: each hole spans two, with a full separating column.
    for j in range(nv):
        for k in range(nn):
            hole = any(a <= j < a+2 for a in finger_j0) and 1 <= k < 3
            if hole:
                continue
            face([lattice_vertex(nu,j,k),lattice_vertex(nu,j+1,k),
                  lattice_vertex(nu,j+1,k+1),lattice_vertex(nu,j,k+1)],
                 [(0.74+j/nv*.22,0.28+k/nn*.12),
                  (0.74+(j+1)/nv*.22,0.28+k/nn*.12),
                  (0.74+(j+1)/nv*.22,0.28+(k+1)/nn*.12),
                  (0.74+j/nv*.22,0.28+(k+1)/nn*.12)])

    profile = ((0.05,0.94),(0.14,1.0),(0.25,1.02),(0.31,0.96),
               (0.47,0.87),(0.51,0.93),(0.57,0.87),
               (0.72,0.78),(0.77,0.83),(0.83,0.75),
               (0.93,0.61),(0.98,0.31))
    if d["fuller_finger_profile"]:
        profile=((.05,.99),(.14,1.02),(.25,1.025),(.31,1.0),
                 (.47,.93),(.51,.96),(.57,.93),(.72,.88),(.77,.91),
                 (.83,.85),(.93,.67),(.98,.35))

    def sweep(root, center_path, side_vector, widths, name, atlas_column):
        prev = root
        region = regions.setdefault(name, [])
        region.extend(root)
        for t, radius in profile:
            if name == "thumb":
                radius += d["thumb_base_flare"]*max(0,1-t/.30)**2
            center = center_path(t)
            side = side_vector(t) if callable(side_vector) else side_vector
            ring = []
            for q in range(8):
                theta = (-3*pi/4) + q*pi/4
                # Blend the first ring to the actual square socket, avoiding
                # abrupt palm-to-finger steps while keeping topology explicit.
                delta = tuple(side[a]*cos(theta)*widths[0]*0.5*radius
                              + (sin(theta)*widths[1]*0.5*radius if a == 2 else 0)
                              for a in range(3))
                ring.append(vertex(tuple(center[a]+delta[a] for a in range(3)),name))
            tprev = 0 if prev is root else previous_t
            for q in range(8):
                q1=(q+1)%8
                face([prev[q],ring[q],ring[q1],prev[q1]],
                     [(0.02+atlas_column*.19+q/8*.17,0.52+tprev*.44),
                      (0.02+atlas_column*.19+q/8*.17,0.52+t*.44),
                      (0.02+atlas_column*.19+(q+1)/8*.17,0.52+t*.44),
                      (0.02+atlas_column*.19+(q+1)/8*.17,0.52+tprev*.44)])
            prev, previous_t = ring, t
        tip = vertex(center_path(1.0),name)
        for q in range(8):
            face([prev[q],tip,prev[(q+1)%8]],
                 [(0.02+atlas_column*.19+q/8*.17,0.52+previous_t*.44),
                  (0.02+atlas_column*.19+(q+.5)/8*.17,.96),
                  (0.02+atlas_column*.19+(q+1)/8*.17,0.52+previous_t*.44)])

    for fi, j0 in enumerate(finger_j0):
        # CCW around distal normal U, starting on palm/little edge.
        pairs = ((j0,1),(j0+1,1),(j0+2,1),(j0+2,2),
                 (j0+2,3),(j0+1,3),(j0,3),(j0,2))
        root=[lattice_vertex(nu,j,k) for j,k in pairs]
        c=tuple(sum(vertices[ix][a] for ix in root)/8 for a in range(3))
        length=d["finger_lengths"][fi]
        slope=d["finger_splay"][fi]
        def path(t,c=c,length=length,slope=slope):
            return (c[0]+length*t,c[1]+length*slope*t,
                    c[2]-d["curl"]*t*t+d["finger_arch"]*sin(pi*t))
        sweep(root,path,(0,1,0),(d["finger_widths"][fi],d["finger_depths"][fi]),
              ("little","ring","middle","index")[fi],fi)

    # Thumb socket in the positive-V palm wall; side vector points toward -U.
    # The loop winding is consistently chosen; final outward normals are
    # recalculated after optional mirroring by the Blender adapter.
    pairs = ((thumb_i1,1),(3,1),(thumb_i0,1),(thumb_i0,2),
             (thumb_i0,3),(3,3),(thumb_i1,3),(thumb_i1,2))
    root=[lattice_vertex(i,nv,k) for i,k in pairs]
    c=tuple(sum(vertices[ix][a] for ix in root)/8 for a in range(3))
    length=d["thumb_length"]
    def thumb_path(t):
        # Thenar branch begins outward, then turns gently toward the fingers.
        # Small optional root translation unfolds the first palmar sweep ring.
        # It rises over the first ring, stays through ring two, then smoothly
        # returns to zero before the middle of the thumb. Distal silhouette is
        # exactly preserved; this does not alter tolerances or omit faces.
        rise=1-exp(-t/.02)
        f=max(0,min(1,(t-.20)/.30))
        fall=1-f*f*(3-2*f)
        clear=d["thumb_root_clearance"]*rise*fall
        return (c[0]+length*d["thumb_forward"]*(0.25*t+0.75*t*t),
                c[1]+length*d["thumb_outward"]*(1.15*t-0.15*t*t)+clear,
                c[2]-0.0020*t-0.0010*t*t)
    def thumb_side(t):
        u=d["thumb_forward"]*(0.25+1.5*t)
        v=d["thumb_outward"]*(1.15-0.3*t)
        norm=sqrt(u*u+v*v)
        return (-v/norm,u/norm,0)
    sweep(root,thumb_path,thumb_side,(d["thumb_width"],d["thumb_depth"]),"thumb",4)

    audit=mesh_connectivity_audit(vertices,faces)
    if audit["boundary_edges"] or audit["nonmanifold_edges"] or audit["components"] != 1:
        raise ValueError("Hand construction is not one closed surface: " + str(audit))
    return {"vertices":vertices,"faces":faces,"uv_faces":uv_faces,
            "regions":regions,"dimensions":d,"audit":audit}


def mesh_connectivity_audit(vertices, faces):
    edges={}
    neighbours=[set() for _ in vertices]
    for f in faces:
        for a,b in zip(f,f[1:]+f[:1]):
            key=tuple(sorted((a,b)))
            edges[key]=edges.get(key,0)+1
            neighbours[a].add(b); neighbours[b].add(a)
    visited=set(); components=0
    for i in range(len(vertices)):
        if i in visited:
            continue
        components+=1; stack=[i]; visited.add(i)
        while stack:
            for n in neighbours[stack.pop()]:
                if n not in visited:
                    visited.add(n);stack.append(n)
    return {"vertices":len(vertices),"faces":len(faces),
            "quads":sum(len(f)==4 for f in faces),
            "triangles":sum(len(f)-2 for f in faces),
            "boundary_edges":sum(n==1 for n in edges.values()),
            "nonmanifold_edges":sum(n>2 for n in edges.values()),
            "components":components,
            "scope":"connectivity only; visual and self-intersection checks remain necessary"}


def build_hand(name, wrist_origin, frame, dimensions=None, collection=None,
               skin_material=None, subdivision=1):
    """Create one hand in caller's Blender scene without touching original mesh.

    frame is (U,V,N), three perpendicular unit world-space direction vectors.
    Return (object, report). The caller owns cutting the old hand, matching the
    sleeve, UV texture painting, visual checks, saving and export.
    """
    import bpy
    import bmesh
    from mathutils import Vector
    axes=[Vector(a) for a in frame]
    if any(abs(a.length-1)>1e-5 for a in axes):
        raise ValueError("Hand frame axes must be unit length")
    if any(abs(axes[a].dot(axes[b]))>1e-5 for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError("Hand frame must be orthogonal")
    data=hand_mesh_data(dimensions)
    origin=Vector(wrist_origin)
    world=[origin+sum((axes[i]*v[i] for i in range(3)),Vector()) for v in data["vertices"]]
    mesh=bpy.data.meshes.new(name+".EditableTopology")
    mesh.from_pydata(world,[],data["faces"]);mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    uv=mesh.uv_layers.new(name="HandAtlas")
    for poly,coords in zip(mesh.polygons,data["uv_faces"]):
        for li,co in zip(poly.loop_indices,coords):
            uv.data[li].uv=co
    bm=bmesh.new();bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    for p in mesh.polygons:
        p.use_smooth=True
    if skin_material:
        mesh.materials.append(skin_material)
    for region,indices in data["regions"].items():
        vg=obj.vertex_groups.new(name="Edit_"+region)
        vg.add(sorted(set(indices)),1.0,'REPLACE')
    if subdivision:
        sub=obj.modifiers.new("SurfaceFinish_EditableBase",'SUBSURF')
        sub.levels=int(subdivision);sub.render_levels=int(subdivision)
    obj["rebuild_source"]="New quad-dominant five-branch hand; no old R11 geometry"
    obj["inferred_detail"]="Finger separation, palm volume and joint loops conservatively rebuilt"
    obj["local_axes"]="U wrist-to-finger; V little-to-thumb; N palm-to-dorsum"
    report={"object":name,"wrist_origin":list(wrist_origin),"frame":[list(a) for a in frame],
            "dimensions":data["dimensions"],"base_topology":data["audit"],
            "subdivision":subdivision,"bones_added":False,
            "requires_visual_validation":True}
    return obj,report
