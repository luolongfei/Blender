"""Minimal upper-neck continuation into the actual head BVH.

Only upper-neck Z positions change. UVs, XY, winding, materials, lower neck and
all other scene objects remain untouched. No unions, remeshing or source saves.
"""
def apply(target_top=.765,anchor_z=.685):
    import bpy
    from mathutils import Vector,Matrix
    from mathutils.bvhtree import BVHTree
    head=bpy.data.objects['Head_Skin_Quad_Retopo']
    neck=bpy.data.objects['Neck_Clean_Quad_Sleeve']
    identity=Matrix.Identity(4)
    assert all(abs(o.matrix_world[i][j]-identity[i][j])<1e-7
               for o in (head,neck) for i in range(4) for j in range(4))
    old_top=max(v.co.z for v in neck.data.vertices)
    assert abs(old_top-.73)<1e-5,'Delta requires original E7 upper neck, not an already extended neck'
    bvh=BVHTree.FromObject(head,bpy.context.evaluated_depsgraph_get())
    top=[v for v in neck.data.vertices if abs(v.co.z-old_top)<1e-5]
    def signed(co):
        p,n,i,d=bvh.find_nearest(co)
        return (co-p).dot(n)
    before=[signed(v.co) for v in top]
    predicted=[signed(Vector((v.co.x,v.co.y,target_top))) for v in top]
    assert max(predicted)<-.004,'Proposed neck top not safely inside actual head'
    uvs={l.name:[tuple(d.uv) for d in l.data] for l in neck.data.uv_layers}
    old=[tuple(v.co) for v in neck.data.vertices]
    moves=[]
    for v in neck.data.vertices:
        if v.co.z<=anchor_z:continue
        t=max(0,min(1,(v.co.z-anchor_z)/(old_top-anchor_z)))
        dz=(target_top-old_top)*(t*t*(3-2*t))
        v.co.z+=dz;moves.append(dz)
    neck.data.update()
    assert uvs=={l.name:[tuple(d.uv) for d in l.data] for l in neck.data.uv_layers}
    assert all(tuple(v.co)==co for v,co in zip(neck.data.vertices,old) if co[2]<=anchor_z)
    assert all(v.co.x==co[0] and v.co.y==co[1] for v,co in zip(neck.data.vertices,old))
    after=[signed(v.co) for v in top]
    assert max(after)<-.004
    return {'object':neck.name,'oldTopZ':old_top,'newTopZ':max(v.co.z for v in top),
            'anchorZ':anchor_z,'movedVertices':len(moves),'maxZDisplacement':max(moves),
            'outsideTopVerticesBefore':sum(x>0 for x in before),
            'outsideTopVerticesAfter':sum(x>0 for x in after),
            'minTopInteriorSignedDistanceAfter':min(-x for x in after),
            'xyExact':True,'uvExact':True,'lowerNeckExact':True,
            'topologyWindingMaterialsUnchanged':True,
            'diagnosis':'Top opening was exposed behind lower head; same skin material on both parts. Extension hides that opening inside actual head.'}
