"""Local fairing of EXISTING E4 hair mesh rings; no cap/head rebuild.

No new vertices, faces, UVs, pieces or modifiers.  This is a surface polish,
not claimed topology reconstruction.  Source is left untouched by CLI mode.
"""
import math, hashlib, struct, json, sys
from pathlib import Path


def _geometry_hash(obj):
    h=hashlib.sha256()
    for v in obj.data.vertices:h.update(struct.pack('<3d',*v.co))
    for p in obj.data.polygons:
        for index in p.vertices:h.update(struct.pack('<I',index))
    return h.hexdigest()


def _uv_hash(obj):
    h=hashlib.sha256()
    for layer in obj.data.uv_layers:
        h.update(layer.name.encode())
        for uv in layer.data:h.update(struct.pack('<2d',*uv.uv))
    return h.hexdigest()


def _smoothstep(t):
    t=max(0,min(1,t));return t*t*(3-2*t)


def apply_hair_continuity_delta():
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    cap=bpy.data.objects['Hair_Crown_SourceEnvelope']
    head=bpy.data.objects['Head_Skin_Quad_Retopo']
    protected={o.name:{'geometry':_geometry_hash(o),'uv':_uv_hash(o)} for o in (cap,head)}
    # Only the actual E4 cap and head are queried. No earlier analytical head
    # is substituted for the scalp-fit result in the saved production file.
    deps=bpy.context.evaluated_depsgraph_get()
    head_bvh=BVHTree.FromObject(head,deps)
    cap_bvh=BVHTree.FromObject(cap,deps)
    targets=[o for o in bpy.context.scene.objects if o.type=='MESH' and
             (o.name.startswith('Fringe_') or o.name.startswith('Temple_'))]
    if len(targets)!=8:raise RuntimeError(f'Expected the eight E4 fringe/temple meshes; got {len(targets)}')
    records=[]
    for obj in targets:
        if len(obj.data.vertices)%12:raise RuntimeError('Unexpected ring topology: '+obj.name)
        old=[v.co.copy() for v in obj.data.vertices];nr=len(old)//12
        topology=(len(obj.data.vertices),len(obj.data.polygons),len(obj.data.loops))
        uv_before=_uv_hash(obj)
        # Ring correspondence is inherent in the existing mesh. Fair only
        # along length, keeping the cross-section count and ring UVs intact.
        work=[v.copy() for v in old]
        for iteration in range(7 if obj.name.startswith('Fringe_') else 3):
            nxt=[v.copy() for v in work]
            for i in range(2,nr-4):
                t=i/(nr-1)
                mask=_smoothstep((t-.025)/.12)*_smoothstep((.91-t)/.14)
                amount=.60*mask
                for j in range(12):
                    k=i*12+j
                    nxt[k]=work[k]+amount*(.5*(work[k-12]+work[k+12])-work[k])
            work=nxt
        # Roots previously sat as tiny raised caps. Translate whole root
        # rings slightly into the actual cap, never project individual ring
        # vertices to a surface (the rejected v3 operation).
        root_moves=[]
        if obj.name.startswith('Fringe_'):
            for i in range(min(8,nr)):
                weight=1-_smoothstep(i/7)
                if weight<=0:continue
                ids=range(i*12,(i+1)*12)
                center=sum((work[k] for k in ids),Vector())/12
                hit,normal,idx,distance=cap_bvh.find_nearest(center)
                if hit is None:continue
                if normal.dot(center-Vector((0,.02,.855)))<0:normal=-normal
                outer=max((work[k]-hit).dot(normal) for k in ids)
                shift=min(.0028,max(0,outer+.00045))*weight
                if shift>0:
                    for k in ids:work[k]-=normal*shift
                    root_moves.append(shift)
        side_moves=[]
        if obj.name.startswith('Temple_'):
            for i in range(3,nr-5):
                t=i/(nr-1)
                weight=_smoothstep((t-.20)/.16)*_smoothstep((.80-t)/.16)
                if weight<=0:continue
                ids=list(range(i*12,(i+1)*12));center=sum((work[k] for k in ids),Vector())/12
                hit,normal,idx,distance=head_bvh.find_nearest(center)
                if hit is None:continue
                # Only reduce a POSITIVE surface gap. Existing contact is
                # never pushed into the head just to diminish a cast shadow.
                gaps=[]
                for k in ids:
                    p,n,ix,d=head_bvh.find_nearest(work[k])
                    if p is not None:gaps.append((work[k]-p).dot(n))
                if not gaps or min(gaps)<=.003:continue
                shift=min(.002,max(0,min(gaps)-.0028))*weight
                for k in ids:work[k]-=normal*shift
                side_moves.append(shift)
        max_move=max((a-b).length for a,b in zip(work,old))
        if max_move>.0055:raise RuntimeError(f'Unexpected large polish delta for {obj.name}: {max_move}')
        # Assign only existing coordinates. Mesh indices/UV/materials survive.
        for v,p in zip(obj.data.vertices,work):v.co=p
        for p in obj.data.polygons:
            if len(p.vertices)==4:p.use_smooth=True
        obj.data.update()
        if topology!=(len(obj.data.vertices),len(obj.data.polygons),len(obj.data.loops)) or _uv_hash(obj)!=uv_before:
            raise RuntimeError('Polish changed topology or UVs: '+obj.name)
        obj['continuity_polish_v5']='Existing longitudinal ring fairing; original tips and UV preserved; no new geometry'
        records.append({'object':obj.name,'vertices':len(work),'movedVertices':sum((a-b).length>1e-9 for a,b in zip(work,old)),
          'maxDisplacement':max_move,'maxRootEmbed':max(root_moves,default=0),'maxSideGapReduction':max(side_moves,default=0),
          'tipLastFourRingsExact':all(a==b for a,b in zip(work[-48:],old[-48:])),
          'uvExact':True,'topologyExact':True,'geometryBefore':hashlib.sha256(b''.join(struct.pack('<3d',*p) for p in old)).hexdigest(),
          'geometryAfter':_geometry_hash(obj)})
    after={o.name:{'geometry':_geometry_hash(o),'uv':_uv_hash(o)} for o in (cap,head)}
    if after!=protected:raise RuntimeError('Protected cap or fitted head changed')
    return {'scope':'local coordinate polish only; no visual acceptance until same-camera render',
            'capAndFittedHeadExact':True,'protectedHashes':protected,'records':records,
            'newVertices':0,'newFaces':0,'newObjects':0,'newModifiers':0}


if __name__=='__main__':
    import bpy
    ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
    from pipeline.context import workspace_path
    source=workspace_path('Character_Rebuild/stages/E4-final-refined.blend')
    out=workspace_path('Character_Rebuild/agents/hair/E5-hair-continuity-candidate.blend')
    reportpath=workspace_path('Character_Rebuild/reports/hair-continuity-delta-v5.json')
    if out.exists():raise FileExistsError('Preserve existing candidate: '+str(out))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source))
    report=apply_hair_continuity_delta()
    out.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Source changed')
    report.update(source=str(source),sourceSha256=source_hash,sourceUnchanged=True,candidate=str(out),candidateSha256=hashlib.sha256(out.read_bytes()).hexdigest())
    reportpath.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))
