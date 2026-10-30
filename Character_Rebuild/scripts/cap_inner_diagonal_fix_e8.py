"""Fix two buried cap quads whose triangulation crossed the outer shell.

No position change or outer surface edit.  The inner quad triangulation is
matched to the corresponding evaluated outer quad.  All existing UV corners
are copied exactly.  Root must re-export/reimport and run fresh full QA.
"""
import bpy,bmesh,sys,json,hashlib,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.context import workspace_path
import geometry_checks


def outer_evidence(mesh):
    mesh.calc_loop_triangles()
    polygons=[p for p in mesh.polygons if all(v<1344 or v==2688 for v in p.vertices)]
    positions=hashlib.sha256();uv=hashlib.sha256();normals=hashlib.sha256();tri=hashlib.sha256()
    for i in list(range(1344))+[2688]:positions.update(struct.pack('<3d',*mesh.vertices[i].co))
    for p in polygons:
        for v in p.vertices:tri.update(struct.pack('<I',v))
        for li in p.loop_indices:
            normals.update(struct.pack('<3d',*mesh.corner_normals[li].vector))
            for layer in mesh.uv_layers:uv.update(struct.pack('<2d',*layer.data[li].uv))
    # Actual tangent frames on the normal-map UV need the same outer data.
    tangent=hashlib.sha256()
    mesh.calc_tangents(uvmap=mesh.uv_layers.active.name)
    for p in polygons:
        for li in p.loop_indices:
            tangent.update(struct.pack('<3d',*mesh.loops[li].tangent));tangent.update(struct.pack('<d',mesh.loops[li].bitangent_sign))
    return {'outerVertices':positions.hexdigest(),'outerTopology':tri.hexdigest(),'outerLoopUV':uv.hexdigest(),
            'outerCornerNormals':normals.hexdigest(),'outerTangents':tangent.hexdigest(),'outerPolygons':len(polygons)}


def apply_cap_inner_diagonal_fix():
    obj=bpy.data.objects['Hair_Crown_SourceEnvelope'];me=obj.data
    if len(me.vertices)!=2690 or len(me.polygons)!=2752:raise RuntimeError('Unexpected E7 cap topology')
    if me.has_custom_normals:raise RuntimeError('Unexpected custom corner normals; preserve explicitly before editing')
    before=geometry_checks.inspect_mesh(me,obj.matrix_world,self_intersections=True,max_self_triangles=20000,max_self_candidates=300000,tolerance=1e-10)
    evidence_before=outer_evidence(me)
    all_positions_before=[tuple(v.co) for v in me.vertices]
    me.calc_loop_triangles()
    pairs=[(923,2267),(1106,2450)];planned=[]
    for outer,inner in pairs:
        triangles=[tuple(t.vertices) for t in me.loop_triangles if t.polygon_index==outer]
        if len(triangles)!=2:raise RuntimeError('Expected an outer quad')
        if set(me.polygons[inner].vertices)!={v+1344 for v in me.polygons[outer].vertices}:
            raise RuntimeError('Inner shell correspondence changed')
        planned.append({'outerPolygon':outer,'innerPolygon':inner,
                        'newInnerTriangles':[tuple(v+1344 for v in reversed(t)) for t in triangles]})
    bm=bmesh.new();bm.from_mesh(me);bm.faces.ensure_lookup_table();bm.verts.ensure_lookup_table()
    uv_names=list(bm.loops.layers.uv.keys());items=[]
    for item in planned:
        face=bm.faces[item['innerPolygon']]
        uv_by_vertex={loop.vert.index:{name:tuple(loop[bm.loops.layers.uv[name]].uv) for name in uv_names} for loop in face.loops}
        items.append((face,item,uv_by_vertex,face.material_index,face.smooth))
    for face,item,uv_by_vertex,mat,smooth in items:
        bm.faces.remove(face)
        for tri in item['newInnerTriangles']:
            new=bm.faces.new([bm.verts[v] for v in tri]);new.material_index=mat;new.smooth=smooth
            for loop in new.loops:
                for name in uv_names:loop[bm.loops.layers.uv[name]].uv=uv_by_vertex[loop.vert.index][name]
    bm.normal_update();bm.to_mesh(me);bm.free();me.update()
    evidence_after=outer_evidence(me)
    if evidence_after!=evidence_before:raise RuntimeError('Outer shading or UV changed; do not save')
    if [tuple(v.co) for v in me.vertices]!=all_positions_before:raise RuntimeError('A vertex position changed')
    after=geometry_checks.inspect_mesh(me,obj.matrix_world,self_intersections=True,max_self_triangles=20000,max_self_candidates=300000,tolerance=1e-8)
    if after['selfContacts']['contacts']!=0:raise RuntimeError('Contacts remain after inner triangulation: '+json.dumps(after['selfContacts']))
    if after['triangulation']['zeroAreaTriangles'] or after['uvLayers'][0]['zeroAreaTrianglesAtUvTolerance1e-14']:
        raise RuntimeError('Degenerate geometry/UV introduced')
    return {'operation':'Two buried inner quads matched to outer diagonal; no point moved',
            'beforeSelfContacts':before['selfContacts'],'afterSelfContacts':after['selfContacts'],
            'allVertexPositionsExact':True,'outerEvidenceBefore':evidence_before,'outerEvidenceAfter':evidence_after,
            'outerPositionTopologyUVNormalsTangentsExact':True,'changes':planned,
            'polygonsBefore':2752,'polygonsAfter':len(me.polygons),'sourceTrianglesBefore':5376,'sourceTrianglesAfter':len(me.loop_triangles),
            'normalRebakeRequiredForOuterSurface':False,'freshExportValidationRequired':True}


if __name__=='__main__':
    base=workspace_path('Character_Rebuild');source=base/'stages/E7-uv-normals-corrected.blend'
    out=base/'agents/hair/E8-inner-cap-diagonals-candidate.blend'
    if out.exists():raise FileExistsError(out)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source));report=apply_cap_inner_diagonal_fix()
    out.parent.mkdir(exist_ok=True,parents=True);bpy.ops.wm.save_as_mainfile(filepath=str(out))
    report.update(source=str(source),sourceSha256=source_hash,sourceUnchanged=hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,
                  candidate=str(out),candidateSha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (base/'reports/cap-inner-diagonal-fix-e8.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))
