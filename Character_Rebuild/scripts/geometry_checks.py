"""Read-only geometry diagnostics for Character_Rebuild.

Call ``inspect_scene(objects)`` inside Blender. The default inspects author mesh
data; ``evaluated=True`` inspects the current dependency-graph result instead.
No scene, mesh, mode, modifier, image, or file is changed/saved by this module.

Self-contact checks are OPTIONAL, bounded, local, and include touching pairs.
They never certify global freedom from intersections or artistic correctness.
UV seam duplicates and intentional garment openings are observations, not errors.
"""

import math
from collections import Counter, defaultdict


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _length(a):
    return math.sqrt(_dot(a, a))


def _finite(values):
    return all(math.isfinite(float(x)) for x in values)


def triangles_contact(a, b, tolerance=1e-8):
    """Closed triangle/triangle SAT, including coplanar overlaps and touching.

    Inputs are three xyz tuples each, in the SAME coordinate system. Degenerate
    triangles are not suitable for this check and are counted separately. Return
    None for a degenerate/nonfinite input, True for contact, False if separated.
    This is a floating-point diagnostic with a declared tolerance, not exact CSG.
    """
    if not all(_finite(v) for v in (*a, *b)):
        return None
    ea = [_sub(a[(i + 1) % 3], a[i]) for i in range(3)]
    eb = [_sub(b[(i + 1) % 3], b[i]) for i in range(3)]
    na, nb = _cross(ea[0], ea[1]), _cross(eb[0], eb[1])
    if _length(na) <= tolerance * tolerance or _length(nb) <= tolerance * tolerance:
        return None
    axes = [na, nb]
    axes.extend(_cross(x, y) for x in ea for y in eb)
    # Needed for coplanar triangles, whose edge-cross-edge axes are all normal.
    axes.extend(_cross(na, x) for x in ea)
    axes.extend(_cross(nb, x) for x in eb)
    # Recenter to reduce cancellation at large world-coordinate translations.
    origin = a[0]
    ar, br = [_sub(v, origin) for v in a], [_sub(v, origin) for v in b]
    for axis in axes:
        size = _length(axis)
        if size <= 1e-24:
            continue
        axis = tuple(x / size for x in axis)
        pa, pb = [_dot(v, axis) for v in ar], [_dot(v, axis) for v in br]
        if max(pa) < min(pb) - tolerance or max(pb) < min(pa) - tolerance:
            return False
    return True


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _self_contacts(points, triangles, polygon_indices, *, enabled=False,
                   max_triangles=25000, max_candidates=200000,
                   max_samples=30, tolerance=1e-8):
    report = {
        "requested": bool(enabled), "status": "not_requested",
        "scope": "Within one object only; excludes same-polygon, shared-index, and shared-position pairs.",
        "method": "BVH broad phase, then closed triangle SAT; contact includes tangency and coplanar overlap.",
        "toleranceWorldUnits": tolerance,
        "triangleLimit": max_triangles, "candidateLimit": max_candidates,
        "completeWithinDeclaredScope": False,
        "doesNotProveNoGlobalIntersections": True,
    }
    if not enabled:
        return report
    if len(triangles) > max_triangles:
        report["status"] = "skipped_triangle_limit"
        return report
    if not all(_finite(p) for p in points):
        report["status"] = "skipped_nonfinite_positions"
        return report
    if not triangles:
        report.update(status="empty", completeWithinDeclaredScope=True,
                      contacts=0, candidates=0)
        return report
    from mathutils.bvhtree import BVHTree
    tree = BVHTree.FromPolygons(points, triangles, all_triangles=True,
                               epsilon=tolerance)
    overlap = tree.overlap(tree)
    counts = Counter()
    samples = []
    complete = True
    for ia, ib in overlap:
        if ia >= ib:
            continue
        counts["uniqueBroadPhasePairs"] += 1
        if counts["uniqueBroadPhasePairs"] > max_candidates:
            complete = False
            break
        a, b = triangles[ia], triangles[ib]
        if polygon_indices[ia] == polygon_indices[ib]:
            counts["samePolygonExcluded"] += 1
            continue
        if set(a) & set(b):
            counts["sharedIndexExcluded"] += 1
            continue
        av, bv = [points[i] for i in a], [points[i] for i in b]
        if any(_length(_sub(x, y)) <= tolerance for x in av for y in bv):
            counts["sharedPositionExcluded"] += 1
            continue
        counts["narrowPhasePairs"] += 1
        result = triangles_contact(av, bv, tolerance)
        if result is None:
            counts["degeneratePairsExcluded"] += 1
        elif result:
            counts["contacts"] += 1
            if len(samples) < max_samples:
                samples.append({"triangles": [ia, ib],
                                "polygons": [polygon_indices[ia], polygon_indices[ib]]})
    report.update(status="completed" if complete else "candidate_limit_reached",
                  completeWithinDeclaredScope=complete,
                  counts=dict(counts), contactPairSamples=samples,
                  contacts=counts["contacts"])
    return report


def inspect_mesh(mesh, matrix_world=None, *, self_intersections=False,
                 max_self_triangles=25000, max_self_candidates=200000,
                 tolerance=1e-8, sample_limit=30):
    """Inspect a Blender Mesh without editing it; values use world coordinates.

    ``calc_loop_triangles`` refreshes only Blender's derived tessellation cache.
    Tolerance is in world units. Triangle area tolerance is tolerance squared.
    """
    points = [tuple(matrix_world @ v.co) if matrix_world is not None else tuple(v.co)
              for v in mesh.vertices]
    finite_indices = [i for i, p in enumerate(points) if _finite(p)]
    finite_set = set(finite_indices)
    finite_points = [points[i] for i in finite_indices]
    mesh.calc_loop_triangles()
    triangles = [tuple(t.vertices) for t in mesh.loop_triangles]
    triangle_polygons = [t.polygon_index for t in mesh.loop_triangles]
    n = len(points)
    union = _UnionFind(n)
    edge_faces = defaultdict(list)
    polygon_used = set()
    edge_used = set()
    for edge in mesh.edges:
        a, b = tuple(edge.vertices)
        union.join(a, b)
        edge_used.update((a, b))
    for polygon in mesh.polygons:
        vs = list(polygon.vertices)
        polygon_used.update(vs)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            key = (min(a, b), max(a, b))
            edge_faces[key].append((polygon.index, 1 if a < b else -1))
            union.join(a, b)
    comps = defaultdict(list)
    for i in range(n):
        comps[union.find(i)].append(i)
    component_faces = Counter(union.find(p.vertices[0]) for p in mesh.polygons if p.vertices)
    component_tris = Counter(union.find(t[0]) for t in triangles)
    boundary = [k for k, fs in edge_faces.items() if len(fs) == 1]
    branching = [k for k, fs in edge_faces.items() if len(fs) > 2]
    winding = [k for k, fs in edge_faces.items() if len(fs) == 2 and fs[0][1] == fs[1][1]]
    face_edge_set = set(edge_faces)
    loose_edges = [e.index for e in mesh.edges
                   if tuple(sorted(tuple(e.vertices))) not in face_edge_set]
    zero_areas, nonfinite_areas = [], []
    area_sum = 0.0
    volumes = defaultdict(float)
    component_origins = {root: points[ids[0]] for root, ids in comps.items()}
    for i, triangle in enumerate(triangles):
        if not all(j in finite_set for j in triangle):
            nonfinite_areas.append(i)
            continue
        a, b, c = [points[j] for j in triangle]
        area = _length(_cross(_sub(b, a), _sub(c, a))) * 0.5
        if area <= tolerance * tolerance:
            zero_areas.append(i)
        area_sum += area
        root = union.find(triangle[0])
        origin = component_origins[root]
        volumes[root] += _dot(_sub(a, origin),
                              _cross(_sub(b, origin), _sub(c, origin))) / 6.0
    position_groups = Counter(points[i] for i in finite_indices)
    nonfinite_vertex_normals = [v.index for v in mesh.vertices if not _finite(v.normal)]
    zero_vertex_normals = [v.index for v in mesh.vertices
                           if _finite(v.normal) and _length(v.normal) < 1e-12]
    nonfinite_polygon_normals = [p.index for p in mesh.polygons if not _finite(p.normal)]
    nonfinite_corner_normals = None
    corner_normal_error = None
    try:
        nonfinite_corner_normals = sum(not _finite(item.vector) for item in mesh.corner_normals)
    except (AttributeError, RuntimeError) as exc:
        corner_normal_error = type(exc).__name__ + ": " + str(exc)
    uv_reports = []
    for layer in mesh.uv_layers:
        values = [tuple(item.uv) for item in layer.data]
        valid = [v for v in values if _finite(v)]
        zero_uv_triangles = 0
        for triangle in mesh.loop_triangles:
            a, b, c = [values[i] for i in triangle.loops]
            if all(_finite(v) for v in (a, b, c)):
                twice_area = abs((b[0] - a[0]) * (c[1] - a[1]) -
                                 (b[1] - a[1]) * (c[0] - a[0]))
                if twice_area <= 2e-14:
                    zero_uv_triangles += 1
        uv_reports.append({
            "name": layer.name, "loopCount": len(values),
            "nonfiniteLoops": len(values) - len(valid),
            "bounds": [[min(v[j] for v in valid), max(v[j] for v in valid)]
                       for j in range(2)] if valid else None,
            "outsideUnitSquareLoops": sum(any(x < 0 or x > 1 for x in v) for v in valid),
            "zeroAreaTrianglesAtUvTolerance1e-14": zero_uv_triangles,
            "note": "Outside [0,1] can be intentional tiling. This is not a UV overlap or stretch audit.",
        })
    component_reports = []
    for root, ids in sorted(comps.items(), key=lambda kv: len(kv[1]), reverse=True):
        component_reports.append({
            "firstVertex": min(ids), "vertices": len(ids),
            "polygons": component_faces[root], "triangles": component_tris[root],
            "signedVolumeWorldUnitsCubed": volumes[root] if _finite([volumes[root]]) else None,
        })
    return {
        "mesh": mesh.name, "vertices": n, "edges": len(mesh.edges),
        "polygons": len(mesh.polygons), "triangles": len(triangles),
        "polygonSizes": dict(Counter(len(p.vertices) for p in mesh.polygons)),
        "worldBounds": [[min(p[j] for p in finite_points), max(p[j] for p in finite_points)]
                        for j in range(3)] if finite_points else None,
        "surfaceAreaWorldUnitsSquared": area_sum if math.isfinite(area_sum) else None,
        "toleranceWorldUnits": tolerance,
        "positions": {"nonfinite": n - len(finite_indices),
                      "nonfiniteSamples": [i for i in range(n) if i not in finite_set][:sample_limit],
                      "exactCoincidentGroups": sum(c > 1 for c in position_groups.values()),
                      "exactCoincidentExtraVertices": sum(c - 1 for c in position_groups.values()),
                      "note": "Coincident UV/material seam vertices are not automatically defects."},
        "topology": {"boundaryEdges": len(boundary),
                     "branchingEdgesMoreThanTwoFaces": len(branching),
                     "nonmanifoldEdgesIncludingBoundaryAndLoose": len(boundary) + len(branching) + len(loose_edges),
                     "inconsistentTwoFaceWindingEdges": len(winding),
                     "looseEdges": len(loose_edges),
                     "verticesUnusedByEdgesOrFaces": sum(i not in edge_used and i not in polygon_used for i in range(n)),
                     "verticesNotUsedByPolygons": n - len(polygon_used),
                     "boundaryEdgeSamples": boundary[:sample_limit],
                     "branchingEdgeSamples": branching[:sample_limit],
                     "windingEdgeSamples": winding[:sample_limit],
                     "connectedVertexComponents": len(comps),
                     "surfaceComponents": sum(v > 0 for v in component_faces.values()),
                     "componentsLargestFirst": component_reports[:sample_limit],
                     "componentListTruncated": len(component_reports) > sample_limit,
                     "note": "Connectivity is by vertex indices; split seams can produce multiple components. Boundary edges can be intentional garment openings. Vertex-fan manifoldness not tested."},
        "triangulation": {"zeroAreaTriangles": len(zero_areas),
                          "nonfiniteTriangles": len(nonfinite_areas),
                          "zeroAreaTriangleSamples": zero_areas[:sample_limit]},
        "normals": {"nonfiniteVertexNormals": len(nonfinite_vertex_normals),
                    "zeroVertexNormals": len(zero_vertex_normals),
                    "nonfinitePolygonNormals": len(nonfinite_polygon_normals),
                    "nonfiniteCornerNormals": nonfinite_corner_normals,
                    "cornerNormalReadError": corner_normal_error,
                    "outwardOrientationCertified": False,
                    "note": "Winding consistency and signed volume are diagnostics. Outwardness of open/multi-shell surfaces needs visual review."},
        "uvLayers": uv_reports,
        "materials": {"slots": [m.name if m else None for m in mesh.materials],
                      "polygonUse": dict(Counter(p.material_index for p in mesh.polygons)),
                      "invalidMaterialPolygonCount": sum(p.material_index >= len(mesh.materials) for p in mesh.polygons)},
        "selfContacts": _self_contacts(points, triangles, triangle_polygons,
                                       enabled=self_intersections,
                                       max_triangles=max_self_triangles,
                                       max_candidates=max_self_candidates,
                                       max_samples=sample_limit, tolerance=tolerance),
    }


def inspect_scene(objects, *, evaluated=False, self_intersection_objects=(),
                  max_self_triangles=25000, max_self_candidates=200000,
                  tolerance=1e-8, sample_limit=30):
    """Return JSON-serializable diagnostics for explicit Blender objects.

    Pass names in self_intersection_objects to opt in per object. It never checks
    object/object collisions. Current evaluated state is a static sample only.
    The caller owns report persistence and source/export SHA correspondence.
    """
    import bpy
    selected_names = set(self_intersection_objects)
    depsgraph = bpy.context.evaluated_depsgraph_get() if evaluated else None
    result = {"schemaVersion": 1, "blenderVersion": bpy.app.version_string,
              "meshState": "evaluated_current_frame" if evaluated else "author_data",
              "frame": bpy.context.scene.frame_current,
              "readOnly": True, "sceneSaved": False,
              "objects": [], "skippedObjects": [], "errors": [],
              "limitations": [
                  "No artistic acceptance, silhouette identity, texture bleed, or finger-count certification.",
                  "No inter-object collisions, animation, global self-intersection, UV overlap or UV stretch certification.",
                  "Signed volume does not prove outward normals on open or multiple nested shells.",
                  "Self-contact reports include touching, exclude shared positions and are bounded per explicit object.",
              ]}
    for obj in list(objects):
        if obj.type != "MESH":
            result["skippedObjects"].append({"name": obj.name, "type": obj.type})
            continue
        owner = obj.evaluated_get(depsgraph) if evaluated else obj
        temporary_mesh = None
        try:
            mesh = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph) if evaluated else obj.data
            if evaluated:
                temporary_mesh = mesh
            entry = inspect_mesh(mesh, owner.matrix_world,
                                 self_intersections=obj.name in selected_names,
                                 max_self_triangles=max_self_triangles,
                                 max_self_candidates=max_self_candidates,
                                 tolerance=tolerance, sample_limit=sample_limit)
            entry["object"] = obj.name
            entry["matrixWorldFinite"] = all(_finite(row) for row in owner.matrix_world)
            determinant = owner.matrix_world.to_3x3().determinant()
            entry["matrixWorldDeterminant"] = determinant if math.isfinite(determinant) else None
            entry["modifiers"] = [{"name": m.name, "type": m.type,
                                   "viewport": m.show_viewport, "render": m.show_render}
                                  for m in obj.modifiers]
            result["objects"].append(entry)
        except Exception as exc:
            result["errors"].append({"object": obj.name,
                                     "error": type(exc).__name__ + ": " + str(exc)})
        finally:
            if temporary_mesh is not None:
                owner.to_mesh_clear()
    result["totals"] = {key: sum(entry[key] for entry in result["objects"])
                        for key in ("vertices", "edges", "polygons", "triangles")}
    result["totals"]["objectsInspected"] = len(result["objects"])
    result["notAnAcceptanceVerdict"] = True
    return result
