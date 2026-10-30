"""Read-only current frame-one head/neck posture, surface and neckline evidence.

No sculpting, rigging, rendering or Blend save. POSE and REST are evaluated
temporarily in a fresh background process. Full actual head face-group IDs
are exported in original evaluated polygon order, with topology hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
RIG = 'Character_Rig'
BODY = 'R01 Head'
BONES = ('upperChest', 'neck', 'head', 'leftShoulder', 'rightShoulder')
GARMENTS = ('Blouse_Tucked_Bodice', 'Blouse_PeterPan_Collar_L', 'Blouse_PeterPan_Collar_R')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def ah(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode()+a.tobytes()).hexdigest()


def matrix(m):
    return [list(row) for row in m]


def arr(items, property_name, width=1, dtype=np.int32):
    a = np.empty((len(items), width), dtype)
    items.foreach_get(property_name, a.ravel())
    return a if width != 1 else a.ravel()


def boundary_edges(mesh):
    loops = arr(mesh.loops, 'vertex_index')
    starts = arr(mesh.polygons, 'loop_start')
    sizes = arr(mesh.polygons, 'loop_total')
    nxt = np.arange(len(loops))+1
    nxt[starts+sizes-1] = starts
    e = np.sort(np.column_stack((loops, loops[nxt])), axis=1)
    e, counts = np.unique(e, axis=0, return_counts=True)
    return e[counts == 1], loops, starts, sizes


def bone_evidence(rig):
    rows = {}
    for name in BONES:
        if name not in rig.data.bones:
            rows[name] = {'available': False}
            continue
        b, p = rig.data.bones[name], rig.pose.bones[name]
        rows[name] = {'available': True, 'parent': b.parent.name if b.parent else None,
            'headRestWorld': list(rig.matrix_world@b.head_local),
            'tailRestWorld': list(rig.matrix_world@b.tail_local),
            'headPoseWorld': list(rig.matrix_world@p.head),
            'tailPoseWorld': list(rig.matrix_world@p.tail),
            'matrixRestWorld': matrix(rig.matrix_world@b.matrix_local),
            'matrixPoseWorld': matrix(rig.matrix_world@p.matrix),
            'matrixBasis': matrix(p.matrix_basis), 'rotationMode': p.rotation_mode,
            'constraints': [{'name': c.name, 'type': c.type, 'influence': float(c.influence)} for c in p.constraints]}
    return rows


def evaluated(obj, prefix, payload, role):
    import bpy
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    mesh = ev.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
    try:
        m = np.asarray(ev.matrix_world, float)
        local = arr(mesh.vertices, 'co', 3, np.float64)
        world = local@m[:3, :3].T+m[:3, 3]
        edges, loops, starts, sizes = boundary_edges(mesh)
        row = {'name': obj.name, 'prefix': prefix, 'vertices': len(world),
            'polygons': len(mesh.polygons), 'matrixWorld': matrix(ev.matrix_world),
            'boundsWorld': [world.min(0).tolist(), world.max(0).tolist()],
            'loopsSha256': ah(loops), 'faceStartsSha256': ah(starts), 'faceSizesSha256': ah(sizes)}
        if role == 'head':
            if len(world) != 238889 or len(mesh.polygons) != 238672:
                raise RuntimeError('Current head topology changed; do not silently reuse landmark IDs')
            attr = mesh.attributes.get('.sculpt_face_set')
            if attr is None or attr.domain != 'FACE' or attr.data_type != 'INT':
                raise RuntimeError('Full evaluated head FACE/INT .sculpt_face_set missing')
            groups = arr(attr.data, 'value')
            payload[prefix+'_faceGroups'] = groups
            row['faceGroupsSha256'] = ah(groups)
            row['faceGroupsCounts'] = {str(k): int(v) for k, v in zip(*np.unique(groups, return_counts=True))}
            ids = np.array([3214, 45312], np.int32)
            payload[prefix+'_landmarkIds'] = ids
            payload[prefix+'_landmarkWorld'] = world[ids]
            row['landmarkMeanings'] = ['L3 vertex closest to existing cage chin 7972; not an automatic anatomical landmark', 'previous actual crown vertex 45312; maximum Z is separately measured']
            crown = int(np.argmax(world[:, 2]))
            row['actualMaximumZ'] = {'vertexId': crown, 'world': world[crown].tolist()}
            selected = edges
        elif role == 'body':
            p = world[edges]
            selected = edges[(p[:, :, 2].min(1) > 1.22) & (p[:, :, 2].max(1) < 1.31) & (np.abs(p[:, :, 0]).max(1) < .09)]
            row['shoulderSurfacePatchMaxima'] = {}
            for side, sign in [('L', -1), ('R', 1)]:
                ids = np.flatnonzero((world[:, 0]*sign > .09) & (world[:, 0]*sign < .17) & (np.abs(world[:, 1]) < .05))
                if not len(ids):
                    raise RuntimeError('Expected body shoulder surface patch missing')
                i = int(ids[np.argmax(world[ids, 2])])
                row['shoulderSurfacePatchMaxima'][side] = {'vertexId': i, 'world': world[i].tolist()}
            row['shoulderLandmarkLimitation'] = 'Defined surface-patch maxima only; not identified acromion or clavicle landmarks.'
        else:
            # Small actual garment meshes/upper-bodice patch; no raw cage is
            # substituted for the deformed sewn surface. Connectivity remains
            # in full evaluated vertex IDs for exact boundary reconstruction.
            keep = np.ones(len(world), bool) if 'Collar' in obj.name else (world[:, 2] > 1.20) & (np.abs(world[:, 0]) < .14)
            ids = np.flatnonzero(keep).astype(np.int32)
            mesh.calc_loop_triangles()
            tri = arr(mesh.loop_triangles, 'vertices', 3)
            tri = tri[np.all(keep[tri], axis=1)]
            payload[prefix+'_vertexIds'] = ids
            payload[prefix+'_world'] = world[ids]
            payload[prefix+'_trianglesOriginalIds'] = tri
            selected = edges[np.all(keep[edges], axis=1)]
            row['garmentPointLimitation'] = 'Actual garment surface and open boundaries, not anatomical clavicle landmarks.'
        ids = np.unique(selected).astype(np.int32)
        payload[prefix+'_boundaryEdgesOriginalIds'] = selected.astype(np.int32)
        payload[prefix+'_boundaryVertexIds'] = ids
        payload[prefix+'_boundaryWorld'] = world[ids]
        row['selectedBoundaryVertices'] = len(ids)
        return row
    finally:
        ev.to_mesh_clear()


def observe(source_relative, output_relative):
    import bpy
    project = project_path()
    source, output = (project/source_relative).resolve(), (project/output_relative).resolve()
    for p in (source, output):
        p.relative_to((project/'work/reconstruction-r02').resolve())
    npz = output.with_suffix('.npz')
    if source.suffix != '.blend' or output.suffix != '.json' or output.exists() or npz.exists():
        raise RuntimeError('Explicit new JSON/NPZ output and existing Blend source required')
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError('Run only in a fresh background process opened on the explicit source')
    head, rig, scene = bpy.data.objects[HEAD], bpy.data.objects[RIG], bpy.context.scene
    source_sha = sha(source)
    frame = [scene.frame_current, scene.frame_subframe]
    old_pose = rig.data.pose_position
    uv = runpy.run_path(str(Path(__file__).with_name('production_uv_r08.py')))
    pose_contract = uv['_pose_contract']()
    key_values = {k.name: float(k.value) for k in head.data.shape_keys.key_blocks}
    arrays = {}
    report = {'source': source_relative, 'sourceSha256': source_sha, 'helperSha256': sha(__file__),
        'sourceFrame': frame, 'reviewFrame': 1, 'savedBlend': False, 'geometryModified': False,
        'scope': 'Read-only frame-one net head posture, physical head-neck-shoulder surface distances, current actual garment neckline evidence, and full evaluated head face groups. No anatomical reference or appearance acceptance is inferred from numeric values.',
        'states': {}}
    try:
        scene.frame_set(1)
        bpy.context.view_layer.update()
        report['frameOneKeyValues'] = {k.name: float(k.value) for k in head.data.shape_keys.key_blocks}
        action = getattr(rig.animation_data, 'action', None) if rig.animation_data else None
        report['rigAction'] = action.name if action else None
        for pose in ('POSE', 'REST'):
            rig.data.pose_position = pose
            rig.data.update_tag()
            bpy.context.view_layer.update()
            row = {'rigMatrixWorld': matrix(rig.matrix_world), 'posePosition': pose, 'bones': bone_evidence(rig), 'objects': {}}
            for name, role in [(HEAD, 'head'), (BODY, 'body')] + [(n, 'garment') for n in GARMENTS]:
                row['objects'][name] = evaluated(bpy.data.objects[name], pose.lower()+'_'+role+('_'+str(GARMENTS.index(name)) if role == 'garment' else ''), arrays, role)
            report['states'][pose] = row
    finally:
        rig.data.pose_position = old_pose
        rig.data.update_tag()
        scene.frame_set(frame[0], subframe=frame[1])
        bpy.context.view_layer.update()
    if uv['_pose_contract']() != pose_contract or {k.name: float(k.value) for k in head.data.shape_keys.key_blocks} != key_values or sha(source) != source_sha:
        raise RuntimeError('Source, pose or expression state did not restore exactly')
    np.savez_compressed(npz, **arrays)
    report.update(arrays=npz.relative_to(project).as_posix(), arraysSha256=sha(npz), sourcePoseAndExpressionRestored=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('R11_HEAD_NECK_POSTURE_OBSERVED '+json.dumps({'sourceSha256': source_sha, 'output': output_relative, 'arraysSha256': report['arraysSha256']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
    observe(args.source, args.output)
