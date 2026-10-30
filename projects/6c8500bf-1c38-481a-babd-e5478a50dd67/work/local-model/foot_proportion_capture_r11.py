"""Small, read-only current stature/ankle/footwear capture for foot adaptation.

Run only inside the integrator's disposable background Blender. No donor is
appended and no source, rig, mesh, shoe design, material or action is saved.
The rig is temporarily evaluated in REST, then restored before writing evidence.
This captures actual coordinates, not an anthropometric target or acceptance.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

OBSERVER_SHA = 'b5de295c393ba74416b7b095cb8f06e777fb975aa0ab97adbef1850ffd9ce36c'
BASELINE = 'work/reconstruction-r02/foot-asset-observation-r11.json'
BASELINE_JSON_SHA = '67d6afbc595cb757c533d66f53bd494e24b29f172356bda251fb1530b98b336c'
BASELINE_ARRAY_SHA = '202c454f342a15613ede0e03cc2cd668f649fd832759389aecb8b43f3ac45564'
HEAD = 'R11 Female head study'
BODY = 'R01 Head'
RIG = 'Character_Rig'
PREFIXES = ('Body_Complete_Shin_Foot_', 'Sock_', 'Shoe_')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def path(relative):
    p = (project_path()/relative).resolve()
    p.relative_to(project_path().resolve())
    return p


def values(collection, property_name, width, dtype):
    a = np.empty(len(collection)*width, dtype=dtype)
    collection.foreach_get(property_name, a)
    return a.reshape((-1, width)) if width > 1 else a


def world(local, matrix):
    m = np.asarray(matrix, np.float64)
    return np.asarray(local, np.float64) @ m[:3, :3].T + m[:3, 3]


def bounds(points):
    if not len(points) or not np.isfinite(points).all():
        raise RuntimeError('Missing or nonfinite actual geometry')
    lo, hi = points.min(0), points.max(0)
    return {'minimum': lo.tolist(), 'maximum': hi.tolist(),
            'dimensions': (hi-lo).tolist(), 'vertices': len(points)}


def raw_mesh(obj):
    mesh = obj.data
    return {'positions': values(mesh.vertices, 'co', 3, np.float32),
            'edges': values(mesh.edges, 'vertices', 2, np.int32),
            'loops': values(mesh.loops, 'vertex_index', 1, np.int32),
            'polygonStarts': values(mesh.polygons, 'loop_start', 1, np.int32),
            'polygonSizes': values(mesh.polygons, 'loop_total', 1, np.int32)}


def curve_data(obj):
    data = obj.data
    rows = []
    for spline in data.splines:
        row = {'type':spline.type, 'cyclic':spline.use_cyclic_u, 'orderU':spline.order_u,
               'resolutionU':spline.resolution_u}
        if spline.type == 'BEZIER':
            row['points'] = [{'co':list(p.co), 'handleLeft':list(p.handle_left),
                'handleRight':list(p.handle_right), 'leftType':p.handle_left_type,
                'rightType':p.handle_right_type, 'radius':p.radius, 'tilt':p.tilt}
                for p in spline.bezier_points]
        else:
            row['points'] = [{'co':list(p.co), 'radius':p.radius, 'tilt':p.tilt,
                             'weightSoftbody':p.weight_softbody} for p in spline.points]
        rows.append(row)
    return {'splines':rows, 'bevelDepth':data.bevel_depth,
            'bevelResolution':data.bevel_resolution, 'resolutionU':data.resolution_u,
            'dimensions':data.dimensions, 'fillMode':data.fill_mode}


def source_contract(api, objects):
    return {'generic':api['_source_contract'](objects),
            'curves':{o.name:curve_data(o) for o in objects if o.type == 'CURVE'}}


def evaluated(obj, graph):
    ev = obj.evaluated_get(graph)
    mesh = ev.to_mesh(preserve_all_data_layers=True, depsgraph=graph)
    try:
        mesh.calc_loop_triangles()
        points = world(values(mesh.vertices, 'co', 3, np.float32), ev.matrix_world)
        triangles = values(mesh.loop_triangles, 'vertices', 3, np.int32).copy()
        matrix = np.asarray(ev.matrix_world, np.float64).copy()
    finally:
        ev.to_mesh_clear()
    return points, triangles, matrix


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def compact_bones(rig):
    matrix = np.asarray(rig.matrix_world, np.float64)
    rows = {}
    names = ['hips', 'spine', 'chest', 'neck', 'head']
    names += [s+p for s in ('left', 'right') for p in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes')]
    for name in names:
        b = rig.data.bones.get(name)
        if b is None: raise RuntimeError('Missing actual existing bone: '+name)
        endpoints = world(np.asarray([b.head_local[:], b.tail_local[:]], np.float64), matrix)
        rows[name] = {'parent': b.parent.name if b.parent else None,
                      'headWorld': endpoints[0].tolist(), 'tailWorld': endpoints[1].tolist(),
                      'lengthWorld': float(np.linalg.norm(endpoints[1]-endpoints[0])),
                      'restMatrixLocal': np.asarray(b.matrix_local).tolist(),
                      'poseMatrixBasis': np.asarray(rig.pose.bones[name].matrix_basis).tolist()}
    return rows


def capture(source_relative, source_sha, output_relative):
    import bpy
    source, output = path(source_relative), path(output_relative)
    npz_path = output.with_suffix('.npz')
    if output.exists() or npz_path.exists(): raise FileExistsError('Use a new evidence path')
    if not bpy.app.background or bpy.context.mode != 'OBJECT':
        raise RuntimeError('Disposable background Object Mode required')
    if Path(bpy.data.filepath).resolve() != source or sha(source) != source_sha:
        raise RuntimeError('Explicit actual source path and SHA required')
    scene = bpy.context.scene
    if scene.frame_current != 1 or scene.frame_subframe != 0:
        raise RuntimeError('Capture the authored frame 1 source; do not silently reset animation')
    baseline_path = path(BASELINE)
    if sha(baseline_path) != BASELINE_JSON_SHA:
        raise RuntimeError('Frozen actual foot observation metadata identity changed')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline_arrays_path = path(baseline['arrays'])
    if sha(baseline_arrays_path) != BASELINE_ARRAY_SHA or baseline['arraysSha256'] != BASELINE_ARRAY_SHA:
        raise RuntimeError('Frozen actual foot observation array identity changed')
    helper = Path(__file__).with_name('hand_asset_observe_r10.py')
    if sha(helper) != OBSERVER_SHA: raise RuntimeError('Frozen source-contract helper changed')
    api = runpy.run_path(str(helper))
    baseline_objects = {o['name']: (i, o) for i, o in enumerate(baseline['current'])}
    old_arrays = np.load(baseline_arrays_path, allow_pickle=False)
    selected = sorted([o for o in scene.objects if o.name.startswith(PREFIXES)], key=lambda o:o.name)
    required = [n for n in baseline_objects if n != RIG]
    missing = sorted(set(required)-{o.name for o in selected})
    if missing or len(selected) > 256 or any(o.type not in ('MESH','CURVE') for o in selected):
        raise RuntimeError('Reconcile actual footwear inventory: '+json.dumps({'missing':missing,
            'selected':[(o.name,o.type) for o in selected]}))
    head, body, rig = (bpy.data.objects[n] for n in (HEAD, BODY, RIG))
    guarded = selected+[head, body, rig]
    original_ids = set(api['_ids'](bpy))
    before = source_contract(api, guarded)
    old_pose = rig.data.pose_position
    arrays = {}
    result = {'schemaVersion':1, 'mode':'foot-proportion-capture-r11',
        'source':source_relative, 'sourceSha256':source_sha, 'helperSha256':sha(__file__),
        'baseline':BASELINE, 'baselineSha256':sha(baseline_path),
        'baselineArraySha256':BASELINE_ARRAY_SHA, 'frame':1,
        'units':{'system':scene.unit_settings.system, 'scaleLength':scene.unit_settings.scale_length,
                 'lengthUnit':scene.unit_settings.length_unit,
                 'coordinateMeaning':'World Blender coordinates; physical lengths require recorded scaleLength interpretation.'},
        'originalPosePosition':old_pose, 'rigMatrixWorld':np.asarray(rig.matrix_world).tolist(),
        'bones':compact_bones(rig), 'objects':[], 'savedBlend':False,
        'sourceGeometryEdited':False, 'targetFootLengthSelected':False, 'appearanceAccepted':False,
        'scope':'Current actual head crown, body lower-shin patch, feet, complete Shoe_/Sock_ object families and rest bone endpoints. REST rig only; all other active modifiers retained and recorded. No hair bounds used as stature.'}
    old_rig = baseline_objects[RIG][1]
    old_bones = {b['name']:b for b in old_rig['bones']}
    result['baselineLegRigCompatibility'] = {
        'rigMatrixWorldExact':np.array_equal(np.asarray(rig.matrix_world),np.asarray(old_rig['matrixWorld'])),
        'bones':{name:{'restMatrixExact':np.array_equal(np.asarray(row['restMatrixLocal']),np.asarray(old_bones[name]['restMatrixLocal'])),
                       'parentExact':row['parent']==old_bones[name]['parent']}
                 for name,row in result['bones'].items() if name.startswith(('left','right'))}}
    try:
        for i, obj in enumerate(selected):
            prefix = 'footwear_'+str(i)
            raw = raw_mesh(obj) if obj.type == 'MESH' else {}
            row = {'name':obj.name, 'type':obj.type, 'matrixWorld':np.asarray(obj.matrix_world).tolist(),
                'parent':obj.parent.name if obj.parent else None, 'parentBone':obj.parent_bone,
                'hideRender':obj.hide_render, 'hideViewport':obj.hide_viewport,
                'modifiers':[api['_rna'](m) for m in obj.modifiers],
                'groups':[(g.name,g.index) for g in obj.vertex_groups],
                'rawArrayHashes':{k:array_hash(v) for k,v in raw.items()},
                'baselineCompatibility':{}}
            if obj.type == 'CURVE': row['rawCurve'] = curve_data(obj)
            row['materialNames'] = [slot.material.name if slot.material else None for slot in obj.material_slots]
            row['shapeKeys'] = [{'name':k.name,'value':k.value,'mute':k.mute,'relative':k.relative_key.name}
                               for k in obj.data.shape_keys.key_blocks] if obj.data.shape_keys else []
            keys = {'positions':'positions','edges':'edges','loops':'loopVertex',
                    'polygonStarts':'polygonStart','polygonSizes':'polygonSize'}
            if obj.name in baseline_objects:
                j, old = baseline_objects[obj.name]
                row['baselineCompatibility'] = {k:np.array_equal(v, old_arrays['current_'+str(j)+'_raw_'+keys[k]]) for k,v in raw.items()}
                row['baselineCompatibility']['groupNamesAndIndices'] = row['groups'] == [(g['name'],g['index']) for g in old['vertexGroups']]
                row['baselineCompatibility']['modifiersExact'] = row['modifiers'] == old['modifiers']
            else:
                row['baselineCompatibility'] = {'notInEarlierSelectedInventory':True}
            for k,v in raw.items(): arrays[prefix+'_raw_'+k] = v
            weights = [(v.index,g.group,g.weight) for v in obj.data.vertices for g in v.groups] if obj.type == 'MESH' else []
            arrays[prefix+'_weightVertex'] = np.asarray([w[0] for w in weights],np.int32)
            arrays[prefix+'_weightGroup'] = np.asarray([w[1] for w in weights],np.int32)
            arrays[prefix+'_weightValue'] = np.asarray([w[2] for w in weights],np.float32)
            if obj.name in baseline_objects:
                j, old = baseline_objects[obj.name]
                for current,old_key in (('weightVertex','vertex'),('weightGroup','group'),('weightValue','value')):
                    row['baselineCompatibility'][current] = np.array_equal(arrays[prefix+'_'+current],old_arrays[old['weights'][old_key]['array']])
            row['arrayPrefix'] = prefix
            result['objects'].append(row)
        rig.data.pose_position = 'REST'
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        all_bounds = {}
        for row, obj in zip(result['objects'], selected):
            points, triangles, matrix = evaluated(obj, graph)
            prefix = row['arrayPrefix']
            arrays[prefix+'_restWorld'] = points
            arrays[prefix+'_restTriangles'] = triangles
            row['restEvaluatedBounds'] = bounds(points)
            row['restEvaluatedMatrixWorld'] = matrix.tolist()
            all_bounds[obj.name] = row['restEvaluatedBounds']
        points, _, matrix = evaluated(head, graph)
        result['headCrown'] = {'name':head.name, 'bounds':bounds(points),
            'highestActualPoint':points[np.argmax(points[:,2])].tolist(),
            'evaluatedMatrixWorld':matrix.tolist(), 'hairExcluded':True,
            'modifiers':[api['_rna'](m) for m in head.modifiers]}
        points, triangles, _ = evaluated(body, graph)
        chosen = np.flatnonzero((points[triangles,2] <= .50).any(1))
        used = np.unique(triangles[chosen])
        reverse = np.full(len(points),-1,np.int32); reverse[used] = np.arange(len(used))
        arrays['bodyLowerPatch_originalVertexIds'] = used
        arrays['bodyLowerPatch_restWorld'] = points[used]
        arrays['bodyLowerPatch_triangles'] = reverse[triangles[chosen]]
        result['bodyLowerPatch'] = {'name':body.name, 'selection':'Existing triangles with any world vertex Z <= 0.50; no artificial cap',
            'wholeBodyBounds':bounds(points), 'patchVertices':len(used), 'patchTriangles':len(chosen),
            'modifiers':[api['_rna'](m) for m in body.modifiers]}
        crown = result['headCrown']['bounds']['maximum'][2]
        bare_min = min(all_bounds['Body_Complete_Shin_Foot_'+s]['minimum'][2] for s in ('L','R'))
        shoe_min = min(all_bounds[n]['minimum'][2] for n in all_bounds if n.startswith(('Shoe_Outsole_','Shoe_Stacked_Heel_')))
        result['verticalSpans'] = {'bareFootMinimumWorldZ':bare_min, 'shoeBottomMinimumWorldZ':shoe_min,
            'headCrownWorldZ':crown, 'crownMinusExistingBareFoot':crown-bare_min,
            'crownMinusExistingShoeBottom':crown-shoe_min,
            'meaning':'Geometric rest spans. Existing bare-foot/footwear geometry is not already accepted anatomy; neither value is a clinical stature measurement.'}
    finally:
        rig.data.pose_position = old_pose
        bpy.context.view_layer.update()
        after = source_contract(api, guarded)
        result['restoration'] = {'sourceContractExact':before==after,
            'posePositionExact':rig.data.pose_position==old_pose,
            'datablockIdsExact':set(api['_ids'](bpy))==original_ids,
            'frameExact':scene.frame_current==1 and scene.frame_subframe==0,
            'sourceDiskShaExact':sha(source)==source_sha}
        old_arrays.close()
        if not all(result['restoration'].values()):
            raise RuntimeError('Read-only proportion capture did not restore source: '+json.dumps(result['restoration']))
    output.parent.mkdir(parents=True,exist_ok=True)
    with npz_path.open('xb') as f: np.savez_compressed(f,**arrays)
    result['arrays'] = str(npz_path.relative_to(project_path()))
    result['arraysSha256'] = sha(npz_path)
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    return {'report':str(output.relative_to(project_path())), 'arraysSha256':result['arraysSha256'],
            'footwearObjects':len(selected), 'verticalSpans':result['verticalSpans'],
            'restoration':result['restoration'], 'savedBlend':False}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--' in argv: argv = argv[argv.index('--')+1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--source-sha',required=True)
    parser.add_argument('--output',default='work/reconstruction-r02/foot-proportion-capture-r11.json')
    args = parser.parse_args(argv)
    print('R11_FOOT_PROPORTION '+json.dumps(capture(args.source,args.source_sha,args.output),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
