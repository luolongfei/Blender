"""Independent R11 optical-depth candidate inside the unchanged eyelid globe.

The existing outer sclera mesh keeps every vertex/face/UV. Its front central
region becomes a refractive interface; the old externally pasted iris moves
inside, with its same x/z dimensions, topology, UV and packed source image.
This is an explicitly simplified single optical boundary, not a measured eye
or complete corneal/aqueous/lens anatomy. No eye-envelope or blink edits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

HEAD = 'R11 Female head study'
MARKER = 'R11_eye_depth_candidate'
PARAMETERS = dict(irisPlaneBehindOriginalRimMeters=.00050, irisCentralRecessMeters=.00012,
    corneaIOR=1.376, corneaRoughness=.015, limbusTransitionMeters=.00025,
    scleraRoughness=.17, scleraSubsurfaceWeight=.035, scleraSubsurfaceScale=.0007,
    scleraSideTintMaximum=.12)


def _api(name): return runpy.run_path(str(Path(__file__).with_name(name + '.py')))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _array(items, field='co', width=3, dtype=np.float32):
    result = np.empty((len(items), width), dtype); items.foreach_get(field, result.ravel()); return result


def _ah(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes()).hexdigest()


def plan_iris_depth(globe_local, iris_local, scale):
    """Pure array part; actual ray clearance is additionally checked in Blender."""
    globe = np.asarray(globe_local, float); iris = np.asarray(iris_local, float)
    if globe.ndim != 2 or iris.ndim != 2 or globe.shape[1] != 3 or iris.shape[1] != 3 or not .5 < scale < 1.5:
        raise ValueError('Actual eye arrays and positive uniform physical scale required')
    center = (globe.min(0) + globe.max(0)) * .5
    if np.linalg.norm(center) * scale > .00002: raise RuntimeError('Actual eye origin is not its calibrated center')
    radius = np.linalg.norm(iris[:, (0, 2)], axis=1); rim_radius = float(radius.max())
    if not .0055 < rim_radius * scale < .0065: raise RuntimeError('Unexpected physical iris diameter')
    front = float(globe[:, 1].min())
    # The measured donor already has a steep convex anterior region. A plane
    # only 2.1 mm behind its pole would cross the outer iris, so derive the plane
    # from the actual existing cap's most posterior rim, not a spherical guess.
    plane = float(iris[:, 1].max()) + PARAMETERS['irisPlaneBehindOriginalRimMeters'] / scale
    if not .0028 < (plane - front) * scale < .0036:
        raise RuntimeError('Inspect unexpected actual corneal cap depth before recessing the iris')
    result = iris.copy()
    result[:, 1] = plane + PARAMETERS['irisCentralRecessMeters'] * np.clip(1 - radius / rim_radius, 0, 1) ** 2 / scale
    if not np.isfinite(result).all() or np.min(result[:, 1] - iris[:, 1]) * scale < .00025:
        raise RuntimeError('Expected a recessed derivative of the current external iris cap')
    return result.astype(np.float32), dict(physicalIrisRadiusMeters=rim_radius * scale,
        physicalMovementMetersQuantiles=np.quantile((result[:, 1] - iris[:, 1]) * scale, [0, .5, 1]).tolist(),
        irisPlaneBehindFrontMeters=(plane - front) * scale,
        originalCapSagMeters=(float(iris[:, 1].max()) - float(iris[:, 1].min())) * scale,
        centralBowlDepthMeters=PARAMETERS['irisCentralRecessMeters'], xzUnchanged=True)


def _geometry(obj):
    mesh = obj.data
    return dict(co=_ah(_array(mesh.vertices)), loops=_ah(_array(mesh.loops, 'vertex_index', 1, np.int32)),
        starts=_ah(_array(mesh.polygons, 'loop_start', 1, np.int32)),
        sizes=_ah(_array(mesh.polygons, 'loop_total', 1, np.int32)),
        uv={uv.name: _ah(_array(uv.data, 'uv', 2)) for uv in mesh.uv_layers},
        matrix=[list(row) for row in obj.matrix_world], parent=obj.parent.name if obj.parent else None,
        parentInverse=[list(row) for row in obj.matrix_parent_inverse],
        basis=[list(row) for row in obj.matrix_basis], modifiers=[[m.name, m.type] for m in obj.modifiers])


def _math(nt, op, a, b=None):
    node = nt.nodes.new('ShaderNodeMath'); node.operation = op
    for index, value in enumerate((a, b)):
        if value is None: continue
        if isinstance(value, (float, int)): node.inputs[index].default_value = value
        else: nt.links.new(value, node.inputs[index])
    return node.outputs[0]


def _smooth_ramp(nt, value, start, end):
    node = nt.nodes.new('ShaderNodeMapRange'); node.clamp = True; node.interpolation_type = 'SMOOTHSTEP'
    node.inputs['From Min'].default_value = start; node.inputs['From Max'].default_value = end
    node.inputs['To Min'].default_value = 0.; node.inputs['To Max'].default_value = 1.
    nt.links.new(value, node.inputs['Value']); return node.outputs[0]


def _sclera_material(original, iris_radius_local, physical_scale):
    mat = original.copy(); mat.name = 'R11 sclera with internal iris optical window'
    nt = mat.node_tree; shaders = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED']
    outputs = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    if len(shaders) != 1 or len(outputs) != 1: raise RuntimeError('Inspect nonstandard sclera shader graph')
    shader = shaders[0]; output = outputs[0]
    if shader.inputs['Base Color'].is_linked or shader.inputs['Roughness'].is_linked:
        raise RuntimeError('Current sclera differs from the documented uniform source')
    original_color = tuple(shader.inputs['Base Color'].default_value)
    shader.inputs['Roughness'].default_value = PARAMETERS['scleraRoughness']
    shader.inputs['Subsurface Weight'].default_value = PARAMETERS['scleraSubsurfaceWeight']
    shader.inputs['Subsurface Radius'].default_value = (1., .45, .3)
    shader.inputs['Subsurface Scale'].default_value = PARAMETERS['scleraSubsurfaceScale']
    tex = nt.nodes.new('ShaderNodeTexCoord'); xyz = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tex.outputs['Object'], xyz.inputs[0])
    radius2 = _math(nt, 'ADD', _math(nt, 'MULTIPLY', xyz.outputs['X'], xyz.outputs['X']),
        _math(nt, 'MULTIPLY', xyz.outputs['Z'], xyz.outputs['Z']))
    radius = _math(nt, 'SQRT', radius2)
    edge_t = _smooth_ramp(nt, _math(nt, 'ABSOLUTE', xyz.outputs['X']), .007 / physical_scale, .012 / physical_scale)
    tint = nt.nodes.new('ShaderNodeMixRGB'); tint.blend_type = 'MIX'
    nt.links.new(_math(nt, 'MULTIPLY', edge_t, PARAMETERS['scleraSideTintMaximum']), tint.inputs[0])
    tint.inputs[1].default_value = original_color; tint.inputs[2].default_value = (.44, .30, .27, 1.)
    nt.links.new(tint.outputs[0], shader.inputs['Base Color'])
    transition = PARAMETERS['limbusTransitionMeters'] / physical_scale
    opaque_radial = _smooth_ramp(nt, radius, iris_radius_local - transition * .35, iris_radius_local + transition * .65)
    front = _math(nt, 'LESS_THAN', xyz.outputs['Y'], -.004 / physical_scale)
    window = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1., opaque_radial), front)
    glass = nt.nodes.new('ShaderNodeBsdfGlass'); glass.name = 'R11 explicit refractive anterior interface'
    glass.inputs['Color'].default_value = (1., 1., 1., 1.)
    glass.inputs['Roughness'].default_value = PARAMETERS['corneaRoughness']
    glass.inputs['IOR'].default_value = PARAMETERS['corneaIOR']
    mix = nt.nodes.new('ShaderNodeMixShader'); nt.links.new(window, mix.inputs[0])
    nt.links.new(shader.outputs['BSDF'], mix.inputs[1]); nt.links.new(glass.outputs['BSDF'], mix.inputs[2])
    nt.links.new(mix.outputs[0], output.inputs['Surface'])
    mat['R11_optics_scope'] = 'Same exact external globe; regional refractive optical window over recessed iris. Single-interface approximation; no shadow-ray transparency cheat.'
    return mat


def _iris_material(original):
    mat = original.copy(); mat.name = 'R11 recessed diffuse iris'
    nodes = [n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
    if len(nodes) != 1: raise RuntimeError('Expected documented iris Principled shader')
    shader = nodes[0]
    for name, value in [('Coat Weight', 0.), ('Specular IOR Level', 0.), ('Roughness', .32), ('Metallic', 0.), ('Transmission Weight', 0.)]:
        if shader.inputs[name].is_linked: raise RuntimeError('Unexpected linked iris optical input: ' + name)
        shader.inputs[name].default_value = value
    mat['R11_optics_scope'] = 'Same packed image and UV, beneath the separate globe optical interface; no duplicate coat highlight.'
    return mat


def _check_internal_clearance(globe, iris):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    co = _array(globe.data.vertices); faces = [tuple(p.vertices) for p in globe.data.polygons]
    bvh = BVHTree.FromPolygons([Vector(row) for row in co], faces)
    scale = float(globe.matrix_world.to_scale().x); start = float(co[:, 1].min() - .02)
    gaps = []
    for point in _array(iris.data.vertices):
        hit, _, _, _ = bvh.ray_cast(Vector((float(point[0]), start, float(point[2]))), Vector((0, 1, 0)), .08)
        if hit is None: raise RuntimeError('Actual iris point is outside original globe front silhouette')
        gaps.append((float(point[1]) - hit.y) * scale)
    if min(gaps) < .00025 or max(gaps) > .0038:
        raise RuntimeError('Internal iris must retain bounded positive clearance: ' + str([min(gaps), max(gaps)]))
    return dict(frontSurfaceToIrisMetersQuantiles=np.quantile(gaps, [0, .1, .5, .9, 1]).tolist(),
        outerGlobeUnchanged=True, checkedActualIrisVertices=len(gaps))


def apply():
    import bpy
    scene = bpy.context.scene
    if not bpy.app.background or bpy.context.mode != 'OBJECT' or MARKER in scene:
        raise RuntimeError('Fresh independent background optical candidate required')
    source = Path(bpy.data.filepath).resolve(); source.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    if 'R11_gaze_control' in scene:
        raise RuntimeError('Apply optical geometry before installing gaze; do not silently invalidate its eye contract')
    if 'R11_adult_eye_proportion' not in scene: raise RuntimeError('Current 24.6 mm eye candidate required')
    binding = _api('face_fibre_binding_r11'); head = bpy.data.objects[HEAD]
    protected = binding['_protected'](head); before_binding = binding['validate'](neutral=False)
    original = {}; records = {}; materials = []; new_meshes = []; original_material_fake = {}
    material_contract = _api('runtime_face_refresh_r08')['_material_contract']
    try:
        for side in ('L', 'R'):
            globe = bpy.data.objects['R11 Adult sclera ' + side]; iris = bpy.data.objects['R11 Adult iris ' + side]
            if globe.parent != iris.parent or max(abs(globe.matrix_world[i][j] - iris.matrix_world[i][j]) for i in range(4) for j in range(4)) > 2e-6:
                raise RuntimeError('Globe and iris must share the same actual local frame')
            scale = float(globe.matrix_world.to_scale().x)
            if max(abs(v - scale) for v in globe.matrix_world.to_scale()) > 1e-7: raise RuntimeError('Uniform physical eye scale required')
            iris_before = _array(iris.data.vertices)
            new_co, plan = plan_iris_depth(_array(globe.data.vertices), iris_before, scale)
            for obj in (globe, iris):
                if len(obj.data.materials) != 1 or obj.data.shape_keys: raise RuntimeError('Expected single-material unkeyed eye attachment')
                old_material = obj.data.materials[0]
                original_material_fake.setdefault(old_material.name, old_material.use_fake_user)
                original[obj.name] = dict(data=obj.data, fake=obj.data.use_fake_user, geometry=_geometry(obj),
                    material=old_material, materialFake=original_material_fake[old_material.name],
                    materialContract=material_contract(obj.data.materials[0]))
                obj.data.use_fake_user = True; obj.data.materials[0].use_fake_user = True
                obj.data = obj.data.copy(); obj.data.name = 'R11 internal optics ' + obj.name
                new_meshes.append(obj.data)
            iris.data.vertices.foreach_set('co', new_co.ravel()); iris.data.update()
            radius = plan['physicalIrisRadiusMeters'] / scale
            sm = _sclera_material(original[globe.name]['material'], radius, scale)
            im = _iris_material(original[iris.name]['material']); materials.extend((sm, im))
            globe.data.materials[0] = sm; iris.data.materials[0] = im
            bpy.context.view_layer.update()
            if _geometry(globe) != original[globe.name]['geometry']: raise RuntimeError('External eyelid-contact globe geometry changed')
            iris_after = _array(iris.data.vertices)
            if not np.array_equal(iris_before[:, (0, 2)], iris_after[:, (0, 2)]): raise RuntimeError('Iris width/height changed')
            geometry = _geometry(iris); geometry['co'] = original[iris.name]['geometry']['co']
            if geometry != original[iris.name]['geometry']: raise RuntimeError('Iris topology, UV or transform changed')
            records[side] = dict(plan=plan, actualClearance=_check_internal_clearance(globe, iris),
                originalIrisCoordinateSha256=_ah(iris_before), newIrisCoordinateSha256=_ah(iris_after),
                scleraGeometryBefore=original[globe.name]['geometry'], scleraGeometryAfter=_geometry(globe),
                newMaterials=[sm.name, im.name])
        if binding['_protected'](head) != protected: raise RuntimeError('Optics operation changed head, eyelid keys or skin')
        after_binding = binding['validate'](neutral=False)
        for row in original.values():
            if material_contract(row['material']) != row['materialContract']: raise RuntimeError('Original eye material was modified')
        report = dict(source=str(source.relative_to(project_path())), sourceSha256=_sha(source), helperSha256=_sha(__file__),
            parameters=PARAMETERS, eyes=records, bindingBefore=before_binding, bindingAfter=after_binding,
            originals={name: dict(data=row['data'].name, fakeUser=row['fake'], material=row['material'].name,
                materialFakeUser=row['materialFake'], candidateData=bpy.data.objects[name].data.name,
                candidateGeometry=_geometry(bpy.data.objects[name]),
                candidateMaterial=bpy.data.objects[name].data.materials[0].name) for name, row in original.items()},
            headEyelidsLashesBrowsAndOuterEyeGeometryUnchanged=True, sourceSaved=False, appearanceAccepted=False,
            limitations=['Authored single refractive interface, not complete measured corneal thickness/aqueous/lens layers.',
                'Pupil remains the texture on the recessed iris; a geometric pupil opening/cavity is not installed.',
                'Caustic lighting may be costly/noisy; inspect actual current light closeups before adoption. No shadow-ray transparency bypass is added.',
                'No tear meniscus geometry or eyelid material changed. Existing wet-line approximation remains unfinished.',
                'Gaze must be installed on this source afterward; earlier gaze material/geometry contracts cannot be reused.'],
            shaderReferences=['https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/glass.html',
                'https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html'])
        scene[MARKER] = json.dumps(report, ensure_ascii=False)
        return report
    except Exception:
        for name, row in original.items():
            bpy.data.objects[name].data = row['data']; row['data'].use_fake_user = row['fake']; row['material'].use_fake_user = row['materialFake']
        for mesh in new_meshes:
            if mesh.users == 0: bpy.data.meshes.remove(mesh)
        for mat in materials:
            if mat.users == 0: bpy.data.materials.remove(mat)
        bpy.context.view_layer.update()
        raise


def restore():
    import bpy
    scene = bpy.context.scene; report = json.loads(scene[MARKER])
    if 'R11_gaze_control' in scene: raise RuntimeError('Restore/remove dependent gaze before restoring optical source')
    for name, old in report['originals'].items():
        if old['data'] not in bpy.data.meshes: raise RuntimeError('Original eye mesh missing')
        obj = bpy.data.objects[name]
        if obj.data.name != old['candidateData'] or _geometry(obj) != old['candidateGeometry'] or obj.data.materials[0].name != old['candidateMaterial']:
            raise RuntimeError('Later optical/transform work exists; inspect it before restoring: ' + name)
    for name, old in report['originals'].items():
        bpy.data.objects[name].data = bpy.data.meshes[old['data']]
        bpy.data.meshes[old['data']].use_fake_user = old['fakeUser']
        bpy.data.materials[old['material']].use_fake_user = old['materialFakeUser']
    del scene[MARKER]; bpy.context.view_layer.update()
    return _api('face_fibre_binding_r11')['validate'](neutral=False)


def prepare(source_relative, output_relative):
    import bpy
    source = (project_path() / source_relative).resolve(); output = (project_path() / output_relative).resolve()
    for path in (source, output): path.relative_to((project_path() / 'work/reconstruction-r02').resolve())
    report_path = output.with_suffix('.eye-depth.json')
    if source.suffix != '.blend' or output.suffix != '.blend' or output.exists() or report_path.exists():
        raise RuntimeError('Fresh independent output paths required')
    if bpy.data.is_dirty or Path(bpy.data.filepath).resolve() != source: raise RuntimeError('Open exact saved independent source')
    source_sha = _sha(source); report = apply()
    if _sha(source) != source_sha: raise RuntimeError('Source changed')
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True): raise RuntimeError('Save failed')
    report.update(output=str(output.relative_to(project_path())), outputSha256=_sha(output), outputSaved=True)
    with report_path.open('x', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    print('R11_EYE_DEPTH_SAVED ' + json.dumps(dict(output=str(output), sha256=report['outputSha256'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]); prepare(args.source, args.output)
