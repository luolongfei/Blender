"""Simple, exportable anime PBR materials for the NEW Hunyuan rebuild.

No scene/file is opened or saved. No rejected R11 geometry or skin maps are used.
RGB palette inputs are explicitly sRGB; Blender shader values are linear.
The optional iris bake is a new procedural material baked through Blender,
not an enlargement of the original atlas and not a claim of likeness approval.
"""
from pathlib import Path


PALETTE = {
    'skin': ((0.960, 0.805, 0.690), 0.72),
    'skin_back': ((0.960, 0.805, 0.690), 0.72),
    'ear_inner': ((0.875, 0.646, 0.539), 0.76),
    'sclera': ((0.964, 0.940, 0.886), 0.56),
    'iris': ((0.265, 0.248, 0.218), 0.60),
    'mouth': ((0.390, 0.236, 0.190), 0.78),
    'lip': ((0.805, 0.540, 0.445), 0.70),
    'lash': ((0.104, 0.097, 0.088), 0.76),
    'brow': ((0.205, 0.191, 0.169), 0.78),
    'hair': ((0.185, 0.176, 0.160), 0.53),
    'hair_shadow': ((0.145, 0.139, 0.130), 0.58),
    'hair_highlight': ((0.218, 0.205, 0.182), 0.56),
    'blouse': ((0.940, 0.890, 0.805), 0.88),
    'skirt': ((0.310, 0.350, 0.276), 0.88),
    'sock': ((0.935, 0.914, 0.861), 0.89),
    'shoe': ((0.255, 0.207, 0.166), 0.68),
}


def srgb_linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def linear_rgba(rgb):
    return tuple(srgb_linear(float(v)) for v in rgb[:3]) + (1.0,)


def principled_material(name, srgb, roughness=0.75):
    import bpy
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    color = linear_rgba(srgb)
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = roughness
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.0
    mat.diffuse_color = color
    mat['rebuild_style'] = 'anime; opaque simple PBR; no pores or baked lighting'
    return mat


def create_palette(prefix='Rebuilt_', iris_image_path=None):
    """Keys match face.py build_face materials and hair/hands parameters.

    Without iris_image_path the iris is only a flat base material; use the
    explicit bake below or another validated image before final eye approval.
    """
    mats = {key: principled_material(prefix + key, rgb, rough)
            for key, (rgb, rough) in PALETTE.items()}
    if iris_image_path:
        set_basecolor_image(mats['iris'], iris_image_path, uv_name='Iris_UV')
    else:
        mats['iris']['needs_iris_detail'] = True
    return mats


def set_basecolor_image(mat, path, uv_name=None, pack=True):
    """Connect an existing real image directly so glTF can export it."""
    import bpy
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    im = bpy.data.images.load(str(path), check_existing=True)
    im.colorspace_settings.name = 'sRGB'
    if pack:
        im.pack()
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    tex = nodes.new('ShaderNodeTexImage')
    tex.name = 'Rebuilt Base Color Image'
    tex.image = im
    tex.interpolation = 'Linear'
    tex.extension = 'EXTEND'
    if uv_name:
        uv = nodes.new('ShaderNodeUVMap')
        uv.uv_map = uv_name
        links.new(uv.outputs['UV'], tex.inputs['Vector'])
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Metallic'].default_value = 0.0
    mat['basecolor_source'] = str(path)
    return im


def match_flat_skin_to_baked_corner(material, image_path):
    """Match constant skin to the actual 8-bit sRGB bake, without recolouring.

    The corner of this module's face texture is guaranteed outside both cheeks.
    This removes only byte quantisation differences; it is not a fix for bad
    geometry, overlapping surfaces or split normals.
    """
    import bpy
    im = bpy.data.images.load(str(image_path), check_existing=True)
    im.colorspace_settings.name = 'sRGB'
    rgb = tuple(float(v) for v in im.pixels[:3])
    bsdf = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    if bsdf.inputs['Base Color'].is_linked:
        raise ValueError('Expected a constant skin material, not an existing texture network')
    bsdf.inputs['Base Color'].default_value = linear_rgba(rgb)
    material.diffuse_color = linear_rgba(rgb)
    material['matched_baked_srgb_corner'] = rgb
    return {'srgb': rgb, 'linear': linear_rgba(rgb)}


def copy_retained_atlas_material(original, name='Rebuilt_RetainedGarmentAtlas'):
    """Copy, preserve its image and UV links, and fix metallic default.

    This alone does NOT repaint contaminated texels. Do not report it as such.
    """
    mat = original.copy()
    mat.name = name
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs['Metallic'].default_value = 0.0
            node.inputs['Roughness'].default_value = 0.86
        elif node.type == 'TEX_IMAGE' and node.image:
            # This input has only a colour image, no data/normal image.
            node.image.colorspace_settings.name = 'sRGB'
            if not node.image.packed_file:
                node.image.pack()
    mat['rebuild_change'] = 'copied original UV/image; non-metallic PBR correction only'
    return mat


def _iris_nodes(mat):
    """A small UV material, intentionally no random noise or skin realism.

    Iris_UV: centre=(.5,.5), outer ellipse mapped to 0..1. The upper region
    is dark, lower iris softly warm grey, with a compact pupil and quiet glint.
    """
    n, l = mat.node_tree.nodes, mat.node_tree.links
    n.clear()
    uv = n.new('ShaderNodeTexCoord')
    def math_node(op, a, b=None):
        nd = n.new('ShaderNodeMath'); nd.operation = op
        for i, value in enumerate((a, b)):
            if value is None: continue
            if isinstance(value, (int, float)): nd.inputs[i].default_value = value
            else: l.new(value, nd.inputs[i])
        return nd.outputs[0]
    def distance(center):
        nd = n.new('ShaderNodeVectorMath'); nd.operation = 'DISTANCE'
        l.new(uv.outputs['UV'], nd.inputs[0]); nd.inputs[1].default_value = center
        return nd.outputs['Value']
    def mix(f, a, b):
        nd = n.new('ShaderNodeMixRGB'); nd.blend_type = 'MIX'
        l.new(f, nd.inputs[0])
        for i, value in ((1, a), (2, b)):
            if isinstance(value, tuple): nd.inputs[i].default_value = linear_rgba(value)
            else: l.new(value, nd.inputs[i])
        return nd.outputs[0]
    sep = n.new('ShaderNodeSeparateXYZ'); l.new(uv.outputs['UV'], sep.inputs[0])
    lower = math_node('SUBTRACT', 1, sep.outputs['Y'])
    iris = mix(lower, (0.155, 0.149, 0.138), (0.385, 0.360, 0.304))
    radius = distance((0.5, 0.5, 0))
    iris = mix(math_node('GREATER_THAN', radius, .465), iris, (0.108, .103, .095))
    iris = mix(math_node('LESS_THAN', radius, .185), iris, (0.070, .067, .061))
    iris = mix(math_node('LESS_THAN', distance((.36, .73, 0)), .060), iris, (.970, .957, .912))
    emit = n.new('ShaderNodeEmission'); l.new(iris, emit.inputs['Color'])
    emit.inputs['Strength'].default_value = 1
    out = n.new('ShaderNodeOutputMaterial'); l.new(emit.outputs[0], out.inputs['Surface'])
    return mat


def _bake_uv_image(output_path, size, builder, image_name):
    """Bake an authored UV material in an isolated transient scene.

    Invoke serially from Blender. Output is a NEW image, never original atlas.
    Restores active scene and removes transient geometry/material afterwards.
    Image contains actual new graphic detail, not an upscaled source.
    """
    import bpy
    if size not in (512, 1024, 2048):
        raise ValueError('Iris bake supports 512/1024/2048; 1K is normally ample')
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError('Refusing to overwrite iris checkpoint: ' + str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.data.scenes.new('TEMP_IrisMaterialBake')
    old_scene = bpy.context.window.scene if bpy.context.window else None
    mesh = bpy.data.meshes.new('TEMP_IrisBakeQuad')
    mesh.from_pydata([(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)], [], [(0,1,2,3)])
    layer = mesh.uv_layers.new(name='Iris_UV')
    for loop, uv in zip(layer.data, ((0,0),(1,0),(1,1),(0,1))): loop.uv = uv
    obj = bpy.data.objects.new('TEMP_IrisBakeQuad', mesh); scene.collection.objects.link(obj)
    mat = bpy.data.materials.new('TEMP_IrisProcedural'); mat.use_nodes = True
    builder(mat); mesh.materials.append(mat)
    im = bpy.data.images.new(image_name, width=size, height=size, alpha=False)
    im.colorspace_settings.name = 'sRGB'
    target = mat.node_tree.nodes.new('ShaderNodeTexImage'); target.image = im
    mat.node_tree.nodes.active = target
    try:
        scene.render.engine = 'CYCLES'; scene.cycles.samples = 1
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0; scene.view_settings.gamma = 1
        scene.render.bake.use_clear = True; scene.render.bake.margin = 0
        scene.view_layers[0].objects.active = obj
        obj.select_set(True, view_layer=scene.view_layers[0])
        if bpy.context.window: bpy.context.window.scene = scene
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0],
                                       active_object=obj, object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            bpy.ops.object.bake(type='EMIT')
        im.filepath_raw = str(output_path); im.file_format = 'PNG'; im.save(); im.pack()
        return im
    finally:
        if old_scene and bpy.context.window: bpy.context.window.scene = old_scene
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.materials.remove(mat)
        bpy.data.scenes.remove(scene)


def bake_iris_image(output_path, size=1024):
    return _bake_uv_image(output_path, size, _iris_nodes, 'Rebuilt_Iris_BaseColor')


def bake_face_skin_image(face_spec, output_path, size=2048, cheek_strength=.075):
    """Bake quiet anime skin with two faint warm cheek regions, no features.

    Uses actual face_spec front_uv and eye positions. No photo, pores, makeup
    noise, projection from hair, or original blurred face image is involved.
    Parent must visually validate this subtle colour choice after fitting.
    Back-of-head uses a separate flat material, never this frontal UV image.
    """
    box = face_spec['front_uv']
    width, height = box['right']-box['left'], box['top']-box['bottom']
    if min(width,height) <= 0:
        raise ValueError('Invalid front UV bounds')
    def builder(mat):
        n, l = mat.node_tree.nodes, mat.node_tree.links; n.clear()
        uv = n.new('ShaderNodeTexCoord')
        base = n.new('ShaderNodeRGB'); base.outputs[0].default_value = linear_rgba(PALETTE['skin'][0])
        col = base.outputs[0]
        for index, eye in enumerate(face_spec['eyes']):
            cheek = face_spec.get('cheek', {})
            x = cheek.get('centres_x', [e['x']*1.18 for e in face_spec['eyes']])[index]
            z = cheek.get('z', eye['z'] - max(eye['lower']*2.1, .015))
            center = ((x-box['left'])/width, (z-box['bottom'])/height, 0)
            sub = n.new('ShaderNodeVectorMath'); sub.operation='SUBTRACT'
            l.new(uv.outputs['UV'], sub.inputs[0]); sub.inputs[1].default_value=center
            scale = n.new('ShaderNodeVectorMath'); scale.operation='DIVIDE'
            l.new(sub.outputs['Vector'], scale.inputs[0])
            scale.inputs[1].default_value=(cheek.get('width', max(eye['half_width'],.020))/width,
                                           cheek.get('height', max(eye['lower']*1.3,.012))/height,1)
            length=n.new('ShaderNodeVectorMath');length.operation='LENGTH';l.new(scale.outputs['Vector'],length.inputs[0])
            sq=n.new('ShaderNodeMath');sq.operation='MULTIPLY';l.new(length.outputs['Value'],sq.inputs[0]);l.new(length.outputs['Value'],sq.inputs[1])
            inv=n.new('ShaderNodeMath');inv.operation='SUBTRACT';inv.inputs[0].default_value=1;l.new(sq.outputs[0],inv.inputs[1]);inv.use_clamp=True
            power=n.new('ShaderNodeMath');power.operation='MULTIPLY';l.new(inv.outputs[0],power.inputs[0]);l.new(inv.outputs[0],power.inputs[1])
            weight=n.new('ShaderNodeMath');weight.operation='MULTIPLY';l.new(power.outputs[0],weight.inputs[0]);weight.inputs[1].default_value=cheek_strength
            mix=n.new('ShaderNodeMixRGB');l.new(weight.outputs[0],mix.inputs[0]);l.new(col,mix.inputs[1]);mix.inputs[2].default_value=linear_rgba((.91,.61,.50));col=mix.outputs[0]
        emit=n.new('ShaderNodeEmission');l.new(col,emit.inputs['Color'])
        out=n.new('ShaderNodeOutputMaterial');l.new(emit.outputs[0],out.inputs['Surface'])
    return _bake_uv_image(output_path,size,builder,'Rebuilt_FaceSkin_BaseColor')


def material_export_audit(objects):
    """Report sources; no export-success or visual-acceptance inference."""
    result = []
    seen = set()
    for obj in objects:
        if obj.type != 'MESH': continue
        for mat in obj.data.materials:
            if not mat or mat.as_pointer() in seen: continue
            seen.add(mat.as_pointer())
            bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            images = [n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
            result.append({'name': mat.name, 'principled': bool(bsdf),
                           'metallic': bsdf.inputs['Metallic'].default_value if bsdf else None,
                           'images': [{'name': im.name, 'size': list(im.size),
                                       'packed': bool(im.packed_file),
                                       'colorSpace': im.colorspace_settings.name} for im in images]})
    return result


def build_geometry_masked_material(original, regions, name='Rebuilt_BodyRepair_BakeSource'):
    """Local colour repair driven by CONFIRMED world-space surface masks.

    regions: explicit dictionaries with name, center, half_extents, color_srgb,
    and optional feather, strength, normal_direction, normal_min,
    dark_threshold. The caller determines bounds from real diagnostics.
    This shader must be baked before glTF export. It never changes original.
    """
    mat = original.copy(); mat.name = name
    n, l = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(nd for nd in n if nd.type == 'BSDF_PRINCIPLED')
    base = bsdf.inputs['Base Color']
    if base.is_linked:
        original_color = base.links[0].from_socket
    else:
        rgb = n.new('ShaderNodeRGB'); rgb.outputs[0].default_value = base.default_value
        original_color = rgb.outputs[0]
    geo = n.new('ShaderNodeNewGeometry')
    def calc(op, a, b=None):
        nd = n.new('ShaderNodeMath'); nd.operation = op
        for i, val in enumerate((a, b)):
            if val is None: continue
            if isinstance(val, (int, float)): nd.inputs[i].default_value = val
            else: l.new(val, nd.inputs[i])
        return nd.outputs[0]
    def clamp(v):
        return calc('MINIMUM', calc('MAXIMUM', v, 0), 1)
    def smooth(v):
        x = clamp(v)
        return calc('MULTIPLY', calc('MULTIPLY', x, x), calc('SUBTRACT', 3, calc('MULTIPLY', 2, x)))
    separate = n.new('ShaderNodeSeparateXYZ'); l.new(geo.outputs['Position'], separate.inputs[0])
    color = original_color
    for region in regions:
        center, half = region['center'], region['half_extents']
        feather = float(region.get('feather', min(half)*.15))
        if min(half) <= 0 or feather <= 0:
            raise ValueError('Positive mask extents/feather required: ' + region['name'])
        terms = []
        for axis in range(3):
            distance = calc('ABSOLUTE', calc('SUBTRACT', separate.outputs[axis], center[axis]))
            terms.append(smooth(calc('DIVIDE', calc('SUBTRACT', half[axis], distance), feather)))
        mask = calc('MULTIPLY', calc('MULTIPLY', terms[0], terms[1]), terms[2])
        if 'normal_direction' in region:
            dot = n.new('ShaderNodeVectorMath'); dot.operation = 'DOT_PRODUCT'
            l.new(geo.outputs['Normal'], dot.inputs[0])
            dot.inputs[1].default_value = region['normal_direction']
            threshold = region.get('normal_min', .05)
            normal_mask = smooth(calc('DIVIDE', calc('SUBTRACT', dot.outputs['Value'], threshold), .25))
            mask = calc('MULTIPLY', mask, normal_mask)
        if 'dark_threshold' in region:
            lum = n.new('ShaderNodeRGBToBW'); l.new(original_color, lum.inputs[0])
            # Threshold is LINEAR luminance. Only use after real source inspection.
            dark = smooth(calc('DIVIDE', calc('SUBTRACT', region['dark_threshold'], lum.outputs[0]), .05))
            mask = calc('MULTIPLY', mask, dark)
        if 'bright_threshold' in region:
            lum = n.new('ShaderNodeRGBToBW'); l.new(original_color, lum.inputs[0])
            bright = smooth(calc('DIVIDE', calc('SUBTRACT', lum.outputs[0], region['bright_threshold']), .08))
            mask = calc('MULTIPLY', mask, bright)
        mask = calc('MULTIPLY', mask, float(region.get('strength', 1)))
        mix = n.new('ShaderNodeMixRGB'); mix.name = 'Confirmed repair: ' + region['name']
        l.new(mask, mix.inputs[0]); l.new(color, mix.inputs[1])
        mix.inputs[2].default_value = linear_rgba(region['color_srgb'])
        color = mix.outputs[0]
    # Emission bakes solid base colour only, with no directional lights/shadows.
    emit = n.new('ShaderNodeEmission'); l.new(color, emit.inputs['Color'])
    output = next(nd for nd in n if nd.type == 'OUTPUT_MATERIAL')
    l.new(emit.outputs[0], output.inputs['Surface'])
    mat['requires_bake_for_glb'] = True
    mat['repair_regions'] = [r['name'] for r in regions]
    return mat


def bake_repaired_body_atlas(source_object, regions, output_path, size=4096,
                             repack_uv=False, original_uv_name=None):
    """Bake only a duplicated retained mesh and confirmed local colour masks.

    All baking happens on a copy. With repack_uv=False source stays untouched.
    With repack_uv=True, old UVs are explicitly pinned to image nodes and a new
    UV layer is generated on the copy; only after a successful bake is that new
    layer committed to source_object. Old UV layers and original image remain.
    Returns a NEW direct-image Principled material suitable for GLB. Root
    decides whether to assign it after same-camera neutral-light inspection.
    """
    import bpy
    if source_object.type != 'MESH' or not source_object.data.materials:
        raise ValueError('Expected retained mesh with valid material slots')
    if any(mat is None for mat in source_object.data.materials):
        raise ValueError('Empty material slots must be resolved before baking')
    if not source_object.data.uv_layers.active:
        raise ValueError('Original UV layer required; no auto-unwrapping here')
    if size not in (2048,4096):
        raise ValueError('Retained atlas bake supports 2K or 4K')
    path = Path(output_path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite prior bake: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = bpy.data.scenes.new('TEMP_ConfirmedBodyColourBake')
    old_scene = bpy.context.window.scene if bpy.context.window else None
    obj = source_object.copy(); obj.data = source_object.data.copy()
    obj.name = 'TEMP_RetainedBodyBake'; obj.animation_data_clear()
    scene.collection.objects.link(obj)
    obj.matrix_world = source_object.matrix_world.copy()
    obj.hide_render = False; obj.hide_viewport = False
    old_uv_name = original_uv_name or source_object.data.uv_layers.active.name
    if old_uv_name not in obj.data.uv_layers:
        raise ValueError('Original UV layer does not exist: ' + old_uv_name)
    # Explicit old-UV lookup is essential: otherwise changing active UV would
    # resample the source atlas in the NEW UV positions and scramble clothing.
    baked_materials = []
    original_face_materials = [p.material_index for p in obj.data.polygons]
    for index, original in enumerate(source_object.data.materials):
        has_atlas = any(n.type == 'TEX_IMAGE' and n.image for n in original.node_tree.nodes)
        # New repair faces have their own solid material. Their placeholder UV
        # (.5,.5) must NEVER cause them to sample an unrelated old atlas patch.
        per_material_regions = regions if has_atlas else []
        mat = build_geometry_masked_material(original, per_material_regions,
                                             name=f'Rebuilt_BodyBakeSource_{index}_{original.name}')
        original_uv = mat.node_tree.nodes.new('ShaderNodeUVMap')
        original_uv.name = 'Pinned original atlas UV'; original_uv.uv_map = old_uv_name
        for nd in mat.node_tree.nodes:
            if nd.type == 'TEX_IMAGE' and nd.image:
                mat.node_tree.links.new(original_uv.outputs['UV'], nd.inputs['Vector'])
        baked_materials.append(mat)
    obj.data.materials.clear()
    for mat in baked_materials: obj.data.materials.append(mat)
    for polygon, slot in zip(obj.data.polygons, original_face_materials): polygon.material_index = slot
    bake_uv_name = old_uv_name
    im = bpy.data.images.new('Rebuilt_RetainedBody_BaseColor', width=size, height=size, alpha=False)
    im.colorspace_settings.name = 'sRGB'
    for mat in baked_materials:
        target = mat.node_tree.nodes.new('ShaderNodeTexImage'); target.image = im
        mat.node_tree.nodes.active = target
    try:
        scene.render.engine = 'CYCLES'; scene.cycles.samples = 1
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0; scene.view_settings.gamma = 1
        scene.render.bake.use_clear = True; scene.render.bake.margin = 12
        scene.render.bake.use_selected_to_active = False
        scene.view_layers[0].objects.active = obj
        obj.select_set(True, view_layer=scene.view_layers[0])
        if bpy.context.window: bpy.context.window.scene = scene
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0],
                                       active_object=obj, object=obj,
                                       selected_objects=[obj], selected_editable_objects=[obj]):
            if repack_uv:
                layer = obj.data.uv_layers.new(name='Rebuilt_BodyUV')
                bake_uv_name = layer.name
                obj.data.uv_layers.active = layer
                layer.active_render = True
                bpy.ops.object.mode_set(mode='EDIT')
                try:
                    bpy.ops.mesh.select_all(action='SELECT')
                    # A true new per-part UV layout, not texture interpolation.
                    bpy.ops.uv.smart_project(angle_limit=1.1519173063162575,
                                             island_margin=.004, area_weight=0.0,
                                             correct_aspect=True, scale_to_bounds=True)
                finally:
                    bpy.ops.object.mode_set(mode='OBJECT')
                obj.data.calc_loop_triangles()
                uv_data = obj.data.uv_layers[bake_uv_name].data
                zero = []
                for tri in obj.data.loop_triangles:
                    a,b,c = [uv_data[i].uv for i in tri.loops]
                    area = abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x))*.5
                    if area <= 1e-16:
                        zero.append(tri.polygon_index)
                if zero:
                    raise RuntimeError(f'New UV layout still has {len(zero)} degenerate triangles; original preserved')
            bpy.ops.object.bake(type='EMIT')
        im.filepath_raw = str(path); im.file_format = 'PNG'; im.save(); im.pack()
        result = principled_material('Rebuilt_RetainedBody_RepairedAtlas', (1,1,1), .86)
        set_basecolor_image(result, path, bake_uv_name)
        result['rebuild_change'] = 'confirmed surface-local colour repair and new UV bake; no source enlargement'
        result['repair_regions'] = [r['name'] for r in regions]
        result['baked_source_materials'] = [m.name for m in source_object.data.materials]
        if repack_uv:
            # Loop ordering and topology were not changed by smart UV project.
            if len(source_object.data.loops) != len(obj.data.loops):
                raise RuntimeError('Source topology changed during bake; refusing UV commit')
            if source_object.data.users > 1:
                source_object.data = source_object.data.copy()
            layer = source_object.data.uv_layers.new(name=bake_uv_name)
            if layer.name != bake_uv_name:
                raise RuntimeError('New UV name collision on source; refusing ambiguous material lookup')
            for dest, origin in zip(layer.data, obj.data.uv_layers[bake_uv_name].data):
                dest.uv = origin.uv
            source_object.data.uv_layers.active = layer
            layer.active_render = True
            result['original_uv_preserved'] = old_uv_name
            result['repacked_uv'] = bake_uv_name
            result['new_uv_degenerate_triangles'] = 0
        return result
    finally:
        if old_scene and bpy.context.window: bpy.context.window.scene = old_scene
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        for mat in baked_materials: bpy.data.materials.remove(mat)
        bpy.data.scenes.remove(scene)
