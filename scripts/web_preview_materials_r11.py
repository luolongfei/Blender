"""Explicit, lossy PBR adaptation for a disposable R11 web-preview export.

No source material, object or mesh is edited. Call adapt_mesh_materials on an
independent evaluated mesh, while its coordinates are still source-object local.
Shader colour fields are evaluated at mesh corners; procedural bump, scattering
and per-point roughness are not claimed to survive. Requires Blender + NumPy.
"""
import numpy as np

COLOR_ATTRIBUTE = 'WEB_PREVIEW_COLOR'


def _record(report, row):
    if isinstance(report, list):
        report.append(row)
    else:
        report.setdefault('materials', []).append(row)


def _array(items, field, width, dtype=np.float32):
    a = np.empty((len(items), width), dtype=dtype)
    items.foreach_get(field, a.ravel())
    return a


def _constant(value, count):
    a = np.asarray(value, dtype=np.float64).reshape(1, -1)
    return np.broadcast_to(a, (count, a.shape[1]))


def _scalar(a):
    return a if a.shape[1] == 1 else a[:, :3].mean(axis=1, keepdims=True)


def _color(a):
    if a.shape[1] == 4:
        return a
    if a.shape[1] == 1:
        a = np.repeat(a, 3, axis=1)
    return np.concatenate((a[:, :3], np.ones((len(a), 1))), axis=1)


class FieldEvaluator:
    def __init__(self, mesh, source_obj):
        self.mesh, self.source_obj = mesh, source_obj
        self.indices = _array(mesh.loops, 'vertex_index', 1, np.int32).ravel()
        self.local = _array(mesh.vertices, 'co', 3)[self.indices].astype(np.float64)
        self.n = len(self.indices)
        self.cache = {}
        self.notes = set()

    def attribute(self, name):
        a = self.mesh.attributes.get(name)
        if a is None:
            # Blender missing shader attributes resolve to zero.
            self.notes.add('Missing shader attribute resolved to zero: ' + name)
            return np.zeros((self.n, 1))
        fields = {'FLOAT': ('value', 1), 'INT': ('value', 1),
                  'BOOLEAN': ('value', 1), 'FLOAT_COLOR': ('color', 4),
                  'BYTE_COLOR': ('color', 4), 'FLOAT_VECTOR': ('vector', 3)}
        if a.data_type not in fields:
            raise ValueError('Unsupported shader attribute type: ' + name + ' ' + a.data_type)
        field, width = fields[a.data_type]
        values = _array(a.data, field, width)
        if a.domain == 'POINT':
            return values[self.indices]
        if a.domain == 'CORNER':
            return values
        if a.domain == 'FACE':
            counts = _array(self.mesh.polygons, 'loop_total', 1, np.int32).ravel()
            return np.repeat(values, counts, axis=0)
        raise ValueError('Unsupported shader attribute domain: ' + a.domain)

    def input(self, socket):
        if socket.is_linked:
            if len(socket.links) != 1:
                raise ValueError('Ambiguous linked shader input')
            return self.output(socket.links[0].from_socket)
        return _constant(socket.default_value, self.n)

    def output(self, socket):
        key = socket.as_pointer()
        if key in self.cache:
            return self.cache[key]
        node = socket.node
        kind = node.type
        get = lambda i: self.input(node.inputs[i])
        if kind == 'REROUTE':
            out = get(0)
        elif kind == 'VALUE':
            out = _constant(socket.default_value, self.n)
        elif kind == 'RGB':
            out = _constant(socket.default_value, self.n)
        elif kind in {'VERTEX_COLOR', 'ATTRIBUTE'}:
            out = self.attribute(node.layer_name if kind == 'VERTEX_COLOR' else node.attribute_name)
            if socket.name == 'Alpha':
                out = _color(out)[:, 3:4]
            elif socket.name == 'Fac':
                out = _scalar(out)
            elif socket.name == 'Color':
                out = _color(out)
        elif kind == 'MIX_RGB':
            f = np.clip(_scalar(get(0)), 0, 1)
            a, b = _color(get(1)), _color(get(2))
            mode = node.blend_type
            if mode == 'MIX':
                out = a * (1-f) + b * f
            elif mode == 'MULTIPLY':
                out = a * (1-f) + a * b * f
            elif mode == 'ADD':
                out = a + b * f
            elif mode == 'SUBTRACT':
                out = a - b * f
            else:
                raise ValueError('Unsupported base-colour blend: ' + mode)
            out[:, 3] = a[:, 3]
            if node.use_clamp:
                out = np.clip(out, 0, 1)
        elif kind == 'MATH':
            a = _scalar(get(0)); op = node.operation
            b = _scalar(get(1)) if len(node.inputs) > 1 else a
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                if op == 'ADD': out = a+b
                elif op == 'SUBTRACT': out = a-b
                elif op == 'MULTIPLY': out = a*b
                elif op == 'DIVIDE': out = np.divide(a, b, out=np.zeros_like(a), where=abs(b)>1e-30)
                elif op == 'POWER': out = np.power(a, b)
                elif op == 'SQRT': out = np.sqrt(np.maximum(a, 0))
                elif op == 'ABSOLUTE': out = abs(a)
                elif op == 'MINIMUM': out = np.minimum(a, b)
                elif op == 'MAXIMUM': out = np.maximum(a, b)
                elif op == 'LESS_THAN': out = (a < b).astype(float)
                elif op == 'GREATER_THAN': out = (a > b).astype(float)
                elif op == 'MULTIPLY_ADD': out = a*b+_scalar(get(2))
                elif op == 'CLAMP': out = np.clip(a, 0, 1)
                else: raise ValueError('Unsupported scalar operation: ' + op)
            if node.use_clamp:
                out = np.clip(out, 0, 1)
        elif kind == 'MAP_RANGE':
            if node.data_type != 'FLOAT':
                raise ValueError('Only scalar map range supported')
            value, lo, hi, target_lo, target_hi = [_scalar(get(i)) for i in range(5)]
            t = np.divide(value-lo, hi-lo, out=np.zeros_like(value), where=abs(hi-lo)>1e-30)
            mode = node.interpolation_type
            if mode in {'SMOOTHSTEP', 'SMOOTHERSTEP'} or node.clamp:
                t = np.clip(t, 0, 1)
            if mode == 'SMOOTHSTEP': t = t*t*(3-2*t)
            elif mode == 'SMOOTHERSTEP': t = t*t*t*(t*(t*6-15)+10)
            elif mode != 'LINEAR': raise ValueError('Unsupported map-range interpolation: ' + mode)
            out = target_lo + (target_hi-target_lo)*t
        elif kind == 'TEX_COORD':
            if socket.name == 'Object':
                out = self.local
                if node.object is not None and node.object != self.source_obj:
                    transform = np.asarray(node.object.matrix_world.inverted() @ self.source_obj.matrix_world)
                    out = out @ transform[:3, :3].T + transform[:3, 3]
            else:
                raise ValueError('Unsupported shader coordinate: ' + socket.name)
        elif kind == 'VECT_MATH':
            a = get(0)[:, :3]; b = get(1)[:, :3]; op = node.operation
            if op == 'SUBTRACT': out = a-b
            elif op == 'ADD': out = a+b
            elif op == 'MULTIPLY': out = a*b
            elif op == 'DOT_PRODUCT': out = np.sum(a*b, axis=1, keepdims=True)
            elif op == 'LENGTH': out = np.linalg.norm(a, axis=1, keepdims=True)
            elif op == 'SCALE': out = a*_scalar(self.input(node.inputs['Scale']))
            elif op == 'NORMALIZE': out = a/np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30)
            else: raise ValueError('Unsupported vector operation: ' + op)
        elif kind == 'SEPXYZ':
            out = get(0)[:, 'XYZ'.index(socket.name):'XYZ'.index(socket.name)+1]
        elif kind == 'COMBXYZ':
            out = np.concatenate([_scalar(get(i)) for i in range(3)], axis=1)
        else:
            raise ValueError('Unsupported shader field node: ' + kind + ' / ' + node.name)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite shader result: ' + node.name)
        self.cache[key] = out
        return out

    def surface(self, socket):
        node = socket.node
        if node.type == 'REROUTE':
            return self.surface(node.inputs[0].links[0].from_socket)
        if node.type == 'BSDF_PRINCIPLED':
            return {'color': _color(self.input(node.inputs['Base Color'])).copy(),
                    'alpha': _scalar(self.input(node.inputs['Alpha'])),
                    'roughness': _scalar(self.input(node.inputs['Roughness'])),
                    'metallic': _scalar(self.input(node.inputs['Metallic'])),
                    'shader': node}
        if node.type == 'MIX_SHADER':
            f = np.clip(_scalar(self.input(node.inputs[0])), 0, 1)
            a = self.surface(node.inputs[1].links[0].from_socket)
            second = node.inputs[2].links[0].from_node
            if second.type == 'BSDF_GLASS':
                self.notes.add('Anterior corneal refraction replaced by the original shader-window vertex alpha; no refractive optics equivalence.')
                a['alpha'] = a['alpha']*(1-f)
                a['glassWindow'] = True
                return a
            b = self.surface(node.inputs[2].links[0].from_socket)
            result = {k:a[k]*(1-f)+b[k]*f for k in ('color','alpha','roughness','metallic')}
            result['shader'] = a['shader']
            self.notes.add('MixShader PBR parameter interpolation approximates the original BSDF mixture.')
            return result
        raise ValueError('Unsupported source surface: ' + node.type)


def _surface_socket(material):
    outputs = [n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
    if len(outputs) != 1 or len(outputs[0].inputs['Surface'].links) != 1:
        raise ValueError('Ambiguous active material output: ' + material.name)
    return outputs[0].inputs['Surface'].links[0].from_socket


def _image_factor(socket):
    """Recognize an image or a constant linear multiply around one image."""
    if not socket.is_linked:
        return None
    node = socket.links[0].from_node
    if node.type == 'TEX_IMAGE':
        return node, np.ones(4)
    if node.type == 'REROUTE':
        return _image_factor(node.inputs[0])
    if node.type == 'MIX_RGB' and node.blend_type == 'MULTIPLY' and not node.inputs[0].is_linked:
        f = float(node.inputs[0].default_value)
        for image_index, tint_index in ((1,2),(2,1)):
            nested = _image_factor(node.inputs[image_index])
            if nested and not node.inputs[tint_index].is_linked:
                tint = np.asarray(node.inputs[tint_index].default_value, float)
                if image_index == 1 or abs(f-1) < 1e-8:
                    return nested[0], nested[1]*((1-f)+f*tint)
    return None


def _copy_image(nt, source, mesh):
    image = nt.nodes.new('ShaderNodeTexImage'); image.image = source.image
    image.interpolation, image.extension = source.interpolation, source.extension
    if source.image is None:
        raise ValueError('Source image missing: ' + source.name)
    vector = source.inputs['Vector']
    if vector.is_linked:
        upstream = vector.links[0].from_node
        if upstream.type != 'UVMAP':
            raise ValueError('Unsupported image coordinate graph: ' + upstream.type)
        uv_name = upstream.uv_map
    else:
        active = mesh.uv_layers.active
        uv_name = active.name if active else ''
    if not uv_name or mesh.uv_layers.get(uv_name) is None:
        raise ValueError('Image UV missing: ' + uv_name)
    uv = nt.nodes.new('ShaderNodeUVMap'); uv.uv_map = uv_name
    nt.links.new(uv.outputs['UV'], image.inputs['Vector'])
    return image


def _pbr(name, color=(1,1,1,1), roughness=.5, metallic=0):
    import bpy
    material = bpy.data.materials.new('WEB PREVIEW / ' + name)
    material.use_nodes = True
    shader = next(n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Metallic'].default_value = metallic
    material.diffuse_color = color
    material.roughness = roughness
    material.metallic = metallic
    return material, shader


def adapt_mesh_materials(mesh, source_obj, report):
    """Adapt only the independent evaluated mesh; returns newly-created mats."""
    if mesh == source_obj.data:
        raise ValueError('Refuse adapting the source mesh')
    evaluator = FieldEvaluator(mesh, source_obj)
    n = len(mesh.loops)
    colors = np.ones((n, 4), dtype=np.float32)
    slots = list(mesh.materials)
    for index, slot in enumerate(source_obj.material_slots):
        if index < len(slots) and slot.link == 'OBJECT':
            slots[index] = slot.material
    counts = _array(mesh.polygons, 'loop_total', 1, np.int32).ravel()
    poly_material = _array(mesh.polygons, 'material_index', 1, np.int32).ravel()
    loop_material = np.repeat(poly_material, counts)
    created = []
    for index, source in enumerate(slots):
        selected = loop_material == index
        if not selected.any():
            continue
        if source is None or not source.use_nodes:
            raise ValueError('Expected node material: ' + source_obj.name)
        socket = _surface_socket(source)
        shader = socket.node if socket.node.type == 'BSDF_PRINCIPLED' else None
        texture = _image_factor(shader.inputs['Base Color']) if shader else None
        evaluator.cache.clear(); evaluator.notes.clear()
        if texture:
            source_image, factor = texture
            surface = {'color':_constant(factor,n),
                       'alpha':_scalar(evaluator.input(shader.inputs['Alpha'])),
                       'roughness':_scalar(evaluator.input(shader.inputs['Roughness'])),
                       'metallic':_scalar(evaluator.input(shader.inputs['Metallic'])), 'shader':shader}
        else:
            surface = evaluator.surface(socket)
        material_colors = _color(surface['color']).copy()
        material_colors[:,3] *= surface['alpha'][:,0]
        if material_colors[selected,:3].min() < -1e-6 or material_colors[selected,:3].max() > 1.00001:
            raise ValueError('Base colour outside normalized reflectance: ' + source.name)
        colors[selected] = np.clip(material_colors[selected], 0, 1)
        roughness = float(np.mean(surface['roughness'][selected]))
        metallic = float(np.mean(surface['metallic'][selected]))
        material, target = _pbr(source.name, roughness=float(np.clip(roughness,.015,1)), metallic=float(np.clip(metallic,0,1)))
        nt = material.node_tree
        color = nt.nodes.new('ShaderNodeVertexColor'); color.layer_name = COLOR_ATTRIBUTE
        if texture:
            image = _copy_image(nt, source_image, mesh)
            mix = nt.nodes.new('ShaderNodeMixRGB'); mix.blend_type='MULTIPLY'; mix.inputs[0].default_value=1.
            nt.links.new(color.outputs['Color'], mix.inputs[1]); nt.links.new(image.outputs['Color'],mix.inputs[2])
            nt.links.new(mix.outputs[0],target.inputs['Base Color'])
        else:
            nt.links.new(color.outputs['Color'], target.inputs['Base Color'])
        shader = surface['shader']
        for prop in ('IOR','Specular IOR Level','Coat Weight','Coat Roughness','Coat IOR'):
            src = shader.inputs.get(prop); dst=target.inputs.get(prop)
            if src and dst and not src.is_linked:
                dst.default_value = src.default_value
        normal = shader.inputs.get('Normal')
        if normal and normal.is_linked:
            original_normal = normal.links[0].from_node
            if original_normal.type=='NORMAL_MAP' and original_normal.inputs['Color'].is_linked and original_normal.inputs['Color'].links[0].from_node.type=='TEX_IMAGE':
                tex = _copy_image(nt,original_normal.inputs['Color'].links[0].from_node,mesh)
                normal_map=nt.nodes.new('ShaderNodeNormalMap');normal_map.space=original_normal.space;normal_map.uv_map=original_normal.uv_map
                normal_map.inputs['Strength'].default_value=original_normal.inputs['Strength'].default_value
                nt.links.new(tex.outputs['Color'],normal_map.inputs['Color']);nt.links.new(normal_map.outputs['Normal'],target.inputs['Normal'])
            else:
                evaluator.notes.add('Procedural micro-normal/bump not exported.')
        transparent = bool(np.any(colors[selected,3] < .99999))
        if transparent:
            nt.links.new(color.outputs['Alpha'],target.inputs['Alpha'])
            if hasattr(material,'surface_render_method'):
                enums={x.identifier for x in material.bl_rna.properties['surface_render_method'].enum_items}
                if 'DITHERED' in enums:material.surface_render_method='DITHERED'
            elif hasattr(material,'blend_method'):
                enums={x.identifier for x in material.bl_rna.properties['blend_method'].enum_items}
                if 'BLEND' in enums:material.blend_method='BLEND'
            # Do not expose the back wall of the globe through the anterior window.
            material.use_backface_culling = True
        else:
            material.use_backface_culling = bool(source.use_backface_culling)
        ss = shader.inputs.get('Subsurface Weight')
        if ss and (ss.is_linked or float(ss.default_value)>0):
            evaluator.notes.add('Subsurface scattering not reproduced by portable PBR.')
        if np.ptp(surface['roughness'][selected]) > 1e-6:
            evaluator.notes.add('Spatial roughness reduced to the mean on this material region.')
        mesh.materials[index] = material
        material['web_preview_source_material'] = source.name
        material['web_preview_only'] = True
        created.append(material)
        _record(report,{'object':source_obj.name,'source':source.name,'output':material.name,
                        'baseColor':'original image times corner factor' if texture else 'CPU-evaluated shader colour at corners',
                        'linearRGBMin':colors[selected,:3].min(0).tolist(),'linearRGBMax':colors[selected,:3].max(0).tolist(),
                        'alphaRange':[float(colors[selected,3].min()),float(colors[selected,3].max())],
                        'roughnessMean':roughness,'metallicMean':metallic,'limitations':sorted(evaluator.notes)})
    old=mesh.color_attributes.get(COLOR_ATTRIBUTE)
    if old:mesh.color_attributes.remove(old)
    attr=mesh.color_attributes.new(name=COLOR_ATTRIBUTE,type='FLOAT_COLOR',domain='CORNER')
    attr.data.foreach_set('color',colors.ravel())
    mesh.color_attributes.active_color_index=list(mesh.color_attributes).index(attr)
    if hasattr(mesh.color_attributes,'render_color_index'):
        mesh.color_attributes.render_color_index=list(mesh.color_attributes).index(attr)
    return created


def make_curve_material(source_material, report):
    """Return an explicitly approximate portable material for meshed fibres."""
    if source_material is None or not source_material.use_nodes:
        raise ValueError('Expected native fibre node material')
    nodes=source_material.node_tree.nodes
    shader=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
    notes=['Native strand scattering/anisotropy and per-strand random colour approximated by portable PBR.']
    roughness=.62;metallic=0.
    if shader:
        socket=shader.inputs['Base Color']
        if not socket.is_linked:
            color=tuple(socket.default_value)
        elif socket.links[0].from_node.type=='VALTORGB':
            ramp=socket.links[0].from_node.color_ramp
            color=tuple(ramp.evaluate(.5))
            notes.append('Brow random colour ramp represented at its midpoint.')
        else:
            raise ValueError('Unsupported fibre base colour: '+socket.links[0].from_node.type)
        if not shader.inputs['Roughness'].is_linked:roughness=float(shader.inputs['Roughness'].default_value)
        if not shader.inputs['Metallic'].is_linked:metallic=float(shader.inputs['Metallic'].default_value)
    else:
        groups=[n for n in nodes if n.type=='GROUP' and n.inputs.get('Color 1') and n.inputs.get('Color 2')]
        if len(groups)!=1:
            raise ValueError('Unrecognized author fibre shader: '+source_material.name)
        group=groups[0]
        if group.inputs['Color 1'].is_linked or group.inputs['Color 2'].is_linked:
            raise ValueError('Unexpected linked author fibre colours')
        color=tuple((np.asarray(group.inputs['Color 1'].default_value)+np.asarray(group.inputs['Color 2'].default_value))*.5)
        notes.append('Author Color 1 / Color 2 midpoint; spatial root darkening omitted.')
    material,target=_pbr(source_material.name,color=color,roughness=roughness,metallic=metallic)
    target.inputs['Specular IOR Level'].default_value=.25
    material.use_backface_culling=False
    material['web_preview_source_material']=source_material.name;material['web_preview_only']=True
    _record(report,{'source':source_material.name,'output':material.name,'linearRGB':list(color),
                    'roughness':roughness,'limitations':notes})
    return material
