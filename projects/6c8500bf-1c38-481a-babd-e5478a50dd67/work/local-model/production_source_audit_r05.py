"""Read-only inventory of an opened, saved, independent Blender production source.

Run report(output_relative='work/reconstruction-r02/source-audit-r05.json') from
the integrator's Blender session. This module never opens/saves a blend, changes
frames, evaluates physics, packs resources, renders, or declares delivery passed.
Its optional JSON write is the only mutation. No automatic run on module import.
"""
import array
import collections
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'toolchain.lock.json').is_file())
sys.path.insert(0, str(ROOT))
from pipeline.context import project_path

VERSION = 1
DIRECTORY_FILE_LIMIT = 20000
MATERIAL_OBJECT_TYPES = {'MESH', 'CURVES', 'CURVE', 'SURFACE', 'FONT', 'META', 'VOLUME'}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _relative(path, root):
    return path.resolve().relative_to(root.resolve()).as_posix() if _inside(path, root) else None


def _packed(id_block):
    files = getattr(id_block, 'packed_files', None)
    return bool(getattr(id_block, 'packed_file', None)) or bool(files and len(files))


def _path_info(raw, project, library=None, packed=False):
    raw = str(raw or '')
    result = {'storedPath': raw, 'packed': bool(packed), 'resolvedPath': None,
              'projectRelativePath': None, 'exists': None, 'withinProject': None}
    if not raw or raw == '<builtin>':
        result['pathKind'] = 'builtin' if raw == '<builtin>' else 'empty'
        return result
    try:
        path = Path(bpy.path.abspath(raw, library=library)).resolve()
        result.update(resolvedPath=str(path), projectRelativePath=_relative(path, project),
                      exists=path.exists(), withinProject=_inside(path, project),
                      pathKind='blender-relative' if raw.startswith('//') else 'other')
    except (OSError, ValueError, RuntimeError) as exc:
        result['resolutionError'] = str(exc)
    return result


def _directory_inventory(path_info):
    """Count flat point-cache files; do not read simulation payloads or recurse."""
    result = {'directoryExists': False, 'directFileCount': 0, 'directBytes': 0,
              'bphysFileCount': 0, 'sampleNames': [], 'nestedDirectories': [],
              'scanTruncated': False, 'scanLimit': DIRECTORY_FILE_LIMIT,
              'frameNumbersParsedFromNames': None}
    if not path_info.get('resolvedPath'):
        return result
    path = Path(path_info['resolvedPath'])
    if not path.is_dir():
        return result
    result['directoryExists'] = True
    frames = set()
    try:
        with os.scandir(path) as entries:
            for i, entry in enumerate(entries):
                if i >= DIRECTORY_FILE_LIMIT:
                    result['scanTruncated'] = True
                    break
                if entry.is_dir(follow_symlinks=False):
                    result['nestedDirectories'].append(entry.name)
                elif entry.is_file(follow_symlinks=True):
                    result['directFileCount'] += 1
                    result['directBytes'] += entry.stat().st_size
                    if len(result['sampleNames']) < 12:
                        result['sampleNames'].append(entry.name)
                    if entry.name.lower().endswith('.bphys'):
                        result['bphysFileCount'] += 1
                        match = re.search(r'_(\d{6})_\d+\.bphys$', entry.name, re.I)
                        if match:
                            frames.add(int(match.group(1)))
        if frames:
            result['frameNumbersParsedFromNames'] = {
                'first': min(frames), 'last': max(frames), 'uniqueCount': len(frames),
                'scope': 'Filename inventory only; no payload, object-index or coverage validation'}
    except OSError as exc:
        result['scanError'] = str(exc)
    return result


def _node_trees(tree, seen=None):
    seen = set() if seen is None else seen
    if tree is None or tree.as_pointer() in seen:
        return
    seen.add(tree.as_pointer())
    yield tree
    for node in tree.nodes:
        child = getattr(node, 'node_tree', None)
        if child:
            yield from _node_trees(child, seen)


def _uv_requirements(texture):
    """Conservative input tracing; arbitrary node-group interfaces stay unknown."""
    socket = texture.inputs.get('Vector')
    if socket is None:
        return {'activeUV': False, 'namedUVs': [], 'uncertain': True}
    if not socket.is_linked:
        return {'activeUV': True, 'namedUVs': [], 'uncertain': False}
    active = False; named = set(); uncertain = False; seen = set()
    stack = [(link.from_node, link.from_socket) for link in socket.links]
    while stack:
        node, output = stack.pop()
        key = (node.as_pointer(), output.identifier)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 100:
            uncertain = True
            break
        if node.bl_idname == 'ShaderNodeUVMap':
            name = getattr(node, 'uv_map', '')
            if name: named.add(name)
            else: active = True
        elif node.bl_idname == 'ShaderNodeTexCoord':
            active = active or output.name == 'UV'
        elif node.bl_idname in {'ShaderNodeGroup', 'NodeGroupInput', 'ShaderNodeAttribute'}:
            uncertain = True
        else:
            for input_socket in node.inputs:
                stack.extend((link.from_node, link.from_socket) for link in input_socket.links)
    return {'activeUV': active, 'namedUVs': sorted(named), 'uncertain': uncertain}


def _material_row(material):
    textures = []
    for tree in _node_trees(material.node_tree if material.use_nodes else None):
        for node in tree.nodes:
            image = getattr(node, 'image', None)
            if node.bl_idname == 'ShaderNodeTexImage':
                textures.append({'tree': tree.name, 'node': node.name,
                                 'image': image.name if image else None,
                                 'uvRequirement': _uv_requirements(node)})
    return {'name': material.name, 'users': material.users,
            'library': material.library.filepath if material.library else None,
            'useNodes': material.use_nodes, 'imageTextures': textures}


def _active_render_collections(view_layer):
    names = set()
    def visit(layer, blocked=False):
        blocked = blocked or layer.exclude or layer.collection.hide_render
        if not blocked:
            names.add(layer.collection.as_pointer())
        for child in layer.children:
            visit(child, blocked)
    visit(view_layer.layer_collection)
    return names


def _fcurves(action):
    curves = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                curves.extend(bag.fcurves)
    if not curves and hasattr(action, 'fcurves'):
        curves = list(action.fcurves)
    return curves


def _action_row(action):
    curves = _fcurves(action)
    return {'name': action.name, 'users': action.users, 'fakeUser': action.use_fake_user,
            'library': action.library.filepath if action.library else None,
            'frameRange': list(action.frame_range), 'fcurveCount': len(curves),
            'keyframeCount': sum(len(curve.keyframe_points) for curve in curves),
            'sampledPointCount': sum(len(curve.sampled_points) for curve in curves),
            'slots': [{'identifier': slot.identifier, 'targetType': slot.target_id_type}
                      for slot in getattr(action, 'slots', [])],
            'scope': 'Inventory only; names do not establish motion-capture origin or animation quality'}


def _animation_ids():
    seen = set()
    buckets = ('objects', 'meshes', 'curves', 'hair_curves', 'shape_keys', 'materials',
               'node_groups', 'scenes', 'worlds', 'cameras', 'lights', 'armatures',
               'lattices', 'speakers', 'volumes', 'textures', 'movieclips', 'masks',
               'metaballs', 'grease_pencils')
    for bucket in buckets:
        for item in getattr(bpy.data, bucket, []):
            for block in (item, getattr(item, 'node_tree', None)):
                if block is None or block.as_pointer() in seen:
                    continue
                seen.add(block.as_pointer())
                if getattr(block, 'animation_data', None):
                    yield block


def _animations():
    bindings = []; invalid = []; disabled = []; total = 0
    for block in _animation_ids():
        animation = block.animation_data
        binding = {'id': block.name, 'idType': block.bl_rna.identifier,
                   'activeAction': animation.action.name if animation.action else None,
                   'actionSlot': getattr(getattr(animation, 'action_slot', None), 'identifier', None),
                   'nlaTracks': [{'name': track.name, 'mute': track.mute,
                                  'strips': [{'name': strip.name,
                                              'action': strip.action.name if getattr(strip, 'action', None) else None,
                                              'frameStart': strip.frame_start, 'frameEnd': strip.frame_end}
                                             for strip in track.strips]}
                                 for track in animation.nla_tracks]}
        bindings.append(binding)
        for curve in animation.drivers:
            total += 1
            driver = curve.driver
            row = {'id': block.name, 'idType': block.bl_rna.identifier,
                   'dataPath': curve.data_path, 'arrayIndex': curve.array_index,
                   'driverValid': driver.is_valid, 'fcurveValid': curve.is_valid,
                   'muted': curve.mute, 'type': driver.type,
                   'expression': driver.expression,
                   'simpleExpression': getattr(driver, 'is_simple_expression', None),
                   'variables': [{'name': var.name, 'type': var.type,
                                  'targets': [{'id': target.id.name if target.id else None,
                                               'dataPath': target.data_path,
                                               'boneTarget': target.bone_target}
                                              for target in var.targets]}
                                 for var in driver.variables]}
            if not driver.is_valid or not curve.is_valid:
                invalid.append(row)
            elif curve.mute:
                disabled.append(row)
    return {'bindings': bindings, 'driverCount': total, 'invalidDrivers': invalid,
            'mutedValidDrivers': disabled,
            'scope': 'Validity flags at the current evaluated state; audit does not change frame or force dependency evaluation'}


def report(output_relative=None, expected_source_relative=None, allow_master=False):
    """Collect facts and optional JSON. This function never marks delivery passed."""
    if bpy is None:
        raise RuntimeError('Run inside the already opened integrator Blender source')
    if not bpy.data.filepath:
        raise RuntimeError('Save an independent source before auditing')
    project = project_path().resolve(); source = Path(bpy.data.filepath).resolve()
    if not _inside(source, project):raise RuntimeError('Audit a saved source inside the project')
    if source == project / 'work/character.blend' and not (
            allow_master and expected_source_relative is not None and
            (project/expected_source_relative).resolve()==source):
        raise RuntimeError('Master read-only audit requires explicit allow_master and exact expected source')
    if expected_source_relative is not None and source != (project / expected_source_relative).resolve():
        raise RuntimeError('Opened source differs from the requested independent source')
    source_stat = source.stat(); source_sha = _sha256(source)
    dirty_before = bool(bpy.data.is_dirty)
    scene = bpy.context.scene; layer = bpy.context.view_layer
    render_collections = _active_render_collections(layer)
    findings = []; dependencies = []; cloth = []; objects = []
    materials = [_material_row(mat) for mat in bpy.data.materials]
    material_lookup = {row['name']: row for row in materials}

    def finding(kind, owner, detail, **extra):
        findings.append(dict(kind=kind, owner=owner, detail=detail, **extra))

    def dependency(kind, owner, raw, library=None, packed=False, **extra):
        row = dict(kind=kind, owner=owner, **_path_info(raw, project, library, packed), **extra)
        dependencies.append(row)
        if not packed and row['pathKind'] != 'builtin':
            if row['exists'] is False:
                finding('missing-external-dependency', owner, kind, storedPath=row['storedPath'])
            if row['withinProject'] is False:
                finding('dependency-outside-project', owner, kind, resolvedPath=row['resolvedPath'])
        return row

    for library in bpy.data.libraries:
        dependency('linked-library', library.name, library.filepath, library=library.library)

    images = []
    for image in bpy.data.images:
        packed = _packed(image)
        row = {'name': image.name, 'source': image.source, 'users': image.users,
               'packed': packed, 'packedFileCount': len(getattr(image, 'packed_files', [])),
               'colorSpace': image.colorspace_settings.name,
               'hasResidentPixels': image.has_data, 'library': image.library.filepath if image.library else None,
               'files': []}
        if image.source in {'FILE', 'TILED', 'SEQUENCE', 'MOVIE'}:
            if not packed and not image.filepath:
                finding('file-image-without-path-or-packed-data', image.name,
                        'File-backed image has neither a stored path nor packed data', users=image.users)
            paths = [image.filepath]
            if image.source == 'TILED':
                paths = [image.filepath.replace('<UDIM>', str(tile.number)) for tile in image.tiles]
                row['tileNumbers'] = [tile.number for tile in image.tiles]
                if '<UDIM>' not in image.filepath:
                    row['tilePathCoverageUncertain'] = True
            for path in dict.fromkeys(paths):
                row['files'].append(dependency('image', image.name, path, image.library, packed,
                                               imageSource=image.source, users=image.users))
            if image.source in {'SEQUENCE', 'MOVIE'}:
                row['coverageScope'] = 'Stored file only; sequence ranges/movie readability not decoded'
        else:
            row['coverageScope'] = 'Generated or runtime image; no external file inferred'
        images.append(row)

    for bucket in ('cache_files', 'movieclips', 'sounds', 'fonts', 'volumes'):
        for block in getattr(bpy.data, bucket, []):
            raw = getattr(block, 'filepath', '')
            if raw:
                dependency(bucket, block.name, raw, getattr(block, 'library', None), _packed(block))

    for obj in scene.objects:
        try:
            visible = obj.visible_get(view_layer=layer)
        except RuntimeError:
            visible = False
        render_candidate = (not obj.hide_render and
                            any(col.as_pointer() in render_collections for col in obj.users_collection))
        slots = [slot.material.name if slot.material else None for slot in obj.material_slots]
        row = {'name': obj.name, 'type': obj.type, 'visibleInCurrentViewport': visible,
               'renderEligibleInCurrentLayer': render_candidate,
               'objectHideViewport': obj.hide_viewport, 'objectHideRender': obj.hide_render,
               'collections': [col.name for col in obj.users_collection],
               'library': obj.library.filepath if obj.library else None,
               'parent': obj.parent.name if obj.parent else None,
               'materialSlots': slots, 'modifiers': []}
        relevant = visible or render_candidate
        if obj.type in MATERIAL_OBJECT_TYPES and relevant:
            if not slots:
                finding('no-material-slots', obj.name, 'Visible/render-eligible geometry has no material slots')
            elif any(name is None for name in slots):
                finding('empty-material-slot', obj.name, 'One or more material slots are empty',
                        slotIndices=[i for i, name in enumerate(slots) if name is None])
        if obj.type == 'MESH':
            mesh = obj.data; uv_names = [uv.name for uv in mesh.uv_layers]
            material_indices = array.array('i', [0]) * len(mesh.polygons)
            mesh.polygons.foreach_get('material_index', material_indices)
            used_indices = sorted(set(material_indices))
            row['mesh'] = {'vertices': len(mesh.vertices), 'polygons': len(mesh.polygons),
                           'uvLayers': uv_names,
                           'activeRenderUV': next((uv.name for uv in mesh.uv_layers if uv.active_render), None),
                           'usedMaterialIndices': used_indices}
            if relevant:
                for index in used_indices:
                    if index >= len(slots) or slots[index] is None:
                        finding('polygon-material-unassigned', obj.name,
                                'Polygon material index resolves to an absent slot', materialIndex=index)
                for name in set(slots) - {None}:
                    for texture in material_lookup[name]['imageTextures']:
                        requirement = texture['uvRequirement']
                        if texture['image'] is None:
                            finding('texture-node-without-image', obj.name,
                                    'Image Texture node has no image; node may be intentionally unused',
                                    material=name, node=texture['node'])
                        if requirement['activeUV'] and not uv_names:
                            finding('missing-active-uv', obj.name, 'Image Texture requests an active UV map', material=name)
                        for required in requirement['namedUVs']:
                            if required not in uv_names:
                                finding('missing-named-uv', obj.name, 'Material requests a named UV map absent from this mesh',
                                        material=name, uvMap=required)
                        if requirement['uncertain'] and not uv_names:
                            finding('uv-provenance-needs-review', obj.name,
                                    'No UV layers; group/attribute texture coordinates were not resolved', material=name)
        elif obj.type == 'CURVES':
            row['nativeHair'] = {'curves': len(obj.data.curves),
                                 'points': len(obj.data.position_data),
                                 'scope': 'Blender native hair geometry; no VRM/export equivalence established'}
        for modifier in obj.modifiers:
            mod = {'name': modifier.name, 'type': modifier.type,
                   'showViewport': modifier.show_viewport, 'showRender': modifier.show_render}
            group = getattr(modifier, 'node_group', None)
            if group:
                mod['nodeGroup'] = group.name
                mod['nodeTypes'] = dict(collections.Counter(node.bl_idname for node in group.nodes))
            row['modifiers'].append(mod)
            if modifier.type == 'CLOTH':
                cache = modifier.point_cache
                raw = cache.filepath
                cache_row = {'object': obj.name, 'modifier': modifier.name,
                             'useExternal': cache.use_external, 'useDiskCache': cache.use_disk_cache,
                             'isBakedFlag': cache.is_baked, 'frameStart': cache.frame_start,
                             'isOutdatedFlag': cache.is_outdated, 'isFrameSkipFlag': cache.is_frame_skip,
                             'frameEnd': cache.frame_end, 'frameStep': cache.frame_step,
                             'cacheName': cache.name, 'index': cache.index,
                             'storedPath': raw,
                             'scope': 'Cache flags and file inventory only; no cache replay/relocation or collision validation'}
                if raw:
                    path = dependency('cloth-cache-directory', obj.name, raw)
                    cache_row['path'] = path; cache_row['inventory'] = _directory_inventory(path)
                else:
                    cache_row['path'] = None
                    cache_row['inventory'] = None
                    cache_row['note'] = 'No explicit path; internal/default cache location has not been inferred'
                if cache.use_external and (not raw or not cache_row.get('inventory', {}).get('bphysFileCount', 0)):
                    finding('external-cloth-cache-without-inventoried-payload', obj.name,
                            'External cache has no directly inventoried .bphys files')
                cloth.append(cache_row)
            elif modifier.type in {'SOFT_BODY', 'FLUID', 'DYNAMIC_PAINT', 'PARTICLE_SYSTEM'}:
                finding('other-simulation-needs-review', obj.name,
                        'This audit inventories cloth, not this simulation payload', modifier=modifier.name, type=modifier.type)
            filepath = getattr(modifier, 'filepath', '')
            if filepath:
                dependency('modifier-file', obj.name + '/' + modifier.name, filepath)
        objects.append(row)

    for material in materials:
        owners = [obj for obj in objects if material['name'] in obj['materialSlots']]
        material['currentSceneUsers'] = [obj['name'] for obj in owners]
        material['viewportVisibleUsers'] = [obj['name'] for obj in owners if obj['visibleInCurrentViewport']]
        material['renderEligibleUsers'] = [obj['name'] for obj in owners if obj['renderEligibleInCurrentLayer']]

    seen_file_trees = set()
    for bucket in ('materials', 'worlds', 'lights', 'scenes', 'node_groups'):
        for block in getattr(bpy.data, bucket, []):
            root_tree = block if bucket == 'node_groups' else getattr(block, 'node_tree', None)
            for tree in _node_trees(root_tree, seen_file_trees):
                for node in tree.nodes:
                    raw = getattr(node, 'filepath', '')
                    if raw:
                        dependency('node-file', tree.name + '/' + node.name, raw, tree.library)

    shape_keys = []
    for keys in bpy.data.shape_keys:
        shape_keys.append({'name': keys.name, 'users': keys.users,
                           'relative': keys.use_relative,
                           'action': keys.animation_data.action.name if keys.animation_data and keys.animation_data.action else None,
                           'keyCount': len(keys.key_blocks),
                           'keys': [{'name': key.name, 'value': key.value, 'mute': key.mute,
                                     'points': len(key.data),
                                     'relativeKey': key.relative_key.name if key.relative_key else None}
                                    for key in keys.key_blocks]})
    animation = _animations()
    for driver in animation['invalidDrivers']:
        finding('invalid-driver', driver['id'], driver['dataPath'], arrayIndex=driver['arrayIndex'])
    if dirty_before:
        finding('unsaved-scene-state', source.name,
                'Inventory describes current memory; source SHA describes the last saved file')
    end_stat = source.stat()
    changed_on_disk = (source_stat.st_size, source_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns)
    if changed_on_disk:
        finding('source-changed-during-audit', source.name, 'Saved file changed while inventory was collected')
    result = {
        'schemaVersion': VERSION, 'auditStatus': 'inventory-only; delivery acceptance not assessed',
        'recordedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tool': {'blenderVersion': bpy.app.version_string,
                 'script': _relative(Path(__file__), project), 'scriptSha256': _sha256(Path(__file__))},
        'source': {'path': str(source), 'projectRelativePath': _relative(source, project),
                   'sha256': source_sha, 'bytes': source_stat.st_size,
                   'unsavedChangesBeforeAudit': dirty_before,
                   'unsavedChangesAfterAudit': bool(bpy.data.is_dirty),
                   'savedFileChangedDuringAudit': changed_on_disk},
        'context': {'scene': scene.name, 'viewLayer': layer.name,
                    'frame': scene.frame_current + scene.frame_subframe,
                    'fps': scene.render.fps / scene.render.fps_base,
                    'renderEngine': scene.render.engine},
        'summary': {'objectsInCurrentScene': len(objects),
                    'visibleViewportObjects': sum(row['visibleInCurrentViewport'] for row in objects),
                    'renderEligibleObjects': sum(row['renderEligibleInCurrentLayer'] for row in objects),
                    'materials': len(materials), 'images': len(images),
                    'packedImages': sum(row['packed'] for row in images),
                    'linkedLibraries': len(bpy.data.libraries), 'clothCaches': len(cloth),
                    'nativeCurveObjects': sum(row['type'] == 'CURVES' for row in objects),
                    'geometryNodesModifiers': sum(mod['type'] == 'NODES' for row in objects for mod in row['modifiers']),
                    'actions': len(bpy.data.actions), 'shapeKeyDatablocks': len(shape_keys),
                    'invalidDrivers': len(animation['invalidDrivers']),
                    'findingsByKind': dict(collections.Counter(row['kind'] for row in findings))},
        'objects': objects, 'materials': materials, 'images': images,
        'dependencies': dependencies, 'clothCaches': cloth,
        'actions': [_action_row(action) for action in bpy.data.actions],
        'shapeKeys': shape_keys, 'animation': animation, 'findings': findings,
        'scope': [
            'Read-only scene inventory; optional JSON report write only. No blend save, packing, frame changes, render or simulation replay.',
            'Object visibility/material/UV checks cover the current scene and view layer; linked datablocks, images, actions and animation drivers cover inventoried file datablocks.',
            'Render eligibility means object/collection/view-layer visibility flags; camera/frustum, shader opacity, holdout and ray visibility are not certified.',
            'UV tracing is conservative and does not prove texture mapping correctness or shader connectivity to Material Output.',
            'Cloth directory scan is shallow and bounded; file existence is not proof that frames replay correctly or remain valid after relocation.',
            'Native CURVES, Geometry Nodes, embedded guide shape keys and Blender cloth are high-quality Blender source features. This report establishes no VRM capability, compatibility or equivalent exported appearance.',
            'Preserved hidden guides, historical actions and disabled modifiers remain inventory entries; existence does not mean they are enabled for the delivered action.',
            'No full collision/deformation testing, motion authenticity inference, visual likeness review, customer acceptance, licensing determination or remote backup verification.',
            'Image sequences are not decoded across their ranges; geometry-node simulation bakes/file-import socket paths, sequencer media and arbitrary custom-script dependencies need separate review if used.'
        ]}
    if output_relative is not None:
        output = (project / output_relative).resolve()
        if not _inside(output, project) or output.suffix.lower() != '.json':
            raise ValueError('Report output must be a JSON file inside this project')
        output.parent.mkdir(parents=True, exist_ok=True)
        result['writtenReport'] = _relative(output, project)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print('Loaded only. Call report(output_relative=...) in an already opened independent Blender source.')
