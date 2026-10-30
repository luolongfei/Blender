"""Read-only bpy.data inventory. The integrator opens the donor file explicitly.

Do not run this by opening the asset with Auto Run enabled. Prefer Blender's
--disable-autoexec background mode. This code does not save, render, evaluate
modifiers, run embedded Text blocks, or change the current scene.
"""
import json
import bpy


def name(value):
    return value.name if value else None


result = {
    'blendFile': bpy.data.filepath,
    'blenderVersion': bpy.app.version_string,
    'libraries': [lib.filepath for lib in bpy.data.libraries],
    'collections': [{'name': col.name, 'objects': [o.name for o in col.objects],
                     'children': [c.name for c in col.children]}
                    for col in bpy.data.collections],
    'objects': [],
    'images': [{'name': im.name, 'filepath': im.filepath,
                'size': list(im.size), 'packed': bool(im.packed_file)}
               for im in bpy.data.images],
    'texts': [],
}
for obj in bpy.data.objects:
    item = {'name': obj.name, 'type': obj.type, 'parent': name(obj.parent),
            'hideRender': obj.hide_render, 'hideViewport': obj.hide_viewport,
            'data': name(obj.data), 'modifiers': [],
            'materials': [name(slot.material) for slot in obj.material_slots]}
    for modifier in obj.modifiers:
        item['modifiers'].append({'name': modifier.name, 'type': modifier.type,
                                  'nodeGroup': name(getattr(modifier, 'node_group', None))})
    if obj.type == 'CURVES':
        item.update(curves=len(obj.data.curves), points=len(obj.data.position_data),
                    surface=name(obj.data.surface), surfaceUV=obj.data.surface_uv_map,
                    attributes=[{'name': a.name, 'domain': a.domain, 'type': a.data_type}
                                for a in obj.data.attributes])
    elif obj.type == 'CURVE':
        item['splines'] = len(obj.data.splines)
    elif obj.type == 'MESH':
        item.update(vertices=len(obj.data.vertices), polygons=len(obj.data.polygons),
                    uvLayers=[uv.name for uv in obj.data.uv_layers])
    result['objects'].append(item)
for text_block in bpy.data.texts:
    content = text_block.as_string()
    words = ('license', 'licence', 'copyright', 'creativecommons', 'cc-by', 'bystedt')
    if any(word in (text_block.name + content).lower() for word in words):
        result['texts'].append({'name': text_block.name, 'characters': len(content),
                                'content': content[:20000], 'truncated': len(content) > 20000})
    else:
        result['texts'].append({'name': text_block.name, 'characters': len(content)})
print('HAIR_DEMO_READONLY_JSON=' + json.dumps(result, ensure_ascii=False))
