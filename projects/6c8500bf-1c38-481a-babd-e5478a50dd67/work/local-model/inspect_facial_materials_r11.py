"""Read-only actual evaluated material-slot audit of current facial fibres."""
import bpy,json
deps=bpy.context.evaluated_depsgraph_get();rows=[]
for obj in bpy.context.scene.objects:
    if obj.type!='CURVES' or not any(s in obj.name.lower() for s in ('lash','brow')):continue
    ev=obj.evaluated_get(deps)
    def material(m):
        return None if m is None else {'name':m.name,'nodes':[{'name':n.name,'type':n.type,'color':list(n.inputs['Base Color'].default_value) if n.inputs.get('Base Color') else list(n.inputs['Color'].default_value) if n.inputs.get('Color') else None} for n in m.node_tree.nodes] if m.use_nodes else None}
    rows.append({'name':obj.name,'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hideGet':obj.hide_get(),
        'slots':[{'link':s.link,'material':material(s.material)} for s in obj.material_slots],
        'rawDataMaterials':[m.name if m else None for m in obj.data.materials],
        'evaluatedMaterials':[m.name if m else None for m in ev.data.materials],
        'rawAttributes':[(a.name,a.domain,a.data_type) for a in obj.data.attributes],
        'evaluatedAttributes':[(a.name,a.domain,a.data_type) for a in ev.data.attributes],
        'curves':len(ev.data.curves),'points':len(ev.data.position_data)})
print('R11_ACTUAL_FACIAL_MATERIALS '+json.dumps(rows),flush=True)
print('R11_ACTUAL_SCENE_OVERRIDE '+json.dumps({'viewLayers':[{'name':v.name,'override':v.material_override.name if v.material_override else None} for v in bpy.context.scene.view_layers]}),flush=True)
