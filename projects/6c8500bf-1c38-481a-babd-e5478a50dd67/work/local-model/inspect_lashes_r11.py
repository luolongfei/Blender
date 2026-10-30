"""Read-only actual lash geometry and visibility after saved-file reopen."""
import bpy,json,numpy as np
scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update();dg=bpy.context.evaluated_depsgraph_get()
rows=[]
for obj in scene.objects:
    if not obj.name.startswith(('R11 Adult lashes','R11 Adult natural brow')):continue
    data=obj.data;co=np.empty((len(data.position_data),3));data.position_data.foreach_get('vector',co.ravel())
    m=np.asarray(obj.matrix_world);world=co@m[:3,:3].T+m[:3,3]
    radii=np.empty(len(data.attributes['radius'].data));data.attributes['radius'].data.foreach_get('value',radii)
    ev=obj.evaluated_get(dg);ec=np.empty((len(ev.data.position_data),3));ev.data.position_data.foreach_get('vector',ec.ravel());em=np.asarray(ev.matrix_world);ew=ec@em[:3,:3].T+em[:3,3]
    rows.append({'name':obj.name,'hideRender':obj.hide_render,'hideViewport':obj.hide_viewport,'hidden':obj.hide_get(),'visibleCamera':obj.visible_camera,
        'matrix':m.tolist(),'localBounds':[co.min(0).tolist(),co.max(0).tolist()], 'worldBounds':[world.min(0).tolist(),world.max(0).tolist()],
        'evaluatedBounds':[ew.min(0).tolist(),ew.max(0).tolist()],'radiusMinMax':[float(radii.min()),float(radii.max())],
        'attributes':[(a.name,a.data_type,a.domain,len(a.data)) for a in data.attributes],'materials':[x.name for x in data.materials],
        'collections':[(c.name,c.hide_render,c.hide_viewport) for c in obj.users_collection],
        'firstCurve':world[:9].tolist()})
print('R11_LASH_ACTUAL '+json.dumps(rows),flush=True)
