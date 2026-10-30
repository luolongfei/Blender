"""Use native strand curves so the hair BSDF receives real strand tangents."""
import bpy
def run():
    created=[]
    for old in list(bpy.context.scene.objects):
        if old.type!='CURVE' or old.hide_render:continue
        if not old.name.startswith(('R01 Groom','R02 Surface bangs','R02 Curved temple','R02 Twin')):continue
        if old.data.shape_keys:raise RuntimeError('Keep expression-driven curves separate')
        name='R02 Strands / '+old.name
        if bpy.data.objects.get(name):raise RuntimeError('Already converted '+name)
        data=bpy.data.hair_curves.new(name)
        sizes=[len(s.points) for s in old.data.splines]
        data.add_curves(sizes)
        coords=[f for s in old.data.splines for p in s.points for f in p.co[:3]]
        data.position_data.foreach_set('vector',coords)
        radius=data.attributes.new('radius','FLOAT','POINT')
        radius.data.foreach_set('value',[old.data.bevel_depth*p.radius*.85 for s in old.data.splines for p in s.points])
        for mat in old.data.materials:data.materials.append(mat)
        o=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(o)
        o.parent=old.parent;o.parent_type=old.parent_type;o.parent_bone=old.parent_bone;o.matrix_parent_inverse=old.matrix_parent_inverse.copy();o.matrix_basis=old.matrix_basis.copy()
        o['source_groom']=old.name;o['scope']='Native editable strands; dynamic attachment pending'
        old.hide_render=True;old.hide_set(True)
        data.update_tag();created.append({'name':name,'strands':len(sizes),'points':sum(sizes)})
    bpy.context.view_layer.update();print(created);return created
if __name__=='__main__':run()
