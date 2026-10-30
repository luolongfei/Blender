"""Fit retained dental arches to the new lip envelope and add editable gum ridges."""
import bpy,math
from mathutils import Vector

def run():
    if bpy.data.objects.get('R03 Upper gingiva'):raise RuntimeError('Already fitted')
    for name,dy,dz in [('R02 Upper teeth',-.005,-.003),('R02 Lower teeth',-.0095,.0008),('R02 Tongue',-.003,.002)]:
        o=bpy.data.objects[name]
        for v in o.data.vertices:v.co.y+=dy;v.co.z+=dz
    mat=bpy.data.materials.new('R03 natural gingiva');mat.use_nodes=True
    bs=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    bs.inputs['Base Color'].default_value=(.16,.043,.05,1);bs.inputs['Roughness'].default_value=.42
    bs.inputs['Subsurface Weight'].default_value=.15;bs.inputs['Subsurface Scale'].default_value=.0012
    for side in ['Upper','Lower']:
        source=bpy.data.objects['R02 '+side+' teeth'];mesh=source.data
        adjacent=[set() for _ in mesh.vertices]
        for edge in mesh.edges:
            a,b=edge.vertices;adjacent[a].add(b);adjacent[b].add(a)
        left=set(range(len(mesh.vertices)));centers=[]
        while left:
            part={left.pop()};stack=list(part)
            while stack:
                for j in adjacent[stack.pop()]:
                    if j in left:left.remove(j);part.add(j);stack.append(j)
            co=[mesh.vertices[i].co for i in part];q=sum(co,Vector())/len(co)
            q.z=(max(v.z for v in co)+.0007) if side=='Upper' else (min(v.z for v in co)-.0007)
            centers.append(q)
        if len(centers)<6:raise RuntimeError('Expected distinct teeth to derive gingiva')
        yc=max(q.y for q in centers)+.008
        centers.sort(key=lambda q:math.atan2(q.x,yc-q.y))
        data=bpy.data.curves.new('R03 '+side+' gingiva','CURVE');data.dimensions='3D';data.resolution_u=6
        data.bevel_depth=.0022;data.bevel_resolution=3;data.use_fill_caps=True
        spline=data.splines.new('BEZIER');spline.bezier_points.add(len(centers)-1)
        for point,q in zip(spline.bezier_points,centers):
            point.co=q;point.handle_left_type='AUTO';point.handle_right_type='AUTO'
        data.materials.append(mat)
        obj=bpy.data.objects.new(data.name,data);bpy.context.scene.collection.objects.link(obj)
        obj.parent=source.parent;obj.parent_type=source.parent_type;obj.parent_bone=source.parent_bone
        obj.matrix_parent_inverse=source.matrix_parent_inverse.copy();obj.matrix_basis=source.matrix_basis.copy()
    bpy.context.view_layer.update();print('R03_ORAL_FIT_CREATED')

if __name__=='__main__':run()
