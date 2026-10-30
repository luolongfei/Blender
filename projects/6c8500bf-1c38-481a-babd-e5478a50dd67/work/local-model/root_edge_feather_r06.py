"""Narrow alpha feather on the existing R06 scalp substrate only.

No tail, gather, scalp-fibre or forelock geometry changes. Linear subdivision
adds samples without smoothing/moving the substrate surface or its boundary.
The fade uses distance to the actual old open boundary, in world-space metres.
The original mesh and its material slots are retained for restore(). No IO,
render, cache bake or file save occurs here.
"""
import json
import math
import bpy
import bmesh
import numpy as np

OBJECT = 'R06 attached scalp root substrate'
MARKER = 'R06_narrow_root_feather'
ATTRIBUTE = 'R06_root_edge_alpha'
BACKUP = 'R06_root_pre_feather_mesh'


def _world(vertices, matrix):
    return vertices@matrix[:3,:3].T+matrix[:3,3]


def _coords(mesh):
    a=np.empty(len(mesh.vertices)*3,dtype=np.float64)
    mesh.vertices.foreach_get('co',a)
    return a.reshape(-1,3)


def _boundary(mesh, world):
    count={}
    for face in mesh.polygons:
        ids=list(face.vertices)
        for a,b in zip(ids,ids[1:]+ids[:1]):
            key=tuple(sorted((a,b)));count[key]=count.get(key,0)+1
    if any(n>2 for n in count.values()):
        raise RuntimeError('Non-manifold substrate edges require explicit review')
    edges=[key for key,n in count.items() if n==1]
    if not edges:raise RuntimeError('No open root substrate boundary was found')
    return np.array([world[list(edge)] for edge in edges]),len(edges)


def _distance(points,segments):
    a=segments[:,0];v=segments[:,1]-a
    length2=np.maximum((v*v).sum(axis=1),1e-18)
    result=np.empty(len(points),dtype=np.float64)
    for start in range(0,len(points),512):
        q=points[start:start+512]
        delta=q[:,None,:]-a[None,:,:]
        t=np.clip((delta*v[None,:,:]).sum(axis=2)/length2[None,:],0,1)
        closest=a[None,:,:]+t[:,:,None]*v[None,:,:]
        result[start:start+len(q)]=np.sqrt(((q[:,None,:]-closest)**2).sum(axis=2).min(axis=1))
    return result


def run(front_mm=4.0,side_mm=2.5,back_mm=3.0,diagnostics_only=False):
    if not bpy.data.filepath or bpy.data.filepath.replace('\\','/').endswith('/work/character.blend'):
        raise RuntimeError('Use the saved reference-scalp-r06 candidate, not the master')
    if not all(1.5<=v<=5.0 for v in (front_mm,side_mm,back_mm)):
        raise ValueError('This patch is limited to a narrow 1.5–5 mm root edge')
    obj=bpy.data.objects.get(OBJECT)
    if obj is None or obj.type!='MESH':raise RuntimeError('Existing R06 root substrate is required')
    if obj.get(MARKER):raise RuntimeError('Already feathered; restore() before retrying')
    if obj.library or obj.data.library or obj.data.users!=1:
        raise RuntimeError('Local single-user substrate is required')
    if getattr(obj.data,'shape_keys',None) or any(m.show_viewport or m.show_render for m in obj.modifiers):
        raise RuntimeError('Unexpected substrate deformation/modifiers')
    rig=bpy.data.objects.get('Character_Rig');pose=rig.data.pose_position if rig else None
    if rig:rig.data.pose_position='REST'
    bpy.context.view_layer.update()
    prepared=None
    try:
        matrix=np.array(obj.matrix_world)
        original=_coords(obj.data);old_world=_world(original,matrix)
        segments,boundary_count=_boundary(obj.data,old_world)
        max_edge=max(float(np.linalg.norm(old_world[e.vertices[1]]-old_world[e.vertices[0]]))
                     for e in obj.data.edges)
        cuts=max(1,min(12,int(math.ceil(max_edge/.0015))-1))
        prepared=obj.data.copy();prepared.name='R06 scalp substrate narrow feather samples'
        bm=bmesh.new()
        try:
            bm.from_mesh(prepared)
            bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=cuts,use_grid_fill=True,smooth=0.0)
            bm.to_mesh(prepared)
        finally:bm.free()
        prepared.update()
        world=_world(_coords(prepared),matrix)
        distance=_distance(world,segments)
        center=(old_world.min(axis=0)+old_world.max(axis=0))*.5
        xy=world[:,:2]-center[:2]
        radial=np.maximum(np.linalg.norm(xy,axis=1),1e-7)
        front=np.clip(-xy[:,1]/radial,0,1)
        back=np.clip(xy[:,1]/radial,0,1)
        width=(side_mm+(front_mm-side_mm)*front**2+(back_mm-side_mm)*back**2)*.001
        # Only a 0.00–0.18 mm variation inside the transition: no perforation or
        # transparency in the interior. Every true boundary sample remains zero.
        variation=.00018*(.5+.5*np.sin(world@np.array([2471.,3719.,1793.])))
        u=np.clip((distance-variation)/np.maximum(width-variation,.001),0,1)
        alpha=u*u*(3-2*u)
        alpha[distance>=width]=1.0
        alpha[distance<1e-7]=0.0
        attr=prepared.attributes.get(ATTRIBUTE) or prepared.attributes.new(ATTRIBUTE,'FLOAT','POINT')
        attr.data.foreach_set('value',alpha)
        for face in prepared.polygons:face.use_smooth=True
        # A linear subdivision must preserve the perimeter position. Compare its
        # new boundary samples directly against the original boundary segments.
        new_boundary,_=_boundary(prepared,world)
        perimeter_error=float(_distance(new_boundary.reshape(-1,3),segments).max())
        if perimeter_error>2e-6:
            raise RuntimeError('Root boundary moved during linear subdivision: '+str(perimeter_error))
        report={'object':obj.name,'old_vertices':len(original),'new_vertices':len(world),
                'original_boundary_edges':boundary_count,'linear_subdivision_cuts':cuts,
                'fade_width_mm':{'front':front_mm,'side':side_mm,'back':back_mm},
                'maximum_boundary_motion_mm':perimeter_error*1000,
                'fully_opaque_samples':int(np.count_nonzero(alpha>=.999999)),
                'transition_samples':int(np.count_nonzero((alpha>0)&(alpha<.999999))),
                'writesPerformed':not diagnostics_only,
                'preserved':'all existing fibre, forelock, gather and tail geometry; skin and rig',
                'validation':'distance/boundary checks only; final edge appearance requires actual render'}
        if diagnostics_only:
            bpy.data.meshes.remove(prepared);prepared=None
            print(json.dumps(report));return report
        source=next((m for m in obj.data.materials if m),None)
        if source is None or not source.use_nodes:raise RuntimeError('Original root shader is required')
        material=source.copy();material.name='R06 root substrate narrow alpha feather'
        tree=material.node_tree
        output=next((n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
        if output is None or not output.inputs['Surface'].is_linked:
            raise RuntimeError('Root material has no connected surface output')
        old_link=output.inputs['Surface'].links[0];old_surface=old_link.from_socket
        tree.links.remove(old_link)
        attribute=tree.nodes.new('ShaderNodeAttribute');attribute.attribute_name=ATTRIBUTE
        transparent=tree.nodes.new('ShaderNodeBsdfTransparent')
        transparent.inputs['Color'].default_value=(1,1,1,1)
        mix=tree.nodes.new('ShaderNodeMixShader')
        tree.links.new(attribute.outputs['Fac'],mix.inputs[0])
        tree.links.new(transparent.outputs['BSDF'],mix.inputs[1])
        tree.links.new(old_surface,mix.inputs[2])
        tree.links.new(mix.outputs[0],output.inputs['Surface'])
        # Cycles evaluates the transparent shader directly; choose the available
        # dithered/hashed mode as well for a consistent material preview.
        for prop,wanted in [('surface_render_method','DITHERED'),('blend_method','HASHED')]:
            descriptor=material.bl_rna.properties.get(prop)
            if descriptor and wanted in descriptor.enum_items.keys():setattr(material,prop,wanted)
        if hasattr(material,'use_transparent_shadow'):material.use_transparent_shadow=True
        prepared.materials.clear();prepared.materials.append(material)
        original_mesh=obj.data;original_mesh.use_fake_user=True
        obj[BACKUP]=original_mesh.name;obj.data=prepared;obj[MARKER]=True
        obj['R06_root_feather_width_mm']=json.dumps(report['fade_width_mm'])
        prepared=None
        obj.data.update_tag();obj.update_tag();bpy.context.view_layer.update()
        print(json.dumps(report));return report
    finally:
        if prepared is not None and prepared.users==0:bpy.data.meshes.remove(prepared)
        if rig:rig.data.pose_position=pose
        bpy.context.view_layer.update()


def restore():
    obj=bpy.data.objects[OBJECT]
    if not obj.get(MARKER):return False
    old=bpy.data.meshes.get(obj.get(BACKUP,''))
    if old is None:raise RuntimeError('Retained pre-feather root mesh is missing')
    obj.data.use_fake_user=True;obj.data=old;old.use_fake_user=False
    del obj[MARKER];del obj[BACKUP]
    if 'R06_root_feather_width_mm' in obj:del obj['R06_root_feather_width_mm']
    bpy.context.view_layer.update();return True


if __name__=='__main__':run()
