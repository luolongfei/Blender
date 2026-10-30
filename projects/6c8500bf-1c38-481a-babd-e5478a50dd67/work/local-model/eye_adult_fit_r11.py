"""New adult female eye attachments for an independent R11 candidate.

Explicit API: build_adult_eyes(candidate_collection, female_to_target_matrix,
                             parent_object=None)
Reads the actual CC0 female donor array export, creates independent eyes and
materials, and does not edit the head, old eyes, source images, rigs, or scene.
No render/save is performed. Integrator owns saving and actual visual checks.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

DONOR_SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
IRIS_SHA='6bddc86582fa1b4541376082816c4d734deabbbcbe38e65710415823f0566784'
R09_OPTICS_SOURCE_SHA='1c0b0210114170f0ce6e09b266b4a9ded58f29bf1df45a6e2be13e9b5c77aa72'
PREFIX='R11 Adult'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_materials(bpy, texture):
    """Reproduce accepted R09 inputs without retaining unused legacy images."""
    image=bpy.data.images.load(str(texture),check_existing=False)
    image.name='R11 documented brown iris v3'
    image.colorspace_settings.name='sRGB'
    image.pack()
    if not image.packed_file:raise RuntimeError('New iris image must be packed for a self-contained candidate')
    iris=bpy.data.materials.new(PREFIX+' warm iris single reflection')
    iris.use_nodes=True
    tree=iris.node_tree;nodes=tree.nodes
    shader=next(n for n in nodes if n.type=='BSDF_PRINCIPLED')
    shader.inputs['Roughness'].default_value=.14
    shader.inputs['IOR'].default_value=1.5
    shader.inputs['Metallic'].default_value=0.
    shader.inputs['Transmission Weight'].default_value=0.
    shader.inputs['Specular IOR Level'].default_value=0.
    shader.inputs['Coat Weight'].default_value=1.
    shader.inputs['Coat Roughness'].default_value=.03
    shader.inputs['Coat IOR'].default_value=1.376
    shader.inputs['Coat Tint'].default_value=(1.,1.,1.,1.)
    tex=nodes.new('ShaderNodeTexImage');tex.name='R11 documented R07 iris';tex.image=image
    tex.interpolation='Linear';tex.extension='EXTEND'
    tex.label='Workspace procedural R07 v3, unchanged original pixels'
    uv=nodes.new('ShaderNodeUVMap');uv.uv_map='IrisUV';tree.links.new(uv.outputs['UV'],tex.inputs['Vector'])
    mult=nodes.new('ShaderNodeMixRGB');mult.name='R11 warm linear reflectance'
    mult.blend_type='MULTIPLY';mult.inputs[0].default_value=1.
    mult.inputs[2].default_value=(.85,.72,.58,1.)
    tree.links.new(tex.outputs['Color'],mult.inputs[1]);tree.links.new(mult.outputs['Color'],shader.inputs['Base Color'])
    iris.diffuse_color=(.06,.025,.009,1.)
    iris['source_record']='textures/procedural-r07/iris-source-r07.json'
    iris['R11_optics']='R09 warm iris: texture x (0.85,0.72,0.58), base specular 0, sole coat reflection 1.0/roughness .03/IOR1.376'
    iris['R11_texture_sha256']=IRIS_SHA
    sclera=bpy.data.materials.new(PREFIX+' soft warm sclera')
    sclera.use_nodes=True
    sb=next(n for n in sclera.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    sb.inputs['Base Color'].default_value=(.44,.42,.38,1.)
    sb.inputs['Roughness'].default_value=.19
    sb.inputs['IOR'].default_value=1.5
    sb.inputs['Specular IOR Level'].default_value=.5
    sb.inputs['Coat Weight'].default_value=.06
    sb.inputs['Coat Roughness'].default_value=.03
    sb.inputs['Coat IOR'].default_value=1.376
    sclera.diffuse_color=(.44,.42,.38,1.)
    sclera['R11_optics']='Reviewed R09 warm sclera inputs reproduced; no ethnicity-dependent color change'
    return iris,sclera,image


def _source(P):
    import numpy as np
    meta_path=P/'work/reconstruction-r02/female-basemesh-actual-r11.json'
    array_path=P/'work/reconstruction-r02/female-basemesh-actual-r11.npz'
    meta=json.loads(meta_path.read_text(encoding='utf8'))
    if meta['source_sha256']!=DONOR_SHA or _sha(array_path)!=meta['npz_sha256']:
        raise RuntimeError('Actual female donor export identity differs')
    rows={o['name']:o for o in meta['objects']}
    body=np.asarray(rows['GEO-body_female_realistic']['matrix_world'],dtype=np.float64)
    body_inv=np.linalg.inv(body)
    eyes={}
    with np.load(array_path) as arrays:
        for side in ('L','R'):
            row=rows['GEO-body_female_realistic.eye.'+side]
            prefix=row['prefix'];local=arrays[prefix+'_local'].astype(np.float64)
            transform=body_inv@np.asarray(row['matrix_world'],dtype=np.float64)
            co=local@transform[:3,:3].T+transform[:3,3]
            corners=arrays[prefix+'_corner_vertices'];starts=arrays[prefix+'_face_starts'];lengths=arrays[prefix+'_face_lengths']
            faces=[tuple(int(v) for v in corners[start:start+length]) for start,length in zip(starts,lengths)]
            uv=arrays[row['uv_layers'][0]['array']].copy()
            eyes[side]={'row':row,'co':co,'faces':faces,'uv':uv,'matrix_to_body':transform}
    return meta,eyes


def _iris_geometry(co, faces, center, radius, *, segments=96, rings=16, clearance=.00006):
    """Sample the actual donor front surface, then offset 0.06 mm outwards.

    Opaque textured cap plus a single reflective coat is the retained static
    optical approximation. There is no second transparent cornea/glass shell.
    """
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh=BVHTree.FromPolygons([Vector(v) for v in co],faces,all_triangles=False)
    verts=[];uvs=[];misses=[];surface_forward_dots=[]
    cast_start_y=float(co[:,1].min()-.02)
    def sample(x,z,u,v):
        hit,normal,index,distance=bvh.ray_cast(Vector((x,cast_start_y,z)),Vector((0.,1.,0.)),.1)
        if hit is None:
            misses.append([x,z]);return
        surface_forward_dots.append(float(-normal.y))
        verts.append((float(hit.x-center[0]),float(hit.y-center[1]-clearance),float(hit.z-center[2])))
        uvs.append((u,v))
    sample(float(center[0]),float(center[2]),.5,.5)
    for ring in range(1,rings+1):
        f=ring/rings
        for i in range(segments):
            angle=2*math.pi*i/segments
            x=math.cos(angle);z=math.sin(angle)
            sample(float(center[0]+radius*f*x),float(center[2]+radius*f*z),.5+.477*f*x,.5+.477*f*z)
    if misses or len(verts)!=1+rings*segments:
        raise RuntimeError('Iris front-surface ray misses; do not invent fallback eye geometry: '+str(misses[:4]))
    if min(surface_forward_dots)<=0.:
        raise RuntimeError('Donor front surface normal points inward')
    caps=[(0,1+i,1+(i+1)%segments) for i in range(segments)]
    for ring in range(rings-1):
        a=1+ring*segments;b=a+segments
        for i in range(segments):
            j=(i+1)%segments
            caps.append((a+i,b+i,b+j,a+j))
    arr=np.asarray(verts)
    if not np.isfinite(arr).all():raise RuntimeError('Nonfinite iris cap')
    cap_forward_dots=[]
    for face in caps:
        a,b,c=arr[list(face[:3])]
        normal=np.cross(b-a,c-a)
        length=float(np.linalg.norm(normal))
        if length<1e-14:raise RuntimeError('Degenerate iris cap face')
        cap_forward_dots.append(float(-normal[1]/length))
    if min(cap_forward_dots)<=0.:raise RuntimeError('Iris face winding points inward')
    return verts,caps,uvs,{'ray_hits':len(verts),'ray_misses':len(misses),'surface_clearance_source_m':clearance,
                          'radial_segments':segments,'radial_rings':rings,'uv_image_radius':.477,
                          'minimum_donor_front_normal_dot':min(surface_forward_dots),
                          'minimum_iris_front_normal_dot':min(cap_forward_dots)}


def build_adult_eyes(candidate_collection, female_to_target_matrix, parent_object=None):
    import bpy
    import numpy as np
    from mathutils import Matrix
    P=project_path()
    if not bpy.app.background or bpy.context.mode!='OBJECT':
        raise RuntimeError('Separate background candidate in Object Mode required')
    current=Path(bpy.data.filepath).resolve()
    if current==(P/'work/character.blend').resolve() or current.parent==(P/'exports').resolve():
        raise RuntimeError('Do not operate on master or published source')
    try:current.relative_to((P/'work/reconstruction-r02').resolve())
    except ValueError:raise RuntimeError('Load an independent reconstruction-r02 candidate first')
    if isinstance(candidate_collection,str):candidate_collection=bpy.data.collections.get(candidate_collection)
    if candidate_collection is None or candidate_collection.library:
        raise ValueError('Explicit existing local candidate collection required')
    if parent_object is not None and (parent_object.library or parent_object.name not in bpy.data.objects):
        raise ValueError('Explicit local candidate parent required')
    names=[PREFIX+' '+part+' '+side for side in ('L','R') for part in ('eye pivot','sclera','iris')]
    if any(bpy.data.objects.get(n) for n in names):
        raise RuntimeError('Adult eye candidate already exists; preserve it instead of overwriting')
    T=np.asarray(female_to_target_matrix,dtype=np.float64)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-9):
        raise ValueError('Explicit finite affine 4x4 body-space-to-target transform required')
    linear=T[:3,:3];scale=float(np.linalg.norm(linear[:,0]))
    if np.linalg.det(linear)<=0 or not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):
        raise ValueError('Use positive uniform body scale and rotation; no independent eye stretch')
    if not .5<scale<1.5:raise ValueError('Unexpected character scale; inspect source units')
    texture=P/'textures/procedural-r07/iris-brown-r07-v3.png'
    if not texture.is_file() or _sha(texture)!=IRIS_SHA:
        raise RuntimeError('Documented procedural R07 iris source changed')
    meta,eyes=_source(P)
    created_objects=[];created_meshes=[];materials=[];image=None
    report={'source_blend':meta['source'],'source_blend_sha256':DONOR_SHA,'source_array_sha256':meta['npz_sha256'],
            'candidate':str(current.relative_to(P)), 'female_to_target_matrix':T.tolist(),'uniform_scale':scale,
            'parent':parent_object.name if parent_object else None,'collection':candidate_collection.name,
            'texture':'textures/procedural-r07/iris-brown-r07-v3.png','texture_sha256':IRIS_SHA,
            'optics_reference_source_sha256':R09_OPTICS_SOURCE_SHA,
            'optics':'R09 warm iris single reflection inputs reproduced in clean new materials. No legacy unknown-source texture nodes retained.',
            'geometry':'Unchanged actual female eye surface; all source sclera vertices/faces kept. Iris is new radial geometry projected onto the donor front surface. Eye dimensions are never inherited from the old enlarged eyes.',
            'eyes':{},'saved_blend':False,'appearance_accepted':False,
            'limitations':['Static centered forward gaze only; no eye bone or eyelid changes performed.','Opaque surface iris with one clearcoat is an optical approximation, not a separate refractive cornea and iris chamber.','The v3 texture supplies the pupil visually; the pupil is not a physical aperture.','Lid contact, closed blink, look directions, iris visibility and catchlights require actual candidate checks.']}
    try:
        iris_mat,sclera_mat,image=_new_materials(bpy,texture);materials=[iris_mat,sclera_mat]
        report['materials']=[m.name for m in materials]
        report['image']={'name':image.name,'packed':bool(image.packed_file),'colorspace':image.colorspace_settings.name,
                         'source_project_relative_path':str(texture.relative_to(P)),'source_sha256':IRIS_SHA}
        for side,data in eyes.items():
            co=data['co'];faces=data['faces'];mn=co.min(axis=0);mx=co.max(axis=0)
            center=(mn+mx)*.5
            # Fit diagnostics are recorded, but bbox centre is used for the
            # exact donor's slightly non-spherical surface and limb placement.
            A=np.column_stack([2*co,np.ones(len(co))]);rhs=np.einsum('ij,ij->i',co,co)
            fit=np.linalg.lstsq(A,rhs,rcond=None)[0]
            sphere_radius=float(math.sqrt(max(0.,fit[3]+np.dot(fit[:3],fit[:3]))))
            effective_diameter=float((mx[0]-mn[0]+mx[2]-mn[2])*.5)
            iris_radius=effective_diameter*.245
            cap_v,cap_f,cap_uv,cap_report=_iris_geometry(co,faces,center,iris_radius)
            pivot=bpy.data.objects.new(PREFIX+' eye pivot '+side,None);created_objects.append(pivot)
            candidate_collection.objects.link(pivot);pivot.empty_display_type='PLAIN_AXES';pivot.empty_display_size=.006*scale
            pivot.parent=parent_object
            # Resolve the actual parent evaluation before assigning a world
            # matrix. The donor probe with parent=None cannot cover this path.
            bpy.context.view_layer.update()
            pivot.matrix_world=Matrix(T.tolist())@Matrix.Translation(center.tolist())
            pivot['R11_eye_role']='adult_eye_rotation_pivot';pivot['R11_donor_side']=side
            made={}
            for part,positions,polys,uvs,material in [
                ('sclera',(co-center).tolist(),faces,data['uv'],sclera_mat),
                ('iris',cap_v,cap_f,cap_uv,iris_mat),
            ]:
                mesh=bpy.data.meshes.new(PREFIX+' '+part+' mesh '+side);created_meshes.append(mesh)
                mesh.from_pydata(positions,[],polys);mesh.update()
                obj=bpy.data.objects.new(PREFIX+' '+part+' '+side,mesh);created_objects.append(obj)
                candidate_collection.objects.link(obj);obj.parent=pivot;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
                mesh.materials.append(material)
                for poly in mesh.polygons:poly.use_smooth=True
                uv=mesh.uv_layers.new(name='IrisUV' if part=='iris' else 'DonorUV')
                if part=='sclera':
                    uv.data.foreach_set('uv',np.asarray(uvs,dtype=np.float32).ravel())
                    attr=mesh.attributes.new('R11_donor_vertex_index','INT','POINT')
                    attr.data.foreach_set('value',np.arange(len(co),dtype=np.int32))
                else:
                    for poly in mesh.polygons:
                        for li in poly.loop_indices:uv.data[li].uv=uvs[mesh.loops[li].vertex_index]
                obj['R11_eye_role']=part;obj['R11_donor_side']=side
                obj['R11_source_blend_sha256']=DONOR_SHA
                obj['R11_source_geometry']='GEO-body_female_realistic.eye.'+side if part=='sclera' else 'New workspace radial iris cap fitted to official female donor'
                obj['R11_eyes_not_final_approved']=True
                made[part]=obj.name
            target_center=linear@center+T[:3,3]
            target_coordinates=co@linear.T+T[:3,3]
            target_iris=(np.asarray(cap_v)+center)@linear.T+T[:3,3]
            report['eyes'][side]={
                'objects':{'pivot':pivot.name,**made},'source_object':data['row']['name'],
                'source_matrix_world':data['row']['matrix_world'],'source_matrix_to_body_space':data['matrix_to_body'].tolist(),
                'center_body_space':center.tolist(),'center_target_world':target_center.tolist(),
                'source_body_space_bounds':[mn.tolist(),mx.tolist()],
                'source_diameters_m':(mx-mn).tolist(),'target_diameters_m':((mx-mn)*scale).tolist(),
                'target_sclera_world_bounds':[target_coordinates.min(axis=0).tolist(),target_coordinates.max(axis=0).tolist()],
                'target_iris_world_bounds':[target_iris.min(axis=0).tolist(),target_iris.max(axis=0).tolist()],
                'sphere_fit_diagnostic':{'center_body_space':fit[:3].tolist(),'radius_m':sphere_radius,
                                         'rms_m':float(np.sqrt(np.mean((np.linalg.norm(co-fit[:3],axis=1)-sphere_radius)**2)))},
                'iris_source_radius_m':iris_radius,'iris_target_radius_m':iris_radius*scale,
                'iris_to_globe_diameter_ratio':.49,
                'pupil_texture_to_iris_diameter_ratio':.145/.477,
                'pupil_approx_target_diameter_m':2*iris_radius*scale*.145/.477,
                'forward_axis_body_space':[0,-1,0],'forward_axis_target_world':(linear@np.array([0.,-1.,0.])/scale).tolist(),
                'surface_cap':cap_report,'sclera_vertices':len(co),'sclera_faces':len(faces),
                'iris_vertices':len(cap_v),'iris_faces':len(cap_f),
            }
            pivot['R11_eye_fit_record']=json.dumps(report['eyes'][side],ensure_ascii=False)
        bpy.context.view_layer.update()
        # Only newly created objects are inspected or assigned; source head,
        # previous eye geometry/materials, and image pixels remain untouched.
        for obj in created_objects:
            if obj.type=='MESH' and (obj.data.users!=1 or obj.library or obj.data.library):
                raise RuntimeError('Independent local eye mesh required')
        report['actual_world_checks']={}
        for side,eye in report['eyes'].items():
            checks={}
            pivot=bpy.data.objects[eye['objects']['pivot']]
            pivot_error=float(np.max(np.abs(np.asarray(pivot.matrix_world.translation)-np.asarray(eye['center_target_world']))))
            if pivot_error>5e-7:raise RuntimeError('Actual parented eye pivot differs from requested world centre')
            checks['pivot_maximum_world_error_m']=pivot_error
            for part in ('sclera','iris'):
                obj=bpy.data.objects[eye['objects'][part]]
                co=np.empty(len(obj.data.vertices)*3,dtype=np.float64)
                obj.data.vertices.foreach_get('co',co);co=co.reshape(-1,3)
                matrix=np.asarray(obj.matrix_world,dtype=np.float64)
                actual=co@matrix[:3,:3].T+matrix[:3,3]
                bounds=np.array([actual.min(axis=0),actual.max(axis=0)])
                expected=np.asarray(eye['target_'+part+'_world_bounds'])
                error=float(np.max(np.abs(bounds-expected)))
                if error>5e-7:
                    raise RuntimeError(f'Actual parented {part} world bounds differ by {error:g} m; do not save this candidate')
                checks[part]={'actual_world_bounds':bounds.tolist(),'maximum_bound_error_m':error,
                              'matrix_world':matrix.tolist()}
            report['actual_world_checks'][side]=checks
        if _sha(texture)!=IRIS_SHA:raise RuntimeError('Original iris pixels changed')
        return report
    except Exception:
        for obj in reversed(created_objects):
            if obj.name in bpy.data.objects:bpy.data.objects.remove(obj,do_unlink=True)
        for mesh in created_meshes:
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        for material in materials:
            if material.users==0:bpy.data.materials.remove(material)
        if image is not None and image.users==0:bpy.data.images.remove(image)
        raise


if __name__=='__main__':
    print('Loaded only. Call build_adult_eyes(collection, body_to_target_matrix, parent_object=None) explicitly in an independent candidate.')
