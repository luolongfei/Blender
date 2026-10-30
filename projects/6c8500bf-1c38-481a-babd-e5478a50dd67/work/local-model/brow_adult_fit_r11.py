"""Optional natural brow fibres for the new adult female head.

No import-time work, save, render, head edits, rig edits, or existing-brow edits.
build_adult_brows(collection, head_object, female_to_target_matrix,
                  parent_object=None) creates two independent native Curves.
Roots and every fibre point are ray-fitted to the actual evaluated candidate
head. Parenting follows object motion only; facial shape following is pending.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
PREFIX='R11 Adult natural brow'
DONOR_SHA='3c121505651140ceb4d69fd1d8923f7788ffadd81672f5be14845a5f2c75c137'
AUTHOR_ARRAY_SHA='2728bbb2c3bd948b6ec8ce426db2d41dbeb68da6844a21d7d48e3a8dd5ed297f'


def _smooth(t):
    t=max(0.,min(1.,t));return t*t*(3.-2.*t)


def _arc(t):
    """Soft cubic arch in verified donor body-space, never an old brow fit.

    Actual L3 surface samples show the convex brow ridge at z~1.556–1.564 m
    medially, dropping toward z~1.550 m at the temple. These are staging
    coordinates on this particular mesh, not population facial proportions.
    """
    # Cubic Bezier has broad tangent transitions instead of a pointed peak.
    controls=((.0140,1.5585),(.0270,1.5660),(.0440,1.5640),(.0590,1.5555))
    a=1.-t
    x=a*a*a*controls[0][0]+3*a*a*t*controls[1][0]+3*a*t*t*controls[2][0]+t*t*t*controls[3][0]
    z=a*a*a*controls[0][1]+3*a*a*t*controls[1][1]+3*a*t*t*controls[2][1]+t*t*t*controls[3][1]
    dx=3*a*a*(controls[1][0]-controls[0][0])+6*a*t*(controls[2][0]-controls[1][0])+3*t*t*(controls[3][0]-controls[2][0])
    dz=3*a*a*(controls[1][1]-controls[0][1])+6*a*t*(controls[2][1]-controls[1][1])+3*t*t*(controls[3][1]-controls[2][1])
    return x,z,dz/dx


def plan_brow_fibres(side, count=320):
    """Deterministic stratified roots and short directional fibres, no Blender."""
    if side not in ('L','R') or not 180<=count<=500:
        raise ValueError('Explicit L/R and a moderate fibre count required')
    sign=1. if side=='L' else -1.
    rng=random.Random(2026102811+(0 if side=='L' else 73))
    # Density is modest at the inner head, fuller in the middle and softer at
    # the narrow tail. A stratified inverse CDF avoids clumps and random gaps.
    bins=1024
    cdf=[0.]
    for j in range(bins):
        t=(j+.5)/bins
        cdf.append(cdf[-1]+.46+.74*math.sin(math.pi*t)**.8)
    total=cdf[-1];cdf=[v/total for v in cdf]
    plans=[];cursor=0
    for i in range(count):
        q=(i+.22+.56*rng.random())/count
        while cursor<bins-1 and cdf[cursor+1]<q:cursor+=1
        fraction=(q-cdf[cursor])/max(cdf[cursor+1]-cdf[cursor],1e-12)
        t=(cursor+fraction)/bins
        x,z,slope=_arc(t)
        row=rng.triangular(-1.,1.,0.)
        half_width=(.00205+.00065*math.sin(math.pi*t))*(1.-.72*_smooth((t-.58)/.42))
        z+=row*half_width
        # Inner hairs rise and fan out. Through the body, upper/lower rows
        # gently converge; the tail follows the declining anatomical arc.
        transition=_smooth(t/.30)
        dx=.18+.82*transition
        dz=.98*(1.-transition)+(slope-.33*row)*transition
        dz+=rng.uniform(-.12,.12)
        norm=math.hypot(dx,dz);dx/=norm;dz/=norm
        length=(.00345+rng.uniform(-.00065,.00065))*(1.-.38*_smooth((t-.73)/.27))
        radius=.000028*rng.uniform(.73,1.13)*(1.-.24*_smooth((t-.65)/.35))
        path=[]
        for j in range(6):
            u=j/5.
            xx=x+dx*length*u
            zz=z+dz*length*u+.00012*math.sin(math.pi*u)*(1.-t)
            path.append((sign*xx,zz))
        plans.append({'t':t,'row':row,'root_body_xz':[sign*x,z],'path_body_xz':path,
                      'length_source_m':length,'radius_source_m':radius})
    return plans


def _material(bpy):
    material=bpy.data.materials.new(PREFIX+' black brown fibres')
    material.use_nodes=True
    tree=material.node_tree;tree.nodes.clear()
    out=tree.nodes.new('ShaderNodeOutputMaterial')
    hair=tree.nodes.new('ShaderNodeBsdfHairPrincipled');hair.parametrization='COLOR'
    hair.inputs['Color'].default_value=(.010,.0065,.0045,1.)
    hair.inputs['Roughness'].default_value=.53
    if 'Radial Roughness' in hair.inputs:hair.inputs['Radial Roughness'].default_value=.58
    if 'IOR' in hair.inputs:hair.inputs['IOR'].default_value=1.55
    tree.links.new(hair.outputs[0],out.inputs['Surface'])
    material.diffuse_color=(.010,.0065,.0045,1.)
    material['R11_source']='New workspace authored material; no image input, no inherited brow material'
    return material


def build_adult_brows(candidate_collection, head_object, female_to_target_matrix,
                      parent_object=None, *, fibres_per_side=320):
    import bpy
    import numpy as np
    from mathutils import Matrix,Vector
    from mathutils.bvhtree import BVHTree
    P=project_path()
    if not bpy.app.background or bpy.context.mode!='OBJECT':
        raise RuntimeError('Use explicit background Object Mode on an independent candidate')
    path=Path(bpy.data.filepath).resolve()
    try:path.relative_to((P/'work/reconstruction-r02').resolve())
    except ValueError:raise RuntimeError('Do not operate on master, official donor, or published source')
    if isinstance(candidate_collection,str):candidate_collection=bpy.data.collections.get(candidate_collection)
    if isinstance(head_object,str):head_object=bpy.data.objects.get(head_object)
    if candidate_collection is None or candidate_collection.library:
        raise ValueError('Existing independent candidate collection required')
    if head_object is None or head_object.type!='MESH' or head_object.library:
        raise ValueError('Actual local new adult head mesh required')
    if head_object.name=='R01 Head':raise ValueError('Do not fit these brows to the rejected old face')
    if parent_object is not None and (parent_object.library or parent_object.name not in bpy.data.objects):
        raise ValueError('Local candidate parent only')
    if any(bpy.data.objects.get(PREFIX+' '+s) for s in ('L','R')):
        raise RuntimeError('Preserve existing adult brow candidate instead of overwriting')
    T=np.asarray(female_to_target_matrix,dtype=np.float64)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-8):
        raise ValueError('Finite explicit donor-body-to-target matrix required')
    scale=float(np.linalg.norm(T[:3,0]));linear=T[:3,:3]
    if not .5<scale<1.5 or np.linalg.det(linear)<=0 or not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):
        raise ValueError('Uniform positive character scale; no brow-only stretch')
    bpy.context.view_layer.update()
    evaluated=head_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me=evaluated.data
    mesh_co=np.empty(len(me.vertices)*3,dtype=np.float64);me.vertices.foreach_get('co',mesh_co);mesh_co=mesh_co.reshape(-1,3)
    matrix=np.asarray(evaluated.matrix_world,dtype=np.float64)
    world=mesh_co@matrix[:3,:3].T+matrix[:3,3]
    faces=[tuple(p.vertices) for p in me.polygons]
    bvh=BVHTree.FromPolygons([Vector(v) for v in world],faces,all_triangles=False)
    ray_direction=Vector((linear@np.array([0.,1.,0.])/scale).tolist())
    outward=-ray_direction
    source_observation=P/'work/reconstruction-r02/female-multires-actual-r11.json'
    if not source_observation.is_file() or json.loads(source_observation.read_text(encoding='utf8'))['npz_sha256']!=AUTHOR_ARRAY_SHA:
        raise RuntimeError('Verified author Multires observation required for this brow placement')
    prepared=[]
    for side in ('L','R'):
        plans=plan_brow_fibres(side,fibres_per_side)
        paths=[];radii=[];roots=[];surface_faces=[];clearances=[];normal_dots=[]
        for index,plan in enumerate(plans):
            strand=[];strand_radii=[]
            for j,(x,z) in enumerate(plan['path_body_xz']):
                u=j/5.
                origin=linear@np.array([x,-.35,z])+T[:3,3]
                hit,normal,face_index,distance=bvh.ray_cast(Vector(origin.tolist()),ray_direction,.6*scale)
                if hit is None or not np.isfinite(tuple(hit)).all():
                    raise RuntimeError(f'Actual new brow surface ray miss {side} strand {index} point {j}; no fallback floating geometry')
                dot=float(normal.dot(outward));normal_dots.append(dot)
                if dot<=.15:raise RuntimeError('Unexpected backside/temple normal in brow region')
                radius=plan['radius_source_m']*scale*(.75+.25*_smooth(u/.15))*max(.035,(1.-u)**.62)
                clearance=plan['radius_source_m']*scale+.000015*scale+(.00015*math.sin(math.pi*u)+.00006*u)*scale
                point=hit+normal*clearance
                strand.append(tuple(point));strand_radii.append(radius);clearances.append(clearance)
                if j==0:
                    roots.append({'source_body_xz':[x,z],'world_skin':list(hit),'world_root':list(point),
                                  'world_normal':list(normal),'evaluated_face_index':int(face_index)})
                    surface_faces.append(int(face_index))
            paths.append(strand);radii.extend(strand_radii)
        coords=np.asarray(paths,dtype=np.float32).reshape(-1,3)
        prepared.append((side,plans,coords,np.asarray(radii,dtype=np.float32),roots,surface_faces,clearances,normal_dots))
    # All surface rays succeed before creation. We only create fresh Curves and
    # material datablocks; no head property/attribute/key/modifier is touched.
    created=[];data_blocks=[];material=None
    report={'head':head_object.name,'head_evaluated_vertices':len(me.vertices),'head_evaluated_faces':len(me.polygons),
            'head_matrix_world':matrix.tolist(),'candidate':str(path.relative_to(P)),
            'female_to_target_matrix':T.tolist(),'uniform_scale':scale,
            'donor_source_sha256':DONOR_SHA,'author_l3_array_sha256':AUTHOR_ARRAY_SHA,
            'brows':{},'saved_blend':False,'appearance_accepted':False,
            'purpose':'Optional static natural eyebrow fibres on the actual new adult female head; first face render need not wait for this module.',
            'limitations':['Parenting follows object motion only; brow roots do not yet follow facial shape keys or eyebrow muscle deformation.','Evaluated face indices and neutral roots are evidence for this candidate, not a finished stable surface-deform binding.','Neutral closeup and hairline/eyelid visual review remain required; density/arch are an authored candidate, not user acceptance.']}
    try:
        material=_material(bpy)
        report['material']=material.name
        for side,plans,coords,radii,roots,surface_faces,clearances,normal_dots in prepared:
            data=bpy.data.hair_curves.new(PREFIX+' strands '+side);data_blocks.append(data)
            data.add_curves([6]*len(plans));data.position_data.foreach_set('vector',coords.ravel())
            radius=data.attributes.new('radius','FLOAT','POINT');radius.data.foreach_set('value',radii)
            root_faces=data.attributes.new('R11_neutral_surface_face','INT','CURVE');root_faces.data.foreach_set('value',surface_faces)
            # Six closely sampled surface points keep the native curve short
            # and follow the actual skin; no evaluated surface is flattened.
            data.materials.append(material)
            obj=bpy.data.objects.new(PREFIX+' '+side,data);created.append(obj);candidate_collection.objects.link(obj)
            obj.matrix_world=Matrix.Identity(4)
            if parent_object is not None:
                # Geometry is already in target world space. A deliberate
                # parent inverse preserves it and subsequently follows motion.
                bpy.context.view_layer.update()
                obj.parent=parent_object;obj.parent_type='OBJECT'
                obj.matrix_parent_inverse=parent_object.matrix_world.inverted()
                obj.matrix_basis=Matrix.Identity(4)
            obj['R11_brow_role']='independent_neutral_fibres'
            obj['R11_facial_shape_following']='pending; object parent motion only'
            obj['R11_source']='Current workspace generated fibre paths, surface fitted to official CC0 female donor derivative'
            obj['R11_root_record']=json.dumps(roots,ensure_ascii=False)
            bpy.context.view_layer.update()
            if not np.allclose(np.asarray(obj.matrix_world),np.eye(4),atol=2e-6):
                raise RuntimeError('Parenting changed actual brow world coordinates')
            report['brows'][side]={'object':obj.name,'type':obj.type,'fibres':len(plans),'points':len(coords),
                                  'world_bounds':[coords.min(axis=0).tolist(),coords.max(axis=0).tolist()],
                                  'source_root_span_body_xz':[plans[0]['root_body_xz'],plans[-1]['root_body_xz']],
                                  'root_world_positions':[r['world_root'] for r in roots],
                                  'radius_world_m':[float(radii.min()),float(radii.max())],
                                  'strand_source_length_m':[min(p['length_source_m'] for p in plans),max(p['length_source_m'] for p in plans)],
                                  'skin_clearance_world_m':[min(clearances),max(clearances)],
                                  'minimum_front_normal_dot':min(normal_dots),'parent':obj.parent.name if obj.parent else None,
                                  'actual_world_matrix':np.asarray(obj.matrix_world).tolist(),
                                  'expression_following':False}
        return report
    except Exception:
        for obj in reversed(created):
            if obj.name in bpy.data.objects:bpy.data.objects.remove(obj,do_unlink=True)
        for data in data_blocks:
            if data.users==0:bpy.data.hair_curves.remove(data)
        if material is not None and material.users==0:bpy.data.materials.remove(material)
        raise


if __name__=='__main__':
    print('Loaded only. Optional build_adult_brows(collection, new_head, donor_to_target, parent_object=None). No source save or render.')
