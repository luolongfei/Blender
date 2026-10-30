"""Native, recorded late-blink correction for the newly authored upper lashes.

Each source builds its own fields on its actual current eyelids. The roots
are placed on a visible patch belonging to the NEUTRAL UPPER lid, not on the
lower lid; shafts then exit forward. Existing raw data and base maps stay exact.
Half blink and below have exactly zero corrective weight.
"""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

MARKER='R11_upper_lash_late_surface'
MODIFIER='R11 upper lash late surface'
ANCHORS=(.65,.8,.9,1.)
NAMES=tuple('R11 Adult lashes upper '+side for side in ('L','R'))
PUBLIC=('R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile','R11 warm closed smile')


def api(name):
    path=Path(__file__).with_name(name+'.py')
    if name=='upper_lash_corrective_validation_r11' and sha(path)!='36a8713e93a962504b6f1546fb466afd70e3a7af394c449b40176b32999e3c0a':
        raise RuntimeError('Frozen composed-validation helper differs')
    return runpy.run_path(str(path))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bvh(world,triangles):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([Vector(v) for v in world],[tuple(t) for t in triangles],all_triangles=True)


def capture_head(head,binding):
    import bpy
    result=binding['_head_mesh'](head,triangulate=True)
    mesh=head.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    ids=np.empty(len(mesh.loop_triangles),np.int32);mesh.loop_triangles.foreach_get('polygon_index',ids)
    result['trianglePolygons']=ids
    return result


def points(obj,binding):
    import bpy
    e=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return binding['_transform'](binding['_array'](e.data.position_data,'vector'),e.matrix_world)


def upper_patch(neutral,design,eye_center):
    """Polygon identities of the actual neutral upper ciliary skin ribbon."""
    roots=design['roots'];xs=np.asarray([r['newRoot'][0] for r in roots]);zs=np.asarray([r['actualApertureUpperZ'] for r in roots])
    order=np.argsort(xs);xs=xs[order];zs=zs[order]
    tris=neutral['world'][neutral['triangles']];centers=tris.mean(1)
    normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);normal_lengths=np.linalg.norm(normals,axis=1)
    safe=normal_lengths>1e-12;normals[safe]/=normal_lengths[safe,None]
    edge=np.interp(centers[:,0],xs,zs)
    keep=safe&(centers[:,0]>=xs.min()-.0015)&(centers[:,0]<=xs.max()+.0015)&(centers[:,2]>=edge-.0001)&(centers[:,2]<=edge+.0035)
    keep&=(centers[:,1]<eye_center[1]-.002)&(normals[:,1]<.05)
    polygons=np.unique(neutral['trianglePolygons'][keep])
    if len(polygons)<100:raise RuntimeError('Insufficient actual upper-lid patch polygons')
    return polygons


def rotation(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    v=np.cross(a,b);c=float(np.clip(a@b,-1,1));s=float(np.linalg.norm(v))
    if s<1e-10:
        if c<0:raise RuntimeError('Ambiguous lash exit rotation')
        return np.eye(3)
    k=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+k+k@k*((1-c)/(s*s))


def corrected_paths(base,radius,skin,upper,globe):
    """Reattach roots to visible upper skin and prevent the folded-down shaft."""
    from mathutils import Vector
    result=[];root_records=[];normal_root=np.asarray(base).reshape(156,9,3);radius=np.asarray(radius).reshape(156,9)
    forward=Vector((0.,1.,0.))
    def ray(surface,x,z):return surface.ray_cast(Vector((float(x),-.35,float(z))),forward,.7)
    for i,path in enumerate(normal_root):
        root=path[0];candidates=[]
        # The sample is deliberately confined to the identified upper skin.
        # A frontal ray through the whole head alone could put lashes on lower lid.
        for dz in np.r_[0.,np.linspace(-.0002,.0018,25)]:
            hit,n,index,d=ray(upper,root[0],root[2]+dz)
            if hit is None or n.y>=-.03:continue
            all_hit=ray(skin,hit.x,hit.z)[0]
            if all_hit is not None and all_hit.y<hit.y-.00004:continue
            q=np.asarray(hit)+np.asarray(n)*max(.000025,float(radius[i,0])*.55)
            near,nn,fi,dist=skin.find_nearest(Vector(q))
            if near is None or (Vector(q)-near).dot(nn)<-.000015:continue
            g=globe.find_nearest(Vector(q))
            if g[0] is None or (Vector(q)-g[0]).dot(g[1])<float(radius[i,0])+.00003:continue
            shift=q-root
            if np.linalg.norm(shift)>.0025:continue
            score=float(shift@shift)+float(shift[2]**2)*.25
            candidates.append((score,q,np.asarray(n),np.asarray(hit)))
        if not candidates:raise RuntimeError('No visible upper-lid surface for late lash root '+str(i))
        score,new_root,n,skin_point=min(candidates,key=lambda item:item[0])
        initial=path[1]-root;initial/=np.linalg.norm(initial)
        target=initial.copy()
        if -target[1]<.85:
            lateral=float(np.clip(target[0],-.30,.30));vertical=np.sign(target[2])*np.sqrt(max(0.,1-.85**2-lateral*lateral))
            target=np.asarray((lateral,-.85,vertical))
        target/=np.linalg.norm(target)
        rotation_angle=float(np.arccos(np.clip(initial@target,-1.,1.)))
        if rotation_angle>np.pi*.5:raise RuntimeError('Late lash exit requires a reversal '+str(i))
        r=rotation(initial,target);candidate=new_root+(path-root)@r.T
        if not np.allclose(np.linalg.norm(np.diff(candidate,axis=0),axis=1),np.linalg.norm(np.diff(path,axis=0),axis=1),rtol=0.,atol=1e-10):
            raise RuntimeError('Late lash orientation changed segment length '+str(i))
        # Small local forward clearance correction; the root remains exactly
        # on its separately identified upper-skin attachment.
        required=np.zeros(9)
        for j in range(1,9):
            limit=[]
            for surface in (skin,globe):
                h=ray(surface,candidate[j,0],candidate[j,2])[0]
                if h is not None:limit.append(h.y-float(radius[i,j])-.00008)
            if limit:required[j]=max(0.,float(candidate[j,1])-min(limit))
        smooth=required.copy()
        for j in range(1,9):
            for k in range(1,9):smooth[j]=max(smooth[j],required[k]*.55**abs(j-k))
        if smooth.max()>.001:raise RuntimeError('Late lash additional forward clearance exceeds 1 mm '+str(i))
        candidate[:,1]-=smooth;candidate[0]=new_root
        displacement=np.linalg.norm(candidate-path,axis=1)
        result.append(candidate)
        root_records.append({'strand':i,'originalRoot':root.tolist(),'root':new_root.tolist(),'skinPoint':skin_point.tolist(),'skinNormal':n.tolist(),
                             'shiftMeters':float(np.linalg.norm(new_root-root)),'maximumShaftCorrectionMeters':float(displacement.max()),
                             'orientationRotationDegrees':float(np.degrees(rotation_angle)),'maximumAdditionalForwardClearanceMeters':float(smooth.max()),
                             'initialForwardBefore':float(-initial[1]),'initialForwardAfter':float(-target[1])})
    return np.asarray(result).reshape(-1,3),root_records


def build_group(obj,lookup,validator):
    """Create the native modifier, cleaning partial construction on failure."""
    import bpy
    before_groups={group.as_pointer() for group in bpy.data.node_groups}
    before_modifiers={modifier.as_pointer() for modifier in obj.modifiers}
    before_drivers={curve.as_pointer() for curve in obj.animation_data.drivers} if obj.animation_data else set()
    try:return _build_group(obj,lookup,validator)
    except Exception:
        if obj.animation_data:
            for curve in list(obj.animation_data.drivers):
                if curve.as_pointer() not in before_drivers:obj.driver_remove(curve.data_path)
        for modifier in list(obj.modifiers):
            if modifier.as_pointer() not in before_modifiers:obj.modifiers.remove(modifier)
        for group in list(bpy.data.node_groups):
            if group.as_pointer() not in before_groups and group.users==0:bpy.data.node_groups.remove(group)
        raise


def _build_group(obj,lookup,validator):
    import bpy
    group=bpy.data.node_groups.new('R11 upper lash late surface '+obj.name[-1],'GeometryNodeTree');nodes=group.nodes;links=group.links
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry');group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    identifiers={}
    for name in tuple('weight'+str(i) for i in range(4))+('warm',):
        socket=group.interface.new_socket(name=name,in_out='INPUT',socket_type='NodeSocketFloat');socket.default_value=0.;identifiers[name]=socket.identifier
        if hasattr(socket,'force_non_field'):socket.force_non_field=True
    source=nodes.new('NodeGroupInput');output=nodes.new('NodeGroupOutput');position=nodes.new('GeometryNodeInputPosition');index=nodes.new('GeometryNodeInputIndex')
    info=nodes.new('GeometryNodeObjectInfo');info.transform_space='ORIGINAL';info.inputs['Object'].default_value=lookup;info.inputs['As Instance'].default_value=False
    def field(name):
        attribute=nodes.new('GeometryNodeInputNamedAttribute');attribute.data_type='FLOAT_VECTOR';attribute.inputs['Name'].default_value=name
        sample=nodes.new('GeometryNodeSampleIndex');sample.data_type='FLOAT_VECTOR';sample.domain='POINT';sample.clamp=False
        links.new(info.outputs['Geometry'],sample.inputs['Geometry']);links.new(attribute.outputs['Attribute'],sample.inputs['Value']);links.new(index.outputs[0],sample.inputs['Index'])
        return sample.outputs['Value']
    def scale(value,factor):
        node=nodes.new('ShaderNodeVectorMath');node.operation='SCALE';links.new(value,node.inputs[0]);links.new(factor,node.inputs['Scale']);return node.outputs['Vector']
    def add(a,b):
        node=nodes.new('ShaderNodeVectorMath');node.operation='ADD';links.new(a,node.inputs[0]);links.new(b,node.inputs[1]);return node.outputs['Vector']
    current=position.outputs['Position']
    for i in range(4):current=add(current,scale(add(field('r11_lash_late_n_'+str(i)),scale(field('r11_lash_late_w_'+str(i)),source.outputs['warm'])),source.outputs['weight'+str(i)]))
    setpos=nodes.new('GeometryNodeSetPosition');links.new(source.outputs['Geometry'],setpos.inputs['Geometry']);links.new(current,setpos.inputs['Position']);links.new(setpos.outputs['Geometry'],output.inputs['Geometry'])
    modifier=obj.modifiers.new(MODIFIER,'NODES');modifier.node_group=group
    expressions=validator['driver_expressions']()
    for logical,identifier in identifiers.items():
        modifier[identifier]=0.
        if identifier+'_use_attribute' in modifier:modifier[identifier+'_use_attribute']=False
        curve=obj.driver_add('modifiers["'+modifier.name+'"]["'+identifier+'"]');driver=curve.driver;driver.type='SCRIPTED'
        curve.mute=False;driver.use_self=False
        for curve_modifier in list(curve.modifiers):curve.modifiers.remove(curve_modifier)
        while driver.variables:driver.variables.remove(driver.variables[0])
        variable=driver.variables.new();variable.type='SINGLE_PROP';variable.name='w' if logical=='warm' else 'b'
        target=variable.targets[0];target.id_type='KEY';target.id=bpy.data.objects['R11 Female head study'].data.shape_keys
        control='R11 warm closed smile' if logical=='warm' else 'R11 blink '+obj.name[-1]
        target.data_path='key_blocks["'+control+'"].value';driver.expression=expressions[logical]
    return modifier,group,identifiers


def remove_current():
    import bpy
    scene=bpy.context.scene
    if MARKER not in scene:return
    record=json.loads(scene[MARKER])
    for name,entry in record['objects'].items():
        obj=bpy.data.objects[name];modifier=obj.modifiers.get(entry['modifier'])
        if modifier:
            for identifier in entry['interfaceIdentifiers'].values():
                try:obj.driver_remove('modifiers["'+modifier.name+'"]["'+identifier+'"]')
                except (TypeError,RuntimeError):pass
            obj.modifiers.remove(modifier)
        lookup=bpy.data.objects.get(entry['lookupObject'])
        if lookup:
            mesh=lookup.data;bpy.data.objects.remove(lookup,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
        group=bpy.data.node_groups.get(entry['nodeGroup'])
        if group and group.users==0:bpy.data.node_groups.remove(group)
    del scene[MARKER];bpy.context.view_layer.update()


def install(head,binding,changes,payload):
    import bpy
    if MARKER in bpy.context.scene:raise RuntimeError('Do not stack late upper-lash corrections')
    validator=api('upper_lash_corrective_validation_r11');scene=bpy.context.scene;keys=head.data.shape_keys.key_blocks
    previous={name:float(keys[name].value) for name in PUBLIC};objects={side:bpy.data.objects['R11 Adult lashes upper '+side] for side in ('L','R')}
    def controls(blink,warm):
        for name in PUBLIC:keys[name].value=blink if name.startswith('R11 blink ') else warm if name=='R11 warm closed smile' else 0.
        bpy.context.view_layer.update()
    record={'schemaVersion':1,'objects':{},'anchors':list(ANCHORS),'helperSha256':sha(__file__),
            'scope':'Current-source late upper-lash correction from actual upper skin; raw paths/maps/radii preserved. Neutral and blink <= .5 have exactly zero weight.',
            'source':str(Path(bpy.data.filepath).resolve().relative_to(project_path().resolve())),'sourceSha256':sha(bpy.data.filepath),
            'authoredStates':[]}
    generated={};created_orphans=[]
    try:
        controls(0.,0.);neutral=capture_head(head,binding)
        eyes={side:binding['_head_mesh'](bpy.data.objects['R11 Adult sclera '+side],triangulate=True) for side in ('L','R')}
        globes={side:make_bvh(eyes[side]['world'],eyes[side]['triangles']) for side in ('L','R')}
        polygons={}
        for side,obj in objects.items():
            design=next(row['design'] for row in changes if row['object']==obj.name)
            center=(eyes[side]['world'].min(0)+eyes[side]['world'].max(0))*.5
            polygons[side]=upper_patch(neutral,design,center);payload['late_'+side+'_upperPatchPolygonIds']=polygons[side]
            generated[side]={}
        for i,anchor in enumerate(ANCHORS):
            for warm in (0.,1.):
                controls(anchor,warm);current=capture_head(head,binding);skin=make_bvh(current['world'],current['triangles'])
                state={'blink':anchor,'warm':warm,'upperLashes':{}}
                for side,obj in objects.items():
                    selected=np.isin(current['trianglePolygons'],polygons[side]);upper=make_bvh(current['world'],current['triangles'][selected])
                    base=points(obj,binding);radius=binding['_array'](obj.data.attributes['radius'].data,'value',1).ravel()
                    scale=float(np.linalg.svd(np.asarray(obj.matrix_world)[:3,:3],compute_uv=False).max())
                    corrected,roots=corrected_paths(base,radius*scale,skin,upper,globes[side])
                    delta=(corrected-base)@np.linalg.inv(np.asarray(obj.matrix_world)[:3,:3]).T
                    if warm==0:generated[side]['r11_lash_late_n_'+str(i)]=np.asarray(delta,np.float32)
                    else:generated[side]['r11_lash_late_w_'+str(i)]=np.asarray(delta-generated[side]['r11_lash_late_n_'+str(i)],np.float32)
                    payload['late_'+side+'_anchor_'+str(i)+'_warm_'+str(int(warm))+'_baseWorld']=base
                    payload['late_'+side+'_anchor_'+str(i)+'_warm_'+str(int(warm))+'_targetWorld']=corrected
                    state['upperLashes'][side]={'roots':roots,'maximumCorrectionMeters':float(np.linalg.norm(corrected-base,axis=1).max())}
                record['authoredStates'].append(state)
                print('R11_LASH_LATE_FIELD '+json.dumps({'blink':anchor,'warm':warm}),flush=True)
        controls(0.,0.)
        for side,obj in objects.items():
            mesh=bpy.data.meshes.new('R11 upper lash late lookup '+side);mesh.from_pydata([(0.,0.,0.)]*len(obj.data.position_data),[],[]);mesh.update()
            lookup=bpy.data.objects.new(mesh.name,mesh);scene.collection.objects.link(lookup);lookup.hide_render=True
            # A point-only lookup has no render surface. Keep it evaluable for Object Info.
            lookup.display_type='WIRE';created_orphans.append((lookup,None,obj,None))
            hashes={}
            for field,value in generated[side].items():
                mesh.attributes.new(field,'FLOAT_VECTOR','POINT').data.foreach_set('vector',value.ravel());hashes[field]=binding['_ah'](value);payload['late_'+side+'_'+field]=value
            modifier,group,identifiers=build_group(obj,lookup,validator);created_orphans[-1]=(lookup,group,obj,modifier)
            bpy.context.view_layer.update()
            entry={'lookupObject':lookup.name,'lookupVertexCount':len(obj.data.position_data),'fieldsSha256':hashes,
                   'modifier':modifier.name,'nodeGroup':group.name,'interfaceIdentifiers':identifiers,
                   'nodeGroupContractSha256':binding['_hash'](validator['capture_group'](group)),
                   'driverContractSha256':binding['_hash'](validator['capture_drivers'](obj,modifier,identifiers))}
            record['objects'][obj.name]=entry
        scene[MARKER]=json.dumps(record,ensure_ascii=False);created_orphans=[]
        return record
    except Exception:
        if MARKER in scene:remove_current()
        for lookup,group,obj,modifier in reversed(created_orphans):
            if modifier is not None:
                if obj.animation_data:
                    prefix=modifier.path_from_id()+'['
                    for curve in list(obj.animation_data.drivers):
                        if curve.data_path.startswith(prefix):obj.driver_remove(curve.data_path)
                obj.modifiers.remove(modifier)
            mesh=lookup.data;bpy.data.objects.remove(lookup,do_unlink=True)
            if mesh.users==0:bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:bpy.data.node_groups.remove(group)
        raise
    finally:
        for name,value in previous.items():keys[name].value=value
        bpy.context.view_layer.update()
