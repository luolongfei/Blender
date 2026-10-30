"""Independent neutral adult-eye surface study; never changes eyelid geometry.

The actual front-visible eye contour supplies a submillimeter colour/roughness
field on the CURRENT evaluated head. A packed point-index bank adds only two
attributes after Multires/Mask, so later same-topology expressions carry the
field with their vertices. Changing subdivision level or Mask requires a new
bank. apply()/restore() do not save; prepare() saves a distinct candidate.
No lashes, eyebrows, lights, globe position, skin source maps or optics change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

HEAD='R11 Female head study'
MARKER='R11_eye_margin_surface_candidate'
PREFIX='R11 eye margin surface'
UPPER='R11 upper eyelid surface tint'
LOWER='R11 lower eyelid surface tint'
PARAMETERS={'upperSupportRadiusMeters':.00042,'lowerSupportRadiusMeters':.00028,
    'upperRelativeLinearRGB':[.94,.915,.91],
    'lowerRelativeLinearRGB':[.995,.975,.970],
    'upperRoughnessReduction':.025,'lowerRoughnessReduction':.060,
    'irisRelativeReflectanceGain':[.82,.86,.92]}


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _array(items,field,width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype=dtype);items.foreach_get(field,a.ravel());return a
def _ah(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()
def _hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _path(relative):
    P=project_path().resolve();p=(P/relative).resolve();p.relative_to(P/'work/reconstruction-r02')
    if p.suffix!='.blend':raise ValueError('Independent reconstruction Blend path required')
    return p


def compact_polyline_field(points,polyline,radius):
    """Pure-array distance to segments, compact smooth falloff; not a blur."""
    points=np.asarray(points,float);line=np.asarray(polyline,float)
    best=np.full(len(points),np.inf)
    for a,b in zip(line[:-1],line[1:]):
        edge=b-a;length2=float(edge@edge)
        if length2<1e-16:continue
        t=np.clip((points-a)@edge/length2,0.,1.)
        best=np.minimum(best,np.linalg.norm(points-(a+t[:,None]*edge),axis=1))
    u=np.clip(best/radius,0.,1.)
    return (1.-u*u*(3.-2.*u)).astype(np.float32)


def _evaluated(obj):
    import bpy
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());me=ev.data
    co=_array(me.vertices,'co');m=np.asarray(ev.matrix_world,dtype=float)
    return {'co':co,'world':co@m[:3,:3].T+m[:3,3],
        'loops':_array(me.loops,'vertex_index',1,np.int32).ravel(),
        'starts':_array(me.polygons,'loop_start',1,np.int32).ravel(),
        'sizes':_array(me.polygons,'loop_total',1,np.int32).ravel(),
        'material':_array(me.polygons,'material_index',1,np.int32).ravel(),
        'matrix':m}


def _topology(data):return {k:_ah(data[k]) for k in ('loops','starts','sizes','material')}


def _fields(head,slot,data):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    def surface(d):
        faces=[tuple(d['loops'][s:s+n]) for s,n in zip(d['starts'],d['sizes'])]
        return BVHTree.FromPolygons([Vector(v) for v in d['world']],faces)
    skin=surface(data);direction=Vector((0,1,0));world=data['world']
    allowed=np.zeros(len(world),bool)
    for start,count,material in zip(data['starts'],data['sizes'],data['material']):
        if material==slot:allowed[data['loops'][start:start+count]]=True
    fields={UPPER:np.zeros(len(world),np.float32),LOWER:np.zeros(len(world),np.float32)};evidence={}
    for side in ('L','R'):
        eye=_evaluated(bpy.data.objects['R11 Adult sclera '+side]);globe=surface(eye)
        lo=eye['world'].min(0);hi=eye['world'].max(0)
        origin_y=float(min(world[:,1].min(),lo[1])-.1)
        def ray(bvh,x,z):return bvh.ray_cast(Vector((float(x),origin_y,float(z))),direction,.5)
        def visible(x,z):
            g=ray(globe,x,z)[0]
            if g is None:return False
            p=ray(skin,x,z)[0]
            return p is None or g.y<p.y-1e-7
        contour=[]
        for x in np.linspace(lo[0]+.00025,hi[0]-.00025,161):
            zs=np.linspace(lo[2]+.00008,hi[2]-.00008,97)
            ids=np.flatnonzero([visible(x,z) for z in zs])
            if len(ids)<3 or ids[0]==0 or ids[-1]==len(zs)-1:continue
            if np.any(np.diff(ids)!=1):raise RuntimeError('Non-contiguous visible eye aperture '+side)
            row=[float(x)]
            for i,j in ((int(ids[0]),int(ids[0]-1)),(int(ids[-1]),int(ids[-1]+1))):
                inside=float(zs[i]);outside=float(zs[j])
                for _ in range(13):
                    middle=(inside+outside)*.5
                    if visible(x,middle):inside=middle
                    else:outside=middle
                row.append((inside+outside)*.5)
            contour.append(row)
        contour=np.asarray(contour)
        if len(contour)<90:raise RuntimeError('Insufficient measured frontal aperture '+side)
        sign=1 if side=='L' else -1
        medial=contour[0,0] if sign==1 else contour[-1,0]
        lateral=contour[-1,0] if sign==1 else contour[0,0]
        side_record={}
        for label,column,dz,name,radius in (
            ('upper',2,1,UPPER,PARAMETERS['upperSupportRadiusMeters']),
            ('lower',1,-1,LOWER,PARAMETERS['lowerSupportRadiusMeters'])):
            line=[]
            for t in np.linspace(.035,.975,193):
                x=medial+(lateral-medial)*t;z=float(np.interp(x,contour[:,0],contour[:,column]))
                # The frontmost actual skin just outside the globe's aperture,
                # not the socket control loop or outer lash growth line.
                point,normal,index,distance=ray(skin,x,z+dz*.00006)
                if point is None:raise RuntimeError('Measured eye-margin skin ray miss')
                line.append(tuple(point))
            line=np.asarray(line);roi=allowed&np.all(world>=line.min(0)-radius,axis=1)&np.all(world<=line.max(0)+radius,axis=1)
            ids=np.flatnonzero(roi);values=compact_polyline_field(world[ids],line,radius)
            # Avoid colouring another fold or the inner socket across a narrow
            # three-dimensional gap. Only the same front-visible skin receives it.
            for k in np.flatnonzero(values>0):
                p=world[ids[k]];hit=ray(skin,p[0],p[2])[0]
                if hit is None or abs(float(hit.y)-p[1])>.00012:values[k]=0.
            if np.count_nonzero(values) < 20 or float(values.max())<.35:
                raise RuntimeError('Evaluated density cannot represent the bounded eye-margin field '+side+' '+label)
            fields[name][ids]=np.maximum(fields[name][ids],values)
            side_record[label]={'actualContourWorld':line.tolist(),'supportRadiusMeters':radius,
                'maximumFullSupportWidthMeters':2*radius,'affectedVertices':int(np.count_nonzero(values)),
                'maximumStrength':float(values.max()),'colourIsSkinRelative':True}
        evidence[side]=side_record
    if sum(np.count_nonzero(x) for x in fields.values())>8000:raise RuntimeError('Unexpectedly broad eye-margin field')
    return fields,evidence


def _bank_and_modifier(bpy,head,fields):
    name=PREFIX+' index bank';mesh=bpy.data.meshes.new(name)
    count=len(next(iter(fields.values())));mesh.vertices.add(count)
    mesh.vertices.foreach_set('co',np.zeros(count*3,np.float32));mesh.update()
    for label,values in fields.items():mesh.attributes.new(label,'FLOAT','POINT').data.foreach_set('value',values)
    bank=bpy.data.objects.new(name,mesh);head.users_collection[0].objects.link(bank);bank.hide_render=True
    bank.display_type='WIRE';bank['R11_non_surface_attribute_carrier']=True
    group=bpy.data.node_groups.new(PREFIX+' attributes only','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=group.nodes;links=group.links;source=nodes.new('NodeGroupInput');output=nodes.new('NodeGroupOutput')
    obj=nodes.new('GeometryNodeObjectInfo');obj.transform_space='ORIGINAL';obj.inputs['Object'].default_value=bank;obj.inputs['As Instance'].default_value=False
    index=nodes.new('GeometryNodeInputIndex');geometry=source.outputs['Geometry']
    for label in fields:
        read=nodes.new('GeometryNodeInputNamedAttribute');read.data_type='FLOAT';read.inputs['Name'].default_value=label
        sample=nodes.new('GeometryNodeSampleIndex');sample.data_type='FLOAT';sample.domain='POINT';sample.clamp=False
        links.new(obj.outputs['Geometry'],sample.inputs['Geometry']);links.new(read.outputs['Attribute'],sample.inputs['Value']);links.new(index.outputs['Index'],sample.inputs['Index'])
        write=nodes.new('GeometryNodeStoreNamedAttribute');write.data_type='FLOAT';write.domain='POINT';write.inputs['Name'].default_value=label
        links.new(geometry,write.inputs['Geometry']);links.new(sample.outputs['Value'],write.inputs['Value']);geometry=write.outputs['Geometry']
    links.new(geometry,output.inputs['Geometry'])
    modifier=head.modifiers.new(PREFIX+' attributes only','NODES');modifier.node_group=group
    return bank,group,modifier


def _skin_material(original):
    candidate=original.copy();candidate.name=PREFIX+' skin';nt=candidate.node_tree
    shader=next(n for n in nt.nodes if n.type=='BSDF_PRINCIPLED')
    color=shader.inputs['Base Color'];rough=shader.inputs['Roughness']
    if len(color.links)!=1 or len(rough.links)!=1:raise RuntimeError('Current skin must retain its linked colour and roughness maps')
    c=color.links[0].from_socket;r=rough.links[0].from_socket
    for label,colour_key,rough_key in ((UPPER,'upperRelativeLinearRGB','upperRoughnessReduction'),(LOWER,'lowerRelativeLinearRGB','lowerRoughnessReduction')):
        attribute=nt.nodes.new('ShaderNodeAttribute');attribute.attribute_name=label
        tint=nt.nodes.new('ShaderNodeMixRGB');tint.blend_type='MIX';tint.inputs[1].default_value=(1.,1.,1.,1.);tint.inputs[2].default_value=(*PARAMETERS[colour_key],1.)
        nt.links.new(attribute.outputs['Fac'],tint.inputs[0])
        product=nt.nodes.new('ShaderNodeMixRGB');product.blend_type='MULTIPLY';product.inputs[0].default_value=1.
        nt.links.new(c,product.inputs[1]);nt.links.new(tint.outputs[0],product.inputs[2]);c=product.outputs[0]
        amount=nt.nodes.new('ShaderNodeMath');amount.operation='MULTIPLY';amount.inputs[1].default_value=PARAMETERS[rough_key];nt.links.new(attribute.outputs['Fac'],amount.inputs[0])
        subtract=nt.nodes.new('ShaderNodeMath');subtract.operation='SUBTRACT';nt.links.new(r,subtract.inputs[0]);nt.links.new(amount.outputs[0],subtract.inputs[1]);r=subtract.outputs[0]
    nt.links.new(c,color);nt.links.new(r,rough)
    candidate['R11_eye_margin_scope']='Skin-relative submillimeter tint/roughness only; no wetline shell, geometry, normal or optical-layer changes.'
    return candidate


def _state(head,slot,skip_modifier=None):
    skin=_api('natural_skin_adult_r11');uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    state=skin['_head_state'](head,slot,uv,face);attr=_api('runtime_torso_cull_r07')
    # The pre-existing natural skin maps are protected here, too.
    # Built-in topology arrays are already covered explicitly above; their
    # internal INT32_2D representation is not a user pigment attribute.
    builtin={'position','.edge_verts','.corner_vert','.corner_edge'}
    state['attributes']=[(a.name,a.domain,a.data_type,_ah(attr['_attribute'](a))) for a in head.data.attributes if a.name not in builtin]
    state['modifiers']=[{k:v for k,v in uv['_scalar_rna'](m).items() if k!='is_active'} for m in head.modifiers if m.name!=skip_modifier]
    return state


def _external(bpy,head,eye_names,bank_name):
    rows=[]
    for o in bpy.context.scene.objects:
        if o==head or o.name==bank_name:continue
        rows.append({'name':o.name,'id':o.as_pointer(),'data':getattr(getattr(o,'data',None),'as_pointer',lambda:None)(),
            'parent':o.parent.as_pointer() if o.parent else None,'world':[list(r) for r in o.matrix_world],
            'materials':[] if o.name in eye_names else [s.material.as_pointer() if s.material else None for s in o.material_slots],
            'modifiers':[(m.name,m.type) for m in o.modifiers],
            'pose':[(b.name,[list(r) for r in b.matrix_basis]) for b in o.pose.bones] if o.type=='ARMATURE' else None})
    return _hash({'objects':rows,'actions':[(a.name,a.as_pointer()) for a in bpy.data.actions],
        'frame':bpy.context.scene.frame_current,'subframe':bpy.context.scene.frame_subframe})


def apply(head_name=HEAD,refine_iris=True):
    import bpy
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode source required')
    head=bpy.data.objects[head_name]
    if head.library or head.data.library or head.data.users!=1 or len(head.data.vertices)!=10582:raise RuntimeError('Complete local R11 adult head required')
    if MARKER in bpy.context.scene or any(x.name.startswith(PREFIX) for x in bpy.data.objects):raise RuntimeError('Retain/restore the earlier eye-margin candidate')
    if head.data.shape_keys and any(abs(k.value)>1e-8 for k in head.data.shape_keys.key_blocks[1:]):raise RuntimeError('Neutral new-head expression values required to author anatomical colour support')
    multires=[m for m in head.modifiers if m.type=='MULTIRES']
    if len(multires)!=1 or multires[0].levels!=3 or multires[0].render_levels!=3:raise RuntimeError('Actual viewport/render L3 topology must match the index bank')
    slots=[i for i,s in enumerate(head.material_slots) if s.material and s.material.name=='R11 natural warm skin and light makeup study']
    if len(slots)!=1:raise RuntimeError('The current natural-skin layer must be present and retained')
    slot=slots[0];original=head.material_slots[slot].material;qa=_api('face_contact_audit_r11');face=_api('runtime_face_refresh_r08')
    source_sha=_sha(source);before=_state(head,slot);eye_names=['R11 Adult iris '+s for s in ('L','R')]
    external=_external(bpy,head,eye_names,PREFIX+' index bank');old_material_hash=face['_material_contract'](original)
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'helperSha256':_sha(__file__),
        'head':head_name,'skinSlot':slot,'originalSkinMaterial':original.name,'parameters':PARAMETERS,
        'appearanceAccepted':False,'sourceSaved':False,'scope':'Submillimeter skin-relative eye-margin colour/roughness and optional copied iris colour multiplier only.',
        'limits':['Dense map is indexed to this exact L3 and Mask topology. Regenerate after Mask/subdivision/topology change; source guide bank has no render surface.',
            'Same-topology expressions carry these vertex attributes, but blink/gaze contact and lash binding remain unvalidated.',
            'No geometric tear meniscus or eyelid thickness correction is created. Thick pale eyelid shape remains a separate visible review item.',
            'Sclera, brows, lashes, source skin maps, original optical layers, eye geometry, face shape, lights, UV and animation are untouched.'],
        'requiredReview':'Same-light eyes macro and front/quarter. Reject if a dark drawn outline, pink stripe or excess shine replaces the pale rim. No acceptance from field statistics.'}
    assignments=[];materials=[];original_flags={};bank=group=modifier=None;success=False
    try:
        with qa['rest_evaluation']():
            data=_evaluated(head);fields,evidence=_fields(head,slot,data)
            bank,group,modifier=_bank_and_modifier(bpy,head,fields)
            candidate=_skin_material(original);materials.append(candidate)
            original_flags[original.as_pointer()]=original.use_fake_user
            assignments.append((head,slot,original,original_flags[original.as_pointer()]));original.use_fake_user=True;head.material_slots[slot].material=candidate
            iris_report=[]
            if refine_iris:
                cache={}
                for name in eye_names:
                    obj=bpy.data.objects[name]
                    if len(obj.material_slots)!=1:raise RuntimeError('Explicit single-material adult iris required')
                    old=obj.material_slots[0].material
                    if old.as_pointer() not in cache:
                        new=old.copy();new.name=PREFIX+' deeper brown iris';materials.append(new)
                        node=new.node_tree.nodes.get('R11 warm linear reflectance')
                        if not node or node.type!='MIX_RGB' or node.blend_type!='MULTIPLY' or node.inputs[2].is_linked:raise RuntimeError('Existing R11 iris reflectance node changed')
                        old_value=list(node.inputs[2].default_value);new_value=[old_value[i]*PARAMETERS['irisRelativeReflectanceGain'][i] for i in range(3)]+[old_value[3]]
                        node.inputs[2].default_value=new_value;cache[old.as_pointer()]=new
                        iris_report.append({'original':old.name,'candidate':new.name,'originalMultiplier':old_value,'candidateMultiplier':new_value})
                    original_flags.setdefault(old.as_pointer(),old.use_fake_user)
                    assignments.append((obj,0,old,original_flags[old.as_pointer()]));old.use_fake_user=True;obj.material_slots[0].material=cache[old.as_pointer()]
            bpy.context.view_layer.update();after=_evaluated(head)
            if _topology(after)!=_topology(data) or not np.array_equal(after['co'],data['co']) or not np.array_equal(after['matrix'],data['matrix']):raise RuntimeError('Attribute-only candidate changed actual evaluated geometry')
            evaluated=head.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
            for label,values in fields.items():
                attribute=evaluated.attributes.get(label)
                if not attribute or not np.array_equal(_array(attribute.data,'value',1).ravel(),values):raise RuntimeError('Actual GN attribute parity failed: '+label)
            report.update(actualContour=evidence,pointCount=len(data['co']),topology=_topology(data),actualGeometryParityMaxMeters=0.,
                candidateAttributeHashes={k:_ah(v) for k,v in fields.items()},actualAttributeParityExact=True,
                attributeBank=bank.name,attributeBankMesh=bank.data.name,nodeGroup=group.name,modifier=modifier.name,
                topologySettings=[{'name':m.name,'type':m.type,'levels':m.levels,'renderLevels':m.render_levels} for m in multires],
                originalSkinMaterialSha256=old_material_hash,iris=iris_report)
        after_state=_state(head,slot,modifier.name)
        if _hash(before)!=_hash(after_state):raise RuntimeError('Original head contract changed: '+json.dumps(_api('natural_skin_adult_r11')['_differences'](before,after_state)))
        if _external(bpy,head,eye_names,bank.name)!=external or face['_material_contract'](original)!=old_material_hash or _sha(source)!=source_sha:raise RuntimeError('External objects/actions, original skin or source file changed')
        report['assignments']=[{'object':o.name,'slot':s,'original':old.name,'originalFakeUser':fake,'candidate':o.material_slots[s].material.name,
            'candidateSha256':face['_material_contract'](o.material_slots[s].material),'originalSha256':face['_material_contract'](old)} for o,s,old,fake in assignments]
        report['validationScope']='Raw head coordinates/keys/UV/weights/maps and actual evaluated topology/positions; actual field equality; external object/data/action identities, materials and current rig pose. Full hair/cloth cache arrays are not re-read.'
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            for obj,index,old,fake in reversed(assignments):obj.material_slots[index].material=old;old.use_fake_user=fake
            if modifier:head.modifiers.remove(modifier)
            if bank:
                mesh=bank.data;bpy.data.objects.remove(bank,do_unlink=True)
                if mesh.users==0:bpy.data.meshes.remove(mesh)
            if group and group.users==0:bpy.data.node_groups.remove(group)
            for material in materials:
                if material.users==0:bpy.data.materials.remove(material)
            bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);face=_api('runtime_face_refresh_r08')
    for item in report['assignments']:
        obj=bpy.data.objects[item['object']];old=bpy.data.materials[item['original']];new=bpy.data.materials[item['candidate']]
        if obj.material_slots[item['slot']].material!=new or face['_material_contract'](new)!=item['candidateSha256'] or face['_material_contract'](old)!=item['originalSha256']:raise RuntimeError('Material changed after the candidate; retain those edits before restore')
    head=bpy.data.objects[report['head']];bank=bpy.data.objects[report['attributeBank']]
    for name,digest in report['candidateAttributeHashes'].items():
        if _ah(_array(bank.data.attributes[name].data,'value',1).ravel())!=digest:raise RuntimeError('Dense eye-margin map was edited; retain it before restore')
    for item in report['assignments']:
        old=bpy.data.materials[item['original']];new=bpy.data.materials[item['candidate']]
        bpy.data.objects[item['object']].material_slots[item['slot']].material=old;old.use_fake_user=item['originalFakeUser'];new.use_fake_user=True
    head.modifiers.remove(head.modifiers[report['modifier']]);group=bpy.data.node_groups[report['nodeGroup']];group.use_fake_user=True
    mesh=bank.data;mesh.use_fake_user=True;bpy.data.objects.remove(bank,do_unlink=True)
    del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'restoredOriginalAssignments':True,'retainedCandidateMaterialsAndMap':True,'sourceSaved':False}


def prepare(source_relative,output_relative,refine_iris=True):
    import bpy
    source=_path(source_relative);output=_path(output_relative);report_path=output.with_suffix('.eye-surface.json')
    if source==output or output.exists() or report_path.exists():raise RuntimeError('Use a new output; preserve all historical candidates')
    if Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh explicit source required')
    report=apply(refine_iris=refine_iris)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),sourceSaved=False,candidateSaved=True)
    with report_path.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_EYE_SURFACE_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--keep-iris',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,not a.keep_iris)
