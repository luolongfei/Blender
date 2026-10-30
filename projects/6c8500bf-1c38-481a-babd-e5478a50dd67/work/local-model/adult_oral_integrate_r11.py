"""Create fitted CC0 teeth, gingiva and tongue on the current native adult jaw.

Adds editable derived meshes and retains the exact imported donor hidden.
Preserves donor UVs when present, source face/vertex IDs and the old oral
assets. The head's continuous original oral bag remains; donor bag157 is not
duplicated. Production native jaw controls all lower parts, with a softer
posterior attachment for gingiva and tongue. No main-source overwrite.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HEAD='R11 Female head study';JAW='R11 jaw open';MARKER='R11_complete_oral_asset_fit'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def array(items,field='co',width=3,dtype=np.float32):
    a=np.empty((len(items),width),dtype);items.foreach_get(field,a.ravel());return a
def transform(v,m):
    m=np.asarray(m,float);return np.asarray(v)@m[:3,:3].T+m[:3,3]


def material(name,kind):
    import bpy
    m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;l=m.node_tree.links
    bs=next(n for n in n if n.type=='BSDF_PRINCIPLED')
    if kind=='teeth':colour=(.67,.62,.52,1.);rough=.26;sss=.045;radius=(1.,.65,.35);scale=.00045
    elif kind=='tongue':colour=(.28,.065,.074,1.);rough=.36;sss=.18;radius=(1.,.35,.20);scale=.0010
    else:colour=(.24,.054,.063,1.);rough=.31;sss=.16;radius=(1.,.4,.25);scale=.0010
    bs.inputs['Base Color'].default_value=colour;bs.inputs['Roughness'].default_value=rough
    bs.inputs['Subsurface Weight'].default_value=sss;bs.inputs['Subsurface Radius'].default_value=radius;bs.inputs['Subsurface Scale'].default_value=scale
    bs.inputs['IOR'].default_value=1.52 if kind=='teeth' else 1.38;bs.inputs['Specular IOR Level'].default_value=.4
    bs.inputs['Coat Weight'].default_value=.06;bs.inputs['Coat Roughness'].default_value=.12
    if kind=='tongue':
        tex=n.new('ShaderNodeTexCoord');noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=1400.;noise.inputs['Detail'].default_value=2.
        l.new(tex.outputs['Object'],noise.inputs['Vector']);bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00006
        l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
    m['R11_scope']='Physical oral material candidate; natural enamel / moist mucosa, not appearance acceptance.'
    return m


def create_part(name,row,arrays,donor,collection,parent,head,jaw_report,materials):
    import bpy
    from mathutils import Matrix
    ids=arrays[name+'_source_ids'];fi=arrays[name+'_source_face_ids'];world=arrays[name+'_world'];local=transform(world,np.linalg.inv(np.asarray(parent.matrix_world,float)))
    loops=arrays[name+'_loops'];starts=arrays[name+'_face_starts'];sizes=arrays[name+'_face_sizes'];faces=[loops[s:s+c].tolist() for s,c in zip(starts,sizes)]
    mesh=bpy.data.meshes.new(row['objectName']+' mesh');mesh.from_pydata(local.tolist(),[],faces);mesh.update()
    obj=bpy.data.objects.new(row['objectName'],mesh);collection.objects.link(obj);obj.parent=parent;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
    for poly in mesh.polygons:poly.use_smooth=True
    a=mesh.attributes.new('R11 oral donor vertex ID','INT','POINT');a.data.foreach_set('value',ids)
    a=mesh.attributes.new('R11 oral donor face ID','INT','FACE');a.data.foreach_set('value',fi)
    a=mesh.attributes.new('.sculpt_face_set','INT','FACE');a.data.foreach_set('value',arrays[name+'_face_sets'])
    source_loop_ids=np.concatenate([np.arange(donor.data.polygons[int(i)].loop_start,donor.data.polygons[int(i)].loop_start+donor.data.polygons[int(i)].loop_total) for i in fi])
    for original in donor.data.uv_layers:
        uv=mesh.uv_layers.new(name=original.name);raw=array(original.data,'uv',2);uv.data.foreach_set('uv',raw[source_loop_ids].ravel())
    original_crease=donor.data.attributes.get('crease_edge')
    if original_crease is not None:
        values=array(original_crease.data,'value',1).ravel();source_edges={tuple(sorted(e.vertices)):float(values[e.index]) for e in donor.data.edges}
        new=mesh.attributes.new('crease_edge','FLOAT','EDGE');new.data.foreach_set('value',np.asarray([source_edges.get(tuple(sorted((int(ids[e.vertices[0]]),int(ids[e.vertices[1]])))),0.) for e in mesh.edges],np.float32))
    kind='teeth' if name.endswith('teeth') else ('tongue' if name=='tongue' else 'gingiva');mesh.materials.append(materials[kind])
    sub=obj.modifiers.new('R11 oral fine surface','SUBSURF');sub.subdivision_type='CATMULL_CLARK';sub.levels=2;sub.render_levels=2
    if name in ('lower_gingiva','tongue'):
        # The dental margin/tongue tip follows the mandible fully. Posterior
        # mucosa blends toward its pharyngeal attachment instead of being rigid.
        t=np.clip((world[:,1]+.045)/.015,0,1);t=t*t*(3-2*t);weights=(1.-.7*t).astype(np.float32)
        attribute=mesh.attributes.new('R11 mandibular rotation weight','FLOAT','POINT');attribute.data.foreach_set('value',weights)
        tree=api('adult_jaw_open_r11')['make_nodes'](head,jaw_report['estimatedPivotLocal'],jaw_report['axisLocal'],jaw_report['translationLocal'])
        mod=obj.modifiers.new('R11 oral mandibular arc','NODES');mod.node_group=tree
        obj['R11_posterior_jaw_weight_minimum']=float(weights.min())
    obj['R11_oral_donor_part']=name;obj['R11_source_face_sets']=json.dumps(row['sourceFaceSets']);obj['R11_editable_derived_mesh']=True
    return obj


def actual_clearance(head,objects,strength):
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    head.data.shape_keys.key_blocks[JAW].value=strength;bpy.context.view_layer.update();qa=api('face_contact_audit_r11');h=qa['_evaluated_mesh'](head)
    bvh=BVHTree.FromPolygons([Vector(p) for p in h['world']],h['faces'],all_triangles=False);rows={}
    for name,obj in objects.items():
        data=qa['_evaluated_mesh'](obj);world=data['world'];sample_ids=np.arange(len(world)) if strength==0 else np.unique(np.linspace(0,len(world)-1,min(3000,len(world))).astype(np.int32))
        signed=[]
        for i in sample_ids:
            co=Vector(world[i]);hit,normal,index,distance=bvh.find_nearest(co)
            if hit is None:raise RuntimeError('Actual oral/head nearest query failed')
            signed.append(float((co-hit).dot(normal.normalized())))
        s=np.asarray(signed);rows[name]={'evaluatedVertices':len(world),'sampledVertices':len(sample_ids),
            'signedHeadDistanceMetersP0P01P50P99P100':np.percentile(s,[0,1,50,99,100]).tolist(),
            'pointsInsideHeadBeyondHalfMillimeter':int((s<-.0005).sum()),'worldBounds':[world.min(0).tolist(),world.max(0).tolist()]}
    return {'jawStrength':strength,'objects':rows,'scope':'Actual evaluated Subsurf2 vertices versus actual current L3 head BVH. Neutral uses all oral vertices; open states use at most3000 per part. Not full edge-face collision or visual acceptance.'}


def prepare(source_relative,output_relative,plan_relative):
    import bpy
    source=(P/source_relative).resolve();output=(P/output_relative).resolve();rp=output.with_suffix('.oral-fit.json');planpath=(P/plan_relative).resolve()
    for path in (source,output,rp,planpath):path.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh independent source required')
    if output.exists() or rp.exists() or MARKER in bpy.context.scene:raise RuntimeError('Retain prior oral candidates')
    plan=json.loads(planpath.read_text(encoding='utf8'));ap=P/plan['arrays']
    if sha(ap)!=plan['arraysSha256']:raise RuntimeError('Oral fit plan arrays changed')
    with np.load(ap) as n:arrays={k:n[k].copy() for k in n.files}
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if JAW not in keys or 'R11_native_mandibular_motion' not in bpy.context.scene:raise RuntimeError('Install the actual native adult jaw before dental fitting')
    if any(abs(k.value)>1e-7 for k in keys if k.name.startswith(('R11 blink','R11 natural soft smile','R11 warm closed smile',JAW))):raise RuntimeError('Neutral face required for dental fitting')
    jaw_report=json.loads(bpy.context.scene['R11_native_mandibular_motion']);jaw_parent=bpy.data.objects[jaw_report['attachmentCarrier']];cranial=bpy.data.objects[jaw_report['cranialParent']]
    donor_path=P/plan['donor']
    if sha(donor_path)!=plan['donorSha256']:raise RuntimeError('Exact installed CC0 donor required')
    original_head=api('adult_head_bind_observe_r11')['data_contract'](head);uv=api('production_uv_r08');pose=uv['_pose_contract']();before_file=sha(source)
    pools=('objects','collections','meshes','materials','node_groups','images','shape_keys');old={name:{v.as_pointer() for v in getattr(bpy.data,name)} for name in pools}
    saved=False;report={}
    try:
        with api('face_contact_audit_r11')['rest_evaluation']():
            mix,_=uv['current_mix'](head);current=transform(mix,head.matrix_world);skin=api('natural_skin_adult_r11');topology,land,_=skin['_donor']()
            ids=np.flatnonzero(skin['_vertices_of_groups'](topology,{7,8}));error=float(np.linalg.norm(current[ids]-arrays['source_observed_cage_world'][ids],axis=1).max())
            if error>2e-6:raise RuntimeError('Actual neutral oral cage changed; refit before appending dental data: '+str(error))
            with bpy.data.libraries.load(str(donor_path),link=False) as (src,dst):dst.objects=['Jaw - Realistic']
            donor=dst.objects[0]
            if donor is None or donor.type!='MESH' or not np.array_equal(array(donor.data.vertices),arrays['donor_reference_local']):raise RuntimeError('Actual donor differs from observed original mesh')
            backup=bpy.data.collections.new('R11 original oral donor retained');bpy.context.scene.collection.children.link(backup);backup.objects.link(donor)
            donor.name='R11 Original CC0 Jaw donor (hidden)';donor.hide_render=True;donor.hide_viewport=True;donor.hide_set(True);donor.data.use_fake_user=True
            donor['R11_source_path']=plan['donor'];donor['R11_source_sha256']=plan['donorSha256'];donor['R11_license']='CC0'
            col=bpy.data.collections.new('R11 Adult complete oral anatomy');bpy.context.scene.collection.children.link(col)
            mats={k:material('R11 natural '+k+' material',k) for k in ('teeth','gingiva','tongue')}
            objects={}
            for name,row in plan['parts'].items():
                if row['objectName'] in bpy.data.objects:raise RuntimeError('Retain existing derived oral object '+row['objectName'])
                parent=jaw_parent if name=='lower_teeth' else cranial
                objects[name]=create_part(name,row,arrays,donor,col,parent,head,jaw_report,mats)
            bpy.context.view_layer.update();checks=[actual_clearance(head,objects,s) for s in (0.,.5,1.)]
            keys[JAW].value=0.;bpy.context.view_layer.update()
            neutral=checks[0]['objects'];critical={name:r for name,r in neutral.items() if r['signedHeadDistanceMetersP0P01P50P99P100'][0]<(-.0003 if name.endswith('teeth') else -.0005)}
            report={'source':source_relative,'sourceSha256':before_file,'plan':plan_relative,'planSha256':sha(planpath),'planArraysSha256':plan['arraysSha256'],
                'donor':plan['donor'],'donorSha256':plan['donorSha256'],'license':'CC0','originalDonorObject':donor.name,
                'parts':plan['parts'],'actualNeutralOralCageErrorMeters':error,'actualClearanceChecks':checks,
                'newObjects':[o.name for o in objects.values()],'originalHeadBagRetained':True,'oldOralObjectsRetainedHidden':True,
                'subdivisionLevels':2,'uvPolicy':'Exact original donor corner UVs transferred per retained source face when UV layers exist.',
                'jawBehavior':'Lower teeth on mandibular carrier; lower gingiva and tongue use same native rotation with posterior mucosal weighting. Upper teeth/gingiva follow cranial carrier.',
                'helperSha256':sha(__file__),'appearanceAccepted':False,'fullOralCollisionPassed':False,
                'criticalNeutralClearanceFailures':critical,
                'limits':['Requires genuine neutral/open/smile mouth images before adoption.','Sparse open-state vertex tests do not prove continuous self-collision freedom.',
                          'Existing upper inner-oral versus lower-lip intersections are a separate head-surface defect, not hidden or repaired by the dental assets.',
                          'Warm-smile and phoneme controls require new combined checks; these measurements cover jaw alone.']}
            if critical:
                report['savedBlend']=False
                with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
                raise RuntimeError('Actual subdivided neutral oral geometry requires refitting; diagnostic report saved')
        if api('adult_head_bind_observe_r11')['data_contract'](head)!=original_head or uv['_pose_contract']()!=pose or sha(source)!=before_file:raise RuntimeError('Dental fitting changed original head, pose or source')
        bpy.context.scene[MARKER]=json.dumps(report,ensure_ascii=False)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Oral candidate save failed')
        report.update(output=output_relative,outputSha256=sha(output),savedBlend=True)
        with rp.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        saved=True;print('R11_ORAL_FIT_SAVED '+json.dumps({'output':output_relative,'sha256':report['outputSha256']}),flush=True)
    finally:
        keys[JAW].value=0.
        if not saved:
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            fresh=[v for name in pools for v in getattr(bpy.data,name) if v.as_pointer() not in old[name]]
            if fresh:bpy.data.batch_remove(ids=fresh)
        bpy.context.view_layer.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);prepare(a.source,a.output,a.plan)
