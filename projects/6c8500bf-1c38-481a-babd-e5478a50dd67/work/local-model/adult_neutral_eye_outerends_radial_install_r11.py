"""Install the reviewed angular outer-eye shape on the actual current source.

Current-source fields and true post-deformation tessellation are measured in
this process. A separately classified inner contact deficit is retained for
the next integrated midpath pass; this is not complete physical acceptance.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';KEY='R11 refined outer eye corners';WARM='R11 warm closed smile';MARKER='R11_refined_radial_outer_eye_candidate'
STATES=[{'id':'blink_'+str(int(b*100)),'L':b,'R':b,'warm':0.} for b in (0.,.2,.35,.5,.8,.9,1.)]+[{'id':'warm_early','L':.2,'R':.2,'warm':1.},{'id':'warm_late','L':.9,'R':.5,'warm':1.}]
HELPERS={'adult_neutral_eye_outerends_radial_r11_v3.py':'9a508aff3d1dfd60814fe56658216adbc219c13bed3314e44ca22efe617d1394',
         'adult_neutral_eye_outerends_radial_nodes_r11.py':'7753497ac3ecbe32136b505f2aa576ab693ba34d08fa3a3c200580da8811dda6'}

def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Independent reconstruction path required')
    return p
def fibres(binding):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();out={}
    for name in binding['OBJECTS']:
        obj=bpy.data.objects[name].evaluated_get(dg)
        out[name]=binding['_transform'](binding['_array'](obj.data.position_data,'vector'),obj.matrix_world)
    return out
def quality(a,b,tri,active):
    tri=tri[active[tri].any(1)];p=a[tri];q=b[tri]
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);l0=np.linalg.norm(n0,axis=1);l1=np.linalg.norm(n1,axis=1);ok=l0>1e-14
    cosine=np.sum(n0[ok]*n1[ok],axis=1)/(l0[ok]*l1[ok]);ratio=l1[ok]/l0[ok]
    if not len(l0):return {'changedTriangleCount':0}
    return {'changedTriangleCount':len(tri),'minimumNormalAlignment':float(cosine.min()),'areaRatioRange':[float(ratio.min()),float(ratio.max())],
            'relativeNormalOver90DegreeCount':int((cosine<=0).sum()),'minimumBeforeAreaSquareMm':float(l0.min()*.5e6),
            'minimumAfterAreaSquareMm':float(l1.min()*.5e6),'minimumAfterAreaSquareMeters':float(l1.min()*.5),
            'interpretation':'Relative normal rotation and area ratio are measurements, not a topological inversion test.'}

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

def actual_layers(before,after,eyes,contact):
    """Independent real-globe checks and covered inner-contact classification."""
    closest=api('upper_lid_relief_surface_r11')['closest_surface'];normal=contact['normals'](before['world'],before['triangles'])
    front_class=api('adult_blink_inner_thickness_support_r11')['FrontTriangleIndex'];front=front_class(after['world'],after['triangles'])
    active=np.linalg.norm(after['world']-before['world'],axis=1)>1e-9;rows={};pending=[]
    for side in ('L','R'):
        eye=eyes[side];center=(eye['world'].min(0)+eye['world'].max(0))*.5
        ids=np.flatnonzero(active & (np.sign(center[0])*before['world'][:,0]>0.))
        old=closest(before['world'][ids],eye['world'],eye['triangles']);new=closest(after['world'][ids],eye['world'],eye['triangles'])
        newly_inside=(old['signed']>=-2e-8)&(new['signed']<-2e-8);local=ids[newly_inside]
        exposed=[];inner=[]
        if len(local):
            hit_y,tri_rows,_=front.front(after['world'][local]);cover=after['world'][local,1]-hit_y
            hit_normal=np.zeros(len(local));valid=np.isfinite(hit_y)&(tri_rows>=0)
            hit_normal[valid]=front.normal_y[tri_rows[valid]]
            for j,idx in enumerate(local):
                record={'globalVertexId':int(after['originalVertexIds'][idx]),'beforeNormalY':float(normal[idx,1]),
                        'newActualSignedDistanceMeters':float(new['signed'][newly_inside][j]),
                        'frontAxisOuterCoverMeters':float(cover[j]) if valid[j] else None,
                        'coverTriangleNormalY':float(hit_normal[j]) if valid[j] else None}
                if normal[idx,1]>.1 and valid[j] and hit_normal[j]<-.1 and cover[j]>.00008:
                    inner.append(record);pending.append(record['globalVertexId'])
                else:exposed.append(record)
        rows[side]={'checkedChangedPoints':len(ids),'newInsideCount':int(newly_inside.sum()),'coveredInnerPending':inner,
                    'newExposedOrUnclassifiedInside':exposed,
                    'minimumAfterActualSignedMeters':float(new['signed'].min()) if len(ids) else None}
        if exposed:raise RuntimeError('Actual outer-eye shape introduced exposed/unclassified globe penetration: '+json.dumps(rows[side],default=serial))
    return rows,sorted(set(pending))

def prepare(source_relative,output_relative):
    import bpy
    source=path(source_relative,'.blend');output=path(output_relative,'.blend');report_path=output.with_suffix('.outer-eye.json');attempt=output.with_suffix('.outer-eye-attempt.json');payload=output.with_suffix('.outer-eye.npz')
    if source==output or any(p.exists() for p in (output,report_path,attempt,payload)) or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh current source and independent candidate required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if KEY in keys or MARKER in bpy.context.scene:raise RuntimeError('Retain existing outer-eye candidate')
    if 'R11 refined upper lid roll' not in keys or abs(keys['R11 refined upper lid roll'].value-1.)>1e-7:raise RuntimeError('Use the installed selected roll source or its intact successor')
    if any(m.type=='ARMATURE' for m in head.modifiers):raise RuntimeError('Current installation is unbound REST only; later cranial query migration is required')
    for filename,expected in HELPERS.items():
        if sha(Path(__file__).with_name(filename))!=expected:raise RuntimeError('Frozen reviewed helper changed: '+filename)
    review=path('work/reconstruction-r02/adult-neutral-eye-outerends-radial-r11-v3-review.json','.json')
    if sha(review)!='572914cf24ea26e2e72cc5abcb428c305276ef49ac28247753cd1bae1ee4b6c6':raise RuntimeError('Independent classified review changed')
    if not json.loads(review.read_text(encoding='utf8')).get('surfaceRefinementCandidateReady'):raise RuntimeError('Classified outer-surface candidate review required')
    controls=['R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile',WARM]
    if any(name not in keys or abs(keys[name].value)>1e-7 for name in controls):raise RuntimeError('Neutral public expressions required; static keys may remain active')
    if any(m.type=='NODES' and m.node_group and m.node_group.name.startswith('R11 middle blink arc and inner contact') for m in head.modifiers):raise RuntimeError('Rebuild midpath from the selected new static source; old middle layer is not a prerequisite')
    capture=api('adult_blink_recompute_r11');surface=api('adult_neutral_eye_outerends_radial_r11_v3');nodes=api('adult_neutral_eye_outerends_radial_nodes_r11');shared=api('upper_lid_relief_install_r11')
    qa=api('face_contact_audit_r11');uv=api('production_uv_r08');natural=api('natural_smile_adult_r11');groom=api('posterior_groom_r09');face=api('runtime_face_refresh_r08');binding=api('face_fibre_binding_r11_v2');contact=api('adult_blink_midpath_contact_r11')
    source_sha=sha(source);old_keys=natural['_key_records'](head.data,uv);old_data=groom['_data_hash'](head.data);old_mods=shared['modifier_contract'](head,uv);old_external=shared['external_contract'](bpy,head)
    old_materials=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots];old_active=head.active_shape_key_index
    modifier=group=cache=mesh=key=None;success=False;diagnostics={}
    report={'source':source_relative,'sourceSha256':source_sha,'helperHashes':{**HELPERS,Path(__file__).name:sha(__file__),
            'face_fibre_binding_r11_v2.py':sha(Path(__file__).with_name('face_fibre_binding_r11_v2.py'))},'scope':'New actual current-source angular outer-eye shape with independently classified inner-contact follow-up.','actualChecks':[],'candidateSaved':False,'technicalAcceptance':False,'allPhysicalContactPassed':False,'appearanceAccepted':False,
        'requiredInnerContactGlobalVertexIds':[206629],'requiredInnerContactPolygonIds':[196561,196562,196567,196568],
        'limits':['The current early fade can reopen the old outer slit: final middle blink must be rebuilt from this new source before motion acceptance.','Covered inner contact deficits require the next true-globe inner layer; this candidate is not complete physical acceptance.','Late-state exact preservation does not certify prior source lash or contact defects.','Actual eye A/B and a quarter/face view are required before visual adoption.']}
    def persist():
        np.savez_compressed(payload,**diagnostics);report['arrays']=str(payload.relative_to(project_path()));report['arraysSha256']=sha(payload);attempt.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=serial),encoding='utf-8')
    def values(state,strength=None):
        for name in controls:keys[name].value=state['warm'] if name==WARM else (state['L'] if name=='R11 blink L' else state['R'] if name=='R11 blink R' else 0.)
        if strength is not None:keys[KEY].value=strength
        head.update_tag();bpy.context.view_layer.update()
    try:
        with qa['rest_evaluation']():
            values(STATES[0]);neutral=capture['_snapshot'](head);topology=shared['polygon_contract'](head)
            eyes={s:capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+s]) for s in ('L','R')}
            contours={s:capture['_measure'](neutral,eyes[s]) for s in ('L','R')}
            fields,arrays,plan=surface['plan'](neutral['world'],neutral['triangles'],neutral['originalVertexIds'],eyes,contours)
            diagnostics.update(arrays);report.update(plan=plan,currentNeutralContours=contours,polygonContract=topology);persist()
            dense={s:np.zeros((len(neutral['fullWorld']),3)) for s in ('L','R')}
            for s in dense:dense[s][neutral['originalVertexIds']]=fields[s]
            selected=(dense['L'][:,0]>0)|(dense['R'][:,0]>0);baseline={};old_fibres={}
            for state in STATES:
                values(state);snap=capture['_snapshot'](head);baseline[state['id']]=snap
                if shared['polygon_contract'](head)!=topology:raise RuntimeError('Existing expressions change polygon connections')
                if min(state['L'],state['R'])>=.5:old_fibres[state['id']]=fibres(binding)
                diagnostics['before_'+state['id']]=snap['fullWorld']
                diagnostics['beforeTriangles_'+state['id']]=snap['triangles']
                print('R11_RADIAL_OUTER_EYE_BASELINE '+state['id'],flush=True)
            values(STATES[0]);key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',capture['_array'](head.data.shape_keys.reference_key.data).ravel())
            matrix=np.asarray(head.matrix_world,float);scale=float(np.linalg.norm(matrix[:3,0]))
            if not np.allclose(matrix[:3,:3],np.eye(3)*scale,atol=2e-7):raise RuntimeError('Current upright uniform REST head frame required')
            modifier,group,cache,mesh=nodes['build'](head,len(neutral['fullWorld']),neutral['originalVertexIds'],fields,scale,plan)
            values(STATES[0],0.);disabled=capture['_snapshot'](head)
            if not np.array_equal(disabled['fullWorld'],neutral['fullWorld']):raise RuntimeError('Zero new Key control failed exact bypass')
            for state in STATES:
                values(state,1.);before=baseline[state['id']];after=capture['_snapshot'](head);diagnostics['after_'+state['id']]=after['fullWorld']
                diagnostics['afterTriangles_'+state['id']]=after['triangles'];diagnostics['afterOriginalTriangleIds_'+state['id']]=after['originalTriangleIds'];persist()
                if shared['polygon_contract'](head)!=topology or not np.array_equal(before['originalVertexIds'],after['originalVertexIds']):raise RuntimeError('Actual angular layer changed true polygon topology/ROI')
                delta=after['fullWorld']-before['fullWorld'];amount=np.linalg.norm(delta,axis=1);late=min(state['L'],state['R'])>=.5
                if not np.array_equal(before['fullWorld'][~selected],after['fullWorld'][~selected]):raise RuntimeError('Actual angular layer moved points outside its true xEnvelope support')
                row={'state':state,'maximumActualMeters':float(amount.max()),'outsideSupportExact':True,'xMaximumMeters':float(abs(delta[:,0]).max()),'lateFullHeadExact':bool(np.array_equal(before['fullWorld'],after['fullWorld'])) if late else None}
                if row['xMaximumMeters']>2e-7:raise RuntimeError('Fixed-X angular mapping changed actual skin X')
                if late:
                    current=fibres(binding);row['lateSixFibresExact']=all(np.array_equal(old_fibres[state['id']][n],current[n]) for n in current)
                    if not row['lateFullHeadExact'] or not row['lateSixFibresExact']:raise RuntimeError('Late blink or six brow/lash outputs changed')
                else:
                    mapped={s:dense[s][before['originalVertexIds']] for s in dense}
                    expected,measure=surface['apply'](before['world'],before['triangles'],mapped,eyes,plan,blink={'L':state['L'],'R':state['R']})
                    actual=after['world']-before['world'];error=float(np.linalg.norm(actual-expected,axis=1).max());active=np.linalg.norm(actual,axis=1)>1e-10
                    row.update(pureNativeMaximumDifferenceMeters=error,physicalReference=measure)
                    diagnostics['expectedDelta_'+state['id']]=expected;diagnostics['nativeDelta_'+state['id']]=actual
                    if error>2e-6:raise RuntimeError('Native actual angular field differs from current-source pure reference: '+str(error))
                    if any(measure[s]['rejectedCount'] for s in measure):raise RuntimeError('Actual current radial support has a rejected active query')
                    # True current tessellation is the source for after-shape
                    # intersections. A legal diagonal change is not topology.
                    row['actualQuality']=quality(before['world'],after['world'],after['triangles'],active)
                    if row['actualQuality'].get('minimumAfterAreaSquareMeters',1.)<=1e-12:raise RuntimeError('Actual angular layer has a numerically degenerate triangle')
                    old_cross=contact['changed_intersections'](before['world'],before['triangles'],active)
                    new_cross=contact['changed_intersections'](after['world'],after['triangles'],active)
                    row.update(priorChangedTriangleCrossings=len(old_cross),newChangedTriangleCrossings=len(new_cross),actualPostTessellationUsed=True)
                    diagnostics['crossingPairs_'+state['id']]=np.asarray(sorted(new_cross),np.int32).reshape(-1,2)
                    if new_cross:raise RuntimeError('Actual current-tessellation angular surface has local triangle intersections')
                    row['actualGlobeAndLayers'],pending=actual_layers(before,after,eyes,contact)
                    report['requiredInnerContactGlobalVertexIds']=sorted(set(report['requiredInnerContactGlobalVertexIds'])|set(pending))
                    if state['id'] in ('blink_0','warm_early'):row['fibreBinding']=binding['validate'](neutral=False)
                report['actualChecks'].append(row);persist();print('R11_RADIAL_OUTER_EYE_ACTUAL '+json.dumps({'state':state['id'],'maximumMeters':row['maximumActualMeters']}),flush=True)
            values(STATES[0],1.);head.active_shape_key_index=old_active
        now=natural['_key_records'](head.data,uv);now['blocks'].pop(KEY)
        if now!=old_keys or groom['_data_hash'](head.data)!=old_data:raise RuntimeError('Original keys/drivers/raw data changed')
        if [face['_material_contract'](s.material) if s.material else None for s in head.material_slots]!=old_materials or shared['modifier_contract'](head,uv,modifier)!=old_mods or shared['external_contract'](bpy,head,cache)!=old_external or sha(source)!=source_sha:raise RuntimeError('Existing material/modifier/external/source state changed')
        report.update(controlKey=KEY,defaultStrength=1.,modifier=modifier.name,nodeGroup=group.name,lookupObject=cache.name,surfaceRefinementChecksPassed=True,modifierOrder=[m.name for m in head.modifiers]);persist();bpy.context.scene[MARKER]=json.dumps(report,default=serial)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent angular outer-eye candidate save failed')
        report.update(candidateSaved=True,output=str(output.relative_to(project_path())),outputSha256=sha(output))
        with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,default=serial)
        success=True;print('R11_RADIAL_OUTER_EYE_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True);return report
    except Exception as exc:report['guardError']=str(exc);persist();raise
    finally:
        if not success:
            values(STATES[0]);head.active_shape_key_index=old_active
            if modifier is not None:head.modifiers.remove(modifier)
            if key is not None:head.shape_key_remove(key)
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
            if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:bpy.data.node_groups.remove(group)
            bpy.context.view_layer.update()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output)
