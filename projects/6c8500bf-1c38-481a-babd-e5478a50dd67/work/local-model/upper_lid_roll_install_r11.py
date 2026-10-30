"""Install only the reviewed upper-roll candidate; outer-end drafts excluded.

Recompute its field from this actual current source, then validate neutral,
early blink and exact late-state exclusion. This is a static shape candidate;
the final blink path and inner contact require their next integrated pass.
"""
import argparse,hashlib,json,runpy,sys
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study';KEY='R11 refined upper lid roll';WARM='R11 warm closed smile';MARKER='R11_refined_upper_lid_roll_candidate'
STATES=[{'id':'blink_'+str(int(b*100)),'L':b,'R':b,'warm':0.} for b in (0.,.2,.5,.8,.9,1.)]+[{'id':'warm_early','L':.2,'R':.2,'warm':1.},{'id':'warm_late','L':.9,'R':.5,'warm':1.}]

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
    return {'minimumNormalAlignment':float(cosine.min()),'areaRatioRange':[float(ratio.min()),float(ratio.max())],'nonpositiveNormals':int((cosine<=0).sum())}

def prepare(source_relative,output_relative):
    import bpy
    source=path(source_relative,'.blend');output=path(output_relative,'.blend');report_path=output.with_suffix('.upper-roll.json');attempt=output.with_suffix('.upper-roll-attempt.json');payload=output.with_suffix('.upper-roll.npz')
    if source==output or any(p.exists() for p in (output,report_path,attempt,payload)) or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Fresh current source and independent candidate required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if KEY in keys or MARKER in bpy.context.scene:raise RuntimeError('Retain existing roll candidate')
    controls=['R11 blink L','R11 blink R','R11 gentle smile','R11 natural soft smile',WARM]
    if any(name not in keys or abs(keys[name].value)>1e-7 for name in controls):raise RuntimeError('Neutral public expressions required; static keys may remain active')
    if any(m.type=='NODES' and m.node_group and m.node_group.name.startswith('R11 middle blink arc and inner contact') for m in head.modifiers):raise RuntimeError('Rebuild midpath from the selected new static source; old middle layer is not a prerequisite')
    capture=api('adult_blink_recompute_r11');surface=api('upper_lid_roll_surface_r11');nodes=api('upper_lid_roll_nodes_r11');shared=api('upper_lid_relief_install_r11')
    qa=api('face_contact_audit_r11');uv=api('production_uv_r08');natural=api('natural_smile_adult_r11');groom=api('posterior_groom_r09');face=api('runtime_face_refresh_r08');binding=api('face_fibre_binding_r11');contact=api('adult_blink_midpath_contact_r11')
    source_sha=sha(source);old_keys=natural['_key_records'](head.data,uv);old_data=groom['_data_hash'](head.data);old_mods=shared['modifier_contract'](head,uv);old_external=shared['external_contract'](bpy,head)
    old_materials=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots];old_active=head.active_shape_key_index
    modifier=group=cache=mesh=key=None;success=False;diagnostics={}
    report={'source':source_relative,'sourceSha256':source_sha,'helperHashes':{n:sha(Path(__file__).with_name(n)) for n in ('upper_lid_roll_install_r11.py','upper_lid_roll_surface_r11.py','upper_lid_roll_nodes_r11.py')},'scope':'Upper-roll only. Rejected outer-end fields are neither imported nor installed.','actualChecks':[],'candidateSaved':False,'technicalAcceptance':False,'appearanceAccepted':False,
        'limits':['New neutral shape and local early-blink evidence only. Final blink/inner-contact integration remains required.','Late-state exact preservation does not certify prior source lash or contact defects.','Actual eye A/B and a quarter/face view are required before visual adoption.']}
    def persist():
        np.savez_compressed(payload,**diagnostics);report['arrays']=str(payload.relative_to(project_path()));report['arraysSha256']=sha(payload);attempt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def values(state,strength=None):
        for name in controls:keys[name].value=state['warm'] if name==WARM else (state['L'] if name=='R11 blink L' else state['R'] if name=='R11 blink R' else 0.)
        if strength is not None:keys[KEY].value=strength
        head.update_tag();bpy.context.view_layer.update()
    try:
        with qa['rest_evaluation']():
            values(STATES[0]);neutral=capture['_snapshot'](head);topology=shared['polygon_contract'](head)
            eyes={s:capture['_snapshot'](bpy.data.objects['R11 Adult sclera '+s]) for s in ('L','R')}
            contours={s:capture['_measure'](neutral,eyes[s]) for s in ('L','R')}
            weights,arrays,plan=surface['plan'](neutral['world'],neutral['triangles'],neutral['originalVertexIds'],eyes,contours)
            diagnostics.update(arrays);report.update(plan=plan,currentNeutralContours=contours,polygonContract=topology);persist()
            dense={s:np.zeros(len(neutral['fullWorld'])) for s in ('L','R')}
            for s in dense:dense[s][neutral['originalVertexIds']]=weights[s]
            selected=(dense['L']>0)|(dense['R']>0);baseline={};old_fibres={}
            for state in STATES:
                values(state);snap=capture['_snapshot'](head);baseline[state['id']]=snap
                if shared['polygon_contract'](head)!=topology:raise RuntimeError('Existing expressions change polygon connections')
                if min(state['L'],state['R'])>=.5:old_fibres[state['id']]=fibres(binding)
                diagnostics['before_'+state['id']]=snap['fullWorld']
                print('R11_UPPER_ROLL_BASELINE '+state['id'],flush=True)
            values(STATES[0]);key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=head.data.shape_keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',capture['_array'](head.data.shape_keys.reference_key.data).ravel())
            matrix=np.asarray(head.matrix_world,float);scale=float(np.linalg.norm(matrix[:3,0]))
            if not np.allclose(matrix[:3,:3],np.eye(3)*scale,atol=2e-7):raise RuntimeError('Current upright uniform REST head frame required')
            modifier,group,cache,mesh=nodes['build'](head,len(neutral['fullWorld']),neutral['originalVertexIds'],weights,scale)
            values(STATES[0],0.);disabled=capture['_snapshot'](head)
            if not np.array_equal(disabled['fullWorld'],neutral['fullWorld']):raise RuntimeError('Zero new Key control failed exact bypass')
            for state in STATES:
                values(state,1.);before=baseline[state['id']];after=capture['_snapshot'](head);diagnostics['after_'+state['id']]=after['fullWorld'];persist()
                if shared['polygon_contract'](head)!=topology or not np.array_equal(before['originalVertexIds'],after['originalVertexIds']):raise RuntimeError('Actual roll changed topology/ROI')
                delta=after['fullWorld']-before['fullWorld'];amount=np.linalg.norm(delta,axis=1);late=min(state['L'],state['R'])>=.5
                if not np.array_equal(before['fullWorld'][~selected],after['fullWorld'][~selected]):raise RuntimeError('Actual roll moved points outside its external upper-lid support')
                row={'state':state,'maximumActualMeters':float(amount.max()),'outsideSupportExact':True,'xZMaximumMeters':float(abs(delta[:,[0,2]]).max()),'lateFullHeadExact':bool(np.array_equal(before['fullWorld'],after['fullWorld'])) if late else None}
                if row['xZMaximumMeters']>2e-7 or amount.max()>.00065+2e-6:raise RuntimeError('Actual roll exceeds Y-only physical bounds')
                if late:
                    current=fibres(binding);row['lateSixFibresExact']=all(np.array_equal(old_fibres[state['id']][n],current[n]) for n in current)
                    if not row['lateFullHeadExact'] or not row['lateSixFibresExact']:raise RuntimeError('Late blink or six brow/lash outputs changed')
                else:
                    mapped={s:dense[s][before['originalVertexIds']] for s in dense}
                    expected,measure=surface['apply'](before['world'],before['triangles'],mapped,eyes,{'L':state['L'],'R':state['R']})
                    actual=after['world']-before['world'];error=float(np.linalg.norm(actual-expected,axis=1).max());active=np.linalg.norm(actual,axis=1)>1e-10
                    row.update(pureNativeMaximumDifferenceMeters=error,physicalReference=measure)
                    if error>2e-6:raise RuntimeError('Native actual roll differs from current-source pure reference: '+str(error))
                    row['actualQuality']=quality(before['world'],after['world'],before['triangles'],active)
                    if row['actualQuality']['nonpositiveNormals'] or row['actualQuality']['minimumNormalAlignment']<.5 or row['actualQuality']['areaRatioRange'][0]<.45 or row['actualQuality']['areaRatioRange'][1]>1.8:raise RuntimeError('Actual roll creates a fold or excess strain')
                    old_cross=contact['changed_intersections'](before['world'],before['triangles'],active);new_cross=contact['changed_intersections'](after['world'],before['triangles'],active)
                    added=new_cross-old_cross;row.update(priorChangedTriangleCrossings=len(old_cross),newChangedTriangleCrossings=len(new_cross),addedCrossings=len(added))
                    if added:raise RuntimeError('Actual external roll creates new local triangle intersections')
                    row['fibreBinding']=binding['validate'](neutral=False)
                report['actualChecks'].append(row);persist();print('R11_UPPER_ROLL_ACTUAL '+json.dumps({'state':state['id'],'maximumMeters':row['maximumActualMeters']}),flush=True)
            values(STATES[0],1.);head.active_shape_key_index=old_active
        now=natural['_key_records'](head.data,uv);now['blocks'].pop(KEY)
        if now!=old_keys or groom['_data_hash'](head.data)!=old_data:raise RuntimeError('Original keys/drivers/raw data changed')
        if [face['_material_contract'](s.material) if s.material else None for s in head.material_slots]!=old_materials or shared['modifier_contract'](head,uv,modifier)!=old_mods or shared['external_contract'](bpy,head,cache)!=old_external or sha(source)!=source_sha:raise RuntimeError('Existing material/modifier/external/source state changed')
        report.update(controlKey=KEY,defaultStrength=1.,modifier=modifier.name,nodeGroup=group.name,lookupObject=cache.name,localGeometryChecksPassed=True,modifierOrder=[m.name for m in head.modifiers]);persist();bpy.context.scene[MARKER]=json.dumps(report)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent roll candidate save failed')
        report.update(candidateSaved=True,output=str(output.relative_to(project_path())),outputSha256=sha(output))
        with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        success=True;print('R11_UPPER_ROLL_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True);return report
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
