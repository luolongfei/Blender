"""Install one reversible actual-L3 lower-lip / mental-slope candidate.

Uses the reviewed actual f501 two-state observation, allowing newer hair,
material or other disjoint changes. Only actual support plus two-ring geometry
is a dependency; an unrelated whole-Blend checksum is never used as a lock.
Runs native current-source neutral/half/full-warm checks, saves diagnostics
before a guard can fail, then saves only a fresh independent candidate.
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
HEAD='R11 Female head study';WARM='R11 warm closed smile';CONTROL='R11 lower face continuity'
MARKER='R11_lower_face_continuity_candidate'
DEFAULT_OBSERVATION='work/reconstruction-r02/lower-face-current-surface-r11.json'


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Explicit independent reconstruction artifact required')
    return p


def snapshot(head):
    import bpy
    dg=bpy.context.evaluated_depsgraph_get();obj=head.evaluated_get(dg);mesh=obj.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
    a=api('lower_face_surface_observe_r11')['array']
    try:
        sizes=a(mesh.polygons,'loop_total',1,np.int32).ravel()
        if not np.all(sizes==4):raise RuntimeError('Current fixed-L3 quad head required')
        faces=a(mesh.loops,'vertex_index',1,np.int32).ravel().reshape(-1,4)
        matrix=np.asarray(obj.matrix_world,float);world=a(mesh.vertices,'co',3)@matrix[:3,:3].T+matrix[:3,3]
        region=mesh.attributes.get('.sculpt_face_set');lip=mesh.attributes.get('R11 anatomical lip colour support')
        if region is None or region.domain!='FACE' or lip is None or lip.domain!='POINT':raise RuntimeError('Actual anatomical attributes required')
        return {'world':world,'faces':faces,'groups':a(region.data,'value',1,np.int32).ravel(),
            'lipSupport':a(lip.data,'value',1).ravel(),'matrix':matrix}
    finally:obj.to_mesh_clear()


def dependency_check(now,reference,arrays,dep_ids,full_topology_hash):
    ah=api('lower_face_surface_observe_r11')['ah']
    if ah(now['faces'])!=full_topology_hash:raise RuntimeError('Actual fixed-L3 polygon connection differs from observed topology')
    ids=reference['originalVertexIds'];index=np.searchsorted(ids,dep_ids)
    if np.any(index>=len(ids)) or not np.array_equal(ids[index],dep_ids):raise RuntimeError('Observed support neighborhood missing from expression ROI')
    amount=np.linalg.norm(now['world'][dep_ids]-reference['world'][index],axis=1)
    lip_delta=np.max(abs(now['lipSupport'][dep_ids]-reference['lipSupport'][index]))
    if amount.max()>2e-7 or lip_delta>2e-7:raise RuntimeError('Current lower-face support differs from actual observation: '+str(float(amount.max())))
    return {'dependencyActualVertices':len(dep_ids),'maximumWorldDifferenceMeters':float(amount.max()),'maximumLipSupportDifference':float(lip_delta)}


def actual_check(before,after,offset,arrays,surface,control,warm,oral,eyes,boundary):
    if not np.array_equal(before['faces'],after['faces']) or not np.array_equal(before['groups'],after['groups']):raise RuntimeError('Native continuity field changed actual topology/anatomical regions')
    expected=offset*control;delta=after['world']-before['world'];active=np.any(offset!=0,axis=1);actual=np.linalg.norm(delta,axis=1)
    expected_error=float(np.linalg.norm(delta-expected,axis=1).max());outside=float(actual[~active].max())
    outside_exact=bool(np.array_equal(before['world'][~active],after['world'][~active]))
    if control==0 and not np.array_equal(before['world'],after['world']):raise RuntimeError('Zero continuity control does not return the exact original geometry')
    if expected_error>3e-7 or not outside_exact:raise RuntimeError('Native actual field differs from reviewed support/vector: '+str([expected_error,outside]))
    groups={'innerOral':oral,'eyeRegion':eyes,'trueOpenNeckBoundary':boundary,
        'protectedObservedContactChinBase':arrays['originalVertexIds'][arrays['protected']]}
    protected={name:{'actualVertices':len(ids),'exact':bool(np.array_equal(before['world'][ids],after['world'][ids])),
        'maximumDifferenceMeters':float(actual[ids].max()) if len(ids) else 0.} for name,ids in groups.items()}
    if any(not row['exact'] for row in protected.values()):raise RuntimeError('Actual protected inner mouth/eye/contact/chin-base/neck moved: '+str(protected))
    xz=float(abs(delta[:,[0,2]]).max())
    if xz>2e-7:raise RuntimeError('Height or width was changed by Y-only continuity field')
    report={'warmSmile':warm,'continuityControl':control,'maximumActualDisplacementMeters':float(actual.max()),
        'expectedVectorMaximumErrorMeters':expected_error,'outsideSupportMaximumMeters':outside,'outsideSupportExact':outside_exact,'xZMaximumDifferenceMeters':xz,
        'protectedRegions':protected,'fullPolygonConnectionExact':True}
    if control:
        quality=surface['quality'](before['world'],after['world'],before['faces'],active)
        report['actualSurfaceQuality']=quality
        if quality['nonpositiveNormalCount'] or quality['minimumNormalAlignment']<.75 or quality['triangleAreaRatioRange'][0]<.55 or quality['triangleAreaRatioRange'][1]>1.7:
            raise RuntimeError('Native actual lower-face field folds/strains: '+str(quality))
    return report


def prepare(source_relative,output_relative,observation_relative=DEFAULT_OBSERVATION):
    import bpy
    source=path(source_relative,'.blend');output=path(output_relative,'.blend');observation=path(observation_relative,'.json')
    report_path=output.with_suffix('.lower-face-continuity.json');attempt=output.with_suffix('.lower-face-continuity-attempt.json');payload=output.with_suffix('.lower-face-continuity.npz')
    if source==output or any(p.exists() for p in (output,report_path,attempt,payload)) or not bpy.app.background or bpy.data.is_dirty or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Fresh saved source and fresh independent candidate paths required')
    head=bpy.data.objects[HEAD];keys=head.data.shape_keys.key_blocks
    if CONTROL in head or MARKER in bpy.context.scene or WARM not in keys:raise RuntimeError('Existing warm smile and a fresh continuity candidate required')
    if any(abs(k.value)>1e-7 for k in keys if 'blink' in k.name.lower() or 'smile' in k.name.lower() or k.name=='R11 jaw open'):
        raise RuntimeError('Neutral current blink/smile/jaw values required')
    metadata=json.loads(observation.read_text(encoding='utf-8'));observed_path=path(metadata['arrays'],'.npz')
    if sha(observed_path)!=metadata['arraysSha256']:raise RuntimeError('Observed actual-surface payload changed')
    with np.load(observed_path) as npz:data={k:npz[k].copy() for k in npz.files}
    surface=api('lower_face_continuity_surface_r11');_,land,_=api('natural_skin_adult_r11')['_donor']()
    arrays,plan=surface['plan'](data,land);active_roi=np.any(arrays['worldOffsets']!=0,axis=1);neighborhood=active_roi.copy()
    for _ in range(2):neighborhood[np.unique(arrays['faces'][neighborhood[arrays['faces']].any(1)])]=True
    dep_ids=arrays['originalVertexIds'][neighborhood]
    uv=api('production_uv_r08');natural=api('natural_smile_adult_r11');groom=api('posterior_groom_r09');face=api('runtime_face_refresh_r08')
    shared=api('upper_lid_relief_install_r11');qa=api('face_contact_audit_r11')
    source_sha=sha(source);key_before=natural['_key_records'](head.data,uv);data_before=groom['_data_hash'](head.data)
    material_before=[face['_material_contract'](s.material) if s.material else None for s in head.material_slots]
    modifiers_before=shared['modifier_contract'](head,uv);external_before=shared['external_contract'](bpy,head);active_key=head.active_shape_key_index
    modifier=group=cache=mesh=None;success=False
    report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,'observation':observation_relative,
        'observationSourceSha256':metadata['sourceSha256'],'observationArraysSha256':metadata['arraysSha256'],
        'helperHashes':{name:sha(Path(__file__).with_name(name)) for name in ('lower_face_continuity_install_r11.py','lower_face_continuity_surface_r11.py','lower_face_continuity_nodes_r11.py')},
        'plan':plan,'actualChecks':[],'dependencyChecks':[],'candidateSaved':False,'technicalAcceptance':False,'appearanceAccepted':False,
        'limits':['Local current neutral/half/full-warm surface checks do not certify all speech/jaw combinations.',
            'Fixed mouth contact and inner surfaces preserve their prior condition; this does not certify existing oral intersections are resolved.',
            'The candidate requires same-light front/quarter/side inspection before appearance acceptance.']}
    diagnostic={key:value for key,value in arrays.items()};diagnostic['dependencyOriginalVertexIds']=dep_ids
    def persist():
        np.savez_compressed(payload,**diagnostic);report['arrays']=str(payload.relative_to(project_path()));report['arraysSha256']=sha(payload)
        attempt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def values(smile,control=None):
        keys[WARM].value=smile
        if control is not None:head[CONTROL]=control
        head.update_tag();bpy.context.view_layer.update()
    try:
        persist()
        with qa['rest_evaluation']():
            baseline={}
            for smile in (0.,.5,1.):
                values(smile);baseline[smile]=snapshot(head)
                diagnostic['beforeWarm'+str(smile)+'World']=baseline[smile]['world']
                if smile in (0.,1.):
                    reference=surface['state'](data,'state_'+str(int(smile)))
                    report['dependencyChecks'].append(dependency_check(baseline[smile],reference,arrays,dep_ids,metadata['states'][int(smile)]['fullTopologySha256']))
                print('R11_LOWER_FACE_CONTINUITY_BASELINE '+str(smile),flush=True)
            values(0.)
            now=baseline[0.];full_count=len(now['world']);offset=np.zeros_like(now['world']);offset[arrays['originalVertexIds']]=arrays['worldOffsets']
            oral=np.unique(now['faces'][np.isin(now['groups'],[7,8])]);eyes=np.unique(now['faces'][np.isin(now['groups'],[2,3])]);boundary=data['state_0_openBoundaryOriginalVertexIds']
            if np.any(offset[oral]) or np.any(offset[eyes]) or np.any(offset[boundary]):raise RuntimeError('Planned field overlaps protected actual anatomy')
            head[CONTROL]=0.;head.id_properties_ui(CONTROL).update(min=0.,max=1.,description='Reversible external lower-lip to chin continuity. Neutral height, mouth contact and chin base retained.')
            modifier,group,cache,mesh=api('lower_face_continuity_nodes_r11')['build'](head,full_count,arrays['originalVertexIds'],arrays['worldOffsets'],now['matrix'])
            values(0.,0.);disabled=snapshot(head);diagnostic['disabledWorld']=disabled['world'];persist()
            report['actualChecks'].append(actual_check(now,disabled,offset,arrays,surface,0.,0.,oral,eyes,boundary))
            for smile in (0.,.5,1.):
                values(smile,1.);changed=snapshot(head);diagnostic['afterWarm'+str(smile)+'World']=changed['world'];persist()
                row=actual_check(baseline[smile],changed,offset,arrays,surface,1.,smile,oral,eyes,boundary)
                report['actualChecks'].append(row);persist()
                print('R11_LOWER_FACE_CONTINUITY_ACTUAL '+json.dumps({'warm':smile,'maximumMeters':row['maximumActualDisplacementMeters']}),flush=True)
            values(0.,1.);head.active_shape_key_index=active_key
        if natural['_key_records'](head.data,uv)!=key_before or groom['_data_hash'](head.data)!=data_before:raise RuntimeError('Original keys/drivers/raw head data changed')
        if [face['_material_contract'](s.material) if s.material else None for s in head.material_slots]!=material_before:raise RuntimeError('Original head materials changed')
        if shared['modifier_contract'](head,uv,modifier)!=modifiers_before or shared['external_contract'](bpy,head,cache)!=external_before:raise RuntimeError('Existing modifier/scene/pose changed')
        if sha(source)!=source_sha:raise RuntimeError('Original saved source changed')
        report.update(control=CONTROL,controlValue=1.,modifier=modifier.name,nodeGroup=group.name,lookupObject=cache.name,
            modifierOrder=[m.name for m in head.modifiers],localGeometryChecksPassed=True,originalKeysDriversRawDataMaterialsPreserved=True,
            technicalAcceptance=False,appearanceAccepted=False)
        persist();bpy.context.scene[MARKER]=json.dumps(report)
        if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Independent continuity candidate save failed')
        report.update(candidateSaved=True,output=str(output.relative_to(project_path())),outputSha256=sha(output))
        with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        success=True;print('R11_LOWER_FACE_CONTINUITY_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
        return report
    except Exception as exc:
        report['guardError']=str(exc);persist();raise
    finally:
        if not success:
            keys[WARM].value=0.;head.active_shape_key_index=active_key
            if modifier is not None:head.modifiers.remove(modifier)
            if CONTROL in head:del head[CONTROL]
            if MARKER in bpy.context.scene:del bpy.context.scene[MARKER]
            if cache is not None:bpy.data.objects.remove(cache,do_unlink=True)
            if mesh is not None and mesh.users==0:bpy.data.meshes.remove(mesh)
            if group is not None and group.users==0:bpy.data.node_groups.remove(group)
            bpy.context.view_layer.update()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--observation',default=DEFAULT_OBSERVATION)
    a=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.observation)
