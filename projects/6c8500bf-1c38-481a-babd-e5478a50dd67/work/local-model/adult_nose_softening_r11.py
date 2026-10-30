"""Local R11 nasal profile candidate on the real female topology.

Authored after actual9fc66 front/quarter inspection: soften the long straight
dorsum/tip wedge. This is one removable shape key, not an ethnicity template,
whole-face warp, automatic appearance acceptance or change to original data.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
HEAD='R11 Female head study'
KEY='R11 softer nasal profile'
MARKER='R11_softer_nasal_profile_candidate'
# Actual anterior centerline, from lower columella to nasal root. Localized
# posterior motion in meters; no8mm global nose/eye deformation from oldR02.
ANCHORS=[8010,8011,8012,8013,8014,8015,2832,8016,8017,8018,8068,8019]
BACK=np.array([0.,.0002,.0008,.0015,.0019,.0018,.0018,.0015,.0012,.0008,.0003,0.])
UP=np.array([0.,0.,.00015,.00035,.00045,.0003,.0001,0.,0.,0.,0.,0.])

def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):return _api('natural_skin_adult_r11')['_sha'](path)
def _array(data):return _api('lower_face_proportion_r11')['_array'](data)

def _profile(z,heights,values):
    out=np.zeros_like(z,dtype=float)
    for i in range(len(heights)-1):
        take=(z>=heights[i])&(z<=heights[i+1]);t=np.clip((z[take]-heights[i])/(heights[i+1]-heights[i]),0.,1.)
        t=t*t*(3.-2.*t);out[take]=values[i]*(1-t)+values[i+1]*t
    return out

def plan(current,T,strength=1.):
    if not 0.<strength<=1.:raise ValueError('Explicit0..1 nasal candidate strength required')
    skin=_api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']()
    current=np.asarray(current,float);T=np.asarray(T,float);linear=T[:3,:3]
    if current.shape!=(10582,3):raise RuntimeError('Complete actual female cage required')
    scale=np.linalg.norm(linear[:,0])
    if not np.allclose(linear,np.eye(3)*scale,atol=8e-7):raise RuntimeError('Upright REST source required')
    world=current@linear.T+T[:3,3];anchors=world[ANCHORS];heights=anchors[:,2]
    if np.any(np.diff(heights)<=0) or np.max(abs(anchors[:,0]))>.001 or not 1.398<world[8014,2]<1.405:
        raise RuntimeError('Actual nasal centerline differs; inspect the source')
    facial=skin['_vertices_of_groups'](donor,{17});graph=skin['_graph'](world,donor['edges'],facial)
    distance=skin['_distance'](graph,ANCHORS,.020)
    width=1.-skin['_smooth']((distance-.003)/.015)
    # Both directions share a compact surface field. The nasal-root and
    # columella endpoints are zero; neither cheeks nor lips are broadly scaled.
    delta_world=np.zeros_like(world)
    delta_world[:,1]=_profile(world[:,2],heights,BACK)*width
    delta_world[:,2]=_profile(world[:,2],heights,UP)*width
    pinned=skin['_vertices_of_groups'](donor,{2,3,4,5,7,8,22})
    lip=np.asarray(land['mouth']['vermilionStripVertexIndices'],np.int32)
    pinned[lip]=True
    # Protect the actual lip volume plus one support ring; the visible nasal
    # base-to-philtrum transition remains measured on the evaluated surface.
    lip_pin=np.zeros(len(world),bool);lip_pin[lip]=True
    lip_pin[np.unique(donor['edges'][lip_pin[donor['edges']].any(1)])]=True
    pinned|=lip_pin|~facial|(world[:,2]<=heights[0])|(world[:,2]>=heights[-1])
    delta_world[pinned]=0.;delta_world*=strength
    delta=(delta_world@np.linalg.inv(linear).T).astype(np.float32)
    selected=np.linalg.norm(delta,axis=1)>1e-10
    if not 30<int(selected.sum())<400 or np.linalg.norm(delta_world,axis=1).max()>.0021:
        raise RuntimeError('Unexpected local nasal scope/amplitude')
    tri=_api('lower_face_proportion_r11')['_triangles'](donor['corner_vertices'],donor['face_starts'],donor['face_lengths'])
    tri=tri[selected[tri].any(1)];a=current[tri];b=(current+delta)[tri]
    n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    area0=np.linalg.norm(n0,axis=1);area1=np.linalg.norm(n1,axis=1);valid=area0>1e-14
    alignment=np.sum(n0[valid]*n1[valid],axis=1)/(area0[valid]*area1[valid]);ratio=area1[valid]/area0[valid]
    if alignment.min()<=0 or ratio.min()<.5 or ratio.max()>1.6:raise RuntimeError('Raw nasal surface folded/strained')
    return delta,{'provenance':provenance,'strength':strength,'anchors':ANCHORS,'anchorWorldBefore':anchors.tolist(),
        'anchorWorldDelta':delta_world[ANCHORS].tolist(),'affectedCageVertices':int(selected.sum()),
        'maximumWorldDisplacementMeters':float(np.linalg.norm(delta_world,axis=1).max()),
        'rawPatchMinimumNormalAlignment':float(alignment.min()),'rawPatchAreaRatioRange':[float(ratio.min()),float(ratio.max())],
        'pinnedLipSupportCageVertices':int(lip_pin.sum()),'shapeIntent':'Reduce long straight nasal dorsum/tip wedge, retaining real nostrils, root and nasal-base support; no nostril fill, iris change, new mesh or texture repaint.',
        'requiredImages':['Same-light front,quarter,full-side at the same neutral expression','Nostril/base and lip profile close-up; reject crushed nostril walls, flattened tip or hard shoulder at nasal root'],
        'appearanceAccepted':False}

def _validate(before,after):
    a=before['data'];b=after['data'];v0=np.asarray(a['world']);v1=np.asarray(b['world']);faces=np.asarray(a['faces'],np.int32)
    if a['faces']!=b['faces'] or not np.array_equal(a['groups'],b['groups']):raise RuntimeError('Actual nasal topology/regions changed')
    diff=np.linalg.norm(v1-v0,axis=1)
    fixed=np.zeros(len(v0),bool);fixed[np.unique(faces[np.isin(a['groups'],[2,3,4,5,7,8,22])])]=True
    # This enclosure is deliberately wider than the changed nasal controls.
    # It tests all untouched face/neck surfaces, not just an isolated tip point.
    envelope=(abs(v0[:,0])<.035)&(v0[:,2]>1.382)&(v0[:,2]<1.441)&(v0[:,1]<-.075)
    fixed|=~envelope
    max_fixed=float(diff[fixed].max())
    if max_fixed>1e-5:raise RuntimeError('Actual lip/oral/eye/ear or outside-nose surface changed by more than10um: '+str(max_fixed))
    old=before['lipSections']['sections'];new=after['lipSections']['sections']
    gap=np.asarray([abs(x['minimumSurfaceGapMeters']-y['minimumSurfaceGapMeters']) for x,y in zip(old,new)])
    if gap.max()>1e-5 or any(x['sectionCrossings']!=y['sectionCrossings'] for x,y in zip(old,new)):
        raise RuntimeError('Actual lip-contact structure changed during the nose pass')
    region=faces[(diff[faces]>1e-7).any(1)];tri=np.concatenate((region[:,[0,1,2]],region[:,[0,2,3]]))
    p=v0[tri];q=v1[tri];n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
    ar0=np.linalg.norm(n0,axis=1);ar1=np.linalg.norm(n1,axis=1);valid=ar0>1e-14
    align=np.sum(n0[valid]*n1[valid],axis=1)/(ar0[valid]*ar1[valid]);ratio=ar1[valid]/ar0[valid]
    if align.min()<=0 or ratio.min()<.4 or ratio.max()>1.8:raise RuntimeError('Actual L3 nasal surface folded/strained')
    return {'actualMaximumWorldDisplacementMeters':float(diff.max()),'protectedActualMaximumMeters':max_fixed,
        'maximumLipContactGapChangeMeters':float(gap.max()),'actualPatchMinimumNormalAlignment':float(align.min()),
        'actualPatchAreaRatioRange':[float(ratio.min()),float(ratio.max())],'actualLipSectionsBefore':before['lipSections'],
        'actualLipSectionsAfter':after['lipSections'],'appearanceAccepted':False,
        'limits':'Checks actual unchanged regions, lip sections and local L3 triangle strain; nostril depth/wall contact and appearance still require real closeups. No full self-collision or final likeness claim.'}

def apply(strength):
    import bpy
    low=_api('lower_face_proportion_r11');lip=_api('upper_lip_contour_r11');skin=_api('natural_skin_adult_r11');uv=_api('production_uv_r08')
    face=_api('runtime_face_refresh_r08');qa=_api('face_contact_audit_r11');binding=_api('face_fibre_binding_r11');gentle=_api('gentle_smile_adult_r11')
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode source required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    if MARKER in bpy.context.scene or KEY in keys.key_blocks or mesh.users!=1 or head.library or mesh.library:raise RuntimeError('Fresh independent real female source required')
    if any(abs(k.value)>1e-7 for k in keys.key_blocks if any(w in k.name.lower() for w in ('smile','blink'))):raise RuntimeError('Use neutral expression; retain static proportion keys')
    bpy.context.view_layer.update();prior=low['_keys'](mesh,uv,skin);contract=low['_nonkey'](head,face)
    external=gentle['_external_identity'](bpy,head);source_sha=_sha(source);active=head.active_shape_key_index;key=None;success=False
    try:
        with qa['rest_evaluation']():
            before=lip['_actual'](head);delta,report=plan(before['cage'],np.asarray(head.matrix_world),strength)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',(_array(keys.reference_key.data)+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            inactive=binding['_head_mesh'](head)
            if np.max(np.linalg.norm(inactive['world']-np.asarray(before['data']['world']),axis=1))>2e-8:raise RuntimeError('Inactive nasal key changed neutral surface')
            key.value=1.;bpy.context.view_layer.update();after=lip['_actual'](head);validation=_validate(before,after);fibre=binding['validate'](neutral=False)
            key.value=0.;bpy.context.view_layer.update()
        now=low['_keys'](mesh,uv,skin);now['blocks'].pop(KEY)
        if now!=prior or gentle['_hash'](low['_nonkey'](head,face))!=gentle['_hash'](contract):raise RuntimeError('Original head coordinates/keys/maps/materials/UV/weights changed')
        if gentle['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('External rig/object/action or source changed')
        key.value=1.;bpy.context.view_layer.update()
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),key=KEY,keyValue=1.,
            newKeyCoordinatesSha256=skin['_ah'](_array(key.data)),originalActiveShapeKeyIndex=active,existingKeysPreserved=list(prior['blocks']),
            actualValidation=validation,actualFibreBindingValidation=fibre,currentMixMetadata=before['mixMetadata'],savedBlend=False)
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()

def prepare(source_relative,output_relative,strength):
    import bpy
    low=_api('lower_face_proportion_r11');source=low['_path'](source_relative,'.blend');output=low['_path'](output_relative,'.blend');record=output.with_suffix('.nose-contour.json')
    if output.exists() or record.exists() or source==output or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('New independent output and fresh explicit source required')
    report=apply(strength)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Nasal candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),savedBlend=True)
    with record.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_NOSE_SOFTENING_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--strength',type=float,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.strength)
