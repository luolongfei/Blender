"""A new local2mm-per-side lower-jaw candidate on the adopted short R11 face.

Preserves the rounded central chin, lip volume, eye anatomy and low neck. It
does not replace the existing earlier jaw key, shorten the face again, alter
textures or install a whole-face scaling operation. Actual images decide.
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
KEY='R11 rounded lower jaw taper'
MARKER='R11_rounded_lower_jaw_taper_candidate'
MAXIMUM=.002
def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _sha(path):return _api('natural_skin_adult_r11')['_sha'](path)
def _array(data):return _api('lower_face_proportion_r11')['_array'](data)

def plan(current,T,strength=1.):
    if not 0.<strength<=1.:raise ValueError('Explicit0..1 taper strength required')
    skin=_api('natural_skin_adult_r11');donor,land,provenance=skin['_donor']()
    current=np.asarray(current,float);T=np.asarray(T,float);linear=T[:3,:3];scale=np.linalg.norm(linear[:,0])
    if current.shape!=(10582,3) or not np.allclose(linear,np.eye(3)*scale,atol=8e-7):raise RuntimeError('Actual full short-face cage in upright REST required')
    world=current@linear.T+T[:3,3];chin=world[7972].copy();mouth=world[8005].copy()
    if not 1.332<chin[2]<1.340 or not .027<mouth[2]-chin[2]<.041:raise RuntimeError('This taper is for the adopted shortened lower face')
    allowed=skin['_vertices_of_groups'](donor,{1,17,22});graph=skin['_graph'](world,donor['edges'],allowed)
    lip=np.asarray(land['mouth']['vermilionStripVertexIndices'],np.int32)
    distance=skin['_distance'](graph,lip,.024);smooth=skin['_smooth']
    x=world[:,0]-chin[0];z=world[:,2];y=world[:,1]
    # Lower face height is unchanged; side contour converges while the central
    #14mm either side retains the adopted rounded anterior chin projection.
    weight=smooth((abs(x)-.014)/.019)
    weight*=smooth((z-(chin[2]-.006))/.012)
    weight*=1.-smooth((z-(chin[2]+.017))/.025)
    weight*=1.-smooth((y+.005)/.030)
    weight*=smooth((distance-.009)/.014)
    protected=skin['_vertices_of_groups'](donor,{2,3,4,5,7,8})
    protected[lip]=True
    lip_support=np.zeros(len(world),bool);lip_support[lip]=True
    lip_support[np.unique(donor['edges'][lip_support[donor['edges']].any(1)])]=True
    protected|=lip_support|(abs(x)<=.014)|(z<=chin[2]-.006)|(z>=chin[2]+.042)|~allowed
    weight[protected]=0.
    delta_world=np.zeros_like(world);delta_world[:,0]=-np.sign(x)*MAXIMUM*strength*weight
    delta=(delta_world@np.linalg.inv(linear).T).astype(np.float32);changed=np.linalg.norm(delta,axis=1)>1e-10
    if not 80<int(changed.sum())<600:raise RuntimeError('Unexpected local lower-jaw scope: '+str(changed.sum()))
    edges=donor['edges'];old=current[edges[:,1]]-current[edges[:,0]];new=old+delta[edges[:,1]]-delta[edges[:,0]];active=changed[edges].any(1)
    ratio=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
    tri=_api('lower_face_proportion_r11')['_triangles'](donor['corner_vertices'],donor['face_starts'],donor['face_lengths']);tri=tri[changed[tri].any(1)]
    a=current[tri];b=(current+delta)[tri];n0=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);n1=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    ar0=np.linalg.norm(n0,axis=1);ar1=np.linalg.norm(n1,axis=1);valid=ar0>1e-14;align=np.sum(n0[valid]*n1[valid],axis=1)/(ar0[valid]*ar1[valid])
    if align.min()<=0 or ratio[active].min()<.75 or ratio[active].max()>1.25:raise RuntimeError('Raw local taper folded or strained')
    return delta,{'provenance':provenance,'strength':strength,'maximumRequestedPerSideMeters':MAXIMUM*strength,
        'maximumActualCageWorldMeters':float(np.linalg.norm(delta_world,axis=1).max()),'affectedCageVertices':int(changed.sum()),
        'currentChinAnchorWorld':chin.tolist(),'currentMouthAnchorWorld':mouth.tolist(),'pinnedCentralChinHalfWidthMeters':.014,
        'rawEdgeRatioQuantiles':np.quantile(ratio[active],[0,.05,.5,.95,1]).tolist(),'rawMinimumNormalAlignment':float(align.min()),
        'rawAreaRatioRange':[float((ar1[valid]/ar0[valid]).min()),float((ar1[valid]/ar0[valid]).max())],
        'scope':'Only localX convergence of the lower lateral face, based on the current shortened anatomy. Original static keys, rounded central chin projection, lip cage plus one ring and Y/Z are retained.',
        'requiredImages':['Front and quarter neutral against the same saved source','Full smile and side: retain cheek volume, rounded chin and continuous jaw-to-neck curve'],
        'appearanceAccepted':False}

def _validate(before,after,plan_report):
    a=before['data'];b=after['data'];v0=np.asarray(a['world']);v1=np.asarray(b['world']);faces=np.asarray(a['faces'],np.int32)
    if a['faces']!=b['faces'] or not np.array_equal(a['groups'],b['groups']):raise RuntimeError('Actual head topology/regions changed')
    delta=v1-v0;diff=np.linalg.norm(delta,axis=1);chin=np.asarray(plan_report['currentChinAnchorWorld']);mouth=np.asarray(plan_report['currentMouthAnchorWorld'])
    eye_ear=np.zeros(len(v0),bool);eye_ear[np.unique(faces[np.isin(a['groups'],[2,3,4,5])])]=True
    oral=np.zeros(len(v0),bool);oral[np.unique(faces[np.isin(a['groups'],[7,8])])]=True
    central=(abs(v0[:,0]-chin[0])<.010)&(v0[:,2]<mouth[2]-.006)&(v0[:,1]<-.075)
    low_neck=v0[:,2]<chin[2]-.010
    upper_face=v0[:,2]>chin[2]+.050
    protected=eye_ear|oral|central|low_neck|upper_face
    max_eye_ear=float(diff[eye_ear].max());max_protected=float(diff[protected].max())
    if max_eye_ear>2e-6 or max_protected>2e-5:raise RuntimeError('Actual protected eye/ear or chin/oral/neck surface moved: '+str([max_eye_ear,max_protected]))
    segments0=before['lipSections']['sections'];segments1=after['lipSections']['sections']
    gap=np.asarray([abs(x['minimumSurfaceGapMeters']-y['minimumSurfaceGapMeters']) for x,y in zip(segments0,segments1)])
    if gap.max()>1e-5 or any(x['sectionCrossings']!=y['sectionCrossings'] for x,y in zip(segments0,segments1)):
        raise RuntimeError('Local jaw taper changed actual lip contact')
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,3]],faces[:,[3,0]]]),axis=1)
    unique,count=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[count==1]);boundary_error=float(diff[boundary].max())
    if boundary_error>2e-6:raise RuntimeError('Actual neck seam moved')
    local=faces[(diff[faces]>1e-7).any(1)];tri=np.concatenate([local[:,[0,1,2]],local[:,[0,2,3]]]);p=v0[tri];q=v1[tri]
    n0=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n1=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);ar0=np.linalg.norm(n0,axis=1);ar1=np.linalg.norm(n1,axis=1);valid=ar0>1e-14
    align=np.sum(n0[valid]*n1[valid],axis=1)/(ar0[valid]*ar1[valid]);area=ar1[valid]/ar0[valid]
    if align.min()<=0 or area.min()<.5 or area.max()>1.6:raise RuntimeError('Actual local jaw patch folded/strained')
    bands=[]
    for offset in (.004,.011,.019,.027):
        level=chin[2]+offset;roi=(abs(v0[:,2]-level)<.0006)&(v0[:,1]<.025)
        x0=[float(v0[roi,0].min()),float(v0[roi,0].max())];x1=[float(v1[roi,0].min()),float(v1[roi,0].max())]
        bands.append({'worldZ':float(level),'beforeXBounds':x0,'afterXBounds':x1,'totalWidthReductionMeters':float((x0[1]-x0[0])-(x1[1]-x1[0]))})
    return {'actualMaximumWorldDisplacementMeters':float(diff.max()),'actualEyesEarsMaximumMeters':max_eye_ear,'actualProtectedMaximumMeters':max_protected,
        'actualNeckBoundaryMaximumMeters':boundary_error,'actualLipGapMaximumChangeMeters':float(gap.max()),
        'actualPatchMinimumNormalAlignment':float(align.min()),'actualPatchAreaRatioRange':[float(area.min()),float(area.max())],
        'actualLowerFaceWidthBands':bands,'appearanceAccepted':False,
        'limits':'Band widths and local continuity do not establish beauty or full expression compatibility. Actual neutral/full-smile front/quarter/side still required.'}

def apply(strength):
    import bpy
    low=_api('lower_face_proportion_r11');lip=_api('upper_lip_contour_r11');skin=_api('natural_skin_adult_r11');uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    qa=_api('face_contact_audit_r11');binding=_api('face_fibre_binding_r11');gentle=_api('gentle_smile_adult_r11')
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve());head=bpy.data.objects[HEAD];mesh=head.data;keys=mesh.shape_keys
    if MARKER in bpy.context.scene or KEY in keys.key_blocks or mesh.users!=1 or mesh.library or head.library:raise RuntimeError('Fresh independent real female source required')
    if any(abs(k.value)>1e-7 for k in keys.key_blocks if any(w in k.name.lower() for w in ('smile','blink'))):raise RuntimeError('Neutral expression required; retain static face keys')
    bpy.context.view_layer.update();prior=low['_keys'](mesh,uv,skin);contract=low['_nonkey'](head,face);external=gentle['_external_identity'](bpy,head)
    source_sha=_sha(source);active=head.active_shape_key_index;key=None;success=False
    try:
        with qa['rest_evaluation']():
            before=lip['_actual'](head);delta,report=plan(before['cage'],np.asarray(head.matrix_world),strength)
            key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=keys.reference_key;key.slider_min=0.;key.slider_max=1.;key.value=0.
            key.data.foreach_set('co',(_array(keys.reference_key.data)+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
            if np.max(np.linalg.norm(binding['_head_mesh'](head)['world']-np.asarray(before['data']['world']),axis=1))>2e-8:raise RuntimeError('Inactive taper changed geometry')
            key.value=1.;bpy.context.view_layer.update();after=lip['_actual'](head);validation=_validate(before,after,report);fibre=binding['validate'](neutral=False)
            key.value=0.;bpy.context.view_layer.update()
        now=low['_keys'](mesh,uv,skin);now['blocks'].pop(KEY)
        if now!=prior or gentle['_hash'](low['_nonkey'](head,face))!=gentle['_hash'](contract):raise RuntimeError('Original keys or head material/UV/weights/state changed')
        if gentle['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:raise RuntimeError('External rig/action/object or source changed')
        key.value=1.;bpy.context.view_layer.update();report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,helperSha256=_sha(__file__),
            key=KEY,keyValue=1.,newKeyCoordinatesSha256=skin['_ah'](_array(key.data)),originalActiveShapeKeyIndex=active,
            existingKeysPreserved=list(prior['blocks']),actualValidation=validation,actualFibreBindingValidation=fibre,savedBlend=False)
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            head.active_shape_key_index=active;bpy.context.view_layer.update()

def prepare(source_relative,output_relative,strength):
    import bpy
    low=_api('lower_face_proportion_r11');source=low['_path'](source_relative,'.blend');output=low['_path'](output_relative,'.blend');record=output.with_suffix('.jaw-taper.json')
    if output.exists() or record.exists() or source==output or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:raise RuntimeError('Fresh saved source and new independent output required')
    report=apply(strength)
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Jaw candidate save failed')
    report.update(output=str(output.relative_to(project_path())),outputSha256=_sha(output),savedBlend=True)
    with record.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_JAW_TAPER_SAVED '+json.dumps({'output':report['output'],'sha256':report['outputSha256']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--strength',type=float,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(a.source,a.output,a.strength)
