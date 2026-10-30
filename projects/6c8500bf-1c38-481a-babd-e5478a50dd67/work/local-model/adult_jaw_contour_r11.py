"""Independent lower-jaw contour candidate for the actual R11 female head.

One bounded, removable static shape key; no eye/nose/lip/neck translation,
mesh replacement, material change or import-time Blender work. Existing keys
and their relative bases are retained. This is a candidate for actual images,
not an assertion that a narrower face is automatically a better face.
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
KEY='R11 lower jaw soft contour'
MARKER='R11_lower_jaw_contour_candidate'
MAXIMUM_WORLD=.0032


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _array(items):
    a=np.empty((len(items),3),np.float32);items.foreach_get('co',a.ravel());return a


def _path(relative,suffix):
    p=(project_path()/relative).resolve();p.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if p.suffix!=suffix:raise ValueError('Independent reconstruction artifact required')
    return p


def plan(current_basis,female_to_target):
    skin=_api('natural_skin_adult_r11');donor,landmarks,provenance=skin['_donor']()
    neutral=np.asarray(current_basis,np.float32);points=np.asarray(donor['local'],float)
    T=np.asarray(female_to_target,float);linear=T[:3,:3];scale=float(np.linalg.norm(linear[:,0]))
    if neutral.shape!=(10582,3) or not np.allclose(linear.T@linear,np.eye(3)*scale*scale,atol=1e-8):
        raise RuntimeError('Actual female cage and uniform source mapping required')
    mouth=np.asarray(landmarks['mouth']['openingRimCandidate']['sourceVertexIndices'],np.int32)
    lip=np.asarray(landmarks['mouth']['vermilionStripVertexIndices'],np.int32)
    center=points[mouth].mean(0);x=points[:,0]-center[0];y=points[:,1];z=points[:,2]
    lower=skin['_vertices_of_groups'](donor,{22})
    chin_candidates=lower&(abs(x)<.018)&(y<-.09)
    chin=float(points[chin_candidates,2].min());height=float(center[2]-chin)
    q=(z-chin)/height
    # Maximum at the lateral lower jaw, below the mouth. The upper cheek,
    # central chin and entire mouth remain structurally distinct from it.
    weight=np.exp(-((q-.48)/.43)**4)
    weight*=skin['_smooth']((q+.06)/.23)*(1.-skin['_smooth']((q-.94)/.30))
    weight*=skin['_smooth']((abs(x)-.012)/.026)
    weight*=skin['_smooth']((-y-.032)/.055)
    allowed=skin['_vertices_of_groups'](donor,{1,17,22})
    graph=skin['_graph'](points,donor['edges'],allowed)
    distance=skin['_distance'](graph,lip,.024)
    weight*=skin['_smooth']((distance-.008)/.016)
    protected=skin['_vertices_of_groups'](donor,{2,3,4,5,7,8})
    protected[lip]=True
    protected|=(z>=center[2]+.015)|(z<=chin-.004)|(abs(x)<=.012)
    weight[~allowed|protected]=0.
    delta=np.zeros_like(points);delta[:,0]=-np.sign(x)*MAXIMUM_WORLD/scale*weight
    target=delta@linear.T;selected=np.linalg.norm(delta,axis=1)>1e-10
    maximum=float(np.linalg.norm(target,axis=1).max())
    if maximum>MAXIMUM_WORLD+1e-9 or not 150<selected.sum()<850:
        raise RuntimeError('Unexpected jaw-candidate scope or amplitude')
    edges=donor['edges'];old=neutral[edges[:,1]].astype(float)-neutral[edges[:,0]]
    new=old+delta[edges[:,1]]-delta[edges[:,0]]
    affected=selected[edges].any(1);ratio=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
    tris=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face=donor['corner_vertices'][start:start+count]
        if selected[face].any():tris.extend((int(face[0]),int(face[k]),int(face[k+1])) for k in range(1,len(face)-1))
    tris=np.asarray(tris,np.int32);a=neutral[tris].astype(float);b=(neutral+delta)[tris]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    area0=np.linalg.norm(na,axis=1);area1=np.linalg.norm(nb,axis=1);valid=area0>1e-12
    alignment=np.sum(na[valid]*nb[valid],axis=1)/(area0[valid]*area1[valid])
    if alignment.min()<=0 or ratio[affected].min()<.75 or ratio[affected].max()>1.25:
        raise RuntimeError('Jaw candidate raw cage folds or changes edge lengths excessively')
    report={'provenance':provenance,'maximumWorldDisplacementMeters':maximum,
        'affectedVertices':int(selected.sum()),'sourceMouthCenter':center.tolist(),'sourceChinFloor':chin,
        'worldChangedBounds':[(neutral[selected]@linear.T+T[:3,3]).min(0).tolist(),(neutral[selected]@linear.T+T[:3,3]).max(0).tolist()],
        'rawEdgeRatioQuantiles':np.quantile(ratio[affected],[0,.05,.5,.95,1]).tolist(),
        'rawNormalAlignmentMinimum':float(alignment.min()),'rawAreaRatioRange':[float((area1[valid]/area0[valid]).min()),float((area1[valid]/area0[valid]).max())],
        'lipNoseEyesEarsAndNeckPinned':True,'centralChinProjectionUnchanged':True,
        'appearanceAccepted':False,'evaluatedMultiresReviewPending':True,
        'scope':'Only X-directed lower-lateral face convergence below the mouth; up to 3.2 mm world per side. Y/Z, central chin, actual lip bodies, nose and upper cheeks are not scaled.',
        'requiredReview':['Same-light front and quarter against the source; reject if jaw becomes pointed, pinched or visually older.',
            'Side view must retain lip/chin projection and a continuous submandibular-to-neck contour.',
            'Inspect neutral and existing gentle smile separately; this is not a smile or a completed facial-expression system.']}
    return delta.astype(np.float32),report


def _keys(mesh):
    if not mesh.shape_keys:return None
    return {k.name:{'coords':_array(k.data),'relative':k.relative_key.name,'value':k.value,'mute':k.mute,
        'min':k.slider_min,'max':k.slider_max,'group':k.vertex_group} for k in mesh.shape_keys.key_blocks}


def apply(head_name=HEAD):
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    source=Path(bpy.data.filepath).resolve();source.relative_to((project_path()/'work/reconstruction-r02').resolve())
    head=bpy.data.objects[head_name];mesh=head.data
    if mesh.library or mesh.users!=1 or len(mesh.vertices)!=10582 or bpy.context.scene.get(MARKER):
        raise RuntimeError('New independent R11 full-cage candidate required')
    skin=_api('natural_skin_adult_r11');smile=_api('gentle_smile_adult_r11');uv=_api('production_uv_r08');face=_api('runtime_face_refresh_r08')
    donor,_,_=skin['_donor']()
    if not np.array_equal(skin['_array'](mesh.loops,'vertex_index',1,np.int32).ravel(),donor['corner_vertices']):
        raise RuntimeError('Female author topology changed')
    prior=_keys(mesh);neutral=_array(mesh.shape_keys.reference_key.data if mesh.shape_keys else mesh.vertices)
    T=np.asarray(json.loads(head['R11_female_to_target']),float);delta,report=plan(neutral,T)
    bpy.context.view_layer.update();before=smile['_nonkey_head_state'](head,uv,face,skin)
    external=smile['_external_identity'](bpy,head);active=head.active_shape_key_index;source_sha=_sha(source)
    created_basis=mesh.shape_keys is None;key=None;success=False
    try:
        if created_basis:head.shape_key_add(name='Basis',from_mix=False)
        basis=mesh.shape_keys.reference_key
        key=head.shape_key_add(name=KEY,from_mix=False);key.relative_key=basis;key.slider_min=0.;key.slider_max=1.;key.value=0.
        key.data.foreach_set('co',(neutral+delta).ravel());head.active_shape_key_index=active;bpy.context.view_layer.update()
        after=smile['_nonkey_head_state'](head,uv,face,skin)
        if smile['_hash'](after)!=smile['_hash'](before):
            raise RuntimeError('Original non-key head changed: '+json.dumps(skin['_differences'](before,after)))
        if prior:
            now=_keys(mesh)
            for name,old in prior.items():
                new=now[name]
                if not np.array_equal(old['coords'],new['coords']) or any(old[k]!=new[k] for k in old if k!='coords'):
                    raise RuntimeError('Existing key or relative basis changed: '+name)
        if smile['_external_identity'](bpy,head)!=external or _sha(source)!=source_sha:
            raise RuntimeError('Other object/rig/attachment state or source file changed')
        key.value=1.;bpy.context.view_layer.update()
        report.update(source=str(source.relative_to(project_path())),sourceSha256=source_sha,head=head_name,key=KEY,keyValue=1.,
            createdBasis=created_basis,originalActiveShapeKeyIndex=active,helperSha256=_sha(Path(__file__)),
            basisSha256=skin['_ah'](neutral),newKeySha256=skin['_ah'](_array(key.data)),deltaSha256=skin['_ah'](delta),
            existingKeysUnchanged=list(prior) if prior else [],writtenInMemory=True,savedBlend=False)
        bpy.context.scene[MARKER]=json.dumps(report);success=True;return report
    finally:
        if not success:
            if key is not None and mesh.shape_keys and KEY in mesh.shape_keys.key_blocks:head.shape_key_remove(mesh.shape_keys.key_blocks[KEY])
            if created_basis and mesh.shape_keys and len(mesh.shape_keys.key_blocks)==1:head.shape_key_clear()
            head.active_shape_key_index=active;bpy.context.view_layer.update()


def restore():
    import bpy
    if MARKER not in bpy.context.scene:return False
    report=json.loads(bpy.context.scene[MARKER]);head=bpy.data.objects[report['head']];key=head.data.shape_keys.key_blocks[KEY]
    if _api('natural_skin_adult_r11')['_ah'](_array(key.data))!=report['newKeySha256']:
        raise RuntimeError('Retain edited contour key before restoring')
    head.shape_key_remove(key)
    if report['createdBasis'] and len(head.data.shape_keys.key_blocks)==1:head.shape_key_clear()
    head.active_shape_key_index=report['originalActiveShapeKeyIndex'];del bpy.context.scene[MARKER];bpy.context.view_layer.update()
    return {'removedKey':KEY,'savedBlend':False}


def prepare(source_relative,output_relative):
    import bpy
    source=_path(source_relative,'.blend');output=_path(output_relative,'.blend');report_path=output.with_suffix('.json')
    if source==output or output.exists() or report_path.exists() or Path(bpy.data.filepath).resolve()!=source or bpy.data.is_dirty:
        raise RuntimeError('Fresh explicit source and new output/report required')
    report=apply()
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(output)):raise RuntimeError('Candidate save failed')
    report.update(savedBlend=True,output=str(output.relative_to(project_path())),outputSha256=_sha(output))
    with report_path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('R11_JAW_CONTOUR_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
