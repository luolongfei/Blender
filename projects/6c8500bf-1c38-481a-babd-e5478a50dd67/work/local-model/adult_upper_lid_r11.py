"""Independent static upper-lid coverage candidate for the real R11 eye fit.

Rotates a continuous upper-lid tissue neighborhood along the unchanged globe,
with pinned lower lids/canthi. Requires the actual brow/lash surface binding.
It is not a blink implementation or proof of all-expression eye contact.
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
KEY='R11 gentle upper lid coverage'
MARKER='R11_upper_lid_coverage_candidate'
MAXIMUM_ANGLE=.075


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
    neutral=np.asarray(current_basis,np.float32);T=np.asarray(female_to_target,float)
    world=neutral@T[:3,:3].T+T[:3,3]
    if neutral.shape!=(10582,3):raise RuntimeError('Complete R11 female cage required')
    audit_path=project_path()/'work/reconstruction-r02/adult-eyelid-visible-audit-r11.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    with np.load(project_path()/'work/reconstruction-r02/adult-neck-observation-r11.npz') as source:
        original=source['rest_00_raw_world']
    eye_zone=(world[:,2]>1.410)&(world[:,2]<1.463)&(abs(world[:,0])<.065)&(world[:,1]<-.060)
    if np.max(np.abs(world[eye_zone]-original[eye_zone]))>3e-7:
        raise RuntimeError('Eye-region Basis changed; renew the actual visible-contour observation')
    allowed=skin['_vertices_of_groups'](donor,{2,3,17})
    graph=skin['_graph'](world,donor['edges'],allowed)
    displacement=np.zeros_like(world);side_reports={}
    for side,row in audit['eyes'].items():
        contour=row['contour'];xs=np.array([q['x'] for q in contour]);cent=np.asarray(row['centerWorld'])
        # Diagnostic neighbors seed a smooth local tissue field only. They
        # are not treated as a new rim loop, contact line or closure boundary.
        seeds=sorted({int(i) for q in contour[10:-10] for i in q['upper']['nearestRawDiagnosticVertices'][:2]})
        distance=skin['_distance'](graph,seeds,.012)
        weight=1.-skin['_smooth']((distance-.0028)/.0082)
        t=(world[:,0]-xs[0])/(xs[-1]-xs[0]);across=np.sin(np.pi*np.clip(t,0.,1.))**.8
        corner=skin['_smooth']((t-.07)/.13)*(1.-skin['_smooth']((t-.80)/.13))
        # Pin the complete lower lid and taper on the inner upper-fold side;
        # both anterior sheets of upper-lid thickness share the same rotation.
        upper=skin['_smooth']((world[:,2]-cent[2]-.0006)/.0015)
        anterior=skin['_smooth']((cent[1]-world[:,1]-.0005)/.004)
        weight*=across*corner*upper*anterior
        weight[~allowed]=0.
        theta=MAXIMUM_ANGLE*weight;v=world-cent;delta=np.zeros_like(v)
        delta[:,1]=v[:,1]*np.cos(theta)-v[:,2]*np.sin(theta)-v[:,1]
        delta[:,2]=v[:,1]*np.sin(theta)+v[:,2]*np.cos(theta)-v[:,2]
        displacement+=delta
        selected=np.linalg.norm(delta,axis=1)>1e-10
        side_reports[side]={'seedDiagnosticIndices':seeds,'affectedRawVertices':int(selected.sum()),
            'maximumWorldDisplacementMeters':float(np.linalg.norm(delta,axis=1).max()),
            'maximumRotationRadians':float(theta.max()),'centralUpperCoverageBeforeMeters':row['centralUpperIrisCoverage'],
            'rotationAxisWorld':[1,0,0],'globeCenterUnchanged':cent.tolist()}
    delta=displacement@np.linalg.inv(T[:3,:3]).T
    protected_ids=sorted({int(i) for row in landmarks['eyes'].values() for i in row['orderedLowerInnerToOuter']})
    if np.any(delta[protected_ids]!=0.):raise RuntimeError('Original lower socket/canthus controls moved')
    selected=np.linalg.norm(delta,axis=1)>1e-10
    if [side_reports[s]['affectedRawVertices'] for s in ('L','R')]!=[42,42] or np.linalg.norm(displacement,axis=1).max()>.0013:
        raise RuntimeError('Unexpected upper-lid scope or amplitude')
    edges=donor['edges'];old=neutral[edges[:,1]].astype(float)-neutral[edges[:,0]];new=old+delta[edges[:,1]]-delta[edges[:,0]]
    affected=selected[edges].any(1);ratio=np.linalg.norm(new,axis=1)/np.maximum(np.linalg.norm(old,axis=1),1e-12)
    tris=[]
    for start,count in zip(donor['face_starts'],donor['face_lengths']):
        face=donor['corner_vertices'][start:start+count]
        if selected[face].any():tris.extend((int(face[0]),int(face[k]),int(face[k+1])) for k in range(1,len(face)-1))
    tris=np.asarray(tris,np.int32);a=neutral[tris].astype(float);b=(neutral+delta)[tris]
    na=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);nb=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]);aa=np.linalg.norm(na,axis=1);bb=np.linalg.norm(nb,axis=1);valid=aa>1e-12
    alignment=np.sum(na[valid]*nb[valid],axis=1)/(aa[valid]*bb[valid])
    if alignment.min()<=0 or ratio[affected].min()<.70 or ratio[affected].max()>1.30:
        raise RuntimeError('Upper-lid candidate raw cage fold or excessive strain')
    return delta.astype(np.float32),{'provenance':provenance,'visibleContourAuditSha256':_sha(audit_path),
        'affectedRawVertices':int(selected.sum()),'sides':side_reports,
        'rawEdgeRatioQuantiles':np.quantile(ratio[affected],[0,.05,.5,.95,1]).tolist(),
        'rawNormalAlignmentMinimum':float(alignment.min()),'appearanceAccepted':False,'evaluatedReviewPending':True,
        'scope':'One static upper-lid tissue rotation along the unchanged eye, with continuous inner/outer thickness and geodesic orbital fade. Lower lids, canthi and all non-eye geometry pinned.',
        'limits':['Does not implement blinking or prove eye contact under gaze changes.',
            'Actual L3 visibility and existing fibre binding are checked after activation; inspect the actual eye macro for thickness and lash collisions.']}


def actual_visible_summary():
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    qa=_api('face_contact_audit_r11');result={}
    with qa['rest_evaluation']():
        dg=bpy.context.evaluated_depsgraph_get()
        def surface(obj):
            ev=obj.evaluated_get(dg);mesh=ev.data;co=_array(mesh.vertices).astype(float);matrix=np.asarray(ev.matrix_world)
            world=co@matrix[:3,:3].T+matrix[:3,3]
            return BVHTree.FromPolygons([Vector(p) for p in world],[tuple(p.vertices) for p in mesh.polygons]),world
        skin,_=surface(bpy.data.objects[HEAD])
        def hit(surface,x,z):return surface.ray_cast(Vector((float(x),-.35,float(z))),Vector((0,1,0)),.7)[0]
        for side in ('L','R'):
            globe,world=surface(bpy.data.objects['R11 Adult sclera '+side]);lo=world.min(0);hi=world.max(0);center=(lo+hi)*.5
            rows=[]
            for x in center[0]+np.array([-.008,-.004,0.,.004,.008]):
                zs=np.linspace(lo[2]+.0001,hi[2]-.0001,121)
                def visible(z):
                    g=hit(globe,x,z);h=hit(skin,x,z)
                    return g is not None and (h is None or g.y<h.y-1e-7)
                ids=np.flatnonzero([visible(z) for z in zs])
                if len(ids)<3 or ids[0]==0 or ids[-1]==len(zs)-1 or np.any(np.diff(ids)!=1):raise RuntimeError('Invalid actual visible eye aperture')
                ends=[]
                for inside,outside in ((ids[0],ids[0]-1),(ids[-1],ids[-1]+1)):
                    a=zs[inside];b=zs[outside]
                    for _ in range(13):
                        mid=(a+b)*.5
                        if visible(mid):a=mid
                        else:b=mid
                    ends.append(float((a+b)*.5))
                rows.append({'x':float(x),'lower':ends[0],'upper':ends[1]})
            result[side]=rows
    return result


def _keys(mesh):
    if not mesh.shape_keys:return None
    return {k.name:{'coords':_array(k.data),'relative':k.relative_key.name,'value':k.value,'mute':k.mute,
        'min':k.slider_min,'max':k.slider_max,'group':k.vertex_group} for k in mesh.shape_keys.key_blocks}


def apply(head_name=HEAD):
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT':raise RuntimeError('Independent background Object Mode candidate required')
    if 'R11_face_fibre_binding' not in bpy.context.scene:raise RuntimeError('Bind actual new brows/lashes before enabling this upper-lid candidate')
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
    visible_before=actual_visible_summary()
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
        visible_after=actual_visible_summary();coverage={}
        for side in ('L','R'):
            before_rows=visible_before[side];after_rows=visible_after[side]
            dz=before_rows[2]['upper']-after_rows[2]['upper']
            lower_error=max(abs(a['lower']-b['lower']) for a,b in zip(before_rows,after_rows))
            if not .00025<dz<.0012 or lower_error>5e-6:
                raise RuntimeError('Actual upper-lid coverage/lower-lid guard failed '+json.dumps({'side':side,'upperDrop':dz,'lowerError':lower_error}))
            coverage[side]={'centralUpperDropMeters':dz,'maximumLowerLidChangeMeters':lower_error}
        binding=_api('face_fibre_binding_r11')['validate'](neutral=False)
        report.update(actualVisibleBefore=visible_before,actualVisibleAfter=visible_after,actualCoverage=coverage,actualFibreBindingValidation=binding)
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
    print('R11_UPPER_LID_COVERAGE_SAVED '+json.dumps({'output':report['output'],'outputSha256':report['outputSha256']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);prepare(args.source,args.output)
