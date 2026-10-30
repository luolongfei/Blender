"""One bounded reversible posterior root + attached-fibre surface correction.

Uses the measured, source-hashed R09 diagnostic arrays. No new PN fit, new hair,
new topology, rig edits or tail solve. Only the posterior root positions and
the nearby existing scalp/gather points move, at most 2 mm. The root perimeter,
front Y<=0.01 m and final two samples of every gather remain exact.

run() changes memory only after complete actual BVH clearance checks.
apply_and_save() additionally saves an explicitly new candidate and JSON.
restore() restores retained original data and scalp runtime provenance.
No execution on import and no render. Adoption requires matched actual images.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path

SHELL='R06 attached scalp root substrate'
SCALP='R06 continuous scalp flow'
GATHERS=('R06 full gathered roots L','R06 full gathered roots R')
TAILS=('R02 Strands / R02 Twin tail L','R02 Strands / R02 Twin tail R')
NAMES=(SHELL,SCALP)+GATHERS
MARKER='R09_posterior_surface_transport'
RECORD='R09_surface_transport_restore'
BACKUP='R09_root_projection_original_data'
SOURCE='work/reconstruction-r02/reference-forelock-flow-r09.blend'
DIAGNOSTIC='work/reconstruction-r02/root-surface-diagnostic-r09.json'
MAX_MOVE=.002
GAP_TOLERANCE=.000002


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def _path(relative,suffix):
    path=(project_path()/relative).resolve();path.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if path.suffix!=suffix:raise ValueError('Wrong candidate file type')
    return path


def _smooth(t):
    t=np.clip(t,0,1);return t*t*(3-2*t)


def _protected(bpy,groom,uv):
    result=groom['_protected'](bpy,uv)
    for name in (SHELL,)+GATHERS:result.pop(name,None)
    return result


def _nonposition(data,groom):
    attributes={name:{'type':row['type'],'domain':row['domain'],'hash':groom['_array_hash'](row['values'])}
                for name,row in groom['_attributes'](data).items() if name!='position'}
    if hasattr(data,'polygons'):
        topology=[list(face.vertices) for face in data.polygons]
    else:topology=[[curve.first_point_index,curve.points_length] for curve in data.curves]
    return {'attributes':attributes,'topologySha256':hashlib.sha256(json.dumps(topology).encode()).hexdigest(),
            'materials':[(material.name,material.as_pointer()) if material else None for material in data.materials]}


def _attachment(obj,uv):
    return {'parent':uv['_plain'](obj.parent),'parentType':obj.parent_type,'parentBone':obj.parent_bone,
            'basis':uv['_plain'](obj.matrix_basis),'parentInverse':uv['_plain'](obj.matrix_parent_inverse)}


def _query(tree,point,vertices,triangles,alpha,diag):
    from mathutils import Vector
    hit,normal,index,distance=tree.find_nearest(Vector(point))
    if hit is None:raise RuntimeError('Root closest-surface query failed')
    hit=np.array(hit);normal=np.array(normal)
    if np.dot(normal,hit-diag['CENTER'])<0:normal=-normal
    bary=np.clip(diag['_barycentric'](hit,vertices[triangles[index]]),0,1);bary/=bary.sum()
    return hit,normal,int(index),bary,float(distance),float(bary@alpha[triangles[index]])


def _transport(obj,root,new_root,triangles,alpha,old_tree,new_tree,diag,groom):
    raw=groom['_values'](obj.data.position_data,'vector')
    world=groom['_world'](raw,obj.matrix_world);radii=groom['_values'](obj.data.attributes['radius'].data,'value',1).ravel()
    matrix=np.array(obj.matrix_world)
    if np.max(np.abs(matrix[:3,:3].T@matrix[:3,:3]-np.eye(3)))>1e-5:
        raise RuntimeError('Expected rigid unit-scale existing fibres: '+obj.name)
    pinned=np.zeros(len(world),bool)
    if obj.name in GATHERS:
        for curve in obj.data.curves:
            end=curve.first_point_index+curve.points_length;pinned[end-2:end]=True
    displacement=new_root-root;output=world.copy();local=raw.copy();before=np.empty(len(world));after=np.empty(len(world))
    nearby=np.zeros(len(world),bool);failures=[];max_move=0.
    inverse=np.array(obj.matrix_world.inverted());rounding=0.
    for i,point in enumerate(world):
        hit,normal,index,bary,distance,opacity=_query(old_tree,point,root,triangles,alpha,diag)
        before[i]=np.dot(point-hit,normal)-radii[i]
        # This smooth spatial field vanishes before the freely hanging gather
        # reaches the unchanged tail join. Do not truncate a visible strand.
        amount=1-float(_smooth((distance-.004)/.004))
        if pinned[i] or point[1]<=.010:amount=0.
        if amount:
            delta=bary@displacement[triangles[index]]*amount
            if np.any(delta):
                wanted=point+delta
                local[i]=(wanted@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32)
                output[i]=local[i].astype(float)@matrix[:3,:3].T+matrix[:3,3]
                rounding=max(rounding,float(np.abs(output[i]-wanted).max()))
                max_move=max(max_move,float(np.linalg.norm(output[i]-point)))
        new_hit,new_normal,_,_,new_distance,new_opacity=_query(new_tree,output[i],new_root,triangles,alpha,diag)
        after[i]=np.dot(output[i]-new_hit,new_normal)-radii[i]
        nearby[i]=new_opacity>=.95 and new_distance<.01
        # A source point already buried is not relabeled as fixed. Preserve its
        # old bound; every initially exposed point must remain exposed.
        lower_bound=min(0.,before[i])
        if nearby[i] and after[i]<lower_bound-GAP_TOLERANCE:
            if len(failures)<12:failures.append({'pointIndex':i,'beforeMeters':float(before[i]),
                'afterMeters':float(after[i]),'oldWorld':point.tolist(),'newWorld':output[i].tolist()})
        if i and i%250000==0:print('R09_ROOT_TRANSPORT_POINTS '+obj.name+' '+str(i)+'/'+str(len(world)),flush=True)
    newly_buried=nearby&(before>=0)&(after < -GAP_TOLERANCE)
    worse_existing=nearby&(before<0)&(after<before-GAP_TOLERANCE)
    if max_move>MAX_MOVE+1e-8:raise RuntimeError('Fibre transport exceeded 2 mm: '+obj.name)
    if newly_buried.any() or worse_existing.any():
        raise RuntimeError('Bounded surface transport lost actual clearance: '+json.dumps({'object':obj.name,
            'newlyBuried':int(newly_buried.sum()),'worseExisting':int(worse_existing.sum()),'examples':failures}))
    # Clearance above was queried with the actual float32 local coordinates
    # transformed back to world space, exactly as the saved Curves will use.
    local[pinned]=raw[pinned];actual_world=groom['_world'](local,obj.matrix_world)
    if rounding>3e-7:raise RuntimeError('Fibre local-space rounding exceeded the reviewed tolerance')
    max_before=max_after=extra=0.
    for curve in obj.data.curves:
        ids=slice(curve.first_point_index,curve.first_point_index+curve.points_length)
        old_step=np.linalg.norm(np.diff(world[ids],axis=0),axis=1)
        new_step=np.linalg.norm(np.diff(actual_world[ids],axis=0),axis=1)
        max_before=max(max_before,float(old_step.max()));max_after=max(max_after,float(new_step.max()))
        extra=max(extra,float((new_step-old_step).max()))
    if extra>.0008 or max_after>max_before+.0008:
        raise RuntimeError('Surface field introduced an excessive local strand step: '+obj.name)
    return local,{'object':obj.name,'curves':len(obj.data.curves),'checkedPoints':len(world),
        'changedPoints':int((np.linalg.norm(actual_world-world,axis=1)>1e-8).sum()),
        'maximumDisplacementMeters':float(np.linalg.norm(actual_world-world,axis=1).max()),
        'beforeOpaqueNearbyClearanceMeters':diag['_summary'](before[nearby]),
        'afterOpaqueNearbyClearanceMeters':diag['_summary'](after[nearby]),
        'newlyBuriedPointCount':int(newly_buried.sum()),'worseAlreadyBuriedPointCount':int(worse_existing.sum()),
        'alreadyBuriedBeforeCount':int((nearby&(before<0)).sum()),
        'negativeAfterIncludingNumericalToleranceCount':int((nearby&(after<0)).sum()),
        'pointClearanceToleranceMeters':GAP_TOLERANCE,'localFloatRoundingMaximumMeters':rounding,
        'maximumPointStepBeforeMeters':max_before,'maximumPointStepAfterMeters':max_after,
        'maximumExtraSegmentMeters':extra,'gatherFinalTwoSamplesExact':bool(np.array_equal(local[pinned],raw[pinned]))}


def _check_other_visible_curves(bpy,root,new_root,triangles,alpha,old_tree,new_tree,diag,groom):
    rows=[];low=root.min(axis=0)-.008;high=root.max(axis=0)+.008
    for obj in bpy.context.scene.objects:
        if obj.type!='CURVES' or obj.name in NAMES or obj.name in TAILS or obj.hide_render:continue
        if obj.parent is None or obj.parent.name!='Character_Rig' or obj.parent_bone!='head':continue
        if any(m.show_viewport or m.show_render for m in obj.modifiers):
            raise RuntimeError('Visible head hair modifiers need an explicit clearance strategy: '+obj.name)
        world=groom['_world'](groom['_values'](obj.data.position_data,'vector'),obj.matrix_world)
        candidates=np.flatnonzero(np.all((world>=low)&(world<=high),axis=1)&(world[:,1]>.010))
        if not len(candidates):continue
        radius=obj.data.attributes.get('radius')
        if radius is None:raise RuntimeError('Visible head curve radius missing: '+obj.name)
        radii=groom['_values'](radius.data,'value',1).ravel();newly=0;worse=0;worst=[]
        for i in candidates:
            q=world[i]
            old_hit,n,_,_,_,_= _query(old_tree,q,root,triangles,alpha,diag)
            new_hit,nn,_,_,distance,opacity=_query(new_tree,q,new_root,triangles,alpha,diag)
            before=float(np.dot(q-old_hit,n)-radii[i]);after=float(np.dot(q-new_hit,nn)-radii[i])
            if opacity>=.95 and distance<.01 and after<min(0.,before)-GAP_TOLERANCE:
                newly+=int(before>=0);worse+=int(before<0)
                if len(worst)<8:worst.append({'point':int(i),'before':before,'after':after})
        row={'object':obj.name,'candidatePointsChecked':len(candidates),'newlyBuried':newly,'worseExisting':worse,'geometryChanged':False}
        rows.append(row)
        if newly or worse:raise RuntimeError('Unchanged visible head hair would lose clearance: '+json.dumps({'summary':row,'examples':worst}))
    return rows


def run(source_relative=SOURCE,diagnostic_relative=DIAGNOSTIC,diagnostics_only=False):
    import bpy
    source=_path(source_relative,'.blend');diagnostic=_path(diagnostic_relative,'.json')
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=source:
        raise RuntimeError('Open the exact independent diagnosed source in background Object Mode')
    measured=json.loads(diagnostic.read_text(encoding='utf-8'));source_sha=_sha(source)
    if measured['sourceSha256']!=source_sha or (project_path()/measured['source']).resolve()!=source:
        raise RuntimeError('Diagnostic belongs to a different saved source')
    numeric=(project_path()/measured['diagnosticArrays']['path']).resolve()
    numeric.relative_to((project_path()/'work/reconstruction-r02').resolve())
    if _sha(numeric)!=measured['diagnosticArrays']['sha256']:raise RuntimeError('Diagnostic numeric payload changed')
    with np.load(numeric,allow_pickle=False) as file:arrays={name:file[name].copy() for name in file.files if name.startswith('root_')}
    diag=_api('root_surface_diagnostic_r09');groom=_api('posterior_groom_r09');uv=_api('production_uv_r08')
    objects={name:bpy.data.objects[name] for name in NAMES};shell=objects[SHELL];rig=bpy.data.objects['Character_Rig']
    if any(any(key in obj for key in (MARKER,RECORD,BACKUP)) for obj in objects.values()):raise RuntimeError('A surface candidate already exists; preserve or restore it')
    for obj in objects.values():
        if obj.library or obj.data.library or obj.data.users!=1:raise RuntimeError('Local single-user root/fibre data required')
        if any(m.show_viewport or m.show_render for m in obj.modifiers):raise RuntimeError('Root/scalp/gather modifiers require explicit integration')
    if shell.data.shape_keys:raise RuntimeError('Root substrate unexpectedly has morphs; do not alter its key system')
    if groom['_data_hash'](shell.data)!=measured['rootDataSha256']:raise RuntimeError('Diagnosed root data changed')
    bpy.context.view_layer.update();protected=_protected(bpy,groom,uv)
    original={name:obj.data for name,obj in objects.items()}
    hashes={name:groom['_data_hash'](obj.data) for name,obj in objects.items()}
    contracts={name:_nonposition(obj.data,groom) for name,obj in objects.items()}
    attachments={name:_attachment(obj,uv) for name,obj in objects.items()}
    fake={name:data.use_fake_user for name,data in original.items()}
    provenance=objects[SCALP].get('RT09_runtime_group_provenance')
    groups=objects[SCALP].get('RT09_runtime_group_counts')
    pose=rig.data.pose_position;prepared={};committed=False
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        raw=groom['_values'](shell.data.vertices,'co');root=groom['_world'](raw,shell.matrix_world)
        if root.shape!=arrays['root_world'].shape or np.abs(root-arrays['root_world']).max()>2e-7:
            raise RuntimeError('Current root does not match the diagnosed REST coordinates')
        triangles=arrays['root_triangles'];boundary=arrays['root_boundary_indices'];alpha=arrays['root_alpha'].astype(float)
        # Restrict the measured correction to posterior scalp. Forehead and
        # forelock root Y<=10 mm remain exact; posterior Y>=45 mm gets the
        # measured PN correction, with one smooth transition between them.
        posterior=_smooth((root[:,1]-.010)/.035)
        delta=(arrays['root_proposed_world'].astype(float)-arrays['root_world'].astype(float))*posterior[:,None]
        delta[boundary]=0.;new_root=root+delta
        if np.linalg.norm(delta,axis=1).max()>MAX_MOVE+1e-8:raise RuntimeError('Measured root correction exceeds 2 mm')
        new_local=groom['_world'](new_root,shell.matrix_world.inverted()).astype(np.float32)
        unchanged=np.all(delta==0,axis=1);new_local[unchanged]=raw[unchanged];new_local[boundary]=raw[boundary]
        new_root=groom['_world'](new_local,shell.matrix_world)
        old_norm=np.cross(root[triangles[:,1]]-root[triangles[:,0]],root[triangles[:,2]]-root[triangles[:,0]])
        new_norm=np.cross(new_root[triangles[:,1]]-new_root[triangles[:,0]],new_root[triangles[:,2]]-new_root[triangles[:,0]])
        if np.any(np.sum(old_norm*new_norm,axis=1)<=0):raise RuntimeError('Root correction flipped/degenerate triangle')
        old_tree=diag['_tree'](root,triangles);new_tree=diag['_tree'](new_root,triangles)
        positions={};hair=[]
        for name in (SCALP,)+GATHERS:
            print('R09_ROOT_TRANSPORT_BEGIN '+name,flush=True)
            positions[name],row=_transport(objects[name],root,new_root,triangles,alpha,old_tree,new_tree,diag,groom)
            hair.append(row);print('R09_ROOT_TRANSPORT_CLEARANCE '+json.dumps(row),flush=True)
        other=_check_other_visible_curves(bpy,root,new_root,triangles,alpha,old_tree,new_tree,diag,groom)
        for name,obj in objects.items():
            candidate=obj.data.copy();candidate.name='R09 curved posterior / '+obj.data.name;prepared[name]=candidate
            if name==SHELL:candidate.vertices.foreach_set('co',new_local.ravel());candidate.update()
            else:candidate.position_data.foreach_set('vector',positions[name].ravel());candidate.update_tag()
            if _nonposition(candidate,groom)!=contracts[name]:raise RuntimeError('Non-position data changed: '+name)
        rig.data.pose_position=pose;bpy.context.view_layer.update()
        if _protected(bpy,groom,uv)!=protected:raise RuntimeError('Protected head/tail/forelock/rig changed')
        for name,data in original.items():
            if groom['_data_hash'](data)!=hashes[name]:raise RuntimeError('Original retained data changed: '+name)
        report={'source':str(source.relative_to(project_path())),'sourceSha256':source_sha,
            'diagnostic':str(diagnostic.relative_to(project_path())),'diagnosticSha256':_sha(diagnostic),
            'diagnosticArraysSha256':_sha(numeric),'objects':list(NAMES),
            'method':'Measured PN posterior root correction plus the same barycentric surface displacement field on existing nearby scalp/gather fibres.',
            'posteriorWindowWorldY':{'unchangedBelow':.010,'fullAbove':.045},
            'fieldDistanceWindowMeters':{'fullBelow':.004,'zeroAbove':.008},
            'rootMaximumDisplacementMeters':float(np.linalg.norm(new_root-root,axis=1).max()),
            'rootBoundaryVertexCount':len(boundary),'rootBoundaryCoordinatesExact':bool(np.array_equal(new_local[boundary],raw[boundary])),
            'rootFrontCoordinatesExact':bool(np.array_equal(new_local[root[:,1]<=.010],raw[root[:,1]<=.010])),
            'sourceDataHashes':hashes,'candidateDataHashes':{name:groom['_data_hash'](data) for name,data in prepared.items()},
            'retainedOriginalDataNames':{name:data.name for name,data in original.items()},
            'curveCountsOrderRadiiMaterialsUVAlphaUnchanged':True,'rootTopologyUnchanged':True,
            'hairAllStoredPointClearance':hair,'unchangedOtherVisibleHeadHair':other,
            'nativeTailPointsAndAttachmentsUnchanged':True,'headGeometryMorphsWeightsAndRigUnchanged':True,
            'scalpRuntimeGroupCountsUnchanged':groups,'affectedRuntimeRegions':['root','scalp','gather'],
            'savedBlend':False,'writtenInMemory':not diagnostics_only,'appearanceAccepted':False,
            'limits':['All stored points of scalp and both gather objects are checked against the actual proposed root BVH, with 2 micrometre numerical tolerance.',
                      'This is not continuous between-point curve or full-body collision verification.',
                      'Only existing surface-attached geometry moves by at most 2 mm; native tails, their pin joins, the rig and body actions remain unchanged.',
                      'If adopted, migrate root/scalp/gather geometry and rebake root normal plus scalp/gather projected atlases. Existing UV/alpha are preserved.',
                      'Review matched back/rear-quarter/crown plus front silhouette before visual adoption. Restore this pass before restoring posterior append.']}
        if _sha(source)!=source_sha:raise RuntimeError('Source file changed externally')
        if diagnostics_only:return report
        restore={'objects':{name:{'data':original[name].name,'hash':hashes[name],'fakeUser':fake[name]} for name in NAMES},
                 'scalpRuntimeProvenance':provenance,'scalpRuntimeGroups':groups,
                 'sourceSha256':source_sha,'candidateDataHashes':report['candidateDataHashes']}
        for name,obj in objects.items():
            original[name].use_fake_user=True;obj.data=prepared[name];obj[MARKER]=source_sha;obj[BACKUP]=original[name].name
        shell[RECORD]=json.dumps(restore,ensure_ascii=False)
        objects[SCALP]['RT09_runtime_group_provenance']=(provenance or '')+'; R09 measured posterior root transport <=2 mm, existing group order/count unchanged.'
        for obj in objects.values():obj.data.update_tag();obj.update_tag()
        bpy.context.view_layer.update()
        if _protected(bpy,groom,uv)!=protected:raise RuntimeError('Protected data changed on candidate assignment')
        if any(_attachment(obj,uv)!=attachments[name] for name,obj in objects.items()):raise RuntimeError('Object attachment changed')
        report['protectedDataUnchanged']=True;committed=True;return report
    finally:
        rig.data.pose_position=pose
        if not committed:
            for name,obj in objects.items():
                if obj.data!=original[name]:obj.data=original[name]
                original[name].use_fake_user=fake[name]
                for key in (MARKER,RECORD,BACKUP):
                    if key in obj:del obj[key]
            if provenance is None:
                if 'RT09_runtime_group_provenance' in objects[SCALP]:del objects[SCALP]['RT09_runtime_group_provenance']
            else:objects[SCALP]['RT09_runtime_group_provenance']=provenance
            for name,data in prepared.items():
                if data.users==0:
                    (bpy.data.meshes if name==SHELL else bpy.data.hair_curves).remove(data)
        bpy.context.view_layer.update()


def apply_and_save(source_relative=SOURCE,output_relative='work/reconstruction-r02/reference-root-flow-r09.blend',
                   report_relative='work/reconstruction-r02/root-surface-transport-r09.json'):
    import bpy
    source=_path(source_relative,'.blend');target=_path(output_relative,'.blend');report_path=_path(report_relative,'.json')
    if source==target or target.exists() or report_path.exists():raise RuntimeError('New independent candidate and report paths required')
    report=run(source_relative=source_relative)
    if target.exists():raise RuntimeError('Candidate path appeared during preparation; do not overwrite it')
    bpy.ops.wm.save_as_mainfile(filepath=str(target));report['savedBlend']=True
    report.update(output=str(target.relative_to(project_path())),outputSha256=_sha(target))
    with report_path.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


def restore():
    import bpy
    shell=bpy.data.objects[SHELL]
    if RECORD not in shell:return False
    record=json.loads(shell[RECORD]);groom=_api('posterior_groom_r09');old={}
    for name,row in record['objects'].items():
        data=(bpy.data.meshes if name==SHELL else bpy.data.hair_curves).get(row['data'])
        if data is None or groom['_data_hash'](data)!=row['hash']:raise RuntimeError('Original surface data missing or changed: '+name)
        old[name]=data
    retained={}
    for name,data in old.items():
        obj=bpy.data.objects[name];retained[name]=obj.data.name;obj.data.use_fake_user=True
        obj.data=data;data.use_fake_user=record['objects'][name]['fakeUser']
        if MARKER in obj:del obj[MARKER]
        if BACKUP in obj:del obj[BACKUP]
    obj=bpy.data.objects[SCALP]
    for key,value in (('RT09_runtime_group_provenance',record['scalpRuntimeProvenance']),('RT09_runtime_group_counts',record['scalpRuntimeGroups'])):
        if value is None:
            if key in obj:del obj[key]
        else:obj[key]=value
    del shell[RECORD];bpy.context.view_layer.update()
    return {'restored':list(old),'retainedCandidateData':retained,'savedBlend':False}


if __name__=='__main__':print('Loaded only. Explicit run/apply_and_save; one measured bounded candidate, no render.')
