"""Integrator-only staged R09 master promotion; no work on import.

1. Fresh background Blender, explicit adopted source open:
   prepare(source_relative, expected_source_sha256, stage_id)
2. Fresh background Blender, returned stagedBlend open:
   verify_staged(stage_record_relative)
3. Ordinary Python, no Blender writer running:
   install(stage_record_relative, expected_checkpoint_sha256)
4. Fresh background Blender, work/character.blend open:
   verify_master(stage_record_relative)

Only install replaces the fixed master (verified local old-master backup,
same-directory staging, atomic replacement). It immediately records the actual
new master with fresh-master verification pending. verify_master performs fresh
finite checks and completes that provenance record, never appearance acceptance.
No subprocess, Blender launch, render, bake, export or historical-candidate write.
"""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path().resolve();MASTER=P/'work/character.blend';STATE=P/'project-state.json'
MIGRATION=ROOT/'asset-manifests'/f'{P.name}.json'
HAIR_FRAMES=(1,55,109.375,151,247,301)


def _api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))
def _now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def _read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def _rel(path):return path.resolve().relative_to(P).as_posix()


def _path(relative):
    path=(P/relative).resolve();path.relative_to(P)
    return path


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _identity(path):
    stat=path.stat();sha=_sha(path);after=path.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError('File changed during hashing: '+str(path))
    return {'path':_rel(path),'bytes':stat.st_size,'sha256':sha}


def _new_json(path,data):
    with path.open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)


def _copy_verified(source,target,expected):
    if source.resolve()==target.resolve():raise RuntimeError('Refuse self-copy')
    before=_identity(source)
    if before['sha256']!=expected:raise RuntimeError('Copy source SHA changed')
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    if _sha(target)!=expected or _identity(source)!=before:raise RuntimeError('Copy verification failed; preserve partial evidence')
    return _identity(target)


def _stable(value):
    if isinstance(value,dict):return {k:_stable(v) for k,v in value.items() if not k.lower().endswith('pointer')}
    if isinstance(value,list):return [_stable(v) for v in value]
    return value


def _digest(value):return hashlib.sha256(json.dumps(_stable(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _assert_open(path):
    import bpy
    if not bpy.app.background or bpy.context.mode!='OBJECT' or Path(bpy.data.filepath).resolve()!=path:
        raise RuntimeError('Integrator must open the exact saved file in a fresh background Blender process')
    if bpy.data.is_dirty:raise RuntimeError('Fresh saved file required; do not promote unsaved edits')
    if bpy.context.scene.frame_current!=1 or bpy.context.scene.frame_subframe!=0:
        raise RuntimeError('Save the adopted body source at frame 1 before integration')


def _dependency_signature(audit):
    return sorted([{'kind':row['kind'],'owner':row['owner'],'packed':row['packed'],
        'resolved':os.path.normcase(row['resolvedPath']) if row.get('resolvedPath') else None,
        'exists':row.get('exists'),'withinProject':row.get('withinProject')}
        for row in audit['dependencies']],key=lambda r:json.dumps(r,sort_keys=True))


def _content_signature(audit):
    import bpy
    import numpy as np
    uv=_api('production_uv_r08');hair=_api('posterior_groom_r09')
    head=bpy.data.objects['R01 Head'];contract=uv['_original_contract'](head,tuple(l.name for l in head.data.uv_layers))
    contract.pop('sha256');contract.pop('shapeKeysSha256')
    contract['objectSettings'].pop('dimensions',None)
    arrays=[]
    for key in head.data.shape_keys.key_blocks:
        values=np.empty((len(key.data),3),np.float32);key.data.foreach_get('co',values.ravel())
        arrays.append({'name':key.name,'sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                       'value':key.value,'mute':key.mute,'relative':key.relative_key.name})
    visible={row['name'] for row in audit['objects'] if row['visibleInCurrentViewport'] or row['renderEligibleInCurrentLayer']}
    groom={obj.name:hair['_data_hash'](obj.data) for obj in bpy.context.scene.objects
           if obj.name in visible and (obj.type=='CURVES' or obj.name==hair['SHELL'])}
    actions={action.name:_stable(uv['_action_state'](action)) for action in bpy.data.actions}
    return {'headContractSha256':_digest(contract),'headContractStable':_stable(contract),
            'headKeys':arrays,'visibleGroom':groom,
            'actionSha256':_digest(actions),'actionFingerprints':actions,'dependencies':_dependency_signature(audit),
            'scope':'Head coordinates/all expression arrays/UV/material/weights, active visible native groom, all action curves, and resolved dependency identities. Native cloth/hair coordinates are checked separately at finite frames.'}


def _audit_gate(audit):
    import bpy
    active={row['name'] for row in audit['objects'] if row['visibleInCurrentViewport'] or row['renderEligibleInCurrentLayer']}
    active.add('Character_Rig')
    for obj in bpy.context.scene.objects:
        if obj.name in active and getattr(obj.data,'shape_keys',None):active.add(obj.data.shape_keys.name)
    rig=bpy.data.objects['Character_Rig'];native=_api('native_hair_playback_r04');guide=native['_active_guide'](rig)
    active.update((guide.name,guide.data.shape_keys.name))
    materials={m['name'] for m in audit['materials'] if m['renderEligibleUsers'] or m['viewportVisibleUsers']}
    images={texture['image'] for m in audit['materials'] if m['name'] in materials for texture in m['imageTextures']}
    # Environment/light images can be active without a mesh material owner.
    audit_api=_api('production_source_audit_r05')
    trees=[getattr(bpy.context.scene.world,'node_tree',None)]
    trees.extend(getattr(obj.data,'node_tree',None) for obj in bpy.context.scene.objects if obj.type=='LIGHT')
    for tree in trees:
        for nested in audit_api['_node_trees'](tree):
            for node in nested.nodes:
                image=getattr(node,'image',None)
                if image is not None:images.add(image.name)
    required_owners=active|images|materials
    fatal=[];non_surface_guides=[]
    fatal_kinds={'no-material-slots','empty-material-slot','polygon-material-unassigned','missing-active-uv','missing-named-uv',
                 'invalid-driver','file-image-without-path-or-packed-data','missing-external-dependency','dependency-outside-project'}
    for finding in audit['findings']:
        # Native playback intentionally stores positions in a render-hidden,
        # vertex-only mesh. It is read by the strand GN graph, has no surface
        # that could display a material, and must stay evaluable in viewport.
        if (finding['owner']==guide.name and finding['kind']=='no-material-slots'
                and guide.type=='MESH' and guide.hide_render
                and len(guide.data.polygons)==0 and len(guide.data.edges)==0
                and native['RECORD'] in guide):
            non_surface_guides.append({'object':guide.name,'vertices':len(guide.data.vertices),
                'polygons':0,'edges':0,'hiddenFromRender':True,'role':'native strand playback coordinate carrier'})
            continue
        if finding['kind'] in fatal_kinds and finding['owner'] in required_owners:fatal.append(finding)
    if audit['summary']['linkedLibraries']:fatal.append({'kind':'linked-libraries-present','count':audit['summary']['linkedLibraries']})
    if audit['source']['savedFileChangedDuringAudit']:fatal.append({'kind':'source-changed-during-audit'})
    if fatal:raise RuntimeError('Active-source audit unresolved: '+json.dumps(fatal,ensure_ascii=False))
    return {'activeDependencyAndDriverGate':'passed','verifiedNonSurfaceGuides':non_surface_guides,
            'historicalOrConservativeFindings':audit['findings'],
            'scope':'Inactive historical resources remain reported. This is not a blanket no-findings or delivery acceptance claim.'}


def _check_loaded(path):
    """Fresh inventory + bounded native parity, preserving in-memory metadata."""
    import bpy
    bpy.context.view_layer.update()
    audit_api=_api('production_source_audit_r05')
    audit=audit_api['report'](expected_source_relative=_rel(path),allow_master=path==MASTER)
    gate=_audit_gate(audit);signature=_content_signature(audit)
    merge=_api('cloth_merge_r09');cloth=merge['C'];native=merge['H']
    rig=cloth['_rig'](readonly_master=path==MASTER);scene=bpy.context.scene
    if len([obj for obj in scene.objects if obj.type=='ARMATURE'])!=1:raise RuntimeError('Expected one production scene armature')
    if (scene.frame_start,scene.frame_end)!=(1,301) or abs(scene.render.fps/scene.render.fps_base-30)>1e-6:
        raise RuntimeError('Expected matching body clip 1-301 at 30 fps')
    obj=bpy.data.objects[cloth['NATIVE']]
    if obj.hide_viewport or obj.hide_render or obj.hide_get():raise RuntimeError('Matched native cloth must be visible')
    if obj.parent!=rig:raise RuntimeError('Native cloth parenting changed')
    provenance=json.loads(obj['r09_combined_cloth_provenance'])
    for name in provenance['hiddenOldClothObjects']:
        old=bpy.data.objects.get(name)
        if old and (not old.hide_render or not old.hide_viewport):raise RuntimeError('Old cloth became visible: '+name)
    guide=native['_active_guide'](rig);saved_record=guide[native['RECORD']]
    record=json.loads(saved_record);cache=record['cacheRelativePath']
    try:
        identity=merge['_matched_hair'](rig)
        # Its full return is written into the validation JSON. Keep the CLI
        # compact instead of repeating the helper's entire JSON on stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            hair=native['validate_evaluated'](cache_relative=cache,frames=list(HAIR_FRAMES))
        cloth_check=merge['verify_loaded'](P/'work/reconstruction-r02/cloth-body-r09',obj,readonly_master=path==MASTER)
    finally:
        guide[native['RECORD']]=saved_record
        scene.frame_set(1);bpy.context.view_layer.update()
    # The parity helpers may dirty metadata while evaluating; none is saved by
    # verification, and the original guide record is restored above.
    return {'source':_identity(path),'audit':audit,'gate':gate,'contentSignature':signature,
            'bodyAction':identity['action'],'matchedHairIdentity':identity,'hairParity':hair,'clothParity':cloth_check,
            'requiredEvidenceFiles':[{'path':cache,'sha256':_sha(_path(cache))},
                {'path':'work/reconstruction-r02/cloth-body-r09/native-report.json',
                 'sha256':_sha(P/'work/reconstruction-r02/cloth-body-r09/native-report.json')}],
            'scope':'Finite native Blender body/hair/cloth and dependency checks; no full collision, continuous-video, VRM, likeness or client acceptance.'}


def prepare(source_relative,expected_source_sha256,stage_id):
    """Stage only. The master and project-state are not written here."""
    import bpy
    if not re.fullmatch(r'r09-[a-z0-9][a-z0-9-]{1,40}',stage_id):raise ValueError('Use a unique explicit r09- stage identifier')
    source=_path(source_relative);source.relative_to(P/'work/reconstruction-r02')
    _assert_open(source)
    if _sha(source)!=expected_source_sha256:raise RuntimeError('Explicit adopted source SHA differs')
    state=_read(STATE)
    if state.get('masterIntegrationR09'):raise RuntimeError('A master integration already exists; preserve or explicitly roll back it first')
    master=_identity(MASTER)
    if master['sha256']!=state['workingMaster']['sha256']:raise RuntimeError('Master differs from tracked state')
    migration=_read(MIGRATION);old=[row for row in migration['files'] if row['path']=='work/character.blend']
    if len(old)!=1 or old[0]['sha256']!=master['sha256'] or old[0]['bytes']!=master['bytes']:
        raise RuntimeError('First R09 integration requires the actual unchanged migration master checkpoint')
    directory=P/'work/master-integration-r09'/stage_id;directory.mkdir(parents=True,exist_ok=False)
    staged=P/'work'/('character-staged-'+stage_id+'.blend')
    if staged.exists():raise RuntimeError('Historical staged candidate exists; use another explicit stage ID')
    backup=_copy_verified(MASTER,directory/'migration-master.blend',master['sha256'])
    state_backup=_copy_verified(STATE,directory/'project-state-before.json',_sha(STATE))
    shutil.copyfile(MIGRATION,directory/'migration-manifest.json')
    shutil.copyfile(ROOT/'toolchain.lock.json',directory/'toolchain.lock.json')
    checkpoint={'recordedAt':_now(),'stageId':stage_id,'adoptedSource':_identity(source),'previousMaster':master,
        'verifiedMasterBackup':backup,'verifiedStateBackup':state_backup,
        'migrationManifestSha256':_sha(MIGRATION),'copiedMigrationManifestSha256':_sha(directory/'migration-manifest.json'),
        'toolchainLockSha256':_sha(ROOT/'toolchain.lock.json'),'copiedToolchainLockSha256':_sha(directory/'toolchain.lock.json'),
        'scope':'Verified local migration-master and state rollback checkpoint. Existing adopted source remains preserved. Not a whole-project freeze or off-machine backup.'}
    if checkpoint['migrationManifestSha256']!=checkpoint['copiedMigrationManifestSha256'] or checkpoint['toolchainLockSha256']!=checkpoint['copiedToolchainLockSha256']:
        raise RuntimeError('Checkpoint metadata copy verification failed')
    cp=directory/'checkpoint.json';_new_json(cp,checkpoint)
    source_check=_check_loaded(source);_new_json(directory/'source-validation.json',source_check)
    if _identity(source)!=checkpoint['adoptedSource'] or _identity(MASTER)!=master or _sha(STATE)!=state_backup['sha256']:
        raise RuntimeError('Source/master/state changed during staged validation')
    # Same directory as the eventual master means paths are already relocated
    # for work/character.blend. Saving never overwrites the adopted source.
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(staged),relative_remap=True):
        raise RuntimeError('Staged save failed')
    if _identity(source)!=checkpoint['adoptedSource']:raise RuntimeError('Adopted source changed during staged save')
    record={'stageId':stage_id,'checkpoint':_rel(cp),'checkpointSha256':_sha(cp),
        'sourceValidation':_rel(directory/'source-validation.json'),'sourceValidationSha256':_sha(directory/'source-validation.json'),
        'stagedBlend':_identity(staged),'master':_rel(MASTER),'status':'staged; fresh-staged-reopen-required',
        'masterWritten':False,'projectStateWritten':False,'appearanceAccepted':False}
    path=directory/'stage.json';_new_json(path,record)
    return dict(record,stageRecord=_rel(path))


def _stage(relative):
    path=_path(relative);path.relative_to(P/'work/master-integration-r09')
    record=_read(path);cp=_path(record['checkpoint'])
    if _sha(cp)!=record['checkpointSha256']:raise RuntimeError('Checkpoint SHA changed')
    checkpoint=_read(cp)
    for key in ('verifiedMasterBackup','verifiedStateBackup'):
        row=checkpoint[key]
        if _identity(_path(row['path']))!=row:raise RuntimeError('Rollback checkpoint changed: '+key)
    validation=_path(record['sourceValidation'])
    if _sha(validation)!=record['sourceValidationSha256']:raise RuntimeError('Saved source validation changed')
    return path,record,checkpoint,_read(validation)


def verify_staged(stage_record_relative):
    path,record,checkpoint,source_check=_stage(stage_record_relative)
    staged=_path(record['stagedBlend']['path']);_assert_open(staged)
    if _identity(staged)!=record['stagedBlend']:raise RuntimeError('Staged source changed after save')
    output=path.parent/'staged-reopen.json'
    if output.exists():raise RuntimeError('Preserve prior fresh verification')
    check=_check_loaded(staged)
    if check['source']!=record['stagedBlend']:raise RuntimeError('Staged source changed during fresh verification')
    if check['contentSignature']!=source_check['contentSignature']:
        difference=_api('perioral_delta_r09')['_contract_diff'](source_check['contentSignature'],check['contentSignature'])
        _new_json(path.parent/'staged-signature-difference.json',{'source':check['source'],'differences':difference})
        raise RuntimeError('Relocated stage signature differs; see staged-signature-difference.json before promotion')
    report={'stageRecord':stage_record_relative,'stageRecordSha256':_sha(path),'checkpointSha256':record['checkpointSha256'],
        'source':record['stagedBlend'],'verification':check,'status':'fresh-staged-finite-verification-passed',
        'masterWritten':False,'appearanceAccepted':False}
    _new_json(output,report);return report


def _write_state(state,expected_sha,temporary):
    if _sha(STATE)!=expected_sha:raise RuntimeError('Project state changed concurrently; do not overwrite it')
    _new_json(temporary,state)
    if _sha(STATE)!=expected_sha:raise RuntimeError('Project state changed before replacement; preserve staged state')
    os.replace(temporary,STATE)


def install(stage_record_relative,expected_checkpoint_sha256):
    """Ordinary Python, single integrator/no Blender writer: atomic master update."""
    if 'bpy' in sys.modules:raise RuntimeError('Install must run in ordinary Python, outside a Blender process')
    path,record,cp,source_check=_stage(stage_record_relative)
    if record['checkpointSha256']!=expected_checkpoint_sha256:raise RuntimeError('Supply the explicit verified checkpoint SHA')
    reopen=path.parent/'staged-reopen.json';proof=_read(reopen)
    if proof['stageRecordSha256']!=_sha(path) or proof['checkpointSha256']!=expected_checkpoint_sha256 or proof['status']!='fresh-staged-finite-verification-passed':
        raise RuntimeError('Fresh staged verification does not identify this exact checkpoint')
    staged=_path(record['stagedBlend']['path'])
    if _identity(staged)!=record['stagedBlend'] or proof['source']!=record['stagedBlend']:raise RuntimeError('Staged bytes changed since reopen')
    if proof['verification']['source']!=record['stagedBlend'] or proof['verification']['contentSignature']!=source_check['contentSignature']:
        raise RuntimeError('Staged proof does not preserve the exact adopted-source signature')
    if _identity(MASTER)!=cp['previousMaster'] or _sha(STATE)!=cp['verifiedStateBackup']['sha256']:
        raise RuntimeError('Master/state changed since checkpoint; re-evaluate before replacing')
    if _identity(_path(cp['adoptedSource']['path']))!=cp['adoptedSource']:raise RuntimeError('Adopted source changed')
    output=path.parent/'installed.json'
    if output.exists():raise RuntimeError('This integration was already installed; do not replay')
    temporary=P/'work'/('.character-publishing-'+record['stageId']+'.blend')
    _copy_verified(staged,temporary,record['stagedBlend']['sha256'])
    if _identity(MASTER)!=cp['previousMaster'] or _sha(STATE)!=cp['verifiedStateBackup']['sha256']:
        raise RuntimeError('Master/state changed during copy; publication stopped before replacement')
    os.replace(temporary,MASTER)
    actual=_identity(MASTER)
    if actual['sha256']!=record['stagedBlend']['sha256']:raise RuntimeError('Installed master SHA mismatch')
    result={'stageRecord':stage_record_relative,'stageRecordSha256':_sha(path),'checkpointSha256':expected_checkpoint_sha256,
        'installedAt':_now(),'adoptedSource':cp['adoptedSource'],'master':actual,
        'stagedReopen':{'path':_rel(reopen),'sha256':_sha(reopen)},'previousMasterBackup':cp['verifiedMasterBackup'],
        'status':'master-installed; fresh-master-reopen-pending','exportsWritten':False,'appearanceAccepted':False}
    _new_json(output,result)
    state=_read(STATE);state.setdefault('priorMasterHistory',[]).append(cp['previousMaster'])
    state['workingMaster']=dict(actual,integratedFrom=cp['adoptedSource'],savedAtSource=_now())
    state['masterIntegrationR09']=dict(result,record=_rel(output),recordSha256=_sha(output))
    state['stage']='r09-production-master-integrated; delivery-validation-in-progress'
    for name in ('reconstructionR02','reconstructionR09'):
        data=state.setdefault(name,{})
        data['currentSavedCandidate']=cp['adoptedSource'];data['integratedIntoMaster']=True;data['masterIntegrated']=True
        data['masterVerifiedUnchanged']=False;data['integratedMaster']=actual
    state.setdefault('exportCorrespondence',{})['staleAgainstCurrentMaster']=True
    state['exportCorrespondence']['currentMasterSha256']=actual['sha256']
    state['currentExportsStatus']='Historical exports retained byte-for-byte; no new VRM/Blender export is certified for this R09 master.'
    state['remainingR09Delivery']=['Fresh reopen of installed master','Final continuous matched body/hair/cloth video',
        'Refresh and reopen actual VRM/runtime from final adopted source','Final front/quarter/profile/back and expression appearance review',
        'Verified delivery package and dependency relocation; no new off-machine backup claimed']
    for name in ('reconstructionR02','reconstructionR09'):
        state[name]['remaining']=list(state['remainingR09Delivery'])
        state[name]['latestMasterIntegration']=_rel(output)
    state.setdefault('migration',{})['historicalManifestDoesNotCoverR09Master']=True
    state['migration']['r09LocalRollbackCheckpoint']={'path':record['checkpoint'],'sha256':expected_checkpoint_sha256,
        'wholeProjectFrozen':False,'offMachineCopyVerified':False}
    state['knownLimitationsBeforeR09Integration']=state.get('knownLimitations',[])
    state['knownLimitations']=['R09 production master integrated; formal exports and old migration manifest remain historical.',
        'Native CURVES/GN/embedded cloth playback are Blender-source features, not established VRM capabilities.',
        'Checks are finite frames and dependency inventory, not full collision or continuous motion acceptance.',
        'Final likeness/appearance, continuous video, runtime export/viewer and delivery package remain pending.',
        'No new client acceptance, author/license determination, performance guarantee or off-machine backup is implied.']
    state.setdefault('acceptance',{}).update(appearanceReviewedByUser=False,photorealisticClientDeliveryReady=False,
        technicalChecks='Exact staged source verified; installed-master fresh reopen pending. See masterIntegrationR09.')
    state['recordedAt']=_now()
    _write_state(state,cp['verifiedStateBackup']['sha256'],path.parent/'project-state-installing.json')
    return result


def verify_master(stage_record_relative):
    path,record,cp,source_check=_stage(stage_record_relative);_assert_open(MASTER)
    installed=_read(path.parent/'installed.json')
    if _identity(MASTER)!=installed['master']:raise RuntimeError('Master changed since installation')
    output=path.parent/'master-reopen.json'
    if output.exists():raise RuntimeError('Preserve prior master verification')
    check=_check_loaded(MASTER)
    if check['source']!=installed['master']:raise RuntimeError('Master changed during fresh verification')
    if check['contentSignature']!=source_check['contentSignature']:
        difference=_api('perioral_delta_r09')['_contract_diff'](source_check['contentSignature'],check['contentSignature'])
        _new_json(path.parent/'master-signature-difference.json',{'source':check['source'],'differences':difference})
        raise RuntimeError('Installed master signature differs; see master-signature-difference.json')
    result={'master':_identity(MASTER),'source':cp['adoptedSource'],'checkpointSha256':record['checkpointSha256'],
        'verification':check,'status':'fresh-master-finite-verification-passed; delivery-work-remains',
        'blendSaved':False,'appearanceAccepted':False,'offMachineCopyVerified':False}
    _new_json(output,result)
    expected=_sha(STATE);state=_read(STATE)
    if state['workingMaster']['sha256']!=result['master']['sha256']:raise RuntimeError('State describes a different master')
    state['masterIntegrationR09'].update(status=result['status'],freshMasterReopen={'path':_rel(output),'sha256':_sha(output)})
    state['remainingR09Delivery']=[value for value in state['remainingR09Delivery'] if value!='Fresh reopen of installed master']
    for name in ('reconstructionR02','reconstructionR09'):state[name]['remaining']=list(state['remainingR09Delivery'])
    state['acceptance']['technicalChecks']='Fresh installed-master finite body/hair/cloth parity and exact-source dependency checks recorded; no appearance or runtime equivalence claim.'
    state['recordedAt']=_now();_write_state(state,expected,path.parent/'project-state-verified.json')
    return result


def rollback(stage_record_relative):
    """Explicit root-only filesystem rollback; requires exact installed bytes/state.

    Preserve current master and state before replacing them from the checkpoint.
    No automatic rollback hides a failed candidate, and no candidate is deleted.
    """
    path,record,cp,_=_stage(stage_record_relative);installed=_read(path.parent/'installed.json')
    if _identity(MASTER)!=installed['master']:raise RuntimeError('Master was edited after install; preserve/resolve those edits first')
    state=_read(STATE)
    if state.get('masterIntegrationR09',{}).get('checkpointSha256')!=record['checkpointSha256']:
        raise RuntimeError('Project state no longer identifies this integration')
    current_sha=_sha(STATE)
    _copy_verified(MASTER,path.parent/'rolled-back-r09-master.blend',installed['master']['sha256'])
    _copy_verified(STATE,path.parent/'rolled-back-r09-project-state.json',current_sha)
    temporary=P/'work'/('.character-rollback-'+record['stageId']+'.blend')
    _copy_verified(_path(cp['verifiedMasterBackup']['path']),temporary,cp['previousMaster']['sha256'])
    os.replace(temporary,MASTER)
    old=_read(_path(cp['verifiedStateBackup']['path']))
    old.setdefault('rolledBackIntegrations',[]).append({'checkpoint':record['checkpoint'],'at':_now(),
        'preservedR09Master':_rel(path.parent/'rolled-back-r09-master.blend')})
    _write_state(old,current_sha,path.parent/'project-state-rollback.json')
    result={'master':_identity(MASTER),'rollback':True,'historicalCandidatesDeleted':False}
    _new_json(path.parent/'rollback.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Explicit staged R09 master integration; no render or export.')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare',help='Background Blender: validate adopted source and save a new staged source')
    p.add_argument('--source',required=True);p.add_argument('--source-sha',required=True);p.add_argument('--stage',required=True)
    p=commands.add_parser('verify-staged',help='Fresh background Blender: exact staged source already open')
    p.add_argument('--record',required=True)
    p=commands.add_parser('install',help='Ordinary Python: atomically install verified staged bytes and record pending reopen')
    p.add_argument('--record',required=True);p.add_argument('--checkpoint-sha',required=True)
    p=commands.add_parser('verify-master',help='Fresh background Blender: fixed master already open')
    p.add_argument('--record',required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        result=prepare(args.source,args.source_sha,args.stage)
        summary={'command':args.command,'stageRecord':result['stageRecord'],'stagedBlend':result['stagedBlend'],
                 'checkpointSha256':result['checkpointSha256'],'status':result['status'],'masterWritten':False}
    elif args.command=='verify-staged':
        result=verify_staged(args.record)
        summary={'command':args.command,'report':_rel(_path(args.record).parent/'staged-reopen.json'),
                 'source':result['source'],'status':result['status'],'masterWritten':False}
    elif args.command=='install':
        result=install(args.record,args.checkpoint_sha)
        summary={'command':args.command,'report':_rel(_path(args.record).parent/'installed.json'),
                 'master':result['master'],'status':result['status']}
    else:
        result=verify_master(args.record)
        summary={'command':args.command,'report':_rel(_path(args.record).parent/'master-reopen.json'),
                 'master':result['master'],'status':result['status']}
    print('R09_MASTER_INTEGRATION '+json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:])
