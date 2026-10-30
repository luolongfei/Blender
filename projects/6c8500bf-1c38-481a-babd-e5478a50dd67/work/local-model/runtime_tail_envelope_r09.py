"""Measured, reversible R09 tail/upper-sleeve collision candidate.

No automatic run/save/export. Based on locked Three VRM CPU skin/collision
measurements from a525dd96... and actual sleeve/opacity geometry. Does not move
hair, rewrite weights, disable simulation, or modify stiffness/gravity/drag.
"""
import hashlib,importlib.util,json,sys,uuid
from pathlib import Path
import numpy as np
try:import bpy
except ImportError:bpy=None
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
GROUP='R09 measured puff sleeves';MARKER='RT09_measured_tail_envelope_backup'
VRM_SHA='a525dd967002848abd874db60297e115ef22a972bc502a01b2703592efa47bb6'

def _api(name):
    spec=importlib.util.spec_from_file_location('tailenv09_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def _path(relative):
    path=(project_path()/relative).resolve();path.relative_to((project_path()/'work/runtime-r06').resolve());return path

def _meshes():
    return {o.name:(o.data.as_pointer(),tuple(m.as_pointer() if m else 0 for m in o.data.materials))
            for o in bpy.context.scene.objects if o.type=='MESH'}

def _parameters(spring):
    return [{'bone':j.node.bone_name,'hitRadius':float(j.hit_radius),'stiffness':float(j.stiffness),
        'gravityPower':float(j.gravity_power),'gravityDir':list(j.gravity_dir),'dragForce':float(j.drag_force)} for j in spring.joints]

def apply(config_relative='work/runtime-r06/r09-tail-fit-measured-candidate.json'):
    review=_api('runtime_hair_review_r09');rig=review._guard();sb=rig.data.vrm_addon_extension.spring_bone1
    if rig.get(MARKER) or any(g.vrm_name==GROUP for g in sb.collider_groups):raise RuntimeError('Measured candidate already exists')
    cfg_path=_path(config_relative);config=json.loads(cfg_path.read_text(encoding='utf-8'))
    audit_path=_path('work/runtime-r06/r09-tail-motion-fit-measured60.json')
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['inputSha']!=VRM_SHA or audit['overrides']!=config:raise RuntimeError('Candidate differs from the actually measured CPU run')
    audit30_path=_path('work/runtime-r06/r09-tail-motion-fit-measured30.json')
    audit30=json.loads(audit30_path.read_text(encoding='utf-8'))
    if audit30['inputSha']!=VRM_SHA or audit30['overrides']!=config:raise RuntimeError('Candidate differs from the measured 30 Hz run')
    comparison_path=_path('work/runtime-r06/r09-tail-profile-comparison.json')
    comparison=json.loads(comparison_path.read_text(encoding='utf-8'))
    if set(config)!={'hitRadiiBySide','sleeveCapsules'}:raise RuntimeError('Unexpected candidate parameters')
    if any(len(config['hitRadiiBySide'][s])!=6 for s in ('L','R')) or len(config['sleeveCapsules'])!=2:raise RuntimeError('Expected two six-joint tails and two sleeve capsules')
    if any(not .01<=v<=.05 for values in config['hitRadiiBySide'].values() for v in values):raise RuntimeError('Candidate radius out of measured bounds')
    springs={s.vrm_name:s for s in sb.springs};target={name:springs[name] for name in review.SPRINGS}
    source={name:_parameters(spring) for name,spring in target.items()}
    old_refs={name:review._refs(spring) for name,spring in target.items()}
    for name,rows in source.items():
        side=name[-1];expected=[f'RT06 tail_{side}_{i:02d}' for i in range(6)]+[f'RT06 tail_{side}_05_CharakitEnd']
        if [r['bone'] for r in rows]!=expected:raise RuntimeError('Unexpected runtime tail chain')
        if not set(review.GROUPS).issubset({r['name'] for r in old_refs[name]}):raise RuntimeError('Apply existing shoulder/back group link fix first')
        for row in rows[:6]:
            if any(abs(row[key]-value)>2e-6 for key,value in [('hitRadius',.014),('stiffness',.7),('gravityPower',.12),('dragForce',.42)]):raise RuntimeError('Current tail parameters differ from audited baseline')
    bone_names={name:getattr(rig.data.vrm_addon_extension.vrm1.humanoid.human_bones,attr).node.bone_name
        for name,attr in [('leftUpperArm','left_upper_arm'),('rightUpperArm','right_upper_arm')]}
    if any(name not in rig.data.bones for name in bone_names.values()):raise RuntimeError('Mapped raw upper-arm bones missing')
    before_meshes=_meshes();refresh=_api('runtime_face_refresh_r08');before_rig=refresh._rig_contract(rig)
    backup={'parameters':source,'refs':old_refs,'colliderUUIDs':[],'colliderObjects':[],'groupUUID':None,
        'candidateConfig':config_relative,'candidateConfigSha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest()}
    rig[MARKER]=json.dumps(backup,ensure_ascii=False);pose=rig.data.pose_position
    try:
        rig.data.pose_position='REST';bpy.context.view_layer.update()
        # Installed VRM 4.10.2 exposes capsule offsets directly in bone-local
        # coordinates and emits those values unchanged (vrm1_exporter.py:770).
        # The model's bone-local axes are preserved through glTF export; no
        # extra world-space Blender Z/Y conversion is applied here.
        for row in config['sleeveCapsules']:
            name='RT09 measured sleeve / '+row['bone']
            if bpy.data.objects.get(name):raise RuntimeError('Candidate collider object name exists')
            collider=sb.colliders.add();collider.uuid=uuid.uuid4().hex;backup['colliderUUIDs'].append(collider.uuid)
            collider.node.bone_name=bone_names[row['bone']];collider.shape_type=collider.SHAPE_TYPE_CAPSULE.identifier
            collider.reset_bpy_object(bpy.context,rig);collider.bpy_object.name=name;collider.bpy_object.hide_render=True
            shape=collider.shape.capsule;shape.offset=row['offset'];shape.tail=row['tail'];shape.radius=row['radius']
            bpy.context.view_layer.update()
            if not np.allclose(list(shape.offset),row['offset'],rtol=0,atol=2e-6) or not np.allclose(list(shape.tail),row['tail'],rtol=0,atol=2e-6) or abs(shape.radius-row['radius'])>2e-6:raise RuntimeError('Capsule properties did not retain measured local coordinates')
            backup['colliderObjects'].extend([o.name for o in (*collider.bpy_object.children,collider.bpy_object)])
            rig[MARKER]=json.dumps(backup,ensure_ascii=False)
        group=sb.add_collider_group();group.vrm_name=GROUP;backup['groupUUID']=group.uuid
        for uid in backup['colliderUUIDs']:group.colliders.add().collider_uuid=uid
        rig[MARKER]=json.dumps(backup,ensure_ascii=False)
        for name,spring in target.items():
            spring.collider_groups.add().collider_group_uuid=group.uuid
            for i,value in enumerate(config['hitRadiiBySide'][name[-1]]):spring.joints[i].hit_radius=value
        bpy.context.view_layer.update()
        if _meshes()!=before_meshes or refresh._rig_contract(rig)!=before_rig:raise RuntimeError('Mesh or rig-rest state unexpectedly changed')
        for name,spring in target.items():
            rows=_parameters(spring)
            for before,after in zip(source[name],rows):
                if {k:v for k,v in before.items() if k!='hitRadius'}!={k:v for k,v in after.items() if k!='hitRadius'}:raise RuntimeError('Non-radius spring setting changed')
            if spring.joints[-1].hit_radius!=source[name][-1]['hitRadius']:raise RuntimeError('Terminal marker changed')
        collisions=[{'time':f['time'],'label':f['label'],'sides':[{'side':r['side'],'visibleIntersectionPairs':r['alphaFilteredIntersectionPairs'],'bodyObjects':r['byBody']} for r in f['garmentContact']['sides']]} for f in audit['samples'] if 'garmentContact' in f]
        return review._report('r09-tail-measured-envelope',{'config':config,'before':source,
            'after':{name:_parameters(s) for name,s in target.items()},'colliderGroup':GROUP,
            'localCoordinateEvidence':'Installed 4.10.2 exporter emits capsule.offset/tail directly; values fitted in corresponding actual raw upper-arm bone frame.',
            'cpuAudit':str(audit_path.relative_to(project_path())),'cpuAuditSha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'cpuAudit30Hz':str(audit30_path.relative_to(project_path())),'cpuAudit30HzSha256':hashlib.sha256(audit30_path.read_bytes()).hexdigest(),
            'profileComparison':str(comparison_path.relative_to(project_path())),'profileComparisonSha256':hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
            'candidateSelection':'Retain original measured 66 mm sleeves. Smaller two/three capsule profiles were measured and rejected; their tip rise was worse.',
            'comparisonScope':comparison['limits'],
            'cpuSourceVrmSha256':VRM_SHA,'cpuSelectedContactCounts':collisions,
            'meshWeightsRigRestMaterialsExpressions':'Unchanged','physicsEnabled':True,
            'limits':'Adopted integration candidate, not final acceptance. Full 12 s at 30/60 Hz found no sleeve/collar crossing at 0.2 s sampled frames, with alpha > 0.15. Sparse lower-tail/bodice contacts remain (8 sampled frames, at most 14 pairs). Tip rise and whole-bundle forward lift remain visible. Final combined VRM needs actual viewer review; finite samples are not a zero-collision proof.'})
    except Exception:
        rig[MARKER]=json.dumps(backup,ensure_ascii=False);restore();raise
    finally:
        rig.data.pose_position=pose;bpy.context.view_layer.update()

def restore():
    review=_api('runtime_hair_review_r09');rig=review._guard();raw=rig.get(MARKER)
    if not raw:return False
    backup=json.loads(raw);sb=rig.data.vrm_addon_extension.spring_bone1;springs={s.vrm_name:s for s in sb.springs}
    for name,rows in backup['parameters'].items():
        spring=springs[name];review._set_refs(spring,backup['refs'][name])
        for joint,row in zip(spring.joints,rows):joint.hit_radius=row['hitRadius']
    for index in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[index].uuid==backup['groupUUID']:sb.collider_groups.remove(index)
    for index in reversed(range(len(sb.colliders))):
        if sb.colliders[index].uuid in backup['colliderUUIDs']:
            obj=sb.colliders[index].bpy_object
            if obj:
                # This collider UUID was created by this transaction. Its
                # generated empties are safe to remove even if a setter failed
                # before their names were recorded in the backup list.
                for item in (*obj.children,obj):bpy.data.objects.remove(item,do_unlink=True)
            sb.colliders.remove(index)
    del rig[MARKER];bpy.context.view_layer.update();return True

if __name__=='__main__':print('Loaded only. apply() in saved runtime copy, save NEW checkpoint explicitly, then export/visual review. restore() reverts only this candidate.')
