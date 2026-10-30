"""Apply only the reviewed fringe/temple shape correction to a saved v8 groom.

No fresh v6 rebuild, density change, tail change, or native trim reconstruction.
The caller independently checks the complete dense before/after geometry.
"""
import hashlib,runpy,sys
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').is_file());sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
P=project_path();HERE=Path(__file__).resolve().parent
MAIN='R11 Donor / long hair main';ROLE='R11 groom role'
LAYOUT_SHA='c04dd8a94f2a714fc4f3e32ce635c1f1e0b553c749beb553baedfa2ae933e3e5'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def apply(*,expected_source_relative):
    source=(P/expected_source_relative).resolve();source.relative_to((P/'work/reconstruction-r02').resolve())
    if not bpy.app.background or Path(bpy.data.filepath).resolve()!=source:raise RuntimeError('Explicit independent candidate source required')
    obj=bpy.data.objects[MAIN]
    if not obj.get('R11_reference_fringe_v8') or obj.get('R11_reference_local_v9'):raise RuntimeError('Saved measured v8-derived groom required')
    if bpy.data.objects['Character_Rig'].data.pose_position!='REST':raise RuntimeError('Caller must enter neutral REST')
    modifier=next(m for m in obj.modifiers if m.type=='NODES' and m.show_render)
    if modifier.node_group.nodes.get('R11 reference fringe native selection trim') is None:raise RuntimeError('Measured native v8 trim missing')
    layout_file=HERE/'reference_fringe_shape_v9_r11.py'
    if sha(layout_file)!=LAYOUT_SHA:raise RuntimeError('Reviewed local v9 layout changed')
    layout=runpy.run_path(str(layout_file));fit=runpy.run_path(str(HERE/'donor_groom_adult_r11.py'))
    segment_eye=runpy.run_path(str(HERE/'reference_front_groom_v6_r11.py'))['segment_eye']
    _,skin,head_report=fit['_head_trees'](bpy.data.objects['R11 Female head study']);center=layout['CENTER']
    def ray(direction):
        d=Vector(direction);hit,normal,_,distance=skin.ray_cast(Vector(center)+d*.30,-d,.60)
        return (None,None) if hit is None else (np.asarray(hit),np.asarray(normal))
    def front(x,z):
        hit,normal,_,distance=skin.ray_cast(Vector((x,-.60,z)),Vector((0.,1.,0.)),1.20)
        return (None,None) if hit is None else (np.asarray(hit),np.asarray(normal))
    eyes=[]
    for side in ('L','R'):
        eye=bpy.data.objects['R11 Adult sclera '+side];q=fit['_world'](fit['_coords'](eye),eye.matrix_world)
        eyes.append(((q.min(0)+q.max(0))*.5,float((q.max(0)-q.min(0)).max())*.5))
    original=obj.data;old_local=fit['_coords'](obj);old_world=fit['_world'](old_local,obj.matrix_world)
    starts=np.array([c.first_point_index for c in original.curves],np.int32);lengths=np.array([c.points_length for c in original.curves],np.int32)
    if (len(starts),len(old_local))!=(346,6770):raise RuntimeError('Unexpected source guide topology')
    attr=original.attributes.get(ROLE)
    if attr is None or attr.data_type!='INT' or attr.domain!='CURVE':raise RuntimeError('Actual source guide roles missing')
    old_roles=np.empty(len(starts),np.int32);attr.data.foreach_get('value',old_roles);roles=old_roles.copy()
    target=np.isin(old_roles,layout['TARGET_ROLES'])
    if int(target.sum())!=98 or int(np.sum(old_roles==13))!=38:raise RuntimeError('Actual v8 fringe partition changed; review before applying')
    # Start from the actual local float32 data. Never round-trip unselected
    # geometry through world matrices; even a harmless conversion can move it.
    local=old_local.copy();selected_points=np.zeros(len(local),bool);inverse=np.asarray(obj.matrix_world.inverted());rows=[];max_push=0.
    for i in np.flatnonzero(target):
        start=int(starts[i]);length=int(lengths[i]);root=old_world[start]
        new,row=layout['design'](root,int(i),length,int(old_roles[i]),ray,front)
        for j in range(1,len(new)):
            hit,n,_,distance=skin.find_nearest(Vector(new[j]));signed=float((Vector(new[j])-hit).dot(n))
            if signed<.00025:
                push=.00028-signed
                if push>.004:raise RuntimeError('Local fringe requires excessive raw skin correction '+str((int(i),j,push)))
                new[j]+=np.asarray(n)*push;max_push=max(max_push,push)
        eye_gap=min(segment_eye(new,c,r) for c,r in eyes)
        if eye_gap<.001:raise RuntimeError('Local raw fringe enters actual eye envelope '+str((int(i),eye_gap)))
        local[start:start+length]=fit['_world'](new,inverse).astype(np.float32);local[start]=old_local[start]
        selected_points[start:start+length]=True;roles[i]=row['newRole'];row['eyeEnvelopeGapMeters']=eye_gap;rows.append(row)
    if not np.array_equal(local[~selected_points],old_local[~selected_points]) or not np.array_equal(local[starts],old_local[starts]):raise RuntimeError('Raw local edit changed a protected point/root')
    original.use_fake_user=True;obj.data=original.copy();obj.data.name='R11 original-reference local fringe v9'
    obj.data.position_data.foreach_set('vector',local.ravel());obj.data.attributes[ROLE].data.foreach_set('value',roles);obj.data.update_tag()
    carriers=[]
    for carrier in sorted((o for o in bpy.data.objects if o.get('R11_groom_role_root_source')==obj.name),key=lambda o:o.name):
        old=carrier.data;positions=np.asarray([tuple(v.co) for v in old.vertices]);distance=np.linalg.norm(positions[:,None,:]-old_local[starts][None,:,:],axis=2);indices=distance.argmin(1)
        if float(distance[np.arange(len(indices)),indices].max())>2e-5:raise RuntimeError('Actual source role carrier no longer matches guide roots')
        previous=np.empty(len(indices),np.int32);old.attributes[ROLE].data.foreach_get('value',previous)
        if not np.array_equal(previous,old_roles[indices]):raise RuntimeError('Actual source carrier role values disagree with source guide roles')
        changed=previous!=roles[indices]
        if np.any(changed & (previous!=13)):raise RuntimeError('A carrier changed outside the authorized central role split')
        old.use_fake_user=True;carrier.data=old.copy();carrier.data.name='R11 v9 central split / '+old.name
        carrier.data.attributes[ROLE].data.foreach_set('value',roles[indices]);carrier.data.update()
        carriers.append({'object':carrier.name,'points':len(indices),'changedCentralRoleValues':int(changed.sum()),'positionsPreservedExact':True,'priorDataRetained':old.name})
    if not carriers:raise RuntimeError('No actual role root carriers found')
    obj.update_tag();bpy.context.view_layer.update()
    obj['R11_reference_local_v9']='Original portrait: wide-left plus fine-right central locks, right forehead opening, soft left temple; no tail/face/density changes'
    return {'source':expected_source_relative,'sourceSha256':sha(source),'helperSha256':sha(__file__),'layoutHelperSha256':sha(layout_file),
        'reference':'references/user-original-portrait-2026-10-28.png','head':head_report,'changedRawGuides':len(rows),'changedGuideIndices':np.flatnonzero(target).tolist(),
        'guides':rows,'rawTargetRoles':list(layout['TARGET_ROLES']),'maximumRawSkinCorrectionMeters':max_push,
        'rawNonTargetPositionsPreservedExact':True,'rawRootsPreservedExact':True,'rawNonTargetPositionsSha256':hashlib.sha256(old_local[~selected_points].tobytes()).hexdigest(),
        'rawBeforeRoles':old_roles.tolist(),'rawAfterRoles':roles.tolist(),'roleCarriers':carriers,'retainedMainGuideData':original.name,
        'densityNodesTailsWispsFaceEdited':False,'appearanceAccepted':False,'stage':'Local original-reference v9 candidate; complete actual dense geometry and visual review required.'}

if __name__=='__main__':print('Explicit saved v8-derived candidate apply only.')
