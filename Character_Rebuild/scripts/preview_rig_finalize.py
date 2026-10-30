"""Apply bounded clips, export, then reimport and inspect the actual GLB."""
import bpy,sys,json,hashlib,math,importlib.util
from pathlib import Path
from mathutils import Vector
from mathutils.kdtree import KDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.context import workspace_path
from common import neutral_review
OUT=workspace_path('Character_Rebuild/interactive');WORK=OUT/'work';CHECKS=OUT/'checks';CHECKS.mkdir(exist_ok=True)
VERIFY_ONLY='--verify-only' in sys.argv
bpy.ops.wm.open_mainfile(filepath=str(WORK/'rigged-rest.blend'))
arm=bpy.data.objects['PreviewHumanoid']
model=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.name.startswith('Reference_') and not o.hide_render]
module_file=OUT/'scripts/motions.py'
spec=importlib.util.spec_from_file_location('preview_motions',module_file);motions=importlib.util.module_from_spec(spec);spec.loader.exec_module(motions)
manifest=motions.create_preview_actions(arm)
(WORK/'motion-integration.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf8')
scene=bpy.context.scene
def set_action(name,frame):
    action=bpy.data.actions.get(name)
    assert action,name
    arm.animation_data_create();arm.animation_data.action=action
    if action.slots:arm.animation_data.action_slot=action.slots[0]
    arm.animation_data.use_nla=False
    scene.frame_set(frame);bpy.context.view_layer.update()

# Retain key-pose measurements from the editable source for imported comparison.
rest={}
arm.data.pose_position='REST';bpy.context.view_layer.update()
for ob in model:
    e=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());me=e.to_mesh()
    rest[ob.name]=[e.matrix_world@v.co for v in me.vertices];e.to_mesh_clear()
arm.data.pose_position='POSE';bpy.context.view_layer.update()
def measured_pose(action,frame):
    set_action(action,frame);rows=[];deps=bpy.context.evaluated_depsgraph_get()
    for ob in model:
        e=ob.evaluated_get(deps);me=e.to_mesh();points=[e.matrix_world@v.co for v in me.vertices]
        assert all(math.isfinite(c) for p in points for c in p)
        bbox=[[min(p[i] for p in points),max(p[i] for p in points)] for i in range(3)]
        rows.append({'name':ob.name,'vertices':len(points),'bbox':bbox,'maxRestDisplacement':max((p-r).length for p,r in zip(points,rest[ob.name]))})
        e.to_mesh_clear()
    return {'action':action,'frame':frame,'objects':rows}

actions={a.name:tuple(a.frame_range) for a in bpy.data.actions if a.name in ['idle','wave','look','walk','Tpose']}
assert len(actions)==5,actions
qa=[]
for name in ['idle','wave','look','walk','Tpose']:
    start,end=actions[name]
    for fraction in [0,.25,.5,.75,1]:qa.append(measured_pose(name,round(start+(end-start)*fraction)))
(WORK/'posed-geometry-report.json').write_text(json.dumps({'poses':qa,'checks':'Finite evaluated vertices and displacements at five phases per clip. Not a global collision certificate.'},indent=2),encoding='utf8')
scene=neutral_review();scene.render.resolution_x=640;scene.render.resolution_y=640;scene.eevee.taa_render_samples=48
def render_pose(name,frame,eye=(0,-5,0),target=(0,0,0),scale=2.2,suffix='front'):
    set_action(name,frame);cam=scene.camera;cam.location=eye
    cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale
    scene.render.filepath=str(CHECKS/(name+'-'+suffix+'.png'));bpy.ops.render.render(write_still=True)
if not VERIFY_ONLY:
    for name in ['idle','wave','look','walk']:
        lo,hi=actions[name];frame=round(lo+(hi-lo)*.25)
        render_pose(name,frame)
    render_pose('wave',round(actions['wave'][0]+(actions['wave'][1]-actions['wave'][0])*.25),(3,-5,.2),(0,0,.1),2.2,'quarter')
set_action('idle',1)
for ob in bpy.context.scene.objects:ob.select_set(False)
for ob in model:ob.hide_set(False);ob.select_set(True)
arm.select_set(True);bpy.context.view_layer.objects.active=arm
for im in bpy.data.images:
    if im.source=='FILE' and im.has_data and not im.packed_file:im.pack()
blend=OUT/'rigged-preview.blend';glb=OUT/'animated-preview.glb'
if not VERIFY_ONLY:
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    set_action('Tpose',1)
    bpy.ops.export_scene.gltf(filepath=str(glb),export_format='GLB',use_selection=True,export_apply=True,export_texcoords=True,export_normals=True,export_materials='EXPORT',export_image_format='AUTO',export_yup=True,export_animations=True,export_animation_mode='ACTIONS',export_skins=True,export_anim_single_armature=True,export_def_bones=False,export_cameras=False,export_lights=False,export_anim_slide_to_zero=True,export_optimize_animation_size=False)
source_reports={'blendSha256':hashlib.sha256(blend.read_bytes()).hexdigest(),'glbSha256':hashlib.sha256(glb.read_bytes()).hexdigest(),'sourceStaticSha256':arm.get('source_static_sha256'),'actions':actions,'meshCount':len(model),'boneCount':len(arm.data.bones)}
# Required clean-scene import: all resulting geometry, bindings and texture data
# are read from the newly exported bytes, never from cached source datablocks.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(glb))
rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
custom_shapes={pb.custom_shape for r in rigs for pb in r.pose.bones if pb.custom_shape}
imported=[o for o in bpy.context.scene.objects if o.type=='MESH' and o not in custom_shapes]
assert all(o.name not in rest and not any(m.type=='ARMATURE' for m in o.modifiers) for o in custom_shapes)
print('IMPORTER_BONE_DISPLAY_HELPERS',[(o.name,o.type) for o in custom_shapes])
assert len(imported)==47 and len(rigs)==1,(len(imported),len(rigs))
rig=rigs[0]
assert all(any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers) for o in imported)
used_images={n.image for o in imported for mat in o.data.materials if mat and mat.use_nodes for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image}
images=[]
for im in used_images:
    assert len(im.pixels[:4])==4 and im.has_data
    images.append({'name':im.name,'size':list(im.size),'packed':bool(im.packed_file)})
assert len(images)==5
weight_audit=[]
for o in imported:
    unweighted=0;maxerr=0;maximum=0
    for v in o.data.vertices:
        d=[g.weight for g in v.groups if g.weight>1e-7]
        if not d:unweighted+=1
        maxerr=max(maxerr,abs(sum(d)-1));maximum=max(maximum,len(d))
    assert unweighted==0 and maxerr<1e-5 and maximum<=4,(o.name,unweighted,maxerr,maximum)
    weight_audit.append({'object':o.name,'unweighted':unweighted,'maxWeightError':maxerr,'maxInfluences':maximum})
assert {a.name for a in bpy.data.actions}==set(actions),[a.name for a in bpy.data.actions]
rig.data.pose_position='REST';bpy.context.view_layer.update()
rest_equivalence=[]
for o in imported:
    assert o.name in rest,o.name
    e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh()
    imported_points=[e.matrix_world@v.co for v in m.vertices];e.to_mesh_clear()
    source_points=rest[o.name]
    trees=[]
    for pts in [source_points,imported_points]:
        tree=KDTree(len(pts))
        for i,p in enumerate(pts):tree.insert(p,i)
        tree.balance();trees.append(tree)
    distance=max(max(trees[0].find(p)[2] for p in imported_points),max(trees[1].find(p)[2] for p in source_points))
    assert distance<2e-6,(o.name,'rest geometry mismatch',distance)
    rest_equivalence.append({'object':o.name,'bidirectionalMaximumVertexDistance':distance})
rig.data.pose_position='POSE';bpy.context.view_layer.update()
rig.animation_data_create();rig.animation_data.use_nla=False
reimported_poses=[]
for name in ['idle','wave','look','walk','Tpose']:
    action=bpy.data.actions[name];rig.animation_data.action=action
    if action.slots:rig.animation_data.action_slot=action.slots[0]
    source_sample=next(p for p in qa if p['action']==name and p['frame']==round(actions[name][0]+(actions[name][1]-actions[name][0])*.25))
    source_fraction=(source_sample['frame']-actions[name][0])/max(actions[name][1]-actions[name][0],1)
    frame=action.frame_range[0]+(action.frame_range[1]-action.frame_range[0])*source_fraction
    bpy.context.scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update()
    matrices=[list(b.matrix.translation) for b in rig.pose.bones]
    assert all(math.isfinite(c) for row in matrices for c in row)
    points=[]
    for o in imported:
        e=o.evaluated_get(bpy.context.evaluated_depsgraph_get());m=e.to_mesh()
        points.extend(e.matrix_world@v.co for v in m.vertices);e.to_mesh_clear()
    assert all(math.isfinite(c) for p in points for c in p)
    bbox=[[min(p[i] for p in points),max(p[i] for p in points)] for i in range(3)]
    source_bbox=[[min(o['bbox'][i][0] for o in source_sample['objects']),max(o['bbox'][i][1] for o in source_sample['objects'])] for i in range(3)]
    bbox_error=max(abs(bbox[i][j]-source_bbox[i][j]) for i in range(3) for j in range(2))
    assert bbox_error<2e-4,(name,'reimport pose mismatch',bbox_error,bbox,source_bbox)
    reimported_poses.append({'name':name,'frame':frame,'bbox':bbox,'sourceBbox':source_bbox,'sourceBboxMaxError':bbox_error})
source_reports.update({'reimportVerified':True,'importerBoneDisplayHelpers':[o.name for o in custom_shapes],'reimportImages':images,'reimportWeights':weight_audit,'reimportRestEquivalence':rest_equivalence,'reimportActionSamples':reimported_poses,'notes':['GLB fresh-import verified for mesh count, armature modifiers, packed image decoding, normalized skin weights and five animated action samples.','Rest geometry compared bidirectionally against original evaluated source points, allowing glTF UV seam vertex duplication.','Visual key poses are source renders; root performs independent web runtime inspection.','No full dynamic cloth/hair or exhaustive inter-object collision verification.']})
(WORK/'export-reimport-report.json').write_text(json.dumps(source_reports,indent=2),encoding='utf8')
print('ANIMATED_PREVIEW_REIMPORTED',json.dumps({k:v for k,v in source_reports.items() if k in ['blendSha256','glbSha256','actions','meshCount','boneCount','reimportVerified']}))
