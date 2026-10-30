import bpy,sys,json,hashlib,shutil
from pathlib import Path
from mathutils import Vector,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,render_views,neutral_review
import geometry_checks,materials
stage=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'E-textured-refinement.blend'
assert Path(stage).name==stage and Path(stage).suffix=='.blend', 'Pass one stage filename inside Character_Rebuild/stages'
source=OUT/'stages'/stage
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(source))
model=[o for o in bpy.context.scene.objects if o.type=='MESH' and not o.name.startswith('Reference_') and not o.hide_render]
assert model and all('Original_Character'!=o.name for o in model)
# This rebuild promises mesh hair. Do not silently discard a new curve/particle
# replacement if a later stage introduces one before this export is rerun.
unsupported=[o.name for o in bpy.context.scene.objects if o.type in {'CURVE','CURVES','SURFACE','META','FONT'} and not o.hide_render and not o.name.startswith('Reference_')]
assert not unsupported, 'Renderable non-mesh geometry needs explicit conversion: '+str(unsupported)
modifier_mismatch=[]
for o in model:
    for m in o.modifiers:
        if m.show_viewport!=m.show_render or (m.type=='SUBSURF' and m.show_viewport and m.levels!=m.render_levels):
            modifier_mismatch.append({'object':o.name,'modifier':m.name,'type':m.type})
assert not modifier_mismatch, 'Source review and evaluated export modifier settings differ: '+str(modifier_mismatch)
for o in bpy.context.scene.objects:o.select_set(False)
for o in model:o.hide_set(False);o.select_set(True)
bpy.context.view_layer.objects.active=model[0]
used_images={n.image for o in model for mat in o.data.materials if mat and mat.use_nodes for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image}
for im in used_images:
    # Packed images are decoded lazily after opening a .blend; force a genuine
    # pixel read before distinguishing missing data from data not yet decoded.
    sample=im.pixels[:4]
    assert len(sample)==4 and im.has_data and min(im.size)>0, 'A model texture cannot decode: '+im.name
for im in bpy.data.images:
    if im.source=='FILE' and im.has_data and not im.packed_file:im.pack()
assert all(im.packed_file for im in used_images), 'A model texture is not packed'
neutral_review()
bpy.context.scene.camera.location=(0,-5,0)
bpy.context.scene.camera.rotation_euler=(Vector((0,0,0))-bpy.context.scene.camera.location).to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera.data.ortho_scale=2.2
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_rotation=Quaternion((1,0,0),1.57079632679)
            area.spaces.active.region_3d.view_location=(0,0,0)
            area.spaces.active.region_3d.view_distance=3.2
            area.spaces.active.shading.type='MATERIAL'
blend=OUT/'character-rebuilt.blend';glb=OUT/'character-rebuilt-textured.glb'
bpy.context.scene['rebuild_input']='character-hq-textured.glb'
bpy.context.scene['rebuild_input_sha256']='105b77790ff4b382b0ef83189827d774a596f4a8069cefdb8911ad85f93af3b2'
bpy.context.scene['rebuild_scope']='Static anime character, T-pose, editable replacement head/hands/hair, retained repaired garments; no rig/expressions'
assert hashlib.sha256((OUT.parent/'character-hq-textured.glb').read_bytes()).hexdigest()==bpy.context.scene['rebuild_input_sha256'], 'Original input changed'
# Fixed delivery names may be updated, but keep a verified previous copy.
for existing in (blend,glb):
    if existing.exists():
        digest=hashlib.sha256(existing.read_bytes()).hexdigest()
        archived=OUT/'stages'/'delivery-history'/(existing.stem+'-'+digest[:16]+existing.suffix)
        archived.parent.mkdir(parents=True,exist_ok=True)
        if not archived.exists():shutil.copy2(existing,archived)
        assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest
bpy.ops.export_scene.gltf(filepath=str(glb),export_format='GLB',use_selection=True,export_apply=True,export_texcoords=True,export_normals=True,export_materials='EXPORT',export_image_format='AUTO',export_yup=True,export_animations=False,export_cameras=False,export_lights=False)
for o in model:o.select_set(False)
bpy.context.view_layer.objects.active=None
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
rep=geometry_checks.inspect_scene(model,evaluated=True,self_intersection_objects=[o.name for o in model],max_self_triangles=200000,max_self_candidates=3000000)
rep['materialAudit']=materials.material_export_audit(model)
rep['sourceStage']='Character_Rebuild/stages/'+stage
rep['sourceStageSha256']=source_sha
rep['exportObjectNames']=sorted(o.name for o in model)
rep['excludedSceneObjects']=[{'name':o.name,'type':o.type,'hideRender':o.hide_render} for o in bpy.context.scene.objects if o not in model]
rep['meshHairConfirmedBeforeExport']=any(o.name.startswith(('Hair_','Fringe_','Temple_','Ponytail_','BackSweep_')) for o in model)
rep['sourceStageUnchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
rep['exportOptions']={'format':'GLB','selectionOnly':True,'applyModifiers':True,'texcoords':True,'normals':True,'materials':'EXPORT','imageFormat':'AUTO','yUp':True,'animations':False,'cameras':False,'lights':False}
rep['modelImages']=[{'name':im.name,'size':list(im.size),'packed':bool(im.packed_file),'hasData':im.has_data,'colorSpace':im.colorspace_settings.name} for im in sorted(used_images,key=lambda x:x.name)]
rep['inputUnchanged']=hashlib.sha256((OUT.parent/'character-hq-textured.glb').read_bytes()).hexdigest()==bpy.context.scene['rebuild_input_sha256']
rep['blendSha256']=hashlib.sha256(blend.read_bytes()).hexdigest();rep['glbSha256']=hashlib.sha256(glb.read_bytes()).hexdigest()
(OUT/'reports/final-source-geometry.json').write_text(json.dumps(rep,indent=2,default=str),encoding='utf-8')
assert not rep['errors'] and rep['inputUnchanged'] and rep['sourceStageUnchanged'], 'Source audit failed; consult final-source-geometry.json'
assert rep['meshHairConfirmedBeforeExport'], 'No expected hair mesh was selected'
contact_issues=[{'object':e['object'],'status':e['selfContacts']['status'],'contacts':e['selfContacts'].get('contacts')} for e in rep['objects'] if e['selfContacts']['status']!='completed' or e['selfContacts'].get('contacts',0)]
assert not contact_issues, 'Per-mesh self-contact check requires review before final rendering: '+str(contact_issues)
render_views('checks/final')
print('EXPORT_WRITTEN',str(glb),'REIMPORT_STILL_REQUIRED')
