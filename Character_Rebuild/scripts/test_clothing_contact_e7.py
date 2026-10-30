import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT
from clothing_contact_e7 import repair_clothing_contacts_e7,sat_object
source=OUT/'stages/E7-uv-normals-corrected.blend'
sha=hashlib.sha256(source.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(source))
report=repair_clothing_contacts_e7()
dest=OUT/'work/clothing-contact-e7';dest.mkdir(parents=True,exist_ok=True)
candidate=dest/'clothing-contact-corrected.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(candidate))
names=['Rebuild_Body_Clothing_SourceRetained','Rebuilt_FrontCollar_R']
for o in bpy.context.scene.objects:o.select_set(False)
for name in names:
    obj=bpy.data.objects[name];obj.hide_set(False);obj.hide_render=False;obj.select_set(True)
bpy.context.view_layer.objects.active=bpy.data.objects[names[0]]
subset=dest/'clothing-contact-subset.glb'
bpy.ops.export_scene.gltf(filepath=str(subset),export_format='GLB',use_selection=True,
                         export_apply=True,export_texcoords=True,export_normals=True,
                         export_materials='EXPORT',export_image_format='AUTO',export_yup=True,
                         export_animations=False,export_cameras=False,export_lights=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(subset))
report['freshSubsetGLBReimportSAT']={name:sat_object(bpy.data.objects[name]) for name in names}
assert all(r.get('status')=='completed' and r.get('contacts')==0 for r in report['freshSubsetGLBReimportSAT'].values())
report['source']=str(source);report['sourceSha256']=sha
report['sourceUnchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==sha
report['candidate']=str(candidate);report['candidateSha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
report['subsetGLB']=str(subset);report['subsetGLBSha256']=hashlib.sha256(subset.read_bytes()).hexdigest()
report['rendered']=False;report['reasonNoRender']='Only one sub-0.1mm-source-unit seam point and hidden collar underside changed; visible front collar top unchanged.'
(dest/'correction-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'afterSAT':report['afterSAT'],'freshSubsetGLBReimportSAT':report['freshSubsetGLBReimportSAT'],
                  'sourceUnchanged':report['sourceUnchanged'],'candidate':str(candidate)},indent=2))
