import bpy,sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,render_views,neutral_review
import geometry_checks,materials
# Fresh empty scene: no inherited geometry, UV, external texture or material.
bpy.ops.wm.read_factory_settings(use_empty=True)
glb=OUT/'character-rebuilt-textured.glb'
source_report=json.loads((OUT/'reports/final-source-geometry.json').read_text(encoding='utf-8'))
glb_sha=hashlib.sha256(glb.read_bytes()).hexdigest()
assert source_report['glbSha256']==glb_sha, 'GLB differs from the recorded export'
assert source_report['blendSha256']==hashlib.sha256((OUT/'character-rebuilt.blend').read_bytes()).hexdigest(), 'Editable blend differs from the recorded export'
bpy.ops.import_scene.gltf(filepath=str(glb))
model=[o for o in bpy.context.scene.objects if o.type=='MESH']
model_names=[o.name for o in model]
report=geometry_checks.inspect_scene(model,evaluated=True,self_intersection_objects=model_names,max_self_triangles=200000,max_self_candidates=3000000)
report['glbSha256']=glb_sha
report['blendSha256']=source_report['blendSha256']
report['sourceStageSha256']=source_report['sourceStageSha256']
report['cleanFactoryImport']=True
report['meshNames']=model_names
report['materials']=materials.material_export_audit(model)
for im in bpy.data.images:
    sample=im.pixels[:4]
    assert len(sample)==4,'An embedded image cannot decode: '+im.name
report['images']=[{'name':im.name,'size':list(im.size),'packed':bool(im.packed_file),'hasData':im.has_data} for im in bpy.data.images]
report['armatures']=sum(o.type=='ARMATURE' for o in bpy.context.scene.objects)
report['excludedReferenceHelpers']=not any(n.startswith('Reference_') for n in model_names)
report['hairMeshCount']=sum(n.startswith(('Hair_','Fringe_','Temple_','Ponytail_','BackSweep_')) for n in model_names)
report['handsPresent']=all(any(n.startswith(x) for n in model_names) for x in ['Rebuilt_Hand_L','Rebuilt_Hand_R'])
report['sourceMeshNames']=source_report['exportObjectNames']
report['missingSourceMeshes']=sorted(set(source_report['exportObjectNames'])-set(model_names))
report['unexpectedMeshes']=sorted(set(model_names)-set(source_report['exportObjectNames']))
report['evaluatedTriangleCountMatchesSource']=report['totals']['triangles']==source_report['totals']['triangles']
report['cameraCountBeforeReviewSetup']=sum(o.type=='CAMERA' for o in bpy.context.scene.objects)
report['lightCountBeforeReviewSetup']=sum(o.type=='LIGHT' for o in bpy.context.scene.objects)
def total_bounds(entries):
    boxes=[e['worldBounds'] for e in entries if e.get('worldBounds')]
    return [[min(b[i][0] for b in boxes),max(b[i][1] for b in boxes)] for i in range(3)] if boxes else None
source_bounds=total_bounds(source_report['objects']);imported_bounds=total_bounds(report['objects'])
report['sourceWorldBounds']=source_bounds;report['reimportWorldBounds']=imported_bounds
report['maxWorldBoundsDifference']=max(abs(a-b) for sa,sb in zip(source_bounds,imported_bounds) for a,b in zip(sa,sb)) if source_bounds and imported_bounds else None
report['geometryAndTransformBoundsMatch']=report['maxWorldBoundsDifference'] is not None and report['maxWorldBoundsDifference']<1e-5
report['verificationFailures']=[]
def require(condition,message):
    if not condition:report['verificationFailures'].append(message)
require(not report['errors'],'Mesh diagnostics raised errors')
require(all(e['selfContacts']['status']=='completed' and e['selfContacts'].get('contacts',0)==0 for e in report['objects']),'A mesh has internal self-contacts or an incomplete check')
require(report['excludedReferenceHelpers'],'Reference geometry was exported')
require(report['handsPresent'],'A rebuilt hand is missing')
require(report['hairMeshCount']>0,'Expected hair meshes are missing')
require(not report['missingSourceMeshes'] and not report['unexpectedMeshes'],'Mesh names differ from selected source export')
require(report['evaluatedTriangleCountMatchesSource'],'Evaluated source and imported GLB triangle counts differ')
require(report['geometryAndTransformBoundsMatch'],'World bounds differ from the source export')
require(report['armatures']==0,'Unexpected armature in static delivery')
require(report['cameraCountBeforeReviewSetup']==0 and report['lightCountBeforeReviewSetup']==0,'Export contains review camera/light helpers')
require(bool(report['images']) and all(im['packed'] and im['hasData'] and min(im['size'])>0 for im in report['images']),'Embedded model images failed to load')
report['structuralReimportPassed']=not report['verificationFailures']
report['visualReviewCompleted']=False
report['fiveFingerVisualCheckCompleted']=False
report['reviewImagesRendered']=False
(OUT/'reports/clean-reimport.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
assert report['structuralReimportPassed'], 'Clean reimport failed: '+str(report['verificationFailures'])
# Reuse only the exact review light/camera/world settings, never source materials.
with bpy.data.libraries.load(str(OUT/'character-rebuilt.blend'),link=False) as (src,dst):
    dst.objects=[n for n in src.objects if n.startswith('Review_')]
    dst.worlds=['Neutral_Review_World']
for o in dst.objects:
    if o:bpy.context.scene.collection.objects.link(o)
bpy.context.scene.world=dst.worlds[0]
bpy.context.scene.camera=next(o for o in dst.objects if o.type=='CAMERA')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'stages/F-clean-glb-reimport.blend'))
render_views('checks/reimport',['front','back','face-front','face-side','face-other-side','face-back-quarter','hand-top','hand-palm','hand-other-top','hand-other-palm'])
report['reviewImagesRendered']=True
report['glbUnchangedDuringReimport']=hashlib.sha256(glb.read_bytes()).hexdigest()==glb_sha
(OUT/'reports/clean-reimport.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
assert report['glbUnchangedDuringReimport'], 'GLB changed while reimport was checked'
print('CLEAN_REIMPORT_READY_FOR_VISUAL_REVIEW')
