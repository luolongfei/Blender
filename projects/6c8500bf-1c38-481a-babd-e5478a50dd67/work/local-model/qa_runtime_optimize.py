from pathlib import Path
import bpy,json,sys
from mathutils import Matrix
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runtime_optimize,vrm_helpers
import final_export
from bpy_extras.node_shader_utils import PrincipledBSDFWrapper
rig=bpy.data.objects['Character_Rig'];meshes=[o for o in bpy.data.collections['CHARACTER · Original geometry'].all_objects if o.type=='MESH']
for bone in rig.pose.bones:bone.matrix_basis=Matrix.Identity(4)
for obj in meshes:
 for modifier in obj.modifiers:
  if modifier.type=='SUBSURF':modifier.levels=modifier.render_levels=0
bpy.context.view_layer.update()
plan=json.loads(rig['charakit_spring_plan_v1'])
normal_requirements={}
for mat in {mat for obj in meshes for mat in obj.data.materials}:
 w=PrincipledBSDFWrapper(mat);t=w.normalmap_texture
 if t and t.image:normal_requirements[mat.name]={'image':t.image.name,'strength':w.normalmap_strength}
vrm_helpers._enabled().apply_mtoon(meshes)
meshes,plan,report=runtime_optimize.optimize(rig,meshes,plan)
report['bindings']=vrm_helpers.audit_bindings(rig,meshes)
spring_api=vrm_helpers._module('spring-setup.py','optimize_spring_audit')
report['springAudit']=spring_api.audit(rig,meshes)
vrm_helpers.export_review(rig,meshes,HERE/'qa-runtime-optimized.vrm',convert_materials=False)
doc=final_export.glb_document(HERE/'qa-runtime-optimized.vrm')
report['exportedStatistics']=final_export.glb_stats(doc)
report['normalChecks']=final_export.assert_normal_textures(doc,normal_requirements,'QA Runtime VRM')
final_export.assert_mesh_coverage(doc,meshes,'QA Runtime VRM')
(HERE/'qa-runtime-spring-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
(HERE/'qa-runtime-optimization.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('RUNTIME_OPTIMIZATION',report['sourceMeshCount'],report['runtimeMeshCount'],report['beforeTriangles'],report['afterTriangles'])
