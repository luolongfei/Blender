import bpy,sys,json,hashlib,array
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,neutral_review
import neck_join_delta,geometry_checks
src=OUT/'stages/E7-uv-normals-corrected.blend'
source_sha='fb285cbe33d0015f185cdf6a83e28e059e5043726b4640e347c2000117923cdf'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(src)==source_sha
bpy.ops.wm.open_mainfile(filepath=str(src))
def protected():
    result={}
    for ob in bpy.context.scene.objects:
        if ob.type!='MESH' or ob.name=='Neck_Clean_Quad_Sleeve':continue
        vertices=array.array('f',[0])*(len(ob.data.vertices)*3)
        loops=array.array('i',[0])*len(ob.data.loops)
        ob.data.vertices.foreach_get('co',vertices)
        ob.data.loops.foreach_get('vertex_index',loops)
        result[ob.name]=hashlib.sha256(vertices.tobytes()+loops.tobytes()+str(tuple(tuple(r) for r in ob.matrix_world)).encode()).hexdigest()
    return result
baseline=protected()
report=neck_join_delta.apply()
report['sourceSha256']=source_sha
report['allOtherMeshGeometryTransformsExact']=protected()==baseline
assert report['allOtherMeshGeometryTransformsExact']
neck=bpy.data.objects['Neck_Clean_Quad_Sleeve']
bpy.context.view_layer.update()
report['geometry']=geometry_checks.inspect_scene([neck],evaluated=True,self_intersection_objects=[neck.name])
assert report['geometry']['objects'][0]['selfContacts']['contacts']==0
scene=neutral_review();scene.render.resolution_x=640;scene.render.resolution_y=640
cam=scene.camera;cam.location=(1.5,4,.86);target=Vector((0,.025,.79))
cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=.59
dest=OUT/'checks/neck-join';dest.mkdir(parents=True,exist_ok=True)
scene.render.filepath=str(dest/'face-back-quarter.png')
bpy.ops.render.render(write_still=True)
work=OUT/'work';work.mkdir(exist_ok=True)
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(work/'neck-join-candidate.blend'))
report['sourceUnchanged']=digest(src)==source_sha
report['savedCandidate']=str(work/'neck-join-candidate.blend')
(OUT/'reports/neck-join-candidate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('NECK_JOIN_DONE',json.dumps({k:v for k,v in report.items() if k!='geometry'}))
