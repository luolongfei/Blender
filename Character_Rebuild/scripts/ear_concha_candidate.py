import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import OUT,neutral_review
import ear_concha,geometry_checks
source=OUT/'stages/E4-final-refined.blend'
expected='7a406fbbcf337e4098ebc98697766cdb3d88d8966286779c92a5e58e3e6928aa'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(source)==expected
bpy.ops.wm.open_mainfile(filepath=str(source))
def other_mesh_hashes():
    result={}
    for ob in bpy.context.scene.objects:
        if ob.type=='MESH' and not ob.name.startswith('Ear_'):
            result[ob.name]=hashlib.sha256(str((tuple(tuple(v.co) for v in ob.data.vertices),tuple(tuple(p.vertices) for p in ob.data.polygons),tuple(tuple(row) for row in ob.matrix_world))).encode()).hexdigest()
    return result
before=other_mesh_hashes()
report=ear_concha.apply()
report['source']=str(source);report['sourceSha256']=expected
report['otherMeshGeometryAndTransformsExact']=before==other_mesh_hashes()
assert report['otherMeshGeometryAndTransformsExact']
bpy.context.view_layer.update()
obs=[bpy.data.objects['Ear_L'],bpy.data.objects['Ear_R']]
report['geometry']=geometry_checks.inspect_scene(obs,evaluated=True,self_intersection_objects=[o.name for o in obs])
for o in report['geometry']['objects']:
    assert o['selfContacts']['contacts']==0 and o['topology']['nonmanifoldEdgesIncludingBoundaryAndLoose']==0
    assert o['uvLayers'] and o['triangulation']['zeroAreaTriangles']==0
scene=neutral_review();scene.render.resolution_x=640;scene.render.resolution_y=640
cam=scene.camera;target=Vector((.111,.023,.767));cam.location=(3,.023,.767)
cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=.12
dest=OUT/'checks/ear-concha';dest.mkdir(parents=True,exist_ok=True)
scene.render.filepath=str(dest/'ear-side.png')
bpy.ops.render.render(write_still=True)
work=OUT/'work';work.mkdir(exist_ok=True)
bpy.ops.file.pack_all();bpy.ops.wm.save_as_mainfile(filepath=str(work/'ear-concha-candidate.blend'))
report['sceneSaved']=str(work/'ear-concha-candidate.blend')
report['sourceUnchanged']=digest(source)==expected
(OUT/'reports/ear-concha-candidate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('EAR_CONCHA_DONE',json.dumps({k:report[k] for k in ['sourceUnchanged','otherMeshGeometryAndTransformsExact','removedInnerPills']}))
