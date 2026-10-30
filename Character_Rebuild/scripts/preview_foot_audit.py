"""Read-only sampled foot contact verification for the preview's short step."""
import bpy,sys,json,math,importlib.util
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.context import workspace_path
OUT=workspace_path('Character_Rebuild/interactive')
bpy.ops.wm.open_mainfile(filepath=str(OUT/'rigged-preview.blend'))
arm=bpy.data.objects['PreviewHumanoid'];body=bpy.data.objects['Rebuild_Body_Clothing_SourceRetained']
spec=importlib.util.spec_from_file_location('motions',OUT/'scripts/motions.py');motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
arm.animation_data.action=bpy.data.actions['walk'];arm.animation_data.action_slot=bpy.data.actions['walk'].slots[0];arm.animation_data.use_nla=False
shoe_ids={side:[v.index for v in body.data.vertices if v.co.z<-.93 and (v.co.x>0)==(side=='Left')] for side in ['Left','Right']}
rest_sole={side:min(body.data.vertices[i].co.z for i in ids) for side,ids in shoe_ids.items()}
rows=[]
for t in [0,.35,.7,1.05,1.4,1.75,2.1,2.45,2.8]:
    expected=motion.apply_preview_pose(arm,'walk',t)
    frame=1+30*t;bpy.context.scene.frame_set(math.floor(frame),subframe=frame-math.floor(frame));bpy.context.view_layer.update()
    e=body.evaluated_get(bpy.context.evaluated_depsgraph_get());me=e.to_mesh()
    for side in ['Left','Right']:
        info=expected['legs'][side];actual=arm.matrix_world@arm.pose.bones[side+'Foot'].head
        error=(actual-Vector(info['ankleTarget'])).length
        sole=min((e.matrix_world@me.vertices[i].co).z for i in shoe_ids[side])
        sole_error=sole-(rest_sole[side]+info['lift'])
        rows.append({'seconds':t,'side':side,'stance':info['stance'],'ankleTargetError':error,'soleZ':sole,'expectedSoleZ':rest_sole[side]+info['lift'],'soleError':sole_error})
        assert error<1e-5 and abs(sole_error)<1e-5,(t,side,error,sole_error)
    e.to_mesh_clear()
report={'samples':rows,'maxAnkleTargetError':max(r['ankleTargetError'] for r in rows),'maxSoleHeightError':max(abs(r['soleError']) for r in rows),'scope':'Nine times per foot; short in-place walk, original shoe sole baseline, no exhaustive collision simulation.'}
(OUT/'work/foot-contact-report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print('FOOT_CONTACT_VERIFIED',json.dumps({k:v for k,v in report.items() if k!='samples'}))
