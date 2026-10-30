"""Save matched mocap, native skirt playback and embedded twin-tail dynamics."""
import bpy,runpy,sys,json,hashlib
from pathlib import Path
from mathutils import Vector
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model';rig=bpy.data.objects['Character_Rig']
target=p/'work/reconstruction-r02/reference-combined-walk-r07.blend'
if target.exists():raise RuntimeError('Retain the existing combined source')
action='R06 CMU 07_01 walk - cloth preroll'
if rig.animation_data.action.name!=action:raise RuntimeError('Expected preserved mocap walk')
if 'R07 Native Walk Cloth playback' not in bpy.data.objects:raise RuntimeError('Native cloth required')
head=bpy.data.objects['R01 Head'];keys=head.data.shape_keys
if keys.animation_data and keys.animation_data.action:raise RuntimeError('Existing face action needs explicit integration')
keys.animation_data_create()
for side,offset in [('L',0.),('R',.5)]:
    key=keys.key_blocks['R02_blink_'+side]
    for frame,value in [(1.,0.),(35.+offset,0.),(38.+offset,.82),(39.+offset,1.),(41.+offset,.45),(43.+offset,0.),(80.,0.)]:
        key.value=value;key.keyframe_insert(data_path='value',frame=frame)
face_action=keys.animation_data.action;face_action.name='R07 Walk authored natural blink';face_action.use_fake_user=True
api=runpy.run_path(str(folder/'secondary_hair_r02.py'))
colliders=api['_collider_definitions'](rig)
center=rig.data.bones['head'].matrix_local.inverted()@Vector((0,.00202,1.441285))
for c in colliders:
    if c['name']=='head':c.update(localCenter=list(center),radius=.084)
cache='work/reconstruction-r02/hair-walk-r07/R06-CMU-07_01-walk---cloth-preroll.guides.npz'
if not (p/cache).exists():
    report=api['bake_actions'](actions=[action],frame_ranges={action:(1,80)},output='work/reconstruction-r02/hair-walk-r07',allow_authored_ik=True,colliders=colliders)
    cache=report['actions'][0]['cache'].replace('\\','/')
else:
    print('R07_REUSE_EXISTING_HAIR_CACHE',cache,flush=True)
native=runpy.run_path(str(folder/'native_hair_playback_r04.py'))
native['run'](cache_relative=cache);native['enable_for_action'](select=True,frame=1)
parity=native['validate_evaluated'](cache_relative=cache,frames=[1,15.375,35,61,80])
bpy.context.scene.frame_start=1;bpy.context.scene.frame_end=80
bpy.context.scene.frame_set(1);bpy.ops.wm.save_as_mainfile(filepath=str(target))
result={'source':str(target.relative_to(p)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'bodyAction':action,'faceAction':face_action.name,'faceProvenance':'Authored blink, not captured facial performance',
        'hairCache':cache,'hairParity':parity,'cloth':'R07 Native Walk Cloth playback',
        'pending':'Fresh-file reopen, visual cloth/hair contact and continuous video'}
(p/'work/reconstruction-r02/combined-walk-r07.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('R07_COMBINED_WALK_SAVED',str(target),flush=True)
