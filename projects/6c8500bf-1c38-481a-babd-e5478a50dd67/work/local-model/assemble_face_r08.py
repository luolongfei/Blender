"""Apply bounded R08 face/brow changes on the matched R07 walk copy, then review."""
import bpy, sys, json, hashlib, runpy
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model'
source=p/'work/reconstruction-r02/reference-combined-walk-r07.blend'
if Path(bpy.data.filepath).resolve()!=source.resolve():raise RuntimeError('Exact combined R07 source required')
target=p/'work/reconstruction-r02/reference-face-flow-r08.blend'
if target.exists():raise FileExistsError(target)
scene=bpy.context.scene;scene.frame_set(1);bpy.context.view_layer.update()
face=runpy.run_path(str(folder/'face_transition_r08.py'))['run']()
brow=runpy.run_path(str(folder/'brow_flow_r08.py'))['run']()
bpy.ops.wm.save_as_mainfile(filepath=str(target))
(p/'work/reconstruction-r02/face-brow-actual-r08.json').write_text(json.dumps({
    'source':source.relative_to(p).as_posix(),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'output':target.relative_to(p).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
    'face':face,'brow':brow,'appearanceReview':'pending actual two-angle render'},indent=2),encoding='utf-8')
# Temporary neutral render configuration only; saved candidate keeps matched walk.
bpy.data.objects['Character_Rig']['R04_native_hair_enabled']=0.0
for o in bpy.data.objects:
    if o.type=='MESH' and o.data.shape_keys and o.data.shape_keys.animation_data:
        o.data.shape_keys.animation_data.action=None
sys.argv=[str(folder/'reference_render_r02.py'),'--','face-flow-r08','face-front,face-quarter']
runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
