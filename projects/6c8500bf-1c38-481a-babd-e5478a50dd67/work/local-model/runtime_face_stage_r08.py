"""Explicit CLI stages for an R08 refresh of the already baked R07 runtime."""
import bpy,json,runpy,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model';root=p/'work/runtime-r06'
stage=sys.argv[sys.argv.index('--')+1]
refresh='r08-retry1'
def api(name):return runpy.run_path(str(folder/(name+'.py')))
def save(name):
    path=root/name
    if path.exists():raise FileExistsError(path)
    bpy.ops.wm.save_as_mainfile(filepath=str(path));print('R08_SAVED',path,flush=True)
if stage=='prepare':
    result=api('runtime_face_refresh_r08')['prepare_payload'](source_relative='work/reconstruction-r02/reference-face-flow-r08.blend',stage=refresh)
elif stage=='swap':
    result=api('runtime_face_refresh_r08')['replace'](manifest_relative='work/runtime-r06/'+refresh+'-face-refresh.json',runtime_relative='work/runtime-r06/r07-material-sock.blend',recull=True)
    save('r08-geometry-face-refresh.blend')
elif stage in {'skin','iris'}:
    result=api('runtime_material_stage_r06')['run_stage'](stage,output_root='work/runtime-r06/channel-bakes-r08')
    save('r08-material-'+stage+'.blend')
elif stage=='review':
    result=api('runtime_cull_pose_review_r07')['run'](render=True,output_relative='previews/reconstruction-r02/runtime-cull-poses-r08')
    sys.argv=[str(folder/'reference_render_r02.py'),'--','runtime-face-r08','face-front,face-quarter']
    runpy.run_path(str(folder/'reference_render_r02.py'),run_name='__main__')
elif stage=='export':
    export=api('runtime_export_r06')
    lineage=['work/reconstruction-r02/reference-face-flow-r08.blend','work/runtime-r06/r07-material-sock.blend','work/runtime-r06/'+refresh+'-face-payload.blend']
    export['prepare'](source_lineage=lineage,require_baked=True)
    save('r08-export-ready.blend')
    export['prepare'](source_lineage=lineage,require_baked=True)
    result=export['export'](output_relative='work/runtime-r06/preview-export-r08')
else:raise ValueError(stage)
print('R08_STAGE_DONE',stage,flush=True)
