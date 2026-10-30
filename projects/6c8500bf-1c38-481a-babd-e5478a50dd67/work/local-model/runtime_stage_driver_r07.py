"""Integrator-owned CLI for reversible R07 runtime checkpoints."""
import bpy, json, runpy, sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'toolchain.lock.json').exists())
sys.path.insert(0,str(ROOT))
from pipeline.context import project_path
p=project_path();folder=p/'work/local-model';root=p/'work/runtime-r06'
args=sys.argv[sys.argv.index('--')+1:];stage=args[0]
def save(name):
    target=root/name
    if target.exists():raise FileExistsError(target)
    bpy.ops.wm.save_as_mainfile(filepath=str(target))
    print('R07_RUNTIME_SAVED',target,flush=True)
if stage=='cull':
    api=runpy.run_path(str(folder/'runtime_torso_cull_r07.py'))
    api['audit']();report=api['apply_candidate']()
    save('r07-geometry-cull-candidate.blend')
elif stage in {'skin','root','iris','blouse','skirt','sock'}:
    api=runpy.run_path(str(folder/'runtime_material_stage_r06.py'))
    if not (root/'material-stage-plan.json').exists():api['plan']()
    output_root=args[1] if len(args)>1 else 'work/runtime-r06/channel-bakes-r07'
    report=api['run_stage'](stage,output_root=output_root)
    saved=runpy.run_path(str(folder/'runtime_fresh_r07.py'))['save_material_stage'](stage)
    print('R07_MATERIAL_SAVED',json.dumps(saved),flush=True)
else:raise ValueError(stage)
print('R07_RUNTIME_STAGE_DONE',stage,flush=True)
